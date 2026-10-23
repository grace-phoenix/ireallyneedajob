#!/usr/bin/env python3
"""
Async web server + Discord bot in ONE process (one asyncio event loop).

REQUIREMENTS
    Python 3.10+
    pip install -U discord.py aiohttp

ENVIRONMENT VARIABLES
    DISCORD_BOT_TOKEN   (required) the bot token
    AUTHORISED_USER_ID  (required) numeric Discord user ID; the only account
                        whose slash commands are honoured
    HOST                (optional) interface to bind, default 0.0.0.0
    PORT                (optional) port to listen on, default 8080

HOW TO RUN
    Linux / macOS:
        export DISCORD_BOT_TOKEN="your-token"
        export AUTHORISED_USER_ID="123456789012345678"
        python discord_web_bot.py
    Windows PowerShell:
        $env:DISCORD_BOT_TOKEN="your-token"
        $env:AUTHORISED_USER_ID="123456789012345678"
        python discord_web_bot.py

COMMANDS (slash commands, authorised user only)
    /pageset path content      serve text at a path
    /fileset path file         serve an attached file at a path (slash-command
                               attachment option)
    /analytics duration        e.g. 30m, 8h, 4d
    /formset path              register a POST endpoint at a path
    /setchannel channel        where form POSTs get delivered
    /viewforms                 show submissions that could NOT be delivered

KEY DESIGN DECISIONS (also commented where they apply)
  * Everything lives in RAM (class State). Restarting the process wipes pages,
    files, analytics, the configured channel and the undelivered-form queue.
  * The web server and the bot are independent tasks. The bot task retries
    forever (120 s pause) on ANY failure, including HTTP 429; the web server
    keeps serving while the bot is down.
  * Access control is a single CommandTree.interaction_check. Other users'
    commands are ignored (no reply at all; Discord shows them its generic
    "application did not respond" notice, which is inherent to ignoring).
  * Stored state for forms is explicit: State.undelivered holds ONLY
    submissions that could not be delivered. A delivered submission is never
    stored. /viewforms is read-only.
"""

import asyncio
import logging
import mimetypes
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import discord
from aiohttp import web
from discord import app_commands

log = logging.getLogger("webbot")

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
RETRY_DELAY_SECONDS = 120        # required: wait 120 s between bot start attempts
DISCORD_CHUNK = 1900             # text per Discord message (limit is 2000; room for code fences)
MAX_HITS = 200_000               # safety cap so analytics can't grow RAM forever (oldest dropped)
MAX_UNDELIVERED = 500            # safety cap on the undelivered-form queue (oldest dropped, logged)
MAX_ANALYTICS_LINES = 300        # per-request lines shown by /analytics (summary always covers all)
MAX_BODY_BYTES = 256 * 1024      # request body cap for the web server
DELIVERY_TIMEOUT = 20            # seconds to wait for Discord before queueing a form


# ----------------------------------------------------------------------------
# In-memory state
# ----------------------------------------------------------------------------
@dataclass(slots=True)
class StoredFile:
    data: bytes
    content_type: str


@dataclass(slots=True)
class Hit:
    """One web request, as recorded by the analytics middleware."""
    ts: float
    method: str
    path: str
    query: str
    status: int
    ip: str
    user_agent: str
    referer: str
    forwarded_for: str  # useful behind a reverse proxy, where `ip` is the proxy


@dataclass(slots=True)
class Submission:
    ts: float
    path: str
    ip: str
    body: str


class State:
    def __init__(self, owner_id: int):
        self.owner_id = owner_id
        self.pages: dict[str, str] = {}
        self.files: dict[str, StoredFile] = {}
        self.form_paths: set[str] = set()
        self.channel_id: int | None = None
        # Stored form state: ONLY undelivered submissions (see module docstring).
        self.undelivered: deque[Submission] = deque(maxlen=MAX_UNDELIVERED)
        self.hits: deque[Hit] = deque(maxlen=MAX_HITS)
        # The currently running bot instance (None while it is down/restarting).
        self.bot: discord.Client | None = None


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def norm_path(raw: str) -> str:
    """'hello/' -> '/hello'. Used for both stored paths and incoming requests."""
    path = raw.strip()
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    return path


def fmt_ts(ts: float, short: bool = False) -> str:
    dt = datetime.fromtimestamp(ts, timezone.utc)
    return dt.strftime("%m-%d %H:%M:%S" if short else "%Y-%m-%d %H:%M:%S UTC")


def with_charset(content_type: str) -> str:
    if content_type.startswith("text/") and "charset" not in content_type:
        return content_type + "; charset=utf-8"
    return content_type


def chunk_text(text: str, limit: int = DISCORD_CHUNK) -> list[str]:
    """Split text into pieces <= limit chars, preferring line boundaries."""
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.split("\n"):
        pieces = [line[i:i + limit] for i in range(0, len(line), limit)] or [""]
        for piece in pieces:
            if buf and size + 1 + len(piece) > limit:
                chunks.append("\n".join(buf))
                buf, size = [], 0
            size += len(piece) + (1 if buf else 0)
            buf.append(piece)
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [""]


def code_blocks(text: str) -> list[str]:
    """Wrap text in code fences, split into Discord-sized messages.
    Code blocks keep columns aligned and stop user-supplied text (form fields,
    User-Agent strings...) from being interpreted as Discord markdown."""
    text = text.replace("```", "``\u200b`")  # a stray fence would break out of the block
    return [f"```\n{c}\n```" for c in chunk_text(text)]


_DURATION_RE = re.compile(r"^\s*(\d{1,6})\s*([mhd])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def parse_duration(text: str) -> int | None:
    """'30m' / '8h' / '4d' -> seconds, or None if invalid."""
    match = _DURATION_RE.match(text)
    if not match:
        return None
    amount = int(match.group(1))
    if amount <= 0:
        return None
    return amount * _UNIT_SECONDS[match.group(2).lower()]


# ----------------------------------------------------------------------------
# Web server (aiohttp)
# ----------------------------------------------------------------------------
@web.middleware
async def analytics_middleware(request: web.Request, handler):
    """Records EVERY request (hits, misses, errors) with its origin.
    Request bodies are deliberately not recorded here, so form contents are
    never retained by analytics."""
    state: State = request.app["state"]
    started = time.time()
    status = 500  # assumed unless the handler returns or raises an HTTP error
    try:
        response = await handler(request)
        status = response.status
        return response
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        headers = request.headers
        state.hits.append(Hit(
            ts=started,
            method=request.method,
            path=clip(request.path, 300),
            query=clip(request.query_string, 300),
            status=status,
            ip=request.remote or "unknown",
            user_agent=clip(headers.get("User-Agent", ""), 200),
            referer=clip(headers.get("Referer", ""), 300),
            forwarded_for=clip(headers.get("X-Forwarded-For", ""), 200),
        ))


async def read_form_body(request: web.Request) -> str:
    """Turn a POST body into readable text. Uploaded files inside multipart
    forms are only named, not forwarded."""
    if request.content_type in ("application/x-www-form-urlencoded", "multipart/form-data"):
        data = await request.post()
        lines = []
        for key, value in data.items():
            if isinstance(value, str):
                lines.append(f"{key}: {value}")
            else:  # aiohttp FileField
                lines.append(f"{key}: <file {getattr(value, 'filename', '?')}>")
        return "\n".join(lines)
    raw = await request.read()  # JSON, plain text, anything else: show as text
    return raw.decode("utf-8", errors="replace")


def render_submission(sub: Submission) -> str:
    return (
        f"Form submission to {sub.path}\n"
        f"Received: {fmt_ts(sub.ts)}\n"
        f"From: {sub.ip}\n"
        f"---\n"
        f"{sub.body or '(empty body)'}"
    )


async def deliver_form(state: State, sub: Submission) -> bool:
    """Try to post the submission to the configured Discord channel.
    Returns True only if every message was sent. (If a multi-message post
    fails part-way it is queued whole, so /viewforms may show a repeat of the
    part that did get through.)"""
    bot = state.bot
    if bot is None or state.channel_id is None or not bot.is_ready():
        return False
    channel = bot.get_channel(state.channel_id)
    if not isinstance(channel, discord.abc.Messageable):
        return False

    async def send_all():
        for block in code_blocks(render_submission(sub)):
            # AllowedMentions.none(): a form field containing @everyone must not ping anyone.
            await channel.send(block, allowed_mentions=discord.AllowedMentions.none())

    try:
        await asyncio.wait_for(send_all(), timeout=DELIVERY_TIMEOUT)
        return True
    except Exception as exc:  # HTTPException, Forbidden, timeout, ...
        log.warning("Form delivery failed (%s: %s)", type(exc).__name__, exc)
        return False


async def handle_form(request: web.Request, state: State, path: str) -> web.Response:
    try:
        body = await read_form_body(request)
    except web.HTTPException:
        raise  # e.g. 413 body too large
    except Exception:
        raise web.HTTPBadRequest(text="Could not read form body\n")

    sub = Submission(ts=time.time(), path=path, ip=request.remote or "unknown", body=body)
    if await deliver_form(state, sub):
        return web.Response(text="Delivered\n")  # delivered -> `sub` is simply discarded

    # Not delivered: this is the ONLY place a submission is kept.
    if len(state.undelivered) == state.undelivered.maxlen:
        log.warning("Undelivered-form queue full; dropping the oldest submission")
    state.undelivered.append(sub)
    return web.Response(text="Received (will be viewable by the owner)\n", status=202)


async def handle_request(request: web.Request) -> web.Response:
    state: State = request.app["state"]
    path = norm_path(request.path)

    if request.method == "POST":
        if path in state.form_paths:
            return await handle_form(request, state, path)
        raise web.HTTPNotFound()

    if request.method in ("GET", "HEAD"):
        page = state.pages.get(path)
        if page is not None:
            # Content type guessed from the path's extension (/data.json, /notes.txt);
            # anything without a known extension is served as HTML.
            ctype = with_charset(mimetypes.guess_type(path)[0] or "text/html")
            return web.Response(body=page.encode("utf-8"), headers={"Content-Type": ctype})
        stored = state.files.get(path)
        if stored is not None:
            return web.Response(body=stored.data, headers={"Content-Type": stored.content_type})
        if path in state.form_paths:
            raise web.HTTPMethodNotAllowed(request.method, ["POST"])
        raise web.HTTPNotFound()

    raise web.HTTPMethodNotAllowed(request.method, ["GET", "HEAD", "POST"])


def build_web_app(state: State) -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_BYTES, middlewares=[analytics_middleware])
    app["state"] = state
    # One catch-all route: paths are added/removed at runtime by the bot, so we
    # look them up in State instead of registering aiohttp routes dynamically.
    app.router.add_route("*", "/{tail:.*}", handle_request)
    return app


async def run_web(state: State, host: str, port: int) -> None:
    runner = web.AppRunner(build_web_app(state), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("Web server listening on %s:%s", host, port)
    try:
        await asyncio.Event().wait()  # run until cancelled
    finally:
        await runner.cleanup()


# ----------------------------------------------------------------------------
# Discord bot (discord.py)
# ----------------------------------------------------------------------------
class OwnerOnlyTree(app_commands.CommandTree):
    """Single choke point for access control: every slash command passes
    through interaction_check, and only the authorised user gets through."""

    def __init__(self, client: discord.Client, owner_id: int):
        super().__init__(client)
        self.owner_id = owner_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id  # False => silently ignored

    async def on_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        log.error("Command error: %r", error, exc_info=error)
        message = "Something went wrong; check the bot's log."
        try:  # only the authorised user can reach here, so replying is safe
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            pass


def build_report(label: str, hits: list[Hit], cutoff: float) -> str:
    since = fmt_ts(cutoff)
    if not hits:
        return f"No requests recorded in the last {label} (since {since})."
    ips = Counter(h.ip for h in hits)
    paths = Counter(h.path for h in hits)
    statuses = Counter(h.status for h in hits)

    shown = hits[-MAX_ANALYTICS_LINES:]
    lines = [
        f"Analytics: last {label} (since {since})",
        f"Requests: {len(hits)} | Unique IPs: {len(ips)}",
        "Status codes: " + ", ".join(f"{s} x{n}" for s, n in sorted(statuses.items())),
        "",
        "Top paths:",
        *[f"  {n:>6}  {p}" for p, n in paths.most_common(10)],
        "",
        "Top IPs:",
        *[f"  {n:>6}  {ip}" for ip, n in ips.most_common(5)],
        "",
        f"Requests, oldest first (times UTC; showing {len(shown)} of {len(hits)}):",
    ]
    for h in shown:
        parts = [
            fmt_ts(h.ts, short=True),
            f"{h.method} {h.path}" + (f"?{h.query}" if h.query else ""),
            str(h.status),
            h.ip,
        ]
        if h.user_agent:
            parts.append(f"UA: {h.user_agent}")
        if h.referer:
            parts.append(f"Ref: {h.referer}")
        if h.forwarded_for:
            parts.append(f"XFF: {h.forwarded_for}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def register_commands(tree: app_commands.CommandTree, state: State) -> None:
    # All replies are ephemeral (visible only to the invoker): analytics and
    # form contents include IP addresses and user-submitted data.

    @tree.command(name="pageset", description="Serve text content at a URL path")
    @app_commands.describe(path="URL path, e.g. /hello", content="Text (or HTML) to serve")
    async def pageset(interaction: discord.Interaction, path: str, content: str):
        # Discord caps a string option at 6000 characters.
        p = norm_path(path)
        state.files.pop(p, None)  # a path serves one thing: last command wins
        state.pages[p] = content
        await interaction.response.send_message(
            f"Page stored at `{p}` ({len(content)} characters).", ephemeral=True)

    @tree.command(name="fileset", description="Serve an attached file at a URL path")
    @app_commands.describe(path="URL path, e.g. /logo.png", file="The file to serve")
    async def fileset(interaction: discord.Interaction, path: str, file: discord.Attachment):
        # Size/count limits (<10 MB, <30 files) are guaranteed by the requirements,
        # so they are not enforced here.
        await interaction.response.defer(ephemeral=True, thinking=True)
        data = await file.read()
        p = norm_path(path)
        ctype = with_charset(
            file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream")
        state.pages.pop(p, None)
        state.files[p] = StoredFile(data=data, content_type=ctype)
        await interaction.followup.send(
            f"File `{file.filename}` ({len(data)} bytes, {ctype}) stored at `{p}`.", ephemeral=True)

    @tree.command(name="analytics", description="Show recorded web requests for a time window")
    @app_commands.describe(duration="Window as a number + m/h/d, e.g. 30m, 8h, 4d")
    async def analytics(interaction: discord.Interaction, duration: str):
        seconds = parse_duration(duration)
        if seconds is None:
            await interaction.response.send_message(
                "Use a number followed by `m`, `h` or `d`, e.g. `30m`, `8h`, `4d`.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        cutoff = time.time() - seconds
        hits = [h for h in state.hits if h.ts >= cutoff]
        report = build_report(duration.strip().lower(), hits, cutoff)
        for block in code_blocks(report):  # split across as many messages as needed
            await interaction.followup.send(block, ephemeral=True)

    @tree.command(name="formset", description="Register a POST endpoint at a URL path")
    @app_commands.describe(path="URL path that will accept POST requests, e.g. /contact")
    async def formset(interaction: discord.Interaction, path: str):
        p = norm_path(path)
        state.form_paths.add(p)
        note = "" if state.channel_id else " No channel is set yet; use `/setchannel`."
        await interaction.response.send_message(f"POST endpoint registered at `{p}`.{note}", ephemeral=True)

    @tree.command(name="setchannel", description="Choose the channel that receives form submissions")
    @app_commands.describe(channel="A text channel in a server the bot has joined")
    async def setchannel(interaction: discord.Interaction, channel: discord.TextChannel):
        # Re-resolve through the bot's own cache: if the bot isn't in that server
        # it can't see the channel, so get_channel() returns None. Only the ID is
        # stored, so this works whichever channel object type the option resolves to.
        target = interaction.client.get_channel(channel.id)
        guild = getattr(target, "guild", None)
        if target is None or guild is None:
            await interaction.response.send_message(
                "I can't see that channel. The bot must be a member of its server.", ephemeral=True)
            return
        me = guild.me
        if me is not None:
            perms = target.permissions_for(me)
            if not (perms.view_channel and perms.send_messages):
                await interaction.response.send_message(
                    "I don't have permission to view and send messages in that channel.", ephemeral=True)
                return
        state.channel_id = target.id
        await interaction.response.send_message(
            f"Form submissions will be posted to {target.mention} ({guild.name}).", ephemeral=True)

    @tree.command(name="viewforms", description="Show form submissions that could not be delivered")
    async def viewforms(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        if not state.undelivered:
            await interaction.followup.send("No undelivered submissions.", ephemeral=True)
            return
        parts = [f"{len(state.undelivered)} undelivered submission(s), oldest first "
                 "(kept in RAM until the process restarts):"]
        for i, sub in enumerate(state.undelivered, 1):
            parts.append(f"\n=== #{i} ===\n{render_submission(sub)}")
        for block in code_blocks("\n".join(parts)):
            await interaction.followup.send(block, ephemeral=True)


class WebBot(discord.Client):
    def __init__(self, state: State):
        # Default intents are enough: slash commands need no privileged intent.
        super().__init__(intents=discord.Intents.default())
        self.tree = OwnerOnlyTree(self, state.owner_id)
        register_commands(self.tree, state)

    async def setup_hook(self) -> None:
        # Runs after login, before connecting. If syncing fails (e.g. 429) the
        # exception propagates out of start() and the retry loop handles it.
        await self.tree.sync()


async def run_bot_forever(state: State, token: str) -> None:
    """Start the bot; on ANY failure (HTTP 429, bad connection, ...) wait
    120 s and try again, indefinitely. A fresh client is built each attempt
    because a closed discord.Client can't be reused. Once connected,
    discord.py reconnects on its own after ordinary network drops."""
    while True:
        bot = WebBot(state)
        state.bot = bot
        try:
            await bot.start(token)
            log.warning("Bot stopped; restarting")
        except Exception as exc:
            log.error("Bot failed (%s: %s)", type(exc).__name__, exc)
        finally:
            state.bot = None
            if not bot.is_closed():
                await bot.close()
        log.info("Retrying bot start in %d seconds", RETRY_DELAY_SECONDS)
        await asyncio.sleep(RETRY_DELAY_SECONDS)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
async def run_all(token: str, owner_id: int, host: str, port: int) -> None:
    state = State(owner_id)
    # Both run in the same event loop and start together.
    await asyncio.gather(run_web(state, host, port), run_bot_forever(state, token))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    owner = os.environ.get("AUTHORISED_USER_ID", "").strip()
    if not token or not owner.isdigit():
        sys.exit("Set DISCORD_BOT_TOKEN and AUTHORISED_USER_ID (a numeric Discord user ID).")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    try:
        asyncio.run(run_all(token, int(owner), host, port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
