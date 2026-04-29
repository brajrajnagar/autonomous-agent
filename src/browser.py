"""HTTP-based browser tools: visit a URL and search the web.

Designed for token-efficient consumption by the agent:
- Boilerplate stripping via `trafilatura` (nav/scripts/ads removed).
- Output as Markdown so structure (headings/code/lists) survives truncation.
- Pagination via `offset` so long pages can be read in 4kB chunks.
- Per-process URL cache so re-visits in the same session cost nothing.
- Optional deny-host list (`AGENT_BROWSER_DENY_HOSTS`) to block obvious
  mistakes (localhost, internal corp domains, etc.).

No JS rendering — most documentation, READMEs, blogs, and forum posts
work fine without it. Add Playwright in a follow-up if you hit a real wall.
"""

import hashlib
import os
import urllib.parse
from typing import Dict, List, Optional

import requests
import trafilatura
from bs4 import BeautifulSoup

from state import ToolResult


_USER_AGENT = (
    "Mozilla/5.0 (compatible; AutonomousAgent/1.0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_REQUEST_TIMEOUT = 20

# Per-process URL cache: { url_hash -> full extracted markdown }.
# Resets on every Python invocation, which matches the "per-session" scope.
_URL_CACHE: Dict[str, str] = {}

# Per-process rewrite cache: { original_query -> rewritten_query }.
# Avoids re-paying for the LLM rewrite call when the agent searches the same
# vague query twice in one session.
_REWRITE_CACHE: Dict[str, str] = {}

# Lazy OpenAI client used for query rewriting; created on first use so that
# importing browser doesn't require an API key.
_REWRITER_CLIENT = None


_REWRITE_PROMPT = """Rewrite the user's query as a high-signal web search.

Rules:
- Replace vague time words ("latest", "current", "recent", "now", "today") with concrete dates. Today's date is {today}, so include the year and month when relevant.
- When the query is about a specific tool/platform/library, add a `site:` qualifier toward that platform's docs (e.g. site:docs.python.org, site:pytorch.org).
- Prefer specificity (concrete names, dates, version numbers) over breadth.
- Output 3 to 12 keywords, lowercase preferred.
- Keep proper nouns capitalized when they are official names.

Output the rewritten query as plain text only. No quotes, no preamble, no explanation, no extra lines.

USER QUERY: {query}"""


_OPERATOR_INDICATORS = ('"', 'site:', 'intitle:', 'inurl:', 'filetype:', ' OR ', ' -', '+')


def _looks_already_specific(query: str) -> bool:
    """True when the query has explicit search operators — the user (or
    agent) already knows what they want, so skip rewriting."""
    return any(ind in query for ind in _OPERATOR_INDICATORS)


def _get_rewriter_client():
    """Lazily build (and cache) an OpenAI client for query rewriting."""
    global _REWRITER_CLIENT
    if _REWRITER_CLIENT is None:
        from openai import OpenAI
        _REWRITER_CLIENT = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _REWRITER_CLIENT


def rewrite_query(query: str) -> str:
    """Return a higher-signal version of `query` for web search.

    Falls back to the original query on any error (disabled, no API key,
    LLM failure, malformed output, query already has operators). Cached
    per-process by query string so repeated searches don't re-pay.
    """
    if not query or not query.strip():
        return query
    if os.getenv("AGENT_SEARCH_REWRITE_ENABLED", "true").lower() != "true":
        return query
    if _looks_already_specific(query):
        return query
    if query in _REWRITE_CACHE:
        return _REWRITE_CACHE[query]
    if not os.getenv("OPENAI_API_KEY"):
        return query

    try:
        from datetime import datetime
        client = _get_rewriter_client()
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = _REWRITE_PROMPT.format(query=query.strip(), today=today)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You rewrite user queries for web search. Output only the rewritten query as plain text, with no quotes or explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            # Reasoning models (Qwen3.5 etc.) consume budget on internal CoT
            # before emitting visible content, so be generous here.
            max_tokens=1500,
        )
        content = (response.choices[0].message.content or "").strip()
        # Strip wrapping quotes / single line.
        content = content.strip('"\'`').splitlines()[0].strip() if content else ""
        # Defensive: empty or absurdly long output → fall back.
        if not content or len(content) > 200 or content.lower() == query.strip().lower():
            return query
        _REWRITE_CACHE[query] = content
        return content
    except Exception:
        # Any failure (network, parse, etc.) → preserve the original query.
        return query

# Phrases that indicate the page didn't actually deliver content (paywall,
# JS-required app shell, captcha challenge, rate-limit page, etc.). When
# the *extracted* content is short and matches one of these, we treat the
# fetch as failed so the agent picks a different source instead of citing
# garbage as if it were an article. Keep these specific to avoid flagging
# real articles that mention "captcha" or "javascript" in passing.
_BLOCK_SENTINELS = (
    "you need to enable javascript",
    "javascript is required",
    "please enable javascript",
    "enable javascript to view",
    "checking your browser",
    "please verify you are a human",
    "are you a robot",
    "access denied",
    "403 forbidden",
    "rate limit exceeded",
    "too many requests",
    "cloudflare",  # paired with shortness check below
    "captcha",
)
# Only flag content as "blocked" if it's short — long pages that
# legitimately mention these phrases in body text shouldn't trip the check.
_BLOCK_MAX_CHARS = 1500


def _looks_blocked(content: str) -> Optional[str]:
    """If the extracted content matches a known block/JS-required sentinel
    AND is short, return the matched phrase. Else None."""
    if not content or len(content) > _BLOCK_MAX_CHARS:
        return None
    lower = content.lower()
    for phrase in _BLOCK_SENTINELS:
        if phrase in lower:
            return phrase
    return None


def _deny_hosts() -> List[str]:
    raw = os.getenv("AGENT_BROWSER_DENY_HOSTS", "")
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _is_denied(url: str) -> bool:
    blocked = _deny_hosts()
    if not blocked:
        return False
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not host:
        return False
    # Exact match or subdomain match (foo.example.com matches "example.com").
    return any(host == b or host.endswith("." + b) for b in blocked)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def visit_url(url: str, max_chars: int = 4000, offset: int = 0) -> ToolResult:
    """Fetch a URL via HTTP, extract main content as Markdown, paginate.

    On first fetch the page is downloaded, boilerplate-stripped, and cached
    in `_URL_CACHE`. Subsequent calls (any `offset`) read from cache with
    no network. The slice `[offset:offset+max_chars]` is returned, with a
    truncation footer pointing at the next offset when there's more.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ToolResult(
            success=False, output="",
            error="URL must start with http:// or https://",
        )
    if _is_denied(url):
        return ToolResult(
            success=False, output="",
            error=f"URL host is in AGENT_BROWSER_DENY_HOSTS deny list: {url}",
        )
    if max_chars <= 0:
        return ToolResult(success=False, output="", error="max_chars must be > 0")
    if offset < 0:
        return ToolResult(success=False, output="", error="offset must be >= 0")

    key = _hash_url(url)
    if key in _URL_CACHE:
        full = _URL_CACHE[key]
    else:
        try:
            resp = requests.get(
                url, timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
            )
        except requests.RequestException as e:
            return ToolResult(success=False, output="", error=f"Fetch error: {e}")
        if resp.status_code >= 400:
            return ToolResult(
                success=False, output="",
                error=f"HTTP {resp.status_code} from {url}",
            )

        # Primary path: trafilatura extracts main content, drops boilerplate,
        # and emits Markdown so headings/code/lists survive truncation.
        extracted = trafilatura.extract(
            resp.text,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            no_fallback=False,
        )
        if not extracted:
            # Fallback: title + body text via BeautifulSoup. Pages without
            # extractable article-like content (single-page apps, image
            # galleries) still yield something useful.
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            body = soup.get_text(separator="\n", strip=True)
            extracted = f"# {title}\n\n{body}" if title else body
        full = extracted or "(no extractable content)"

        # If the page came back as a JS-required shell, paywall, captcha,
        # or rate-limit page, treat it as a fetch failure so the agent
        # tries a different source instead of citing the boilerplate.
        blocked = _looks_blocked(full)
        if blocked:
            return ToolResult(
                success=False, output="",
                error=(f"Page blocked or JS-required (matched '{blocked}'): {url}. "
                       f"Try a different source — this URL won't yield article content."),
            )
        _URL_CACHE[key] = full

    total = len(full)
    if offset >= total:
        return ToolResult(
            success=True,
            output=(f"[browser_visit] {url}\n\n"
                    f"(empty: offset={offset} >= total length {total})"),
        )

    slice_end = min(offset + max_chars, total)
    chunk = full[offset:slice_end]
    output = f"[browser_visit] {url}\n\n{chunk}"
    if slice_end < total:
        output += (
            f"\n\n--- [truncated; {total} chars total. "
            f"Use offset={slice_end} to continue reading.] ---"
        )
    return ToolResult(success=True, output=output)


def search_web(query: str, max_results: int = 5) -> ToolResult:
    """Search the web via DuckDuckGo HTML; return top-N {title, url, snippet}.

    Before sending, the query is passed through `rewrite_query` to add date
    qualifiers, site filters, and concrete keywords (no-op when disabled or
    when the query already contains explicit operators).

    Uses the no-API-key HTML endpoint. Pair with `visit_url` on the most
    relevant URL returned. Free but flaky — if DuckDuckGo serves a captcha
    page or rate-limits, this returns an empty/error result.
    """
    if not query.strip():
        return ToolResult(success=False, output="", error="Empty search query.")
    if max_results <= 0 or max_results > 20:
        return ToolResult(success=False, output="", error="max_results must be between 1 and 20.")

    original_query = query.strip()
    effective_query = rewrite_query(original_query)

    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": effective_query, "kl": "us-en"},
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return ToolResult(success=False, output="", error=f"Search error: {e}")
    if resp.status_code >= 400:
        return ToolResult(
            success=False, output="",
            error=f"DuckDuckGo returned HTTP {resp.status_code}",
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for r in soup.select(".result, .web-result")[:max_results]:
        title_el = r.select_one(".result__title, .result__a")
        url_el = r.select_one(".result__url")
        snippet_el = r.select_one(".result__snippet")
        title = title_el.get_text(strip=True) if title_el else ""
        url_text = url_el.get_text(strip=True) if url_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        # DuckDuckGo wraps result links via a /l/?uddg=<encoded URL> redirector;
        # extract the underlying URL when present, otherwise fall back to the
        # cleaner displayed url text.
        href = ""
        anchor = r.select_one("a.result__a")
        if anchor and anchor.get("href"):
            href = anchor["href"]
            if "uddg=" in href:
                qs = urllib.parse.urlparse(href).query
                inner = urllib.parse.parse_qs(qs).get("uddg", [""])[0]
                if inner:
                    href = urllib.parse.unquote(inner)

        if title or href or url_text:
            results.append({
                "title": title,
                "url": href or url_text,
                "snippet": snippet,
            })

    rewrite_note = (
        f"\n[rewritten as: {effective_query}]"
        if effective_query != original_query else ""
    )

    if not results:
        return ToolResult(
            success=True,
            output=(f"[web_search] {original_query}{rewrite_note}\n\nNo results parsed — "
                    f"DuckDuckGo may have returned a captcha or changed its HTML."),
        )

    lines = [f"[web_search] {original_query}{rewrite_note}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r['snippet']:
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return ToolResult(success=True, output="\n".join(lines).rstrip())
