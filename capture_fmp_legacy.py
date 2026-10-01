"""
Capture the FMP LEGACY API docs as a single self-contained, offline .html file.

The legacy docs (https://site.financialmodelingprep.com/developer/docs/legacy)
are one long page navigated by #anchors, GATED behind your FMP login. FMP's
anti-bot blocks Playwright-LAUNCHED browsers (the login form won't even render).

So instead this launches your REAL Chrome with only a remote-debugging port --
no automation flags, so login renders normally -- you sign in by hand, and
Playwright merely ATTACHES to that Chrome over CDP to capture the page. The
debug profile (.chrome_debug/) persists your login, so re-runs skip sign-in.

Usage (run yourself in a terminal -- it opens a visible Chrome window):
    python capture_fmp_legacy.py

Requires: pip install playwright   (Google Chrome must be installed)
"""

import base64
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

URL = "https://site.financialmodelingprep.com/developer/docs/legacy"
HERE = Path(__file__).resolve().parent
DEBUG_PROFILE = HERE / ".chrome_debug"
OUT = HERE / "fmp_legacy_docs.html"
PORT = 9222

resources: dict[str, tuple[str, bytes]] = {}  # url -> (content_type, raw_bytes)


def find_chrome() -> str | None:
    cands = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def port_up() -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/json/version", timeout=1)
        return True
    except Exception:
        return False


def data_uri(content_type: str, raw: bytes) -> str:
    ct = (content_type or "application/octet-stream").split(";")[0].strip()
    return f"data:{ct};base64,{base64.b64encode(raw).decode('ascii')}"


def looks_logged_in(page) -> bool:
    try:
        txt = page.evaluate("document.body.innerText") or ""
    except Exception:
        return False
    if "can't find that page" in txt.lower() or "404" in txt[:200]:
        return False
    return len(txt) > 3000


def autoscroll(page):
    prev = -1
    for _ in range(60):
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(350)
        h = page.evaluate("document.body.scrollHeight")
        if h == prev:
            break
        prev = h
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(800)


def inline_css(css_text: str, css_url: str) -> str:
    def repl(m):
        ref = m.group(1).strip().strip("'\"")
        if ref.startswith("data:"):
            return m.group(0)
        abs_ref = urljoin(css_url, ref)
        if abs_ref in resources:
            ct, raw = resources[abs_ref]
            return f"url({data_uri(ct, raw)})"
        return m.group(0)
    return re.sub(r"url\(([^)]+)\)", repl, css_text)


def inline_html(html: str) -> str:
    def repl_link(m):
        href = urljoin(URL, m.group(1))
        if href in resources:
            ct, raw = resources[href]
            return f"<style>{inline_css(raw.decode('utf-8', 'ignore'), href)}</style>"
        return m.group(0)
    html = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\'][^>]*>',
                  repl_link, html, flags=re.I)
    html = re.sub(r'<link[^>]+href=["\']([^"\']+)["\'][^>]*rel=["\']stylesheet["\'][^>]*>',
                  repl_link, html, flags=re.I)

    def repl_img(m):
        src = urljoin(URL, m.group(2))
        if src in resources:
            ct, raw = resources[src]
            return f'{m.group(1)}="{data_uri(ct, raw)}"'
        return m.group(0)
    html = re.sub(r'(src|href)=["\']([^"\']+\.(?:png|jpe?g|gif|svg|webp|ico)[^"\']*)["\']',
                  repl_img, html, flags=re.I)

    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.I | re.S)
    html = re.sub(r"<script\b[^>]*/>", "", html, flags=re.I)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f"<!-- Offline snapshot of {URL} captured {stamp} -->\n"
            + html.replace("<head>", f'<head>\n<base href="{URL}">', 1))


def ensure_chrome_running():
    if port_up():
        print(f"Found a Chrome already listening on debug port {PORT}.")
        return
    chrome = find_chrome()
    if not chrome:
        print("Could not find Google Chrome. Launch it yourself with:\n"
              f'  chrome.exe --remote-debugging-port={PORT} '
              f'--user-data-dir="{DEBUG_PROFILE}" "{URL}"\n'
              "then re-run this script.", file=sys.stderr)
        sys.exit(1)
    print("Launching your real Chrome with a debug port (no automation flags)...")
    subprocess.Popen([
        chrome,
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={DEBUG_PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        URL,
    ])
    for _ in range(30):
        if port_up():
            return
        time.sleep(0.5)
    print("Chrome did not expose the debug port in time.", file=sys.stderr)
    sys.exit(1)


def get_docs_page(browser):
    """Find (or create) a tab and return it, navigated nowhere yet."""
    ctxs = browser.contexts
    ctx = ctxs[0] if ctxs else browser.new_context()
    pages = ctx.pages
    for pg in pages:
        if "financialmodelingprep.com" in (pg.url or ""):
            return ctx, pg
    return ctx, (pages[0] if pages else ctx.new_page())


def main():
    ensure_chrome_running()

    print("\n" + "=" * 70)
    print("  In the Chrome window that opened:")
    print("    1. Sign in to FMP (the login page renders normally here)")
    print("    2. You don't need to do anything else -- this script will")
    print("       detect the authenticated docs page and capture it.")
    print("  Waiting (up to 10 minutes)...")
    print("=" * 70 + "\n", flush=True)

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{PORT}")
        ctx, page = get_docs_page(browser)

        # Poll until the authenticated legacy docs render.
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if "/developer/docs/legacy" not in (page.url or ""):
                    if looks_logged_in(page):  # logged in elsewhere -> go to docs
                        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
                        page.wait_for_timeout(3000)
                if "/developer/docs/legacy" in (page.url or "") and looks_logged_in(page):
                    break
            except Exception:
                pass
            time.sleep(3)
            print(f"  ...waiting for sign-in ({int(deadline - time.time())}s left)", flush=True)
        else:
            print("\nTimed out. Finish signing in and re-run "
                  "(your login is saved in .chrome_debug).", file=sys.stderr)
            sys.exit(1)

        print("Authenticated. Reloading to capture all resources...")
        page.on("response", lambda r: _grab(r))
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception:
            pass  # SPA may keep connections open; the DOM is what matters
        page.wait_for_timeout(5000)
        autoscroll(page)
        html = page.content()
        # do NOT close the browser -- it's the user's Chrome.

    print(f"Captured {len(resources)} resources. Inlining...")
    OUT.write_text(inline_html(html), encoding="utf-8")
    mb = OUT.stat().st_size / 1_048_576
    print(f"\nSaved: {OUT}  ({mb:.1f} MB)")
    print("Open it in any browser, fully offline. You can close the Chrome window.")


def _grab(resp):
    try:
        ct = resp.headers.get("content-type", "")
        if any(k in ct for k in ("css", "image", "font", "woff", "svg")):
            resources[resp.url] = (ct, resp.body())
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
