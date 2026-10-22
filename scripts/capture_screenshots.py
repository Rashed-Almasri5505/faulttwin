"""Screenshots and the demo GIF for the README and the Devpost gallery (docs/images/), from the running app.

Start the app without keys first (recorded diagnoses only, no API cost):
    FAULTTWIN_OFFLINE=1 streamlit run app.py --server.port 8620
then:
    python scripts/capture_screenshots.py [--url http://localhost:8620]

Drives a headless Microsoft Edge or Google Chrome (already installed; a throwaway profile is used) through the
DevTools protocol, opens deep links such as ?case=AU_open&at=30, and crops the parts of the page it needs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/microsoft-edge"]
PORT = 9333
WIDTH = 1440
LOAD_S = 7.0          # page load and first render before a capture


class Page:
    def __init__(self, ws: websockets.ClientConnection) -> None:
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params) -> dict:
        self.n += 1
        my = self.n
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def shot(self, url: str, height: int, wait_s: float = LOAD_S) -> Image.Image:
        await self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height, deviceScaleFactor=1,
                        mobile=False)
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait_s)
        return await self.capture()

    async def capture(self) -> Image.Image:
        data = (await self.call("Page.captureScreenshot", format="png"))["data"]
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    async def box(self, js: str, pad: int = 8) -> tuple[int, int, int, int]:
        """The bounding box of the element a JavaScript expression returns (viewport pixels)."""
        res = await self.call("Runtime.evaluate", returnByValue=True, expression=(
            f"(() => {{ const e = {js}; if (!e) return null; const r = e.getBoundingClientRect();"
            " return [r.left, r.top, r.right, r.bottom]; })()"))
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JavaScript error: {res['exceptionDetails'].get('exception', {}).get('description')}")
        v = res.get("result", {}).get("value")
        if not v:
            raise RuntimeError(f"element not found: {js}")
        return (max(0, int(v[0]) - pad), max(0, int(v[1]) - pad), int(v[2]) + pad, int(v[3]) + pad)


def by_text(text: str, container: str) -> str:
    """JavaScript: the nearest `container` around the first element whose text contains `text`."""
    t = json.dumps(text)                    # headings first: the status strip repeats the diagnosis in a <p>
    return (f"([...document.querySelectorAll('h4')].find(e => e.textContent.includes({t})) ?? "
            f"[...document.querySelectorAll('th, p, summary')].find(e => e.textContent.includes({t})))"
            f"?.closest({json.dumps(container)})")


BLOCK = "[data-testid='stVerticalBlock']"          # a bordered container's own block
EXPANDER = "[data-testid='stExpander']"


CHARTS = "document.querySelectorAll('[data-testid=\"stPlotlyChart\"]')"


def caption_bar(img: Image.Image, text: str) -> Image.Image:
    """A plain title bar above an image (gallery images are seen without their page)."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    for size in range(30, 15, -1):                 # the largest bold size that fits the image width
        try:
            font = ImageFont.truetype("segoeuib.ttf", size)
        except OSError:
            break
        if font.getlength(text) <= img.width - 48:
            break
    out = Image.new("RGB", (img.width, img.height + 64), "#FFFFFF")
    ImageDraw.Draw(out).text((24, 14), text, fill="#1F2933", font=font)
    out.paste(img, (0, 64))
    return out


def side_by_side(left: Image.Image, right: Image.Image, height: int) -> Image.Image:
    left = left.resize((round(left.width * height / left.height), height), Image.LANCZOS)
    right = right.resize((round(right.width * height / right.height), height), Image.LANCZOS)
    out = Image.new("RGB", (left.width + right.width + 24, height), "#FFFFFF")
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 24, 0))
    return out


def browser() -> str:
    path = next((p for p in BROWSERS if Path(p).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not path:
        raise SystemExit("No Edge or Chrome found for headless screenshots.")
    return path


def save(img: Image.Image, name: str, box: tuple[int, int, int, int] | None = None, width: int | None = None) -> None:
    if box:
        img = img.crop(box)
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(OUT / name, optimize=True)
    print(f"{name}: {img.width}x{img.height}")


async def main(base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="faulttwin_shots_")
    proc = subprocess.Popen([browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--force-prefers-reduced-motion", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=1).read())
                target = next(p for p in pages if p.get("type") == "page")
                break
            except Exception:  # noqa: BLE001 - the browser is still starting
                time.sleep(0.2)
        else:
            raise SystemExit("The headless browser did not start.")
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=64 * 2 ** 20) as ws:
            page = Page(ws)
            await page.call("Page.enable")
            await page.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])

            # 1. The live plant, healthy (no recording)
            save(await page.shot(f"{base}/", 900), "01_live_healthy.png")
            # 2. AU_open mid-investigation: deformed Park's vector, the agent at work
            mid = await page.shot(f"{base}/?case=AU_open&at={22 - LOAD_S}", 1900)
            save(mid, "02_au_open_investigating.png", (0, 0, WIDTH, 1900))
            # 3. AU_open finished: final card, ranking, interlock, overlays, sources, work order
            end = await page.shot(f"{base}/?case=AU_open&at=200", 5200)
            save(end, "03_au_open_diagnosis.png", (0, 0, WIDTH, 3600))
            card = end.crop(await page.box(by_text("Upper switch of leg A open", BLOCK)))
            charts = [await page.box(f"{CHARTS}[{k}]") for k in range(3)]
            plant = end.crop((charts[0][0], charts[0][1], charts[2][2], max(c[3] for c in charts)))
            park = end.crop(charts[1])
            sources = end.crop(await page.box(by_text("Sources (via Tavily)", EXPANDER)))
            order = end.crop(await page.box(by_text("Technician work order", EXPANDER)))
            save(order, "07_work_order.png")
            # 4. The look-alike: a 3 A sensor offset the textbook detector calls an open switch
            lk = await page.shot(f"{base}/?case=sensor_offset_B_3A&at=200", 3000)
            save(lk, "04_sensor_offset_lookalike.png", (0, 0, WIDTH, 3000))
            lk_card = lk.crop(await page.box(by_text("Current sensor of phase B", BLOCK)))
            lk_monitor = lk.crop(await page.box(by_text("Edge monitor", BLOCK), pad=0))
            lk_park = lk.crop(await page.box(f"{CHARTS}[1]"))
            # 5. How it works and 6. Results
            save(await page.shot(f"{base}/?view=how", 2400), "05_how_it_works.png")
            res = await page.shot(f"{base}/?view=results", 3600)
            save(res, "06_results.png")
            summary = res.crop(await page.box("[...document.querySelectorAll('th')].find(e => e.textContent.trim() === "
                                              "'Method')?.closest('table')", pad=12))

            # Devpost gallery (5 images; the first is the thumbnail)
            save(caption_bar(side_by_side(park, card, 640), "Open switch AU: Park's vector deforms; the agent's checked "
                             "diagnosis"), "gallery_1_thumbnail.png", width=1600)
            save(caption_bar(mid.crop((0, 0, WIDTH, 1800)), "The live plant and the agent at work (recorded run)"),
                 "gallery_2_agent_at_work.png", width=1440)
            top = side_by_side(lk_park, lk_card, 600)
            monitor = lk_monitor.resize((top.width, round(lk_monitor.height * top.width / lk_monitor.width)),
                                        Image.LANCZOS)
            look = Image.new("RGB", (top.width, top.height + monitor.height + 16), "#FFFFFF")
            look.paste(top, (0, 0))
            look.paste(monitor, (0, top.height + 16))
            save(caption_bar(look, "A 3 A sensor offset: the textbook detector says BU_open; FaultTwin says sensor "
                             "offset"), "gallery_3_lookalike.png", width=1600)
            save(caption_bar(side_by_side(sources, order, 900), "Tavily evidence and the Ultra work order"),
                 "gallery_4_tavily_and_work_order.png", width=1600)
            cm = Image.open(ROOT / "eval" / "results" / "confusion_matrix.png").convert("RGB")
            stacked = Image.new("RGB", (max(summary.width, 1400), summary.height + 560), "#FFFFFF")
            stacked.paste(summary, (0, 0))
            cm_small = cm.resize((1400, round(cm.height * 1400 / cm.width)), Image.LANCZOS)
            stacked.paste(cm_small, (0, summary.height + 20))
            save(caption_bar(stacked, "Evaluation on 24 fixed-seed scenarios (every run reported)"),
                 "gallery_5_results.png", width=1600)
            save(plant, "plant_row.png")
            # GIF: the AU_open replay from the healthy lead-in to the diagnosis, plant row and status
            frames = []
            for t in [0, 2, 4, 6, 9, 12, 15, 18, 22, 26, 30, 34, 38, 42]:
                img = await page.shot(f"{base}/?case=AU_open&at={max(0.0, t - LOAD_S)}", 900)
                frames.append(img.resize((960, 600), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE))
                print(f"frame at {t} s")
            frames[0].save(OUT / "faulttwin_demo.gif", save_all=True, append_images=frames[1:], duration=700, loop=0,
                           optimize=True)
            print("faulttwin_demo.gif:", len(frames), "frames")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8620")
    asyncio.run(main(ap.parse_args().url.rstrip("/")))
    sys.exit(0)
