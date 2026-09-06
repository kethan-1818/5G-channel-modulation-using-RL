#!/usr/bin/env python3
"""Generate a widescreen DevOps roadmap infographic (winding-road style)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

ROOT = Path(__file__).parent
LOGOS = ROOT / "assets" / "logos"
OUT = ROOT / "assets" / "devops_roadmap_infographic.png"

W, H = 2400, 1350
BG = (236, 239, 243, 255)
ROAD = (55, 62, 72, 255)
ROAD_EDGE = (40, 45, 52, 255)
DASH = (255, 255, 255, 230)
CARD = (255, 255, 255, 245)
SHADOW = (20, 25, 35, 55)
INK = (28, 35, 48, 255)
MUTED = (90, 100, 115, 255)
BANNER = (55, 62, 72, 255)

HEADER_COLORS = {
    "orange": (245, 130, 32),
    "red": (220, 68, 68),
    "purple": (124, 92, 212),
    "amber": (232, 145, 48),
    "blue": (56, 126, 214),
    "navy": (36, 72, 140),
    "green": (46, 160, 90),
    "teal": (32, 150, 160),
    "olive": (120, 140, 55),
}


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_logo(name, size):
    p = LOGOS / f"{name}.png"
    if not p.exists():
        return None
    im = Image.open(p).convert("RGBA")
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im


def rounded_rect(draw, box, r, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def bezier(p0, p1, p2, p3, n=400):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def point_on_path(pts, t):
    i = min(int(t * (len(pts) - 1)), len(pts) - 1)
    return pts[i]


def draw_road(base, pts, width=78):
    """Stamp overlapping circles along the path to form a smooth road."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = width // 2
    for x, y in pts[::2]:
        d.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=ROAD_EDGE)
    for x, y in pts[::2]:
        d.ellipse((x - r, y - r, x + r, y + r), fill=ROAD)
    # dashed center line
    dash_on = True
    acc = 0
    last = pts[0]
    for p in pts[4::4]:
        dx, dy = p[0] - last[0], p[1] - last[1]
        dist = math.hypot(dx, dy)
        acc += dist
        if dash_on and acc < 22:
            d.line([last, p], fill=DASH, width=5)
        if acc >= 22:
            acc = 0
            dash_on = not dash_on
        last = p
    base.alpha_composite(layer)


def pin(draw, x, y, color):
    r = 14
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (255,), outline=(255, 255, 255, 255), width=4)


def paste_logo(canvas, name, xy, size):
    logo = load_logo(name, size)
    if logo:
        canvas.alpha_composite(logo, dest=(int(xy[0]), int(xy[1])))
        return True
    return False


def card(canvas, draw, x, y, w, h, title, lines, header, logos, pin_xy=None):
    # shadow
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    rounded_rect(sd, (x + 6, y + 10, x + w + 6, y + h + 10), 18, SHADOW)
    sh = sh.filter(ImageFilter.GaussianBlur(8))
    canvas.alpha_composite(sh)

    rounded_rect(draw, (x, y, x + w, y + h), 18, CARD, outline=(220, 224, 230, 255), width=2)
    # colored header tab
    tab_w = min(210, w - 24)
    draw.rounded_rectangle((x + 14, y - 10, x + 14 + tab_w, y + 28), radius=10, fill=header + (255,))
    tw = font(18, True)
    draw.text((x + 24, y - 5), title, font=tw, fill=(255, 255, 255, 255))

    body = font(20, True)
    small = font(16, False)
    ty = y + 42
    for i, line in enumerate(lines):
        draw.text((x + 20, ty), line, font=body if i == 0 else small, fill=INK if i == 0 else MUTED)
        ty += 28 if i == 0 else 24

    lx = x + 18
    ly = y + h - 62
    for name in logos:
        paste_logo(canvas, name, (lx, ly), 48)
        lx += 56


def checkered_flag(canvas, x, y):
    flag = Image.new("RGBA", (70, 48), (0, 0, 0, 0))
    d = ImageDraw.Draw(flag)
    cell = 12
    for i in range(6):
        for j in range(4):
            c = (20, 20, 20, 255) if (i + j) % 2 == 0 else (245, 245, 245, 255)
            d.rectangle((i * cell, j * cell, (i + 1) * cell, (j + 1) * cell), fill=c)
    canvas.alpha_composite(flag, dest=(x, y))
    pole = ImageDraw.Draw(canvas)
    pole.rectangle((x - 8, y - 8, x - 2, y + 90), fill=(40, 160, 80, 255))


def main():
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # subtle horizontal paper lines
    for y in range(0, H, 28):
        draw.line((0, y, W, y), fill=(228, 231, 236, 255), width=1)

    # Title
    title_f = font(42, True)
    sub_f = font(20, False)
    title = "DevOps Roadmap: Beginner’s Guide to Cloud Automation"
    draw.text((70, 36), title, font=title_f, fill=INK)
    draw.text((70, 92), "Friday Deployment Club  ·  Day 2  ·  Screenshot this map", font=sub_f, fill=MUTED)

    # S-curve road through the canvas
    pts = bezier((180, 210), (900, 180), (1500, 280), (1680, 420), n=280)
    pts += bezier((1680, 420), (1850, 560), (1400, 640), (720, 700), n=280)[1:]
    pts += bezier((720, 700), (180, 760), (280, 980), (1980, 1080), n=320)[1:]
    draw_road(canvas, pts, width=86)

    # Stop positions along the road (t in 0..1)
    stops = [
        dict(t=0.04, side="left", title="Foundations", lines=["Linux, Bash, Git Basics"],
             header=HEADER_COLORS["orange"], logos=["linux", "bash", "git"]),
        dict(t=0.13, side="right", title="Source Control", lines=["Git Workflows & PRs"],
             header=HEADER_COLORS["red"], logos=["git", "github"]),
        dict(t=0.24, side="left", title="Containerization", lines=["Docker & Images"],
             header=HEADER_COLORS["purple"], logos=["docker"]),
        dict(t=0.34, side="right", title="CI/CD Pipelines", lines=["Jenkins / GitHub Actions"],
             header=HEADER_COLORS["amber"], logos=["jenkins", "githubactions"]),
        dict(t=0.45, side="left", title="Cloud Platforms", lines=["AWS Core Services", "EC2 · VPC · IAM · S3 · ALB", "ECS / EKS Basics"],
             header=HEADER_COLORS["blue"], logos=["amazonaws", "googlecloud", "microsoftazure"], w=380, h=200),
        dict(t=0.55, side="right", title="Infrastructure as Code", lines=["Terraform / IaC", "CloudFormation"],
             header=HEADER_COLORS["orange"], logos=["terraform", "cloudformation"], w=340),
        dict(t=0.66, side="left", title="Config & Automation", lines=["Ansible / SSM"],
             header=HEADER_COLORS["navy"], logos=["ansible"]),
        dict(t=0.76, side="right", title="Kubernetes", lines=["K8s & Helm"],
             header=HEADER_COLORS["blue"], logos=["kubernetes", "helm"]),
        dict(t=0.84, side="left", title="Security", lines=["IAM & Scanning"],
             header=HEADER_COLORS["red"], logos=["security"]),
        dict(t=0.90, side="right", title="Observability", lines=["Dashboards & Logging"],
             header=HEADER_COLORS["amber"], logos=["prometheus", "grafana"]),
        dict(t=0.97, side="right", title="GitOps", lines=["ArgoCD / Flux"],
             header=HEADER_COLORS["olive"], logos=["argocd", "flux", "git"]),
    ]

    draw2 = ImageDraw.Draw(canvas)
    for s in stops:
        px, py = point_on_path(pts, s["t"])
        pin(draw2, px, py, s["header"])
        cw = s.get("w", 310)
        ch = s.get("h", 130)
        if s["side"] == "left":
            cx = max(40, px - cw - 70)
        else:
            cx = min(W - cw - 40, px + 70)
        cy = max(130, min(H - 160, py - ch / 2))
        # connector
        draw2.line((px, py, cx + (cw if s["side"] == "left" else 0), cy + 20), fill=s["header"] + (200,), width=3)
        card(canvas, draw2, cx, cy, cw, ch, s["title"], s["lines"], s["header"], s["logos"])

    # finish flag near end of road
    fx, fy = point_on_path(pts, 0.995)
    checkered_flag(canvas, int(fx) - 10, int(fy) - 70)

    # footer banner
    draw.rounded_rectangle((40, 1248, 1960, 1320), radius=16, fill=BANNER)
    draw.text((80, 1264), "Build, Automate, Monitor & Deploy.", font=font(28, True), fill=(255, 255, 255, 255))
    draw.text((1980, 1280), "Kethan Gummalla", font=font(16, False), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT, "PNG", quality=95)
    print(f"Wrote {OUT} {canvas.size}")


if __name__ == "__main__":
    main()
