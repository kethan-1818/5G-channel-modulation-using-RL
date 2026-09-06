#!/usr/bin/env python3
"""Designed on-the-road DevOps infographic (preview before PPT)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

ROOT = Path(__file__).parent
LOGOS = ROOT / "assets" / "logos"
OUT = ROOT / "assets" / "devops_roadmap_infographic.png"

W, H = 3600, 2025

BG = (236, 240, 245, 255)
INK = (30, 38, 50, 255)
MUTED = (96, 108, 122, 255)
WHITE = (255, 255, 255, 255)
ASPHALT = (58, 66, 78, 255)
EDGE = (40, 46, 56, 255)
DASH = (255, 220, 80, 240)  # highway yellow


def font(size, bold=False):
    p = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(p, size)


def load_logo(name, size):
    path = LOGOS / f"{name}.png"
    if not path.exists():
        return None
    return Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)


def catmull(points, n_per=80):
    """Smooth path through waypoints."""
    pts = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(n_per):
            t = j / n_per
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    out.append(points[-1])
    return out


def draw_road(canvas, pts, width=118):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    re, r = width // 2 + 8, width // 2
    for x, y in pts[::2]:
        d.ellipse((x - re, y - re, x + re, y + re), fill=EDGE)
    for x, y in pts[::2]:
        d.ellipse((x - r, y - r, x + r, y + r), fill=ASPHALT)
    acc, on, last = 0.0, True, pts[0]
    for p in pts[4::4]:
        dist = math.hypot(p[0] - last[0], p[1] - last[1])
        if on:
            d.line([last, p], fill=DASH, width=7)
        acc += dist
        if acc >= 28:
            acc, on = 0.0, not on
        last = p
    canvas.alpha_composite(layer)


def card_shadow(canvas, box):
    x0, y0, x1, y1 = box
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0 + 5, y0 + 12, x1 + 5, y1 + 12), radius=20, fill=(18, 26, 38, 40))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(9)))


# Designed rest-stop coordinates (x, y) on the highway
# Gentle valley in the middle, even spacing, room for cards above/below
PINS = [
    (260, 860),
    (560, 780),
    (860, 720),
    (1160, 780),
    (1460, 920),
    (1760, 1080),
    (2060, 1120),
    (2360, 1000),
    (2660, 860),
    (2960, 760),
    (3280, 720),
]

# side: "up" or "down"
STOPS = [
    ("01", "Foundations", "Linux, Bash, Git Basics", ["linux", "bash", "git"], (245, 130, 32), "up"),
    ("02", "Source Control", "Git Workflows & PRs", ["git", "github"], (220, 68, 68), "down"),
    ("03", "Containerization", "Docker & Images", ["docker"], (124, 92, 212), "up"),
    ("04", "CI/CD Pipelines", "Jenkins / GitHub Actions", ["jenkins", "githubactions"], (232, 145, 48), "down"),
    ("05", "Cloud Platforms", "EC2 · VPC · IAM · S3 · ECS/EKS", ["amazonaws", "googlecloud", "microsoftazure"], (56, 126, 214), "up"),
    ("06", "Infrastructure as Code", "Terraform / CloudFormation", ["terraform", "cloudformation"], (245, 130, 32), "down"),
    ("07", "Config & Automation", "Ansible / SSM", ["ansible"], (36, 72, 140), "up"),
    ("08", "Kubernetes", "K8s & Helm", ["kubernetes", "helm"], (56, 126, 214), "down"),
    ("09", "Security", "IAM & Scanning", ["security"], (220, 68, 68), "up"),
    ("10", "Observability", "Dashboards & Logging", ["prometheus", "grafana"], (245, 130, 32), "down"),
    ("11", "GitOps", "ArgoCD / Flux", ["argocd", "flux"], (46, 160, 90), "up"),
]


def main():
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    for y in range(0, H, 34):
        draw.line((0, y, W, y), fill=(226, 230, 236, 255), width=1)

    draw.text((80, 32), "DevOps Roadmap: Beginner’s Guide to Cloud Automation",
              font=font(50, True), fill=INK)
    draw.text((80, 100), "Friday Deployment Club   ·   Day 2   ·   Screenshot this map",
              font=font(22, False), fill=MUTED)

    # extend road a bit past first/last pin
    waypoints = [(80, 920)] + PINS + [(3520, 700)]
    path = catmull(waypoints, n_per=70)
    draw_road(canvas, path, width=120)

    CW, CH = 318, 176
    GAP = 78  # air between card and pin (road edge ~60px)

    # connectors + cards
    for (px, py), stop in zip(PINS, STOPS):
        num, title, subtitle, logos, color, side = stop
        if side == "up":
            cx, cy = px - CW / 2, py - GAP - CH
        else:
            cx, cy = px - CW / 2, py + GAP
        cx = max(28, min(W - CW - 28, cx))

        # short stem from pin to card (does not cross other cards)
        if side == "up":
            x1, y1 = px, cy + CH
        else:
            x1, y1 = px, cy
        draw.line((px, py, x1, y1), fill=color + (200,), width=5)

        card_shadow(canvas, (cx, cy, cx + CW, cy + CH))
        draw.rounded_rectangle((cx, cy, cx + CW, cy + CH), radius=18, fill=WHITE,
                               outline=(222, 228, 236, 255), width=2)
        tab_w = min(228, CW - 20)
        draw.rounded_rectangle((cx + 14, cy - 14, cx + 14 + tab_w, cy + 24), radius=11, fill=color + (255,))
        draw.text((cx + 24, cy - 8), title, font=font(17, True), fill=WHITE)
        draw.text((cx + 20, cy + 42), subtitle, font=font(17, True), fill=INK)

        n = len(logos)
        ls = 54
        total = n * ls + max(n - 1, 0) * 14
        sx = cx + (CW - total) / 2
        ly = cy + CH - 70
        for k, name in enumerate(logos):
            logo = load_logo(name, ls)
            if logo:
                canvas.alpha_composite(logo, (int(sx + k * (ls + 14)), int(ly)))

        # pin
        r = 24
        draw.ellipse((px - r - 5, py - r - 5, px + r + 5, py + r + 5), fill=WHITE)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color + (255,))
        f = font(17, True)
        bb = draw.textbbox((0, 0), num, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((px - tw / 2, py - th / 2 - 1), num, font=f, fill=WHITE)

    # START plaque to the left of pin 01
    sx, sy = PINS[0]
    draw.rounded_rectangle((sx - 168, sy - 22, sx - 48, sy + 22), radius=11, fill=(46, 168, 88, 255))
    draw.text((sx - 150, sy - 14), "START", font=font(16, True), fill=WHITE)

    # finish flag
    fx, fy = PINS[-1]
    flag = Image.new("RGBA", (90, 60), (0, 0, 0, 0))
    fd = ImageDraw.Draw(flag)
    cell = 15
    for i in range(6):
        for j in range(4):
            c = (22, 22, 22, 255) if (i + j) % 2 == 0 else (250, 250, 250, 255)
            fd.rectangle((i * cell, j * cell, (i + 1) * cell, (j + 1) * cell), fill=c)
    canvas.alpha_composite(flag, (int(fx + 40), int(fy - 86)))
    draw.rectangle((fx + 34, fy - 90, fx + 40, fy + 70), fill=(46, 168, 88, 255))

    draw.rounded_rectangle((70, 1918, 2860, 1994), radius=16, fill=(48, 56, 68, 255))
    draw.text((100, 1936), "Build, Automate, Monitor & Deploy.", font=font(28, True), fill=WHITE)
    draw.text((3000, 1946), "Kethan Gummalla", font=font(20, False), fill=MUTED)

    canvas.convert("RGB").save(OUT, "PNG", optimize=True)
    print(f"Wrote {OUT} {canvas.size}")


if __name__ == "__main__":
    main()
