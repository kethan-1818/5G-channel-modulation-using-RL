#!/usr/bin/env python3
"""Clean 16:9 DevOps roadmap infographic — equal cards, numbered path."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).parent
LOGOS = ROOT / "assets" / "logos"
OUT = ROOT / "assets" / "devops_roadmap_infographic.png"

W, H = 3200, 1800

BG = (246, 248, 252, 255)
WHITE = (255, 255, 255, 255)
INK = (20, 30, 46, 255)
MUTED = (96, 110, 126, 255)
LINE = (226, 232, 240, 255)
FOOTER = (20, 30, 46, 255)

# Phase accents (match legend)
P1 = (245, 130, 32)    # Foundations
P2 = (124, 92, 212)    # Containers & CI/CD
P3 = (56, 126, 214)    # Cloud
P4 = (79, 70, 180)     # Infra
P5 = (34, 160, 98)     # Security & Monitoring
GOAL = (236, 72, 153)


def font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_logo(name, size):
    p = LOGOS / f"{name}.png"
    if not p.exists():
        return None
    return Image.open(p).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)


def shadow(canvas, box, radius=20):
    x0, y0, x1, y1 = box
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius=radius, fill=(15, 25, 40, 28))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(10)))


STOPS = [
    (0, 0, "01", "Foundations", "Linux · Bash · Git", ["linux", "bash", "git"], P1),
    (0, 1, "02", "Source Control", "Branches · PRs · Reviews", ["git", "github"], P1),
    (0, 2, "03", "Containerization", "Docker & Images", ["docker"], P2),
    (0, 3, "04", "CI / CD", "Jenkins · GitHub Actions", ["jenkins", "githubactions"], P2),
    (1, 0, "05", "Cloud Platforms", "EC2 · VPC · IAM · S3 · ECS / EKS", ["amazonaws", "googlecloud", "microsoftazure"], P3),
    (1, 1, "06", "Infrastructure as Code", "Terraform · CloudFormation", ["terraform", "cloudformation"], P4),
    (1, 2, "07", "Config & Automation", "Ansible · AWS SSM", ["ansible"], P4),
    (1, 3, "08", "Kubernetes", "K8s · Helm", ["kubernetes", "helm"], P4),
    (2, 0, "09", "Security", "IAM · Scanning · Secrets", ["security"], P5),
    (2, 1, "10", "Observability", "Prometheus · Grafana · Logs", ["prometheus", "grafana"], P5),
    (2, 2, "11", "GitOps", "ArgoCD · Flux", ["argocd", "flux", "git"], P5),
]


def main():
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, W, 10), fill=(34, 211, 238, 255))

    draw.text((80, 32), "DevOps Roadmap", font=font(58, True), fill=INK)
    draw.text((80, 106), "Beginner’s Guide to Cloud Automation    ·    Friday Deployment Club    ·    Day 2",
              font=font(24, False), fill=MUTED)

    legend = [
        (P1, "Foundations"),
        (P2, "Containers & CI/CD"),
        (P3, "Cloud"),
        (P4, "Infra & Orchestration"),
        (P5, "Security & Monitoring"),
    ]
    lx = 80
    for color, label in legend:
        draw.rounded_rectangle((lx, 158, lx + 22, 180), radius=5, fill=color + (255,))
        draw.text((lx + 32, 154), label, font=font(20, True), fill=MUTED)
        lx += 430

    margin_x, top = 72, 210
    footer_h = 118
    gap_x, gap_y = 48, 56
    cols, rows = 4, 3
    usable_w = W - 2 * margin_x
    usable_h = H - top - footer_h
    cw = (usable_w - (cols - 1) * gap_x) / cols
    ch = (usable_h - (rows - 1) * gap_y) / rows

    def box(r, c):
        x = margin_x + c * (cw + gap_x)
        y = top + r * (ch + gap_y)
        return x, y, x + cw, y + ch

    # arrows in the gutters (not behind cards)
    def chevron(x, y, color, direction="right"):
        if direction == "right":
            draw.polygon([(x - 10, y - 12), (x + 12, y), (x - 10, y + 12)], fill=color + (220,))
        elif direction == "down":
            draw.polygon([(x - 12, y - 10), (x + 12, y - 10), (x, y + 12)], fill=color + (220,))

    # row 1 arrows
    for c in range(3):
        x0, y0, x1, y1 = box(0, c)
        nx0, _, _, _ = box(0, c + 1)
        mid_x = (x1 + nx0) / 2
        mid_y = (y0 + y1) / 2
        draw.line((x1 + 6, mid_y, nx0 - 8, mid_y), fill=(180, 190, 205, 255), width=5)
        chevron(nx0 - 10, mid_y, (140, 150, 165), "right")
    # down from 04 to 08 then... we'll go 04 → down-left visually as wrap to 05
    # wrap: from card 04 bottom to card 05 top via left margin
    a = box(0, 3)
    b = box(1, 0)
    draw.line((a[2] - cw / 2, a[3] + 6, a[2] - cw / 2, a[3] + gap_y / 2), fill=(180, 190, 205, 255), width=5)
    y_mid = a[3] + gap_y / 2
    draw.line((b[0] + cw / 2, y_mid, a[2] - cw / 2, y_mid), fill=(180, 190, 205, 255), width=5)
    draw.line((b[0] + cw / 2, y_mid, b[0] + cw / 2, b[1] - 8), fill=(180, 190, 205, 255), width=5)
    chevron(b[0] + cw / 2, b[1] - 10, (140, 150, 165), "down")

    for c in range(3):
        x0, y0, x1, y1 = box(1, c)
        nx0, _, _, _ = box(1, c + 1)
        mid_y = (y0 + y1) / 2
        draw.line((x1 + 6, mid_y, nx0 - 8, mid_y), fill=(180, 190, 205, 255), width=5)
        chevron(nx0 - 10, mid_y, (140, 150, 165), "right")

    a = box(1, 3)
    b = box(2, 0)
    y_mid = a[3] + gap_y / 2
    draw.line((a[2] - cw / 2, a[3] + 6, a[2] - cw / 2, y_mid), fill=(180, 190, 205, 255), width=5)
    draw.line((b[0] + cw / 2, y_mid, a[2] - cw / 2, y_mid), fill=(180, 190, 205, 255), width=5)
    draw.line((b[0] + cw / 2, y_mid, b[0] + cw / 2, b[1] - 8), fill=(180, 190, 205, 255), width=5)
    chevron(b[0] + cw / 2, b[1] - 10, (140, 150, 165), "down")

    for c in range(3):
        x0, y0, x1, y1 = box(2, c)
        nx0, _, _, _ = box(2, c + 1)
        mid_y = (y0 + y1) / 2
        draw.line((x1 + 6, mid_y, nx0 - 8, mid_y), fill=(180, 190, 205, 255), width=5)
        chevron(nx0 - 10, mid_y, (140, 150, 165), "right")

    logo_size = 64
    for r, c, num, title, subtitle, logos, accent in STOPS:
        x0, y0, x1, y1 = box(r, c)
        shadow(canvas, (x0, y0, x1, y1))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=WHITE, outline=LINE, width=2)
        # left accent
        draw.rounded_rectangle((x0, y0, x0 + 12, y1), radius=8, fill=accent + (255,))
        draw.rectangle((x0 + 6, y0, x0 + 12, y1), fill=accent + (255,))

        # number
        bx, by = x0 + 32, y0 + 28
        draw.rounded_rectangle((bx, by, bx + 70, by + 44), radius=12, fill=accent + (255,))
        nf = font(22, True)
        bbox = draw.textbbox((0, 0), num, font=nf)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((bx + (70 - tw) / 2, by + (44 - th) / 2 - 2), num, font=nf, fill=WHITE)

        draw.text((x0 + 116, y0 + 30), title, font=font(28, True), fill=INK)
        draw.text((x0 + 116, y0 + 72), subtitle, font=font(20, False), fill=MUTED)

        n = max(len(logos), 1)
        total = n * logo_size + (n - 1) * 16
        start_x = x0 + (cw - total) / 2
        ly = y0 + 128
        for i, name in enumerate(logos):
            logo = load_logo(name, logo_size)
            if logo:
                canvas.alpha_composite(logo, dest=(int(start_x + i * (logo_size + 16)), int(ly)))

    # Goal card
    x0, y0, x1, y1 = box(2, 3)
    shadow(canvas, (x0, y0, x1, y1))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=FOOTER)
    draw.rounded_rectangle((x0, y0, x0 + 12, y1), radius=8, fill=GOAL + (255,))
    draw.rectangle((x0 + 6, y0, x0 + 12, y1), fill=GOAL + (255,))
    draw.text((x0 + 36, y0 + 28), "GOAL", font=font(20, True), fill=GOAL + (255,))
    for i, word in enumerate(["Build", "Automate", "Monitor", "Deploy"]):
        draw.text((x0 + 36, y0 + 68 + i * 40), word, font=font(32, True), fill=WHITE)

    draw.rounded_rectangle((72, H - 92, W - 72, H - 28), radius=16, fill=FOOTER)
    draw.text((104, H - 78), "Build   ·   Automate   ·   Monitor   ·   Deploy",
              font=font(28, True), fill=WHITE)
    draw.text((W - 430, H - 74), "Kethan Gummalla", font=font(22, False), fill=(176, 186, 198, 255))

    canvas.convert("RGB").save(OUT, "PNG", optimize=True)
    print(f"Wrote {OUT} {canvas.size}")


if __name__ == "__main__":
    main()
