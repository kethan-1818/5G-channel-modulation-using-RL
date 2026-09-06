#!/usr/bin/env python3
"""Generate Friday Deployment Club — Day 2 DevOps Roadmap PowerPoint."""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

ASSETS = Path(__file__).parent / "assets" / "logos"
INFOGRAPHIC = Path(__file__).parent / "assets" / "devops_roadmap_infographic.png"
OUTPUT = Path(__file__).parent / "FDC_Day2_DevOps_Roadmap.pptx"

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0B, 0x11, 0x20)
BG_CARD  = RGBColor(0x14, 0x1E, 0x32)
CYAN     = RGBColor(0x22, 0xD3, 0xEE)
WHITE    = RGBColor(0xF8, 0xFA, 0xFC)
GRAY     = RGBColor(0x94, 0xA3, 0xB8)
ORANGE   = RGBColor(0xFB, 0x92, 0x3C)
GREEN    = RGBColor(0x34, 0xD3, 0x99)
PURPLE   = RGBColor(0xA7, 0x8B, 0xFA)
PINK     = RGBColor(0xF4, 0x72, 0xB6)
RED      = RGBColor(0xF8, 0x71, 0x71)
BLUE     = RGBColor(0x60, 0xA5, 0xFA)
YELLOW   = RGBColor(0xFA, 0xCC, 0x15)

SECTION_COLORS = {
    "foundations": CYAN,
    "containers": ORANGE,
    "cloud": YELLOW,
    "infra": PURPLE,
    "security": GREEN,
    "goal": PINK,
}


def logo(name: str) -> Path | None:
    p = ASSETS / f"{name}.png"
    return p if p.exists() else None


def set_slide_bg(slide, color: RGBColor = BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill: RGBColor, line: RGBColor | None = None, radius=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                     left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text, size=24, color=WHITE,
                bold=False, align=PP_ALIGN.LEFT, font_name="Calibri"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    run = p.runs[0]
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return box


def add_badge(slide, left, top, text, color: RGBColor, width=Inches(1.2)):
    h = Inches(0.35)
    shape = add_rect(slide, left, top, width, h, color, radius=True)
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    run = p.runs[0]
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = BG
    run.font.name = "Calibri"
    return shape


def add_tool_icon(slide, left, top, name: str, label: str, icon_size=Inches(0.65)):
    path = logo(name)
    if path:
        slide.shapes.add_picture(str(path), left, top, icon_size, icon_size)
    else:
        add_rect(slide, left, top, icon_size, icon_size, BG_CARD, GRAY, radius=True)
    add_textbox(slide, left - Inches(0.1), top + icon_size + Inches(0.05),
                icon_size + Inches(0.2), Inches(0.35), label, size=10, color=GRAY,
                align=PP_ALIGN.CENTER)


def add_tool_row(slide, tools: list[tuple[str, str]], y, x_start=Inches(0.8), spacing=Inches(1.35)):
    for i, (icon, label) in enumerate(tools):
        add_tool_icon(slide, x_start + i * spacing, y, icon, label)


def slide_header(slide, stop_num: str, title: str, subtitle: str, accent: RGBColor):
    add_rect(slide, Inches(0), Inches(0), Inches(13.33), Inches(0.08), accent)
    add_badge(slide, Inches(0.6), Inches(0.35), stop_num, accent, width=Inches(1.0))
    add_textbox(slide, Inches(1.75), Inches(0.28), Inches(10), Inches(0.6),
                title, size=32, color=WHITE, bold=True)
    add_textbox(slide, Inches(0.6), Inches(1.05), Inches(11.5), Inches(0.5),
                subtitle, size=16, color=GRAY)


def slide_infographic(prs):
    """Full-bleed winding-road roadmap (screenshot this)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    if INFOGRAPHIC.exists():
        slide.shapes.add_picture(str(INFOGRAPHIC), Inches(0), Inches(0),
                                 Inches(13.333), Inches(7.5))
    else:
        add_textbox(slide, Inches(0.6), Inches(3), Inches(12), Inches(1),
                    "Roadmap infographic missing — run generate_roadmap_infographic.py",
                    size=18, color=GRAY, align=PP_ALIGN.CENTER)


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_rect(slide, Inches(0), Inches(5.5), Inches(13.33), Inches(2), BG_CARD)
    add_rect(slide, Inches(0), Inches(0), Inches(0.12), Inches(7.5), CYAN)

    add_textbox(slide, Inches(0.8), Inches(1.6), Inches(11), Inches(0.5),
                "FRIDAY DEPLOYMENT CLUB", size=18, color=CYAN, bold=True)
    add_textbox(slide, Inches(0.8), Inches(2.2), Inches(11), Inches(1.2),
                "DevOps Roadmap", size=54, color=WHITE, bold=True)
    add_textbox(slide, Inches(0.8), Inches(3.5), Inches(11), Inches(0.6),
                "DAY 2  ·  Your Complete Learning Map", size=22, color=GRAY)

    add_textbox(slide, Inches(0.8), Inches(5.85), Inches(11), Inches(0.5),
                "Build  ·  Automate  ·  Monitor  ·  Deploy", size=20, color=WHITE, bold=True)


def slide_roadmap_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_textbox(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
                "The DevOps Roadmap", size=36, color=WHITE, bold=True)
    add_textbox(slide, Inches(0.6), Inches(0.95), Inches(12), Inches(0.4),
                "5 phases  ·  10 stops  ·  One learning path", size=16, color=GRAY)

    phases = [
        ("PHASE 1", "Foundations", "Linux · Bash · Git", SECTION_COLORS["foundations"],
         ["1 Foundations", "2 Source Control"]),
        ("PHASE 2", "Containers & CI/CD", "Docker · Jenkins · GitHub Actions", SECTION_COLORS["containers"],
         ["3 Containerization", "4 CI/CD"]),
        ("PHASE 3", "Cloud", "AWS Core Services", SECTION_COLORS["cloud"],
         ["5 Cloud Platforms"]),
        ("PHASE 4", "Infrastructure & Orchestration", "IaC · Config · K8s · GitOps", SECTION_COLORS["infra"],
         ["6 IaC", "7 Config", "8 Kubernetes", "10 GitOps"]),
        ("PHASE 5", "Security & Monitoring", "Secure · Observe · Alert", SECTION_COLORS["security"],
         ["9 Security", "10 Observability"]),
    ]

    x, y = Inches(0.55), Inches(1.55)
    card_w, card_h = Inches(2.35), Inches(4.6)
    gap = Inches(0.18)

    for i, (phase, title, sub, color, stops) in enumerate(phases):
        cx = x + i * (card_w + gap)
        card = add_rect(slide, cx, y, card_w, card_h, BG_CARD, color, radius=True)
        add_badge(slide, cx + Inches(0.15), y + Inches(0.15), phase, color, width=Inches(0.95))
        add_textbox(slide, cx + Inches(0.15), y + Inches(0.6), card_w - Inches(0.3), Inches(0.7),
                    title, size=14, color=WHITE, bold=True)
        add_textbox(slide, cx + Inches(0.15), y + Inches(1.2), card_w - Inches(0.3), Inches(0.5),
                    sub, size=10, color=GRAY)

        for j, stop in enumerate(stops):
            sy = y + Inches(1.85) + j * Inches(0.55)
            dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, cx + Inches(0.2), sy + Inches(0.08),
                                         Inches(0.12), Inches(0.12))
            dot.fill.solid()
            dot.fill.fore_color.rgb = color
            dot.line.fill.background()
            add_textbox(slide, cx + Inches(0.42), sy, card_w - Inches(0.55), Inches(0.35),
                        stop, size=11, color=WHITE)

    # Goal banner
    banner = add_rect(slide, Inches(0.55), Inches(6.35), Inches(12.2), Inches(0.75), BG_CARD, PINK, radius=True)
    add_textbox(slide, Inches(0.55), Inches(6.45), Inches(12.2), Inches(0.55),
                "GOAL →  BUILD   ·   AUTOMATE   ·   MONITOR   ·   DEPLOY",
                size=18, color=PINK, bold=True, align=PP_ALIGN.CENTER)


def slide_foundations(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 1", "Foundations", "Linux  ·  Bash  ·  Git  ·  ~2–3 weeks", SECTION_COLORS["foundations"])

    card = add_rect(slide, Inches(0.6), Inches(1.8), Inches(12.1), Inches(4.5), BG_CARD, SECTION_COLORS["foundations"], radius=True)

    tools = [("linux", "Linux"), ("bash", "Bash"), ("git", "Git")]
    add_tool_row(slide, tools, Inches(2.5), x_start=Inches(4.5), spacing=Inches(2.0))

    bullets = [
        "Navigate Linux terminals & file systems",
        "Automate tasks with Bash scripts",
        "Track code changes with Git",
    ]
    for i, b in enumerate(bullets):
        add_textbox(slide, Inches(1.2), Inches(3.8 + i * 0.55), Inches(10.5), Inches(0.4),
                    f"▸  {b}", size=16, color=WHITE)

    add_textbox(slide, Inches(0.6), Inches(6.5), Inches(12), Inches(0.4),
                "Series Days 3–9 cover these in depth", size=13, color=GRAY, align=PP_ALIGN.CENTER)


def slide_source_control(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 2", "Source Control", "Git for real-world team workflows", SECTION_COLORS["foundations"])

    add_tool_icon(slide, Inches(5.85), Inches(2.2), "git", "Git", icon_size=Inches(1.0))

    topics = ["Branches", "Pull Requests", "Code Reviews", "Merge Conflicts"]
    for i, t in enumerate(topics):
        col, row = i % 2, i // 2
        add_rect(slide, Inches(1.5 + col * 5.5), Inches(3.8 + row * 1.1),
                 Inches(4.8), Inches(0.75), BG_CARD, CYAN, radius=True)
        add_textbox(slide, Inches(1.7 + col * 5.5), Inches(3.95 + row * 1.1),
                    Inches(4.4), Inches(0.5), t, size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)


def slide_containerization(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 3", "Containerization", "Package apps + dependencies into portable containers", SECTION_COLORS["containers"])

    add_tool_icon(slide, Inches(5.85), Inches(2.0), "docker", "Docker", icon_size=Inches(1.1))

    flow = ["Your App", "→", "Container Image", "→", "Runs Anywhere"]
    for i, item in enumerate(flow):
        x = Inches(1.0 + i * 2.3)
        if item == "→":
            add_textbox(slide, x, Inches(4.2), Inches(0.5), Inches(0.5), item, size=28, color=ORANGE, align=PP_ALIGN.CENTER)
        else:
            add_rect(slide, x, Inches(3.9), Inches(1.8), Inches(0.9), BG_CARD, ORANGE, radius=True)
            add_textbox(slide, x, Inches(4.05), Inches(1.8), Inches(0.6), item, size=13, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(0.6), Inches(5.5), Inches(12), Inches(0.5),
                "Solves: \"It works on my machine\"", size=20, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)


def slide_cicd(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 4", "CI / CD", "Automate build · test · deploy pipelines", SECTION_COLORS["containers"])

    tools = [("jenkins", "Jenkins"), ("githubactions", "GitHub Actions")]
    add_tool_row(slide, tools, Inches(2.3), x_start=Inches(4.3), spacing=Inches(2.5))

    steps = ["Push Code", "Build", "Test", "Deploy"]
    for i, s in enumerate(steps):
        x = Inches(1.2 + i * 2.9)
        add_rect(slide, x, Inches(4.0), Inches(2.2), Inches(0.85), BG_CARD, ORANGE, radius=True)
        add_textbox(slide, x, Inches(4.15), Inches(2.2), Inches(0.55), s, size=15, color=WHITE, align=PP_ALIGN.CENTER)
        if i < 3:
            add_textbox(slide, x + Inches(2.2), Inches(4.15), Inches(0.7), Inches(0.5),
                        "→", size=22, color=ORANGE, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(0.6), Inches(5.6), Inches(12), Inches(0.4),
                "Less manual work  ·  Faster releases  ·  Fewer mistakes", size=15, color=GRAY, align=PP_ALIGN.CENTER)


def slide_cloud(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 5", "Cloud Platforms — AWS", "Core services every DevOps engineer needs", SECTION_COLORS["cloud"])

    add_tool_row(slide, [("amazonaws", "AWS"), ("googlecloud", "GCP"), ("microsoftazure", "Azure")],
                Inches(1.7), x_start=Inches(8.4), spacing=Inches(1.4))

    services = [
        ("EC2", "Virtual Machines"),
        ("VPC", "Networking"),
        ("IAM", "Access & Permissions"),
        ("S3", "Object Storage"),
        ("ALB", "Load Balancer"),
        ("ECS", "Containers"),
        ("EKS", "Kubernetes"),
    ]
    for i, (name, desc) in enumerate(services):
        col, row = i % 4, i // 4
        x = Inches(0.8 + col * 3.1)
        y = Inches(2.8 + row * 1.65)
        add_rect(slide, x, y, Inches(2.8), Inches(1.2), BG_CARD, YELLOW, radius=True)
        add_textbox(slide, x + Inches(0.15), y + Inches(0.15), Inches(2.5), Inches(0.45),
                    name, size=20, color=YELLOW, bold=True)
        add_textbox(slide, x + Inches(0.15), y + Inches(0.6), Inches(2.5), Inches(0.4),
                    desc, size=12, color=GRAY)

    add_textbox(slide, Inches(0.6), Inches(6.4), Inches(12), Inches(0.4),
                "One video · One concept · One service at a time", size=13, color=GRAY, align=PP_ALIGN.CENTER)


def slide_iac(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 6", "Infrastructure as Code", "Define infrastructure in code — repeatable & version-controlled", SECTION_COLORS["infra"])

    tools = [("terraform", "Terraform"), ("amazonaws", "CloudFormation")]
    add_tool_row(slide, tools, Inches(2.5), x_start=Inches(4.0), spacing=Inches(2.8))

    add_textbox(slide, Inches(1.5), Inches(4.2), Inches(10), Inches(0.5),
                "Code  →  Plan  →  Apply  →  Reusable Infrastructure", size=18, color=PURPLE, align=PP_ALIGN.CENTER)


def slide_config(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 7", "Configuration & Automation", "Terraform creates it · Ansible configures it", SECTION_COLORS["infra"])

    tools = [("ansible", "Ansible"), ("amazonaws", "AWS SSM")]
    add_tool_row(slide, tools, Inches(2.5), x_start=Inches(4.0), spacing=Inches(2.8))

    tasks = ["Install software", "Create users", "Manage settings", "Multi-server automation"]
    for i, t in enumerate(tasks):
        add_textbox(slide, Inches(1.5 + (i % 2) * 5.5), Inches(4.0 + (i // 2) * 0.65),
                    Inches(5), Inches(0.4), f"▸  {t}", size=15, color=WHITE)


def slide_kubernetes(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 8", "Kubernetes & Helm", "Run, scale & manage containers at production scale", SECTION_COLORS["infra"])

    tools = [("kubernetes", "Kubernetes"), ("helm", "Helm")]
    add_tool_row(slide, tools, Inches(2.3), x_start=Inches(4.0), spacing=Inches(2.8))

    feats = [("Auto-restart", "Failed containers"), ("Auto-scale", "Traffic spikes"), ("Helm charts", "App deployment")]
    for i, (title, sub) in enumerate(feats):
        x = Inches(1.0 + i * 3.8)
        add_rect(slide, x, Inches(4.0), Inches(3.4), Inches(1.1), BG_CARD, PURPLE, radius=True)
        add_textbox(slide, x + Inches(0.15), Inches(4.15), Inches(3.1), Inches(0.4), title, size=16, color=PURPLE, bold=True)
        add_textbox(slide, x + Inches(0.15), Inches(4.55), Inches(3.1), Inches(0.4), sub, size=12, color=GRAY)


def slide_security(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 9", "Security", "Protect access, scan for vulnerabilities, stay compliant", SECTION_COLORS["security"])

    areas = [
        ("IAM Security", "Least privilege · Roles · Policies"),
        ("Vulnerability Scanning", "Images · Dependencies · Configs"),
        ("Secrets Management", "Credentials · Keys · Tokens"),
    ]
    for i, (title, sub) in enumerate(areas):
        y = Inches(2.2 + i * 1.35)
        add_rect(slide, Inches(1.5), y, Inches(10.3), Inches(1.05), BG_CARD, GREEN, radius=True)
        add_textbox(slide, Inches(1.75), y + Inches(0.12), Inches(9.5), Inches(0.4), title, size=18, color=GREEN, bold=True)
        add_textbox(slide, Inches(1.75), y + Inches(0.52), Inches(9.5), Inches(0.4), sub, size=13, color=GRAY)


def slide_observability(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "STOP 10", "Observability", "Metrics · Dashboards · Logs · Alerts", SECTION_COLORS["security"])

    tools = [("prometheus", "Prometheus"), ("grafana", "Grafana")]
    add_tool_row(slide, tools, Inches(2.2), x_start=Inches(3.5), spacing=Inches(2.5))

    add_rect(slide, Inches(8.5), Inches(2.2), Inches(1.0), Inches(1.0), BG_CARD, GREEN, radius=True)
    add_textbox(slide, Inches(8.5), Inches(2.45), Inches(1.0), Inches(0.5), "Logs", size=14, color=GREEN, align=PP_ALIGN.CENTER)

    pillars = ["What's happening?", "Why is it slow?", "What broke?", "Alert the team"]
    for i, p in enumerate(pillars):
        add_textbox(slide, Inches(1.5 + (i % 2) * 5.5), Inches(4.0 + (i // 2) * 0.7),
                    Inches(5), Inches(0.4), f"▸  {p}", size=15, color=WHITE)


def slide_gitops(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    slide_header(slide, "GITOPS", "Git as the Source of Truth for Deployments", "ArgoCD · Flux · Git-driven deploys", SECTION_COLORS["infra"])

    tools = [("argocd", "ArgoCD"), ("flux", "Flux"), ("git", "Git")]
    add_tool_row(slide, tools, Inches(2.3), x_start=Inches(3.2), spacing=Inches(2.3))

    add_textbox(slide, Inches(1.0), Inches(4.2), Inches(11), Inches(0.5),
                "Git commit  →  Sync  →  Cluster matches desired state", size=18, color=PURPLE, align=PP_ALIGN.CENTER)


def slide_goal(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, Inches(0.6), Inches(0.5), Inches(12), Inches(0.7),
                "The DevOps Goal", size=36, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    goals = [("BUILD", CYAN), ("AUTOMATE", ORANGE), ("MONITOR", GREEN), ("DEPLOY", PINK)]
    for i, (word, color) in enumerate(goals):
        x = Inches(0.9 + i * 3.0)
        add_rect(slide, x, Inches(2.5), Inches(2.6), Inches(2.6), BG_CARD, color, radius=True)
        add_textbox(slide, x, Inches(3.35), Inches(2.6), Inches(0.8), word, size=26, color=color, bold=True, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(0.6), Inches(5.8), Inches(12), Inches(0.5),
                "That's the core of DevOps", size=18, color=GRAY, align=PP_ALIGN.CENTER)


def slide_beyond(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, Inches(0.6), Inches(0.4), Inches(12), Inches(0.7),
                "Beyond DevOps → AI & Modern Tech", size=32, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    topics = ["MLOps", "AI Pipelines", "Local LLMs", "AI Agents", "Emerging Tech"]
    for i, t in enumerate(topics):
        col, row = i % 3, i // 3
        x = Inches(1.2 + col * 3.7)
        y = Inches(1.8 + row * 1.5)
        add_rect(slide, x, y, Inches(3.2), Inches(1.0), BG_CARD, BLUE, radius=True)
        add_textbox(slide, x, y + Inches(0.28), Inches(3.2), Inches(0.5), t, size=18, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(0.6), Inches(5.5), Inches(12), Inches(0.5),
                "Friday Deployment Club — building modern tech skills step by step", size=14, color=GRAY, align=PP_ALIGN.CENTER)


def slide_challenge(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, Inches(0.6), Inches(0.4), Inches(12), Inches(0.7),
                "Day 2 Challenge", size=36, color=PINK, bold=True, align=PP_ALIGN.CENTER)

    steps = [
        ("1", "Screenshot the roadmap"),
        ("2", "Circle your top 3 topics"),
        ("3", "Comment them below 👇"),
    ]
    for i, (num, text) in enumerate(steps):
        y = Inches(1.8 + i * 1.3)
        add_badge(slide, Inches(2.5), y, num, PINK, width=Inches(0.55))
        add_textbox(slide, Inches(3.3), y - Inches(0.02), Inches(8), Inches(0.5), text, size=22, color=WHITE)

    add_textbox(slide, Inches(0.6), Inches(5.8), Inches(12), Inches(0.5),
                "NEXT → Day 3: Linux Terminal · File System · Live Demo", size=16, color=CYAN, align=PP_ALIGN.CENTER)

    add_textbox(slide, Inches(0.6), Inches(6.4), Inches(12), Inches(0.4),
                "Learn it. Build it. Ship it.", size=14, color=GRAY, align=PP_ALIGN.CENTER)


def slide_series_map(prs):
    """Full visual roadmap — screenshot-friendly single slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.55),
                "DEVOPS ROADMAP 2025", size=28, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    stops = [
        (1, "Foundations", "Linux · Bash · Git", SECTION_COLORS["foundations"],
         [("linux", ""), ("bash", ""), ("git", "")]),
        (2, "Source Control", "Branches · PRs · Reviews", SECTION_COLORS["foundations"],
         [("git", "")]),
        (3, "Containerization", "Docker", SECTION_COLORS["containers"],
         [("docker", "")]),
        (4, "CI / CD", "Jenkins · GH Actions", SECTION_COLORS["containers"],
         [("jenkins", ""), ("githubactions", "")]),
        (5, "Cloud — AWS", "EC2 · VPC · IAM · S3 · ECS · EKS", SECTION_COLORS["cloud"],
         [("amazonaws", ""), ("googlecloud", ""), ("microsoftazure", "")]),
        (6, "Infrastructure as Code", "Terraform · CloudFormation", SECTION_COLORS["infra"],
         [("terraform", ""), ("cloudformation", "")]),
        (7, "Config & Automation", "Ansible · AWS SSM", SECTION_COLORS["infra"],
         [("ansible", ""), ("amazonaws", "")]),
        (8, "Kubernetes & Helm", "Scale · Manage · Deploy", SECTION_COLORS["infra"],
         [("kubernetes", ""), ("helm", "")]),
        (9, "Security", "IAM · Scanning · Secrets", SECTION_COLORS["security"],
         []),
        (10, "Observability", "Prometheus · Grafana · Logs", SECTION_COLORS["security"],
         [("prometheus", ""), ("grafana", "")]),
    ]

    cols = 2
    for idx, (num, title, sub, color, icons) in enumerate(stops):
        col, row = idx % cols, idx // cols
        x = Inches(0.4 + col * 6.5)
        y = Inches(0.75 + row * 1.22)
        w, h = Inches(6.2), Inches(1.05)

        card = add_rect(slide, x, y, w, h, BG_CARD, color, radius=True)
        add_badge(slide, x + Inches(0.1), y + Inches(0.12), f"#{num}", color, width=Inches(0.45))

        add_textbox(slide, x + Inches(0.65), y + Inches(0.08), Inches(2.8), Inches(0.35),
                    title, size=12, color=WHITE, bold=True)
        add_textbox(slide, x + Inches(0.65), y + Inches(0.42), Inches(3.2), Inches(0.3),
                    sub, size=9, color=GRAY)

        ix = x + Inches(4.0)
        for j, (icon, _) in enumerate(icons[:4]):
            path = logo(icon)
            if path:
                slide.shapes.add_picture(str(path), ix + j * Inches(0.42), y + Inches(0.22),
                                         Inches(0.38), Inches(0.38))

    # GitOps row
    y = Inches(6.85)
    add_rect(slide, Inches(0.4), y, Inches(12.5), Inches(0.55), BG_CARD, PURPLE, radius=True)
    add_textbox(slide, Inches(0.6), y + Inches(0.08), Inches(2), Inches(0.35), "GitOps", size=12, color=PURPLE, bold=True)
    add_textbox(slide, Inches(2.5), y + Inches(0.1), Inches(4), Inches(0.35), "ArgoCD · Flux · Git-driven deploys", size=10, color=GRAY)
    for j, icon in enumerate(["argocd", "flux", "git"]):
        path = logo(icon)
        if path:
            slide.shapes.add_picture(str(path), Inches(10.5 + j * 0.45), y + Inches(0.08), Inches(0.38), Inches(0.38))


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_infographic(prs)          # winding-road visual (screenshot this)
    slide_roadmap_overview(prs)
    slide_series_map(prs)
    slide_foundations(prs)
    slide_source_control(prs)
    slide_containerization(prs)
    slide_cicd(prs)
    slide_cloud(prs)
    slide_iac(prs)
    slide_config(prs)
    slide_kubernetes(prs)
    slide_security(prs)
    slide_observability(prs)
    slide_gitops(prs)
    slide_goal(prs)
    slide_beyond(prs)
    slide_challenge(prs)

    prs.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
