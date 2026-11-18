"""Build h-cbm-eeg.pdf: progress report for the professor (overview of idea + pipeline, then 5 presenter parts).

Run: docs/report/build.sh   (uses .venv-report with reportlab; chart from docs/report/make_chart.py)
Human-in-the-loop / human validation are not done yet and are not mentioned in the report.
"""
import json
import os
import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
PAGES = {k: v for k, v in json.loads(sys.argv[2]).items()} if len(sys.argv) > 2 else {}
pg = lambda k: PAGES.get(k, "?")  # noqa: E731
FD = "/usr/share/fonts/liberation-sans-fonts/"
pdfmetrics.registerFont(TTFont("LS", FD + "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LS-B", FD + "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LS-I", FD + "LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LS-BI", FD + "LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFontFamily("LS", normal="LS", bold="LS-B", italic="LS-I", boldItalic="LS-BI")

MAROON, BEIGE, INK, INK2, RULE = colors.HexColor("#73303A"), colors.HexColor("#F4EBDA"), colors.HexColor("#1D1B1A"), \
    colors.HexColor("#55514C"), colors.HexColor("#D9D2C5")
st = dict(
    body=ParagraphStyle("body", fontName="LS", fontSize=10, leading=13.5, textColor=INK, spaceAfter=5),
    small=ParagraphStyle("small", fontName="LS", fontSize=8.5, leading=11.5, textColor=INK2, spaceAfter=4),
    h1=ParagraphStyle("h1", fontName="LS-B", fontSize=19, leading=23, textColor=MAROON, spaceAfter=4),
    h2=ParagraphStyle("h2", fontName="LS-B", fontSize=12.5, leading=16, textColor=MAROON, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    kicker=ParagraphStyle("kicker", fontName="LS-B", fontSize=9, leading=12, textColor=INK2, spaceAfter=2),
    presenter=ParagraphStyle("presenter", fontName="LS", fontSize=10, leading=13, textColor=INK2, spaceAfter=10),
    cell=ParagraphStyle("cell", fontName="LS", fontSize=8.8, leading=11.5, textColor=INK),
    cellb=ParagraphStyle("cellb", fontName="LS-B", fontSize=8.8, leading=11.5, textColor=INK),
    head=ParagraphStyle("head", fontName="LS-B", fontSize=8.8, leading=11.5, textColor=colors.white),
    bullet=ParagraphStyle("bullet", fontName="LS", fontSize=10, leading=13.2, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
    key=ParagraphStyle("key", fontName="LS", fontSize=10.5, leading=15, textColor=INK),
    title=ParagraphStyle("title", fontName="LS-B", fontSize=40, leading=46, textColor=MAROON, alignment=TA_LEFT),
    sub=ParagraphStyle("sub", fontName="LS", fontSize=15, leading=20, textColor=INK, spaceAfter=6),
)
W = A4[0] - 4 * cm


def P(t, s="body"):
    return Paragraph(t, st[s])


def bullets(items):
    return [Paragraph(t, st["bullet"], bulletText="•") for t in items]


def table(rows, widths, bold_first_col=False, zebra=True):
    data = [[Paragraph(c, st["head"]) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(c, st["cellb"] if (bold_first_col and j == 0) else st["cell"]) for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), MAROON), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2.6),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FAF6EE")) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


def keybox(text, label="Key message"):
    t = Table([[Paragraph(f"<b>{label}.</b> {text}", st["key"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BEIGE), ("LINEBEFORE", (0, 0), (0, -1), 3, MAROON),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def part(n, title, minutes, key):
    return [PageBreak(), P(f"PART {n} OF 5", "kicker"), P(title, "h1"), Spacer(1, 4), keybox(key), Spacer(1, 8)]


def fig(path, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width_cm * cm
    return KeepTogether([Image(path, width=w, height=w * ih / iw), P(caption, "small")])


def pipeline_diagram():
    H = 300
    d = Drawing(W, H)
    bw, bh, x0 = 300, 30, (W - 300) / 2
    boxes = [("1 · EEG window", "4 s · 256 Hz · 30 channels (12 for DREAMER)", "plain"),
             ("2 · EEGPT encoder (frozen) + spectral features", "2,048 + 292 numbers per window", "plain"),
             ("3 · Level 1 · Neurophysiological", "theta / alpha / gamma power × 12 channels + frontal asymmetry (37)", "concept"),
             ("4 · Level 2 · Affective", "valence · arousal · dominance", "concept"),
             ("5 · Level 3 · Cognitive appraisal", "novelty · goal conduciveness · coping", "concept"),
             ("6 · Emotion classifier", "7 classes, sees only Level 3", "plain"),
             ("7 · Qwen LLM explanation", "generated only from the predicted concepts", "dark")]
    y = H - bh - 2
    for i, (t1, t2, kind) in enumerate(boxes):
        fill = {"plain": colors.white, "concept": BEIGE, "dark": MAROON}[kind]
        c1 = {"plain": INK, "concept": MAROON, "dark": colors.white}[kind]
        c2 = colors.white if kind == "dark" else INK2
        d.add(Rect(x0, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=MAROON, strokeWidth=1))
        d.add(String(x0 + bw / 2, y + 18, t1, fontName="LS-B", fontSize=9, fillColor=c1, textAnchor="middle"))
        d.add(String(x0 + bw / 2, y + 7, t2, fontName="LS", fontSize=7.5, fillColor=c2, textAnchor="middle"))
        if i < len(boxes) - 1:
            d.add(Line(x0 + bw / 2, y, x0 + bw / 2, y - 9, strokeColor=INK2, strokeWidth=1))
            d.add(Polygon([x0 + bw / 2 - 3, y - 7, x0 + bw / 2 + 3, y - 7, x0 + bw / 2, y - 11], fillColor=INK2, strokeColor=INK2))
        y -= bh + 11
    # brace label for the concept bottleneck
    yt, yb = H - 2 - 2 * (bh + 11), H - 2 - 4 * (bh + 11) - bh
    xb = x0 + bw + 8
    d.add(Line(xb, yt, xb, yb, strokeColor=MAROON, strokeWidth=1.2))
    d.add(Line(xb - 4, yt, xb, yt, strokeColor=MAROON, strokeWidth=1.2)); d.add(Line(xb - 4, yb, xb, yb, strokeColor=MAROON, strokeWidth=1.2))
    d.add(String(xb + 5, (yt + yb) / 2 + 4, "concept", fontName="LS-B", fontSize=8, fillColor=MAROON))
    d.add(String(xb + 5, (yt + yb) / 2 - 6, "bottleneck", fontName="LS-B", fontSize=8, fillColor=MAROON))
    return d


def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setFont("LS", 8); c.setFillColor(INK2)
        c.drawString(2 * cm, A4[1] - 1.2 * cm, "H-CBM-EEG · Progress report")
        c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(2 * cm, A4[1] - 1.35 * cm, A4[0] - 2 * cm, A4[1] - 1.35 * cm)
    c.restoreState()


story = []
# ------------------------------------------------------------------ cover
story += [Spacer(1, 2.2 * cm), P("PROGRESS REPORT", "kicker"), P("H-CBM-EEG", "title"), Spacer(1, 4),
          P("Hierarchical Concept Bottleneck Model for Trustworthy EEG-to-Emotion LLMs", "sub"),
          P("Progress report", "presenter"), P("<b>Team members:</b> Pratham, Param, Neel, Arju, Leon", "body"), Spacer(1, 10),
          keybox("The model pipeline now runs end to end: EEG → frozen EEGPT encoder → three concept levels → emotion → "
                 "Qwen explanation. Trained on the FACED dataset, the concept model recognises 7 emotions in "
                 "<b>people it has never seen</b> with <b>35.7% accuracy (0.295 macro-F1)</b>, <b>43.5% per 30-second clip</b>; "
                 "random guessing scores 14%. It is 3.6 points of macro-F1 behind an opaque baseline, the measured price of "
                 "interpretability.", "Where we are"),
          Spacer(1, 14), P("Contents", "h2"),
          table([["", "Section", "Pages"],
                 ["", "<b>Overview: the idea and the pipeline</b> (read first)", pg("O")],
                 ["1", "The project and how the plan evolved", pg("1")],
                 ["2", "Data: datasets, audits, preprocessing, concept targets", pg("2")],
                 ["3", "Models and training pipeline", pg("3")],
                 ["4", "Results and how we improved them", pg("4")],
                 ["5", "Generalisation, explanations and limitations", pg("5")]],
                [0.06, 0.82, 0.12], bold_first_col=True, zebra=False),
          Spacer(1, 14), P("Status at a glance", "h2"),
          table([["Stage", "Status"],
                 ["Data inspection, audits, preprocessing (4 datasets, 12 and 30 channels)", "Done"],
                 ["EEGPT embeddings, concept targets, models, training, evaluation", "Done"],
                 ["Zero-shot testing on three unseen datasets", "Done"],
                 ["Qwen explanations + FaithScore", "Done for the earlier DENS model; to rerun for the FACED model"],
                 ["Results write-up and slide updates", "Next"]], [0.68, 0.32])]

# ------------------------------------------------------------------ overview: idea + pipeline
story += [PageBreak(), P("OVERVIEW", "kicker"), P("The idea and the pipeline", "h1"), Spacer(1, 4),
          keybox("Make an EEG emotion recogniser that <b>shows its reasoning</b>. Instead of mapping brain signals straight "
                 "to an emotion through a black box, the model must pass through three levels of human-meaningful concepts, "
                 "and an LLM explains the result using only those concepts.", "In one sentence"),
          P("The problem", "h2"),
          P("Recent systems that combine EEG with large language models (for example E²-LLM) can predict an emotion from brain "
            "signals and describe it in fluent text. But the step from EEG to emotion is an opaque neural network, so nobody "
            "can check <i>why</i> it decided something, and the text explanation is not guaranteed to reflect what the model "
            "actually used. For clinical use, where a doctor must be able to trust and verify a decision, this is a serious gap."),
          P("The idea: a hierarchical concept bottleneck", "h2"),
          P("A <b>concept bottleneck model</b> (CBM) forces a network to first predict concepts that people understand, and to "
            "make its final decision <i>only</i> from those concepts. We stack three levels, following Scherer's Component "
            "Process Model of emotion (an emotion arises from bodily signals, a felt affective state, and how a person "
            "appraises the situation):"),
          table([["Level", "Question it answers", "Concepts", "What the model learns from"],
                 ["1 · Neurophysiological", "What is the brain doing?", "Theta, alpha and gamma power in 12 regions + frontal alpha asymmetry (37)", "Measured directly from the EEG"],
                 ["2 · Affective", "How does the person feel?", "Valence (pleasant–unpleasant), arousal (calm–excited), dominance (in control or not)", "Participants' own ratings"],
                 ["3 · Cognitive appraisal", "How do they judge the situation?", "Novelty, goal conduciveness (good or bad for me), coping potential", "Appraisal-theory profiles per emotion"],
                 ["Emotion", "Which emotion is it?", "7 classes: happy, tenderness, anger, fear, disgust, sadness, neutral", "Film-clip category"]],
                [0.20, 0.19, 0.36, 0.25], True),
          Spacer(1, 4),
          P("Because each step is a readable number (for example \"valence 0.2, unpleasant\"), a person can see how the model "
            "reached its decision, and an LLM can explain it from those numbers alone."),
          P("How we train and test", "h2")]
story += bullets(["<b>Training data:</b> FACED (119 people, 28 emotional film clips each). <b>Testing:</b> 24 people held out "
                  "completely, never used for training or for any tuning decision.",
                  "<b>Fair comparison:</b> a flat black-box network (input → emotion, no concepts) gets exactly the same inputs, "
                  "splits and tuning. The difference between the two is the cost or benefit of interpretability.",
                  "<b>Generalisation:</b> the trained model is also tested, unchanged, on three other datasets it has never "
                  "seen: DEAP, DREAMER and DENS."])
story += [P("The pipeline", "h2"), pipeline_diagram(),
          P("The emotion classifier sees only Level 3, so there is no hidden shortcut around the concepts; an automated test "
            "proves this.", "small")]
story += [P("What each stage does", "h2"),
          table([["Stage", "Input → output", "How"],
                 ["1 · Preprocessing", "Raw EEG → 4-s windows, 30 channels, 256 Hz", "Band-pass 4–45 Hz, 50 Hz notch, common average reference, each person scaled to their own baseline, noisy windows flagged"],
                 ["2 · Feature extraction", "Window → 2,340 numbers", "EEGPT, a transformer pretrained on large EEG collections, kept frozen (2,048 numbers, normalised per person), plus 292 classic spectral features (band powers and left–right asymmetries)"],
                 ["3–5 · Concept levels", "Features → 37 → 3 → 3 concepts", "Small neural networks, each seeing only the previous level; trained to match the measured / rated / theory targets above"],
                 ["6 · Emotion classifier", "3 appraisal values → 7 emotion probabilities", "Small network; predictions averaged over 3 neighbouring windows of the same clip"],
                 ["7 · Explanation", "Concepts + emotion → text", "Qwen2.5-1.5B-Instruct, instructed to use only the supplied concepts"]],
                [0.20, 0.27, 0.53], True),
          P("Example: one real test window", "h2"),
          P("A 4-second window from a FACED subject the model never saw during training (output of our demo script):", "small"),
          table([["Stage", "What the model produced"],
                 ["Level 1", "Theta and alpha power low in frontal and temporal regions (close to the values measured from the EEG)"],
                 ["Level 2", "Valence 0.58 (pleasant), arousal 0.55, dominance 0.55"],
                 ["Level 3", "Novelty +0.37, goal conduciveness +0.50, coping +0.38 (positive, manageable situation)"],
                 ["Emotion", "<b>happy</b> (most likely of 7). Correct: the clip was a happy clip"],
                 ["Explanation", "Qwen writes a short paragraph from the three concept levels above"]], [0.18, 0.82], True),
          ]

# ------------------------------------------------------------------ part 1
story += part(1, "The project and how the plan evolved", 4,
              "The research goal and architecture are unchanged from our proposal; what changed is the data. SEED-VII was "
              "unavailable, DENS proved too small, and FACED is now the training set, with DEAP, DREAMER and DENS as unseen "
              "test sets.")
story += [P("The problem we address", "h2"),
          table([["Challenge in today's EEG-to-LLM systems", "Consequence", "Our answer"],
                 ["Hallucinated explanations", "Fluent text that is not grounded in the EEG; faithfulness cannot be verified", "Explanations generated only from concepts; FaithScore test"],
                 ["Black-box projection (opaque MLP)", "Clinicians cannot see intermediate reasoning or psychological concepts", "Three-level concept bottleneck"]],
                [0.28, 0.40, 0.32], True),
          P("Research questions and hypotheses", "h2"),
          table([["", "Question / hypothesis"],
                 ["RQ1", "Does the concept model match a flat model in-dataset and generalise better to new datasets? "
                         "<b>H1</b>: within 2 points of macro-F1 of flat; <b>H2</b>: better zero-shot valence/arousal on other datasets"],
                 ["RQ2", "Are its explanations more faithful? <b>H3</b>: higher FaithScore than flat, placebo false-positive rate below 15%"]],
                [0.08, 0.92], True),
          P("What changed since the proposal deck", "h2"),
          table([["Deck (7 Oct)", "Now", "Why"],
                 ["Train on SEED-VII", "Train on <b>FACED</b> (123 subjects, 28 film clips, 9 emotions)", "SEED-VII not available. DENS was tried first (20–21 Oct) but has only 475 labelled epochs; results were below chance"],
                 ["Test on DEAP / DREAMER", "Test zero-shot on DEAP, DREAMER <b>and DENS</b>", "DENS kept as an extra unseen dataset"],
                 ["7 classes incl. surprise", "7 FACED classes: happy, tenderness, anger, fear, disgust, sadness, neutral", "FACED has no surprise"],
                 ["Compare with E²-LLM's numbers", "Compare with our own flat baseline on identical features and splits", "Different dataset, so their numbers are not comparable"],
                 ["62 EEG channels", "30 channels (12 for DREAMER)", "Only these channels exist across our datasets"]],
                [0.24, 0.40, 0.36]),
          P("Timeline", "h2"),
          table([["Step", "Milestone"],
                 ["1", "Repository, dataset inspection, preprocessing pipeline"],
                 ["2", "Full pipeline built; real EEGPT checkpoint verified on the GPU"],
                 ["3", "DENS as training set, full run (weak results)"],
                 ["4", "FACED as training set; audit of the weak results"],
                 ["5", "Four improvements: best results so far"]], [0.10, 0.90], True)]

# ------------------------------------------------------------------ part 2
story += part(2, "Data: datasets, audits, preprocessing, concept targets", 4,
              "Four public EEG emotion datasets, one shared preprocessing pipeline, and a data audit before any training. The audit "
              "caught real problems (a modified DEAP copy, truncated DENS recordings, unreliable FACED raters) and each has a fixed, "
              "documented rule.")
story += [P("Datasets and roles", "h2"),
          table([["Dataset", "Role", "Subjects", "4-s windows", "Labels"],
                 ["FACED", "Train / validate / in-dataset test", "119", "23,324", "9 categories → 7 classes; valence, arousal"],
                 ["DENS", "Zero-shot test", "40", "1,900", "5 of the 7 classes; valence, arousal, dominance"],
                 ["DEAP", "Zero-shot test", "32", "19,200", "Valence, arousal, dominance (1–9)"],
                 ["DREAMER", "Zero-shot test", "23", "6,210", "Valence, arousal, dominance (1–5); 16 of 18 clips mapped to classes"]],
                [0.13, 0.25, 0.11, 0.13, 0.38], True),
          P("FACED split: 24 subjects held out for testing and 5 subject-wise folds for training/validation, fixed before any "
            "training. No person ever appears in both training and testing.", "small"),
          P("Preprocessing (identical for every dataset)", "h2")]
story += bullets(["Band-pass 4–45 Hz, 50 Hz notch, common average reference, resampling to 256 Hz.",
                  "4-second windows (1,024 samples); 7 windows per 30-s FACED clip.",
                  "Per-subject z-scoring; artifact windows flagged by a subject-relative amplitude rule.",
                  "Two montages: the 12 channels all four datasets share, and FACED's 30 channels (DEAP has all 30; DENS's "
                  "128-electrode cap is mapped to them, mean 6° off; DREAMER has only the 12)."])
story += [P("Data audit findings", "h2"),
          table([["Dataset", "Finding", "What we do"],
                 ["DEAP", "Our local copy differs from the official release on 439 of 1,280 trials (ratings flipped to 9−x)", "Labels taken from the genuine ratings file; re-download before publication"],
                 ["DENS", "Official release has truncated EEG for 30 of 40 subjects", "Use the authors' 475 preprocessed epochs (all 40 subjects)"],
                 ["FACED", "27 of 119 subjects rate valence unlike everyone else (r < 0.5)", "Their valence/arousal loss is masked in training; rule fixed in advance"],
                 ["FACED", "Clip labels checked against ratings", "Correct: negative clips valence 1.5–2.0, neutral ~3.1, positive 4.0–5.2"],
                 ["DREAMER", "No emotion labels in the file", "Clip → emotion table transcribed from the paper"]],
                [0.12, 0.48, 0.40], True),
          P("Concept targets (what each level is trained to predict)", "h2"),
          table([["Level", "Concepts", "Source of the target"],
                 ["1 · Neurophysiological", "Theta, alpha, gamma power on 12 channels + frontal alpha asymmetry (37)", "Measured from the EEG itself"],
                 ["2 · Affective", "Valence, arousal, dominance", "Participants' own ratings (FACED dominance is a pseudo-label, down-weighted)"],
                 ["3 · Appraisal", "Novelty, goal conduciveness, coping", "Per-category profiles from Scherer's appraisal theory: <b>draft, awaiting approval</b>"]],
                [0.22, 0.42, 0.36], True)]

# ------------------------------------------------------------------ part 3
story += part(3, "Models and training pipeline", 4,
              "A frozen foundation model (EEGPT) turns each window into numbers; small trainable heads predict the concepts and "
              "the emotion. We compare the concept model with a flat black-box baseline on identical inputs and splits, under rules "
              "that keep the test data out of every decision.")
story += [P("Models", "h2"),
          table([["Model", "What it is", "Role"],
                 ["EEGPT encoder", "Pretrained EEG transformer (25.3M parameters), frozen; checkpoint hash verified", "Feature extractor"],
                 ["Main CBM (level123)", "EEGPT → L1 (37) → L2 (3) → L3 (3) → emotion (7); each stage sees only the previous one", "Proposed model"],
                 ["Flat baseline", "2-layer MLP + GELU, input → 7 emotions directly", "Black-box comparison (as in E²-LLM)"],
                 ["Ablation variants", "Emotion read from L1 only, from L2, from L2+L3, or with a bypass of the bottleneck", "Show what each level contributes"]],
                [0.22, 0.56, 0.22], True),
          P("Best-run input per window: EEGPT embedding (2,048, normalised per subject) + 292 spectral features (band powers, "
            "relative powers, 13 left–right asymmetries over 30 channels), averaged over 3 neighbouring windows of the same clip.", "small"),
          P("Training recipe", "h2"),
          table([["Aspect", "Choice"],
                 ["Loss", "Sum of the four levels' losses (MSE for concepts, class-weighted cross-entropy for emotion), balanced automatically by learned weights"],
                 ["Optimiser", "AdamW, learning rate and dropout chosen by a small grid search on validation folds"],
                 ["Stopping", "30-epoch warm-up, then early stopping on validation macro-F1 (patience 20, max 200 epochs)"],
                 ["Evaluation", "5 fold models averaged; confidence intervals by resampling subjects, not windows"]], [0.18, 0.82], True),
          P("Safeguards against fooling ourselves", "h2")]
story += bullets(["Splits are by subject; test subjects and the zero-shot datasets are never used for any choice "
                  "(model selection, settings, EEGPT input scale). An automated leakage test checks this.",
                  "Every protocol decision is logged before results are seen (29 decisions, D1–D29).",
                  "42 automated tests, including one proving the emotion head can only be reached through the concepts."])
story += [P("Problems we found and fixed", "h2"),
          table([["Problem", "Fix"],
                 ["The concept model got stuck predicting one class on the small DENS data", "Smaller batches and a 30-epoch warm-up before early stopping"],
                 ["A model method accidentally overrode PyTorch's internal one, breaking GPU training", "Renamed"],
                 ["Class count hard-coded for DENS; two file-handling slips in new code", "Fixed and covered by tests before any affected result"]],
                [0.58, 0.42])]

# ------------------------------------------------------------------ part 4
story += part(4, "Results and how we improved them", 5,
              "First results were weak, so we audited the whole pipeline. There was no bug in code, labels or metrics, but two "
              "fixable weaknesses. Fixing them raised the concept model from 21.6% to 35.7% accuracy on unseen subjects.")
story += [P("From the first attempt to the current best", "h2"),
          table([["Run", "What changed", "CBM accuracy / macro-F1", "Flat accuracy / macro-F1"],
                 ["DENS (21 Oct)", "First full run, 6 classes, 8 test subjects", "15.1% / 0.135", "12.5% / 0.100"],
                 ["faced12 (22 Oct)", "FACED as training set, 12 channels", "21.6% / 0.197", "25.2% / 0.227"],
                 ["faced30 (23 Oct)", "+ all 30 channels + per-subject normalisation", "24.1% / 0.228", "30.5% / 0.266"],
                 ["<b>faced30sx</b> (23 Oct)", "+ spectral features + smoothing over neighbouring windows", "<b>35.7% / 0.295</b>", "<b>37.9% / 0.331</b>"]],
                [0.17, 0.41, 0.21, 0.21], True),
          fig(os.path.join(HERE, "progression.png"), 14.5, "FACED test subjects, 7 classes, per window. DENS used 6 classes on a different test set and is not shown.")]
story += [P("What the audit found", "h2")]
story += bullets(["<b>EEGPT embeddings mostly encode who the person is, not what they feel:</b> subject identity explains 12.2% of "
                  "their variance, the emotion 0.3%. Normalising embeddings per subject raised a simple probe from 0.204 to 0.225.",
                  "<b>Using only 12 channels cost about 0.05 macro-F1:</b> a published-style baseline (30 channels, spectral "
                  "features, linear SVM) scored 0.291 on raw FACED but 0.240 on our 12 channels.",
                  "<b>Hand-made spectral features carried more emotion information than the frozen EEGPT,</b> so the best run "
                  "uses both.",
                  "Labels, metrics, units and time handling were all checked and correct."])
story += [P("Best result in context (FACED test subjects, 7 classes)", "h2"),
          table([["Model or rule", "Accuracy", "Macro-F1", "Balanced acc.", "Per-clip acc."],
                 ["Uniform random guess", "13.9%", "0.131", "0.143", "—"],
                 ["Always guess 'happy' (largest class)", "32.2%", "0.070", "0.143", "—"],
                 ["Published-style baseline (30 ch, SVM)", "32.0%", "0.291", "—", "—"],
                 ["<b>Main CBM</b>", "<b>35.7%</b>", "<b>0.295</b>", "<b>0.303</b>", "<b>43.5%</b>"],
                 ["Flat baseline", "37.9%", "0.331", "0.324", "42.7%"]], [0.40, 0.14, 0.14, 0.16, 0.16]),
          P("<b>Per window</b> scores each 4-second window separately (the strict main metric). <b>Per clip</b> averages the 7 "
            "window predictions of a subject's 30-second clip into one prediction. Published FACED models reach 41–58% balanced "
            "accuracy on 9 classes, but they use all 32 channels and end-to-end training, so the numbers are not directly comparable.", "small"),
          P("Does interpretability cost accuracy? (H1 and ablation)", "h2"),
          table([["Variant (best run)", "flat", "L1 only", "bypass", "L1→L2", "L2+L3", "<b>L1→L2→L3 (main)</b>"],
                 ["Test macro-F1", "0.331", "0.322", "0.319", "0.312", "0.303", "<b>0.295</b>"]],
                [0.22, 0.11, 0.11, 0.11, 0.11, 0.11, 0.23], True, zebra=False),
          P("Each added concept level costs about one point. H1 (within 2 points of flat) is not met per window "
            "(−3.6 points, 95% CI −5.5 to −1.6); per clip the two models are equal.", "small")]

# ------------------------------------------------------------------ part 5
story += part(5, "Generalisation, explanations and limitations", 4,
              "On datasets the model has never seen, accuracy stays near chance, because the recordings differ strongly. The "
              "explanation stage works but must be rerun for the new model.")
story += [P("Generalisation to unseen datasets (zero-shot, H2)", "h2"),
          table([["Test set", "Concept model: valence / arousal accuracy", "Concept model vs flat (significant differences only)"],
                 ["DEAP", "50.8% / 54.9% (4 quadrants 27.3%)", "Arousal +1.7 points; valence (via emotion head) −1.1 points"],
                 ["DENS", "51.6% / 59.7%", "Valence (via emotion head) +5.2 points"],
                 ["DREAMER", "50.4% / 59.8% (7 classes about 25%)", "No significant difference"]], [0.14, 0.38, 0.48], True),
          P("Chance is about 50%. H2 is mostly not supported: the datasets differ strongly (a classifier tells them apart 68% of the time).", "small"),
          P("Explanations (Qwen2.5-1.5B, from the predicted concepts only) and FaithScore", "h2")]
story += bullets(["FaithScore was measured on the earlier DENS model: when valence was flipped, the concept model's explanations "
                  "changed correctly 60–81% of the time (two automatic judges), but a placebo change also altered wording 30–47% of "
                  "the time, so H3 is not yet supported. Qwen 1.5B sometimes contradicts the facts it is given.",
                  "For the FACED model this stage still has to be rerun (about 40 minutes on the GPU)."])
story += [P("Hypotheses: where we stand", "h2"),
          table([["", "Status"],
                 ["H1 accuracy close to flat", "Not met per window (−3.6 points); met per clip"],
                 ["H2 better transfer to new datasets", "Mostly not supported (2 better, 1 worse, rest equal)"],
                 ["H3 faithful explanations", "Not yet (DENS model); rerun pending"]], [0.35, 0.65], True),
          P("Limitations", "h2")]
story += bullets(["The interpretable model costs 3.6 points of macro-F1 per window.",
                  "Per-subject normalisation and smoothing use the test person's unlabelled windows (standard for FACED, but stated).",
                  "Similar emotions have almost identical valence/arousal ratings; Level 3 targets are pseudo-labels from appraisal theory."])

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm, bottomMargin=1.7 * cm,
                      title="H-CBM-EEG progress report", author="Pratham, Param, Neel, Arju, Leon", subject="Progress report")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=on_page)])
doc.build(story)
print("wrote", OUT)
