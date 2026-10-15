from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
from .config import DISCLAIMER


def metrics_frame(m: dict) -> pd.DataFrame:
    skip = {"sector_weights", "warnings", "return_convention"}
    return pd.DataFrame([{"metric": k, "value": v} for k, v in m.items() if k not in skip])


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def to_pdf(title: str, metrics: dict, comparison: pd.DataFrame | None, ai_report: dict | None, data_info: dict, warnings: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4); st = getSampleStyleSheet(); el = []
    el += [Paragraph(title, st["Title"]), Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", st["Normal"]), Spacer(1, 8),
           Paragraph(DISCLAIMER, st["Italic"]), Spacer(1, 10)]
    el.append(Paragraph("Data snapshot", st["Heading2"]))
    for k, v in data_info.items():
        el.append(Paragraph(f"<b>{k}:</b> {v}", st["Normal"]))
    def tbl(rows):
        t = Table(rows, hAlign="LEFT"); t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)])); return t
    fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and v != v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
    el += [Spacer(1, 8), Paragraph("Portfolio metrics", st["Heading2"]), tbl([["Metric", "Value"]] + [[r.metric, fmt(r.value)] for r in metrics_frame(metrics).itertuples()])]
    if comparison is not None and len(comparison):
        el += [Spacer(1, 8), Paragraph("Scenario comparison", st["Heading2"]),
               tbl([["Metric", "Current", "Proposed", "Abs chg", "Rel chg %"]] + [[r.metric, fmt(r.current), fmt(r.proposed), fmt(r.abs_change), fmt(r.rel_change_pct)] for r in comparison.itertuples()])]
    if ai_report:
        el.append(Paragraph("Explanation", st["Heading2"]))
        for k, items in ai_report.items():
            el.append(Paragraph(k.replace("_", " ").title(), st["Heading3"]))
            for it in (items if isinstance(items, list) else [items]):
                el.append(Paragraph(f"• {it}", st["Normal"]))
    if warnings:
        el.append(Paragraph("Warnings", st["Heading2"]))
        for w in warnings:
            el.append(Paragraph(f"• {w}", st["Normal"]))
    doc.build(el)
    return buf.getvalue()
