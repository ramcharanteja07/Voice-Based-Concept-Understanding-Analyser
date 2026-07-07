from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def build_report(reference_text, transcribed_text, waveform_path, metrics, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Reference Concept</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(reference_text, styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Student Transcription</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(transcribed_text, styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Audio Visualization</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    img = Image(waveform_path, width=450, height=150)
    img.hAlign = 'CENTER'
    content.append(img)
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Evaluation Summary</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    data = [["Metric", "Value"]] + [[k, v] for k, v in metrics.items()]

    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    content.append(table)
    doc.build(content)
    return output_path