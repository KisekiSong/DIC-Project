"""Render report.md as a two-page A4 PDF and check the result.

Run from the project root after the pipeline: .venv/bin/python report/build_report.py
Figure 1 is drawn from output/clustering.json, so it always shows the model the text describes."""
import json
import re
import sys
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from profiles import colour_of  # noqa: E402  same pattern colours as the HTML report
SOURCE = HERE / 'report.md'
OUTPUT = HERE / 'report.pdf'
FIGURE = HERE / 'build' / 'figure1.png'
MAX_PAGES = 2
FONTS = Path('/System/Library/Fonts/Supplemental')
# Display names used in Table 1 and Figure 1, keyed by the labels clustering.py derives from the centres.
PATTERN_NAMES = {
    'daytime plateau 09-22h, medium base': 'Broad daytime plateau (09–22 h)',
    'evening peak 19-23h, medium base': 'Evening peak (19–23 h)',
    'flat all day': 'Flat all day',
    'daytime plateau 11-22h, low base': 'Low-base plateau (11–22 h)',
    'daytime plateau 09-19h, medium base': 'Office hours (09–19 h)',
}
# Markers keep the curves apart if the page is printed in black and white.
MARKERS = ['o', 's', '^', 'D', 'v']

for name, filename in (('Serif', 'Times New Roman.ttf'), ('SerifBold', 'Times New Roman Bold.ttf'),
                       ('SerifItalic', 'Times New Roman Italic.ttf')):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
pdfmetrics.registerFontFamily('Serif', normal='Serif', bold='SerifBold', italic='SerifItalic')

STYLES = {
    'title': ParagraphStyle('title', fontName='SerifBold', fontSize=15, leading=18, spaceAfter=3),
    'meta': ParagraphStyle('meta', fontName='Serif', fontSize=9.5, leading=11.5, spaceAfter=6),
    'heading': ParagraphStyle('heading', fontName='SerifBold', fontSize=11, leading=13, spaceBefore=5,
                              spaceAfter=2, keepWithNext=True),
    'body': ParagraphStyle('body', fontName='Serif', fontSize=10, leading=12, spaceAfter=3.5, alignment=TA_LEFT,
                           allowWidows=0, allowOrphans=0),
    'caption': ParagraphStyle('caption', fontName='Serif', fontSize=9, leading=10.8, spaceBefore=2, spaceAfter=2),
    'cell': ParagraphStyle('cell', fontName='Serif', fontSize=9, leading=10.5),
    'reference': ParagraphStyle('reference', fontName='Serif', fontSize=9, leading=10.8, spaceAfter=1.5),
}


def inline(text):
    """Escape HTML and turn *italic* into ReportLab markup."""
    return re.sub(r'\*([^*]+)\*', r'<i>\1</i>', escape(text))


def caption(text):
    """Bold 'Table 1.' / 'Figure 1.' label followed by the caption text."""
    label, _, rest = text.partition('. ')
    return Paragraph(f'<b>{inline(label)}.</b> {inline(rest)}', STYLES['caption'])


def draw_figure(path):
    font_manager.fontManager.addfont(str(FONTS / 'Times New Roman.ttf'))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5})
    result = json.loads((ROOT / 'output' / 'clustering.json').read_text())
    share = {c['cluster']: result['train_shares'][str(c['cluster'])] for c in result['centres']}
    centres = sorted(result['centres'], key=lambda c: share[c['cluster']], reverse=True)
    unknown = [c['label'] for c in centres if c['label'] not in PATTERN_NAMES]
    assert not unknown, f'name these patterns in PATTERN_NAMES: {unknown}'
    fig, ax = plt.subplots(figsize=(6.6, 2.15))
    for centre, marker in zip(centres, MARKERS):
        values = [v * 100 for v in centre['centre']]
        label = f"{PATTERN_NAMES[centre['label']]}, {share[centre['cluster']]:.0%}"
        ax.plot(range(24), values, color=colour_of(centre['cluster']), marker=marker, markersize=2.8,
                linewidth=1.3, label=label)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('% of daily energy')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def table(rows):
    cells = []
    for row in rows:
        cells.append([Paragraph(inline(cell), STYLES['cell']) for cell in row])
    widths = [None] * len(rows[0])
    widths[0] = 6.6 * cm
    grid = Table(cells, colWidths=widths, hAlign='LEFT')
    grid.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return grid


def table_rows(block):
    rows = []
    for line in block.splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if not all(set(cell) <= {'-', ':'} for cell in cells):  # skip the |---| separator row
            rows.append(cells)
    return rows


def flowables(source):
    story, pending_caption, in_references = [], None, False
    for block in re.split(r'\n\s*\n', source.strip()):
        block = block.strip()
        if block.startswith('# '):
            story.append(Paragraph(inline(block[2:]), STYLES['title']))
        elif block.startswith('## '):
            in_references = block[3:] == 'References'
            story.append(Paragraph(inline(block[3:]), STYLES['heading']))
        elif block.startswith('ID2221'):
            story.append(Paragraph(inline(block), STYLES['meta']))
        elif re.match(r'Table \d+\. ', block):
            pending_caption = caption(block)
        elif block.startswith('|'):
            story.append(KeepTogether([pending_caption, table(table_rows(block)), Spacer(1, 4)]))
            pending_caption = None
        elif block.startswith('!['):
            text = re.match(r'!\[(.+)\]\(figure:centres\)', block).group(1)
            draw_figure(FIGURE)
            image = Image(str(FIGURE), width=16.4 * cm, height=16.4 * cm * 0.26, kind='proportional')
            story.append(KeepTogether([Spacer(1, 2), image, caption(text), Spacer(1, 3)]))
        else:
            style = STYLES['reference'] if in_references else STYLES['body']
            story.append(Paragraph(inline(block), style))
    return story


def check(source):
    assert not re.search(r'\b(?:TODO|TBD|PLACEHOLDER|XXX)\b', source), 'placeholder left in report.md'
    for ref in sorted(set(re.findall(r'\[(\d+)\]', source))):
        assert source.count(f'[{ref}]') >= 2, f'reference [{ref}] is listed but never cited'
    pages = len(PdfReader(OUTPUT).pages)
    assert pages <= MAX_PAGES, f'{pages} pages; the report may have at most {MAX_PAGES}'
    words = len(re.findall(r"[A-Za-z0-9][\w'.%–-]*", source))
    print(f'{OUTPUT.relative_to(ROOT)}: {pages} page(s), about {words} words in the source')


def main():
    source = SOURCE.read_text(encoding='utf-8')
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                 topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                                 title='Discovering Daily Electricity Use Patterns with Apache Spark')
    document.build(flowables(source))
    check(source)


if __name__ == '__main__':
    main()
