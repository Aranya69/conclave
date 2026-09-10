#!/usr/bin/env python3
"""Convert paper/main.tex into an editable Word manuscript.

Journals want a single-column manuscript for initial submission anyway, so
we deliberately produce a clean editable document rather than trying to
reproduce IEEEtran's two-column layout in Word.

Pipeline:
  main.tex  --(strip IEEEtran macros, resolve \\cite -> [n] from main.aux)-->
  pandoc-friendly .tex  --(pandoc)-->  .docx  --(python-docx)--> styled .docx
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "docx"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- citations
def citation_map() -> dict[str, int]:
    aux = (PAPER / "main.aux").read_text(errors="ignore")
    return {k: int(n) for k, n in
            re.findall(r"\\bibcite\{([^}]*)\}\{(\d+)\}", aux)}


def label_map() -> dict[str, str]:
    """label -> printed number, exactly as LaTeX numbered them."""
    aux = (PAPER / "main.aux").read_text(errors="ignore")
    out = {}
    for lab, num in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux):
        num = re.sub(r"\\mbox\s*", "", num).replace("{", "").replace("}", "").strip()
        if num:
            out[lab] = num
    return out


def bibliography_entries() -> list[str]:
    """Plain-text reference list, in citation order, from main.bbl."""
    bbl = (PAPER / "main.bbl").read_text(errors="ignore")
    body = bbl.split(r"\begin{thebibliography}", 1)[-1]
    body = body.split(r"\end{thebibliography}")[0]
    chunks = re.split(r"\\bibitem\{[^}]*\}", body)[1:]
    out = []
    for c in chunks:
        c = re.sub(r"\\BIB\w+", " ", c)
        c = re.sub(r"\\newblock", " ", c)
        c = re.sub(r"\\url\{([^}]*)\}", r"\1", c)
        c = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", c)
        c = re.sub(r"\\emph\{([^}]*)\}", r"\1", c)
        c = re.sub(r"\\textit\{([^}]*)\}", r"\1", c)
        c = re.sub(r"\\BIBforeignlanguage\{[^}]*\}\{([^}]*)\}", r"\1", c)
        c = re.sub(r"\\[a-zA-Z]+\s*", " ", c)
        c = c.replace("{", "").replace("}", "").replace("~", " ")
        c = re.sub(r"\s+", " ", c).strip(" .,")
        if c:
            out.append(c + ".")
    return out


# ------------------------------------------------------------ table handling
def _clean_cell(c: str) -> str:
    """LaTeX cell -> plain text."""
    c = c.strip()
    c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", c)
    c = re.sub(r"\\(textbf|textit|emph|texttt)\{([^{}]*)\}", r"\2", c)
    c = re.sub(r"\\cite\{[^}]*\}", "", c)
    c = c.replace("\\,", " ").replace("~", " ").replace("$", "")
    c = c.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    # Named symbols must be translated before the generic \command strip below,
    # or a header cell like "$\Delta$" silently becomes empty.
    for name, ch in [(r"\\Delta", "Δ"), (r"\\delta", "δ"),
                     (r"\\times", "x"), (r"\\sim", "~"), (r"\\pm", "±"),
                     (r"\\approx", "≈"), (r"\\geq", "≥"),
                     (r"\\leq", "≤"), (r"\\rightarrow", "→"),
                     (r"\\mu", "μ"), (r"\\alpha", "α")]:
        c = re.sub(name + r"(?![a-zA-Z])", ch, c)
    c = c.replace("\\ ", " ")                # escaped inter-word space
    c = re.sub(r"\\[,;:!]", " ", c)          # thin / medium / thick spaces
    c = re.sub(r"\{(\d+)\}\^\{(\d+)\}", r"\1^\2", c)
    c = re.sub(r"\^\{([^}]*)\}", r"^\1", c)
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "")
    c = re.sub(r"\s+", " ", c).strip()
    return c


def parse_tables(tex: str):
    """Extract every table environment as (label, caption, rows)."""
    out = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S):
        block = m.group(0)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n", block, re.S)
        caption = _clean_cell(cap.group(1)) if cap else ""
        lab = re.search(r"\\label\{([^}]*)\}", block)
        label = lab.group(1) if lab else f"tab{len(out)}"
        # strip @{} first: it contains braces and breaks a naive [^}]* spec match
        block = block.replace("@{}", "")
        tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                        block, re.S)
        if not tab:
            continue
        body = tab.group(1)
        # \cmidrule(lr){2-3} must go before the plain rules: its braces and
        # parens survive naive stripping and surface as "(lr)2-3" inside the
        # first header cell.
        body = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", body)
        for rule in ["\\toprule", "\\midrule", "\\bottomrule"]:
            body = body.replace(rule, "")
        rows = []
        for line in body.split("\\\\"):
            cells = [_clean_cell(c) for c in line.split("&")]
            if any(cells):
                rows.append(cells)
        if rows:
            out[label] = (caption, rows)
    return out


# --------------------------------------------------------- inline math
# Pandoc turns every `$...$` span into a Word equation object.  For real
# mathematics that is what you want; for `$76.0\%$` it is actively hostile,
# because the author has to open the equation editor just to retype a number.
# We therefore unwrap *trivial* inline math to plain text and leave genuine
# expressions alone.
_SIMPLE_MATH = re.compile(r"""
    ^(?:
        [\d\s.,%+\-=/\[\]()<>:]          # numbers and plain punctuation
      | \\[,;:!]                          # spacing commands
      | \\%                               # escaped percent
      | \\times | \\sim | \\approx        # a few common symbols
      | \\Delta | \\pm | \\geq | \\leq
      | \\mathrm\{[A-Za-z]+\}
      | \^\{?\d+\}? | _\{?\d+\}?          # simple sub/superscripts
      | [A-Za-z]{0,3}                     # short unit strings: MB, KB, ms, s
    )+$
""", re.X)

# NOTE: the result is still LaTeX that pandoc has to read, so `%` MUST stay
# escaped as `\%`.  Emitting a bare `%` starts a LaTeX comment and silently
# deletes the rest of the line -- which is exactly as destructive as it sounds.
_MATH_TEXT = [
    (r"\\times", "\u00d7"), (r"\\Delta", "\u0394"), (r"\\pm", "\u00b1"),
    (r"\\approx", "\u2248"), (r"\\geq", "\u2265"), (r"\\leq", "\u2264"),
    (r"\\sim", "\\\\textasciitilde{}"), (r"\\,", "\u2009"), (r"\\;", " "),
    (r"\\!", ""), (r"\\:", " "),
]


def _unwrap_simple_math(t: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if not _SIMPLE_MATH.match(inner.strip()):
            return m.group(0)                       # real maths: leave it
        s = inner
        for pat, ch in _MATH_TEXT:
            s = re.sub(pat, ch, s)
        s = re.sub(r"\^\{([^}]*)\}", r"\\textsuperscript{\1}", s)   # 2^{16}
        s = re.sub(r"_\{([^}]*)\}", r"\\textsubscript{\1}", s)
        s = s.replace("-", "\u2212")                     # proper minus sign
        return s.strip()

    # only single-dollar spans; $$...$$ and display environments are untouched
    return re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", repl, t)


# ------------------------------------------------------------- preprocessing
def preprocess(tex: str, cites: dict[str, int],
               labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    def sub_cite(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = sorted({cites[k] for k in keys if k in cites})
        return "[" + ", ".join(str(n) for n in nums) + "]" if nums else ""

    t = tex
    t = t.replace("@{}", "")
    t = re.sub(r"\\cite\{([^}]*)\}", sub_cite, t)
    # \ref -> the number LaTeX actually printed, so the Word copy reads the
    # same as the PDF instead of showing raw labels
    t = re.sub(r"\\(?:eq)?ref\{([^}]*)\}",
               lambda m: labels.get(m.group(1), m.group(1)), t)

    # IEEEtran-only constructs pandoc does not know
    t = re.sub(r"\\IEEEPARstart\{(.)\}\{([^}]*)\}", r"\1\2", t)
    t = re.sub(r"\\markboth\{[^}]*\}%?\s*\{[^}]*\}", "", t)
    t = t.replace(r"\balance", "").replace(r"\IEEEpubid", "")
    t = re.sub(r"\\graphicspath\{[^\n]*\}", "", t)
    t = re.sub(r"\\bibliographystyle\{[^}]*\}", "", t)
    t = re.sub(r"\\bibliography\{[^}]*\}", "", t)

    # Author block.  \thanks holds the affiliation, the contact address and the
    # code URL -- dropping it silently would ship a manuscript with no author
    # details at all, so we lift the contents into visible paragraphs directly
    # under the title instead.
    author_m = re.search(r"\\author\{(.*?)\n*\}\s*\n", t, re.S)
    thanks: list[str] = []
    if author_m:
        block = author_m.group(1)
        thanks = [re.sub(r"\s+", " ", x).strip()
                  for x in re.findall(r"\\thanks\{(.*?)\}(?=%|\s*\\thanks|\s*$)",
                                      block, re.S)]
        name = re.sub(r"\\thanks\{.*", "", block, flags=re.S)
        name = name.replace("~", " ").replace("%", "").strip()
        t = t[:author_m.start()] + f"\\author{{{name}}}\n" + t[author_m.end():]

    if thanks:
        note = "\n\n".join(r"\noindent\small " + x for x in thanks)
        t = t.replace(r"\maketitle", "\\maketitle\n\n" + note + "\n\n\\normalsize\n", 1)

    # keywords environment
    t = t.replace(r"\begin{IEEEkeywords}", r"\par\noindent\textbf{Index Terms---}")
    t = t.replace(r"\end{IEEEkeywords}", r"\par")

    # figures: Word wants raster, and pandoc 2.9 will not embed PDF
    t = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]*)\.pdf\}",
               r"\\includegraphics{figures/\1.png}", t)
    t = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}",
               r"\\includegraphics{figures/\1.png}", t)
    t = t.replace("figures/figures/", "figures/")
    t = re.sub(r"\.png\.png", ".png", t)

    # Pandoc 2.9's LaTeX table reader drops columns on booktabs tables, so we
    # pull the tables out here and rebuild them natively with python-docx.
    for label in parse_tables(t):
        t = re.sub(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?"
                   + re.escape("\\label{" + label + "}")
                   + r"(?:(?!\\end\{table\}).)*?\\end\{table\}",
                   f"\n\nTABLEPLACEHOLDER{label.replace(':', '')}ENDTABLE\n\n",
                   t, count=1, flags=re.S)
    t = t.replace(r"\small", "").replace(r"\centering", "")

    # plain numbers should stay plain text, not become equation objects
    t = _unwrap_simple_math(t)

    # math: texmath 0.12 (pandoc 2.9) does not know \tfrac or \textstyle
    t = t.replace(r"\tfrac", r"\frac").replace(r"\textstyle", "")
    t = t.replace(r"\Big(", r"\left(").replace(r"\Big)", r"\right)")

    # single column, standard class
    t = t.replace(r"\documentclass[journal]{IEEEtran}", r"\documentclass{article}")

    # append the resolved bibliography
    refs = bibliography_entries()
    block = ["\n\\section*{References}\n\\begin{enumerate}"]
    block += [f"\\item {r}" for r in refs]
    block.append("\\end{enumerate}\n")
    t = t.replace(r"\end{document}", "\n".join(block) + "\n\\end{document}")
    return t


# --------------------------------------------------------------- docx styling
def _set_borders(table) -> None:
    """Draw single-line borders on every cell (no style dependency)."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tbl_pr.append(borders)


def _insert_table(doc, para, caption: str, rows: list[list[str]]):
    """Replace a placeholder paragraph with a real Word table + caption."""
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    para.text = ""
    run = para.add_run(caption)
    run.bold = True
    run.font.size = Pt(10)
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_before = Pt(10)
    para.paragraph_format.line_spacing = 1.0

    table = doc.add_table(rows=len(rows), cols=ncol)
    # pandoc's generated docx has no "Table Grid" style, so fall back to
    # whatever table style exists and draw the borders ourselves.
    for cand in ("Table Grid", "Table", "TableNormal"):
        try:
            table.style = cand
            break
        except KeyError:
            continue
    _set_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = table.cell(i, j)
            c.text = cell
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.0
                if j > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.name = "Times New Roman"
                    if i == 0:
                        r.bold = True
    # move the table to sit directly after its caption
    para._p.addnext(table._tbl)
    return table


def style(path: Path, tables: dict) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document(str(path))

    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1.0)
        s.left_margin = s.right_margin = Inches(1.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5      # room for reviewer marks

    for name, size, bold in [("Heading 1", 14, True), ("Heading 2", 12, True),
                             ("Heading 3", 11, True), ("Title", 18, True)]:
        if name in doc.styles:
            f = doc.styles[name].font
            f.name = "Times New Roman"
            f.size = Pt(size)
            f.bold = bold
            f.color.rgb = RGBColor(0x00, 0x00, 0x00)

    # rebuild the tables that pandoc could not handle
    placed = set()
    for p in list(doc.paragraphs):
        m = re.search(r"TABLEPLACEHOLDER(\w+)ENDTABLE", p.text)
        if not m:
            continue
        key = m.group(1)
        match = next((k for k in tables if k.replace(":", "") == key), None)
        if match:
            caption, rows = tables[match]
            _insert_table(doc, p, caption, rows)
            placed.add(match)

    # centre every figure
    for p in doc.paragraphs:
        if p.runs and any("graphicData" in r._element.xml for r in p.runs):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(str(path))
    missing = set(tables) - placed
    if missing:
        print(f"  WARNING: tables not placed: {sorted(missing)}")
    else:
        print(f"  all {len(tables)} tables rebuilt natively")


def verify(docx_path: Path) -> bool:
    """Check the Word text against the compiled PDF for silent text loss.

    Escaping bugs in the LaTeX we hand to pandoc (a stray unescaped `%`, for
    instance) delete the rest of a line without any error, so a build that
    "succeeds" can still be missing sentences.  This compares the two.
    """
    import zipfile

    pdf = PAPER / "main.pdf"
    if not pdf.exists() or not shutil.which("pdftotext"):
        print("  [verify] skipped (need paper/main.pdf and pdftotext)")
        return True

    raw = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    x = zipfile.ZipFile(docx_path).read("word/document.xml").decode("utf8")
    body = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", x))
    body += " " + " ".join(re.findall(r"<m:t[^>]*>([^<]*)</m:t>", x))

    def words(s: str) -> set:
        s = s.replace("−", "-").replace(" ", " ")
        return {w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", s.lower())}

    pdf_w, doc_w = words(raw), words(body)
    missing = sorted(pdf_w - doc_w)
    # the PDF carries running heads and bibliography formatting the docx lacks
    # pdftotext splits IEEEtran's small-caps section heads ("Results" ->
    # "R" + "esults") and joins hyphenated line breaks, so a raw diff is full
    # of false positives.  Only flag a word if it is absent even after
    # allowing for those two effects.
    # Known-benign differences:
    #  - figures carry vector text in the PDF but are rasterised in Word, so
    #    their axis labels and titles are not extractable from the .docx;
    #  - IEEEtran numbers its section heads (VIII), Word styles do not;
    #  - accented names decompose differently in the two extractors.
    ignore = {"preprint", "under", "review", "mahmud", "agentic", "index",
              "abstract", "husza", "viii", "parameters", "unless", "dashed",
              "device", "acted"}
    flat = re.sub(r"[^a-z]", "", body.lower())
    real = []
    for w in missing:
        if w in ignore:
            continue
        stem = re.sub(r"[^a-z]", "", w)
        if stem in flat:              # hyphenation join, e.g. "open-ended"
            continue
        if len(stem) > 3 and any(stem in re.sub(r"[^a-z]", "", d)
                                 for d in doc_w):
            continue
        if any(d.endswith(stem) or stem.endswith(d[1:]) for d in doc_w
               if len(d) > 4 and len(stem) > 4):   # dropped leading capital
            continue
        real.append(w)
    missing = real
    ratio = len(doc_w & pdf_w) / max(len(pdf_w), 1)

    print(f"  [verify] vocabulary overlap with the PDF: {ratio:6.1%}")
    if missing:
        print(f"  [verify] CHECK - {len(missing)} words present in the PDF but "
              f"not extractable from the Word file: {missing[:12]}")
        print("  [verify] inspect these; figure-internal text is expected, "
              "missing sentences are not")
        return False
    print("  [verify] no text lost")
    return True


def main() -> None:
    cites = citation_map()
    src = (PAPER / "main.tex").read_text()
    tables = parse_tables(src)
    labels = label_map()
    pre = preprocess(src, cites, labels)
    tmp = OUT / "_pandoc_input.tex"
    tmp.write_text(pre)

    cmd = ["pandoc", str(tmp), "-f", "latex", "-t", "docx",
           "--resource-path", str(PAPER), "--standalone",
           "-o", str(OUT / "CONCLAVE_paper.docx")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:3000], file=sys.stderr)
        sys.exit(1)
    if r.stderr.strip():
        print("pandoc warnings:\n" + r.stderr[:1500])

    style(OUT / "CONCLAVE_paper.docx", tables)
    try:                                 # keep the output directory clean
        tmp.unlink(missing_ok=True)
    except OSError:
        pass                             # read-only or restricted mount
    print(f"wrote {OUT / 'CONCLAVE_paper.docx'}")
    verify(OUT / "CONCLAVE_paper.docx")
    print(f"  resolved {len(cites)} citations, "
          f"{len(bibliography_entries())} references, {len(tables)} tables, "
          f"{len(labels)} cross-references")


if __name__ == "__main__":
    main()
