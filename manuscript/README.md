# Textbook Manuscript Export

This directory holds the compiled Microsoft Word manuscript for the textbook
**AI in Biological Sciences: Theory, Applications, Practice, and Society**
(DaScient Press).

## Files

| File | Description |
|------|-------------|
| `AI_in_Biological_Sciences.docx` | Full manuscript: front matter + all 24 chapters across the 8 Parts, with equations, formulas, code blocks, tables, and figures-as-diagram source. |

## How it is generated

The `.docx` is compiled directly from the chapter Markdown under
`docs/source/chapters/` in the exact order declared by the MkDocs `nav`
(`mkdocs.yml`), preserving the Part groupings (Part I–VIII).

```bash
# from the repository root
make manuscript
# or, equivalently
python tools/build_manuscript.py --output manuscript/AI_in_Biological_Sciences.docx
```

The compiler ([`tools/build_manuscript.py`](../tools/build_manuscript.py))
preserves the structure expected by the DaScient, Inc. textbook pipeline:

- **Headings / subheadings / sections** mapped to Word heading styles so Word's
  Table of Contents field can paginate them automatically.
- **Equations & formulas** kept verbatim (including inline `$…$` math and the
  Unicode symbols used in the sources, e.g. `H(X) = -Σ p(x) log₂ p(x)`).
- **Code blocks** rendered in a shaded monospace style with a language label;
  Mermaid blocks are preserved as labelled diagram source.
- **Tables, lists, block quotes** and inline `code`/**bold**/*italic*.

## Inputs still owned by DaScient, Inc. (TBD)

The front matter intentionally contains `[TBD - DaScient, Inc.]` placeholders for
values the publisher supplies before final processing:

- ISBN (print and e-book)
- Copyright holder / context and edition
- Final paginated Table of Contents (a live Word TOC field is embedded —
  right-click it in Word and choose **Update Field** to populate page numbers)

Update these in `tools/build_manuscript.py` (the constants and front-matter
section) and re-run `make manuscript`.
