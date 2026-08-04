"""
Bibliography plugin for Pelican.

Adds a ``bibliography::<file>.bib[]`` directive to AsciiDoc pages. The named
file is resolved relative to ``content/static/`` and rendered newest-first,
grouped under year headings, as title / authors / venue blocks.

Three optional fields drive the site rendering and are ignored by LaTeX:
``venue`` (short label such as ACL or COLING; falls back to the full
booktitle), ``code`` (repo URL, rendered as a "code" link), and ``award``
(shows a trophy before the title and the award name after the venue).

This is a pure-Python renderer: it parses the .bib with pybtex and decodes
LaTeX markup with pylatexenc. It deliberately does NOT shell out to ``bibtex``
or use a .bst file, so the site builds without a TeX installation.

The .bib is a shared source of truth with the Overleaf CV, which renders the
same file through biblatex. Fields only the CV cares about (``keywords``,
``editor``, ``abstract``, ``ISBN``) are simply ignored here.
"""
import html
import os
import re

from pelican import signals
from pybtex.database.input import bibtex
from pylatexenc.latex2text import LatexNodes2Text

from .asciidoc_reader import AsciiDocReader

# Name to render in bold in author lists (i.e. the site owner).
BOLD_LAST = 'Gessler'
BOLD_FIRST = 'Luke'

_L2T = LatexNodes2Text(keep_comments=False, math_mode='text')


def register():
    """Register the plugin with Pelican."""
    signals.readers_init.connect(add_asciidoc_directive)


def add_asciidoc_directive(readers):
    """Replace the default AsciiDocReader with our enhanced version."""
    original_reader = readers.reader_classes.get('adoc', AsciiDocReader)
    readers.reader_classes['adoc'] = (
        lambda *args, **kwargs: BibliographyAsciiDocReader(original_reader, *args, **kwargs)
    )


def latex_to_text(value):
    """Decode a raw BibTeX field into plain Unicode text.

    Handles brace-protected capitals ({NLP} -> NLP), accents ({\\'e} -> é) and
    escapes (\\& -> &). Collapses the line wrapping that .bib files carry.
    """
    if not value:
        return ''
    text = _L2T.latex_to_text(value)
    return re.sub(r'\s+', ' ', text).strip()


def _esc(value):
    return html.escape(value, quote=False)


def _format_person(person):
    """Format one author, bolding the site owner."""
    first = latex_to_text(' '.join(person.first_names + person.middle_names))
    last = latex_to_text(' '.join(person.prelast_names + person.last_names))
    name = _esc(f'{first} {last}'.strip())
    if last == BOLD_LAST and first.startswith(BOLD_FIRST):
        return f'<strong>{name}</strong>'
    return name


def _format_authors(persons):
    """Join authors with serial-comma 'and'."""
    names = [_format_person(p) for p in persons]
    if not names:
        return ''
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + f', and {names[-1]}'


def entry_year(entry):
    """Extract a sortable integer year from an entry, 0 if absent."""
    match = re.search(r'(\d{4})', entry.fields.get('year', ''))
    return int(match.group(1)) if match else 0


def _entry_url(fields):
    """Preferred link target for an entry: explicit URL, else the DOI."""
    url = latex_to_text(fields.get('url', ''))
    if not url and fields.get('doi'):
        url = 'https://doi.org/' + latex_to_text(fields['doi'])
    return url


def _venue_label(entry):
    """Short venue label. Falls back to the full booktitle/journal if the
    entry has no `venue` field, so a new entry is never left blank."""
    fields = entry.fields
    venue = latex_to_text(fields.get('venue', ''))
    if venue:
        return venue
    if entry.type == 'article':
        return latex_to_text(fields.get('journal', '') or fields.get('journaltitle', ''))
    return latex_to_text(fields.get('booktitle', ''))


def format_entry(key, entry):
    """Render one entry as a title / authors / venue block."""
    fields = entry.fields
    rows = []

    title = _esc(latex_to_text(fields.get('title', '')))
    award = latex_to_text(fields.get('award', ''))
    url = _entry_url(fields)
    if url:
        title = f'<a href="{html.escape(url, quote=True)}">{title}</a>'
    if award:
        title = f'<span class="pub-award" title="{_esc(award)}">&#127942;</span> {title}'
    rows.append(f'<div class="pub-title">{title}</div>')

    authors = _format_authors(entry.persons.get('author', []))
    if authors:
        rows.append(f'<div class="pub-authors">{authors}</div>')

    # Venue line: "VENUE (Award) · code"
    meta = []
    venue = _venue_label(entry)
    if venue:
        label = _esc(venue)
        if award:
            label += f' ({_esc(award)})'
        meta.append(f'<span class="pub-venue">{label}</span>')
    code = latex_to_text(fields.get('code', ''))
    if code:
        meta.append(f'<a href="{html.escape(code, quote=True)}">code</a>')
    if meta:
        rows.append('<div class="pub-meta">' + ' &middot; '.join(meta) + '</div>')

    anchor = html.escape(key, quote=True)
    return f'<div class="pub" id="bib-{anchor}">' + ''.join(rows) + '</div>'


def render_bibliography(bib_path):
    """Parse a .bib file and render it grouped under year headings."""
    parser = bibtex.Parser()
    bib_data = parser.parse_file(bib_path)

    # Newest first; ties broken by citekey so builds are reproducible.
    ordered = sorted(
        bib_data.entries.items(),
        key=lambda kv: (-entry_year(kv[1]), kv[0]),
    )

    out = []
    current_year = None
    for key, entry in ordered:
        year = entry_year(entry)
        if year != current_year:
            current_year = year
            label = str(year) if year else 'Other'
            out.append(f'<h2 class="pub-year">{label}</h2>')
        out.append(format_entry(key, entry))
    return '<div class="bibliography">' + ''.join(out) + '</div>'


class BibliographyAsciiDocReader:
    """AsciiDoc reader that expands ``bibliography::<file>[]`` directives."""

    DIRECTIVE = re.compile(r'bibliography::([^\[]+?)\[([^\]]*?)\]')

    def __init__(self, original_reader_class, *args, **kwargs):
        self.reader = original_reader_class(*args, **kwargs)
        self._original_read = self.reader.read
        self.reader.read = self._read_with_bibliography

    def _read_with_bibliography(self, source_path):
        content, metadata = self._original_read(source_path)
        content = self._process_directives(content, os.path.dirname(source_path))
        return content, metadata

    def _process_directives(self, content, base_path):
        def replace(match):
            bib_file = match.group(1).strip()
            bib_path = os.path.join(base_path, 'static', bib_file)
            if not os.path.exists(bib_path):
                return f'<div class="error">Error: BibTeX file not found: {_esc(bib_file)}</div>'
            try:
                return render_bibliography(bib_path)
            except Exception as exc:  # noqa: BLE001 - surface any parse failure on the page
                return f'<div class="error">Error rendering {_esc(bib_file)}: {_esc(str(exc))}</div>'

        return self.DIRECTIVE.sub(replace, content)

    def __getattr__(self, name):
        return getattr(self.reader, name)
