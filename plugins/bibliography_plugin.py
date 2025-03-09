"""
Bibliography Plugin for Pelican
Adds support for bibliography directives in AsciiDoc files.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

from pelican import signals
from .asciidoc_reader import AsciiDocReader
from pybtex.database.input import bibtex
from pybtex.database import BibliographyData
from pybtex.backends import html
from pybtex.style.formatting import BaseStyle
from pybtex.style.template import field, optional, words, sentence, join, tag, optional_field
from pybtex.richtext import Text
from pybtex.backends.html import Backend

def register():
    """Register the plugin with Pelican."""
    signals.readers_init.connect(add_asciidoc_directive)

def add_asciidoc_directive(readers):
    """Replace the default AsciiDocReader with our enhanced version."""
    original_reader = readers.reader_classes.get('adoc', AsciiDocReader)
    readers.reader_classes['adoc'] = lambda *args, **kwargs: BibliographyAsciiDocReader(original_reader, *args, **kwargs)

class CustomStyle(BaseStyle):
    """Custom bibliography style to match the desired format."""
    
    def format_names(self, persons):
        """Format the names of persons in the bibliography entry."""
        from pybtex.style.template import join, words, field
        from pybtex.style.names import plain
        
        name_style = plain.NameStyle()
        formatted_names = []
        
        for person in persons:
            formatted_names.append(name_style.format(person, abbr=False))
        
        if not formatted_names:
            return words()
        
        return join(formatted_names, ' and ')
    
    def format_title(self, entry, which_field):
        title = entry.fields.get(which_field, "")
        if not title:
            return Text()
        
        # Format the title
        formatted_title = Text(title)
        
        # If there's a URL, make the title a hyperlink
        url = entry.fields.get('url', '')
        if url:
            return tag('a', {'href': url})[formatted_title]
        
        return formatted_title
    
    def format_entry(self, label, entry, bib_data):
        # Prepare the components
        author = self.format_names(entry.persons.get('author', []))
        year = field('year')
        title = self.format_title(entry, 'title')
        
        # For conference/proceedings papers
        booktitle = field('booktitle')
        pages = optional_field('pages', prefix='pages ', suffix='')
        address = optional_field('address', prefix='', suffix='')
        publisher = optional_field('publisher', prefix='', suffix='')
        
        # Format the parts of the citation
        parts = [
            sentence(author, period=True),
            sentence(year, period=True),
            sentence(title, period=True),
            words('In', booktitle, optional[words('pages', pages)], address),
            sentence(publisher)
        ]
        
        return join(parts)

class BibliographyAsciiDocReader:
    """Enhanced AsciiDoc reader that supports bibliography directives."""
    
    def __init__(self, original_reader_class, *args, **kwargs):
        """Initialize the reader with the original AsciiDoc reader."""
        # Create an instance of the original reader
        self.reader = original_reader_class(*args, **kwargs)
        # Add our functionality to the reader's process
        self._original_read = self.reader.read
        self.reader.read = self._read_with_bibliography
        
        # Path to the bundled acl.bst file
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.bst_path = os.path.join(self.plugin_dir, 'acl.bst')
        
        # Check if the BST file exists
        if not os.path.exists(self.bst_path):
            print("WARNING: acl.bst not found in plugin directory. The bibliography plugin may not work correctly.")
    
    def _read_with_bibliography(self, source_path):
        """Read the source file and process any bibliography directives."""
        content, metadata = self._original_read(source_path)
        
        # Process any bibliography directives in the content
        content = self._process_bibliography_directives(content, os.path.dirname(source_path))
        
        return content, metadata
    
    def _process_bibliography_directives(self, content, base_path):
        """Find and process bibliography directives in the content."""
        # Pattern for bibliography directive
        pattern = r'bibliography::([^\[]+?)\[([^\]]*?)\]'
        
        def replace_bibliography(match):
            bib_file_path = match.group(1).strip()
            # Style options are ignored since we bundle the acl.bst
            
            full_bib_path = os.path.join(base_path, "static", bib_file_path)
            
            if not os.path.exists(full_bib_path):
                return f'<div class="error">Error: BibTeX file not found: {bib_file_path}</div>'
            
            return self._process_with_bibtex(full_bib_path)
        
        return re.sub(pattern, replace_bibliography, content)
    
    def _process_with_bibtex(self, bib_path):
        """Process bibliography with the bundled ACL BST file."""
        # Create a temporary directory for our work
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define temp file paths
            temp_dir_path = Path(temp_dir)
            tex_path = temp_dir_path / "temp.tex"
            aux_path = temp_dir_path / "temp.aux"
            bbl_path = temp_dir_path / "temp.bbl"
            temp_bib_path = temp_dir_path / "temp.bib"
            temp_bst_path = temp_dir_path / "temp.bst"
            
            # Copy the bib file
            with open(bib_path, 'r', encoding='utf-8') as src, \
                    open(temp_bib_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            
            # Copy the bundled BST file
            with open(self.bst_path, 'r', encoding='utf-8') as src, \
                    open(temp_bst_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            
            # Create the LaTeX document
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(r'''\documentclass{article}
\begin{document}
\nocite{*}
\bibliographystyle{temp}
\bibliography{temp}
\end{document}
''')
            
            # Create the aux file
            with open(aux_path, 'w', encoding='utf-8') as f:
                f.write(r'''\relax
\citation{*}
\bibstyle{temp}
\bibdata{temp}
''')
            
            # Run bibtex
            bibtex_proc = subprocess.run(
                ["bibtex", "temp"],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            
            if bibtex_proc.returncode != 0:
                return f'<div class="error">BibTeX error: {bibtex_proc.stderr}</div>'
            
            # Read the generated BBL file
            if os.path.exists(bbl_path):
                return self._process_bbl_file(bbl_path)
            else:
                return '<div class="error">BibTeX did not generate a BBL file</div>'
    
    def _process_bbl_file(self, bbl_path):
        """Extract and convert bibliography entries from a BBL file to HTML."""
        with open(bbl_path, 'r', encoding='utf-8') as f:
            bbl_content = f.read()
        
        # Pattern to match the bibliography environment
        bib_env_pattern = r'\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}'
        bib_match = re.search(bib_env_pattern, bbl_content, re.DOTALL)
        
        if bib_match:
            # Get the content excluding the begin/end tags and any arguments
            bib_entries_content = bib_match.group(1).strip()
            
            # Split into individual bibliography entries
            bib_items = re.split(r'(?=\\bibitem)', bib_entries_content)
            bib_items = [item for item in bib_items if item.strip()]
            
            # Extract year from each entry for sorting
            def extract_year(entry):
                # Look for a year pattern (4 digits, typically in brackets or after certain markers)
                year_match = re.search(r'(\b|\D)(19|20)\d{2}(\b|\D)', entry)
                if year_match:
                    # Extract just the 4-digit year from the match
                    digit_match = re.search(r'(19|20)\d{2}', year_match.group(0))
                    if digit_match:
                        return int(digit_match.group(0))
                return 0  # Default for entries without a year
            
            # Sort entries by year in reverse chronological order
            sorted_items = sorted(bib_items, key=extract_year, reverse=True)
            
            # Convert each LaTeX entry to HTML
            html_entries = [self._latex_to_html(entry) for entry in sorted_items]
            
            # Join all entries into a single HTML string
            bib_entries_html = ''.join(html_entries)
            
            return f'<div class="bibliography"><ul>{bib_entries_html}</ul></div>'
        else:
            return '<div class="error">Could not extract bibliography entries from BibTeX output</div>'
    
    def _latex_to_html(self, latex_text):
        """Convert LaTeX formatting to HTML."""
        # Remove LaTeX command definitions
        latex_text = re.sub(r'\\providecommand\{\\natexlab\}\[1\]\{#1\}', '', latex_text)
        
        # Replace LaTeX commands with HTML
        html_text = latex_text
        
        # Handle \bibitem
        html_text = re.sub(
            r'\\bibitem\[([^\]]*)\]\{([^}]*)\}',
            r'<li class="bibliography-entry" id="bib-\2">',
            html_text
        )
        
        # Close each bibliography entry li
        html_text = re.sub(r'(?=\\bibitem|$)', r'</li>', html_text)
        
        # Remove tildes (non-breaking spaces in LaTeX)
        html_text = html_text.replace('~', ' ')
        
        # Process \href commands
        html_text = self._process_href_commands(html_text)
        
        # Remove LaTeX-specific commands
        html_text = re.sub(r'\\newblock', '', html_text)
        html_text = re.sub(r'\\natexlab\{([^}]*)\}', '', html_text)
        
        # Handle LaTeX emphasis
        html_text = re.sub(r'\\emph\{([^}]*)\}', r'<em>\1</em>', html_text)
        
        # Remove braces
        prev_text = ""
        while prev_text != html_text:
            prev_text = html_text
            html_text = re.sub(r'\{([^{}]*?)\}', r'\1', html_text)
        
        # Handle special characters
        html_text = html_text.replace('--', '–')
        html_text = html_text.replace('---', '—')
        
        # Clean up remaining LaTeX commands
        html_text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', html_text)
        
        # Clean up whitespace
        html_text = re.sub(r'\s+', ' ', html_text)
        html_text = html_text.strip()
        
        return html_text
    
    def _process_href_commands(self, text):
        """Process \href LaTeX commands to HTML <a> tags, handling nested braces properly."""
        result = ""
        i = 0
        while i < len(text):
            # Look for \href
            href_match = text.find('\\href', i)
            if href_match == -1:
                # No more \href found
                result += text[i:]
                break
            
            # Add text up to \href
            result += text[i:href_match]
            i = href_match
            
            # Find URL opening brace
            url_start = text.find('{', i)
            if url_start == -1:
                # Malformed \href without opening brace
                result += text[i]
                i += 1
                continue
            
            # Find URL closing brace with brace counting
            url_end = self._find_matching_brace(text, url_start)
            if url_end == -1:
                # Unclosed URL brace
                result += text[i]
                i += 1
                continue
            
            # Extract URL
            url = text[url_start+1:url_end]
            
            # Find title opening brace
            title_start = text.find('{', url_end + 1)
            if title_start == -1:
                # No title brace found
                result += text[i:url_end+1]
                i = url_end + 1
                continue
            
            # Find title closing brace with brace counting
            title_end = self._find_matching_brace(text, title_start)
            if title_end == -1:
                # Unclosed title brace
                result += text[i:title_start+1]
                i = title_start + 1
                continue
            
            # Extract title
            title = text[title_start+1:title_end]
            
            # Create an HTML link
            result += f'<a href="{url}">{title}</a>'
            
            # Move past this \href
            i = title_end + 1
        
        return result
    
    def _find_matching_brace(self, text, start_pos):
        """Find the matching closing brace for an opening brace at start_pos."""
        brace_count = 1
        j = start_pos + 1
        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return j
            j += 1
        return -1
    
    # Delegate all other method calls to the original reader
    def __getattr__(self, name):
        return getattr(self.reader, name) 