import datetime

CURRENT_YEAR = datetime.datetime.now().year

AUTHOR = 'Luke Gessler'
SITENAME = 'Luke Gessler'
SITEURL = "https://lgessler.com"

# Basic configuration
PATH = "content"
TIMEZONE = 'America/New_York'
DEFAULT_LANG = 'en'

# Disable all feed generation
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Remove unused blog features
LINKS = ()
SOCIAL = ()
DEFAULT_PAGINATION = False
RELATIVE_URLS = True

# Static content handling
STATIC_PATHS = ['static']
STATIC_SAVE_AS = '{path}'
STATIC_URL = '{path}'

# Extension support
PLUGIN_PATHS = ['plugins']
PLUGINS = ['asciidoc_reader', 'plugins.bibliography_plugin']
READERS = {'asc': 'asciidoc_reader.AsciiDocReader'}
ASCIIDOC_OPTIONS = []
ASCIIDOC_BACKEND = 'html5'

# Turn off default templates (including index)
DIRECT_TEMPLATES = []

# Enable pages in menu
DISPLAY_PAGES_ON_MENU = True

# Custom menu configuration
MENUITEMS = (
    ('About', '/index.html'),
    ('Research', '/research.html'),  
    ('Teaching', '/teaching.html'),
    ('Contact', '/contact.html'),
)

# Disable categories and tags
CATEGORY_SAVE_AS = ''
AUTHOR_SAVE_AS = ''
TAG_SAVE_AS = ''

# Treat all content as pages - include the root content directory
ARTICLE_PATHS = []
PAGE_PATHS = ['']

# URL and path configurations
PATH_METADATA = '(?P<path_no_ext>.*)\..*'
SLUG_REGEX_SUBSTITUTIONS = [(r'[^\w/]+', '-')]
PAGE_URL = '{path_no_ext}.html'
PAGE_SAVE_AS = '{path_no_ext}.html'
INDEX_URL = '{path_no_ext}/index.html'
INDEX_SAVE_AS = '{path_no_ext}/index.html'

# Use basename for slug generation
SLUGIFY_SOURCE = 'basename'

# Use path as translation identifier
TRANSLATION_ID_METADATA = 'path'

# Field formatting
FORMATTED_FIELDS = ['summary', 'title', 'path', 'url', 'save_as']

# Custom content processor for handling AsciiDoc files
def content_filter(path, metadata):
    """Debug and process metadata for AsciiDoc files"""
    print(f"Processing {path}")
    print(f"Initial metadata: {metadata}")
    
    # Special handling for index.adoc files
    if path.endswith('index.adoc'):
        metadata['save_as'] = path.replace('index.adoc', 'index.html')
        metadata['url'] = path.replace('index.adoc', '')
        if metadata['url'] == '':
            metadata['url'] = '/'
        return True
    # For all other .adoc files
    elif path.endswith('.adoc'):
        html_path = path.replace('.adoc', '.html')
        metadata['save_as'] = html_path
        metadata['url'] = html_path
        return True
    return False

PROCESS_METADATA = content_filter

# Theme
THEME = 'themes/academic'