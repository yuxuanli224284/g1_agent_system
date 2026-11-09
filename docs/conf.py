# Sphinx configuration for the g1_agent_system documentation (Chinese).
# Build locally:  sphinx-build -b html docs docs/_build/html

project = "g1_agent_system"
author = "Yuxuan Li"
copyright = "2026, Yuxuan Li"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
language = "zh_CN"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "attrs_inline",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_title = "g1_agent_system 文档"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "use_repository_button": False,
    "show_toc_level": 2,
}
