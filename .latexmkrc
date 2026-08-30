$pdf_mode = 5;          # xelatex -> pdf
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;        # always rerun bibtex when the .bib files change
$pdf_previewer = 'open -a Preview %O %S';
