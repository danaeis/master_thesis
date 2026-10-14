#!/bin/bash
# Build with the Mac's installed MacTeX, preserving any existing PDF on failure.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$PWD"
export PATH="/Library/TeX/texbin:/usr/local/texlive/2025/bin/universal-darwin:$PATH"
finish() {
  code=$?
  if [ "$code" -ne 0 ]; then
    printf '\nBuild failed. The source files and any previous PDF were preserved.\n'
  fi
  if [ -t 0 ]; then read -r -p 'Press Return to close this window.' _ || true; fi
}
trap finish EXIT
for tool in latexmk xelatex kpsewhich; do
  if ! command -v "$tool" >/dev/null; then
    printf 'Required MacTeX tool is missing: %s\n' "$tool"
    exit 1
  fi
done
for package in xepersian.sty bidi.sty; do
  if ! kpsewhich "$package" >/dev/null; then
    printf 'Required TeX package is missing: %s\n' "$package"
    exit 1
  fi
done
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/thesis-build.XXXXXX")"
printf 'Compiling with: %s\nBuild directory: %s\n' "$(command -v xelatex)" "$BUILD"
# Duplicate the project so clean/build operations never remove its previous PDF.
rsync -a --exclude='.git' --exclude='snapshots' --exclude='thesis.pdf' "$PROJECT/" "$BUILD/"
cd "$BUILD"
latexmk -C thesis.tex
set +e
latexmk -xelatex -f -interaction=nonstopmode thesis.tex >build-console.log 2>&1
status=$?
set -e
cp build-console.log "$PROJECT/thesis-build-console.log"
if [ -f thesis.log ]; then cp thesis.log "$PROJECT/thesis-build-diagnostics.log"; fi
if [ "$status" -ne 0 ] || [ ! -s thesis.pdf ]; then
  tail -60 build-console.log
  printf '\nDiagnostics saved in the thesis folder. Build files retained at %s\n' "$BUILD"
  exit 1
fi
if command -v pdfinfo >/dev/null; then pdfinfo thesis.pdf; fi
if command -v pdftotext >/dev/null; then
  pdftotext thesis.pdf "$BUILD/thesis-verification.txt"
  test -s "$BUILD/thesis-verification.txt"
fi
cp thesis.pdf "$PROJECT/thesis.pdf.new"
mv -f "$PROJECT/thesis.pdf.new" "$PROJECT/thesis.pdf"
printf '\nPDF produced successfully: %s/thesis.pdf\n' "$PROJECT"
printf '\nRemaining layout or LaTeX warnings:\n'
grep -E 'Warning|Overfull|Underfull' thesis.log || true
open "$PROJECT/thesis.pdf"
