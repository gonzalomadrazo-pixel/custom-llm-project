# Add your training material here

Put your PDF, TXT and Markdown (.md) files in this folder. Subfolders work too.
Keep this README here: the loader ignores it. It does not follow links or execute
code in your documents.

- Default: CORPUS = "classroom" combines the teaching sentences with these files.
- Files only: set CORPUS = "folder" before running. This needs at least 100 unique passages.
- Run All after adding/changing files. The model retrains from scratch, then export
  its new checkpoint.json to the embedding viewer. Adding files alone changes no weights.
- In Colab, run sections 1 and 2 once to create /content/corpus. Upload files into that folder
  using the Files sidebar, then Run All. Opening a notebook from GitHub does not copy
  the repository's folders or upload your computer's files automatically.

PDFs must contain selectable text. Scans need OCR first; encrypted, corrupt and
textless files stop the import with an explanation. Empty pages in otherwise readable
PDFs are reported. Complex layouts can extract in the wrong reading order: inspect
the saved corpus.txt and file previews.

Long text is split into passages of at most 47 word/punctuation tokens, with no
overlap or silent truncation. Markdown is read as plain text, so formatting may remain.
The small model keeps the 509 most frequent training token types; other words become
UNK. Inspect the reported coverage before trusting the model's output.

Limits: 50 supported files, 25 MB per file, 100 MB total, 200 pages per PDF and
2 million extracted characters per file. Unsupported/hidden files and symbolic links
are not ingested. These are classroom limits, not a large-document pipeline.

Files here are ignored by Git by default to prevent accidental publication. The
results ZIP STILL contains extracted text, file names/hashes and model weights.
Use only material you may process/share, and review the ZIP before publishing.
