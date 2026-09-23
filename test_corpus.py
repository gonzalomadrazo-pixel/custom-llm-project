"""Test the exact notebook loader with real PDF bytes and UTF-8 text fixtures.
Run: python test_corpus.py. Fixtures stay in temporary directories, not corpus/.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from run_evals import load_suite, reject_eval_leakage, validate_corpus_location
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT/"custom_llm.py").read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"word_tokens", "chunk_text", "load_corpus_folder"}]
namespace = {"Path":Path, "hashlib":hashlib, "re":re,
             "language_suite":load_suite(), "reject_eval_leakage":reject_eval_leakage,
             "validate_corpus_location":validate_corpus_location}
exec(compile(ast.Module(body=functions,type_ignores=[]),"notebook-loader","exec"),namespace)
load_folder, chunk_text, word_tokens = (namespace[n] for n in ["load_corpus_folder","chunk_text","word_tokens"])

def make_pdf(path, text=None, blank_page=False, password=None):
    """Minimal test input with a real extractable PDF text stream, not mocked extraction."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612,height=792)
    if text is not None:
        font = DictionaryObject({NameObject("/Type"):NameObject("/Font"),NameObject("/Subtype"):NameObject("/Type1"),NameObject("/BaseFont"):NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):font})})
        safe = text.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 720 Td ({safe}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if blank_page:
        writer.add_blank_page(width=612,height=792)
    if password:
        writer.encrypt(password)
    writer.write(path)

class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def test_mixed_formats_and_nested_files(self):
        (self.root/"nested").mkdir()
        (self.root/"notes.TXT").write_text("Café customers order products.",encoding="utf-8")
        (self.root/"nested/notes.md").write_text("# Product notes\nBuyers compare price and quality.")
        make_pdf(self.root/"report.pdf","Clients review delivery and service.")
        chunks, manifest = load_folder(self.root)
        self.assertEqual(len(manifest["files"]),3)
        self.assertIn("café customers order products .",chunks)
        self.assertIn("clients review delivery and service .",chunks)
        self.assertIn("buyers compare price and quality .",chunks)
        self.assertTrue(all(len(row["sha256"])==64 for row in manifest["files"]))
    def test_chunking_preserves_all_tokens(self):
        text=" ".join(f"word{i}" for i in range(125))
        chunks=chunk_text(text)
        self.assertEqual([len(word_tokens(c)) for c in chunks],[47,47,31])
        self.assertEqual(word_tokens(" ".join(chunks)),word_tokens(text))
    def test_blank_pdf_and_corrupt_pdf_fail_clearly(self):
        make_pdf(self.root/"scan.pdf")
        with self.assertRaisesRegex(ValueError,"scan.pdf.*OCR"):
            load_folder(self.root)
        (self.root/"scan.pdf").unlink()
        (self.root/"corrupt.pdf").write_bytes(b"not a PDF")
        with self.assertRaisesRegex(ValueError,"corrupt.pdf"):
            load_folder(self.root)
    def test_mixed_pdf_pages_warn(self):
        make_pdf(self.root/"mixed.pdf","Readable customer sentence.",blank_page=True)
        chunks, manifest=load_folder(self.root)
        self.assertIn("readable customer sentence .",chunks)
        self.assertIn("Page 2",manifest["files"][0]["warnings"][0])
    def test_encrypted_pdf(self):
        make_pdf(self.root/"locked.pdf","A report.",password="test-fixture")
        with self.assertRaisesRegex(ValueError,"locked.pdf.*encrypted"):
            load_folder(self.root)
    def test_empty_and_invalid_utf8(self):
        (self.root/"empty.txt").write_text("  ")
        with self.assertRaisesRegex(ValueError,"empty.txt.*no readable text"):
            load_folder(self.root)
        (self.root/"empty.txt").unlink()
        (self.root/"invalid.txt").write_bytes(b"\xff\xfe\xff")
        with self.assertRaisesRegex(ValueError,"invalid.txt"):
            load_folder(self.root)
    def test_ignored_files_and_no_link_fetching(self):
        (self.root/"README.md").write_text("Ignore instruction file.")
        (self.root/".hidden.txt").write_text("Ignore hidden text.")
        (self.root/"image.png").write_bytes(b"not used")
        (self.root/"links.md").write_text("[Product](https://example.invalid/no-network)\nprint('not executed')")
        (self.root/"linked.txt").symlink_to(self.root/"links.md")
        _,manifest=load_folder(self.root)
        self.assertEqual([f["file"] for f in manifest["files"]],["links.md"])
        self.assertEqual(len(manifest["ignored"]),2)
    def test_empty_folder(self):
        chunks,manifest=load_folder(self.root)
        self.assertEqual(chunks,[])
        self.assertEqual(manifest["files"],[])
    def test_duplicates_are_reported(self):
        (self.root/"repeat.txt").write_text("Customer buys product.\nCustomer buys product.")
        _,manifest=load_folder(self.root)
        self.assertEqual(manifest["files"][0]["passages"],2)
        self.assertEqual(manifest["files"][0]["unique_passages"],1)

if __name__=="__main__":
    unittest.main(verbosity=2)
