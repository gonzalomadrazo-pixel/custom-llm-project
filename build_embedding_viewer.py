"""Refresh the standalone viewer's bundled reference data, without dependencies."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "examples/reference"
checkpoint_bytes = (RUN / "checkpoint.json").read_bytes()
checkpoint = json.loads(checkpoint_bytes)
inspection = json.loads((RUN / "inspection.json").read_text())
config = checkpoint["config"]
after = checkpoint["weights"]["wte"]
before = checkpoint["initial_embeddings"]
probe = inspection["token_id"]
assert before[probe] == inspection["embedding_before"], "Initialization differs from recorded evidence"
assert after[probe] == inspection["embedding_after"], "Checkpoint differs from recorded evidence"
payload = {
    "tokens": checkpoint["vocabulary"],
    "before": before,
    "after": after,
    "steps": checkpoint["completed_steps"],
    "name": "nanoGPT · classroom sentences",
    "tokenizer": "word",
    "run": RUN.name,
    "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
    "initialization": "Actual full embedding table captured before training; recorded probe verified.",
}
assert len(payload["tokens"]) == len(after)
page = ROOT / "embedding-viewer.html"
html = page.read_text()
packed = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
html, count = re.subn(r'(<script id="model-data" type="application/json">).*?(</script>)',
                     lambda m: m[1] + packed + m[2], html, count=1, flags=re.S)
assert count == 1
page.write_text(html)
print(f"Embedded {len(after)} actual {len(after[0])}D vectors, before and after {payload['steps']} steps.")
print(f"Checkpoint SHA-256: {payload['sha256']}")
