"""Validate frozen upstream sources before model initialization."""
from common import ROOT, VENDOR, read, sha

def check_native():
    for row in read(ROOT/'provenance/vendor_files.json')['files']:
        assert sha(VENDOR/row['path'])==row['sha256'], f"Vendor mismatch: {row['path']}"
