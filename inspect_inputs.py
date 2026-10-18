import csv
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path("extracted")

# Read the project documentation without installing extra libraries
for filename in ["FlowCast_PRD.docx", "FlowCast_Data_Dictionary.docx"]:
    path = root / filename
    print("\n" + "=" * 70)
    print(f"DOCUMENT: {filename}")
    print("=" * 70)

    with zipfile.ZipFile(path) as doc:
        xml = doc.read("word/document.xml")

    tree = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    for paragraph in tree.findall(".//w:p", ns):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", ns)
        )
        if text.strip():
            print(text)

# Inspect each CSV: columns, sample records, and total rows
for path in sorted(Path("data/raw").glob("*.csv")):
    print("\n" + "=" * 70)
    print(f"DATASET: {path.name}")
    print("=" * 70)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print("Columns:", reader.fieldnames)

        count = 0
        for row in reader:
            if count < 3:
                print("Sample:", row)
            count += 1

    print("Total data rows:", count)
