import os
import time
import requests

URL = "https://ndownloader.figshare.com/files/38030910"
OUTPUT = "data/raw/RDD2022_India.zip"
ENTRY_START = 609341032
SIZE = 526709145
CHUNK = 8 * 1024 * 1024

os.makedirs("data/raw", exist_ok=True)
session = requests.Session()

def get_range(start, end):
    last_error = None
    for attempt in range(5):
        try:
            response = session.get(
                URL,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 180),
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206, got {response.status_code}"
                )
            expected = end - start + 1
            if len(response.content) != expected:
                raise RuntimeError(
                    f"Expected {expected} bytes, received {len(response.content)}"
                )
            return response.content
        except Exception as error:
            last_error = error
            print(f"\nRequest failed ({attempt + 1}/5): {error}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"All retries failed: {last_error}")

print("Checking the India ZIP entry...", flush=True)
header = get_range(ENTRY_START, ENTRY_START + 4095)

if header[:4] != b"PK\x03\x04":
    raise RuntimeError("The remote data does not start with a ZIP header.")

filename_length = int.from_bytes(header[26:28], "little")
extra_length = int.from_bytes(header[28:30], "little")
data_start = ENTRY_START + 30 + filename_length + extra_length

print(f"India ZIP data begins at byte {data_start:,}", flush=True)

if os.path.exists(OUTPUT):
    downloaded = os.path.getsize(OUTPUT)
    if downloaded > SIZE:
        raise RuntimeError("Existing file is larger than expected; inspect it first.")
else:
    downloaded = 0

print(f"Starting from {downloaded:,} / {SIZE:,} bytes", flush=True)

with open(OUTPUT, "ab") as output:
    while downloaded < SIZE:
        amount = min(CHUNK, SIZE - downloaded)
        start = data_start + downloaded
        chunk = get_range(start, start + amount - 1)
        output.write(chunk)
        output.flush()
        downloaded += len(chunk)

        print(
            f"\rProgress: {downloaded / SIZE * 100:5.1f}% "
            f"({downloaded / 1_000_000:.1f} / {SIZE / 1_000_000:.1f} MB)",
            end="",
            flush=True,
        )

print("\nDownload finished. Checking exact file size...", flush=True)
actual = os.path.getsize(OUTPUT)
if actual != SIZE:
    raise RuntimeError(f"Size mismatch: expected {SIZE}, got {actual}")

print("Size verified. Now run: unzip -t " + OUTPUT, flush=True)
