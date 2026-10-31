from remotezip import RemoteZip
import os
import sys
import time

URL = "https://ndownloader.figshare.com/files/38030910"
MEMBER = "RDD2022/India.zip"
OUTPUT = "data/raw/RDD2022_India.zip"
EXPECTED_SIZE = 526_709_145

os.makedirs("data/raw", exist_ok=True)

try:
    with RemoteZip(URL) as archive:
        info = archive.getinfo(MEMBER)
        print(f"India ZIP size: {info.file_size:,} bytes", flush=True)

        downloaded = 0
        last_report = time.time()

        with archive.open(MEMBER) as source, open(OUTPUT, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)

                if time.time() - last_report >= 2:
                    percent = downloaded / info.file_size * 100
                    print(
                        f"\rDownloaded {downloaded:,} / {info.file_size:,} bytes "
                        f"({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
                    last_report = time.time()

        print(f"\nFinished writing {downloaded:,} bytes.", flush=True)

        if downloaded != info.file_size:
            raise RuntimeError("Downloaded size does not match the archive entry.")

    print("Download complete. Now verify the ZIP with: unzip -t " + OUTPUT)

except KeyboardInterrupt:
    print("\nDownload interrupted. Partial file is not a valid ZIP yet.")
    sys.exit(1)
except Exception as exc:
    print(f"\nDownload failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
