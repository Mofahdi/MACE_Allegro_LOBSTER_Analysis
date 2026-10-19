import pandas as pd
import requests
import time
import zipfile
import io
from pathlib import Path

# Load material ID -> NOMAD entry ID mapping
df = pd.read_csv("nomad_entry_ids.csv", index_col=0)

# Files to download from each NOMAD entry
files = [
    "CONTCAR.gz",
    "CHARGE.lobster.gz",
    "ICOHPLIST.lobster.gz",
    "ICOBILIST.lobster.gz",
]

# Output directory
output_dir = Path("LOBSTER_data")
output_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://nomad-lab.eu/prod/v1/api/v1"

with requests.Session() as session:
    for material_id, entry_id in df["nomad_entry_id"].items():

        material_id = str(material_id)
        entry_id = str(entry_id).strip()

        material_dir = output_dir / material_id
        material_dir.mkdir(parents=True, exist_ok=True)

        url = f"{base_url}/entries/{entry_id}/raw"

        try:
            response = session.get(
                url,
                params={"include_files": files},
                timeout=300,
            )
            response.raise_for_status()

            # Extract files directly into the material folder
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                root = material_dir.resolve()

                for member in zf.infolist():
                    if member.is_dir():
                        continue

                    # Discard all internal directory components
                    filename = Path(member.filename).name

                    if not filename:
                        continue

                    target = material_dir / filename

                    # Prevent duplicate or unsafe path issues
                    if not target.resolve().is_relative_to(root):
                        raise ValueError(
                            f"Unsafe archive filename: {member.filename}"
                        )

                    # Extract individual file
                    with zf.open(member) as src:
                        with target.open("wb") as dst:
                            while chunk := src.read(1024 * 1024):
                                dst.write(chunk)

            print(f"Downloaded {material_id}: {entry_id}")
            time.sleep(5)

        except (
            requests.RequestException,
            zipfile.BadZipFile,
            ValueError,
            OSError,
        ) as e:
            print(f"Failed {material_id} ({entry_id}): {e}")

print("Download process finished.")