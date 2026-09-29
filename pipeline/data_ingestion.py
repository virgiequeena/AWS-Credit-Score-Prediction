from pathlib import Path
import pandas as pd

# Base directory
BASE_DIR = Path(__file__).parent

# Define folders
RAW_DIR = BASE_DIR / "data"
INGESTED_DIR = BASE_DIR / "ingested"

# Define files
DATA_FILE   = RAW_DIR / "credit_data_raw.csv"
OUTPUT_FILE = INGESTED_DIR / "credit_data.csv"

def ingest_data():
    # Ensure output folder exists
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data (single file, target included)
    df = pd.read_csv(DATA_FILE)

    # Basic validation
    assert not df.empty, "Dataset is empty"

    # Save ingested data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Data ingested: {DATA_FILE} → {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest_data()
