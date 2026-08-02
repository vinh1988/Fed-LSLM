import pandas as pd
import re
from pathlib import Path

def convert_roco_id_to_filename(roco_id: str) -> str:
    """
    Convert ROCO reasoning ID format to filename format.
    ROCOv2_2023_train_000001 -> 000000.jpg
    ROCOv2_2023_train_000002 -> 000001.jpg
    """
    if pd.isna(roco_id):
        return ""
    # Extract the numeric part from ROCOv2_2023_train_XXXXXX
    match = re.search(r'ROCOv2_2023_train_(\d+)', str(roco_id))
    if match:
        num = int(match.group(1))
        # Subtract 1 to convert to 0-indexed
        new_num = num - 1
        # Format as 6-digit with .jpg extension
        return f"{new_num:06d}.jpg"
    return str(roco_id)

def fix_roco_reasoning_csv(input_csv: Path, output_csv: Path):
    """
    Fix ROCO reasoning CSV by converting image_id to match filename format.
    """
    print(f"Loading ROCO reasoning CSV from {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Original samples: {len(df)}")
    
    # Convert image_id to filename format
    df['image_name'] = df['image_id'].apply(convert_roco_id_to_filename)
    
    # Save with new column
    df.to_csv(output_csv, index=False)
    print(f"Saved fixed CSV to {output_csv}")
    print(f"Output samples: {len(df)}")
    
    # Show sample conversions
    print("\nSample conversions:")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['image_id']
        converted = df.iloc[i]['image_name']
        print(f"  {original} -> {converted}")

if __name__ == "__main__":
    data_root = Path("processed_data")
    input_csv = data_root / "ROCO" / "roco_reasoning_train.csv"
    output_csv = data_root / "ROCO" / "roco_reasoning_train_fixed.csv"
    
    fix_roco_reasoning_csv(input_csv, output_csv)
