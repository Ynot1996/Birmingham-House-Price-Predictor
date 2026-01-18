import pandas as pd


def process_local_csv():
    # Load the big file you just downloaded
    input_file = 'uk_2025.csv'
    output_file = 'birmingham_prices_real_2025.csv'

    # Official column names for Land Registry PPD
    columns = ['ID', 'Price', 'Date', 'Postcode', 'Type', 'Old_New', 'Duration', 'PAON', 'SAON', 'Street', 'Locality',
        'City', 'District', 'County', 'PPD_Category', 'Record_Status']

    print(f"Reading {input_file}...")

    # We use chunksize to prevent memory errors with the large 100MB+ file
    chunks = pd.read_csv(input_file, names=columns, chunksize=50000)

    bham_data = []
    for chunk in chunks:
        # Filter where District is BIRMINGHAM (must be uppercase)
        filtered = chunk[chunk['District'] == 'BIRMINGHAM']
        bham_data.append(filtered)

    if bham_data:
        final_df = pd.concat(bham_data)
        final_df.to_csv(output_file, index=False)
        print(f"Success! Filtered {len(final_df)} Birmingham records into {output_file}")
    else:
        print("No Birmingham records found. Check if the district name is correct.")


if __name__ == "__main__":
    process_local_csv()