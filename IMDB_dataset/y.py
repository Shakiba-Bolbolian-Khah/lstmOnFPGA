import pandas as pd

# Read the CSV file
input_file = "imdb/y_test.csv"   # Change this to your actual file name
output_file = "imdb/y.csv"  # Name of the output file

# Read the file as a single-column DataFrame
df = pd.read_csv(input_file, header=None)

# Convert values to integers
df = df.astype(int)

# Save to a new CSV file without the index
df.to_csv(output_file, index=False, header=False)

print(f"Converted file saved as {output_file}")
