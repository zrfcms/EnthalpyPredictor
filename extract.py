import sqlite3
import csv
import os

# Connect to the SQLite database
conn = sqlite3.connect('data/Supplementary_Data.db')

# Create a cursor object
cursor = conn.cursor()

# Query the first 10 rows from the Formation_Enthalpy table
cursor.execute("SELECT ID, NAME, STRUCT, Formation_energy, Source FROM Formation_Enthalpy LIMIT 10")
rows = cursor.fetchall()

# Save the data to a CSV file
csv_file = 'formation_enthalpy.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'NAME', 'STRUCT', 'Formation_energy', 'Source'])  # Write header
    writer.writerows(rows)  # Write data

print(f"Data has been saved to {csv_file}")

# Save the STRUCT column content to POSCAR files
output_dir = 'POSCAR'
os.makedirs(output_dir, exist_ok=True)

for row in rows:
    struct_content = row[2]
    poscar_filename = f"{row[0]}.POSCAR"
    poscar_filepath = os.path.join(output_dir, poscar_filename)
    
    with open(poscar_filepath, mode='w') as file:
        file.write(struct_content)
    
    print(f"STRUCT content has been saved to {poscar_filepath}")

# Close the cursor and the connection
cursor.close()
conn.close()
