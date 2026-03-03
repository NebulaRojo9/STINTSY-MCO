
import numpy as np
import csv

def import_csv_to_numpy(filename):
    rows = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader, None) # necessary to read header row in strings
        rows.append(header)

        for row in reader:
            clean = []
            for cell in row:
                cell = cell.strip()
                if cell == '':
                    clean.append(np.nan)
                else:
                    try:
                        clean.append(float(cell))
                    except ValueError:
                        clean.append(cell)
            rows.append(clean)

    return np.array(rows, dtype=object) 