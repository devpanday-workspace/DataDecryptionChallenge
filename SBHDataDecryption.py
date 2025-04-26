import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Load Excel sheets
file_path = r'C:\Users\USER\OneDrive\Documents\DataOG(1).xlsx'
sheet1 = pd.read_excel(file_path, sheet_name='DataOG').iloc[1:20001]
sheet2 = pd.read_excel(file_path, sheet_name='protected_data_challenge(1)').iloc[1:20001]

vectors1 = sheet1.iloc[:, 4:17].values
vectors2 = sheet2.iloc[:, 4:17].values

used_rows2 = set()
matches = []
threshold = 0.7

print("Greedily matching each row in Sheet1 to best in Sheet2...")

for i in tqdm(range(len(vectors1))):
    best_j = -1
    best_sim = -1

    for j in range(len(vectors2)):
        if j in used_rows2:
            continue

        sim = 1 - cosine(vectors1[i], vectors2[j])
        if sim > best_sim:
            best_sim = sim
            best_j = j

    if best_sim >= threshold:
        used_rows2.add(best_j)
        matches.append({
            'Sheet1_Index': i + 2,
            'Sheet2_Index': best_j + 2,
            'Sheet1_Column2': sheet1.iloc[i, 1],
            'Sheet2_Column2': sheet2.iloc[best_j, 1],
            'Cosine_Similarity': best_sim
        })

# Save results
result_df = pd.DataFrame(matches)
result_df.to_excel(r'C:\Users\USER\OneDrive\Documents\data_matches.xlsx', index=False)

# Estimate accuracy
total_possible = min(len(sheet1), len(sheet2))
accuracy = len(matches) / total_possible
print(f"Matches found: {len(matches)}")
print(f"Estimated accuracy: {accuracy:.4f}")
