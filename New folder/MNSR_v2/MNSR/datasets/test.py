import json

with open("truthful_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total questions:", len(data))

#  376 in truthful_qa.json
# 478 in strategyqa.json
# 500 in halueval_qa.json
# 500 in gsm8k.json

# from pathlib import Path

# # Path to your file
# file_path = r"C:\Users\user2\Desktop\projects\MNSR\New folder\MNSR_v2\MNSR\datasets\truthful_qa.json"

# # Keep everything up to this line number (1-based)
# keep_until_line = 2476

# path = Path(file_path)

# # Read all lines
# with open(path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # Keep only the specified number of lines
# lines = lines[:keep_until_line]

# # Write back to the same file
# with open(path, "w", encoding="utf-8") as f:
#     f.writelines(lines)

# print(f"Kept the first {keep_until_line} lines.")