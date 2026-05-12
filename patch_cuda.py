import os
import re

device_files = {
    "recommenders/data.py": 90,
    "recommenders/models/base/abstract_model.py": 23,
    "recommenders/models/base/abstract_RS.py": 23,
    "simulation/base/abstract_arena.py": 26,
}

for fp, line_num in device_files.items():
    with open(fp, 'r') as f:
        lines = f.readlines()
    old = lines[line_num - 1]
    new = old.replace(
        'self.device = torch.device(args.cuda)',
        'self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")'
    )
    lines[line_num - 1] = new
    with open(fp, 'w') as f:
        f.writelines(lines)
    print(f"Fixed device in {fp}")

cuda_files = [
    "recommenders/data.py",
    "recommenders/models/MF.py",
    "recommenders/models/LightGCN.py",
    "recommenders/models/MultVAE.py",
    "recommenders/models/InfoNCE.py",
    "recommenders/models/base/abstract_model.py",
    "recommenders/models/base/abstract_RS.py",
    "simulation/base/abstract_arena.py",
]

for fp in cuda_files:
    if not os.path.exists(fp):
        continue
    with open(fp, 'r') as f:
        content = f.read()
    new_content = re.sub(r'\.cuda\(self\.device\)', '.to(self.device)', content)
    if new_content != content:
        with open(fp, 'w') as f:
            f.write(new_content)
        print(f"Patched .cuda() in {fp}")

print("Done.")
