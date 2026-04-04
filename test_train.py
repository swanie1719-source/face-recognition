# test_train.py
import ast

for fname in ['train.py', 'data/ms1m_dataset.py']:
    with open(fname, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        ast.parse(source)
        print(f"{fname}: 语法正确 ✓")
    except SyntaxError as e:
        print(f"{fname}: 语法错误 → {e}")