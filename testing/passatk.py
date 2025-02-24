from evaluate import load
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"  #  This metric exists to run untrusted model-generated code. Use robust security sandbox


code_eval = load("code_eval")
test_cases = ["assert add(2,3)==5"]
candidates = [["def add(a,b): return a*b", "def add(a, b): return a+b"]]
pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1, 2])
print(pass_at_k)


candidates = [["def add(a,b): return a*b"]]
pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
print(pass_at_k)