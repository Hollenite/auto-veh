import re

with open("pytest_out.txt", "r", encoding="utf-16le") as f:
    text = f.read()

failures = []
for line in text.split("\n"):
    if line.startswith("FAILED"):
        failures.append(line.strip())

with open("failures.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(failures))
