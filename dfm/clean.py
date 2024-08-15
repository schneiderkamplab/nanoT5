import json
import sys

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"Cleaning {arg}")
        ds = []
        with open(arg, "rt") as f:
            for line in f:
                d = json.loads(line)
                for key in list(d.keys()):
                    if not key == "text":
                        del d[key]
                ds.append(d)
        with open(arg, "wt") as f:
            for d in ds:
                f.write(f"{json.dumps(d)}\n")
        print(f"Cleaned {arg}")
