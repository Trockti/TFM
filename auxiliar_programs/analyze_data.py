def main():
    import os
    import json

    dirName = "./data"
    definitions = {}
    for file in os.listdir(dirName):
        if file.endswith(".json"):
            print(f"Processing {file}...")
            result = os.path.join(dirName, file)
            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)
                definitions.update({k.lower(): v for k, v in data.items()})
    with open("./data/definitions.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()