import pandas as pd
import json

def CsvToJson(csvFilepath: str, jsonFilepath: str) -> None:
      csv = pd.read_csv(csvFilepath, sep=",", header=0, index_col=False)
      dataFrame = pd.DataFrame(csv)
      dataFrame.to_json(jsonFilepath, orient="records", date_format="epoch", double_precision=10, force_ascii=True, date_unit="ms", default_handler=None)

def CleanData(jsonFilepath: str) -> None:
      with open(jsonFilepath, "r") as f:
            data: list[dict] = json.load(f)
            f.close()

      dataCleaned: list[dict] = []
      for entry in data:
            if entry["LandAverageTemperature"] == None:
                  dataCleaned = []
            else:
                  dataCleaned.append({entry["dt"]: entry["LandAverageTemperature"]})

      jsonFilepathNew: str = jsonFilepath[:-5] + "_Cleaned.json"
      with open(jsonFilepathNew, "w") as f:
            json.dump(dataCleaned, f)
            f.close()

def CsvToEntries(csvFilepath: str) -> None:
      jsonFilepath: str = csvFilepath[:-4] + ".json"
      CsvToJson(csvFilepath, jsonFilepath)

      jsonFilepathNew: str = jsonFilepath[:-5] + "_Cleaned.json"
      CleanData(jsonFilepathNew)

      with open(jsonFilepathNew, "r") as f:
            data: list[dict] = json.load(f)
            f.close()
      
      entries: list = []
      for entry in data:
            value = float(list(entry.values())[0])
            entries.append(value)
      return entries

if __name__ == "__main__":
      pass