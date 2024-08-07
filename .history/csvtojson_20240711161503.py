import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

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

def CsvToEntries(csvFilepath: str) -> list[float]:
      jsonFilepath: str = csvFilepath[:-4] + ".json"
      CsvToJson(csvFilepath, jsonFilepath)
      
      CleanData(jsonFilepath)

      jsonFilepathNew: str = jsonFilepath[:-5] + "_Cleaned.json"
      with open(jsonFilepathNew, "r") as f:
            data: list[dict] = json.load(f)
            f.close()
      
      entries: dict = {}
      for entry in data:
            value = float(list(entry.values())[0])
            date = list(entry.keys())[0]
            entries[date] = value
      return entries

def AdjustForYears(entries: dict) -> dict:
      values: list = list(entries.values())
      period: int = 12
      entriesAdjusted: dict = {list(entries.keys())[index]: [np.mean(values[index:index + period])] for index in range(0, len(values) - period, period)}
      return entriesAdjusted

if __name__ == "__main__":
      entries: dict = CsvToEntries("/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.csv")
      entries = AdjustForYears(entries)
      df: pd.DataFrame = pd.read_csv("/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.csv")
      df.plot(kind="bar", x="index", y="LandAverageTemperature")
      plt.show()
