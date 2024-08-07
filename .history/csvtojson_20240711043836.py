import pandas as pd
import json

def CsvToJson(csvFilepath: str, jsonFilepath) -> None:
      csv = pd.read_csv(csvFilepath, sep=",", header=0, index_col=False)
      dataFrame = pd.DataFrame(csv)
      dataFrame.to_json(jsonFilepath, orient="records", date_format="epoch", double_precision=10, force_ascii=True, date_unit="ms", default_handler=None)

def CleanData(jsonFilepath) -> None:
      with open(jsonFilepath, "r") as f:
            data: list[dict] = json.load(f)
            f.close()
      dataCleaned: list[dict] = []
      for entry in data:
            if entry["LandAverageTemperature"] == "null":
                  dataCleaned = []
            dataCleaned[entry["dt"]] = entry["LandAverageTemperature"]
      print(dataCleaned)

def main() -> None:
      csvFilepath = "/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.csv"
      jsonFilepath = "/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.json"
      CleanData()


if __name__ == "__main__":
      main()