import pandas as pd
import json

def CsvToJson(csvFilepath: str, jsonFilepath) -> None:
      csv = pd.read_csv(csvFilepath, sep=",", header=0, index_col=False)
      dataFrame = pd.DataFrame(csv)
      dataFrame.to_json(jsonFilepath, orient="records", date_format="epoch", double_precision=10, force_ascii=True, date_unit="ms", default_handler=None)

def CleanData(jsonFilepath) -> None:
      with open(jsonFilepath, "r") as f:
            data = json.load(f)
            f.close()
      dataCleaned = 
      for entry in data:
            

def main() -> None:
      csvFilepath = "/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.csv"
      jsonFilepath = "/Users/MOPOLLIKA/python_StockDL/globaltemperature/GlobalLandTemperatures_GlobalTemperatures.json"
      


if __name__ == "__main__":
      main()