import pandas as pd

def CsvToJson(csvFilepath: str, jsonFilepath) -> None:
      csv = pd.read_csv(csvFilepath, sep=",", header=0, index_col=False)
      dataFrame = pd.DataFrame(csv)
      dataFrame.to_json(jsonFilepath, orient="records", date_format="epoch", double_precision=10, force_ascii=True, date_unit="ms", default_handler=None)
      return

def main() -> None:
      csvFilepath = "tests/test1.csv"


if __name__ == "__main__":
      main()