import csv
import json
import pandas as pd

def CsvToJson(csvFilepath: str, jsonFilepath) -> None:
      csv = pd.read_csv(csv)