import csv
import json
import pandas as pd

def CsvToJson(filepath: str):
      csv = pd.read_csv