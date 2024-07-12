import json
import csv

def json_to_csv(json_file_path, csv_file_path):
    # Read the JSON file
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Extract the fieldnames from the first item (if data is a list of dictionaries)
    if isinstance(data, list) and len(data) > 0:
        fieldnames = data[0].keys()
    else:
        raise ValueError("JSON file must contain a list of dictionaries")
    
    # Write the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        csv_writer.writeheader()
        
        # Write the data
        for row in data:
            csv_writer.writerow(row)

def dictUpdateFunc(dct):
    dct.update({"Target": float(dct["4. close"])})
    return dct

if __name__ == "__main__":
      # Example usage
      with open("/Users/MOPOLLIKA/python_StockDL/stockdata/ibmDaily.json", "r") as f:
          prices = json.load(f)
      prices = list(prices["Time Series (Daily)"].values())
      prices = list(map(dictUpdateFunc, prices))
      with open("/Users/MOPOLLIKA/python_StockDL/stockdata/ibmDailyToCsv.json", "w") as ph:
          json.dump(prices, ph)
      json_file_path = "/Users/MOPOLLIKA/python_StockDL/stockdata/ibmDailyToCsv.json"
      csv_file_path = "/Users/MOPOLLIKA/python_StockDL/test1.csv"
      json_to_csv(json_file_path, csv_file_path)
