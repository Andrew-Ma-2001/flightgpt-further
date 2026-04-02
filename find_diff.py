import json

json_1 = '/home/yjy/flightgpt/FlightGPT/data/training_data/citynav_train_data.json'
# json_2 = '/home/yjy/flightgpt/FlightGPT/data/citynav/citynav_train_seen.json'
# json_3 = '/home/yjy/flightgpt/FlightGPT/refine_citynav/processed_citynav/citynav_train_seen.json'


json_2 = '/home/yjy/flightgpt/FlightGPT/data/cityrefer/processed_descriptions.json'
json_3 = '/home/yjy/flightgpt/FlightGPT/refine_citynav/cityrefer/processed_descriptions.json'
with open(json_1, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open(json_2, 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

with open(json_3, 'r', encoding='utf-8') as f3:
    data3 = json.load(f3)

# 打印第一个样例
import pprint

pp = pprint.PrettyPrinter(indent=2, width=100, compact=False, sort_dicts=False)

print("=== data1[0] ===")
pp.pprint(data1[0])

# print("\n=== data2[0] (without 'trajectory') ===")
# data2_0 = {k: v for k, v in data2[1].items() if k != 'trajectory'}
# pp.pprint(data2_0)

# print("\n=== data3[0] (without 'trajectory') ===")
# data3_0 = {k: v for k, v in data3[1].items() if k != 'trajectory'}
# pp.pprint(data3_0)

key = 'birmingham_block_1'
key2 = '1'

print("=== data2[key2] 的 value ===")
if key2 in data2:
    print(f"key: {key2}")
    print("value:")
    pp.pprint(data2[key2])
else:
    print(f"{key2} not found in data2.")

print("=== data3[key2] 的 value ===")
if key2 in data3:
    print(f"key: {key2}")
    print("value:")
    pp.pprint(data3[key2])
else:
    print(f"{key2} not found in data3.")