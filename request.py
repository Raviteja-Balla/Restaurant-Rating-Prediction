import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'votes':2, 'cost':9, 'book_table':1,'online_order':1,'cuisine_count':1})

print(r.json())