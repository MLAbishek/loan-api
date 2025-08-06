import requests

BASE_URL = "https://sap-f2dc.onrender.com"

features = [1000, 500, 50, 8, 300, 1200, "Sunny", 0, 1]
label = 123


def print_response(resp):
    print("Status code:", resp.status_code)
    try:
        print("JSON:", resp.json())
    except Exception:
        print("Raw response:", resp.text)


print("Testing /submit ...")
resp = requests.post(f"{BASE_URL}/submit", json={"features": features, "label": label})
print_response(resp)

print("\nTesting /predict ...")
resp = requests.post(
    f"{BASE_URL}/predict",
    json={"features": [1000, 500, 50, 8, 300, 1200, "Sunny", 0, 1]},
)
print_response(resp)

print("\nTesting /retrain ...")
resp = requests.post(f"{BASE_URL}/retrain")
print_response(resp)
