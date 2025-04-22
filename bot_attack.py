import requests
import time

url = "http://127.0.0.1:5000/predict"

# Malicious payload designed to trigger 'Is Account Takeover' = 1
data = {
    "ASN": 500126,
    "Login Hour": 11,
    "IP Address": 167963945,
    "User Agent String": 252152,
    "Browser Name and Version": 2463,
    "OS Name and Version": 7,
    "Country": "BR",
    "Device Type": "desktop"
}

# Simulate repeated attack attempts
for i in range(10):  # Adjust the number of attempts
    response = requests.post(url, json=data)
    print(f"Attempt {i+1}: {response.json()}")
    time.sleep(1)  # Pause between attempts to mimic real attacks
