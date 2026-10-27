import requests

BASE_URL = "http://localhost:5000"

url = f"{BASE_URL}/generate"
payload = {"prompt": "Tell me a story about a brave knight."}

try:
    print(f"Sending POST to {url}...")
    response = requests.post(url, json=payload, timeout=30, stream=True)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)
    print()
except requests.exceptions.ConnectionError:
    print(f"ERR Connection refused at {url}. Make sure the Docker container or Flask app is running.")
except Exception as e:
    print(f"ERR: {e}")