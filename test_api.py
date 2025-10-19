import os
import sys
import json
import pandas as pd
import requests

# Config section
# API_URL can be overridden for Docker/remote use
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"
CSV_PATH = "data/future_unseen_examples.csv"

def check_api_health():
    """Check if the API is up and running before testing."""
    try:
        response = requests.get(HEALTH_URL)
        response.raise_for_status()
        print(f"Success! API is reachable at {API_URL}")
    except Exception as e:
        print(f"Fail! API is not reachable at {API_URL}")
        print(e)
        sys.exit(1)

def main():
    check_api_health()

    # Load the data
    df = pd.read_csv(CSV_PATH, dtype={'zipcode': str})

    # Select a subset of rows for testing
    examples = df.sample(10, random_state=42).to_dict(orient="records")

    for i, example in enumerate(examples, start=1):
        try:
            # Send POST request
            response = requests.post(PREDICT_URL, json=example)

            # Check response
            if response.status_code == 200:
                print("\nSuccess!")
                print(f"Input {i}: {example}")
                print(f"Response {i}: {json.dumps(response.json(), indent=2)}")
            
            # Fail (but still got a response from server)
            else:
                print("\nFail!")
                print(f"Input {i}: {example}")
                print("Error:", response.status_code, response.text)

        # Fail (no response at all: network error, timeout, connection refused, etc.)
        except requests.exceptions.RequestException as e:
            print("\nRequest failed entirely.")
            print(f"Input {i}: {example}")
            print("Exception:", e)

if __name__ == "__main__":
    main()