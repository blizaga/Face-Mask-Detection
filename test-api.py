import requests
import os


# Define the URL of the API
url = "http://localhost:8000/detect_mask"

# Define the path to the image file
image_path = os.path.join("example", "mask.jpeg")

# Open an image file in binary mode
image_file = open(image_path, "rb")

# Define the multipart/form-data payload
payload = {"image": image_file}

# Send a POST request to the API
response = requests.post(url, files=payload)

# Close the image file
image_file.close()

# Print the response from the API
print(response.json())