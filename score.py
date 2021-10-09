import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = {"data": [{"age": 75.0, "anaemia": 0, "creatinine_phosphokinase": 582, "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 1, "platelets": 265000.0, "serum_creatinine": 1.9, "serum_sodium": 130, "sex": 1, "smoking": 0, "time": 4}, {"age": 55.0, "anaemia": 0, "creatinine_phosphokinase": 7861, "diabetes": 0, "ejection_fraction": 38, "high_blood_pressure": 0, "platelets": 263358.03, "serum_creatinine": 1.1, "serum_sodium": 136, "sex": 1, "smoking": 0, "time": 6}, {"age": 65.0, "anaemia": 0, "creatinine_phosphokinase": 146, "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 0, "platelets": 162000.0, "serum_creatinine": 1.3, "serum_sodium": 129, "sex": 1, "smoking": 1, "time": 7}, {"age": 50.0, "anaemia": 1, "creatinine_phosphokinase": 111, "diabetes": 0, "ejection_fraction": 20, "high_blood_pressure": 0, "platelets": 210000.0, "serum_creatinine": 1.9, "serum_sodium": 137, "sex": 1, "smoking": 0, "time": 7}, {"age": 65.0, "anaemia": 1, "creatinine_phosphokinase": 160, "diabetes": 1, "ejection_fraction": 20, "high_blood_pressure": 0, "platelets": 327000.0, "serum_creatinine": 2.7, "serum_sodium": 116, "sex": 0, "smoking": 0, "time": 8}, {"age": 90.0, "anaemia": 1, "creatinine_phosphokinase": 47, "diabetes": 0, "ejection_fraction": 40, "high_blood_pressure": 1, "platelets": 204000.0, "serum_creatinine": 2.1, "serum_sodium": 132, "sex": 1, "smoking": 1, "time": 8}, {"age": 75.0, "anaemia": 1, "creatinine_phosphokinase": 246, "diabetes": 0, "ejection_fraction": 15, "high_blood_pressure": 0, "platelets": 127000.0, "serum_creatinine": 1.2, "serum_sodium": 137, "sex": 1, "smoking": 0, "time": 10}, {"age": 60.0, "anaemia": 1, "creatinine_phosphokinase": 315, "diabetes": 1, "ejection_fraction": 60, "high_blood_pressure": 0, "platelets": 454000.0, "serum_creatinine": 1.1, "serum_sodium": 131, "sex": 1, "smoking": 1, "time": 10}, {"age": 65.0, "anaemia": 0, "creatinine_phosphokinase": 157, "diabetes": 0, "ejection_fraction": 65, "high_blood_pressure": 0, "platelets": 263358.03, "serum_creatinine": 1.5, "serum_sodium": 138, "sex": 0, "smoking": 0, "time": 10}, {"age": 80.0, "anaemia": 1, "creatinine_phosphokinase": 123, "diabetes": 0, "ejection_fraction": 35, "high_blood_pressure": 1, "platelets": 388000.0, "serum_creatinine": 9.4, "serum_sodium": 133, "sex": 1, "smoking": 1, "time": 10}]}

body = str.encode(json.dumps(data))

url = 'http://9a5d892d-7ed6-4093-946e-c6c42f7a035d.southcentralus.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
