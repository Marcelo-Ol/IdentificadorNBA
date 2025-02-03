import config
import requests
import pandas
import cv2
import pytesseract
import numpy as np
import re

API_KEY = config.ROBOFLOW_API_KEY # Altere para sua chave de API Roboflow
MODEL_ID = "custom-workflow-object-detection-h43rn/3"
IMAGE_PATH = "images/Celtics_2.jpg"
jogadores = pandas.read_csv('BostonCelticsRoster.csv')

url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"

with open(IMAGE_PATH, "rb") as image_file:
    image_bytes = image_file.read()

response = requests.post(url, files={"file": image_bytes})

data = response.json()
image = cv2.imread(IMAGE_PATH)

# Filtrar os objetos preditos que são números
number_predictions = [pred for pred in data.get("predictions", []) if pred["class"] == "number"]

if number_predictions:
    for pred in number_predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])

        # Cortar a imagem para obter o número identificado e converter para cinza
        cropped_number = image[y-h//2:y+h//2, x-w//2:x+w//2]
        gray = cv2.cvtColor(cropped_number, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        raw_text = pytesseract.image_to_string(gray, config="--psm 7").strip()

        match = re.search(r"\d+", raw_text)
        recognized_number = match.group() if match else ""
      
        if recognized_number.isdigit():
            detected_number = int(recognized_number)
            #Encontrar o jogador com o número identificado no .CSV
            player_name = jogadores[jogadores["No."] == detected_number]["Player"].values

            if(len(player_name) > 0):
                print(f"Jogador: {player_name[0]}, Numero: {detected_number}")
                cv2.putText(image, f"{player_name[0]} ({detected_number})", (x-120, y+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 0, 255), 2)
                cv2.imwrite('resultado.jpg', image)
            else:
             print(f"Jogador não encontrado, Numero: {detected_number}")
else:
    print("Nenhum numero identificado na imagem")