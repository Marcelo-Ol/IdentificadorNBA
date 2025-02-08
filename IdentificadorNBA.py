import cv2
import config
from ultralytics import YOLO
import pytesseract
from PIL import Image
import pandas
import re

API_KEY = config.ROBOFLOW_API_KEY
MODEL_ID = "custom-workflow-object-detection-h43rn/3"
video = cv2.VideoCapture('c:/Users/marce/Downloads/CelticsHighlights ‐ Feito com o Clipchamp.mp4')
jogadores = pandas.read_csv('BostonCelticsRoster.csv')
model = YOLO(MODEL_ID)

url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    results = model.predict(frame)

    for result in results:
        x, y, w, h = int(result["x"]), int(result["y"]), int(result["width"]), int(result["height"])

        # Cortar a imagem para obter o número identificado e converter para cinza
        cropped_number = video[y-h//2:y+h//2, x-w//2:x+w//2]
        gray = cv2.cvtColor(cropped_number, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))

        raw_text = pytesseract.image_to_string(gray, config="--psm 7").strip()

        match = re.search(r"\d+", raw_text)
        recognized_number = match.group() if match else ""
        print(f"OCR detectou: '{raw_text}', Número extraído: '{recognized_number}'")
        if recognized_number.isdigit():
            detected_number = int(recognized_number)
            #Encontrar o jogador com o número identificado no .CSV
            player_name = jogadores[jogadores["No."] == detected_number]["Player"].values

            if(len(player_name) > 0):
                print(f"Jogador: {player_name[0]}, Numero: {detected_number}")
                cv2.putText(video, f"{player_name[0]} ({detected_number})", (x-120, y+120), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)
                cv2.rectangle(video, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 0, 255), 2)
                cv2.imwrite('resultado.jpg', video)
            else:
             print(f"Jogador não encontrado, Numero: {detected_number}")
        else:
            print("Nenhum numero identificado na imagem")

    cv2.imshow('NBAClip', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()