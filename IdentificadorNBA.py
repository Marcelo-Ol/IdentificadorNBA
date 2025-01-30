import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import pandas

video = cv2.VideoCapture('c:/Users/marce/Downloads/CelticsHighlights ‐ Feito com o Clipchamp.mp4')
model = YOLO('yolov8n.pt')
jogadores = pandas.read_csv('BostonCelticsRoster.csv')

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # recorta região do torso
        torso = frame[y1:y2, x1:x2]
        # extrai o número da camisa
        numero = pytesseract.image_to_string(Image.fromarray(torso), config='--psm 6')
        # consulta número no csv
        if numero in jogadores['No.'].astype(str).values:
            nome_jogador = jogadores.loc[jogadores['No.'] == int(numero), 'Player'].values[0]
        # escreve nome no frame
        cv2.putText(frame, nome_jogador, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('NBAClip', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()