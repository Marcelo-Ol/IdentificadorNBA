import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import pandas

image = cv2.imread("images/Celtics_1.jpg")
model = YOLO('yolov8n.pt')
jogadores = pandas.read_csv('BostonCelticsRoster.csv')

results = model.predict(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

for box in boxes:
    x1, y1, x2, y2 = map(int, box)

    # recorta região do torso
    torso = image[y1:y2, x1:x2]
    cv2.imwrite("recorte_torso.jpg", torso)

    # extrai o número da camisa
    numero = pytesseract.image_to_string(Image.fromarray(torso), config='--psm 6').strip()
    print(f"Número: '{numero}'")

    # consulta número no csv
    if numero.isdigit() and int(numero) in jogadores['No.'].values:
        nome_jogador = jogadores[jogadores['No.'] == int(numero)]['Player'].values[0]
        print(f"Jogador: '{nome_jogador}'")

        # escreve nome na imagem 
        cv2.putText(image, nome_jogador, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    else:
        print("Jogador não encontrado no CSV")

cv2.imwrite('resultado.jpg', image)
print("Processamento concluído. Verifique 'resultado_teste.jpg' e 'recorte_torso.jpg'.")