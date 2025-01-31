import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import pandas

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO('yolov8n.pt')
jogadores = pandas.read_csv('BostonCelticsRoster.csv')
image = cv2.imread("images/Celtics_1.jpg")
altura, largura = image.shape[:2]

results = model.predict(image)
boxes = results[0].boxes.xyxy.cpu().numpy()

for box in boxes:
    x1, y1, x2, y2 = map(int, box)

    # ignora pessoas na arquibancada
    altura = y2 - y1
    if y2 < altura * 0.25:
        continue
    centro = (y1 + y2) / 2
    if centro < altura * 0.3:
        continue

    # recorta região do torso
    torso = image[y1:y1+100, x1:x2]
    cv2.imwrite("recorte_torso.jpg", torso)

    # pré-processamento
    torso_resize = cv2.resize(torso, (300, 100))
    gray = cv2.cvtColor(torso_resize, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # extrai o número da camisa
    numero = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
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