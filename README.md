# Reconhecimento de Jogadores da NBA em Clipes de Jogo 🏀

Este projeto utiliza visão computacional e OCR para identificar jogadores da NBA em clipes de jogo, exibindo seus nomes em tempo real. 

## Funcionalidades

- 🎥 **Processamento de vídeo**: Carrega e exibe clipes da NBA.
- 👤 **Detecção de jogadores**: Usa YOLOv8 para localizar jogadores em cada frame.
- 🔢 **Reconhecimento de números**: Extrai o número da camisa com Tesseract OCR.
- 📊 **Consulta de dados**: Mapeia número e time para o nome do jogador usando um arquivo CSV.

## Tecnologias

- **Python** (linguagem principal)
- **OpenCV** (processamento de vídeo)
- **YOLOv8** (detecção de objetos)
- **Tesseract OCR** (leitura de números)
- **Pandas** (gerenciamento do banco de dados de jogadores)

## Instalação

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/Marcelo-Ol/IdentificadorNBA.git
   cd IdentificadorNBA

2. **Instale as dependências**:

```bash
pip install opencv-python ultralytics pytesseract pandas