# Reconhecimento de Jogadores da NBA em Imagens 🏀

Este projeto utiliza **visão computacional**, **OCR** e a **NBA API** para identificar jogadores do Boston Celtics a partir de imagens, exibindo seus nomes com base no número da camisa.

## 🚀 Funcionalidades
✅ **Detecção de números**: Usa um modelo de visão computacional para identificar os números nas camisas.  
✅ **Reconhecimento de texto (OCR)**: Converte a imagem do número para texto usando **Tesseract OCR**.  
✅ **Mapeamento de jogadores**: Relaciona o número identificado com o nome do jogador utilizando a **NBA API**.  
✅ **Anotação da imagem**: Exibe o nome do jogador e destaca a região detectada na imagem. 

## 🛠️ Tecnologias Utilizadas
- **Python** (linguagem principal)
- **OpenCV** (processamento de imagem)
- **Roboflow API** (modelo de detecção de números)
- **Tesseract OCR** (extração de texto)
- **NBA API** (obtenção dos nomes dos jogadores)
- **Pandas** (manipulação de dados do elenco)

## 📥 Instalação e Uso

1. **Clone o repositório**:
   ```sh
   git clone https://github.com/Marcelo-Ol/IdentificadorNBA.git
   cd IdentificadorNBA
   ```

2. **Instale as dependências**:
```sh
pip install opencv-python pytesseract pandas requests nba_api
```

3. **Configure a API Key da Roboflow**:
Crie um arquivo config.py e adicione sua chave:
```python
ROBOFLOW_API_KEY = "SUA_CHAVE_AQUI"
```

4. **Execute o código**:
```sh
python IdentificadorNBAimagem.py
```

O script processará a imagem, identificará o número da camisa e exibirá o nome do jogador correspondente.

## 📌 Próximos Passos
- Suporte para mais times além do Boston Celtics.
- Melhorias na detecção e OCR para aumentar a precisão.
- Processamento de vídeos, além de imagens estáticas.
