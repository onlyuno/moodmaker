import requests

url = 'https://drive.google.com/file/d/12CSTGv0Gx8IXoCN4XZ8isSDsNe8wVoF1/view?usp=drive_link'
r = requests.get(url)
with open('model_trained.pth', 'wb') as f:
    f.write(r.content)