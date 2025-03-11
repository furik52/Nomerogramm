# To save lib u're using
# pip freeze > .\venv\requirements.txt
import subprocess
subprocess.run(["powershell", "-noexit", 'pip freeze > .\\venv\\requirements.txt'])