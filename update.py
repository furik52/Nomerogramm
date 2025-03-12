# To save lib u're using
import subprocess
subprocess.run(["powershell", 'pip freeze > .\\venv\\requirements.txt'])
# check https://pypi.org/project/opencv-python/
# pip install opencv-python
# pip install opencv-contrib-python