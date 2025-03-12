import subprocess, os
# Get-ExecutionPolicy -List
subprocess.run(["powershell", '.\\venv\\ExecutionPolicyUndefined.bat'])
os.system('cls')
subprocess.run(["powershell", "-noexit", 'deactivate'])