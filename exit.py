import subprocess
subprocess.run(["powershell", "-noexit", '.\\venv\\ExecutionPolicyUndefined.bat'])
subprocess.run(["powershell", "-noexit", 'deactivate'])