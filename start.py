import subprocess, os, sys

init = './venv/init.ps1'
update = './venv/update.ps1'
activate = './venv/activate.ps1'
command = init

os.system('.\\venv\\ExecutionPolicy.bat')

if os.path.isdir("./venv/venv") and ('-u' in sys.argv or '--update' in sys.argv):
    command = update
elif os.path.exists("./venv/venv/Scripts/Activate.ps1"):
    command = activate

if command != activate:
    subprocess.Popen([r'C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe',
                    '-ExecutionPolicy',
                    'Bypass',
                    command])
os.system('cls')

if command == activate:
    subprocess.run(["powershell", "-noexit", '.\\venv\\venv\\Scripts\\Activate.ps1'])
else:
    print('Waiting for lib to install', 'Restart \'start.py\' to activate venv.', sep='\n')
    exit()