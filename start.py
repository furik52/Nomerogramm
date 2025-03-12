import subprocess, os

init = './venv/init.ps1'
update = './venv/update.ps1'
activate = './venv/activate.ps1'
command = init

os.system('.\\venv\\ExecutionPolicy.bat')

if os.path.isdir("./venv/venv"):
    command = str(input('Enter the script(update/activate, default = activate):'))
if command == 'init' or command == 'i':
    command = init
elif command == 'update' or command == 'u':
    command = update
else:
    command = activate

if not(os.path.isdir("./venv/venv")):
    command = init

if command != activate:
    subprocess.Popen([r'C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe',
                    '-ExecutionPolicy',
                    'Bypass',
                    command])
os.system('cls')

if os.path.isdir("./venv/venv/Scripts") and command == activate:
    subprocess.run(["powershell", "-noexit", '.\\venv\\venv\\Scripts\\Activate.ps1'])
else:
    print('Waiting for lib to install', 'Restart \'start.py\' to activate venv.', sep='\n')
    exit()