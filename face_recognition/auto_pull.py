#-*- coding: iso-8859-1 -*-
import os
import time

# Configuracion del repositorio
REPO_PATH = r"/home/eon/Documents/data-ras/Face_encodings/"
BRANCH = "main"
CHECK_INTERVAL =10 # Intervalo de tiempo para verificar cambios en segundos

def git_fetch():
    os.system(f"cd {REPO_PATH} && git fetch")

def check_for_updates():
    status = os.popen(f"cd {REPO_PATH} && git status").read()
    # print(status)
    if "Your branch is behind" in status:
        return True
    return False

def git_pull():
    os.system(f"cd {REPO_PATH} && git pull")
    print("Repository updated with latest changes.")

if __name__ == "__main__":
    try:
        while True:
            git_fetch()
            if check_for_updates():
                git_pull()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("Stopped by user")
