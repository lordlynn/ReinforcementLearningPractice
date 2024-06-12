import subprocess

for i in range (20):
    print("episode: " + str(i) + "\n\n")
    subprocess.run(['python', 'carRacing.py'])

