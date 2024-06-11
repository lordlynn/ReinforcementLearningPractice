import matplotlib.pyplot as plt
import numpy as np

scores  = []


# Read scores in from file
with open("scores.txt", "r") as file: 
    temp = file.readlines(-1)



for i in range(len(temp)):
    if (temp[i] == "\n"):
        continue

    scores.append(float(temp[i]))



# Smooth version
smoothed = []
windowLen = 30
for i in range(windowLen, len(scores)):
    smoothed.append(np.mean(scores[i-windowLen:i]))


plt.figure(1)
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Raw Scores")



plt.figure(2)
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Moving Average Scores")


plt.show()