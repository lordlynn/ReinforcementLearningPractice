import ReplayBuffer
import numpy as np

buffLen = 1000

# 96 * 96 * 3 * 4 * 2 * 100 = 22,118,400
# 4 * 3 * 100 = 1,200
states = np.array(np.random.random((buffLen, 96, 96, 3)), dtype=np.int32)
newStates = np.array(np.random.random((buffLen, 96, 96, 3)), dtype=np.int32)
rewards = np.array(np.random.random(buffLen), dtype=np.int32)
actions = np.array([2 for i in range(buffLen)], dtype=np.int32)
terminal = np.array(np.random.random((buffLen)), dtype=np.int32)


buff = ReplayBuffer.ReplayBuffer(maxSize=buffLen, inputShape=(96,96,3), nActions=4, discrete=True)



for i in range(buffLen):
    buff.store_transition(states[i], actions[i], rewards[i], newStates[i], terminal[i])

buffFile = "buffTest"
buff.load_buffer(buffFile)

pass