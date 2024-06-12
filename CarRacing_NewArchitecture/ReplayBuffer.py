import numpy as np
import pickle

class GameFrames(object):
    def __init__(self, inputShape, stateType, actionType, consecutive=4):
        self.consecutive = consecutive
        self.inputShape = inputShape
        self.stateType = stateType
        self.actionType = actionType

        self.gameFrames = np.array([], dtype=stateType)
        self.action = np.array([], dtype=actionType)
        self.reward = np.array([], dtype=np.float32)
        self.done = np.array([], dtype=np.int8)

        self.firstFrame = False
        

    def addFrame(self, frame, action, reward, done):
        if (self.firstFrame == False):
            self.gameFrames = frame
            self.gameFrames = np.expand_dims(self.gameFrames, axis=2)
            self.action = np.array([action], dtype=self.actionType)
            self.reward = np.array([reward], dtype=np.float32)
            self.done = np.array([done], dtype=np.int8)
            self.firstFrame = True
        else:
            frame = np.expand_dims(frame, axis=2)
            self.gameFrames = np.append(self.gameFrames, frame, axis=2)
            self.action = np.append(self.action, action)
            self.reward = np.append(self.reward, reward)
            self.done = np.append(self.done, done)
        

    def removeFrame(self):
        self.gameFrames = self.gameFrames[:,:,1:]
        self.action = self.action[1:]
        self.reward = self.reward[1:]
        self.done = self.done[1:]

        # If there are no frames in the game
        if (len(self.done) == 0):
            return 0
        
        return 1


    def getFrameBatch(self, batchSize):
        nFrames = self.gameFrames.shape[2]

        if (nFrames < (self.consecutive + batchSize)):
            return None
        
        batch = np.random.choice(nFrames-self.consecutive-1, batchSize)

        currFrame = np.array([self.gameFrames[:,:,b:b+self.consecutive] for b in batch], dtype=self.stateType)
        nextFrame = np.array([self.gameFrames[:,:,b+1:b+self.consecutive+1] for b in batch], dtype=self.stateType)

        action = np.array([self.action[b+self.consecutive] for b in batch], dtype=self.actionType)
        reward = np.array([self.reward[b+self.consecutive] for b in batch], dtype=np.float32)
        done = np.array([self.done[b+self.consecutive] for b in batch], dtype=np.int8)

        return currFrame, nextFrame, action, reward, done

    def getFrames(self):
        try:
            nFrames = self.gameFrames.shape[2]
        except:
            return None
        
        if (nFrames < self.consecutive):
            return None
        
        return self.gameFrames[:,:, -1 * self.consecutive:]

    def getSize(self):
        return len(self.done)

# Intended for use with sets of consecutive images
class ReplayBuffer(object):
    def __init__(self, memSize, inputShape, nActions, consecutive=4 , discrete=False, stateType=np.float32): 
        self.memMax = memSize
        self.discrete = discrete
        self.consecutive = consecutive
        self.inputShape = inputShape
        self.stateType = stateType
        self.actionType = np.int8 if self.discrete else np.float32
        self.nActions = nActions 
        self.games = []
        self.memPtr = 0
        self.rollOver = 0
    

    def store_transition(self, state, action, reward, done):
        # If at end of array roll pointer back to start
        if (self.memPtr >= self.memMax):
            self.memPtr = 0
            self.rollOver = 1
            print("**Replay buffer pointer rollover")
        

        # If we need to start deleting frames
        if (self.rollOver):
            # Delete first frame from the last game
            if (self.games[0].removeFrame() == 0):
                # If no frames left, delete the game
                del self.games[0]


        # Add the current frame to the gameFrames
        self.currentGame.addFrame(state, action, reward, 1 - int(done))

        self.memPtr += 1


    def sample_buffer(self, batchSize):
        # Get number of frames in each game
        sizes = [g.getSize() for g in self.games]

        # Calculate number of samples that can be had from each game
        sampleSizes = [s - 4 if (s-4 > 0) else 0 for s in sizes]
        total = np.sum(sampleSizes) 

        # Calculate proportions for each game based on number of samples
        gameProportion = [weight / total for weight in sampleSizes]

        # Uniformly select number of samples to use for each game 
        gameSamples = np.random.choice(len(self.games), batchSize, p=gameProportion)
        gameSamples = np.bincount(gameSamples, minlength=len(self.games))

        # Resample if the number of samples from a game is larger than available
        while (np.any((sampleSizes - gameSamples) < 0)):
            gameSamples = np.random.choice(len(self.games), batchSize, p=gameProportion)

            gameSamples = np.bincount(gameSamples, minlength=len(self.games))

            print("**Resample batch probabilities")

        # Use distribution to create a batch from multiple games
        first = True
        for i in range(len(self.games)):
            if (gameSamples[i] == 0):
                continue

            cF, nF, a, r, d = self.games[i].getFrameBatch(gameSamples[i])
            
            if (first):
                first = False
                currentFrames = cF
                nextFrames = nF
                action = a
                reward = r
                done = d
            else:
                currentFrames = np.append(currentFrames, cF, axis=0)
                nextFrames = np.append(nextFrames, cF, axis=0)
                action = np.append(action, a)
                reward = np.append(reward, r)
                done = np.append(done, d)

        return currentFrames, action, reward, nextFrames, done


    def sampleForAction(self):
        return self.currentGame.getFrames()
    
    
    def newGame(self):
        self.currentGame = GameFrames(self.inputShape, self.stateType, self.actionType, self.consecutive)
        self.games.append(self.currentGame)


    def saveGames(self, filename):
        with open(filename + "_games.pkl", "wb") as file:
            pickle.dump(self.games, file)

        with open(str(filename) + "_pointer.pkl", 'wb') as file:
            pickle.dump([self.memPtr, self.rollOver, self.memMax], file)
            

    def loadGames(self, filename):
        with open(filename + "_games.pkl", "rb") as file:
            self.games = pickle.load(file)

        with open(str(filename) + "_pointer.pkl", 'rb') as file:
            temp = pickle.load(file)
            self.memPtr = temp[0]
            self.rollOver = temp[1]
            self.memMax = temp[2]