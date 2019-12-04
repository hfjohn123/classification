import util

class kNNClassifier:
    def __init__(self, legalLabels, k):
        self.legalLabels = legalLabels
        self.type = "k Nearest Neighbors"
        self.k = k
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.tdata= trainingData
        self.tlable = trainingLabels
    def classify(self, testData):
        res = []
        for test in testData:
            guesses = []
            for i,data in enumerate(self.tdata):
                sim = data*test
                guesses.append((sim,i))
            guesses.sort(reverse=True)
            k_guess = []
            for x in range(int(self.k)):
                k_guess.append(guesses[x][1])
            ans = util.Counter()
            for x in k_guess:
                ans[self.tlable[x]] += 1
            res.append(ans.argMax())
        return res
