from copy import deepcopy
import json
import sys
from bs4 import BeautifulSoup
import string
import re
import numpy as np

class Preprocessor:
    def __init__(self):
        self.stopWords = {}
        
    def urlRemoval(self, sentence):
        """Function to remove the HTML tags and URLs from reviews using BeautifulSoup

        Args:
            sentence (string): Sentence from which we remove the HTML tags and URLs

        Returns:
            string: Sentence which does not contain any HTML tags and URLs
        """
        return BeautifulSoup(sentence, 'lxml').get_text() 

    def nonAlphabeticRemoval(self, sentence):
        """Function to remove non-alphabetic characters from the sentence. 
        Note - We do not remove spaces from the sentence however extra spaces are removed in a different function

        Args:
            sentence (string): Sentence from which we remove the non-alphabetic characters

        Returns:
            string: Sentence from which non-alphabetic characters have been removed
        """
        return re.sub(r"[^a-zA-Z ]+", "", sentence)  #This will also remove numbers.

    def removeExtraSpaces(self, sentence):
        """Remove extra spaces from the sentence

        Args:
            sentence (string): Sentence from which we remove extra spaces

        Returns:
            string: Sentence from which extra spaces have been removed
        """
        return ' '.join(sentence.split())

    def removePunctuation(self, sentence):
        """Function to remove punctuations from a sentence

        Args:
            sentence (string): Sentence from which punctuations have to be removed

        Returns:
            string: Sentence from which punctuations have been removed
        """
        for value in string.punctuation:
            if value in sentence:
                sentence = sentence.replace(value, ' ')
        return sentence.strip()

    def convertToLowercase(self, sentence):
        return sentence.lower()
    
    def generateWordTokens(self, fileContent):
        wordTokens = dict()
        for sentence in fileContent:
            for word in sentence.rstrip('\n').split(' ')[3:]:
                wordTokens[word] = wordTokens.get(word, 0) + 1
        wordTokensList = [(key, value) for key, value in wordTokens.items()]
        wordTokensList = sorted(wordTokensList, key=lambda x:x[1], reverse=True)
        # print(wordTokensList[:20])
        
        for key,value in wordTokensList[:20]:    # Top - k stopwords.
            self.stopWords[key] = value # Easier to index in dictionary than search in list
        return wordTokens
    
    def stopWordRemoval(self, sentence):
        newSentence = []
        for word in sentence.rstrip('\n').split(' '):
            if self.stopWords.get(word) == None:
                newSentence.append(word)
        return ' '.join(newSentence)

class Dataset:
    def __init__(self, fileContent, uniqueWords):
        self.fileContent = fileContent
        self.uniqueWords = {value: idx for idx, value in enumerate(uniqueWords)}
        self.labelsPosNeg = [0] * len(self.fileContent)
        self.labelsTrueFake = [0] * len(self.fileContent)
        self.inputs = []

    def generateLabels(self):
        posNeg = {'Pos': 1 , 'Neg': -1}
        trueFake = {'True': 1, 'Fake': -1}

        for idx, sentence in enumerate(self.fileContent):
            sentenceLabels = sentence.rstrip('\n').split(' ')[1:3]
            self.labelsTrueFake[idx] = trueFake[sentenceLabels[0]]
            self.labelsPosNeg[idx] = posNeg[sentenceLabels[1]]
        return np.asarray(self.labelsPosNeg), np.asarray(self.labelsTrueFake)
    
    def generateInputs(self):
        for sentence in self.fileContent:
            oneHotSentence = [0] * len(self.uniqueWords)
            for word in sentence.rstrip('\n').split(' ')[3:]:
                if self.uniqueWords.get(word):
                    oneHotSentence[self.uniqueWords[word]] += 1 #Skipping unseen words for now
            self.inputs.append(oneHotSentence)
        self.inputs = np.array([np.array(input) for input in self.inputs])
        return self.inputs

def readInputFile(filepath):
    fileContent = open(filepath, 'r')
    fileContent = fileContent.readlines()
    return fileContent

def printDictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key} : {value}')

def preProcessing(fileContent, training = True):
    preprocessor = Preprocessor()
    sentenceList = []
    preprocessedFileContent = deepcopy(fileContent)
    for idx, sentence in enumerate(preprocessedFileContent):
        sentenceList = sentence.rstrip('\n').split(' ')
        intermediateSentence = ''
        if training:
            intermediateSentence = ' '.join(sentenceList[3:])
        else:
            intermediateSentence = ' '.join(sentenceList[1:])
        intermediateSentence = preprocessor.convertToLowercase(intermediateSentence)
        intermediateSentence = preprocessor.urlRemoval(intermediateSentence)
        intermediateSentence = preprocessor.removeExtraSpaces(intermediateSentence)
        intermediateSentence = preprocessor.removePunctuation(intermediateSentence)
        intermediateSentence = preprocessor.nonAlphabeticRemoval(intermediateSentence)
        intermediateSentence = preprocessor.removeExtraSpaces(intermediateSentence)
        if training:
            preprocessedFileContent[idx] = ' '.join(sentenceList[0:3]) + ' ' + intermediateSentence
        else:
            preprocessedFileContent[idx] = sentenceList[0] + ' ' + intermediateSentence

    wordTokens = preprocessor.generateWordTokens(preprocessedFileContent)
    # print(f'Before stop word removal: {preprocessedFileContent[0]}')
    for idx, sentence in enumerate(preprocessedFileContent):
        preprocessedFileContent[idx] = preprocessor.stopWordRemoval(sentence)
    # print(f'After stop word removal: {preprocessedFileContent[0]}')
    return preprocessedFileContent, wordTokens

def writeParameters(filePath, weightDictionary):
    with open(filePath, 'w', encoding='utf-8') as outputFile:
        # json.dump(weightDictionary, outputFile, indent=2, ensure_ascii=False)
        json.dump(weightDictionary, outputFile, ensure_ascii=False)
    
class SLPVanilla:
    def __init__(self, inputs, targets, numEpochs):
        # print(f'Shape of inputs {inputs.shape}')
        self.weights = np.zeros((inputs.shape[1], 1))
        self.bias = 0
        self.inputs = inputs
        self.targets = np.reshape(targets, (targets.shape[0], 1))
        self.numEpochs = numEpochs

    def train(self):
        for epoch in range(self.numEpochs):
            activation = np.dot(self.inputs, self.weights) + self.bias
            for idx, (input, label) in enumerate(zip(self.inputs, self.targets)): #dot product of label and activation might speed it up. Use np.where next
                if label * activation[idx][0] <= 0:    # Make labels -1, 1 for true/false
                    value = label * input
                    value = np.reshape(value, (value.shape[0], 1))
                    self.weights = self.weights + value
                    self.bias = self.bias + label
        return self.weights.flatten(), self.bias.flatten()

class SLPAveraged:
    def __init__(self, inputs, targets, numEpochs):
        # print(f'Shape of inputs {inputs.shape}')
        self.weights = np.zeros((inputs.shape[1], 1))
        self.bias = 0

        self.cachedWeights = np.zeros((inputs.shape[1], 1))
        self.cachedBias = 0

        self.inputs = inputs
        self.targets = np.reshape(targets, (targets.shape[0], 1))
        self.numEpochs = numEpochs
        self.counter = 1

    def train(self):
        for epoch in range(self.numEpochs):
            activation = np.dot(self.inputs, self.weights) + self.bias
            for idx, (input, label) in enumerate(zip(self.inputs, self.targets)): #dot product of label and activation might speed it up. Use np.where next
                if label * activation[idx][0] <= 0:    # Make labels -1, 1 for true/false
                    value = label * input
                    value = np.reshape(value, (value.shape[0], 1))

                    self.weights = self.weights + value
                    self.bias = self.bias + label

                    self.cachedWeights = self.cachedWeights + (value * self.counter)
                    self.cachedBias = self.bias + label * self.counter
                self.counter += 1

        self.weights = self.weights - (1/self.counter) * self.cachedWeights
        self.bias = self.bias - (1/self.counter) * self.cachedBias
        return self.weights.flatten(), self.bias.flatten()

if __name__ == "__main__":
    inputFilePath = sys.argv[1]
    fileContent = readInputFile(inputFilePath)
    
    #Preprocessing
    preprocessedFileContent, wordTokens = preProcessing(fileContent)
    uniqueWords = wordTokens.keys()

    #Create inputs and labels
    dataset = Dataset(preprocessedFileContent, uniqueWords)
    inputs = dataset.generateInputs()
    targetsPosNeg, targetsTrueFake = dataset.generateLabels()


    slpTrueFakeVanillaCLF = SLPVanilla(inputs, targetsTrueFake, 100)
    weightsTrueFakeVanilla, biasTrueFakeVanilla = slpTrueFakeVanillaCLF.train()

    slpPosNegVanillaCLF = SLPVanilla(inputs, targetsPosNeg, 100)
    weightsPosNegVanilla, biasPosNegVanilla = slpPosNegVanillaCLF.train()

    slpTrueFakeAveragedCLF = SLPAveraged(inputs, targetsTrueFake, 100)
    weightsTrueFakeAveraged, biasTrueFakeAveraged = slpTrueFakeAveragedCLF.train()

    slpPosNegAveragedCLF = SLPAveraged(inputs, targetsPosNeg, 100)
    weightsPosNegAveraged, biasPosNegAveraged = slpPosNegAveragedCLF.train()

    weightsDictionaryVanilla = {'tf':{'weights':weightsTrueFakeVanilla.tolist(), 'bias':biasTrueFakeVanilla.tolist()}, 'pn':{'weights':weightsPosNegVanilla.tolist(), 'bias':biasPosNegVanilla.tolist()}, 'uniqueWords': list(uniqueWords)}
    weightsDictionaryAveraged = {'tf':{'weights':weightsTrueFakeAveraged.tolist(), 'bias':biasTrueFakeAveraged.tolist()}, 'pn':{'weights':weightsPosNegAveraged.tolist(), 'bias':biasPosNegAveraged.tolist()}, 'uniqueWords': list(uniqueWords)}
    
    writeParameters('vanillamodel.txt', weightsDictionaryVanilla)
    writeParameters('averagedmodel.txt', weightsDictionaryAveraged)