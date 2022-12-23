import json
import sys
import numpy as np
import perceplearn

def readModelParameters(filePath):
    try:
        parameters, weightsTrueFake, weightsPosNeg, biasTrueFake, biasPosNeg, uniqueWords = [], [], [], [], [], []
        with open(filePath, 'r', encoding='utf-8', errors='ignore') as inputFile:
            parameters = json.loads(inputFile.read())
            trueFake, posNeg = parameters['tf'], parameters['pn']

            weightsTrueFake = trueFake['weights']
            weightsPosNeg = posNeg['weights']

            biasTrueFake = trueFake['bias']
            biasPosNeg = posNeg['bias']
            
            uniqueWords = parameters['uniqueWords']

        weightsTrueFake = np.array([np.array(weight) for weight in weightsTrueFake])
        weightsPosNeg = np.array([np.array(weight) for weight in weightsPosNeg])
        biasTrueFake = np.asarray(biasTrueFake)
        biasPosNeg = np.asarray(biasPosNeg)

        return weightsTrueFake, weightsPosNeg, biasTrueFake, biasPosNeg, uniqueWords
    except:
        print('Failure in reading model parameters from file')

def readInputFile(filepath):
    fileContent = open(filepath, 'r')
    fileContent = fileContent.readlines()
    return fileContent

def writeOutputFile(fileContent, outputFileName):
    file = open(outputFileName, 'w')
    for sentence in fileContent:
        try:
            file.write(sentence +'\n')
        except:
            print(f'Error for: {sentence}')
            break
    file.close()

def predict(weightsTrueFake, weightsPosNeg, biasTrueFake, biasPosNeg, input):
    # print(input.shape, weightsTrueFake.shape, biasTrueFake.shape)
    # print(biasTrueFake)
    activationTF = np.dot(input, weightsTrueFake) + biasTrueFake
    activationPN = np.dot(input, weightsPosNeg) + biasPosNeg

    outputTF = 'True' if activationTF > 0 else 'Fake'
    outputPN = 'Pos' if activationPN > 0 else 'Neg'
    return outputTF, outputPN

if __name__=='__main__':
    modelFilePath, inputFilePath = sys.argv[1], sys.argv[2]
    weightsTrueFake, weightsPosNeg, biasTrueFake, biasPosNeg, uniqueWords = readModelParameters(modelFilePath)
    fileContent = readInputFile(inputFilePath)

    preprocessedFileContent, _ = perceplearn.preProcessing(fileContent, training=False)
    # print(f'After preprocessing: {preprocessedFileContent[0]}\n')

    dataset = perceplearn.Dataset(preprocessedFileContent, uniqueWords)
    inputs = dataset.generateInputs()
    # print(inputs.shape, len(uniqueWords))
    finalOutput = []
    for idx, oneHotSentence in enumerate(inputs):
        outputTF, outputPN = predict(weightsTrueFake, weightsPosNeg, biasTrueFake, biasPosNeg, oneHotSentence)
        newSentence = fileContent[idx].rstrip('\n').split(' ')
        newSentence = newSentence[0] + ' ' + outputTF + ' ' + outputPN
        finalOutput.append(newSentence)
    
    writeOutputFile(finalOutput, 'percepoutput.txt')
    