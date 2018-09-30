
import math
import argparse

# Arguments format as shared in assignment
parser = argparse.ArgumentParser()
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)

args = vars(parser.parse_args())
Xtrain_name = args['f1']
Xtest_name = args['f2']
Out_name = args['o']

totalMailCount = 0
hamWordCount = 0
spamWordCount = 0

mailCount = {} # to track number of ham or spam mails received
wordCounts = {} # track of how many times word appears in spam/ham
prior = {} # store the prior probability estimation values
conditional = {} # store the conditional probabilities

dictionary = set() # a set of unique words present in training set.

def initialize():
    global totalMailCount
    global hamWordCount
    global spamWordCount
    global mailCount
    global wordCounts
    global prior
    global conditional
    global dictionary

    totalMailCount = 0
    hamWordCount = 0
    spamWordCount = 0
    mailCount = {}
    wordCounts = {}
    prior = {}
    conditional = {}
    dictionary = set()

    trainer = open(Xtrain_name,'r')

    for line in trainer:
        tokens = line.split(" ")[2:] # dont include ID and mail type
        for (index, word) in enumerate(tokens):
            if index % 2 == 0:
                tokens += [word]

        for item in tokens:
            dictionary.add(item)

    trainer.close()

def BinomialClassifier():

    global dictionary
    global totalMailCount
    global mailCount
    global wordCounts
    global prior
    global conditional
    global hamWordCount
    global spamWordCount

    # reset DS for this round of calculations
    print("For BinomialClassifier:")
    wordCounts.setdefault("spam", {})
    wordCounts.setdefault("ham", {})

    for word in dictionary:
        wordCounts["spam"].setdefault(word, 0)
        wordCounts["ham"].setdefault(word, 0)

    # reading the words as well as count of occurances
    trainer = open(Xtrain_name, 'r')

    for line in trainer:
        totalMailCount += 1
        tokens = line.split(" ")
        type = tokens[1]
        mailCount.setdefault(type, 0)
        mailCount[type] += 1

        for i in range(2, len(tokens), 2):
            wordCounts[type][tokens[i]] += 1

    # calculate the prior and conditional probability
    for word, count in wordCounts["spam"].items():
        spamWordCount += count

    for word, count in wordCounts["ham"].items():
        hamWordCount += count

    for type, count in mailCount.items():
        prior.setdefault(type, 0.0)
        prior[type] = float(count/totalMailCount)
        print("prior prob: "+str(float(count/totalMailCount)))

    conditional.setdefault("spam", {})
    conditional.setdefault("ham", {})

    for type, attribute in wordCounts.items():
        for word, count in attribute.items():
            conditional[type].setdefault(word, 0)
            conditional[type][word] = float((count + 1) / float(mailCount[type] + len(dictionary)))


def MultinomialClassifier():
    global dictionary
    global totalMailCount
    global mailCount
    global wordCounts
    global prior
    global conditional
    global hamWordCount
    global spamWordCount

    # reset DS for next calculations
    print("For MultinomialClassifier:")
    wordCounts.setdefault("spam", {})
    wordCounts.setdefault("ham", {})

    for word in dictionary:
        wordCounts["spam"].setdefault(word, 0)
        wordCounts["ham"].setdefault(word, 0)

    trainer = open(Xtrain_name, 'r')

    for line in trainer:
        totalMailCount += 1
        tokens = line.split(" ")
        type = tokens[1]
        mailCount.setdefault(type, 0)
        mailCount[type] += 1

        for i in range(2, len(tokens), 2):
            wordCounts[type][tokens[i]] += int(tokens[i + 1])

        tokens = []

    for word, count in wordCounts["ham"].items():
        hamWordCount += count
    for word, count in wordCounts["spam"].items():
        spamWordCount += count

    for type, count in mailCount.items():
        prior.setdefault(type, 0.0)
        prior[type] = float(count / totalMailCount)

    conditional.setdefault("ham", {})
    conditional.setdefault("spam", {})

    for type, item in wordCounts.items():
        for word, count in item.items():
            conditional[type].setdefault(word, 0)
            if (type in "spam"):
                conditional[type][word] = float(
                    (count + 1) / float(spamWordCount + len(dictionary)))
            elif (type in "ham"):
                conditional[type][word] = float(
                    (count + 1) / float(hamWordCount + len(dictionary)))


def performClassification():
    global conditional
    global hamWordCount
    global spamWordCount

    testData = open(Xtest_name,'r')
    output = open(Out_name,'w')

    actualClassList = list()
    calculatedClassList = list()

    # RM: iterate over every test set email
    for eachLine in testData:
        tokens = eachLine.split(" ")
        testID = tokens[0]
        actualClass = tokens[1]

        predictedClass = ""
        wordList = []
        for index, word in enumerate(tokens[2:]):
            if index % 2 == 0:
                wordList += [word]

        pHam = 0.0
        pSpam = 0.0

        # Note: summing up logs of probabilities rather than multiplying to avoid floating-point underflow.
        for word in wordList:
            if word not in dictionary:
                pHam += math.log10(1 / float(hamWordCount + len(dictionary)))
                pSpam += math.log10(1 / float(spamWordCount + len(dictionary)))
            else:
                pHam += math.log10(conditional["ham"][word])
                pSpam += math.log10(conditional["spam"][word])

        if pSpam > pHam:
            predictedClass = "spam"
        else:
            predictedClass = "ham"

        actualClassList.append(actualClass)
        calculatedClassList.append(predictedClass)

        # Output file must be in format <ID> <Spam/Ham>, based on piazza note
        output.write(testID + " " + predictedClass + "\n")

    testData.close()
    output.close()

    messageBox(actualClassList, calculatedClassList)

def messageBox(groundTruthList, predictedTruthList):

    correctSpam = 0
    correctHam = 0
    incorrectSpam = 0
    incorrectHam = 0


    for actual,predicted in zip(groundTruthList,predictedTruthList):
        if (predicted in "spam"):
            if(actual in "spam"):
                correctSpam += 1
            elif(actual in "ham"):
                incorrectSpam += 1

        elif (predicted in "ham"):
            if(actual in "ham"):
                correctHam += 1
            elif (actual in "spam"):
                incorrectHam += 1

    precision = float(correctSpam/float(correctSpam+incorrectSpam)) * 100
    recall = float(correctSpam/float(correctSpam+incorrectHam)) * 100
    fmeasure = (2*precision*recall)/(precision+recall) # harmonic mean of precision and recall

    print("Precision: " + str(precision)+" ,"+"Recall: " + str(recall)+" ,"+"F-Measure: "+ str(fmeasure))

    print("")

if __name__ == "__main__":
    # note: binomial classifier outputs data to the output file name received as arguments
    initialize()
    BinomialClassifier()
    performClassification()

    # # however for multinomial, we create a separate file.
    # initialize()
    # Out_name += "_multinomial"
    # MultinomialClassifier()
    # # for type, count in mailCount.items():
    # #     print("type: "+str(type)+" , "+"count: "+str(count)+" , "+"total: "+str(total))
    # performClassification()