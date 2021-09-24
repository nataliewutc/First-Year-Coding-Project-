'''Welikala, Vinuk
Surachescomson, Inkarat
Wu, Natalie
Moustiri, Younes'''

import numpy as np
from numpy import random as rnd
from math import log
import matplotlib.pyplot as plt
import ast
import time
import os


#File will be saved in the same location as where this code is saved
''' 
Since:
k1.1 = 0.36
k2.1 = 3.6*10**7
k3.1 = k4.1
k3.2 = k3.3 = k4.2 = k4.3
k5.1 = k5.2 = k5.3 = k5.4
Then:
input for InputKx = [(k31 = k41) , (k32 = k33 = k42 = k43), (k51 = k52 = k53 = k54)]
dictionary for self.Kx values = {"K11": input[0] ...."K53": input[4],"K54": input[4] }
so we only input 4 values as a list for self.Kx.
'''


def createFolders():
    '''Creates a set of folders, which are named from the elements of the folder list if the folders do not exist already'''
    for i in folders:
        try:
            if not os.path.exists(i):
                os.makedirs(i)
        except OSError:
            print('Error: Creating directory. ' + i)

class Timer:
    '''This class is used as a timer as we can use it to start a timer, reset it and get the time at any point in the code'''
    def __init__(self):
        self.startTime = time.time()  # timer = Timer() [initialize the timer]

    def startTimer(self):
        self.startTime = time.time()  # timer.startTimer() [resets the timer to time=0]

    def getTime(self):
            return (time.time() - self.startTime)  # timer.getTime() [returns the time in seconds as an integer rounded down to the nearest second]






class REACTION:
    '''Contains the files, plots, reactions and evolving the system'''
    def __init__(self, I, T, InputKx, directory, M=5):
        ''' This function performs the reaction:
        -It takes I (concentration of initiators), T (concentration of RAFT agents), directory (the name of the files), and a preset value of M (concentration of monomers) as an input
        -Creates a dictionary with all the k values. Works out rate of reactions the probibility for each reaction.
        -Then finds the reactants and products.
        -Evolves the system.'''
        self.directory = directory
        V = 0.1*10**-8 #Volume (changed so that we can change the number of molecules. If this is not done then the number of molecules is too big to take away 1 molecule.)
        Na = 6022 * (10 ** 20) #Avogadro's constant
        alpha = V * Na # the conversion factor that allows us to go from the concentration of each specie to their number of molecules
        self.Kx = {
        "k1.1":0.36/3600,
        "k2.1": 36 * 10 ** (6)/3600,
        "k3.1":InputKx[0]/3600,
        "k3.2":InputKx[1]/3600,
        "k3.3":InputKx[1]/3600,
        "k4.1":InputKx[0]/3600,
        "k4.2":InputKx[1]/3600,
        "k4.3":InputKx[1]/3600,
        "k5.1":InputKx[2]/3600,
        "k5.2":InputKx[2]/3600,
        "k5.3":InputKx[2]/3600,
        "k5.4":InputKx[2]/3600} #dictionary for all values of kx, which are divided by 3600 to get the reaction per second.
        self.recordsMade = 0  # records the number of times results are recorded (to choose files to plot)
        self.filesSaved = []  # records the file names that are saved
        self.timeOfFilesSaved = []  # records the time that each files are saved at
        self.recordsToPlot = []  # records the file names + locations of records to be plotted
        self.numOfRecordsToPlot = []  # records the number of the files to plot
        self.averageLength = []  # average length of radicals in each time file is saved
        self.fileName = ''  # name of files to be saved
        self.fracDict = {}  # dictionary of fraction of living radicals
        self.updatedTime = 0 #the time at
        recordingPeriod = 10  # period of time to wait between recording results
        recordingTime = 10  # when updated time exceeds this limit, results are recorded and 'recordingTime' is incremented by 'recordingPeriod'

        #Validation to check that only the accepted values of Kx, I and T are used.
        if I != 10 ** (-3) and I != 10 ** (-2):
            raise ValueError("Your I value was not one of the accepted I values: Accepted values are 10**-3 or 10**-2 units: M")
        if T != 10 ** -3 and T != 10 ** -2 and T != 0:
            raise ValueError(
                "Your T value was not one of the accepted T values: Accepted values are 10**-3 or 10**-2 or 0 units: M")
        if self.Kx["k3.1"] < 36 * 10 ** (8)/3600 or self.Kx["k3.1"] > 36 * 10 ** (10)/3600:
            raise ValueError(
                "Your K31 (= K41) value was not one of the accepted K31 values: Accepted values are numbers between 3.6 * 10 ** (9) and 3.6 * 10 ** (11) units: h^-1*M^-1")
        if self.Kx["k3.2"] < 18 * 10 ** (7)/3600 or self.Kx["k3.2"] > 18 * 10 ** (9)/3600:
            raise ValueError(
                "Your K32 (= K33 = K42 = K43) value was not one of the accepted K32 values: Accepted values are numbers between 18 * 10 ** (7) and 18 * 10 ** (9) units: h^-1*M^-1")
        if self.Kx["k5.1"] < 36 * 10 ** (9)/3600 or self.Kx["k5.1"] > 36 * 10 ** (11)/3600:
            raise ValueError(
                "Your K51 (= K52 = K53 = K54) value was not one of the accepted K51 values: Accepted values are between numbers 3.6 * 10 ** (10) and 3.6 * 10 ** (12) units: h^-1*M^-1")
        #The number of molecules of M, I and T, given their concentrations
        self.M = float(round(M * alpha))
        self.I = float(round(I * alpha))
        self.T = float(round(T * alpha))
        # Dictionary for all the molecules of different types. Key = n, Value = number of molecules
        self.chainRadicals = {} #R_n•
        self.chainTRadicals = {} #•TR_n
        self.chainT = {0: self.T} #TR_n
        self.chainTChainRadicals = {} #R_n•TR_n
        self.chainTChainSeparate = {}
        self.subChainTChain = np.array([])
        self.Polymer = {} #P_n
        while timer.getTime() < 3600:
            self.Rn = sum(self.chainRadicals.values())
            self.TRn = sum(self.chainTRadicals.values())
            self.Tn = sum(self.chainT.values())
            self.RnTRn = sum(self.chainTChainRadicals.values())
            self.Rx = {11: self.Kx["k1.1"] * (2 * self.I * (1 / (alpha))), #Rate of reaction for each reaction
                       21: self.Kx["k2.1"] * self.Rn * self.M * (1 / (alpha)) ** 2,
                       32: self.Kx["k3.2"] * self.TRn * (1 / (alpha)),
                       33: self.Kx["k3.3"] * self.TRn * (1 / (alpha)),
                       41: self.Kx["k4.1"] * self.Rn * self.Tn * (1 / (alpha)) ** 2,# Reaction 41 accounts for reaction 31 too
                       42: self.Kx["k4.2"] * 2 * self.RnTRn * (1 / (alpha)),
                       # Rx[42] accounts for both reactions 4.2 and 4.3, thus multiplied by 2. These have the same rate of reaction (50% chance of each of the reaction occuring), which we will get into later on. see (1).
                       51: self.Kx["k5.1"] * self.Rn * self.Rn * (1 / (alpha)) ** 2,
                       52: self.Kx["k5.2"] * self.Rn * self.Rn * (1 / (alpha)) ** 2,
                       54: self.Kx["k5.4"] * self.RnTRn * self.Rn * (1 / (alpha)) ** 2}  # Rx[54] accounts for both reactions 5.3 and 5.4 as self.Rn includes self.RadicalMol[0], which is a radical with no monomers attatched

            if self.Rn == 1:
                self.Rx[51] = 0
                self.Rx[52] = 0 #We need atleast 2 chainRadicals for the reaction to occur

            self.Rt = sum(self.Rx.values()) # cumulative rate of all reactions
            r = rnd.random()
            tau = (1 / self.Rt) * log(1 / r)  # time tau passed from last reaction (simulated time)

            self.updatedTime = self.updatedTime + tau

            if self.Rt == 0:
                break #Stops the reaction if the rate of reaction = 0
            self.Px = {} #Probability of the reactions occuring

            for i in self.Rx.keys():
                self.Px[i] = self.Rx[i] / self.Rt #calculating the probability for each reaction
            self.reaction = rnd.choice(list(self.Rx.keys()), replace=True, p=list(self.Px.values()))#chooses a random reaction based on current reactants available

            if self.reaction == 11: #Reaction 1.1
                self.product = 0 #the value of self.product is the length of the polymer chain produced which is 0 in this case
                n = (list(self.chainRadicals.keys())).count(self.product) #gives a value of n=0 or n=1 based on whether there is a radical with chain length of 0
                if n == 0:
                    self.chainRadicals[self.product] = 2 * 1 #this will create a new key:value pair in the chainRadicals dictionary. 2 free radicals are produced from one inititaor, hence 2 * 1 radicals
                else:
                    self.chainRadicals[self.product] += 2 * 1 #this will add to the current key:value pair in the chainRadicals dictionary
                self.I -= 1 #subtracts 1 initiator molecule from the total number of initiators


            elif self.reaction == 21: #Reaction 2.1
                self.reactant = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values()))) / self.Rn) #this chooses a random reactant available to this reaction which in this case is everything in the chainRadicals dictionary
                self.product = self.reactant + 1 #adds 1 to the length of product, as polymer chain is extended by 1 after the reaction happens
                n = (list(self.chainRadicals.keys())).count(self.product)
                if n == 0:
                    self.chainRadicals[self.product] = 1
                else:
                    self.chainRadicals[self.product] += 1
                self.chainRadicals[self.reactant] -= 1
                self.M -= 1 #subtracts 1 monomer from the total number of monomers


            elif self.reaction == 32:  # Reaction 3.2
                self.reactant = rnd.choice(list(self.chainTRadicals.keys()), replace=True, p=(np.array(list(self.chainTRadicals.values())) / self.TRn))  # picks a random reactant from the chainTRadicals dictionary
                self.product = self.reactant  # length of chain is the same for the reactant and product
                n = (list(self.chainRadicals.keys())).count(self.product)
                if n == 0:
                    self.chainRadicals[self.product] = 1
                else:
                    self.chainRadicals[self.product] += 1
                self.chainT[0] += 1  # T is the 2nd product of the reaction.
                self.chainTRadicals[self.reactant] -= 1  # subtracts the initial reactant used from the chainTRadicals dictionary
            elif self.reaction == 33:  # Reaction 3.3
                self.reactant = rnd.choice(list(self.chainTRadicals.keys()), replace=True,p=(np.array(list(self.chainTRadicals.values())) / self.TRn))
                self.product = self.reactant  # length of chain is the same for the reactant and product
                n = (list(self.chainT.keys())).count(self.product)
                if n == 0:
                    self.chainT[self.product] = 1
                else:
                    self.chainT[self.product] += 1
                self.chainRadicals[0] += 1  # R0 is the 2nd product of the reaction.
                self.chainTRadicals[self.reactant] -= 1  # subtracts the initial reactant used from the chainTRadicals dictionary
            ###
            elif self.reaction == 41:  # Reaction 4.1 and 3.1 combined
                self.reactant1 = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses first reactant from chainRadicals dictionary
                self.reactant2 = rnd.choice(list(self.chainT.keys()), replace=True, p=(np.array(list(self.chainT.values())) / self.Tn))  # chooses second reactant from chainT dictionary
                if self.reactant2 == 0:  # checks if chain for reactant2 is 0, if so then reaction 3.1 is taking place
                    self.product = self.reactant1  # product has same chain length as reactant1
                    n = (list(self.chainTRadicals.keys())).count(self.product)
                    if n == 0:
                        self.chainTRadicals[self.product] = 1
                    else:
                        self.chainTRadicals[self.product] += 1
                else:  # reaction 4.1 begins here
                    if self.reactant1 < self.reactant2:
                        self.product = f"{self.reactant1}T{self.reactant2}"
                    else:
                        self.product = f"{self.reactant2}T{self.reactant1}"  # orders it, so that we can't have 2T1 and 1T2, we only have 1T2, which accounts for both
                    n = (list(self.chainTChainRadicals.keys())).count(self.product)
                    if n == 0:
                        self.chainTChainRadicals[self.product] = 1
                    else:
                        self.chainTChainRadicals[self.product] += 1
                self.chainRadicals[self.reactant1] -= 1  # subtracts reactant from dictionary
                self.chainT[self.reactant2] -= 1
            elif self.reaction == 42:  # This includes reaction 4.3 as well as reaction 4.2. We find which reaction occurs in this statement too. The only difference between the 2 reactions is the product that contains the RAFT agent
                self.reactant = rnd.choice(list(self.chainTChainRadicals.keys()), replace=True,p=(np.array(list(self.chainTChainRadicals.values())) / self.RnTRn))
                self.sub_reactant = np.array(self.reactant.split("T"),np.int)  # Each sub_reactant = different block's length
                i = rnd.randint(0,2)  # The radical has a chain before the raft agent and after the raft agent. There is a 50% chance that reaction 4.2 occurs and a 50% chance reaction 4.3 occurs. We use this to find which one happens.
                if i == 0:
                    self.productT = self.sub_reactant[0]
                    self.productNoT = self.sub_reactant[1]
                else:
                    self.productT = self.sub_reactant[1]
                    self.productNoT = self.sub_reactant[0]
                n = (list(self.chainT.keys())).count(self.productT)
                if n == 0:
                    self.chainT[self.productT] = 1
                else:
                    self.chainT[self.productT] += 1
                n = (list(self.chainRadicals.keys())).count(self.productNoT)
                if n == 0:
                    self.chainRadicals[self.productNoT] = 1
                else:
                    self.chainRadicals[self.productNoT] += 1
                self.chainTChainRadicals[self.reactant] -= 1  # subtracts reactant from dictionary
            elif self.reaction == 42:  # This includes reaction 4.3 as well as reaction 4.2. We find which reaction occurs in this statement too. The only difference between the 2 reactions is the product that contains the RAFT agent
                self.reactant = rnd.choice(list(self.chainTChainRadicals.keys()), replace=True,p=(np.array(list(self.chainTChainRadicals.values())) / self.RnTRn))
                self.sub_reactant = np.array(self.reactant.split("T"),np.int)  # Each sub_reactant = different block's length
                i = rnd.randint(0,2)  # The radical has a chain before the raft agent and after the raft agent. There is a 50% chance that reaction 4.2 occurs and a 50% chance reaction 4.3 occurs. We use this to find which one happens.
                if i == 0:
                    self.productT = self.sub_reactant[0]
                    self.productNoT = self.sub_reactant[1]
                else:
                    self.productT = self.sub_reactant[1]
                    self.productNoT = self.sub_reactant[0]
                n = (list(self.chainT.keys())).count(self.productT)
                if n == 0:
                    self.chainT[self.productT] = 1
                else:
                    self.chainT[self.productT] += 1
                n = (list(self.chainRadicals.keys())).count(self.productNoT)
                if n == 0:
                    self.chainRadicals[self.productNoT] = 1
                else:
                    self.chainRadicals[self.productNoT] += 1
                self.chainTChainRadicals[self.reactant] -= 1  # subtracts reactant from dictionary
            elif self.reaction == 51:  # Reaction 5.1
                self.reactant1 = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses first reactant from chainRadicals dictionary
                self.chainRadicals[self.reactant1] -= 1  # subtracts reactant1 from dictionary
                self.Rn = sum(self.chainRadicals.values())  # calculates a new value of Rn for choosing the second reactant
                self.reactant2 = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses second reactant from chainRadicals dictionary
                self.chainRadicals[self.reactant2] -= 1  # subtracts reactant2 from dictionary
                self.product = self.reactant1 + self.reactant2  # the product chain length is the sum of the reactant chain lengths
                n = (list(self.Polymer.keys())).count(self.product)  # checks Polymer dictionary to see if it the product already exists
                if n == 0:
                    self.Polymer[self.product] = 1
                else:
                    self.Polymer[self.product] += 1
            elif self.reaction == 52:  # Reaction 5.2
                self.reactant1 = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses first reactant from chainRadicals dictionary
                self.chainRadicals[self.reactant1] -= 1  # subtracts reactant1 from dictionary
                self.Rn = sum(self.chainRadicals.values())  # calculates a new value of Rn for choosing the second reactant
                self.reactant2 = rnd.choice(list(self.chainRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses second reactant from chainRadicals dictionary
                self.chainRadicals[self.reactant2] -= 1  # subtracts reactant2 from dictionary
                self.product1 = self.reactant1  # one of the products has the same chain length as one of the reactants
                self.product2 = self.reactant2  # the other product has the same chain length as the other reactant
                n1 = (list(self.Polymer.keys())).count(self.product1)  # checks Polymer dictionary to see if it the product already exists
                if n1 == 0:
                    self.Polymer[self.product1] = 1
                else:
                    self.Polymer[self.product1] += 1
                n2 = (list(self.Polymer.keys())).count(self.product2)  # checks Polymer dictionary to see if it the product already exists
                if n2 == 0:
                    self.Polymer[self.product2] = 1
                else:
                    self.Polymer[self.product2] += 1

            elif self.reaction == 53:  # Reaction 5.3 and 5.4 combined
                self.reactant1 = rnd.choice(list(self.chainTRadicals.keys()), replace=True, p=(np.array(list(self.chainRadicals.values())) / self.Rn))  # chooses first reactant from chainRadicals dictionary
                self.reactant2 = rnd.choice(list(self.chainTChainRadicals.keys()), replace=True, p=(np.array(list(self.chainTChainRadicals.values())) / self.RnTRn))  # chooses second reactant from chainTChainRadicals dictionary
                self.product = np.array(self.reactant2.split("T"),np.int).sum() + self.reactant1  # the keys in the chainTChainRadicals dictionary are written as strings in the format aTb where T is raft constant. The T is removed and the chains either side are summed with the other reactant to get the product length
                n = (list(self.Polymer.keys())).count(self.product)  # checks Polymer dictionary to see if it the product already exists
                if n == 0:
                    self.Polymer[self.product] = 1
                else:
                    self.Polymer[self.product] += 1
                self.chainRadicals[self.reactant1] -= 1  # subtracts reactant1 from dictionary
                self.chainTChainRadicals[self.reactant2] -= 1  # subtracts reactant2 from dictionary

            if timer.getTime() >= recordingTime: #Everytime the time goes past a multiple of 10/is a multiple of 10.
                for i in self.chainTChainRadicals.keys():
                    for j in range(self.chainTChainRadicals[i]):
                        self.subChainTChainRadicals = np.array(np.append(self.subChainTChain, i.split("T")), np.int)
                for i in set(self.subChainTChain):
                    list(self.subChainTChain).count(i)
                    self.chainTChainSeparate[i] = list(self.subChainTChain).count(i) #lines 307-312 creates a dictionary with the chains on either side of the RAFT agent seperated puts the chains with the same length together
                self.recordsMade += 1
                self.timeOfFilesSaved.append(timer.getTime())

                self.chainTNoT=self.chainT.copy()
                self.chainTNoT.pop(0) #creating a new dictionary with no singular RAFT agent molecules, this will be used instead of the previous dictionary for the files

                avrRad = sum([k * v for k, v in self.chainRadicals.items()])  # calculate the combined lengths of radicals in each dict by multiplying each key with value and adding it all up
                avrChain = sum([d * s for d, s in self.chainTRadicals.items()])
                avrT = sum([g * h for g, h in self.chainTNoT.items()])
                avrPoly = sum([o * c for o, c in self.Polymer.items()])
                avrChainT = sum([a * b for a, b in self.chainTChainSeparate.items()])
                total = sum(self.chainRadicals.values()) + sum(self.chainTRadicals.values()) + sum(self.chainTNoT.values()) + sum(self.Polymer.values()) + sum(self.chainTChainSeparate.values())  # total number of living radicals
                avr = (avrRad + avrChain + avrT + avrPoly + avrChainT) / total  # Calculate average length of radicals
                self.averageLength.append(avr)

                self.fileName = '{0}\system_{1:05d}'.format(self.directory,self.recordsMade)

                data = open(self.fileName, 'w+')


                fracDictKeys = set()  # creates keys for the dictionary of fractions
                k1 = list(self.chainRadicals.keys())
                k2 = list(self.chainTRadicals.keys())
                k3 = list(self.chainTNoT.keys())
                k4 = list(self.Polymer.keys())
                k5 = list(self.chainTChainSeparate.keys())
                fracDictKeys.update(k1, k2, k3, k4, k5) #A set is created with all the different keys (different lengths)
                self.fracDict = dict.fromkeys(fracDictKeys, 0)

                # check if the keys of each dictionary match those of fracDict,
                # if yes add the value of dictionry to corresponding key in fracDict
                self.fracDict = {key: self.fracDict.get(key, 0) + self.chainRadicals.get(key, 0)
                                 for key in set(self.fracDict) | set(self.chainRadicals)}
                self.fracDict = {key: self.fracDict.get(key, 0) + self.chainTRadicals.get(key, 0)
                                 for key in set(self.fracDict) | set(self.chainTRadicals)}
                self.fracDict = {key: self.fracDict.get(key, 0) + self.chainTNoT.get(key, 0)
                                 for key in set(self.fracDict) | set(self.chainTNoT)}
                self.fracDict = {key: self.fracDict.get(key, 0) + self.Polymer.get(key, 0)
                                 for key in set(self.fracDict) | set(self.Polymer)}
                self.fracDict = {key: self.fracDict.get(key, 0) + self.chainTChainSeparate.get(key, 0)
                                 for key in set(self.fracDict) | set(self.chainTChainSeparate)}

                Ln = sum(self.fracDict.values())  # calculate the fraction of living radicals for each length
                fraction = list(np.array(list(self.fracDict.values())) / Ln)
                zipFrac = zip(fracDictKeys, fraction)
                self.fracDict = dict(zipFrac)
                fractionExport = str(self.fracDict)
                data.write(f"Length of radicals and the corresponding fractions\n{fractionExport}")
                data.close()
                self.filesSaved.append(self.fileName)

                if self.recordsMade % 5 == 0:  # take a file to plot every 5 files exported
                    self.recordsToPlot.append(self.fileName)
                    self.numOfRecordsToPlot.append(self.recordsMade)
            if timer.getTime() > recordingTime:
                recordingTime += recordingPeriod


    def dataToPlot(self):
        '''Plots a P(n) vs n graph for every 5 files created
        '''
        for i in self.recordsToPlot:  # plot each file added to this list
            plotData = open(i, 'r')
            plotData = ast.literal_eval(
                (plotData.readlines()[1]))  # Converts string back in to dictionary of fractions
            x = list(plotData.keys())[1:]  # n
            y = list(plotData.values())[1:]  # P(n)
            plt.plot(x, y, 'bo')
            plt.xlabel('n')
            plt.ylabel('P(n)')
            plt.title('P(n) vs n')
            currentPlotNum = self.numOfRecordsToPlot.pop(0)
            plt.savefig('{0}\pnplot_{1:05d}.jpg'.format(self.directory,currentPlotNum))
            plt.close()

    def widthPlot(self):
        '''Plots the W vs time graph'''
        listOfW = list() # list of W values for each file saved
        for i in self.filesSaved:
            widthData = open(i, 'r')
            widthData = ast.literal_eval(widthData.readlines()[1])
            avrLen = self.averageLength.pop(0)  # avrLen = <n>
            W = 0 # width of the distribution
            for index in range(len(widthData)):  # calculate W = Σ (n-<n>)^2 * P(n)
                width = ((int(list(widthData.keys())[index]) - avrLen) ** 2) * list(widthData.values())[index]  # W for each n calculated
                W = W + width  # summing W for all n
            listOfW.append(W)
        # W vs Time of the system
        x1 = self.timeOfFilesSaved
        y1 = listOfW
        plt.figure()
        plt.plot(x1, y1, 'go')
        plt.xlabel(' Time (s)')
        plt.ylabel('W')
        plt.title('Polydispersity in the system')
        plt.savefig('{0}\W_plot.jpg'.format(self.directory))
        plt.close()


I = [10**-2,10**-3]
T = [0,10**-2,10**-3]
InputKx = [36 * 10 ** (9),18 * 10 ** (8),36 * 10 ** (10)]

#We have multiple reactions for each of the different combinations of I and T. Hence we have 6 folders (1 for each combination) and we run the reactions and plot functions 6 times
folders = ['time_series_data1','time_series_data2','time_series_data3','time_series_data4','time_series_data5','time_series_data6']
createFolders() #creates new folders with names as fiven in the folders list, which is where we save the files.
timer = Timer()
firstReaction = REACTION(I[0], T[0], InputKx, folders[0])
firstReaction.dataToPlot()
firstReaction.widthPlot()

timer.startTimer()
secondReaction = REACTION(I[0], T[1], InputKx, folders[1])
secondReaction.dataToPlot()
secondReaction.widthPlot()

timer.startTimer()
thirdReaction = REACTION(I[0], T[2], InputKx, folders[2])
thirdReaction.dataToPlot()
thirdReaction.widthPlot()

timer.startTimer()
forthReaction = REACTION(I[1], T[0], InputKx, folders[3])
forthReaction.dataToPlot()
forthReaction.widthPlot()

timer.startTimer()
fifthReaction = REACTION(I[1], T[1], InputKx, folders[4])
fifthReaction.dataToPlot()
fifthReaction.widthPlot()

timer.startTimer()
sixthReaction = REACTION(I[1], T[2], InputKx, folders[5])
sixthReaction.dataToPlot()
sixthReaction.widthPlot()
