import gspread
from oauth2client.service_account import ServiceAccountCredentials
import dateutil.parser
import datetime
#import matplotlib.pyplot as plt
import numpy
import os
import difflib

#Globals============================================

debug = False

color_map = ['b', 'g', 'r', 'c', 'm', 'y',  'k', 'w']
markers =['o',#	circle marker
'v',#	triangle_down marker
'^'	,#triangle_up marker
'<'	,#triangle_left marker
'>'	,#triangle_right marker
'1'	,#tri_down marker
'2'	,#tri_up marker
'3'	,#tri_left marker
'4'	,#tri_right marker
's'	,#square marker
'p'	,#pentagon marker
'*'	,#star marker
'h'	,#hexagon1 marker
'H'	,#hexagon2 marker
'+'	,#plus marker
'x'	,#x marker
'D'	,#diamond marker
'd'	,#thin_diamond marker
'|'	,#vline marker
'_'	,#hline marker
]


desiredTrainees = None
desiredEarliestDate = None
desiredLatestDate = None
workSheets = None
sheet = None
minDate = datetime.datetime.now()
maxDate = datetime.datetime(2015, 1, 1, 0, 0, 0)

Trainees = []
daysOfWeekLine = {'m': 0, 'tu': 0, 'w': 0, 'th': 0, 'f': 0} 
daysOfWeekComp = {'m': 0, 'tu': 0, 'w': 0, 'th': 0, 'f': 0}
days = ['m', 'tu', 'w', 'th', 'f']  

#Classes=======================================================================
class TraineeRecord:
    def __init__(self, date, section, numberOfLineItems, compentancy, compentancyPercentage, completedComp = 'n'):
        #removing all special characters
        #day = ''.join(c for c in day if c.isalnum())
        numberOfLineItems = ''.join(c for c in numberOfLineItems if c.isalnum())
        compentancy = ''.join(c for c in compentancy if c.isalnum())
        compentancyPercentage = ''.join(c for c in compentancyPercentage if c.isalnum())
        section = ''.join(c for c in section if c.isalnum())

        if date == '' or date == None:
            date = '1/1/2010'

        try:
            self.date = dateutil.parser.parse(date)
        except:
            self.date = datetime.datetime.now()
            
        self.day = days[self.date.weekday() % 5]
        self.valid = True
        self.section = section
        self.completedComp = completedComp

        try:
            self.numberOfLineItems = int(numberOfLineItems)
        except:
            self.numberOfLineItems = 0


        self.compentancy = compentancy

        try:
            self.compentancyPercentage = int(compentancyPercentage)
        except:
            self.compentancyPercentage = 0

    def __str__(self):
        return "Date: %s\tDay of Week: %s\tSection: %s\t# Line Items: %d\tCurrent Comp: %s\tEstimated Percentage Completed: %d\tCurrent Comp Completed:%s".format(
              self.date.strftime('%m/%d/%Y'), self.day, self.section, self.numberOfLineItems, self.compentancy, self.compentancyPercentage, self.completedComp)

class Trainee:
    def __init__(self, worksheet):
        self.name = worksheet._title
        self.lineVel = 0
        self.compVel = 0
        self.lineIntercpt = 0
        self.compIntercept = 0
        self.minDate = None
        self.maxDate = None
        self.numRecords = 0;
        self.numberOfLineItemsCompleted = 0
        self.numberOfCompsCompleted = 0
        self.averageLineVel = 0
        self.averageCompVel = 0

        dates = worksheet.col_values(1)[1:-1]
        #days = worksheet.col_values(2)[1:-1]
        sections = worksheet.col_values(3)[1:-1]
        numberOfLineItems = worksheet.col_values(4)[1:-1]
        compentancies = worksheet.col_values(5)[1:-1]
        compentancyPercentage = worksheet.col_values(6)[1:-1]
        completedComp = worksheet.col_values(7)[1:-1]

        index = 0
        self.records = []
        for date in dates:

            if date != None and date != '' and  date != ' ':
                currRecord = TraineeRecord(date, sections[index], numberOfLineItems[index],
                                           compentancies[index], compentancyPercentage[index],
                                           'y' if completedComp[index].lower() == 'y' else 'n')

                self.numberOfLineItemsCompleted = self.numberOfLineItemsCompleted + currRecord.numberOfLineItems
                if currRecord.completedComp.lower() == 'y':
                    self.numberOfCompsCompleted = self.numberOfCompsCompleted+1

                if self.minDate == None:
                        self.minDate = currRecord.date
                else:
                    self.minDate = min(self.minDate, currRecord.date)
                if self.maxDate == None:
                    self.maxDate = currRecord.date
                else:
                    self.maxDate = max(self.maxDate, currRecord.date)

                self.records.append(currRecord)
                self.numRecords = self.numRecords +1

            index = index+1

    def setLineVel(self, lineVel):
        self.lineVel = lineVel

    def setCompVel(self, compVel):
        self.compVel = compVel

    def computeLineItemVel(self):

        if self.numRecords<2:
            print("%s lacks sufficient records to do regression."%self.name)
            return [],[],[]

        dates = [record.date.date() for record in self.records if record.valid]
        lineItems = [record.numberOfLineItems for record in self.records if record.valid]

        #The instruction below is needed in the case that there are two or more records with the same date.
        dateLineItems= Trainee.__ConsolidateIntoUniqueDict(dates,lineItems)
        dates = list(dateLineItems.keys())
        lineItems = list(dateLineItems.values())

        xvals = numpy.arange(len(dates))  # this should account for weekends and leave
        coeff = numpy.polyfit(xvals, numpy.array(sumList(lineItems)), 1)
        self.averageLineVel = sum(lineItems)/len(dates)
        #self.lineVel = coeff[0] if coeff[0] > 0.001 else 0
        self.lineVel = coeff[0]
        self.lineIntercpt = coeff[1]
        return dates, lineItems, xvals

    def computeCompVel(self):
        if self.numRecords<2:
            print("%s lacks sufficient records to do regression.",self.name)
            return [],[],[]

        dates = [record.date.date() for record in self.records if record.valid]
        compPercentages = [record.compentancyPercentage for record in self.records if record.valid]

        # The instruction below is needed in the case that there are two or more records with the same date.
        dateLineItems = Trainee.__ConsolidateIntoUniqueDict(dates, compPercentages)
        dates = list(dateLineItems.keys())
        compPercentages = list(dateLineItems.values())

        xvals = numpy.arange(len(dates))  # this should account for weekends and leave
        coeff = numpy.polyfit(xvals, numpy.array(sumList(compPercentages)), 1)
        self.averageCompVel = sum(compPercentages)/ len(dates)
        #self.compVel = coeff[0] if coeff[0] > 0.001 else 0
        self.compVel = coeff[0]
        self.compIntercept = coeff[1]
        return dates, compPercentages, xvals

    def FilterTrainee(self, lowerDate, upperDate):

        if upperDate >= self.maxDate and lowerDate <= self.minDate:
            return

        if lowerDate > self.minDate:
            self.minDate = lowerDate
        if upperDate < self.maxDate:
            self.maxDate = upperDate

        # filteredRecords = []

        for record in self.records:
            if record.date <= upperDate and record.date >= lowerDate:
                record.valid = True
            else:
                self.numRecords = self.numRecords - 1
                record.valid = False
                self.numberOfLineItemsCompleted = self.numberOfLineItemsCompleted - record.numberOfLineItems
                if record.completedComp.lower() == 'y':
                    self.numberOfCompsCompleted = self.numberOfCompsCompleted - 1

    def ResetRecords(self):
        for record in self.records:
            record.valid = True
            if record.date < self.minDate:
                self.minDate = record.date
            if record.date > self.maxDate:
                self.maxDate = record.date

    @staticmethod
    def __ConsolidateIntoUniqueDict(keys,values):
        retDict = {}
        index = 0
        for key in keys:
            if key not in retDict:
                retDict[key] = values[index]
            else:
                retDict[key] = values[index]+retDict[key]
            index = index + 1

        return retDict


class TraineeCounts:
    validCCompNames = ['C 1', 'C 2', 'C 3', 'C 4', 'C Comp 1', 'C Comp 2', 'C Comp 3', 'C Comp 4']
    validPyCompNames = ['Py 1', 'Py 2 ', 'P 1', 'P 2 ', 'pyhton 1', 'pyhton 2', 'Py Comp 1', 'Py Comp 2 ', 'P Comp 1', 'P Comp 2 ', 'pyhton Comp1', 'pyhton Comp 2']
    validAsmCommpNames = ['Asm 1', 'Asm 2', 'Asm 3', 'A 1', 'A 2', 'A 3', 'assembly 1', 'assembly 2', 'assembly 3', 'Asm Comp 1', 'Asm Comp 2', 'Asm Comp 3', 'A Comp 1', 'A Comp 2', 'A Comp 3', 'assembly Comp 1', 'assembly Comp 2', 'assembly Comp 3']
    validCapNames = ['Cap', 'capstone']

    cComps = 'C Comps'
    pyComps = 'Python \nComps'
    asmComp = 'Assembly \nComps'
    capProj = 'Capstone \nProject'

    def __init__(self, Trainee, HistoricSheet=None):
        self.name = Trainee.name

        self.traineeComps = {TraineeCounts.cComps: 0,
                             TraineeCounts.pyComps: 0,
                             TraineeCounts.asmComp: 0,
                             TraineeCounts.capProj: 0}

        self.historicalComps = {TraineeCounts.cComps: 0,
                                TraineeCounts.pyComps: 0,
                                TraineeCounts.asmComp: 0,
                                TraineeCounts.capProj: 0}

        self.traineeSections = {'100': 0,
                                '101': 0,
                                '200': 0,
                                '201': 0,
                                '202': 0,
                                # '203':0,
                                '204': 0,
                                'Debug': 0}  # Same column as 203

        self.historicSections = {'100': 0,
                                 '101': 0,
                                 '200': 0,
                                 '201': 0,
                                 '202': 0,
                                 # '203':0,
                                 '204': 0,
                                 'Debug': 0}  # Same column as 203

        self.compsMax = {TraineeCounts.cComps: [0,[-1,-1]],
                    TraineeCounts.pyComps: [0,[-1,-1]],
                    TraineeCounts.asmComp: [0,[-1,-1]],
                    TraineeCounts.capProj: [0,[-1,-1]]}

        self.sectionsMax = {'100': [0,[-1,-1]], #(value and position)
                       '101': [0,[-1,-1]],
                       '200': [0,[-1,-1]],
                       '201': [0,[-1,-1]],
                       '202': [0,[-1,-1]],
                       # '203':[0,[-1,-1]],
                       '204': [0,[-1,-1]],
                       'Debug': [0,[-1,-1]]}  # Same Row as 203



        self.sections = list(self.traineeSections.keys())
        self.comps = list(self.traineeComps.keys())

        self.historicCells = None
        self.targetCells = None

        for record in Trainee.records:


            try:
                section = difflib.get_close_matches(record.section, self.sections)[0]
                self.traineeSections[section] = self.traineeSections[section] + record.numberOfLineItems
            except:
                if record.section != '' and record.section != None and record.section != ' ' \
                        and TraineeCounts.CleanAndLowerStr(record.section) != 'na':
                    print("Could not find key: %s in traineeSections."%record.section)

            if record.completedComp.lower() == 'y':
                currComp = self.__CurrComp(record)
                if currComp != 'err':
                    try:
                        self.traineeComps[currComp] = self.traineeComps[currComp] + 1
                    except:
                        print("Could not find key: %s in traineeComps." % currComp)


        if HistoricSheet != None:
            HistoricSheet = HistoricSheet.worksheets()[0]
            self.historicCells = HistoricSheet.get_all_values()
            self.UpdateHistoricCounts(HistoricSheet, self.historicSections)
            self.UpdateHistoricCounts(HistoricSheet, self.historicalComps)

    def __FindNameCell(self, sheet):

        try:
            return sheet.find(self.name)
        except:
            print("%s not present in Historical Sheet." % self.name)

        return None

    def UpdateHistoricCounts(self, sheet, dictionary):
        nameCellHistorical = TraineeCounts.findPositionOfCell(self.name, self.historicCells)

        if nameCellHistorical[0] == -1:
            print("Could not find %s in historical sheet." % self.name)
            return

        tempCellKey = None
        tempCellVal = None

        for key in dictionary.keys():

            tempCellKey = TraineeCounts.findPositionOfCell(key, self.historicCells)

            if tempCellKey[0] == -1:
                print("%s not found in historic sheet." % key)
                continue

            tempCellVal = sheet.cell(nameCellHistorical[0], tempCellKey[1])
            value = ''.join(c for c in tempCellVal.value if c.isalnum())
            # print("key: %s\tvalue: %s"%(key,tempCellVal.value))
            if value == '' or value == None:
                value = '0'

            dictionary[key] = int(value)

    def UpdateAllHistoricCounts(self, sheet):
        self.UpdateHistoricCounts(sheet, self.historicSections)
        self.UpdateHistoricCounts(sheet, self.historicalComps)

    def UpdateJQRTracker(self, TargetSheet):


        TargetSheet = TargetSheet.worksheets()[0]

        self.targetCells = TargetSheet.get_all_values()

        nameCellTarget = TraineeCounts.findPositionOfCell(self.name, self.targetCells)

        if nameCellTarget[0] == -1:
            print("Could not find %s in target sheet." % self.name)
            return

        self.__FindMaxVals(nameCellTarget)

        update = 'y'

        for key in self.traineeSections.keys():

            tempCellKey = self.sectionsMax[key][1]
            if tempCellKey[0] == -1:
                print("%s not found in target sheet." % key)
                continue

            inputVal = self.traineeSections[key] + self.historicSections[key]

            if self.sectionsMax[key][0] < inputVal:
                update = input("%s has a new value (%d) greater than max for section %s (%d). Contninue (y/n)?"% (self.name, inputVal, key, self.sectionsMax[key][0]))

            if update.lower() == 'y':
                TargetSheet.update_cell(nameCellTarget[0], tempCellKey[1], inputVal)

        for key in self.traineeComps.keys():

            tempCellKey = self.compsMax[key][1]


            if tempCellKey[0] == -1:
                print("%s not found in target sheet." % key)
                continue

            inputVal = self.traineeComps[key] + self.historicalComps[key]

            if self.compsMax[key][0] < inputVal:
                update = input("%s has a new value (%d) greater than max for %s (%d). Contninue (y/n)?"%( self.name, inputVal,key, self.compsMax[key][0]))

            if update.lower() == 'y':
                TargetSheet.update_cell(nameCellTarget[0], tempCellKey[1], inputVal)


    @staticmethod
    def findPositionOfCell(value, cells):
        row = 0
        col = 0
        for l in cells:
            col = 0
            for val in l:
                if TraineeCounts.CleanAndLowerStr(value) == TraineeCounts.CleanAndLowerStr(val):
                    return (row + 1, col + 1)
                col = col + 1
            row = row + 1

        return (-1, -1)

    def __CurrComp(self, record):

        comp = TraineeCounts.CleanAndLowerStr(record.compentancy)


        if comp in TraineeCounts.CleanAndLowerStr(TraineeCounts.validCCompNames):
            return TraineeCounts.cComps
        if comp in TraineeCounts.CleanAndLowerStr(TraineeCounts.validPyCompNames):
            return TraineeCounts.pyComps
        if comp in TraineeCounts.CleanAndLowerStr(TraineeCounts.validAsmCommpNames):
            return TraineeCounts.asmComp
        if comp in TraineeCounts.CleanAndLowerStr(TraineeCounts.validCapNames):
            return TraineeCounts.capProj

        try:
            bestGuess = difflib.get_close_matches(comp, self.comps)[0]
        except:
            print('%s is an invalid comp name.' % record.compentancy)
            return 'err'

        print('%s is an invalid comp name. Best guess is: %s' % record.compentancy, bestGuess)
        return bestGuess


    def __FindMaxVals(self, nameCellTarget):


        for key in self.sectionsMax.keys():
            # The following finds the section cell right above the corresposding name row
            if key == 'Debug':
                tempCell = TraineeCounts.findPositionOfCell('203', reversed(self.targetCells[0:nameCellTarget[0]]))
                tempCell2 = TraineeCounts.findPositionOfCell('Debug', reversed(self.targetCells[0:nameCellTarget[0]]))

                tempCell = tempCell if tempCell[0]<tempCell2[0] else tempCell2

                if tempCell[0] == -1:
                    tempCell = tempCell2
            else:
                tempCell = TraineeCounts.findPositionOfCell(key, reversed(self.targetCells[0:nameCellTarget[0]]))

            if tempCell[0] == -1:
                print("Could not find key %s."%key)
                continue
            row = nameCellTarget[0] - tempCell[0] - 1
            value = self.targetCells[row][tempCell[1] - 1]
            try:
                self.sectionsMax[key][0] = int(value)
                self.sectionsMax[key][1] = [row, tempCell[1]]
            except:
                print("Error finding max value for %s." % key)


        for key in self.compsMax.keys():
            # The following finds the section cell right above the corresposding name row
            tempCell = TraineeCounts.findPositionOfCell(key, reversed(self.targetCells[0:nameCellTarget[0]]))
            if tempCell[0] == -1:
                print("Could not find key %s."%key)
                continue
            row = nameCellTarget[0] - tempCell[0] + 1
            value = self.targetCells[row][tempCell[1] - 1]
            try:
                self.compsMax[key][0] = int(value)
                self.compsMax[key][1] = [row, tempCell[1]]
            except:
                print("Error finding max value for %s." % key)


    def __GetMaxVal(self, nameCellTarget, dictionary):

        for key in dictionary.keys():
            #The following finds the section cell right above the corresposding name row
            tempCell = TraineeCounts.findPositionOfCell(key, reversed(self.targetCells[0:nameCellTarget[0]]))
            row = nameCellTarget[0] - tempCell[0] -1
            value = self.targetCells[row][tempCell[1]-1]
            dictionary[key] = int(value)


    @staticmethod
    def CleanAndLowerStr(string):
        return ''.join(c.lower() for c in ''.join(string) if c.isalnum() or c == '.')

#Functions=============================================================
'''
@fn RetrieveSpreadSheet(jsonFile, spreadSheetName)
@brief Function to open a spreadsheet from the google server
@param jsonFile name of the .jsonFile key to access the google drive file
@param spreadSheetName name of the spreadsheet to be opened
@return returns google spreadsheet object from the gspread API
'''
def RetrieveSpreadSheet(jsonFile = 'JQR Self Progress-7a72c0d519ad.json', spreadSheetName = "JQR Self Progress"):
    # use creds to create a client to interact with the Google Drive API
    #global sheet
    if not os.path.exists(jsonFile):
        print("%s could not be located." % jsonFile)
        exit(-1)

    scope = ['https://spreadsheets.google.com/feeds']
    try:
        # Using Oath2client API to authenticate to google drive
        creds = ServiceAccountCredentials.from_json_keyfile_name(jsonFile, scope)
        client = gspread.authorize(creds)
        sheet = client.open(spreadSheetName)
    except:
        print("Could not retrive Spread Sheet %s"%spreadSheetName)
        exit(-1)

    return sheet




def sumList(myList):
    # currSum = 0
    first = True
    retList = []
    for i in myList:
        if first:
            currSum = i
            first = False
        else:
            currSum = currSum + i

        retList.append(currSum)

    return retList


def PlotLineVelocity(figure, trainee, color, marker):
    plt.figure(figure.number)

    dates, lineItems, xvalsP = trainee.computeLineItemVel()

    xvals = [x + (dates[0] - minDate.date()).days for x in range(len(dates))]
    yvals = sumList(lineItems)

    plt.plot(xvals, yvals, color + marker, label=trainee.name + "; VEL:" + "{0: .3f}".format(trainee.lineVel))
    plt.plot(xvals, (trainee.lineVel * xvalsP) + trainee.lineIntercpt, color)

    plt.legend(loc=0, prop={'size': 20})

    return xvals, dates

    #return slopeintercept


def PlotCompVelocity(figure, trainee, color, marker):
    plt.figure(figure.number)

    dates, compPercentages, xvalsP = trainee.computeCompVel()


    xvals = [x + (dates[0]-minDate.date()).days for x in range(len(dates))]
    yvals = sumList(compPercentages)

    plt.plot(xvals, yvals, color + marker, label=trainee.name +
                        "; Latest Comp:" + GetLatestComp(trainee)+"; VEL:" + "{0: .3f}".format(trainee.compVel))
    plt.plot(xvals, (trainee.compVel * xvalsP) + trainee.compIntercept, color)

    plt.legend(loc=0, prop={'size': 20})

    return xvals, dates


def GetLatestComp(trainee):

    for index in range(len(trainee.records)-1, -1, -1):
        if trainee.records[index].valid:
            currComp = trainee.records[index].compentancy
            if currComp != '' or currComp != None or currComp != ' ':
                return currComp

    return None


def CreateTableOfVelocities(file=None):
    global Trainees

    formatStr = '{:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}\n'

    table = []

    table.append(
        formatStr.format('NAME', 'LINE ITEM', 'LINE ITEM', 'COMP PROGRESS', 'COMP PROGRESS', 'LINE ITEMS', 'COMPS'))
    table.append(
        formatStr.format(' ', 'VELOCITY ML', 'VELOCITY AVG', '% VELOCITYM ML', '% VELOCITY AVG', 'COMPLETED', 'COMPLETED'))

    lineVelocities = []
    compVelocities = []
    lineVelocitiesAVG = []
    compVelocitiesAVG = []

    for trainee in Trainees:
        if trainee.numRecords > 0:
            lineVelocities.append(trainee.lineVel)
            compVelocities.append(trainee.compVel)
            lineVelocitiesAVG.append(trainee.averageLineVel)
            compVelocitiesAVG.append(trainee.averageCompVel)

            floatFormat =  "{0: .3f}"

            lineVelStr = floatFormat.format(trainee.lineVel)
            compVelStr = floatFormat.format(trainee.compVel)
            linVelAvg = floatFormat.format(trainee.averageLineVel)
            compVelAvg = floatFormat.format(trainee.averageCompVel)
            table.append(
                formatStr.format(trainee.name, lineVelStr, linVelAvg,
                                 compVelStr, compVelAvg,
                                 trainee.numberOfLineItemsCompleted,
                                 trainee.numberOfCompsCompleted))

    statString = minDate.strftime("%Y-%m-%d") +\
              ' to ' + maxDate.strftime("%Y-%m-%d") +'\n'+\
              '\nLine Velocity ML:\tMean = {:.3f}\tSTD = {:.3f}\nComp Velocity ML:\tMean = {:.3f}\tSTD = {:.3f}\nLine Velocity AVG:\tMean = {:.3f}\tSTD = {:.3f}\nComp Velocity AVG:\tMean = {:.3f}\tSTD = {:.3f}\n\n '.format(
        numpy.mean(numpy.array(lineVelocities)),
        numpy.std(numpy.array(lineVelocities)),
        numpy.mean(numpy.array(compVelocities)),
        numpy.std(numpy.array(compVelocities)),
        numpy.mean(numpy.array(lineVelocitiesAVG)),
        numpy.std(numpy.array(lineVelocitiesAVG)),
        numpy.mean(numpy.array(compVelocitiesAVG)),
        numpy.std(numpy.array(compVelocitiesAVG)))

    table = [statString] + table + ["\n*ML = Machine Learning Algorihtm Learned Over Desired Time Span\n*AVG = Average Value Over Desired Time Span"]

    table = ''.join(table)

    print(table)

    if (file != None):
        f = open(file, "w+")
        f.write(table)
        f.close()

def CreateDayOfWeekDistributions():
    global days
    global minDate
    global maxDate
    global Trainees
    global daysOfWeekLine
    global daysOfWeekComp

    UpdateDayCount(daysOfWeekLine, daysOfWeekComp)

    plt.figure("daysOfWeekLine", figsize=(10, 7))
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Number Of Line Items', fontsize=16)
    plt.title('N51 Number of Line Items Completed '+ minDate.strftime("%Y-%m-%d")+
              ' to '+maxDate.strftime("%Y-%m-%d"), fontsize=16)
    plt.bar(range(5), [daysOfWeekLine[day] for day in days], tick_label = days)

    plt.figure("daysOfWeekComp", figsize=(10, 7))
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Total Comp Progress', fontsize=16)
    plt.title('N51 Total Comp Progress Completed ' + minDate.strftime("%Y-%m-%d") +
              ' to ' + maxDate.strftime("%Y-%m-%d"), fontsize=16)
    plt.bar(range(5), [daysOfWeekComp[day] for day in days], tick_label=days)

def CreateCompVelocityDistribution(compProgressVelocities):
    print()

def CreateLineItemVelocityDistribution(daysOfWeek):
    print()

def UpdateDayCount(daysOfWeekLine, daysOfWeekComp):
    global days
    global Trainees

    for currTrainee in Trainees:
        for record in currTrainee.records:
            if record.valid:
                currDay = record.day.lower()
                if currDay in days:
                    daysOfWeekComp[currDay] = daysOfWeekComp[currDay] + record.compentancyPercentage
                    daysOfWeekLine[currDay] = daysOfWeekLine[currDay] + record.numberOfLineItems
'''
@fn SetUpDates()
@brief Retrieved desired date range from the user
@details sets the global params desiredEarliestDate and desiredLatestDate with the user specified dates.
'''
def SetUpDates():
    global desiredEarliestDate
    global desiredLatestDate

    # The earliest date that someone can filter records on
    earliestDate = datetime.datetime.min
    # The latest date that someone can filter records on (ie today)
    latestDate = datetime.datetime.now()

    # Get date range for data points
    badDate = True
    while(badDate):
        desiredEarliestDate = input(
            "Please provide the begin date in \"M/D/YYYY\" format (ie 5/11/1989) to filter\n\t\"all\"\tif you dont want to filter:")

        if desiredEarliestDate.lower() == "all":
            desiredEarliestDate = earliestDate
            desiredLatestDate = latestDate
            return
        else:
            try:
                desiredEarliestDate = dateutil.parser.parse(desiredEarliestDate)
                badDate = False
            except:
                print("Invalid early date entered %s" % desiredEarliestDate)

    badDate = True
    while (badDate):
        desiredLatestDate = input(
            "Please provide the latest date in \"M/D/YYYY\" format (ie 5/11/2018) to filter\n\t\"today\"\tif you dont want to filter:")

        if desiredLatestDate.lower() == "today":
            desiredLatestDate = latestDate
            return
        else:
            try:
                desiredLatestDate = dateutil.parser.parse(desiredLatestDate)
                badDate = False
            except:
                print("Invalid late date entered %s" % desiredEarliestDate)
'''
@fn GetListOfTraineeObjects
@brief Reads in the list of current trainees from the spreadsheet then retrieves data for the user desired trainee
@param sheet the sheet object retrieved from the google drive
'''
def GetListOfTraineeObjects(sheet):

    global Trainees
    global desiredTrainees
    global workSheets

    workSheets = sheet.worksheets()

    # Get Subset of trainees one would like analytics on
    allTrainees = []
    for currWorkSheet in workSheets:
        allTrainees.append(currWorkSheet._title)  # The title of each worksheet is a trainees name

    if 'Example' in allTrainees:
        allTrainees.remove('Example')
    if 'BLANK' in allTrainees:
        allTrainees.remove('BLANK')

    desiredTrainees = input(
        "Please indicate in semicolon seperated format who the desired trainees are.\n\n\tie \"Gamboa,Allan\";\"Basior,Greg\";...\n\n" +
        "\tIf all trainees are desired say:\n\n\t\"all\"\n\n" +
        "\tAvailable Trainees are:\n\n\t" + (','.join(allTrainees)).replace(',', '\n\t') + "\n")

    if desiredTrainees.lower() == 'all':
        desiredTrainees = ';'.join(allTrainees)

    desiredTrainees = desiredTrainees.lower()
    desiredTrainees = desiredTrainees.split(';')

    for currWorkSheet in workSheets:
        # todo check that trainee is in list
        if currWorkSheet._title.lower() in desiredTrainees:
            print("Creating Object For %s" % currWorkSheet._title.lower())
            currTrainee = Trainee(currWorkSheet)
            Trainees.append(currTrainee)
    if len(Trainees) == 0:
        print("No trainees selected or could be found from those desired.")
        exit(-1)


def FilterTraineesDateRanges():
    global Trainees
    global minDate
    global maxDate
    global desiredEarliestDate
    global desiredLatestDate

    for currTrainee in Trainees:
        currTrainee.FilterTrainee(desiredEarliestDate, desiredLatestDate)
        if currTrainee.minDate < minDate:
            minDate = currTrainee.minDate
        if currTrainee.maxDate > maxDate:
            maxDate = currTrainee.maxDate



        #filteredRecords.append(record)

    #trainee.records = filteredRecords


def MakePlots():
    # ============================================================================================
    # Below we are both filtering the data according to desired date range and also creating plots
    # ============================================================================================


    global Trainees
    global daysOfWeekComp
    global daysOfWeekLine
    global minDate
    global maxDate

    #lineItemVelocities = []
    #compProgressVelocities = []
    #actualTrainees = []



    lineItemFigure = plt.figure("lineItemFigure", figsize=(18, 16))

    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Number Of Line Items', fontsize=16)
    plt.title('Line Item Velocity ' + minDate.strftime("%Y-%m-%d") +
              ' to ' + maxDate.strftime("%Y-%m-%d"), fontsize=20)

    CompVelFigure = plt.figure("CompVelFigure", figsize=(18, 16))

    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Competency Progress Percentage', fontsize=16)
    plt.title('Competency Progress Velocity ' + minDate.strftime("%Y-%m-%d") +
              ' to ' + maxDate.strftime("%Y-%m-%d"), fontsize=20)

    count = 0



    for currTrainee in Trainees:

        if currTrainee.numRecords > 1:

            PlotLineVelocity(lineItemFigure, currTrainee, color_map[count % len(color_map)],
                                                markers[count // len(color_map)])
            xvals, dates = PlotCompVelocity(CompVelFigure, currTrainee, color_map[count % len(color_map)],
                                            markers[count // len(color_map)])

            if count == 0:
                minx = xvals[0]
                maxx = xvals[-1]
                mindate = dates[0]
                maxdate = dates[-1]
            else:
                if xvals[0] < minx:
                    minx = xvals[0]
                if dates[0] < mindate:
                    mindate = dates[0]
                if xvals[-1] > maxx:
                    maxx = xvals[-1]
                if dates[-1] > maxdate:
                    maxdate = dates[-1]

            count = count + 1
            # lineItemVelocities.append(lineItemVelocity)
            # compProgressVelocities.append(compVelocity)
            # actualTrainees.append(currTrainee.name)
    plt.figure(lineItemFigure.number)
    plt.xticks(range(minx,maxx), [mindate+datetime.timedelta(days = count) for count in range(maxx-minx)])
    plt.locator_params(axis='x', nbins=10)

    plt.figure(CompVelFigure.number)
    plt.xticks(range(minx, maxx), [mindate + datetime.timedelta(days=count) for count in range(maxx - minx)])
    plt.locator_params(axis='x', nbins=10)

def UpdateJQRTracker(selfTrackerSheetName='JQR Self Progress',
                     historicalTrackerSheetName='Historical Training Tracker',
                     targetSheetName='Training Tracker',
                     traineesWanted = "all"):
    selfTrackersheet = RetrieveSpreadSheet(spreadSheetName=selfTrackerSheetName)
    workSheets = selfTrackersheet.worksheets()

    # All trainees currently loaded
    if TraineeCounts.CleanAndLowerStr(traineesWanted) == "all":
        allLoadedTrainees = [trainee.name.lower() for trainee in Trainees]
    else:
        allLoadedTrainees = []
        for currWorkSheet in workSheets:
            currName  = currWorkSheet._title
            if TraineeCounts.CleanAndLowerStr(currName) not in TraineeCounts.CleanAndLowerStr(traineesWanted):
                allLoadedTrainees.append(currName)  # The title of each worksheet is a trainees name

    allLoadedTrainees = TraineeCounts.CleanAndLowerStr(''.join(allLoadedTrainees))

    for currWorkSheet in workSheets:
        # todo check that trainee is in list
        currName = currWorkSheet._title.lower()
        if currName not in allLoadedTrainees and currName not in ['blank', 'example']:
            print("Creating Trainee Object for %s" % currWorkSheet._title.lower())
            currTrainee = Trainee(currWorkSheet)
            Trainees.append(currTrainee)

    historicalTrackerSheet = RetrieveSpreadSheet(spreadSheetName=historicalTrackerSheetName)

    targetSheet = RetrieveSpreadSheet(spreadSheetName=targetSheetName)

    for trainee in Trainees:

        global debug


        print("Creating count object for  %s."%trainee.name)

        tempCounts = TraineeCounts(trainee, historicalTrackerSheet)

        if debug:
            print("Self Tracker Sections/Comps:")
            print(tempCounts.traineeSections)
            print(tempCounts.traineeComps)

            print("Historical Sections/Comps:")
            print(tempCounts.historicSections)
            print(tempCounts.historicalComps)

            a = input("Press <ENTER> to continue with update.")

        print("Updating %s.\n" % trainee.name)
        tempCounts.UpdateJQRTracker(targetSheet)

def main():


    # Getting the spread sheet from google drive
    global sheet
    sheet = RetrieveSpreadSheet()
    SetUpDates()
    GetListOfTraineeObjects(sheet)
    FilterTraineesDateRanges()
    MakePlots()
    CreateTableOfVelocities()
    CreateDayOfWeekDistributions()

    updateTracker = input("Would you like to update the JQR tracker (y/n):")

    if updateTracker.lower() == 'y':
        UpdateJQRTracker(selfTrackerSheetName='JQR Self Progress',
                     historicalTrackerSheetName='Historical Training Tracker',
                     targetSheetName='Training Tracker.xlsx')




if __name__ == "__main__":
    main()



