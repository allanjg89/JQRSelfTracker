import gspread
from oauth2client.service_account import ServiceAccountCredentials
import dateutil.parser
import datetime
import matplotlib.pyplot as plt
import numpy
import os

#Globals============================================

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
    def __init__(self, date, day, section, numberOfLineItems, compentancy, compentancyPercentage, completedComp = 'n'):
        #removing all special characters
        day = ''.join(c for c in day if c.isalnum())
        numberOfLineItems = ''.join(c for c in numberOfLineItems if c.isalnum())
        compentancy = ''.join(c for c in compentancy if c.isalnum())
        compentancyPercentage = ''.join(c for c in compentancyPercentage if c.isalnum())
        section = ''.join(c for c in section if c.isalnum())

        if date == '' or date == None:
            date = '1/1/2010'

        self.date = dateutil.parser.parse(date)
        self.day = day
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
        days = worksheet.col_values(2)[1:-1]
        sections = worksheet.col_values(3)[1:-1]
        numberOfLineItems = worksheet.col_values(4)[1:-1]
        compentancies = worksheet.col_values(5)[1:-1]
        compentancyPercentage = worksheet.col_values(6)[1:-1]
        completedComp = worksheet.col_values(7)[1:-1]

        index = 0
        self.records = []
        for date in dates:

            if date != None and date != '' and  date != ' ':
                currRecord = TraineeRecord(date, days[index], sections[index], numberOfLineItems[index],
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
        dates = [record.date.date() for record in self.records if record.valid]
        lineItems = [record.numberOfLineItems for record in self.records if record.valid]
        xvals = numpy.arange(len(dates))  # this should account for weekends and leave
        coeff = numpy.polyfit(xvals, numpy.array(sumList(lineItems)), 1)
        self.averageLineVel = sum(lineItems)/len(dates)
        self.lineVel = coeff[0] if coeff[0] > 0.001 else 0
        self.lineIntercpt = coeff[1]
        return dates, lineItems, xvals

    def computeCompVel(self):
        dates = [record.date.date() for record in self.records if record.valid]
        compPercentages = [record.compentancyPercentage for record in self.records if record.valid]
        xvals = numpy.arange(len(dates))  # this should account for weekends and leave
        coeff = numpy.polyfit(xvals, numpy.array(sumList(compPercentages)), 1)
        self.averageCompVel = sum(compPercentages)/ len(dates)
        self.compVel = coeff[0] if coeff[0] > 0.001 else 0
        self.compIntercept = coeff[1]
        return dates, compPercentages, xvals


class TraineeCounts:
    validCCompNames = ['C1', 'C2', 'C3', 'C4']
    ValidPyCompNames = ['Py1', 'Py2']
    ValidAsmCommpNames = ['Asm1', 'Asm2', 'Asm3']
    ValidCapNames = ['Cap']

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
                                'Debug': 0}  # Same Row as 203

        self.historicSections = {'100': 0,
                                 '101': 0,
                                 '200': 0,
                                 '201': 0,
                                 '202': 0,
                                 # '203':0,
                                 '204': 0,
                                 'Debug': 0}  # Same Row as 203

        self.historicCells = None
        self.targetCells = None

        for record in Trainee.records:
            if record.section != '' and record.section != None and record.section != ' ':

                try:
                    self.traineeSections[record.section] = self.traineeSections[record.section] + record.numberOfLineItems
                except:
                    print("Could not find key: %s in traineeSections.\n"%record.section)

                if record.completedComp == 'y':
                    currComp = self.__CurrComp(record)
                    if currComp != 'err':
                        try:
                            self.traineeComps[currComp] = self.traineeComps[currComp] + 1
                        except:
                            print("Could not find key: %s in traineeComps.\n" % currComp)

        if HistoricSheet != None:
            HistoricSheet = HistoricSheet.worksheets()[0]
            self.historicCells = HistoricSheet.get_all_values()
            self.UpdateHistoricCounts(HistoricSheet, self.historicSections)
            self.UpdateHistoricCounts(HistoricSheet, self.historicalComps)

    def __FindNameCell(self, sheet):

        try:
            return sheet.find(self.name)
        except:
            print("%s not present in Historical Sheet.\n" % self.name)

        return None

    def UpdateHistoricCounts(self, sheet, dictionary):
        nameCellHistorical = TraineeCounts.findPositionOfCell(self.name, self.historicCells)

        if nameCellHistorical[0] == -1:
            print("Could not find %s in historical sheet.\n" % self.name)
            return

        tempCellKey = None
        tempCellVal = None

        for key in dictionary.keys():

            tempCellKey = TraineeCounts.findPositionOfCell(key, self.historicCells)

            if tempCellKey[0] == -1:
                print("%s not found in historic sheet.\n" % key)
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
            print("Could not find %s in target sheet.\n" % self.name)
            return

        tempCellKey = None
        for key in self.traineeSections.keys():

            tempCellKey = TraineeCounts.findPositionOfCell(key, self.targetCells)
            if tempCellKey[0] == -1:
                print("%s not found in target sheet.\n" % key)
                continue

            TargetSheet.update_cell(nameCellTarget[0], tempCellKey[1],
                                    self.traineeSections[key] + self.historicSections[key])

        for key in self.traineeComps.keys():

            tempCellKey = TraineeCounts.findPositionOfCell(key, self.targetCells)
            if tempCellKey[0] == -1:
                print("%s not found in target sheet.\n" % key)
                continue

            TargetSheet.update_cell(nameCellTarget[0], tempCellKey[1],
                                    self.traineeComps[key] + self.historicalComps[key])

    @staticmethod
    def findPositionOfCell(value, cells):
        row = 0
        col = 0
        for l in cells:
            col = 0
            for val in l:
                if ''.join(c for c in value if c.isalnum()) == ''.join(c for c in val if c.isalnum()):
                    return (row + 1, col + 1)
                col = col + 1
            row = row + 1

        return (-1, -1)

    def __CurrComp(self, record):

        if record.compentancy in TraineeCounts.validCCompNames:
            return TraineeCounts.cComps
        if record.compentancy in TraineeCounts.validCCompNames:
            return TraineeCounts.pyComps
        if record.compentancy in TraineeCounts.validCCompNames:
            return TraineeCounts.asmComp
        if record.compentancy in TraineeCounts.validCCompNames:
            return TraineeCounts.capProj

        print('%s is an invalid comp name.\n' % record.compentancy)
        return 'err'

    #Functions=============================================================
def RetrieveSpreadSheet(jsonFile = 'JQR Self Progress-7a72c0d519ad.json', spreadSheetName = "JQR Self Progress"):
    # use creds to create a client to interact with the Google Drive API
    #global sheet
    if not os.path.exists(jsonFile):
        print("%s could not be located.\n" % jsonFile)
        exit(-1)

    scope = ['https://spreadsheets.google.com/feeds']
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(jsonFile, scope)
        client = gspread.authorize(creds)
        sheet = client.open(spreadSheetName)
    except:
        print("Could not retrive Spread Sheet %s\n"%spreadSheetName)
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
        currSum = currSum + i
        retList.append(currSum)

    return retList


def PlotLineVelocity(figure, trainee, color, marker):
    plt.figure(figure.number)

    #xvals = numpy.array([(date - dates[0]).days for date in dates])
    dates, lineItems, xvalsP = trainee.computeLineItemVel()

    #plt.xticks(xvals, dates)

    #plt.locator_params(axis='x', nbins=10)


    xvals = [x + (dates[0] - minDate.date()).days for x in range(len(dates))]

    plt.plot(xvals, sumList(lineItems), color + marker, label=trainee.name + "; VEL:" + str(trainee.lineVel)[0:6])
    plt.plot(xvals, (trainee.lineVel * xvalsP) + trainee.lineIntercpt, color)
    plt.xticks(xvals, dates)
    plt.locator_params(axis='x', nbins=10)
    plt.legend(loc=0, prop={'size': 20})

    #return slopeintercept


def PlotCompVelocity(figure, trainee, color, marker):
    plt.figure(figure.number)

    dates, compPercentages, xvalsP = trainee.computeCompVel()

    #plt.xticks(xvals, dates)

    #plt.locator_params(axis='x', nbins=10)

    xvals = [x + (dates[0]-minDate.date()).days for x in range(len(dates))]

    plt.plot(xvals, sumList(compPercentages), color + marker, label=trainee.name +
                        "; Latest Comp:" + GetLatestComp(trainee)+"; VEL:" + str(trainee.compVel)[0:6])
    plt.plot(xvals, (trainee.compVel * xvalsP) + trainee.compIntercept, color)
    plt.xticks(xvals, dates)
    plt.locator_params(axis='x', nbins=10)
    plt.legend(loc=0, prop={'size': 20})

    #return slope


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
        formatStr.format(' ', 'VELOCITY ML', 'VELOCITY AVG', 'VELOCITYM ML', 'VELOCITY AVG', 'COMPLETED', 'COMPLETED'))

    lineVelocities = []
    compVelocities = []
    lineVelocitiesAVG = []
    compVelocitiesAVG = []

    for trainee in Trainees:
        lineVelocities.append(trainee.lineVel)
        compVelocities.append(trainee.compVel)
        lineVelocitiesAVG.append(trainee.averageLineVel)
        compVelocitiesAVG.append(trainee.averageCompVel)

        lineVelStr = str(trainee.lineVel)
        compVelStr = str(trainee.compVel)
        linVelAvg = str(trainee.averageLineVel)
        compVelAvg = str(trainee.averageCompVel)
        table.append(
            formatStr.format(trainee.name, lineVelStr[0:min(6, len(lineVelStr))], linVelAvg[0:min(6, len(linVelAvg))],
                             compVelStr[0:min(6, len(compVelStr))], compVelAvg[0:min(6, len(compVelAvg))],
                             trainee.numberOfLineItemsCompleted,
                             trainee.numberOfCompsCompleted))

    statString = '\nLine Velocity ML:\tMean = {:.3f}\tSTD = {:.3f}\nComp Velocity ML:\tMean = {:.3f}\tSTD = {:.3f}\nLine Velocity AVG:\tMean = {:.3f}\tSTD = {:.3f}\nComp Velocity AVG:\tMean = {:.3f}\tSTD = {:.3f}\n\n '.format(
        numpy.mean(numpy.array(lineVelocities)),
        numpy.std(numpy.array(lineVelocities)),
        numpy.mean(numpy.array(compVelocities)),
        numpy.std(numpy.array(compVelocities)),
        numpy.mean(numpy.array(lineVelocitiesAVG)),
        numpy.std(numpy.array(lineVelocitiesAVG)),
        numpy.mean(numpy.array(compVelocitiesAVG)),
        numpy.std(numpy.array(compVelocitiesAVG)))

    table = [statString] + table

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

def SetUp(sheet):
    global desiredTrainees
    global desiredEarliestDate
    global desiredLatestDate
    #global sheet
    global workSheets


    # The earliest date that someone can filter records on
    earliestDate = datetime.datetime.min
    # The latest date that someone can filter records on (ie today)
    latestDate = datetime.datetime.now()

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
        "Please indicate in semicolon seperated format who the desired trainees are.\n\n\tie \"Gamboa,Allan\"; \"Basior,Greg\";...\n\n" +
        "\tIf all trainees are desired say:\n\n\t\"all\"\n\n" +
        "\tAvailable Trainees are:\n\n\t" + (','.join(allTrainees)).replace(',', '\n\t') + "\n")

    if desiredTrainees.lower() == 'all':
        desiredTrainees = ';'.join(allTrainees)

    desiredTrainees = desiredTrainees.lower()
    desiredTrainees = desiredTrainees.split(';')

    # Get date range for data points
    desiredEarliestDate = input(
        "Please provide the begin date in \"M/D/YYYY\" format (ie 5/11/1989) to filter\n\t\"all\"\tif you dont want to filter:")

    if desiredEarliestDate.lower() != "all":
        desiredLatestDate = input(
            "Please provide the latest date in \"M/D/YYYY\" format (ie 5/11/2018) to filter\n\t\"today\"\tif you dont want to filter:")
        desiredEarliestDate = datetime.datetime.strptime(desiredEarliestDate, '%m/%d/%Y')

        if desiredLatestDate.lower() == "today":
            desiredLatestDate = latestDate
        else:
            desiredLatestDate = datetime.datetime.strptime(desiredLatestDate, '%m/%d/%Y')
    else:
        desiredEarliestDate = earliestDate
        desiredLatestDate = latestDate



def GetListOfTraineeObjects():

    global Trainees
    global desiredTrainees

    for currWorkSheet in workSheets:
        # todo check that trainee is in list
        if currWorkSheet._title.lower() in desiredTrainees:
            print("Creating Object For %s\n" % currWorkSheet._title.lower())
            currTrainee = Trainee(currWorkSheet)
            Trainees.append(currTrainee)
    if len(Trainees) == 0:
        print("No trainees selected or could be found from those desired.\n")
        exit(-1)


def FilterTraineesDateRanges():
    global Trainees
    global minDate
    global maxDate
    global desiredEarliestDate
    global desiredLatestDate

    for currTrainee in Trainees:
        FilterTrainee(currTrainee, desiredEarliestDate, desiredLatestDate)
        if currTrainee.minDate < minDate:
            minDate = currTrainee.minDate
        if currTrainee.maxDate > maxDate:
            maxDate = currTrainee.maxDate

def FilterTrainee(trainee, lowerDate, upperDate):

    if upperDate >= trainee.maxDate and lowerDate <= trainee.minDate:
        return

    if lowerDate > trainee.minDate:
        trainee.minDate = lowerDate
    if upperDate < trainee.maxDate:
        trainee.maxDate = upperDate

    #filteredRecords = []

    for record in trainee.records:
        if record.date <= upperDate and record.date >= lowerDate:
            record.valid = True
        else:
            trainee.numRecords = trainee.numRecords - 1
            record.valid = False
            trainee.numberOfLineItemsCompleted = trainee.numberOfLineItemsCompleted - record.numberOfLineItems
            if record.completedComp.lower() == 'y':
                trainee.numberOfCompsCompleted = trainee.numberOfCompsCompleted - 1

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

        PlotLineVelocity(lineItemFigure, currTrainee, color_map[count % len(color_map)],
                                            markers[count // len(color_map)])
        PlotCompVelocity(CompVelFigure, currTrainee, color_map[count % len(color_map)],
                                        markers[count // len(color_map)])
        count = count + 1
        # lineItemVelocities.append(lineItemVelocity)
        # compProgressVelocities.append(compVelocity)
        # actualTrainees.append(currTrainee.name)


def UpdateJQRTracker(selfTrackerSheetName='JQR Self Progress',
                     historicalTrackerSheetName='Historical Training Tracker',
                     targetSheetName='Training Tracker'):
    selfTrackersheet = RetrieveSpreadSheet(spreadSheetName=selfTrackerSheetName)
    workSheets = selfTrackersheet.worksheets()

    # All trainees currently loaded
    allLoadedTrainees = [trainee.name.lower() for trainee in Trainees]

    for currWorkSheet in workSheets:
        # todo check that trainee is in list
        if currWorkSheet._title.lower() not in allLoadedTrainees and currWorkSheet._title.lower() not in ['blank',
                                                                                                          'example']:
            print("Creating Object For in %s\n" % currWorkSheet._title.lower())
            currTrainee = Trainee(currWorkSheet)
            Trainees.append(currTrainee)

    historicalTrackerSheet = RetrieveSpreadSheet(spreadSheetName=historicalTrackerSheetName)

    targetSheet = RetrieveSpreadSheet(spreadSheetName=targetSheetName)

    for trainee in Trainees:
        print("Creating count object for  %s."%trainee.name)

        tempCounts = TraineeCounts(trainee, historicalTrackerSheet)

        print("Self Tracker Sections/Comps:")
        print(tempCounts.traineeSections)
        print(tempCounts.traineeComps)

        print("Historical Sections/Comps:")
        print(tempCounts.historicSections)
        print(tempCounts.historicalComps)

        print("Updating %s.\n"%trainee.name)
        tempCounts.UpdateJQRTracker(targetSheet)

def main():
    # Getting the spread sheet from google drive
    global sheet
    sheet = RetrieveSpreadSheet(spreadSheetName="Testing")
    SetUp(sheet)
    GetListOfTraineeObjects()
    FilterTraineesDateRanges()
    MakePlots()
    CreateTableOfVelocities()
    CreateDayOfWeekDistributions()

    updateTracker = input("Would you like to update the JQR tracker (y/n):")

    if updateTracker.lower() == 'y':
        UpdateJQRTracker()



if __name__ == "__main__":
    main()



