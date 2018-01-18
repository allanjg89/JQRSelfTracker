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
        self.lineVel = coeff[0]
        self.lineIntercpt = coeff[1]
        return dates, lineItems, xvals

    def computeCompVel(self):
        dates = [record.date.date() for record in self.records if record.valid]
        compPercentages = [record.compentancyPercentage for record in self.records if record.valid]
        xvals = numpy.arange(len(dates))  # this should account for weekends and leave
        coeff = numpy.polyfit(xvals, numpy.array(sumList(compPercentages)), 1)
        self.compVel = coeff[0]
        self.compIntercept = coeff[1]
        return dates, compPercentages, xvals



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

    formatStr = '{:^20} {:^20} {:^20} {:^20} {:^20}\n'

    table = []

    table.append(formatStr.format('NAME', 'LINE ITEM VELOCITY', 'COMP PROGRESS VELOCITY', 'LINE ITEMS COMPLETED', 'COMPS COMPLETED'))

    lineVelocities = []
    compVelocities = []

    for trainee in Trainees:
        lineVelocities.append(trainee.lineVel)
        compVelocities.append(trainee.compVel)

        lineVelStr = str(trainee.lineVel)
        compVelStr = str(trainee.compVel)
        table.append(formatStr.format(trainee.name, lineVelStr[0:min(6, len(lineVelStr))],
                                      compVelStr[0:min(6, len(compVelStr))], trainee.numberOfLineItemsCompleted,
                                      trainee.numberOfCompsCompleted))

    statString = '\nLine Velocity:\tMean = {:.3f}\tSTD = {:.3f}\nComp Velocity:\tMean = {:.3f}\tSTD = {:.3f}\n\n\n '.format(
        numpy.mean(numpy.array(lineVelocities)),
        numpy.std(numpy.array(lineVelocities)),
        numpy.mean(numpy.array(compVelocities)),
        numpy.std(numpy.array(compVelocities)))

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

    plt.xlabel('Arbritrary Days', fontsize=16)
    plt.ylabel('Number Of Line Items', fontsize=16)
    plt.title('Line Item Velocity ' + minDate.strftime("%Y-%m-%d") +
              ' to ' + maxDate.strftime("%Y-%m-%d"), fontsize=20)

    CompVelFigure = plt.figure("CompVelFigure", figsize=(18, 16))

    plt.xlabel('Arbritrary Days', fontsize=16)
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
    CreateLineItemVelocityDistribution()
    CreateCompVelocityDistribution()



if __name__ == "__main__":
    main()



