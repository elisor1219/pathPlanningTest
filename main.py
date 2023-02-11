import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import csv
import sys
sys.setrecursionlimit(1500)



# PARAMETERS
degreeSearch_ = 130 # degrees
_degreeSearchRad_ = degreeSearch_ * np.pi / 180
searchRadius_ = 5
sideSearchX_ = 2.5
sideSearchY_ = 2.5

def main():



    # Import a map of cones or build one
    test = plt.axis([-15, 40, -70, 80])
    plt.grid()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    #conesPos = plt.ginput(-1,-1)
    #plt.plot([x[0] for x in conesPos], [x[1] for x in conesPos], 'bo')
    
    carPos = np.array([0.0, 0.0])
    carDirection = 0.0


    trueConesPos = np.zeros((0, 2))
    # Import a map of cones from a file
    fileName = np.zeros((4), dtype='object')
    fileName[0] = 'hairpins_increasing_difficulty.csv'
    fileName[1] = 'comp_2021.csv'
    fileName[2] = 'fseast_2022.csv'
    fileName[3] = 'vargarda.csv'

    with open(fileName[3], newline='') as csvfile:
        coneReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        skipFirstRowAndSecondRow = 2
        for row in coneReader:
            if skipFirstRowAndSecondRow > 0:
                if skipFirstRowAndSecondRow == 1:
                    carPos[0] = float(row[1])
                    carPos[1] = float(row[2])
                    carDirection = float(row[3])
                skipFirstRowAndSecondRow -= 1
                continue
            xPosTrue = float(row[1])
            yPosTrue = float(row[2])
            trueConesPos = np.append(trueConesPos, [[xPosTrue, yPosTrue]], axis=0)
            #print('xPosTrue = ', xPosTrue, ' yPosTrue = ', yPosTrue)
    #print('conesPos = ', conesPos)
    #print('trueConesPos = ', trueConesPos)

    conesPos = trueConesPos

    # Convert to numpy array
    conesPos = np.array(conesPos)

    print('conesPos = ', conesPos)

    # Plot the cones
    plt.plot([x[0] for x in conesPos], [x[1] for x in conesPos], 'bo')

    # Plot the car
    #carDirection is probably in radians
    carRectangle = drawCar(carPos, carDirection)
    sideConesFinderLeft = drawSideConesFinder(carPos, 1.2, side = 'left')
    sideConesFinderRight = drawSideConesFinder(carPos, 1.2, side = 'right')

    # Find the start cones
    startConeIndexLeft = findStartCone(conesPos, carPos, sideConesFinderLeft)
    print('startConeIndexLeft = ', startConeIndexLeft)
    startConeIndexRight = findStartCone(conesPos, carPos, sideConesFinderRight)
    print('startConeIndexRight = ', startConeIndexRight)

    # TODO: Parallelize left and right side
    # Find the left side line of the track
    print('- - - - - - - - - - - - - - - - - - LEFT - - - - - - - - - - - - - - - - - -')
    startConeIndex = startConeIndexLeft
    print('startConeIndex = ', startConeIndex)
    print('conesPos[startConeIndex,:] = ', conesPos[startConeIndex,:])
    sideLineIndexLeft = findSideLine(conesPos, conesPos[startConeIndex,:], previousCone = conesPos[startConeIndex,:])
    sideLineIndexLeft = np.delete(sideLineIndexLeft, -1) # type: ignore
    sideLineIndexLeft = np.append(startConeIndex, sideLineIndexLeft)
    #print('sideLineIndex = ', sideLineIndex)

    print('- - - - - - - - - - - - - - - - - - RIGHT - - - - - - - - - - - - - - - - - -')
    # Find the right side line of the track
    startConeIndex = startConeIndexRight
    print('startConeIndex = ', startConeIndex)
    print('conesPos[startConeIndex,:] = ', conesPos[startConeIndex,:])
    sideLineIndexRight = findSideLine(conesPos, conesPos[startConeIndex,:], previousCone = conesPos[startConeIndex,:])
    sideLineIndexRight = np.delete(sideLineIndexRight, -1) # type: ignore
    sideLineIndexRight = np.append(startConeIndex, sideLineIndexRight)
    #print('sideLineIndex = ', sideLineIndex)
        
    plt.show()

    plt.figure()
    plt.axis([-15, 40, -70, 80])
    plt.grid()
    plt.plot([x[0] for x in conesPos], [x[1] for x in conesPos], 'bo')

    # Plot the left side line
    sideLine = np.zeros((sideLineIndexLeft.size, 2))
    for inx, val in enumerate(sideLineIndexLeft):
        sideLine[inx,:] = conesPos[val,:]
    plt.plot([x[0] for x in sideLine], [x[1] for x in sideLine], 'b--')

    # Plot the right side line
    sideLine = np.zeros((sideLineIndexRight.size, 2))
    for inx, val in enumerate(sideLineIndexRight):
        sideLine[inx,:] = conesPos[val,:]
    plt.plot([x[0] for x in sideLine], [x[1] for x in sideLine], 'y--')

    plt.show()

#TODO: What to do if there is no cone on the side
def findStartCone(conesPos, carPos, searchPolygon):
    # Find the start cone

    # Test if there is a cone in polynomial
    polygon = mpltPath.Path(searchPolygon)
    inOrOnPolygon = polygon.contains_points(conesPos)
    inOrOnPolygon = np.array(inOrOnPolygon)
    listOfValidConesIndex = np.where(inOrOnPolygon == True)[0]
    print('list of cones inside the FOV = ', listOfValidConesIndex)

    if listOfValidConesIndex.size == 0:
        print('ERROR: No cone found in findStartCone')
        return -1

    # Get the closest cone
    closestLocalConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - carPos, axis=1))
    print('closestConeIndex = ', closestLocalConeIndex)

    return listOfValidConesIndex[closestLocalConeIndex]

    


#TODO: Fix rotation
def drawCar(carPos, carDirection):
    # Draw a car given a position and a direction
    print('carDirection = ', carDirection)
    carLength = 1.525
    carWidth = 0.9

    # Make a rectangel with the car dimensions
    carRectangle = np.array([[carPos[0] - carLength/2, carPos[1] - carWidth/2],
                                [carPos[0] - carLength/2, carPos[1] + carWidth/2],
                                [carPos[0] + carLength/2, carPos[1] + carWidth/2],
                                [carPos[0] + carLength/2, carPos[1] - carWidth/2],
                                [carPos[0] - carLength/2, carPos[1] - carWidth/2]])

    # Rotate the rectangle
    #carRectangle = carRectangle - carPos
    #carRectangle = np.matmul(carRectangle, np.array([[np.cos(carDirection), -np.sin(carDirection)],
    #                                                [np.sin(carDirection), np.cos(carDirection)]]))
    #carRectangle = carRectangle + carPos

    # Plot the rectangle
    plt.plot(carRectangle[:,0], carRectangle[:,1], 'r-')
    plt.plot(carPos[0], carPos[1], 'r*')

    return carRectangle

#TODO: Fix rotation
def drawSideConesFinder(carPos, carDirection, side):
    # Draw the side cones finder
    sideRectangle = np.zeros((5, 2))
    if side == 'left':
        sideRectangle = np.array([[carPos[0], carPos[1]],
                                    [carPos[0] + sideSearchX_, carPos[1]],
                                    [carPos[0] + sideSearchX_, carPos[1] + sideSearchY_],
                                    [carPos[0], carPos[1] + sideSearchY_],
                                    [carPos[0], carPos[1]]])
    elif side == 'right':
        sideRectangle = np.array([[carPos[0], carPos[1]],
                                    [carPos[0] + sideSearchX_, carPos[1]],
                                    [carPos[0] + sideSearchX_, carPos[1] - sideSearchY_],
                                    [carPos[0], carPos[1] - sideSearchY_],
                                    [carPos[0], carPos[1]]])

    # Rotate the rectangle
    #sideRectangle = sideRectangle - carPos
    #sideRectangle = np.matmul(sideRectangle, np.array([[np.cos(carDirection), -np.sin(carDirection)],
    #                                                [np.sin(carDirection), np.cos(carDirection)]]))
    #sideRectangle = sideRectangle + carPos

    # Plot the rectangle
    plt.plot(sideRectangle[:,0], sideRectangle[:,1], 'g-')

    return sideRectangle





# TODO: Remove the previousCone from the list of cones
# FIXED TODO: Seams to be a bug where some cones are not found in the FOV
#       Think this only happens when we go above some amount of cones 
#       CONCLUSION: Think that the polygon sometimes has the startCone in it
#                   and sometimes not. Think this is a bug in the contains_points function
# TODO: Maybe change this to a iterative function instead of recursive
# Probebly add a pointer to bestPathIndex in C++ to make it easier
# TODO: Add if statement to check for multiple cones in the same FOV and weigh the angle
#       to the distance to the cone. This should remove the problem where it is hard to
#       find the cone in a hairpin turn
def findSideLine(conesPos, startCone, previousCone, bestPathIndex = [], alreadyUsedCones = []):
    # Find the side line given the cones and a start cone (Recusive function)
    
    print('- - - - - - - - - - - - - - findSideLine - - - - - - - - - - - - - -')


    # Find the current cone
    print('startCone = ', startCone)
    currentConeIndex = np.argmin(np.linalg.norm(conesPos[:,:] - startCone, axis=1))
    print('currentConeIndex = ', currentConeIndex)

    if currentConeIndex in alreadyUsedCones:
        print('currentConeIndex = ', currentConeIndex)
        print('alreadyUsedCones = ', alreadyUsedCones)
        print('currentConeIndex is already used')
        return

    # Calculate the angle from the previous cone to the current cone
    angle = 0

    print('startCone = ', startCone)
    
    prevCone = previousCone
    print('prevCone = ', prevCone)
    angleRad = np.arctan2(startCone[1] - prevCone[1], startCone[0] - prevCone[0])
    print('angleRad = ', angleRad)
    angle = angleRad * 180 / np.pi

    # Test if there is a cone in the FOV
    searchPolygon = drawLineOfSight(startCone, angle)
    polygon = mpltPath.Path(searchPolygon)
    inOrOnPolygon = polygon.contains_points(conesPos)
    inOrOnPolygon = np.array(inOrOnPolygon)
    listOfValidConesIndex = np.where(inOrOnPolygon == True)[0]
    print('list of cones inside the FOV = ', listOfValidConesIndex)

    # If there is cone in the FOV, return
    if listOfValidConesIndex.size == 0:
        print('No cone in the FOV')
        return

    # If there is more than one cone in the FOV, find the closest cone

    # If the startCone is in the list, remove it
    currentConeIndexLocal = np.where(listOfValidConesIndex == currentConeIndex)[0]
    print('currentConeIndexLocal = ', currentConeIndexLocal)
    if currentConeIndexLocal.size > 0:
        print('currentConeIndex in listOfValidConesIndex')
        listOfValidConesIndex = np.delete(listOfValidConesIndex, currentConeIndexLocal)  
    else:
        print('currentConeIndex not in listOfValidConesIndex')
        print('------------------------------------------------------------------------------')

    print('listOfValidConesIndex = ', listOfValidConesIndex)

    # If the startCone was the only cone in the FOV, return
    if listOfValidConesIndex.size == 0:
        print('Only the startCone in the FOV')
        return

    # Find the closest cone
    closestConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - startCone, axis=1))
    closestCone = conesPos[listOfValidConesIndex[closestConeIndex],:]
    print('closestCone = ', closestCone, ' at index ', listOfValidConesIndex[closestConeIndex])

    # Find the side line
    previousCone = startCone
    alreadyUsedCones = np.append(alreadyUsedCones, currentConeIndex)
    plt.plot(conesPos[currentConeIndex,0], conesPos[currentConeIndex,1], 'go')

    bestSideLineIndexRet = findSideLine(conesPos, closestCone, previousCone, bestPathIndex, alreadyUsedCones)

    bestSideLineIndex = np.append(listOfValidConesIndex[closestConeIndex], bestSideLineIndexRet)


    return bestSideLineIndex
    



# turnAngleFromCar is the angle from the car, the car is always facing 0 degrees (maybe)
def drawLineOfSight(conePos, turnAngleFromCar):
    radius = searchRadius_
    FOV = _degreeSearchRad_
    turnAngleFromCarRad = turnAngleFromCar * np.pi / 180
    numberOfPointsOnCircle = 10
    x, y = conePos[0], conePos[1]

    rightFOV = -FOV/2 + turnAngleFromCarRad
    leftFOV = FOV/2 + turnAngleFromCarRad

    theta = np.linspace(rightFOV, leftFOV, numberOfPointsOnCircle)
    x_circle = radius * np.cos(theta) + x
    y_circle = radius * np.sin(theta) + y

    xCord = np.zeros((numberOfPointsOnCircle+2,1))
    xCord[0] = x
    xCord[-1] = x
    for inx, val in enumerate(x_circle):
        xCord[inx+1] = val

    yCord = np.zeros((numberOfPointsOnCircle+2,1))
    yCord[0] = y
    yCord[-1] = y
    for inx, val in enumerate(y_circle):
        yCord[inx+1] = val

    plt.plot(xCord, yCord, 'r-')

    return np.concatenate((xCord, yCord), axis=1)


    




    




if __name__ == '__main__':
    main()
