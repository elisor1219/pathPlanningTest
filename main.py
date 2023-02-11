import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath


# PARAMETERS
degreeSearch_ = 120 # degrees
_degreeSearchRad_ = degreeSearch_ * np.pi / 180
searchRadius_ = 3

def main():



    # Import a map of cones or build one
    plt.axis([-10, 10, -10, 10])
    plt.grid()
    conesPos = plt.ginput(-1,-1)
    plt.plot([x[0] for x in conesPos], [x[1] for x in conesPos], 'bo')

    # Convert to numpy array
    conesPos = np.array(conesPos)

    # TODO: Find start cone right and left
    startConeIndex = np.where(conesPos == conesPos[0,:])[0][0]
    sideLineIndex = findSideLine(conesPos, conesPos[startConeIndex,:], previousCone = conesPos[startConeIndex,:])
    sideLineIndex = np.delete(sideLineIndex, -1) # type: ignore
    sideLineIndex = np.append(startConeIndex, sideLineIndex)
    print('sideLineIndex = ', sideLineIndex)
    #for cone in conesPos:
    #    temp = drawLineOfSight(cone, 0)
    #    plt.plot(temp[:,0], temp[:,1], 'r-')
    #    temp2 = mpltPath.Path(temp)
    #    testCone = np.array([cone])
    #    #print(temp2.contains_points(testCone))
        

    plt.show()

    plt.figure()
    plt.axis([-10, 10, -10, 10])
    plt.grid()
    plt.plot([x[0] for x in conesPos], [x[1] for x in conesPos], 'bo')
    sideLine = np.zeros((sideLineIndex.size, 2))
    for inx, val in enumerate(sideLineIndex):
        sideLine[inx,:] = conesPos[val,:]
    plt.plot([x[0] for x in sideLine], [x[1] for x in sideLine], 'b--')
    plt.show()


# TODO: Seams to be a bug where some cones are not found in the FOV
#       Think this only happens when we go above some amount of cones 
#       CONCLUSION: Think that the polygon sometimes has the startCone in it
#                   and sometimes not. Think this is a bug in the contains_points function
# TODO: Maybe change this to a iterative function instead of recursive
# Probebly add a pointer to bestPathIndex in C++ to make it easier
def findSideLine(conesPos, startCone, previousCone, bestPathIndex = None):
    # Find the side line given the cones and a start cone (Recusive function)
    
    print('- - - - - - - - - - - - - - findSideLine - - - - - - - - - - - - - -')

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

    # If there is only one cone in the FOV, return the cone
    # This is the startCone
    if listOfValidConesIndex.size == 1:
        print('Only one cone in FOV')
        return

    # If there is more than one cone in the FOV, find the closest cone
    # The closest cone is the startCone, so remove it from the list
    closestConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - startCone, axis=1))
    closestCone = conesPos[listOfValidConesIndex[closestConeIndex],:]
    print('closestCone = ', closestCone, ' at index ', listOfValidConesIndex[closestConeIndex])

    # Remove the start cone from the list
    listOfValidConesIndex = np.delete(listOfValidConesIndex, closestConeIndex)
    print('listOfValidConesIndex = ', listOfValidConesIndex)

    # Find the closest cone
    closestConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - startCone, axis=1))
    closestCone = conesPos[listOfValidConesIndex[closestConeIndex],:]
    print('closestCone = ', closestCone, ' at index ', listOfValidConesIndex[closestConeIndex])

    # Find the side line
    previousCone = startCone
    bestSideLineIndexRet = findSideLine(conesPos, closestCone, previousCone, bestPathIndex)

    bestSideLineIndex = np.append(listOfValidConesIndex[closestConeIndex], bestSideLineIndexRet)


    return bestSideLineIndex
    
    ## Know that this works
    ## Find the closest cone
    #closestConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - startCone, axis=1))
    ## Remove the start cone from the list
    #listOfValidConesIndex = np.delete(listOfValidConesIndex, closestConeIndex)
    ## Find the closest cone
    #closestConeIndex = np.argmin(np.linalg.norm(conesPos[listOfValidConesIndex,:] - startCone, axis=1))
    ## Find the closest cone
    #closestCone = conesPos[listOfValidConesIndex[closestConeIndex],:]
    ## Draw the line of sight
    #temp = drawLineOfSight(closestCone, 0)
    #plt.plot(temp[:,0], temp[:,1], 'r-')
    ## Find the side line
    #findSideLine(conesPos, closestCone)


    

    # Find the side line


    #return listOfValidConesIndex

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
