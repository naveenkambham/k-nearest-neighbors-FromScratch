"""
Title:Implement KNN Classifier algorithm and run various test cases to understand the algorithm.
Developer : Naveen Kambham
ID:nkk300
Description: This is a simple KNN classifier implemented using nparrays. It has various test cases like adding the feed back
             to training data and sorting the data and changing different K values. Here I have used euclidean distance as a
             distance metric.
"""

"""
Importing the required libraries. timeit - for performance recording, pandas - for data frames, math - for math operations
matplotlib-for plotting functionality, numpy - for data structures
"""
import pandas as pd
import math
import operator
import timeit
import matplotlib.pyplot as plt
import numpy as np

""" euclidean_distance(row1, row2, no_of_parameters) : [in] points, no of featurs [out] distance between given points
Function to measure the distance between two points row1 - first point, row2 - second point,
no_of_parameter -number of Features to be included in measuring the distance, like 2 for distance
between (x2,y2),(x3,y3) and 3 for distance between (x2,y2,z2),(x3,y3,z3)
"""


def euclidean_distance(row1, row2, no_of_parameters):
    distance = 0
    # iterate through the parameters
    for i in range( no_of_parameters ):
        # sum up the distance
        distance += pow( (row1[i] - row2[i]), 2 )
    return math.sqrt( (distance) )


""" find_Neighbours(trainingset, testcase, k, no_of_features) : [in] trainingset, testcase, k, no_of_features
[out] list of neighbours with distances.
Function to find the K-neighbours for the give datapoint in training Set.
"""
def find_Neighbours(trainingset, testcase, k, no_of_features):
    # List to store the classification and distance
    distances = []
    # Iterate the traing data
    for row in (trainingset):
        # Compute the distance and then add the class label, distance to list
        distances.append( (row[2], euclidean_distance( row, testcase, no_of_features )) )

    # Sort the distances based on distance
    distances.sort( key=operator.itemgetter( 1 ) )

    # Store the neighbours in list
    neighbours = []
    # Take the K neighbours
    for i in range( k ):
        # Append the neighbour
        neighbours.append( (distances[i]) )
    return neighbours


""" get_classification(neighbours): [in]: neighbours [out]: classification label
Function to return the classification label among the given neighbours.
"""


def get_classification(neighbours):
    # Dictionary to count the votes
    class_votes = {}
    # Iterate the neighbours
    for i in range( len( neighbours ) ):
        # Read the classification label
        response = neighbours[i][-2]
        # Check if the label is already in dictionary
        if response in class_votes:
            # Already this label is recorded increase the vote
            class_votes[response] += 1
        else:
            # Add the label to the dictionary
            class_votes[response] = 1
    # Sort the votes based on votes
    sortedVotes = sorted( class_votes.items( ), key=operator.itemgetter( 1 ), reverse=True )
    # Return the class label with more no of votes
    return sortedVotes[0][0]


"""get_classificationExtended(neighbors): [in]: neighbours [out]: classification label
This method also considers the distance which will be useful when we encounter two class labels with same no of votes
"""


def get_classificationExtended(neighbors):
    # Dictionary to count the votes
    class_votes = {}
    for i in range( len( neighbors ) ):
        # get the label first
        response = neighbors[i][-2]

        # If the label already presents then increase its rank and add the distance to existing distance sum
        if response in class_votes:
            temp_list = class_votes[response]
            temp_list[0] += 1
            # for distance
            temp_list[1] += neighbors[i][-1]
            # already this label is recorded increase the vote
            class_votes[response] = temp_list
        # If the label not presents then add the label to dictionary and store rank, distance
        else:
            class_votes[response] = [1, neighbors[i][-1]]  # add the label to the list

    # sort the votes based on rank
    sortedVotes = sorted( class_votes.items( ), key=operator.itemgetter( 1 ), reverse=True )

    # Take the list at top i.e label with most no of votes
    classified_label = sortedVotes[0][0]
    classified_labelrank = sortedVotes[0][1]

    # Below code is to check cases like two labels might have same rank in that case we will use the distance metric to
    # decide the label
    for i, value in sortedVotes:
        # If no of votes are equal and distance is less than the classified label then
        # reset the classfied label to current one as this is nearer to the data point.
        if value[0] == classified_labelrank[0] and value[1] < classified_labelrank[1]:
            classified_labelrank = value
            classified_label = i

    return classified_label


"""KNN(training_Data, unlabeled_Data, K, feedBack_Classification): [in] : Training Data, Unlabeled data, K value,
feedBack_classification - bolean flag whether to add the predicted data point to training data or not. True - To add , False - To skip
"""
def KNN(training_Data,unlabeled_Data,K,feedBack_Classification):
    # Iterate through the unlabeled data and take each data point
    for row in (unlabeled_Data):
        # Find the label for the data point
        label = get_classificationExtended(find_Neighbours(training_Data,row,K,2))
        # Update the label to the data point
        row[3] = label
        if (feedBack_Classification == "True"):
            # Feed back is true, so add the updated row to training data
            training_Data = np.append( training_Data, [np.array( [row[0], row[1], row[3]] ).tolist( )], axis=0 )
    # Print the accuracy
    print( "Accuracy:", accuracy( unlabeled_Data ) )
    return unlabeled_Data


'''confusionMatrix(label, trueLabel) : [in] label, truelabel [out] confusion matrix
Method  for computing the Confusion Matrix
'''
def confusionMatrix(label,trueLabel):
    matrix = np.zeros( (5, 5), dtype=int )
    for true, pred in zip( label, trueLabel ):
        matrix[int( true ) - 1][int( pred ) - 1] += 1
    return matrix


''' accuracy(data): [in] data [out] accuracy
Method to check the accuracy of KNN
'''
def accuracy(data):
    correct = 0
    for row in data:
        # See if the predicted and true labels are correct or not
        if row[2] == row[3]:
            # If correct increase the count
            correct += 1
    return correct / len( data ) * 100


''' Histogram(label, trueLabel, title): [in] predicted Label, true label, title for histogram [out] Histogram
Method for Plotting Histogram
'''
def Histogram(label, trueLabel, title):
    #Plot the histogram
    plt.hist( [label, trueLabel], label=['Pred Label', 'True Label'] )
    # Give X,Y labels and titles.
    plt.xlabel( 'Label' )
    plt.ylabel( 'Count' )
    plt.legend( )
    plt.title( title )
    plt.show( block="True" )


''' ScatterPlot(X, Y, label, title): [in] X, Y , Label, title [out] Scatter Plot
Method for scatter plotting
'''
def ScatterPlot(X, Y, label, title):
    # make scatter plot
    plt.scatter( X, Y, c=label )
    plt.title( title )
    plt.show( block="True" )


def Main():
    # Read the data set
    df = pd.read_csv(
        'C:\Education\ML\AssignMent\DataSet\knnDataSet.csv' )  # Read the CSV file and store the table in Pandas Data Frame

    # problem 1: Scatter Plot

    # Dictionary to colour different class labels with different colours
    use_colours = {1.0:"darkblue",2.0: "black", 3.0: "green", 4.0: "orange",5.0:"red" }
    for index,row in df.iterrows():
        # If L is null then give + mark other wise give O marker, colour will be picked up by dictionary
        if (pd.isnull(row['L'])):
           plt.scatter(row['x'],row['y'],c=use_colours[row['TL']],marker="+",s=180)
        else:
           plt.scatter(row['x'],row['y'],c=use_colours[row['TL']],marker="o",s=180)
    # Plot the Labels, title
    plt.xlabel('X', fontsize=14, color='red')
    plt.ylabel('Y',fontsize=14, color='red')
    plt.title('Scatter Plot for Classes 1,2,3,4,5(O-Labeled Data)',fontsize=18, color='red')

    # Creates 3 Rectangles for legend information
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc=use_colours[1.0])
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc=use_colours[2.0])
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc=use_colours[3.0])
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc=use_colours[4.0])
    p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc=use_colours[5.0])

    # Adds the legend into plot
    plt.legend((p1, p2, p3,p4,p5), ('1.0', '2.0', '3.0','4.0','5.0'), loc='best')
    plt.show()

    # Problem 3: Testing the KNN clasifier for K values 1, 5, 10, 20 and feedback classification False
    df['L'] = df['L'].astype( 'category' )
    df['TL'] = df['TL'].astype( 'category' )
    # Drop NaN rows
    trainSet = df.dropna( )

    # Drop training data to get the testing data and drop unnecessary columns
    testSet = df.drop( trainSet.index )
    del trainSet['TL'], trainSet['Unnamed: 0']
    del testSet['L'], testSet['Unnamed: 0']
    trainSet = trainSet.values
    testSet = testSet.values

    # adding a zero value column. This is needed to update the label value after running KNN
    # If we add new column while iterating since numpy creates continuous memory allocation
    # for faster performance, it will recreate the entire array again which will cause performance
    # hit, hence it is recommended to preallocate the array.
    new_col = np.zeros( (len( testSet ), 1) )
    testSet = np.append( testSet, new_col, axis=1 )

    print( "Problem 3:" )
    # Run KNN for different K values with feed Back classification as false
    for k in [1, 5, 10, 20]:
        print( "Testing K-", k, "Classifier: FeedBack_Classification -False" )
        # Timer to print the performance
        start = timeit.default_timer( )
        # Run KNN
        LabelData = KNN( trainSet, testSet, k, "False" )
        # Take the actual labels
        L_actual = [row[2] for row in LabelData]
        # Take the predicted labels
        L_predicted = [row[3] for row in LabelData]
        # Use these values for creating confusion matrix
        print( "Confusion Matrix:" )
        print( confusionMatrix( L_actual, L_predicted ) )

        # using the same values draw histogram and scatter plot
        Histogram(L_actual,L_predicted,"Histogram for KNN-"+k.__str__())
        X =[row[0] for row in LabelData]
        Y =[row[1] for row in LabelData]
        ScatterPlot(X,Y,L_predicted,"Scatter Plot for KNN-"+k.__str__())
        print( "Time to execute the classifier:", timeit.default_timer( ) - start )
        print( "-------------------------------------------------------------------------------------" )

    # Problem 4: Sort the data by TL column and set true for the feed back classification
    # Read the CSV file and store the table in Pandas Data Frame
    df2 = pd.read_csv( 'C:\Education\ML\AssignMent\DataSet\knnDataSet.csv' )
    # Sorting the df by "TL" Value in Descending order
    df2 = df2.sort_values( ['TL'], ascending=[True] )
    # Collect the traing data and testing data
    df2['L'] = df2['L'].astype( 'category' )
    df2['TL'] = df2['TL'].astype( 'category' )
    trainSet2 = df2.dropna( )
    testSet2 = df2.drop( trainSet2.index )
    del trainSet2['TL'], trainSet2['Unnamed: 0']
    del testSet2['L'], testSet2['Unnamed: 0']
    trainSet2 = trainSet2.values
    testSet2 = testSet2.values

    # adding a zero value column. This is needed to update the label value after running KNN
    # If we add new column while iterating since numpy creates continuous memory allocation
    # for faster performance, it will recreate the entire array again which will cause performance
    # hit, hence it is recommended to preallocate the array.
    new_col2 = np.zeros( (len( testSet2 ), 1) )
    testSet2 = np.append( testSet2, new_col2, axis=1 )

    print( "Problem 4:" )
    # Running the Classifer for Different K values and Feedback_Classification is true
    for k in [1, 5, 10, 20]:
        print( "Testing K-", k, "Classifier: FeedBack_Classification -True" )
        # timeit to calculate elapsed time
        start2 = timeit.default_timer( )
        # Run classfier
        LabelData2 = KNN( trainSet2, testSet2, k, "True" )

        # Get actual labels and predicted labels for confusion matrix and to plot histogram,scatter plot
        L_actual2 = [row[2] for row in LabelData2]
        L_predicted2 = [row[3] for row in LabelData2]
        from sklearn.metrics import confusion_matrix
        print( "Confusion Matrix:" )
        print( confusionMatrix( L_actual2, L_predicted2 ) )

        Histogram( L_actual2, L_predicted2, "Histogram for KNN-" + k.__str__( ) )
        X2 = [row[0] for row in LabelData2]
        Y2 = [row[1] for row in LabelData2]
        ScatterPlot( X2, Y2, L_predicted2, "Scatter Plot for KNN-" + k.__str__( ) )
        print( "Time to execute the classifier:", timeit.default_timer( ) - start2 )
        print( "-------------------------------------------------------------------------------------" )

    # Problem 5 Randomizing the order of Unlabeld nodes, using feed back classificaion as true
    # Read the CSV file and store the table in Pandas Data Frame
    df3 = pd.read_csv( 'C:\Education\ML\AssignMent\DataSet\knnDataSet.csv' )
    # Sorting the df by "TL" Value
    df3['L'] = df3['L'].astype( 'category' )
    df3['TL'] = df3['TL'].astype( 'category' )
    trainSet3 = df3.dropna( )

    #prepare test data, training data
    testSet3 = df3.drop( trainSet3.index )
    del trainSet3['TL'], trainSet3['Unnamed: 0']
    del testSet3['L'], testSet3['Unnamed: 0']
    trainSet3 = trainSet3.values
    testSet3 = testSet3.values

    # adding a zero value column. This is need to update the label value after running KNN
    # If we add new column while iterating since numpy does continous memory allocation
    # for faster performance, it will recreate the entire array again which will cause performance
    # hit, hence it is recommended to preallot the array.
    new_col3 = np.zeros( (len( testSet3 ), 1) )
    testSet3 = np.append( testSet3, new_col3, axis=1 )
    np.random.shuffle( testSet3 )

    # Running the Classifer for Different K values and Feedback_Classification is true
    print( "Problem 5:" )
    for k in [1, 5, 10, 20]:
        print( "Testing K-", k, "Classifier: FeedBack_Classification -True , Unlabeled data : Random Order" )
        start3 = timeit.default_timer( )
        # Run KNN
        LabelData3 = KNN( trainSet3, testSet3, k, "True" )

        # Get the actual and predicted values for confusion matrix, histogram, scatter plots
        L_actual3 = [row[2] for row in LabelData3]
        L_predicted3 = [row[3] for row in LabelData3]
        from sklearn.metrics import confusion_matrix
        print( "Confusion Matrix:" )
        print( confusionMatrix( L_actual3, L_predicted3 ) )

        Histogram(L_actual3,L_predicted3,"Histogram for KNN-"+k.__str__())
        X3 =[row[0] for row in LabelData3]
        Y3 =[row[1] for row in LabelData3]
        ScatterPlot(X3,Y3,L_predicted3,"Scatter Plot for KNN-"+k.__str__())
        print( "Time to execute the classifier:", timeit.default_timer( ) - start3 )
        print( "-------------------------------------------------------------------------------------" )


Main( )
