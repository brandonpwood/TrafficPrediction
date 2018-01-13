import numpy as np
import csv
import math
import nn1
'''
    Goals:
        Create graph of squared total error as a function of time
        Create a distribution of random cross validation errors
        Create a graph of the radial error as a function of time
        Compute cross validation and radial errors after training
'''
# CV Error = 559.517000403
# Radial Error = 257.019675443

# Define Metparameters for training
dataPoints = 10000
shape = (1, 10, 2)
inScale = 1/1000
outScale = 1

# Open csv file and extract data
def grabData(points, inscale, outscale):
    # Function to convert data into time in minutes from 2000
    def format_time(date, time):
        new_date = [0, 0, 0]
        num = 0
        item = ''
        for x in date:
            if x == '/':
                item  = ''
                num += 1
            else:
                item += x
            new_date[num] = item

        years = int(new_date[2]) - 2000
        months = int(new_date[1])
        days = int(new_date[0])

        new_time = [0, 0]
        num = 0
        item  = ''
        for x in time:
            if x == ':':
                item  = ''
                num += 1
            else:
                item += x
            new_time[num] = item
        hours =  int(new_time[0])
        minutes =  int(new_time[1])
        total = (60*365*24*years + 60*30*24*months + 60*24*days + hours + minutes)*inscale # Scale down
        return total

    # Open from csv.
    data = []
    with open('collisions.csv', 'r') as f:
        cr = csv.reader(f, dialect = 'excel')
        num = 1
        dataPoints = points
        for row in cr:
            if num != 1 and row[4] and row[5]:
                    x = float(row[4])
                    y = float(row[5])
                    x = x%1 * outscale
                    y = y%1 * outscale
                    data.append([format_time(row[0], row[1]), x, y])
            else:
                dataPoints += 1

            num += 1
            if num > dataPoints:
                break

    # Sort into CV data and normal data
    data = np.array(data)
    np.random.shuffle(data)

    # Crop
    cv_size = math.floor(len(data)/10)
    cv_data = data[:cv_size]
    data = data[cv_size:]

    # Split into targets and inputs
    targets = data[:, 1:]
    inputs  = data[:, :1]
    cv_targets = cv_data[:, 1:]
    cv_inputs  = cv_data[:, :1]
    return cv_inputs, cv_targets, inputs, targets

# Initialize Data
cv_inputs, cv_targets, inputs, targets = grabData(dataPoints, inScale, outScale)

# Initialize neural network
nn = nn1.nn1(shape)

# Train and save results
nn.learn(inputs, targets, cv_inputs, cv_targets, outScale)

with open('errors.csv', 'w') as f:
    wr = csv.writer(f, dialect = 'excel')
    wr.writerow(nn.errors)

with open('radials.csv', 'w') as f:
    wr = csv.writer(f, dialect = 'excel')
    wr.writerow(nn.radials)

# Generate distribution of cv_errors:
dist = []
for i in range(1000):
    nn = nn1.nn1(shape)
    dist.append(nn.cvError(cv_inputs, cv_targets))

with open('dist.csv', 'w') as f:
    wr = csv.writer(f, dialect = 'excel')
    wr.writerow(dist)
