import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Set of variable
n=20000 #chunk size
chnklst=[] # Empty list of chunks

# Load data set and chunking
for i in pd.read_csv("ratings.csv", chunksize=n):
    chnklst.append(i)
ratings = pd.concat(chnklst, ignore_index=True) # concatanation of chunks

ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')  # Convert timestamp

# Random Split
train,test = train_test_split(ratings, train_size=0.8, random_state=42)

# Use Random Split dataset
# Create a matrix with rows and columns representing user and the movie respectively. The entries of the metrix represents the ratings
RateMtrx = train.pivot(index='userId', columns='movieId', values='rating')
RateMtrx = RateMtrx.apply(lambda row: row.fillna(row.mean()), axis=1) # Fill null values using mean value

# Perform SVD
Svd = TruncatedSVD(n_components=30, random_state=42)
U = Svd.fit_transform(RateMtrx.values)
Sg = np.diag(Svd.singular_values_)  # Convert singular values into diagonal form
Vt = Svd.components_

# Reconstruction of RateMtrx2
RateMtrx2 = np.dot(U, Vt)

# Normalize Predicted Ratings
Min = ratings['rating'].min()
Max = ratings['rating'].max()

# Clip predictions to match the rating scale (1-5)
RateMtrx2 = np.clip(RateMtrx2, Min, Max)

# Convertion of test set into binary (If rating greater than or equal 3 then 1, Else 0)
def binaryrating(data):
    data['BinaryRating'] = (data['rating'] >= 3).astype(int)
    return data

test = binaryrating(test)

def RatePrediction(userid, movieid):
    default=Min # default rating
    if userid in RateMtrx.index and movieid in RateMtrx.columns:
        i=RateMtrx.index.get_loc(userid)
        j=RateMtrx.columns.get_loc(movieid) # Taking the row and column numbers from the Original matrix
        return RateMtrx2[i,j]
    else:
        return default

# Generate predictions for test set
test['RatingPrediction'] = test.apply(lambda row: RatePrediction(row['userId'], row['movieId']), axis=1)
test['BinaryPrediction'] = (test['RatingPrediction'] >= 3).astype(int)

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(test['rating'], test['RatingPrediction']))
print ("\nRandom Split")
print(f'RMSE: {rmse:.4f}')

# Precision, Recall, F1-score
precision, recall, f1score, x = precision_recall_fscore_support(test['BinaryRating'], test['BinaryPrediction'], average='binary')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1score:.4f}')

# For a particular userId, recommending top five movies
usrId = 18
UsrIndx = RateMtrx.index.get_loc(usrId)
predicted_ratings = pd.Series(RateMtrx2[UsrIndx], index=RateMtrx.columns)
print("\nRandom Split Top Five Movies Recommendation for the User ID ", usrId)
print(predicted_ratings.sort_values(ascending=False).head(6))

# Temporal Split

# Temporal Split (Using 8th quantile timestamp as the split point)
time_threshold = ratings['timestamp'].quantile(0.8)
trainTemp = ratings[ratings['timestamp'] <= time_threshold]
testTemp = ratings[ratings['timestamp'] > time_threshold].copy()

# Create a matrix
RateMtrx3 = trainTemp.pivot(index='userId', columns='movieId', values='rating')
RateMtrx3 = RateMtrx3.apply(lambda row: row.fillna(row.mean()), axis=1)

# svd
Svd = TruncatedSVD(n_components=30, random_state=42)
U = Svd.fit_transform(RateMtrx3.values)
Sg = np.diag(Svd.singular_values_)
Vt = Svd.components_

# Reconstruction of RateMtrx4
RateMtrx4 = np.dot(U, Vt)

# Normalize Predicted Ratings
Min = ratings['rating'].min()
Max = ratings['rating'].max()

# Clip predictions to match the rating scale (1-5)
RateMtrx4 = (RateMtrx4 - Min) / (Max - Min) * 4 + 1

# Convert test set into binary labels
def binaryrating(data):
    data['BinaryRating'] = (data['rating'] >= 3).astype(int)
    return data

testTemp = binaryrating(testTemp)

def RatePrediction(userid, movieid):
    default=Min
    if userid in RateMtrx3.index and movieid in RateMtrx3.columns:
        i=RateMtrx3.index.get_loc(userid)
        j=RateMtrx3.columns.get_loc(movieid)
        return RateMtrx4[i,j]
    else:
        return default

# Generate predictions for test set
testTemp['RatingPrediction'] = testTemp.apply(lambda row: RatePrediction(row['userId'], row['movieId']), axis=1)
testTemp['BinaryPrediction'] = (testTemp['RatingPrediction'] >= 3).astype(int)

# Root Mean Square Error
rmse = np.sqrt(mean_squared_error(testTemp['rating'], testTemp['RatingPrediction']))
print ("\nTemporal Split")
print(f'RMSE: {rmse:.4f}')

# Precision, Recall, F1-score
precision, recall, f1score, support = precision_recall_fscore_support(testTemp['BinaryRating'], testTemp['BinaryPrediction'], average='binary')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1score:.4f}')

# For a particular userId, recommending top five movies
usrId = 18
UsrIndx = RateMtrx3.index.get_loc(usrId)
predicted_ratings = pd.Series(RateMtrx4[UsrIndx], index=RateMtrx3.columns)
print("\nTemporal Split Top Five Movies Recommendation for the User ID ", usrId)
print(predicted_ratings.sort_values(ascending=False).head(6))