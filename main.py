
import numpy as np
import pandas as pd
import collections
import sklearn
import sklearn.manifold
import random
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np

# Load each data set (users, movies, and ratings).
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names= users_cols, encoding= 'latin-1')# read user data from u.user use | as the separator, names=user_cols, and 'latin-1' as the encoding

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv( 'u.data', sep= '\t', names=ratings_cols, encoding='latin-1')# Ratings data from u.data, use '\t' as separator, names=ratings_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies =pd.read_csv('u.item',sep='|',names=movies_cols, encoding='latin-1') # read movies information from u.item. Look at the file to figure out the appropriate separator. Use names=movie_cols, and encoding as 'latin-1'

# Since the ids in the dataset start at 1, we shift them to start at 0.
users["user_id"] = users["user_id"].apply(lambda x: str(x-1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))


#retrieves user's vector
def getUserMovieProfile(user):
    userInfo = users[users["user_id"] == user]
    usersProfile = userInfo.loc[:,['user_id', 'age', 'sex', 'occupation']]
    return usersProfile

#helper method to get movies user has watched
def getUserMovieProfile(user):
    userInfo = ratings[ratings["user_id"] == user]
    usersMovies = np.array(userInfo.loc[:,['movie_id','rating']])
    return usersMovies

#finds movie title given an array of movie ids
def getMovieRecTitle(movieRec):

    #holds movie title
    movieTitleRec = []

    #looks through movie data set but only the columns of movie id and title
    movieList = np.array(movies.loc[:,['movie_id','title']])
    #print(movieList)

    #loops through first 10 movieids to search for titles
    for movieid in movieRec[0:10]:
        #print('10 Movies: ')
        #print(movieid)
        for x in movieList:
            if x[0] == movieid:
                #updates matching movieids to movie rec lists
                movieTitleRec.append(x[1])

    #prints out top 10 reccomentation s
    print("Your top 10 reccomendations are: ")
    print(movieTitleRec)



#returns common movies between two users
def getCommonMovies(user1, user2):
    #user1 = getUserMovieProfile(user1)
    #user2 = getUserMovieProfile(user2)
    common_movies = []
    for item in user1:
        for item2 in user2:
            if item[0] == item2[0]:
                common_movies.append(item[0])
    #print(common_movies)
    return common_movies

#method computes simiilarity between two users
def compute_cosine_similarity(user1,user2):

    #retrieves array of userid, movies user watched, and rating
    user1 = getUserMovieProfile(user1)
    user2= getUserMovieProfile(user2)

    #creates local common movies between users with helper function
    common = getCommonMovies(user1,user2)
    #print(common_movies)

    user1rating = []
    user2rating = []

    #finds rating of user1 and user2 give their common movies
    for movie in getCommonMovies(user1,user2):
        for rating in user1:
            if movie == rating[0]:
                user1rating.append(rating[1])

    for movie in getCommonMovies(user1,user2):
        for rating in user2:
            if movie == rating[0]:
                user2rating.append(rating[1])

    #reshapes array in order to compute cosine similarity
    user1rating = np.array([user1rating]).reshape(1, -1)
    user2rating = np.array([user2rating]).reshape(1, -1)

    #returns each users ratings and their common movies
    print(user1rating)
    print(user2rating)
    print('Common Movies')
    #print(getCommonMovies(user1,user2))

    #makes sure users have common movies before computing cosine similarity
    if len(common) == 0:
        return 0

    print("users cose similariry is: ")
    cos_sim = cosine_similarity(user1rating, user2rating)


    print(cos_sim)
    return cos_sim

def getMovieReccs(similarUsers,input_user):

    # arrays holding all similarity scores and the overall score which is the probablility a user will like a movie
    overall_scores = {}
    similarity_scores = {}

    similarUsersId = []

    #gets userid for similar users
    for profiles in similarUsers:
        #print(profiles[0])
        similarUsersId.append(profiles[0])

    print("User similar to you based on cluster are: ")
    print(similarUsersId)


    # loops through every user in dataset
    for user in [x for x in similarUsersId if x != input_user]:
        #print(user)
        #appends cosine similarities to a dictionary
        similarity_score = compute_cosine_similarity(user, input_user)

        if similarity_score <= 0:
            continue

        #creates list of movies input user has not watched
        filtered_list = []

        #gets movies that user hasnt watched in common
        for x in getUserMovieProfile(user):
            if x[0] not in getCommonMovies(user, input_user):
                filtered_list.append(x)

        # calculates probability of whether user will like that unwatched movie
        for item in filtered_list:
            overall_scores.update({item[0]: item[1] * similarity_score})
            similarity_scores.update({item[0]: similarity_score})

    # checks to see if input user even has any matches to reccomend
    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalizing the scores
    movie_scores = np.array([[score / similarity_scores[item], item]
                                 for item, score in overall_scores.items()], dtype=object)

    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations as movie ids
    movie_recommendations = [movie for _, movie in movie_scores]
    print('Your Movie Reccomendations:')
    #calls for previous method
    getMovieRecTitle(movie_recommendations)
    return movie_recommendations

#find cluster user is in
def getUserCluster(user):
    for x in range(num_clusters):
        cluster = np.array(X[Kmean.labels_ == x])
        for y in cluster:
            if y[0] == user: #(WORKS)
                simiUser = np.array(X[Kmean.labels_ == x])
                getMovieReccs(simiUser,user_id)


if __name__ == '__main__':
    #Pick a random user
    #pick a random row from u.user to compare
    rand_user = users.sample()

    #stores user id of random user
    user_id = rand_user['user_id'].iloc[0]

    #list rand_user's list of movies
    rand_user_movies = ratings.loc[ratings['user_id'] == user_id]
    #print(rand_user_movies)
    rand_user_movieids = np.array(rand_user_movies.loc[:,['movie_id','rating']])
    print("Random USER ID")
    print(user_id)

    #encodes female or male
    encoded_gender = pd.get_dummies(users['sex'])
    #print('encoded_gender')

    #lists 21 different occupation types
    #print(users['occupation'].value_counts())

    #creates label encoder to encode occupation
    labelencoder = LabelEncoder()
    users['occupation_encoded'] = labelencoder.fit_transform(users['occupation'])

    #print(users[['occupation', 'occupation_encoded']].sample(20))

    #adds the encoded categories to the original data frame
    users.update(users['occupation_encoded'])
    #users.update(encoded_gender)
    users = users.join(encoded_gender)
    #print(users.head(2))

    #retrieve encoded users vector
    #def getUserEncodedProfile(user):
    #userInfo = users[users["user_id"] == user]
    #usersProfile = userInfo.loc[:, ['user_id', 'age', 'occupation_encoded','F', 'M']]
    #print(usersProfile)

    X = users.loc[:, ['user_id', 'age', 'occupation_encoded','F', 'M']]
    #X = np.array(users.loc[:, ['age', 'occupation_encoded','F', 'M']])
    #print(X)

    #id_map = {x[0]:0, X[1],1}

    #id_map = {id: y for y, id in row for row in X}
    #id_map = {}
    #for index in range(len(X)):
    #print(index)
    #id_map[index] = X.iloc[index]
    #id_map[index] = X[index]

    #print(id_map)

    num_clusters = 7

    #kmean classifier
    Kmean = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    #Kmean.fit(X)
    Kmean.fit_predict(X)

    # Plot the centers of clusters
    cluster_centers = Kmean.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='d', s=120, linewidths=1, color='black',
            zorder=12, facecolors='white')
    #plt.show()
    getUserCluster(user_id)
