
import pandas as pd

############################################
# User-Based Collaborative Filtering
#############################################

# VERİ SETİNİN HAZIRLANMASI

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('movie.csv')
    rating = pd.read_csv('rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index # 1000 den az comment alan filmleri filtrelemek
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# rastgele bir id seçelim ve bu kişi üzerinden ilerleyelim

random_user = int(pd.Series(user_movie_df.index).sample(1).values)

user_df = user_movie_df[user_movie_df.index == random_user]

kullanıcının_izledikleri = user_df.columns[user_df.notna().any()].tolist()

#check
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == 'Aladdin (1992)']


# Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek

pd.set_option('display.max_columns', 5)

movies_watched_df = user_movie_df[kullanıcının_izledikleri]

len(kullanıcının_izledikleri)
## Her bir kullanıcı kaç tane benzer film izlediğini bulmak için notnull olan değerleri topluyoruz

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

perc = round(len(kullanıcının_izledikleri) * 0.6) # kullanıcının izlediklerinin en az %60'ını izlemiş olmalarını istiyoruz

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

users_same_movies.count()

# Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
type(final_df)

corr_df = final_df.T.corr().unstack().sort_values()
corr_df.shape
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()


# yüzde 65 ve üzeri korelasyona sahip kullanıcılar:

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65) & (corr_df["user_id_2"] != random_user)][
    ["user_id_2", "corr"]].reset_index(drop=True)

#korelasyon olarak aldığımız benzer puanları vereceğini düşündüklerimizi ifade ediyor
# ne kadar fazla yakını puan verdiklerinin korelasyonu tutuluyor.

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('rating.csv')

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')


# Weighted Average Recommendation Score'un Hesaplanması

# weighted_rating'in hesaplanması.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.7]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.6].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

    #MoveId  weighted_rating    TITLE
# 0	    1023	3.299888	Winnie the Pooh and the Blustery Day (1968)
# 1	    1956	2.931275	Ordinary People (1980)
# 2	    2870	2.931275	Barefoot in the Park (1967)
# 3	    3451	2.931275	Guess Who's Coming to Dinner (1967)
# 4	    3244	2.931275	Goodbye Girl, The (1977)
# 5	    3145	2.931275	Cradle Will Rock (1999)
# 6	    3097	2.931275	Shop Around the Corner, The (1940)
# 7	    3028	2.931275	Taming of the Shrew, The (1967)
# 8	    946	    2.931275	To Be or Not to Be (1942)
# 9	    2929	2.931275	Reds (1981)
# 10	955	    2.931275	Bringing Up Baby (1938)
# 11	3551	2.931275	Marathon Man (1976)
# 12	2565	2.931275	King and I, The (1956)
# 13	2205	2.931275	Mr. & Mrs. Smith (1941)
# 14	1188	2.931275	Strictly Ballroom (1992)
# 15	2022	2.931275	Last Temptation of Christ, The (1988)
# 16	1820	2.931275	Proposition, The (1998)
# 17	1409	2.931275	Michael (1996)
# 18	3497	2.931275	Max Dugan Returns (1983)
# 19	3185	2.931275	Snow Falling on Cedars (1999)











