#######################################################################################################################
# Class for article similarity based Recommender System model
class article_similarity_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.article_id = None
        self.cooccurence_matrix = None
        self.articles_dict = None
        self.rev_articles_dict = None
        self.article_similarity_recommendations = None
    # find unique articles corresponding to a given user
    def find_user_articles(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_articles = list(user_data[self.article_id].unique())
        
        return user_articles
    
    # find unique users for a given article
    def find_article_users(self, article):
        article_data = self.train_data[self.train_data[self.article_id] == article]
        article_users = set(article_data[self.user_id].unique())       
        
        return article_users
    
    # find unique articles in the training data
    def find_all_articles_train_data(self):
        all_articles = list(self.train_data[self.article_id].unique())        
        return all_articles
    
    # Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_articles, all_articles):
        # find users for all articles in user_article
        user_articles_users = []        
        for i in range(0, len(user_articles)):
            user_articles_users.append(self.find_article_users(user_articles[i]))     
        # Initialize the article cooccurence matrix of size len(user_articles) X len(articles)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_articles), len(all_articles))), float)   
        # Calculate similarity between user articles and all unique articles in the training data
        for i in range(0,len(all_articles)):
            # Calculate unique readers (users) of article  i
            articles_i_data = self.train_data[self.train_data[self.article_id] == all_articles[i]]
            users_i = set(articles_i_data[self.user_id].unique())
             
            for j in range(0,len(user_articles)):          
                # find unique readers (users) of article  j
                users_j = user_articles_users[j]
                # Calculate intersection of readers of articles i and j
                users_intersection = users_i.intersection(users_j)
                # Calculate cooccurence_matrix[i,j]
                if len(users_intersection) != 0:
                # Calculate union of readers of articles i and j
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0                    
        return cooccurence_matrix
    # Use the cooccurence matrix to make top recommendations
    def finding_top_recommendations(self, user, cooccurence_matrix, all_articles, user_articles):
        # Calculate a weighted average of the scores in cooccurence matrix for all user articles.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
        # Sort the indices of user_sim_scores based upon their value also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        # Create a dataframe from the following
        columns = ['user_id','article','user_name', 'score', 'rank'] 
        df = pd.DataFrame(columns=columns)
        # Fill the dataframe with top 10 article based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_articles[sort_index[i][1]] not in user_articles and rank <= 10:
                df.loc[len(df)]=[user,all_articles[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        # Handle the case where there are no recommendations     
        if df.shape[0] == 0:
            print("This user has no articles for training the article-similarity based model.")
            return -1
        else:
            return df
    # Create the article similarity based recommender system 
    def create(self, train_data, user_id, article_id):
        self.train_data = train_data
        self.user_id = user_id
        self.article_id = article_id
    # Use the article similarity based recommender system to make recommendations
    def recommend(self, user):
        #  find all unique articles for this user
        user_articles = self.find_user_articles(user)
        print("No. of unique articles for the user: %d" % len(user_articles))

        #  find all unique articles in the training data
        all_articles = self.find_all_articles_train_data()
        print("no. of unique articles in the training set: %d" % len(all_articles))

        #  Construct article cooccurence matrix of size len(user_articles) X len(articles)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_articles, all_articles)        
        # D. Use the cooccurence matrix to make recommendations
        df_recommendations = self.finding_top_recommendations(user, cooccurence_matrix, all_articles, user_articles)                
        return df_recommendations
    # find similar articles to given articles
    def find_similar_articles(self, article_list):
        user_articles = article_list
        # B. find all unique articles in the training data
        all_articles = self.find_all_articles_train_data()    
        print("no. of unique articles in the training set: %d" % len(all_articles))
        # C. Construct article cooccurence matrix of size len(user_articles) X len(articles)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_articles, all_articles)
        # D. Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.finding_top_recommendations(user, cooccurence_matrix, all_articles, user_articles) 
        return df_recommendations
