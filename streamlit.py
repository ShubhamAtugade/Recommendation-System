
#Step 1: Importing necessary libraries
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD 
from surprise.model_selection import cross_validate 

#Step 2: Load the dataset into jupyter notebook
data = pd.read_csv('OnlineRetail.csv')

#Step 3: Viewing 5 rows of data
data.head(5)

#Step 4: Creating a Reader object and specifying the rating scale
reader = Reader(rating_scale=(0, data['Quantity'].max()))

#Step 5: Creating the dataset from the pandas dataframe
data_for_surprise = Dataset.load_from_df(data[['CustomerID', 'StockCode', 'Quantity']], reader)

#Step 6: Using the Singular value decomposition (SVD) algorithm for collaborative filtering
algo = SVD()

#Step 7: Evaluating the algorithm with cross-validation
cross_validate(algo, data_for_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#Step 8: Training the model on the entire dataset
trainset = data_for_surprise.build_full_trainset()
algo.fit(trainset)

#Step 9: Function to get top n recommendations for a g iven customer
def top_recommendations(customer_id, n=15):

    customer_id = float(customer_id)
    
    #list of all products
    all_products = data['Description'].unique()
    
    #list of products the customer has already bought
    purchased_products = data[data['CustomerID'] == customer_id]['Description'].unique()
    
    #list of products the customer has not bought yet 
    products_to_predict = [product_description for product_description in all_products if product_description not in purchased_products] 
    
    # Predict the ratings for all products the customer has not bought yet
    predictions = [algo.predict(customer_id, product_description) for product_description in products_to_predict]
    
    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est)
        
    # top N recommendations
    top_recommendations = [pred.iid for pred in predictions[:n]]
    
    return top_recommendations


#Step 10: Getting the recommendated product list
st.title("Product Recommendation System For Enterprise")
st.write("This product recommendation system uses customer id to recommend product to particular customer.")
customer_id = st.number_input('Enter the Customer ID')
if customer_id not in data['CustomerID'].unique():
            st.error(f"Customer ID {customer_id} not found in the data.")
else:
    top_product_recommendations = top_recommendations(customer_id, n=5)
    st.write(f'\nTop 10 recommendated products for {customer_id}:')

    for product in top_product_recommendations:
        st.write(product)
