import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt 
#import plotly  
#from plotly import graph_objs as go 
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from statsmodels.tsa.arima.model import ARIMA
import datetime
# import pmdarima as pm
# from sklearn.metrics import mean_absolute_error as mae
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import acf,pacf
import itertools 


### Naming the app with an icon
img = Image.open('Raeb_lluB.jpg')
st.set_page_config(page_title = "Raeb_lluB",
    page_icon=img)

## Functions for creating the app

### Checks for stationarity and returns the order of differencing required


## The main function

def main():
    st.title("Raeb_lluB")
    st.image("Raeb_lluB.jpg",width=300)
    st.subheader("A Bull Bear Affair")
    nav= st.sidebar.radio("Navigation",["HOME","PREDICTION","PORTFOLIO"])

    ## Loading the data
    uploaded_file = st.file_uploader("File_Upload")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("prices.csv")

    ### Pre-processing
    #rp = pd.read_csv('prices.csv') #raw prices
    rp = data.copy()
    # Pre-Processing
    rp = rp.rename(columns={ rp.columns[1]: "DATE" })
    rp['DATE']=pd.to_datetime(rp['DATE']).dt.date
    rp = rp[['DATE','SYMBOL',' CLOSE_PRICE']]
    processed_prices = rp.pivot_table(index='DATE',columns='SYMBOL',values=' CLOSE_PRICE')
    processed_prices = processed_prices.dropna(axis=1)
    stock_info = pd.read_csv('ind_nifty500list.csv')
    Selected_stocks = stock_info['Symbol'].to_list()
    orignalstocks = processed_prices.columns
    def common_member(a, b):
        result = [i for i in a if i in b]
        return result
    common_stocks = common_member(orignalstocks,Selected_stocks)
    res = [i for i in common_stocks]
    processed_prices = processed_prices[res]
    processed_prices = processed_prices.reset_index() 
    stock_info = stock_info[stock_info['Symbol'].isin(res)]
    
    
    # Moving average prediction Model
    a = st.text_input("Enter for how many days do you want to calculate the Moving Average",value = 10)
    Rolling_Width = int(a)
    # Read processed Stock prices
    #stocks_prices = pd.read_csv('processed_prices.csv')
    stocks_prices = processed_prices.copy()
    stocks_prices_len = stocks_prices.shape[0] -1
    stocks_prices = stocks_prices[stocks_prices['DATE'] == stocks_prices['DATE'][stocks_prices_len]]
    output_df = stocks_prices.transpose()
    date = output_df[stocks_prices_len][0]
    output_df = output_df.iloc[1:]
    output_df.reset_index(inplace = True)
    output_df['Date'] = date
    output_df.columns = ["Symbol", "Price", "Date"]
    #stock_info = pd.read_csv('Stock_info.csv')
    stock_info = stock_info[['Symbol', 'Industry']]
    output_df = output_df.merge(stock_info, on = "Symbol", how = "left")
    output_df = output_df[['Date', 'Symbol', 'Industry', 'Price']]
    ## Adding Moving averages for n days(user input)
    stocks_prices1 = processed_prices.copy()
    #stocks_prices1 = pd.read_csv('processed_prices.csv')

    # Cretaing an empty dataframe to store the Moving average values
    stocks_prices2 = pd.DataFrame()
    stocks_prices2['DATE'] = stocks_prices1['DATE']
    # Setting date as index
    stocks_prices1 = stocks_prices1.set_index('DATE')
    stocks_prices2 = stocks_prices2.set_index('DATE')
    cols = stocks_prices1.columns

    for i in range(stocks_prices1.shape[1]):
        stocks_prices2[cols[i]] = stocks_prices1[cols[i]].rolling(Rolling_Width).mean()

    stocks_prices2.reset_index(inplace = True)
    stocks_prices2 = stocks_prices2.tail(1)
    stocks_prices2 = stocks_prices2.transpose()

    stocks_prices2 = stocks_prices2.iloc[1:]
    stocks_prices2.reset_index(inplace = True)
    stocks_prices2.reset_index(inplace = True,drop = True)
    stocks_prices2.columns = ['Symbol', (str(Rolling_Width)+ '_MA')]

    # # output_df = pd.read_csv('Predicted_Price.csv').drop({'Unnamed: 0'},axis = 1)
    output_df = output_df.merge(stocks_prices2, on = 'Symbol', how = "left")
    output_df.drop(columns = ["Price"],inplace = True)
    output_df.rename(columns = {stocks_prices2.columns[1]:'Price'},inplace =True)
  
    ## Output 
    b = st.text_input("Enter the Amount you want to invest",value = 1000000)
    total_investment= float(b)
    df = output_df.copy()
    df2 = processed_prices.copy()
    df2['date']=df2['DATE']
    df2.drop('DATE',axis=1,inplace=True)
    df2=df2.sort_values(by='date', ascending=True)
    df2 = df2.set_index(['date'])
    numeric_cols = df2.select_dtypes(exclude=['object']).columns
    percentage_return = ((df2[numeric_cols].iloc[-1] - df2[numeric_cols].iloc[0]) / df2[numeric_cols].iloc[-1]) * 100
    df3=percentage_return.to_frame()
    columns=['Symbol','return']
    df3 = df3.reset_index(drop=False)
    df3.columns=['Symbol','return']
    merged_df = pd.merge(df, df3, on='Symbol', how='left')
    df1=merged_df.copy()

    df1=df1[df1['Price']<=0.05*total_investment]
    df1=df1[df1['return']>0]
    industry_groups = df1.groupby('Industry')['Symbol'].count()
    df = df1.sort_values('return', ascending=False)
    df=df[['Symbol', 'Industry', 'Price', 'return']].reset_index()
    df.drop('index',axis=1,inplace=True)
    sectors={key:0 for key in set(df['Industry'])}

    def optimized_constrained_stock_mgmt(df,total_investment):
    
        selected_symbols = []
        solution_df=[]
        sectors={key:0 for key in set(df['Industry'])}
        sector_wise_allocation=0.20*total_investment
        symbol_wise_allocation=0.05*total_investment
        total_price = 0
        for symbol, row in df.iterrows():
            solution={}
            if total_price <=total_investment:
            
                if sectors[row['Industry']]+symbol_wise_allocation<=sector_wise_allocation:
                
                    if total_investment-total_price>=symbol_wise_allocation:
                        #print('HI')
                        num_of_symbols=symbol_wise_allocation//row['Price']
                        sectors[row['Industry']]+=row['Price']*num_of_symbols
            
                        selected_symbols.append((row['Symbol'],symbol_wise_allocation//row['Price']))
                        total_price+=row['Price']*int(symbol_wise_allocation//row['Price'])
                        solution_df.append([row["Symbol"],num_of_symbols,row['Price'],row['Price']*num_of_symbols,row['Industry']])
                    else:
                        #print("by")
                        num_of_symbols=(total_investment-total_price)//row['Price']
                        sectors[row['Industry']]+=row['Price']*num_of_symbols
                        selected_symbols.append((row['Symbol'],num_of_symbols))
                        total_price+=row['Price']*num_of_symbols
                        
                    
                        solution_df.append([row["Symbol"],num_of_symbols,row['Price'],row['Price']*num_of_symbols,row['Industry']])
                else:
                    #print('There')
                    num_of_symbols=(sector_wise_allocation-sectors[row['Industry']])//row["Price"]
                    sectors[row['Industry']]+=row['Price']*num_of_symbols
                    selected_symbols.append((row['Symbol'],num_of_symbols))
                    total_price+=row['Price']*num_of_symbols
                    
                    
                    solution_df.append([row["Symbol"],num_of_symbols,row['Price'],row['Price']*num_of_symbols,row['Industry']])
            
        return pd.DataFrame(solution_df, columns=["symbol",'num_of_symbol','price','amount_invested','sector'])
    
    sol=optimized_constrained_stock_mgmt(df,total_investment)
    #sol['flag']=sol['num_of_symbol']*sol['price']==sol['amount_invested']
    sol=sol[sol['num_of_symbol']>0.1]
    sector = pd.DataFrame(sol.groupby("sector")["amount_invested"].sum()).reset_index().sort_values('amount_invested',ascending =False).reset_index(drop=True)
    



    if nav == "HOME":
        st.subheader("HOME PAGE")
        st.subheader("Imported Data")
        st.write(rp.head())
        st.subheader("Processed Data")
        st.write(processed_prices.tail(5))
        
        l=list(processed_prices.columns)
        del l[0]
        st.subheader("How did you stock look like last year")
        d=st.selectbox("Select your Stock",l)
        st.line_chart(processed_prices,x=processed_prices.columns[0],y=d)
        st.subheader("Compare your stock with any other stock")
        col2=st.multiselect("Select Products for Comparision",l,[l[0],l[2]])
        st.line_chart(processed_prices,x=processed_prices.columns[0],y=col2)





    elif nav == "PREDICTION"  :

        st.header("Your Product Prediction")
        st.write(output_df.head(5))
        
        #col1,col2 = st.columns(2)
        #with col1:
            #st.write("Data used for building the model")
            #st.write(a)
        #with col2:
            #st.write("Data used for testings the model")
            #st.write(b)
        
        st.download_button("Download Output",output_df.to_csv(),file_name = "All_stock_prediction.csv",mime='text/csv')


    else:
        st.subheader("Portfolio Prediction")
        sol = sol.round(decimals=0)
        sector = sector.round(decimals=0)
        st.bar_chart(data=sector, x="sector", y="amount_invested")
        st.write("Stock wise distribution")
        st.write(sol)
        st.write("Sector wise distribution")
        st.write(sector)
         
        st.download_button("Portfolio",sol.to_csv(),file_name = "Portfolio.csv",mime='text/csv')
        st.download_button("Portfolio_Sector",sector.to_csv(),file_name = "Sector.csv",mime='text/csv')



if __name__ == '__main__':
	main()
