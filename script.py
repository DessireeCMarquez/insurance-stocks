
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Insurance Stock Prediction App')

stocks = ('AIG','AON', 'BRK-A','CB','MML','PRU')
selected_stocks=st.selectbox('Select dataset for prediction',stocks)

n_years= st.slider('Years of prediction:',1,4)
period= n_years* 365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    return data

data_load_state=st.text('Load data...')
data= load_data(selected_stocks)
data_load_state.text('Loading data...Done!')


st.subheader('Raw data')
st.write(data.tail())
print(data)


st.subheader('Time series data since 01 January 2015 to actual date')
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open',line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()    

##forecasting
df_train= data[['Date','Close']]
df_train=df_train.rename(columns={'Date':'ds','Close':'y'})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())
print(data.tail())



st.write('Forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Forecast component')
fig2=m.plot_components(forecast)
st.write(fig2)
