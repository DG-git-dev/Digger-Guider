import qlib
import pandas as pd
import os
from qlib.contrib.evaluate import risk_analysis, backtest
from qlib.contrib.strategy.strategy import TopkDropoutStrategy
from tqdm import tqdm
qlib.init()


input_path = '/data1/v-liuyan/project/D_G/out/kdd21/'
r_box = []

STRATEGY_CONFIG = {
    "topk": 50,
    "n_drop": 5,
}
BACKTEST_CONFIG = {
    "verbose": False,
    "limit_threshold": 0.095,
    "account": 100000000,
    "benchmark": "SH000300",
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}
# use default strategy
strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
out_file = 'trade_sim_20210127_ndrop_5'
filenames = ['pred_mlp_2',
 'pred_cnn_v2_2',
 'pred_alstm_2',
 'pred_cnn_rnn_v2_1',
 'pred_arima_2',
 'pred_linear_2',
 'pred_adv_alstm_2',
 'pred_rnn_v2_2',
 'pred_sfm_6',
 'pred_rnn_v1_2',
 'pred_transformer_2',
 'pred_ensemble_v50fine']

for filename in tqdm(filenames):

    strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
    pred_test = pd.read_pickle(input_path + filename +'.pkl')
    pred_test= pred_test.reset_index('datetime')
    pred_test= pred_test.reset_index('instrument')
    pred_test = pred_test.set_index(['instrument','datetime'])
    pred_test = pd.DataFrame(index=pred_test.index, data=pred_test['score'].values,columns=['score'])
    
    
    print(pred_test.head())
    report, positions = backtest(pred_test, strategy=strategy, **BACKTEST_CONFIG)
    r = (report['return'] - report['cost']).dropna()
    r_box.append(r)
    print(r.head())
df = pd.concat(r_box,axis=1)
df.columns = filenames
df['CSI300'] = report['bench']
df.to_csv(out_file + '.csv')


import plotly.offline as of
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv(out_file + '.csv')
of.offline.init_notebook_mode(connected=True)
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df.head()

def plot(df):
    trace0 = go.Scatter(
        x=df.index,
        y=(df['CSI300'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        marker=dict(color="#9D755D"),
        # marker=dict(color="#FF97FF")
        name='Benchmark'
    )

    trace1_1 = go.Scatter(
        x=df.index,
        y=(df['pred_arima_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        marker=dict(color="#FF9DA6"),
        name='ARIMA'
    )

    trace1_2 = go.Scatter(
        x=df.index,
        y=(df['pred_linear_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#FFA15A"),
        name='Linear Regression'
    )

    trace2_1 = go.Scatter(
        x=df.index,
        y=(df['pred_mlp_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#AB63FA"),
        name='MLP'
    )

    trace2_2 = go.Scatter(
        x=df.index,
        y=(df['pred_rnn_v1_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#19D3F3"),
        name='Daily RNN'
    )

    trace2_3 = go.Scatter(
        x=df.index,
        y=(df['pred_sfm_6'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#EF553B"),
        name='SFM'
    )

    trace2_4 = go.Scatter(
        x=df.index,
        y=(df['pred_alstm_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#FF97FF"),
        name='ALSTM'
    )    

    trace2_5 = go.Scatter(
        x=df.index,
        y=(df['pred_adv_alstm_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        marker=dict(color="#BCBD22"),
        name='Adv-ALSTM'
    )   
    trace2_6 = go.Scatter(
        x=df.index,
        y=(df['pred_transformer_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        marker=dict(color="#FF97FF"),
        name='Transformer'
    ) 

    trace3_1 = go.Scatter(
        x=df.index,
        y=(df['pred_cnn_v2_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="FECB52"),
        name='Coarse-grained Model'
    )

    trace3_2 = go.Scatter(
        x=df.index,
        y=(df['pred_rnn_v2_2'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        # marker=dict(color="#BAB0AC"),
        name='Fine-grained Model'
    )

    trace3_3 = go.Scatter(
        x=df.index,
        y=(df['pred_ensemble_v50fine'] + 1).cumprod().values,
        line=dict(width=4),
        marker=dict(color="#B6E880"),
        mode='lines',
        name='Ensemble Model'
    )

    trace3_4 = go.Scatter(
        x=df.index,
        y=(df['pred_cnn_rnn_v2_1'] + 1).cumprod().values,
        line=dict(width=4),
        mode='lines',
        marker=dict(color="#EF553B"),
        name='Digger-Guider'
    )


    layout = go.Layout(
        # margin=go.Margin(l=0,r=0,b=0,t=0,pad=0),
        plot_bgcolor='#E6E6FA',
        yaxis=dict(title='Cumulative Profit',
                   tickfont=dict(size=34)),
        xaxis=dict(title='Date',
                   tickfont=dict(size=34),
                   tickangle=30,
                   range=["2016-01", "2020-06"],
                   tickvals=["2016-01", "2016-04", "2016-07", "2016-10", "2017-01", "2017-04", "2017-07", "2017-10",\
                    "2018-01", "2018-04",  "2018-07", "2018-10", "2019-01", "2019-04",  "2019-07", "2019-10",\
                    "2020-01","2020-04","2020-06"]),
        xaxis_tickformat = '%b %Y',
              font=dict(size=34, family="Times New Roman"),
        # legend=dict(x=0.01, y=0.98,
        #             font=dict(size=32),bgcolor="rgba(255,255,255,0.5)"),
        legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
        font=dict(size=32),
        bgcolor="rgba(255,255,255,0.5)"),
        autosize=False, width=2000, height=1000
    )
    data = [trace0, trace1_1, trace1_2, trace2_1, trace2_2, trace2_3,
        trace2_4, trace2_5, trace2_6, trace3_1, trace3_2, trace3_3, trace3_4]
    fig = go.Figure(data=data, layout=layout)
    fig.write_image(out_file+".pdf",
       width=2000,
       height=1000)
plot(df)