"""
Keven Disen
111433335
CSE 332
Lab 3
10/26/2020
"""
import math

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from dash.dependencies import Input, Output
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CSE332Covid.csv')
cat = df.drop(columns=['hospitalizedcurrently', 'pending', 'deathincrease', 'recovered',
                       'death','positiveincrease','negativeincrease', 'positive', 'negative', 'death', 'total', 'num'])

USstate_abb = {
    'AL': 'Alabama',
    'AS': 'American Samoa',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Conneticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Lousiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'MP': 'Northern Mariana Islands',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PW': 'Palau',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VI': 'Virgin Islands',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}

df['state'] = df['state'].map(lambda x: USstate_abb[x] if x in list(USstate_abb.keys()) else x)

df2 = df.drop(columns=['date', 'state', 'num'])

#fill the NaN with 0
df2['death'] = df2['death'].fillna(0)
df2['recovered'] = df2['recovered'].fillna(0)
df2['positive'] = df2['positive'].fillna(0)
df2['pending'] = df2['pending'].fillna(0)
df2['hospitalizedcurrently'] = df2['hospitalizedcurrently'].fillna(0)
df2['pending'] = df2['pending'].fillna(0)


numerical = df2
df3 = df2.drop(columns=['hospitalizedcurrently', 'pending', 'deathincrease', 'recovered'])

#rename columns to fit regular!!!!!!!!!!!!!!!
df3.rename(columns={'deathincrease':'deathInc', 'hospitalizedcurrently':'hospitalized',
                    'negativeincrease':'negInc', 'positiveincrease':'posInc'}, inplace=True)


#correlate
correlate1 = df2.corr()
correlate2 = df3.abs().corr()


#heatmap
heatmap = px.imshow(correlate1, origin='lower', zmin=-1, title="10x10 Correlation Matrix" )

#Scatter
scatter = px.scatter_matrix(df3.abs(),dimensions=['posInc','negative','positive','death', 'total'],
                            height=800, color=df['state'],title='5x5 Scatter Matrix')

#PC (Greatest correlation) -> px.parallel_coordinates
parallel = px.parallel_coordinates(correlate2, dimensions=['posInc','negative','positive','death', 'negInc'], title="Parallel Coordinates")
#last day data was recorded
df5 = df.iloc[8137:8192]
parallel2 = px.parallel_categories(df5, dimensions=['state', 'date'], color='total',
                                  width=1100, height=1700, labels={'state':'States', 'date':'Dates'},
                                   title="Total by State on 7/29/2020")



#PCS and SCREE -> PCA)
pca = PCA(n_components=2)
numerical1 = numerical.iloc[1450:1500] #Only works with 50 points, Data too big
components = pca.fit_transform(numerical1)
fig = px.scatter(components,x=0, y=1, labels={'0':'X Axis', '1':'Y Axis'}, title="PCA Plot")

#Scree
scree = px.bar(numerical1.corr(), labels={'index':'X Axis', 'value':'Y Axis'}, title="Scree Plot")

#BIPLOT
columns = ['negative', 'positive', 'positiveincrease', 'negativeincrease',
           'pending', 'hospitalizedcurrently', 'recovered', 'death', 'deathincrease',
           'total']

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

biplot = px.scatter(components, x=0, y=1, title="Biplot")

for i, feature in enumerate(columns):
    biplot.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    biplot.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )

#MDS
            #using Euclidean Distance
mdseu = MDS(n_components=2, dissimilarity='euclidean', random_state=True, metric=True)
components = mdseu.fit_transform(numerical1)
md = px.scatter(components, x=0, y=1, title="MDS Plot Display of Data using Euclidean Distance")

            #1-|correlation| distance
mdd = MDS(n_components=2, dissimilarity='precomputed', metric=True, random_state=True)
numerical1 = numerical1.corr()
numerical1 = numerical1.abs()
numerical1list = numerical1.values.tolist()
for i in range(len(numerical1list)):
    for j in range(len(numerical1list)):
        numerical1list[i][j] = 1 - numerical1list[i][j]

updatedDf = pd.DataFrame(numerical1list)
updatedDf.columns = numerical1.columns
print(updatedDf)


transformed = mdd.fit_transform(updatedDf)

mds = px.scatter(transformed, x=0, y=1, title="MDS Plot using 1-|correlation| distance",
                 labels={'0':'X axis', '1':'Y axis'})


#Dash code
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1("Keven Disen's CSE 332 Lab 3"),
    html.P("Welcome to my CSE 332 Lab 3 WebServer", style={'text-align':'center'}),

    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Correlation Matrix', value='tab1', children=[
            dcc.Graph(id="heat", figure = heatmap)]),
        dcc.Tab(label='Scatter Matrix', value='tab2', children=[
            dcc.Graph(id='scat', figure=scatter)]),
        dcc.Tab(label='Parallel Coordinates',value='tab3', children=[
            dcc.Graph(id='para', figure=parallel),
            dcc.Graph(id='paracat', figure=parallel2)]),
        dcc.Tab(label='PCA Plot',value='tab5', children=[
            dcc.Graph(id='pca', figure=fig),
            dcc.Graph(id='scree', figure=scree)]),
        dcc.Tab(label='Biplot',value='tab6', children=[
            dcc.Graph(id='bip', figure=biplot)]),
        dcc.Tab(label='MDS Plots',value='tab7', children=[
            dcc.Graph(id='mds', figure=md),
            dcc.Graph(id='mdds', figure=mds)])
    ])])

if __name__ == '__main__':
    app.run_server(debug=True)