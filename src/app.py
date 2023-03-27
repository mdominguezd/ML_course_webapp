#!/usr/bin/env python
# coding: utf-8
import dash
from dash import Dash, dcc, html, callback, callback_context, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import random
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import geopandas as gpd
import pandas as pd

import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor

from Functions import read_data, fit_regression, get_metrics, get_error_n_predicted_GeoJSON, read_data_cls, open_raster, calculate_ndvi, data_clean_up, train_RFC, normalize_images, apply_RFC, conf_matrix, read_data_clu, my_kmeans

# Read the data that will be used
X_train, X_test, y_train, y_test = read_data()
im1, im2, im1gt, im2gt = read_data_cls()
data_cities, regions, feature_names_clu = read_data_clu()
# CLean up dat for assignment 2
X_tr, X_te, y_tr, y_te = data_clean_up(im1, im2, im1gt, im2gt)
# To avoid reading the geojson everytime we update the map
nuts2 = gpd.read_file('https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2016_4326_LEVL_2.geojson')

seed = 123
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

server = app.server

load_figure_template(["slate"])
color = 'darkgreen'
font_s = {'font-family' : 'bahnschrift'}

# Feature names for classification
feature_names = [
                 'Red',
                 'Green',
                 'Blue',
                 'NIR',
                 'NDVI',
                 'NDVI_loc_avg',
                 'Red_loc_avg',
                 'Green_loc_avg',
                 'Blue_loc_avg',
                 'NIR_loc_avg',
                 'EF1',
                 'EF2',
                 'EF3',
                 'EF4',
                 'EF5',
                 'EF6',
                 'EF7',
                 'EF8']
feature_names_keep = ['Blue',
                 'NDVI',
                 'NDVI_loc_avg',
                 'Red_loc_avg',
                 'Green_loc_avg',
                 'Blue_loc_avg',
                 'NIR_loc_avg',
                 'EF1',
                 'EF2',
                 'EF4',
                 'EF5',
                 'EF6',
                 'EF7',
                 'EF8']

# Colors for classes
class_colors = [[0,'white'],[1/6,'gray'],[2/6,'black'],[3/6,'darkgreen'],[4/6,'lightgreen'],[5/6,'goldenrod'],[6/6,'blue']]

app.layout = html.Div([html.H1('Machine Learning Dashbord', style=font_s),
                       html.Div([''], style = {'height':15, 'width':1900, 'background-color':color}),
                       html.Div([html.Div([' '], style = {'width':15}),
                                 html.Div([html.Div([' '], style = {'height':20}),
                                           html.H2('Regression algorithms', style = font_s),
                                           html.H3('Choose a model:', style = font_s),
                                           dcc.Dropdown(['Linear Regression', 'KNN Regression', 'Random Forest Regression'],
                                                        'Linear Regression',
                                                        id = 'Model',
                                                        style={'font-family' : 'bahnschrift',
                                                               'width':440, 
                                                               'color': 'black', 
                                                               'background-color':'lightgray'}),
                                           html.H4('Additional parameters:', style = font_s),
                                           html.Div([html.B('   K=', style = {'font-family' : 'bahnschrift','width':440}),
                                                     daq.NumericInput(min=1,
                                                                      max=30,
                                                                      value=3,
                                                                      style = {'font-family' : 'bahnschrift'},
                                                                      id='K')], 
                                                    style={'display':'flex', 'width':440}),
                                           html.Div([html.B('   Number of trees =', style = {'font-family' : 'bahnschrift','width':440}),
                                                     daq.NumericInput(min=20,
                                                                      max=300,
                                                                      value=100,
                                                                      style = {'font-family' : 'bahnschrift'},
                                                                      id='T')], 
                                                    style={'display':'flex', 'width':440}),
                                           html.Div([html.B('   Max depth of trees =', style = {'font-family' : 'bahnschrift','width':440}),
                                                     daq.NumericInput(min=1,
                                                                      max=10,
                                                                      value=None,
                                                                      style = {'font-family' : 'bahnschrift'},
                                                                      id='D')], 
                                                    style={'display':'flex', 'width':440}),
                                           html.Div([html.B('   YEAR=', style = {'font-family' : 'bahnschrift','width':440}),
                                                     daq.NumericInput(min=2012,
                                                                      max=2018,
                                                                      value=2012,
                                                                      style = {'font-family' : 'bahnschrift'},
                                                                      id='YEAR')], 
                                                    style={'display':'flex', 'width':440}),
                                           html.H5('Metrics:', style = font_s),
                                           dash_table.DataTable(id= 'metrics_table', 
                                                                style_header={'backgroundColor': color, 'color':'lightgray','fontWeight': 'bold'},
                                                                style_cell={'textAlign': 'center', 'backgroundColor':'lightgray', 'color':'black'},
                                                                style_table={'width':440}, cell_selectable = False, 
                                                                style_as_list_view=True)]),
                                 html.Div([' '], style = {'width':15}),
                                 html.Div([' '], style = {'width':20,'background-color':color}),
                                 html.Div([' '], style = {'width':15}),
                                 html.Div([html.Div([html.H3('Crop yield predictions', style = font_s),
                                                     dcc.Loading(id = 'loading2',
                                                                 children = [dcc.Graph(id="Yield_Pred",
                                                                                       style = {'width':650, 'height' : 800})]),
                                                    ]),
                                           html.Div([' '], style = {'width':70}),
                                           html.Div([html.H3('Error estimations', style = font_s),
                                                     dcc.Loading(id = 'loading3',
                                                                 children = [dcc.Graph(id="Error",
                                                                                       style = {'width':650, 'height' : 800})])
                                                    ], style = {'height':15})
                                           ], style = {'display':'flex', 'width': 1370, 'height' : 800}
                                         )],
                                style={'display':'flex', 'width':1900, 'height':845, 'overflow':'auto'}),
                       html.Div([''], style = {'height':20, 'background-color':color}),
#                        html.Div([html.Div([' '], style = {'width':15}),
#                                  html.Div([html.Div([' '], style = {'height':20}),
#                                            html.H2('Classification algorithm', style = font_s),
#                                            html.H3('Feature selection:'),
#                                            dcc.Dropdown(options=feature_names,
#                                                         value = feature_names_keep,
#                                                         multi = True,
#                                                         id='features_cl',
#                                                         style = {'width':440, 'color' : color}),
#                                            html.H3('Choose the parameters:', style = font_s),
#                                            html.Div([html.B('   Number of trees =', style = {'font-family' : 'bahnschrift','width':320}),
#                                                      dcc.Dropdown(options=[10, 25, 50, 100],
#                                                                   value = 25,
#                                                                   style = {'font-family' : 'bahnschrift','width':120},
#                                                                   id = 'cl_T')], 
#                                                     style={'display':'flex', 'width':440}),
#                                            html.Div([html.B('   Minimum number of leaf samples =', style = {'font-family' : 'bahnschrift','width':440}),
#                                                      dcc.Dropdown(options=[10, 25, 50, 100],
#                                                                   value = 25,
#                                                                   style = {'font-family' : 'bahnschrift','width':120},
#                                                                   id = 'cl_L')],
#                                                     style={'display':'flex', 'width':440}),
#                                            html.Div([html.B('   RUN', style = {'font-family' : 'bahnschrift','width':440}),
#                                                      daq.BooleanSwitch(id = 'RUN', theme = 'dark', color = color, on=True)],
#                                                     style={'display':'flex', 'width':440}),
                                           
#                                            dcc.Loading(id = 'loading4', 
#                                                        children = [html.H5('Metrics:', style = font_s),
#                                                                    dash_table.DataTable(id= 'acc_table', style_header={'backgroundColor': color, 'color':'lightgray','fontWeight': 'bold'},
#                                                                                         style_cell={'textAlign': 'center', 'backgroundColor':'lightgray', 'color':'black'},
#                                                                                         style_table={'width':440}, cell_selectable = False, style_as_list_view=True),
#                                                                    html.H5('Confusion matrix:', style = font_s),
#                                                                    dcc.Graph(id = 'conf_mat', style = {'width' : 440})],
#                                                        type = 'default'
#                                                       )
#                                           ]),
#                                  html.Div([' '], style = {'width':15}),
#                                  html.Div([' '], style = {'width':20,'background-color':color}),
#                                  html.Div([' '], style = {'width':15}),
#                                  html.Div([html.Div([html.H3('True image', style = {'font-family':'bahnschrift','width' : 475}),
#                                                      html.H3('True classes', style = {'font-family':'bahnschrift','width' : 425}),
#                                                      html.H3('Classified image', style = font_s),
#                                                     ], style = {'display':'flex', 'width': 1370}),
#                                            dcc.Loading(id = 'Loading', 
#                                                        children = [html.Div([dcc.Graph(id = 'True_img', style = {'width':420, 'height' : 800},),
#                                                                              dcc.Graph(id="Classified_img", style = {'width':950, 'height' : 800})],
#                                                                             style = {'display':'flex', 'height':800})], type = 'default')
#                                           ], 
#                                          ),
#                                 ], style = {'display':'flex'}),
                       # html.Div([''], style = {'height':20, 'background-color':color}),
                       html.Div([html.Div([' '], style = {'width':15}),
                                 html.Div([html.Div([' '], style = {'height':20}),
                                           html.H2('Clustering algorithm', style = font_s),
                                           html.H3('Choose the paramters:'),
                                           html.Div([html.B('   K =', style = {'font-family' : 'bahnschrift','width':440}),
                                                     daq.NumericInput(id = 'K-means',
                                                                      value = 4, 
                                                                      min = 2,
                                                                      max = 6)], 
                                                    style={'display':'flex', 'width':440}),
                                          ]),
                                 html.Div([' '], style = {'width':15}),
                                 html.Div([' '], style = {'width':20,'background-color':color}),
                                 html.Div([' '], style = {'width':15}),
                                 html.Div([dcc.Loading(id='ldg',
                                                       children = [dcc.Graph(id = 'pie', style = {'width':1370}),
                                                                   dcc.Graph(id = 'bar', style = {'width':1370})
                                                                  ])])
                                ], style = {'display':'flex'}),
], style = {'overflow':'auto', 'height':1000})


@app.callback(
    [Output('metrics_table', 'data'),Output('K', 'disabled'), Output('T', 'disabled'), Output('D', 'disabled'), Output('Yield_Pred', 'figure'), Output('Error','figure')],
    [Input('Model','value'),Input('K','value'), Input('T','value'), Input('D','value'), Input('YEAR','value')]
)

def update_metrics_n_map(model, K, T, D, year):
    """
        Objective: Take the model, the parameters and the year selected apply the regression and return the estimations as maps
    """
    if model == 'Linear Regression':
        K_not_needed = True
        T_not_needed = True
        D_not_needed = True
        
        model, y_pred = fit_regression('LR', X_train, X_test, y_train, y_test)
        df, df_e = get_error_n_predicted_GeoJSON(nuts2, 'LR', year)
        
    elif model == 'KNN Regression':
        K_not_needed = False
        T_not_needed = True
        D_not_needed = True
        
        model, y_pred = fit_regression('KNN', X_train, X_test, y_train, y_test, K)
        df, df_e = get_error_n_predicted_GeoJSON(nuts2, 'KNN', year, K)
        
    elif model == 'Random Forest Regression':
        K_not_needed = True
        T_not_needed = False
        D_not_needed = False
        
        model, y_pred = fit_regression('RFR', X_train, X_test, y_train, y_test,T = T,D = D)
        df, df_e = get_error_n_predicted_GeoJSON(nuts2, 'RFR', year,T = T,D = D)
    
    fig = px.choropleth_mapbox(df, 
                               geojson=df.geometry, 
                               locations = df.index,
                               color='est_crop_yield',
                               color_continuous_scale="greens",
                               range_color=(35, 55),
                               mapbox_style="carto-positron",
                               hover_data = ['NAME_LATN','est_crop_yield'],
                               zoom=5.8, center = {"lat": 52.05, "lon": 5.67}
                              )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
    fig_e = px.choropleth_mapbox(df_e, 
                                 geojson=df_e.geometry, 
                                 locations = df_e.index,
                                 color='Error',
                                 color_continuous_scale="RdBu_r",
                                 range_color=(-12, 12),
                                 mapbox_style="carto-positron",
                                 hover_data = ['NAME_LATN','Error'],
                                 zoom=5.8, center = {"lat": 52.05, "lon": 5.67}
                                )
    
    fig_e.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return get_metrics(y_test,y_pred), K_not_needed, T_not_needed, D_not_needed, fig, fig_e

@app.callback(
    [Output('acc_table', 'data'), Output('True_img','figure'), Output('Classified_img','figure'), Output('conf_mat', 'figure')],
    [Input('features_cl','value'), Input('cl_T','value'), Input('cl_L','value'), Input('RUN','on')]
)
def update_Cls(features, Trees, min_Leaf, on):
    """
        Objective: Take the fatures used and the parameters selected, fit a classification algorithm, aply it to a test dataset and return th results as images.
    """
    global X_tr, X_te, y_tr, y_te, feature_names, im2, im2gt, class_colors
    
    if not on:
        raise PreventUpdate
    else:
        ind = [i in features for i in feature_names]
        
        X_tr_new = X_tr[:,ind]
        X_te_new = X_te[:,ind]
        
        print('Training model...')
        
        mod, t, preds, acc = train_RFC(X_tr_new, X_te_new, y_tr, y_te, n_trees = Trees, min_samples=min_Leaf)
        
        print('\nDone\n\nApplying model to test set...')

        classified_img = apply_RFC(mod, im2[ind,:,:], im2gt)
        
        print('Done')

        table = pd.DataFrame([acc, t], index = ['Test accuracy', 'Time taken [s]']).T.apply(lambda df:round(df,2))
        table = table.to_dict('records')

        classified_img[im2gt[0] == 0]  = 0

        im = np.swapaxes(im2, 0, 2)
        im = np.swapaxes(im, 0, 1)

        im = (im - np.min(im[:,:,:3]))/(np.max(im[:,:,:3])-np.min(im[:,:,:3]))

        fig_img = px.imshow(im[:,:,:3])

        fig_pred = px.imshow(np.array([im2gt[0,:,:],classified_img]),
                             facet_col=0, 
                             color_continuous_scale=class_colors, 
                             zmin = 0,
                             zmax = 6)
    
        conf_mat = px.imshow(conf_matrix(y_te, preds),
                             x = ['roads', 'buildings', 'trees', ' grass', 'water'],
                             y = ['roads', 'buildings', 'trees', ' grass', 'water'],
                             color_continuous_scale=[[0,'white'],[1,'black']]
                            )
    
    return table, fig_img, fig_pred, conf_mat

@app.callback(
    [Output('pie','figure'),Output('bar','figure')],
    Input('K-means','value')
)
def update_clus(K):
    
    global regions, data_cities, feature_names_clu
    
    cols = ['darkred' , 'darkblue', 'indianred', 'darkgreen', 'lightseagreen', 'red', 'goldenrod','tomato', 'orange']
    regs = ['Central Europe', 'East Asia', 'Eastern Europe', 'Latin America','North America', 'Northern Europe', 'Oceania', 'Southern Europe','Western Europe']

    cols_dict = dict(zip(regs, cols))
    
    cluster_assignments, cluster_centroids = my_kmeans(data_cities,K=K)
    
    pie = make_subplots(rows=1, cols=K, specs=[[{"type": "domain"} for i in range(K)]])
    bar = make_subplots(rows=1, cols=K)
    
    for i in range(K):
        reg, count = np.unique(regions[cluster_assignments == i], return_counts=True)

        labels = [ (reg[j] + '\n' + str(round(100*count[j]/np.sum(count), 0)) + '%' if 100*count[j]/np.sum(count) > 2.5 else ' ') for j in range(len(reg))]
        cols = [cols_dict[k] for k in reg]

        pie.add_trace(
            go.Pie(values = count, labels = reg, hole = 0.6, title = 'Cluster = '+str(i+1)),
            row = 1,
            col = i+1
        )
        
        bar.add_trace(
            go.Bar(x = feature_names_clu, y = cluster_centroids[i,:]),
            row = 1,
            col = i+1
        )

    
    return pie, bar

if __name__ == '__main__':
    app.run_server(port = 8080)


