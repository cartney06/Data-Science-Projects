import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns # Using seaborn for visualization
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

### COVID-19 Outcomes by Testing Cohorts: Cases, Hospitalizations, and Deaths
"""This will be the dataset to develop the overall NYC timeseries for Covid positivity"""

#### Import Data

chd = pd.read_csv('https://data.cityofnewyork.us/resource/cwmx-mvra.csv')

#### Data Cleanup
# convert specimen_date to dt
chd['specimen_date_dt'] = [pd.to_datetime(row).strftime('%Y-%m-%d') for row in chd['specimen_date']]
chd['dt'] = [pd.to_datetime(row).strftime('%Y-%m-%d') for row in chd['extract_date']]

# Add NYC data label to a new column called Boro
chd['boro'] = 'NYC'

# Get max dt of data load
max_date = chd['dt'].max()
chd = chd[(chd['dt'] == chd['dt'].max())]

# Sort DF by specimen_date
chd.sort_values(by=['specimen_date'], inplace=True)
chd.reset_index(inplace=True, drop=True)

# Add % Tested Positive
chd['pct_tested_positive'] = chd['number_confirmed']/chd['number_tested']


# Add Running Total of Tests, Confirmed Tests & Deaths
chd['cumsum_number_tested'] = chd['number_tested'].cumsum()
chd['cumsum_number_confirmed'] = chd['number_confirmed'].cumsum()
chd['cumsum_number_deaths'] = chd['number_deaths'].cumsum()
chd['cumsum_number_hospitalized'] = chd['number_hospitalized'].cumsum()

# Add Percentage Total of Tests Confirmed Tests Cumulative
chd['cumsum_pct_tested_positive'] = chd['cumsum_number_confirmed']/chd['cumsum_number_tested']

# Add Rolling 7-Day Moving Averages of Positve Test Percentage
chd['7-day_rolling_avg_pct_tested_positive'] = chd['number_confirmed'].rolling(7).sum()/chd['number_tested'].rolling(7).sum()

# Store topline numbers in variables
total_cases = format(chd['number_confirmed'].sum(), ',')
total_deaths = format(chd['number_deaths'].sum(), ',')
total_hospitalizations = format(chd['number_hospitalized'].sum(), ',')
total_tests = format(chd['number_tested'].sum(), ',')

topline_values = ['Cases', 'Tests', 'Hospitalizations', 'Deaths'] 
topline_data = (total_cases, total_deaths, total_hospitalizations, total_tests)

# Store in new dataframe
df = pd.DataFrame(data=topline_data).T
df.columns = topline_values

# Export Data to Local Folder
chd.to_csv('data_output/chd.csv', index=False)

#### Build Dashboard

# external CSS stylesheets
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://raw.githubusercontent.com/plotly/dash-sample-apps/master/apps/dash-web-trader/assets/style.css']


# Sheet styling/configuration and formatting
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}

DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
    "#188463",
    "#157658",
    "#11684d",
    "#10523e",
]

DEFAULT_OPACITY = 0.8

# Create a table to store topline information within the layout
def generate_table(dataframe, max_rows=2):
    return html.Div(children=
    [
        html.H3(
            html.Table([
        html.Thead(
            html.Tr([html.Th(col, style={"text-align":"center"}, ) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col], style={"text-align":"center"}) for col in dataframe.columns
                ], style={'border-style':'hidden'}) for i in range(min(len(dataframe), max_rows))
            ])
        ], style={"margin-left":"auto", 'margin-right':'auto'},)
    )
    ])

##### App Layout
app = dash.Dash(__name__, title='NYC COVID-19 Dashboard', external_stylesheets=external_stylesheets)
app.renderer = 'var renderer = new DashRenderer();'
app.layout = html.Div( children=[
    html.H1("Time Series and Demographic Analysis of COVID-19 in New York City"),
    html.H2("Reporting Dashboard"),
    html.Hr(),
    html.H4("New York City Totals"),
    html.P("As of "+max_date),

    # HTML table generated above here
    html.Div(generate_table(df), style={"text-align":"center", 'color':colors['text']}),
    html.Hr(),
    html.Br(),



    dcc.Dropdown(id="select_metric",
                options=[
                    {"label": "Confirmed Cases", "value":'number_confirmed'},
                    {"label": "Hospitalizations", "value":'number_hospitalized'},
                    {"label": "Tested", "value":'number_tested'},
                    {"label": "Deaths", "value":'number_deaths'}],
                 value= "number_confirmed",
                 multi=False,
                 style={"width":"40%"}
                ),
    html.Div(id="output_container", children=[]),
    html.Br(),
    dcc.Graph(id="covid_19_chart", figure={}),
    
    html.P("Source: NYC Department of Health, "+ max_date, style={'font-size':'small', 'text-align':'left', 'font-style':'italic', 'color':colors['text']})

])



# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id='covid_19_chart', component_property='figure'),
    [Input(component_id='select_metric', component_property='value')]
)

def update_graph(metric_selected):
    print(metric_selected)
    print(type(metric_selected))

    # Make copy of dataframe
    dff = chd.copy()

    # Filter dataframe copy for boro selection
    #dff = dff[(dff['boro'] == boro_selected)]

    # Filter dataframe copy for when 7-day average is not null
    dff = dff[(dff['7-day_rolling_avg_pct_tested_positive'].notnull())]

    # Now convert column names to output display values
    dff_dict = {'number_confirmed':'Confirmed Cases',
    'number_hospitalized':'Hospitalizations',
    'number_tested':'Tested',
    'number_deaths':'Deaths'}

    
    # Now use plotly to create chart/graph

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=dff['specimen_date_dt'], y=dff[metric_selected], name=dff_dict[metric_selected], marker=dict(color='yellow', opacity=0.6)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dff['specimen_date_dt'], y=dff['7-day_rolling_avg_pct_tested_positive'], name="7-Day Postive Test Rate Avg", mode='lines+markers', line=dict(width=1), marker=dict(size=4) ),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title=dict(text="COVID-19 "+ dff_dict[metric_selected] + " by Day", x=0.5)
    ) 

    # Set x-axis title
    fig.update_xaxes(title_text="Date", showline=False)

    # Set y-axes titles and hoverformat
    fig.update_yaxes(title_text=dff_dict[metric_selected], showgrid=False, secondary_y=False, showticklabels=True, type='linear', tickformat=',')
    fig.update_yaxes(hoverformat=",.2%", showgrid=False, secondary_y=True, showticklabels=False, zeroline=False)
   
    # Change theme, remove grid, format hover labels, and center legend
    fig.update_layout(hovermode='x unified', hoverlabel=dict(namelength=35), xaxis=dict(visible=True, zeroline=False), yaxis=dict(showgrid=False), autosize=True, template='plotly_dark', legend=dict(yanchor='bottom',x=0.5, xanchor='center', orientation="h", y=1.02))
    fig.update_layout(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)













