#--------------------------------------------------------
# Libraries

# import dash
# import pandas as pd
# import numpy as np
# import plotly.express as px
import plotly.graph_objects as go
# from jupyter_dash import JupyterDash
# import dash_core_components as dcc
# import dash_html_components as html
# import chart_studio.plotly as py
# from plotly.offline import iplot
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

#--------------------------------------------------------
dfname = pd.read_csv('/Users/22danielf/Desktop/VA_Files/uploadedDF.csv')
#--------------------------------------------------------
# App Layout

app.layout = html.Div([


	html.H1("Visualizing Vaccination Trend", style={'text-align': 'center'}),

	dcc.Dropdown(id='locationdropdown_idname',
				options=[
					{"label": "Kent County (Delaware)", "value": "Kent County (Delaware)"},
					{"label": "New Castle County (Delaware)", "value": "New Castle County (Delaware)"},
					{"label": "Sussex County (Delaware)", "value": "Sussex Count (Delaware)"},
					{"label": "Unknown County (Delaware)", "value": "Unknown County (Delaware)"}
				],
				multi=False,
				value = "Kent County (Delaware)",
				style = {'width' : "50%"},
				searchable = True
		),


	html.Br(),

	dcc.Graph(id="linegraph_idname", figure={})

])

#--------------------------------------------------------
# Connect the Plotly Graph with Dash components
@app.callback(
	Output(component_id='linegraph_idname', component_property='figure'),
	Input(component_id='locationdropdown_idname', component_property='value')
)

def update_graph(location_selected):


	df_copy = dfname.copy()
	df_copy = df_copy[df_copy['County_Name'] == location_selected]

	# Plotly Express
	fig = px.line(
		df_copy,
		x="Date",
		y=df_copy.columns[1:3],
		title='Graph  Title'
	)

	return fig

#--------------------------------------------------------
# Run
if __name__ == '__main__':
    app.run_server(debug=True)