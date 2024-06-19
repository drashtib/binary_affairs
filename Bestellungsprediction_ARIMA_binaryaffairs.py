import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import random
import string

# Set random seed for reproducibility
np.random.seed(42)

# Define the date range
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame(index=dates)

# Generate unique IDs
def generate_uid():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))

# Paints count
paints_counts = {'holz': 355, 'metall': 487, 'kunststoff': 27, 'andere': 1275 - 355 - 487 - 27}

# Create 25 products with shared unique IDs but different categories
product_categories = ['holz', 'metall', 'kunststoff', 'andere']
unique_ids = [generate_uid() for _ in range(25)]
products = {f'{category}_{uid}': paints_counts[category] for uid in unique_ids for category in product_categories}

# Generate realistic demand data with specific seasonal patterns
def generate_demand(dates, category, proportion):
    base_demand = 5 * proportion  # Base demand proportional to initial stock, scaled down to prevent large lambda
    weekly_pattern = 1 + 0.1 * np.sin(2 * np.pi * dates.dayofweek / 7)
    
    if category == 'holz':
        monthly_pattern = np.where(dates.month.isin([6, 7, 8]), 1.5, 1)  # 50% increase in summer
    elif category == 'metall':
        monthly_pattern = np.where(dates.month.isin([11, 12, 1, 2]), 1.75, 1)  # 75% increase in winter
    else:
        monthly_pattern = np.ones(len(dates))  # No significant seasonality for other categories
    
    demand = base_demand * weekly_pattern * monthly_pattern
    demand = np.clip(demand, a_min=0, a_max=None)  # Ensure no negative demand
    return demand

# Define initial stock for warehouses
initial_stock = {'A': 576, 'B': 1275}

# Define base incoming and outgoing paints
base_incoming = 25
base_outgoing = 43

# Generate warehouse data
warehouses = ['A', 'B']
warehouse_data = {warehouse: data.copy() for warehouse in warehouses}

for warehouse, df in warehouse_data.items():
    df['eingehende_farben'] = np.random.poisson(lam=base_incoming, size=len(df))
    df['ausgehende_farben'] = np.random.poisson(lam=base_outgoing, size=len(df))
    
    # Collect data in a dictionary
    columns_data = {}
    
    for product, initial_stock in products.items():
        category, uid = product.split('_')
        proportion = initial_stock / sum(paints_counts.values())
        
        # Generate demand with specified seasonal patterns
        columns_data[f'{product}_Bestellung'] = generate_demand(dates, category, proportion)
        
        # Generate sales based on demand
        columns_data[f'{product}_verkauf'] = np.random.poisson(lam=np.clip(columns_data[f'{product}_Bestellung'], 0, 5))
        
        # Calculate stock levels
        columns_data[f'{product}_bestand'] = initial_stock - np.cumsum(columns_data[f'{product}_verkauf'])
        
        # Ensure varnishes are not kept in storage for more than 3 months
        if category == 'lack':
            columns_data[f'{product}_bestand'] = pd.Series(columns_data[f'{product}_bestand']).shift(90).fillna(0).values

    # Convert columns_data dictionary to DataFrame and concatenate with existing DataFrame
    columns_df = pd.DataFrame(columns_data, index=df.index)
    warehouse_data[warehouse] = pd.concat([df, columns_df], axis=1)

# Adjust stock levels for usage lifecycle
for warehouse, df in warehouse_data.items():
    for product, initial_stock in products.items():
        category, uid = product.split('_')
        proportion = initial_stock / sum(paints_counts.values())
        
        # 25% of paints used within the first 4 weeks
        usage_within_4_weeks = 0.25 * df[f'{product}_bestand'].rolling(window=28).sum().fillna(0)
        
        # Usage lifecycle of 8 days
        usage_lifecycle = usage_within_4_weeks / 8
        
        # Adjust stock levels
        df[f'{product}_bestand'] -= usage_lifecycle

# Create a Dash application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the dashboard
app.layout = dbc.Container([
    html.H1("Lacke Bestellung Prediction Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Lager auswählen"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='warehouse-dropdown',
                        options=[{'label': f'Lager {warehouse}', 'value': warehouse} for warehouse in warehouses],
                        value='A'
                    )
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Produkttyp auswählen"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='product-dropdown',
                        options=[{'label': category.capitalize(), 'value': category} for category in product_categories],
                        value='holz'
                    )
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Produkt-ID auswählen"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='uid-dropdown',
                        options=[],
                        value=None
                    )
                ])
            ])
        ], width=4)
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='stock-forecast-graph')
                ])
            ])
        ])
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='available-stock-graph')
                ])
            ])
        ])
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='actual-vs-predicted-graph')
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='demand-forecast-graph')
                ])
            ])
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='demand-vs-actual-graph')
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fehlermetriken"),
                    html.P(id='mae'),
                    html.P(id='mse'),
                    html.P(id='mape'),
                    html.H4("Modelleffizienz"),
                    html.P(id='efficiency')
                ])
            ])
        ])
    ])
], fluid=True)

# Callback to update UID dropdown based on selected product type
@app.callback(
    Output('uid-dropdown', 'options'),
    Input('product-dropdown', 'value')
)
def set_uid_options(selected_product):
    filtered_uids = [uid for uid in unique_ids if f'{selected_product}_{uid}' in products]
    return [{'label': uid, 'value': uid} for uid in filtered_uids]

# Callback to update UID dropdown value
@app.callback(
    Output('uid-dropdown', 'value'),
    Input('uid-dropdown', 'options')
)
def set_uid_value(available_options):
    if available_options:
        return available_options[0]['value']
    return None

# Callback to update the graphs based on the selected product type and unique ID
@app.callback(
    [Output('stock-forecast-graph', 'figure'), 
     Output('available-stock-graph', 'figure'), 
     Output('actual-vs-predicted-graph', 'figure'),
     Output('demand-forecast-graph', 'figure'),
     Output('demand-vs-actual-graph', 'figure'),
     Output('mae', 'children'),
     Output('mse', 'children'),
     Output('mape', 'children'),
     Output('efficiency', 'children')],
    [Input('warehouse-dropdown', 'value'), 
     Input('product-dropdown', 'value'), 
     Input('uid-dropdown', 'value')]
)
def update_graph(selected_warehouse, selected_product, selected_uid):
    try:
        print("Callback started")
        df = warehouse_data[selected_warehouse]
        print(f"DataFrame for warehouse {selected_warehouse} loaded successfully")
        
        if selected_uid is None:
            # Aggregated data for selected product type
            product_columns = [f'{selected_product}_{uid}_bestand' for uid in unique_ids if f'{selected_product}_{uid}' in products]
            demand_columns = [f'{selected_product}_{uid}_Bestellung' for uid in unique_ids if f'{selected_product}_{uid}' in products]
            df['total_stock'] = df[product_columns].sum(axis=1)
            df['total_demand'] = df[demand_columns].sum(axis=1)
            title = f'{selected_product.capitalize()} Lagerbestand in Lager {selected_warehouse}'
        else:
            # Data for selected product with unique ID
            product_key = f'{selected_product}_{selected_uid}'
            df['total_stock'] = df[f'{product_key}_bestand']
            df['total_demand'] = df[f'{product_key}_Bestellung']
            title = f'{selected_product.capitalize()} Lagerbestand ({selected_uid}) in Lager {selected_warehouse}'
        
        print(f"Title set to: {title}")
        
        # Split data into training and test sets
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        
        # Train the model on the training set for stock
        model_stock = auto_arima(train['total_stock'], seasonal=True, m=12, trace=False, suppress_warnings=True, stepwise=True)
        
        # Forecast on the test set for stock
        forecast_test_stock = model_stock.predict(n_periods=len(test))
        forecast_index_test_stock = test.index
        forecast_series_test_stock = pd.Series(forecast_test_stock, index=forecast_index_test_stock)
        
        # Forecast for the next 6 months for stock
        model_stock.update(df['total_stock'])
        forecast_stock = model_stock.predict(n_periods=180)
        forecast_index_stock = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')
        forecast_series_stock = pd.Series(forecast_stock, index=forecast_index_stock)
        
        # Calculate error metrics for stock
        mae_stock = mean_absolute_error(test['total_stock'], forecast_series_test_stock)
        mse_stock = mean_squared_error(test['total_stock'], forecast_series_test_stock)
        mape_stock = mean_absolute_percentage_error(test['total_stock'], forecast_series_test_stock)
        
        print(f"Stock model metrics calculated: MAE = {mae_stock}, MSE = {mse_stock}, MAPE = {mape_stock}")
        
        # Train the model on the training set for demand
        model_demand = auto_arima(train['total_demand'], seasonal=True, m=12, trace=False, suppress_warnings=True, stepwise=True)
        
        # Forecast on the test set for demand
        forecast_test_demand = model_demand.predict(n_periods=len(test))
        forecast_index_test_demand = test.index
        forecast_series_test_demand = pd.Series(forecast_test_demand, index=forecast_index_test_demand)
        
        # Forecast for the next 6 months for demand
        model_demand.update(df['total_demand'])
        forecast_demand = model_demand.predict(n_periods=180)
        forecast_index_demand = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')
        forecast_series_demand = pd.Series(forecast_demand, index=forecast_index_demand)
        
        # Calculate error metrics for demand
        mae_demand = mean_absolute_error(test['total_demand'], forecast_series_test_demand)
        mse_demand = mean_squared_error(test['total_demand'], forecast_series_test_demand)
        mape_demand = mean_absolute_percentage_error(test['total_demand'], forecast_series_test_demand)
        
        print(f"Demand model metrics calculated: MAE = {mae_demand}, MSE = {mse_demand}, MAPE = {mape_demand}")
        
        # Calculate model efficiency
        efficiency_stock = 1 - mape_stock
        efficiency_demand = 1 - mape_demand
        efficiency_text = f'Lagerbestandsmodell Effizienz: {efficiency_stock:.2%}\nBestellungmodell Effizienz: {efficiency_demand:.2%}'
        
        # Create traces for the stock forecast plot
        trace_actual_stock = go.Scatter(x=df.index, y=df['total_stock'], mode='lines', name='Aktueller Bestand')
        trace_forecast_stock = go.Scatter(x=forecast_series_stock.index, y=forecast_series_stock, mode='lines', name='Vorhersage', line=dict(dash='dash'))
        
        forecast_layout_stock = go.Layout(
            title=title,
            xaxis=dict(title='Datum'),
            yaxis=dict(title='Lagerbestand'),
            showlegend=True
        )

        # Create traces for the available stock plot
        available_stock_traces = []
        if selected_uid is None:
            for uid in unique_ids:
                product_key = f'{selected_product}_{uid}'
                if product_key in df:
                    available_stock_traces.append(
                        go.Scatter(x=df.index, y=df[f'{product_key}_bestand'], mode='lines', name=f'{product_key} Bestand')
                    )
        else:
            product_key = f'{selected_product}_{selected_uid}'
            available_stock_traces.append(
                go.Scatter(x=df.index, y=df[f'{product_key}_bestand'], mode='lines', name=f'{product_key} Bestand')
            )
        
        available_stock_layout = go.Layout(
            title=f'Verfügbarer Lagerbestand in Lager {selected_warehouse}',
            xaxis=dict(title='Datum'),
            yaxis=dict(title='Lagerbestand'),
            showlegend=True
        )
        
        # Create traces for the actual vs predicted plot
        combined_actual_stock = pd.concat([df['total_stock'], forecast_series_stock])
        combined_dates_stock = pd.concat([pd.Series(df.index), pd.Series(forecast_series_stock.index)])
        
        actual_vs_predicted_trace_actual_stock = go.Scatter(x=combined_dates_stock, y=combined_actual_stock, mode='lines', name='Aktueller Bestand')
        actual_vs_predicted_trace_forecast_stock = go.Scatter(x=forecast_series_stock.index, y=forecast_series_stock, mode='lines', name='Vorhersage', line=dict(dash='dash'))

        actual_vs_predicted_layout_stock = go.Layout(
            title=f'Aktueller vs. vorhergesagter {title}',
            xaxis=dict(title='Datum'),
            yaxis=dict(title='Lagerbestand'),
            showlegend=True
        )
        
        # Create traces for the demand forecast plot
        demand_forecast_trace = go.Scatter(x=forecast_series_demand.index, y=forecast_series_demand, mode='lines', name='Bestellungvorhersage', line=dict(dash='dash'))
        demand_forecast_layout = go.Layout(
            title=f'Bestellungvorhersage für {title}',
            xaxis=dict(title='Datum'),
            yaxis=dict(title='Bestellung'),
            showlegend=True
        )
        
        # Create traces for the demand vs actual plot
        demand_vs_actual_trace_actual = go.Scatter(x=df.index, y=df['total_demand'], mode='lines', name='Aktuelle Bestellung')
        demand_vs_actual_trace_forecast = go.Scatter(x=forecast_series_demand.index, y=forecast_series_demand, mode='lines', name='Vorhersage Bestellung', line=dict(dash='dash'))
        demand_vs_actual_layout = go.Layout(
            title=f'Aktuelle vs. Predicted Bestellung für {title}',
            xaxis=dict(title='Datum'),
            yaxis=dict(title='Bestellung'),
            showlegend=True
        )

        print("Returning graph data")
        return (
            {'data': [trace_actual_stock, trace_forecast_stock], 'layout': forecast_layout_stock}, 
            {'data': available_stock_traces, 'layout': available_stock_layout}, 
            {'data': [actual_vs_predicted_trace_actual_stock, actual_vs_predicted_trace_forecast_stock], 'layout': actual_vs_predicted_layout_stock},
            {'data': [demand_forecast_trace], 'layout': demand_forecast_layout},
            {'data': [demand_vs_actual_trace_actual, demand_vs_actual_trace_forecast], 'layout': demand_vs_actual_layout},
            f'Lagerbestand - Mittlerer absoluter Fehler (MAE): {mae_stock:.2f}',
            f'Lagerbestand - Mittlerer quadratischer Fehler (MSE): {mse_stock:.2f}',
            f'Lagerbestand - Mittlerer absoluter Prozentfehler (MAPE): {mape_stock:.2%}',
            efficiency_text
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return {}, {}, {}, {}, {}, "Fehler", "Fehler", "Fehler", "Fehler"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)  # Change the port if needed
