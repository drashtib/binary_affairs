import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

def create_video():
    # Generate example data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    total_stock = np.random.randint(300, 500, size=(365,))
    df = pd.DataFrame({'total_stock': total_stock}, index=dates)

    # Generate forecast data
    forecast_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')  # Shorten forecast period for simplicity
    forecast_stock = np.random.randint(100, 300, size=(30,))
    forecast_series = pd.Series(forecast_stock, index=forecast_dates)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    def init():
        ax.clear()
        ax.plot(df.index, df['total_stock'], label='Aktueller Bestand')
        ax.set_xlabel('Datum')
        ax.set_ylabel('Lagerbestand')
        ax.legend()
        ax.set_title('Lagerbestand Vorhersage')

    def update(frame):
        ax.clear()
        ax.plot(df.index, df['total_stock'], label='Aktueller Bestand')
        ax.plot(forecast_series.index[:frame], forecast_series[:frame], label='Vorhersage', linestyle='--')
        ax.set_xlabel('Datum')
        ax.set_ylabel('Lagerbestand')
        ax.legend()
        ax.set_title('Lagerbestand Vorhersage')

    ani = animation.FuncAnimation(fig, update, frames=len(forecast_series), init_func=init, repeat=False)

    # Save the animation
    ani.save('assets/forecast_video.mp4', writer='ffmpeg', fps=5)  # Reduce fps for simplicity

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Button('Generate Video', id='generate-video-button', n_clicks=0),
            html.Div(id='video-container')
        ]),
    ]),
])

@app.callback(
    Output('video-container', 'children'),
    Input('generate-video-button', 'n_clicks')
)
def update_video(n_clicks):
    if n_clicks > 0:
        create_video()
        return html.Video(src='/assets/forecast_video.mp4', controls=True, autoPlay=True)
    return html.Div()

if not os.path.exists('assets'):
    os.makedirs('assets')

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
