"""
Minimal Dashboard - Absolute Simplest Version
Run: python src/minimal_dashboard.py
"""

from flask import Flask, render_template_string, request, jsonify
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)

# Store data
STORE_SALES = {20: 2401395, 4: 2136003, 14: 2054575, 1: 1643691, 6: 1652635, 5: 312481}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Walmart Forecast</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        .container { max-width: 800px; margin: auto; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #2c3e50; color: white; }
        .card { background: #667eea; color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .loading { display: none; color: #666; }
    </style>
</head>
<body>
<div class="container">
    <h1>Walmart Demand Forecast</h1>
    
    <label>Store:</label>
    <select id="store">
        <option value="20">Store 20</option>
        <option value="4">Store 4</option>
        <option value="14">Store 14</option>
        <option value="1">Store 1</option>
        <option value="6">Store 6</option>
        <option value="5">Store 5</option>
    </select>
    
    <label>Weeks:</label>
    <select id="weeks">
        <option value="4">4 weeks</option>
        <option value="8" selected>8 weeks</option>
        <option value="13">13 weeks</option>
    </select>
    
    <button onclick="getForecast()">Get Forecast</button>
    
    <div id="loading" class="loading">Loading...</div>
    <div id="results"></div>
</div>

<script>
    function formatMoney(v) {
        return '$' + Math.round(v).toLocaleString();
    }
    
    async function getForecast() {
        const store = document.getElementById('store').value;
        const weeks = document.getElementById('weeks').value;
        
        document.getElementById('loading').style.display = 'block';
        document.getElementById('results').innerHTML = '';
        
        const response = await fetch('/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ store_id: parseInt(store), weeks: parseInt(weeks) })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            let html = '<div class="card">Total: ' + formatMoney(data.total) + '</div>';
            html += '<div class="card">Weekly Average: ' + formatMoney(data.avg) + '</div>';
            html += '<table><thead><tr><th>Week</th><th>Date</th><th>Forecast</th></tr></thead><tbody>';
            
            for (let i = 0; i < data.forecasts.length; i++) {
                html += '<tr><td>' + (i+1) + '</td><td>' + data.dates[i] + '</td><td>' + formatMoney(data.forecasts[i]) + '</td></tr>';
            }
            html += '</tbody></table>';
            document.getElementById('results').innerHTML = html;
        } else {
            document.getElementById('results').innerHTML = '<div style="color:red">Error: ' + data.error + '</div>';
        }
        
        document.getElementById('loading').style.display = 'none';
    }
    
    getForecast();
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    store_id = data.get('store_id', 20)
    weeks = data.get('weeks', 8)
    
    baseline = STORE_SALES.get(store_id, 1500000)
    
    forecasts = []
    dates = []
    
    for week in range(1, weeks + 1):
        seasonality = 1 + 0.12 * np.sin(2 * np.pi * week / 52)
        forecast_val = baseline * seasonality
        forecasts.append(round(forecast_val, 2))
        dates.append((datetime.now() + timedelta(weeks=week)).strftime('%Y-%m-%d'))
    
    return jsonify({
        'forecasts': forecasts,
        'dates': dates,
        'total': sum(forecasts),
        'avg': sum(forecasts) / len(forecasts)
    })

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MINIMAL DASHBOARD")
    print("=" * 50)
    print("\nInstall Flask first: pip install flask")
    print("Then open: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, port=5000)