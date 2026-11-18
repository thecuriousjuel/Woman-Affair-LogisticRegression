import os
from flask import Flask, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    rate_marriage = float(request.form['rate_marriage'])
    age = float(request.form['age'])
    yrs_married = float(request.form['yrs_married'])
    children = float(request.form['children'])
    religious = float(request.form['religious'])
    educ = float(request.form['educ'])
    occupation = float(request.form['occupation'])
    occupation_husb = float(request.form['occupation_husb'])

    data = pd.DataFrame([[rate_marriage, age, yrs_married, children, religious, educ, occupation, occupation_husb]],
    columns=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ', 'occupation', 'occupation_husb'])

    data['married_to_age'] = data['yrs_married'] / data['age']
    data.drop(['yrs_married', 'age'], axis=1, inplace=True)

    full_pipeline = joblib.load(os.path.join(BASE_DIR, 'model', 'full_pipeline.pkl'))
    prepared_data = full_pipeline.transform(data)

    log_reg = joblib.load(os.path.join(BASE_DIR, 'model', 'logistic_regression.pkl'))
    output = log_reg.predict(prepared_data)

    return render_template('index.html', output=output)


if __name__ == '__main__':
    os.environ.pop("WERKZEUG_SERVER_FD", None)
    os.environ.pop("WERKZEUG_RUN_MAIN", None)
    port = int(os.environ.get("PORT", os.environ.get("AFFAIR_PORT", 5206)))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"🚀 Woman Affair Classifier running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
