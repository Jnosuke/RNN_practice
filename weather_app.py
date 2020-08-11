import pickle
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import temp_prefigure_app as tmp_pre

# Flaskでwebアプリ作成
app = Flask(__name__)

@app.route('/')
def app_root():
    # HTMLフォームを表示
    return html("""
    <div><form action="/predict">
    年: <select name='year'>
            <option value='2021'>2021</option>
            <option value='2022'>2022</option>
            <option value='2023'>2023</option>
            <option value='2024'>2024</option>
            <option value='2025'>2025</option>
            <option value='2026'>2026</option>
            <option value='2027'>2027</option>
            <option value='2028'>2028</option>
            <option value='2029'>2029</option>
            <option value='2030'>2030</option>
        </select><br>
    月: <select name='month'>
            <option value='1'>1</option>
            <option value='2'>2</option>
            <option value='3'>3</option>
            <option value='4'>4</option>
            <option value='5'>5</option>
            <option value='6'>6</option>
            <option value='7'>7</option>
            <option value='8'>8</option>
            <option value='9'>9</option>
            <option value='10'>10</option>
            <option value='11'>11</option>
            <option value='12'>12</option>
        </select><br>
    <input type='submit' value='予測'>
    </form></div>
    """)

@app.route('/predict')
def app_predict():
    # 受信したフォーム引数を読み込む
    year = request.args.get('year')
    month = request.args.get('month')
    # 予測して結果を表示する
    y_pred = tmp_pre.predict_weather(year=year, month=month)

    return html("""
    <h1>{}年 {}月 平均気温{}℃</h1>
    <a href='/'>戻る</a>
    """.format(year, month, round(y_pred[month], 1)))

def html(body):
    return """
    <html><head><style>
    * { padding:8px; margin:4px; }
    div {border: 1px; solid silver; }
    h1 { border-bottom: 3px solid silver; }
    </style></head><body>
    <h1>平均気温予測</h1>
    """ + body + """
    </body></html>"""

if __name__ == "__main__":
    # webサーバーを起動
    app.run(port=8888, host='0.0.0.0')
