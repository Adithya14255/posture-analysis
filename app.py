from flask import Flask,render_template


app = Flask(__name__)

@app.route('/')
def index():
    print("whut")
    return render_template('index.html',title='Home')

if __name__ == '__main__':
    app.run(debug=True,host='8.8.8.8',port=5000)