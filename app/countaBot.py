from flask import Flask, render_template, request, url_for, jsonify
import counter_pipeline

app = Flask(__name__)

@app.route('/argue/', methods=['POST'])
def my_argument():
    print("received!")

    input_json = request.json
    print('data from client:', input_json)

    counter = {
        "counter": counter_pipeline.counter(input_json["topic"], input_json["claim"])
    }

    return jsonify(counter)

    # return "fuck you"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
