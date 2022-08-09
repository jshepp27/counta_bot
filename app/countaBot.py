from flask import Flask, render_template, request, url_for, jsonify
import counter_pipeline

app = Flask(__name__)

@app.route('/argue/', methods=['POST'])
def my_argument():
    print("received!")
    # for key in request.form.keys():
    #     print(f"key: {key} => value: {request.form[key]}")

    claim = request.form["claim"]
    topic = request.form["topic"]

    #print(claim, topic)

    # input_json = request.json
    # claim = input_json["claim"]
    # topic = input_json["topic"]

    # print('data from client:', input_json)

    counter = {
        "counter": counter_pipeline.counter(str(topic), str(claim))
    }

    return jsonify(counter)

    # return jsonify(counter)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
