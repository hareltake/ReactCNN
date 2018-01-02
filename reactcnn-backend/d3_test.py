#!/usr/bin/env python
from json import dumps
from flask import Flask, g, Response, request, render_template
import json

app = Flask(__name__)

@app.route("/")
def get_index():
    return render_template('index2.html')

@app.route("/getCorrData")
def get_corr_data():
	layer = request.args['l']
	f = open('static/corr_layer_' + layer + '.csv')
	lines = f.readlines()
	data = {}
	for i in range(len(lines)):
		data[i] = lines[i].strip().split(',')
	f.close()
	return json.dumps(data)


if __name__ == '__main__':
    app.run()
