#!/usr/bin/env python
from json import dumps
from flask import Flask, g, Response, request, render_template
import json

app = Flask(__name__)

@app.route("/")
def get_index():
    return render_template('louvain_cluster.html')

@app.route("/getCorrData")
def get_corr_data():
	f = open('corr_layer_10.csv')
	lines = f.readlines()
	data = {}
	for i in range(len(lines)):
		data[i] = lines[i].strip().split(',')
	f.close()
	return json.dumps(data)


if __name__ == '__main__':
    app.run()
