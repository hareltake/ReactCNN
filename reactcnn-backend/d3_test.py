#!/usr/bin/env python
from json import dumps
from flask import Flask, g, Response, request, render_template

app = Flask(__name__)

@app.route("/")
def get_index():
    return render_template('louvain_cluster.html')



if __name__ == '__main__':
    app.run()
