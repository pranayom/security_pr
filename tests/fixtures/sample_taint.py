"""Sample Flask routes with taint flows for testing."""

import os
import subprocess
import pickle

from flask import Flask, request

app = Flask(__name__)


@app.route("/sql_injection")
def sql_injection():
    user_id = request.args.get("id")
    cursor.execute("SELECT * FROM users WHERE id = " + user_id)
    return "ok"


@app.route("/command_injection")
def command_injection():
    cmd = request.form.get("cmd")
    os.system(cmd)
    return "ok"


@app.route("/eval_injection")
def eval_injection():
    expr = request.args.get("expr")
    result = eval(expr)
    return str(result)


@app.route("/taint_propagation")
def taint_propagation():
    """Taint flows through variable reassignment."""
    data = request.json
    username = data.get("username")
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return "ok"


@app.route("/subprocess_injection")
def subprocess_injection():
    filename = request.args.get("file")
    subprocess.call("cat " + filename, shell=True)
    return "ok"


@app.route("/safe_route")
def safe_route():
    """This route is safe — no tainted data flows to sinks."""
    name = "hardcoded"
    cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
    return "ok"


@app.route("/pickle_injection")
def pickle_injection():
    data = request.data
    obj = pickle.loads(data)
    return str(obj)


def input_to_system():
    """Taint from input() to os.system."""
    cmd = input("Enter command: ")
    os.system(cmd)
