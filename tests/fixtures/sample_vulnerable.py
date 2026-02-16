"""Sample file with known vulnerabilities for testing."""

import os
import subprocess
import pickle
import hashlib
import yaml


# SQL Injection — string formatting
def get_user_bad(cursor, user_id):
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")


# SQL Injection — concatenation
def search_users(cursor, name):
    cursor.execute("SELECT * FROM users WHERE name = '" + name + "'")


# Command Injection — os.system
def run_command(cmd):
    os.system(cmd)


# Command Injection — subprocess shell=True
def run_shell(cmd):
    subprocess.call(cmd, shell=True)


# Command Injection — eval
def dangerous_eval(expr):
    return eval(expr)


# Hardcoded password
DB_PASSWORD = "super_secret_password_123"
password = "hardcoded_pass"

# Hardcoded AWS key
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"

# Hardcoded API key
api_key = "sk-1234567890abcdef"

# Insecure deserialization — pickle
def load_data(data):
    return pickle.loads(data)


# Insecure deserialization — yaml
def load_yaml_config(content):
    return yaml.load(content)


# Weak crypto — MD5
def hash_password_bad(password):
    return hashlib.md5(password.encode()).hexdigest()


# SSL verify disabled
def fetch_url(url):
    import requests
    return requests.get(url, verify=False)


# Debug mode
DEBUG = True


# Path traversal
def read_file(filename):
    user_input = input("filename: ")
    f = open("/data/" + user_input)
    return f.read()
