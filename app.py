# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import codecs
import json
app = Flask(__name__)


def getUser(aname, apwd):
	inf = codecs.open('user.txt', 'r', 'utf-8')
	lines = inf.readlines()
	proxys = []
	ishas = False
	for i in lines:
		i = i.replace('\n','')
		arr = i.split(':')
		name = arr[0]
		pwd = arr[1]
		print(type(name))
		print(name)
		print(type(aname))
		print(aname)
		if name == aname:
			print('username is correct')
			if pwd == apwd:
				print('pwd is correct')
				ishas = True
				pass
			pass
	if ishas == True:
		return json.dumps({"msg":"Login"})
		pass
	return json.dumps({"msg":"Login Failed"})


@app.route("/login", methods=['GET'])
def login():
	name = request.args.get('name','erro')
	pwd = request.args.get('pwd','erro')
	return getUser(name, pwd)


@app.route("/regist",methods=['GET'])
def regist():
	name = request.args.get('name','erro')
	pwd = request.args.get('pwd','erro')
	# return amain.getzhanghu()
	return insertuser(name, pwd)


def insertuser(name, pwd):
	f1 = open('user.txt','a')
	f1.write(name+':'+pwd+'\n')
	return json.dumps({"msg":"Success"})



@app.route("/")
def index():
	return render_template('index.html')

@app.route("/userLogin.html")
def userLogin():
	return render_template('userLogin.html')

@app.route("/userReg.html")
def userReg():
	return render_template('userReg.html')



@app.route("/main.html")
def main():
	return render_template('main.html')


if __name__ == '__main__':
	app.run()

