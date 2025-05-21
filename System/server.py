from flask import Flask, request, jsonify
from flask_cors import CORS  # 处理跨域问题
# from model import sentiment  # 导入你的情感分析类


from socket import *
import numpy as np
import PIL

tcp_socket = socket(AF_INET,SOCK_STREAM)
# 2.准备连接服务器，建立连接
serve_ip = "10.1.114.127"
serve_port = 6002  #端口，比如8000
tcp_socket.connect((serve_ip,serve_port))  # 连接服务器，建立连接,参数是元组形式
print("connect success")


def send_mesage(tcp_socket,sentence):
    flag = tcp_socket.send(sentence.encode("gbk"))
    print("send",sentence)

    recive_data = tcp_socket.recv(1024).decode("gbk")
    return recive_data


app = Flask(__name__)
CORS(app)  # 允许所有跨域请求
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        sentence = data.get('text', '')
        
        # 调用情感分析模块
        # tcp_socket.connect((serve_ip,serve_port))  # 连接服务器，建立连接,参数是元组形式
        flag = tcp_socket.send(sentence.encode("gbk"))
        print(sentence)
        analysis_result = tcp_socket.recv(1024).decode("gbk")
        # analysis_result = send_mesage(tcp_socket=tcp_socket, sentence=sentence)  # 假设sem需要实例化
        # tcp_socket.close()
        
        return jsonify({
            "status": "success",
            "result": analysis_result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    app.run(port=5000, debug=False)
    # tcp_socket.close()