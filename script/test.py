from transformers import BertForSequenceClassification, BertTokenizer
import torch

from socket import *

model = BertForSequenceClassification.from_pretrained("/home/hfxia/数据挖掘/outputs/home/hfxia/25acl/model/bert-base-uncased/checkpoint-40000")
tokenizer = BertTokenizer.from_pretrained("/home/hfxia/数据挖掘/outputs/home/hfxia/25acl/model/bert-base-uncased/checkpoint-40000")

# 1.创建套接字
tcp_server = socket(AF_INET,SOCK_STREAM)
#绑定ip，port
#这里ip默认本机
address = ('',6002)
tcp_server.bind(address)
# 启动被动连接
print("ready to connect")
tcp_server.listen(5)  
client_socket, clientAddr = tcp_server.accept()
print("connect success")


while (1):
    print("ready to recive")
    recive_data = client_socket.recv(1024).decode("gbk")
    print("recive",recive_data)
    inputs = tokenizer(recive_data,return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    print(prediction.item())
    if prediction.item() == 0:
        answer = "负面意义"
    elif prediction.item() == 1:
        answer = "正面意义"
    else:
        answer = "模型有误"
    flag = client_socket.send(answer.encode("gbk"))
    

client_socket.close()
tcp_server.close()

