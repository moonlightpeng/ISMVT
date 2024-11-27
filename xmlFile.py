#encoding:utf-8
from xml.dom.minidom import Document


doc = Document()  #创建DOM文档对象
item = doc.createElement('DOCUMENT') #创建根元素
item.setAttribute('content_method', "full")#设置命名空间
#引用本地XML Schema
doc.appendChild(item)
# # ############item:Python处理XML之Minidom################
# item = doc.createElement('item')
# item.setAttribute('genre','XML')
# DOCUMENT.appendChild(item)

# 设置阈值
key = doc.createElement('thresholdValue')
key_text = doc.createTextNode('28') #元素内容写入
key.appendChild(key_text)
# DOCUMENT.appendChild(item)
item.appendChild(key)



# 设置一个像素代表多少mm
DPI_value = doc.createElement('DPI')
price_text = doc.createTextNode('0.025')
DPI_value.appendChild(price_text)
item.appendChild(DPI_value)


display = doc.createElement('display')
item.appendChild(display)
display_url = doc.createElement('url')
display_title = doc.createElement('title')
display_url_text = doc.createTextNode('https://www.baidu.com/')
display_title_text = doc.createTextNode('Good')
display.appendChild(display_url)
display.appendChild(display_title)
display_url.appendChild(display_url_text)
display_title.appendChild(display_title_text)
item.appendChild(display)

########### 将DOM对象doc写入文件
f = open('config.xml', 'w')
#f.write(doc.toprettyxml(indent = '\t', newl = '\n', encoding = 'utf-8'))
doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
f.close()
print("Fine")