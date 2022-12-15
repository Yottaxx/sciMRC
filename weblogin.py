#Beijing Institute of Tech
import http.cookiejar
import urllib.request
import urllib.parse
import time

#created by yangzhi
#modified by SYX

#header information
#Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36


login_address = 'http://10.0.0.55:801/srun_portal_pc.php'
user_name = '3120201096'
password = 'ZHANG1998718'

def getOpener(head):
    cj = http.cookiejar.CookieJar()
    pro = urllib.request.HTTPCookieProcessor(cj)
    opener = urllib.request.build_opener(pro)
    header = []
    for key, value in head.items():
        elem = (key, value)
        header.append(elem)
    opener.addheaders = header
    return opener

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
web_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
}
opener = getOpener(web_headers)
post_dict = {
    'username': user_name,
    'password': password,
    'action': 'login',
    'ac_id': '1',
    'user_ip': '',
    'nas_ip': '',
    'user_mac': '',
    'save_me': '1',
    'ajax': '1'
}
post_data = urllib.parse.urlencode(post_dict).encode()
op = opener.open(login_address, post_data)
data = op.read()
print(data.decode('utf8'))