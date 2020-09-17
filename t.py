import requests
import json
if __name__ == '__main__':
    url = "http://www.true-think.com/api/match_info"
    match_ID = 144
    datas = json.dumps({'game':'{}'.format(match_ID)})
    r = requests.post(url,  data=datas)
    info = r.json()['data']
    print()
