import time
import urllib.error
import urllib.request

head = 'https://static.myfigurecollection.net/upload/pictures/2012/12/30'
ids = range(596464, 596883 + 1)
ext = '.jpeg'

dl_dir = 'C:/Users/arimitsu/Desktop/images'

errors = []

for i in ids:
    url = '/'.join((head, str(i))) + ext
    path = '/'.join((dl_dir, str(i))) + ext

    try:
        with urllib.request.urlopen(url) as res, open(path, mode='wb') as f:
            f.write(res.read())
    except urllib.error.URLError as e:
        errors.append(url)

    time.sleep(0.5)
