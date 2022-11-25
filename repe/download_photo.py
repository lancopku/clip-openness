import json
import os
import base64
import requests as req
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# req.DEFAULT_RETRIES = 5
s = req.session()
s.keep_alive = False
json_path = "./CIFAR100_laion5B_retrieval_1000/"
file_names = os.listdir(json_path)


def download(file_name):
    file_name = json_path + file_name
    download_successful = 0
    download_failure = 0
    count = 0
    Image_class = file_name.split("/")[-1].replace('.json', '')

    folder_path = "./laion5B_retrieval/" + Image_class
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("Image_class: ", Image_class, flush=True)

    f = open(file_name, "r")
    f_failure_record = open(folder_path + "/" + "f_failure_log.txt", "w")
    for line in f.readlines():
        count += 1
        # if count == 10:
        #     break
        image_url = json.loads(line)["url"]
        image_id = json.loads(line)["id"]
        # print(image_url)
        try:
            response = req.get(image_url)
            image = Image.open(BytesIO(response.content))
            ls_f = base64.b64encode(BytesIO(response.content).read()).decode('utf-8')
            imgdata = base64.b64decode(ls_f)
            f_image = open(os.path.join(folder_path, '%d.jpg' % image_id), 'wb')
            f_image.write(imgdata)
            f_image.close()
            download_successful += 1
        except:
            f_failure_record.write(image_url)
            f_failure_record.write("\n")
            download_failure += 1
            continue

    print(Image_class + " download_totally: " + str(count), flush=True)
    print(Image_class + " download_successful: " + str(download_successful), flush=True)
    print(Image_class + " download_failure: " + str(download_failure), flush=True)
    print("\n", flush=True)


with ThreadPoolExecutor(128) as executor:
    res = executor.map(download, file_names)
