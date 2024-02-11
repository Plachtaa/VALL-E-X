""" from itertools import chain

from bs4 import BeautifulSoup
xml_handle = open("D:\\MachineCourse\\Graduation_Project\\dev\\xml\\utf8\\0D58D15F-A9C0-4E65-9270-A36A9E016B18.xml", "r")
soup = BeautifulSoup(xml_handle, "xml")

for segment in soup.find_all("segment"):
        #id=segment["id"] + "_" + segment["starttime"] + ":" + segment["endtime"],
        recording_id=segment["id"].split("_utt")[0].replace("_", "-"),
        start=float(segment["starttime"]),
        #duration=round(
         #       float(segment["endtime"]) - float(segment["starttime"]), ndigits=8
         #   ),
        #channel=0,
        text=" ".join(
                [
                    element.string
                    for element in segment.find_all("element")
                    if element.string is not None
                ]
            ),
        language="Arabic",
        #speaker=int(match(r"\w+speaker(\d+)\w+", segment["who"]).group(1)),
         """


""" with open("D:\\MachineCourse\\dataset\\train\\audio_ann_sum.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ls = [l.split("|") for l in lines]
        ls_T = list(zip(*ls))
        del ls_T[3]
        print(ls_T[3]) """


""" import h5py
_archive = h5py.File("D:\\MachineCourse\\dataset\\train\\audio_sum.hdf5", "r")
print(_archive)
h5_paths="D:\\MachineCourse\\dataset\\train\\audio_sum.hdf5"
h5_path = h5_paths


# print(h5_path)
sub = _archive[h5_path]
#print(len(_archive))
#this is the error #
 """





#from bs4 import BeautifulSoup as bss4
#url="D:\\MachineCourse\\Graduation_Project\\dev\\xml\\utf8\\01BE8E7B-C179-42E3-8521-109C2C732334.xml"

#xml_handle = open(url, "r")
#soup=bss4(xml_handle,'lxml')






































""" from tokenizers import Tokenizer, models
import json

try:
    # Load the JSON configuration from the file
         with open("D:\\MachineCourse\\Graduation_Project\\VALL-E-X\\utils\\g2p\\bpe_69.json", "r", encoding="utf-8") as file:
           tokenizer_config = file.read()

    # Parse the JSON configuration
           tokenizer_config = json.loads(tokenizer_config)

    # Create a BPE model from the vocab and merges in the configuration
           bpe_model = models.BPE(
          vocab=tokenizer_config["model"]["vocab"],
          merges=tokenizer_config["model"]["merges"]
     )

    # Create the tokenizer instance
           tokenizer = Tokenizer(bpe_model)

           print("Tokenizer loaded successfully.")

except Exception as e:
    # If an error occurs, print the error message
             print(f"Error loading tokenizer from : {e}")


text="hello everybody"

cptpho_tokens = tokenizer.encode(text).ids
print(cptpho_tokens)

 """


































