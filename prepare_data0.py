import os
import json
import glob

with open("project.json","r",encoding='utf-8') as d:
    str_json = d.read()
    train_json = json.loads(str_json)
STARTSCORE =train_json["use_score_num"]
num_classes = train_json["num_labels"]
image_path =train_json["image_path"]
meta_path =train_json["metadata_path"]

if STARTSCORE > 0:
    USESCORE = True
else:
    USESCORE = False

#image_path = r"C:/D/512px" # danbooru2021/512px

#meta_path = r"E:/work/danbooru/0/danbooru2021/metadata"  # danbooru2021/metadata



def process_Once(pic_dic):

        labels_nums =[]
        if "id" not in pic_dic.keys():
            return [],"1"
        pic_md5 = pic_dic["md5"]
        pic_ext = pic_dic["file_ext"]
        pic_id = pic_dic["id"]
        pic_tags = pic_dic["tag_string"]
        pic_score = pic_dic["score"]
        if USESCORE:
            if int(pic_score) < STARTSCORE:
                return [],"2"
        """
        output_name = '{}.{}'.format(pic_md5, pic_ext)
        output_path = os.path.join(image_path, pic_md5[0:2])
        input_file = os.path.join(output_path, output_name)
        """
        
        pic_in_name = "{}.{}".format(pic_id,pic_ext)
        input_path = os.path.join(image_path,pic_id.zfill(3)[-3:].zfill(4))
        input_file = os.path.join(input_path,pic_in_name)

        
        
        if  os.path.exists(input_file):

            tag_list = pic_tags.split(" ")
            for i in tag_list:
                if i in tags:
                    labels_nums.append(str(tags.index(i)))
                    
            return labels_nums,input_file
        else:
            return [],"4" #return 4/4





def processImages():
    test_dict = dict()

    json_file_list = glob.glob(os.path.join(meta_path, "posts*.json"))
    for json_filename in json_file_list:
        print(json_filename)
        with open(json_filename,"r",encoding="utf-8") as f:
            raw_json_lines = f.readlines()
        with open("images_all.txt","a",encoding='utf-8') as d:
            for i_line in raw_json_lines:
                pic_dic = json.loads(i_line)
                labels,path=process_Once(pic_dic)
                if len(labels) <1:
                    if len(path)>4:
                        path = "5"
                    test_dict[path] = test_dict.get(path,0) +1
                else:
                    test_dict["-1"] = test_dict.get("-1",0) +1
                if len(labels) > 1:
                    d.write("{}\t{}".format(path,"\t".join(labels)))
                    d.write("\n")
            d.close()
            print(test_dict)

class Tags():
    def __init__(self,):
        self.tags_dict = dict()
        """
        self.tags_dict = dict()
        tags_json_path = os.path.join(meta_path,"tags000000000000.json")
        with open(json_filename, "r", encoding="utf-8") as f:
            raw_json_lines = f.readlines()
        for raw_json in raw_json_lines:
            raw_json = json.loads(raw_json) 
            self.tags_dict[raw_json["name"]] = 0        
        """

    def countTags(self,pic_dic):
        tags = pic_dic["tag_string"].split(" ")
        for i_tag in tags:
            self.tags_dict[i_tag] = self.tags_dict.get(i_tag,0) + 1

    def getMostTags(self,number):
        items = list(self.tags_dict.items())
        items.sort(key=lambda x:x[1], reverse=True)
        self.most_items =items[:number]

    def saveFile(self):
        with open("tags6000.json","w",encoding="utf-8") as f:
            f.write(json.dumps(self.tags_dict))
            f.close()



def processCount():
    Tc = Tags()
    json_file_list = glob.glob(os.path.join(meta_path,"posts*.json"))
    for json_filename in json_file_list:
        print(json_filename)
        with open(json_filename,"r",encoding="utf-8") as f:
            raw_json_lines = f.readlines()

        for i_line in raw_json_lines:
            pic_dic = json.loads(i_line)

            Tc.countTags(pic_dic)
    Tc.saveFile()
    return  Tc.tags_dict

if __name__ == '__main__':

    tags_json = processCount()
    '''
    with open('tags6000.json',"r",encoding='utf-8') as f:
        string = f.read()
    tags_json = json.loads(string)'''

    tags_items = list(tags_json.items())
    tags_items.sort(key=lambda x: x[1], reverse=True)
    tags_items = tags_items[:num_classes]
    tags = [tag_i[0] for tag_i in tags_items ]
    with open("tags_all.txt","w",encoding='utf-8') as f:
        f.write("\n".join(tags))
    processImages()


