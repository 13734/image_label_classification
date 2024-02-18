import os
import json
import glob

with open("project.json","r",encoding='utf-8') as d:
    str_json = d.read()
    train_json = json.loads(str_json)
STARTSCORE =train_json["use_score_num"]
num_classes = train_json["num_labels"]


if STARTSCORE > 0:
    USESCORE = True
else:
    USESCORE = False

tags_json_file = r"E:\work\danbooru\0\tags6000.json"
dest_path =r"../MyDataset/images"
meta_path = r"E:\work\danbooru\0\danbooru2021\metadata"
with open(tags_json_file,"r",encoding='utf-8') as f:
    string = f.read()


def process_Once(pic_dic):

        labels_nums =[]
        if "id" not in pic_dic.keys():
            return [],""
        pic_md5 = pic_dic["md5"]
        pic_ext = pic_dic["file_ext"]
        pic_id = pic_dic["id"]
        pic_tags = pic_dic["tag_string"]
        pic_score = pic_dic["score"]
        if USESCORE:
            if int(pic_score) < STARTSCORE:
                return [],""
        """
        output_name = '{}.{}'.format(pic_md5, pic_ext)
        output_path = os.path.join(dest_path, pic_md5[0:2])
        output_file = os.path.join(output_path, output_name)
        """
        image_path = r"C:\D\512px"
        pic_in_name = "{}.{}".format(pic_id,pic_ext)
        input_path = os.path.join(image_path,pic_id.zfill(3)[-3:].zfill(4))
        input_file = os.path.join(input_path,pic_in_name)
        output_file = input_file
        
        
        if  os.path.exists(output_file):

            tag_list = pic_tags.split(" ")
            for i in tag_list:
                if i in tags:
                    labels_nums.append(str(tags.index(i)))
                    
            return labels_nums,output_file
        else:
            return [],output_file





def processImages():
    json_file_list = glob.glob(os.path.join(meta_path, "posts*.json"))
    for json_filename in json_file_list:
        print(json_filename)
        with open(json_filename,"r",encoding="utf-8") as f:
            raw_json_lines = f.readlines()
        with open("images_all.txt","a",encoding='utf-8') as d:
            for i_line in raw_json_lines:
                pic_dic = json.loads(i_line)
                labels,path=process_Once(pic_dic)
                if len(labels) > 1:
                    d.write("{}\t{}".format(path,"\t".join(labels)))
                    d.write("\n")
            d.close()

if __name__ == '__main__':

    tags_json = json.loads(string)
    tags_items = list(tags_json.items())
    tags_items.sort(key=lambda x: x[1], reverse=True)
    tags_items = tags_items[:num_classes]
    tags = [tag_i[0] for tag_i in tags_items ]
    with open("tags_all.txt","w",encoding='utf-8') as f:
        f.write("\n".join(tags))
    processImages()


