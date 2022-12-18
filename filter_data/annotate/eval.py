import datasets
import jsonlines
import evaluate
import sys
from transformers import T5Tokenizer
from tqdm import tqdm

bleu = evaluate.load("bleu")

acc = evaluate.load('accuracy')

pathG = sys.argv[1]
pathP = sys.argv[2]

id2subsetType = {
    0: "Overall",
    1: "Yes|No",
    2: "Generative",
    3: "Extractive",
    4: "Unanswerable"
}


def eval(pathP, pathG, removeTask=-1, onlyTask=-1):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    subsetTypeDict = {
        0: {
            "g": [],
            "p": []
        },
        1: {
            "g": [],
            "p": []
        },
        2: {
            "g": [],
            "p": []
        },
        3: {
            "g": [],
            "p": []
        },
        4: {
            "g": [],
            "p": []
        }
    }
    count = 0
    with jsonlines.open(pathG) as gReader, jsonlines.open(pathP) as pReader:
        for item in tqdm(gReader):
            item["tgt"] = item["tgt"].strip()

            if onlyTask != -1:
                if int(item["task"]) != removeTask:
                    continue
            elif removeTask != -1:
                if int(item["task"]) == removeTask:
                    continue

            count += 1
            if item["type"] == 1:
                if item["tgt"].lower().strip().startswith("yes"):
                    subsetTypeDict[1]["g"].append(0)
                else:
                    subsetTypeDict[1]["g"].append(1)
            elif item["type"] == 2:
                subsetTypeDict[2]["g"].append(item["tgt"])
            elif item["type"] == 3:
                subsetTypeDict[3]["g"].append(item["tgt"])
            elif item["type"] == 4:
                subsetTypeDict[4]["g"].append(item["tgt"])
                #
                # if item["tgt"].lower().strip().startswith("unanswerable"):
                #     subsetTypeDict[4]["g"].append(0)
                # else:
                #     subsetTypeDict[4]["g"].append(1)
            subsetTypeDict[0]["g"].append(item["tgt"])

        for item in tqdm(pReader):
            item["tgt"] = item["tgt"].strip()
            if onlyTask != -1:
                if int(item["task"]) != removeTask:
                    continue
            elif removeTask != -1:
                if int(item["task"]) == removeTask:
                    continue

            if item["type"] == 1:
                if item["tgt"].lower().startswith("yes"):
                    subsetTypeDict[1]["p"].append(0)
                else:
                    subsetTypeDict[1]["p"].append(1)
            elif item["type"] == 2:
                subsetTypeDict[2]["p"].append(item["tgt"])
            elif item["type"] == 3:
                subsetTypeDict[3]["p"].append(item["tgt"])
            elif item["type"] == 4:
                subsetTypeDict[4]["p"].append(item["tgt"])
                # if item["tgt"].lower().strip().startswith("unanswerable"):
                #     subsetTypeDict[4]["p"].append(0)
                # else:
                #     subsetTypeDict[4]["p"].append(1)
            subsetTypeDict[0]["p"].append(item["tgt"])

    # 1 yes
    # 2 generative
    # 3 extractive
    # 4 unanswerable
    # 0 overall
    result = {1: {}, 2: {}, 3: {}, 4: {}, 0: {}}
    for item in subsetTypeDict.keys():
        if item == 1:
            subMetric = acc.compute(predictions=subsetTypeDict[item]["p"], references=subsetTypeDict[item]["g"])
        else:

            p = subsetTypeDict[item]["p"]
            g = subsetTypeDict[item]["g"]

            if len(p) != 0:
                # p = tokenizer(p, max_length=128, truncation=True)["input_ids"]
                # print(len(p))
                # p = tokenizer.batch_decode(p, skip_special_tokens=True)
                #
                # g = tokenizer(g, max_length=128, truncation=True)["input_ids"]
                # print(len(g))
                # g = tokenizer.batch_decode(g, skip_special_tokens=True)
                g = [[item] for item in g]

                assert len(p) == len(g), f"{len(p)} {len(g)}"
                # print(p[-10:])
                # print(g[-10:])
                subMetric1 = bleu.compute(predictions=p, references=g, max_order=1)
                subMetric4 = bleu.compute(predictions=p, references=g, max_order=4)
                subMetric = {'BLEU1': subMetric1, 'BLEU4': subMetric4}
            else:
                subMetric = {'BLEU1': 0.0, 'BLEU4': 0.0}

        for key in subMetric.keys():
            result[item][key] = subMetric[key]
    print(f"count: {count}")
    return result


for i in [-1, 1, 2, 3]:
    print(f"----Removing Task {i}------")
    result = eval(pathP, pathG, removeTask=i)
    for key in result.keys():
        print("-----------------")
        print(id2subsetType[key])
        print(result[key])
