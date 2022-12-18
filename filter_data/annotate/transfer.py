import json

import jsonlines
from tqdm import tqdm

qtype2id = {}

with open("./class") as file:
    line = file.readline()
    while line:
        qtype2id[line.replace(" ", "").strip("\n")] = len(qtype2id)
        line = file.readline()

type2id = {"yes or no": 1, "总结概括": 2,"从原文抽取": 3,  "unanswerable": 4}
for dataset in ["Train", "Dev", "Test"]:
    with open("qasperFormatData{}AnEvidenceOnly.json".format(dataset), "r") as file:
        data = json.load(file)

    full = []
    tgt = []
    task = []
    type = []
    qType = []

    for key in tqdm(data):
        text = []
        for section in data[key]['full_text']:
            text.extend(section['paragraphs'])

        for questions in data[key]['qas']:
            question = questions['question']
            answer = questions['answers'][0]['answer']['free_form_answer']
            if questions["questionType"] == "论文实验条件相关":
                questions["questionType"] = "论文实验条件要求"
            full.append("question: " + question + "context: " + " ".join(text))
            tgt.append(answer)

            if questions['answers'][0]['answer']['unanswerable']:
                type.append(type2id["unanswerable"])
                task.append(int(questions["annotateType"]))
                print(questions["questionType"])
                qType.append(qtype2id[questions["questionType"].replace(" ", "").strip("\n")])
                continue
            elif questions['answers'][0]['answer']['answerType'] in type2id.keys():
                type.append(type2id[questions['answers'][0]['answer']['answerType']])
            else:
                print(questions['answers'][0]['answer']['answerType'])

            task.append(int(questions["annotateType"]))
            qType.append(qtype2id[questions["questionType"].replace(" ", "").strip("\n")])

    with jsonlines.open("clean{}all.json".format(dataset), "w") as writer, \
            jsonlines.open("clean{}withoutYes.json".format(dataset), "w") as writerWY, \
            jsonlines.open("clean{}withoutGenerative.json".format(dataset), "w") as writerWG, \
            jsonlines.open("clean{}withoutExtractive.json".format(dataset), "w") as writerWE, \
            jsonlines.open("clean{}withoutUnanswerable.json".format(dataset), "w") as writerWU, \
            jsonlines.open("clean{}withoutTask1.json".format(dataset), "w") as writerW1, \
            jsonlines.open("clean{}withoutTask2.json".format(dataset), "w") as writerW2, \
            jsonlines.open("clean{}withoutTask3.json".format(dataset), "w") as writerW3:

        for i in range(len(full)):
            item = {"full": full[i], "tgt": tgt[i], "task": task[i], "type": type[i], "qType": qType[i]}
            writer.write(item)
            if task[i] != 1:
                writerW1.write(item)
            if task[i] != 2:
                writerW2.write(item)
            if task[i] != 3:
                writerW3.write(item)

            if type[i] != 1:
                writerWY.write(item)
            if type[i] != 2:
                writerWG.write(item)
            if type[i] != 3:
                writerWE.write(item)
            if type[i] != 4:
                writerWU.write(item)
