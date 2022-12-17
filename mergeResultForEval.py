import copy
import os

import jsonlines
from tqdm import tqdm

goldenDev = []
with jsonlines.open("./filter_data/annotate/cleanDevall.json") as reader:
    for item in reader:
        goldenDev.append(item)

goldenTest = []
with jsonlines.open("./filter_data/annotate/cleanTestall.json") as reader:
    for item in reader:
        goldenTest.append(item)

golden = {
    "test": goldenTest,
    "dev": goldenDev
}

resultNameList = "all withoutTask3 withoutExtractive withoutGenerative withoutUnanswerable withoutYes withoutTask1 withoutTask2 "
resultNameList = resultNameList.strip(" ").split(" ")
for model in ["t5"]:
    for name in resultNameList:
        for dataset in ["dev", "test"]:
            prediction = []
            if model == "t5":
                if dataset == "test":
                    path = f"{name}-paperSummarization-QA-{model}-1e4-16epoch-16bsz"
                else:
                    path = f"{name}-paperSummarization-QA-{model}-1e4-16epoch-16bsz-{dataset}"
            else:
                path = f"{name}-paperSummarization-QA-{model}-2e5-16epoch-2x8bsz-{dataset}"

            with open(f'{path}/generated_predictions.txt') as file:
                line = file.readline()
                while line:
                    prediction.append(line.strip("\n"))
                    line = file.readline()

                tempGolden = copy.deepcopy(golden[dataset])
            assert len(tempGolden) == len(prediction)
            with jsonlines.open(f"{path}/prediction.json", "w") as writer:
                for i in tqdm(range(len(tempGolden))):
                    tempGolden[i]["tgt"] = prediction[i]
                    writer.write(tempGolden[i])
            prefixEvalPath = "./filter_data/annotate"
            f = os.popen(
                f"python3 {prefixEvalPath}/eval.py {prefixEvalPath}/clean{dataset.capitalize()}all.json {path}/prediction.json")
            print(f.read())
