import json, os

video_dir = "/network/rit/lab/tyu-ayu/TYT/LLaVA-NeXT-7B/divprune_image/my_tasks/tasks/msrvttqa/MSRVTT_Zero_Shot_QA/videos/all"
q_path = "/network/rit/lab/tyu-ayu/TYT/LLaVA-NeXT-7B/divprune_image/my_tasks/tasks/msrvttqa/MSRVTT_Zero_Shot_QA/test_q.json"
a_path = "/network/rit/lab/tyu-ayu/TYT/LLaVA-NeXT-7B/divprune_image/my_tasks/tasks/msrvttqa/MSRVTT_Zero_Shot_QA/test_a.json"
out_path = "/network/rit/lab/tyu-ayu/TYT/LLaVA-NeXT-7B/divprune_image/my_tasks/tasks/msrvttqa/msrvttqa_test.json"

qs = json.load(open(q_path))
ans = json.load(open(a_path))


# Build map: question_id -> {"answer": ..., "type": ...}
ans_map = {}
if isinstance(ans, list):
    for x in ans:
        qid = x["question_id"]
        ans_map[qid] = {
            "answer": x.get("answer", ""),
            "type": x.get("type", None),
        }
else:
    # if it's dict keyed by question_id, normalize it
    for qid, x in ans.items():
        if isinstance(x, dict):
            ans_map[qid] = {"answer": x.get("answer", ""), "type": x.get("type", None)}
        else:
            ans_map[qid] = {"answer": x, "type": None}

data = []
missing = 0

for q in qs:
    vid = q["video_name"]
    qid = q.get("question_id", q.get("qid", None))

    a = ans_map.get(qid, {})
    answer = a.get("answer", q.get("answer", ""))
    qtype = a.get("type", q.get("type", None))

    item = {
        "video_name": vid,
        "video": os.path.join(video_dir, f"{vid}.mp4"),
        "question": q["question"],
        "answer": answer,
        "question_id": qid,
        "type": qtype,
    }

    if qid not in ans_map:
        missing += 1

    data.append(item)

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print("Wrote", out_path, "num=", len(data), "missing_answers=", missing)