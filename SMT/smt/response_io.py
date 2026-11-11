from __future__ import annotations

import json
from pathlib import Path


def save_response_json(path,vo_file,query,candidate_ids,stats,server_query_time_s=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    data={
        "vo_file":str(vo_file),
        "query":query,
        "candidate_ids":[x.hex() for x in candidate_ids],
        "stats":{name:getattr(stats,name) for name in stats.__dataclass_fields__},
        "server_query_time_s":server_query_time_s,
    }
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")


def load_response_json(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    data["candidate_ids"]=[bytes.fromhex(x) for x in data["candidate_ids"]]
    return data
