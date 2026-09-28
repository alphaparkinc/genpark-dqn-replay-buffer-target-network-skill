import sys
import json
from client import DQNReplayTarget

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-dqn-replay-buffer-target-network-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "simulate_dqn_replay_step",
                        "description": "Simulate DQN experience replay sampling and Polyak target network update",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "transitions": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "[s, a, r, s_next, done]"
                                    }
                                },
                                "tau": {"type": "number", "default": 0.1},
                                "batch_size": {"type": "integer", "default": 4}
                            },
                            "required": ["transitions"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "simulate_dqn_replay_step":
            trans = args.get("transitions", [])
            tau = args.get("tau", 0.1)
            bs = args.get("batch_size", 4)
            dqn = DQNReplayTarget(tau=tau)
            for s, a, r, s_next, done in trans:
                dqn.push(int(s), int(a), float(r), int(s_next), bool(done))
            td_loss = dqn.update_batch(batch_size=min(bs, len(trans)))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"mean_td_loss": td_loss, "q_table": dqn.q_table, "target_q_table": dqn.target_q_table})}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
