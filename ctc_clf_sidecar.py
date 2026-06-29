#!/usr/bin/env python3
"""
ctc_clf_sidecar.py — Python sidecar for CTCMemory (cognitive-loop-framework)

Reads JSON from stdin, executes the requested method, writes JSON to stdout.

Methods:
  ingest_cycle    — Index a cognitive cycle record into the CTC graph
  reconstruct     — Run MRAgent active reconstruction for a question
  temporal_query  — Get events in a date range
  event_keywords  — Get cues for an event
  edges_by_tag    — Follow cue→tag→content edges
"""

import json
import logging
import sys
from pathlib import Path

# Add evolva-mragent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


def handle_ingest_cycle(args: dict) -> dict:
    """Index a cognitive cycle record into the CTC graph."""
    from evolva_mragent.memory.system import MemorySystem
    from evolva_mragent.memory.indexer import CognitiveLoopIndexer
    from evolva_mragent.memory.persistence import MemoryPersistence
    from evolva_mragent.llm.controller import LLMController

    cycle = args.get("cycle", {})
    db_path = Path(args.get("db_path", Path.home() / ".codebase-memory" / "ctc_clf_graph.db"))

    if not cycle:
        return {"ok": False, "error": "cycle is required"}

    # Load existing graph or create new
    persistence = MemoryPersistence()
    memory = None
    if db_path.exists():
        try:
            memory = persistence.load(str(db_path))
        except Exception as e:
            logger.warning(f"Failed to load existing graph: {e}")
            memory = MemorySystem()
    else:
        memory = MemorySystem()
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # Index the cycle
    llm = LLMController()
    indexer = CognitiveLoopIndexer(llm=llm, memory=memory)
    indexer.index_cycles([cycle])

    # Save
    persistence.save(memory, str(db_path))
    return {"ok": True, "events": len(memory.episode_events)}


def handle_reconstruct(args: dict) -> dict:
    """Run MRAgent active reconstruction for a question."""
    from evolva_mragent.memory.persistence import MemoryPersistence
    from evolva_mragent.memory.controller import MemoryController
    from evolva_mragent.llm.controller import LLMController
    from evolva_mragent.agent.reconstruct import ActiveReconstructionAgent
    from evolva_mragent.prompts.base import Prompts

    question = args.get("question", "")
    db_path = Path(args.get("db_path", Path.home() / ".codebase-memory" / "ctc_clf_graph.db"))

    if not question:
        return {"error": "question is required", "answer": "", "confidence": "low"}

    if not db_path.exists():
        return {
            "error": f"CTC graph not found at {db_path}. Ingest some cycles first.",
            "answer": "",
            "confidence": "low",
            "question": question,
            "supports": [],
            "reasoning": "",
            "tool_calls_made": 0,
            "rounds": 0,
            "evidence_texts": [],
        }

    persistence = MemoryPersistence()
    memory = persistence.load(str(db_path))
    controller = MemoryController(memory)
    llm = LLMController()
    agent = ActiveReconstructionAgent(
        controller=controller,
        llm=llm,
        system_prompt=Prompts.AGENT_SYSTEM_PROMPT,
    )

    result = agent.reconstruct(question)
    return result.to_dict()


def handle_temporal_query(args: dict) -> dict:
    """Get events in a date range."""
    import sqlite3

    db_path = Path(args.get("db_path", Path.home() / ".codebase-memory" / "ctc_clf_graph.db"))
    start_date = args.get("start_date", "")
    end_date = args.get("end_date", "")

    if not db_path.exists():
        return {"start_date": start_date, "end_date": end_date, "events": [], "count": 0}

    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """SELECT t.date_str, t.event_id, e.text, e.origin, e.domain
               FROM timeline t
               JOIN episode_events e ON t.event_id = e.event_id
               WHERE t.date_str >= ? AND t.date_str <= ?
               ORDER BY t.date_str DESC LIMIT 100""",
            (start_date, end_date),
        ).fetchall()
        conn.close()
        events = [
            {"date": r[0], "event_id": r[1], "text": r[2], "origin": r[3], "domain": r[4]}
            for r in rows
        ]
        return {"start_date": start_date, "end_date": end_date, "events": events, "count": len(events)}
    except Exception as e:
        return {"start_date": start_date, "end_date": end_date, "events": [], "count": 0, "error": str(e)}


def handle_event_keywords(args: dict) -> dict:
    """Get cues for an event."""
    import sqlite3

    db_path = Path(args.get("db_path", Path.home() / ".codebase-memory" / "ctc_clf_graph.db"))
    event_id = args.get("event_id", "")

    if not db_path.exists():
        return {"event_id": event_id, "keywords": []}

    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT key_id FROM event_to_keys WHERE event_id = ?", (event_id,)
        ).fetchall()
        keywords = []
        for (key_id,) in rows:
            kn = conn.execute(
                "SELECT key_id, text, tag_list FROM key_nodes WHERE key_id = ?", (key_id,)
            ).fetchone()
            if kn:
                keywords.append({"key": kn[0], "text": kn[1], "tags": json.loads(kn[2] or "[]")})
        conn.close()
        return {"event_id": event_id, "keywords": keywords}
    except Exception as e:
        return {"event_id": event_id, "keywords": [], "error": str(e)}


def handle_edges_by_tag(args: dict) -> dict:
    """Follow cue→tag→content edges."""
    import sqlite3

    db_path = Path(args.get("db_path", Path.home() / ".codebase-memory" / "ctc_clf_graph.db"))
    key = args.get("key", "")
    tag = args.get("tag", "")

    if not db_path.exists():
        return {"key": key, "tag": tag, "events": []}

    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT key_id, tag_dict FROM key_nodes WHERE LOWER(key_id) = LOWER(?)",
            (key,)
        ).fetchall()
        events = []
        for (_, tag_dict_str) in rows:
            tag_dict = json.loads(tag_dict_str or "{}")
            event_ids = tag_dict.get(tag, [])
            for eid in event_ids:
                ev = conn.execute(
                    "SELECT event_id, text, origin, time FROM episode_events WHERE event_id = ?",
                    (eid,)
                ).fetchone()
                if ev:
                    events.append({"event_id": ev[0], "text": ev[1], "origin": ev[2], "time": ev[3]})
        conn.close()
        return {"key": key, "tag": tag, "events": events}
    except Exception as e:
        return {"key": key, "tag": tag, "events": [], "error": str(e)}


HANDLERS = {
    "ingest_cycle": handle_ingest_cycle,
    "reconstruct": handle_reconstruct,
    "temporal_query": handle_temporal_query,
    "event_keywords": handle_event_keywords,
    "edges_by_tag": handle_edges_by_tag,
}


def main():
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"error": "Empty input"}))
            return

        request = json.loads(raw)
        method = request.get("method", "")
        args = request.get("args", {})

        handler = HANDLERS.get(method)
        if not handler:
            print(json.dumps({"error": f"Unknown method: {method}"}))
            return

        response = handler(args)
        print(json.dumps(response))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}))
    except Exception as e:
        logger.exception("Sidecar error")
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
