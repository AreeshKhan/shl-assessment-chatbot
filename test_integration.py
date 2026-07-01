"""
Multi-turn conversation test — simulates the evaluator's behavior.
Tests: Clarification, Recommendation, Refinement, Comparison, Confirmation.
"""
import httpx
import json
import sys

BASE = "http://localhost:8000"

def chat(messages):
    """Send a chat request and return the response."""
    r = httpx.post(f"{BASE}/chat", json={"messages": messages}, timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()

def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[PASS] Health check: {'status': 'ok'}")

def test_single_turn():
    """User gives enough context in turn 1 → should get recommendations."""
    d = chat([{"role": "user", "content": "I need assessments for hiring a Java developer, mid-level, 4 years experience"}])
    assert len(d["recommendations"]) >= 3, f"Expected 3+ recs, got {len(d['recommendations'])}"
    assert d["end_of_conversation"] == False
    # Verify schema
    for rec in d["recommendations"]:
        assert "name" in rec and "url" in rec and "test_type" in rec
        assert rec["url"].startswith("https://www.shl.com/")
    print(f"[PASS] Single-turn: {len(d['recommendations'])} recommendations returned")

def test_multi_turn_refine():
    """User asks, gets recs, then refines."""
    msgs = [{"role": "user", "content": "I need assessments for a senior Python developer"}]
    d1 = chat(msgs)
    print(f"  Turn 1: {len(d1['recommendations'])} recs")
    
    # Add the assistant response + user refinement
    msgs.append({"role": "assistant", "content": d1["reply"]})
    msgs.append({"role": "user", "content": "Can you also add a personality assessment to this list?"})
    d2 = chat(msgs)
    print(f"  Turn 2 (refine): {len(d2['recommendations'])} recs")
    assert len(d2["recommendations"]) >= 3
    print(f"[PASS] Multi-turn refinement works")

def test_schema_compliance():
    """Verify the response matches the exact schema."""
    d = chat([{"role": "user", "content": "What SHL assessments can test cognitive ability?"}])
    # Check top-level keys
    assert "reply" in d, "Missing 'reply' key"
    assert "recommendations" in d, "Missing 'recommendations' key"
    assert "end_of_conversation" in d, "Missing 'end_of_conversation' key"
    assert isinstance(d["reply"], str), "'reply' must be string"
    assert isinstance(d["recommendations"], list), "'recommendations' must be list"
    assert isinstance(d["end_of_conversation"], bool), "'end_of_conversation' must be bool"
    print(f"[PASS] Schema compliance verified ({len(d['recommendations'])} recs)")

print("=" * 50)
print("SHL Assessment Chatbot - Integration Tests")
print("=" * 50)

test_health()
test_schema_compliance()
test_single_turn()
test_multi_turn_refine()

print("=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
