"""Quick test of the /chat endpoint."""
import httpx
import json

r = httpx.post(
    "http://localhost:8000/chat",
    json={
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer who works with stakeholders. What assessments should I use?"}
        ]
    },
    timeout=30,
)

d = r.json()
print(f"Status: {r.status_code}")
print(f"Reply: {d['reply'][:300]}...")
print(f"\nRecommendations ({len(d['recommendations'])}):")
for i, rec in enumerate(d["recommendations"]):
    print(f"  {i+1}. {rec['name']} [{rec['test_type']}]")
    print(f"     {rec['url']}")
print(f"\nend_of_conversation: {d['end_of_conversation']}")
