"""
Verification script for ChatDev-style agent collaboration system
Run this after installing dependencies to ensure everything is working
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("ChatDev Agent System - Installation Verification")
print("=" * 60)

# Check 1: Dependencies
print("\n1. Checking dependencies...")
dependencies = {
    "sqlglot": "SQL parser",
    "pydantic": "Data validation",
    "opentelemetry": "Observability (optional)"
}

for package, description in dependencies.items():
    try:
        __import__(package)
        print(f"   ✅ {package:20s} - {description}")
    except ImportError:
        print(f"   ❌ {package:20s} - {description} [MISSING]")
        if package in ["sqlglot", "pydantic"]:
            print(f"      → Install with: pip install {package}")

# Check 2: Local modules
print("\n2. Checking local modules...")
modules = {
    "agent_contracts": "Pydantic schemas for agent communication",
    "sql_validator": "Parser-based SQL validation",
    "agent_memory": "Structured memory management",
    "atomic_chat": "Multi-turn dialogue system",
    "chatchain": "Main orchestration"
}

all_ok = True
for module, description in modules.items():
    try:
        __import__(module)
        print(f"   ✅ {module:20s} - {description}")
    except ImportError as e:
        print(f"   ❌ {module:20s} - {description}")
        print(f"      Error: {e}")
        all_ok = False

# Check 3: Test CTE validation
print("\n3. Testing CTE validation (your specific bug)...")
try:
    from sql_validator import Validator

    sql = """
    WITH recent_reviews AS (
        SELECT game_id, COUNT(*) AS review_count
        FROM games_reviews
        WHERE review_date >= '2024-01-01'
        GROUP BY game_id
    )
    SELECT g.game_name, r.review_count
    FROM recent_reviews r
    JOIN games_info g ON r.game_id = g.game_id
    ORDER BY r.review_count DESC LIMIT 1
    """

    validator = Validator(
        catalog={
            "games_reviews": ["game_id", "review_text", "review_score", "review_date", "helpful_count"],
            "games_info": ["game_id", "game_name", "description", "score", "ratings_count"]
        }
    )

    result = validator.analyze_sql(sql)

    if result["ok"]:
        print(f"   ✅ CTE validation works!")
        print(f"      Sources: {result['lineage'].sources}")
        print(f"      CTEs: {result['lineage'].ctes}")
    else:
        print(f"   ❌ CTE validation failed: {result['errors']}")
        all_ok = False

except Exception as e:
    print(f"   ❌ CTE test failed: {e}")
    all_ok = False

# Check 4: Test Pydantic validation
print("\n4. Testing Pydantic validation...")
try:
    from agent_contracts import DSProposal, AMCritique, JudgeVerdict

    # Test DSProposal
    proposal = DSProposal(
        goal="Test query",
        sql="SELECT * FROM test",
        assumptions=["Column exists"],
        expected_schema=[],
        risk_flags=[]
    )
    print(f"   ✅ DSProposal validation works")

    # Test AMCritique
    critique = AMCritique(
        decision="approve",
        reasons=["Looks good"],
        required_changes=[],
        nonnegotiables=[]
    )
    print(f"   ✅ AMCritique validation works")

    # Test JudgeVerdict
    verdict = JudgeVerdict(
        verdict="pass",
        severity="MINOR",
        evidence=["No issues"],
        required_actions=[]
    )
    print(f"   ✅ JudgeVerdict validation works")

except Exception as e:
    print(f"   ❌ Pydantic validation failed: {e}")
    all_ok = False

# Final summary
print("\n" + "=" * 60)
if all_ok:
    print("✅ ALL CHECKS PASSED - System ready to use!")
    print("\nNext steps:")
    print("1. Integrate ChatChain into your app.py")
    print("2. See CHATDEV_INTEGRATION_README.md for usage examples")
else:
    print("❌ SOME CHECKS FAILED - Please fix the issues above")
    print("\nCommon fixes:")
    print("1. Install missing dependencies: pip install sqlglot pydantic")
    print("2. Make sure you're in the e:\\AIAgent directory")
    print("3. Check file permissions")
print("=" * 60)
