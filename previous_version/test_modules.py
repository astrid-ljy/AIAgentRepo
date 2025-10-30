"""
Test script to verify the modular structure works correctly.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")

    try:
        import config
        print("✓ config module imported")
    except Exception as e:
        print(f"✗ config module failed: {e}")

    try:
        import prompts
        print("✓ prompts module imported")
    except Exception as e:
        print(f"✗ prompts module failed: {e}")

    try:
        import database
        print("✓ database module imported")
    except Exception as e:
        print(f"✗ database module failed: {e}")

    try:
        import sql_generation
        print("✓ sql_generation module imported")
    except Exception as e:
        print(f"✗ sql_generation module failed: {e}")

    try:
        import core
        print("✓ core module imported")
    except Exception as e:
        print(f"✗ core module failed: {e}")

    try:
        import ui
        print("✓ ui module imported")
    except Exception as e:
        print(f"✗ ui module failed: {e}")

    try:
        import agents
        print("✓ agents module imported")
    except Exception as e:
        print(f"✗ agents module failed: {e}")

def test_basic_functionality():
    """Test basic functionality without Streamlit."""
    print("\\nTesting basic functionality...")

    try:
        from sql_generation import generate_fallback_sql
        result = generate_fallback_sql("what is the top selling product", {})
        print(f"✓ SQL generation works: {result[:50]}...")
    except Exception as e:
        print(f"✗ SQL generation failed: {e}")

    try:
        from core import infer_default_model_plan
        plan = infer_default_model_plan("cluster customers", {})
        print(f"✓ Model plan inference works: {plan}")
    except Exception as e:
        print(f"✗ Model plan inference failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
    print("\\n🎉 Module testing completed!")