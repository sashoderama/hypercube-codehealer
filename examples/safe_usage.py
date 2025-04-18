import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.code_healer import CodeHealer
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    healer = CodeHealer()
    
    # Test 1: Safe code
    safe_code = """print("Hello Secure World")"""
    safe_result = await healer.execute_with_healing(safe_code)
    print("Safe code result:", safe_result)
    
    # Test 2: Dangerous code
    dangerous_code = """import os\nos.system('rm -rf /')"""
    try:
        result = await healer.execute_with_healing(dangerous_code)
    except Exception as e:
        print(f"Dangerous code blocked: {e}")
    
    # Test 3: Rate limits
    print("\nTesting rate limits:")
    for i in range(12):
        res = await healer.execute_with_healing("print('Test')")
        print(f"Attempt {i+1}: {'Success' if res['success'] else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())
