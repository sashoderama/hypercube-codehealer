from src.code_healer import CodeHealer
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    healer = CodeHealer()
    
    # Test 1: Safe code
    safe_code = """
    def hello():
        print("Secure operation")
    """
    safe_result = await healer.execute_with_healing(safe_code)
    print("Safe code result:", safe_result)

    # Test 2: Dangerous code
    dangerous_code = """
    def dangerous():
        os.system('rm -rf /')
    """
    try:
        dangerous_result = await healer.execute_with_healing(dangerous_code)
    except Exception as e:
        print("Dangerous code blocked:", str(e))

    # Test 3: Rate limit check
    print("\nTesting rate limits:")
    for i in range(12):
        result = await healer.execute_with_healing("print('Test')")
        print(f"Attempt {i+1}: {'Success' if result['success'] else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())
