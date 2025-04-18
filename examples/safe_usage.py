import asyncio
from src.code_healer import CodeHealer

async def main():
    healer = CodeHealer()
    result = await healer.execute_with_healing("print('hello')")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
