import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path BEFORE importing our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.extractor import EnhancedKnowledgeExtractor  # noqa: E402

load_dotenv()

async def main():
    """Test the enhanced knowledge extraction"""
    print("🔬 Testing Enhanced Knowledge Extraction")
    print("="*50)

    extractor = EnhancedKnowledgeExtractor()

    # Test with PC algorithm
    algorithm = "Peter-Clark (PC)"
    print(f"🎯 Algorithm: {algorithm}")

    try:
        knowledge = await extractor.extract_enhanced_knowledge(algorithm)

        print("\n📊 Knowledge extracted successfully!")
        print(f"📏 Length: {len(knowledge)} characters")
        print("📄 Preview (first 300 chars):")
        print("-" * 40)
        print(knowledge[:300] + "..." if len(knowledge) > 300 else knowledge)
        print("-" * 40)

        # Save to file for inspection
        output_file = Path("extracted_knowledge.md")
        with open(output_file, 'w') as f:
            f.write("# Enhanced Knowledge Extraction Result\n\n")
            f.write(f"**Algorithm:** {algorithm}\n\n")
            f.write(f"**Extracted on:** {asyncio.get_event_loop().time()}\n\n")
            f.write("---\n\n")
            f.write(knowledge)

        print(f"\n💾 Full knowledge saved to: {output_file}")
        print("✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        print(f"🔍 Error type: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
