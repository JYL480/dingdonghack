"""
Test Neo4j Aura connection
This script verifies that the Neo4j database connection is working properly.
"""

import sys
from neo4j import GraphDatabase

# Connection settings from config
URI = "neo4j+ssc://cbd4231c.databases.neo4j.io"  # Using +ssc for self-signed cert compatibility
AUTH = ("neo4j", "ub7z9Hg_hCFRXjatBqE6xlKjYm8yPC7Vw8xEhNasVQ4")

def test_connection():
    """Test Neo4j connection and basic operations"""
    driver = None
    try:
        print("Testing Neo4j Aura connection...")
        print(f"URI: {URI}")
        print(f"Username: {AUTH[0]}")
        print("-" * 50)
        
        # Create driver
        print("Creating driver...")
        driver = GraphDatabase.driver(URI, auth=AUTH)
        
        # Verify connectivity
        print("Verifying connectivity...")
        driver.verify_connectivity()
        print("[SUCCESS] Connection verified!")
        
        # Test basic query
        print("\nRunning test query...")
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS num").single()
            print(f"[SUCCESS] Query result: {result['num']}")
        
        # Check database content
        print("\nChecking database content...")
        with driver.session(database="neo4j") as session:
            # Count nodes
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()
            print(f"Total nodes: {node_count['count']}")
            
            # Count relationships
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            print(f"Total relationships: {rel_count['count']}")
            
            # List node labels
            labels = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
            if labels['labels']:
                print(f"Node labels: {', '.join(labels['labels'])}")
            else:
                print("No node labels found (empty database)")
        
        print("\n[SUCCESS] All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if your Neo4j Aura instance is running at https://console.neo4j.io")
        print("2. Verify the instance is not paused (resume if needed)")
        print("3. Ensure credentials in config.py are correct")
        print("4. Check network connectivity")
        return False
        
    finally:
        if driver:
            driver.close()
            print("\nDriver closed.")

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)