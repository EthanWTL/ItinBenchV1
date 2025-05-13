import argparse

def main():
    parser = argparse.ArgumentParser(description="A simple script with arguments")
    
    # Add arguments
    parser.add_argument('--name', type=str, help='Your name', required=True)
    parser.add_argument('--age', type=int, help='Your age', default=0)
    
    # Parse arguments
    args = parser.parse_args()

    # Use the arguments
    print(f"Hello, {args.name}! You are {args.age} years old.")

if __name__ == "__main__":
    main()