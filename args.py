import sys

# Check if enough arguments are passed
if len(sys.argv) < 3:
    print("Usage: python script.py <num1> <num2>")
    sys.exit(1)

# Read arguments
n1 = int(sys.argv[1])
n2 = int(sys.argv[2])

# Perform operations
r1 = n1 + n2
r2 = n1 - n2
r3 = n1 * n2
r4 = n1 / n2

# Output results
print("add", r1)
print("sub", r2)
print("mul", r3)
print("div", r4)

