class Calculator:
    def add(self, num1, num2):
        return num1 + num2

    def subtract(self, num1, num2):
        return num1 - num2

    def multiply(self, num1, num2):
        return num1 * num2

    def divide(self, num1, num2):
        if num2 == 0:
            return "Error: Division by zero"
        return num1 / num2

    def is_prime(self, num):
        if num <= 1:
            return False
        elif num <= 3:
            return True
        elif num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    def fibonacci(self, n):
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        else:
            fib_sequence = [0, 1]
            while len(fib_sequence) < n:
                next_num = fib_sequence[-1] + fib_sequence[-2]
                fib_sequence.append(next_num)
            return fib_sequence

    def factorial(self, n):
        if n < 0:
            return "Error: Factorial is undefined for negative numbers"
        elif n == 0:
            return 1
        else:
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

# Example usage:
calc = Calculator()

# Perform arithmetic operations
result_add = calc.add(5, 3)
result_subtract = calc.subtract(10, 4)
result_multiply = calc.multiply(6, 2)
result_divide = calc.divide(8, 2)

# Check for prime number
is_prime = calc.is_prime(17)

# Calculate Fibonacci sequence
fib_sequence = calc.fibonacci(10)

# Calculate factorial
factorial_result = calc.factorial(5)

print("Addition:", result_add)
print("Subtraction:", result_subtract)
print("Multiplication:", result_multiply)
print("Division:", result_divide)
print("Is Prime:", is_prime)
print("Fibonacci Sequence:", fib_sequence)
print("Factorial:", factorial_result)
