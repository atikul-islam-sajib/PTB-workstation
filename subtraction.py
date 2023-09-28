class Substraction:
    def __init__(self, number1, number2):
        self.number1 = number1
        self.number2 = number2
        
        print(self.subtraction())
    
    def subtraction(self):
        return "Sub result = {} ".format(self.number1 - self.number2)
    
if __name__ == "__main__":
    add = Substraction(100, 20)