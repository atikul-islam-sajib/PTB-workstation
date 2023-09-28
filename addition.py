class Addition:
    def __init__(self, number1, number2):
        self.number1 = number1
        self.number2 = number2
        
        print(self.addition())
    
    def addition(self):
        return "Addition = {} ".format(self.number1 + self.number2)
    
    def mul(self):
        return "mul = {}".format(self.number1 * self.number2)
    
if __name__ == "__main__":
    add = Addition(10, 20)