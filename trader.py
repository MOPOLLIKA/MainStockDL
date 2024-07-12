
class Trader():
      def __init__(self, money=0, stocks={}):
            self.money = money
            self.stocks = stocks # format {"ticker": {"qty": qty, "priceBuy": priceBuy, "priceCurr": priceCurr}}