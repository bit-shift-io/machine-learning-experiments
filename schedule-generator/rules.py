
class Rule:
    """Interface for rules"""

    def reward(self, data) -> float:
        """Compute a score based on some results"""
        pass

    def is_valid(self, data) -> bool:
        """Determine if the incomming data set conforms to the rule"""
        pass


class Day(Rule):
    def reward(self, data) -> float:
        return 0.0

    def is_valid(self, data) -> bool:
        return False


class Days(Rule):
    def __init__(self, days):
        self.days = days

    def reward(self, data) -> float:
        return 0.0

    def is_valid(self, data) -> bool:
        return False


class Subject(Rule):
    pass

