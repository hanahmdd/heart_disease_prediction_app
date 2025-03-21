
from experta import Fact, KnowledgeEngine, Rule, DefFacts

class HeartDiseaseRules(KnowledgeEngine):
    @DefFacts()
    def _initial_facts(self):
        yield Fact(action="find_risk")

    @Rule(Fact(cholesterol=lambda x: x > 240), Fact(age=lambda x: x > 50))
    def high_risk_rule1(self):
        self.declare(Fact(risk="high"))

    @Rule(Fact(trestbps=lambda x: x > 140), Fact(smoking="Yes"))
    def high_risk_rule2(self):
        self.declare(Fact(risk="high"))

    @Rule(Fact(exercise="Regular"), Fact(BMI=lambda x: x < 25))
    def low_risk_rule3(self):
        self.declare(Fact(risk="low"))
    