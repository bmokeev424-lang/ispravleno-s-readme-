import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_fuzzy_risk(row):
    prof = ctrl.Antecedent(np.arange(0, 101, 1), 'prof')
    pay = ctrl.Antecedent(np.arange(1, 11, 1), 'pay')
    fin = ctrl.Consequent(np.arange(0, 101, 1), 'fin')

    prof['low'] = fuzz.trimf(prof.universe, [0, 0, 50])
    prof['high'] = fuzz.trimf(prof.universe, [50, 100, 100])
    pay['short'] = fuzz.trimf(pay.universe, [1, 1, 5])
    pay['long'] = fuzz.trimf(pay.universe, [5, 10, 10])
    fin['low'] = fuzz.trimf(fin.universe, [0, 0, 50])
    fin['high'] = fuzz.trimf(fin.universe, [50, 100, 100])

    rules_fin = [ctrl.Rule(prof['high'] & pay['short'], fin['low']),
                 ctrl.Rule(prof['low'] & pay['long'], fin['high']),
                 ctrl.Rule(prof['high'] & pay['long'], fin['low']),
                 ctrl.Rule(prof['low'] & pay['short'], fin['high'])]

    sim_fin = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_fin))
    try:
        sim_fin.input['prof'] = row['profitability']
        sim_fin.input['pay'] = row['payback_period']
        sim_fin.compute()
        fin_val = sim_fin.output['fin']
    except:
        fin_val = 50

    mkt_val = (row['market_volatility'] + row['competition_level'] * 10) / 2
    org_val = 100 - row['capability_score'] / 10
    final_risk = (fin_val * 0.4) + (mkt_val * 0.4) + (org_val * 0.2)
    return min(max(final_risk, 0), 100)