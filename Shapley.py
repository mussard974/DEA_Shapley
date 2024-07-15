import pulp as pl
import numpy as np
from itertools import chain, combinations
import scipy.stats as ss
import scipy.special
fact = scipy.special.factorial

class ShapleyModel(object):
    
    def __init__(self, outputs = None, constraint = None, g_inputs = None, g_outputs = None):
        self.outputs = outputs
        self.constraint = constraint
        self.g_inputs = g_inputs
        self.g_outputs = g_outputs

    def powerset(self, iterable):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        xs = list(iterable)
        return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

    def dea_ddf(self, X, y):
        if self.outputs == "one_dimension":
            if self.g_inputs is None:
                self.g_inputs = np.ones((X.shape[1]))
            if self.g_outputs is None:
                self.g_outputs = 1
            num_dmus, num_inputs = X.shape
            ddf_values = np.zeros(num_dmus)
            for k in range(num_dmus):
                prob = pl.LpProblem(f"DMU_{k}_DDF", pl.LpMaximize)
                lambdas = pl.LpVariable.dicts("Lambda", (range(num_dmus)), lowBound=0)
                theta = pl.LpVariable("Theta", lowBound=0)
                # Objective function: maximize theta
                prob += theta
                prob += pl.lpSum([lambdas[j] * float(y[j]) for j in range(num_dmus)]) >= float(y[k]) + theta * float(self.g_outputs)
                for i in range(num_inputs):
                    prob += pl.lpSum([lambdas[j] * float(X[j, i]) for j in range(num_dmus)]) <= float(X[k, i]) - theta * float(self.g_inputs[i])
                # Constraints:  
                if self.constraint == "DRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) <= 1
                elif self.constraint == "IRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) >= 1
                elif self.constraint == "VRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) == 1
                elif self.constraint == "CRS":
                    pass
                solver = pl.PULP_CBC_CMD(msg=True)
                prob.solve(solver)
                # Store the result
                if pl.LpStatus[prob.status] == "Optimal":
                    ddf_values[k] = theta.varValue
                else:
                    ddf_values[k] = "NO"  # In case no optimal solution is found
            return ddf_values
        
        else:
            num_dmus, num_inputs = X.shape
            num_outputs = y.shape[1]
            ddf_values = np.zeros(num_dmus)
            if self.g_inputs is None:
                self.g_inputs = np.ones((X.shape[1]))
            if self.g_outputs is None:
                self.g_outputs = 1
            
            for k in range(num_dmus):
                # Create the linear programming problem
                prob = pl.LpProblem(f"DMU_{k}_DDF", pl.LpMaximize)
                lambdas = pl.LpVariable.dicts("Lambda", (range(num_dmus)), lowBound=0)
                theta = pl.LpVariable("Theta", lowBound=0)
                # Objective function: maximize theta
                prob += theta
                for r in range(num_outputs):
                    prob += pl.lpSum([lambdas[j] * float(y[j, r]) for j in range(num_dmus)]) >= float(y[k, r]) + theta * float(self.g_outputs[r])
                for i in range(num_inputs):
                    prob += pl.lpSum([lambdas[j] * float(X[j, i]) for j in range(num_dmus)]) <= float(X[k, i]) - theta * float(self.g_inputs[i])

                # Constraints:
                if self.constraint == "DRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) <= 1
                elif self.constraint == "IRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) >= 1
                elif self.constraint == "VRS": 
                    prob += pl.lpSum([lambdas[j] for j in range(num_dmus)]) == 1
                elif self.constraint == "CRS":
                    pass

                # Solve the problem using CBC solver explicitly
                solver = pl.PULP_CBC_CMD(msg=True)
                prob.solve(solver)

                # Store the result
                if pl.LpStatus[prob.status] == "Optimal":
                    ddf_values[k] = theta.varValue
                else:
                    ddf_values[k] = "NO" # In case no optimal solution is found

            return ddf_values

    def fit(self, method, X, y, n_permutations=30):
        self.method = method
        
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
    
        if self.method == "ES":
            return self.ES(X, y)
                
        if self.method == "shapley":
            return self.shapley(X, y)
        
        if self.method == "permutation":
            return self.permutation_importance(X, y, n_permutations)

    def ES(self, X, y):
        n, k = X.shape
        values = np.zeros_like(X)
        for i in range(k):
            values[:, i] = self.dea_ddf(X[:, i][:, np.newaxis], y)
        v_n = self.dea_ddf(X, y)
        v_n = v_n[:, np.newaxis]
        row_sums = values.sum(axis=1)
        result = values + ((v_n - row_sums[:, np.newaxis]) / k)
        return result    

    def shapley(self, X, y):
        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        n, k = X.shape

        variables = [i for i in range(X.shape[1])]
        counter = np.zeros(k)
        values = np.zeros_like(X)

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)
            
            if mask_s.sum() == 0:
                v_s = 0
            else:
                v_s = self.dea_ddf(X[:, mask_s.astype('bool')], y)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    b_si, b_s = 1, 1
                    performance = self.dea_ddf(X[:, mask_si.astype('bool')], y)
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
        
        return np.array(values)

    def permutation_importance(self, X, y, n_permutations=30):
        # Score de base sans permutation
        base_scores = self.dea_ddf(X, y)
        
        # Matrice pour stocker les importances des features
        feature_importances = np.zeros_like(X)
        
        # Pour chaque colonne (feature) de X
        for col in range(X.shape[1]):
            permuted_scores = np.zeros((n_permutations, X.shape[0]))

            # Faire n_permutations permutations
            for i in range(n_permutations):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, col])
                permuted_scores[i, :] = self.dea_ddf(X_permuted, y)

            # Calculer la moyenne des scores permutés pour la colonne en question
            mean_permuted_score = permuted_scores.mean(axis=0)
            
            # Calculer la différence avec le score de base pour obtenir l'importance de la feature
            feature_importances[:, col] = base_scores - mean_permuted_score

        return feature_importances

