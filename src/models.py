import pymc as pm
import pytensor.tensor as pt
import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor.conv as ptconv

CHANNEL_PRIORS = {
    "spend_channel_1": 0.05,
    "spend_channel_2": 0.05,
    "spend_channel_3": 0.05,
    "spend_channel_4": 0.05,
    "spend_channel_5": 0.05,
    "spend_channel_6": 0.05,
    "spend_channel_7": 0.05,
}


def scale(x, y):
    return x * y.sum() / x.sum()

def logistic_curve(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def cyclic_component(t, a0,a1,b1,period=20):
    return a0 + a1*np.cos(2*np.pi*t/period) + b1*np.sin(2*np.pi*t/period)

def mean_normalize(data):
    return data / data.mean()

def _apply_delayed_adstock_tensor(data, weights):
    weights = weights[::-1]
    n_data = data.shape[0]
    n_weights = weights.shape[0]
    data = data.reshape((1, 1, 1, n_data) ) # Reshape for conv2d
    weights = weights.reshape((1, 1, 1, n_weights))  # Reshape for conv2d
    res = ptconv.conv2d(data, weights, border_mode='full')
    res = res.reshape((n_data + (n_weights - 1),))
    res = res[:n_data]
    return res

def black_friday(n, offset=13, period=52):
    res = np.zeros(n)
    for i in range(n):
        if i % period == offset:
            res[i] = 1
    return res


def create_baseline_mmm(media, channel_priors = CHANNEL_PRIORS):
    with pm.Model() as baseline_mmm:
        target = mean_normalize(media['revenue'])
        media_contributions = []
        for channel ,channel_prior in channel_priors.items():
             # define coefficient
            media_channel = pt.as_tensor_variable(media[channel])
            media_channel = mean_normalize(media_channel)
            scaled_channel = scale(media_channel, target)
            channel_coefficient = pm.TruncatedNormal(f"coefficient_{channel}", mu=channel_prior, sigma=0.1, lower=0, upper=0.5)
            channel_contribution = pm.Deterministic(f"contribution_{channel}", channel_coefficient * scaled_channel)
            media_contributions.append(channel_contribution)

        sigma = pm.Uniform("sigma", lower=0, upper=0.5)

        revenue = pm.Normal("revenue",
                    mu = sum(media_contributions),
                    sigma = sigma,
                    observed=target)
    
    return baseline_mmm

def create_mmm(media, channel_priors = CHANNEL_PRIORS):
    with pm.Model() as mmm:
        target = mean_normalize(media['revenue'])
        media_contributions = []
        for channel ,channel_prior in channel_priors.items():
             # define coefficient
            media_channel = pt.as_tensor_variable(media[channel])
            alpha = pm.Uniform(f"alpha_{channel}", lower=0.3, upper=0.9)
            L = 5 # fixed length
            theta = pm.DiscreteUniform(f"theta_{channel}", lower=0, upper=L-1)
            weights = pm.Deterministic(f'weights_{channel}',alpha ** ((pt.arange(L)) - theta) ** 2)
            media_transformed_channel = pm.Deterministic(f'media_transformed_{channel}', _apply_delayed_adstock_tensor(media_channel, weights))
            
            media_transformed_channel = mean_normalize(media_transformed_channel)
            scaled_channel = scale(media_transformed_channel, target) # Is this actually necessary?
            channel_coefficient = pm.TruncatedNormal(f"coefficient_{channel}", mu=channel_prior, sigma=0.1, lower=0, upper=0.5)
            channel_contribution = pm.Deterministic(f"contribution_{channel}", channel_coefficient * scaled_channel)
            media_contributions.append(channel_contribution)
        
        # trend component
        mean = target.mean()
        t = np.arange(len(target))
        t0 = t[len(target) // 2]
        capacity_coeff = pm.Normal("capacity_coef",mu=1.2, sigma=0.3)
        capacity = mean * capacity_coeff
        growth_rate = pm.TruncatedNormal("growth_rate", mu=-0.02, sigma=0.1, lower=-0.15, upper=0.0)
        trend = logistic_curve(t, capacity, growth_rate, t0)

        # cyclic year component

        cos_ceof = pm.Normal("cos_coef", mu=-1, sigma=0.3)
        sin_coef = pm.Normal("sin_coef", mu=2, sigma=0.3)
        BASE = 0
        period=52
        year = cyclic_component(t, BASE,cos_ceof,sin_coef,period=period)

        year_coeff = pm.Normal("year_coeff", mu=0.1, sigma=0.05)
        year = pm.Deterministic("year", year * year_coeff * mean)
        

        trend_year = pm.Deterministic("trend_year", trend + year)

        n = len(target)
        bf_index = black_friday(n)
        bf_coef = pm.Normal("black_friday_coef", mu=1, sigma=0.2)
        bf_contrib = pm.Deterministic("black_friday_contribution",bf_index*bf_coef*trend_year)
        trend_year += bf_contrib


        sigma = pm.Uniform("sigma", lower=0, upper=0.5)

        revenue = pm.Normal("revenue",
                    mu = trend_year  + sum(media_contributions),
                    sigma = sigma,
                    observed=target)
        
    return mmm


def prior_predictive(mmm):
    with mmm:
        prior_samples = pm.sample_prior_predictive(100)
    predicted = prior_samples.prior_predictive["revenue"].mean(axis=1)[0]
    return predicted

def posterior_predictive(mmm):
    with mmm:
        trace = pm.sample(tune=2000, idata_kwargs={"log_likelihood": True})
    
    posterior = pm.sample_posterior_predictive(trace, mmm)
    predictions = posterior['posterior_predictive']['revenue'].mean(axis=0).mean(axis=0)
    return trace, predictions

