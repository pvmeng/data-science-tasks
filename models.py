import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

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

def black_friday(x, coef, offset=13):
    res = np.zeros(len(x))
    for i in range(len(x)):
        if i % 52 == offset:
            res[i] = coef * x[i]
    return res

def _delayed_adstock(alpha, theta, L):
    return alpha**((np.ones(L).cumsum()-1)-theta)**2

def _apply_delayed_adstock(spend_data, weights):
    # Ensure weights sum to 1
    weights /= weights.sum()
    # Calculate the weighted sum using convolution
    adstocked_spend = np.convolve(spend_data, weights[::-1], mode='full')[:len(spend_data)]
    return adstocked_spend

def model_trend(media):
    channel_priors = CHANNEL_PRIORS
    with pm.Model() as mmm:
        # target = media['revenue'] / media['revenue'].mean()
        target = mean_normalize(media['revenue'])
        media_contributions = []
        for channel ,channel_prior in channel_priors.items():
             # define coefficient
            alpha = pm.Uniform(f"alpha_{channel}", lower=0.3, upper=0.9)
            # L = pm.DiscreteUniform(f"L_{channel}", lower=2, upper=6)
            L = 5
            theta = pm.DiscreteUniform(f"theta_{channel}", lower=0, upper=L)
            weights = pm.Deterministic(f'weights_{channel}',alpha ** ((pt.arange(L)) - theta) ** 2)
            media = pt.as_tensor_variable(media[channel])
            media_transformed = pm.Deterministic(f'media_transformed_{channel}', pm.math.conv2d(media[None, :], weights[::-1][None, :], border_mode='full')[0, :len(media)])
            media_channel = mean_normalize(media_transformed)


            media_channel = pt.as_tensor_variable(media_channel)
            scaled_channel = scale(media_channel, target) # Is this actually necessary?
            channel_coefficient = pm.TruncatedNormal(f"coefficient_{channel}", mu=channel_prior, sigma=0.1, lower=0, upper=0.5)
            channel_contribution = pm.Deterministic(f"contribution_{channel}", channel_coefficient * scaled_channel)
            media_contributions.append(channel_contribution)
        
        # trend component
        mean = target.mean()
        t = np.arange(len(target))
        t0 = t[len(target) // 2]
        capacity_coeff = pm.Normal("capacity_coef",mu=1.6, sigma=0.2)
        capacity = mean * capacity_coeff
        growth_rate = pm.TruncatedNormal("growth_rate", mu=-0.02, sigma=0.1, lower=-0.15, upper=0.0)
        trend = logistic_curve(t, capacity, growth_rate, t0)

        # cyclic year component
        BASE = 0
        cos_ceof = pm.Normal("cos_coef", mu=0, sigma=0.5)
        sin_coef = pm.Normal("sin_coef", mu=5.0, sigma=1)
        period=52
        year = cyclic_component(t, BASE,cos_ceof,sin_coef,period=period)

        year_coeff = pm.Normal("year_coeff", mu=0.1, sigma=0.05)
        year = pm.Deterministic("year", year * year_coeff * mean)
        

        trend_year = pm.Deterministic("trend_year", trend + year)
        bf = black_friday(t, 0.01)
        trend_year += bf


        sigma = pm.Uniform("sigma", lower=0, upper=0.5)

        revenue = pm.Normal("revenue",                               
                                    mu = trend_year  + sum(media_contributions),
                                    sigma = sigma,
                                    observed=target)
        
    return mmm



def shift_numpy_array(arr, lag):
    if lag <= 0:
        return arr
    # Create an array of zeros with length `lag`

    lag_fill = np.full(lag, np.mean(arr[:lag])*0.5)
    # Concatenate the lag zeros with the truncated original array
    shifted_arr = np.concatenate((lag_fill, arr[:-lag]))
    
    return shifted_arr

# geometric decay
def adstock_shifted(series, decay_rate, retention_length, shift, linear_shift=True):
    result = np.zeros(len(series))
    if linear_shift:
        series = shift_numpy_array(series, shift)
    for i in range(1, len(series)):

        retention_values = np.array([series[j] for j in range(max(1, i - retention_length+1), i+1)])
        if linear_shift:
            retention_weights = np.array([decay_rate**i for i in range(min(i, retention_length))])[::-1]
        else:
            retention_weights = np.array([decay_rate**((i-shift)**2) for i in range(min(i, retention_length))])[::-1]
        result[i] = np.sum(retention_values * retention_weights)
    return result

def delay(media, lag=5, retention_rate=0.8, retention_length=4, linear_shift=True):
    media_transformed = media.copy()
    for channel in CHANNEL_PRIORS.keys():
        media_transformed[channel] = adstock_shifted(media[channel], retention_rate, retention_length, lag, linear_shift=linear_shift)
    return media_transformed

def _delayed_adstock(alpha, theta, L):
    return alpha**((np.ones(L).cumsum()-1)-theta)**2

def _apply_delayed_adstock(spend_data, weights):
    # Ensure weights sum to 1
    weights /= weights.sum()
    # Calculate the weighted sum using convolution
    adstocked_spend = np.convolve(spend_data, weights[::-1], mode='full')[:len(spend_data)]
    return adstocked_spend

def delayed_adstock(media, alpha=0.8, theta=2, L=5):
    weights = _delayed_adstock(alpha, theta, L)
    media_transformed = media.copy()
    for channel in CHANNEL_PRIORS.keys():
        media_transformed[channel] = _apply_delayed_adstock(media[channel], weights)
    return media_transformed


def load_media():
    media = pd.read_csv('data/MMM_test_data.csv', )
    media["start_of_week"] = pd.to_datetime(media["start_of_week"], format="%d-%m-%y")
    # 24-07-22
    media.sort_values(by="start_of_week", inplace=True)
    media.reset_index(drop=True, inplace=True)
    return media

def plot_media(media):
    return media.plot(x='start_of_week', title='MMM test data transformed', legend=True)

def plot_prior_predictive(mmm, media):
    with mmm:
        prior_samples = pm.sample_prior_predictive(100)
        
    observed = media['revenue'] / media['revenue'].mean()
    predicted = prior_samples.prior_predictive["revenue"].mean(axis=1)[0]

    dates = media['start_of_week']

    plt.figure(figsize=(10, 6))
    plt.plot(observed, label='Observed')
    plt.plot(predicted, label='Predicted')
    plt.xticks(np.arange(0, len(dates), 10), dates.dt.date[::10], rotation=45)
    plt.ylabel('Revenue')
    plt.title('Observed vs Predicted')
    plt.legend()
    plt.show()

def plot_posterior(mmm, media, channel_priors=CHANNEL_PRIORS):
    with mmm:
        trace = pm.sample(tune=2000)
    
    posterior = pm.sample_posterior_predictive(trace, mmm)
    predictions = posterior['posterior_predictive']['revenue'].mean(axis=0).mean(axis=0) * media['revenue'].mean()

    dates = media['start_of_week']

    # media_decomp = pd.DataFrame({i:np.array(trace['posterior']["contribution_"+str(i)]).mean(axis=(0,1)) for i in channel_priors.keys()}, index=dates) * media['revenue'].mean()

    plt.plot(media['revenue'])
    plt.plot(predictions)
    # xticks 
    plt.xticks(np.arange(0, len(dates), 10), dates.dt.date[::10], rotation=45)
    plt.title("Model Fit")
    
    plt.show()

    summary = az.summary(trace, round_to=2)
    return summary