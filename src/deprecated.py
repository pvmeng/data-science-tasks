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




def _delayed_adstock(alpha, theta, L):
    return alpha**((np.ones(L).cumsum()-1)-theta)**2

def _apply_delayed_adstock(data, weights):
    # Ensure weights sum to 1
    weights /= weights.sum()
    # Calculate the weighted sum using convolution
    adstocked_spend = np.convolve(data, weights[::-1], mode='full')[:len(data)]
    
    return adstocked_spend