import numpy as np
import pandas as pd
from scipy import stats


def ttest_season(table, variables, class_column, classes, get_greater=False, get_percentage=False, logger=None):
    """Performs two sample t-test for each variable in each season and returns the result in a DataFrame.

    This function splits the input data based on the specified classes, performs two-sample t-test on each variable in each season, and returns the p-value(s) in a DataFrame.
    The t-test compares the means of two independent samples to determine if there is a significant difference between their means.

    Args:
        table (pandas DataFrame): The input data.
        variables (list of dictionaries): List of dictionaries where each dictionary represents a variable and its properties.
        class_column (str): Column name representing the class label in the input data.
        classes (list of str): List of two class labels to split the input data into.
        get_greater (bool, optional): If True, returns the class with greater mean for each variable in each season along with the p-value. Default is False.
        get_percentage (bool, optional): If True, returns the p-value in percentage. Default is False.
        logger (logging.Logger, optional): The logger instance. If None, logging is disabled.

    Returns:
        pd.DataFrame: A DataFrame containing the p-value(s) for each variable in each season.
        If `get_greater` is True, the class with greater mean is also returned.
        If `get_percentage` is True, p-values are returned in percentage.

    """

    a = table[table[class_column] == classes[0]]
    b = table[table[class_column] == classes[1]]
    logger.debug("%s shape = %s", classes[0], str(a.shape)) if logger is not None else None
    logger.debug("%s shape = %s", classes[1], str(b.shape)) if logger is not None else None

    seasons = set(table["Season"])
    logger.debug(seasons) if logger is not None else None

    output = pd.DataFrame(columns=["Variable", *seasons])
    logger.debug("\n%s", output.to_string()) if logger is not None else None

    for element in variables:
        var = next(iter(element.keys()))
        flatten = next(iter(element.values()))
        
        features = [col for col in table if col.startswith(var)]
        logger.debug("%s shape is = %s", var, str(table[features].shape)) if logger is not None else None
        
        pvalues = {"Variable": var}
        logger.debug(pvalues) if logger is not None else None
        
        for season in seasons:
            if(flatten == True):
                a_features = a[a.Season == season][features].values.flatten()
                b_features = b[b.Season == season][features].values.flatten()
            else:
                a_features = a[a.Season == season][features]
                b_features = b[b.Season == season][features]
            
            
            logger.debug("%s in %s for %s shape = %s", classes[0], season, var, str(a_features.shape)) if logger is not None else None
            logger.debug("%s in %s for %s shape = %s", classes[1], season, var, str(b_features.shape)) if logger is not None else None

            _, pvalue = stats.ttest_ind(a_features, b_features, equal_var=False, nan_policy='omit')                    
            logger.debug ("%s in %s = %s before fix_zero", var, season, pvalue) if logger is not None else None
            pvalue = fix_zero_value(np.array(pvalue), a_features, b_features, logger=logger, variable=var)
            logger.info ("%s in %s = %s", var, season, pvalue) if logger is not None else None
            
            if get_percentage:
                pvalue = find_percentage(pvalue)

            if get_greater:
                greater = find_greater(a_features, b_features, classes, logger)
                logger.debug ("%s is greater", greater) if logger is not None else None
                result = pvalue, greater
            else:
                result = pvalue

            pvalues[season] = result
            
        logger.debug(pvalues) if logger is not None else None
        output = pd.concat([output, pd.DataFrame([pvalues])], ignore_index=True) # non serve più
        logger.debug("\n%s", output.to_string()) if logger is not None else None

    return output

def fix_zero_value(pvalue, a, b, logger=None, variable=None):
    """Fix zero value of a given p-value.

    This function will replace zero values of a p-value with a minimum value from a t-test array, computed using two input arrays `a` and `b`.
    If the p-value is a simple array, it will return the minimum value.
    If it's multidimensional, it will replace the zero value with the minimum value for each corresponding element in the p-value array.

    Args:
        pvalue (numpy.ndarray): The p-value array to be fixed.
        a (numpy.ndarray): The first input array for t-test computation.
        b (numpy.ndarray): The second input array for t-test computation.
        logger (logging.Logger, optional): A logging instance to log messages. If None, no logging will occur. Defaults to None.
        variable (str, optional): The name of the variable, used for logging purposes. Defaults to None.

    Returns:
        numpy.ndarray: The fixed p-value array.
    """
    if pvalue.size > 1:
        logger.debug("%s is multidimensional" % (variable)) if logger is not None else None
        
        if (0 in pvalue):
            logger.debug(pvalue) if logger is not None else None

            for element in np.where(pvalue == 0):
                logger.debug(element) if logger is not None else None

                ttest_array = [stats.ttest_ind(a, b[:i], equal_var=False, nan_policy='omit')[1] for i in range(200, b.shape[0], 200)]
                logger.debug("ttest_array: %s" % (ttest_array)) if logger is not None else None
                
                tmp = list(filter(lambda num: num != 0, ttest_array))
                logger.debug("tmp: %s" % (tmp)) if logger is not None else None

                pvalue[element] = min(tmp) if len(tmp) > 0 else np.nan
    else:
        logger.debug("%s is simple array" % (variable)) if logger  is not None else None
        
        if (pvalue == 0): 
            ttest_array = [stats.ttest_ind(a, b[:i], equal_var=False, nan_policy='omit')[1] for i in range(200, b.shape[0], 200)]
            logger.debug("ttest_array: %s" % (ttest_array)) if logger is not None else None
            
            tmp = list(filter(lambda num: num != 0, ttest_array))
            logger.debug("tmp: %s" % (tmp)) if logger is not None else None

            pvalue = min(tmp) if len(tmp) > 0 else np.nan

    return pvalue

def find_greater(a, b, behaviours, logger=None):
    """Compares two arrays to find which one is greater according to a t-test.

    This function takes two arrays and two behaviours, and performs a two-tailed t-test to determine which of the arrays is greater, if any.
    If the p-value of the test is below a threshold (default 0.05), the function returns the behaviour corresponding to the greater array.

    Args:
        a (np.array): The first input array.
        b (np.array): The second input array.
        behaviours (list of str): A list of two strings that represent the two behaviours being compared.
        logger (logging.Logger, optional): A logger to log debug information. Default is None.

    Returns:
        str: The behaviour corresponding to the greater array, or None if both arrays are equal (p-value is greater than or equal to the threshold).
    """
    result = None

    _, a_is_greater = stats.ttest_ind(a, b,  equal_var=False, nan_policy='omit', alternative='greater')
    _, b_is_greater = stats.ttest_ind(a, b,  equal_var=False, nan_policy='omit', alternative='less')

    a_is_greater = fix_zero_value(np.array(a_is_greater), a, b, logger)
    b_is_greater = fix_zero_value(np.array(b_is_greater), a, b, logger)

    minimum = np.min([a_is_greater, b_is_greater])

    if minimum < 0.05:
        if a_is_greater == minimum:
            logger.debug("a > b") if logger is not None else None
            result = behaviours[0]
        else:
            logger.debug("b > a") if logger is not None else None
            result = behaviours[1]
    
    logger.debug("greater is = %s", result) if logger is not None else None
    return result

def find_percentage(array, threshold=0.05):
    """Calculates the percentage of values in an array below a threshold.

    This function takes an input array and a threshold and returns the percentage of values in the array that are below the threshold.

    Args:
        array (np.array): The input array.
        threshold (float, optional): The threshold value. Default is 0.05.

    Returns:
        float: The percentage of values in the input array below the threshold.
    """
    count = 0
    for value in array:
        if value < threshold:
            count += 1
    return count / len(array) * 100