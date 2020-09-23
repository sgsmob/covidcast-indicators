import sys
from os.path import join 
import re
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from datetime import date, datetime, timedelta
from .datafetcher import *
import math

import pdb

negated_regex_dict = {
    'county': '^(?!\d{5}).*$',
    'hrr': '^(?!\d{1,3}).*$',
    'msa': '^(?!\d{5}).*$',
    'state': '^(?![a-z]{2}).*$',
    'national': '(?!usa).*$'
}

class ValidationError(Exception):
    """ Error raised when validation check fails. """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def make_date_filter(start_date, end_date):
    start_code = int(start_date.strftime("%Y%m%d"))
    end_code = int(end_date.strftime("%Y%m%d"))
    def f(filename, match):
        if not match: return False
        code = int(match.groupdict()['date'])
        return code > start_code and code < end_code
    return f

def validate_daily(df_to_test, nameformat, max_check_lookbehind, generation_date):
    """
    Perform some automated format & sanity checks of inputs.
    
    Arguments:
        - df_to_test: pandas dataframe of CSV source data
        - nameformat: CSV name; for example, "20200624_county_smoothed_nohh_cmnty_cli.csv"
        - max_check_lookbehind: number of days back to perform sanity checks, starting from the last date appearing in df_to_test
        - generation_date: date that this df_to_test was generated; typically 1 day after the last date in df_to_test

    Returns:
        - None  
    """
    
    if (not isinstance(max_check_lookbehind, timedelta)):
        raise ValidationError(max_check_lookbehind, f"max_check_lookbehind ({max_check_lookbehind}) must be of type datetime.timedelta")

    if( not isinstance(generation_date, date) or generation_date > date.today()):
        raise ValidationError(generation_date, f"generation.date ({generation.date}) must be a datetime.date type and not in the future.")
    
    pattern_found = filename_regex.match(nameformat)
    if (not nameformat or not pattern_found):
        raise ValidationError(nameformat, 'nameformat ({nameformat}) not recognized')

    if not isinstance(df_to_test, pd.DataFrame):
        raise ValidationError(nameformat, 'df_to_test must be a pandas dataframe.')

    # TODO: check column names and types in df_to_test. Currently skipped since load_csv() specifies field names and types on read. Extra columns will simply be ignored during later processing.


def check_bad_geo_id(df_to_test, geo_type):
    """
    Check validity of geo type and values, according to regex pattern.
    
    Arguments:
        - df_to_test: pandas dataframe of CSV source data
        - geo_type: string from CSV name specifying geo type (state, county, etc) of data

    Returns:
        - None  
    """
    if geo_type not in negated_regex_dict:
        raise ValidationError(geo_type,"Unrecognized geo type")
    
    def find_all_unexpected_geo_ids(df_to_test, negated_regex):
        unexpected_geos = [ugeo[0] for ugeo in df_to_test['geo_id'].str.findall(negated_regex) if len(ugeo) > 0]
        if(len(unexpected_geos) > 0):
            raise ValidationError(unexpected_geos,"Non-conforming geo_ids exist!")
    
    find_all_unexpected_geo_ids(df_to_test, negated_regex_dict[geo_type])

def check_missing_dates(daily_filenames, sdate, edate):
    number_of_dates = edate - sdate + timedelta(days=1)
    date_seq = {sdate + timedelta(days=x) for x in range(number_of_dates.days)}
    unique_dates = set()

    for daily_filename in daily_filenames:
        unique_dates.add(datetime.strptime(daily_filename[0][0:8], '%Y%m%d'))

    check_dateholes = list(date_seq.difference(unique_dates))
    check_dateholes.sort()
    
    if check_dateholes:
        print("Missing dates are observed; if these dates are already in the API they would not be updated")
        print(check_dateholes)

def check_bad_val(df_to_test, signal_type):
    """
    Check value field for validity.
    
    Arguments:
        - df_to_test: pandas dataframe of CSV source data
        - signal_type: string from CSV name specifying signal type (smoothed_cli, etc) of data

    Returns:
        - None  
    """
    proportion_option = True if 'prop' in signal_type or 'pct' in signal_type else False

    if proportion_option:
        if (not df_to_test[(df_to_test['val'] > 100)].empty):
            raise ValidationError(signal_type, "val column can't have any cell greater than 100")

    if (df_to_test['val'].isnull().values.any()):
        raise ValidationError(None,"val column can't have any cell that is NA")
    
    if (not df_to_test[(df_to_test['val'] < 0)].empty):
        raise ValidationError(None,"val column can't have any cell smaller than 0")

def check_bad_se(df_to_test, missing_se_allowed):
    """
    Check standard errors for validity.
    
    Arguments:
        - df_to_test: pandas dataframe of CSV source data
        - missing_se_allowed: boolean specified in params.json

    Returns:
        - None  
    """
    df_to_test.eval('se_upper_limit = (val * sample_size + 50)/(sample_size + 1)', inplace=True)

    df_to_test['se']= df_to_test['se'].round(3)
    df_to_test['se_upper_limit'] = df_to_test['se_upper_limit'].round(3)

    if not missing_se_allowed:
        if (df_to_test['se'].isnull().values.any()):
            raise ValidationError(None, "se must not be NA")
        
        result = df_to_test.query('~((se > 0) & (se < 50) & (se <= se_upper_limit))')

        if not result.empty:
            raise ValidationError(None, "se must be in (0,min(50,val*(1+eps))]")     

    elif missing_se_allowed:
        result = df_to_test.query('~(se.isnull() | ((se > 0) & (se < 50) & (se <= se_upper_limit)))')

        if not result.empty:
            raise ValidationError(None, "se must be NA or in (0,min(50,val*(1+eps))]")
    
    result = df_to_test.query('(val == 0) & (se == 0)')

    if not result.empty:
        raise ValidationError(None, "when signal value is 0, se must be non-zero. please use Jeffreys correction to generate an appropriate se")     

def check_bad_sample_size(df_to_test, minimum_sample_size, missing_sample_size_allowed):
    if not missing_sample_size_allowed:
        if (df_to_test['sample_size'].isnull().values.any()):
            raise ValidationError(None, "sample_size must not be NA")
        
        result = df_to_test.query('(sample_size < @minimum_sample_size)')

        if not result.empty:
            raise ValidationError(None, "sample size must be >= {minimum_sample_size}")    

    elif missing_sample_size_allowed:
        result = df_to_test.query('~(sample_size.isnull() | (sample_size >= @minimum_sample_size))')

        if not result.empty:
            raise ValidationError(None, "sample size must be NA or >= {minimum_sample_size}")

def check_min_allowed_max_date(max_date, generation_date, weighted_option='unweighted'):
    switcher = {
        'unweighted': timedelta(days=1),
        'weighted': timedelta(days=4)
    }
    # Get the function from switcher dictionary
    thres = switcher.get(weighted_option, lambda: "Invalid weighting option")

    if (max_date < generation_date - thres):
        raise ValidationError(None, "latest date of generated file seems too long ago")

def check_max_allowed_max_date(max_date, generation_date):
    if (max_date < generation_date - timedelta(days=1)):
        raise ValidationError(None, "latest date of generated file seems too recent")

def reldiff_by_min(x, y):
    return (x - y) / min(x,y)

def check_rapid_change(df_to_test, df_to_compare, checking_date, date_list, sig, geo):
    """
    Compare number of obervations per day in test dataframe vs reference dataframe.
    
    Arguments:
        - df_to_test: pandas dataframe of "recent" CSV source data
        - df_to_compare: pandas dataframe of reference data, either from the COVIDcast API or semirecent data
        - checking_date
        - date_list
        - sig
        - geo

    Returns:
        - None  
    """
    test_rows_per_reporting_day = df_to_test[df_to_test['time_value'] == checking_date].shape[0]
    reference_rows_per_reporting_day = df_to_compare.shape[0] / len(date_list)
    
    if(abs(reldiff_by_min(test_rows_per_reporting_day, reference_rows_per_reporting_day)) > 0.35):
        raise ValidationError((checking_date,sig,geo), "Number of rows per day (-with-any-rows) seems to have changed rapidly (reference vs recent window of data)")

def check_avg_val_diffs(df_to_test, df_to_compare, smooth_option):
    """
    Compare average values in test dataframe vs reference dataframe.
    
    Arguments:
        - df_to_test: pandas dataframe of "recent" CSV source data
        - df_to_compare: pandas dataframe of reference data, either from the COVIDcast API or semirecent data
        - smooth_option: "raw" or "smoothed", choosen according to smoothing of signal (e.g. 7dav is "smoothed")

    Returns:
        - None  
    """
    # Average each of val, se, and sample_size over all dates for a given geo_id. Ignores NA values by default.
    df_to_test = df_to_test.groupby(['geo_id'], as_index=False)[['val', 'se', 'sample_size']].mean()
    df_to_test["type"] = "test"

    df_to_compare = df_to_compare.groupby(['geo_id'], as_index=False)[['val', 'se', 'sample_size']].mean()
    df_to_compare["type"] = "reference"

    df_all = pd.concat([df_to_compare, df_to_test])

    # For each variable type (val, se, and sample size) where not missing, calculate the relative mean difference and mean absolute difference between the test data and the reference data across all geographic regions.
    df_all = pd.melt(df_all, id_vars=["geo_id", "type"], value_vars=["val","se","sample_size"]).pivot(index=("geo_id", "variable"), columns="type", values="value").reset_index(("geo_id","variable")).dropna().assign(
            type_diff=lambda x: x["test"] - x["reference"],
            abs_type_diff=lambda x: abs(x["type_diff"])
        ).groupby("variable", as_index=False).agg(
            mean_type_diff=("type_diff", "mean"),
            mean_abs_type_diff=("abs_type_diff", "mean"),
            mean_test_var=("test", "mean"),
            mean_ref_var=("reference", "mean")
        ).assign(
            mean_stddiff=lambda x: 2 * x["mean_type_diff"] / (x["mean_test_var"] + x["mean_ref_var"]),
            mean_stdabsdiff=lambda x: 2 * x["mean_abs_type_diff"] / (x["mean_test_var"] + x["mean_ref_var"])
        )[["variable", "mean_stddiff", "mean_stdabsdiff"]]

    # Set thresholds for raw and smoothed variables.
    classes = ['mean_stddiff', 'val_mean_stddiff', 'mean_stdabsdiff']
    raw_thresholds = pd.DataFrame([0.50, 0.30, 0.80], classes).T

    smoothed_thresholds = raw_thresholds.apply(lambda x: x/(math.sqrt(7) * 1.5))

    switcher = {
    'raw': raw_thresholds,
    'smoothed': smoothed_thresholds,
    }

    # Get the selected thresholds from switcher dictionary
    thres = switcher.get(smooth_option, lambda: "Invalid smoothing option")

    # Check if the calculated mean differences are high, compared to the thresholds.
    mean_stddiff_high = (abs(df_all["mean_stddiff"]) > thres["mean_stddiff"]).bool() or ((df_all["variable"] == "val").bool() and (abs(df_all["mean_stddiff"]) > thres["val_mean_stddiff"]).bool())
    mean_stdabsdiff_high = (df_all["mean_stdabsdiff"] > thres["mean_stdabsdiff"]).bool()
    
    flag = mean_stddiff_high or mean_stdabsdiff_high

    if flag:
        raise ValidationError((mean_stddiff_high, mean_stdabsdiff_high), 'Average differences in variables by geo_id between recent & refernce data (either semirecent or from API) seem' \
              + 'large --- either large increase tending toward one direction or large mean absolute' \
              + 'difference, relative to average values of corresponding variables.  For the former' \
              + 'check, tolerances for `val` are more restrictive than those for other columns.')

def validate(export_dir, start_date, end_date, data_source, params, generation_date = date.today()):
    """
    Perform data checks.
    
    Arguments:

        - generation_date: date that this df_to_test was generated; typically 1 day after the last date in df_to_test
        - max_check_lookbehind: number of days back to perform sanity checks, starting from the last date appearing in df_to_test
        - sanity_check_rows_per_day
        - sanity_check_value_diffs: 
        - check_vs_working

    Returns:
        - None  
    """
    # Get user settings from params or if not provided, set default.
    max_check_lookbehind = timedelta(days=params.get("ref_window_size", 7)) 
    minimum_sample_size = params.get('minimum_sample_size', 100)
    missing_se_allowed = params.get('missing_se_allowed', False)
    missing_sample_size_allowed = params.get('missing_sample_size_allowed', False)

    sanity_check_rows_per_day = params.get('sanity_check_rows_per_day', True)
    sanity_check_value_diffs = params.get('sanity_check_value_diffs', True)
    check_vs_working = params.get('check_vs_working', True)


    export_files = read_filenames(export_dir)
    date_filter = make_date_filter(start_date, end_date)
    # List of tuples of CSV names and regex match objects.
    validate_files = [(f, m) for (f, m) in export_files if date_filter(f,m)]

    all_frames = []
    
    # TODO: What does unweighted vs weighted mean? 7dav vs not? Best place for these checks?
    check_min_allowed_max_date(end_date, generation_date, weighted_option='unweighted')
    check_max_allowed_max_date(end_date, generation_date)

    # First, check file formats
    check_missing_dates(validate_files, start_date, end_date)
    for filename, match in validate_files:
        df = load_csv(join(export_dir, filename))

        validate_daily(df, filename, max_check_lookbehind, generation_date)
        check_bad_geo_id(df, match.groupdict()['geo_type'])
        check_bad_val(df, match.groupdict()['signal'])
        check_bad_se(df, missing_se_allowed)
        check_bad_sample_size(df, minimum_sample_size, missing_sample_size_allowed)

        # Get geo_type, date, and signal name as specified by CSV name.
        df['geo_type'] = match.groupdict()['geo_type']
        df['date'] = match.groupdict()['date']
        df['signal'] = match.groupdict()['signal']

        # Add current CSV data to all_frames.
        all_frames.append(df)    

    # TODO: Multi-indexed dataframe for a given (signal, geo_type)
    all_frames = pd.concat(all_frames)
 
    # Get all expected combinations of geo_type and signal. 
    geo_sig_cmbo = get_geo_sig_cmbo(data_source)

    # Get list of dates we expect to see in the CSV data.
    date_slist = df['date'].unique().tolist()
    date_list = list(map(lambda x: datetime.strptime(x, '%Y%m%d'), date_slist))

    # Get list of CSV names.
    filenames = [name_match_pair[0] for name_match_pair in validate_files] 

    ## recent_lookbehind: start from the check date and working backward in time,
    ## how many days do we include in the window of date to check for anomalies?
    ## Choosing 1 day checks just the check data itself.
    recent_lookbehind = timedelta(days=1)

    ## semirecent_lookbehind: starting from the check date and working backward
    ## in time, how many days -- before subtracting out the "recent" days ---
    ## do we use to form the reference statistics?
    semirecent_lookbehind = timedelta(days=7)

    # # TODO: Check recent data against semirecent and API data.
    # start_checking_date = max(min(all_frames["date"]) + semirecent_lookbehind - 1, 
    #     max(all_frames["date"]) - max_check_lookbehind + 1)
    # end_checking_date = max(all_frames["date"])

    # if (start_checking_date > end_checking_date):
    #     raise ValidationError((start_checking_date, end_checking_date), "not enough days included in the dataframe to perform sanity checks")


    # for checking_date in range(start_checking_date, end_checking_date):
    #     known_irregularities = get_known_irregularities(checking_date, filename)

    #     recent_cutoff_date = checking_date - recent_lookbehind + 1
    #     semirecent_cutoff_date = checking_date - semirecent_lookbehind + 1

    #     recent_df_to_test = df_to_test.query('date <= @checking_date & date >= @recent_cutoff_date')
    #     semirecent_df_to_test = df_to_test.query('date <= @checking_date & date < @recent_cutoff_date & date >= @semirecent_cutoff_date')

    #     if (recent_df_to_test["se"].isnull().mean() > 0.5):
    #         print('Recent se values are >50% NA')

    #     if sanity_check_rows_per_day:
    #         check_rapid_change(recent_df_to_test, semirecent_df_to_test)

    #     recent_df_to_test["recency"] = "recent"
    #     semirecent_df_to_test["recency"] = "semirecent"

    #     recency_df = pd.concat([recent_df_to_test, semirecent_df_to_test])

    #     # TODO: Continue with check_avg_val_diffs() here.





    smooth_option_regex = re.compile(r'([^_]+)')

    kroc = 0

    # TODO: Improve efficiency by grouping all_frames by geo and sig instead of reading data in again via read_geo_sig_cmbo_files().
    for recent_df, geo, sig in read_geo_sig_cmbo_files(geo_sig_cmbo, export_dir, filenames, date_slist):
         
        m = smooth_option_regex.match(sig)
        smooth_option = m.group(1)

        if smooth_option not in ('raw', 'smoothed'):
            smooth_option = 'smoothed' if '7dav' in sig or 'smoothed' in sig else 'raw'
        
        #recent_df.set_index("time_value", inplace = True)
        print("Printing recent_df scenes:", recent_df.shape)
        print(recent_df)
        for checking_date in date_list:
            #print(recent_df.loc[checking_date,:])
            # -recent- dataframe run backwards from the checking_date
            recent_end_date = checking_date - recent_lookbehind
            recent_begin_date = checking_date - max_check_lookbehind
            recent_api_df = covidcast.signal(data_source, sig, recent_begin_date, recent_end_date, geo)

            # Replace None with NA to make numerical manipulation easier.
            recent_api_df.replace(to_replace=[None], value=np.nan, inplace=True) 

            # Rename columns.
            recent_api_df.rename(columns={'geo_value': "geo_id", 'stderr': 'se', 'value': 'val'}, inplace = True)
            recent_api_df.drop(['direction', 'issue', 'lag'], axis=1, inplace = True)
            
            column_names = ["geo_id", "val", "se", "sample_size", "time_value"]

            recent_api_df = recent_api_df.reindex(columns=column_names)

            if (recent_df["se"].isnull().mean() > 0.5):
                print('Recent se values are >50% NA')

            if sanity_check_rows_per_day:
                check_rapid_change(recent_df, recent_api_df, checking_date, date_list, sig, geo)

            if sanity_check_value_diffs:
                check_avg_val_diffs(recent_df, recent_api_df, smooth_option)

        kroc += 1
        if kroc == 2:  
            break
    sys.exit()
    

