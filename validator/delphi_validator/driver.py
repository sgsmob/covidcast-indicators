from fbsurveyvalidation import *
from datafetcher import *

# Defining start date and end date for the last fb-survey pipeline execution
survey_sdate = "2020-06-13"
survey_edate = "2020-06-15"
dtobj_sdate = datetime.strptime(survey_sdate, '%Y-%m-%d')
dtobj_edate = datetime.strptime(survey_edate, '%Y-%m-%d')
print(dtobj_sdate.date())
print(dtobj_edate.date())


# Collecting all filenames
daily_filnames = read_filenames("../data")

fbsurvey_validation(daily_filnames, dtobj_sdate, dtobj_edate)
