% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/contingency_variables.R
\name{remap_response}
\alias{remap_response}
\title{Convert numeric response codes in a single survey item to values specified in
map. Returns as-is for numeric columns.}
\usage{
remap_response(df, col_var, map_old_new, default = NULL, response_type = "b")
}
\arguments{
\item{df}{Data frame of individual response data.}

\item{col_var}{Name of response var to recode}

\item{map_old_new}{Named vector of new values we want to use; names are the
original response codes}

\item{default}{Default to use if value is not explicitly remapped in
`map_old_new`; often `NA`, `NA_character_`, etc. See `recode`
[documentation](https://rdrr.io/cran/dplyr/man/recode.html) for more info}

\item{response_type}{Str indicating if response is binary, multiple choice, or
multi-select.}
}
\value{
list of data frame of individual response data with newly mapped column
}
\description{
Maps for recoding are set manually in `remap_responses`.
}
