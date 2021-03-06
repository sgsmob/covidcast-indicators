% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/contingency_variables.R
\name{make_human_readable}
\alias{make_human_readable}
\title{Rename question codes to informative descriptions.}
\usage{
make_human_readable(input_data)
}
\arguments{
\item{input_data}{Data frame of individual response data}
}
\value{
Data frame with descriptive column names
}
\description{
Column names beginning with "b_" are binary (T/F/NA); with "t_" are user-
entered text; with "n_" are user-entered numeric; with "mc_" are multiple
choice (where only a single response can be selected); and with "ms_" are
so-called multi-select, where multiple responses can be selected.
}
\details{
Only binary responses with a third "I don't know" option are mapped from
response codes to interpretable values. Multiple choice, multi-select, and
pure binary (yes/no) questions use the original numeric response codes.
}
