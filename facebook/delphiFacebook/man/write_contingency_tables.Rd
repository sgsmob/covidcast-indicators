% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/contingency_write.R
\name{write_contingency_tables}
\alias{write_contingency_tables}
\title{Write csv file for sharing with researchers.}
\usage{
write_contingency_tables(data, params, geo_level, groupby_vars)
}
\arguments{
\item{data}{a data frame to save; must contain the columns "geo_id", "val",
"se", "sample_size", and grouping variables. The first four are saved in the
output; day is used for spliting the data into files.}

\item{params}{a named list, containing the value "export_dir" indicating the
directory where the csv should be saved}

\item{geo_level}{name of the geographic level; used for naming the output file}

\item{groupby_vars}{character vector of column names used for grouping to
calculate aggregations; used for naming the output file}
}
\description{
CSV name includes date specifying start of time period aggregated, geo level,
and grouping variables.
}
