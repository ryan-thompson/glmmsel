% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glmmsel.R
\name{predict.glmmsel}
\alias{predict.glmmsel}
\title{Predict function for \code{glmmsel} object}
\usage{
\method{predict}{glmmsel}(object, x.new, cluster.new, lambda = NULL, ...)
}
\arguments{
\item{object}{an object of class \code{glmmsel}}

\item{x.new}{a matrix of new values for the predictors}

\item{cluster.new}{a vector identifying the clusters that the rows of \code{x.new} belong to}

\item{lambda}{a value of the regularisation parameter}

\item{...}{any other arguments}
}
\value{
A matrix of predictions.
}
\description{
Generates predictions for new data using a specified value of the regularisation
parameter.
}
\author{
Ryan Thompson <ryan.thompson-1@uts.edu.au>
}
