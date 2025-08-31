my_packages <- c("dplyr", "ggplot2")
install_if_missing <- function(p) {
  if (!p %in% rownames(installed.packages())) {
    install.packages(p, clean=TRUE, quiet=TRUE)
  }
}
invisible(sapply(my_packages, install_if_missing))