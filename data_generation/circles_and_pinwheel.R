library(tidyverse)
library(patchwork)

## circles
generate_circles <- function(path, N = 1000){
  set.seed(0)
  z <- runif(N, 0, 1.8*pi)
  x <- sample(seq(-pi, pi, length=6)[-1], N, replace=TRUE)
  y1 <- cos(z) + 1.4*cos(x) + 0.05*rnorm(N)
  y2 <- sin(z) + 1.4*sin(x) + 0.05*rnorm(N)
  df_data <- data.frame(y1, y2, z_true = as.numeric(scale(z)), x = as.numeric(scale(x)))
  write_csv(df_data, path)
}

generate_distorted_circles <- function(path, N = 1000){
  N <- 1000
  set.seed(0)
  z <- runif(N, 0, 1.8*pi)
  x <- sample(seq(-pi, pi, length=6)[-1], N, replace=TRUE)
  y1 <- cos(z) + 1.4*cos(x) + 0.05*rnorm(N)
  y2 <- sin(z) + 1.4*sin(x) + sin(z)*sin(0.25*x) + 0.05*rnorm(N)
  df_data <- data.frame(y1, y2, z_true = as.numeric(scale(z)), x = as.numeric(scale(x)))
  write_csv(df_data, path)
}

## pinwheel
generate_pinwheel <- function(path, N = 1000){
  set.seed(0)
  n_classes <- 5
  rate <- 0.25
  rads <- 0.0 + seq(0, 2*pi, length=n_classes+1)[-1]
  # features <- cbind(rnorm(N, 1, 0.3), rnorm(N, 0, 0.05))
  cluster <- sample(c(1, 2, 3), N, replace=TRUE)
  features_x <- runif(N, 0.1, 1.7)
  # features_x <- runif(N, c(0.1, 0.8, 1.4)[cluster], c(0.5, 1.2, 1.7)[cluster])
  features_y <- runif(N, -0.1, 0.1)
  features <- cbind(features_x, features_y)
  angles0 <- sort(sample(rads, N, replace = TRUE))
  angles <- angles0 + rate * exp(features[, 1])
  Y <- matrix(0, N, 2)
  for(i in 1:N){
    rotations <- rbind(c(cos(angles[i]), -sin(angles[i])), c(sin(angles[i]), cos(angles[i])))
    Y[i, ] <- features[i, ] %*% rotations
  }
  covariate <- as.numeric(scale(angles0))
  
  df_data <- data.frame(y1 = Y[, 1], y2 = Y[, 2], x = covariate, z_true = as.numeric(scale(features_x)))
  write_csv(df_data, path)
}


generate_circles("data/circles.csv")
generate_pinwheel("data/pinwheel.csv")


df_data <- read_csv("data/pinwheel.csv")
# df_data <- read_csv("data/circles.csv")

p1 <- df_data %>%
  ggplot(aes(z, y1, col=x)) +
  geom_point() +
  scale_color_viridis_c() +
  labs(title = "(z, y1)")
p2 <- df_data %>%
  ggplot(aes(z, y2, col=x)) +
  geom_point() +
  scale_color_viridis_c() +
  labs(title = "(z, y2)")
p0 <- df_data %>%
  ggplot(aes(y1, y2, col=x)) +
  geom_point() +
  scale_color_viridis_c() +
  labs(title = "(y1, y2)")

p1 + p2 + p0 + plot_layout(widths = c(1, 1, 1.5))
