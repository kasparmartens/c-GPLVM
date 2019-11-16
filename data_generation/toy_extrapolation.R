library(tidyverse)

generate_toy_data <- function(N = 500){
  z <- runif(N, -3, 3)
  c <- sample(c(-1, 0, 1), N, TRUE)
  
  y <- sin(z) + c + 0.1*rnorm(N)
  
  data.frame(y, z, c) %>%
    filter(!(c == 1 & z > -1.5), !(c == 0 & z > -0.5))
}

df <- generate_toy_data(250)

write_csv(df, "data/extrapolation.csv")

df %>% 
  ggplot(aes(z, y, col=c)) +
  geom_point() +
  scale_color_viridis_c() +
  theme_classic()
