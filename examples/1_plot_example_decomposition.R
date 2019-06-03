library(tidyverse)

df <- read_csv("output/toy_decomposition.csv")

# plot overall prediction
df %>%
  ggplot(aes(z, f, col=x, group=x)) +
  geom_path() +
  scale_color_viridis_c() +
  theme_classic() +
  labs(title = "Aggregate c-GPLVM mapping")

# now plot decomposition

# f_z
df %>%
  ggplot(aes(z, f_z, group=x)) +
  geom_path() +
  theme_classic() +
  labs(title = "Decomposition: f(z)")

# f_x
df %>%
  ggplot(aes(z, f_x, col=x, group=x)) +
  geom_path() +
  scale_color_viridis_c() +
  theme_classic() +
  labs(title = "Decomposition: f(x)")

# interaction f_{zx}
df %>%
  ggplot(aes(z, f_int, col=x, group=x)) +
  geom_path() +
  scale_color_viridis_c() +
  theme_classic() +
  labs(title = "Decomposition: f(z, x)")
