library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)

losses_real <- read_csv("finetune_HSIx2_val3_losses.csv", show_col_types=FALSE)
losses_synth <- read_csv("finetune_HSIx2_val3_synth_losses.csv", show_col_types=FALSE)


df_real <- losses_real %>%
  mutate(model="Real") %>%
  pivot_longer(cols=c(2:5),
               names_to="type",
               values_to="value")

df_synth <- losses_synth %>%
  mutate(model="Synthetic") %>%
  pivot_longer(cols=c(2:5),
               names_to="type",
               values_to="value")





df <- rbind(df_real, df_synth)

# Drop adv_loss
df2 <- df %>%
  filter(!(type=="adv_loss")) %>%
  mutate(type=recode(type, 
                     "pix_loss"="(a) Pixel Loss", 
                     "perc_loss"="(b) Perceptual Loss",
                     "psnr"="(c) PSNR")) %>%
  rename("Epoch"=epoch,
         "Model"=model,
         "Value"=value) %>%
  filter(Epoch<=2000)


ggplot(df2, aes(x=Epoch, y=Value, color=Model)) +
  geom_smooth(method="loess", se=FALSE, span=0.035, na.rm=TRUE) +
  facet_wrap(~type, scales="free_y", ncol=3) +
  theme(
    strip.text.x = element_text(size=12, color='#000000', margin=margin(b=8)),
    strip.background = element_rect(fill=NA),
    axis.title.x = element_text(size=12, color='#000000', margin=margin(t=8)),
    axis.title.y = element_text(size=12, color='#000000', margin=margin(r=8)),
    axis.ticks = element_line(color='#000000'),
    panel.border = element_rect(color='#000000', fill=NA),
    panel.background = element_rect(fill=NA),
    axis.text=element_text(size=9, color='#000000')
  )


