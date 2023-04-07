### LOAD PACKAGES

library(tidyverse)
library(readxl)
library(readr)
library(pacman)
library(shadowtext)
library(ggrepel)
library(sf)
library("rnaturalearth")
library("rnaturalearthdata")
library(viridis)
library(shadowtext)

install.packages('devtools')
devtools::install_github('bbc/bbplot')
library('bbplot')

pacman::p_load('dplyr', 'tidyr', 'gapminder',
               'ggplot2',  'ggalt',
               'forcats', 'R.utils', 'png', 
               'grid', 'ggpubr', 'scales',
               'bbplot')


SAVE_ici<-TRUE

### LOAD THE DATA

villes_gis<-st_read('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/code_R/data/CentersShp_4326.shp') %>% 
  mutate(Continent = str_to_title(str_replace(Continent, "_", " ")))

#run results_analysis/results_analysis.py first
results <-read_excel('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/results_20221221.xlsx') #test_congestion2.xlsx, robustness_20230106.xlsx

villes<-results %>% 
  left_join(villes_gis,by=c('City')) %>% 
  st_as_sf()

### DEFINE FUNCTIONS TO PLOT THE MAPS AND DRAW THE CHARTS

vincent_style<-function (position_legend="top",position_titre_horiz=0,position_titre_vertic=0) {
  font <- "Helvetica"
  ggplot2::theme(
    plot.title = ggplot2::element_text(family = font,
                                       size = 28, face = "bold", color = "#222222",hjust = position_titre_horiz,vjust = position_titre_vertic), 
    plot.subtitle = ggplot2::element_text(family = font,
                                          size = 22, margin = ggplot2::margin(9, 0, 9, 0)), 
    plot.caption = ggplot2::element_blank(),
    legend.position = position_legend, 
    legend.text.align = 0, 
    legend.background = ggplot2::element_blank(),
    legend.title = ggplot2::element_blank(), 
    legend.key = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(family = font, size = 18,
                                        color = "#222222"), 
    axis.title = ggplot2::element_blank(),
    axis.text = ggplot2::element_blank(), 
    axis.ticks = ggplot2::element_blank(),
    axis.line = ggplot2::element_blank(), 
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major = ggplot2::element_blank(),
    panel.background = ggplot2::element_rect(fill = "aliceblue"),
    strip.background = ggplot2::element_rect(fill = "white"),
    strip.text = ggplot2::element_text(size = 22, hjust = 0),
    plot.margin = unit(c(0,0,0,0),"mm"))
}

vincent_style_zoom<-function () {
  ggplot2::theme(axis.title.x=element_blank(),
                 axis.text.x=element_blank(),
                 axis.ticks.x=element_blank(),
                 axis.title.y=element_blank(),
                 axis.text.y=element_blank(),
                 axis.ticks.y=element_blank(),
                 panel.grid.minor = ggplot2::element_blank(),
                 panel.grid.major = ggplot2::element_blank(),
                 panel.background = ggplot2::element_rect(color='black',fill = "aliceblue"),
                 panel.border = ggplot2::element_rect(colour = "black", fill=NA, size=0.5),
                 plot.margin = unit(c(0,0,-6,-6),"mm"))
}

desat <- function(cols, sat=0.5) {
  X <- diag(c(1, sat, 1)) %*% rgb2hsv(col2rgb(cols))
  hsv(X[1,], X[2,], X[3,])
}

world <- ne_countries(scale = "medium", returnclass = "sf")

draw_my_map<-function(villes_ici,title_ici="Test",subtitle_ici="",myfunc,funcargs=list(),SAVE_ici=TRUE,position_legend="top",position_titre_horiz=0,position_titre_vertic=0,nom_specifique_fichier=NULL,texte_en_plus_nom_fichier="")
{
  if (is.null(nom_specifique_fichier))
  {nom_specifique_fichier<-title_ici}
  
  world_map<-ggplot(data=world) + 
    vincent_style(position_legend=position_legend,position_titre_horiz=position_titre_horiz,position_titre_vertic=position_titre_vertic)+
    geom_sf(fill = "antiquewhite1", lwd=0.1)+
    geom_point(data=villes_ici,shape=21,alpha=0.9,size=3,
               aes(fill=a_tracer, geometry = geometry),color='grey60',
               stat = "sf_coordinates"
    ) +
    #geom_sf(data=villes,aes(color=Continent))+
    coord_sf(ylim = c(-65,90), expand = FALSE)+  
    xlab('') + ylab('') +
    labs(title=title_ici,subtitle=subtitle_ici) +
    do.call(myfunc, funcargs)
  
  zoom_europe<-ggplot(data=world) + 
    geom_sf(fill = "antiquewhite1", lwd=0.1) + 
    geom_point(data=villes_ici,shape=21,alpha=0.9,size=3,
               aes(fill=a_tracer, geometry = geometry),color='grey60',
               stat = "sf_coordinates",
               show.legend = FALSE
    ) +
    coord_sf(ylim = c(35,60), xlim = c(-10,30), expand = FALSE) + #Europe
    xlab('') + ylab('') +
    vincent_style_zoom()+
    do.call(myfunc, funcargs)
  
  world_with_Europe <- ggplotGrob(zoom_europe)
  world_with_Europe<-world_map+
    annotation_custom(grob = world_with_Europe, 
                      xmin = -165, xmax = -95 , ymin = -65 , ymax = 30)
  
  plot(world_with_Europe,
       width = 27, 
       height = 11, 
       units = "cm")
  
  if (SAVE_ici) {
    ggsave(paste0("./outputs/",nom_specifique_fichier,texte_en_plus_nom_fichier,".png"),plot=world_with_Europe,
           width = 2666/300,
           height= 1675/300,
           unit="in")
  }
}

draw_graph<-function(villes_ici,title_ici="test",SAVE_ici=TRUE,nom_specifique_fichier=NULL)
{
  if (is.null(nom_specifique_fichier))
  {nom_specifique_fichier<-title_ici}
  
  p <- ggplot(villes_ici, aes(y = emissions_2035_a_tracer_var, x = welfare_2035_a_tracer_var, fill = Continent)) +
    geom_point(size = 5,alpha=0.8,color='grey25',shape=21) +
    geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
    geom_vline(xintercept = 0, size = 0.5, colour="#333333") +
    #scale_fill_manual(values = country_colors) +
    bbc_style() +
    labs(title=title_ici,
         subtitle = "")+
    theme(
      panel.grid.major.x = element_line(color="#cbcbcb"),
      axis.title = ggplot2::element_text(family = "Helvetica", size = 18, color = "#222222") )+
    xlab("Welfare variation (%)")+
    ylab("Emissions variation (%)")
  
  if (SAVE_ici) {
    ggsave(paste0("./outputs/graphe",nom_specifique_fichier,".png"),plot=p,
           width = 2666/300,
           height= 1875/300,
           unit="in")
  }
}

draw_graph_without<-function(villes_ici,title_ici="test",SAVE_ici=TRUE,nom_specifique_fichier=NULL)
{
  if (is.null(nom_specifique_fichier))
  {nom_specifique_fichier<-title_ici}
  
  p <- ggplot(villes_ici, aes(y = expenses_2035_a_tracer_var, x = health_2035_a_tracer_var, fill = Continent)) +
    geom_point(size = 5,alpha=0.8,color='grey25',shape=21) +
    geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
    geom_vline(xintercept = 0, size = 0.5, colour="#333333") +
    #scale_fill_manual(values = country_colors) +
    bbc_style() +
    labs(title=title_ici,
         subtitle = "")+
    theme(
      panel.grid.major.x = element_line(color="#cbcbcb"),
      axis.title = ggplot2::element_text(family = "Helvetica", size = 18, color = "#222222") )+
    xlab("Welfare variations due to\nhealth co-benefits")+
    ylab("Welfare variations due to\nvariations in households expenses")+
    scale_x_continuous(labels = function(x) paste0(x, "%"))+
    scale_y_continuous(labels = function(x) paste0(x, "%"))
  
  if (SAVE_ici) {
    ggsave(paste0("./outputs/graphe",nom_specifique_fichier,"without.png"),plot=p,
           width = 2666/300,
           height= 1875/300,
           unit="in")
  }
}

### PLOT CITY SAMPLE

title_ici="Cities studied"

world_map<-ggplot(data=world) + 
  vincent_style()+
  geom_sf(fill = "antiquewhite1", lwd=0.1)+
  geom_point(data=villes,shape=21,alpha=0.9,
             aes(fill=Continent, geometry = geometry),color='grey15',
             stat = "sf_coordinates"
  ) +
  #geom_sf(data=villes,aes(color=Continent))+
  coord_sf(ylim = c(-65,90), expand = FALSE)+  
  xlab('') + ylab('') +
  labs(title=title_ici)

zoom_europe<-ggplot(data=world) + 
  geom_sf(fill = "antiquewhite1", lwd=0.1) + 
  geom_point(data=villes,shape=21,alpha=0.9,
             aes(fill=Continent, geometry = geometry),color='grey15',
             stat = "sf_coordinates",
             show.legend = FALSE
  ) +
  coord_sf(ylim = c(35,60), xlim = c(-10,30), expand = FALSE) + #Europe
  xlab('') + ylab('') +
  vincent_style_zoom()

world_with_Europe <- ggplotGrob(zoom_europe)
world_with_Europe<-world_map+
  annotation_custom(grob = world_with_Europe, 
                    xmin = -165, xmax = -95 , ymin = -65 , ymax = 30)


plot(world_with_Europe,
     width = 27, 
     height = 11, 
     units = "cm")

if (SAVE_ici) {
  ggsave(paste0("./outputs/",title_ici,".png"),plot=world_with_Europe,
         width = 2666/300,
         height= 1675/300,
         unit="in")
}

### FIGURE 3/S11/S12/S21/S22

texte_en_plus_nom_fichier<-"_intensity_quartile"
sub_title_ici<-"Welfare variation (%)"
title_ici<-"Restrictive land-use regulations"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(welfare_2035_UGB_var, breaks=c(-Inf, -5,0, 5, Inf),
                      labels=c("less than\n-5%","between\n-5% and 0","between\n0 and +5%","more\nthan +5%")))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_manual,funcargs=list(values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),drop = FALSE),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"Fuel tax"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(welfare_2035_CT_var, breaks=c(-Inf, -0.5,0, 0.5, Inf),
                      labels=c("less than\n-0.5%","between\n-0.5% and 0","between\n0 and +0.5%","more\nthan +0.5%")))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_manual,funcargs=list(values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),drop = FALSE),position_legend = "top",position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"Fuel efficiency policy"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(welfare_2035_FE_var, breaks=c(-Inf, -0.5,0, 0.5, Inf),
                      labels=c("less than\n-0.5%","between\n-0.5% and 0","between\n0 and +0.5%","more\nthan +0.5%")))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_manual,funcargs=list(values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),drop = FALSE),position_legend = "top",position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"BRT"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(welfare_2035_BRT_var, breaks=c(-Inf, -1,0,1, Inf),
                      labels=c("less than\n-1%","between\n-1% and 0","between\n0 and +1%","more\nthan +1%")))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_manual,funcargs=list(values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),drop = FALSE),position_legend = "top",position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)


texte_en_plus_nom_fichier<-"emission_intensity_quartile"
sub_title_ici<-"CO2 transport emission variation (%)"
title_ici<-"Restrictive land-use regulations"


villes_ici<-villes %>% 
  mutate(a_tracer=cut(emissions_2035_UGB_var, breaks=c(-Inf,-10,-5,-1,Inf),
                      labels=c("more\nthan -10%","between\n-10% and -5%","between\n-5% and -1%","less than\n-1%%")
  ))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"Fuel tax"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(emissions_2035_CT_var, breaks=c(-Inf,-10,-5,-1,Inf),
                      labels=c("more\nthan -10%","between\n-10% and -5%","between\n-5% and -1%","less than\n-1%%")
  ))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"Fuel efficiency policy"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(emissions_2035_FE_var, breaks=c(-Inf,-10,-5,-1,Inf),
                      labels=c("more\nthan -10%","between\n-10% and -5%","between\n-5% and -1%","less than\n-1%%")
  ))
#draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(),position_legend = "none",position_titre_horiz=0.5,position_titre_vertic=-6,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

title_ici<-"BRT"
villes_ici<-villes %>% 
  mutate(a_tracer=cut(emissions_2035_BRT_var, breaks=c(-Inf,-10,-5,-1,Inf),
                      labels=c("more\nthan -10%","between\n-10% and -5%","between\n-5% and -1%","less than\n-1%%")
  ))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)

texte_en_plus_nom_fichier<-"_intensity_quartile"
sub_title_ici<-"Welfare variation (%)"
title_ici<-"Impact of the combined four policies"
nom_specifique_fichier<-"all policies"
villes_ici<-villes %>% 
  mutate(a_tracer=welfare_2035_all_var) %>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf, -5,0,5, Inf),
                      labels=c("less than\n-5%",
                               "between\n-5% and 0",
                               "between\n0 and +5%",
                               "more\nthan +5%")))
draw_my_map(villes_ici=villes_ici,
            title_ici=title_ici,
            subtitle_ici=sub_title_ici,
            myfunc=scale_fill_manual,
            funcargs=list(
              values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),
              drop = FALSE
            ),
            position_legend = "top",
            position_titre_horiz=0.,
            position_titre_vertic=0,
            texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,
            nom_specifique_fichier=nom_specifique_fichier
)

title_ici<-"Impact of the combined four policies"
nom_specifique_fichier<-"all policies"
texte_en_plus_nom_fichier<-"emission_intensity_quartile"
sub_title_ici<-"CO2 transport emission variation (%)"
villes_ici<-villes %>% 
  mutate(a_tracer=emissions_2035_all_var)%>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf,-30,-20,-10,Inf),
                      labels=c("more\nthan -30%","between\n-30% and -20%","between\n-20% and -10%","less than\n-10%")
  ))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,nom_specifique_fichier=nom_specifique_fichier)

### FIGURE S17/S28

title_ici="Fuel Tax"
villes_ici %>% mutate(
  emissions_2035_a_tracer_var=emissions_2035_CT_var,
  welfare_2035_a_tracer_var=welfare_2035_CT_var
) %>% draw_graph(.,title_ici,SAVE_ici)

title_ici="Fuel Efficiency policy"
villes_ici %>% mutate(
  emissions_2035_a_tracer_var=emissions_2035_FE_var,
  welfare_2035_a_tracer_var=welfare_2035_FE_var
) %>% draw_graph(.,title_ici,SAVE_ici)

title_ici="BRT"
villes_ici %>% mutate(
  emissions_2035_a_tracer_var=emissions_2035_BRT_var,
  welfare_2035_a_tracer_var=welfare_2035_BRT_var
) %>% draw_graph(.,title_ici,SAVE_ici)

title_ici="Restrictive land-use regulations"
villes_ici %>% mutate(
  emissions_2035_a_tracer_var=emissions_2035_UGB_var,
  welfare_2035_a_tracer_var=welfare_2035_UGB_var
) %>% draw_graph(.,title_ici,SAVE_ici)

welfare_var_mean<-villes %>% 
  st_drop_geometry() %>% 
  mutate(welfare_pondere=welfare_2035_all_var*population_2035,
         welfare_pondere_without=welfare_2035_all_var_without*population_2035,
         emissions_pondere=emissions_2035_all_var*population_2035) %>% 
  summarise(welfare_pondere=sum(welfare_pondere)/sum(population_2035),
            welfare_pondere_without=sum(welfare_pondere_without)/sum(population_2035),
            emissions_pondere=sum(emissions_pondere)/sum(population_2035)) 

texte_en_plus_nom_fichier<-"_intensity_quartile"
sub_title_ici<-"Welfare variation (%)"
title_ici<-"Impact of the combined four policies"
#title_ici<-""
nom_specifique_fichier<-"all policies"
villes_ici<-villes %>% 
  mutate(a_tracer=welfare_2035_all_var) %>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf, -5,0,5, Inf),
                      labels=c("less than\n-5%",
                               "between\n-5% and 0",
                               "between\n0 and +5%",
                               "more\nthan +5%")))
draw_my_map(villes_ici=villes_ici,
            title_ici=title_ici,
            subtitle_ici=sub_title_ici,
            myfunc=scale_fill_manual,
            funcargs=list(
              values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),
              drop = FALSE
            ),
            position_legend = "top",
            position_titre_horiz=0.,
            position_titre_vertic=0,
            texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,
            nom_specifique_fichier=nom_specifique_fichier
)


title_ici<-"Impact of the combined four policies"
nom_specifique_fichier<-"all policies"
texte_en_plus_nom_fichier<-"emission_intensity_quartile"
sub_title_ici<-"CO2 transport emission variation (%)"
villes_ici<-villes %>% 
  mutate(a_tracer=emissions_2035_all_var)%>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf,-30,-20,-10,Inf),
                      labels=c("more\nthan -30%","between\n-30% and -20%","between\n-20% and -10%","less than\n-10%")
  ))
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,nom_specifique_fichier=nom_specifique_fichier)

### FIGURE S15/S26

title_ici<-"Impact of the combined four policies"
nom_specifique_fichier<-"all_policies"

villes_ici %>% mutate(
  #emissions_2035_a_tracer_var=emissions_2035_CT_var+emissions_2035_FE_var+emissions_2035_UGB_var+emissions_2035_BRT_var,
  #welfare_2035_a_tracer_var=welfare_2035_CT_var+welfare_2035_FE_var+welfare_2035_UGB_var+welfare_2035_BRT_var
  emissions_2035_a_tracer_var=emissions_2035_all_var,
  welfare_2035_a_tracer_var=welfare_2035_all_var
) %>% draw_graph(.,title_ici,SAVE_ici,nom_specifique_fichier=nom_specifique_fichier)


### FIGURE 1/S23/S32

tmp<-villes %>% 
  st_drop_geometry() %>%
  select(c(City,population_2035,
           emissions_2035_CT_var,welfare_2035_CT_var,
           emissions_2035_FE_var,welfare_2035_FE_var,
           emissions_2035_BRT_var,welfare_2035_BRT_var,
           emissions_2035_UGB_var,welfare_2035_UGB_var,
           emissions_2035_all_var,welfare_2035_all_var,
           emissions_2035_all_welfare_increasing_var,welfare_2035_all_welfare_increasing_var,welfare_2035_all_welfare_increasing_var_without,
           welfare_2035_CT_var,welfare_2035_CT_var_without,
           welfare_2035_FE_var,welfare_2035_FE_var_without,
           welfare_2035_BRT_var,welfare_2035_BRT_var_without,
           welfare_2035_UGB_var,welfare_2035_UGB_var_without,
           welfare_2035_all_var,welfare_2035_all_var_without,))%>%
  rename(`emissions_2035_welfareincreasing_var`=emissions_2035_all_welfare_increasing_var,
         `welfare_2035_welfareincreasing_var`=welfare_2035_all_welfare_increasing_var,
         `welfare_2035_welfareincreasing_var_without`=welfare_2035_all_welfare_increasing_var_without) %>% 
  pivot_longer(-c(1,2))%>% 
  separate(name,into=c("variable",NA,"policy",NA,"health")) %>% 
  mutate(what=case_when(
    variable=="emissions" ~ "Emissions\nvariations",
    (variable=="welfare")&(health=="without") ~ "Households\nexpenses",
    (variable=="welfare")&(is.na(health)) ~ "Total impact\non welfare",
    TRUE ~ "attention erreur"
  )) %>% 
  select(-health) %>% 
  pivot_wider(names_from=what,values_from = value) %>% 
  mutate("Health\nco-benefits"=`Total impact\non welfare`-`Households\nexpenses`)%>% 
  #select(-c(total)) %>% 
  pivot_longer(-seq(1,4)) %>% 
  filter(!is.na(value))

tmp2<-tmp %>% 
  group_by(variable,policy,name) %>% 
  summarise(median=median(value),
            quartile25=quantile(value,0.25,na.rm=TRUE),
            quartile75=quantile(value,0.75,na.rm=TRUE),
            mean=sum(value*population_2035)/sum(population_2035))%>% 
  mutate(y_label=ifelse((variable=="emissions")&(name=="Emissions\nvariations"),20,12))

p<-tmp%>% 
  ggplot(aes(x=policy,y=value,fill=variable))+
  geom_jitter(aes(fill=variable),
              position=position_jitterdodge(dodge.width = 0.9,),
              shape=21,
              alpha=0.3,
              color="black")+
  geom_boxplot(outlier.shape = NA, alpha=0.5,position = position_dodge(width = .9))+
  geom_label(data=tmp2,
             aes(x = policy, 
                 y = y_label,
                 label = ifelse(mean>0,paste0("+",round(mean, digits=1),"%"),paste0(round(mean, digits=1),"%")),
                 # label = paste0(
                 # ifelse(median>0,paste0("+",round(median, digits=1),"%"),paste0(round(median, digits=1),"%")),
                 #  "\n",
                 #  "(",round(quartile25, digits=1),"% / ",round(quartile75, digits=1),"%)"
                 # )
             ),
             hjust = 0.5, 
             vjust = 1, 
             label.size = NA, 
             fill=NA,
             family="Helvetica", 
             size = 6,
             position = position_dodge(width = .9),
             face="bold")+
  bbc_style()+
  geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
  #scale_fill_manual(values = country_colors) +
  bbc_style() +
  facet_grid(rows=vars(name),scales="free_y")+
  #ylim(-25,10)+
  scale_x_discrete(labels=c("All 4 policies\ncombined","Bus Rapid\nTransit","Fuel\nTax","Fuel\nEfficiency","Land-use\nregulations","Welfare-\nincreasing\npolicy\nportfolio"))+
  labs(title="",
       subtitle = "")+
  theme(panel.spacing.y = unit(2,"line"),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.position = "top")+
  scale_y_continuous(labels = function(x) paste0(x, "%"))



p

if (SAVE_ici) {
  ggsave(paste0("./outputs/box_plot_policies_version5.png"),plot=p,
         # width = 2666/300,
         # height= 1875/300,
         height = 2666/300*1.3,
         width= 2666/300*1.3,
         unit="in")
}

### FIGURE S13/S24

tmp<-villes %>% 
  st_drop_geometry() %>%
  select(c(City,population_2035,
           welfare_2035_CT_var,welfare_2035_CT_var_without,
           welfare_2035_FE_var,welfare_2035_FE_var_without,
           welfare_2035_BRT_var,welfare_2035_BRT_var_without,
           welfare_2035_UGB_var,welfare_2035_UGB_var_without,
           welfare_2035_all_var,welfare_2035_all_var_without,
           welfare_2035_all_welfare_increasing_var,welfare_2035_all_welfare_increasing_var_without,))%>%
  rename(
    `welfare_2035_welfareincreasing_var`=welfare_2035_all_welfare_increasing_var,
    `welfare_2035_welfareincreasing_var_without`=welfare_2035_all_welfare_increasing_var_without) %>% 
  pivot_longer(-c(1,2))%>% 
  separate(name,into=c("variable",NA,"policy",NA,"health")) %>% 
  mutate(health=if_else(is.na(health),"Total","Households\nexpenses")) %>% 
  pivot_wider(names_from=health,values_from = value) %>% 
  mutate("Health\nco-benefits"=Total-`Households\nexpenses`) %>% 
  #select(-c(total)) %>% 
  pivot_longer(-seq(1,4))

tmp2<-tmp %>% 
  group_by(policy,name) %>% 
  summarise(median=median(value),
            quartile25=quantile(value,0.25),
            quartile75=quantile(value,0.75))

p<-tmp%>% 
  ggplot(aes(x=policy,y=value,fill=name))+
  geom_jitter(aes(fill=name),
              position=position_jitterdodge(dodge.width = 0.9,),
              shape=21,
              alpha=0.3,
              color="black")+
  geom_boxplot(outlier.shape = NA, alpha=0.5,position = position_dodge(width = .9))+
  geom_label(data=tmp2,
             aes(x = policy, 
                 y = 15,
                 # label = ifelse(mean>0,paste0("+",round(mean, digits=1),"%"),paste0(round(mean, digits=1),"%")),
                 label = paste0(
                   ifelse(median>0,paste0("+",round(median, digits=1),"%"),paste0(round(median, digits=1),"%")),
                   "\n",
                   "(",round(quartile25, digits=1),"% / ",round(quartile75, digits=1),"%)"
                 )),
             hjust = 0.5, 
             vjust = 1, 
             label.size = NA, 
             fill=NA,
             family="Helvetica", 
             size = 5,
             position = position_dodge(width = .9),
             face="bold")+
  bbc_style()+
  geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
  #scale_fill_manual(values = country_colors) +
  bbc_style() +
  facet_grid(rows=vars(name),scales="free_y")+
  #ylim(-25,10)+
  scale_x_discrete(labels=c("All 4 policies\ncombined","Bus Rapid\nTransit","Fuel\nTax","Fuel\nEfficiency","Land-use\nregulations","Welfare-\nincreasing\npolicy\nportfolio"))+
  labs(title="Decomposition of welfare variations",
       subtitle = "Impact of variations in expenses, and of health co-benefits")+
  theme(panel.spacing.y = unit(2,"line"),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.position = "none")+
  scale_y_continuous(labels = function(x) paste0(x, "%"))



p 

if (SAVE_ici) {
  ggsave(paste0("./outputs/box_plot_health.png"),plot=p,
         width = 2666/300*1.2,
         #height= 1875/300,
         height = 2666/300*1.2,
         unit="in")
}

### FIGURE S14/S25

title_ici="Fuel Tax"
villes_ici %>% mutate(
  expenses_2035_a_tracer_var=welfare_2035_CT_var_without,
  health_2035_a_tracer_var=welfare_2035_CT_var-welfare_2035_CT_var_without
) %>% draw_graph_without(.,title_ici,SAVE_ici)

title_ici="Fuel Efficiency policy"
villes_ici %>% mutate(
  expenses_2035_a_tracer_var=welfare_2035_FE_var_without,
  health_2035_a_tracer_var=welfare_2035_FE_var-welfare_2035_FE_var_without
) %>% draw_graph_without(.,title_ici,SAVE_ici)

title_ici="BRT"
villes_ici %>% mutate(
  expenses_2035_a_tracer_var=welfare_2035_BRT_var_without,
  health_2035_a_tracer_var=welfare_2035_BRT_var-welfare_2035_BRT_var_without
) %>% draw_graph_without(.,title_ici,SAVE_ici)

title_ici="Restrictive land-use regulations"
villes_ici %>% mutate(
  expenses_2035_a_tracer_var=welfare_2035_UGB_var_without,
  health_2035_a_tracer_var=welfare_2035_UGB_var-welfare_2035_UGB_var_without
) %>% draw_graph_without(.,title_ici,SAVE_ici)


### FIGURE 2/S33

var_cobenefits<-results[c("var_active_modes_CT",
                         "var_active_modes_FE",
                         "var_active_modes_UGB"  ,
                         "var_active_modes_BRT",
                         "var_active_modes_all" ,
                         "var_air_pollution_CT",
                         "var_air_pollution_FE",
                         "var_air_pollution_UGB",
                         "var_air_pollution_BRT",
                         "var_air_pollution_all",
                         "var_car_accidents_CT",
                         "var_car_accidents_FE",
                         "var_car_accidents_UGB",
                         "var_car_accidents_BRT",
                         "var_car_accidents_all",
                         "var_noise_CT",
                         "var_noise_FE",
                         "var_noise_UGB",
                         "var_noise_BRT",
                         "var_noise_all",
                         "var_active_modes_all_welfare_increasing",
                         "var_air_pollution_all_welfare_increasing",
                         "var_car_accidents_all_welfare_increasing",
                         "var_noise_all_welfare_increasing",
                         "var_rent_all_welfare_increasing",
                         "var_tcost_all_welfare_increasing",
                         "var_disp_income_all_welfare_increasing",
                         "var_disp_income_CT",
                         "var_disp_income_BRT",
                         "var_disp_income_all",
                         "var_rent_CT",
                         "var_rent_FE",
                         "var_rent_UGB",
                         "var_rent_BRT",
                         "var_rent_all",
                         "var_tcost_CT",
                         "var_tcost_FE",
                         "var_tcost_UGB",
                         "var_tcost_BRT",
                         "var_tcost_all",   
                         "population_2035")] %>%
  mutate(var_disp_income_FE=0) %>% 
  mutate(var_disp_income_UGB=0)


p<-var_cobenefits %>% 
  pivot_longer(-c(population_2035),values_to = "Variation") %>%
  mutate(financial=ifelse(grepl("cost",name) | grepl("rent",name)| grepl("income",name),"Variations in\nfinancial expenses","Health co-benefit")) %>%
  mutate(policy=case_when(
    grepl("welfare_increasing",name) ~ "Welfare increasing",
    grepl("CT",name) ~ "Fuel\nTax",
    grepl("FE",name) ~ "Fuel\nEfficiency",
    grepl("BRT",name) ~ "Bus Rapid\nTransit",
    grepl("UGB",name) ~ "Land-use\nregulations",
    TRUE ~ "All policies"
  )) %>% 
  mutate(benefit=case_when(
    grepl("active_modes",name) ~ "Active mode use",
    grepl("air_pollution",name) ~ "Exposure to air pollution",
    grepl("car_accidents",name) ~ "Car accidents",
    grepl("noise",name) ~ "Exposure to noise",
    grepl("rent",name) ~ "Average housing costs",
    grepl("tcost",name) ~ "Average transportation costs",
    grepl("disp_income",name) ~ "Disposable income",
    TRUE ~"zz" # juste pour v?rifier qu'il n'y a pas de probl?me
  ))%>% 
  filter(!(policy %in% c("Welfare increasing","All policies"))) %>% 
  mutate(benefit=factor(benefit, levels=c("Active mode use",
                                          "Car accidents",
                                          "Exposure to air pollution",
                                          "Exposure to noise",
                                          "Average housing costs",
                                          "Average transportation costs",
                                          "Disposable income"
  ))) %>% 
  ggplot(aes(x=policy,y=Variation,fill=financial))+
  geom_boxplot(outlier.shape = NA, alpha=0.7,position = position_dodge(width = .9))+
  geom_jitter(aes(fill=financial),
              position=position_jitterdodge(dodge.width = 0.9,),
              shape=21,
              alpha=0.3,
              color="black")+
  bbc_style()+
  geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
  #scale_fill_manual(values = country_colors) +
  bbc_style() +
  #ylim(-25,10)+
  facet_wrap('benefit',scales = "free_y")+
  scale_y_continuous(labels = function(x) paste0(x, "%"))+
  labs(title="Impact of each policy on welfare components",
       subtitle = "Average impact of each policy")+
  theme(panel.spacing.y = unit(2,"line"),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.position = "top")+
  scale_fill_manual(values = c( "#1380A1","#FAAB18"))

p

if (SAVE_ici) {
  ggsave(paste0("./outputs/decompo2_fine_welfare.png"),plot=p,
         width = 2666/150,
         height= 1875/150,
         unit="in")
}

### FIGURE S18/S29

tmp_mean<-villes %>% 
  st_drop_geometry() %>%
  select(c(City,population_2035,
           welfare_2035_all_welfare_increasing_var,
           emissions_2035_all_welfare_increasing_var)) %>%
  summarise(welfare_2035_all_welfare_increasing_var=sum(welfare_2035_all_welfare_increasing_var*population_2035)/sum(population_2035),
            emissions_2035_all_welfare_increasing_var=sum(emissions_2035_all_welfare_increasing_var*population_2035)/sum(population_2035)) %>% 
  pivot_longer(everything(),values_to = "mean")%>% 
  separate(name,into=c("variable",NA,"policy",NA))


p<-villes %>% 
  st_drop_geometry() %>%
  select(c(City,population_2035,
           welfare_2035_all_welfare_increasing_var,
           emissions_2035_all_welfare_increasing_var)) %>%
  pivot_longer(-c(1,2)) %>% 
  separate(name,into=c("variable",NA,"policy",NA))%>% 
  ggplot(aes(x="",y=value,fill=variable))+
  geom_jitter(aes(fill=variable),
              shape=21,
              alpha=0.5,
              color="black",
              size=4)+
  geom_boxplot(outlier.shape = NA, alpha=0.3,width=1)+
  bbc_style()+
  geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
  bbc_style() +
  # facet_grid(cols=vars(variable),scales="free_x")+
  facet_wrap(.~variable,scales="free_y")+
  #ylim(-25,10)+
  labs(title="Impact of the welfare-increasing\npolicy portfolio",
       subtitle = "Global variation with respect to BAU scenario")+
  theme(panel.spacing.y = unit(2,"line"),panel.border = element_rect(colour = "black", fill=NA, size=1),legend.position="None")+
  scale_y_continuous(labels = function(x) paste0(x, "%"))

p

if (SAVE_ici) {
  ggsave(paste0("./outputs/box_plot_policies_increasing.png"),plot=p,
         width = 2666/350,
         height= 1875/350,
         #height = 2666/300,
         unit="in")
}


### FIGURE S20/S31

nom_specifique_fichier="XY welfare increasing"
title_ici="Impact of the welfare-increasing\npolicy portfolio"
texte_en_plus_nom_fichier<-"_intensity_quartile"
sub_title_ici<-"Welfare variation (%)"
villes_ici<-villes %>% 
  mutate(a_tracer=welfare_2035_all_welfare_increasing_var) %>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf, -5,0,5, Inf),
                      labels=c("less than\n-5%",
                               "between\n-5% and 0",
                               "between\n0 and +5%",
                               "more\nthan +5%")))
#draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(),position_legend = "none",position_titre_horiz=0.5,position_titre_vertic=-6,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_manual,funcargs=list(values = c( "#D7191C",  "#FDAE61", "#A6D96A","#1A9641"),drop = FALSE),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,nom_specifique_fichier=nom_specifique_fichier)


### FIGURE S19/S30

nom_specifique_fichier="XY welfare increasing"
title_ici="Impact of the welfare-increasing\npolicy portfolio"
texte_en_plus_nom_fichier<-"emission_intensity_quartile"
sub_title_ici<-"CO2 transport emission variation (%)"
villes_ici<-villes %>% 
  mutate(a_tracer=emissions_2035_all_welfare_increasing_var)%>% 
  mutate(a_tracer=cut(a_tracer, breaks=c(-Inf,-30,-20,-10,Inf),
                      labels=c("more\nthan -30%","between\n-30% and -20%","between\n-20% and -10%","less than\n-10%")
  ))
#draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(),position_legend = "none",position_titre_horiz=0.5,position_titre_vertic=-6,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier)
draw_my_map(villes_ici=villes_ici,title_ici=title_ici,subtitle_ici=sub_title_ici,myfunc=scale_fill_viridis_d,funcargs=list(drop = FALSE,direction=1),position_legend = "top",
            position_titre_horiz=0.,position_titre_vertic=0,texte_en_plus_nom_fichier=texte_en_plus_nom_fichier,nom_specifique_fichier=nom_specifique_fichier)


### FIGURE S16/S27

mod <- lm( log(-emissions_2035_all_welfare_increasing_var) ~ welfare_2035_all_welfare_increasing_var, data = villes) 
summary(mod)

p <- ggplot(villes, aes(y = emissions_2035_all_welfare_increasing_var, x = welfare_2035_all_welfare_increasing_var, fill = Continent)) +
  geom_point(size = 5,alpha=0.8,color='grey25',shape=21) +
  geom_hline(yintercept = 0, size = 0.5, colour="#333333") +
  geom_vline(xintercept = 0, size = 0.5, colour="#333333") +
  #scale_fill_manual(values = country_colors) +
  bbc_style() +
  labs(title="Impact of the welfare-increasing\npolicy portfolio",
       subtitle = "")+
  theme(
    panel.grid.major.x = element_line(color="#cbcbcb"),
    axis.title = ggplot2::element_text(family = "Helvetica", size = 18, color = "#222222") )+
  geom_smooth(aes(y = emissions_2035_all_welfare_increasing_var, x = welfare_2035_all_welfare_increasing_var,fill="black"),method = "lm", col = "black",linetype="dashed")+
  xlab("Welfare variation (%)")+
  ylab("Emissions variation (%)")


p

if (SAVE_ici) {
  ggsave(paste0("./outputs/graphe","XY welfare increasing regression",".png"),plot=p,
         width = 2666/300,
         height= 1875/300,
         unit="in")
}
