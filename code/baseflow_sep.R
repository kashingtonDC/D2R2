library(EcoHydRology)

get_baseflow <- function(df){
    bfs<-BaseflowSeparation(df, passes=3)
    return(bfs)
}