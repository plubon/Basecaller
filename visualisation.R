library(hdf5r)
library(ggplot2)
h5_file <- H5File$new('~/praca_magisterska/Dane/squiggled/chrM/DEAMERNANOPORE_20161117_FNFAB43577_MN16450_sequencing_run_MA_821_R9_4_NA12878_11_17_16_88738_ch155_read1046_strand.fast5',mode='r+')
events <- h5_file[['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']]
signal <- h5_file[['Raw/Reads/Read_1046/Signal']][]
read_start <- h5attr(events, 'read_start_rel_to_raw')
events_df <- events[]
events_df$start <- events_df$start + read_start
read_end <- events_df[nrow(events_df), 'start'] + events_df[nrow(events_df), 'length']
signal_analyzed <- signal[read_start:read_end]
signal_analyzed <- (signal_analyzed-mean(signal_analyzed))/sd(signal_analyzed)
data <- data.frame(signal=signal_analyzed, x= 1:length(signal_analyzed))
data$base <- rep(events_viz$base, events_viz$length)
events_viz <- events_df[cumsum(events_df$length)<300,]
data$base <- c(rep(events_df$base, events_df$length), 'A')
ggplot(data=data[1:300,], aes(x=x,y=signal,colour=base))+geom_line(aes(group=1))+geom_vline(xintercept = cumsum(events_viz$length), linetype="dotted")

       