# Makefile for Samoan Passage overflow energy and momentum study.

PYTHON=python3

# By default display the help listing below. Note: First target is always the
# default target, that's why help is displayed when just typing make without 
# providing a target. All comments starting with a second # will be included in
# the help listing.

# HELP ============================================================{{{
## help                : Show this help section
.PHONY : help
help : Makefile
	@echo "Please use \`make <target>' where <target> is one of"
	@sed -n 's/^##//p' $<
# }}}

# ALL ============================================================={{{
## all                 : Run everything
.PHONY : all
all: data calcs plots
# }}}


# RETRIEVE DATA ==================================================={{{
DATA_TOWYO_IN=data/in/towyo/nc/sp_2012_towyo_104.nc\
			  data/in/towyo/nc/sp_2014_towyo_011.nc\
			  data/in/towyo/nc/sp_2014_towyo_012.nc


DATA_BATHY_IN=data/in/bathy/sp_bathy.nc

DATA_MP_IN=data/in/mp/nc/sp_mp_T01.nc

DATA_CTD_IN=data/in/ctd_ladcp/sp_2012_ctd_ladcp.nc\
			data/in/ctd_ladcp/sp_2014_ctd_ladcp.nc

$(DATA_TOWYO_IN) &: code/00_retrieve_data.py
	$(PYTHON) code/00_retrieve_data.py --towyo

$(DATA_BATHY_IN) &: code/00_retrieve_data.py
	$(PYTHON) code/00_retrieve_data.py --bathy

$(DATA_MP_IN) &: code/00_retrieve_data.py
	$(PYTHON) code/00_retrieve_data.py --mp

$(DATA_CTD_IN) &: code/00_retrieve_data.py
	$(PYTHON) code/00_retrieve_data.py --ctd

## retrieve-data-towyo : Sync towyo data from data repository
retrieve-data-towyo: $(DATA_TOWYO_IN)
## retrieve-data-bathy : Sync bathymetric data from data repository
retrieve-data-bathy: $(DATA_BATHY_IN)
## retrieve-data-mp    : Sync MP data from data repository
retrieve-data-mp: $(DATA_MP_IN)
## retrieve-data-ctd   : Sync CTD/LADCP data from data repository
retrieve-data-ctd: $(DATA_CTD_IN)
# }}}

# PREPARE DATA ===================================================={{{
DATA_TOWYO=data/out/ty2012in.nc data/out/ty2014in.nc

DATA_EXTRA=data/out/ctd_downstream.nc

DATA_MODEL=data/model_data.nc

DATA_MODEL_SNAPSHOTS=data/model_snapshot.nc\
					 data/model_energy_snapshot.nc

DATA=$(DATA_TOWYO) $(DATA_MODEL) $(DATA_EXTRA)

# --- some convenience naming
## data                : Generate all initial data files
data: $(DATA)
##   data-towyo        : Generate towyo data files
data-towyo: $(DATA_TOWYO)
##   data-model        : Generate model data files
data-model: $(DATA_MODEL)

$(DATA_TOWYO) &: code/01_load_towyo_data.py code/nslib/io.py
	$(PYTHON) $<

$(DATA_EXTRA): code/01_downstream_density.py code/nslib/io.py
	$(PYTHON) $<

$(DATA_MODEL): code/00_retrieve_data.py
	$(PYTHON) code/00_retrieve_data.py --model_extracted

# $(DATA_MODEL): model_extract_data.py nslib/model.py
# 	$(PYTHON) $<

$(DATA_MODEL_SNAPSHOTS): model_extract_snapshot.py data/model_data.nc data/model_energy.nc
	$(PYTHON) $<
# }}}

# ANALYSIS ========================================================{{{
## calcs               : Run all calculations
.PHONY : calcs
calcs : towyo-calcs model-calcs mod-energ-budget

##   towyo-calcs       : Calculate energetics terms for towyo data
TOWYO-CALCS-OUT=data/out/ty2012.nc data/out/ty2014.nc\
				data/out/ape12.nc data/out/ape14.nc\
				data/out/ape12s.nc data/out/ape14s.nc\
				data/out/towyo_energy_budget.nc

towyo-calcs : $(TOWYO-CALCS-OUT)

$(TOWYO-CALCS-OUT) &: code/02_nb_towyo_analysis.py code/nslib/towyo.py $(DATA_TOWYO)
	$(PYTHON) $<


##   model-calcs       : Calculate energetics terms for model data
MODEL-CALCS-OUT=data/out/model_bottom_pressure.nc\
				data/out/model_energy.nc\
				$(MODEL-WAVE-FLUXES)

model-calcs : $(MODEL-CALCS-OUT)

##   model-wave-fluxes : Calculate model wave fluxes from local means
MODEL-WAVE-FLUXES=data/out/model_small_scale_fluxes.nc\
				  data/out/model_small_scale_vwf_integrated.nc\
				  data/out/model_hp_fluxes.nc\
				  data/out/model_hp_vwf_integrated.nc
model-wave-fluxes    : $(MODEL-WAVE-FLUXES)
$(MODEL-WAVE-FLUXES) : code/model_wave_flux_calcs.py code/nslib/model_wave_fluxes.py
	$(PYTHON) $<

data/out/model_bottom_pressure.nc: code/model_form_drag.py $(DATA_MODEL) data/out/model_energy.nc
	$(PYTHON) $<

data/out/model_energy.nc: code/model_energy_calcs.py $(DATA_MODEL) $(MODEL-WAVE-FLUXES)
	$(PYTHON) $<

##   mod-energ-budget  : Calculate model energy budget
MODEL-ENERGY-BUDGET=data/out/model_energy_budget_results_layer.nc\
					data/out/model_energy_budget_results_box.nc
mod-energ-budget: $(MODEL-ENERGY-BUDGET)
$(MODEL-ENERGY-BUDGET): code/model_energy_budget.py $(DATA_MODEL) data/out/model_energy.nc
	$(PYTHON) $<
# }}}

# PLOTS ==========================================================={{{
## plots               : Generate all plots
.PHONY : plots
plots : plot-map plot-mp plot-energetics plot-model plot-ty-sketch plot-bernoulli

# MAPS {{{
##   plot-map          : Plot maps
plot-map: fig/map_overview.png
fig/map_overview.png: code/plot_overview_map.py
	$(PYTHON) $<
# }}}

# MP {{{
##   plot-mp           : Plot moored profiler time series
plot-mp: fig/T1_time_series.png
fig/T1_time_series.png: code/plot_mp_time_series.py
	$(PYTHON) $<
# }}}

# ENERGETICS {{{
##   plot-energetics   : Plot energetics fields and integrated quantities
ENERGETICS_PLOTS=fig/energetics_fields.png\
				 fig/energetics_fields_integrated.png\
				 fig/vertical_wave_flux_integrated.png\
				 fig/energy_budget_results.png
plot-energetics: $(ENERGETICS_PLOTS) $(DATA) $(DATA_MODEL_SNAPSHOTS)

fig/energetics_fields.png: plot_energetics_fields.py\
						   nslib/plt.py\
						   $(DATA_MODEL_SNAPSHOTS)
	$(PYTHON) $<
fig/energetics_fields_integrated.png: plot_energetics_fields_integrated.py\
								   	  nslib/plt.py\
						   			  $(DATA_MODEL_SNAPSHOTS)
	$(PYTHON) $<
fig/vertical_wave_flux_integrated.png: plot_vertical_wave_flux_integrated.py\
									   nslib/plt.py\
									   $(DATA-TOWYO)\
						   			   $(DATA_MODEL_SNAPSHOTS)
	$(PYTHON) $<

##   plot-enrg-bdgt    : Plot energy budget results
plot-enrg-bdgt: fig/energy_budget_results.png
fig/energy_budget_results.png: plot_energy_budget_results.py\
	                           nslib/plt.py\
							   $(MODEL-ENERGY-BUDGET) $(TOWYO-CALCS-OUT)
	$(PYTHON) $<
# }}}

# TOWYO ENERGETICS SKETCH {{{
##   plot-ty-sketch    : Plot towyo energetics sketch
ENERGETICS_SKETCH=fig/towyo_sketch_energetics.png
plot-ty-sketch: $(ENERGETICS_SKETCH)
$(ENERGETICS_SKETCH) : code/plot_towyo_sketch_energetics.py code/nslib/plt.py
	$(PYTHON) $<
# }}}

# BERNOULLI FLUX {{{
##   plot-bernoulli    : Plot Bernoulli flux estimates
plot-bernoulli: fig/bernoulli_transport.png
fig/bernoulli_transport.png: code/bernoulli_flux.py code/nslib/bernoulli.py
	$(PYTHON) $<
# }}}

# MODEL {{{
##   plot-model        : Plot model figures
MODEL_PLOTS=fig/model_steadiness.png\
			fig/model_snapshot.png\
			fig/model_internal_wave_fluxes.png
plot-model : $(MODEL_PLOTS) $(DATA-MODEL)

##   plot-mod-stead    : Plot model steadiness
plot-mod-stead: fig/model_steadiness.png
fig/model_steadiness.png : plot_model_steadiness.py nslib/model.py
	$(PYTHON) $<

##   plot-mod-snapsh   : Plot model snapshot
plot-mod-snapsh: fig/model_snapshot.png
fig/model_snapshot.png : plot_model_snapshot.py nslib/model.py
	$(PYTHON) $<

##   plot-mod-iw-flux  : Plot model internal wave flux snapshot & time series
plot-mod-iw-flux: fig/model_internal_wave_fluxes.png
fig/model_internal_wave_fluxes.png: plot_model_internal_wave_fluxes.py nslib/model.py
	$(PYTHON) $<
# }}}

# }}}

