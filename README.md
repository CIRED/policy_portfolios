# Environmental and welfare gains via urban transport policy portfolios across 120 cities
### Charlotte LIOTTA, Vincent VIGUIE, Felix CREUTZIG (*Minor Revisions* at *Nature Sustainability*)

This code allows to calibrate a model derived from the standard urban model of urban economics on 120 cities worldwide, and to simulate the impact of 4 transport emissions mitigation policies as well as of policy portfolios on these cities.

main.py allows for model calibration and policies simulations, and main_policy_portfolios.py allows for model calibration and policy portfolios simulations. robustness/main_congestion.py, robustness/main_policy_portfolios_congestion.py and main_policy_portfolios_robustness.py allow for robustness checks.

### Data availability

The main dataset used for the calibration of the model is available here: https://zenodo.org/record/7086267.
It is also fully described in the following data paper:
Lepetit, Q., Viguié, V., Liotta, C. A gridded dataset on densities, real estate prices, transport, and land use inside 192 worldwide urban areas. *Data in Brief* 47, 108962 (2023). https://doi.org/10.1016/j.dib.2023.108962

Complementary data are needed to run the simulations:
- World bank data on gasoline prices (https://data.worldbank.org/indicator/EP.PMP.SGAS.CD, accessed in April 2023)
- Data on average car fuel consumption per country from the International Energy Agency 2019 report *Fuel Economy in Major Car Markets* (https://www.iea.org/reports/fuel-economy-in-major-car-markets, accessed in April 2023).
- Data on the monetary cost of public transport from:<ul>
      <li>Numbeo (https://www.numbeo.com/cost-of-living/country_price_rankings?itemId=18, accessed in April 2023)
      <li>kiwi.fr (https://www.kiwi.com/stories/cheapest-and-most-expensive-public-transport-revealed, accessed in April 2023)
      <li> Wordatlas (https://www.worldatlas.com/articles/cost-of-public-transportation-around-the-world.html, accessed in April 2023). </ul>
- Data on income:<ul>
      <li>OECD GDP per capita at the city scale data (https://stats.oecd.org/Index.aspx?DataSetCode=CITIES, accessed in April 2023)
      <li>Brookings GDP per capita at the city scale data (*Redefining Global Cities* 2016 report from the Brookings Institution, https://www.brookings.edu/research/redefining-global-cities)
      <li>World Bank GDP per capita at the country scale data (https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.CD, accessed in April 2023).  </ul>
- Agricultural GDP and agricultural areas by country data from the FAO (https://www.fao.org/faostat/en/#home, accessed in April 2023).
- Population growth scenarios from the *World Urbanization Prospects – the 2018 Revision* of the United Nations (https://population.un.org/wup/Download, accessed in April 2023).
- Marginal costs of air pollution, noise and car accidents from the report *External costs of transport in Europe*, van Essen et al., 2019. https://trid.trb.org/view/1646234, accessed in April 2019.


Complementary datasets are needed for calibrations’ validation:
- Modal shares data from the following datasets:<ul>
      <li>EPOMM (https://epomm.eu, accessed in April 2023)
      <li>CDP (https://data.cdp.net, accessed in April 2023)
      <li>Deloitte (report *The 2019 Deloitte City Mobility Index*, Dixon et al., 2019. https://www2.deloitte.com/content/dam/Deloitte/br/Documents/consumer-business/City-Mobility-Index-2019.pdf, accessed in April 2023) </ul>
- Emissions datasets from the following papers:<ul>
      <li>Moran, D. et al. Carbon footprints of 13 000 cities. *Environ. Res. Lett.* 13, 064041 (2018). DOI 10.1088/1748-9326/aac72a.
      <li>Nangini, C. et al. A global dataset of CO 2 emissions and ancillary data related to emissions for 343 cities. *Scientific Data* 6, 180280 (2019). https://doi.org/10.1038/sdata.2018.280
      <li>Kona, A. et al. Global Covenant of Mayors, a dataset of GHG emissions for 6,200 cities in Europe and the Southern Mediterranean. *Earth System Science Data Discussions* 1–17 (2021) doi:10.5194/essd-2021-67. </ul>
