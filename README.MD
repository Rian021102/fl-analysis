# FLUID ANALYSIS USING OPEN HOLE LOG
## Problem Background
After a well reaches its target depth, logging tools are typically employed to acquire data. These tools include the Gamma Ray (measures gamma rays emitted by the formation), Resistivity (measures the resistivity of fluid and formation), Neutron Log (measures the number of neutrons that travel back to the tool), and Density Log (measures the density of the rock). The next step is usually running RFT/MDT tool to obtain formation pressure and analyze the fluid in the formation based on log interpretation. However, there are instances when running RFT/MDT is not feasible due to various reasons such as exceeding the initial budget, wellbore stability issues, and more. This results in ambiguity in determining if the target formation contains hydrocarbons, which in turn hinders the completion strategy due to uncertainties. This project aims to explore the possibility of employing a machine learning model trained on log data and corresponding fluid classifications (based on historical fluid analysis or production) to address this issue.
## Input Data
The input data is LAS file with this format:
DEPTH.FT                   :  Depth
TVDSS.FT                   :  Reference TVD sub-sea log.
GR   .GAPI                 :  Gamma Ray
RT   .OHMM                 :  True Resistivity
NPHI .V/V                  :  Neutron Porosity
RHOB .G/C3                 :  Bulk Density
VCALCITE.V/V                  :  Calcite Volume Fraction
VQUARTZ.V/V                  :  Quartz Volume Fraction
VSH  .V/V                  :  Wet Cl/Sh 1 Volume Fraction
VCOAL.V/V                  :  Coal Volume Fraction
VORGSH.V/V                  :  Organic Shale Volume Fraction
VGAS .V/V                  :  X Gas Volume Fraction
VOIL .V/V                  :  X Oil Volume Fraction
VWATER.V/V                  :  X Water Volume Fraction
PHIT .V/V                  :  Total Porosity
PHIE .V/V                  :  Limited Effective Porosity
SWE  .V/V                  :  Effective Water Saturation
PERM .MD                   :  Permeability

After thorough experimentation and analysis, it has been concluded that only 6 features are necessary (including depth and TVD subsea depth) to train our model. The provided source code in this project drops the other columns not required for training the model. The deployment currently utilizes Streamlit to present the OH log data, including the prediction results.
## Limitations and Room for Improvement
It's important to acknowledge that the model's training data is limited to just two wells, and the testing data is drawn from other wells. While the results have shown promise, they may not fully reflect the entire field's characteristics. To improve the model's confidence and generalization, gathering more data with a wide range of feature values is essential. This will ensure that the model can account for various details pertaining to the field, leading to more reliable predictions in practical scenarios.
