# FPL Model

This repository contains a Fantasy Premier League (FPL) modeling and automation toolkit. The main focus is on selecting the optimal FPL squad based on recent player performance, with features for data sourcing, feature engineering, automated squad selection, and optional email reporting.

## Features
- **Automated FPL Squad Selection**: Selects the best 15-man squad based on points per game over a configurable number of recent gameweeks.
- **Data Sourcing**: Fetches and processes FPL data for analysis.
- **Feature Engineering**: Calculates advanced metrics such as mean points per game.
- **Squad Reporting**: Outputs the selected squad in a readable table format and optionally emails the report.
- **Configuration**: Easily adjust parameters such as the number of gameweeks to consider and email credentials.

## Project Structure
```
meta/
  configuration/         # Config and credentials (YAML)
  data/                  # Cleaned and merged FPL data (CSV)
  squads/                # Saved squad selections (CSV)
notebooks/               # Jupyter notebooks for analysis and prototyping
src/                     # Source code for data, features, squad selection, and reporting
  data_sourcing.py
  feature_engineering.py
  select_fpl_squad.py
  email_report.py
  fpl_bot_v1r0.py        # Main bot script
```

## Getting Started
1. **Install Requirements**
   - Python 3.8+
   - Install dependencies:
     ```sh
     pip install pandas tabulate tqdm pyyaml
     ```
2. **Configure Email (Optional)**
   - Edit `meta/configuration/email_credentials.yml` with your sender and recipient email details.
3. **Run the Bot**
   - From the `src/` directory:
     ```sh
     python fpl_bot_v1r0.py
     ```
   - The selected squad will be printed in the console. Uncomment the `email_squad` line in `main()` to enable email reports.

## Customization
- Change the number of gameweeks considered by editing the `num_gws_to_check` variable in `fpl_bot_v1r0.py`.
- Adjust data sources or feature engineering logic in the respective modules in `src/`.

## Notebooks
- Explore the `notebooks/` directory for exploratory data analysis, API prototyping, and additional insights.

## License
This project is for personal and educational use. Please review and update licensing as needed for your use case.
