# Nobos Dataset Manager
Work in progress, still some leftover paths etc.

## Installation
1. Setup POSTGRESQL Database (Docker example in db_docker_example)
2. Copy config_template.py to config.py
3. Setup database connection in config.py
4. Run create_table.py
5. Run create_action_gt_view.sql on database
6. After new import: Run delete_skeletons_outside_img.sql on database
7. See examples in nobos_dataset_manager/examples/dataset_export_examples and dataset_import_examples