bucket         = "aucml-terraform-state"
key            = "aucml/prod/terraform.tfstate"
region         = "eu-central-1"
dynamodb_table = "aucml-terraform-locks"
encrypt        = true
