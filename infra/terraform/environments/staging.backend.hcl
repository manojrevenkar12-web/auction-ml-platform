bucket         = "aucml-terraform-state"
key            = "aucml/staging/terraform.tfstate"
region         = "eu-central-1"
dynamodb_table = "aucml-terraform-locks"
encrypt        = true
