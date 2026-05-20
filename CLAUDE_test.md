# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OntologyCreation uses Pydantic AI agents to automatically extract OWL ontology TBox definitions (class hierarchies) from unstructured text. The pipeline fetches data from DBpedia/Wikipedia, feeds Wikipedia summaries to a Claude-powered agent, then evaluates extraction quality against DBpedia ground truth.
