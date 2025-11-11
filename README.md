# RKT Project AI

An automated workflow system that transforms project ideas into Product Requirement Documents (PRDs) and execution plans, then pushes them directly to Azure DevOps as structured work items.

## Overview

RKT Project AI streamlines the project initialization process by automating three critical steps:

1. **Idea → PRD**: Convert a raw project idea (JSON format) into a comprehensive Product Requirements Document
2. **PRD → Plan**: Generate a detailed, step-by-step execution plan from the PRD
3. **Plan → Azure DevOps**: Automatically create User Stories and Tasks in Azure DevOps Boards

This eliminates manual data entry, reduces errors, and accelerates project kickoff from days to hours.

## Features

- ✅ **AI-Powered PRD Generation**: Uses OpenAI GPT models to create structured PRDs with goals, requirements, risks, and success criteria
- ✅ **Intelligent Plan Creation**: Automatically breaks down PRDs into actionable steps with acceptance criteria
- ✅ **Azure DevOps Integration**: Creates projects, features, user stories, and tasks with proper hierarchy
- ✅ **Flexible Output Formats**: Generates both JSON (machine-readable) and Markdown (human-readable) outputs
- ✅ **Process Template Support**: Works with Agile, Scrum, CMMI, and Basic process templates
- ✅ **Auto-creates Azure Resources**: Automatically creates projects, classification paths, and parent features if they don't exist

## Prerequisites

- Python 3.7+
- OpenAI API key
- Azure DevOps Personal Access Token (PAT)
- `.env` file with required credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/trekcrew1/RKT_Project_AI.git
cd RKT_Project_AI
```

2. Install required dependencies:
```bash
pip install openai python-dotenv python-dateutil requests
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
PAT_TOKEN=your_azure_devops_pat_token
DEFAULT_MODEL=gpt-4.1
```

## Project Structure

```
RKT_Project_AI/
├── main.py                      # Main orchestration script
├── make_prd_from_idea.py        # Generate PRD from idea JSON
├── make_plan_from_prd.py        # Generate execution plan from PRD
├── push_plan_to_azure.py        # Push plan to Azure DevOps
├── ideas/                       # Input: Project idea JSON files
│   └── TimeTracker.json
├── out/
│   ├── jsons/                   # PRD JSON outputs
│   ├── prds/                    # PRD Markdown outputs
│   └── plans/                   # Execution plan outputs
├── logs/                        # Application logs
└── .env                         # Environment configuration (not in repo)
```

## Usage

### Quick Start

Edit `main.py` to configure your project:

```python
project_name = 'TimeTracker'
project_location = r'ideas\TimeTracker.json'
```

Then run the complete workflow:

```bash
python main.py
```

### Step-by-Step Usage

#### 1. Generate PRD from Idea

```python
import make_prd_from_idea

make_prd_from_idea.main(
    file_name='TimeTracker',
    input_path='ideas/TimeTracker.json',
    overwrite=True
)
```

**Output**:
- `out/jsons/TimeTracker.prd.json`
- `out/prds/TimeTracker.md`

#### 2. Generate Execution Plan from PRD

```python
import make_plan_from_prd

make_plan_from_prd.main(
    project_name='TimeTracker',
    prd_path='out/prds/TimeTracker.md',
    overwrite=True,
    max_steps='auto',      # Automatic step optimization
    max_steps_hard=20      # Hard cap to preserve tasks
)
```

**Output**:
- `out/plans/TimeTracker.plan.json`
- `out/plans/TimeTracker.plan.md`

#### 3. Push Plan to Azure DevOps

```python
import push_plan_to_azure
import os

push_plan_to_azure.run_push_to_azure(
    plan='out/plans/TimeTracker.plan.json',
    org='trekcrew',
    project='TimeTracker',
    pat=os.getenv('PAT_TOKEN'),
    process='agile',
    work_item_type=None,  # Auto-select based on process
    parent_feature_title='MVP — Core Time Tracking'
)
```

## Input Format

Create idea files in the `ideas/` directory as JSON:

```json
{
  "date": "2025-08-19T14:27:00-04:00",
  "url": "https://example.com",
  "heading": "Project Title",
  "main_text": "Project description...",
  "trend_analysis": "Market trends...",
  "community_signals": "User feedback...",
  "full_text": "Complete project details..."
}
```

## Command-Line Usage

### Generate PRD

```bash
python make_prd_from_idea.py TimeTracker
```

### Generate Plan

```bash
python make_plan_from_prd.py --prd out/prds/TimeTracker.md
python make_plan_from_prd.py --prd out/prds/TimeTracker.md --max-steps auto --max-steps-hard 20
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `PAT_TOKEN` | Azure DevOps Personal Access Token | - |
| `DEFAULT_MODEL` | OpenAI model to use | `gpt-4.1` |

### Azure DevOps Process Templates

The system supports all Azure DevOps process templates:

- **Agile**: User Story → Task
- **Scrum**: Product Backlog Item → Task
- **CMMI**: Requirement → Task
- **Basic**: Issue → Task

## Benefits

- 🚀 **Speed**: Project setup in hours instead of days
- 🎯 **Accuracy**: Eliminates manual transcription errors
- 📊 **Consistency**: Standardized PRD and backlog structure
- 🔄 **Scalability**: Handle multiple projects without PM bottleneck
- 💡 **Focus**: Teams build instead of doing data entry

## Error Handling

The system includes comprehensive error handling:
- Validates Azure DevOps credentials
- Auto-creates missing projects and classification paths
- Retries failed API calls with exponential backoff
- Provides detailed error messages and logs

## Development

### Running Tests

(Add test instructions when available)

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

(Add license information)

## Author

**trekcrew1**  
GitHub: [@trekcrew1](https://github.com/trekcrew1)

## Acknowledgments

Built with:
- [OpenAI API](https://platform.openai.com/) for AI-powered content generation
- [Azure DevOps REST API](https://learn.microsoft.com/en-us/rest/api/azure/devops/) for project management integration

## Support

For issues or questions, please open an issue on GitHub.

---

**Note**: See `proposal.md` for the complete business case and detailed process documentation.
