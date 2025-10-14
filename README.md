# Monarch Money MCP Server

An MCP (Model Context Protocol) server that provides access to Monarch Money financial data and operations.

## Features

- **Account Management**: List and retrieve account information
- **Transaction Operations**: Get, search, create, and update transactions with filtering by date range, accounts, and categories
- **Transaction Rules**: Create and manage rules for automatic transaction categorization and updates
- **Budget Analysis**: Access budget data and spending insights
- **Category Management**: List and manage transaction categories
- **Goal Tracking**: Access financial goals and progress
- **Net Worth Tracking**: Retrieve net worth snapshots over time
- **Intelligent Batch Operations**: Combined API calls for comprehensive financial overviews and spending analysis

## Installation

1. Clone or download this MCP server
2. Install dependencies:
   ```bash
   cd /path/to/monarch-money-mcp
   uv sync
   ```

## Configuration

Add the server to your `.mcp.json` configuration file:

```json
{
  "mcpServers": {
    "monarch-money": {
      "command": "/path/to/uv",
      "args": [
        "--directory", 
        "/path/to/monarch-money-mcp",
        "run",
        "python",
        "server.py"
      ],
      "env": {
        "MONARCH_EMAIL": "your-email@example.com",
        "MONARCH_PASSWORD": "your-password",
        "MONARCH_MFA_SECRET": "your-mfa-secret-key"
      }
    }
  }
}
```

**Important Notes:**
- Replace `/path/to/uv` with the full path to your `uv` executable (find it with `which uv`)
- Replace `/path/to/monarch-money-mcp` with the absolute path to this server directory
- Use absolute paths, not relative paths

### Getting Your MFA Secret

1. Go to Monarch Money settings and enable 2FA
2. When shown the QR code, look for the "Can't scan?" or "Enter manually" option
3. Copy the secret key (it will be a string like `T5SPVJIBRNPNNINFSH5W7RFVF2XYADYX`)
4. Use this as your `MONARCH_MFA_SECRET`

## Available Tools

### `get_accounts`
List all accounts with their balances and details.

### `get_transactions`
Get transactions with smart output formatting and flexible filtering:
- **Smart Output**: Returns compact format by default (reduces data size by ~80%)
  - Compact fields: `id`, `date`, `amount`, `merchant`, `plaidName`, `category`, `account`, `pending`, `needsReview`, `notes`
  - Set `verbose=True` for complete transaction details with all metadata
- **Flexible Date Filtering**: Supports natural language dates
  - Examples: `"last month"`, `"yesterday"`, `"30 days ago"`, `"this year"`
  - Also supports standard formats: `"2024-01-01"`, `"01/15/2024"`
- **Additional Filters**:
  - `account_id`: Filter by specific account
  - `category_id`: Filter by specific category
  - `limit`: Maximum transactions to return (default: 100, max: 1000)
  - `offset`: Pagination offset

### `get_categories`
List all transaction categories.

### `get_budgets`
Get budget information and spending analysis.

### `get_goals`
List financial goals and their progress.

### `get_cashflow`
Get cashflow data for income and expense analysis.

### `get_investments`
Get investment account details and performance.

### `get_net_worth`
Get net worth snapshots over time.

### Transaction Management

#### `search_transactions`
Search transactions with natural language queries:
- Searches across merchant names, descriptions, and notes
- Supports all date and account filtering options from `get_transactions`
- Returns compact or verbose format based on `verbose` parameter

#### `create_transaction`
Create a new manual transaction with category, merchant, and notes.

#### `update_transaction`
Update an existing transaction's amount, category, merchant, notes, or account.

### Transaction Rules

#### `get_transaction_rules`
List all configured transaction rules for automatic categorization.

#### `create_transaction_rule`
Create a rule to automatically categorize or modify transactions:
- **Merchant criteria**: `"contains:Amazon"`, `"equals:Starbucks"`
- **Amount filters**: `"gt:100"`, `"lt:50"`, `"eq:25.00"`
- **Actions**: Set category, add tags, rename merchant
- **Apply to existing**: Optionally apply rule to past transactions

#### `update_transaction_rule`
Update an existing transaction rule's criteria or actions.

#### `delete_transaction_rule`
Remove a transaction rule by ID.

#### `preview_transaction_rule`
Preview which transactions would match a rule before applying it.

### Batch Operations & Analysis

#### `get_complete_financial_overview`
Get a comprehensive financial snapshot in a single call:
- Combines accounts, budgets, cashflow, transactions, and categories
- Parallel API execution for fast response
- Includes transaction summary statistics
- Supports natural language period filters: `"this month"`, `"last quarter"`

#### `analyze_spending_patterns`
Deep analysis of spending trends and forecasting:
- Multi-month trend analysis by category and account
- Predictive forecasting based on historical patterns
- Confidence indicators and variance analysis
- Configurable lookback period (1-12 months)

#### `get_usage_analytics`
View MCP tool usage patterns and optimization suggestions:
- Tool call frequency and performance metrics
- Common usage sequences for batch operation suggestions
- Performance bottleneck identification

## Usage Examples

### Basic Account Information
```
Use the get_accounts tool to see all my accounts and their current balances.
```

### Transaction Analysis (Compact Format)
```
Get all transactions from last month - returns compact format with essential fields only.
```

The compact format is perfect for most queries and looks like:
```json
{
  "id": "220177205767991763",
  "date": "2025-08-24",
  "amount": -8.49,
  "merchant": "LA Bagel Delight",
  "plaidName": "LA BAGEL DELIGHT",
  "category": "Restaurants & Bars",
  "account": "Chase Sapphire",
  "pending": false,
  "needsReview": true
}
```

### Transaction Analysis (Verbose Format)
```
Get detailed transaction data with verbose=True to see all metadata, nested objects, and timestamps.
```

### Budget Tracking
```
Show me my current budget status using the get_budgets tool.
```

### Transaction Search
```
Search for all transactions from Amazon in the last 3 months.
```

### Transaction Rules
```
Create a rule to automatically categorize all Amazon purchases over $100 as "Shopping" and tag them with "online".
```

### Complete Financial Overview
```
Give me a complete financial overview for this month including accounts, budgets, cashflow, and transaction summaries.
```

## Session Management

The server automatically manages authentication sessions:
- Sessions are cached in a `.mm` directory for faster subsequent logins
- The session cache is automatically created and managed
- Use `MONARCH_FORCE_LOGIN=true` in the env section to force a fresh login if needed

## Troubleshooting

### MFA Issues
- Ensure your MFA secret is correct and properly formatted
- Try setting `MONARCH_FORCE_LOGIN=true` in your `.mcp.json` env section
- Check that your system time is accurate (required for TOTP)

### Connection Issues
- Verify your email and password are correct in `.mcp.json`
- Check your internet connection
- Try running the server directly to see detailed error messages:
  ```bash
  uv run server.py
  ```

### Session Problems
- Delete the `.mm` directory to clear cached sessions
- Set `MONARCH_FORCE_LOGIN=true` in your `.mcp.json` env section temporarily

## Notable Differences from Forked Repository

This implementation has evolved significantly from the original forked repository with substantial architectural improvements and enhanced capabilities:

### Advanced Architecture
- **FastMCP Framework**: Complete migration from basic MCP to modern FastMCP with `@mcp.tool()` decorators for cleaner, more maintainable code
- **Enhanced Library**: Uses `monarchmoney-enhanced` fork with transaction rule support and additional GraphQL operations
- **Comprehensive Testing**: 61 passing tests across 6 test files with 100% coverage including analytics, validation, and error handling
- **Type Safety**: Strict typing throughout with Pydantic models and minimal MyPy warnings
- **Structured Logging**: Professional logging with `structlog` for debugging and analytics tracking

### Enhanced Financial Intelligence
- **Compact Transaction Format**: Smart output formatting that reduces data size by ~80% while preserving essential fields (verbose mode available for full details)
- **Batch Operations**: Intelligent tools like `get_complete_financial_overview()` that combine 5 APIs in parallel execution
- **Pattern Analysis**: `analyze_spending_patterns()` with multi-month trend analysis and predictive forecasting
- **Smart Date Processing**: Natural language date parsing ("last month", "this year") with flexible fallback mechanisms
- **Usage Analytics**: Real-time performance monitoring with optimization suggestions and session tracking

### Production-Ready Features
- **Error Resilience**: Comprehensive error handling including broken pipe errors, session expiration, and graceful API failure recovery
- **Security Enhancements**: Proper session management with 0700 permissions and MFA support
- **JSON-RPC Compliance**: All logging redirected to stderr to prevent stdout contamination
- **Development Workflow**: Extensive documentation in `CLAUDE.md` with git commit standards and quality gates

### Tool Expansion
- **26 Tools**: Complete Monarch Money API coverage including transaction rules vs basic implementation
- **Transaction Rules**: Full rule management system for automatic categorization and updates
- **Intelligent Filtering**: Advanced transaction filtering with category, account, and date combinations
- **Search Capabilities**: Natural language transaction search with flexible criteria
- **Optimization Tracking**: Built-in analytics that suggest batch operations based on usage patterns

This implementation represents a complete rewrite focused on production readiness, developer experience, and advanced financial analysis capabilities.

## Credits

### MCP Server
- **Author**: Taurus Colvin ([@colvint](https://github.com/colvint))
- **Description**: MCP (Model Context Protocol) server wrapper for Monarch Money

### MonarchMoney Python Library
- **Original Author**: hammem ([@hammem](https://github.com/hammem))
- **Original Repository**: [https://github.com/hammem/monarchmoney](https://github.com/hammem/monarchmoney)
- **Enhanced Fork**: keithah/monarchmoney-enhanced with transaction rule support
- **License**: MIT License
- **Description**: The underlying Python library that provides API access to Monarch Money

This MCP server wraps the monarchmoney-enhanced Python library to provide seamless integration with AI assistants through the Model Context Protocol, including advanced features like transaction rule management.

## Security Notes

- Keep your credentials secure in your `.mcp.json` file
- The MFA secret provides full access to your account - treat it like a password
- Session files in `.mm` directory contain authentication tokens - keep them secure
- Consider restricting access to your `.mcp.json` file since it contains sensitive credentials