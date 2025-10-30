# 💬 Agent Dialogue Display - What You'll See

## Overview

When you enable **"Use ChatDev-style agents"**, the system displays **detailed turn-by-turn dialogue** between agents on your screen. This makes the AI's thinking process completely transparent!

---

## Example: "Which game has most reviews in recent 3 months?"

### What You'll See On Screen:

---

**User:** Which game has most reviews in recent 3 months?

---

**System:** 🤖 Using ChatDev-style multi-agent system...

**System:** 🎯 Planning analysis approach...

---

**System:**

## 💬 Planning Phase: AM ↔ DS Negotiation

### 🔵 Turn 1: DS Proposes

**Goal:** Find the game with the highest number of reviews in the last 3 months

**SQL:**
```sql
WITH recent_reviews AS (
    SELECT game_id, COUNT(*) AS review_count
    FROM games_reviews
    WHERE TRY_CAST(review_date AS TIMESTAMP) >= (
        SELECT MAX(TRY_CAST(review_date AS TIMESTAMP)) - INTERVAL '3 months'
        FROM games_reviews
    )
    GROUP BY game_id
)
SELECT g.game_name, r.review_count
FROM recent_reviews r
JOIN games_info g ON r.game_id = g.game_id
ORDER BY r.review_count DESC
LIMIT 1
```

**Risk Flags:** Complex CTE with subquery, may be slow on large datasets

### 🟠 Turn 1: AM Reviews

**Decision:** ✅ APPROVE

**Reasons:**
  - SQL structure is correct
  - Properly uses MAX(date) - INTERVAL for historical data
  - CTE makes query readable
  - Appropriate use of COUNT aggregation
  - Correct JOIN on game_id

### ✅ **Result: Consensus Reached**
**Total turns:** 1

---

**System:** 🔍 Validating SQL against schema...

**System:** ✅ Schema validation passed

**System:**

## 📝 Approved SQL

```sql
WITH recent_reviews AS (
    SELECT game_id, COUNT(*) AS review_count
    FROM games_reviews
    WHERE TRY_CAST(review_date AS TIMESTAMP) >= (
        SELECT MAX(TRY_CAST(review_date AS TIMESTAMP)) - INTERVAL '3 months'
        FROM games_reviews
    )
    GROUP BY game_id
)
SELECT g.game_name, r.review_count
FROM recent_reviews r
JOIN games_info g ON r.game_id = g.game_id
ORDER BY r.review_count DESC
LIMIT 1
```

**Validation Results:**
- ✅ Read-only: Yes
- ✅ Source tables: games_reviews, games_info
- ✅ CTEs: recent_reviews
- ✅ Schema compliance: Verified

---

**System:** ⚙️ Executing approved SQL...

**System:** ✅ Execution successful (1 rows)

---

**System:** ⚖️ Reviewing results quality...

---

**System:**

## 💬 Review Phase: Judge ↔ DS Quality Check

### 🔵 Turn 1: DS Proposes

**Goal:** Present results from approved query

**SQL:** [results shown]

### 🟠 Turn 1: Judge Reviews

**Decision:** ✅ APPROVE

**Reasons:**
  - Query executed successfully
  - Results match expected schema
  - Single row returned as expected (LIMIT 1)
  - Data quality verified

### ✅ **Result: Consensus Reached**
**Total turns:** 1

---

**System:** ✅ Analysis complete!

**System:** ✅ ChatDev analysis complete!

---

## Example with Revision: "Show customer lifetime value"

If DS makes a mistake, you'll see the dialogue:

---

### 🔵 Turn 1: DS Proposes

**Goal:** Calculate customer lifetime value

**SQL:**
```sql
SELECT customer_id, SUM(price) as ltv
FROM orders
GROUP BY customer_id
ORDER BY ltv DESC
```

**Risk Flags:** Assuming 'price' column exists

### 🟠 Turn 1: AM Reviews

**Decision:** 🔄 REVISE

**Reasons:**
  - Column 'price' not verified in schema
  - Should check column_mappings for actual revenue column
  - Missing consideration for refunds/returns

**Required Changes:**
  - Verify actual column name for order value
  - Check if refunds table exists and should be subtracted
  - Add time period filter for meaningful LTV calculation

---

### 🔵 Turn 2: DS Proposes (Revised)

**Goal:** Calculate customer lifetime value using correct schema

**SQL:**
```sql
SELECT
    customer_id,
    SUM(order_amount) - COALESCE(SUM(refund_amount), 0) as ltv,
    COUNT(DISTINCT order_id) as order_count
FROM orders
LEFT JOIN refunds USING (order_id)
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
GROUP BY customer_id
ORDER BY ltv DESC
```

**Risk Flags:** None

### 🟠 Turn 2: AM Reviews

**Decision:** ✅ APPROVE

**Reasons:**
  - Correct schema usage verified
  - Refunds properly handled
  - Time period filter applied
  - Additional metrics (order_count) valuable

### ✅ **Result: Consensus Reached**
**Total turns:** 2

---

## Key Visual Elements

### Emojis Used:
- 🔵 **DS (Assistant)** - Proposes solutions
- 🟠 **AM (Instructor)** - Reviews and critiques
- ✅ **Approve** - Ready to execute
- 🔄 **Revise** - Needs changes
- 🚫 **Block** - Cannot proceed
- 💬 **Dialogue** - Conversation display
- 📝 **SQL** - Code display
- 🎯 **Goal** - Objective
- ⚠️ **Risk Flags** - Potential issues

### Color Coding (in Streamlit):
- **Blue** = DS proposals
- **Orange** = AM critiques
- **Green** = Success/Approval
- **Yellow** = Warnings
- **Red** = Errors/Blocks

---

## Benefits of Seeing Dialogue

### 1. **Transparency**
- You see exactly how agents negotiate
- Understand why certain decisions were made
- See what was revised and why

### 2. **Learning**
- Learn SQL best practices from AM's critiques
- Understand schema validation process
- See how agents handle edge cases

### 3. **Trust**
- No black box - everything is visible
- Can verify agent reasoning
- Catch issues before execution

### 4. **Debugging**
- If something goes wrong, you see where
- Can provide better feedback to improve prompts
- Understand bottlenecks in the system

---

## Comparison: Old vs New

### Old System (No Dialogue):
```
User: Which game has most reviews?
System: 🧠 Planning analysis approach...
System: 🔬 Executing analysis...
System: ❌ Schema Validation Failed
• Table 'recent_reviews' not found in schema
```

**Problem:** You don't know what went wrong or why!

### New System (With Dialogue):
```
User: Which game has most reviews?
System: 💬 Planning Phase: AM ↔ DS Negotiation
  [Shows full dialogue with SQL, reasoning, decisions]
System: 📝 Approved SQL
  [Shows validated SQL with lineage info]
System: ✅ Execution successful
```

**Solution:** Complete transparency at every step!

---

## How to Enable

1. Start your app: `streamlit run app.py`
2. Check the box in sidebar: **"Use ChatDev-style agents"**
3. Ask any question
4. Watch the beautiful dialogue unfold! 🎭

---

**The dialogue display is AUTOMATIC - no configuration needed!** Just enable the checkbox and enjoy watching your AI agents collaborate in real-time! 🚀
