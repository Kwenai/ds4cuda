/*
 * tool_calls.h — OpenAI tool schema validation + DSML <-> OpenAI converter.
 *
 * Validates an incoming `tools` array against the OpenAI shape and
 * converts it to the DSML serialization the model emits (see
 * src/server/anthropic_endpoint.c + the prompt-rendering call sites in
 * inference_engine.cu). The reverse direction (parsing DSML tool blocks
 * out of the model output and rewriting them into OpenAI/Anthropic
 * tool_use blocks) is exposed via the same module.
 *
 * OpenAI tool entry shape:
 *   {
 *     "type": "function",
 *     "function": {
 *       "name": "<str>",
 *       "description": "<str?>",
 *       "parameters": { ...JSON Schema... }
 *     }
 *   }
 *
 * DSML invocation shape (cite ds4_server.c:1578):
 *   <｜DSML｜tool_calls>
 *   <｜DSML｜invoke name="$NAME">
 *   <｜DSML｜parameter name="$KEY" string="true|false">$VAL</｜DSML｜parameter>
 *   ...
 *   </｜DSML｜invoke>
 *   </｜DSML｜tool_calls>
 *
 * The validator returns 0 on success, <0 with err[0] populated on failure.
 * The converters are best-effort — well-formed input passes through, ill-
 * formed input returns NULL or an empty string and does not crash.
 */
#ifndef DS4CUDA_TOOL_CALLS_H
#define DS4CUDA_TOOL_CALLS_H

#include <stdbool.h>
#include <stddef.h>

#include "../cjson_min/cjson_min.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Validate an OpenAI `tools` JSON array. Returns 0 on success, -1 on
 * malformation. err / err_len receives a human-readable explanation
 * (truncation safe). */
int ds4cuda_validate_openai_tools(const cjson *tools, char *err, size_t err_len);

/* For each function-typed entry in `tools`, append a DSML-flavored line to
 * `out` (caller-owned heap buffer). Layout matches ds4_server.c
 * append_dsml_text_escaped + append_tools_prompt_text. Returns a malloc()'d
 * string (possibly empty). NULL on internal alloc failure. Skeleton:
 * we emit only the "Available Tool Schemas" footer text — the system
 * prompt boilerplate (the long "## Tools" block) is left to the integrated build
 * because it embeds reasoning-mode flags that aren't exposed at this level. */
char *ds4cuda_dsml_render_tools(const cjson *openai_tools);

/* Stub conversion: take a single OpenAI assistant message's `tool_calls`
 * array (i.e. {id, type:"function", function:{name, arguments}}) and emit
 * the equivalent DSML <｜DSML｜tool_calls>...</｜DSML｜tool_calls> block. The
 * arguments string is parsed as JSON to enumerate keys; an unparsable
 * arguments value falls back to a single string=\"false\" parameter named
 * "args" carrying the raw JSON. */
char *ds4cuda_dsml_render_assistant_tool_calls(const cjson *tool_calls_array);

/* Parse a single DSML <｜DSML｜invoke name="..."> ... </｜DSML｜invoke>
 * block out of `text` and emit the equivalent OpenAI tool_call JSON
 * fragment {"id":"call_X","type":"function","function":{...}}. Returns
 * NULL when no invoke block is present. The parser is *strict*: it
 * scans for the literal opening and closing markers and captures the
 * name + each parameter element. */
char *ds4cuda_openai_render_tool_call(const char *dsml_text);

/* ------------------------------------------------------------------ */
/* Anthropic-flavored entry points.                                    */
/*                                                                     */
/* Anthropic /v1/messages "tools" entries do not have a "type:function" */
/* wrapper. The shape is flat:                                          */
/*   {"name": ..., "description": ..., "input_schema": { ...JSON Sch }} */
/* The DSML rendering is identical to the OpenAI flavor (we only emit  */
/* one JSON-flavored line per tool); we expose a separate function so   */
/* the endpoint code reads as a 1:1 mapping from input to output.       */
/* ------------------------------------------------------------------ */

/* Validate an Anthropic tools array (returns 0 on success, -1 with err
 * populated on failure). NULL tools is accepted as "no tools". */
int ds4cuda_validate_anthropic_tools(const cjson *tools, char *err, size_t err_len);

/* Render an Anthropic tools array into the same DSML "schema list" used
 * by the OpenAI flavor. Caller frees with free(); never NULL on success
 * (an empty input yields an empty allocated string). */
char *ds4cuda_dsml_render_anthropic_tools(const cjson *anthropic_tools);

/* Convert a single Anthropic content_block of type=="tool_use" into the
 * DSML <｜DSML｜tool_calls> block fragment (one invoke). The Anthropic
 * shape is {"type":"tool_use","id":"toolu_...","name":"X","input":{...}}.
 * Unlike OpenAI (where arguments is a *string* of JSON), Anthropic's
 * "input" is already a JSON object, so we walk its keys directly.
 *
 * Returns malloc()'d DSML text; "" if the block is empty/invalid. NULL on
 * alloc failure. */
char *ds4cuda_dsml_render_anthropic_tool_use(const cjson *block);

/* Convert an Anthropic assistant content array (which may contain a mix
 * of {type:"text"} and {type:"tool_use"} blocks) into a single DSML
 * payload that can be rendered as the assistant turn body: the text parts
 * are concatenated first, then a single <｜DSML｜tool_calls> block is
 * appended carrying every tool_use entry.
 *
 * Returns malloc()'d UTF-8 string. */
char *ds4cuda_dsml_render_anthropic_assistant_blocks(const cjson *content_array);

/* Parse a single DSML <｜DSML｜invoke ...> block out of `dsml_text` and
 * build an Anthropic content_block JSON object
 *   {"type":"tool_use","id":"toolu_X","name":"X","input":{...}}
 * Returns malloc()'d JSON string; NULL when the input contains no invoke
 * block. */
char *ds4cuda_anthropic_render_tool_use(const char *dsml_text);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_TOOL_CALLS_H */
