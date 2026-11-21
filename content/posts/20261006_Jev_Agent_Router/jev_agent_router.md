+++
title = 'Choosing a Model for Every Coding-Agent Task'
date = 2026-10-06T00:00:00+08:00
draft = false
summary = "A small router can help coding agents choose a model for each subtask. The useful engineering is in the guardrails around that choice."
tags = ["LLM", "AI Agents", "Model Routing", "Claude Code", "Codex", "OpenRouter", "Cost Control"]
+++

The classifier is the easy part. The harder question is what happens around its answer.

Coding agents can hand pieces of work to subagents: one might inspect a bug, another might write a test, and another might review a design. Those tasks do not all need the same model. Sending every task to the most capable model can waste money; choosing a model by hand each time adds friction.

I built Jev Agent Router to make that choice when a subagent starts. Jev is a classifier model; I call it through OpenRouter and give it the task title and brief. It chooses among four tiers, from straightforward work to the rare case where a previous attempt at the same task has already fallen short. The main coding-agent conversation keeps its model; only the new subagent's model can change.

## Keep the input small

The router sends Jev the task title and brief, not the whole conversation transcript. That keeps the decision focused on the work the subagent has been asked to do.

Jev returns a tier choice. The router applies the relevant harness configuration to turn that choice into a model for Claude Code or Codex. The point is to pick the least costly tier that looks capable of doing the job, while still leaving the model choice configurable.

<figure>
  <img src="/images/jev-agent-router/jev-router-design.svg" alt="Design diagram: a subagent spawn flows through its task brief, Jev, the configured model ladder, an approval gate for the top Claude Code tier, and the decision log.">
  <figcaption>Design diagram: the route from a subagent request to a model choice.</figcaption>
</figure>

## The guardrails are the useful part

A classifier can be wrong. So the router can put a configured tier behind an approval step. In the default Claude Code configuration, the top tier, Fable, is gated. The router also appends decisions to a JSONL log so I can inspect what it chose and why. If I want it out of the way, there is a kill switch: set `JEV_ROUTER=off` or turn the mode off in the config.

The Claude Code and Codex integrations are separate adapters around a shared decision core. They run at subagent spawn time and apply a model to that subagent. The main conversation is left alone. The project uses Node's built-in APIs and `fetch`; it has no npm dependencies.

## What the local log says

Over eleven days, from September 23 to October 4, 2026, my local log recorded 239 Claude Code routing decisions. Of those, 231 included an OpenRouter call; their median logged latency was 688 milliseconds.

That is a measure of the extra wait, not a measure of whether Jev picked the best model. I am not treating it as an accuracy result. The more important lesson for me is the one I keep coming back to: the classifier is the easy part. An approval gate, a decision log, and a kill switch are what make the routing usable.

The code is [on GitHub](https://github.com/vflores-io/jev-agent-router).
