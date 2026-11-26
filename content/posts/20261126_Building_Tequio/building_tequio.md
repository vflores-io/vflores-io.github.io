+++
title = 'Tequio: How I Build with AI'
date = 2026-11-26T12:00:00+05:30
draft = false
summary = "Tequio takes a rough idea through investigation, implementation and review. I’ve been using it to improve itself."
tags = ["AI Agents", "Software Engineering", "Open Source", "Tequio"]
+++

My coding agent could write the code. I kept wanting a clearer answer to **“what did we actually finish?”**

A feature could stretch across several sessions. The plan would change. A check would pass, then another fix would change the code it covered. Picking the work back up meant reconstructing what still held.

So I built **[Tequio](https://github.com/katalytics-labs/tequio)**: a workflow that keeps the brief, tasks, checks and reviews connected. Then I used it to improve Tequio. Tequioception, apparently.

<figure>
  <img src="/images/tequio/workflow.svg" alt="Tequio workflow: Briefing captures intent, Explore investigates, Design chooses an approach, Tasks defines bounded work, and Build implements, checks and reviews. Decisions, evidence and remaining work stay with the project across sessions.">
  <figcaption>Start with an idea. Keep the decisions and evidence as the work moves forward.</figcaption>
</figure>

## A few questions before the rabbit hole

The first version assumed you already had a product brief. I did, so I missed the gap.

I added a briefing step for someone arriving with just an idea. The agent asks about the people, problem and main use case, then stops when there is enough to investigate. Smaller details can wait. **Getting started shouldn’t require designing the entire product.**

You can say something as rough as “I want a tool to organise my research.” The first conversation should help you work out whose research, what is currently painful, and what a useful first version would do. Explore then investigates the gaps. You can also bring an existing brief and skip straight to exploration.

That split grew out of using the thing. I wanted help shaping an idea, without being asked to settle every database field before we understood the problem.

## “Done” needs something behind it

Take a simple example: archived items keep appearing in a task list.

Tequio helps turn that into a bounded task: reproduce it, define the expected behaviour, fix it, check that ordinary items still appear, and have a separate reviewer inspect the result. Relevant changes can make earlier evidence stale, so an old pass cannot quietly stand in for a current check.

The agent does the engineering. Tequio’s instructions guide the work; its local Python CLI records evidence and guards task transitions. Human judgement and genuine review still matter.

Coming back the next day, I want to ask **“where are we?”** and get a useful answer: what is finished, what is waiting for review, and what can happen next. A completed task should leave enough behind to inspect that answer. Otherwise I’m back to scrolling through a conversation and hoping I remember which “all checks passed” was the relevant one.

Tequio installs into the project, and the coding agent uses it through instructions and local commands. The everyday interaction can stay conversational: “help me shape this idea,” “investigate this change,” or “build the next ready task.” The structure sits underneath those requests.

## The part I’m still tuning

The process takes time. Investigation and independent review earn some of it. Repeated questions and oversized handoffs deserve scrutiny.

Dogfooding has been useful here too. If Tequio makes improving Tequio feel like filling in paperwork, I get to experience that immediately. The aim is to keep the useful friction: enough to catch a bad assumption or an incomplete fix, while keeping the next step easy to understand.

I’m using Tequio on my own work and trimming that friction as I find it. I haven’t measured a speed improvement. What I value so far is being able to see what passed, what changed and what still needs attention.

**[Tequio is open source, under MIT.](https://github.com/katalytics-labs/tequio)** The README includes a prompt you can give your coding agent to help install it. Try one small change first; tell me where the workflow helps and where it gets in your way.
