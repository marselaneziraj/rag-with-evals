# AI and Data Governance Playbook: DPIAs, Vendor Review and Incident Response

This playbook is the operational companion to the GDPR and AI Act handbooks. It
describes how this organisation runs assessments, reviews vendors and handles
incidents. Where the law sets a deadline, the internal deadline is deliberately
shorter.

## Section 1 - When an impact assessment is mandatory

An assessment is opened whenever a new processing activity meets any of the
following triggers: automated decision-making with legal or similarly
significant effects; large-scale processing of special categories of data;
systematic monitoring of a publicly accessible area; matching or combining data
sets from separate processing operations; processing of data concerning
vulnerable data subjects including children and employees; innovative use of a
new technology; or processing that prevents data subjects from exercising a
right or using a service. Two triggers mean an assessment is required; one
trigger means the privacy team makes a documented judgement call. The internal
rule is stricter than the legal minimum because the cost of an unnecessary
assessment is one week of work and the cost of a missing one is a reportable
compliance failure.

## Section 2 - The eight steps of an assessment

Step one, describe the processing: data categories, subjects, volumes, flows,
retention and recipients. Step two, record the lawful basis and, for special
categories, the Article 9 exception. Step three, assess necessity and
proportionality: state what less intrusive alternative was considered and why
it was rejected. Step four, map risks to data subjects using the likelihood and
severity scale in Section 3. Step five, identify mitigations and the residual
risk after each. Step six, consult: the data protection officer always, workers'
representatives where employees are affected, and a sample of data subjects
where feasible. Step seven, record the decision and the sign-off. Step eight,
schedule review: every twelve months, or immediately on material change to the
processing, the model or the vendor.

## Section 3 - Risk scoring scale

Likelihood is scored from 1 to 4: remote, possible, likely, almost certain.
Severity is scored from 1 to 4: limited inconvenience, significant
inconvenience, significant consequences, irreversible consequences. The product
gives a risk score from 1 to 16. A residual score of 1 to 3 is accepted by the
product owner. A score of 4 to 7 requires sign-off by the data protection
officer. A score of 8 to 11 requires sign-off by the governance committee. A
residual score of 12 or above may not be accepted internally and requires prior
consultation of the supervisory authority before processing begins.

## Section 4 - Vendor and model due diligence

Before any third-party model or data processor is used in production, the
procurement checklist must be completed and stored with the contract. The
checklist covers: the processing locations and any transfers outside the EEA
with the transfer mechanism relied on; the subprocessor list and the notice
period for changes; whether customer data is used for training and how that is
contractually excluded; the retention period for prompts, outputs and logs; the
security certifications held and their scope; the incident notification time
promised in the contract, which may not exceed 24 hours; the audit rights
granted; and, for AI vendors, the model documentation required by Articles 11
and 13 of the AI Act. A vendor that cannot state its training-data provenance
is not approved for processing personal data, irrespective of commercial
pressure.

## Section 5 - Incident response runbook

An incident is any confirmed or suspected event affecting the confidentiality,
integrity or availability of personal data, or any unintended behaviour of a
deployed AI system that affects individuals. The on-call engineer raises the
incident within 30 minutes of detection. The incident commander is appointed
within 1 hour and is never the same person as the engineer remediating. The
severity assessment is completed within 4 hours and determines the clock: for a
personal data breach the supervisory authority must be notified within 72 hours
of awareness, and the internal target is 48 hours to leave margin for the legal
review. Communication to affected individuals, where the risk is high, is
drafted in parallel and not after notification. A post-incident review is held
within 10 working days, is blameless, and produces a written action list with
named owners and dates. Every incident is recorded in the breach register
whether or not it was notifiable, because the register is the evidence of the
accountability principle.

## Section 6 - Where the AI Act and the GDPR overlap

The two regimes are cumulative, not alternative. A recruitment screening model
is high-risk under Annex III of the AI Act and, at the same time, performs
automated decision-making under Article 22 of the GDPR: the AI Act requires
human oversight by design, while the GDPR gives the individual a right to
obtain human intervention after the fact. A fundamental rights impact
assessment under Article 27 of the AI Act does not replace a data protection
impact assessment under Article 35 of the GDPR; where both are required they
are run as one exercise with two outputs. The AI Act's logging requirement of
at least six months and the GDPR's storage limitation principle pull in
opposite directions, and the resolution recorded here is to retain logs for the
minimum period required by the AI Act, pseudonymised where the log does not
need to identify an individual.

## Section 7 - Governance committee and review cadence

The governance committee meets monthly and comprises the data protection
officer, the head of engineering, the security lead, a product representative
and a legal representative. It approves assessments scoring 8 to 11, reviews
the model register quarterly, and reviews the incident register at every
meeting. The model register lists every deployed model with its purpose, risk
classification, owner, date of last evaluation, evaluation results and the date
of the next scheduled review. A model whose evaluation is more than six months
old is flagged, and a model whose evaluation is more than twelve months old is
withdrawn from production until re-evaluated.

## Section 8 - Documentation that must exist before launch

No system processing personal data is launched without the following artefacts
in the governance repository: the record of processing activity; the impact
assessment and its sign-off; the lawful basis note; the retention schedule; the
data flow diagram including subprocessors; the security review; the model card
where an AI model is used, including the evaluation results and known
limitations; the human oversight procedure naming the role responsible; the
rollback plan; and the user-facing transparency notice. The launch checklist is
reviewed by the data protection officer, and an exception requires written
approval from the governance committee recorded in the minutes.
