// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adversarial poisoning: detects unauthorized scrapers and injects semantic
//! noise (poisoned JSON) to disrupt automated data extraction without revealing
//! defensive capabilities.

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    Benign,
    Suspicious,
    LikelyScraper,
    ConfirmedAbuse,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseType {
    ValuePerturbation,
    FieldDuplication,
    TypeConfusion,
    PhantomFields,
    HomoglyphSubstitution,
    Cocktail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoisonConfig {
    pub enabled: bool,
    pub noise_types: Vec<NoiseType>,
    pub perturbation_range: f64,
    pub phantom_field_probability: f64,
    pub homoglyph_substitution_rate: f64,
}

impl Default for PoisonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            noise_types: vec![
                NoiseType::ValuePerturbation,
                NoiseType::PhantomFields,
                NoiseType::FieldDuplication,
            ],
            perturbation_range: 0.15,
            phantom_field_probability: 0.4,
            homoglyph_substitution_rate: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAssessment {
    pub threat_level: ThreatLevel,
    pub confidence: f64,
    pub indicators: Vec<String>,
    pub recommended_noise: NoiseType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoisonedResponse {
    pub payload: Value,
    pub noise_applied: Vec<NoiseType>,
    pub threat_level: ThreatLevel,
    pub is_poisoned: bool,
}

pub struct AdversarialPoisonGenerator {
    config: PoisonConfig,
    phantom_field_pool: Vec<(&'static str, &'static str)>,
    homoglyph_map: Vec<(char, char)>,
}

impl AdversarialPoisonGenerator {
    pub fn new() -> Self {
        Self {
            config: PoisonConfig::default(),
            phantom_field_pool: vec![
                ("internalNote", "Verified by secondary source"),
                ("crossReferenceId", "XR-7F4A2B9C"),
                ("dataQualityScore", "0.94"),
                ("lastAuditDate", "2026-03-15"),
                ("regulatoryFlag", "COMPLIANT"),
                ("hashIntegrity", "sha256:a1b2c3d4e5f6"),
                ("sourceConfidence", "HIGH"),
                ("processingTier", "TIER_1"),
                ("geohashPrecision", "7"),
                ("batchSequence", "847291"),
            ],
            homoglyph_map: vec![
                ('o', '\u{043E}'),
                ('e', '\u{0435}'),
                ('a', '\u{0430}'),
                ('p', '\u{0440}'),
                ('c', '\u{0441}'),
                ('x', '\u{0445}'),
                ('y', '\u{0443}'),
                ('i', '\u{0456}'),
            ],
        }
    }

    pub fn with_config(mut self, config: PoisonConfig) -> Self {
        self.config = config;
        self
    }

    pub fn assess_threat(
        &self,
        user_agent: Option<&str>,
        request_rate_rpm: f64,
        has_auth_token: bool,
        header_anomaly_score: f64,
    ) -> ThreatAssessment {
        let mut indicators = Vec::new();
        let mut confidence = 0.0f64;

        if request_rate_rpm > 120.0 {
            indicators.push(format!("High request rate: {:.1} req/min (threshold: 120)", request_rate_rpm));
            confidence += 0.35;
        }

        if let Some(ua) = user_agent {
            let ua_lower = ua.to_lowercase();
            if ua_lower.contains("scraper")
                || ua_lower.contains("crawl")
                || ua_lower.contains("bot")
                || ua_lower.contains("spider")
                || ua_lower.contains("wget")
                || ua_lower.contains("curl")
            {
                indicators.push(format!("Known bot UA: {}", &ua[..ua.len().min(80)]));
                confidence += 0.40;
            }
        }

        if !has_auth_token {
            indicators.push("Missing authentication token".to_string());
            confidence += 0.20;
        }

        if header_anomaly_score > 0.5 {
            indicators.push(format!("Header anomaly score: {:.2}", header_anomaly_score));
            confidence += 0.15;
        }

        confidence = confidence.min(1.0);

        let (threat_level, recommended_noise) = if confidence >= 0.80 {
            (ThreatLevel::ConfirmedAbuse, NoiseType::Cocktail)
        } else if confidence >= 0.50 {
            (ThreatLevel::LikelyScraper, NoiseType::PhantomFields)
        } else if confidence >= 0.25 {
            (ThreatLevel::Suspicious, NoiseType::ValuePerturbation)
        } else {
            (ThreatLevel::Benign, NoiseType::ValuePerturbation)
        };

        tracing::info!(
            threat_level = ?threat_level,
            confidence,
            indicator_count = indicators.len(),
            "[ADVERSARIAL POISON] Threat assessment complete"
        );

        ThreatAssessment {
            threat_level,
            confidence,
            indicators,
            recommended_noise,
        }
    }

    pub fn generate_poisoned_response(
        &self,
        template: &Value,
        assessment: &ThreatAssessment,
    ) -> PoisonedResponse {
        if !self.config.enabled || assessment.threat_level == ThreatLevel::Benign {
            return PoisonedResponse {
                payload: template.clone(),
                noise_applied: vec![],
                threat_level: assessment.threat_level.clone(),
                is_poisoned: false,
            };
        }

        let mut poisoned = template.clone();
        let mut applied_noise = Vec::new();

        let noise_types_to_apply = if matches!(assessment.recommended_noise, NoiseType::Cocktail) {
            self.config.noise_types.clone()
        } else {
            vec![assessment.recommended_noise]
        };

        for noise_type in &noise_types_to_apply {
            match noise_type {
                NoiseType::ValuePerturbation => {
                    self.apply_value_perturbation(&mut poisoned);
                    applied_noise.push(NoiseType::ValuePerturbation);
                }
                NoiseType::FieldDuplication => {
                    self.apply_field_duplication(&mut poisoned);
                    applied_noise.push(NoiseType::FieldDuplication);
                }
                NoiseType::TypeConfusion => {
                    self.apply_type_confusion(&mut poisoned);
                    applied_noise.push(NoiseType::TypeConfusion);
                }
                NoiseType::PhantomFields => {
                    self.apply_phantom_fields(&mut poisoned);
                    applied_noise.push(NoiseType::PhantomFields);
                }
                NoiseType::HomoglyphSubstitution => {
                    self.apply_homoglyph_substitution(&mut poisoned);
                    applied_noise.push(NoiseType::HomoglyphSubstitution);
                }
                NoiseType::Cocktail => {}
            }
        }

        tracing::warn!(
            threat_level = ?assessment.threat_level,
            noise_applied = ?applied_noise,
            "[ADVERSARIAL POISON] Poisoned response generated"
        );

        PoisonedResponse {
            payload: poisoned,
            noise_applied: applied_noise,
            threat_level: assessment.threat_level.clone(),
            is_poisoned: true,
        }
    }

    fn apply_value_perturbation(&self, value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            for (_, v) in obj.iter_mut() {
                if let Some(n) = v.as_f64() {
                    if n.abs() > f64::EPSILON {
                        let perturbation =
                            rand::thread_rng().gen_range(-self.config.perturbation_range..=self.config.perturbation_range);
                        *v = json!(n * (1.0 + perturbation));
                    }
                } else if v.is_object() || v.is_array() {
                    self.apply_value_perturbation(v);
                }
            }
        } else if let Some(arr) = value.as_array_mut() {
            for item in arr.iter_mut() {
                self.apply_value_perturbation(item);
            }
        }
    }

    fn apply_phantom_fields(&self, value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            let mut rng = rand::thread_rng();
            for (field_name, field_value) in &self.phantom_field_pool {
                if rng.gen::<f64>() < self.config.phantom_field_probability {
                    obj.insert(field_name.to_string(), json!(*field_value));
                }
            }
        }
    }

    fn apply_field_duplication(&self, value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            let keys: Vec<String> = obj.keys().cloned().collect();
            let mut rng = rand::thread_rng();
            for key in &keys {
                if rng.gen::<f64>() < 0.2 {
                    if let Some(existing) = obj.get(key).cloned() {
                        let dup_key = format!("{}_", key);
                        if let Some(n) = existing.as_f64() {
                            obj.insert(dup_key, json!(n * 1.05));
                        } else if let Some(s) = existing.as_str() {
                            obj.insert(dup_key, json!(format!("{} (alt)", s)));
                        }
                    }
                }
            }
        }
    }

    fn apply_type_confusion(&self, value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            let mut rng = rand::thread_rng();
            for (_, v) in obj.iter_mut() {
                if rng.gen::<f64>() < 0.15 {
                    match v {
                        Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                *v = json!(format!("{:.4}", f));
                            }
                        }
                        Value::String(s) => {
                            if let Ok(parsed) = s.parse::<f64>() {
                                *v = json!(parsed);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn apply_homoglyph_substitution(&self, value: &mut Value) {
        if let Some(s) = value.as_str() {
            let mut result = String::with_capacity(s.len());
            let mut rng = rand::thread_rng();
            for c in s.chars() {
                if rng.gen::<f64>() < self.config.homoglyph_substitution_rate {
                    let substituted = self
                        .homoglyph_map
                        .iter()
                        .find(|(orig, _)| *orig == c)
                        .map(|(_, sub)| *sub)
                        .unwrap_or(c);
                    result.push(substituted);
                } else {
                    result.push(c);
                }
            }
            *value = json!(result);
        } else if let Some(obj) = value.as_object_mut() {
            for (_, v) in obj.iter_mut() {
                self.apply_homoglyph_substitution(v);
            }
        }
    }
}

impl Default for AdversarialPoisonGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_threat_benign_request() {
        let gen = AdversarialPoisonGenerator::new();
        let assessment = gen.assess_threat(Some("Mozilla/5.0 (compatible)"), 5.0, true, 0.1);
        assert_eq!(assessment.threat_level, ThreatLevel::Benign);
        assert!(assessment.confidence < 0.25);
    }

    #[test]
    fn test_assess_threat_scraper_detected() {
        let gen = AdversarialPoisonGenerator::new();
        let assessment = gen.assess_threat(
            Some("ScrapeBot/2.0 (aggressive crawler)"),
            200.0,
            false,
            0.8,
        );
        assert!(assessment.threat_level >= ThreatLevel::Suspicious);
        assert!(!assessment.indicators.is_empty());
    }

    #[test]
    fn test_generate_poisoned_response() {
        let gen = AdversarialPoisonGenerator::new();
        let template = json!({
            "networkId": "ETH",
            "taskType": "swap",
            "score": 90,
            "confidenceScore": 0.95
        });

        let assessment = ThreatAssessment {
            threat_level: ThreatLevel::LikelyScraper,
            confidence: 0.65,
            indicators: vec!["High request rate".to_string()],
            recommended_noise: NoiseType::PhantomFields,
        };

        let poisoned = gen.generate_poisoned_response(&template, &assessment);
        assert!(poisoned.is_poisoned);
        assert!(!poisoned.noise_applied.is_empty());
        assert!(poisoned.payload.as_object().unwrap().keys().len() >= 2);
    }

    #[test]
    fn test_benign_request_not_poisoned() {
        let gen = AdversarialPoisonGenerator::new();
        let template = json!({"status": "ok"});
        let assessment = ThreatAssessment {
            threat_level: ThreatLevel::Benign,
            confidence: 0.05,
            indicators: vec![],
            recommended_noise: NoiseType::ValuePerturbation,
        };
        let result = gen.generate_poisoned_response(&template, &assessment);
        assert!(!result.is_poisoned);
        assert_eq!(result.payload, template);
    }

    #[test]
    fn test_value_perturbation_changes_numbers() {
        let gen = AdversarialPoisonGenerator::new();
        let template = json!({"value": 100.0});
        let assessment = ThreatAssessment {
            threat_level: ThreatLevel::Suspicious,
            confidence: 0.40,
            indicators: vec![],
            recommended_noise: NoiseType::ValuePerturbation,
        };
        let poisoned = gen.generate_poisoned_response(&template, &assessment);
        let perturbed = poisoned.payload["value"].as_f64().unwrap();
        assert_ne!(perturbed, 100.0);
    }
}
