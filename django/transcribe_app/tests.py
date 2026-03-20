# transcribe_app/tests.py
# (C) 2024–2026
# Licensed under Apache License Version 2.0
#
# Comprehensive test suite for the SUSI Translator transcribe_app.
# Covers: utility functions, serializers, and REST API integration tests.

import time
import json
import base64
import numpy as np
from unittest.mock import patch, MagicMock

from django.test import TestCase, override_settings
from rest_framework.test import APIClient
from rest_framework import status

from transcribe_app.serializers import (
    TranscribeInputSerializer,
    TranscribeResponseSerializer,
    TranscriptResponseSerializer,
    ListTranscriptsResponseSerializer,
    SizeResponseSerializer,
)
from transcribe_app.transcribe_utils import (
    is_valid,
    clean_old_transcripts,
    add_to_audio_stack,
    get_transcripts,
    transcriptsd,
    audio_stacks,
)


# =============================================================================
# Unit Tests — is_valid()
# =============================================================================

class IsValidTestCase(TestCase):
    """Tests for the transcript validation function in transcribe_utils.py."""

    def test_valid_normal_sentence(self):
        """A normal English sentence should be valid."""
        self.assertTrue(is_valid("Hello, this is a test sentence."))

    def test_valid_single_word(self):
        """A single valid ASCII word should be valid."""
        self.assertTrue(is_valid("Hello"))

    def test_valid_sentence_with_numbers(self):
        """A sentence containing numbers and punctuation should be valid."""
        self.assertTrue(is_valid("The event starts at 3:00 PM in Room 42."))

    def test_empty_string_is_invalid(self):
        """An empty string should be invalid."""
        self.assertFalse(is_valid(""))

    def test_none_is_invalid(self):
        """None input should be invalid."""
        self.assertFalse(is_valid(None))

    def test_forbidden_phrase_thank_you(self):
        """Transcripts containing 'thank you' should be invalid (hallucination)."""
        self.assertFalse(is_valid("Thank you for watching this video."))

    def test_forbidden_phrase_thanks_for_watching(self):
        """Transcripts containing 'thanks for watching' should be invalid."""
        self.assertFalse(is_valid("Thanks for watching!"))

    def test_forbidden_phrase_bye(self):
        """Transcripts containing 'bye!' should be invalid."""
        self.assertFalse(is_valid("Bye!"))

    def test_forbidden_phrase_click_click(self):
        """Transcripts with 'click click' should be invalid."""
        self.assertFalse(is_valid("click click"))

    def test_forbidden_phrase_cough_cough(self):
        """Transcripts with 'cough cough' should be invalid."""
        self.assertFalse(is_valid("cough cough"))

    def test_forbidden_exact_string_eh(self):
        """The exact string 'eh.' should be invalid."""
        self.assertFalse(is_valid("eh."))

    def test_forbidden_exact_string_bye_dot(self):
        """The exact string 'bye.' should be invalid."""
        self.assertFalse(is_valid("bye."))

    def test_forbidden_exact_string_its_fine(self):
        """The exact string \"it's fine\" should be invalid."""
        self.assertFalse(is_valid("it's fine"))

    def test_long_word_over_40_chars(self):
        """Transcripts with words longer than 40 characters should be invalid."""
        long_word = "a" * 41
        self.assertFalse(is_valid(f"This has a {long_word} word."))

    def test_word_exactly_40_chars_is_valid(self):
        """A word of exactly 40 characters should still be valid."""
        word_40 = "a" * 40
        self.assertTrue(is_valid(f"This has a {word_40} word."))

    def test_non_ascii_only_is_invalid(self):
        """A transcript with only non-ASCII or space characters should be invalid."""
        self.assertFalse(is_valid("   "))  # only spaces (code > 32 check fails)

    def test_forbidden_korean_characters(self):
        """Transcripts with specific Korean characters should be invalid."""
        self.assertFalse(is_valid("뉴스"))

    def test_forbidden_phrase_case_insensitive(self):
        """Forbidden phrase check should be case-insensitive."""
        self.assertFalse(is_valid("THANK YOU very much"))

    def test_valid_sentence_not_matching_forbidden(self):
        """A sentence with 'bye' (without dot) as a substring in a valid sentence should be valid."""
        self.assertTrue(is_valid("I will sit nearby and listen."))


# =============================================================================
# Unit Tests — clean_old_transcripts()
# =============================================================================

class CleanOldTranscriptsTestCase(TestCase):
    """Tests for the transcript cleanup function in transcribe_utils.py."""

    def setUp(self):
        """Clear transcript storage before each test."""
        transcriptsd.clear()

    def tearDown(self):
        """Clear transcript storage after each test."""
        transcriptsd.clear()

    def test_old_transcripts_are_removed(self):
        """Transcripts older than 2 hours should be removed."""
        tenant = "test_tenant_old"
        three_hours_ago = str(int((time.time() - 3 * 3600) * 1000))
        transcriptsd[tenant] = {
            three_hours_ago: {'transcript': 'Old transcript', 'translated': False}
        }

        clean_old_transcripts()

        # The tenant should be cleaned up entirely
        self.assertNotIn(tenant, transcriptsd)

    def test_recent_transcripts_are_kept(self):
        """Transcripts less than 2 hours old should be kept."""
        tenant = "test_tenant_recent"
        now = str(int(time.time() * 1000))
        transcriptsd[tenant] = {
            now: {'transcript': 'Recent transcript', 'translated': False}
        }

        clean_old_transcripts()

        self.assertIn(tenant, transcriptsd)
        self.assertIn(now, transcriptsd[tenant])

    def test_mixed_old_and_recent(self):
        """Only old transcripts should be removed; recent ones should remain."""
        tenant = "test_tenant_mixed"
        three_hours_ago = str(int((time.time() - 3 * 3600) * 1000))
        now = str(int(time.time() * 1000))
        transcriptsd[tenant] = {
            three_hours_ago: {'transcript': 'Old', 'translated': False},
            now: {'transcript': 'Recent', 'translated': False},
        }

        clean_old_transcripts()

        self.assertIn(tenant, transcriptsd)
        self.assertNotIn(three_hours_ago, transcriptsd[tenant])
        self.assertIn(now, transcriptsd[tenant])

    def test_empty_tenant_is_removed(self):
        """If all transcripts are removed from a tenant, the tenant should be removed."""
        tenant = "test_tenant_empty"
        old_time = str(int((time.time() - 4 * 3600) * 1000))
        transcriptsd[tenant] = {
            old_time: {'transcript': 'Old transcript', 'translated': False}
        }

        clean_old_transcripts()

        self.assertNotIn(tenant, transcriptsd)


# =============================================================================
# Unit Tests — add_to_audio_stack() & get_transcripts()
# =============================================================================

class AudioStackAndTranscriptsTestCase(TestCase):
    """Tests for audio stack management and transcript retrieval."""

    def setUp(self):
        """Clear storage before each test."""
        transcriptsd.clear()
        audio_stacks.clear()

    def tearDown(self):
        """Clear storage after each test."""
        transcriptsd.clear()
        audio_stacks.clear()

    def test_add_to_new_tenant_creates_queue(self):
        """Adding audio to a new tenant should create a queue."""
        add_to_audio_stack("new_tenant", "chunk_1", "dGVzdA==", "en", "de")
        self.assertIn("new_tenant", audio_stacks)
        self.assertFalse(audio_stacks["new_tenant"].empty())

    def test_add_multiple_chunks_same_tenant(self):
        """Adding multiple chunks to the same tenant should all go into the same queue."""
        add_to_audio_stack("tenant_a", "chunk_1", "dGVzdA==", "en", "de")
        add_to_audio_stack("tenant_a", "chunk_2", "dGVzdA==", "en", "de")
        self.assertEqual(audio_stacks["tenant_a"].qsize(), 2)

    def test_add_to_different_tenants(self):
        """Each tenant should have its own isolated audio queue."""
        add_to_audio_stack("tenant_x", "chunk_1", "dGVzdA==", "en", "de")
        add_to_audio_stack("tenant_y", "chunk_1", "dGVzdA==", "en", "fr")
        self.assertIn("tenant_x", audio_stacks)
        self.assertIn("tenant_y", audio_stacks)

    def test_get_transcripts_unknown_tenant(self):
        """Getting transcripts for a non-existent tenant should return empty dict."""
        result = get_transcripts("nonexistent_tenant")
        self.assertEqual(result, {})

    def test_get_transcripts_known_tenant(self):
        """Getting transcripts for a known tenant should return its transcript dict."""
        tenant = "known_tenant"
        chunk_id = str(int(time.time() * 1000))
        transcriptsd[tenant] = {
            chunk_id: {'transcript': 'Hello world', 'translated': False}
        }
        result = get_transcripts(tenant)
        self.assertIn(chunk_id, result)
        self.assertEqual(result[chunk_id]['transcript'], 'Hello world')


# =============================================================================
# Unit Tests — Serializers
# =============================================================================

class TranscribeInputSerializerTestCase(TestCase):
    """Tests for the TranscribeInputSerializer."""

    def test_valid_data_all_fields(self):
        """Serializer should be valid when all fields are provided."""
        data = {
            'audio_b64': 'dGVzdGluZw==',
            'chunk_id': '1234567890',
            'tenant_id': 'my_tenant',
            'translate_from': 'en',
            'translate_to': 'de',
        }
        serializer = TranscribeInputSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_valid_data_required_only(self):
        """Serializer should be valid with only required fields; defaults should apply."""
        data = {
            'audio_b64': 'dGVzdGluZw==',
            'chunk_id': '1234567890',
        }
        serializer = TranscribeInputSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data.get('tenant_id'), '0000')

    def test_invalid_missing_audio(self):
        """Serializer should be invalid when audio_b64 is missing."""
        data = {
            'chunk_id': '1234567890',
        }
        serializer = TranscribeInputSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('audio_b64', serializer.errors)

    def test_invalid_missing_chunk_id(self):
        """Serializer should be invalid when chunk_id is missing."""
        data = {
            'audio_b64': 'dGVzdGluZw==',
        }
        serializer = TranscribeInputSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('chunk_id', serializer.errors)

    def test_invalid_empty_data(self):
        """Serializer should be invalid with empty data."""
        serializer = TranscribeInputSerializer(data={})
        self.assertFalse(serializer.is_valid())


class TranscribeResponseSerializerTestCase(TestCase):
    """Tests for the TranscribeResponseSerializer."""

    def test_valid_response(self):
        """Response serializer should accept valid response data."""
        data = {
            'chunk_id': '1234567890',
            'tenant_id': '0000',
            'status': 'processing',
        }
        serializer = TranscribeResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)


class SizeResponseSerializerTestCase(TestCase):
    """Tests for the SizeResponseSerializer."""

    def test_valid_size(self):
        """Size serializer should accept an integer size."""
        serializer = SizeResponseSerializer(data={'size': 5})
        self.assertTrue(serializer.is_valid(), serializer.errors)


# =============================================================================
# Integration Tests — API Views
# =============================================================================

class TranscribeAPITestCase(TestCase):
    """Integration tests for the /api/transcribe endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()
        audio_stacks.clear()

    def tearDown(self):
        transcriptsd.clear()
        audio_stacks.clear()

    def test_transcribe_valid_payload(self):
        """POST /api/transcribe with valid data should return 200."""
        # Generate minimal valid audio (1 second of silence as int16)
        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        payload = {
            'audio_b64': audio_b64,
            'chunk_id': str(int(time.time() * 1000)),
            'tenant_id': 'test_api',
        }

        response = self.client.post(
            '/api/transcribe',
            data=json.dumps(payload),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'processing')
        self.assertEqual(response.data['tenant_id'], 'test_api')

    def test_transcribe_invalid_payload_missing_audio(self):
        """POST /api/transcribe without audio_b64 should return 400."""
        payload = {
            'chunk_id': '1234567890',
        }
        response = self.client.post(
            '/api/transcribe',
            data=json.dumps(payload),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_transcribe_invalid_payload_missing_chunk_id(self):
        """POST /api/transcribe without chunk_id should return 400."""
        payload = {
            'audio_b64': 'dGVzdA==',
        }
        response = self.client.post(
            '/api/transcribe',
            data=json.dumps(payload),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_transcribe_default_tenant_id(self):
        """POST /api/transcribe without tenant_id should default to '0000'."""
        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        payload = {
            'audio_b64': audio_b64,
            'chunk_id': str(int(time.time() * 1000)),
        }
        response = self.client.post(
            '/api/transcribe',
            data=json.dumps(payload),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['tenant_id'], '0000')


class GetLatestTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/get_latest_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_no_transcripts_returns_empty(self):
        """GET /api/get_latest_transcript with no data should return empty response."""
        response = self.client.get('/api/get_latest_transcript', {'tenant_id': 'empty_tenant'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {})

    def test_returns_latest_transcripts(self):
        """GET /api/get_latest_transcript should return the most recent transcripts."""
        tenant = "api_test_tenant"
        now = int(time.time() * 1000)
        transcriptsd[tenant] = {
            str(now - 3000): {'transcript': 'First sentence.', 'translated': False},
            str(now - 2000): {'transcript': 'Second sentence.', 'translated': False},
            str(now - 1000): {'transcript': 'Third sentence.', 'translated': False},
        }

        response = self.client.get('/api/get_latest_transcript', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Should contain transcript entries
        self.assertGreater(len(response.data), 0)


class ListTranscriptsAPITestCase(TestCase):
    """Integration tests for the /api/list_transcripts endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_list_empty_tenant(self):
        """GET /api/list_transcripts for empty tenant should return empty list."""
        response = self.client.get('/api/list_transcripts', {'tenant_id': 'no_data'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcripts'], [])

    def test_list_transcripts_with_data(self):
        """GET /api/list_transcripts should return all transcripts for the tenant."""
        tenant = "list_test"
        now = int(time.time() * 1000)
        transcriptsd[tenant] = {
            str(now - 1000): {'transcript': 'Line one.', 'translated': False},
            str(now): {'transcript': 'Line two.', 'translated': False},
        }

        response = self.client.get('/api/list_transcripts', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['transcripts']), 2)

    def test_list_transcripts_with_range_filter(self):
        """GET /api/list_transcripts with from/until params should filter results."""
        tenant = "range_test"
        now = int(time.time() * 1000)
        chunk_1 = str(now - 5000)
        chunk_2 = str(now - 3000)
        chunk_3 = str(now - 1000)
        transcriptsd[tenant] = {
            chunk_1: {'transcript': 'Too early.', 'translated': False},
            chunk_2: {'transcript': 'In range.', 'translated': False},
            chunk_3: {'transcript': 'Also in range.', 'translated': False},
        }

        response = self.client.get('/api/list_transcripts', {
            'tenant_id': tenant,
            'from': chunk_2,
            'until': chunk_3,
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['transcripts']), 2)


class TranscriptsSizeAPITestCase(TestCase):
    """Integration tests for the /api/transcripts_size endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_size_empty_tenant(self):
        """GET /api/transcripts_size for empty tenant should return size 0."""
        response = self.client.get('/api/transcripts_size', {'tenant_id': 'none'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['size'], 0)

    def test_size_with_data(self):
        """GET /api/transcripts_size should return the correct count."""
        tenant = "size_test"
        now = int(time.time() * 1000)
        transcriptsd[tenant] = {
            str(now - 2000): {'transcript': 'A.', 'translated': False},
            str(now - 1000): {'transcript': 'B.', 'translated': False},
            str(now): {'transcript': 'C.', 'translated': False},
        }

        response = self.client.get('/api/transcripts_size', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['size'], 3)


class GetTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/get_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_get_existing_transcript(self):
        """GET /api/get_transcript with valid chunk_id should return the transcript."""
        tenant = "get_test"
        chunk_id = str(int(time.time() * 1000))
        transcriptsd[tenant] = {
            chunk_id: {'transcript': 'Found this.', 'translated': False}
        }

        response = self.client.get('/api/get_transcript', {
            'tenant_id': tenant,
            'chunk_id': chunk_id,
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'Found this.')

    def test_get_nonexistent_chunk_id(self):
        """GET /api/get_transcript with unknown chunk_id should return empty transcript."""
        tenant = "get_miss_test"
        transcriptsd[tenant] = {
            '999': {'transcript': 'Something.', 'translated': False}
        }

        response = self.client.get('/api/get_transcript', {
            'tenant_id': tenant,
            'chunk_id': '000',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], '')

    def test_get_transcript_empty_tenant(self):
        """GET /api/get_transcript for empty tenant should return chunk_id '-1'."""
        response = self.client.get('/api/get_transcript', {
            'tenant_id': 'empty',
            'chunk_id': '12345',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['chunk_id'], '-1')


class DeleteTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/delete_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_delete_existing_transcript(self):
        """GET /api/delete_transcript should remove and return the transcript."""
        tenant = "del_test"
        chunk_id = str(int(time.time() * 1000))
        transcriptsd[tenant] = {
            chunk_id: {'transcript': 'Delete me.', 'translated': False}
        }

        response = self.client.get('/api/delete_transcript', {
            'tenant_id': tenant,
            'chunk_id': chunk_id,
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'Delete me.')
        # Verify it's gone
        self.assertNotIn(chunk_id, transcriptsd.get(tenant, {}))

    def test_delete_nonexistent_transcript(self):
        """GET /api/delete_transcript with unknown chunk_id should return empty."""
        response = self.client.get('/api/delete_transcript', {
            'tenant_id': 'del_empty',
            'chunk_id': '99999',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], '')


class GetFirstTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/get_first_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_get_first_transcript(self):
        """GET /api/get_first_transcript should return the earliest transcript."""
        tenant = "first_test"
        now = int(time.time() * 1000)
        chunk_early = str(now - 5000)
        chunk_late = str(now - 1000)
        transcriptsd[tenant] = {
            chunk_early: {'transcript': 'First!', 'translated': False},
            chunk_late: {'transcript': 'Second!', 'translated': False},
        }

        response = self.client.get('/api/get_first_transcript', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'First!')
        self.assertEqual(response.data['chunk_id'], chunk_early)

    def test_get_first_transcript_with_from_param(self):
        """GET /api/get_first_transcript with 'from' should skip older chunks."""
        tenant = "first_from_test"
        now = int(time.time() * 1000)
        chunk_early = str(now - 5000)
        chunk_mid = str(now - 3000)
        chunk_late = str(now - 1000)
        transcriptsd[tenant] = {
            chunk_early: {'transcript': 'Too early.', 'translated': False},
            chunk_mid: {'transcript': 'This one.', 'translated': False},
            chunk_late: {'transcript': 'Later one.', 'translated': False},
        }

        response = self.client.get('/api/get_first_transcript', {
            'tenant_id': tenant,
            'from': chunk_mid,
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'This one.')

    def test_get_first_transcript_empty(self):
        """GET /api/get_first_transcript on empty tenant should return chunk_id '-1'."""
        response = self.client.get('/api/get_first_transcript', {'tenant_id': 'empty_first'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['chunk_id'], '-1')


class PopFirstTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/pop_first_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_pop_first_removes_transcript(self):
        """GET /api/pop_first_transcript should return and remove the first transcript."""
        tenant = "pop_first_test"
        now = int(time.time() * 1000)
        chunk_early = str(now - 5000)
        chunk_late = str(now - 1000)
        transcriptsd[tenant] = {
            chunk_early: {'transcript': 'Pop me.', 'translated': False},
            chunk_late: {'transcript': 'Keep me.', 'translated': False},
        }

        response = self.client.get('/api/pop_first_transcript', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'Pop me.')
        # Verify it was removed
        self.assertNotIn(chunk_early, transcriptsd.get(tenant, {}))
        # The other one should still be there
        self.assertIn(chunk_late, transcriptsd[tenant])


class PopLatestTranscriptAPITestCase(TestCase):
    """Integration tests for the /api/pop_latest_transcript endpoint."""

    def setUp(self):
        self.client = APIClient()
        transcriptsd.clear()

    def tearDown(self):
        transcriptsd.clear()

    def test_pop_latest_removes_transcript(self):
        """GET /api/pop_latest_transcript should return and remove the latest transcript."""
        tenant = "pop_latest_test"
        now = int(time.time() * 1000)
        chunk_early = str(now - 5000)
        chunk_late = str(now - 1000)
        transcriptsd[tenant] = {
            chunk_early: {'transcript': 'Keep me.', 'translated': False},
            chunk_late: {'transcript': 'Pop me.', 'translated': False},
        }

        response = self.client.get('/api/pop_latest_transcript', {'tenant_id': tenant})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transcript'], 'Pop me.')
        # Verify it was removed
        self.assertNotIn(chunk_late, transcriptsd.get(tenant, {}))
        # The earlier one should still be there
        self.assertIn(chunk_early, transcriptsd[tenant])

    def test_pop_latest_empty_tenant(self):
        """GET /api/pop_latest_transcript on empty tenant should return empty."""
        response = self.client.get('/api/pop_latest_transcript', {'tenant_id': 'pop_empty'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['chunk_id'], '-1')
