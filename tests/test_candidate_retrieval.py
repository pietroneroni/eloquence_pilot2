"""Offline regressions for the course pipeline; no LLM or external services required."""
import contextlib
import copy
import io
import json
import re
import unittest
from unittest.mock import patch

from university_counseling import AgentWithRag as rag
from university_counseling.agents_setup import sanitize_expert_output
from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever


def request(region='Sardegna', city=None, strict=False):
    return {'query':'ecosistemi e conservazione della natura', 'location':{
        'region':region, 'city':city, 'strict':strict, 'requirement':city or region or ''}}


def course(code, title, region='Sardegna', level='Corso di Laurea', cities=()):
    return {'key':('university',code), 'score':-0.2, 'id':code,
            'meta':{'UNIVERSITY':'University', 'COURSE':f'[{code}] {title}', 'COURSE_CODE':code,
                    'TYPE':level, 'TYPE_STATUS':'declared' if level else 'unknown',
                    'REGION':region, 'CITIES':list(cities), 'SOURCES':['https://example.test/course']},
            'text':'Contenuti del corso. Sono possibili studi magistrali successivi.'}


class Retriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return copy.deepcopy(sorted([h for h in self.hits if not kwargs.get('region') or h['meta']['REGION']==kwargs['region']], key=lambda h:-h['score']))


class CandidateRetrievalTest(unittest.TestCase):
    def setUp(self):
        rag.set_dialog_language("Italian")
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.retriever = Retriever([
            course('001','SCIENZE FORESTALI E AMBIENTALI', cities=['Sassari']),
            course('002','SCIENZE NATURALI'), course('003','BIOLOGIA', region='Sicilia'),
            course('004','ECOLOGIA', level='Laurea Magistrale'),
        ])
        self.flow = rag.UniversityCounselorFlowOrchestrator(retriever=self.retriever, dialog_language='Italian',
                                                          candidate_pool_size=6,max_ctx_chars=2200)
        self.flow._slots.intended_level='bachelor'
        self.flow._listener_memory={'explicit':{},'inferred':{'field_of_interest':['environmental_science','life_sciences']}}
        rag._LAST_RAG_STATE.retrieval_request=request()

    def tearDown(self):
        self.output.__exit__(None,None,None)
        rag.clear_last_rag_state()

    def retrieve(self, req=None):
        if req is not None:rag._LAST_RAG_STATE.retrieval_request=req
        return self.flow._get_llm_candidate_pool(self.flow._build_query())

    def test_forestry_survives_and_declared_master_does_not(self):
        result, options=self.retrieve()
        self.assertTrue(any('[001]' in o for o in options))
        self.assertFalse(any('[004]' in o for o in options))
        self.assertLessEqual(len(result.short_ctx),2200)
        self.assertEqual(len(options),len(set(options)))
        self.assertEqual({q for q,_ in self.retriever.calls}, {request()["query"], 'environmental science', 'life sciences'})

    def test_flexible_region_keeps_alternatives_and_strict_region_excludes_them(self):
        _, options=self.retrieve()
        self.assertIn('[003]',options[-1])
        _, options=self.retrieve(request(strict=True))
        self.assertEqual(len(options),2)
        self.assertFalse(any('[003]' in o for o in options))

    def test_strict_city_requires_documented_campus(self):
        _, options=self.retrieve(request(city='Sassari',strict=True))
        self.assertEqual(options,['University | [001] SCIENZE FORESTALI E AMBIENTALI'])
        _, options=self.retrieve(request(city='Aosta',strict=True))
        self.assertEqual(options,[])

    def test_unknown_level_is_preserved_as_unverified(self):
        self.retriever.hits=[course('005','SCIENZE AMBIENTALI',level='')]
        result,options=self.retrieve()
        self.assertEqual(len(options),1)
        self.assertIn('TIPO: unknown',result.short_ctx)

    def test_distinct_codes_with_similar_titles_are_not_merged(self):
        self.retriever.hits=[course('001','SCIENZE AMBIENTALI'),course('002','SCIENZE AMBIENTALI')]
        _,options=self.retrieve()
        self.assertEqual(len(options),2)

    def test_formatter_does_not_invent_missing_choices(self):
        result,options=self.retrieve()
        rag.update_last_rag_state(result.short_ctx,phase='advice',advice_options=options)
        single = sanitize_expert_output('Riepilogo: Proposte.\n1. '+options[0])
        self.assertIn(options[0], single)
        self.assertNotIn('2.', single)
        rag.update_last_rag_state(result.short_ctx,phase='advice',advice_options=options)
        text='Riepilogo: Proposte.\n'+'\n'.join(f'{i+1}. {o}' for i,o in enumerate(options))
        formatted=sanitize_expert_output(text)
        self.assertEqual(sanitize_expert_output(formatted),formatted)

    def test_zero_choices_remains_a_resumable_advice_turn(self):
        self.flow._advice_given=True
        self.flow._flow_stage='qa'
        rag._LAST_RAG_STATE.advice_location_policy=request(city='Aosta',strict=True)['location']
        rag.update_last_rag_state('',phase='advice',advice_options=[])
        text=sanitize_expert_output('Aggiornamento: Non posso verificare una sede compatibile.')
        self.assertIn('Vuoi modificare la ricerca?',text)
        self.assertEqual(rag._LAST_RAG_STATE.advice_options,[])
        self.assertIn('POST_ADVICE_STATEMENT',self.flow.instruct([], 'Ritiro il vincolo'))

    def test_control_preserves_exact_region(self):
        rag.update_last_rag_state('',phase='background_validate_final')
        payload={'intent':'background_valid','retrieval':request()}
        text=rag.strip_and_apply_turn_control('<TURN_CONTROL>'+json.dumps(payload)+'</TURN_CONTROL>')
        self.assertIn('Sardegna',text)
        self.assertEqual(rag._LAST_RAG_STATE.retrieval_request,request())

    def test_invalid_request_rejected_before_state_mutation(self):
        bad={'intent':'profile_update','temporary':False,'region':'Toscana','retrieval':request()}
        previous=copy.deepcopy(rag._LAST_RAG_STATE.retrieval_request)
        with self.assertRaises(rag.TurnControlError):rag.parse_turn_control('<TURN_CONTROL>'+json.dumps(bad)+'</TURN_CONTROL>','qa_statement')
        self.assertEqual(rag._LAST_RAG_STATE.retrieval_request,previous)

    def test_temporary_exploration_then_reset_keeps_confirmed_request(self):
        self.retrieve()
        confirmed=copy.deepcopy(self.flow._confirmed_retrieval_request)
        temporary={'intent':'profile_update','temporary':True,'region':'Toscana','retrieval':request('Toscana')}
        rag.update_last_rag_state('',phase='qa_statement')
        rag.strip_and_apply_turn_control('<TURN_CONTROL>'+json.dumps(temporary)+'</TURN_CONTROL>')
        with patch.object(self.flow,'_instruct',return_value='test'):
            self.flow.instruct([],rag.INTERNAL_PROFILE_REFRESH_TOKEN)
        self.assertEqual(self.flow._confirmed_retrieval_request,confirmed)
        self.assertEqual(rag._LAST_RAG_STATE.retrieval_request['location']['region'],'Toscana')
        reset={'intent':'profile_update','temporary':True,'reset_exploration':True,'retrieval':confirmed}
        rag.update_last_rag_state('',phase='qa_statement')
        rag.strip_and_apply_turn_control('<TURN_CONTROL>'+json.dumps(reset)+'</TURN_CONTROL>')
        with patch.object(self.flow,'_instruct',return_value='test'):
            self.flow.instruct([],rag.INTERNAL_PROFILE_REFRESH_TOKEN)
        self.assertEqual(self.flow._exploration_patch,{})
        self.assertEqual(rag._LAST_RAG_STATE.retrieval_request,confirmed)
        self.flow.reset()
        self.assertEqual(self.flow._confirmed_retrieval_request,{})

    def test_logged_economics_title_abbreviation_is_canonicalized(self):
        options = [
            'University of Foggia | [1366] ECONOMIA AZIENDALE AMMINISTRAZIONE DELLE AZIENDE/MANAGEMENT',
            'University of Trento | [0117G] ECONOMIA E MANAGEMENT',
            'University of Trento | [0116G] GESTIONE AZIENDALE',
        ]
        rag.update_last_rag_state('', phase='advice', advice_options=options)
        raw = 'Riepilogo: Proposte.\n1. University of Foggia | [1366] ECONOMIA AZIENDALE\n2. '+options[1]+'\n3. '+options[2]
        result = sanitize_expert_output(raw)
        self.assertEqual(rag._LAST_RAG_STATE.advice_options, options)
        self.assertIn(options[0], result)
        self.assertEqual(sanitize_expert_output(result), result)

    def test_identity_matching_preserves_order_and_rejects_wrong_university(self):
        options = ['University of Sassari | [A106] INGEGNERIA INFORMATICA Inglese Italiano',
                   'University of Trento | [INT - TPAL] TECNICHE DELLA PREVENZIONE']
        answer = '1. University of Trento | [INT - TPAL] PREVENZIONE 2. University of Sassari | [A106] INGEGNERIA INFORMATICA'
        self.assertEqual(rag._extract_advice_options_from_answer(answer, options), options[::-1])
        self.assertEqual(rag._extract_advice_options_from_answer('1. University of Trento | [A106] INGEGNERIA INFORMATICA', options), [])
        self.assertEqual(rag._extract_advice_options_from_answer('1. University of Sassari | [OTHER] INFORMATICA', options), [])

    def test_duplicate_or_ambiguous_codes_do_not_create_choices(self):
        option = 'University | [001] BIOLOGIA'
        self.assertEqual(rag._extract_advice_options_from_answer('1. '+option+' 2. '+option, [option]), [option])
        self.assertEqual(rag._extract_advice_options_from_answer('1. '+option, [option, 'University | [001] ALTRO']), [])

    def test_background_final_prompt_has_complete_parseable_control(self):
        self.flow._bg_i = len(self.flow.BG_QUESTIONS)-1
        prompt = self.flow._build_background_validation_prompt('vorrei studiare in Lombardia')
        controls = re.findall(r'<TURN_CONTROL>.*?</TURN_CONTROL>', prompt)
        valid = [c for c in controls if 'background_valid' in c]
        self.assertEqual(len(valid), 1)
        parsed = rag.parse_turn_control(valid[0], 'background_validate_final')
        self.assertIsInstance(parsed['retrieval'], dict)
        self.assertNotIn('query', {k:v for k,v in parsed.items() if k!='retrieval'})
        self.assertIn('Do not output LISTENER_PATCH in this phase.', prompt)

    def test_profile_update_example_is_complete_json(self):
        prompt = self.flow._retrieval_request_instruction('profile_update')
        control = re.search(r'<TURN_CONTROL>.*?</TURN_CONTROL>', prompt)[0]
        self.assertEqual(rag.parse_turn_control(control, 'qa_statement')['intent'], 'profile_update')

    def test_flattened_logged_control_gets_valid_json_repair_example(self):
        raw = '<TURN_CONTROL>{"intent":"background_valid","query":"current study interests and goals in the student language","location":{"region":"Lombardia","city":null,"strict":false,"requirement":""}}</TURN_CONTROL>'
        with self.assertRaises(rag.TurnControlError) as caught:
            rag.parse_turn_control(raw, 'background_validate_final')
        example = re.search(r'<TURN_CONTROL>.*?</TURN_CONTROL>', str(caught.exception))[0]
        self.assertIsInstance(rag.parse_turn_control(example, 'background_validate_final')['retrieval'],dict)

    def test_first_advice_is_recap_despite_existing_location(self):
        self.flow._listener_memory['explicit']['academic_background']='liceo scientifico'
        self.flow._flow_stage='advice'
        rag._LAST_RAG_STATE.advice_location_policy=request()['location']
        prompt = self.flow.instruct([], rag.INTERNAL_RIASEC_CONTINUE_TOKEN)
        self.assertFalse(rag._LAST_RAG_STATE.advice_is_followup)
        self.assertIn('Start with Riepilogo:',prompt)
        self.assertNotIn('These are follow-up proposals',prompt)
        prompt = self.flow.instruct([], rag.INTERNAL_PROFILE_REFRESH_TOKEN)
        self.assertTrue(rag._LAST_RAG_STATE.advice_is_followup)
        self.assertIn('Start with Aggiornamento:',prompt)

    def test_refresh_after_empty_first_advice_is_still_followup(self):
        self.flow._listener_memory['explicit']['academic_background']='liceo scientifico'
        self.flow._flow_stage='advice'
        self.retriever.hits=[]
        self.flow.instruct([],rag.INTERNAL_RIASEC_CONTINUE_TOKEN)
        self.assertFalse(rag._LAST_RAG_STATE.advice_is_followup)
        self.assertEqual(rag._LAST_RAG_STATE.advice_options,[])
        self.flow.instruct([],rag.INTERNAL_PROFILE_REFRESH_TOKEN)
        self.assertTrue(rag._LAST_RAG_STATE.advice_is_followup)

    def test_relevant_courses_survive_flexible_local_preference(self):
        self.flow.candidate_pool_size = 2
        self.retriever.hits = [course('local', 'AGRIBUSINESS', region='Toscana'),
                               course('design1', 'DESIGN', region='Sardegna'),
                               course('design2', 'DESIGN', region='Lazio')]
        self.retriever.hits[0]['score'] = -0.8
        self.retriever.hits[1]['score'] = -0.2
        self.retriever.hits[2]['score'] = -0.3
        req = request('Toscana'); req['query'] = 'design di oggetti di uso quotidiano'
        _, options = self.retrieve(req)
        self.assertEqual(options, ['University | [design1] DESIGN', 'University | [design2] DESIGN'])
        _, options = self.retrieve(dict(req, location=dict(req['location'], strict=True)))
        self.assertEqual(options, ['University | [local] AGRIBUSINESS'])

    def test_previous_course_is_rejected_even_when_fewer_proposals_allowed(self):
        _, options = self.retrieve()
        rag.update_last_rag_state('', phase='advice', advice_options=options)
        with self.assertRaisesRegex(rag.TurnControlError, 'CURRENT OPTION_FACTS'):
            sanitize_expert_output('Aggiornamento: ricerca.\n1. '+options[0]+'\n2. University | [previous] ALTRO')
        self.assertEqual(rag._LAST_RAG_STATE.advice_options, options)
        with self.assertRaises(rag.TurnControlError):
            sanitize_expert_output('Riepilogo: ricerca.\n1. '+options[0]+'\n2. '+options[0])

    def test_qa_accepts_known_code_in_prose_but_rejects_wrong_university(self):
        option = 'University of Foggia | [1366] ECONOMIA AZIENDALE AMMINISTRAZIONE DELLE AZIENDE/MANAGEMENT'
        rag.update_last_rag_state('Gestione delle imprese.', phase='qa_other', advice_options=[option])
        for title in [option, 'University of Foggia | [1366] ECONOMIA AZIENDALE']:
            answer = title+' è coerente con i tuoi interessi.'
            self.assertEqual(rag.enforce_grounding_or_fallback(answer), answer)
        wrong = 'University of Trento | [1366] ECONOMIA AZIENDALE è coerente.'
        self.assertIn('Non posso confermare', rag.enforce_grounding_or_fallback(wrong))

    def test_unsupported_italian_fees_are_not_kept_when_entire_reply_removed(self):
        rag.update_last_rag_state('Corso di informatica.', phase='qa_other')
        answer = 'Le tasse potrebbero essere di 1.500-2.000 euro.'
        result = rag.enforce_grounding_or_fallback(answer)
        self.assertNotIn('1.500', result)
        self.assertIn('non è disponibile', result)
        self.assertNotIn('settembre', rag.enforce_grounding_or_fallback('La scadenza è a settembre.'))

    def test_empty_acknowledgement_rejected_before_state_mutation(self):
        rag.update_last_rag_state('', phase='qa_statement')
        with self.assertRaisesRegex(rag.TurnControlError, 'content_request'):
            rag.strip_and_apply_turn_control('<TURN_CONTROL>{"intent":"acknowledgement"}</TURN_CONTROL>')
        self.assertEqual(rag._LAST_RAG_STATE.phase, 'qa_statement')

    def test_refresh_feedback_provides_its_own_schema(self):
        control = {'intent':'search_refresh', 'retrieval':request()}
        with self.assertRaises(rag.TurnControlError) as caught:
            rag.parse_turn_control('<TURN_CONTROL>'+json.dumps(control)+'</TURN_CONTROL>', 'qa_statement')
        example = re.search(r'<TURN_CONTROL>.*?</TURN_CONTROL>', str(caught.exception))[0]
        self.assertEqual(rag.parse_turn_control(example, 'qa_statement'), {'intent':'search_refresh'})

    def test_complete_json_survives_missing_or_escaped_closing_tag(self):
        control = '<TURN_CONTROL>'+json.dumps({'intent':'profile_update','temporary':False,'retrieval':request()})+chr(92)+'</TURN_CONTROL>'
        self.assertEqual(rag.parse_turn_control(control, 'qa_statement')['retrieval'], request())
        self.assertEqual(rag.parse_turn_control(control.replace(chr(92)+'</TURN_CONTROL>', ''), 'qa_statement')['retrieval'], request())
        with self.assertRaises(rag.TurnControlError):
            rag.parse_turn_control('<TURN_CONTROL>{"intent":"profile_update",', 'qa_statement')
        with self.assertRaises(rag.TurnControlError):
            rag.parse_turn_control(control.replace(chr(92), 'trailing garbage'), 'qa_statement')

    def test_location_update_has_one_region_source(self):
        rag.update_last_rag_state('', phase='qa_statement')
        control = {'intent':'profile_update','temporary':False,'retrieval':request('Toscana')}
        rag.strip_and_apply_turn_control('<TURN_CONTROL>'+json.dumps(control)+'</TURN_CONTROL>')
        self.assertEqual(rag._LAST_RAG_STATE.profile_update['patch']['explicit']['region'], 'Toscana')
        self.assertEqual(rag._LAST_RAG_STATE.retrieval_request, request('Toscana'))

    def test_withdrawn_interest_clears_memory_and_slot(self):
        self.flow._listener_memory['inferred']['field_of_interest'] = ['economics_business']
        self.flow._sync_slots_from_listener_memory()
        self.flow._listener_memory = rag.apply_listener_patch(self.flow._listener_memory, {'inferred':{'field_of_interest':[]}})
        self.flow._sync_slots_from_listener_memory()
        self.assertEqual(self.flow._get_ranked_field_of_interest(), [])
        self.assertIsNone(self.flow._slots.field_of_interest)

    def test_first_recap_contains_activity_evidence(self):
        self.flow._listener_memory['explicit']['academic_background'] = 'liceo scientifico'
        self.flow._flow_stage = 'advice'
        self.flow._riasec_answers = ['non mi piace organizzare'] * 8
        prompt = self.flow.instruct([], rag.INTERNAL_RIASEC_CONTINUE_TOKEN)
        self.assertIn('non mi piace organizzare', prompt)
        self.assertIn('never turn it into a positive preference', prompt)

    def course_evidence(self, rows):
        class Table:
            def search(self): return self
            def where(self,*args,**kwargs): return self
            def select(self,*args): return self
            def limit(self,*args): return self
            def count_rows(self): return len(rows)
            def to_list(self): return rows
        retriever = object.__new__(LanceDBUniversityRetriever)
        retriever.table = Table(); retriever._course_evidence_cache = {}
        return retriever._course_evidence(rows[0]['metadata'])

    def test_declared_class_detected_without_incidental_master(self):
        meta = {'UNIVERSITY':'Sapienza', 'COURSE_CODE':'33440', 'COURSE':'[33440] ECONOMIA'}
        evidence = self.course_evidence([{'metadata':meta,'text':'Classe di laurea: LM-56\nAccesso con laurea triennale.'}])
        self.assertEqual(rag._infer_program_level(evidence['TYPE']), 'master')
        evidence = self.course_evidence([{'metadata':meta,'text':'Classe di laurea: L-18\nProsecuzione con laurea magistrale.'}])
        self.assertEqual(rag._infer_program_level(evidence['TYPE']), 'bachelor')

    def test_declared_postgraduate_course_is_not_an_unknown_bachelor_candidate(self):
        meta = {'UNIVERSITY':'University','COURSE_CODE':'12P01',
                'COURSE':'[12P01] CONSULENZA SOCIO-RELAZIONALE Corso di Perfezionamento a crediti'}
        evidence = self.course_evidence([{'metadata':meta, 'text':meta['COURSE']}])
        self.assertEqual(rag._infer_program_level(evidence['TYPE']), 'postgraduate')

    def test_flexible_search_uses_comparable_national_ranks(self):
        self.retrieve()
        self.assertTrue(self.retriever.calls)
        self.assertTrue(all(not args.get('region') and not args.get('macroarea') for _,args in self.retriever.calls))
        self.retriever.calls.clear()
        self.retrieve(request(strict=True))
        self.assertTrue(all(args['region']=='Sardegna' for _,args in self.retriever.calls))

    def test_inline_campus_comes_from_course_not_university(self):
        meta = {'UNIVERSITY':'University of Sassari','COURSE_CODE':'1175','COURSE':'[1175] SCIENZE FORESTALI Inglese Italiano'}
        evidence = self.course_evidence([{'metadata':meta, 'text':'[1175] SCIENZE FORESTALI Corso di Laurea\nCorso Interateneo No Accesso Libero Sede NUORO Dipartimento AGRARIA'}])
        self.assertEqual(evidence['CITIES'], ['NUORO'])
        self.assertEqual(rag._infer_program_level(evidence['TYPE']), 'bachelor')

    def test_course_evidence_does_not_use_incidental_master_or_university_city(self):
        class Table:
            def search(self):return self
            def where(self,*args,**kwargs):return self
            def select(self,*args):return self
            def limit(self,*args):return self
            def count_rows(self):return 2
            def to_list(self):return [
                {'metadata':{'UNIVERSITY':'University of Sassari','COURSE_CODE':'001','COURSE':'[001] SCIENZE FORESTALI'},
                 'text':'[001] SCIENZE FORESTALI Corso di Laurea\nPossibile accesso a laurea magistrale.'}]
        retriever=object.__new__(LanceDBUniversityRetriever)
        retriever.table=Table();retriever._course_evidence_cache={}
        evidence=retriever._course_evidence({'UNIVERSITY':'University of Sassari','COURSE_CODE':'001','COURSE':'[001] SCIENZE FORESTALI'})
        self.assertEqual(evidence['TYPE'],'Corso di Laurea')
        self.assertEqual(evidence['CITIES'],[])


if __name__=='__main__':unittest.main()
