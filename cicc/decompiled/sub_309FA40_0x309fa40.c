// Function: sub_309FA40
// Address: 0x309fa40
//
void __fastcall sub_309FA40(_QWORD *a1)
{
  signed __int64 v2; // r12
  _QWORD *v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  _QWORD *v6; // rax
  __m128i *v7; // rdx
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  _QWORD *v15; // rax
  _WORD *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  __m128i *v19; // rdx
  __m128i v20; // xmm0
  _QWORD *v21; // rax
  _WORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  _QWORD *v27; // rax
  _WORD *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rax
  __m128i *v31; // rdx
  __m128i v32; // xmm0
  _QWORD *v33; // rax
  _WORD *v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rax
  __m128i *v37; // rdx
  __m128i v38; // xmm0
  _QWORD *v39; // rax
  __m128i *v40; // rdx
  __int64 v41; // rdi
  __m128i v42; // xmm0
  __int64 v43; // rax
  _WORD *v44; // rdx
  __int64 v45; // rdi
  __int64 v46; // rax
  _WORD *v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // rax
  _WORD *v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rax
  _WORD *v53; // rdx
  signed __int64 v54; // r12
  _QWORD *v55; // rax
  __m128i *v56; // rdx
  __m128i v57; // xmm0
  _QWORD *v58; // rax
  _WORD *v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // rax
  __m128i *v62; // rdx
  _QWORD *v63; // rax
  _WORD *v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rax
  __m128i *v67; // rdx
  __m128i v68; // xmm0
  _QWORD *v69; // rax
  _WORD *v70; // rdx
  __int64 v71; // rdi
  __int64 v72; // rax
  void *v73; // rdx
  _QWORD *v74; // rax
  _WORD *v75; // rdx
  __int64 v76; // rdi
  __int64 v77; // rax
  void *v78; // rdx
  _QWORD *v79; // rax
  _WORD *v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // rax
  __m128i *v83; // rdx
  __m128i v84; // xmm0
  _QWORD *v85; // rax
  __m128i *v86; // rdx
  __int64 v87; // rdi
  __m128i v88; // xmm0
  __int64 v89; // rax
  _WORD *v90; // rdx
  __int64 v91; // rdi
  __int64 v92; // rax
  _WORD *v93; // rdx
  __int64 v94; // rdi
  __int64 v95; // rax
  _WORD *v96; // rdx
  __int64 v97; // rdi
  __int64 v98; // rax
  _WORD *v99; // rdx

  if ( *a1 )
  {
    v2 = a1[4] + a1[3] + a1[1] + a1[2];
    v3 = sub_CB72A0();
    v4 = (__m128i *)v3[4];
    if ( v3[3] - (_QWORD)v4 <= 0x2Bu )
    {
      sub_CB6200((__int64)v3, "===== Alias Analysis Evaluator Report =====\n", 0x2Cu);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4288730);
      qmemcpy(&v4[2], "eport =====\n", 12);
      *v4 = si128;
      v4[1] = _mm_load_si128((const __m128i *)&xmmword_4288740);
      v3[4] += 44LL;
    }
    if ( v2 )
    {
      v9 = sub_CB72A0();
      v10 = (_WORD *)v9[4];
      v11 = (__int64)v9;
      if ( v9[3] - (_QWORD)v10 <= 1u )
      {
        v11 = sub_CB6200((__int64)v9, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v10 = 8224;
        v9[4] += 2LL;
      }
      v12 = sub_CB59F0(v11, v2);
      v13 = *(__m128i **)(v12 + 32);
      if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0x1Eu )
      {
        sub_CB6200(v12, " Total Alias Queries Performed\n", 0x1Fu);
      }
      else
      {
        v14 = _mm_load_si128((const __m128i *)&xmmword_4288780);
        qmemcpy(&v13[1], "ries Performed\n", 15);
        *v13 = v14;
        *(_QWORD *)(v12 + 32) += 31LL;
      }
      v15 = sub_CB72A0();
      v16 = (_WORD *)v15[4];
      v17 = (__int64)v15;
      if ( v15[3] - (_QWORD)v16 <= 1u )
      {
        v17 = sub_CB6200((__int64)v15, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v16 = 8224;
        v15[4] += 2LL;
      }
      v18 = sub_CB59F0(v17, a1[1]);
      v19 = *(__m128i **)(v18 + 32);
      if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 0x13u )
      {
        sub_CB6200(v18, " no alias responses ", 0x14u);
      }
      else
      {
        v20 = _mm_load_si128((const __m128i *)&xmmword_4288790);
        v19[1].m128i_i32[0] = 544433523;
        *v19 = v20;
        *(_QWORD *)(v18 + 32) += 20LL;
      }
      sub_309F080(a1[1], v2);
      v21 = sub_CB72A0();
      v22 = (_WORD *)v21[4];
      v23 = (__int64)v21;
      if ( v21[3] - (_QWORD)v22 <= 1u )
      {
        v23 = sub_CB6200((__int64)v21, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v22 = 8224;
        v21[4] += 2LL;
      }
      v24 = sub_CB59F0(v23, a1[2]);
      v25 = *(__m128i **)(v24 + 32);
      if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0x14u )
      {
        sub_CB6200(v24, " may alias responses ", 0x15u);
      }
      else
      {
        v26 = _mm_load_si128((const __m128i *)&xmmword_42887A0);
        v25[1].m128i_i32[0] = 1936028526;
        v25[1].m128i_i8[4] = 32;
        *v25 = v26;
        *(_QWORD *)(v24 + 32) += 21LL;
      }
      sub_309F080(a1[2], v2);
      v27 = sub_CB72A0();
      v28 = (_WORD *)v27[4];
      v29 = (__int64)v27;
      if ( v27[3] - (_QWORD)v28 <= 1u )
      {
        v29 = sub_CB6200((__int64)v27, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v28 = 8224;
        v27[4] += 2LL;
      }
      v30 = sub_CB59F0(v29, a1[3]);
      v31 = *(__m128i **)(v30 + 32);
      if ( *(_QWORD *)(v30 + 24) - (_QWORD)v31 <= 0x18u )
      {
        sub_CB6200(v30, " partial alias responses ", 0x19u);
      }
      else
      {
        v32 = _mm_load_si128((const __m128i *)&xmmword_42887B0);
        v31[1].m128i_i8[8] = 32;
        v31[1].m128i_i64[0] = 0x7365736E6F707365LL;
        *v31 = v32;
        *(_QWORD *)(v30 + 32) += 25LL;
      }
      sub_309F080(a1[3], v2);
      v33 = sub_CB72A0();
      v34 = (_WORD *)v33[4];
      v35 = (__int64)v33;
      if ( v33[3] - (_QWORD)v34 <= 1u )
      {
        v35 = sub_CB6200((__int64)v33, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v34 = 8224;
        v33[4] += 2LL;
      }
      v36 = sub_CB59F0(v35, a1[4]);
      v37 = *(__m128i **)(v36 + 32);
      if ( *(_QWORD *)(v36 + 24) - (_QWORD)v37 <= 0x15u )
      {
        sub_CB6200(v36, " must alias responses ", 0x16u);
      }
      else
      {
        v38 = _mm_load_si128((const __m128i *)&xmmword_42887C0);
        v37[1].m128i_i32[0] = 1702063727;
        v37[1].m128i_i16[2] = 8307;
        *v37 = v38;
        *(_QWORD *)(v36 + 32) += 22LL;
      }
      sub_309F080(a1[4], v2);
      v39 = sub_CB72A0();
      v40 = (__m128i *)v39[4];
      v41 = (__int64)v39;
      if ( v39[3] - (_QWORD)v40 <= 0x31u )
      {
        v41 = sub_CB6200((__int64)v39, "  Alias Analysis Evaluator Pointer Alias Summary: ", 0x32u);
      }
      else
      {
        v42 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v40[3].m128i_i16[0] = 8250;
        *v40 = v42;
        v40[1] = _mm_load_si128((const __m128i *)&xmmword_42887D0);
        v40[2] = _mm_load_si128((const __m128i *)&xmmword_42887E0);
        v39[4] += 50LL;
      }
      v43 = sub_CB59F0(v41, 100LL * a1[1] / v2);
      v44 = *(_WORD **)(v43 + 32);
      v45 = v43;
      if ( *(_QWORD *)(v43 + 24) - (_QWORD)v44 <= 1u )
      {
        v45 = sub_CB6200(v43, "%/", 2u);
      }
      else
      {
        *v44 = 12069;
        *(_QWORD *)(v43 + 32) += 2LL;
      }
      v46 = sub_CB59F0(v45, 100LL * a1[2] / v2);
      v47 = *(_WORD **)(v46 + 32);
      v48 = v46;
      if ( *(_QWORD *)(v46 + 24) - (_QWORD)v47 <= 1u )
      {
        v48 = sub_CB6200(v46, "%/", 2u);
      }
      else
      {
        *v47 = 12069;
        *(_QWORD *)(v46 + 32) += 2LL;
      }
      v49 = sub_CB59F0(v48, 100LL * a1[3] / v2);
      v50 = *(_WORD **)(v49 + 32);
      v51 = v49;
      if ( *(_QWORD *)(v49 + 24) - (_QWORD)v50 <= 1u )
      {
        v51 = sub_CB6200(v49, "%/", 2u);
      }
      else
      {
        *v50 = 12069;
        *(_QWORD *)(v49 + 32) += 2LL;
      }
      v52 = sub_CB59F0(v51, 100LL * a1[4] / v2);
      v53 = *(_WORD **)(v52 + 32);
      if ( *(_QWORD *)(v52 + 24) - (_QWORD)v53 <= 1u )
      {
        sub_CB6200(v52, "%\n", 2u);
      }
      else
      {
        *v53 = 2597;
        *(_QWORD *)(v52 + 32) += 2LL;
      }
    }
    else
    {
      v6 = sub_CB72A0();
      v7 = (__m128i *)v6[4];
      if ( v6[3] - (_QWORD)v7 <= 0x30u )
      {
        sub_CB6200((__int64)v6, "  Alias Analysis Evaluator Summary: No pointers!\n", 0x31u);
      }
      else
      {
        v8 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v7[3].m128i_i8[0] = 10;
        *v7 = v8;
        v7[1] = _mm_load_si128((const __m128i *)&xmmword_4288760);
        v7[2] = _mm_load_si128((const __m128i *)&xmmword_4288770);
        v6[4] += 49LL;
      }
    }
    v54 = a1[8] + a1[6] + a1[5] + a1[7];
    if ( v54 )
    {
      v58 = sub_CB72A0();
      v59 = (_WORD *)v58[4];
      v60 = (__int64)v58;
      if ( v58[3] - (_QWORD)v59 <= 1u )
      {
        v60 = sub_CB6200((__int64)v58, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v59 = 8224;
        v58[4] += 2LL;
      }
      v61 = sub_CB59F0(v60, v54);
      v62 = *(__m128i **)(v61 + 32);
      if ( *(_QWORD *)(v61 + 24) - (_QWORD)v62 <= 0x1Fu )
      {
        sub_CB6200(v61, " Total ModRef Queries Performed\n", 0x20u);
      }
      else
      {
        *v62 = _mm_load_si128((const __m128i *)&xmmword_4288810);
        v62[1] = _mm_load_si128((const __m128i *)&xmmword_4288820);
        *(_QWORD *)(v61 + 32) += 32LL;
      }
      v63 = sub_CB72A0();
      v64 = (_WORD *)v63[4];
      v65 = (__int64)v63;
      if ( v63[3] - (_QWORD)v64 <= 1u )
      {
        v65 = sub_CB6200((__int64)v63, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v64 = 8224;
        v63[4] += 2LL;
      }
      v66 = sub_CB59F0(v65, a1[5]);
      v67 = *(__m128i **)(v66 + 32);
      if ( *(_QWORD *)(v66 + 24) - (_QWORD)v67 <= 0x15u )
      {
        sub_CB6200(v66, " no mod/ref responses ", 0x16u);
      }
      else
      {
        v68 = _mm_load_si128((const __m128i *)&xmmword_4288830);
        v67[1].m128i_i32[0] = 1702063727;
        v67[1].m128i_i16[2] = 8307;
        *v67 = v68;
        *(_QWORD *)(v66 + 32) += 22LL;
      }
      sub_309F080(a1[5], v54);
      v69 = sub_CB72A0();
      v70 = (_WORD *)v69[4];
      v71 = (__int64)v69;
      if ( v69[3] - (_QWORD)v70 <= 1u )
      {
        v71 = sub_CB6200((__int64)v69, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v70 = 8224;
        v69[4] += 2LL;
      }
      v72 = sub_CB59F0(v71, a1[6]);
      v73 = *(void **)(v72 + 32);
      if ( *(_QWORD *)(v72 + 24) - (_QWORD)v73 <= 0xEu )
      {
        sub_CB6200(v72, (unsigned __int8 *)" mod responses ", 0xFu);
      }
      else
      {
        qmemcpy(v73, " mod responses ", 15);
        *(_QWORD *)(v72 + 32) += 15LL;
      }
      sub_309F080(a1[6], v54);
      v74 = sub_CB72A0();
      v75 = (_WORD *)v74[4];
      v76 = (__int64)v74;
      if ( v74[3] - (_QWORD)v75 <= 1u )
      {
        v76 = sub_CB6200((__int64)v74, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v75 = 8224;
        v74[4] += 2LL;
      }
      v77 = sub_CB59F0(v76, a1[7]);
      v78 = *(void **)(v77 + 32);
      if ( *(_QWORD *)(v77 + 24) - (_QWORD)v78 <= 0xEu )
      {
        sub_CB6200(v77, (unsigned __int8 *)" ref responses ", 0xFu);
      }
      else
      {
        qmemcpy(v78, " ref responses ", 15);
        *(_QWORD *)(v77 + 32) += 15LL;
      }
      sub_309F080(a1[7], v54);
      v79 = sub_CB72A0();
      v80 = (_WORD *)v79[4];
      v81 = (__int64)v79;
      if ( v79[3] - (_QWORD)v80 <= 1u )
      {
        v81 = sub_CB6200((__int64)v79, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v80 = 8224;
        v79[4] += 2LL;
      }
      v82 = sub_CB59F0(v81, a1[8]);
      v83 = *(__m128i **)(v82 + 32);
      if ( *(_QWORD *)(v82 + 24) - (_QWORD)v83 <= 0x14u )
      {
        sub_CB6200(v82, (unsigned __int8 *)" mod & ref responses ", 0x15u);
      }
      else
      {
        v84 = _mm_load_si128((const __m128i *)&xmmword_4288840);
        v83[1].m128i_i32[0] = 1936028526;
        v83[1].m128i_i8[4] = 32;
        *v83 = v84;
        *(_QWORD *)(v82 + 32) += 21LL;
      }
      sub_309F080(a1[8], v54);
      v85 = sub_CB72A0();
      v86 = (__m128i *)v85[4];
      v87 = (__int64)v85;
      if ( v85[3] - (_QWORD)v86 <= 0x2Bu )
      {
        v87 = sub_CB6200((__int64)v85, "  Alias Analysis Evaluator Mod/Ref Summary: ", 0x2Cu);
      }
      else
      {
        v88 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        qmemcpy(&v86[2], "ef Summary: ", 12);
        *v86 = v88;
        v86[1] = _mm_load_si128((const __m128i *)&xmmword_4288890);
        v85[4] += 44LL;
      }
      v89 = sub_CB59F0(v87, 100LL * a1[5] / v54);
      v90 = *(_WORD **)(v89 + 32);
      v91 = v89;
      if ( *(_QWORD *)(v89 + 24) - (_QWORD)v90 <= 1u )
      {
        v91 = sub_CB6200(v89, "%/", 2u);
      }
      else
      {
        *v90 = 12069;
        *(_QWORD *)(v89 + 32) += 2LL;
      }
      v92 = sub_CB59F0(v91, 100LL * a1[6] / v54);
      v93 = *(_WORD **)(v92 + 32);
      v94 = v92;
      if ( *(_QWORD *)(v92 + 24) - (_QWORD)v93 <= 1u )
      {
        v94 = sub_CB6200(v92, "%/", 2u);
      }
      else
      {
        *v93 = 12069;
        *(_QWORD *)(v92 + 32) += 2LL;
      }
      v95 = sub_CB59F0(v94, 100LL * a1[7] / v54);
      v96 = *(_WORD **)(v95 + 32);
      v97 = v95;
      if ( *(_QWORD *)(v95 + 24) - (_QWORD)v96 <= 1u )
      {
        v97 = sub_CB6200(v95, "%/", 2u);
      }
      else
      {
        *v96 = 12069;
        *(_QWORD *)(v95 + 32) += 2LL;
      }
      v98 = sub_CB59F0(v97, 100LL * a1[8] / v54);
      v99 = *(_WORD **)(v98 + 32);
      if ( *(_QWORD *)(v98 + 24) - (_QWORD)v99 <= 1u )
      {
        sub_CB6200(v98, "%\n", 2u);
      }
      else
      {
        *v99 = 2597;
        *(_QWORD *)(v98 + 32) += 2LL;
      }
    }
    else
    {
      v55 = sub_CB72A0();
      v56 = (__m128i *)v55[4];
      if ( v55[3] - (_QWORD)v56 <= 0x37u )
      {
        sub_CB6200((__int64)v55, "  Alias Analysis Mod/Ref Evaluator Summary: no mod/ref!\n", 0x38u);
      }
      else
      {
        v57 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v56[3].m128i_i64[0] = 0xA216665722F646FLL;
        *v56 = v57;
        v56[1] = _mm_load_si128((const __m128i *)&xmmword_42887F0);
        v56[2] = _mm_load_si128((const __m128i *)&xmmword_4288800);
        v55[4] += 56LL;
      }
    }
  }
}
