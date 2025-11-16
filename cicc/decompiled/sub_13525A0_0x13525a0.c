// Function: sub_13525A0
// Address: 0x13525a0
//
void __fastcall sub_13525A0(_QWORD *a1, const char *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0
  __int64 v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rdi
  __m128i v12; // xmm0
  __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // rdi
  const char *v16; // rsi
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // rdi
  __m128i v20; // xmm0
  __int64 v21; // rax
  _WORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rax
  _WORD *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __m128i v34; // xmm0
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  _WORD *v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdx
  __m128i v42; // xmm0
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  _WORD *v46; // rdx
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdx
  __m128i v50; // xmm0
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rax
  __m128i *v54; // rdx
  __int64 v55; // rdi
  __m128i v56; // xmm0
  __int64 v57; // rax
  _WORD *v58; // rdx
  __int64 v59; // rdi
  __int64 v60; // rax
  _WORD *v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rax
  _WORD *v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // r12
  __int64 v68; // rax
  __m128i *v69; // rdx
  __m128i v70; // xmm0
  __int64 v71; // rax
  _WORD *v72; // rdx
  __int64 v73; // rdi
  const char *v74; // rsi
  __int64 v75; // rax
  __m128i *v76; // rdx
  __int64 v77; // rdi
  __int64 v78; // rax
  _WORD *v79; // rdx
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rdx
  __m128i v83; // xmm0
  __int64 v84; // rdi
  __int64 v85; // rdx
  __int64 v86; // rax
  _WORD *v87; // rdx
  __int64 v88; // rdi
  __int64 v89; // rax
  void *v90; // rdx
  __int64 v91; // rdi
  __int64 v92; // rdx
  __int64 v93; // rax
  _WORD *v94; // rdx
  __int64 v95; // rdi
  __int64 v96; // rax
  void *v97; // rdx
  __int64 v98; // rdi
  __int64 v99; // rdx
  __int64 v100; // rax
  _WORD *v101; // rdx
  __int64 v102; // rdi
  __int64 v103; // rax
  __int64 v104; // rdx
  __m128i v105; // xmm0
  __int64 v106; // rdi
  __int64 v107; // rdx
  __int64 v108; // rax
  _WORD *v109; // rdx
  __int64 v110; // rdi
  __int64 v111; // rax
  __m128i *v112; // rdx
  __int64 v113; // rdi
  __int64 v114; // rdx
  __int64 v115; // rax
  _WORD *v116; // rdx
  __int64 v117; // rdi
  __int64 v118; // rax
  __int64 v119; // rdx
  __m128i v120; // xmm0
  __int64 v121; // rdi
  __int64 v122; // rdx
  __int64 v123; // rax
  _WORD *v124; // rdx
  __int64 v125; // rdi
  __int64 v126; // rax
  __int64 v127; // rdx
  __m128i v128; // xmm0
  __int64 v129; // rdi
  __int64 v130; // rdx
  __int64 v131; // rax
  _WORD *v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rax
  __m128i *v135; // rdx
  __m128i v136; // xmm0
  __int64 v137; // rdi
  __int64 v138; // rdx
  __int64 v139; // rax
  __m128i *v140; // rdx
  __int64 v141; // rdi
  __m128i v142; // xmm0
  __int64 v143; // rax
  _WORD *v144; // rdx
  __int64 v145; // rdi
  __int64 v146; // rax
  _WORD *v147; // rdx
  __int64 v148; // rdi
  __int64 v149; // rax
  _WORD *v150; // rdx
  __int64 v151; // rdi
  __int64 v152; // rax
  _WORD *v153; // rdx
  __int64 v154; // rdi
  __int64 v155; // rax
  _WORD *v156; // rdx
  __int64 v157; // rdi
  __int64 v158; // rax
  _WORD *v159; // rdx
  __int64 v160; // rdi
  __int64 v161; // rax
  _WORD *v162; // rdx
  __int64 v163; // rdi
  __int64 v164; // rax
  _WORD *v165; // rdx

  if ( *a1 )
  {
    v4 = a1[4] + a1[3] + a1[1] + a1[2];
    v5 = sub_16E8CB0(a1, a2, a3);
    v6 = *(__m128i **)(v5 + 24);
    v7 = v5;
    if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x2Bu )
    {
      a2 = "===== Alias Analysis Evaluator Report =====\n";
      sub_16E7EE0(v5, "===== Alias Analysis Evaluator Report =====\n", 44);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4288730);
      qmemcpy(&v6[2], "eport =====\n", 12);
      *v6 = si128;
      v6[1] = _mm_load_si128((const __m128i *)&xmmword_4288740);
      *(_QWORD *)(v5 + 24) += 44LL;
    }
    if ( v4 )
    {
      v13 = sub_16E8CB0(v7, a2, v6);
      v14 = *(_WORD **)(v13 + 24);
      v15 = v13;
      if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
      {
        v15 = sub_16E7EE0(v13, "  ", 2);
      }
      else
      {
        *v14 = 8224;
        *(_QWORD *)(v13 + 24) += 2LL;
      }
      v16 = (const char *)v4;
      v17 = sub_16E7AB0(v15, v4);
      v18 = *(__m128i **)(v17 + 24);
      v19 = v17;
      if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 0x1Eu )
      {
        v16 = " Total Alias Queries Performed\n";
        sub_16E7EE0(v17, " Total Alias Queries Performed\n", 31);
      }
      else
      {
        v20 = _mm_load_si128((const __m128i *)&xmmword_4288780);
        qmemcpy(&v18[1], "ries Performed\n", 15);
        *v18 = v20;
        *(_QWORD *)(v17 + 24) += 31LL;
      }
      v21 = sub_16E8CB0(v19, v16, v18);
      v22 = *(_WORD **)(v21 + 24);
      v23 = v21;
      if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 1u )
      {
        v23 = sub_16E7EE0(v21, "  ", 2);
      }
      else
      {
        *v22 = 8224;
        *(_QWORD *)(v21 + 24) += 2LL;
      }
      v24 = sub_16E7AB0(v23, a1[1]);
      v25 = *(_QWORD *)(v24 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v24 + 16) - v25) <= 0x13 )
      {
        sub_16E7EE0(v24, " no alias responses ", 20);
      }
      else
      {
        v26 = _mm_load_si128((const __m128i *)&xmmword_4288790);
        *(_DWORD *)(v25 + 16) = 544433523;
        *(__m128i *)v25 = v26;
        *(_QWORD *)(v24 + 24) += 20LL;
      }
      v27 = a1[1];
      sub_1351F50(v27, v4, v25);
      v29 = sub_16E8CB0(v27, v4, v28);
      v30 = *(_WORD **)(v29 + 24);
      v31 = v29;
      if ( *(_QWORD *)(v29 + 16) - (_QWORD)v30 <= 1u )
      {
        v31 = sub_16E7EE0(v29, "  ", 2);
      }
      else
      {
        *v30 = 8224;
        *(_QWORD *)(v29 + 24) += 2LL;
      }
      v32 = sub_16E7AB0(v31, a1[2]);
      v33 = *(_QWORD *)(v32 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v32 + 16) - v33) <= 0x14 )
      {
        sub_16E7EE0(v32, " may alias responses ", 21);
      }
      else
      {
        v34 = _mm_load_si128((const __m128i *)&xmmword_42887A0);
        *(_DWORD *)(v33 + 16) = 1936028526;
        *(_BYTE *)(v33 + 20) = 32;
        *(__m128i *)v33 = v34;
        *(_QWORD *)(v32 + 24) += 21LL;
      }
      v35 = a1[2];
      sub_1351F50(v35, v4, v33);
      v37 = sub_16E8CB0(v35, v4, v36);
      v38 = *(_WORD **)(v37 + 24);
      v39 = v37;
      if ( *(_QWORD *)(v37 + 16) - (_QWORD)v38 <= 1u )
      {
        v39 = sub_16E7EE0(v37, "  ", 2);
      }
      else
      {
        *v38 = 8224;
        *(_QWORD *)(v37 + 24) += 2LL;
      }
      v40 = sub_16E7AB0(v39, a1[3]);
      v41 = *(_QWORD *)(v40 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v40 + 16) - v41) <= 0x18 )
      {
        sub_16E7EE0(v40, " partial alias responses ", 25);
      }
      else
      {
        v42 = _mm_load_si128((const __m128i *)&xmmword_42887B0);
        *(_BYTE *)(v41 + 24) = 32;
        *(_QWORD *)(v41 + 16) = 0x7365736E6F707365LL;
        *(__m128i *)v41 = v42;
        *(_QWORD *)(v40 + 24) += 25LL;
      }
      v43 = a1[3];
      sub_1351F50(v43, v4, v41);
      v45 = sub_16E8CB0(v43, v4, v44);
      v46 = *(_WORD **)(v45 + 24);
      v47 = v45;
      if ( *(_QWORD *)(v45 + 16) - (_QWORD)v46 <= 1u )
      {
        v47 = sub_16E7EE0(v45, "  ", 2);
      }
      else
      {
        *v46 = 8224;
        *(_QWORD *)(v45 + 24) += 2LL;
      }
      v48 = sub_16E7AB0(v47, a1[4]);
      v49 = *(_QWORD *)(v48 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v48 + 16) - v49) <= 0x15 )
      {
        sub_16E7EE0(v48, " must alias responses ", 22);
      }
      else
      {
        v50 = _mm_load_si128((const __m128i *)&xmmword_42887C0);
        *(_DWORD *)(v49 + 16) = 1702063727;
        *(_WORD *)(v49 + 20) = 8307;
        *(__m128i *)v49 = v50;
        *(_QWORD *)(v48 + 24) += 22LL;
      }
      v51 = a1[4];
      sub_1351F50(v51, v4, v49);
      v53 = sub_16E8CB0(v51, v4, v52);
      v54 = *(__m128i **)(v53 + 24);
      v55 = v53;
      if ( *(_QWORD *)(v53 + 16) - (_QWORD)v54 <= 0x31u )
      {
        v55 = sub_16E7EE0(v53, "  Alias Analysis Evaluator Pointer Alias Summary: ", 50);
      }
      else
      {
        v56 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v54[3].m128i_i16[0] = 8250;
        *v54 = v56;
        v54[1] = _mm_load_si128((const __m128i *)&xmmword_42887D0);
        v54[2] = _mm_load_si128((const __m128i *)&xmmword_42887E0);
        *(_QWORD *)(v53 + 24) += 50LL;
      }
      v57 = sub_16E7AB0(v55, 100LL * a1[1] / v4);
      v58 = *(_WORD **)(v57 + 24);
      v59 = v57;
      if ( *(_QWORD *)(v57 + 16) - (_QWORD)v58 <= 1u )
      {
        v59 = sub_16E7EE0(v57, "%/", 2);
      }
      else
      {
        *v58 = 12069;
        *(_QWORD *)(v57 + 24) += 2LL;
      }
      v60 = sub_16E7AB0(v59, 100LL * a1[2] / v4);
      v61 = *(_WORD **)(v60 + 24);
      v62 = v60;
      if ( *(_QWORD *)(v60 + 16) - (_QWORD)v61 <= 1u )
      {
        v62 = sub_16E7EE0(v60, "%/", 2);
      }
      else
      {
        *v61 = 12069;
        *(_QWORD *)(v60 + 24) += 2LL;
      }
      v63 = sub_16E7AB0(v62, 100LL * a1[3] / v4);
      v64 = *(_WORD **)(v63 + 24);
      v65 = v63;
      if ( *(_QWORD *)(v63 + 16) - (_QWORD)v64 <= 1u )
      {
        v65 = sub_16E7EE0(v63, "%/", 2);
      }
      else
      {
        *v64 = 12069;
        *(_QWORD *)(v63 + 24) += 2LL;
      }
      a2 = (const char *)(100LL * a1[4] / v4);
      v66 = sub_16E7AB0(v65, a2);
      v10 = *(__m128i **)(v66 + 24);
      v11 = v66;
      if ( *(_QWORD *)(v66 + 16) - (_QWORD)v10 <= 1u )
      {
        a2 = "%\n";
        sub_16E7EE0(v66, "%\n", 2);
      }
      else
      {
        v10->m128i_i16[0] = 2597;
        *(_QWORD *)(v66 + 24) += 2LL;
      }
    }
    else
    {
      v9 = sub_16E8CB0(v7, a2, v6);
      v10 = *(__m128i **)(v9 + 24);
      v11 = v9;
      if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0x30u )
      {
        a2 = "  Alias Analysis Evaluator Summary: No pointers!\n";
        sub_16E7EE0(v9, "  Alias Analysis Evaluator Summary: No pointers!\n", 49);
      }
      else
      {
        v12 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v10[3].m128i_i8[0] = 10;
        *v10 = v12;
        v10[1] = _mm_load_si128((const __m128i *)&xmmword_4288760);
        v10[2] = _mm_load_si128((const __m128i *)&xmmword_4288770);
        *(_QWORD *)(v9 + 24) += 49LL;
      }
    }
    v67 = a1[12] + a1[11] + a1[10] + a1[9] + a1[8] + a1[6] + a1[5] + a1[7];
    if ( v67 )
    {
      v71 = sub_16E8CB0(v11, a2, v10);
      v72 = *(_WORD **)(v71 + 24);
      v73 = v71;
      if ( *(_QWORD *)(v71 + 16) - (_QWORD)v72 <= 1u )
      {
        v73 = sub_16E7EE0(v71, "  ", 2);
      }
      else
      {
        *v72 = 8224;
        *(_QWORD *)(v71 + 24) += 2LL;
      }
      v74 = (const char *)v67;
      v75 = sub_16E7AB0(v73, v67);
      v76 = *(__m128i **)(v75 + 24);
      v77 = v75;
      if ( *(_QWORD *)(v75 + 16) - (_QWORD)v76 <= 0x1Fu )
      {
        v74 = " Total ModRef Queries Performed\n";
        sub_16E7EE0(v75, " Total ModRef Queries Performed\n", 32);
      }
      else
      {
        *v76 = _mm_load_si128((const __m128i *)&xmmword_4288810);
        v76[1] = _mm_load_si128((const __m128i *)&xmmword_4288820);
        *(_QWORD *)(v75 + 24) += 32LL;
      }
      v78 = sub_16E8CB0(v77, v74, v76);
      v79 = *(_WORD **)(v78 + 24);
      v80 = v78;
      if ( *(_QWORD *)(v78 + 16) - (_QWORD)v79 <= 1u )
      {
        v80 = sub_16E7EE0(v78, "  ", 2);
      }
      else
      {
        *v79 = 8224;
        *(_QWORD *)(v78 + 24) += 2LL;
      }
      v81 = sub_16E7AB0(v80, a1[5]);
      v82 = *(_QWORD *)(v81 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v81 + 16) - v82) <= 0x15 )
      {
        sub_16E7EE0(v81, " no mod/ref responses ", 22);
      }
      else
      {
        v83 = _mm_load_si128((const __m128i *)&xmmword_4288830);
        *(_DWORD *)(v82 + 16) = 1702063727;
        *(_WORD *)(v82 + 20) = 8307;
        *(__m128i *)v82 = v83;
        *(_QWORD *)(v81 + 24) += 22LL;
      }
      v84 = a1[5];
      sub_1351F50(v84, v67, v82);
      v86 = sub_16E8CB0(v84, v67, v85);
      v87 = *(_WORD **)(v86 + 24);
      v88 = v86;
      if ( *(_QWORD *)(v86 + 16) - (_QWORD)v87 <= 1u )
      {
        v88 = sub_16E7EE0(v86, "  ", 2);
      }
      else
      {
        *v87 = 8224;
        *(_QWORD *)(v86 + 24) += 2LL;
      }
      v89 = sub_16E7AB0(v88, a1[6]);
      v90 = *(void **)(v89 + 24);
      if ( *(_QWORD *)(v89 + 16) - (_QWORD)v90 <= 0xEu )
      {
        sub_16E7EE0(v89, " mod responses ", 15);
      }
      else
      {
        qmemcpy(v90, " mod responses ", 15);
        *(_QWORD *)(v89 + 24) += 15LL;
      }
      v91 = a1[6];
      sub_1351F50(v91, v67, (__int64)v90);
      v93 = sub_16E8CB0(v91, v67, v92);
      v94 = *(_WORD **)(v93 + 24);
      v95 = v93;
      if ( *(_QWORD *)(v93 + 16) - (_QWORD)v94 <= 1u )
      {
        v95 = sub_16E7EE0(v93, "  ", 2);
      }
      else
      {
        *v94 = 8224;
        *(_QWORD *)(v93 + 24) += 2LL;
      }
      v96 = sub_16E7AB0(v95, a1[7]);
      v97 = *(void **)(v96 + 24);
      if ( *(_QWORD *)(v96 + 16) - (_QWORD)v97 <= 0xEu )
      {
        sub_16E7EE0(v96, " ref responses ", 15);
      }
      else
      {
        qmemcpy(v97, " ref responses ", 15);
        *(_QWORD *)(v96 + 24) += 15LL;
      }
      v98 = a1[7];
      sub_1351F50(v98, v67, (__int64)v97);
      v100 = sub_16E8CB0(v98, v67, v99);
      v101 = *(_WORD **)(v100 + 24);
      v102 = v100;
      if ( *(_QWORD *)(v100 + 16) - (_QWORD)v101 <= 1u )
      {
        v102 = sub_16E7EE0(v100, "  ", 2);
      }
      else
      {
        *v101 = 8224;
        *(_QWORD *)(v100 + 24) += 2LL;
      }
      v103 = sub_16E7AB0(v102, a1[8]);
      v104 = *(_QWORD *)(v103 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v103 + 16) - v104) <= 0x14 )
      {
        sub_16E7EE0(v103, " mod & ref responses ", 21);
      }
      else
      {
        v105 = _mm_load_si128((const __m128i *)&xmmword_4288840);
        *(_DWORD *)(v104 + 16) = 1936028526;
        *(_BYTE *)(v104 + 20) = 32;
        *(__m128i *)v104 = v105;
        *(_QWORD *)(v103 + 24) += 21LL;
      }
      v106 = a1[8];
      sub_1351F50(v106, v67, v104);
      v108 = sub_16E8CB0(v106, v67, v107);
      v109 = *(_WORD **)(v108 + 24);
      v110 = v108;
      if ( *(_QWORD *)(v108 + 16) - (_QWORD)v109 <= 1u )
      {
        v110 = sub_16E7EE0(v108, "  ", 2);
      }
      else
      {
        *v109 = 8224;
        *(_QWORD *)(v108 + 24) += 2LL;
      }
      v111 = sub_16E7AB0(v110, a1[9]);
      v112 = *(__m128i **)(v111 + 24);
      if ( *(_QWORD *)(v111 + 16) - (_QWORD)v112 <= 0xFu )
      {
        sub_16E7EE0(v111, " must responses ", 16);
      }
      else
      {
        *v112 = _mm_load_si128((const __m128i *)&xmmword_4288850);
        *(_QWORD *)(v111 + 24) += 16LL;
      }
      v113 = a1[9];
      sub_1351F50(v113, v67, (__int64)v112);
      v115 = sub_16E8CB0(v113, v67, v114);
      v116 = *(_WORD **)(v115 + 24);
      v117 = v115;
      if ( *(_QWORD *)(v115 + 16) - (_QWORD)v116 <= 1u )
      {
        v117 = sub_16E7EE0(v115, "  ", 2);
      }
      else
      {
        *v116 = 8224;
        *(_QWORD *)(v115 + 24) += 2LL;
      }
      v118 = sub_16E7AB0(v117, a1[11]);
      v119 = *(_QWORD *)(v118 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v118 + 16) - v119) <= 0x13 )
      {
        sub_16E7EE0(v118, " must mod responses ", 20);
      }
      else
      {
        v120 = _mm_load_si128((const __m128i *)&xmmword_4288860);
        *(_DWORD *)(v119 + 16) = 544433523;
        *(__m128i *)v119 = v120;
        *(_QWORD *)(v118 + 24) += 20LL;
      }
      v121 = a1[11];
      sub_1351F50(v121, v67, v119);
      v123 = sub_16E8CB0(v121, v67, v122);
      v124 = *(_WORD **)(v123 + 24);
      v125 = v123;
      if ( *(_QWORD *)(v123 + 16) - (_QWORD)v124 <= 1u )
      {
        v125 = sub_16E7EE0(v123, "  ", 2);
      }
      else
      {
        *v124 = 8224;
        *(_QWORD *)(v123 + 24) += 2LL;
      }
      v126 = sub_16E7AB0(v125, a1[10]);
      v127 = *(_QWORD *)(v126 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v126 + 16) - v127) <= 0x13 )
      {
        sub_16E7EE0(v126, " must ref responses ", 20);
      }
      else
      {
        v128 = _mm_load_si128((const __m128i *)&xmmword_4288870);
        *(_DWORD *)(v127 + 16) = 544433523;
        *(__m128i *)v127 = v128;
        *(_QWORD *)(v126 + 24) += 20LL;
      }
      v129 = a1[10];
      sub_1351F50(v129, v67, v127);
      v131 = sub_16E8CB0(v129, v67, v130);
      v132 = *(_WORD **)(v131 + 24);
      v133 = v131;
      if ( *(_QWORD *)(v131 + 16) - (_QWORD)v132 <= 1u )
      {
        v133 = sub_16E7EE0(v131, "  ", 2);
      }
      else
      {
        *v132 = 8224;
        *(_QWORD *)(v131 + 24) += 2LL;
      }
      v134 = sub_16E7AB0(v133, a1[12]);
      v135 = *(__m128i **)(v134 + 24);
      if ( *(_QWORD *)(v134 + 16) - (_QWORD)v135 <= 0x19u )
      {
        sub_16E7EE0(v134, " must mod & ref responses ", 26);
      }
      else
      {
        v136 = _mm_load_si128((const __m128i *)&xmmword_4288880);
        qmemcpy(&v135[1], "responses ", 10);
        *v135 = v136;
        *(_QWORD *)(v134 + 24) += 26LL;
      }
      v137 = a1[12];
      sub_1351F50(v137, v67, (__int64)v135);
      v139 = sub_16E8CB0(v137, v67, v138);
      v140 = *(__m128i **)(v139 + 24);
      v141 = v139;
      if ( *(_QWORD *)(v139 + 16) - (_QWORD)v140 <= 0x2Bu )
      {
        v141 = sub_16E7EE0(v139, "  Alias Analysis Evaluator Mod/Ref Summary: ", 44);
      }
      else
      {
        v142 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        qmemcpy(&v140[2], "ef Summary: ", 12);
        *v140 = v142;
        v140[1] = _mm_load_si128((const __m128i *)&xmmword_4288890);
        *(_QWORD *)(v139 + 24) += 44LL;
      }
      v143 = sub_16E7AB0(v141, 100LL * a1[5] / v67);
      v144 = *(_WORD **)(v143 + 24);
      v145 = v143;
      if ( *(_QWORD *)(v143 + 16) - (_QWORD)v144 <= 1u )
      {
        v145 = sub_16E7EE0(v143, "%/", 2);
      }
      else
      {
        *v144 = 12069;
        *(_QWORD *)(v143 + 24) += 2LL;
      }
      v146 = sub_16E7AB0(v145, 100LL * a1[6] / v67);
      v147 = *(_WORD **)(v146 + 24);
      v148 = v146;
      if ( *(_QWORD *)(v146 + 16) - (_QWORD)v147 <= 1u )
      {
        v148 = sub_16E7EE0(v146, "%/", 2);
      }
      else
      {
        *v147 = 12069;
        *(_QWORD *)(v146 + 24) += 2LL;
      }
      v149 = sub_16E7AB0(v148, 100LL * a1[7] / v67);
      v150 = *(_WORD **)(v149 + 24);
      v151 = v149;
      if ( *(_QWORD *)(v149 + 16) - (_QWORD)v150 <= 1u )
      {
        v151 = sub_16E7EE0(v149, "%/", 2);
      }
      else
      {
        *v150 = 12069;
        *(_QWORD *)(v149 + 24) += 2LL;
      }
      v152 = sub_16E7AB0(v151, 100LL * a1[8] / v67);
      v153 = *(_WORD **)(v152 + 24);
      v154 = v152;
      if ( *(_QWORD *)(v152 + 16) - (_QWORD)v153 <= 1u )
      {
        v154 = sub_16E7EE0(v152, "%/", 2);
      }
      else
      {
        *v153 = 12069;
        *(_QWORD *)(v152 + 24) += 2LL;
      }
      v155 = sub_16E7AB0(v154, 100LL * a1[9] / v67);
      v156 = *(_WORD **)(v155 + 24);
      v157 = v155;
      if ( *(_QWORD *)(v155 + 16) - (_QWORD)v156 <= 1u )
      {
        v157 = sub_16E7EE0(v155, "%/", 2);
      }
      else
      {
        *v156 = 12069;
        *(_QWORD *)(v155 + 24) += 2LL;
      }
      v158 = sub_16E7AB0(v157, 100LL * a1[10] / v67);
      v159 = *(_WORD **)(v158 + 24);
      v160 = v158;
      if ( *(_QWORD *)(v158 + 16) - (_QWORD)v159 <= 1u )
      {
        v160 = sub_16E7EE0(v158, "%/", 2);
      }
      else
      {
        *v159 = 12069;
        *(_QWORD *)(v158 + 24) += 2LL;
      }
      v161 = sub_16E7AB0(v160, 100LL * a1[11] / v67);
      v162 = *(_WORD **)(v161 + 24);
      v163 = v161;
      if ( *(_QWORD *)(v161 + 16) - (_QWORD)v162 <= 1u )
      {
        v163 = sub_16E7EE0(v161, "%/", 2);
      }
      else
      {
        *v162 = 12069;
        *(_QWORD *)(v161 + 24) += 2LL;
      }
      v164 = sub_16E7AB0(v163, 100LL * a1[12] / v67);
      v165 = *(_WORD **)(v164 + 24);
      if ( *(_QWORD *)(v164 + 16) - (_QWORD)v165 <= 1u )
      {
        sub_16E7EE0(v164, "%\n", 2);
      }
      else
      {
        *v165 = 2597;
        *(_QWORD *)(v164 + 24) += 2LL;
      }
    }
    else
    {
      v68 = sub_16E8CB0(v11, a2, v10);
      v69 = *(__m128i **)(v68 + 24);
      if ( *(_QWORD *)(v68 + 16) - (_QWORD)v69 <= 0x37u )
      {
        sub_16E7EE0(v68, "  Alias Analysis Mod/Ref Evaluator Summary: no mod/ref!\n", 56);
      }
      else
      {
        v70 = _mm_load_si128((const __m128i *)&xmmword_4288750);
        v69[3].m128i_i64[0] = 0xA216665722F646FLL;
        *v69 = v70;
        v69[1] = _mm_load_si128((const __m128i *)&xmmword_42887F0);
        v69[2] = _mm_load_si128((const __m128i *)&xmmword_4288800);
        *(_QWORD *)(v68 + 24) += 56LL;
      }
    }
  }
}
