// Function: sub_30C4110
// Address: 0x30c4110
//
_BYTE *__fastcall sub_30C4110(signed __int64 *a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rdi
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __m128i *v8; // rdx
  __m128i v9; // xmm0
  __int64 v10; // rdi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rdi
  __int64 v20; // rdi
  _BYTE *v21; // rax
  void *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __m128i *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rdi
  _BYTE *v29; // rax
  void *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __m128i *v34; // rdx
  __m128i v35; // xmm0
  __int64 v36; // rdi
  __int64 v37; // rdi
  _BYTE *v38; // rax
  __m128i *v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rdi
  __int64 v42; // rdi
  _BYTE *v43; // rax
  _BYTE *result; // rax
  __m128i *v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __m128i *v49; // rdx
  __m128i v50; // xmm0
  __int64 v51; // rdi
  __int64 v52; // rdi
  _BYTE *v53; // rax
  __m128i *v54; // rdx
  __m128i v55; // xmm0
  __int64 v56; // rdi
  __int64 v57; // rdi
  _BYTE *v58; // rax
  __m128i *v59; // rdx
  __m128i v60; // xmm0
  __int64 v61; // rdi
  __int64 v62; // rdi
  _BYTE *v63; // rax
  __m128i *v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rdi
  _BYTE *v67; // rax
  __m128i *v68; // rdx
  __m128i v69; // xmm0
  __int64 v70; // rdi
  __int64 v71; // rdi
  _BYTE *v72; // rax
  __m128i *v73; // rdx
  __int64 v74; // rdi
  __int64 v75; // rdi
  _BYTE *v76; // rax
  __m128i *v77; // rdx
  __m128i v78; // xmm0
  __int64 v79; // rdi
  __int64 v80; // rdi
  _BYTE *v81; // rax
  __m128i *v82; // rdx
  __m128i v83; // xmm0
  __int64 v84; // rdi
  __int64 v85; // rdi
  _BYTE *v86; // rax
  __m128i *v87; // rdx
  __m128i v88; // xmm0
  __int64 v89; // rdi
  __int64 v90; // rdi
  _BYTE *v91; // rax
  __m128i *v92; // rdx
  __m128i v93; // xmm0
  __int64 v94; // rdi
  __int64 v95; // rdi
  _BYTE *v96; // rax
  __m128i *v97; // rdx
  __m128i v98; // xmm0
  __int64 v99; // rdi
  __int64 v100; // rdi
  _BYTE *v101; // rax
  __m128i *v102; // rdx
  __m128i v103; // xmm0
  __int64 v104; // rdi
  __int64 v105; // rdi
  _BYTE *v106; // rax
  __m128i *v107; // rdx
  __m128i v108; // xmm0
  __int64 v109; // rdi
  __int64 v110; // rdi
  _BYTE *v111; // rax
  __m128i *v112; // rdx
  __m128i v113; // xmm0
  __int64 v114; // rdi
  __int64 v115; // rdi
  _BYTE *v116; // rax
  __m128i *v117; // rdx
  __m128i v118; // xmm0
  __int64 v119; // rdi
  __int64 v120; // rdi
  _BYTE *v121; // rax
  __m128i *v122; // rdx
  __m128i v123; // xmm0
  __int64 v124; // rdi
  __int64 v125; // rdi
  _BYTE *v126; // rax
  __m128i *v127; // rdx
  __m128i v128; // xmm0
  __int64 v129; // rdi
  __int64 v130; // rdi
  _BYTE *v131; // rax
  __m128i *v132; // rdx
  __m128i v133; // xmm0
  __int64 v134; // rdi
  __int64 v135; // rdi
  _BYTE *v136; // rax
  __m128i *v137; // rdx
  __m128i v138; // xmm0
  __int64 v139; // rdi
  __int64 v140; // rdi
  _BYTE *v141; // rax
  __m128i *v142; // rdx
  __m128i v143; // xmm0
  __int64 v144; // rdi
  __int64 v145; // rdi
  _BYTE *v146; // rax
  __m128i *v147; // rdx
  __m128i v148; // xmm0
  __int64 v149; // rdi
  __int64 v150; // rdi
  _BYTE *v151; // rax
  __m128i *v152; // rdx
  __m128i v153; // xmm0
  __int64 v154; // rdi
  __int64 v155; // rdi
  _BYTE *v156; // rax
  __m128i *v157; // rdx
  __m128i v158; // xmm0
  __int64 v159; // rdi
  __int64 v160; // rdi
  _BYTE *v161; // rax
  __m128i *v162; // rdx
  __int64 v163; // rdi
  __int64 v164; // rdi
  _BYTE *v165; // rax
  __m128i *v166; // rdx
  __m128i v167; // xmm0
  __int64 v168; // rdi
  __int64 v169; // rdi
  _BYTE *v170; // rax
  __m128i *v171; // rdx
  __m128i v172; // xmm0
  __int64 v173; // rdi
  __int64 v174; // rdi
  _BYTE *v175; // rax
  __m128i *v176; // rdx
  __m128i v177; // xmm0
  __int64 v178; // rdi
  __int64 v179; // rdi
  _BYTE *v180; // rax
  __m128i *v181; // rdx
  __m128i v182; // xmm0
  __int64 v183; // rdi
  __int64 v184; // rdi
  _BYTE *v185; // rax
  __m128i *v186; // rdx
  __m128i v187; // xmm0
  __int64 v188; // rdi
  __int64 v189; // rdi
  _BYTE *v190; // rax
  __m128i *v191; // rdx
  __m128i v192; // xmm0
  __int64 v193; // rdi
  __int64 v194; // rdi
  _BYTE *v195; // rax
  __m128i *v196; // rdx
  __m128i v197; // xmm0
  __int64 v198; // rdi
  __int64 v199; // rdi
  _BYTE *v200; // rax
  __m128i *v201; // rdx
  __m128i v202; // xmm0
  __int64 v203; // rdi
  __int64 v204; // rdi
  _BYTE *v205; // rax
  __m128i *v206; // rdx
  __m128i v207; // xmm0
  __int64 v208; // rdi
  __int64 v209; // rdi
  _BYTE *v210; // rax
  __m128i *v211; // rdx
  __m128i v212; // xmm0
  __int64 v213; // rdi
  __int64 v214; // rdi
  _BYTE *v215; // rax

  v2 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0x10u )
  {
    v5 = sub_CB6200(a2, (unsigned __int8 *)"BasicBlockCount: ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CC390);
    v2[1].m128i_i8[0] = 32;
    v5 = a2;
    *v2 = si128;
    *(_QWORD *)(a2 + 32) += 17LL;
  }
  v6 = sub_CB59F0(v5, *a1);
  v7 = *(_BYTE **)(v6 + 32);
  if ( *(_BYTE **)(v6 + 24) == v7 )
  {
    sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v7 = 10;
    ++*(_QWORD *)(v6 + 32);
  }
  v8 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 0x28u )
  {
    v10 = sub_CB6200(a2, "BlocksReachedFromConditionalInstruction: ", 0x29u);
  }
  else
  {
    v9 = _mm_load_si128((const __m128i *)&xmmword_44CC3A0);
    v8[2].m128i_i8[8] = 32;
    v10 = a2;
    v8[2].m128i_i64[0] = 0x3A6E6F6974637572LL;
    *v8 = v9;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_44CC3B0);
    *(_QWORD *)(a2 + 32) += 41LL;
  }
  v11 = sub_CB59F0(v10, a1[1]);
  v12 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == v12 )
  {
    sub_CB6200(v11, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v12 = 10;
    ++*(_QWORD *)(v11 + 32);
  }
  v13 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v13) <= 5 )
  {
    v14 = sub_CB6200(a2, "Uses: ", 6u);
  }
  else
  {
    *(_DWORD *)v13 = 1936028501;
    v14 = a2;
    *(_WORD *)(v13 + 4) = 8250;
    *(_QWORD *)(a2 + 32) += 6LL;
  }
  v15 = sub_CB59F0(v14, a1[2]);
  v16 = *(_BYTE **)(v15 + 32);
  if ( *(_BYTE **)(v15 + 24) == v16 )
  {
    sub_CB6200(v15, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v16 = 10;
    ++*(_QWORD *)(v15 + 32);
  }
  v17 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v17 <= 0x1Eu )
  {
    v19 = sub_CB6200(a2, "DirectCallsToDefinedFunctions: ", 0x1Fu);
  }
  else
  {
    v18 = _mm_load_si128((const __m128i *)&xmmword_44CC3C0);
    v19 = a2;
    qmemcpy(&v17[1], "inedFunctions: ", 15);
    *v17 = v18;
    *(_QWORD *)(a2 + 32) += 31LL;
  }
  v20 = sub_CB59F0(v19, a1[3]);
  v21 = *(_BYTE **)(v20 + 32);
  if ( *(_BYTE **)(v20 + 24) == v21 )
  {
    sub_CB6200(v20, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v21 = 10;
    ++*(_QWORD *)(v20 + 32);
  }
  v22 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v22 <= 0xEu )
  {
    v23 = sub_CB6200(a2, "LoadInstCount: ", 0xFu);
  }
  else
  {
    v23 = a2;
    qmemcpy(v22, "LoadInstCount: ", 15);
    *(_QWORD *)(a2 + 32) += 15LL;
  }
  v24 = sub_CB59F0(v23, a1[4]);
  v25 = *(_BYTE **)(v24 + 32);
  if ( *(_BYTE **)(v24 + 24) == v25 )
  {
    sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v25 = 10;
    ++*(_QWORD *)(v24 + 32);
  }
  v26 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 <= 0xFu )
  {
    v27 = sub_CB6200(a2, "StoreInstCount: ", 0x10u);
  }
  else
  {
    v27 = a2;
    *v26 = _mm_load_si128((const __m128i *)&xmmword_44CC3D0);
    *(_QWORD *)(a2 + 32) += 16LL;
  }
  v28 = sub_CB59F0(v27, a1[5]);
  v29 = *(_BYTE **)(v28 + 32);
  if ( *(_BYTE **)(v28 + 24) == v29 )
  {
    sub_CB6200(v28, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v29 = 10;
    ++*(_QWORD *)(v28 + 32);
  }
  v30 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v30 <= 0xDu )
  {
    v31 = sub_CB6200(a2, "MaxLoopDepth: ", 0xEu);
  }
  else
  {
    v31 = a2;
    qmemcpy(v30, "MaxLoopDepth: ", 14);
    *(_QWORD *)(a2 + 32) += 14LL;
  }
  v32 = sub_CB59F0(v31, a1[6]);
  v33 = *(_BYTE **)(v32 + 32);
  if ( *(_BYTE **)(v32 + 24) == v33 )
  {
    sub_CB6200(v32, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v33 = 10;
    ++*(_QWORD *)(v32 + 32);
  }
  v34 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v34 <= 0x12u )
  {
    v36 = sub_CB6200(a2, "TopLevelLoopCount: ", 0x13u);
  }
  else
  {
    v35 = _mm_load_si128((const __m128i *)&xmmword_44CC3E0);
    v34[1].m128i_i8[2] = 32;
    v36 = a2;
    v34[1].m128i_i16[0] = 14964;
    *v34 = v35;
    *(_QWORD *)(a2 + 32) += 19LL;
  }
  v37 = sub_CB59F0(v36, a1[7]);
  v38 = *(_BYTE **)(v37 + 32);
  if ( *(_BYTE **)(v37 + 24) == v38 )
  {
    sub_CB6200(v37, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v38 = 10;
    ++*(_QWORD *)(v37 + 32);
  }
  v39 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v39 <= 0x16u )
  {
    v41 = sub_CB6200(a2, "TotalInstructionCount: ", 0x17u);
  }
  else
  {
    v40 = _mm_load_si128((const __m128i *)&xmmword_44CC3F0);
    v39[1].m128i_i8[6] = 32;
    v41 = a2;
    v39[1].m128i_i32[0] = 1853189955;
    v39[1].m128i_i16[2] = 14964;
    *v39 = v40;
    *(_QWORD *)(a2 + 32) += 23LL;
  }
  v42 = sub_CB59F0(v41, a1[8]);
  v43 = *(_BYTE **)(v42 + 32);
  if ( *(_BYTE **)(v42 + 24) == v43 )
  {
    sub_CB6200(v42, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v43 = 10;
    ++*(_QWORD *)(v42 + 32);
  }
  if ( LOBYTE(qword_502F088[8]) )
  {
    v45 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v45 <= 0x1Fu )
    {
      v46 = sub_CB6200(a2, "BasicBlocksWithSingleSuccessor: ", 0x20u);
    }
    else
    {
      v46 = a2;
      *v45 = _mm_load_si128((const __m128i *)&xmmword_44CC400);
      v45[1] = _mm_load_si128((const __m128i *)&xmmword_44CC410);
      *(_QWORD *)(a2 + 32) += 32LL;
    }
    v47 = sub_CB59F0(v46, a1[9]);
    v48 = *(_BYTE **)(v47 + 32);
    if ( *(_BYTE **)(v47 + 24) == v48 )
    {
      sub_CB6200(v47, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v48 = 10;
      ++*(_QWORD *)(v47 + 32);
    }
    v49 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v49 <= 0x1Du )
    {
      v51 = sub_CB6200(a2, "BasicBlocksWithTwoSuccessors: ", 0x1Eu);
    }
    else
    {
      v50 = _mm_load_si128((const __m128i *)&xmmword_44CC420);
      v51 = a2;
      qmemcpy(&v49[1], "woSuccessors: ", 14);
      *v49 = v50;
      *(_QWORD *)(a2 + 32) += 30LL;
    }
    v52 = sub_CB59F0(v51, a1[10]);
    v53 = *(_BYTE **)(v52 + 32);
    if ( *(_BYTE **)(v52 + 24) == v53 )
    {
      sub_CB6200(v52, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v53 = 10;
      ++*(_QWORD *)(v52 + 32);
    }
    v54 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v54 <= 0x25u )
    {
      v56 = sub_CB6200(a2, "BasicBlocksWithMoreThanTwoSuccessors: ", 0x26u);
    }
    else
    {
      v55 = _mm_load_si128((const __m128i *)&xmmword_44CC430);
      v54[2].m128i_i32[0] = 1936879475;
      v56 = a2;
      v54[2].m128i_i16[2] = 8250;
      *v54 = v55;
      v54[1] = _mm_load_si128((const __m128i *)&xmmword_44CC440);
      *(_QWORD *)(a2 + 32) += 38LL;
    }
    v57 = sub_CB59F0(v56, a1[11]);
    v58 = *(_BYTE **)(v57 + 32);
    if ( *(_BYTE **)(v57 + 24) == v58 )
    {
      sub_CB6200(v57, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v58 = 10;
      ++*(_QWORD *)(v57 + 32);
    }
    v59 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v59 <= 0x21u )
    {
      v61 = sub_CB6200(a2, "BasicBlocksWithSinglePredecessor: ", 0x22u);
    }
    else
    {
      v60 = _mm_load_si128((const __m128i *)&xmmword_44CC400);
      v59[2].m128i_i16[0] = 8250;
      v61 = a2;
      *v59 = v60;
      v59[1] = _mm_load_si128((const __m128i *)&xmmword_44CC450);
      *(_QWORD *)(a2 + 32) += 34LL;
    }
    v62 = sub_CB59F0(v61, a1[12]);
    v63 = *(_BYTE **)(v62 + 32);
    if ( *(_BYTE **)(v62 + 24) == v63 )
    {
      sub_CB6200(v62, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v63 = 10;
      ++*(_QWORD *)(v62 + 32);
    }
    v64 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v64 <= 0x1Fu )
    {
      v65 = sub_CB6200(a2, "BasicBlocksWithTwoPredecessors: ", 0x20u);
    }
    else
    {
      v65 = a2;
      *v64 = _mm_load_si128((const __m128i *)&xmmword_44CC420);
      v64[1] = _mm_load_si128((const __m128i *)&xmmword_44CC460);
      *(_QWORD *)(a2 + 32) += 32LL;
    }
    v66 = sub_CB59F0(v65, a1[13]);
    v67 = *(_BYTE **)(v66 + 32);
    if ( *(_BYTE **)(v66 + 24) == v67 )
    {
      sub_CB6200(v66, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v67 = 10;
      ++*(_QWORD *)(v66 + 32);
    }
    v68 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v68 <= 0x27u )
    {
      v70 = sub_CB6200(a2, "BasicBlocksWithMoreThanTwoPredecessors: ", 0x28u);
    }
    else
    {
      v69 = _mm_load_si128((const __m128i *)&xmmword_44CC430);
      v70 = a2;
      v68[2].m128i_i64[0] = 0x203A73726F737365LL;
      *v68 = v69;
      v68[1] = _mm_load_si128((const __m128i *)&xmmword_44CC470);
      *(_QWORD *)(a2 + 32) += 40LL;
    }
    v71 = sub_CB59F0(v70, a1[14]);
    v72 = *(_BYTE **)(v71 + 32);
    if ( *(_BYTE **)(v71 + 24) == v72 )
    {
      sub_CB6200(v71, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v72 = 10;
      ++*(_QWORD *)(v71 + 32);
    }
    v73 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v73 <= 0xFu )
    {
      v74 = sub_CB6200(a2, "BigBasicBlocks: ", 0x10u);
    }
    else
    {
      v74 = a2;
      *v73 = _mm_load_si128((const __m128i *)&xmmword_44CC480);
      *(_QWORD *)(a2 + 32) += 16LL;
    }
    v75 = sub_CB59F0(v74, a1[15]);
    v76 = *(_BYTE **)(v75 + 32);
    if ( *(_BYTE **)(v75 + 24) == v76 )
    {
      sub_CB6200(v75, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v76 = 10;
      ++*(_QWORD *)(v75 + 32);
    }
    v77 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v77 <= 0x12u )
    {
      v79 = sub_CB6200(a2, "MediumBasicBlocks: ", 0x13u);
    }
    else
    {
      v78 = _mm_load_si128((const __m128i *)&xmmword_44CC490);
      v77[1].m128i_i8[2] = 32;
      v79 = a2;
      v77[1].m128i_i16[0] = 14963;
      *v77 = v78;
      *(_QWORD *)(a2 + 32) += 19LL;
    }
    v80 = sub_CB59F0(v79, a1[16]);
    v81 = *(_BYTE **)(v80 + 32);
    if ( *(_BYTE **)(v80 + 24) == v81 )
    {
      sub_CB6200(v80, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v81 = 10;
      ++*(_QWORD *)(v80 + 32);
    }
    v82 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v82 <= 0x11u )
    {
      v84 = sub_CB6200(a2, "SmallBasicBlocks: ", 0x12u);
    }
    else
    {
      v83 = _mm_load_si128((const __m128i *)&xmmword_44CC4A0);
      v84 = a2;
      v82[1].m128i_i16[0] = 8250;
      *v82 = v83;
      *(_QWORD *)(a2 + 32) += 18LL;
    }
    v85 = sub_CB59F0(v84, a1[17]);
    v86 = *(_BYTE **)(v85 + 32);
    if ( *(_BYTE **)(v85 + 24) == v86 )
    {
      sub_CB6200(v85, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v86 = 10;
      ++*(_QWORD *)(v85 + 32);
    }
    v87 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v87 <= 0x15u )
    {
      v89 = sub_CB6200(a2, "CastInstructionCount: ", 0x16u);
    }
    else
    {
      v88 = _mm_load_si128((const __m128i *)&xmmword_44CC4B0);
      v87[1].m128i_i32[0] = 1953396079;
      v89 = a2;
      v87[1].m128i_i16[2] = 8250;
      *v87 = v88;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v90 = sub_CB59F0(v89, a1[18]);
    v91 = *(_BYTE **)(v90 + 32);
    if ( *(_BYTE **)(v90 + 24) == v91 )
    {
      sub_CB6200(v90, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v91 = 10;
      ++*(_QWORD *)(v90 + 32);
    }
    v92 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v92 <= 0x1Eu )
    {
      v94 = sub_CB6200(a2, "FloatingPointInstructionCount: ", 0x1Fu);
    }
    else
    {
      v93 = _mm_load_si128((const __m128i *)&xmmword_44CC4C0);
      v94 = a2;
      qmemcpy(&v92[1], "tructionCount: ", 15);
      *v92 = v93;
      *(_QWORD *)(a2 + 32) += 31LL;
    }
    v95 = sub_CB59F0(v94, a1[19]);
    v96 = *(_BYTE **)(v95 + 32);
    if ( *(_BYTE **)(v95 + 24) == v96 )
    {
      sub_CB6200(v95, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v96 = 10;
      ++*(_QWORD *)(v95 + 32);
    }
    v97 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v97 <= 0x18u )
    {
      v99 = sub_CB6200(a2, "IntegerInstructionCount: ", 0x19u);
    }
    else
    {
      v98 = _mm_load_si128((const __m128i *)&xmmword_44CC4D0);
      v97[1].m128i_i8[8] = 32;
      v99 = a2;
      v97[1].m128i_i64[0] = 0x3A746E756F436E6FLL;
      *v97 = v98;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v100 = sub_CB59F0(v99, a1[20]);
    v101 = *(_BYTE **)(v100 + 32);
    if ( *(_BYTE **)(v100 + 24) == v101 )
    {
      sub_CB6200(v100, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v101 = 10;
      ++*(_QWORD *)(v100 + 32);
    }
    v102 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v102 <= 0x18u )
    {
      v104 = sub_CB6200(a2, "ConstantIntOperandCount: ", 0x19u);
    }
    else
    {
      v103 = _mm_load_si128((const __m128i *)&xmmword_44CC4E0);
      v102[1].m128i_i8[8] = 32;
      v104 = a2;
      v102[1].m128i_i64[0] = 0x3A746E756F43646ELL;
      *v102 = v103;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v105 = sub_CB59F0(v104, a1[21]);
    v106 = *(_BYTE **)(v105 + 32);
    if ( *(_BYTE **)(v105 + 24) == v106 )
    {
      sub_CB6200(v105, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v106 = 10;
      ++*(_QWORD *)(v105 + 32);
    }
    v107 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v107 <= 0x17u )
    {
      v109 = sub_CB6200(a2, "ConstantFPOperandCount: ", 0x18u);
    }
    else
    {
      v108 = _mm_load_si128((const __m128i *)&xmmword_44CC4F0);
      v109 = a2;
      v107[1].m128i_i64[0] = 0x203A746E756F4364LL;
      *v107 = v108;
      *(_QWORD *)(a2 + 32) += 24LL;
    }
    v110 = sub_CB59F0(v109, a1[22]);
    v111 = *(_BYTE **)(v110 + 32);
    if ( *(_BYTE **)(v110 + 24) == v111 )
    {
      sub_CB6200(v110, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v111 = 10;
      ++*(_QWORD *)(v110 + 32);
    }
    v112 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v112 <= 0x15u )
    {
      v114 = sub_CB6200(a2, "ConstantOperandCount: ", 0x16u);
    }
    else
    {
      v113 = _mm_load_si128((const __m128i *)&xmmword_44CC500);
      v112[1].m128i_i32[0] = 1953396079;
      v114 = a2;
      v112[1].m128i_i16[2] = 8250;
      *v112 = v113;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v115 = sub_CB59F0(v114, a1[23]);
    v116 = *(_BYTE **)(v115 + 32);
    if ( *(_BYTE **)(v115 + 24) == v116 )
    {
      sub_CB6200(v115, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v116 = 10;
      ++*(_QWORD *)(v115 + 32);
    }
    v117 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v117 <= 0x18u )
    {
      v119 = sub_CB6200(a2, "InstructionOperandCount: ", 0x19u);
    }
    else
    {
      v118 = _mm_load_si128((const __m128i *)&xmmword_44CC510);
      v117[1].m128i_i8[8] = 32;
      v119 = a2;
      v117[1].m128i_i64[0] = 0x3A746E756F43646ELL;
      *v117 = v118;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v120 = sub_CB59F0(v119, a1[24]);
    v121 = *(_BYTE **)(v120 + 32);
    if ( *(_BYTE **)(v120 + 24) == v121 )
    {
      sub_CB6200(v120, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v121 = 10;
      ++*(_QWORD *)(v120 + 32);
    }
    v122 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v122 <= 0x17u )
    {
      v124 = sub_CB6200(a2, "BasicBlockOperandCount: ", 0x18u);
    }
    else
    {
      v123 = _mm_load_si128((const __m128i *)&xmmword_44CC520);
      v124 = a2;
      v122[1].m128i_i64[0] = 0x203A746E756F4364LL;
      *v122 = v123;
      *(_QWORD *)(a2 + 32) += 24LL;
    }
    v125 = sub_CB59F0(v124, a1[25]);
    v126 = *(_BYTE **)(v125 + 32);
    if ( *(_BYTE **)(v125 + 24) == v126 )
    {
      sub_CB6200(v125, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v126 = 10;
      ++*(_QWORD *)(v125 + 32);
    }
    v127 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v127 <= 0x18u )
    {
      v129 = sub_CB6200(a2, "GlobalValueOperandCount: ", 0x19u);
    }
    else
    {
      v128 = _mm_load_si128((const __m128i *)&xmmword_44CC530);
      v127[1].m128i_i8[8] = 32;
      v129 = a2;
      v127[1].m128i_i64[0] = 0x3A746E756F43646ELL;
      *v127 = v128;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v130 = sub_CB59F0(v129, a1[26]);
    v131 = *(_BYTE **)(v130 + 32);
    if ( *(_BYTE **)(v130 + 24) == v131 )
    {
      sub_CB6200(v130, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v131 = 10;
      ++*(_QWORD *)(v130 + 32);
    }
    v132 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v132 <= 0x16u )
    {
      v134 = sub_CB6200(a2, "InlineAsmOperandCount: ", 0x17u);
    }
    else
    {
      v133 = _mm_load_si128((const __m128i *)&xmmword_44CC540);
      v132[1].m128i_i8[6] = 32;
      v134 = a2;
      v132[1].m128i_i32[0] = 1853189955;
      v132[1].m128i_i16[2] = 14964;
      *v132 = v133;
      *(_QWORD *)(a2 + 32) += 23LL;
    }
    v135 = sub_CB59F0(v134, a1[27]);
    v136 = *(_BYTE **)(v135 + 32);
    if ( *(_BYTE **)(v135 + 24) == v136 )
    {
      sub_CB6200(v135, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v136 = 10;
      ++*(_QWORD *)(v135 + 32);
    }
    v137 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v137 <= 0x15u )
    {
      v139 = sub_CB6200(a2, "ArgumentOperandCount: ", 0x16u);
    }
    else
    {
      v138 = _mm_load_si128((const __m128i *)&xmmword_44CC550);
      v137[1].m128i_i32[0] = 1953396079;
      v139 = a2;
      v137[1].m128i_i16[2] = 8250;
      *v137 = v138;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v140 = sub_CB59F0(v139, a1[28]);
    v141 = *(_BYTE **)(v140 + 32);
    if ( *(_BYTE **)(v140 + 24) == v141 )
    {
      sub_CB6200(v140, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v141 = 10;
      ++*(_QWORD *)(v140 + 32);
    }
    v142 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v142 <= 0x14u )
    {
      v144 = sub_CB6200(a2, "UnknownOperandCount: ", 0x15u);
    }
    else
    {
      v143 = _mm_load_si128((const __m128i *)&xmmword_44CC560);
      v142[1].m128i_i32[0] = 980708981;
      v144 = a2;
      v142[1].m128i_i8[4] = 32;
      *v142 = v143;
      *(_QWORD *)(a2 + 32) += 21LL;
    }
    v145 = sub_CB59F0(v144, a1[29]);
    v146 = *(_BYTE **)(v145 + 32);
    if ( *(_BYTE **)(v145 + 24) == v146 )
    {
      sub_CB6200(v145, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v146 = 10;
      ++*(_QWORD *)(v145 + 32);
    }
    v147 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v147 <= 0x12u )
    {
      v149 = sub_CB6200(a2, "CriticalEdgeCount: ", 0x13u);
    }
    else
    {
      v148 = _mm_load_si128((const __m128i *)&xmmword_44CC570);
      v147[1].m128i_i8[2] = 32;
      v149 = a2;
      v147[1].m128i_i16[0] = 14964;
      *v147 = v148;
      *(_QWORD *)(a2 + 32) += 19LL;
    }
    v150 = sub_CB59F0(v149, a1[30]);
    v151 = *(_BYTE **)(v150 + 32);
    if ( *(_BYTE **)(v150 + 24) == v151 )
    {
      sub_CB6200(v150, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v151 = 10;
      ++*(_QWORD *)(v150 + 32);
    }
    v152 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v152 <= 0x15u )
    {
      v154 = sub_CB6200(a2, "ControlFlowEdgeCount: ", 0x16u);
    }
    else
    {
      v153 = _mm_load_si128((const __m128i *)&xmmword_44CC580);
      v152[1].m128i_i32[0] = 1953396079;
      v154 = a2;
      v152[1].m128i_i16[2] = 8250;
      *v152 = v153;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v155 = sub_CB59F0(v154, a1[31]);
    v156 = *(_BYTE **)(v155 + 32);
    if ( *(_BYTE **)(v155 + 24) == v156 )
    {
      sub_CB6200(v155, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v156 = 10;
      ++*(_QWORD *)(v155 + 32);
    }
    v157 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v157 <= 0x19u )
    {
      v159 = sub_CB6200(a2, "UnconditionalBranchCount: ", 0x1Au);
    }
    else
    {
      v158 = _mm_load_si128((const __m128i *)&xmmword_44CC590);
      v159 = a2;
      qmemcpy(&v157[1], "nchCount: ", 10);
      *v157 = v158;
      *(_QWORD *)(a2 + 32) += 26LL;
    }
    v160 = sub_CB59F0(v159, a1[32]);
    v161 = *(_BYTE **)(v160 + 32);
    if ( *(_BYTE **)(v160 + 24) == v161 )
    {
      sub_CB6200(v160, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v161 = 10;
      ++*(_QWORD *)(v160 + 32);
    }
    v162 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v162 <= 0xFu )
    {
      v163 = sub_CB6200(a2, "IntrinsicCount: ", 0x10u);
    }
    else
    {
      v163 = a2;
      *v162 = _mm_load_si128((const __m128i *)&xmmword_44CC5A0);
      *(_QWORD *)(a2 + 32) += 16LL;
    }
    v164 = sub_CB59F0(v163, a1[33]);
    v165 = *(_BYTE **)(v164 + 32);
    if ( *(_BYTE **)(v164 + 24) == v165 )
    {
      sub_CB6200(v164, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v165 = 10;
      ++*(_QWORD *)(v164 + 32);
    }
    v166 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v166 <= 0x10u )
    {
      v168 = sub_CB6200(a2, "DirectCallCount: ", 0x11u);
    }
    else
    {
      v167 = _mm_load_si128((const __m128i *)&xmmword_44CC5B0);
      v166[1].m128i_i8[0] = 32;
      v168 = a2;
      *v166 = v167;
      *(_QWORD *)(a2 + 32) += 17LL;
    }
    v169 = sub_CB59F0(v168, a1[34]);
    v170 = *(_BYTE **)(v169 + 32);
    if ( *(_BYTE **)(v169 + 24) == v170 )
    {
      sub_CB6200(v169, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v170 = 10;
      ++*(_QWORD *)(v169 + 32);
    }
    v171 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v171 <= 0x12u )
    {
      v173 = sub_CB6200(a2, "IndirectCallCount: ", 0x13u);
    }
    else
    {
      v172 = _mm_load_si128((const __m128i *)&xmmword_44CC5C0);
      v171[1].m128i_i8[2] = 32;
      v173 = a2;
      v171[1].m128i_i16[0] = 14964;
      *v171 = v172;
      *(_QWORD *)(a2 + 32) += 19LL;
    }
    v174 = sub_CB59F0(v173, a1[35]);
    v175 = *(_BYTE **)(v174 + 32);
    if ( *(_BYTE **)(v174 + 24) == v175 )
    {
      sub_CB6200(v174, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v175 = 10;
      ++*(_QWORD *)(v174 + 32);
    }
    v176 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v176 <= 0x18u )
    {
      v178 = sub_CB6200(a2, "CallReturnsIntegerCount: ", 0x19u);
    }
    else
    {
      v177 = _mm_load_si128((const __m128i *)&xmmword_44CC5D0);
      v176[1].m128i_i8[8] = 32;
      v178 = a2;
      v176[1].m128i_i64[0] = 0x3A746E756F437265LL;
      *v176 = v177;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v179 = sub_CB59F0(v178, a1[36]);
    v180 = *(_BYTE **)(v179 + 32);
    if ( *(_BYTE **)(v179 + 24) == v180 )
    {
      sub_CB6200(v179, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v180 = 10;
      ++*(_QWORD *)(v179 + 32);
    }
    v181 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v181 <= 0x16u )
    {
      v183 = sub_CB6200(a2, "CallReturnsFloatCount: ", 0x17u);
    }
    else
    {
      v182 = _mm_load_si128((const __m128i *)&xmmword_44CC5E0);
      v181[1].m128i_i32[0] = 1853189955;
      v181[1].m128i_i16[2] = 14964;
      v183 = a2;
      v181[1].m128i_i8[6] = 32;
      *v181 = v182;
      *(_QWORD *)(a2 + 32) += 23LL;
    }
    v184 = sub_CB59F0(v183, a1[37]);
    v185 = *(_BYTE **)(v184 + 32);
    if ( *(_BYTE **)(v184 + 24) == v185 )
    {
      sub_CB6200(v184, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v185 = 10;
      ++*(_QWORD *)(v184 + 32);
    }
    v186 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v186 <= 0x18u )
    {
      v188 = sub_CB6200(a2, "CallReturnsPointerCount: ", 0x19u);
    }
    else
    {
      v187 = _mm_load_si128((const __m128i *)&xmmword_44CC5F0);
      v186[1].m128i_i8[8] = 32;
      v188 = a2;
      v186[1].m128i_i64[0] = 0x3A746E756F437265LL;
      *v186 = v187;
      *(_QWORD *)(a2 + 32) += 25LL;
    }
    v189 = sub_CB59F0(v188, a1[38]);
    v190 = *(_BYTE **)(v189 + 32);
    if ( *(_BYTE **)(v189 + 24) == v190 )
    {
      sub_CB6200(v189, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v190 = 10;
      ++*(_QWORD *)(v189 + 32);
    }
    v191 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v191 <= 0x1Au )
    {
      v193 = sub_CB6200(a2, "CallReturnsVectorIntCount: ", 0x1Bu);
    }
    else
    {
      v192 = _mm_load_si128((const __m128i *)&xmmword_44CC600);
      v193 = a2;
      qmemcpy(&v191[1], "rIntCount: ", 11);
      *v191 = v192;
      *(_QWORD *)(a2 + 32) += 27LL;
    }
    v194 = sub_CB59F0(v193, a1[39]);
    v195 = *(_BYTE **)(v194 + 32);
    if ( *(_BYTE **)(v194 + 24) == v195 )
    {
      sub_CB6200(v194, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v195 = 10;
      ++*(_QWORD *)(v194 + 32);
    }
    v196 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v196 <= 0x1Cu )
    {
      v198 = sub_CB6200(a2, "CallReturnsVectorFloatCount: ", 0x1Du);
    }
    else
    {
      v197 = _mm_load_si128((const __m128i *)&xmmword_44CC600);
      v198 = a2;
      qmemcpy(&v196[1], "rFloatCount: ", 13);
      *v196 = v197;
      *(_QWORD *)(a2 + 32) += 29LL;
    }
    v199 = sub_CB59F0(v198, a1[40]);
    v200 = *(_BYTE **)(v199 + 32);
    if ( *(_BYTE **)(v199 + 24) == v200 )
    {
      sub_CB6200(v199, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v200 = 10;
      ++*(_QWORD *)(v199 + 32);
    }
    v201 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v201 <= 0x1Eu )
    {
      v203 = sub_CB6200(a2, "CallReturnsVectorPointerCount: ", 0x1Fu);
    }
    else
    {
      v202 = _mm_load_si128((const __m128i *)&xmmword_44CC600);
      v203 = a2;
      qmemcpy(&v201[1], "rPointerCount: ", 15);
      *v201 = v202;
      *(_QWORD *)(a2 + 32) += 31LL;
    }
    v204 = sub_CB59F0(v203, a1[41]);
    v205 = *(_BYTE **)(v204 + 32);
    if ( *(_BYTE **)(v204 + 24) == v205 )
    {
      sub_CB6200(v204, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v205 = 10;
      ++*(_QWORD *)(v204 + 32);
    }
    v206 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v206 <= 0x1Bu )
    {
      v208 = sub_CB6200(a2, "CallWithManyArgumentsCount: ", 0x1Cu);
    }
    else
    {
      v207 = _mm_load_si128((const __m128i *)&xmmword_44CC610);
      v208 = a2;
      qmemcpy(&v206[1], "mentsCount: ", 12);
      *v206 = v207;
      *(_QWORD *)(a2 + 32) += 28LL;
    }
    v209 = sub_CB59F0(v208, a1[42]);
    v210 = *(_BYTE **)(v209 + 32);
    if ( *(_BYTE **)(v209 + 24) == v210 )
    {
      sub_CB6200(v209, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v210 = 10;
      ++*(_QWORD *)(v209 + 32);
    }
    v211 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v211 <= 0x1Du )
    {
      v213 = sub_CB6200(a2, "CallWithPointerArgumentCount: ", 0x1Eu);
    }
    else
    {
      v212 = _mm_load_si128((const __m128i *)&xmmword_44CC620);
      v213 = a2;
      qmemcpy(&v211[1], "rgumentCount: ", 14);
      *v211 = v212;
      *(_QWORD *)(a2 + 32) += 30LL;
    }
    v214 = sub_CB59F0(v213, a1[43]);
    v215 = *(_BYTE **)(v214 + 32);
    if ( *(_BYTE **)(v214 + 24) == v215 )
    {
      sub_CB6200(v214, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v215 = 10;
      ++*(_QWORD *)(v214 + 32);
    }
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
