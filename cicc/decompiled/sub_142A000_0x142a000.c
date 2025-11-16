// Function: sub_142A000
// Address: 0x142a000
//
__int64 __fastcall sub_142A000(__int64 a1, __int64 a2)
{
  char **v3; // rbx
  char **v4; // r12
  void *v5; // rdi
  char v6; // r9
  char *v7; // rax
  char *v8; // rsi
  __int64 v9; // rdx
  const char *v10; // r14
  __int64 v11; // rax
  size_t v12; // rdx
  size_t v13; // r8
  const char *v14; // rsi
  __int64 v15; // rax
  size_t v16; // rdx
  _BYTE *v17; // rax
  void *v18; // rdx
  char *v19; // r15
  const char *v20; // rax
  size_t v21; // rdx
  __m128i *v22; // rdx
  __m128i si128; // xmm0
  __int64 v24; // rdi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdi
  const char *v32; // rcx
  size_t v33; // rdx
  size_t v34; // r8
  __int64 v35; // rax
  const char *v36; // rsi
  __int64 v37; // rax
  size_t v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  _BYTE *v41; // rax
  void *v42; // rdx
  __int64 v43; // r14
  __int64 v44; // r15
  __int64 v45; // rdi
  const char *v46; // rax
  size_t v47; // rdx
  void *v48; // rdi
  unsigned int v49; // r15d
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // rdi
  __int64 v53; // rax
  size_t v54; // rdx
  _WORD *v55; // rdi
  const char *v56; // rsi
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r12
  __m128i v60; // xmm0
  __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // rdi
  const char *v64; // rax
  size_t v65; // rdx
  void *v66; // rdi
  unsigned int v67; // r14d
  __int64 v68; // rax
  __int64 v69; // rdi
  const char *v70; // rcx
  size_t v71; // rdx
  size_t v72; // r8
  __int64 v73; // rax
  const char *v74; // rsi
  __int64 v75; // rax
  size_t v76; // rdx
  __int64 v77; // rdi
  __int64 v78; // rdx
  _BYTE *v79; // rax
  __int64 v80; // rbx
  __m128i *v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // r14
  __int64 v84; // rdi
  __int64 v85; // rax
  size_t v86; // rdx
  _WORD *v87; // rdi
  const char *v88; // rsi
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 *v91; // rbx
  __int64 result; // rax
  __int64 v93; // rdx
  __int64 v94; // r15
  __int64 v95; // rdi
  __int64 v96; // rdx
  unsigned int v97; // r14d
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rdi
  const char *v101; // rcx
  size_t v102; // rdx
  size_t v103; // r8
  __int64 v104; // rax
  const char *v105; // rsi
  __int64 v106; // rax
  size_t v107; // rdx
  _BYTE *v108; // rax
  const char *v109; // rax
  size_t v110; // rdx
  __m128i *v111; // rdx
  __m128i v112; // xmm0
  __int64 v113; // rdi
  __int64 v114; // rdi
  _BYTE *v115; // rax
  _BYTE *v116; // rdx
  __int64 v117; // r14
  __int64 v118; // r15
  __int64 v119; // rax
  size_t v120; // rdx
  _WORD *v121; // rdi
  const char *v122; // rsi
  unsigned __int64 v123; // rax
  void *v124; // rdi
  void *v125; // rdx
  __int64 v126; // rdi
  __int64 v127; // rdi
  _BYTE *v128; // rax
  _BYTE *v129; // rax
  __int64 v130; // r14
  __int64 v131; // rdi
  const char *v132; // rax
  size_t v133; // rdx
  void *v134; // rdi
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v138; // [rsp+8h] [rbp-48h]
  size_t v139; // [rsp+8h] [rbp-48h]
  size_t v140; // [rsp+10h] [rbp-40h]
  size_t v141; // [rsp+10h] [rbp-40h]
  const char *v142; // [rsp+10h] [rbp-40h]
  size_t v143; // [rsp+10h] [rbp-40h]
  size_t v144; // [rsp+10h] [rbp-40h]
  size_t v145; // [rsp+10h] [rbp-40h]
  size_t v146; // [rsp+18h] [rbp-38h]
  size_t v147; // [rsp+18h] [rbp-38h]
  const char *v148; // [rsp+18h] [rbp-38h]
  size_t v149; // [rsp+18h] [rbp-38h]
  size_t v150; // [rsp+18h] [rbp-38h]
  size_t v151; // [rsp+18h] [rbp-38h]
  const char *v152; // [rsp+18h] [rbp-38h]
  size_t v153; // [rsp+18h] [rbp-38h]
  __int64 *v154; // [rsp+18h] [rbp-38h]

  v3 = *(char ***)(a1 + 160);
  v4 = &v3[*(unsigned int *)(a1 + 168)];
  if ( v4 != v3 )
  {
    while ( 1 )
    {
      v18 = *(void **)(a2 + 24);
      v19 = *v3;
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v18 <= 0xDu )
      {
        sub_16E7EE0(a2, "Compile unit: ", 14);
      }
      else
      {
        qmemcpy(v18, "Compile unit: ", 14);
        *(_QWORD *)(a2 + 24) += 14LL;
      }
      v20 = (const char *)sub_14E77F0(*((unsigned int *)v19 + 6));
      if ( v21 )
        break;
      v22 = *(__m128i **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v22 <= 0x10u )
      {
        v24 = sub_16E7EE0(a2, "unknown-language(", 17);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_428C2D0);
        v22[1].m128i_i8[0] = 40;
        v24 = a2;
        *v22 = si128;
        *(_QWORD *)(a2 + 24) += 17LL;
      }
      v25 = sub_16E7A90(v24, *((unsigned int *)v19 + 6));
      v26 = *(_BYTE **)(v25 + 24);
      if ( *(_BYTE **)(v25 + 16) == v26 )
      {
        sub_16E7EE0(v25, ")", 1);
        goto LABEL_5;
      }
      *v26 = 41;
      ++*(_QWORD *)(v25 + 24);
      v6 = *v19;
      if ( *v19 == 15 )
      {
LABEL_24:
        v8 = v19;
        v7 = v19;
        goto LABEL_8;
      }
LABEL_6:
      v7 = *(char **)&v19[-8 * *((unsigned int *)v19 + 2)];
      if ( !v7 )
      {
        v10 = byte_3F871B3;
        v13 = 0;
        v16 = 0;
        v14 = byte_3F871B3;
        goto LABEL_15;
      }
      v8 = *(char **)&v19[-8 * *((unsigned int *)v19 + 2)];
LABEL_8:
      v9 = *((unsigned int *)v7 + 2);
      v10 = *(const char **)&v8[8 * (1 - v9)];
      if ( v10 )
      {
        v11 = sub_161E970(*(_QWORD *)&v8[8 * (1 - v9)]);
        v6 = *v19;
        v10 = (const char *)v11;
        v13 = v12;
      }
      else
      {
        v13 = 0;
      }
      v14 = *(const char **)&v19[-8 * *((unsigned int *)v19 + 2)];
      if ( v6 == 15 )
        goto LABEL_13;
      if ( v14 )
      {
        v14 = *(const char **)&v14[-8 * *((unsigned int *)v14 + 2)];
LABEL_13:
        if ( v14 )
        {
          v147 = v13;
          v15 = sub_161E970(v14);
          v13 = v147;
          v14 = (const char *)v15;
        }
        else
        {
          v16 = 0;
        }
        goto LABEL_15;
      }
      v16 = 0;
      v14 = byte_3F871B3;
LABEL_15:
      sub_1429E60(a2, v14, v16, v10, v13, 0);
      v17 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v17 >= *(_QWORD *)(a2 + 16) )
      {
        ++v3;
        sub_16E7DE0(a2, 10);
        if ( v4 == v3 )
          goto LABEL_27;
      }
      else
      {
        ++v3;
        *(_QWORD *)(a2 + 24) = v17 + 1;
        *v17 = 10;
        if ( v4 == v3 )
          goto LABEL_27;
      }
    }
    v5 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 < v21 )
    {
      sub_16E7EE0(a2, v20);
    }
    else
    {
      v146 = v21;
      memcpy(v5, v20, v21);
      *(_QWORD *)(a2 + 24) += v146;
    }
LABEL_5:
    v6 = *v19;
    if ( *v19 == 15 )
      goto LABEL_24;
    goto LABEL_6;
  }
LABEL_27:
  v27 = *(__int64 **)(a1 + 240);
  v28 = &v27[*(unsigned int *)(a1 + 248)];
  if ( v27 != v28 )
  {
    while ( 1 )
    {
      v42 = *(void **)(a2 + 24);
      v43 = *v27;
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v42 <= 0xBu )
      {
        v44 = sub_16E7EE0(a2, "Subprogram: ", 12);
      }
      else
      {
        v44 = a2;
        qmemcpy(v42, "Subprogram: ", 12);
        *(_QWORD *)(a2 + 24) += 12LL;
      }
      v45 = *(_QWORD *)(v43 + 8 * (2LL - *(unsigned int *)(v43 + 8)));
      if ( v45 )
      {
        v46 = (const char *)sub_161E970(v45);
        v48 = *(void **)(v44 + 24);
        if ( *(_QWORD *)(v44 + 16) - (_QWORD)v48 < v47 )
        {
          sub_16E7EE0(v44, v46);
        }
        else if ( v47 )
        {
          v149 = v47;
          memcpy(v48, v46, v47);
          *(_QWORD *)(v44 + 24) += v149;
        }
      }
      v49 = *(_DWORD *)(v43 + 24);
      if ( *(_BYTE *)v43 == 15 )
      {
        if ( !v43 )
        {
          v32 = byte_3F871B3;
          v34 = 0;
          v38 = 0;
          v36 = byte_3F871B3;
          goto LABEL_37;
        }
        v31 = *(_QWORD *)(v43 + 8 * (1LL - *(unsigned int *)(v43 + 8)));
        if ( !v31 )
        {
          v35 = v43;
          v32 = 0;
          v34 = 0;
LABEL_35:
          v36 = *(const char **)(v35 - 8LL * *(unsigned int *)(v35 + 8));
          if ( v36 )
          {
            v140 = v34;
            v148 = v32;
            v37 = sub_161E970(*(_QWORD *)(v35 - 8LL * *(unsigned int *)(v35 + 8)));
            v32 = v148;
            v34 = v140;
            v36 = (const char *)v37;
          }
          else
          {
            v38 = 0;
          }
          goto LABEL_37;
        }
      }
      else
      {
        v29 = *(unsigned int *)(v43 + 8);
        v30 = *(_QWORD *)(v43 - 8 * v29);
        if ( !v30 )
        {
          v32 = byte_3F871B3;
          v34 = 0;
          goto LABEL_33;
        }
        v31 = *(_QWORD *)(v30 + 8 * (1LL - *(unsigned int *)(v30 + 8)));
        if ( !v31 )
        {
          v32 = 0;
          v34 = 0;
          goto LABEL_33;
        }
      }
      v32 = (const char *)sub_161E970(v31);
      v34 = v33;
      v35 = v43;
      if ( *(_BYTE *)v43 != 15 )
      {
        v29 = *(unsigned int *)(v43 + 8);
LABEL_33:
        v35 = *(_QWORD *)(v43 - 8 * v29);
      }
      if ( v35 )
        goto LABEL_35;
      v38 = 0;
      v36 = byte_3F871B3;
LABEL_37:
      sub_1429E60(a2, v36, v38, v32, v34, v49);
      v39 = *(_QWORD *)(v43 + 8 * (3LL - *(unsigned int *)(v43 + 8)));
      if ( v39 && (sub_161E970(v39), v40) )
      {
        v50 = *(_QWORD *)(a2 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v50) <= 2 )
        {
          v51 = sub_16E7EE0(a2, " ('", 3);
        }
        else
        {
          *(_BYTE *)(v50 + 2) = 39;
          v51 = a2;
          *(_WORD *)v50 = 10272;
          *(_QWORD *)(a2 + 24) += 3LL;
        }
        v52 = *(_QWORD *)(v43 + 8 * (3LL - *(unsigned int *)(v43 + 8)));
        if ( !v52 )
          goto LABEL_86;
        v53 = sub_161E970(v52);
        v55 = *(_WORD **)(v51 + 24);
        v56 = (const char *)v53;
        v57 = *(_QWORD *)(v51 + 16) - (_QWORD)v55;
        if ( v54 > v57 )
        {
          v51 = sub_16E7EE0(v51, v56);
LABEL_86:
          v55 = *(_WORD **)(v51 + 24);
          if ( *(_QWORD *)(v51 + 16) - (_QWORD)v55 <= 1u )
            goto LABEL_87;
          goto LABEL_64;
        }
        if ( v54 )
        {
          v150 = v54;
          memcpy(v55, v56, v54);
          v58 = *(_QWORD *)(v51 + 16);
          v55 = (_WORD *)(v150 + *(_QWORD *)(v51 + 24));
          *(_QWORD *)(v51 + 24) = v55;
          v57 = v58 - (_QWORD)v55;
        }
        if ( v57 <= 1 )
        {
LABEL_87:
          sub_16E7EE0(v51, "')", 2);
          goto LABEL_39;
        }
LABEL_64:
        *v55 = 10535;
        *(_QWORD *)(v51 + 24) += 2LL;
        v41 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v41 < *(_QWORD *)(a2 + 16) )
          goto LABEL_40;
LABEL_65:
        ++v27;
        sub_16E7DE0(a2, 10);
        if ( v28 == v27 )
          break;
      }
      else
      {
LABEL_39:
        v41 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v41 >= *(_QWORD *)(a2 + 16) )
          goto LABEL_65;
LABEL_40:
        ++v27;
        *(_QWORD *)(a2 + 24) = v41 + 1;
        *v41 = 10;
        if ( v28 == v27 )
          break;
      }
    }
  }
  v59 = *(_QWORD *)(a1 + 320);
  v138 = v59 + 8LL * *(unsigned int *)(a1 + 328);
  if ( v59 != v138 )
  {
    while ( 1 )
    {
      v80 = *(_QWORD *)(*(_QWORD *)v59 - 8LL * *(unsigned int *)(*(_QWORD *)v59 + 8LL));
      v81 = *(__m128i **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v81 > 0x10u )
      {
        v60 = _mm_load_si128((const __m128i *)&xmmword_428C2E0);
        v81[1].m128i_i8[0] = 32;
        v61 = a2;
        *v81 = v60;
        *(_QWORD *)(a2 + 24) += 17LL;
      }
      else
      {
        v61 = sub_16E7EE0(a2, "Global variable: ", 17);
      }
      v62 = *(unsigned int *)(v80 + 8);
      v63 = *(_QWORD *)(v80 + 8 * (1 - v62));
      if ( v63 )
      {
        v64 = (const char *)sub_161E970(v63);
        v66 = *(void **)(v61 + 24);
        if ( *(_QWORD *)(v61 + 16) - (_QWORD)v66 < v65 )
        {
          sub_16E7EE0(v61, v64);
          v62 = *(unsigned int *)(v80 + 8);
        }
        else
        {
          if ( v65 )
          {
            v151 = v65;
            memcpy(v66, v64, v65);
            *(_QWORD *)(v61 + 24) += v151;
          }
          v62 = *(unsigned int *)(v80 + 8);
        }
      }
      v67 = *(_DWORD *)(v80 + 24);
      v68 = *(_QWORD *)(v80 + 8 * (2 - v62));
      if ( !v68 )
      {
        v70 = byte_3F871B3;
        v72 = 0;
        v76 = 0;
        v74 = byte_3F871B3;
        goto LABEL_79;
      }
      v69 = *(_QWORD *)(v68 + 8 * (1LL - *(unsigned int *)(v68 + 8)));
      if ( !v69 )
        break;
      v70 = (const char *)sub_161E970(v69);
      v72 = v71;
      v73 = *(_QWORD *)(v80 + 8 * (2LL - *(unsigned int *)(v80 + 8)));
      if ( v73 )
      {
        v74 = *(const char **)(v73 - 8LL * *(unsigned int *)(v73 + 8));
        if ( v74 )
          goto LABEL_78;
LABEL_176:
        v76 = 0;
        goto LABEL_79;
      }
      v76 = 0;
      v74 = byte_3F871B3;
LABEL_79:
      sub_1429E60(a2, v74, v76, v70, v72, v67);
      v77 = *(_QWORD *)(v80 + 8 * (5LL - *(unsigned int *)(v80 + 8)));
      if ( v77 && (sub_161E970(v77), v78) )
      {
        v82 = *(_QWORD *)(a2 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v82) <= 2 )
        {
          v83 = sub_16E7EE0(a2, " ('", 3);
        }
        else
        {
          *(_BYTE *)(v82 + 2) = 39;
          v83 = a2;
          *(_WORD *)v82 = 10272;
          *(_QWORD *)(a2 + 24) += 3LL;
        }
        v84 = *(_QWORD *)(v80 + 8 * (5LL - *(unsigned int *)(v80 + 8)));
        if ( !v84 )
          goto LABEL_136;
        v85 = sub_161E970(v84);
        v87 = *(_WORD **)(v83 + 24);
        v88 = (const char *)v85;
        v89 = *(_QWORD *)(v83 + 16) - (_QWORD)v87;
        if ( v86 > v89 )
        {
          v83 = sub_16E7EE0(v83, v88);
LABEL_136:
          v87 = *(_WORD **)(v83 + 24);
          if ( *(_QWORD *)(v83 + 16) - (_QWORD)v87 <= 1u )
            goto LABEL_137;
          goto LABEL_96;
        }
        if ( v86 )
        {
          v153 = v86;
          memcpy(v87, v88, v86);
          v90 = *(_QWORD *)(v83 + 16);
          v87 = (_WORD *)(v153 + *(_QWORD *)(v83 + 24));
          *(_QWORD *)(v83 + 24) = v87;
          v89 = v90 - (_QWORD)v87;
        }
        if ( v89 <= 1 )
        {
LABEL_137:
          sub_16E7EE0(v83, "')", 2);
          goto LABEL_81;
        }
LABEL_96:
        *v87 = 10535;
        *(_QWORD *)(v83 + 24) += 2LL;
        v79 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v79 < *(_QWORD *)(a2 + 16) )
          goto LABEL_82;
LABEL_97:
        v59 += 8;
        sub_16E7DE0(a2, 10);
        if ( v138 == v59 )
          goto LABEL_98;
      }
      else
      {
LABEL_81:
        v79 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v79 >= *(_QWORD *)(a2 + 16) )
          goto LABEL_97;
LABEL_82:
        v59 += 8;
        *(_QWORD *)(a2 + 24) = v79 + 1;
        *v79 = 10;
        if ( v138 == v59 )
          goto LABEL_98;
      }
    }
    v70 = 0;
    v72 = 0;
    v74 = *(const char **)(v68 - 8LL * *(unsigned int *)(v68 + 8));
    if ( v74 )
    {
LABEL_78:
      v141 = v72;
      v152 = v70;
      v75 = sub_161E970(v74);
      v70 = v152;
      v72 = v141;
      v74 = (const char *)v75;
      goto LABEL_79;
    }
    goto LABEL_176;
  }
LABEL_98:
  v91 = *(__int64 **)(a1 + 400);
  result = (__int64)&v91[*(unsigned int *)(a1 + 408)];
  v154 = (__int64 *)result;
  if ( v91 != (__int64 *)result )
  {
    while ( 1 )
    {
      v93 = *(_QWORD *)(a2 + 24);
      v94 = *v91;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v93) <= 4 )
      {
        sub_16E7EE0(a2, "Type:", 5);
      }
      else
      {
        *(_DWORD *)v93 = 1701869908;
        *(_BYTE *)(v93 + 4) = 58;
        *(_QWORD *)(a2 + 24) += 5LL;
      }
      v95 = *(_QWORD *)(v94 + 8 * (2LL - *(unsigned int *)(v94 + 8)));
      if ( v95 )
      {
        sub_161E970(v95);
        if ( v96 )
        {
          v129 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v129 >= *(_QWORD *)(a2 + 16) )
          {
            v130 = sub_16E7DE0(a2, 32);
          }
          else
          {
            v130 = a2;
            *(_QWORD *)(a2 + 24) = v129 + 1;
            *v129 = 32;
          }
          v131 = *(_QWORD *)(v94 + 8 * (2LL - *(unsigned int *)(v94 + 8)));
          if ( v131 )
          {
            v132 = (const char *)sub_161E970(v131);
            v134 = *(void **)(v130 + 24);
            if ( v133 > *(_QWORD *)(v130 + 16) - (_QWORD)v134 )
            {
              sub_16E7EE0(v130, v132);
            }
            else if ( v133 )
            {
              v144 = v133;
              memcpy(v134, v132, v133);
              *(_QWORD *)(v130 + 24) += v144;
            }
          }
        }
      }
      v97 = *(_DWORD *)(v94 + 24);
      if ( *(_BYTE *)v94 != 15 )
        break;
      if ( !v94 )
      {
        v101 = byte_3F871B3;
        v103 = 0;
        v107 = 0;
        v105 = byte_3F871B3;
        goto LABEL_112;
      }
      v100 = *(_QWORD *)(v94 + 8 * (1LL - *(unsigned int *)(v94 + 8)));
      if ( v100 )
        goto LABEL_106;
      v104 = v94;
      v101 = 0;
      v103 = 0;
LABEL_110:
      v105 = *(const char **)(v104 - 8LL * *(unsigned int *)(v104 + 8));
      if ( v105 )
      {
        v139 = v103;
        v142 = v101;
        v106 = sub_161E970(*(_QWORD *)(v104 - 8LL * *(unsigned int *)(v104 + 8)));
        v101 = v142;
        v103 = v139;
        v105 = (const char *)v106;
      }
      else
      {
        v107 = 0;
      }
LABEL_112:
      sub_1429E60(a2, v105, v107, v101, v103, v97);
      v108 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE *)v94 == 11 )
      {
        if ( *(_BYTE **)(a2 + 16) == v108 )
        {
          sub_16E7EE0(a2, " ", 1);
        }
        else
        {
          *v108 = 32;
          ++*(_QWORD *)(a2 + 24);
        }
        v109 = (const char *)sub_14E6F20(*(unsigned int *)(v94 + 52));
        if ( !v110 )
        {
          v111 = *(__m128i **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v111 <= 0x10u )
          {
            v113 = sub_16E7EE0(a2, "unknown-encoding(", 17);
          }
          else
          {
            v112 = _mm_load_si128((const __m128i *)&xmmword_428C2F0);
            v111[1].m128i_i8[0] = 40;
            v113 = a2;
            *v111 = v112;
            *(_QWORD *)(a2 + 24) += 17LL;
          }
          v114 = sub_16E7A90(v113, *(unsigned int *)(v94 + 52));
          v115 = *(_BYTE **)(v114 + 24);
          if ( (unsigned __int64)v115 >= *(_QWORD *)(v114 + 16) )
          {
            sub_16E7DE0(v114, 41);
          }
          else
          {
            *(_QWORD *)(v114 + 24) = v115 + 1;
            *v115 = 41;
          }
          v116 = *(_BYTE **)(a2 + 24);
          goto LABEL_121;
        }
      }
      else
      {
        if ( (unsigned __int64)v108 >= *(_QWORD *)(a2 + 16) )
        {
          sub_16E7DE0(a2, 32);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v108 + 1;
          *v108 = 32;
        }
        v109 = (const char *)sub_14E0540(*(unsigned __int16 *)(v94 + 2));
        if ( !v110 )
        {
          v125 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v125 <= 0xBu )
          {
            v126 = sub_16E7EE0(a2, "unknown-tag(", 12);
          }
          else
          {
            v126 = a2;
            qmemcpy(v125, "unknown-tag(", 12);
            *(_QWORD *)(a2 + 24) += 12LL;
          }
          v127 = sub_16E7A90(v126, *(unsigned __int16 *)(v94 + 2));
          v128 = *(_BYTE **)(v127 + 24);
          if ( *(_BYTE **)(v127 + 16) == v128 )
          {
            sub_16E7EE0(v127, ")", 1);
          }
          else
          {
            *v128 = 41;
            ++*(_QWORD *)(v127 + 24);
          }
          v116 = *(_BYTE **)(a2 + 24);
          goto LABEL_121;
        }
      }
      v124 = *(void **)(a2 + 24);
      if ( v110 > *(_QWORD *)(a2 + 16) - (_QWORD)v124 )
      {
        sub_16E7EE0(a2, v109);
        v116 = *(_BYTE **)(a2 + 24);
      }
      else
      {
        v143 = v110;
        memcpy(v124, v109, v110);
        v116 = (_BYTE *)(*(_QWORD *)(a2 + 24) + v143);
        *(_QWORD *)(a2 + 24) = v116;
      }
LABEL_121:
      if ( *(_BYTE *)v94 == 13 )
      {
        v117 = *(_QWORD *)(v94 + 8 * (7LL - *(unsigned int *)(v94 + 8)));
        if ( v117 )
        {
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v116 <= 0xEu )
          {
            v118 = sub_16E7EE0(a2, " (identifier: '", 15);
          }
          else
          {
            v118 = a2;
            qmemcpy(v116, " (identifier: '", 15);
            *(_QWORD *)(a2 + 24) += 15LL;
          }
          v119 = sub_161E970(v117);
          v121 = *(_WORD **)(v118 + 24);
          v122 = (const char *)v119;
          v123 = *(_QWORD *)(v118 + 16) - (_QWORD)v121;
          if ( v120 > v123 )
          {
            v135 = sub_16E7EE0(v118, v122);
            v121 = *(_WORD **)(v135 + 24);
            v118 = v135;
            v123 = *(_QWORD *)(v135 + 16) - (_QWORD)v121;
          }
          else if ( v120 )
          {
            v145 = v120;
            memcpy(v121, v122, v120);
            v136 = *(_QWORD *)(v118 + 16);
            v121 = (_WORD *)(v145 + *(_QWORD *)(v118 + 24));
            *(_QWORD *)(v118 + 24) = v121;
            v123 = v136 - (_QWORD)v121;
          }
          if ( v123 <= 1 )
          {
            sub_16E7EE0(v118, "')", 2);
          }
          else
          {
            *v121 = 10535;
            *(_QWORD *)(v118 + 24) += 2LL;
          }
          v116 = *(_BYTE **)(a2 + 24);
        }
      }
      if ( (unsigned __int64)v116 >= *(_QWORD *)(a2 + 16) )
      {
        result = sub_16E7DE0(a2, 10);
      }
      else
      {
        result = (__int64)(v116 + 1);
        *(_QWORD *)(a2 + 24) = v116 + 1;
        *v116 = 10;
      }
      if ( v154 == ++v91 )
        return result;
    }
    v98 = *(unsigned int *)(v94 + 8);
    v99 = *(_QWORD *)(v94 - 8 * v98);
    if ( v99 )
    {
      v100 = *(_QWORD *)(v99 + 8 * (1LL - *(unsigned int *)(v99 + 8)));
      if ( v100 )
      {
LABEL_106:
        v101 = (const char *)sub_161E970(v100);
        v103 = v102;
        v104 = v94;
        if ( *(_BYTE *)v94 == 15 )
          goto LABEL_109;
        v98 = *(unsigned int *)(v94 + 8);
      }
      else
      {
        v101 = 0;
        v103 = 0;
      }
    }
    else
    {
      v101 = byte_3F871B3;
      v103 = 0;
    }
    v104 = *(_QWORD *)(v94 - 8 * v98);
LABEL_109:
    if ( !v104 )
    {
      v107 = 0;
      v105 = byte_3F871B3;
      goto LABEL_112;
    }
    goto LABEL_110;
  }
  return result;
}
