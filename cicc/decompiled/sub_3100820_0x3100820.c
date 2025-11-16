// Function: sub_3100820
// Address: 0x3100820
//
__int64 __fastcall sub_3100820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  char **v4; // r12
  __int64 v5; // r13
  void *v6; // rdi
  char v7; // cl
  char *v8; // rdx
  unsigned __int8 v9; // al
  unsigned __int8 v10; // al
  unsigned __int8 *v11; // r15
  __int64 v12; // rax
  size_t v13; // rdx
  size_t v14; // r8
  char *v15; // r14
  unsigned __int8 v16; // al
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  size_t v19; // rdx
  _BYTE *v20; // rax
  void *v21; // rdx
  unsigned __int8 *v22; // rax
  size_t v23; // rdx
  size_t v24; // r15
  __m128i *v25; // rdx
  __m128i si128; // xmm0
  __int64 v27; // rdi
  __int64 v28; // rdi
  _BYTE *v29; // rax
  char **v30; // r12
  char *v31; // rdx
  __int64 v32; // rdi
  void *v33; // rax
  size_t v34; // rdx
  void *v35; // rdi
  char v36; // si
  unsigned int v37; // r14d
  char *v38; // rdx
  unsigned __int8 v39; // al
  unsigned __int8 v40; // al
  __int64 v41; // rdi
  __int64 v42; // rax
  size_t v43; // rdx
  size_t v44; // r8
  unsigned __int8 *v45; // rcx
  char *v46; // rax
  unsigned __int8 v47; // dl
  unsigned __int8 *v48; // rsi
  __int64 v49; // rax
  size_t v50; // rdx
  unsigned __int8 v51; // al
  char *v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // r14
  unsigned __int8 v57; // al
  char *v58; // r13
  __int64 v59; // rdi
  void *v60; // rax
  size_t v61; // rdx
  _WORD *v62; // rdi
  _BYTE *v63; // rax
  void *v64; // rdx
  char *v65; // r15
  __int64 v66; // r14
  unsigned __int8 v67; // al
  char *v68; // r13
  __int64 *v69; // r12
  __int64 v70; // rdi
  void *v71; // rax
  size_t v72; // rdx
  void *v73; // rdi
  unsigned __int8 v74; // al
  bool v75; // dl
  unsigned int v76; // r14d
  __int64 v77; // rdx
  unsigned __int8 v78; // cl
  char v79; // si
  __int64 v80; // rdi
  __int64 v81; // rdi
  size_t v82; // rdx
  size_t v83; // r8
  unsigned __int8 *v84; // rcx
  __int64 v85; // rax
  unsigned __int8 v86; // dl
  unsigned __int8 *v87; // rsi
  __int64 v88; // rax
  size_t v89; // rdx
  unsigned __int8 v90; // al
  __int64 v91; // rdx
  __int64 v92; // rdi
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // r14
  unsigned __int8 v96; // al
  __int64 v97; // r13
  __int64 v98; // rdi
  void *v99; // rax
  size_t v100; // rdx
  _WORD *v101; // rdi
  _BYTE *v102; // rax
  __int64 v103; // rax
  unsigned __int8 v104; // dl
  __int64 *v105; // rax
  __int64 v106; // r15
  __m128i *v107; // rdx
  __m128i v108; // xmm0
  __int64 v109; // r14
  __int64 v110; // r13
  __int64 v111; // rdx
  __int64 *v112; // r12
  __int64 v113; // rdx
  __int64 v114; // r13
  unsigned __int8 v115; // al
  __int64 v116; // r14
  __int64 v117; // rdx
  __int64 v118; // rdi
  __int64 v119; // rdx
  _BYTE *v120; // rax
  __int64 v121; // r15
  unsigned __int8 v122; // al
  __int64 v123; // rdx
  __int64 v124; // rdi
  void *v125; // rax
  size_t v126; // rdx
  void *v127; // rdi
  char v128; // si
  unsigned int v129; // r15d
  unsigned __int8 v130; // al
  __int64 v131; // rdx
  unsigned __int8 v132; // al
  __int64 v133; // rdi
  __int64 v134; // rax
  size_t v135; // rdx
  size_t v136; // r8
  unsigned __int8 *v137; // rcx
  __int64 v138; // rax
  __int64 *v139; // rdx
  unsigned __int8 v140; // dl
  unsigned __int8 *v141; // rsi
  __int64 v142; // rax
  size_t v143; // rdx
  _BYTE *v144; // rax
  unsigned __int8 *v145; // rax
  size_t v146; // rdx
  __m128i *v147; // rdx
  __m128i v148; // xmm0
  __int64 v149; // rdi
  __int64 v150; // rdi
  _BYTE *v151; // rax
  _BYTE *v152; // rdx
  unsigned __int8 v153; // al
  __int64 v154; // r14
  __int64 v155; // r14
  __int64 v156; // r13
  __int64 v157; // rax
  size_t v158; // rdx
  _WORD *v159; // rdi
  unsigned __int8 *v160; // rsi
  unsigned __int64 v161; // rax
  void *v163; // rdi
  unsigned __int16 v164; // ax
  void *v165; // rdx
  __int64 v166; // r15
  unsigned __int16 v167; // ax
  __int64 v168; // rdi
  _BYTE *v169; // rax
  __int64 v170; // rax
  __int64 v171; // rax
  size_t v174; // [rsp+18h] [rbp-48h]
  size_t v175; // [rsp+18h] [rbp-48h]
  size_t v176; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v177; // [rsp+20h] [rbp-40h]
  unsigned __int8 *v178; // [rsp+20h] [rbp-40h]
  size_t v179; // [rsp+20h] [rbp-40h]
  unsigned __int8 *v180; // [rsp+20h] [rbp-40h]
  size_t v181; // [rsp+20h] [rbp-40h]
  size_t v182; // [rsp+20h] [rbp-40h]
  size_t v183; // [rsp+20h] [rbp-40h]
  size_t v184; // [rsp+20h] [rbp-40h]
  size_t v185; // [rsp+20h] [rbp-40h]
  size_t v186; // [rsp+20h] [rbp-40h]
  size_t v187; // [rsp+28h] [rbp-38h]
  char **v188; // [rsp+28h] [rbp-38h]
  __int64 *v189; // [rsp+28h] [rbp-38h]
  __int64 *v190; // [rsp+28h] [rbp-38h]

  sub_AE8D20(a2, a3);
  v3 = *(_QWORD *)(a2 + 688);
  v4 = *(char ***)a2;
  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v5 )
  {
    while ( 1 )
    {
      v21 = *(void **)(v3 + 32);
      v15 = *v4;
      if ( *(_QWORD *)(v3 + 24) - (_QWORD)v21 <= 0xDu )
      {
        sub_CB6200(v3, "Compile unit: ", 0xEu);
      }
      else
      {
        qmemcpy(v21, "Compile unit: ", 14);
        *(_QWORD *)(v3 + 32) += 14LL;
      }
      v22 = (unsigned __int8 *)sub_E0A700(*((_DWORD *)v15 + 4));
      v24 = v23;
      if ( v23 )
      {
        v6 = *(void **)(v3 + 32);
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v6 < v23 )
        {
          sub_CB6200(v3, v22, v23);
        }
        else
        {
          memcpy(v6, v22, v23);
          *(_QWORD *)(v3 + 32) += v24;
        }
      }
      else
      {
        v25 = *(__m128i **)(v3 + 32);
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v25 <= 0x10u )
        {
          v27 = sub_CB6200(v3, "unknown-language(", 0x11u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_428C2D0);
          v25[1].m128i_i8[0] = 40;
          v27 = v3;
          *v25 = si128;
          *(_QWORD *)(v3 + 32) += 17LL;
        }
        v28 = sub_CB59D0(v27, *((unsigned int *)v15 + 4));
        v29 = *(_BYTE **)(v28 + 32);
        if ( *(_BYTE **)(v28 + 24) == v29 )
        {
          sub_CB6200(v28, (unsigned __int8 *)")", 1u);
        }
        else
        {
          *v29 = 41;
          ++*(_QWORD *)(v28 + 32);
        }
      }
      v7 = *v15;
      v8 = v15;
      if ( *v15 != 16 )
      {
        v9 = *(v15 - 16);
        if ( (v9 & 2) != 0 )
        {
          v8 = (char *)**((_QWORD **)v15 - 4);
          if ( !v8 )
            goto LABEL_31;
        }
        else
        {
          v8 = *(char **)&v15[-8 * ((v9 >> 2) & 0xF) - 16];
          if ( !v8 )
          {
LABEL_31:
            v11 = (unsigned __int8 *)byte_3F871B3;
            v14 = 0;
            goto LABEL_13;
          }
        }
      }
      v10 = *(v8 - 16);
      if ( (v10 & 2) != 0 )
      {
        v11 = *(unsigned __int8 **)(*((_QWORD *)v8 - 4) + 8LL);
        if ( !v11 )
          goto LABEL_75;
      }
      else
      {
        v11 = *(unsigned __int8 **)&v8[-8 * ((v10 >> 2) & 0xF) - 8];
        if ( !v11 )
        {
LABEL_75:
          v14 = 0;
          goto LABEL_11;
        }
      }
      v12 = sub_B91420((__int64)v11);
      v7 = *v15;
      v11 = (unsigned __int8 *)v12;
      v14 = v13;
LABEL_11:
      if ( v7 == 16 )
        goto LABEL_15;
      v9 = *(v15 - 16);
LABEL_13:
      if ( (v9 & 2) != 0 )
      {
        v15 = (char *)**((_QWORD **)v15 - 4);
        if ( !v15 )
          goto LABEL_28;
      }
      else
      {
        v15 = *(char **)&v15[-8 * ((v9 >> 2) & 0xF) - 16];
        if ( !v15 )
        {
LABEL_28:
          v19 = 0;
          v17 = (unsigned __int8 *)byte_3F871B3;
          goto LABEL_18;
        }
      }
LABEL_15:
      v16 = *(v15 - 16);
      if ( (v16 & 2) != 0 )
      {
        v17 = (unsigned __int8 *)**((_QWORD **)v15 - 4);
        if ( !v17 )
          goto LABEL_73;
      }
      else
      {
        v17 = *(unsigned __int8 **)&v15[-8 * ((v16 >> 2) & 0xF) - 16];
        if ( !v17 )
        {
LABEL_73:
          v19 = 0;
          goto LABEL_18;
        }
      }
      v187 = v14;
      v18 = sub_B91420((__int64)v17);
      v14 = v187;
      v17 = (unsigned __int8 *)v18;
LABEL_18:
      sub_31005E0(v3, v17, v19, v11, v14, 0);
      v20 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v20 >= *(_QWORD *)(v3 + 24) )
      {
        ++v4;
        sub_CB5D20(v3, 10);
        if ( (char **)v5 == v4 )
          break;
      }
      else
      {
        ++v4;
        *(_QWORD *)(v3 + 32) = v20 + 1;
        *v20 = 10;
        if ( (char **)v5 == v4 )
          break;
      }
    }
  }
  v30 = *(char ***)(a2 + 80);
  v188 = &v30[*(unsigned int *)(a2 + 88)];
  if ( v30 != v188 )
  {
    while ( 1 )
    {
      v64 = *(void **)(v3 + 32);
      v65 = *v30;
      if ( *(_QWORD *)(v3 + 24) - (_QWORD)v64 <= 0xBu )
      {
        v66 = sub_CB6200(v3, "Subprogram: ", 0xCu);
      }
      else
      {
        v66 = v3;
        qmemcpy(v64, "Subprogram: ", 12);
        *(_QWORD *)(v3 + 32) += 12LL;
      }
      v67 = *(v65 - 16);
      v68 = v65 - 16;
      if ( (v67 & 2) != 0 )
        v31 = (char *)*((_QWORD *)v65 - 4);
      else
        v31 = &v68[-8 * ((v67 >> 2) & 0xF)];
      v32 = *((_QWORD *)v31 + 2);
      if ( v32 )
      {
        v33 = (void *)sub_B91420(v32);
        v35 = *(void **)(v66 + 32);
        if ( *(_QWORD *)(v66 + 24) - (_QWORD)v35 >= v34 )
        {
          if ( v34 )
          {
            v179 = v34;
            memcpy(v35, v33, v34);
            *(_QWORD *)(v66 + 32) += v179;
          }
        }
        else
        {
          sub_CB6200(v66, (unsigned __int8 *)v33, v34);
        }
      }
      v36 = *v65;
      v37 = *((_DWORD *)v65 + 4);
      v38 = v65;
      if ( *v65 != 16 )
      {
        v39 = *(v65 - 16);
        if ( (v39 & 2) != 0 )
        {
          v38 = (char *)**((_QWORD **)v65 - 4);
          if ( !v38 )
            goto LABEL_86;
        }
        else
        {
          v38 = *(char **)&v68[-8 * ((v39 >> 2) & 0xF)];
          if ( !v38 )
          {
LABEL_86:
            v45 = (unsigned __int8 *)byte_3F871B3;
            v44 = 0;
            goto LABEL_47;
          }
        }
      }
      v40 = *(v38 - 16);
      if ( (v40 & 2) != 0 )
      {
        v41 = *(_QWORD *)(*((_QWORD *)v38 - 4) + 8LL);
        if ( !v41 )
          goto LABEL_134;
      }
      else
      {
        v41 = *(_QWORD *)&v38[-8 * ((v40 >> 2) & 0xF) - 8];
        if ( !v41 )
        {
LABEL_134:
          v44 = 0;
          goto LABEL_45;
        }
      }
      v42 = sub_B91420(v41);
      v36 = *v65;
      v41 = v42;
      v44 = v43;
LABEL_45:
      v45 = (unsigned __int8 *)v41;
      if ( v36 == 16 )
      {
        v46 = v65;
        v47 = *(v65 - 16);
        if ( (v47 & 2) != 0 )
          goto LABEL_50;
        goto LABEL_81;
      }
      v39 = *(v65 - 16);
LABEL_47:
      if ( (v39 & 2) != 0 )
      {
        v46 = (char *)**((_QWORD **)v65 - 4);
        if ( v46 )
          goto LABEL_49;
      }
      else
      {
        v46 = *(char **)&v68[-8 * ((v39 >> 2) & 0xF)];
        if ( v46 )
        {
LABEL_49:
          v47 = *(v46 - 16);
          if ( (v47 & 2) != 0 )
          {
LABEL_50:
            v48 = (unsigned __int8 *)**((_QWORD **)v46 - 4);
            if ( v48 )
              goto LABEL_51;
LABEL_82:
            v50 = 0;
            goto LABEL_52;
          }
LABEL_81:
          v48 = *(unsigned __int8 **)&v46[-8 * ((v47 >> 2) & 0xF) - 16];
          if ( v48 )
          {
LABEL_51:
            v174 = v44;
            v177 = v45;
            v49 = sub_B91420((__int64)v48);
            v45 = v177;
            v44 = v174;
            v48 = (unsigned __int8 *)v49;
            goto LABEL_52;
          }
          goto LABEL_82;
        }
      }
      v50 = 0;
      v48 = (unsigned __int8 *)byte_3F871B3;
LABEL_52:
      sub_31005E0(v3, v48, v50, v45, v44, v37);
      v51 = *(v65 - 16);
      if ( (v51 & 2) != 0 )
        v52 = (char *)*((_QWORD *)v65 - 4);
      else
        v52 = &v68[-8 * ((v51 >> 2) & 0xF)];
      v53 = *((_QWORD *)v52 + 3);
      if ( !v53 )
        goto LABEL_66;
      sub_B91420(v53);
      if ( !v54 )
        goto LABEL_66;
      v55 = *(_QWORD *)(v3 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v55) <= 2 )
      {
        v56 = sub_CB6200(v3, (unsigned __int8 *)" ('", 3u);
      }
      else
      {
        *(_BYTE *)(v55 + 2) = 39;
        v56 = v3;
        *(_WORD *)v55 = 10272;
        *(_QWORD *)(v3 + 32) += 3LL;
      }
      v57 = *(v65 - 16);
      if ( (v57 & 2) != 0 )
        v58 = (char *)*((_QWORD *)v65 - 4);
      else
        v58 = &v68[-8 * ((v57 >> 2) & 0xF)];
      v59 = *((_QWORD *)v58 + 3);
      if ( v59 )
      {
        v60 = (void *)sub_B91420(v59);
        v62 = *(_WORD **)(v56 + 32);
        if ( *(_QWORD *)(v56 + 24) - (_QWORD)v62 >= v61 )
        {
          if ( v61 )
          {
            v183 = v61;
            memcpy(v62, v60, v61);
            v62 = (_WORD *)(v183 + *(_QWORD *)(v56 + 32));
            *(_QWORD *)(v56 + 32) = v62;
          }
          goto LABEL_64;
        }
        v56 = sub_CB6200(v56, (unsigned __int8 *)v60, v61);
      }
      v62 = *(_WORD **)(v56 + 32);
LABEL_64:
      if ( *(_QWORD *)(v56 + 24) - (_QWORD)v62 <= 1u )
      {
        sub_CB6200(v56, (unsigned __int8 *)"')", 2u);
      }
      else
      {
        *v62 = 10535;
        *(_QWORD *)(v56 + 32) += 2LL;
      }
LABEL_66:
      v63 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v63 >= *(_QWORD *)(v3 + 24) )
      {
        ++v30;
        sub_CB5D20(v3, 10);
        if ( v188 == v30 )
          break;
      }
      else
      {
        ++v30;
        *(_QWORD *)(v3 + 32) = v63 + 1;
        *v63 = 10;
        if ( v188 == v30 )
          break;
      }
    }
  }
  v69 = *(__int64 **)(a2 + 160);
  v189 = &v69[*(unsigned int *)(a2 + 168)];
  if ( v69 != v189 )
  {
    while ( 1 )
    {
      v103 = *v69;
      v104 = *(_BYTE *)(*v69 - 16);
      if ( (v104 & 2) != 0 )
        v105 = *(__int64 **)(v103 - 32);
      else
        v105 = (__int64 *)(v103 - 16 - 8LL * ((v104 >> 2) & 0xF));
      v106 = *v105;
      v107 = *(__m128i **)(v3 + 32);
      if ( *(_QWORD *)(v3 + 24) - (_QWORD)v107 <= 0x10u )
      {
        v109 = sub_CB6200(v3, "Global variable: ", 0x11u);
      }
      else
      {
        v108 = _mm_load_si128((const __m128i *)&xmmword_428C2E0);
        v107[1].m128i_i8[0] = 32;
        v109 = v3;
        *v107 = v108;
        *(_QWORD *)(v3 + 32) += 17LL;
      }
      v74 = *(_BYTE *)(v106 - 16);
      v110 = v106 - 16;
      if ( (v74 & 2) != 0 )
      {
        v70 = *(_QWORD *)(*(_QWORD *)(v106 - 32) + 8LL);
        if ( !v70 )
        {
          v76 = *(_DWORD *)(v106 + 16);
LABEL_95:
          v77 = *(_QWORD *)(*(_QWORD *)(v106 - 32) + 16LL);
          if ( !v77 )
          {
            v84 = (unsigned __int8 *)byte_3F871B3;
            v83 = 0;
LABEL_99:
            v85 = *(_QWORD *)(*(_QWORD *)(v106 - 32) + 16LL);
            if ( !v85 )
              goto LABEL_132;
            goto LABEL_100;
          }
          v78 = *(_BYTE *)(v77 - 16);
          v79 = 1;
          if ( (v78 & 2) != 0 )
            goto LABEL_97;
LABEL_128:
          v80 = *(_QWORD *)(v77 - 16 - 8LL * ((v78 >> 2) & 0xF) + 8);
          if ( !v80 )
            goto LABEL_129;
LABEL_98:
          v81 = sub_B91420(v80);
          v74 = *(_BYTE *)(v106 - 16);
          v83 = v82;
          v84 = (unsigned __int8 *)v81;
          if ( (v74 & 2) != 0 )
            goto LABEL_99;
LABEL_130:
          v111 = v110 - 8LL * ((v74 >> 2) & 0xF);
          goto LABEL_131;
        }
      }
      else
      {
        v70 = *(_QWORD *)(v110 - 8LL * ((v74 >> 2) & 0xF) + 8);
        if ( !v70 )
        {
          v76 = *(_DWORD *)(v106 + 16);
          goto LABEL_126;
        }
      }
      v71 = (void *)sub_B91420(v70);
      v73 = *(void **)(v109 + 32);
      if ( *(_QWORD *)(v109 + 24) - (_QWORD)v73 >= v72 )
      {
        if ( v72 )
        {
          v181 = v72;
          memcpy(v73, v71, v72);
          *(_QWORD *)(v109 + 32) += v181;
        }
        v74 = *(_BYTE *)(v106 - 16);
        v75 = (v74 & 2) != 0;
      }
      else
      {
        sub_CB6200(v109, (unsigned __int8 *)v71, v72);
        v74 = *(_BYTE *)(v106 - 16);
        v75 = (v74 & 2) != 0;
      }
      v76 = *(_DWORD *)(v106 + 16);
      if ( v75 )
        goto LABEL_95;
LABEL_126:
      v77 = *(_QWORD *)(v106 - 8LL * ((v74 >> 2) & 0xF));
      if ( v77 )
      {
        v78 = *(_BYTE *)(v77 - 16);
        v79 = 0;
        if ( (v78 & 2) == 0 )
          goto LABEL_128;
LABEL_97:
        v80 = *(_QWORD *)(*(_QWORD *)(v77 - 32) + 8LL);
        if ( v80 )
          goto LABEL_98;
LABEL_129:
        v83 = 0;
        v84 = (unsigned __int8 *)v80;
        if ( v79 )
          goto LABEL_99;
        goto LABEL_130;
      }
      v84 = (unsigned __int8 *)byte_3F871B3;
      v83 = 0;
      v111 = v110 - 8LL * ((v74 >> 2) & 0xF);
LABEL_131:
      v85 = *(_QWORD *)(v111 + 16);
      if ( !v85 )
      {
LABEL_132:
        v89 = 0;
        v87 = (unsigned __int8 *)byte_3F871B3;
        goto LABEL_103;
      }
LABEL_100:
      v86 = *(_BYTE *)(v85 - 16);
      if ( (v86 & 2) != 0 )
      {
        v87 = **(unsigned __int8 ***)(v85 - 32);
        if ( v87 )
          goto LABEL_102;
      }
      else
      {
        v87 = *(unsigned __int8 **)(v85 - 16 - 8LL * ((v86 >> 2) & 0xF));
        if ( v87 )
        {
LABEL_102:
          v175 = v83;
          v178 = v84;
          v88 = sub_B91420((__int64)v87);
          v84 = v178;
          v83 = v175;
          v87 = (unsigned __int8 *)v88;
          goto LABEL_103;
        }
      }
      v89 = 0;
LABEL_103:
      sub_31005E0(v3, v87, v89, v84, v83, v76);
      v90 = *(_BYTE *)(v106 - 16);
      if ( (v90 & 2) != 0 )
        v91 = *(_QWORD *)(v106 - 32);
      else
        v91 = v110 - 8LL * ((v90 >> 2) & 0xF);
      v92 = *(_QWORD *)(v91 + 40);
      if ( !v92 )
        goto LABEL_117;
      sub_B91420(v92);
      if ( !v93 )
        goto LABEL_117;
      v94 = *(_QWORD *)(v3 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v94) <= 2 )
      {
        v95 = sub_CB6200(v3, (unsigned __int8 *)" ('", 3u);
      }
      else
      {
        *(_BYTE *)(v94 + 2) = 39;
        v95 = v3;
        *(_WORD *)v94 = 10272;
        *(_QWORD *)(v3 + 32) += 3LL;
      }
      v96 = *(_BYTE *)(v106 - 16);
      if ( (v96 & 2) != 0 )
        v97 = *(_QWORD *)(v106 - 32);
      else
        v97 = v110 - 8LL * ((v96 >> 2) & 0xF);
      v98 = *(_QWORD *)(v97 + 40);
      if ( v98 )
      {
        v99 = (void *)sub_B91420(v98);
        v101 = *(_WORD **)(v95 + 32);
        if ( *(_QWORD *)(v95 + 24) - (_QWORD)v101 >= v100 )
        {
          if ( v100 )
          {
            v185 = v100;
            memcpy(v101, v99, v100);
            v101 = (_WORD *)(v185 + *(_QWORD *)(v95 + 32));
            *(_QWORD *)(v95 + 32) = v101;
          }
          goto LABEL_115;
        }
        v95 = sub_CB6200(v95, (unsigned __int8 *)v99, v100);
      }
      v101 = *(_WORD **)(v95 + 32);
LABEL_115:
      if ( *(_QWORD *)(v95 + 24) - (_QWORD)v101 <= 1u )
      {
        sub_CB6200(v95, (unsigned __int8 *)"')", 2u);
      }
      else
      {
        *v101 = 10535;
        *(_QWORD *)(v95 + 32) += 2LL;
      }
LABEL_117:
      v102 = *(_BYTE **)(v3 + 32);
      if ( (unsigned __int64)v102 >= *(_QWORD *)(v3 + 24) )
      {
        ++v69;
        sub_CB5D20(v3, 10);
        if ( v189 == v69 )
          break;
      }
      else
      {
        ++v69;
        *(_QWORD *)(v3 + 32) = v102 + 1;
        *v102 = 10;
        if ( v189 == v69 )
          break;
      }
    }
  }
  v112 = *(__int64 **)(a2 + 240);
  v190 = &v112[*(unsigned int *)(a2 + 248)];
  while ( v190 != v112 )
  {
    v113 = *(_QWORD *)(v3 + 32);
    v114 = *v112;
    if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v113) <= 4 )
    {
      sub_CB6200(v3, "Type:", 5u);
    }
    else
    {
      *(_DWORD *)v113 = 1701869908;
      *(_BYTE *)(v113 + 4) = 58;
      *(_QWORD *)(v3 + 32) += 5LL;
    }
    v115 = *(_BYTE *)(v114 - 16);
    v116 = v114 - 16;
    if ( (v115 & 2) != 0 )
      v117 = *(_QWORD *)(v114 - 32);
    else
      v117 = v116 - 8LL * ((v115 >> 2) & 0xF);
    v118 = *(_QWORD *)(v117 + 16);
    if ( v118 )
    {
      sub_B91420(v118);
      if ( v119 )
      {
        v120 = *(_BYTE **)(v3 + 32);
        if ( (unsigned __int64)v120 >= *(_QWORD *)(v3 + 24) )
        {
          v121 = sub_CB5D20(v3, 32);
        }
        else
        {
          v121 = v3;
          *(_QWORD *)(v3 + 32) = v120 + 1;
          *v120 = 32;
        }
        v122 = *(_BYTE *)(v114 - 16);
        v123 = (v122 & 2) != 0 ? *(_QWORD *)(v114 - 32) : v116 - 8LL * ((v122 >> 2) & 0xF);
        v124 = *(_QWORD *)(v123 + 16);
        if ( v124 )
        {
          v125 = (void *)sub_B91420(v124);
          v127 = *(void **)(v121 + 32);
          if ( *(_QWORD *)(v121 + 24) - (_QWORD)v127 >= v126 )
          {
            if ( v126 )
            {
              v184 = v126;
              memcpy(v127, v125, v126);
              *(_QWORD *)(v121 + 32) += v184;
            }
          }
          else
          {
            sub_CB6200(v121, (unsigned __int8 *)v125, v126);
          }
        }
      }
    }
    v128 = *(_BYTE *)v114;
    v129 = *(_DWORD *)(v114 + 16);
    if ( *(_BYTE *)v114 == 16 )
    {
      if ( !v114 )
      {
        v137 = (unsigned __int8 *)byte_3F871B3;
        v136 = 0;
LABEL_228:
        v143 = 0;
        v141 = (unsigned __int8 *)byte_3F871B3;
        goto LABEL_169;
      }
      v131 = v114;
      v132 = *(_BYTE *)(v114 - 16);
      if ( (v132 & 2) != 0 )
        goto LABEL_159;
    }
    else
    {
      v130 = *(_BYTE *)(v114 - 16);
      if ( (v130 & 2) != 0 )
      {
        v131 = **(_QWORD **)(v114 - 32);
        if ( !v131 )
          goto LABEL_217;
      }
      else
      {
        v131 = *(_QWORD *)(v116 - 8LL * ((v130 >> 2) & 0xF));
        if ( !v131 )
        {
LABEL_217:
          v137 = (unsigned __int8 *)byte_3F871B3;
          v136 = 0;
          if ( (v130 & 2) == 0 )
            goto LABEL_218;
          goto LABEL_163;
        }
      }
      v132 = *(_BYTE *)(v131 - 16);
      if ( (v132 & 2) != 0 )
      {
LABEL_159:
        v133 = *(_QWORD *)(*(_QWORD *)(v131 - 32) + 8LL);
        if ( v133 )
          goto LABEL_160;
        goto LABEL_215;
      }
    }
    v133 = *(_QWORD *)(v131 - 16 - 8LL * ((v132 >> 2) & 0xF) + 8);
    if ( v133 )
    {
LABEL_160:
      v134 = sub_B91420(v133);
      v128 = *(_BYTE *)v114;
      v133 = v134;
      v136 = v135;
      goto LABEL_161;
    }
LABEL_215:
    v136 = 0;
LABEL_161:
    v137 = (unsigned __int8 *)v133;
    v138 = v114;
    if ( v128 == 16 )
      goto LABEL_165;
    v130 = *(_BYTE *)(v114 - 16);
    if ( (v130 & 2) == 0 )
    {
LABEL_218:
      v139 = (__int64 *)(v116 - 8LL * ((v130 >> 2) & 0xF));
      goto LABEL_164;
    }
LABEL_163:
    v139 = *(__int64 **)(v114 - 32);
LABEL_164:
    v138 = *v139;
LABEL_165:
    if ( !v138 )
      goto LABEL_228;
    v140 = *(_BYTE *)(v138 - 16);
    if ( (v140 & 2) != 0 )
    {
      v141 = **(unsigned __int8 ***)(v138 - 32);
      if ( v141 )
        goto LABEL_168;
    }
    else
    {
      v141 = *(unsigned __int8 **)(v138 - 16 - 8LL * ((v140 >> 2) & 0xF));
      if ( v141 )
      {
LABEL_168:
        v176 = v136;
        v180 = v137;
        v142 = sub_B91420((__int64)v141);
        v137 = v180;
        v136 = v176;
        v141 = (unsigned __int8 *)v142;
        goto LABEL_169;
      }
    }
    v143 = 0;
LABEL_169:
    sub_31005E0(v3, v141, v143, v137, v136, v129);
    v144 = *(_BYTE **)(v3 + 32);
    if ( *(_BYTE *)v114 == 12 )
    {
      if ( *(_BYTE **)(v3 + 24) == v144 )
      {
        sub_CB6200(v3, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v144 = 32;
        ++*(_QWORD *)(v3 + 32);
      }
      v145 = (unsigned __int8 *)sub_E09D50(*(_DWORD *)(v114 + 44));
      if ( !v146 )
      {
        v147 = *(__m128i **)(v3 + 32);
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v147 <= 0x10u )
        {
          v149 = sub_CB6200(v3, "unknown-encoding(", 0x11u);
        }
        else
        {
          v148 = _mm_load_si128((const __m128i *)&xmmword_428C2F0);
          v147[1].m128i_i8[0] = 40;
          v149 = v3;
          *v147 = v148;
          *(_QWORD *)(v3 + 32) += 17LL;
        }
        v150 = sub_CB59D0(v149, *(unsigned int *)(v114 + 44));
        v151 = *(_BYTE **)(v150 + 32);
        if ( (unsigned __int64)v151 >= *(_QWORD *)(v150 + 24) )
        {
          sub_CB5D20(v150, 41);
        }
        else
        {
          *(_QWORD *)(v150 + 32) = v151 + 1;
          *v151 = 41;
        }
        v152 = *(_BYTE **)(v3 + 32);
        goto LABEL_178;
      }
    }
    else
    {
      if ( (unsigned __int64)v144 >= *(_QWORD *)(v3 + 24) )
      {
        sub_CB5D20(v3, 32);
      }
      else
      {
        *(_QWORD *)(v3 + 32) = v144 + 1;
        *v144 = 32;
      }
      v164 = sub_AF18C0(v114);
      v145 = (unsigned __int8 *)sub_E02B90(v164);
      if ( !v146 )
      {
        v165 = *(void **)(v3 + 32);
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v165 <= 0xBu )
        {
          v166 = sub_CB6200(v3, "unknown-tag(", 0xCu);
        }
        else
        {
          v166 = v3;
          qmemcpy(v165, "unknown-tag(", 12);
          *(_QWORD *)(v3 + 32) += 12LL;
        }
        v167 = sub_AF18C0(v114);
        v168 = sub_CB59F0(v166, v167);
        v169 = *(_BYTE **)(v168 + 32);
        if ( *(_BYTE **)(v168 + 24) == v169 )
        {
          sub_CB6200(v168, (unsigned __int8 *)")", 1u);
        }
        else
        {
          *v169 = 41;
          ++*(_QWORD *)(v168 + 32);
        }
        v152 = *(_BYTE **)(v3 + 32);
        goto LABEL_178;
      }
    }
    v163 = *(void **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v163 < v146 )
    {
      sub_CB6200(v3, v145, v146);
      v152 = *(_BYTE **)(v3 + 32);
    }
    else
    {
      v182 = v146;
      memcpy(v163, v145, v146);
      v152 = (_BYTE *)(*(_QWORD *)(v3 + 32) + v182);
      *(_QWORD *)(v3 + 32) = v152;
    }
LABEL_178:
    if ( *(_BYTE *)v114 == 14 )
    {
      v153 = *(_BYTE *)(v114 - 16);
      v154 = (v153 & 2) != 0 ? *(_QWORD *)(v114 - 32) : v116 - 8LL * ((v153 >> 2) & 0xF);
      v155 = *(_QWORD *)(v154 + 56);
      if ( v155 )
      {
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v152 <= 0xEu )
        {
          v156 = sub_CB6200(v3, " (identifier: '", 0xFu);
        }
        else
        {
          v156 = v3;
          qmemcpy(v152, " (identifier: '", 15);
          *(_QWORD *)(v3 + 32) += 15LL;
        }
        v157 = sub_B91420(v155);
        v159 = *(_WORD **)(v156 + 32);
        v160 = (unsigned __int8 *)v157;
        v161 = *(_QWORD *)(v156 + 24) - (_QWORD)v159;
        if ( v161 < v158 )
        {
          v170 = sub_CB6200(v156, v160, v158);
          v159 = *(_WORD **)(v170 + 32);
          v156 = v170;
          v161 = *(_QWORD *)(v170 + 24) - (_QWORD)v159;
        }
        else if ( v158 )
        {
          v186 = v158;
          memcpy(v159, v160, v158);
          v171 = *(_QWORD *)(v156 + 24);
          v159 = (_WORD *)(v186 + *(_QWORD *)(v156 + 32));
          *(_QWORD *)(v156 + 32) = v159;
          v161 = v171 - (_QWORD)v159;
        }
        if ( v161 <= 1 )
        {
          sub_CB6200(v156, (unsigned __int8 *)"')", 2u);
        }
        else
        {
          *v159 = 10535;
          *(_QWORD *)(v156 + 32) += 2LL;
        }
        v152 = *(_BYTE **)(v3 + 32);
      }
    }
    if ( (unsigned __int64)v152 >= *(_QWORD *)(v3 + 24) )
    {
      sub_CB5D20(v3, 10);
    }
    else
    {
      *(_QWORD *)(v3 + 32) = v152 + 1;
      *v152 = 10;
    }
    ++v112;
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
