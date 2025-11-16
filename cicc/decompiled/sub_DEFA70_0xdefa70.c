// Function: sub_DEFA70
// Address: 0xdefa70
//
__int64 __fastcall sub_DEFA70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v6; // r15
  __int64 *i; // rbx
  __int64 v8; // rdx
  __int64 v9; // rdx
  _WORD *v10; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 *v14; // rbx
  __m128i si128; // xmm0
  __int64 v16; // r8
  const char *v17; // rax
  size_t v18; // rdx
  __int64 v19; // r8
  unsigned __int8 *v20; // rsi
  _WORD *v21; // rdi
  unsigned __int64 v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // r15
  _WORD *v25; // rdx
  __int64 v26; // rdx
  _WORD *v27; // rdx
  _BYTE *v28; // rax
  __int64 *v29; // rbx
  __m128i v30; // xmm0
  __int64 v31; // r8
  const char *v32; // rax
  size_t v33; // rdx
  __int64 v34; // r8
  unsigned __int8 *v35; // rsi
  _WORD *v36; // rdi
  unsigned __int64 v37; // rax
  _BYTE *v38; // rax
  __m128i *v39; // rdx
  __int64 v40; // r15
  const char *v41; // rsi
  bool v42; // al
  __int64 *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  _WORD *v46; // rdx
  __int64 v47; // r15
  __int64 *v48; // r15
  __int64 *v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // r15
  __int64 *v52; // r15
  __int64 *v53; // rbx
  __int64 v54; // rdi
  __int64 v55; // r15
  __int64 v56; // rdx
  void *v57; // rdx
  __int64 *v58; // r15
  __int64 *v59; // rbx
  __int64 v60; // rdi
  char *v61; // rsi
  __int64 result; // rax
  const char *v63; // rsi
  bool v64; // al
  __int64 *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  _WORD *v68; // rdx
  __int64 v69; // r12
  unsigned int v70; // eax
  __int64 v71; // rax
  unsigned __int8 *v72; // rax
  size_t v73; // rdx
  __int64 v74; // r9
  __int64 v75; // r8
  void *v76; // rdi
  __int64 *v77; // r15
  __int64 *v78; // rbx
  __int64 v79; // rdi
  unsigned __int8 *v80; // rax
  size_t v81; // rdx
  __int64 v82; // r9
  __int64 v83; // r8
  void *v84; // rdi
  __int64 *v85; // r15
  __int64 *v86; // rbx
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // [rsp+0h] [rbp-110h]
  __int64 v91; // [rsp+10h] [rbp-100h]
  __int64 v92; // [rsp+18h] [rbp-F8h]
  __int64 v93; // [rsp+18h] [rbp-F8h]
  __int64 v94; // [rsp+20h] [rbp-F0h]
  __int64 v95; // [rsp+28h] [rbp-E8h]
  __int64 v96; // [rsp+28h] [rbp-E8h]
  __int64 v97; // [rsp+28h] [rbp-E8h]
  __int64 v98; // [rsp+28h] [rbp-E8h]
  __int64 *v99; // [rsp+30h] [rbp-E0h]
  __int64 *v100; // [rsp+30h] [rbp-E0h]
  __int64 v101; // [rsp+48h] [rbp-C8h]
  __int64 v102; // [rsp+48h] [rbp-C8h]
  __int64 v103; // [rsp+48h] [rbp-C8h]
  __int64 v104; // [rsp+48h] [rbp-C8h]
  __int64 v105; // [rsp+48h] [rbp-C8h]
  size_t v106; // [rsp+48h] [rbp-C8h]
  __int64 v107; // [rsp+48h] [rbp-C8h]
  size_t v108; // [rsp+48h] [rbp-C8h]
  __int64 v109; // [rsp+48h] [rbp-C8h]
  __int64 v110; // [rsp+48h] [rbp-C8h]
  __int64 *v111; // [rsp+48h] [rbp-C8h]
  __int64 v112; // [rsp+48h] [rbp-C8h]
  __int64 v113; // [rsp+48h] [rbp-C8h]
  __int64 *v114; // [rsp+48h] [rbp-C8h]
  size_t v115; // [rsp+48h] [rbp-C8h]
  size_t v116; // [rsp+48h] [rbp-C8h]
  __int64 *v117; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v118; // [rsp+58h] [rbp-B8h]
  _BYTE v119[48]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v120; // [rsp+90h] [rbp-80h] BYREF
  __int64 v121; // [rsp+98h] [rbp-78h]
  _BYTE v122[112]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = *(__int64 **)(a3 + 16);
  for ( i = *(__int64 **)(a3 + 8); v6 != i; ++i )
  {
    v8 = *i;
    sub_DEFA70(a1, a2, v8);
  }
  v9 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v9) <= 4 )
  {
    sub_CB6200(a1, (unsigned __int8 *)"Loop ", 5u);
  }
  else
  {
    *(_DWORD *)v9 = 1886351180;
    *(_BYTE *)(v9 + 4) = 32;
    *(_QWORD *)(a1 + 32) += 5LL;
  }
  sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
  v10 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(a1, (unsigned __int8 *)": ", 2u);
  }
  else
  {
    *v10 = 8250;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  v120 = (__int64 *)v122;
  v121 = 0x800000000LL;
  sub_D46D90(a3, (__int64)&v120);
  if ( (_DWORD)v121 != 1 )
    sub_904010(a1, "<multiple exits> ");
  v94 = sub_DCF3A0(a2, (char *)a3, 0);
  if ( !sub_D96A50(v94) )
  {
    sub_904010(a1, "backedge-taken count is ");
    sub_D96060(a1, v94);
    v11 = *(_BYTE **)(a1 + 32);
    if ( *(_BYTE **)(a1 + 24) != v11 )
      goto LABEL_11;
LABEL_95:
    sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
    v13 = (unsigned int)v121;
    v12 = *(_QWORD *)(a1 + 32);
    if ( (unsigned int)v121 > 1uLL )
      goto LABEL_12;
    goto LABEL_96;
  }
  sub_904010(a1, "Unpredictable backedge-taken count.");
  v11 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v11 )
    goto LABEL_95;
LABEL_11:
  *v11 = 10;
  v12 = *(_QWORD *)(a1 + 32) + 1LL;
  v13 = (unsigned int)v121;
  *(_QWORD *)(a1 + 32) = v12;
  if ( v13 > 1 )
  {
LABEL_12:
    v14 = v120;
    v99 = &v120[v13];
    do
    {
      while ( 1 )
      {
        v24 = *v14;
        if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v12) > 0x10 )
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F74ED0);
          *(_BYTE *)(v12 + 16) = 32;
          v16 = a1;
          *(__m128i *)v12 = si128;
          *(_QWORD *)(a1 + 32) += 17LL;
        }
        else
        {
          v16 = sub_CB6200(a1, "  exit count for ", 0x11u);
        }
        v101 = v16;
        v17 = sub_BD5D20(v24);
        v19 = v101;
        v20 = (unsigned __int8 *)v17;
        v21 = *(_WORD **)(v101 + 32);
        v22 = *(_QWORD *)(v101 + 24) - (_QWORD)v21;
        if ( v18 > v22 )
        {
          v44 = sub_CB6200(v101, v20, v18);
          v21 = *(_WORD **)(v44 + 32);
          v19 = v44;
          v22 = *(_QWORD *)(v44 + 24) - (_QWORD)v21;
        }
        else if ( v18 )
        {
          v95 = v101;
          v106 = v18;
          memcpy(v21, v20, v18);
          v19 = v95;
          v45 = *(_QWORD *)(v95 + 24);
          v46 = (_WORD *)(*(_QWORD *)(v95 + 32) + v106);
          *(_QWORD *)(v95 + 32) = v46;
          v21 = v46;
          v22 = v45 - (_QWORD)v46;
        }
        if ( v22 <= 1 )
        {
          sub_CB6200(v19, (unsigned __int8 *)": ", 2u);
        }
        else
        {
          *v21 = 8250;
          *(_QWORD *)(v19 + 32) += 2LL;
        }
        v102 = sub_DBA6E0((__int64)a2, a3, v24, 0);
        sub_D96060(a1, v102);
        if ( sub_D96A50(v102) )
        {
          v41 = (const char *)a3;
          v117 = (__int64 *)v119;
          v118 = 0x600000000LL;
          v105 = sub_DBAF70((__int64)a2, a3, v24, (__int64)&v117, 0);
          v42 = sub_D96A50(v105);
          v43 = v117;
          if ( !v42 )
          {
            v97 = v105;
            v109 = sub_904010(a1, "\n  predicated exit count for ");
            v72 = (unsigned __int8 *)sub_BD5D20(v24);
            v74 = v109;
            v75 = v97;
            v76 = *(void **)(v109 + 32);
            if ( v73 > *(_QWORD *)(v109 + 24) - (_QWORD)v76 )
            {
              v88 = sub_CB6200(v109, v72, v73);
              v75 = v97;
              v74 = v88;
            }
            else if ( v73 )
            {
              v93 = v109;
              v116 = v73;
              memcpy(v76, v72, v73);
              v74 = v93;
              v75 = v97;
              *(_QWORD *)(v93 + 32) += v116;
            }
            v110 = v75;
            sub_904010(v74, ": ");
            sub_D96060(a1, v110);
            v41 = "\n   Predicates:\n";
            sub_904010(a1, "\n   Predicates:\n");
            v43 = v117;
            if ( &v117[(unsigned int)v118] != v117 )
            {
              v111 = v14;
              v77 = &v117[(unsigned int)v118];
              v78 = v117;
              do
              {
                v79 = *v78++;
                v41 = (const char *)a1;
                (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v79 + 24LL))(v79, a1, 4);
              }
              while ( v77 != v78 );
              v14 = v111;
              v43 = v117;
            }
          }
          if ( v43 != (__int64 *)v119 )
            _libc_free(v43, v41);
        }
        v23 = *(_BYTE **)(a1 + 32);
        if ( *(_BYTE **)(a1 + 24) == v23 )
          break;
        *v23 = 10;
        ++v14;
        v12 = *(_QWORD *)(a1 + 32) + 1LL;
        *(_QWORD *)(a1 + 32) = v12;
        if ( v99 == v14 )
          goto LABEL_25;
      }
      ++v14;
      sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
      v12 = *(_QWORD *)(a1 + 32);
    }
    while ( v99 != v14 );
LABEL_25:
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v12) > 4 )
      goto LABEL_26;
    goto LABEL_97;
  }
LABEL_96:
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v12) > 4 )
  {
LABEL_26:
    *(_DWORD *)v12 = 1886351180;
    *(_BYTE *)(v12 + 4) = 32;
    *(_QWORD *)(a1 + 32) += 5LL;
    goto LABEL_27;
  }
LABEL_97:
  sub_CB6200(a1, (unsigned __int8 *)"Loop ", 5u);
LABEL_27:
  sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
  v25 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v25 <= 1u )
  {
    sub_CB6200(a1, (unsigned __int8 *)": ", 2u);
  }
  else
  {
    *v25 = 8250;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  v92 = sub_DCF3A0(a2, (char *)a3, 1);
  if ( sub_D96A50(v92) )
  {
    sub_904010(a1, "Unpredictable constant max backedge-taken count. ");
  }
  else
  {
    sub_904010(a1, "constant max backedge-taken count is ");
    sub_D96060(a1, v92);
    if ( (unsigned __int8)sub_DBA820((__int64)a2, a3) )
      sub_904010(a1, ", actual taken count either this or zero.");
  }
  v26 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v26) <= 5 )
  {
    sub_CB6200(a1, "\nLoop ", 6u);
  }
  else
  {
    *(_DWORD *)v26 = 1869564938;
    *(_WORD *)(v26 + 4) = 8304;
    *(_QWORD *)(a1 + 32) += 6LL;
  }
  sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
  v27 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v27 <= 1u )
  {
    sub_CB6200(a1, (unsigned __int8 *)": ", 2u);
  }
  else
  {
    *v27 = 8250;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  v91 = sub_DCF3A0(a2, (char *)a3, 2);
  if ( sub_D96A50(v91) )
  {
    sub_904010(a1, "Unpredictable symbolic max backedge-taken count. ");
    v28 = *(_BYTE **)(a1 + 32);
    if ( *(_BYTE **)(a1 + 24) != v28 )
    {
LABEL_40:
      *v28 = 10;
      ++*(_QWORD *)(a1 + 32);
      goto LABEL_41;
    }
  }
  else
  {
    sub_904010(a1, "symbolic max backedge-taken count is ");
    sub_D96060(a1, v91);
    if ( (unsigned __int8)sub_DBA820((__int64)a2, a3) )
      sub_904010(a1, ", actual taken count either this or zero.");
    v28 = *(_BYTE **)(a1 + 32);
    if ( *(_BYTE **)(a1 + 24) != v28 )
      goto LABEL_40;
  }
  sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
LABEL_41:
  if ( (unsigned int)v121 > 1uLL )
  {
    v29 = v120;
    v100 = &v120[(unsigned int)v121];
    do
    {
      while ( 1 )
      {
        v39 = *(__m128i **)(a1 + 32);
        v40 = *v29;
        if ( *(_QWORD *)(a1 + 24) - (_QWORD)v39 > 0x1Du )
        {
          v30 = _mm_load_si128((const __m128i *)&xmmword_3F74EE0);
          v31 = a1;
          qmemcpy(&v39[1], "xit count for ", 14);
          *v39 = v30;
          *(_QWORD *)(a1 + 32) += 30LL;
        }
        else
        {
          v31 = sub_CB6200(a1, "  symbolic max exit count for ", 0x1Eu);
        }
        v103 = v31;
        v32 = sub_BD5D20(v40);
        v34 = v103;
        v35 = (unsigned __int8 *)v32;
        v36 = *(_WORD **)(v103 + 32);
        v37 = *(_QWORD *)(v103 + 24) - (_QWORD)v36;
        if ( v33 > v37 )
        {
          v66 = sub_CB6200(v103, v35, v33);
          v36 = *(_WORD **)(v66 + 32);
          v34 = v66;
          v37 = *(_QWORD *)(v66 + 24) - (_QWORD)v36;
        }
        else if ( v33 )
        {
          v96 = v103;
          v108 = v33;
          memcpy(v36, v35, v33);
          v34 = v96;
          v67 = *(_QWORD *)(v96 + 24);
          v68 = (_WORD *)(*(_QWORD *)(v96 + 32) + v108);
          *(_QWORD *)(v96 + 32) = v68;
          v36 = v68;
          v37 = v67 - (_QWORD)v68;
        }
        if ( v37 <= 1 )
        {
          sub_CB6200(v34, (unsigned __int8 *)": ", 2u);
        }
        else
        {
          *v36 = 8250;
          *(_QWORD *)(v34 + 32) += 2LL;
        }
        v104 = sub_DBA6E0((__int64)a2, a3, v40, 2);
        sub_D96060(a1, v104);
        if ( sub_D96A50(v104) )
        {
          v63 = (const char *)a3;
          v117 = (__int64 *)v119;
          v118 = 0x600000000LL;
          v107 = sub_DBAF70((__int64)a2, a3, v40, (__int64)&v117, 2);
          v64 = sub_D96A50(v107);
          v65 = v117;
          if ( !v64 )
          {
            v98 = v107;
            v112 = sub_904010(a1, "\n  predicated symbolic max exit count for ");
            v80 = (unsigned __int8 *)sub_BD5D20(v40);
            v82 = v112;
            v83 = v98;
            v84 = *(void **)(v112 + 32);
            if ( v81 > *(_QWORD *)(v112 + 24) - (_QWORD)v84 )
            {
              v89 = sub_CB6200(v112, v80, v81);
              v83 = v98;
              v82 = v89;
            }
            else if ( v81 )
            {
              v90 = v112;
              v115 = v81;
              memcpy(v84, v80, v81);
              v82 = v90;
              v83 = v98;
              *(_QWORD *)(v90 + 32) += v115;
            }
            v113 = v83;
            sub_904010(v82, ": ");
            sub_D96060(a1, v113);
            v63 = "\n   Predicates:\n";
            sub_904010(a1, "\n   Predicates:\n");
            v65 = v117;
            if ( &v117[(unsigned int)v118] != v117 )
            {
              v114 = v29;
              v85 = &v117[(unsigned int)v118];
              v86 = v117;
              do
              {
                v87 = *v86++;
                v63 = (const char *)a1;
                (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v87 + 24LL))(v87, a1, 4);
              }
              while ( v85 != v86 );
              v29 = v114;
              v65 = v117;
            }
          }
          if ( v65 != (__int64 *)v119 )
            _libc_free(v65, v63);
        }
        v38 = *(_BYTE **)(a1 + 32);
        if ( *(_BYTE **)(a1 + 24) == v38 )
          break;
        *v38 = 10;
        ++v29;
        ++*(_QWORD *)(a1 + 32);
        if ( v100 == v29 )
          goto LABEL_60;
      }
      ++v29;
      sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
    }
    while ( v100 != v29 );
  }
LABEL_60:
  v117 = (__int64 *)v119;
  v118 = 0x400000000LL;
  v47 = sub_DEF8B0(a2, a3, (__int64)&v117);
  if ( v94 != v47 )
  {
    sub_904010(a1, "Loop ");
    sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
    sub_904010(a1, ": ");
    if ( sub_D96A50(v47) )
    {
      sub_904010(a1, "Unpredictable predicated backedge-taken count.");
    }
    else
    {
      sub_904010(a1, "Predicated backedge-taken count is ");
      sub_D96060(a1, v47);
    }
    sub_904010(a1, "\n");
    sub_904010(a1, " Predicates:\n");
    v48 = v117;
    v49 = &v117[(unsigned int)v118];
    if ( v49 != v117 )
    {
      do
      {
        v50 = *v48++;
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v50 + 24LL))(v50, a1, 4);
      }
      while ( v49 != v48 );
    }
  }
  LODWORD(v118) = 0;
  v51 = sub_DBB040((__int64)a2, a3, (__int64)&v117);
  if ( v51 != v92 )
  {
    sub_904010(a1, "Loop ");
    sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
    sub_904010(a1, ": ");
    if ( sub_D96A50(v51) )
    {
      sub_904010(a1, "Unpredictable predicated constant max backedge-taken count.");
    }
    else
    {
      sub_904010(a1, "Predicated constant max backedge-taken count is ");
      sub_D96060(a1, v51);
    }
    sub_904010(a1, "\n");
    sub_904010(a1, " Predicates:\n");
    v52 = v117;
    v53 = &v117[(unsigned int)v118];
    if ( v53 != v117 )
    {
      do
      {
        v54 = *v52++;
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v54 + 24LL))(v54, a1, 4);
      }
      while ( v53 != v52 );
    }
  }
  LODWORD(v118) = 0;
  v55 = sub_DEF990(a2, (char *)a3, (__int64)&v117);
  if ( v55 != v91 )
  {
    v56 = *(_QWORD *)(a1 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v56) <= 4 )
    {
      sub_CB6200(a1, (unsigned __int8 *)"Loop ", 5u);
    }
    else
    {
      *(_DWORD *)v56 = 1886351180;
      *(_BYTE *)(v56 + 4) = 32;
      *(_QWORD *)(a1 + 32) += 5LL;
    }
    sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
    sub_904010(a1, ": ");
    if ( sub_D96A50(v55) )
    {
      sub_904010(a1, "Unpredictable predicated symbolic max backedge-taken count.");
    }
    else
    {
      sub_904010(a1, "Predicated symbolic max backedge-taken count is ");
      sub_D96060(a1, v55);
    }
    sub_904010(a1, "\n");
    v57 = *(void **)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v57 <= 0xCu )
    {
      sub_CB6200(a1, (unsigned __int8 *)" Predicates:\n", 0xDu);
    }
    else
    {
      qmemcpy(v57, " Predicates:\n", 13);
      *(_QWORD *)(a1 + 32) += 13LL;
    }
    v58 = v117;
    v59 = &v117[(unsigned int)v118];
    if ( v59 != v117 )
    {
      do
      {
        v60 = *v58++;
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v60 + 24LL))(v60, a1, 4);
      }
      while ( v59 != v58 );
    }
  }
  v61 = (char *)a3;
  result = sub_DCFA10(a2, (char *)a3);
  if ( (_BYTE)result )
  {
    sub_904010(a1, "Loop ");
    sub_A5BF40(**(unsigned __int8 ***)(a3 + 32), a1, 0, 0);
    sub_904010(a1, ": ");
    v69 = sub_904010(a1, "Trip multiple is ");
    v70 = sub_DE5EA0(a2, a3);
    v71 = sub_CB59D0(v69, v70);
    v61 = "\n";
    result = sub_904010(v71, "\n");
  }
  if ( v117 != (__int64 *)v119 )
    result = _libc_free(v117, v61);
  if ( v120 != (__int64 *)v122 )
    return _libc_free(v120, v61);
  return result;
}
