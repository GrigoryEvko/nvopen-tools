// Function: sub_2EAC530
// Address: 0x2eac530
//
_BYTE *__fastcall sub_2EAC530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6, _QWORD *a7)
{
  _BYTE *v11; // rax
  __int16 v12; // ax
  __int16 v13; // dx
  __int64 v14; // rcx
  void *v15; // rdx
  unsigned __int8 v16; // al
  unsigned __int8 v17; // dl
  unsigned __int8 v18; // al
  unsigned __int64 v19; // rax
  _BYTE *v20; // rdx
  __int64 v21; // r14
  _BYTE *v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  __int16 v25; // ax
  char *v26; // r15
  size_t v27; // r8
  _QWORD *v28; // rax
  signed __int64 v29; // rsi
  __int16 v30; // ax
  const char *v31; // rsi
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // r14
  char v36; // al
  char v37; // bl
  void *v38; // rdx
  __int64 v39; // rdi
  const char *v40; // rbx
  const char *v41; // r15
  const char *v42; // r14
  _QWORD *v43; // rdx
  void *v44; // rdx
  void *v45; // rdx
  __int64 v46; // rdx
  unsigned int v47; // ebx
  _BYTE *result; // rax
  _BYTE *v49; // rax
  __int64 v50; // r8
  const char *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  _BYTE *v55; // rax
  __int64 v56; // r8
  const char *v57; // rax
  __int64 v58; // rax
  _BYTE *v59; // rax
  __int64 v60; // r8
  const char *v61; // rax
  __int64 v62; // rax
  _BYTE *v63; // rax
  __int64 v64; // r8
  const char *v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  char *v68; // rbx
  size_t v69; // rax
  _BYTE *v70; // rdi
  size_t v71; // r14
  unsigned __int64 v72; // rax
  __int64 v73; // r8
  unsigned __int8 *v74; // rbx
  size_t v75; // rax
  _BYTE *v76; // rdi
  size_t v77; // r14
  unsigned __int64 v78; // rax
  __int64 v79; // r8
  char v80; // cl
  unsigned __int64 v81; // rdx
  char v82; // di
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // rbx
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rbx
  unsigned __int64 v87; // rbx
  void *v88; // rdx
  void *v89; // rdx
  __m128i *v90; // rdx
  void *v91; // rdx
  __int64 v92; // rdx
  unsigned __int64 v93; // r14
  __int16 v94; // ax
  const char *v95; // rsi
  __int64 v96; // rax
  __int64 v97; // rax
  unsigned __int64 v98; // rax
  int v99; // eax
  unsigned __int64 v100; // rbx
  unsigned __int64 v101; // rdx
  unsigned __int64 v102; // rbx
  __int64 v103; // rax
  unsigned __int64 v104; // rdx
  char *v105; // rax
  char *v106; // r15
  unsigned int v107; // ecx
  unsigned int v108; // esi
  __int64 v109; // rax
  char *v110; // r14
  size_t v111; // rdx
  _QWORD *(__fastcall *v112)(__int64); // rax
  __int64 v113; // rbx
  __int64 (__fastcall *v114)(__int64, __int64, __int64, __int64 (__fastcall ***)(_QWORD)); // rax
  unsigned __int64 v115; // rdi
  void (*v116)(void); // rax
  __int64 v117; // [rsp+8h] [rbp-58h]
  __int64 v118; // [rsp+8h] [rbp-58h]
  __int64 v119; // [rsp+8h] [rbp-58h]
  __int64 v120; // [rsp+8h] [rbp-58h]
  unsigned __int8 v121; // [rsp+8h] [rbp-58h]
  unsigned __int8 v122; // [rsp+8h] [rbp-58h]
  _QWORD v125[7]; // [rsp+28h] [rbp-38h] BYREF

  v11 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 24) )
  {
    sub_CB5D20(a2, 40);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v11 + 1;
    *v11 = 40;
  }
  v12 = *(_WORD *)(a1 + 32);
  if ( (v12 & 4) != 0 )
  {
    v92 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v92) <= 8 )
    {
      sub_CB6200(a2, "volatile ", 9u);
    }
    else
    {
      *(_BYTE *)(v92 + 8) = 32;
      *(_QWORD *)v92 = 0x656C6974616C6F76LL;
      *(_QWORD *)(a2 + 32) += 9LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 8) != 0 )
  {
    v91 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v91 <= 0xCu )
    {
      sub_CB6200(a2, "non-temporal ", 0xDu);
    }
    else
    {
      qmemcpy(v91, "non-temporal ", 13);
      *(_QWORD *)(a2 + 32) += 13LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x10) != 0 )
  {
    v90 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v90 <= 0xFu )
    {
      sub_CB6200(a2, "dereferenceable ", 0x10u);
    }
    else
    {
      *v90 = _mm_load_si128((const __m128i *)&xmmword_42EB9C0);
      *(_QWORD *)(a2 + 32) += 16LL;
    }
    v12 = *(_WORD *)(a1 + 32);
  }
  if ( (v12 & 0x20) == 0 )
  {
    v13 = v12 & 0x40;
    if ( a7 )
      goto LABEL_8;
LABEL_119:
    if ( v13 )
    {
      sub_904010(a2, "\"MOTargetFlag1\" ");
      v12 = *(_WORD *)(a1 + 32);
    }
    if ( (v12 & 0x80u) != 0 )
    {
      sub_904010(a2, "\"MOTargetFlag2\" ");
      v12 = *(_WORD *)(a1 + 32);
    }
    if ( (v12 & 0x100) != 0 )
    {
      sub_904010(a2, "\"MOTargetFlag3\" ");
      v12 = *(_WORD *)(a1 + 32);
    }
    if ( (v12 & 0x200) != 0 )
    {
      sub_904010(a2, "\"MOTargetFlag4\" ");
      v12 = *(_WORD *)(a1 + 32);
    }
    goto LABEL_12;
  }
  v89 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v89 <= 9u )
  {
    sub_CB6200(a2, "invariant ", 0xAu);
  }
  else
  {
    qmemcpy(v89, "invariant ", 10);
    *(_QWORD *)(a2 + 32) += 10LL;
  }
  v12 = *(_WORD *)(a1 + 32);
  v13 = v12 & 0x40;
  if ( !a7 )
    goto LABEL_119;
LABEL_8:
  if ( v13 )
  {
    v63 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v63 >= *(_QWORD *)(a2 + 24) )
    {
      v64 = sub_CB5D20(a2, 34);
    }
    else
    {
      v64 = a2;
      *(_QWORD *)(a2 + 32) = v63 + 1;
      *v63 = 34;
    }
    v120 = v64;
    v65 = (const char *)sub_2EAAE60((__int64)a7, 64);
    v66 = sub_904010(v120, v65);
    sub_904010(v66, "\" ");
    v12 = *(_WORD *)(a1 + 32);
    if ( (v12 & 0x80u) == 0 )
    {
LABEL_10:
      if ( (v12 & 0x100) == 0 )
        goto LABEL_11;
      goto LABEL_77;
    }
  }
  else if ( (v12 & 0x80u) == 0 )
  {
    goto LABEL_10;
  }
  v59 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v59 >= *(_QWORD *)(a2 + 24) )
  {
    v60 = sub_CB5D20(a2, 34);
  }
  else
  {
    v60 = a2;
    *(_QWORD *)(a2 + 32) = v59 + 1;
    *v59 = 34;
  }
  v119 = v60;
  v61 = (const char *)sub_2EAAE60((__int64)a7, 128);
  v62 = sub_904010(v119, v61);
  sub_904010(v62, "\" ");
  v12 = *(_WORD *)(a1 + 32);
  if ( (v12 & 0x100) != 0 )
  {
LABEL_77:
    v55 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v55 >= *(_QWORD *)(a2 + 24) )
    {
      v56 = sub_CB5D20(a2, 34);
    }
    else
    {
      v56 = a2;
      *(_QWORD *)(a2 + 32) = v55 + 1;
      *v55 = 34;
    }
    v118 = v56;
    v57 = (const char *)sub_2EAAE60((__int64)a7, 256);
    v58 = sub_904010(v118, v57);
    sub_904010(v58, "\" ");
    v12 = *(_WORD *)(a1 + 32);
    if ( (v12 & 0x200) != 0 )
      goto LABEL_69;
LABEL_12:
    if ( (v12 & 1) == 0 )
      goto LABEL_13;
    goto LABEL_72;
  }
LABEL_11:
  if ( (v12 & 0x200) == 0 )
    goto LABEL_12;
LABEL_69:
  v49 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v49 >= *(_QWORD *)(a2 + 24) )
  {
    v50 = sub_CB5D20(a2, 34);
  }
  else
  {
    v50 = a2;
    *(_QWORD *)(a2 + 32) = v49 + 1;
    *v49 = 34;
  }
  v117 = v50;
  v51 = (const char *)sub_2EAAE60((__int64)a7, 512);
  v52 = sub_904010(v117, v51);
  sub_904010(v52, "\" ");
  v12 = *(_WORD *)(a1 + 32);
  if ( (v12 & 1) == 0 )
  {
LABEL_13:
    if ( (v12 & 2) == 0 )
      goto LABEL_14;
    goto LABEL_75;
  }
LABEL_72:
  v53 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v53) <= 4 )
  {
    sub_CB6200(a2, "load ", 5u);
  }
  else
  {
    *(_DWORD *)v53 = 1684107116;
    *(_BYTE *)(v53 + 4) = 32;
    *(_QWORD *)(a2 + 32) += 5LL;
  }
  if ( (*(_WORD *)(a1 + 32) & 2) != 0 )
  {
LABEL_75:
    v54 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v54) <= 5 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"store ", 6u);
    }
    else
    {
      *(_DWORD *)v54 = 1919906931;
      *(_WORD *)(v54 + 4) = 8293;
      *(_QWORD *)(a2 + 32) += 6LL;
    }
  }
LABEL_14:
  v14 = *(unsigned __int8 *)(a1 + 36);
  if ( (_BYTE)v14 == 1 )
    goto LABEL_20;
  if ( !*(_DWORD *)(a4 + 8) )
  {
    v122 = *(_BYTE *)(a1 + 36);
    sub_B6F820(a5);
    v14 = v122;
  }
  v15 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v15 <= 0xAu )
  {
    v121 = v14;
    sub_CB6200(a2, (unsigned __int8 *)"syncscope(\"", 0xBu);
    v14 = v121;
  }
  else
  {
    qmemcpy(v15, "syncscope(\"", 11);
    *(_QWORD *)(a2 + 32) += 11LL;
  }
  sub_C92400(*(unsigned __int8 **)(*(_QWORD *)a4 + 16 * v14), *(_QWORD *)(*(_QWORD *)a4 + 16 * v14 + 8), a2);
  v67 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v67) <= 2 )
  {
    sub_CB6200(a2, "\") ", 3u);
LABEL_20:
    v16 = *(_BYTE *)(a1 + 37);
    v17 = v16 & 0xF;
    if ( (v16 & 0xF) == 0 )
      goto LABEL_21;
    goto LABEL_92;
  }
  *(_BYTE *)(v67 + 2) = 32;
  *(_WORD *)v67 = 10530;
  *(_QWORD *)(a2 + 32) += 3LL;
  v16 = *(_BYTE *)(a1 + 37);
  v17 = v16 & 0xF;
  if ( (v16 & 0xF) == 0 )
    goto LABEL_21;
LABEL_92:
  v68 = (&off_4B91120)[v17];
  if ( !v68 )
  {
    v70 = *(_BYTE **)(a2 + 32);
    v72 = *(_QWORD *)(a2 + 24);
    v73 = a2;
    goto LABEL_96;
  }
  v69 = strlen((&off_4B91120)[v17]);
  v70 = *(_BYTE **)(a2 + 32);
  v71 = v69;
  v72 = *(_QWORD *)(a2 + 24);
  if ( v71 <= v72 - (unsigned __int64)v70 )
  {
    v73 = a2;
    if ( v71 )
    {
      memcpy(v70, v68, v71);
      v72 = *(_QWORD *)(a2 + 24);
      v73 = a2;
      v70 = (_BYTE *)(v71 + *(_QWORD *)(a2 + 32));
      *(_QWORD *)(a2 + 32) = v70;
    }
LABEL_96:
    if ( (unsigned __int64)v70 < v72 )
      goto LABEL_97;
    goto LABEL_151;
  }
  v96 = sub_CB6200(a2, (unsigned __int8 *)v68, v71);
  v70 = *(_BYTE **)(v96 + 32);
  v73 = v96;
  if ( (unsigned __int64)v70 >= *(_QWORD *)(v96 + 24) )
  {
LABEL_151:
    sub_CB5D20(v73, 32);
    v16 = *(_BYTE *)(a1 + 37);
LABEL_21:
    v18 = v16 >> 4;
    if ( !v18 )
      goto LABEL_22;
    goto LABEL_98;
  }
LABEL_97:
  *(_QWORD *)(v73 + 32) = v70 + 1;
  *v70 = 32;
  v18 = *(_BYTE *)(a1 + 37) >> 4;
  if ( !v18 )
    goto LABEL_22;
LABEL_98:
  v74 = (unsigned __int8 *)(&off_4B91120)[v18];
  if ( v74 )
  {
    v75 = strlen((&off_4B91120)[v18]);
    v76 = *(_BYTE **)(a2 + 32);
    v77 = v75;
    v78 = *(_QWORD *)(a2 + 24);
    if ( v77 > v78 - (unsigned __int64)v76 )
    {
      v97 = sub_CB6200(a2, v74, v77);
      v76 = *(_BYTE **)(v97 + 32);
      v79 = v97;
      if ( *(_QWORD *)(v97 + 24) > (unsigned __int64)v76 )
        goto LABEL_103;
      goto LABEL_153;
    }
    v79 = a2;
    if ( v77 )
    {
      memcpy(v76, v74, v77);
      v78 = *(_QWORD *)(a2 + 24);
      v79 = a2;
      v76 = (_BYTE *)(v77 + *(_QWORD *)(a2 + 32));
      *(_QWORD *)(a2 + 32) = v76;
    }
  }
  else
  {
    v76 = *(_BYTE **)(a2 + 32);
    v78 = *(_QWORD *)(a2 + 24);
    v79 = a2;
  }
  if ( v78 > (unsigned __int64)v76 )
  {
LABEL_103:
    *(_QWORD *)(v79 + 32) = v76 + 1;
    *v76 = 32;
    goto LABEL_22;
  }
LABEL_153:
  sub_CB5D20(v79, 32);
LABEL_22:
  v19 = *(_QWORD *)(a2 + 24);
  v20 = *(_BYTE **)(a2 + 32);
  if ( (*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    if ( v19 <= (unsigned __int64)v20 )
    {
      v21 = sub_CB5D20(a2, 40);
    }
    else
    {
      v21 = a2;
      *(_QWORD *)(a2 + 32) = v20 + 1;
      *v20 = 40;
    }
    v125[0] = *(_QWORD *)(a1 + 24);
    sub_34B2640(v125, v21);
    v22 = *(_BYTE **)(v21 + 32);
    if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
    {
      sub_CB5D20(v21, 41);
    }
    else
    {
      *(_QWORD *)(v21 + 32) = v22 + 1;
      *v22 = 41;
    }
LABEL_27:
    v23 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
      goto LABEL_28;
LABEL_36:
    v29 = *(_QWORD *)(a1 + 8);
    if ( !v29 )
      goto LABEL_42;
    v30 = *(_WORD *)(a1 + 32);
    v31 = " into ";
    if ( (v30 & 1) != 0 )
    {
      v31 = " on ";
      if ( (v30 & 2) == 0 )
        v31 = " from ";
    }
    v32 = sub_904010(a2, v31);
    sub_904010(v32, "unknown-address");
    goto LABEL_41;
  }
  if ( v19 - (unsigned __int64)v20 <= 0xB )
  {
    sub_CB6200(a2, "unknown-size", 0xCu);
    goto LABEL_27;
  }
  qmemcpy(v20, "unknown-size", 12);
  *(_QWORD *)(a2 + 32) += 12LL;
  v23 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    goto LABEL_36;
LABEL_28:
  if ( ((v23 >> 2) & 1) == 0 )
  {
    v24 = v23 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v25 = *(_WORD *)(a1 + 32);
      v26 = " into ";
      if ( (v25 & 1) != 0 )
      {
        v26 = " on ";
        if ( (v25 & 2) == 0 )
          v26 = " from ";
      }
      v27 = strlen(v26);
      v28 = *(_QWORD **)(a2 + 32);
      if ( v27 <= *(_QWORD *)(a2 + 24) - (_QWORD)v28 )
      {
        if ( (unsigned int)v27 >= 8 )
        {
          *v28 = *(_QWORD *)v26;
          *(_QWORD *)((char *)v28 + (unsigned int)v27 - 8) = *(_QWORD *)&v26[(unsigned int)v27 - 8];
          v104 = (unsigned __int64)(v28 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v105 = (char *)v28 - v104;
          v106 = (char *)(v26 - v105);
          if ( (((_DWORD)v27 + (_DWORD)v105) & 0xFFFFFFF8) >= 8 )
          {
            v107 = (v27 + (_DWORD)v105) & 0xFFFFFFF8;
            v108 = 0;
            do
            {
              v109 = v108;
              v108 += 8;
              *(_QWORD *)(v104 + v109) = *(_QWORD *)&v106[v109];
            }
            while ( v108 < v107 );
          }
        }
        else if ( (v27 & 4) != 0 )
        {
          *(_DWORD *)v28 = *(_DWORD *)v26;
          *(_DWORD *)((char *)v28 + (unsigned int)v27 - 4) = *(_DWORD *)&v26[(unsigned int)v27 - 4];
        }
        else if ( (_DWORD)v27 )
        {
          *(_BYTE *)v28 = *v26;
          if ( (v27 & 2) != 0 )
            *(_WORD *)((char *)v28 + (unsigned int)v27 - 2) = *(_WORD *)&v26[(unsigned int)v27 - 2];
        }
        *(_QWORD *)(a2 + 32) += v27;
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)v26, v27);
      }
      sub_2F12490(a2, v24, a3);
      v29 = *(_QWORD *)(a1 + 8);
      goto LABEL_42;
    }
LABEL_41:
    v29 = *(_QWORD *)(a1 + 8);
    goto LABEL_42;
  }
  if ( ((v23 >> 2) & 1) == 0 )
    goto LABEL_41;
  v93 = v23 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_41;
  v94 = *(_WORD *)(a1 + 32);
  v95 = " into ";
  if ( (v94 & 1) != 0 )
  {
    v95 = " on ";
    if ( (v94 & 2) == 0 )
      v95 = " from ";
  }
  sub_904010(a2, v95);
  switch ( *(_DWORD *)(v93 + 8) )
  {
    case 0:
      sub_904010(a2, "stack");
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 1:
      sub_904010(a2, "got");
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 2:
      sub_904010(a2, "jump-table");
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 3:
      sub_904010(a2, "constant-pool");
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 4:
      sub_2EAC020(a2, *(_DWORD *)(v93 + 16), 1, a6);
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 5:
      sub_904010(a2, "call-entry ");
      sub_A5C020(*(_BYTE **)(v93 + 16), a2, 0, a3);
      v29 = *(_QWORD *)(a1 + 8);
      break;
    case 6:
      sub_904010(a2, "call-entry &");
      v110 = *(char **)(v93 + 16);
      v111 = 0;
      if ( v110 )
        v111 = strlen(v110);
      sub_A54F00(a2, (unsigned __int8 *)v110, v111);
      v29 = *(_QWORD *)(a1 + 8);
      break;
    default:
      v112 = *(_QWORD *(__fastcall **)(__int64))(*a7 + 1480LL);
      if ( v112 == sub_2EAAED0 )
      {
        v113 = a7[7];
        if ( !v113 )
        {
          v113 = sub_22077B0(8u);
          if ( v113 )
            *(_QWORD *)v113 = &unk_4A29780;
          v115 = a7[7];
          a7[7] = v113;
          if ( v115 )
          {
            v116 = *(void (**)(void))(*(_QWORD *)v115 + 8LL);
            if ( (char *)v116 == (char *)sub_2EAAD20 )
              j_j___libc_free_0(v115);
            else
              v116();
            v113 = a7[7];
          }
        }
      }
      else
      {
        v113 = (__int64)v112((__int64)a7);
      }
      sub_904010(a2, "custom \"");
      v114 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 (__fastcall ***)(_QWORD)))(*(_QWORD *)v113 + 32LL);
      if ( v114 == sub_2EAACB0 )
        (**(void (__fastcall ***)(unsigned __int64, __int64))v93)(v93, a2);
      else
        v114(v113, a2, a3, (__int64 (__fastcall ***)(_QWORD))v93);
      sub_A51310(a2, 0x22u);
      v29 = *(_QWORD *)(a1 + 8);
      break;
  }
LABEL_42:
  sub_2EAC0D0(a2, v29);
  v33 = *(_QWORD *)(a1 + 24);
  if ( (v33 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
LABEL_43:
    v34 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v34 <= 7u )
    {
      v35 = sub_CB6200(a2, ", align ", 8u);
    }
    else
    {
      v35 = a2;
      *v34 = 0x206E67696C61202CLL;
      *(_QWORD *)(a2 + 32) += 8LL;
    }
    v36 = sub_2EAC4F0(a1);
    sub_CB59D0(v35, 1LL << v36);
    goto LABEL_46;
  }
  v80 = *(_BYTE *)(a1 + 24);
  v81 = v33 >> 3;
  v82 = v80 & 2;
  if ( (v80 & 6) == 2 || (v80 & 1) != 0 )
  {
    v83 = HIDWORD(v33);
    if ( v82 )
      v83 = v81 >> 45;
    if ( (v83 + 7) >> 3 )
    {
      if ( (v80 & 6) == 2 )
        goto LABEL_110;
      goto LABEL_167;
    }
  }
  else
  {
    v98 = HIWORD(v33);
    if ( !v82 )
      LODWORD(v98) = HIDWORD(*(_QWORD *)(a1 + 24));
    if ( ((unsigned __int64)((unsigned __int16)((unsigned int)*(_QWORD *)(a1 + 24) >> 8) * (unsigned int)v98) + 7) >> 3 )
    {
LABEL_167:
      if ( (*(_BYTE *)(a1 + 24) & 1) == 0 )
      {
        v99 = (unsigned __int16)(v81 >> 5);
        v100 = v81;
        v101 = v81 >> 45;
        v102 = v100 >> 29;
        if ( v82 )
          LODWORD(v102) = v101;
        v87 = ((unsigned __int64)(unsigned int)(v99 * v102) + 7) >> 3;
LABEL_171:
        if ( 1LL << sub_2EAC4F0(a1) == v87 )
          goto LABEL_46;
        goto LABEL_43;
      }
LABEL_110:
      v84 = v81;
      v85 = v81 >> 29;
      v86 = v84 >> 45;
      if ( !v82 )
        v86 = v85;
      v87 = (v86 + 7) >> 3;
      goto LABEL_171;
    }
  }
LABEL_46:
  v37 = *(_BYTE *)(a1 + 34);
  if ( (unsigned __int8)sub_2EAC4F0(a1) != v37 )
  {
    v38 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v38 <= 0xBu )
    {
      v39 = sub_CB6200(a2, ", basealign ", 0xCu);
    }
    else
    {
      v39 = a2;
      qmemcpy(v38, ", basealign ", 12);
      *(_QWORD *)(a2 + 32) += 12LL;
    }
    sub_CB59D0(v39, 1LL << *(_BYTE *)(a1 + 34));
  }
  v40 = *(const char **)(a1 + 40);
  v41 = *(const char **)(a1 + 56);
  v42 = *(const char **)(a1 + 64);
  if ( v40 )
  {
    v43 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v43 <= 7u )
    {
      sub_CB6200(a2, ", !tbaa ", 8u);
    }
    else
    {
      *v43 = 0x206161627421202CLL;
      *(_QWORD *)(a2 + 32) += 8LL;
    }
    sub_A61DC0(v40, a2, a3, 0);
  }
  if ( v41 )
  {
    v44 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v44 <= 0xEu )
    {
      sub_CB6200(a2, ", !alias.scope ", 0xFu);
    }
    else
    {
      qmemcpy(v44, ", !alias.scope ", 15);
      *(_QWORD *)(a2 + 32) += 15LL;
    }
    sub_A61DC0(v41, a2, a3, 0);
  }
  if ( v42 )
  {
    v45 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v45 <= 0xAu )
    {
      sub_CB6200(a2, ", !noalias ", 0xBu);
    }
    else
    {
      qmemcpy(v45, ", !noalias ", 11);
      *(_QWORD *)(a2 + 32) += 11LL;
    }
    sub_A61DC0(v42, a2, a3, 0);
  }
  if ( *(_QWORD *)(a1 + 72) )
  {
    v46 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v46) <= 8 )
    {
      sub_CB6200(a2, ", !range ", 9u);
    }
    else
    {
      *(_BYTE *)(v46 + 8) = 32;
      *(_QWORD *)v46 = 0x65676E617221202CLL;
      *(_QWORD *)(a2 + 32) += 9LL;
    }
    sub_A61DC0(*(const char **)(a1 + 72), a2, a3, 0);
  }
  v47 = sub_2EAC1E0(a1);
  if ( v47 )
  {
    v88 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v88 <= 0xBu )
    {
      v103 = sub_CB6200(a2, ", addrspace ", 0xCu);
      sub_CB59D0(v103, v47);
    }
    else
    {
      qmemcpy(v88, ", addrspace ", 12);
      *(_QWORD *)(a2 + 32) += 12LL;
      sub_CB59D0(a2, v47);
    }
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
