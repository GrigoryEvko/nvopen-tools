// Function: sub_1AB9F40
// Address: 0x1ab9f40
//
__int64 *__fastcall sub_1AB9F40(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, __int64),
        __int64 a5)
{
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v8; // r14
  _BYTE *v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  _BYTE *v15; // rdi
  const char **v16; // rax
  size_t v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r12
  __int64 v21; // rax
  _BYTE *v22; // rsi
  _QWORD *v23; // rbx
  __int64 v24; // rdx
  _BYTE *v25; // rdi
  const char **v26; // rax
  size_t v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rcx
  _QWORD *v30; // r12
  _BYTE *v31; // rsi
  __int64 v32; // rbx
  __int64 v33; // rdx
  _BYTE *v34; // rdi
  const char **v35; // rax
  __int64 v36; // rcx
  size_t v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // r13
  __int64 v40; // rax
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 v46; // rdx
  _QWORD *v47; // rax
  __int64 v48; // rbx
  _QWORD *v49; // r13
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r14
  _QWORD *v55; // r12
  __int64 v56; // rax
  __int64 *v57; // r14
  __int64 v58; // rdx
  __int16 v59; // r13
  __int64 v60; // rbx
  _QWORD *v61; // r12
  __int64 v62; // rax
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r14
  char v69; // al
  __int64 v70; // rbx
  __int64 v71; // r13
  __int64 v72; // r14
  __int64 v73; // r13
  unsigned int v74; // ebx
  __int64 v75; // rbx
  __int64 v76; // r12
  void *v77; // rax
  size_t v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r13
  __int64 v82; // rsi
  __int64 v83; // r14
  __int16 v84; // ax
  __int64 v85; // rcx
  __int64 v86; // r13
  __int64 v87; // rbx
  __int64 v88; // rdx
  _QWORD *v89; // r12
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r13
  __int64 v93; // rax
  char v94; // al
  __int64 v95; // r13
  __int64 v96; // rdx
  __int64 v97; // r14
  _QWORD *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // r12
  __int64 v101; // r13
  const void *v102; // rax
  __int64 v103; // rdi
  unsigned int v104; // r13d
  size_t v105; // rdx
  __int64 v106; // r14
  unsigned int v107; // esi
  __int64 v108; // r15
  __int64 v110; // r12
  __int64 v111; // r14
  void *v112; // rax
  size_t v113; // rdx
  __int64 v114; // rax
  __int64 v115; // r12
  __int64 v116; // rdx
  __int64 v117; // r12
  __int64 v118; // rdx
  size_t v119; // rdx
  size_t v120; // rdx
  size_t v121; // rdx
  __int64 v122; // [rsp+8h] [rbp-F8h]
  __int64 v123; // [rsp+10h] [rbp-F0h]
  __int64 v124; // [rsp+10h] [rbp-F0h]
  __int64 v126; // [rsp+28h] [rbp-D8h]
  __int64 v127; // [rsp+28h] [rbp-D8h]
  __int64 v128; // [rsp+28h] [rbp-D8h]
  const char **v129; // [rsp+28h] [rbp-D8h]
  __int64 v130; // [rsp+28h] [rbp-D8h]
  char v132; // [rsp+38h] [rbp-C8h]
  char v133; // [rsp+38h] [rbp-C8h]
  __int64 *v134; // [rsp+38h] [rbp-C8h]
  char v135; // [rsp+40h] [rbp-C0h]
  __int64 *i; // [rsp+40h] [rbp-C0h]
  __int16 v137; // [rsp+48h] [rbp-B8h]
  __int64 v138; // [rsp+48h] [rbp-B8h]
  unsigned int v139; // [rsp+48h] [rbp-B8h]
  __int64 v140; // [rsp+48h] [rbp-B8h]
  __int64 v141; // [rsp+48h] [rbp-B8h]
  __int64 *v143; // [rsp+58h] [rbp-A8h]
  __int64 v144; // [rsp+58h] [rbp-A8h]
  __int64 v145; // [rsp+58h] [rbp-A8h]
  int v147; // [rsp+60h] [rbp-A0h]
  unsigned int v148; // [rsp+68h] [rbp-98h]
  __int64 v149; // [rsp+68h] [rbp-98h]
  __int64 j; // [rsp+68h] [rbp-98h]
  __int64 v151; // [rsp+68h] [rbp-98h]
  __int64 v152; // [rsp+68h] [rbp-98h]
  __int64 v153; // [rsp+68h] [rbp-98h]
  __int64 v154; // [rsp+68h] [rbp-98h]
  __int64 v155; // [rsp+68h] [rbp-98h]
  const char *v156; // [rsp+70h] [rbp-90h] BYREF
  __int64 v157; // [rsp+78h] [rbp-88h]
  const char **v158; // [rsp+80h] [rbp-80h] BYREF
  size_t n; // [rsp+88h] [rbp-78h]
  _QWORD v160[14]; // [rsp+90h] [rbp-70h] BYREF

  v8 = *a2;
  v9 = (_BYTE *)a2[22];
  v10 = a2[23];
  v11 = sub_22077B0(736);
  v12 = (_QWORD *)v11;
  if ( v11 )
    sub_1631D60(v11, v9, v10, v8);
  *a1 = (__int64)v12;
  v13 = (_BYTE *)a2[26];
  if ( !v13 )
  {
    LOBYTE(v160[0]) = 0;
    v15 = (_BYTE *)v12[26];
    v119 = 0;
    v158 = (const char **)v160;
LABEL_148:
    v12[27] = v119;
    v15[v119] = 0;
    v16 = v158;
    goto LABEL_8;
  }
  v14 = a2[27];
  v158 = (const char **)v160;
  sub_1AB9B30((__int64 *)&v158, v13, (__int64)&v13[v14]);
  v15 = (_BYTE *)v12[26];
  v16 = (const char **)v15;
  if ( v158 == v160 )
  {
    v119 = n;
    if ( n )
    {
      if ( n == 1 )
        *v15 = v160[0];
      else
        memcpy(v15, v160, n);
      v119 = n;
      v15 = (_BYTE *)v12[26];
    }
    goto LABEL_148;
  }
  v17 = n;
  v18 = v160[0];
  if ( v16 == v12 + 28 )
  {
    v12[26] = v158;
    v12[27] = v17;
    v12[28] = v18;
  }
  else
  {
    v19 = v12[28];
    v12[26] = v158;
    v12[27] = v17;
    v12[28] = v18;
    if ( v16 )
    {
      v158 = v16;
      v160[0] = v19;
      goto LABEL_8;
    }
  }
  v158 = (const char **)v160;
  v16 = (const char **)v160;
LABEL_8:
  n = 0;
  *(_BYTE *)v16 = 0;
  if ( v158 != v160 )
    j_j___libc_free_0(v158, v160[0] + 1LL);
  v20 = *a1;
  v21 = sub_1632FA0((__int64)a2);
  sub_1632B40(v20, v21);
  v22 = (_BYTE *)a2[30];
  v23 = (_QWORD *)*a1;
  if ( !v22 )
  {
    LOBYTE(v160[0]) = 0;
    v121 = 0;
    v158 = (const char **)v160;
    v25 = (_BYTE *)v23[30];
LABEL_152:
    v23[31] = v121;
    v25[v121] = 0;
    v26 = v158;
    goto LABEL_15;
  }
  v24 = a2[31];
  v158 = (const char **)v160;
  sub_1AB9B30((__int64 *)&v158, v22, (__int64)&v22[v24]);
  v25 = (_BYTE *)v23[30];
  v26 = (const char **)v25;
  if ( v158 == v160 )
  {
    v121 = n;
    if ( n )
    {
      if ( n == 1 )
        *v25 = v160[0];
      else
        memcpy(v25, v160, n);
      v121 = n;
      v25 = (_BYTE *)v23[30];
    }
    goto LABEL_152;
  }
  v27 = n;
  v28 = v160[0];
  if ( v26 == v23 + 32 )
  {
    v23[30] = v158;
    v23[31] = v27;
    v23[32] = v28;
  }
  else
  {
    v29 = v23[32];
    v23[30] = v158;
    v23[31] = v27;
    v23[32] = v28;
    if ( v26 )
    {
      v158 = v26;
      v160[0] = v29;
      goto LABEL_15;
    }
  }
  v158 = (const char **)v160;
  v26 = (const char **)v160;
LABEL_15:
  n = 0;
  *(_BYTE *)v26 = 0;
  if ( v158 != v160 )
    j_j___libc_free_0(v158, v160[0] + 1LL);
  v30 = (_QWORD *)*a1;
  v31 = (_BYTE *)a2[11];
  v32 = *a1 + 104;
  if ( !v31 )
  {
    LOBYTE(v160[0]) = 0;
    v120 = 0;
    v158 = (const char **)v160;
    v34 = (_BYTE *)v30[11];
LABEL_150:
    v30[12] = v120;
    v34[v120] = 0;
    v35 = v158;
    goto LABEL_22;
  }
  v33 = a2[12];
  v158 = (const char **)v160;
  sub_1AB9B30((__int64 *)&v158, v31, (__int64)&v31[v33]);
  v34 = (_BYTE *)v30[11];
  v35 = (const char **)v34;
  if ( v158 == v160 )
  {
    v120 = n;
    if ( n )
    {
      if ( n == 1 )
        *v34 = v160[0];
      else
        memcpy(v34, v160, n);
      v120 = n;
      v34 = (_BYTE *)v30[11];
    }
    goto LABEL_150;
  }
  v36 = v160[0];
  v37 = n;
  if ( v34 == (_BYTE *)v32 )
  {
    v30[11] = v158;
    v30[12] = v37;
    v30[13] = v36;
  }
  else
  {
    v38 = v30[13];
    v30[11] = v158;
    v30[12] = v37;
    v30[13] = v36;
    if ( v35 )
    {
      v158 = v35;
      v160[0] = v38;
      goto LABEL_22;
    }
  }
  v158 = (const char **)v160;
  v35 = (const char **)v160;
LABEL_22:
  n = 0;
  *(_BYTE *)v35 = 0;
  if ( v158 != v160 )
    j_j___libc_free_0(v158, v160[0] + 1LL);
  v39 = v30[12];
  if ( v39 )
  {
    v40 = v30[11];
    if ( *(_BYTE *)(v40 + v39 - 1) != 10 )
    {
      v41 = v39 + 1;
      if ( v40 == v32 )
        v42 = 15;
      else
        v42 = v30[13];
      if ( v41 > v42 )
      {
        sub_2240BB0(v30 + 11, v30[12], 0, 0, 1);
        v40 = v30[11];
      }
      *(_BYTE *)(v40 + v39) = 10;
      v43 = v30[11];
      v30[12] = v41;
      *(_BYTE *)(v43 + v39 + 1) = 0;
    }
  }
  v44 = a2[2];
  v143 = a2 + 1;
  if ( (__int64 *)v44 != a2 + 1 )
  {
    do
    {
      v45 = *a1;
      if ( !v44 )
        BUG();
      v126 = *(_QWORD *)(v44 - 32);
      v132 = *(_BYTE *)(v44 + 24) & 1;
      v135 = *(_BYTE *)(v44 - 24) & 0xF;
      v156 = sub_1649960(v44 - 56);
      v157 = v46;
      LOWORD(v160[0]) = 261;
      v158 = &v156;
      v137 = (*(_BYTE *)(v44 - 23) >> 2) & 7;
      v148 = *(_DWORD *)(*(_QWORD *)(v44 - 56) + 8LL) >> 8;
      v47 = sub_1648A60(88, 1u);
      v48 = (__int64)v47;
      if ( v47 )
        sub_15E51E0((__int64)v47, v45, v126, v132, v135, 0, (__int64)&v158, 0, v137, v148, 0);
      sub_15E6480(v48, v44 - 56);
      v49 = sub_1AB9BE0(a3, v44 - 56);
      v50 = v49[2];
      if ( v48 != v50 )
      {
        if ( v50 != 0 && v50 != -8 && v50 != -16 )
          sub_1649B30(v49);
        v49[2] = v48;
        if ( v48 != 0 && v48 != -8 && v48 != -16 )
          sub_164C220((__int64)v49);
      }
      v44 = *(_QWORD *)(v44 + 8);
    }
    while ( v143 != (__int64 *)v44 );
  }
  v51 = a2[4];
  for ( i = a2 + 3; (__int64 *)v51 != i; v51 = *(_QWORD *)(v51 + 8) )
  {
    if ( !v51 )
    {
      v5 = sub_1649960(0);
      LOWORD(v160[0]) = 261;
      v156 = v5;
      v157 = v6;
      v158 = &v156;
      BUG();
    }
    v127 = *a1;
    v156 = sub_1649960(v51 - 56);
    LOWORD(v160[0]) = 261;
    v157 = v52;
    v158 = &v156;
    v133 = *(_BYTE *)(v51 - 24) & 0xF;
    v138 = *(_QWORD *)(v51 - 32);
    v53 = sub_1648B60(120);
    v54 = v53;
    if ( v53 )
      sub_15E2490(v53, v138, v133, (__int64)&v158, v127);
    sub_15E4330(v54, v51 - 56);
    v55 = sub_1AB9BE0(a3, v51 - 56);
    v56 = v55[2];
    if ( v56 != v54 )
    {
      if ( v56 != 0 && v56 != -8 && v56 != -16 )
        sub_1649B30(v55);
      v55[2] = v54;
      if ( v54 != 0 && v54 != -8 && v54 != -16 )
        sub_164C220((__int64)v55);
    }
  }
  v57 = (__int64 *)a2[6];
  v134 = a2 + 5;
  if ( a2 + 5 != v57 )
  {
    while ( 1 )
    {
      v63 = (__int64)(v57 - 6);
      if ( !v57 )
        v63 = 0;
      if ( !a4(a5, v63) )
        break;
      v64 = *a1;
      v156 = sub_1649960(v63);
      v157 = v65;
      LOWORD(v160[0]) = 261;
      v158 = &v156;
      v66 = *(_QWORD *)v63;
      if ( *(_BYTE *)(*(_QWORD *)v63 + 8LL) == 16 )
        v66 = **(_QWORD **)(v66 + 16);
      v60 = sub_15E5860(
              *(_QWORD *)(v63 + 24),
              *(_DWORD *)(v66 + 8) >> 8,
              *(_BYTE *)(v63 + 32) & 0xF,
              (__int64)&v158,
              v64);
      sub_15E4BE0(v60, v63);
      v61 = sub_1AB9BE0(a3, v63);
      v67 = v61[2];
      if ( v60 == v67 )
        goto LABEL_67;
      if ( v67 != -8 && v67 != 0 && v67 != -16 )
        sub_1649B30(v61);
      v61[2] = v60;
      if ( v60 != -8 && v60 != 0 )
      {
LABEL_65:
        if ( v60 != -16 )
          sub_164C220((__int64)v61);
LABEL_67:
        v57 = (__int64 *)v57[1];
        if ( v134 == v57 )
          goto LABEL_79;
      }
      else
      {
        v57 = (__int64 *)v57[1];
        if ( v134 == v57 )
          goto LABEL_79;
      }
    }
    if ( *(_BYTE *)(*(_QWORD *)(v63 + 24) + 8LL) == 12 )
    {
      v141 = *a1;
      v156 = sub_1649960(v63);
      v157 = v91;
      LOWORD(v160[0]) = 261;
      v158 = &v156;
      v92 = *(_QWORD *)(v63 + 24);
      v93 = sub_1648B60(120);
      v60 = v93;
      if ( v93 )
        sub_15E2490(v93, v92, 0, (__int64)&v158, v141);
    }
    else
    {
      v123 = *(_QWORD *)(v63 + 24);
      v128 = *a1;
      v156 = sub_1649960(v63);
      v157 = v58;
      LOWORD(v160[0]) = 261;
      v158 = &v156;
      v59 = (*(_BYTE *)(v63 + 33) >> 2) & 7;
      v139 = *(_DWORD *)(*(_QWORD *)v63 + 8LL) >> 8;
      v60 = (__int64)sub_1648A60(88, 1u);
      if ( v60 )
        sub_15E51E0(v60, v128, v123, 0, 0, 0, (__int64)&v158, 0, v59, v139, 0);
    }
    v61 = sub_1AB9BE0(a3, v63);
    v62 = v61[2];
    if ( v60 == v62 )
      goto LABEL_67;
    if ( v62 != -8 && v62 != 0 && v62 != -16 )
      sub_1649B30(v61);
    v61[2] = v60;
    if ( v60 == 0 || v60 == -8 )
      goto LABEL_67;
    goto LABEL_65;
  }
LABEL_79:
  v68 = a2[2];
  if ( v143 != (__int64 *)v68 )
  {
    while ( 1 )
    {
      v70 = v68 - 56;
      if ( !v68 )
        v70 = 0;
      if ( sub_15E4F60(v70) )
        goto LABEL_83;
      v71 = sub_1AB9BE0(a3, v70)[2];
      if ( !a4(a5, v70) )
        break;
      if ( !sub_15E4F60(v70) )
      {
        v154 = *(_QWORD *)(v70 - 24);
        sub_1B75040(&v158, a3, 0, 0, 0);
        v155 = sub_1B79690(&v158, v154, v118);
        sub_1B75110(&v158);
        sub_15E5440(v71, v155);
      }
      v158 = (const char **)v160;
      n = 0x100000000LL;
      sub_1626D60(v70, (__int64)&v158);
      v129 = &v158[2 * (unsigned int)n];
      if ( v158 != v129 )
      {
        v124 = v70;
        v140 = v71;
        v122 = v68;
        v72 = (__int64)v158;
        do
        {
          v73 = *(_QWORD *)(v72 + 8);
          v74 = *(_DWORD *)v72;
          v72 += 16;
          sub_1B75040(&v156, a3, 4, 0, 0);
          v149 = sub_1B79620(&v156, v73);
          sub_1B75110(&v156);
          sub_16267C0(v140, v74, v149);
        }
        while ( v129 != (const char **)v72 );
        v70 = v124;
        v71 = v140;
        v68 = v122;
      }
      v75 = *(_QWORD *)(v70 + 48);
      if ( v75 )
      {
        v76 = *(_QWORD *)(v71 + 40);
        v77 = (void *)sub_1580C70((_QWORD *)v75);
        v79 = sub_1633B90(v76, v77, v78);
        *(_DWORD *)(v79 + 8) = *(_DWORD *)(v75 + 8);
        *(_QWORD *)(v71 + 48) = v79;
      }
      if ( v158 == v160 )
      {
LABEL_83:
        v68 = *(_QWORD *)(v68 + 8);
        if ( v143 == (__int64 *)v68 )
          goto LABEL_98;
      }
      else
      {
        _libc_free((unsigned __int64)v158);
        v68 = *(_QWORD *)(v68 + 8);
        if ( v143 == (__int64 *)v68 )
          goto LABEL_98;
      }
    }
    v69 = *(_BYTE *)(v71 + 32);
    *(_BYTE *)(v71 + 32) = v69 & 0xF0;
    if ( (v69 & 0x30) != 0 )
      *(_BYTE *)(v71 + 33) |= 0x40u;
    goto LABEL_83;
  }
LABEL_98:
  for ( j = a2[4]; i != (__int64 *)j; j = *(_QWORD *)(j + 8) )
  {
    v80 = 0;
    if ( j )
      v80 = j - 56;
    v81 = v80;
    if ( !sub_15E4F60(v80) )
    {
      v82 = v81;
      v130 = sub_1AB9BE0(a3, v81)[2];
      if ( a4(a5, v81) )
      {
        if ( (*(_BYTE *)(v130 + 18) & 1) != 0 )
          sub_15E08E0(v130, v81);
        v83 = *(_QWORD *)(v130 + 88);
        v84 = *(_WORD *)(v81 + 18);
        if ( (v84 & 1) != 0 )
        {
          sub_15E08E0(v81, v81);
          v84 = *(_WORD *)(v81 + 18);
        }
        v85 = v81;
        v86 = *(_QWORD *)(v81 + 88);
        v87 = v85;
        while ( 1 )
        {
          if ( (v84 & 1) != 0 )
            sub_15E08E0(v87, v82);
          if ( v86 == *(_QWORD *)(v87 + 88) + 40LL * *(_QWORD *)(v87 + 96) )
            break;
          v156 = sub_1649960(v86);
          LOWORD(v160[0]) = 261;
          v157 = v88;
          v158 = &v156;
          sub_164B780(v83, (__int64 *)&v158);
          v82 = v86;
          v89 = sub_1AB9BE0(a3, v86);
          v90 = v89[2];
          if ( v83 != v90 )
          {
            LOBYTE(v82) = v90 != 0;
            if ( v90 != -8 && v90 != 0 && v90 != -16 )
              sub_1649B30(v89);
            v89[2] = v83;
            if ( (v83 & 0xFFFFFFFFFFFFFFF7LL) != 0xFFFFFFFFFFFFFFF0LL )
            {
              if ( v83 )
                sub_164C220((__int64)v89);
            }
          }
          v84 = *(_WORD *)(v87 + 18);
          v86 += 40;
          v83 += 40;
        }
        v158 = (const char **)v160;
        n = 0x800000000LL;
        sub_1AB5B80((_QWORD *)v130, (_BYTE *)v87, a3, 1u, (__int64)&v158, byte_3F871B3, 0, 0, 0);
        if ( (*(_BYTE *)(v87 + 18) & 8) != 0 )
        {
          v115 = sub_15E38F0(v87);
          sub_1B75040(&v156, a3, 0, 0, 0);
          v117 = sub_1B79690(&v156, v115, v116);
          sub_1B75110(&v156);
          sub_15E3D80(v130, v117);
        }
        v110 = *(_QWORD *)(v87 + 48);
        if ( v110 )
        {
          v111 = *(_QWORD *)(v130 + 40);
          v112 = (void *)sub_1580C70(*(_QWORD **)(v87 + 48));
          v114 = sub_1633B90(v111, v112, v113);
          *(_DWORD *)(v114 + 8) = *(_DWORD *)(v110 + 8);
          *(_QWORD *)(v130 + 48) = v114;
        }
        if ( v158 != v160 )
          _libc_free((unsigned __int64)v158);
      }
      else
      {
        v94 = *(_BYTE *)(v130 + 32);
        *(_BYTE *)(v130 + 32) = v94 & 0xF0;
        if ( (v94 & 0x30) != 0 )
          *(_BYTE *)(v130 + 33) |= 0x40u;
        sub_15E3D80(v130, 0);
      }
    }
  }
  if ( v134 != (__int64 *)a2[6] )
  {
    v95 = a2[6];
    do
    {
      v96 = v95 - 48;
      if ( !v95 )
        v96 = 0;
      v97 = v96;
      if ( a4(a5, v96) )
      {
        v98 = sub_1AB9BE0(a3, v97);
        v151 = *(_QWORD *)(v97 - 24);
        v144 = v98[2];
        if ( v151 )
        {
          sub_1B75040(&v158, a3, 0, 0, 0);
          v152 = sub_1B79690(&v158, v151, v99);
          sub_1B75110(&v158);
          sub_15E5930(v144, v152);
        }
      }
      v95 = *(_QWORD *)(v95 + 8);
    }
    while ( v134 != (__int64 *)v95 );
  }
  v100 = a2[10];
  if ( a2 + 9 != (__int64 *)v100 )
  {
    v145 = a3;
    do
    {
      v101 = *a1;
      v102 = (const void *)sub_161F640(v100);
      v103 = v101;
      v104 = 0;
      v106 = sub_1632440(v103, v102, v105);
      v147 = sub_161F520(v100);
      if ( v147 )
      {
        do
        {
          v107 = v104++;
          v108 = sub_161F530(v100, v107);
          sub_1B75040(&v158, v145, 0, 0, 0);
          v153 = sub_1B79620(&v158, v108);
          sub_1B75110(&v158);
          sub_1623CA0(v106, v153);
        }
        while ( v147 != v104 );
      }
      v100 = *(_QWORD *)(v100 + 8);
    }
    while ( a2 + 9 != (__int64 *)v100 );
  }
  return a1;
}
