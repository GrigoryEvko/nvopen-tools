// Function: sub_31E6750
// Address: 0x31e6750
//
__int64 __fastcall sub_31E6750(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // r13
  unsigned int v7; // edx
  unsigned __int64 v8; // rax
  int v9; // esi
  unsigned __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  const __m128i *v15; // rbx
  __m128i *v16; // rax
  _QWORD *v17; // rdi
  void (*v18)(); // rax
  __int64 v19; // rdi
  const char *v20; // rcx
  void (*v21)(); // rax
  char v22; // r12
  int v23; // ebx
  bool v24; // r14
  char v25; // r12
  char v26; // r13
  __int64 v27; // rdi
  void (*v28)(); // rax
  unsigned __int64 v29; // rsi
  int v30; // r12d
  __int64 v31; // rbx
  __int64 v32; // rsi
  char v33; // al
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // r10
  int v38; // ebx
  bool v39; // al
  int v40; // r10d
  int v41; // r9d
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // edx
  __int64 j; // rdi
  bool v47; // al
  __int64 v48; // rsi
  int v49; // edx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // r13
  __int64 v55; // rdi
  void (*v56)(); // rax
  unsigned __int64 v57; // rsi
  __int64 *v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 k; // r12
  __int64 v62; // r13
  void (*v63)(); // rax
  unsigned __int64 v64; // rsi
  __int64 v65; // rdi
  unsigned int v66; // ebx
  void (*v67)(); // rax
  __int64 *v68; // rbx
  __int64 v69; // r14
  void (*v70)(); // rax
  unsigned __int64 v71; // rsi
  __int64 v72; // rdi
  __int64 v73; // r13
  void (*v74)(); // rax
  __int64 v75; // r12
  void (*v76)(); // rax
  unsigned int v77; // eax
  __int64 v78; // rdi
  void (*v79)(); // rax
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // r12
  void (*v83)(); // rax
  unsigned int v84; // eax
  __int64 v85; // r12
  void (*v86)(); // rax
  __int64 v87; // rdi
  void (*v88)(); // rax
  __int64 v89; // rbx
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rsi
  __int64 v95; // rdi
  __int64 (*v96)(); // rax
  __int64 v97; // r14
  unsigned __int64 v98; // rax
  unsigned __int64 v99; // r9
  __int64 v100; // rdi
  int v101; // edx
  __int64 v102; // r12
  char (__fastcall *v103)(__int64, __int64); // rdx
  __int64 i; // r9
  bool v105; // al
  int v106; // eax
  __int64 v108; // rax
  __int64 *v109; // rdx
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rdi
  const void *v114; // rsi
  char *v115; // rbx
  unsigned __int64 v116; // rdi
  unsigned __int8 v118; // [rsp+1Bh] [rbp-C5h]
  char v119; // [rsp+1Ch] [rbp-C4h]
  unsigned __int8 v120; // [rsp+1Dh] [rbp-C3h]
  char v121; // [rsp+1Eh] [rbp-C2h]
  bool v122; // [rsp+1Fh] [rbp-C1h]
  unsigned __int8 v123; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v124; // [rsp+20h] [rbp-C0h]
  __int64 *v125; // [rsp+28h] [rbp-B8h]
  __int64 v126; // [rsp+28h] [rbp-B8h]
  bool v127; // [rsp+30h] [rbp-B0h]
  __int64 v128; // [rsp+38h] [rbp-A8h]
  __int64 v129; // [rsp+38h] [rbp-A8h]
  char v130; // [rsp+40h] [rbp-A0h]
  __int64 *v131; // [rsp+40h] [rbp-A0h]
  __int64 v132; // [rsp+48h] [rbp-98h]
  int v133[4]; // [rsp+50h] [rbp-90h] BYREF
  char v134; // [rsp+60h] [rbp-80h]
  char v135; // [rsp+70h] [rbp-70h]
  char v136; // [rsp+71h] [rbp-6Fh]
  char *v137; // [rsp+80h] [rbp-60h] BYREF
  __int64 v138; // [rsp+88h] [rbp-58h]
  __int64 v139; // [rsp+90h] [rbp-50h]
  unsigned int v140; // [rsp+98h] [rbp-48h]
  _BYTE *v141; // [rsp+A0h] [rbp-40h]
  __int64 v142; // [rsp+A8h] [rbp-38h]
  _BYTE v143[48]; // [rsp+B0h] [rbp-30h] BYREF

  v3 = sub_31DA6B0(a1);
  v4 = sub_E89F00(v3, *(_QWORD *)(a2 + 72));
  v5 = *(_QWORD *)(a1 + 224);
  v6 = v4;
  v7 = *(_DWORD *)(v5 + 128);
  v128 = *(_QWORD *)(a1 + 536);
  v8 = *(_QWORD *)(v5 + 120);
  if ( v7 )
  {
    v11 = v7;
    v10 = 32LL * v7;
    v32 = v8 + v10 - 32;
    v13 = *(_QWORD *)(v32 + 16);
    v7 = *(_DWORD *)(v32 + 24);
    v12 = *(_QWORD *)v32;
    v9 = *(_DWORD *)(v32 + 8);
  }
  else
  {
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
  }
  v140 = v7;
  v14 = v11 + 1;
  v15 = (const __m128i *)&v137;
  v137 = (char *)v12;
  LODWORD(v138) = v9;
  v139 = v13;
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 132) )
  {
    v113 = v5 + 120;
    v114 = (const void *)(v5 + 136);
    if ( v8 > (unsigned __int64)&v137 || (v10 += v8, (unsigned __int64)&v137 >= v10) )
    {
      sub_C8D5F0(v113, v114, v14, 0x20u, v10, v12);
      v8 = *(_QWORD *)(v5 + 120);
      v10 = 32LL * *(unsigned int *)(v5 + 128);
    }
    else
    {
      v115 = (char *)&v137 - v8;
      sub_C8D5F0(v113, v114, v14, 0x20u, v10, v12);
      v8 = *(_QWORD *)(v5 + 120);
      v15 = (const __m128i *)&v115[v8];
      v10 = 32LL * *(unsigned int *)(v5 + 128);
    }
  }
  v16 = (__m128i *)(v10 + v8);
  *v16 = _mm_loadu_si128(v15);
  v16[1] = _mm_loadu_si128(v15 + 1);
  ++*(_DWORD *)(v5 + 128);
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
    *(_QWORD *)(a1 + 224),
    v6,
    0);
  v17 = *(_QWORD **)(a1 + 224);
  v18 = *(void (**)())(*v17 + 120LL);
  v137 = "version";
  LOWORD(v141) = 259;
  if ( v18 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, char **, __int64))v18)(v17, &v137, 1);
    v17 = *(_QWORD **)(a1 + 224);
  }
  v120 = *(_BYTE *)(v17[1] + 1472LL);
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*v17 + 536LL))(v17, v120, 1);
  v19 = *(_QWORD *)(a1 + 224);
  v20 = "feature";
  v21 = *(void (**)())(*(_QWORD *)v19 + 120LL);
  v137 = "feature";
  LOWORD(v141) = 259;
  if ( v21 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v21)(v19, &v137, 1);
  v22 = dword_50360A8;
  v23 = *(_DWORD *)(a1 + 344);
  if ( (dword_50360A8 & 0x11) != 0 && (unsigned int)sub_39FAC40((unsigned int)dword_50360A8) != 1 )
  {
    v108 = sub_B2BE50(*(_QWORD *)a2);
    LOWORD(v141) = 259;
    v137 = "-pgo-anaylsis-map can accept only all or none with no additional values.";
    sub_B6ECE0(v108, (__int64)&v137);
    v22 = dword_50360A8;
  }
  if ( (v22 & 0x10) != 0 )
  {
    v119 = 1;
    v33 = qword_5035FC8;
    v24 = 1;
    v25 = 2;
    v121 = 1;
    v26 = 4;
    v122 = 1;
    goto LABEL_26;
  }
  if ( (v22 & 1) != 0 )
  {
    v26 = 0;
    v25 = 0;
    v24 = 0;
    v119 = 0;
    v130 = qword_5035FC8;
    v121 = 0;
    v122 = 0;
  }
  else
  {
    v24 = (v22 & 2) != 0;
    v20 = (const char *)(v22 & 4);
    v122 = (v22 & 4) != 0;
    v119 = v24;
    if ( (v22 & 8) != 0 )
    {
      v25 = 2 * v122;
      v121 = 1;
      v26 = 4;
      v130 = qword_5035FC8;
      if ( !(_BYTE)qword_5035FC8 )
        goto LABEL_15;
LABEL_27:
      v34 = sub_B2BE50(*(_QWORD *)a2);
      LOWORD(v141) = 259;
      v137 = "BB entries info is required for BBFreq and BrProb features";
      sub_B6ECE0(v34, (__int64)&v137);
      v130 = qword_5035FC8;
      goto LABEL_15;
    }
    v33 = qword_5035FC8;
    v130 = qword_5035FC8;
    if ( (v22 & 4) != 0 )
    {
      v121 = 0;
      v25 = 2;
      v26 = 0;
      v122 = 1;
LABEL_26:
      v130 = v33;
      if ( !v33 )
        goto LABEL_15;
      goto LABEL_27;
    }
    v121 = 0;
    v26 = 0;
    v25 = 0;
    v122 = 0;
  }
LABEL_15:
  v127 = v23 > 1 && *(_DWORD *)(a2 + 588) <= 2u;
  (*(void (__fastcall **)(_QWORD, _QWORD, __int64, const char *))(**(_QWORD **)(a1 + 224) + 536LL))(
    *(_QWORD *)(a1 + 224),
    (char)(v26 | v24 | v25 | (8 * v127) | (16 * v130)),
    1,
    v20);
  if ( v127 )
  {
    v27 = *(_QWORD *)(a1 + 224);
    v28 = *(void (**)())(*(_QWORD *)v27 + 120LL);
    v137 = "number of basic block ranges";
    LOWORD(v141) = 259;
    if ( v28 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, char **, __int64))v28)(v27, &v137, 1);
      v27 = *(_QWORD *)(a1 + 224);
    }
    v29 = *(unsigned int *)(a1 + 344);
    v30 = 0;
    sub_E98EB0(v27, v29, 0);
    v137 = 0;
    v141 = v143;
    v138 = 0;
    v31 = *(_QWORD *)(a2 + 328);
    v139 = 0;
    v140 = 0;
    v142 = 0;
    v132 = a2 + 320;
    if ( v31 == a2 + 320 )
      goto LABEL_43;
    do
    {
      while ( 1 )
      {
        ++v30;
        if ( *(_BYTE *)(v31 + 261) )
          break;
        v31 = *(_QWORD *)(v31 + 8);
        if ( v31 == v132 )
          goto LABEL_80;
      }
      v29 = (unsigned __int64)v133;
      *(_QWORD *)v133 = *(_QWORD *)(v31 + 252);
      *(_DWORD *)sub_31E6450((__int64)&v137, v133) = v30;
      v31 = *(_QWORD *)(v31 + 8);
      v30 = 0;
    }
    while ( v31 != v132 );
  }
  else
  {
    v75 = *(_QWORD *)(a1 + 224);
    v137 = 0;
    v138 = 0;
    v139 = 0;
    v140 = 0;
    v141 = v143;
    v142 = 0;
    v76 = *(void (**)())(*(_QWORD *)v75 + 120LL);
    v136 = 1;
    *(_QWORD *)v133 = "function address";
    v135 = 3;
    if ( v76 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, int *, __int64))v76)(v75, v133, 1);
      v75 = *(_QWORD *)(a1 + 224);
    }
    v77 = sub_31DAFE0(a1);
    sub_E9A500(v75, v128, v77, 0);
    v78 = *(_QWORD *)(a1 + 224);
    v79 = *(void (**)())(*(_QWORD *)v78 + 120LL);
    v136 = 1;
    *(_QWORD *)v133 = "number of basic blocks";
    v135 = 3;
    if ( v79 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, int *, __int64))v79)(v78, v133, 1);
      v78 = *(_QWORD *)(a1 + 224);
    }
    v29 = 0;
    v80 = *(_QWORD *)(a2 + 328);
    v132 = a2 + 320;
    if ( v80 != a2 + 320 )
    {
      do
      {
        v80 = *(_QWORD *)(v80 + 8);
        LODWORD(v29) = v29 + 1;
      }
      while ( v80 != a2 + 320 );
      v29 = (unsigned int)v29;
    }
    sub_E98EB0(v78, v29, 0);
  }
LABEL_80:
  if ( *(_QWORD *)(a2 + 328) != v132 )
  {
    v54 = *(_QWORD *)(a2 + 328);
    do
    {
      v81 = v128;
      if ( !sub_2E31AB0(v54) )
        v81 = sub_2E309C0(v54, v29, v50, v51, v52);
      if ( v127 && (*(_BYTE *)(v54 + 260) || sub_2E31AB0(v54)) )
      {
        v82 = *(_QWORD *)(a1 + 224);
        v83 = *(void (**)())(*(_QWORD *)v82 + 120LL);
        v136 = 1;
        *(_QWORD *)v133 = "base address";
        v135 = 3;
        if ( v83 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, int *, __int64))v83)(v82, v133, 1);
          v82 = *(_QWORD *)(a1 + 224);
        }
        v84 = sub_31DAFE0(a1);
        sub_E9A500(v82, v81, v84, 0);
        v85 = *(_QWORD *)(a1 + 224);
        v86 = *(void (**)())(*(_QWORD *)v85 + 120LL);
        v136 = 1;
        *(_QWORD *)v133 = "number of basic blocks";
        v135 = 3;
        if ( v86 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, int *, __int64))v86)(v85, v133, 1);
          v85 = *(_QWORD *)(a1 + 224);
        }
        *(_QWORD *)v133 = *(_QWORD *)(v54 + 252);
        v29 = *(unsigned int *)sub_31E6450((__int64)&v137, v133);
        sub_E98EB0(v85, v29, 0);
      }
      if ( v130 )
        goto LABEL_42;
      if ( v120 > 1u )
      {
        v87 = *(_QWORD *)(a1 + 224);
        v88 = *(void (**)())(*(_QWORD *)v87 + 120LL);
        v136 = 1;
        *(_QWORD *)v133 = "BB id";
        v135 = 3;
        if ( v88 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, int *, __int64))v88)(v87, v133, 1);
          v87 = *(_QWORD *)(a1 + 224);
        }
        sub_E98EB0(v87, *(unsigned int *)(v54 + 240), 0);
      }
      v89 = 0;
      sub_31DCA60(a1);
      v94 = sub_2E30F60(v54, v81, v90, v91, v92, v93);
      sub_31DCA60(a1);
      v126 = *(_QWORD *)(a1 + 224);
      v95 = *(_QWORD *)(*(_QWORD *)(v54 + 32) + 16LL);
      v96 = *(__int64 (**)())(*(_QWORD *)v95 + 128LL);
      if ( v96 != sub_2DAC790 )
        v89 = ((__int64 (__fastcall *)(__int64, __int64))v96)(v95, v94);
      v97 = v54 + 48;
      v98 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v99 = v98;
      if ( v54 + 48 == v98 )
      {
        LOBYTE(v102) = 0;
LABEL_121:
        LOBYTE(v37) = 0;
        goto LABEL_31;
      }
      if ( !v98 )
        BUG();
      v100 = *(_QWORD *)v98;
      v101 = *(_DWORD *)(v98 + 44);
      if ( (*(_QWORD *)v98 & 4) != 0 )
      {
        v116 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v101 & 4) != 0 )
        {
LABEL_125:
          v103 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)v89 + 1328LL);
          v102 = (*(_QWORD *)(*(_QWORD *)(v116 + 16) + 24LL) >> 5) & 1LL;
          goto LABEL_108;
        }
      }
      else if ( (v101 & 4) != 0 )
      {
        while ( 1 )
        {
          v116 = v100 & 0xFFFFFFFFFFFFFFF8LL;
          v101 = *(_DWORD *)(v116 + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)(v116 + 44) & 4) == 0 )
            break;
          v100 = *(_QWORD *)v116;
        }
      }
      else
      {
        v116 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      }
      if ( (v101 & 8) == 0 )
        goto LABEL_125;
      LOBYTE(v102) = sub_2E88A90(v116, 32, 1);
      v99 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v99 == v97 )
        goto LABEL_121;
      v103 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)v89 + 1328LL);
      if ( !v99 )
        BUG();
LABEL_108:
      if ( (*(_QWORD *)v99 & 4) != 0 )
      {
        if ( v103 != sub_2FDE950 )
          goto LABEL_144;
        v35 = *(_DWORD *)(v99 + 44) & 0xFFFFFF;
        if ( (*(_DWORD *)(v99 + 44) & 4) != 0 )
          goto LABEL_30;
      }
      else
      {
        if ( (*(_BYTE *)(v99 + 44) & 4) != 0 )
        {
          for ( i = *(_QWORD *)v99; ; i = *(_QWORD *)v99 )
          {
            v99 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v99 + 44) & 4) == 0 )
              break;
          }
        }
        if ( v103 != sub_2FDE950 )
        {
LABEL_144:
          LOBYTE(v37) = v103(v89, v99);
          goto LABEL_31;
        }
        v35 = *(_DWORD *)(v99 + 44) & 0xFFFFFF;
      }
      if ( (v35 & 8) != 0 )
      {
        v124 = v99;
        v105 = sub_2E88A90(v99, 32, 1);
        v99 = v124;
        LOBYTE(v37) = v105;
        if ( !v105 )
          goto LABEL_31;
        goto LABEL_117;
      }
LABEL_30:
      v36 = *(_QWORD *)(v99 + 16);
      v37 = (*(_QWORD *)(v36 + 24) >> 5) & 1LL;
      if ( (*(_QWORD *)(v36 + 24) & 0x20LL) == 0 )
        goto LABEL_31;
LABEL_117:
      v106 = *(_DWORD *)(v99 + 44);
      if ( (v106 & 4) != 0 || (v106 & 8) == 0 )
        LOBYTE(v37) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v99 + 16) + 24LL) >> 7;
      else
        LOBYTE(v37) = sub_2E88A90(v99, 128, 1);
LABEL_31:
      v123 = v37;
      v38 = *(unsigned __int8 *)(v54 + 216);
      v39 = sub_2E32580((__int64 *)v54);
      v40 = v123;
      v41 = v39;
      v42 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v43 = v42;
      if ( v42 == v97 )
      {
        v49 = 0;
        goto LABEL_41;
      }
      if ( !v42 )
        BUG();
      v44 = *(_QWORD *)v42;
      v45 = *(_DWORD *)(v42 + 44);
      if ( (v44 & 4) != 0 )
      {
        if ( (v45 & 4) != 0 )
        {
LABEL_123:
          v48 = (*(_QWORD *)(*(_QWORD *)(v43 + 16) + 24LL) >> 11) & 1LL;
          goto LABEL_40;
        }
      }
      else if ( (v45 & 4) != 0 )
      {
        for ( j = v44; ; j = *(_QWORD *)v43 )
        {
          v43 = j & 0xFFFFFFFFFFFFFFF8LL;
          v45 = *(_DWORD *)(v43 + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)(v43 + 44) & 4) == 0 )
            break;
        }
      }
      if ( (v45 & 8) == 0 )
        goto LABEL_123;
      v118 = v41;
      v47 = sub_2E88A90(v43, 2048, 1);
      v40 = v123;
      v41 = v118;
      LOBYTE(v48) = v47;
LABEL_40:
      v49 = 16 * (unsigned __int8)v48;
LABEL_41:
      v29 = v49 | (8 * v41) | (2 * v40) | (unsigned __int8)v102 | (unsigned int)(4 * v38);
      sub_E98EB0(v126, v29, 0);
LABEL_42:
      sub_2E30F60(v54, v29, v50, v51, v52, v53);
      v54 = *(_QWORD *)(v54 + 8);
    }
    while ( v54 != v132 );
  }
LABEL_43:
  if ( v119 )
  {
    v55 = *(_QWORD *)(a1 + 224);
    v56 = *(void (**)())(*(_QWORD *)v55 + 120LL);
    v136 = 1;
    *(_QWORD *)v133 = "function entry count";
    v135 = 3;
    if ( v56 != nullsub_98 )
      ((void (__fastcall *)(__int64, int *, __int64))v56)(v55, v133, 1);
    sub_B2EE70((__int64)v133, *(_QWORD *)a2, 0);
    v57 = 0;
    if ( v134 )
      v57 = *(_QWORD *)v133;
    sub_E98EB0(*(_QWORD *)(a1 + 224), v57, 0);
    if ( !v122 )
    {
LABEL_49:
      v125 = 0;
      if ( !v121 )
        goto LABEL_130;
      goto LABEL_50;
    }
  }
  else if ( !v122 )
  {
    if ( !v121 )
      goto LABEL_130;
    goto LABEL_49;
  }
  v109 = *(__int64 **)(a1 + 8);
  v110 = *v109;
  v111 = v109[1];
  if ( v110 == v111 )
LABEL_155:
    BUG();
  while ( *(_UNKNOWN **)v110 != &unk_503BDA8 )
  {
    v110 += 16;
    if ( v111 == v110 )
      goto LABEL_155;
  }
  v112 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v110 + 8) + 104LL))(
           *(_QWORD *)(v110 + 8),
           &unk_503BDA8);
  v125 = (__int64 *)sub_3503E60(v112);
  if ( v121 )
  {
LABEL_50:
    v58 = *(__int64 **)(a1 + 8);
    v59 = *v58;
    v60 = v58[1];
    if ( v59 == v60 )
LABEL_151:
      BUG();
    while ( *(_UNKNOWN **)v59 != &unk_501F1C8 )
    {
      v59 += 16;
      if ( v60 == v59 )
        goto LABEL_151;
    }
    v129 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v59 + 8) + 104LL))(
             *(_QWORD *)(v59 + 8),
             &unk_501F1C8)
         + 169;
    goto LABEL_55;
  }
  v129 = 0;
LABEL_55:
  for ( k = *(_QWORD *)(a2 + 328); k != v132; k = *(_QWORD *)(k + 8) )
  {
    if ( v122 )
    {
      v62 = *(_QWORD *)(a1 + 224);
      v63 = *(void (**)())(*(_QWORD *)v62 + 120LL);
      v136 = 1;
      *(_QWORD *)v133 = "basic block frequency";
      v135 = 3;
      if ( v63 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, int *, __int64))v63)(v62, v133, 1);
        v62 = *(_QWORD *)(a1 + 224);
      }
      v64 = sub_2E39EA0(v125, k);
      sub_E98EB0(v62, v64, 0);
    }
    if ( v121 )
    {
      v65 = *(_QWORD *)(a1 + 224);
      v66 = *(_DWORD *)(k + 120);
      v67 = *(void (**)())(*(_QWORD *)v65 + 120LL);
      v136 = 1;
      *(_QWORD *)v133 = "basic block successor count";
      v135 = 3;
      if ( v67 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, int *, __int64))v67)(v65, v133, 1);
        v65 = *(_QWORD *)(a1 + 224);
      }
      sub_E98EB0(v65, v66, 0);
      v68 = *(__int64 **)(k + 112);
      v131 = &v68[*(unsigned int *)(k + 120)];
      while ( v131 != v68 )
      {
        v72 = *(_QWORD *)(a1 + 224);
        v73 = *v68;
        v74 = *(void (**)())(*(_QWORD *)v72 + 120LL);
        v136 = 1;
        *(_QWORD *)v133 = "successor BB ID";
        v135 = 3;
        if ( v74 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, int *, __int64))v74)(v72, v133, 1);
          v72 = *(_QWORD *)(a1 + 224);
        }
        sub_E98EB0(v72, *(unsigned int *)(v73 + 240), 0);
        v69 = *(_QWORD *)(a1 + 224);
        v70 = *(void (**)())(*(_QWORD *)v69 + 120LL);
        v136 = 1;
        *(_QWORD *)v133 = "successor branch probability";
        v135 = 3;
        if ( v70 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, int *, __int64))v70)(v69, v133, 1);
          v69 = *(_QWORD *)(a1 + 224);
        }
        ++v68;
        v71 = (unsigned int)sub_2E441D0(v129, k, v73);
        sub_E98EB0(v69, v71, 0);
      }
    }
  }
LABEL_130:
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 168LL))(*(_QWORD *)(a1 + 224));
  if ( v141 != v143 )
    _libc_free((unsigned __int64)v141);
  return sub_C7D6A0(v138, 12LL * v140, 4);
}
