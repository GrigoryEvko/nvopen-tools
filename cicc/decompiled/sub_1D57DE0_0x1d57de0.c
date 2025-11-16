// Function: sub_1D57DE0
// Address: 0x1d57de0
//
__int64 __fastcall sub_1D57DE0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rcx
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 (*v18)(); // rcx
  __int64 v19; // rdx
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 **v24; // rax
  __int64 **v25; // r12
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int128 *v33; // rcx
  __int64 v34; // r13
  __int64 v35; // rbx
  __int64 v36; // r15
  __int64 v37; // r12
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // r13
  unsigned __int64 v43; // rax
  unsigned int v44; // r12d
  unsigned __int64 v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  _QWORD *v48; // r12
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 (*v54)(); // rax
  __int64 (*v55)(void); // rax
  __int64 v56; // rdx
  __int64 *v57; // r14
  __int64 *i; // r12
  __int64 v59; // rdi
  int v60; // r8d
  int v61; // r9d
  __int16 v62; // ax
  __int64 v63; // rax
  __int64 v64; // rax
  int v65; // r12d
  __int64 v66; // rax
  int *v67; // r15
  int v68; // r13d
  int v69; // r12d
  __int64 v70; // r8
  _DWORD *v71; // rax
  int v72; // edi
  int v73; // ecx
  _DWORD *v74; // rdx
  int v75; // edx
  unsigned int j; // r12d
  __int64 v77; // r15
  __int64 v78; // rax
  int v79; // r13d
  __int64 v80; // rdi
  __int64 v81; // rsi
  __int64 *v82; // r14
  __int64 v83; // rsi
  __int64 v84; // rax
  int v85; // r10d
  unsigned int v86; // ecx
  int *v87; // r14
  int v88; // edi
  __int64 v89; // r13
  __int64 v90; // rax
  __int64 v91; // rcx
  _BOOL4 v92; // r11d
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r10
  __int64 v97; // rdx
  __int16 v98; // cx
  __int64 v99; // r12
  __int64 v100; // r14
  __int64 v101; // r13
  __int64 v102; // r12
  __int64 k; // r15
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // r8
  __int64 v108; // rax
  __int64 v109; // rdi
  _BOOL4 v110; // ecx
  _QWORD *v111; // rdx
  unsigned __int64 *v112; // rcx
  __int64 v113; // rdx
  unsigned __int64 v114; // rsi
  _BYTE *v115; // rax
  __int64 v116; // r13
  __int64 v117; // r14
  unsigned __int64 v118; // rdi
  __int64 *v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rdx
  int *v123; // r9
  int *v124; // r14
  int v125; // esi
  int *v126; // rax
  int v127; // r15d
  int *v128; // r12
  __int64 v129; // rbx
  int v130; // r13d
  int *v131; // r8
  int v132; // r10d
  int *v133; // rax
  unsigned int v134; // edx
  int v135; // ecx
  __int64 v136; // rax
  __int64 v137; // rax
  int v138; // eax
  int v139; // edi
  __int64 v140; // rax
  __int64 *v141; // rdx
  __int64 v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rax
  unsigned int v145; // r12d
  __int64 v146; // rdx
  __int64 *v147; // rdx
  __int64 v148; // rax
  __int64 v149; // rdx
  __int64 v150; // rax
  char v151; // al
  unsigned int v152; // ecx
  int v153; // r8d
  int v154; // edi
  _DWORD *v155; // rsi
  int v156; // esi
  unsigned int v157; // r14d
  _DWORD *v158; // rcx
  int v159; // r8d
  __int64 v160; // [rsp+0h] [rbp-C0h]
  bool v162; // [rsp+1Bh] [rbp-A5h]
  int v163; // [rsp+1Ch] [rbp-A4h]
  __int64 v164; // [rsp+20h] [rbp-A0h]
  __int64 v165; // [rsp+20h] [rbp-A0h]
  _BOOL4 v166; // [rsp+20h] [rbp-A0h]
  __int64 v167; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v168; // [rsp+20h] [rbp-A0h]
  __int64 v169; // [rsp+28h] [rbp-98h]
  __int64 v170; // [rsp+28h] [rbp-98h]
  unsigned __int64 v171; // [rsp+28h] [rbp-98h]
  __int64 v172; // [rsp+30h] [rbp-90h]
  __int64 v173; // [rsp+30h] [rbp-90h]
  __int64 v174; // [rsp+38h] [rbp-88h]
  __int64 v175; // [rsp+40h] [rbp-80h]
  int *v176; // [rsp+40h] [rbp-80h]
  __int64 v177; // [rsp+48h] [rbp-78h]
  __int64 v178; // [rsp+48h] [rbp-78h]
  __int64 v179; // [rsp+48h] [rbp-78h]
  __int64 v180; // [rsp+48h] [rbp-78h]
  __int64 v181; // [rsp+58h] [rbp-68h] BYREF
  __int128 v182; // [rsp+60h] [rbp-60h] BYREF
  __int64 v183; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v184; // [rsp+78h] [rbp-48h]

  v11 = a1;
  v12 = *a2;
  *(_QWORD *)(a1 + 256) = a2;
  v174 = v12;
  sub_17000B0(*(_QWORD *)(a1 + 232), v12);
  v163 = *(_DWORD *)(a1 + 304);
  if ( v163 )
  {
    v145 = (unsigned __int8)sub_1636880(a1, v12) ? 0 : v163;
    v163 = *(_DWORD *)(a1 + 304);
    if ( v145 != v163 )
    {
      *(_DWORD *)(a1 + 304) = v145;
      v12 = v145;
      sub_1700730(*(_QWORD *)(a1 + 232), v145);
      v146 = *(_QWORD *)(a1 + 232);
      v162 = (*(_BYTE *)(v146 + 800) & 2) != 0;
      if ( !v145 )
        *(_BYTE *)(v146 + 800) = *(_BYTE *)(v146 + 640) & 2 | *(_BYTE *)(v146 + 800) & 0xFD;
    }
  }
  v13 = *(_QWORD *)(a1 + 256);
  v14 = *(_QWORD *)(v13 + 16);
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 40LL);
  v16 = 0;
  if ( v15 != sub_1D00B00 )
  {
    v16 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v15)(v14, v12, 0);
    v13 = *(_QWORD *)(v11 + 256);
  }
  *(_QWORD *)(v11 + 312) = v16;
  v17 = *(_QWORD *)(v13 + 16);
  v18 = *(__int64 (**)())(*(_QWORD *)v17 + 56LL);
  v19 = 0;
  if ( v18 != sub_1D12D20 )
  {
    v19 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v18)(v17, v12, 0);
    v13 = *(_QWORD *)(v11 + 256);
  }
  *(_QWORD *)(v11 + 320) = v19;
  v20 = *(__int64 **)(v11 + 8);
  *(_QWORD *)(v11 + 264) = *(_QWORD *)(v13 + 40);
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_293:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F9B6E8 )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_293;
  }
  *(_QWORD *)(v11 + 240) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
                             *(_QWORD *)(v21 + 8),
                             &unk_4F9B6E8)
                         + 360;
  if ( (*(_BYTE *)(v174 + 19) & 0x40) != 0 )
  {
    v141 = *(__int64 **)(v11 + 8);
    v142 = *v141;
    v143 = v141[1];
    if ( v142 == v143 )
LABEL_295:
      BUG();
    while ( *(_UNKNOWN **)v142 != &unk_4FC3606 )
    {
      v142 += 16;
      if ( v143 == v142 )
        goto LABEL_295;
    }
    v144 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v142 + 8) + 104LL))(
             *(_QWORD *)(v142 + 8),
             &unk_4FC3606);
    v23 = sub_1D8F610(v144, v174);
  }
  else
  {
    v23 = 0;
  }
  *(_QWORD *)(v11 + 296) = v23;
  v24 = (__int64 **)sub_22077B0(24);
  v25 = v24;
  if ( v24 )
    sub_143A950(v24, (__int64 *)v174);
  v26 = *(_QWORD *)(v11 + 408);
  *(_QWORD *)(v11 + 408) = v25;
  if ( v26 )
  {
    v27 = *(_QWORD *)(v26 + 16);
    if ( v27 )
    {
      sub_1368A00(*(__int64 **)(v26 + 16));
      j_j___libc_free_0(v27, 8);
    }
    j_j___libc_free_0(v26, 24);
  }
  v28 = sub_160F9A0(*(_QWORD *)(v11 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v28 && (v29 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v28 + 104LL))(v28, &unk_4F9E06C)) != 0 )
    v169 = v29 + 160;
  else
    v169 = 0;
  v30 = (__int64)&unk_4F9920C;
  v31 = sub_160F9A0(*(_QWORD *)(v11 + 8), (__int64)&unk_4F9920C, 1u);
  if ( v31
    && (v30 = (__int64)&unk_4F9920C,
        (v32 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v31 + 104LL))(v31, &unk_4F9920C)) != 0) )
  {
    v164 = v32 + 160;
  }
  else
  {
    v164 = 0;
  }
  v33 = &v182;
  v34 = *(_QWORD *)(v174 + 80);
  v177 = v174 + 72;
  if ( v34 != v174 + 72 )
  {
    v160 = v11;
    do
    {
      if ( !v34 )
        BUG();
      v35 = *(_QWORD *)(v34 + 24);
      if ( !v35 )
        BUG();
      if ( *(_BYTE *)(v35 - 8) == 77 )
      {
        v175 = v34;
        v172 = v34 - 24;
        while ( 1 )
        {
          if ( !v35 )
LABEL_290:
            BUG();
          if ( *(_BYTE *)(v35 - 8) != 77 )
            break;
          while ( (*(_DWORD *)(v35 - 4) & 0xFFFFFFF) != 0 )
          {
            v36 = v35 - 24;
            v37 = 0;
            v38 = 8LL * (*(_DWORD *)(v35 - 4) & 0xFFFFFFF);
            while ( 1 )
            {
              v39 = (*(_BYTE *)(v35 - 1) & 0x40) != 0
                  ? *(_QWORD *)(v35 - 32)
                  : v36 - 24LL * (*(_DWORD *)(v35 - 4) & 0xFFFFFFF);
              v40 = *(_QWORD *)(v39 + 3 * v37);
              if ( *(_BYTE *)(v40 + 16) == 5 && (unsigned __int8)sub_1593DF0(v40, v30, v39, v33) )
              {
                v41 = (*(_BYTE *)(v35 - 1) & 0x40) != 0
                    ? *(_QWORD *)(v35 - 32)
                    : v36 - 24LL * (*(_DWORD *)(v35 - 4) & 0xFFFFFFF);
                v42 = *(_QWORD *)(v37 + v41 + 24LL * *(unsigned int *)(v35 + 32) + 8);
                v43 = sub_157EBA0(v42);
                if ( (unsigned int)sub_15F4D60(v43) != 1 )
                  break;
              }
              v37 += 8;
              if ( v38 == v37 )
                goto LABEL_127;
            }
            LODWORD(v183) = 16777217;
            *(_QWORD *)&v182 = v169;
            *((_QWORD *)&v182 + 1) = v164;
            v44 = sub_137DFF0(v42, v172);
            v45 = sub_157EBA0(v42);
            v30 = v44;
            sub_1AAC5F0(v45, v44, &v182, a3, a4, a5, a6, v46, v47, a9, a10);
            v35 = *(_QWORD *)(v175 + 24);
            if ( !v35 )
              goto LABEL_290;
            if ( *(_BYTE *)(v35 - 8) != 77 )
              goto LABEL_43;
          }
LABEL_127:
          v35 = *(_QWORD *)(v35 + 8);
        }
LABEL_43:
        v34 = v175;
      }
      v34 = *(_QWORD *)(v34 + 8);
    }
    while ( v177 != v34 );
    v11 = v160;
  }
  v48 = *(_QWORD **)(v11 + 272);
  v49 = sub_160F9A0(*(_QWORD *)(v11 + 8), (__int64)&unk_4F98E0C, 1u);
  v50 = v49;
  if ( v49 )
    v50 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v49 + 104LL))(v49, &unk_4F98E0C);
  sub_1D17360(v48, *(_QWORD *)(v11 + 256), *(_QWORD *)(v11 + 408), v11, *(_QWORD *)(v11 + 240), v50);
  sub_1FE1BD0(*(_QWORD *)(v11 + 248), v174, *(_QWORD *)(v11 + 256), *(_QWORD *)(v11 + 272));
  if ( byte_4FC1CC0 && *(_DWORD *)(v11 + 304) )
  {
    v147 = *(__int64 **)(v11 + 8);
    v148 = *v147;
    v149 = v147[1];
    if ( v148 == v149 )
LABEL_291:
      BUG();
    while ( *(_UNKNOWN **)v148 != &unk_4F98724 )
    {
      v148 += 16;
      if ( v149 == v148 )
        goto LABEL_291;
    }
    *(_QWORD *)(*(_QWORD *)(v11 + 248) + 32LL) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v148 + 8)
                                                                                           + 104LL))(
                                                   *(_QWORD *)(v148 + 8),
                                                   &unk_4F98724)
                                               + 160;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)(v11 + 248) + 32LL) = 0;
  }
  if ( *(_DWORD *)(v11 + 304) )
  {
    v119 = *(__int64 **)(v11 + 8);
    v120 = *v119;
    v121 = v119[1];
    if ( v120 == v121 )
LABEL_292:
      BUG();
    while ( *(_UNKNOWN **)v120 != &unk_4F96DB4 )
    {
      v120 += 16;
      if ( v121 == v120 )
        goto LABEL_292;
    }
    v51 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v120 + 8) + 104LL))(
                        *(_QWORD *)(v120 + 8),
                        &unk_4F96DB4)
                    + 160);
    *(_QWORD *)(v11 + 288) = v51;
  }
  else
  {
    *(_QWORD *)(v11 + 288) = 0;
    v51 = 0;
  }
  sub_2051350(*(_QWORD *)(v11 + 280), *(_QWORD *)(v11 + 296), v51, *(_QWORD *)(v11 + 240));
  *(_BYTE *)(*(_QWORD *)(v11 + 256) + 345LL) = 0;
  *(_BYTE *)(*(_QWORD *)(v11 + 248) + 41LL) = 0;
  if ( !*(_DWORD *)(v11 + 304) )
  {
LABEL_186:
    v53 = *(_QWORD *)(v11 + 256);
    goto LABEL_55;
  }
  v52 = *(_QWORD *)(v11 + 320);
  v53 = *(_QWORD *)(v11 + 256);
  v54 = *(__int64 (**)())(*(_QWORD *)v52 + 1168LL);
  if ( v54 != sub_1D45FF0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v54)(v52, v53) )
    {
      *(_BYTE *)(*(_QWORD *)(v11 + 248) + 41LL) = 1;
      v116 = *(_QWORD *)(v174 + 80);
      if ( v116 != v177 )
      {
        while ( 1 )
        {
          v117 = v116 - 24;
          if ( !v116 )
            v117 = 0;
          v118 = sub_157EBA0(v117);
          if ( !v118 || !(unsigned int)sub_15F4D60(v118) )
          {
            v151 = *(_BYTE *)(sub_157EBA0(v117) + 16);
            if ( v151 != 31 && v151 != 25 )
              break;
          }
          v116 = *(_QWORD *)(v116 + 8);
          if ( v116 == v177 )
            goto LABEL_186;
        }
        *(_BYTE *)(*(_QWORD *)(v11 + 248) + 41LL) = 0;
        v53 = *(_QWORD *)(v11 + 256);
        goto LABEL_55;
      }
    }
    goto LABEL_186;
  }
LABEL_55:
  v178 = *(_QWORD *)(v53 + 328);
  if ( *(_BYTE *)(*(_QWORD *)(v11 + 248) + 41LL) )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v11 + 320) + 1176LL))(*(_QWORD *)(v11 + 320));
  sub_1D54C20(v11, (__int64 *)v174);
  if ( *(_BYTE *)(v11 + 328) && byte_4FC1DA0 )
  {
    *((_QWORD *)&v182 + 1) = 0x100000006LL;
    *(_QWORD *)&v182 = &unk_49ED058;
    v183 = v174;
    v150 = sub_15E0530(v174);
    sub_16027F0(v150, (__int64)&v182);
  }
  v173 = 0;
  v55 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(v11 + 256) + 16LL) + 112LL);
  if ( v55 != sub_1D00B10 )
    v173 = v55();
  sub_1E6A760(*(_QWORD *)(v11 + 264), v178, v173, *(_QWORD *)(v11 + 312));
  v56 = *(_QWORD *)(v11 + 248);
  if ( *(_BYTE *)(v56 + 41) )
  {
    *((_QWORD *)&v182 + 1) = 0x400000000LL;
    *(_QWORD *)&v182 = &v183;
    v57 = (__int64 *)a2[41];
    for ( i = a2 + 40; i != v57; v57 = (__int64 *)v57[1] )
    {
      while ( 1 )
      {
        if ( v57[12] == v57[11] )
        {
          v59 = sub_1DD5EE0(v57);
          if ( (__int64 *)v59 != v57 + 3 )
          {
            v62 = *(_WORD *)(v59 + 46);
            if ( (v62 & 4) == 0 && (v62 & 8) != 0 )
              LOBYTE(v63) = sub_1E15D00(v59, 8, 1);
            else
              v63 = (*(_QWORD *)(*(_QWORD *)(v59 + 16) + 8LL) >> 3) & 1LL;
            if ( (_BYTE)v63 )
              break;
          }
        }
        v57 = (__int64 *)v57[1];
        if ( i == v57 )
          goto LABEL_75;
      }
      v64 = DWORD2(v182);
      if ( DWORD2(v182) >= HIDWORD(v182) )
      {
        sub_16CD150((__int64)&v182, &v183, 0, 8, v60, v61);
        v64 = DWORD2(v182);
      }
      *(_QWORD *)(v182 + 8 * v64) = v57;
      ++DWORD2(v182);
    }
LABEL_75:
    (*(void (__fastcall **)(_QWORD, __int64, __int128 *))(**(_QWORD **)(v11 + 320) + 1184LL))(
      *(_QWORD *)(v11 + 320),
      v178,
      &v182);
    if ( (__int64 *)v182 != &v183 )
      _libc_free(v182);
    v56 = *(_QWORD *)(v11 + 248);
  }
  v182 = 0u;
  v183 = 0;
  v184 = 0;
  v65 = *(_DWORD *)(v56 + 408);
  if ( !v65 )
    goto LABEL_130;
  v66 = *(_QWORD *)(v11 + 264);
  v67 = *(int **)(v66 + 360);
  v176 = *(int **)(v66 + 368);
  if ( v67 == v176 )
    goto LABEL_95;
  do
  {
    while ( 1 )
    {
      v68 = v67[1];
      if ( v68 )
      {
        v69 = *v67;
        if ( !v184 )
        {
          *(_QWORD *)&v182 = v182 + 1;
LABEL_254:
          sub_1392B70((__int64)&v182, 2 * v184);
          if ( !v184 )
            goto LABEL_287;
          v152 = (v184 - 1) & (37 * v69);
          v75 = v183 + 1;
          v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8LL * v152);
          v153 = *v71;
          if ( v69 != *v71 )
          {
            v154 = 1;
            v155 = 0;
            while ( v153 != -1 )
            {
              if ( !v155 && v153 == -2 )
                v155 = v71;
              v152 = (v184 - 1) & (v154 + v152);
              v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8LL * v152);
              v153 = *v71;
              if ( v69 == *v71 )
                goto LABEL_91;
              ++v154;
            }
            if ( v155 )
              v71 = v155;
          }
          goto LABEL_91;
        }
        LODWORD(v70) = (v184 - 1) & (37 * v69);
        v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8LL * (unsigned int)v70);
        v72 = *v71;
        if ( v69 != *v71 )
          break;
      }
LABEL_81:
      v67 += 2;
      if ( v176 == v67 )
        goto LABEL_94;
    }
    v73 = 1;
    v74 = 0;
    while ( v72 != -1 )
    {
      if ( v72 == -2 && !v74 )
        v74 = v71;
      v70 = (v184 - 1) & ((_DWORD)v70 + v73);
      v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8 * v70);
      v72 = *v71;
      if ( v69 == *v71 )
        goto LABEL_81;
      ++v73;
    }
    if ( v74 )
      v71 = v74;
    *(_QWORD *)&v182 = v182 + 1;
    v75 = v183 + 1;
    if ( 4 * ((int)v183 + 1) >= 3 * v184 )
      goto LABEL_254;
    if ( v184 - HIDWORD(v183) - v75 <= v184 >> 3 )
    {
      sub_1392B70((__int64)&v182, v184);
      if ( !v184 )
      {
LABEL_287:
        LODWORD(v183) = v183 + 1;
        BUG();
      }
      v156 = 1;
      v157 = (v184 - 1) & (37 * v69);
      v75 = v183 + 1;
      v158 = 0;
      v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8LL * v157);
      v159 = *v71;
      if ( v69 != *v71 )
      {
        while ( v159 != -1 )
        {
          if ( v159 == -2 && !v158 )
            v158 = v71;
          v157 = (v184 - 1) & (v156 + v157);
          v71 = (_DWORD *)(*((_QWORD *)&v182 + 1) + 8LL * v157);
          v159 = *v71;
          if ( v69 == *v71 )
            goto LABEL_91;
          ++v156;
        }
        if ( v158 )
          v71 = v158;
      }
    }
LABEL_91:
    LODWORD(v183) = v75;
    if ( *v71 != -1 )
      --HIDWORD(v183);
    *v71 = v69;
    v67 += 2;
    v71[1] = v68;
  }
  while ( v176 != v67 );
LABEL_94:
  v56 = *(_QWORD *)(v11 + 248);
  v65 = *(_DWORD *)(v56 + 408);
  if ( v65 )
  {
LABEL_95:
    for ( j = v65 - 1; ; --j )
    {
      v77 = *(_QWORD *)(*(_QWORD *)(v56 + 400) + 8LL * j);
      v78 = *(_QWORD *)(v77 + 32);
      v79 = *(_BYTE *)v78 == 5
          ? (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v173 + 416LL))(v173, *(_QWORD *)(v11 + 256))
          : *(_DWORD *)(v78 + 8);
      if ( v79 > 0 )
        break;
      v108 = sub_1E69D00(*(_QWORD *)(v11 + 264), (unsigned int)v79);
      if ( v108 )
      {
        v109 = *(_QWORD *)(v108 + 24);
        if ( (*(_BYTE *)v108 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v108 + 46) & 8) != 0 )
            v108 = *(_QWORD *)(v108 + 8);
        }
        v82 = *(__int64 **)(v108 + 8);
        v80 = v109 + 16;
        v81 = v77;
        goto LABEL_100;
      }
LABEL_101:
      if ( v184 )
      {
        v85 = 1;
        v86 = (v184 - 1) & (37 * v79);
        v87 = (int *)(*((_QWORD *)&v182 + 1) + 8LL * v86);
        v88 = *v87;
        if ( v79 == *v87 )
        {
LABEL_103:
          if ( v87 != (int *)(*((_QWORD *)&v182 + 1) + 8LL * v184) )
          {
            v165 = sub_1E69D00(*(_QWORD *)(v11 + 264), (unsigned int)v87[1]);
            v89 = sub_1E16500(v77);
            v90 = sub_1E16510(v77);
            v91 = v165;
            v170 = v90;
            v181 = *(_QWORD *)(v77 + 64);
            if ( v181 )
            {
              sub_1623A60((__int64)&v181, v181, 2);
              v91 = v165;
            }
            v92 = 0;
            if ( **(_WORD **)(v77 + 16) == 12 )
            {
              v115 = *(_BYTE **)(v77 + 32);
              if ( !*v115 )
                v92 = v115[40] == 1;
            }
            if ( !v91 )
              BUG();
            if ( (*(_BYTE *)v91 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v91 + 46) & 8) != 0 )
                v91 = *(_QWORD *)(v91 + 8);
            }
            v166 = v92;
            sub_1E1C1F0(
              v178,
              *(_QWORD *)(v91 + 8),
              (unsigned int)&v181,
              *(_QWORD *)(*(_QWORD *)(v11 + 312) + 8LL) + 768,
              v92,
              v87[1],
              v89,
              v170);
            v93 = (unsigned int)v87[1];
            v94 = *(_QWORD *)(v11 + 264);
            if ( (int)v93 < 0 )
              v95 = *(_QWORD *)(*(_QWORD *)(v94 + 24) + 16 * (v93 & 0x7FFFFFFF) + 8);
            else
              v95 = *(_QWORD *)(*(_QWORD *)(v94 + 272) + 8 * v93);
            if ( v95 )
            {
              if ( (*(_BYTE *)(v95 + 3) & 0x10) != 0 )
              {
                while ( 1 )
                {
                  v95 = *(_QWORD *)(v95 + 32);
                  if ( !v95 )
                    break;
                  if ( (*(_BYTE *)(v95 + 3) & 0x10) == 0 )
                    goto LABEL_113;
                }
              }
              else
              {
LABEL_113:
                v96 = 0;
LABEL_114:
                v97 = *(_QWORD *)(v95 + 16);
                while ( 1 )
                {
                  v95 = *(_QWORD *)(v95 + 32);
                  if ( !v95 )
                    break;
                  if ( (*(_BYTE *)(v95 + 3) & 0x10) == 0 && v97 != *(_QWORD *)(v95 + 16) )
                  {
                    v98 = **(_WORD **)(v97 + 16);
                    if ( v98 == 12 )
                    {
                      v97 = v96;
LABEL_175:
                      v96 = v97;
                      goto LABEL_114;
                    }
                    goto LABEL_119;
                  }
                }
                v98 = **(_WORD **)(v97 + 16);
                if ( v98 == 12 )
                {
                  if ( !v96 )
                    goto LABEL_122;
                  goto LABEL_171;
                }
                v95 = 0;
LABEL_119:
                if ( v98 != 15 || v96 || v178 != *(_QWORD *)(v97 + 24) )
                  goto LABEL_122;
                if ( v95 )
                  goto LABEL_175;
                v96 = v97;
LABEL_171:
                v110 = v166;
                v167 = v96;
                sub_1E1C0A0(
                  *(_QWORD *)(v11 + 256),
                  (unsigned int)&v181,
                  *(_QWORD *)(*(_QWORD *)(v11 + 312) + 8LL) + 768,
                  v110,
                  *(_DWORD *)(*(_QWORD *)(v96 + 32) + 8LL),
                  v89,
                  v170);
                if ( v178 + 24 == (*(_QWORD *)(v178 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
                  v112 = *(unsigned __int64 **)(v178 + 32);
                else
                  v112 = *(unsigned __int64 **)(v167 + 8);
                v168 = v112;
                v171 = (unsigned __int64)v111;
                sub_1DD5BA0(v178 + 16, v111);
                v113 = *(_QWORD *)v171;
                v114 = *v168;
                *(_QWORD *)(v171 + 8) = v168;
                v114 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)v171 = v114 | v113 & 7;
                *(_QWORD *)(v114 + 8) = v171;
                *v168 = *v168 & 7 | v171;
              }
            }
LABEL_122:
            if ( v181 )
              sub_161E7C0((__int64)&v181, v181);
          }
        }
        else
        {
          while ( v88 != -1 )
          {
            v86 = (v184 - 1) & (v85 + v86);
            v87 = (int *)(*((_QWORD *)&v182 + 1) + 8LL * v86);
            v88 = *v87;
            if ( v79 == *v87 )
              goto LABEL_103;
            ++v85;
          }
        }
      }
      if ( !j )
        goto LABEL_130;
      v56 = *(_QWORD *)(v11 + 248);
    }
    v80 = v178 + 16;
    v81 = v77;
    v82 = *(__int64 **)(v178 + 32);
LABEL_100:
    sub_1DD5BA0(v80, v81);
    v83 = *v82;
    v84 = *(_QWORD *)v77;
    *(_QWORD *)(v77 + 8) = v82;
    v83 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v77 = v83 | v84 & 7;
    *(_QWORD *)(v83 + 8) = v77;
    *v82 = v77 | *v82 & 7;
    goto LABEL_101;
  }
LABEL_130:
  v99 = *(_QWORD *)(v11 + 256);
  v100 = *(_QWORD *)(v99 + 328);
  v101 = *(_QWORD *)(v99 + 56);
  v179 = v99 + 320;
  if ( v100 != v99 + 320 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v101 + 65) )
      {
        v99 = *(_QWORD *)(v11 + 256);
        if ( *(_BYTE *)(v99 + 345) )
          break;
      }
      v102 = *(_QWORD *)(v100 + 32);
      for ( k = v100 + 24; k != v102; v102 = *(_QWORD *)(v102 + 8) )
      {
        while ( 1 )
        {
          v104 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 312) + 8LL)
                           + ((unsigned __int64)**(unsigned __int16 **)(v102 + 16) << 6)
                           + 8);
          if ( (v104 & 0x10) != 0 && (v104 & 8) == 0 || (unsigned __int8)sub_1E16450(v102) )
            *(_BYTE *)(v101 + 65) = 1;
          if ( **(_WORD **)(v102 + 16) == 1 )
            *(_BYTE *)(*(_QWORD *)(v11 + 256) + 345LL) = 1;
          if ( (*(_BYTE *)v102 & 4) == 0 )
            break;
          v102 = *(_QWORD *)(v102 + 8);
          if ( k == v102 )
            goto LABEL_143;
        }
        while ( (*(_BYTE *)(v102 + 46) & 8) != 0 )
          v102 = *(_QWORD *)(v102 + 8);
      }
LABEL_143:
      v100 = *(_QWORD *)(v100 + 8);
      if ( v179 == v100 )
      {
        v99 = *(_QWORD *)(v11 + 256);
        break;
      }
    }
  }
  *(_BYTE *)(v99 + 344) = sub_15E3780(v174);
  v105 = *(_QWORD *)(v11 + 248);
  v106 = *(_QWORD *)(v11 + 256);
  if ( *(_DWORD *)(v105 + 520) )
  {
    v122 = *(unsigned int *)(v105 + 528);
    v123 = *(int **)(v105 + 512);
    v124 = &v123[2 * v122];
    v125 = *(_DWORD *)(v105 + 528);
    if ( v123 != v124 )
    {
      v126 = *(int **)(v105 + 512);
      while ( 1 )
      {
        v127 = *v126;
        v128 = v126;
        if ( (unsigned int)*v126 <= 0xFFFFFFFD )
          break;
        v126 += 2;
        if ( v124 == v126 )
          goto LABEL_146;
      }
      if ( v124 != v126 )
      {
        v180 = v11;
        v129 = *(_QWORD *)(v106 + 40);
        while ( 1 )
        {
          v130 = v128[1];
          v131 = &v123[2 * v122];
          v132 = v125 - 1;
          while ( 1 )
          {
            v133 = v131;
            if ( v125 )
            {
              v134 = v132 & (37 * v130);
              v133 = &v123[2 * v134];
              v135 = *v133;
              if ( *v133 != v130 )
              {
                v138 = 1;
                while ( v135 != -1 )
                {
                  v139 = v138 + 1;
                  v140 = v132 & (v134 + v138);
                  v134 = v140;
                  v133 = &v123[2 * v140];
                  v135 = *v133;
                  if ( *v133 == v130 )
                    goto LABEL_208;
                  v138 = v139;
                }
                v133 = v131;
              }
            }
LABEL_208:
            if ( v133 == v124 )
              break;
            v130 = v133[1];
          }
          if ( v127 < 0 )
          {
            if ( v130 < 0 )
            {
              sub_1E69410(
                v129,
                (unsigned int)v130,
                *(_QWORD *)(*(_QWORD *)(v129 + 24) + 16LL * (v127 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                0);
LABEL_211:
              v136 = *(_QWORD *)(*(_QWORD *)(v129 + 24) + 16LL * (v130 & 0x7FFFFFFF) + 8);
              goto LABEL_212;
            }
          }
          else if ( v130 < 0 )
          {
            goto LABEL_211;
          }
          v136 = *(_QWORD *)(*(_QWORD *)(v129 + 272) + 8LL * (unsigned int)v130);
LABEL_212:
          if ( v136 )
          {
            if ( (*(_BYTE *)(v136 + 3) & 0x10) != 0 )
            {
              while ( 1 )
              {
                v136 = *(_QWORD *)(v136 + 32);
                if ( !v136 )
                  break;
                if ( (*(_BYTE *)(v136 + 3) & 0x10) == 0 )
                  goto LABEL_214;
              }
            }
            else
            {
LABEL_214:
              sub_1E69E80(v129, (unsigned int)v127);
            }
          }
          v128 += 2;
          sub_1E69BA0(v129, (unsigned int)v127, (unsigned int)v130);
          if ( v128 == v124 )
            goto LABEL_218;
          while ( (unsigned int)*v128 > 0xFFFFFFFD )
          {
            v128 += 2;
            if ( v124 == v128 )
              goto LABEL_218;
          }
          if ( v128 == v124 )
          {
LABEL_218:
            v11 = v180;
            v106 = *(_QWORD *)(v180 + 256);
            break;
          }
          v127 = *v128;
          v137 = *(_QWORD *)(v180 + 248);
          v122 = *(unsigned int *)(v137 + 528);
          v123 = *(int **)(v137 + 512);
          v125 = *(_DWORD *)(v137 + 528);
        }
      }
    }
  }
LABEL_146:
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v11 + 320) + 968LL))(*(_QWORD *)(v11 + 320), v106);
  sub_1FDEF50(*(_QWORD *)(v11 + 248));
  j___libc_free_0(*((_QWORD *)&v182 + 1));
  if ( v163 != *(_DWORD *)(v11 + 304) )
  {
    *(_DWORD *)(v11 + 304) = v163;
    sub_1700730(*(_QWORD *)(v11 + 232), v163);
    *(_BYTE *)(*(_QWORD *)(v11 + 232) + 800LL) = (2 * v162) | *(_BYTE *)(*(_QWORD *)(v11 + 232) + 800LL) & 0xFD;
  }
  return 1;
}
