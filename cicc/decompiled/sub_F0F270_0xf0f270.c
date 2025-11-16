// Function: sub_F0F270
// Address: 0xf0f270
//
unsigned __int8 *__fastcall sub_F0F270(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  unsigned __int8 *v4; // r15
  __int64 v5; // r14
  _BYTE *v6; // rbx
  __int64 v7; // r8
  int *v8; // rdi
  int v9; // r9d
  int v10; // eax
  _DWORD *v11; // rdi
  _DWORD *v12; // rsi
  unsigned int v13; // edx
  _DWORD *v14; // rdi
  _DWORD *v15; // rsi
  unsigned __int8 *v16; // r13
  __int64 v17; // rbx
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // rax
  __int64 v29; // r15
  __int64 *v30; // rdi
  __int64 *v31; // rax
  _BYTE *v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rcx
  _BYTE *v36; // rdx
  _BYTE *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  _BYTE *v42; // rsi
  char v43; // bl
  _BYTE *v44; // rdi
  unsigned __int64 v45; // r10
  _DWORD *v46; // rcx
  __int64 v47; // rbx
  int v48; // r11d
  unsigned int *v49; // rsi
  __int64 v50; // r8
  __int64 v51; // rdi
  unsigned int *v52; // rax
  unsigned int v53; // r14d
  __int64 v54; // rax
  bool v55; // al
  __int64 v56; // rcx
  int v57; // r11d
  unsigned __int64 v58; // r10
  __int64 v59; // rax
  __int64 v60; // r8
  unsigned __int64 v61; // r10
  int v62; // r11d
  __int64 *v63; // rax
  __int64 *v64; // r13
  __int64 *v65; // rbx
  __int64 *v66; // rbx
  __int64 v67; // rax
  unsigned __int64 v68; // r10
  __int64 v69; // rax
  __int64 v70; // r10
  __int64 v71; // r14
  unsigned __int8 *v72; // rsi
  __int64 *v73; // rdi
  __int64 v74; // r11
  __int64 v75; // r10
  void *v76; // r14
  __int64 v77; // rbx
  __int64 v78; // rcx
  int v79; // eax
  __int64 v80; // rax
  __int64 v81; // rax
  unsigned __int8 *v82; // rax
  __int64 v83; // r15
  __int64 v84; // r10
  unsigned __int8 *v85; // rax
  unsigned __int8 *v86; // r10
  unsigned __int8 *v87; // rax
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 v90; // rax
  void *v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  int v94; // eax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r8
  __int64 v99; // rdx
  char v100; // al
  int v101; // eax
  unsigned __int8 *v102; // rax
  int v103; // eax
  __int64 v104; // r9
  int v105; // eax
  unsigned __int8 *v106; // rax
  __int64 v107; // rdx
  int v108; // eax
  __int64 v109; // rax
  __int64 *v110; // rax
  __int64 *v111; // rdx
  _QWORD *v112; // rax
  __int64 v113; // r13
  __int64 v114; // rbx
  __int64 v115; // rdx
  unsigned int v116; // esi
  unsigned __int64 v117; // rdx
  __int64 *v118; // rbx
  __int64 *v119; // rax
  __int64 v120; // [rsp-10h] [rbp-200h]
  unsigned __int8 *v121; // [rsp+8h] [rbp-1E8h]
  _BYTE *v122; // [rsp+18h] [rbp-1D8h]
  __int64 v123; // [rsp+20h] [rbp-1D0h]
  __int64 v124; // [rsp+28h] [rbp-1C8h]
  unsigned __int64 v125; // [rsp+30h] [rbp-1C0h]
  char *v126; // [rsp+30h] [rbp-1C0h]
  int *v127; // [rsp+38h] [rbp-1B8h]
  int v128; // [rsp+38h] [rbp-1B8h]
  unsigned __int64 v129; // [rsp+38h] [rbp-1B8h]
  unsigned __int64 v130; // [rsp+38h] [rbp-1B8h]
  __int64 v131; // [rsp+38h] [rbp-1B8h]
  __int64 v132; // [rsp+38h] [rbp-1B8h]
  __int64 v133; // [rsp+38h] [rbp-1B8h]
  __int64 v134; // [rsp+38h] [rbp-1B8h]
  int v135; // [rsp+38h] [rbp-1B8h]
  __int64 v136; // [rsp+40h] [rbp-1B0h]
  char v137; // [rsp+40h] [rbp-1B0h]
  char *v138; // [rsp+40h] [rbp-1B0h]
  __int64 v139; // [rsp+40h] [rbp-1B0h]
  __int64 v140; // [rsp+40h] [rbp-1B0h]
  __int64 v141; // [rsp+40h] [rbp-1B0h]
  int v142; // [rsp+48h] [rbp-1A8h]
  unsigned int v143; // [rsp+48h] [rbp-1A8h]
  char *v144; // [rsp+48h] [rbp-1A8h]
  int v145; // [rsp+48h] [rbp-1A8h]
  int v146; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v147; // [rsp+48h] [rbp-1A8h]
  __int64 v148; // [rsp+48h] [rbp-1A8h]
  __int64 v149; // [rsp+48h] [rbp-1A8h]
  __int64 v150; // [rsp+48h] [rbp-1A8h]
  __int64 v151; // [rsp+48h] [rbp-1A8h]
  __int64 v152; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v153; // [rsp+48h] [rbp-1A8h]
  __int64 v154; // [rsp+50h] [rbp-1A0h]
  __int64 v155; // [rsp+50h] [rbp-1A0h]
  __int64 v156; // [rsp+50h] [rbp-1A0h]
  __int64 v157; // [rsp+50h] [rbp-1A0h]
  const void *v158; // [rsp+50h] [rbp-1A0h]
  __int64 v159; // [rsp+50h] [rbp-1A0h]
  unsigned __int8 *v161; // [rsp+58h] [rbp-198h]
  __int64 v162; // [rsp+58h] [rbp-198h]
  int v163; // [rsp+6Ch] [rbp-184h] BYREF
  _QWORD v164[4]; // [rsp+70h] [rbp-180h] BYREF
  _QWORD v165[4]; // [rsp+90h] [rbp-160h] BYREF
  _QWORD v166[4]; // [rsp+B0h] [rbp-140h] BYREF
  __int16 v167; // [rsp+D0h] [rbp-120h]
  _BYTE *v168; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v169; // [rsp+E8h] [rbp-108h]
  _BYTE v170[16]; // [rsp+F0h] [rbp-100h] BYREF
  __int16 v171; // [rsp+100h] [rbp-F0h]
  __int64 *v172; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v173; // [rsp+138h] [rbp-B8h]
  __int64 v174; // [rsp+140h] [rbp-B0h] BYREF
  int v175; // [rsp+148h] [rbp-A8h]
  char v176; // [rsp+14Ch] [rbp-A4h]
  __int16 v177; // [rsp+150h] [rbp-A0h] BYREF

  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL) - 17 > 1 )
    return 0;
  v2 = *((_QWORD *)a2 - 8);
  v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v163 = *a2 - 29;
  if ( *(_BYTE *)v2 == 92 )
  {
    v74 = *(_QWORD *)(v2 - 64);
    if ( v74 )
    {
      v75 = *(_QWORD *)(v2 - 32);
      if ( v75 )
      {
        v76 = *(void **)(v2 + 72);
        v77 = *(unsigned int *)(v2 + 80);
        if ( *v4 == 92 )
        {
          v78 = *((_QWORD *)v4 - 8);
          if ( v78 )
          {
            v157 = *((_QWORD *)v4 - 4);
            if ( v157 )
            {
              if ( v77 == *((_DWORD *)v4 + 20) )
              {
                if ( !(4 * v77)
                  || (v132 = *((_QWORD *)v4 - 8),
                      v139 = *(_QWORD *)(v2 - 32),
                      v149 = *(_QWORD *)(v2 - 64),
                      v79 = memcmp(v76, *((const void **)v4 + 9), 4 * v77),
                      v74 = v149,
                      v75 = v139,
                      v78 = v132,
                      !v79) )
                {
                  v80 = *(_QWORD *)(v2 + 16);
                  if ( v80 )
                  {
                    if ( !*(_QWORD *)(v80 + 8) )
                    {
                      v81 = *((_QWORD *)v4 + 2);
                      if ( v81 )
                      {
                        if ( !*(_QWORD *)(v81 + 8) )
                        {
                          v133 = v78;
                          v140 = v75;
                          v150 = v74;
                          if ( (unsigned __int8)sub_B4F610(v2) )
                          {
                            if ( (unsigned __int8)sub_B4F610((__int64)v4) )
                            {
                              v177 = 257;
                              v82 = (unsigned __int8 *)sub_F0A990(
                                                         *(__int64 **)(a1 + 32),
                                                         v163,
                                                         v150,
                                                         v133,
                                                         (int)v168,
                                                         0,
                                                         (__int64)&v172,
                                                         0);
                              v83 = (__int64)v82;
                              v84 = v140;
                              if ( (unsigned __int8)(*v82 - 42) <= 0x11u )
                              {
                                sub_B45260(v82, (__int64)a2, 1);
                                v84 = v140;
                              }
                              v177 = 257;
                              v85 = (unsigned __int8 *)sub_F0A990(
                                                         *(__int64 **)(a1 + 32),
                                                         v163,
                                                         v84,
                                                         v157,
                                                         (int)v168,
                                                         0,
                                                         (__int64)&v172,
                                                         0);
                              v86 = v85;
                              if ( (unsigned __int8)(*v85 - 42) <= 0x11u )
                              {
                                v161 = v85;
                                sub_B45260(v85, (__int64)a2, 1);
                                v86 = v161;
                              }
                              v162 = (__int64)v86;
                              v177 = 257;
                              v87 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
                              v16 = v87;
                              if ( v87 )
                                sub_B4E9E0((__int64)v87, v83, v162, v76, v77, (__int64)&v172, 0, 0);
                              return v16;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v164[2] = a2;
  v164[1] = &v163;
  v164[0] = a1;
  if ( *(_BYTE *)v2 == 85
    && (v88 = *(_QWORD *)(v2 - 32)) != 0
    && !*(_BYTE *)v88
    && *(_QWORD *)(v88 + 24) == *(_QWORD *)(v2 + 80)
    && *(_DWORD *)(v88 + 36) == 402
    && (v89 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))) != 0 )
  {
    v90 = *(_QWORD *)(v2 + 16);
    if ( *v4 == 85 )
    {
      v97 = *((_QWORD *)v4 - 4);
      if ( v97 )
      {
        if ( !*(_BYTE *)v97 && *(_QWORD *)(v97 + 24) == *((_QWORD *)v4 + 10) && *(_DWORD *)(v97 + 36) == 402 )
        {
          v98 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
          if ( v98 )
          {
            if ( v90 && !*(_QWORD *)(v90 + 8) )
              return (unsigned __int8 *)sub_F0AC30(v164, v89, v98);
            v99 = *((_QWORD *)v4 + 2);
            if ( v99 )
            {
              if ( !*(_QWORD *)(v99 + 8) )
                return (unsigned __int8 *)sub_F0AC30(v164, v89, v98);
            }
            if ( (unsigned __int8 *)v2 == v4 )
            {
              v159 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
              v100 = sub_BD3610(v2, 2);
              v98 = v159;
              if ( v100 )
                return (unsigned __int8 *)sub_F0AC30(v164, v89, v98);
              v90 = *(_QWORD *)(v2 + 16);
            }
          }
        }
      }
    }
    if ( v90 && !*(_QWORD *)(v90 + 8) && sub_9B7DA0((char *)v4, 0xFFFFFFFF, 0) )
      return (unsigned __int8 *)sub_F0AC30(v164, v89, (__int64)v4);
  }
  else if ( sub_9B7DA0((char *)v2, 0xFFFFFFFF, 0) )
  {
    v38 = *((_QWORD *)v4 + 2);
    if ( v38 )
    {
      if ( !*(_QWORD *)(v38 + 8) && *v4 == 85 )
      {
        v39 = *((_QWORD *)v4 - 4);
        if ( v39 )
        {
          if ( !*(_BYTE *)v39 && *(_QWORD *)(v39 + 24) == *((_QWORD *)v4 + 10) && *(_DWORD *)(v39 + 36) == 402 )
          {
            v40 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
            if ( v40 )
              return (unsigned __int8 *)sub_F0AC30(v164, v2, v40);
          }
        }
      }
    }
  }
  if ( !sub_991A70(a2, 0, 0, 0, 0, 0, 0) )
    return 0;
  v165[1] = &v163;
  v165[2] = a2;
  v165[0] = a1;
  if ( *(_BYTE *)v2 == 92 )
  {
    v5 = *(_QWORD *)(v2 - 64);
    if ( v5 )
    {
      v6 = *(_BYTE **)(v2 - 32);
      if ( *v6 == 13 )
      {
        v91 = *(void **)(v2 + 72);
        v92 = *(unsigned int *)(v2 + 80);
        if ( *v4 == 92 )
        {
          v93 = *((_QWORD *)v4 - 8);
          if ( v93 )
          {
            if ( **((_BYTE **)v4 - 4) == 13 && v92 == *((_DWORD *)v4 + 20) )
            {
              if ( !(4 * v92)
                || (v141 = *((_QWORD *)v4 - 8),
                    v151 = *(unsigned int *)(v2 + 80),
                    v158 = *(const void **)(v2 + 72),
                    v94 = memcmp(v158, *((const void **)v4 + 9), 4 * v92),
                    v91 = (void *)v158,
                    v92 = v151,
                    v93 = v141,
                    !v94) )
              {
                if ( *(_QWORD *)(v93 + 8) == *(_QWORD *)(v5 + 8) )
                {
                  v95 = *(_QWORD *)(v2 + 16);
                  if ( v95 )
                  {
                    if ( !*(_QWORD *)(v95 + 8) )
                      return (unsigned __int8 *)sub_F0AAA0(v165, v5, v93, v91, v92);
                  }
                  v96 = *((_QWORD *)v4 + 2);
                  if ( v96 )
                  {
                    if ( !*(_QWORD *)(v96 + 8) )
                      return (unsigned __int8 *)sub_F0AAA0(v165, v5, v93, v91, v92);
                  }
                  if ( (unsigned __int8 *)v2 == v4 )
                    return (unsigned __int8 *)sub_F0AAA0(v165, v5, v93, v91, v92);
                }
              }
            }
          }
        }
      }
      if ( sub_B46D50(a2) )
      {
        if ( v6 )
        {
          v7 = *(unsigned int *)(v2 + 80);
          v8 = *(int **)(v2 + 72);
          v9 = *(_DWORD *)(v2 + 80);
          if ( *v4 == 92 && v6 == *((_BYTE **)v4 - 8) && v5 == *((_QWORD *)v4 - 4) && v7 == *((_DWORD *)v4 + 20) )
          {
            if ( !(4 * v7)
              || (v136 = *(unsigned int *)(v2 + 80),
                  v142 = *(_DWORD *)(v2 + 80),
                  v10 = memcmp(v8, *((const void **)v4 + 9), 4 * v7),
                  v9 = v142,
                  v7 = v136,
                  !v10) )
            {
              if ( v9 == *(_DWORD *)(*(_QWORD *)(v5 + 8) + 32LL) )
              {
                if ( (unsigned __int8)sub_B4EEA0(v8, v7, v9) )
                {
                  v11 = *(_DWORD **)(v2 + 72);
                  v12 = &v11[*(unsigned int *)(v2 + 80)];
                  if ( v12 == sub_F06910(v11, (__int64)v12, dword_3F89984) )
                  {
                    v13 = *((_DWORD *)v4 + 20);
                    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v4 - 8) + 8LL) + 32LL) == v13 )
                    {
                      if ( (unsigned __int8)sub_B4EEA0(*((int **)v4 + 9), v13, v13) )
                      {
                        v14 = (_DWORD *)*((_QWORD *)v4 + 9);
                        v15 = &v14[*((unsigned int *)v4 + 20)];
                        if ( v15 == sub_F06910(v14, (__int64)v15, dword_3F89984) )
                        {
                          v177 = 257;
                          v16 = (unsigned __int8 *)sub_B504D0(v163, v5, (__int64)v6, (__int64)&v172, 0, 0);
                          sub_B45260(v16, (__int64)a2, 1);
                          return v16;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v17 = *((_QWORD *)a2 + 1);
  if ( *(_BYTE *)(v17 + 8) != 17 )
    goto LABEL_25;
  v19 = *((_QWORD *)a2 - 8);
  v20 = *((_QWORD *)a2 - 4);
  v21 = *(_QWORD *)(v19 + 16);
  if ( v21 )
  {
    if ( !*(_QWORD *)(v21 + 8) && *(_BYTE *)v19 == 92 )
    {
      v123 = *(_QWORD *)(v19 - 64);
      if ( v123 )
      {
        if ( **(_BYTE **)(v19 - 32) == 13 )
        {
          v124 = *(unsigned int *)(v19 + 80);
          v127 = *(int **)(v19 + 72);
          if ( *(_BYTE *)v20 <= 0x15u && *(_BYTE *)v20 != 5 )
          {
            if ( !(unsigned __int8)sub_AD6CA0(v20) )
              goto LABEL_38;
            v20 = *((_QWORD *)a2 - 4);
          }
        }
      }
    }
  }
  v22 = *(_QWORD *)(v20 + 16);
  if ( !v22 )
    goto LABEL_25;
  if ( *(_QWORD *)(v22 + 8) )
    goto LABEL_25;
  if ( *(_BYTE *)v20 != 92 )
    goto LABEL_25;
  v123 = *(_QWORD *)(v20 - 64);
  if ( !v123 )
    goto LABEL_25;
  if ( **(_BYTE **)(v20 - 32) != 13 )
    goto LABEL_25;
  v127 = *(int **)(v20 + 72);
  v23 = *(unsigned int *)(v20 + 80);
  v20 = *((_QWORD *)a2 - 8);
  v124 = v23;
  if ( *(_BYTE *)v20 > 0x15u || *(_BYTE *)v20 == 5 || (unsigned __int8)sub_AD6CA0(v20) )
    goto LABEL_25;
LABEL_38:
  v143 = *(_DWORD *)(*(_QWORD *)(v123 + 8) + 32LL);
  if ( *(_DWORD *)(v17 + 32) < v143 )
    goto LABEL_25;
  v24 = *(_QWORD *)(v20 + 8);
  v137 = *v4;
  if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
    v24 = **(_QWORD **)(v24 + 16);
  v25 = sub_ACADE0((__int64 **)v24);
  v172 = &v174;
  v122 = (_BYTE *)v25;
  v173 = 0x1000000000LL;
  if ( v143 > 0x10 )
  {
    sub_C8D5F0((__int64)&v172, &v174, v143, 8u, v26, v27);
    v110 = v172;
    v111 = &v172[v143];
    do
      *v110++ = (__int64)v122;
    while ( v111 != v110 );
  }
  else if ( v143 )
  {
    v28 = &v174;
    do
      *v28++ = (__int64)v122;
    while ( &v174 + v143 != v28 );
  }
  LODWORD(v173) = v143;
  if ( !*(_DWORD *)(v17 + 32) )
  {
LABEL_193:
    v104 = sub_AD3730(v172, (unsigned int)v173);
    v105 = *a2;
    if ( (unsigned int)(v105 - 48) > 1 && (unsigned __int8)(*a2 - 51) > 1u )
    {
      if ( (unsigned int)(v105 - 54) > 2 )
      {
LABEL_196:
        if ( (unsigned __int8)v137 <= 0x15u )
          goto LABEL_197;
        goto LABEL_218;
      }
      if ( (unsigned __int8)v137 > 0x15u )
      {
LABEL_218:
        v109 = v104;
        v104 = v123;
        v123 = v109;
LABEL_197:
        v72 = (unsigned __int8 *)v123;
        v106 = (unsigned __int8 *)sub_F0AAA0(v165, v123, v104, v127, v124);
        v73 = v172;
        v16 = v106;
        if ( v172 != &v174 )
          goto LABEL_119;
        return v16;
      }
    }
    v104 = sub_F0BC20(v163, v104, (unsigned __int8)v137 <= 0x15u);
    goto LABEL_196;
  }
  v121 = v4;
  v29 = 0;
  v154 = *(unsigned int *)(v17 + 32);
  while ( 1 )
  {
    v33 = sub_AD69F0((unsigned __int8 *)v20, (unsigned int)v29);
    v34 = v127[v29];
    if ( (int)v34 >= 0 )
      break;
LABEL_54:
    v35 = *(_QWORD *)(a1 + 88);
    if ( (unsigned __int8)v137 > 0x15u )
    {
      v37 = (_BYTE *)sub_96E6C0(v163, v33, v122, v35);
    }
    else
    {
      v36 = (_BYTE *)v33;
      v33 = (__int64)v122;
      v37 = (_BYTE *)sub_96E6C0(v163, (__int64)v122, v36, v35);
    }
    if ( !v37 || *v37 != 13 )
    {
      v4 = v121;
      v30 = v172;
      goto LABEL_59;
    }
LABEL_52:
    if ( v154 == ++v29 )
      goto LABEL_193;
  }
  v30 = v172;
  v31 = &v172[v34];
  v32 = (_BYTE *)*v31;
  if ( v33 && (*v32 == 13 || v32 == (_BYTE *)v33) && v143 > (unsigned int)v29 )
  {
    *v31 = v33;
    if ( v127[v29] >= 0 )
      goto LABEL_52;
    goto LABEL_54;
  }
  v4 = v121;
LABEL_59:
  if ( v30 != &v174 )
    _libc_free(v30, v33);
LABEL_25:
  if ( !sub_B46CC0(a2) || !sub_B46D50(a2) )
    return 0;
  if ( *v4 == 92 )
  {
    v102 = (unsigned __int8 *)v2;
    v2 = (__int64)v4;
    v4 = v102;
  }
  v41 = *(_QWORD *)(v2 + 16);
  if ( !v41 )
    return 0;
  if ( *(_QWORD *)(v41 + 8) )
    return 0;
  if ( *(_BYTE *)v2 != 92 )
    return 0;
  v155 = *(_QWORD *)(v2 - 64);
  if ( !v155 )
    return 0;
  v42 = *(_BYTE **)(v2 - 32);
  if ( (unsigned __int8)(*v42 - 12) > 1u )
  {
    if ( (unsigned __int8)(*v42 - 9) > 2u )
      return 0;
    v172 = 0;
    v168 = v170;
    v173 = (__int64)&v177;
    v169 = 0x800000000LL;
    v174 = 8;
    v175 = 0;
    v176 = 1;
    v166[0] = &v172;
    v166[1] = &v168;
    v43 = sub_AA8FD0(v166, (__int64)v42);
    if ( v43 )
    {
      while ( 1 )
      {
        v44 = v168;
        if ( !(_DWORD)v169 )
          break;
        v42 = *(_BYTE **)&v168[8 * (unsigned int)v169 - 8];
        LODWORD(v169) = v169 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v166, (__int64)v42) )
          goto LABEL_247;
      }
    }
    else
    {
LABEL_247:
      v44 = v168;
      v43 = 0;
    }
    if ( v44 != v170 )
      _libc_free(v44, v42);
    if ( !v176 )
      _libc_free(v173, v42);
    if ( !v43 )
      return 0;
  }
  v45 = *(unsigned int *)(v2 + 80);
  v46 = *(_DWORD **)(v2 + 72);
  v47 = 4 * v45;
  v48 = *(_DWORD *)(v2 + 80);
  v49 = &v46[v45];
  v50 = (__int64)(4 * v45) >> 2;
  v51 = (__int64)(4 * v45) >> 4;
  if ( !v51 )
  {
    v107 = (__int64)(4 * v45) >> 2;
    v52 = *(unsigned int **)(v2 + 72);
LABEL_201:
    if ( v107 != 2 )
    {
      if ( v107 != 3 )
      {
        if ( v107 != 1 )
          return 0;
        goto LABEL_204;
      }
      if ( *v52 != -1 )
        goto LABEL_205;
      ++v52;
    }
    if ( *v52 != -1 )
      goto LABEL_205;
    ++v52;
LABEL_204:
    if ( *v52 == -1 )
      return 0;
LABEL_205:
    if ( v49 == v52 )
      return 0;
    v53 = *v52;
    if ( v51 )
      goto LABEL_96;
LABEL_207:
    if ( v50 != 2 )
    {
      if ( v50 != 3 )
      {
        if ( v50 != 1 )
          goto LABEL_99;
        goto LABEL_210;
      }
      if ( *v46 != -1 && v53 != *v46 )
        goto LABEL_98;
      ++v46;
    }
    if ( *v46 != -1 && v53 != *v46 )
      goto LABEL_98;
    ++v46;
LABEL_210:
    if ( v53 != *v46 && *v46 != -1 )
      goto LABEL_98;
    goto LABEL_99;
  }
  v52 = *(unsigned int **)(v2 + 72);
  while ( *v52 == -1 )
  {
    if ( v52[1] != -1 )
    {
      ++v52;
      break;
    }
    if ( v52[2] != -1 )
    {
      v52 += 2;
      break;
    }
    if ( v52[3] != -1 )
    {
      v52 += 3;
      break;
    }
    v52 += 4;
    if ( &v46[4 * v51] == v52 )
    {
      v107 = v49 - v52;
      goto LABEL_201;
    }
  }
  if ( v52 == v49 )
    return 0;
  v53 = *v52;
LABEL_96:
  while ( *v46 == -1 || v53 == *v46 )
  {
    v101 = v46[1];
    if ( v53 != v101 && v101 != -1 )
    {
      ++v46;
      break;
    }
    v103 = v46[2];
    if ( v103 != -1 && v53 != v103 )
    {
      v46 += 2;
      break;
    }
    v108 = v46[3];
    if ( v53 != v108 && v108 != -1 )
    {
      v46 += 3;
      break;
    }
    v46 += 4;
    if ( !--v51 )
    {
      v50 = v49 - v46;
      goto LABEL_207;
    }
  }
LABEL_98:
  if ( v49 != v46 )
    return 0;
LABEL_99:
  if ( *((_QWORD *)a2 + 1) != *(_QWORD *)(v155 + 8) )
    return 0;
  v54 = *((_QWORD *)v4 + 2);
  if ( !v54 )
    return 0;
  v16 = *(unsigned __int8 **)(v54 + 8);
  if ( v16 )
    return 0;
  if ( *v4 != v163 + 29 )
    return 0;
  v125 = v45;
  v128 = v48;
  v138 = (char *)*((_QWORD *)v4 - 8);
  if ( !v138 || !*((_QWORD *)v4 - 4) )
    return 0;
  v144 = (char *)*((_QWORD *)v4 - 4);
  v55 = sub_9B7DA0(v144, v53, 0);
  v56 = (__int64)v144;
  v57 = v128;
  v58 = v125;
  if ( v55 )
    goto LABEL_108;
  v126 = v144;
  v129 = v58;
  v145 = v57;
  if ( sub_9B7DA0(v138, v53, 0) )
  {
    v58 = v129;
    v57 = v145;
    v56 = (__int64)v138;
    v138 = v126;
LABEL_108:
    v177 = 257;
    v130 = v58;
    v146 = v57;
    v59 = sub_F0A990(*(__int64 **)(a1 + 32), v163, v155, v56, (int)v168, 0, (__int64)&v172, 0);
    v61 = v130;
    v156 = v59;
    v62 = v146;
    v172 = &v174;
    v173 = 0x800000000LL;
    if ( v130 > 8 )
    {
      v117 = v130;
      v135 = v146;
      v153 = v61;
      sub_C8D5F0((__int64)&v172, &v174, v117, 4u, v60, v120);
      v64 = v172;
      v61 = v153;
      v62 = v135;
      v118 = (__int64 *)((char *)v172 + v47);
      v119 = v172;
      if ( v172 != v118 )
      {
        do
        {
          *(_DWORD *)v119 = v53;
          v119 = (__int64 *)((char *)v119 + 4);
        }
        while ( v118 != v119 );
LABEL_112:
        v64 = v172;
      }
    }
    else
    {
      v63 = &v174;
      v64 = &v174;
      if ( v130 )
      {
        v65 = (__int64 *)((char *)&v174 + v47);
        if ( v65 != &v174 )
        {
          do
          {
            *(_DWORD *)v63 = v53;
            v63 = (__int64 *)((char *)v63 + 4);
          }
          while ( v65 != v63 );
          goto LABEL_112;
        }
      }
    }
    LODWORD(v173) = v62;
    v167 = 257;
    v66 = *(__int64 **)(a1 + 32);
    v147 = v61;
    v67 = sub_ACADE0(*(__int64 ***)(v156 + 8));
    v68 = v147;
    v148 = v67;
    v131 = v68;
    v69 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *, unsigned __int64))(*(_QWORD *)v66[10] + 112LL))(
            v66[10],
            v156,
            v67,
            v64,
            v68);
    v70 = v131;
    v71 = v69;
    if ( !v69 )
    {
      v134 = v148;
      v171 = 257;
      v152 = v70;
      v112 = sub_BD2C40(112, unk_3F1FE60);
      v71 = (__int64)v112;
      if ( v112 )
        sub_B4E9E0((__int64)v112, v156, v134, v64, v152, (__int64)&v168, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v66[11] + 16LL))(
        v66[11],
        v71,
        v166,
        v66[7],
        v66[8]);
      v113 = *v66;
      v114 = *v66 + 16LL * *((unsigned int *)v66 + 2);
      while ( v114 != v113 )
      {
        v115 = *(_QWORD *)(v113 + 8);
        v116 = *(_DWORD *)v113;
        v113 += 16;
        sub_B99FD0(v71, v116, v115);
      }
    }
    v72 = (unsigned __int8 *)v71;
    v171 = 257;
    v16 = (unsigned __int8 *)sub_B504D0(v163, v71, (__int64)v138, (__int64)&v168, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v16) )
    {
      sub_B45230((__int64)v16, (__int64)a2);
      v72 = v4;
      sub_B45560(v16, (unsigned __int64)v4);
    }
    if ( (unsigned __int8)(*(_BYTE *)v156 - 42) <= 0x11u )
    {
      v72 = v16;
      sub_B45260((unsigned __int8 *)v156, (__int64)v16, 1);
    }
    v73 = v172;
    if ( v172 != &v174 )
LABEL_119:
      _libc_free(v73, v72);
  }
  return v16;
}
