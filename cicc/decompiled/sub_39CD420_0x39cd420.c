// Function: sub_39CD420
// Address: 0x39cd420
//
__int64 __fastcall sub_39CD420(__int64 a1, __int64 a2, __int64 a3, bool *a4, unsigned __int64 *a5, unsigned __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rdi
  int v10; // edx
  unsigned int v12; // eax
  int v13; // esi
  __int64 *v14; // rbx
  __int64 v15; // rcx
  const __m128i *v16; // rdi
  __m128i *v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  unsigned int v23; // r13d
  __int64 v24; // rax
  unsigned __int64 v25; // r15
  __int64 v26; // r14
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // r10
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 *v39; // rax
  unsigned __int64 *v40; // rax
  unsigned __int64 *v41; // rax
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // r12
  int v44; // esi
  unsigned __int64 *v45; // rdi
  unsigned int v46; // edx
  unsigned __int64 *v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rbx
  unsigned int v50; // eax
  unsigned __int64 v51; // rbx
  unsigned __int64 v52; // r13
  unsigned int v53; // esi
  unsigned int v54; // eax
  unsigned int v55; // edx
  const void *v56; // r13
  __int64 v57; // r12
  __int64 *v58; // r15
  int v59; // r8d
  int v60; // r9d
  __int64 v61; // r10
  __int64 v62; // rax
  bool v63; // zf
  __int64 v64; // rdx
  __int64 *v65; // r12
  __int64 *v66; // rbx
  __int64 v67; // rsi
  __int64 v68; // r12
  unsigned __int64 *v70; // r8
  int v71; // edi
  unsigned int v72; // ecx
  unsigned __int64 v73; // r9
  __int64 v74; // rcx
  __int64 v75; // rsi
  unsigned __int64 v76; // rdx
  int v77; // r8d
  int v78; // r9d
  __int64 v79; // rax
  int v80; // r8d
  int v81; // r9d
  __int64 v82; // rax
  __int64 v83; // rbx
  __int64 v84; // rdx
  int v85; // r8d
  unsigned __int64 **v86; // r9
  __int64 v87; // rax
  unsigned __int64 **v88; // r13
  unsigned __int64 **v89; // r14
  int v90; // esi
  unsigned __int64 *v91; // r8
  unsigned int v92; // edx
  unsigned __int64 *v93; // rax
  unsigned __int64 v94; // rbx
  __int64 v95; // rax
  unsigned __int64 *v96; // rbx
  unsigned int v97; // esi
  unsigned int v98; // eax
  unsigned __int64 *v99; // rdi
  unsigned int v100; // edx
  int v101; // r10d
  int v102; // ecx
  unsigned __int64 *v103; // rsi
  unsigned int v104; // edx
  int v105; // r10d
  unsigned __int64 *v106; // rax
  unsigned __int64 *v107; // rsi
  int v108; // ecx
  unsigned int v109; // edx
  int v110; // r10d
  int v111; // r10d
  unsigned __int64 *v112; // rsi
  int v113; // ecx
  unsigned int v114; // edx
  unsigned __int64 v115; // rdi
  unsigned __int64 *v116; // rax
  int v117; // r10d
  __int64 v118; // r13
  __int64 v119; // rdx
  __int64 v120; // rbx
  __int64 v121; // rdx
  unsigned __int64 v122; // r14
  unsigned __int64 v123; // rdx
  int v124; // ecx
  unsigned int v125; // esi
  __int64 v126; // r11
  __int64 v127; // r9
  unsigned int v128; // ebx
  __int64 v129; // rdi
  __int64 v130; // rcx
  __int64 v131; // r8
  __int64 *v132; // r13
  __int64 *v133; // rbx
  __int64 v134; // r8
  int v135; // r9d
  __int64 v136; // rax
  unsigned __int64 *v137; // rsi
  int v138; // ecx
  unsigned int v139; // edx
  unsigned __int64 v140; // rdi
  unsigned __int64 *v141; // r11
  size_t v142; // rdx
  int v143; // r13d
  __int64 *v144; // rax
  int v145; // ecx
  int v146; // ebx
  int v147; // ebx
  __int64 v148; // r11
  unsigned int v149; // esi
  __int64 v150; // r9
  int v151; // r8d
  __int64 *v152; // rdi
  int v153; // r11d
  int v154; // r11d
  __int64 v155; // r10
  __int64 *v156; // rsi
  unsigned int v157; // ebx
  int v158; // edi
  __int64 v159; // r8
  int v160; // r10d
  __int64 v162; // [rsp+48h] [rbp-2C8h]
  __int64 v163; // [rsp+48h] [rbp-2C8h]
  const void *v165; // [rsp+58h] [rbp-2B8h]
  __int64 v166; // [rsp+58h] [rbp-2B8h]
  __int64 *v168; // [rsp+68h] [rbp-2A8h]
  __int64 v169; // [rsp+68h] [rbp-2A8h]
  __int64 v170; // [rsp+68h] [rbp-2A8h]
  __int64 v171; // [rsp+70h] [rbp-2A0h] BYREF
  unsigned __int64 v172; // [rsp+78h] [rbp-298h] BYREF
  unsigned __int64 **v173; // [rsp+80h] [rbp-290h] BYREF
  __int64 v174; // [rsp+88h] [rbp-288h]
  _BYTE v175[16]; // [rsp+90h] [rbp-280h] BYREF
  _BYTE v176[48]; // [rsp+A0h] [rbp-270h] BYREF
  __int64 *v177; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v178; // [rsp+D8h] [rbp-238h]
  _BYTE v179[64]; // [rsp+E0h] [rbp-230h] BYREF
  _BYTE *v180; // [rsp+120h] [rbp-1F0h] BYREF
  __int64 v181; // [rsp+128h] [rbp-1E8h]
  _BYTE v182[64]; // [rsp+130h] [rbp-1E0h] BYREF
  __int64 v183; // [rsp+170h] [rbp-1A0h] BYREF
  __int64 v184; // [rsp+178h] [rbp-198h]
  unsigned __int64 *v185; // [rsp+180h] [rbp-190h] BYREF
  unsigned int v186; // [rsp+188h] [rbp-188h]
  __int64 v187; // [rsp+1C0h] [rbp-150h] BYREF
  __int64 v188; // [rsp+1C8h] [rbp-148h]
  unsigned __int64 *v189; // [rsp+1D0h] [rbp-140h] BYREF
  int v190; // [rsp+1D8h] [rbp-138h]
  __int64 v191; // [rsp+210h] [rbp-100h] BYREF
  __int64 v192; // [rsp+218h] [rbp-F8h]
  unsigned __int64 v193; // [rsp+220h] [rbp-F0h] BYREF
  unsigned __int64 v194[22]; // [rsp+260h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a1 + 208);
  v8 = *(_DWORD *)(v7 + 288);
  v171 = 0;
  if ( !v8 )
  {
LABEL_30:
    memset(v194, 0, 0x80u);
    HIDWORD(v194[7]) = 8;
    v194[3] = (unsigned __int64)&v194[1];
    v194[4] = (unsigned __int64)&v194[1];
    v194[6] = (unsigned __int64)&v194[8];
    goto LABEL_31;
  }
  v9 = *(_QWORD *)(v7 + 272);
  v10 = v8 - 1;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = 1;
  v14 = (__int64 *)(v9 + 136LL * v12);
  v15 = *v14;
  if ( a2 != *v14 )
  {
    while ( v15 != -8 )
    {
      LODWORD(a5) = v13 + 1;
      v12 = v10 & (v13 + v12);
      v14 = (__int64 *)(v9 + 136LL * v12);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_3;
      ++v13;
    }
    goto LABEL_30;
  }
LABEL_3:
  v194[5] = 0;
  v194[3] = (unsigned __int64)&v194[1];
  v194[4] = (unsigned __int64)&v194[1];
  v16 = (const __m128i *)v14[3];
  LODWORD(v194[1]) = 0;
  v194[2] = 0;
  if ( v16 )
  {
    v17 = sub_39C76F0(v16, (__int64)&v194[1]);
    v18 = (unsigned __int64)v17;
    do
    {
      v19 = (unsigned __int64)v17;
      v17 = (__m128i *)v17[1].m128i_i64[0];
    }
    while ( v17 );
    v194[3] = v19;
    v20 = v18;
    do
    {
      v21 = v20;
      v20 = *(_QWORD *)(v20 + 24);
    }
    while ( v20 );
    v22 = v14[6];
    v194[4] = v21;
    v194[2] = v18;
    v194[5] = v22;
  }
  v23 = *((_DWORD *)v14 + 16);
  v194[6] = (unsigned __int64)&v194[8];
  v194[7] = 0x800000000LL;
  if ( v23 && &v194[6] != (unsigned __int64 *)(v14 + 7) )
  {
    a5 = &v194[8];
    v142 = 8LL * v23;
    if ( v23 <= 8
      || (sub_16CD150((__int64)&v194[6], &v194[8], v23, 8, (int)&v194[8], v23),
          a5 = (unsigned __int64 *)v194[6],
          (v142 = 8LL * *((unsigned int *)v14 + 16)) != 0) )
    {
      memcpy(a5, (const void *)v14[7], v142);
    }
    LODWORD(v194[7]) = v23;
  }
  if ( (unsigned __int64 *)v194[3] == &v194[1] )
  {
    v31 = v194[5];
  }
  else
  {
    v24 = a3;
    v165 = (const void *)(a3 + 16);
    v25 = v194[3];
    v26 = v24;
    do
    {
      v29 = sub_39CAC10((__int64 *)a1, *(_QWORD *)(v25 + 40), a2, &v171);
      v30 = *(unsigned int *)(v26 + 8);
      if ( (unsigned int)v30 >= *(_DWORD *)(v26 + 12) )
      {
        v163 = v29;
        sub_16CD150(v26, v165, 0, 8, v27, v28);
        v30 = *(unsigned int *)(v26 + 8);
        v29 = v163;
      }
      *(_QWORD *)(*(_QWORD *)v26 + 8 * v30) = v29;
      ++*(_DWORD *)(v26 + 8);
      v25 = sub_220EEE0(v25);
    }
    while ( (unsigned __int64 *)v25 != &v194[1] );
    v31 = v194[5];
    a3 = v26;
  }
  if ( v31 )
  {
    v32 = sub_15AB1E0(*(_BYTE **)(a2 + 8));
    v33 = *(_QWORD *)(v32 + 8 * (4LL - *(unsigned int *)(v32 + 8)));
    v34 = *(_QWORD *)(v33 + 8 * (3LL - *(unsigned int *)(v33 + 8)));
    if ( v34 )
    {
      v35 = *(unsigned int *)(v34 + 8);
      if ( (unsigned int)v35 > 1
        && !*(_QWORD *)(v34 + 8 * ((unsigned int)(v35 - 1) - v35))
        && !sub_39C84F0((_QWORD *)a1) )
      {
        v36 = sub_145CDC0(0x30u, (__int64 *)(a1 + 88));
        v37 = v36;
        if ( v36 )
        {
          *(_QWORD *)(v36 + 8) = 0;
          *(_QWORD *)v36 = v36 | 4;
          *(_QWORD *)(v36 + 16) = 0;
          *(_DWORD *)(v36 + 24) = -1;
          *(_WORD *)(v36 + 28) = 24;
          *(_BYTE *)(v36 + 30) = 0;
          *(_QWORD *)(v36 + 32) = 0;
          *(_QWORD *)(v36 + 40) = 0;
        }
        v38 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v38 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, (int)a5, a6);
          v38 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v38) = v37;
        ++*(_DWORD *)(a3 + 8);
      }
    }
  }
LABEL_31:
  v183 = 0;
  v177 = (__int64 *)v179;
  v178 = 0x800000000LL;
  v181 = 0x800000000LL;
  v39 = (unsigned __int64 *)&v185;
  v180 = v182;
  v184 = 1;
  do
  {
    *v39 = -8;
    v39 += 2;
  }
  while ( v39 != (unsigned __int64 *)&v187 );
  v40 = (unsigned __int64 *)&v189;
  v187 = 0;
  v188 = 1;
  do
    *v40++ = -8;
  while ( v40 != (unsigned __int64 *)&v191 );
  v191 = 0;
  v41 = &v193;
  v192 = 1;
  do
    *v41++ = -8;
  while ( v41 != v194 );
  v42 = v194[6];
  v43 = v194[6] + 8LL * LODWORD(v194[7]);
  if ( v194[6] == v43 )
    goto LABEL_53;
  do
  {
    v51 = *(_QWORD *)(v43 - 8);
    v52 = *(_QWORD *)v51;
    if ( (v184 & 1) != 0 )
    {
      v44 = 3;
      v45 = (unsigned __int64 *)&v185;
    }
    else
    {
      v53 = v186;
      v45 = v185;
      if ( !v186 )
      {
        v54 = v184;
        ++v183;
        a5 = 0;
        v55 = ((unsigned int)v184 >> 1) + 1;
        goto LABEL_48;
      }
      v44 = v186 - 1;
    }
    v46 = v44 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v47 = &v45[2 * v46];
    a6 = *v47;
    if ( v52 == *v47 )
      goto LABEL_41;
    v111 = 1;
    a5 = 0;
    while ( a6 != -8 )
    {
      if ( a6 != -16 || a5 )
        v47 = a5;
      LODWORD(a5) = v111 + 1;
      v46 = v44 & (v111 + v46);
      a6 = v45[2 * v46];
      if ( v52 == a6 )
        goto LABEL_41;
      ++v111;
      a5 = v47;
      v47 = &v45[2 * v46];
    }
    if ( !a5 )
      a5 = v47;
    v54 = v184;
    ++v183;
    v55 = ((unsigned int)v184 >> 1) + 1;
    if ( (v184 & 1) == 0 )
    {
      v53 = v186;
LABEL_48:
      if ( 3 * v53 > 4 * v55 )
        goto LABEL_49;
      goto LABEL_156;
    }
    v53 = 4;
    if ( 4 * v55 < 0xC )
    {
LABEL_49:
      if ( v53 - HIDWORD(v184) - v55 > v53 >> 3 )
        goto LABEL_50;
      sub_39CC920((__int64)&v183, v53);
      if ( (v184 & 1) != 0 )
      {
        v138 = 3;
        v137 = (unsigned __int64 *)&v185;
      }
      else
      {
        v137 = v185;
        if ( !v186 )
        {
LABEL_269:
          LODWORD(v184) = (2 * ((unsigned int)v184 >> 1) + 2) | v184 & 1;
          BUG();
        }
        v138 = v186 - 1;
      }
      v54 = v184;
      v139 = v138 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      a5 = &v137[2 * v139];
      v140 = *a5;
      if ( v52 == *a5 )
        goto LABEL_50;
      LODWORD(a6) = 1;
      v116 = 0;
      while ( v140 != -8 )
      {
        if ( v140 == -16 && !v116 )
          v116 = a5;
        v160 = a6 + 1;
        LODWORD(a6) = v139 + a6;
        v139 = v138 & a6;
        a5 = &v137[2 * (v138 & (unsigned int)a6)];
        v140 = *a5;
        if ( v52 == *a5 )
          goto LABEL_197;
        LODWORD(a6) = v160;
      }
      goto LABEL_195;
    }
LABEL_156:
    sub_39CC920((__int64)&v183, 2 * v53);
    if ( (v184 & 1) != 0 )
    {
      v113 = 3;
      v112 = (unsigned __int64 *)&v185;
    }
    else
    {
      v112 = v185;
      if ( !v186 )
        goto LABEL_269;
      v113 = v186 - 1;
    }
    v54 = v184;
    v114 = v113 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    a5 = &v112[2 * v114];
    v115 = *a5;
    if ( v52 == *a5 )
      goto LABEL_50;
    LODWORD(a6) = 1;
    v116 = 0;
    while ( v115 != -8 )
    {
      if ( !v116 && v115 == -16 )
        v116 = a5;
      v117 = a6 + 1;
      LODWORD(a6) = v114 + a6;
      v114 = v113 & a6;
      a5 = &v112[2 * (v113 & (unsigned int)a6)];
      v115 = *a5;
      if ( v52 == *a5 )
        goto LABEL_197;
      LODWORD(a6) = v117;
    }
LABEL_195:
    if ( v116 )
      a5 = v116;
LABEL_197:
    v54 = v184;
LABEL_50:
    LODWORD(v184) = (2 * (v54 >> 1) + 2) | v54 & 1;
    if ( *a5 != -8 )
      --HIDWORD(v184);
    *a5 = v52;
    a5[1] = v51;
LABEL_41:
    v48 = (unsigned int)v181;
    v49 = v51 & 0xFFFFFFFFFFFFFFFBLL;
    if ( (unsigned int)v181 >= HIDWORD(v181) )
    {
      sub_16CD150((__int64)&v180, v182, 0, 8, (int)a5, a6);
      v48 = (unsigned int)v181;
    }
    v43 -= 8LL;
    *(_QWORD *)&v180[8 * v48] = v49;
    v50 = v181 + 1;
    LODWORD(v181) = v181 + 1;
  }
  while ( v42 != v43 );
  if ( !v50 )
    goto LABEL_53;
  while ( 2 )
  {
    v74 = v50--;
    v75 = *(_QWORD *)&v180[8 * v74 - 8];
    LODWORD(v181) = v50;
    v76 = v75 & 0xFFFFFFFFFFFFFFF8LL;
    v172 = v75 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v75 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_81;
    if ( (v188 & 1) != 0 )
    {
      v70 = (unsigned __int64 *)&v189;
      v71 = 7;
    }
    else
    {
      v70 = v189;
      v71 = v190 - 1;
      if ( !v190 )
        goto LABEL_85;
    }
    v72 = v71 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
    v73 = v70[v72];
    if ( v76 == v73 )
      goto LABEL_81;
    v110 = 1;
    while ( v73 != -8 )
    {
      v72 = v71 & (v110 + v72);
      v73 = v70[v72];
      if ( v76 == v73 )
        goto LABEL_81;
      ++v110;
    }
LABEL_85:
    if ( (v75 & 4) != 0 )
    {
      sub_39CD0B0((__int64)v176, (__int64)&v187, (__int64 *)&v172);
      v79 = (unsigned int)v178;
      if ( (unsigned int)v178 >= HIDWORD(v178) )
      {
        sub_16CD150((__int64)&v177, v179, 0, 8, v77, v78);
        v79 = (unsigned int)v178;
      }
      v177[v79] = v172;
      v50 = v181;
      LODWORD(v178) = v178 + 1;
      goto LABEL_81;
    }
    sub_39CD0B0((__int64)v176, (__int64)&v191, (__int64 *)&v172);
    if ( v176[32] )
    {
      v82 = (unsigned int)v181;
      v83 = v172 | 4;
      if ( (unsigned int)v181 >= HIDWORD(v181) )
      {
        sub_16CD150((__int64)&v180, v182, 0, 8, v80, v81);
        v82 = (unsigned int)v181;
      }
      *(_QWORD *)&v180[8 * v82] = v83;
      v173 = (unsigned __int64 **)v175;
      LODWORD(v181) = v181 + 1;
      v174 = 0x200000000LL;
      v84 = sub_3988770(v172);
      v87 = (unsigned int)v174;
      if ( *(_BYTE *)v84 != 13
        || *(_WORD *)(v84 + 2) != 1
        || (v118 = *(_QWORD *)(v84 + 8 * (4LL - *(unsigned int *)(v84 + 8)))) == 0
        || (v119 = 8LL * *(unsigned int *)(v118 + 8), v120 = v118 - v119, v118 == v118 - v119) )
      {
LABEL_94:
        v88 = v173;
        v89 = &v173[v87];
        if ( v173 == v89 )
          goto LABEL_116;
        while ( 2 )
        {
          v96 = *v88;
          if ( *v88 && *(_BYTE *)v96 != 25 )
            v96 = 0;
          if ( (v184 & 1) != 0 )
          {
            v90 = 3;
            v91 = (unsigned __int64 *)&v185;
          }
          else
          {
            v97 = v186;
            v91 = v185;
            if ( !v186 )
            {
              v98 = v184;
              ++v183;
              v99 = 0;
              v100 = ((unsigned int)v184 >> 1) + 1;
              goto LABEL_108;
            }
            v90 = v186 - 1;
          }
          v92 = v90 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
          v93 = &v91[2 * v92];
          v86 = (unsigned __int64 **)*v93;
          if ( v96 == (unsigned __int64 *)*v93 )
          {
            v94 = v93[1] & 0xFFFFFFFFFFFFFFFBLL;
            goto LABEL_99;
          }
          v101 = 1;
          v99 = 0;
          while ( 1 )
          {
            if ( v86 == (unsigned __int64 **)-8LL )
            {
              LODWORD(v91) = 12;
              v97 = 4;
              if ( !v99 )
                v99 = v93;
              v98 = v184;
              ++v183;
              v100 = ((unsigned int)v184 >> 1) + 1;
              if ( (v184 & 1) != 0 )
              {
LABEL_109:
                if ( (unsigned int)v91 > 4 * v100 )
                {
                  if ( v97 - HIDWORD(v184) - v100 > v97 >> 3 )
                    goto LABEL_111;
                  sub_39CC920((__int64)&v183, v97);
                  if ( (v184 & 1) != 0 )
                  {
                    v108 = 3;
                    v107 = (unsigned __int64 *)&v185;
                    goto LABEL_136;
                  }
                  v107 = v185;
                  if ( v186 )
                  {
                    v108 = v186 - 1;
LABEL_136:
                    v98 = v184;
                    v109 = v108 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
                    v99 = &v107[2 * v109];
                    v91 = (unsigned __int64 *)*v99;
                    if ( v96 == (unsigned __int64 *)*v99 )
                      goto LABEL_111;
                    LODWORD(v86) = 1;
                    v106 = 0;
                    while ( v91 != (unsigned __int64 *)-8LL )
                    {
                      if ( !v106 && v91 == (unsigned __int64 *)-16LL )
                        v106 = v99;
                      v109 = v108 & ((_DWORD)v86 + v109);
                      v99 = &v107[2 * v109];
                      v91 = (unsigned __int64 *)*v99;
                      if ( v96 == (unsigned __int64 *)*v99 )
                        goto LABEL_132;
                      LODWORD(v86) = (_DWORD)v86 + 1;
                    }
LABEL_130:
                    if ( v106 )
                      v99 = v106;
LABEL_132:
                    v98 = v184;
LABEL_111:
                    LODWORD(v184) = (2 * (v98 >> 1) + 2) | v98 & 1;
                    if ( *v99 != -8 )
                      --HIDWORD(v184);
                    *v99 = (unsigned __int64)v96;
                    v94 = 0;
                    v99[1] = 0;
                    v95 = (unsigned int)v181;
                    if ( (unsigned int)v181 < HIDWORD(v181) )
                      goto LABEL_100;
LABEL_114:
                    sub_16CD150((__int64)&v180, v182, 0, 8, (int)v91, (int)v86);
                    v95 = (unsigned int)v181;
                    goto LABEL_100;
                  }
LABEL_270:
                  LODWORD(v184) = (2 * ((unsigned int)v184 >> 1) + 2) | v184 & 1;
                  BUG();
                }
                sub_39CC920((__int64)&v183, 2 * v97);
                if ( (v184 & 1) != 0 )
                {
                  v102 = 3;
                  v103 = (unsigned __int64 *)&v185;
                }
                else
                {
                  v103 = v185;
                  if ( !v186 )
                    goto LABEL_270;
                  v102 = v186 - 1;
                }
                v98 = v184;
                v104 = v102 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
                v99 = &v103[2 * v104];
                v91 = (unsigned __int64 *)*v99;
                if ( v96 == (unsigned __int64 *)*v99 )
                  goto LABEL_111;
                v105 = 1;
                v106 = 0;
                while ( v91 != (unsigned __int64 *)-8LL )
                {
                  if ( v106 || v91 != (unsigned __int64 *)-16LL )
                    v99 = v106;
                  v104 = v102 & (v105 + v104);
                  v86 = (unsigned __int64 **)&v103[2 * v104];
                  v91 = *v86;
                  if ( v96 == *v86 )
                  {
                    v98 = v184;
                    v99 = &v103[2 * v104];
                    goto LABEL_111;
                  }
                  ++v105;
                  v106 = v99;
                  v99 = &v103[2 * v104];
                }
                goto LABEL_130;
              }
              v97 = v186;
LABEL_108:
              LODWORD(v91) = 3 * v97;
              goto LABEL_109;
            }
            if ( v99 || v86 != (unsigned __int64 **)-16LL )
              v93 = v99;
            v92 = v90 & (v101 + v92);
            v141 = &v91[2 * v92];
            v86 = (unsigned __int64 **)*v141;
            if ( v96 == (unsigned __int64 *)*v141 )
              break;
            ++v101;
            v99 = v93;
            v93 = &v91[2 * v92];
          }
          v94 = v141[1] & 0xFFFFFFFFFFFFFFFBLL;
LABEL_99:
          v95 = (unsigned int)v181;
          if ( (unsigned int)v181 >= HIDWORD(v181) )
            goto LABEL_114;
LABEL_100:
          ++v88;
          *(_QWORD *)&v180[8 * v95] = v94;
          LODWORD(v181) = v181 + 1;
          if ( v89 != v88 )
            continue;
          break;
        }
        v89 = v173;
LABEL_116:
        if ( v89 != (unsigned __int64 **)v175 )
          _libc_free((unsigned __int64)v89);
        v50 = v181;
LABEL_81:
        if ( v50 )
          continue;
        break;
      }
      while ( 2 )
      {
        if ( **(_BYTE **)v120 == 9 )
        {
          v123 = *(_QWORD *)(*(_QWORD *)v120 - 8LL * *(unsigned int *)(*(_QWORD *)v120 + 8LL));
          v124 = *(unsigned __int8 *)v123;
          if ( (_BYTE)v124 == 1 )
          {
            v121 = *(_QWORD *)(v123 + 136);
            goto LABEL_171;
          }
          if ( (unsigned int)(v124 - 24) <= 1 )
          {
            v121 = v123 | 4;
LABEL_171:
            if ( (v121 & 4) != 0 )
            {
              v122 = v121 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v121 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( HIDWORD(v174) <= (unsigned int)v87 )
                {
                  sub_16CD150((__int64)&v173, v175, 0, 8, v85, (int)v86);
                  v87 = (unsigned int)v174;
                }
                v173[v87] = (unsigned __int64 *)v122;
                v87 = (unsigned int)(v174 + 1);
                LODWORD(v174) = v174 + 1;
              }
            }
          }
        }
        v120 += 8;
        if ( v118 == v120 )
          goto LABEL_94;
        continue;
      }
    }
    break;
  }
LABEL_53:
  if ( (v192 & 1) == 0 )
    j___libc_free_0(v193);
  if ( (v188 & 1) == 0 )
    j___libc_free_0((unsigned __int64)v189);
  if ( (v184 & 1) == 0 )
    j___libc_free_0((unsigned __int64)v185);
  if ( v180 != v182 )
    _libc_free((unsigned __int64)v180);
  v56 = (const void *)(a3 + 16);
  if ( v177 != &v177[(unsigned int)v178] )
  {
    v168 = &v177[(unsigned int)v178];
    v57 = a3;
    v58 = v177;
    do
    {
      v61 = sub_39CAC10((__int64 *)a1, *v58, a2, &v171);
      v62 = *(unsigned int *)(v57 + 8);
      if ( (unsigned int)v62 >= *(_DWORD *)(v57 + 12) )
      {
        v162 = v61;
        sub_16CD150(v57, v56, 0, 8, v59, v60);
        v62 = *(unsigned int *)(v57 + 8);
        v61 = v162;
      }
      ++v58;
      *(_QWORD *)(*(_QWORD *)v57 + 8 * v62) = v61;
      ++*(_DWORD *)(v57 + 8);
    }
    while ( v168 != v58 );
    a3 = v57;
  }
  v63 = sub_39C84F0((_QWORD *)a1) == 0;
  v64 = *(_QWORD *)(a2 + 8);
  if ( v63 )
  {
    v125 = *(_DWORD *)(a1 + 664);
    v126 = a1 + 640;
    if ( v125 )
    {
      v127 = *(_QWORD *)(a1 + 648);
      v128 = ((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4);
      LODWORD(v129) = (v125 - 1) & v128;
      v130 = v127 + 88LL * (unsigned int)v129;
      v131 = *(_QWORD *)v130;
      if ( v64 == *(_QWORD *)v130 )
      {
LABEL_184:
        v132 = *(__int64 **)(v130 + 8);
        v133 = &v132[*(unsigned int *)(v130 + 16)];
        if ( v132 != v133 )
        {
          do
          {
            v134 = sub_39CC1A0((__int64 *)a1, *v132);
            v136 = *(unsigned int *)(a3 + 8);
            if ( (unsigned int)v136 >= *(_DWORD *)(a3 + 12) )
            {
              v169 = v134;
              sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v134, v135);
              v136 = *(unsigned int *)(a3 + 8);
              v134 = v169;
            }
            ++v132;
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v136) = v134;
            ++*(_DWORD *)(a3 + 8);
          }
          while ( v133 != v132 );
          v64 = *(_QWORD *)(a2 + 8);
        }
        goto LABEL_68;
      }
      v143 = 1;
      v144 = 0;
      while ( v131 != -8 )
      {
        if ( v131 == -16 && !v144 )
          v144 = (__int64 *)v130;
        v129 = (v125 - 1) & ((_DWORD)v129 + v143);
        v130 = v127 + 88 * v129;
        v131 = *(_QWORD *)v130;
        if ( v64 == *(_QWORD *)v130 )
          goto LABEL_184;
        ++v143;
      }
      if ( !v144 )
        v144 = (__int64 *)v130;
      ++*(_QWORD *)(a1 + 640);
      v145 = *(_DWORD *)(a1 + 656) + 1;
      if ( 4 * v145 < 3 * v125 )
      {
        if ( v125 - *(_DWORD *)(a1 + 660) - v145 > v125 >> 3 )
        {
LABEL_218:
          *(_DWORD *)(a1 + 656) = v145;
          if ( *v144 != -8 )
            --*(_DWORD *)(a1 + 660);
          *v144 = v64;
          v144[1] = (__int64)(v144 + 3);
          v144[2] = 0x800000000LL;
          v64 = *(_QWORD *)(a2 + 8);
          goto LABEL_68;
        }
        v166 = v64;
        sub_3991D30(v126, v125);
        v153 = *(_DWORD *)(a1 + 664);
        if ( v153 )
        {
          v154 = v153 - 1;
          v155 = *(_QWORD *)(a1 + 648);
          v64 = v166;
          v156 = 0;
          v157 = v154 & v128;
          v145 = *(_DWORD *)(a1 + 656) + 1;
          v158 = 1;
          v144 = (__int64 *)(v155 + 88LL * v157);
          v159 = *v144;
          if ( v166 != *v144 )
          {
            while ( v159 != -8 )
            {
              if ( !v156 && v159 == -16 )
                v156 = v144;
              v157 = v154 & (v158 + v157);
              v144 = (__int64 *)(v155 + 88LL * v157);
              v159 = *v144;
              if ( v166 == *v144 )
                goto LABEL_218;
              ++v158;
            }
            if ( v156 )
              v144 = v156;
          }
          goto LABEL_218;
        }
LABEL_268:
        ++*(_DWORD *)(a1 + 656);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 640);
    }
    v170 = v64;
    sub_3991D30(v126, 2 * v125);
    v146 = *(_DWORD *)(a1 + 664);
    if ( v146 )
    {
      v64 = v170;
      v147 = v146 - 1;
      v148 = *(_QWORD *)(a1 + 648);
      v149 = v147 & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
      v145 = *(_DWORD *)(a1 + 656) + 1;
      v144 = (__int64 *)(v148 + 88LL * v149);
      v150 = *v144;
      if ( v170 != *v144 )
      {
        v151 = 1;
        v152 = 0;
        while ( v150 != -8 )
        {
          if ( !v152 && v150 == -16 )
            v152 = v144;
          v149 = v147 & (v151 + v149);
          v144 = (__int64 *)(v148 + 88LL * v149);
          v150 = *v144;
          if ( v170 == *v144 )
            goto LABEL_218;
          ++v151;
        }
        if ( v152 )
          v144 = v152;
      }
      goto LABEL_218;
    }
    goto LABEL_268;
  }
LABEL_68:
  sub_39A2570(a1, a3, v64);
  if ( a4 )
    *a4 = *(_DWORD *)(a3 + 8) != 0;
  v65 = *(__int64 **)(a2 + 32);
  v66 = &v65[*(unsigned int *)(a2 + 40)];
  while ( v66 != v65 )
  {
    v67 = *v65++;
    sub_39CE8E0(a1, v67, a3);
  }
  v68 = v171;
  if ( v177 != (__int64 *)v179 )
    _libc_free((unsigned __int64)v177);
  if ( (unsigned __int64 *)v194[6] != &v194[8] )
    _libc_free(v194[6]);
  sub_39C77C0(v194[2]);
  return v68;
}
