// Function: sub_A323F0
// Address: 0xa323f0
//
__int64 *__fastcall sub_A323F0(__int64 *a1, __int64 a2, __int64 a3, _BOOL4 a4)
{
  _BOOL4 v4; // r11d
  __int64 *v5; // r15
  __int64 v6; // rbx
  unsigned int v7; // r8d
  __int64 v8; // r9
  unsigned int v9; // eax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  _QWORD *v12; // rbx
  _QWORD *v13; // r13
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  int v21; // edx
  _QWORD *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // r8
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rbx
  unsigned int v30; // esi
  __int64 v31; // r8
  __int64 v32; // r10
  int v33; // r13d
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 *result; // rax
  int v38; // eax
  __int64 v39; // rbx
  int v40; // edx
  __int64 v41; // r10
  int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // r12
  __int64 v45; // rdi
  __int64 v46; // rax
  __m128i *v47; // rax
  __int64 v48; // r11
  __int64 v49; // rax
  __int16 v50; // si
  _QWORD *v51; // r12
  __int64 *v52; // r8
  _QWORD *m; // rbx
  unsigned __int64 v54; // rsi
  __int64 *v55; // rax
  __int64 *v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rdx
  unsigned int v60; // esi
  __int64 v61; // rdi
  unsigned int v62; // ecx
  __int64 v63; // rbx
  int v64; // eax
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 *v70; // rbx
  __int64 *v71; // r13
  _QWORD *v72; // rdx
  __int64 v73; // rax
  __int64 *v74; // rdi
  __int64 v75; // rbx
  __int64 *v76; // r14
  __int64 *i; // r13
  __int64 v78; // rax
  __int64 v79; // rbx
  __int64 *v80; // r13
  __int64 *v81; // rbx
  __int64 v82; // rsi
  __int64 *v83; // r13
  __int64 *v84; // rbx
  __int64 v85; // rax
  __int64 v86; // r9
  __int64 v87; // rax
  __int64 v88; // r9
  __int64 *v89; // r13
  __int64 *v90; // rbx
  __int64 v91; // rax
  __int64 v92; // r9
  __int64 v93; // rax
  __int64 v94; // r9
  __int64 *v95; // rax
  _BYTE ***v96; // rax
  unsigned int *v97; // rcx
  unsigned int *v98; // rdx
  __int64 v99; // r13
  __int64 v100; // rdi
  __int64 v101; // rax
  __m128i *v102; // rax
  __int64 v103; // r11
  __int64 v104; // rax
  __int16 v105; // si
  __int16 v106; // dx
  __int16 v107; // cx
  _QWORD *v108; // rdx
  _QWORD *v109; // r12
  __int64 *v110; // rax
  __int64 v111; // r14
  __int64 v112; // rbx
  _QWORD *v113; // r15
  __int64 *v114; // r13
  __int64 v115; // rdx
  _QWORD *v116; // r9
  unsigned __int64 v117; // rsi
  _QWORD *v118; // rax
  _QWORD *v119; // rdi
  __int64 v120; // rcx
  __int64 v121; // rdx
  __int64 v122; // r13
  __int64 v123; // rbx
  __int64 j; // r12
  unsigned __int64 v125; // rsi
  __int64 *v126; // rax
  __int64 *v127; // rdi
  __int64 v128; // rcx
  __int64 v129; // rdx
  int v130; // r13d
  __int64 v131; // r14
  int v132; // eax
  int v133; // edx
  _QWORD *v134; // rax
  __int64 v135; // r13
  unsigned int v136; // eax
  __int64 *v137; // rax
  __int64 v138; // rsi
  __int64 *v139; // rax
  __int64 v140; // rsi
  __int64 *v141; // r15
  __int64 v142; // rax
  __int64 v143; // r9
  unsigned __int64 v144; // rdx
  __int64 v145; // rdx
  __int64 v146; // rax
  _QWORD *v147; // rax
  __int64 k; // rdx
  unsigned __int64 v149; // rdi
  __int64 *v150; // rax
  __int64 *v151; // r9
  __int64 v152; // rcx
  __int64 v153; // rax
  unsigned int v154; // eax
  __int64 *v155; // rax
  __int64 v156; // rdx
  __int64 v157; // rax
  unsigned __int64 v158; // r9
  __int64 v159; // rax
  __int64 *v160; // rdx
  __int64 v161; // rax
  __int64 v162; // rdx
  unsigned __int64 v163; // r10
  __int64 v164; // r9
  int v165; // esi
  __int64 v166; // [rsp-18h] [rbp-318h]
  __int64 *v167; // [rsp-10h] [rbp-310h]
  __int64 *v168; // [rsp+0h] [rbp-300h]
  __int64 v169; // [rsp+8h] [rbp-2F8h]
  __int64 v170; // [rsp+10h] [rbp-2F0h]
  __int64 v171; // [rsp+18h] [rbp-2E8h]
  __int64 v172; // [rsp+20h] [rbp-2E0h]
  __int64 v173; // [rsp+20h] [rbp-2E0h]
  __int64 v174; // [rsp+20h] [rbp-2E0h]
  __int64 v175; // [rsp+20h] [rbp-2E0h]
  __int64 *v176; // [rsp+28h] [rbp-2D8h]
  unsigned __int32 v177; // [rsp+30h] [rbp-2D0h]
  bool v178; // [rsp+37h] [rbp-2C9h]
  __int64 v179; // [rsp+48h] [rbp-2B8h]
  _BOOL4 v180; // [rsp+50h] [rbp-2B0h]
  bool v181; // [rsp+50h] [rbp-2B0h]
  int v182; // [rsp+50h] [rbp-2B0h]
  _BOOL4 v183; // [rsp+50h] [rbp-2B0h]
  unsigned int v184; // [rsp+58h] [rbp-2A8h]
  unsigned int v185; // [rsp+58h] [rbp-2A8h]
  __int64 v186; // [rsp+58h] [rbp-2A8h]
  __int64 v187; // [rsp+58h] [rbp-2A8h]
  __int64 v188; // [rsp+58h] [rbp-2A8h]
  __int64 v189; // [rsp+58h] [rbp-2A8h]
  __int64 *v190; // [rsp+58h] [rbp-2A8h]
  __int64 v191[2]; // [rsp+60h] [rbp-2A0h] BYREF
  __int64 v192; // [rsp+70h] [rbp-290h] BYREF
  __int64 *v193; // [rsp+78h] [rbp-288h] BYREF
  __int64 v194; // [rsp+80h] [rbp-280h] BYREF
  unsigned int v195; // [rsp+88h] [rbp-278h]
  __int64 *v196; // [rsp+90h] [rbp-270h] BYREF
  unsigned int v197; // [rsp+98h] [rbp-268h]
  _QWORD *v198; // [rsp+A0h] [rbp-260h] BYREF
  unsigned int v199; // [rsp+A8h] [rbp-258h]
  __int64 *v200; // [rsp+B0h] [rbp-250h]
  __int64 (__fastcall *v201)(__int64 **, _QWORD *); // [rsp+B8h] [rbp-248h]
  __m128i v202; // [rsp+C0h] [rbp-240h] BYREF
  _QWORD v203[70]; // [rsp+D0h] [rbp-230h] BYREF

  v4 = a4;
  v5 = a1;
  v6 = *a1;
  v191[0] = a2;
  v191[1] = a3;
  v7 = *(_DWORD *)(v6 + 24);
  v178 = a4;
  v192 = a3;
  if ( !v7 )
  {
    v202.m128i_i64[0] = 0;
    ++*(_QWORD *)v6;
LABEL_245:
    v183 = v4;
    v165 = 2 * v7;
LABEL_246:
    sub_A32210(v6, v165);
    sub_A27FA0(v6, v191, &v202);
    v4 = v183;
    v133 = *(_DWORD *)(v6 + 16) + 1;
    goto LABEL_144;
  }
  v8 = *(_QWORD *)(v6 + 8);
  v9 = (v7 - 1) & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
  v10 = (__int64 *)(v8 + 8LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_3;
  v130 = 1;
  v131 = 0;
  while ( v11 != -1 )
  {
    if ( v131 || v11 != -2 )
      v10 = (__int64 *)v131;
    v9 = (v7 - 1) & (v130 + v9);
    v11 = *(_QWORD *)(v8 + 8LL * v9);
    if ( a2 == v11 )
      goto LABEL_3;
    ++v130;
    v131 = (__int64)v10;
    v10 = (__int64 *)(v8 + 8LL * v9);
  }
  if ( !v131 )
    v131 = (__int64)v10;
  v202.m128i_i64[0] = v131;
  v132 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v133 = v132 + 1;
  if ( 4 * (v132 + 1) >= 3 * v7 )
    goto LABEL_245;
  if ( v7 - *(_DWORD *)(v6 + 20) - v133 <= v7 >> 3 )
  {
    v183 = v4;
    v165 = v7;
    goto LABEL_246;
  }
LABEL_144:
  *(_DWORD *)(v6 + 16) = v133;
  v134 = (_QWORD *)v202.m128i_i64[0];
  if ( *(_QWORD *)v202.m128i_i64[0] != -1 )
    --*(_DWORD *)(v6 + 20);
  *v134 = v191[0];
  a3 = v192;
LABEL_3:
  v12 = *(_QWORD **)(a3 + 40);
  v13 = &v12[*(unsigned int *)(a3 + 48)];
  if ( v12 != v13 )
  {
    while ( 1 )
    {
      v18 = *v5;
      v19 = *(_QWORD *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
      v198 = (_QWORD *)v19;
      v20 = *(_DWORD *)(v18 + 24);
      if ( !v20 )
        break;
      v14 = *(_QWORD *)(v18 + 8);
      v15 = (v20 - 1) & (((0xBF58476D1CE4E5B9LL * v19) >> 31) ^ (484763065 * v19));
      v16 = (__int64 *)(v14 + 8LL * v15);
      v17 = *v16;
      if ( v19 == *v16 )
      {
LABEL_6:
        if ( v13 == ++v12 )
          goto LABEL_14;
      }
      else
      {
        v182 = 1;
        v41 = 0;
        while ( v17 != -1 )
        {
          if ( v41 || v17 != -2 )
            v16 = (__int64 *)v41;
          v15 = (v20 - 1) & (v182 + v15);
          v17 = *(_QWORD *)(v14 + 8LL * v15);
          if ( v19 == v17 )
            goto LABEL_6;
          ++v182;
          v41 = (__int64)v16;
          v16 = (__int64 *)(v14 + 8LL * v15);
        }
        if ( !v41 )
          v41 = (__int64)v16;
        v202.m128i_i64[0] = v41;
        v42 = *(_DWORD *)(v18 + 16);
        ++*(_QWORD *)v18;
        v21 = v42 + 1;
        if ( 4 * (v42 + 1) >= 3 * v20 )
          goto LABEL_9;
        if ( v20 - *(_DWORD *)(v18 + 20) - v21 > v20 >> 3 )
          goto LABEL_11;
        v180 = v4;
LABEL_10:
        sub_A32210(v18, v20);
        sub_A27FA0(v18, (__int64 *)&v198, &v202);
        v4 = v180;
        v21 = *(_DWORD *)(v18 + 16) + 1;
LABEL_11:
        *(_DWORD *)(v18 + 16) = v21;
        v22 = (_QWORD *)v202.m128i_i64[0];
        if ( *(_QWORD *)v202.m128i_i64[0] != -1 )
          --*(_DWORD *)(v18 + 20);
        ++v12;
        *v22 = v198;
        if ( v13 == v12 )
          goto LABEL_14;
      }
    }
    v202.m128i_i64[0] = 0;
    ++*(_QWORD *)v18;
LABEL_9:
    v180 = v4;
    v20 *= 2;
    goto LABEL_10;
  }
LABEL_14:
  v23 = v5[1];
  v24 = (_QWORD *)(v23 + 48);
  v25 = *(_QWORD **)(v23 + 56);
  if ( !v25 )
    goto LABEL_21;
  v26 = v24;
  do
  {
    while ( 1 )
    {
      v27 = v25[2];
      v28 = v25[3];
      if ( v191[0] <= v25[4] )
        break;
      v25 = (_QWORD *)v25[3];
      if ( !v28 )
        goto LABEL_19;
    }
    v26 = v25;
    v25 = (_QWORD *)v25[2];
  }
  while ( v27 );
LABEL_19:
  if ( v26 == v24 || v191[0] < v26[4] )
  {
LABEL_21:
    v29 = v5[2];
    v30 = *(_DWORD *)(v29 + 24);
    if ( v30 )
      goto LABEL_22;
LABEL_30:
    v202.m128i_i64[0] = 0;
    ++*(_QWORD *)v29;
LABEL_31:
    v181 = v4;
    v30 *= 2;
    goto LABEL_32;
  }
  v29 = v5[2];
  v30 = *(_DWORD *)(v29 + 24);
  v184 = *((_DWORD *)v26 + 10);
  if ( !v30 )
    goto LABEL_30;
LABEL_22:
  v31 = *(_QWORD *)(v29 + 8);
  v32 = 0;
  v33 = 1;
  v34 = (v30 - 1) & (((unsigned int)v192 >> 9) ^ ((unsigned int)v192 >> 4));
  v35 = v31 + 16LL * v34;
  v36 = *(_QWORD *)v35;
  if ( *(_QWORD *)v35 == v192 )
    goto LABEL_23;
  while ( v36 != -4096 )
  {
    if ( !v32 && v36 == -8192 )
      v32 = v35;
    v34 = (v30 - 1) & (v33 + v34);
    v35 = v31 + 16LL * v34;
    v36 = *(_QWORD *)v35;
    if ( v192 == *(_QWORD *)v35 )
      goto LABEL_23;
    ++v33;
  }
  if ( !v32 )
    v32 = v35;
  v202.m128i_i64[0] = v32;
  v64 = *(_DWORD *)(v29 + 16);
  ++*(_QWORD *)v29;
  v40 = v64 + 1;
  if ( 4 * (v64 + 1) >= 3 * v30 )
    goto LABEL_31;
  if ( v30 - *(_DWORD *)(v29 + 20) - v40 > v30 >> 3 )
    goto LABEL_71;
  v181 = v4;
LABEL_32:
  sub_A32030(v29, v30);
  sub_A1A190(v29, &v192, &v202);
  LOBYTE(v4) = v181;
  v40 = *(_DWORD *)(v29 + 16) + 1;
LABEL_71:
  *(_DWORD *)(v29 + 16) = v40;
  v35 = v202.m128i_i64[0];
  if ( *(_QWORD *)v202.m128i_i64[0] != -4096 )
    --*(_DWORD *)(v29 + 20);
  v65 = v192;
  *(_DWORD *)(v35 + 8) = 0;
  *(_QWORD *)v35 = v65;
LABEL_23:
  result = (__int64 *)(v35 + 8);
  *(_DWORD *)result = v184;
  if ( v4 )
    return result;
  v179 = v192;
  v38 = *(_DWORD *)(v192 + 8);
  if ( !v38 )
  {
    v39 = v5[3];
    result = (__int64 *)*(unsigned int *)(v39 + 8);
    if ( (unsigned __int64)result + 1 > *(unsigned int *)(v39 + 12) )
    {
      sub_C8D5F0(v5[3], v39 + 16, (char *)result + 1, 8);
      result = (__int64 *)*(unsigned int *)(v39 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v39 + 8LL * (_QWORD)result) = v179;
    ++*(_DWORD *)(v39 + 8);
    return result;
  }
  v43 = v184;
  v169 = v184;
  if ( v38 != 2 )
  {
    v176 = (__int64 *)v5[1];
    v193 = v176;
    v170 = *v176;
    v66 = *(__int64 **)(v192 + 80);
    if ( v66 )
    {
      v67 = *v66;
      v68 = (v66[1] - *v66) >> 3;
      if ( v68 )
      {
        v202.m128i_i64[0] = v67;
        v43 = 11;
        v202.m128i_i64[1] = v68;
        sub_A1B680(v170, 0xBu, v202.m128i_i64, 0);
      }
    }
    v202.m128i_i64[0] = (__int64)v203;
    v202.m128i_i64[1] = 0x4000000000LL;
    v69 = *(_QWORD **)(v179 + 80);
    if ( v69 )
    {
      v70 = (__int64 *)v69[3];
      v71 = (__int64 *)v69[4];
      if ( v70 == v71 )
        goto LABEL_87;
      v72 = v203;
      v73 = 0;
      v74 = v70 + 2;
      v75 = *v70;
      v76 = v71;
      for ( i = v74; ; i += 2 )
      {
        v72[v73] = v75;
        ++v202.m128i_i32[2];
        v78 = v202.m128i_u32[2];
        v79 = *(i - 1);
        if ( (unsigned __int64)v202.m128i_u32[2] + 1 > v202.m128i_u32[3] )
        {
          sub_C8D5F0(&v202, v203, v202.m128i_u32[2] + 1LL, 8);
          v78 = v202.m128i_u32[2];
        }
        *(_QWORD *)(v202.m128i_i64[0] + 8 * v78) = v79;
        v73 = (unsigned int)++v202.m128i_i32[2];
        if ( v76 == i )
          break;
        v75 = *i;
        if ( v73 + 1 > (unsigned __int64)v202.m128i_u32[3] )
        {
          sub_C8D5F0(&v202, v203, v73 + 1, 8);
          v73 = v202.m128i_u32[2];
        }
        v72 = (_QWORD *)v202.m128i_i64[0];
      }
      v43 = 12;
      sub_A1FB70(v170, 0xCu, (__int64)&v202, 0);
      v69 = *(_QWORD **)(v179 + 80);
      if ( v69 )
      {
LABEL_87:
        v80 = (__int64 *)v69[6];
        v81 = (__int64 *)v69[7];
        if ( v80 == v81 )
          goto LABEL_256;
        v202.m128i_i32[2] = 0;
        do
        {
          v82 = *v80;
          v80 += 2;
          sub_A188E0((__int64)&v202, v82);
          sub_A188E0((__int64)&v202, *(v80 - 1));
        }
        while ( v81 != v80 );
        v43 = 13;
        sub_A1FB70(v170, 0xDu, (__int64)&v202, 0);
        v69 = *(_QWORD **)(v179 + 80);
        if ( v69 )
        {
LABEL_256:
          v83 = (__int64 *)v69[9];
          v84 = (__int64 *)v69[10];
          if ( v83 == v84 )
            goto LABEL_98;
          do
          {
            v85 = 0;
            v202.m128i_i32[2] = 0;
            v86 = *v83;
            if ( !v202.m128i_i32[3] )
            {
              v187 = *v83;
              sub_C8D5F0(&v202, v203, 1, 8);
              v86 = v187;
              v85 = 8LL * v202.m128i_u32[2];
            }
            *(_QWORD *)(v202.m128i_i64[0] + v85) = v86;
            ++v202.m128i_i32[2];
            v87 = v202.m128i_u32[2];
            v88 = v83[1];
            if ( (unsigned __int64)v202.m128i_u32[2] + 1 > v202.m128i_u32[3] )
            {
              v186 = v83[1];
              sub_C8D5F0(&v202, v203, v202.m128i_u32[2] + 1LL, 8);
              v87 = v202.m128i_u32[2];
              v88 = v186;
            }
            v83 += 5;
            *(_QWORD *)(v202.m128i_i64[0] + 8 * v87) = v88;
            ++v202.m128i_i32[2];
            sub_A16520(
              (__int64)&v202,
              (char *)(v202.m128i_i64[0] + 8LL * v202.m128i_u32[2]),
              (char *)*(v83 - 3),
              (char *)*(v83 - 2));
            v43 = 14;
            sub_A1FB70(v170, 0xEu, (__int64)&v202, 0);
          }
          while ( v84 != v83 );
          v69 = *(_QWORD **)(v179 + 80);
          if ( v69 )
          {
LABEL_98:
            v89 = (__int64 *)v69[12];
            v90 = (__int64 *)v69[13];
            while ( v90 != v89 )
            {
              v91 = 0;
              v202.m128i_i32[2] = 0;
              v92 = *v89;
              if ( !v202.m128i_i32[3] )
              {
                v189 = *v89;
                sub_C8D5F0(&v202, v203, 1, 8);
                v92 = v189;
                v91 = 8LL * v202.m128i_u32[2];
              }
              *(_QWORD *)(v202.m128i_i64[0] + v91) = v92;
              ++v202.m128i_i32[2];
              v93 = v202.m128i_u32[2];
              v94 = v89[1];
              if ( (unsigned __int64)v202.m128i_u32[2] + 1 > v202.m128i_u32[3] )
              {
                v188 = v89[1];
                sub_C8D5F0(&v202, v203, v202.m128i_u32[2] + 1LL, 8);
                v93 = v202.m128i_u32[2];
                v94 = v188;
              }
              v89 += 5;
              *(_QWORD *)(v202.m128i_i64[0] + 8 * v93) = v94;
              ++v202.m128i_i32[2];
              sub_A16520(
                (__int64)&v202,
                (char *)(v202.m128i_i64[0] + 8LL * v202.m128i_u32[2]),
                (char *)*(v89 - 3),
                (char *)*(v89 - 2));
              v43 = 15;
              sub_A1FB70(v170, 0xFu, (__int64)&v202, 0);
            }
          }
        }
      }
    }
    v95 = *(__int64 **)(v179 + 88);
    if ( !v95 || *v95 == v95[1] || (v202.m128i_i32[2] = 0, v171 = v95[1], *v95 == v171) )
    {
LABEL_106:
      if ( (_QWORD *)v202.m128i_i64[0] != v203 )
        _libc_free(v202.m128i_i64[0], v43);
      sub_A19FD0(v179, (_QWORD *)v5[8]);
      v96 = (_BYTE ***)v5[1];
      v97 = (unsigned int *)v5[10];
      v203[1] = sub_A31110;
      v203[0] = sub_A15AB0;
      v198 = &v193;
      v98 = (unsigned int *)v5[9];
      v201 = sub_A16110;
      v167 = (__int64 *)v5[12];
      v166 = v5[11];
      v202.m128i_i64[0] = (__int64)v96;
      v200 = (__int64 *)sub_A15AE0;
      sub_A2C250(*v96, v179, *v98, *v97, 0, 0, (__int64)&v198, (__int64)&v202, 0, v166, v167);
      sub_A17130((__int64)&v198);
      sub_A17130((__int64)&v202);
      sub_A188E0(v5[4], v169);
      v99 = v5[4];
      v100 = v5[1] + 152;
      v101 = *(_QWORD *)(v179 + 32);
      v202.m128i_i64[0] = *(_QWORD *)(v179 + 24);
      v202.m128i_i64[1] = v101;
      v102 = sub_A2B8A0(v100, &v202);
      sub_A188E0(v99, v102->m128i_i64[0]);
      v103 = v5[4];
      v104 = *(_QWORD *)(*(_QWORD *)v5[5] + 24LL);
      if ( v104 )
        v178 = sub_A15B10(*(_QWORD *)v104, *(_QWORD *)(v104 + 8), v179) != 0;
      v105 = *(unsigned __int8 *)(v179 + 12);
      LOBYTE(v105) = (unsigned __int8)v105 >> 4;
      sub_A188E0(
        v103,
        *(_BYTE *)(v179 + 12) & 0xF
      | (unsigned __int64)((v105 << 8) & 0x300)
      | (16
       * (((*(_BYTE *)(v179 + 12) & 0x40) != 0)
        | (unsigned __int64)((8 * (*(_BYTE *)(v179 + 13) >> 1)) & 8)
        | (4 * (*(_BYTE *)(v179 + 13) & 1))
        | (2 * (*(_BYTE *)(v179 + 12) >> 7))))
      | ((v178 || (*(_BYTE *)(v179 + 13) & 4) != 0) << 10) & 0x3FC00);
      sub_A188E0(v5[4], *(unsigned int *)(v179 + 56));
      v106 = *(unsigned __int8 *)(v179 + 61);
      v107 = v106;
      LOBYTE(v106) = (unsigned __int8)v106 >> 1;
      sub_A188E0(
        v5[4],
        (2 * ((*(_BYTE *)(v179 + 60) & 2) != 0))
      | (4 * ((*(_BYTE *)(v179 + 60) & 4) != 0))
      | (v106 << 9) & 0x200
      | (v107 << 8) & 0x100
      | (unsigned __int8)(*(_BYTE *)(v179 + 60) >> 7 << 7)
      | (*(_BYTE *)(v179 + 60) >> 6 << 6) & 0x40
      | (32 * (*(_BYTE *)(v179 + 60) >> 5)) & 0x20
      | *(_BYTE *)(v179 + 60) & 1
      | (unsigned __int64)((16 * ((*(_BYTE *)(v179 + 60) & 0x10) != 0)) | (8 * ((*(_BYTE *)(v179 + 60) & 8) != 0))));
      sub_A188E0(v5[4], 0);
      sub_A188E0(v5[4], 0);
      sub_A188E0(v5[4], 0);
      sub_A188E0(v5[4], 0);
      v108 = *(_QWORD **)(v179 + 40);
      v109 = &v108[*(unsigned int *)(v179 + 48)];
      if ( v109 == v108 )
      {
        v122 = 0;
        v111 = 0;
        v112 = 0;
      }
      else
      {
        v185 = 0;
        v110 = v5;
        v111 = 0;
        v112 = 0;
        v113 = *(_QWORD **)(v179 + 40);
        v114 = v110;
        do
        {
          v115 = v114[1];
          v116 = (_QWORD *)(v115 + 48);
          v117 = *(_QWORD *)(*v113 & 0xFFFFFFFFFFFFFFF8LL);
          v118 = *(_QWORD **)(v115 + 56);
          if ( v118 )
          {
            v119 = (_QWORD *)(v115 + 48);
            do
            {
              while ( 1 )
              {
                v120 = v118[2];
                v121 = v118[3];
                if ( v117 <= v118[4] )
                  break;
                v118 = (_QWORD *)v118[3];
                if ( !v121 )
                  goto LABEL_117;
              }
              v119 = v118;
              v118 = (_QWORD *)v118[2];
            }
            while ( v120 );
LABEL_117:
            if ( v119 != v116 && v117 >= v119[4] )
            {
              sub_A188E0(v114[4], *((unsigned int *)v119 + 10));
              if ( (*v113 & 2) != 0 )
                v111 = (unsigned int)(v111 + 1);
              else
                v185 -= ((*(_DWORD *)v113 & 4) == 0) - 1;
              v112 = (unsigned int)(v112 + 1);
            }
          }
          ++v113;
        }
        while ( v109 != v113 );
        v5 = v114;
        v122 = v185;
      }
      *(_QWORD *)(*(_QWORD *)v5[4] + 48LL) = v112;
      *(_QWORD *)(*(_QWORD *)v5[4] + 56LL) = v111;
      *(_QWORD *)(*(_QWORD *)v5[4] + 64LL) = v122;
      v123 = *(_QWORD *)(v179 + 64);
      for ( j = v123 + 16LL * *(unsigned int *)(v179 + 72); j != v123; v123 += 16 )
      {
        if ( (*(_QWORD *)v123 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v125 = *(_QWORD *)(*(_QWORD *)v123 & 0xFFFFFFFFFFFFFFF8LL);
          v126 = (__int64 *)v193[7];
          if ( v126 )
          {
            v127 = v193 + 6;
            do
            {
              while ( 1 )
              {
                v128 = v126[2];
                v129 = v126[3];
                if ( v125 <= v126[4] )
                  break;
                v126 = (__int64 *)v126[3];
                if ( !v129 )
                  goto LABEL_133;
              }
              v127 = v126;
              v126 = (__int64 *)v126[2];
            }
            while ( v128 );
LABEL_133:
            if ( v127 != v193 + 6 && v125 >= v127[4] )
            {
              sub_A188E0(v5[4], *((unsigned int *)v127 + 10));
              sub_A188E0(v5[4], *(_BYTE *)(v123 + 8) & 7 | (unsigned __int64)((8 * (*(_BYTE *)(v123 + 8) >> 3)) & 8));
            }
          }
        }
      }
      v60 = 5;
      v59 = v5[4];
      v62 = *(_DWORD *)v5[13];
      v61 = *(_QWORD *)v5[1];
      goto LABEL_54;
    }
    v168 = v5;
    v177 = 0;
    v135 = *v95;
LABEL_149:
    sub_A188E0((__int64)&v202, *(_QWORD *)v135);
    v195 = *(_DWORD *)(v135 + 16);
    if ( v195 > 0x40 )
      sub_C43780(&v194, v135 + 8);
    else
      v194 = *(_QWORD *)(v135 + 8);
    v197 = *(_DWORD *)(v135 + 32);
    if ( v197 > 0x40 )
      sub_C43780(&v196, v135 + 24);
    else
      v196 = *(__int64 **)(v135 + 24);
    sub_AB4E00(&v198, &v194, 64);
    if ( v195 > 0x40 && v194 )
      j_j___libc_free_0_0(v194);
    v194 = (__int64)v198;
    v136 = v199;
    v199 = 0;
    v195 = v136;
    if ( v197 > 0x40 && v196 )
    {
      j_j___libc_free_0_0(v196);
      v196 = v200;
      v197 = (unsigned int)v201;
      if ( v199 > 0x40 && v198 )
        j_j___libc_free_0_0(v198);
    }
    else
    {
      v196 = v200;
      v197 = (unsigned int)v201;
    }
    v137 = &v194;
    if ( v195 > 0x40 )
      v137 = (__int64 *)v194;
    v138 = *v137;
    if ( *v137 < 0 )
      sub_A188E0((__int64)&v202, -2 * v138 + 1);
    else
      sub_A188E0((__int64)&v202, 2 * v138);
    v139 = (__int64 *)&v196;
    if ( v197 > 0x40 )
      v139 = v196;
    v140 = *v139;
    if ( *v139 < 0 )
      sub_A188E0((__int64)&v202, -2 * v140 + 1);
    else
      sub_A188E0((__int64)&v202, 2 * v140);
    if ( v197 > 0x40 && v196 )
      j_j___libc_free_0_0(v196);
    if ( v195 > 0x40 && v194 )
      j_j___libc_free_0_0(v194);
    v43 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v135 + 48) - *(_QWORD *)(v135 + 40)) >> 4);
    sub_A188E0((__int64)&v202, v43);
    v190 = *(__int64 **)(v135 + 48);
    if ( *(__int64 **)(v135 + 40) == v190 )
    {
LABEL_229:
      v177 = v202.m128i_u32[2];
      goto LABEL_189;
    }
    v141 = *(__int64 **)(v135 + 40);
    while ( 1 )
    {
      v142 = v202.m128i_u32[2];
      v143 = *v141;
      v144 = v202.m128i_u32[2] + 1LL;
      if ( v144 > v202.m128i_u32[3] )
      {
        v43 = (__int64)v203;
        v174 = *v141;
        sub_C8D5F0(&v202, v203, v144, 8);
        v142 = v202.m128i_u32[2];
        v143 = v174;
      }
      *(_QWORD *)(v202.m128i_i64[0] + 8 * v142) = v143;
      v145 = (unsigned int)++v202.m128i_i32[2];
      if ( (v141[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_180;
      v149 = *(_QWORD *)(v141[1] & 0xFFFFFFFFFFFFFFF8LL);
      v150 = (__int64 *)v176[7];
      if ( !v150 )
        goto LABEL_180;
      v151 = v176 + 6;
      do
      {
        while ( 1 )
        {
          v43 = v150[2];
          v152 = v150[3];
          if ( v149 <= v150[4] )
            break;
          v150 = (__int64 *)v150[3];
          if ( !v152 )
            goto LABEL_198;
        }
        v151 = v150;
        v150 = (__int64 *)v150[2];
      }
      while ( v43 );
LABEL_198:
      if ( v151 == v176 + 6 || v149 < v151[4] )
      {
LABEL_180:
        v146 = (unsigned int)v145;
        if ( v177 == (unsigned __int64)(unsigned int)v145 )
        {
          v177 = v145;
        }
        else
        {
          if ( v177 >= (unsigned __int64)(unsigned int)v145 )
          {
            if ( v177 > (unsigned __int64)v202.m128i_u32[3] )
            {
              v43 = (__int64)v203;
              sub_C8D5F0(&v202, v203, v177, 8);
              v146 = v202.m128i_u32[2];
            }
            v147 = (_QWORD *)(v202.m128i_i64[0] + 8 * v146);
            for ( k = v202.m128i_i64[0] + 8LL * v177; (_QWORD *)k != v147; ++v147 )
            {
              if ( v147 )
                *v147 = 0;
            }
          }
          v202.m128i_i32[2] = v177;
        }
LABEL_189:
        v135 += 64;
        if ( v171 == v135 )
        {
          v5 = v168;
          if ( v177 )
          {
            v43 = 25;
            sub_A1FB70(v170, 0x19u, (__int64)&v202, 0);
          }
          goto LABEL_106;
        }
        goto LABEL_149;
      }
      v153 = *((unsigned int *)v151 + 10);
      if ( v145 + 1 > (unsigned __int64)v202.m128i_u32[3] )
      {
        v175 = *((unsigned int *)v151 + 10);
        sub_C8D5F0(&v202, v203, v145 + 1, 8);
        v145 = v202.m128i_u32[2];
        v153 = v175;
      }
      *(_QWORD *)(v202.m128i_i64[0] + 8 * v145) = v153;
      ++v202.m128i_i32[2];
      v195 = *((_DWORD *)v141 + 6);
      if ( v195 > 0x40 )
        sub_C43780(&v194, v141 + 2);
      else
        v194 = v141[2];
      v197 = *((_DWORD *)v141 + 10);
      if ( v197 > 0x40 )
        sub_C43780(&v196, v141 + 4);
      else
        v196 = (__int64 *)v141[4];
      v43 = (__int64)&v194;
      sub_AB4E00(&v198, &v194, 64);
      if ( v195 > 0x40 && v194 )
        j_j___libc_free_0_0(v194);
      v194 = (__int64)v198;
      v154 = v199;
      v199 = 0;
      v195 = v154;
      if ( v197 > 0x40 && v196 )
      {
        j_j___libc_free_0_0(v196);
        v196 = v200;
        v197 = (unsigned int)v201;
        if ( v199 > 0x40 && v198 )
          j_j___libc_free_0_0(v198);
      }
      else
      {
        v196 = v200;
        v197 = (unsigned int)v201;
      }
      v155 = &v194;
      if ( v195 > 0x40 )
        v155 = (__int64 *)v194;
      v156 = v202.m128i_u32[2];
      v157 = *v155;
      v158 = v202.m128i_u32[2] + 1LL;
      if ( v157 < 0 )
      {
        v159 = -2 * v157 + 1;
        if ( v158 <= v202.m128i_u32[3] )
          goto LABEL_218;
      }
      else
      {
        v159 = 2 * v157;
        if ( v158 <= v202.m128i_u32[3] )
          goto LABEL_218;
      }
      v43 = (__int64)v203;
      v172 = v159;
      sub_C8D5F0(&v202, v203, v202.m128i_u32[2] + 1LL, 8);
      v156 = v202.m128i_u32[2];
      v159 = v172;
LABEL_218:
      *(_QWORD *)(v202.m128i_i64[0] + 8 * v156) = v159;
      v160 = (__int64 *)&v196;
      v161 = (unsigned int)++v202.m128i_i32[2];
      if ( v197 > 0x40 )
        v160 = v196;
      v162 = *v160;
      v163 = v161 + 1;
      if ( v162 < 0 )
      {
        v164 = -2 * v162 + 1;
        if ( v163 <= v202.m128i_u32[3] )
          goto LABEL_222;
      }
      else
      {
        v164 = 2 * v162;
        if ( v163 <= v202.m128i_u32[3] )
          goto LABEL_222;
      }
      v43 = (__int64)v203;
      v173 = v164;
      sub_C8D5F0(&v202, v203, v161 + 1, 8);
      v161 = v202.m128i_u32[2];
      v164 = v173;
LABEL_222:
      *(_QWORD *)(v202.m128i_i64[0] + 8 * v161) = v164;
      ++v202.m128i_i32[2];
      if ( v197 > 0x40 && v196 )
        j_j___libc_free_0_0(v196);
      if ( v195 > 0x40 && v194 )
        j_j___libc_free_0_0(v194);
      v141 += 6;
      if ( v190 == v141 )
        goto LABEL_229;
    }
  }
  sub_A188E0(v5[4], v184);
  v44 = v5[4];
  v45 = v5[1] + 152;
  v46 = *(_QWORD *)(v179 + 32);
  v202.m128i_i64[0] = *(_QWORD *)(v179 + 24);
  v202.m128i_i64[1] = v46;
  v47 = sub_A2B8A0(v45, &v202);
  sub_A188E0(v44, v47->m128i_i64[0]);
  v48 = v5[4];
  v49 = *(_QWORD *)(*(_QWORD *)v5[5] + 24LL);
  if ( v49 )
    v178 = sub_A15B10(*(_QWORD *)v49, *(_QWORD *)(v49 + 8), v179) != 0;
  v50 = *(unsigned __int8 *)(v179 + 12);
  LOBYTE(v50) = (unsigned __int8)v50 >> 4;
  sub_A188E0(
    v48,
    *(_BYTE *)(v179 + 12) & 0xF
  | (unsigned __int64)((v50 << 8) & 0x300)
  | (16
   * (((*(_BYTE *)(v179 + 12) & 0x40) != 0)
    | (unsigned __int64)((8 * (*(_BYTE *)(v179 + 13) >> 1)) & 8)
    | (4 * (*(_BYTE *)(v179 + 13) & 1))
    | (2 * (*(_BYTE *)(v179 + 12) >> 7))))
  | ((v178 || (*(_BYTE *)(v179 + 13) & 4) != 0) << 10) & 0x3FC00);
  sub_A188E0(
    v5[4],
    (8 * ((*(_BYTE *)(v179 + 64) >> 3) & 3))
  | *(_BYTE *)(v179 + 64) & 1
  | (2 * ((*(_BYTE *)(v179 + 64) & 2) != 0))
  | (4 * ((*(_BYTE *)(v179 + 64) & 4) != 0)));
  v51 = *(_QWORD **)(v179 + 40);
  v52 = (__int64 *)v5[1];
  for ( m = &v51[*(unsigned int *)(v179 + 48)]; m != v51; ++v51 )
  {
    v54 = *(_QWORD *)(*v51 & 0xFFFFFFFFFFFFFFF8LL);
    v55 = (__int64 *)v52[7];
    if ( v55 )
    {
      v56 = v52 + 6;
      do
      {
        while ( 1 )
        {
          v57 = v55[2];
          v58 = v55[3];
          if ( v54 <= v55[4] )
            break;
          v55 = (__int64 *)v55[3];
          if ( !v58 )
            goto LABEL_49;
        }
        v56 = v55;
        v55 = (__int64 *)v55[2];
      }
      while ( v57 );
LABEL_49:
      if ( v56 != v52 + 6 && v54 >= v56[4] )
      {
        sub_A188E0(v5[4], *((unsigned int *)v56 + 10));
        v52 = (__int64 *)v5[1];
      }
    }
  }
  v59 = v5[4];
  v60 = 6;
  v61 = *v52;
  v62 = *(_DWORD *)v5[6];
LABEL_54:
  sub_A1FB70(v61, v60, v59, v62);
  *(_DWORD *)(v5[4] + 8) = 0;
  v63 = v5[7];
  result = *(__int64 **)v63;
  if ( !*(_QWORD *)(*(_QWORD *)v63 + 32LL) )
  {
    result = (__int64 *)((*(_BYTE *)(v192 + 12) & 0xFu) - 7);
    if ( (unsigned int)result <= 1 )
    {
      sub_A188E0(*(_QWORD *)(v63 + 8), *(_QWORD *)(v192 + 16));
      sub_A1FB70(**(_QWORD **)v63, 9u, *(_QWORD *)(v63 + 8), 0);
      result = *(__int64 **)(v63 + 8);
      *((_DWORD *)result + 2) = 0;
    }
  }
  return result;
}
