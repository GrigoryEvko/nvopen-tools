// Function: sub_19D4270
// Address: 0x19d4270
//
// bad sp value at call has been detected, the output may be wrong!
__int64 __fastcall sub_19D4270(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // r14
  __int64 v25; // rcx
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // r15
  __m128i v29; // xmm0
  __m128i v30; // xmm1
  bool v31; // dl
  bool v32; // zf
  __int64 v33; // rcx
  __m128i *v34; // rdi
  __m128i *v35; // rsi
  __int64 v36; // r10
  bool v37; // cc
  __m128i *v38; // rsi
  __m128i *v39; // rdi
  __int64 j; // rcx
  __m128i *v41; // rdi
  __m128i *v42; // rsi
  __int64 v43; // rcx
  __int64 k; // rcx
  const __m128i *v45; // r13
  __int64 v46; // rbx
  char v47; // dl
  __int64 v48; // r8
  int v49; // r9d
  unsigned int v50; // eax
  __int64 *v51; // rsi
  __m128i v52; // xmm6
  __m128i v53; // xmm7
  __int64 v54; // rcx
  __int64 v55; // r8
  unsigned __int8 v56; // al
  __int64 v57; // r12
  __m128i *v58; // rdi
  __m128i *v59; // rsi
  __m128i v60; // xmm2
  __m128i v61; // xmm3
  __int64 v62; // rax
  __m128i v63; // xmm1
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  int v67; // eax
  unsigned int v68; // eax
  __int64 v69; // r12
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // r15
  __m128i *v73; // rdi
  __m128i *v74; // rsi
  __m128i *v75; // rdi
  __m128i *v76; // rsi
  __m128i *v77; // rdi
  __m128i *v78; // rsi
  __m128i *v79; // rdi
  __m128i *v80; // rsi
  unsigned __int64 v81; // r12
  __int64 v82; // r15
  _BYTE *v83; // rbx
  _QWORD *v84; // rdi
  char v85; // r14
  __int64 v86; // rax
  __int64 v87; // rsi
  unsigned int v88; // r8d
  unsigned int v89; // r14d
  __int64 v90; // rax
  __int64 *v91; // rax
  _QWORD *v92; // r14
  __int64 v93; // rsi
  __int64 v94; // rdx
  __int64 *v95; // rax
  __int64 *v96; // rdi
  int v97; // r8d
  __int64 v98; // r15
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // r12
  __int64 *v102; // r15
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // r12
  __int64 v106; // r8
  __int64 v107; // r8
  __int64 v108; // r8
  __int64 v109; // rdx
  unsigned __int64 v110; // rdx
  __int64 v111; // rax
  unsigned __int64 v112; // rcx
  _QWORD *v113; // rbx
  __int64 v114; // r14
  __int64 v115; // r12
  __m128i v116; // xmm2
  __m128i v117; // xmm3
  int v118; // edx
  int v119; // r14d
  int v120; // r14d
  __int64 v121; // rax
  int v122; // eax
  int v123; // esi
  unsigned int v124; // r14d
  __int64 v125; // rax
  __int64 *v126; // rax
  __int64 v127; // rax
  __m128i *v128; // rdi
  __int64 v129; // rcx
  __int32 *v130; // rsi
  signed __int64 v131; // rax
  __m128i *v132; // rdi
  __int64 v133; // rcx
  __int32 *v134; // rsi
  __m128i *v135; // rdi
  __int64 v136; // rcx
  __int32 *v137; // rsi
  __int64 *v138; // rdx
  int v139; // edi
  __int64 v140; // r8
  int v141; // r10d
  unsigned int m; // edx
  __int64 v143; // r8
  _QWORD *v144; // rcx
  int v145; // edi
  unsigned int i; // eax
  _QWORD *v147; // rdx
  __int64 v148; // r8
  unsigned int v149; // eax
  unsigned int v150; // ecx
  unsigned int v151; // edx
  bool v152; // [rsp+Fh] [rbp-3C1h]
  __int64 v153; // [rsp+30h] [rbp-3A0h]
  const __m128i *v154; // [rsp+38h] [rbp-398h]
  __int64 v155; // [rsp+40h] [rbp-390h]
  __int64 *v156; // [rsp+40h] [rbp-390h]
  _QWORD *v157; // [rsp+48h] [rbp-388h]
  __int64 v158; // [rsp+50h] [rbp-380h]
  _QWORD *v159; // [rsp+60h] [rbp-370h]
  _QWORD *v160; // [rsp+68h] [rbp-368h]
  __int64 v161; // [rsp+68h] [rbp-368h]
  bool v162; // [rsp+68h] [rbp-368h]
  __int64 v163; // [rsp+70h] [rbp-360h]
  __int64 v164; // [rsp+70h] [rbp-360h]
  unsigned int v165; // [rsp+70h] [rbp-360h]
  _QWORD *v166; // [rsp+70h] [rbp-360h]
  unsigned int v167; // [rsp+70h] [rbp-360h]
  _QWORD *v168; // [rsp+78h] [rbp-358h]
  _QWORD *v169; // [rsp+78h] [rbp-358h]
  _QWORD *v170; // [rsp+78h] [rbp-358h]
  __int64 v171; // [rsp+78h] [rbp-358h]
  __int16 v172; // [rsp+80h] [rbp-350h]
  __int64 v173; // [rsp+80h] [rbp-350h]
  __int64 *v174; // [rsp+80h] [rbp-350h]
  _QWORD *v175; // [rsp+80h] [rbp-350h]
  __int16 v176; // [rsp+80h] [rbp-350h]
  __int64 v177; // [rsp+80h] [rbp-350h]
  unsigned __int64 v178; // [rsp+80h] [rbp-350h]
  unsigned int v179; // [rsp+88h] [rbp-348h]
  unsigned __int64 v180; // [rsp+88h] [rbp-348h]
  char v181; // [rsp+88h] [rbp-348h]
  __int64 v182; // [rsp+88h] [rbp-348h]
  unsigned __int64 v183; // [rsp+88h] [rbp-348h]
  __int64 v184; // [rsp+88h] [rbp-348h]
  __int64 v185; // [rsp+90h] [rbp-340h]
  __int64 v186; // [rsp+90h] [rbp-340h]
  int v187; // [rsp+90h] [rbp-340h]
  __int64 v188; // [rsp+98h] [rbp-338h]
  _QWORD *v189; // [rsp+98h] [rbp-338h]
  _QWORD *v190; // [rsp+98h] [rbp-338h]
  _QWORD *v191; // [rsp+98h] [rbp-338h]
  __int64 v192; // [rsp+A0h] [rbp-330h] BYREF
  __int64 v193; // [rsp+A8h] [rbp-328h]
  __int64 v194; // [rsp+B0h] [rbp-320h]
  __int64 v195; // [rsp+B8h] [rbp-318h]
  __m128i v196; // [rsp+C0h] [rbp-310h] BYREF
  __m128i v197; // [rsp+D0h] [rbp-300h] BYREF
  __int64 v198; // [rsp+E0h] [rbp-2F0h]
  __m128i v199[3]; // [rsp+F0h] [rbp-2E0h] BYREF
  __m128i v200; // [rsp+120h] [rbp-2B0h] BYREF
  __m128i v201; // [rsp+130h] [rbp-2A0h] BYREF
  __int64 v202; // [rsp+140h] [rbp-290h]
  __m128i v203; // [rsp+150h] [rbp-280h] BYREF
  __m128i v204; // [rsp+160h] [rbp-270h] BYREF
  __int64 v205; // [rsp+170h] [rbp-260h]
  __m128i v206; // [rsp+180h] [rbp-250h] BYREF
  __m128i v207; // [rsp+190h] [rbp-240h] BYREF
  __int64 v208; // [rsp+1A0h] [rbp-230h]
  char v209; // [rsp+1A8h] [rbp-228h]
  _BYTE *v210; // [rsp+1B0h] [rbp-220h] BYREF
  __int64 v211; // [rsp+1B8h] [rbp-218h]
  _BYTE v212[64]; // [rsp+1C0h] [rbp-210h] BYREF
  __m128i v213; // [rsp+200h] [rbp-1D0h] BYREF
  __m128i v214; // [rsp+210h] [rbp-1C0h] BYREF
  __int64 v215; // [rsp+220h] [rbp-1B0h]
  __m128i v216; // [rsp+250h] [rbp-180h] BYREF
  __m128i v217; // [rsp+260h] [rbp-170h] BYREF
  __int64 v218; // [rsp+270h] [rbp-160h]
  int v219; // [rsp+278h] [rbp-158h]
  __int64 v220; // [rsp+280h] [rbp-150h]
  __int64 v221; // [rsp+288h] [rbp-148h]

  v4 = a2;
  if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
  {
    a2 = 9;
    if ( sub_1625790(v4, 9) )
      return 0;
  }
  v7 = sub_15F2050(v4);
  v8 = sub_1632FA0(v7);
  v9 = *(_QWORD *)(v4 - 48);
  v188 = v8;
  if ( *(_BYTE *)(v9 + 16) != 54 )
  {
LABEL_6:
    v10 = sub_14ABE30((unsigned __int8 *)v9);
    v11 = v10;
    if ( v10 )
    {
      v12 = sub_19D0490((__int64 *)a1, v4, *(_QWORD *)(v4 - 24), v10);
      if ( v12 )
      {
        *a3 = v12 + 24;
        return 1;
      }
      v13 = *(_QWORD *)v9;
      switch ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) )
      {
        case 0xD:
          v20 = 8LL * *(_QWORD *)sub_15A9930(v188, v13);
          goto LABEL_11;
        case 0xE:
          v185 = *(_QWORD *)(v13 + 32);
          v173 = *(_QWORD *)(v13 + 24);
          v180 = (unsigned int)sub_15A9FE0(v188, v173);
          v20 = 8 * v185 * v180 * ((v180 + ((unsigned __int64)(sub_127FA20(v188, v173) + 7) >> 3) - 1) / v180);
LABEL_11:
          v14 = (unsigned __int64)(v20 + 7) >> 3;
          v179 = 1 << (*(unsigned __int16 *)(v4 + 18) >> 1) >> 1;
          if ( v179 )
          {
            v15 = sub_16498A0(v4);
            v16 = *(_QWORD *)(v4 + 48);
            v216.m128i_i64[0] = 0;
            v217.m128i_i64[1] = v15;
            v17 = *(_QWORD *)(v4 + 40);
            v218 = 0;
            v216.m128i_i64[1] = v17;
            v217.m128i_i64[0] = v4 + 24;
            v219 = 0;
            v220 = 0;
            v221 = 0;
            v213.m128i_i64[0] = v16;
            if ( v16 )
            {
              sub_1623A60((__int64)&v213, v16, 2);
              if ( v216.m128i_i64[0] )
                sub_161E7C0((__int64)&v216, v216.m128i_i64[0]);
              v216.m128i_i64[0] = v213.m128i_i64[0];
              if ( v213.m128i_i64[0] )
                sub_1623210((__int64)&v213, (unsigned __int8 *)v213.m128i_i64[0], (__int64)&v216);
            }
            v189 = *(_QWORD **)(v4 - 24);
            v172 = *(_WORD *)(v4 + 18) & 1;
            v18 = sub_1643360((_QWORD *)v217.m128i_i64[1]);
            v19 = (__int64 *)sub_159C470(v18, v14, 0);
            sub_15E7280(v216.m128i_i64, v189, v11, v19, v179, v172, 0, 0, 0);
            sub_14191F0(*(_QWORD *)a1, v4);
            sub_15F20C0((_QWORD *)v4);
            JUMPOUT(0x19D44F6);
          }
          JUMPOUT(0x19D46A0);
        default:
          return 0;
      }
    }
    return 0;
  }
  v21 = *(_QWORD *)(v4 - 48);
  if ( sub_15F32D0(v9)
    || (v181 = *(_BYTE *)(v9 + 18) & 1) != 0
    || (v23 = *(_QWORD *)(v9 + 8)) == 0
    || *(_QWORD *)(v23 + 8)
    || *(_QWORD *)(v9 + 40) != *(_QWORD *)(v4 + 40) )
  {
LABEL_27:
    v9 = *(_QWORD *)(v4 - 48);
    goto LABEL_6;
  }
  v153 = *(_QWORD *)v9;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)v9 + 8LL) - 13 > 1 )
    goto LABEL_151;
  if ( !*(_QWORD *)(a1 + 32) )
LABEL_255:
    sub_4263D6(v21, a2, v22);
  v24 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(a1 + 40))(a1 + 16);
  sub_141EB40(&v196, (__int64 *)v9);
  v25 = *(_QWORD *)(v9 + 32);
  if ( v4 + 24 == v25 )
    goto LABEL_178;
  v174 = (__int64 *)v9;
  v26 = v4 + 24;
  v163 = v4;
  v27 = v25;
  v160 = a3;
  do
  {
    v28 = 0;
    v29 = _mm_loadu_si128(&v196);
    v30 = _mm_loadu_si128(&v197);
    LOBYTE(v219) = 1;
    if ( v27 )
      v28 = v27 - 24;
    v216 = v29;
    v218 = v198;
    v217 = v30;
    if ( (sub_13575E0(v24, v28, &v216, v25) & 2) != 0 )
    {
      v4 = v163;
      v9 = (__int64)v174;
      v164 = v28;
      v31 = v28 != 0;
      v32 = v4 == v28;
      a3 = v160;
      v152 = !v32 && v31;
      if ( !v152 )
      {
        if ( v164 )
        {
          v159 = (_QWORD *)(v164 + 24);
          goto LABEL_111;
        }
        goto LABEL_151;
      }
      sub_141EDF0(v199, v4);
      LOBYTE(v219) = 1;
      v33 = 10;
      v34 = &v216;
      v35 = v199;
      while ( v33 )
      {
        v34->m128i_i32[0] = v35->m128i_i32[0];
        v35 = (__m128i *)((char *)v35 + 4);
        v34 = (__m128i *)((char *)v34 + 4);
        --v33;
      }
      if ( (sub_13575E0(v24, v164, &v216, 0) & 3) != 0 )
        goto LABEL_151;
      v194 = 0;
      v36 = *(_QWORD *)(v4 - 24);
      v195 = 0;
      v192 = 0;
      v37 = *(_BYTE *)(v36 + 16) <= 0x17u;
      v193 = 0;
      if ( !v37 && *(_QWORD *)(v36 + 40) == *(_QWORD *)(v4 + 40) )
      {
        v171 = v36;
        v192 = 1;
        sub_1467110((__int64)&v192, 0);
        if ( !(_DWORD)v195 )
LABEL_272:
          JUMPOUT(0x41CDFA);
        v144 = 0;
        v145 = 1;
        for ( i = (v195 - 1) & (((unsigned int)v171 >> 9) ^ ((unsigned int)v171 >> 4)); ; i = (v195 - 1) & v149 )
        {
          v147 = (_QWORD *)(v193 + 8LL * i);
          v148 = *v147;
          if ( v171 == *v147 )
            break;
          if ( v148 == -8 )
          {
            if ( v144 )
              v147 = v144;
            break;
          }
          if ( v144 || v148 != -16 )
            v147 = v144;
          v149 = v145 + i;
          v144 = v147;
          ++v145;
        }
        LODWORD(v194) = v194 + 1;
        if ( *v147 != -8 )
          --HIDWORD(v194);
        *v147 = v171;
      }
      v38 = v199;
      v39 = &v213;
      for ( j = 10; j; --j )
      {
        v39->m128i_i32[0] = v38->m128i_i32[0];
        v38 = (__m128i *)((char *)v38 + 4);
        v39 = (__m128i *)((char *)v39 + 4);
      }
      v41 = &v217;
      v42 = &v213;
      v43 = 10;
      v216.m128i_i64[0] = (__int64)&v217;
      while ( v43 )
      {
        v41->m128i_i32[0] = v42->m128i_i32[0];
        v42 = (__m128i *)((char *)v42 + 4);
        v41 = (__m128i *)((char *)v41 + 4);
        --v43;
      }
      v210 = v212;
      v216.m128i_i64[1] = 0x800000001LL;
      v211 = 0x800000000LL;
      v213.m128i_i64[0] = (__int64)&v214;
      v213.m128i_i64[1] = 0x800000000LL;
      sub_141EB40(&v200, v174);
      k = v164 + 24;
      v168 = (_QWORD *)(*(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      v159 = (_QWORD *)(v164 + 24);
      if ( v168 == (_QWORD *)(v164 + 24) )
        goto LABEL_99;
      v158 = a1;
      v161 = v4;
      v157 = a3;
      v45 = &v206;
      while ( 1 )
      {
        v209 = 0;
        v46 = (__int64)(v168 - 3);
        if ( !v168 )
          v46 = 0;
        v47 = sub_13575E0(v24, v46, v45, k) & 3;
        if ( (_DWORD)v195 )
        {
          k = (unsigned int)(v195 - 1);
          v50 = k & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v51 = (__int64 *)(v193 + 8LL * v50);
          v48 = *v51;
          if ( *v51 == v46 )
          {
LABEL_55:
            *v51 = -16;
            LODWORD(v194) = v194 - 1;
            ++HIDWORD(v194);
            if ( v47 )
            {
LABEL_56:
              v52 = _mm_loadu_si128(&v200);
              v53 = _mm_loadu_si128(&v201);
              v209 = 1;
              v208 = v202;
              v206 = v52;
              v207 = v53;
              if ( (sub_13575E0(v24, v46, v45, k) & 2) != 0 )
                goto LABEL_201;
              v56 = *(_BYTE *)(v46 + 16);
              if ( v56 > 0x17u && ((v57 = v46 | 4, v56 == 78) || (v57 = v46, v56 == 29)) )
              {
                if ( (v57 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (sub_134F8C0(v24, v164, v57, v54, v55) & 3) != 0 )
                  goto LABEL_201;
                if ( v213.m128i_i32[2] >= (unsigned __int32)v213.m128i_i32[3] )
                  sub_16CD150((__int64)&v213, &v214, 0, 8, v48, v49);
                *(_QWORD *)(v213.m128i_i64[0] + 8LL * v213.m128i_u32[2]) = v57;
                ++v213.m128i_i32[2];
              }
              else
              {
                if ( (unsigned __int8)(v56 - 54) > 1u && v56 != 82 )
                  goto LABEL_201;
                switch ( *(_BYTE *)(v46 + 16) )
                {
                  case '6':
                    sub_141EB40(&v203, (__int64 *)v46);
                    v54 = 10;
                    v58 = (__m128i *)v45;
                    v59 = &v203;
                    while ( v54 )
                    {
                      v58->m128i_i32[0] = v59->m128i_i32[0];
                      v59 = (__m128i *)((char *)v59 + 4);
                      v58 = (__m128i *)((char *)v58 + 4);
                      --v54;
                    }
                    break;
                  case '7':
                    sub_141EDF0(&v203, v46);
                    v54 = 10;
                    v79 = (__m128i *)v45;
                    v80 = &v203;
                    while ( v54 )
                    {
                      v79->m128i_i32[0] = v80->m128i_i32[0];
                      v80 = (__m128i *)((char *)v80 + 4);
                      v79 = (__m128i *)((char *)v79 + 4);
                      --v54;
                    }
                    break;
                  case ':':
                    sub_141F110(&v203, v46);
                    v54 = 10;
                    v77 = (__m128i *)v45;
                    v78 = &v203;
                    while ( v54 )
                    {
                      v77->m128i_i32[0] = v78->m128i_i32[0];
                      v78 = (__m128i *)((char *)v78 + 4);
                      v77 = (__m128i *)((char *)v77 + 4);
                      --v54;
                    }
                    break;
                  case ';':
                    sub_141F3C0(&v203, v46);
                    v54 = 10;
                    v75 = (__m128i *)v45;
                    v76 = &v203;
                    while ( v54 )
                    {
                      v75->m128i_i32[0] = v76->m128i_i32[0];
                      v76 = (__m128i *)((char *)v76 + 4);
                      v75 = (__m128i *)((char *)v75 + 4);
                      --v54;
                    }
                    break;
                  case 'R':
                    sub_141F0A0(&v203, v46);
                    v54 = 10;
                    v73 = (__m128i *)v45;
                    v74 = &v203;
                    while ( v54 )
                    {
                      v73->m128i_i32[0] = v74->m128i_i32[0];
                      v74 = (__m128i *)((char *)v74 + 4);
                      v73 = (__m128i *)((char *)v73 + 4);
                      --v54;
                    }
                    break;
                  default:
                    break;
                }
                v60 = _mm_loadu_si128(&v206);
                v61 = _mm_loadu_si128(&v207);
                v209 = 1;
                v203 = v60;
                v205 = v208;
                v204 = v61;
                if ( (sub_13575E0(v24, v164, v45, v54) & 3) != 0 )
                {
LABEL_201:
                  v9 = (__int64)v174;
                  a1 = v158;
                  v4 = v161;
                  a3 = v157;
                  goto LABEL_104;
                }
                v62 = v216.m128i_u32[2];
                if ( v216.m128i_i32[2] >= (unsigned __int32)v216.m128i_i32[3] )
                {
                  sub_16CD150((__int64)&v216, &v217, 0, 40, v48, v49);
                  v62 = v216.m128i_u32[2];
                }
                v63 = _mm_loadu_si128(&v204);
                v64 = v216.m128i_i64[0] + 40 * v62;
                v65 = v205;
                *(__m128i *)v64 = _mm_loadu_si128(&v203);
                *(_QWORD *)(v64 + 32) = v65;
                *(__m128i *)(v64 + 16) = v63;
                ++v216.m128i_i32[2];
              }
            }
            v66 = (unsigned int)v211;
            if ( (unsigned int)v211 >= HIDWORD(v211) )
            {
              sub_16CD150((__int64)&v210, v212, 0, 8, v48, v49);
              v66 = (unsigned int)v211;
            }
            *(_QWORD *)&v210[8 * v66] = v46;
            v67 = *(_DWORD *)(v46 + 20);
            LODWORD(v211) = v211 + 1;
            v68 = v67 & 0xFFFFFFF;
            if ( v68 )
            {
              v69 = 0;
              v154 = v45;
              v70 = 24LL * v68;
              while ( 1 )
              {
                if ( (*(_BYTE *)(v46 + 23) & 0x40) != 0 )
                  v71 = *(_QWORD *)(v46 - 8);
                else
                  v71 = v46 - 24LL * (*(_DWORD *)(v46 + 20) & 0xFFFFFFF);
                v72 = *(_QWORD *)(v71 + v69);
                if ( *(_BYTE *)(v72 + 16) > 0x17u && *(_QWORD *)(v72 + 40) == *(_QWORD *)(v161 + 40) )
                {
                  if ( !(_DWORD)v195 )
                  {
                    ++v192;
                    goto LABEL_228;
                  }
                  v94 = ((_DWORD)v195 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
                  v95 = (__int64 *)(v193 + 8 * v94);
                  k = *v95;
                  if ( *v95 != v72 )
                  {
                    v96 = 0;
                    v97 = 1;
                    while ( k != -8 )
                    {
                      if ( !v96 && k == -16 )
                        v96 = v95;
                      LODWORD(v94) = (v195 - 1) & (v97 + v94);
                      v95 = (__int64 *)(v193 + 8LL * (unsigned int)v94);
                      k = *v95;
                      if ( v72 == *v95 )
                        goto LABEL_77;
                      ++v97;
                    }
                    if ( v96 )
                      v95 = v96;
                    ++v192;
                    if ( 4 * ((int)v194 + 1) < (unsigned int)(3 * v195) )
                    {
                      k = (unsigned int)v195 >> 3;
                      if ( (int)v195 - HIDWORD(v194) - ((int)v194 + 1) <= (unsigned int)k )
                      {
                        sub_1467110((__int64)&v192, v195);
                        if ( !(_DWORD)v195 )
                          goto LABEL_272;
                        v138 = 0;
                        v139 = 1;
                        for ( k = ((_DWORD)v195 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
                              ;
                              k = ((_DWORD)v195 - 1) & v150 )
                        {
                          v95 = (__int64 *)(v193 + 8LL * (unsigned int)k);
                          v140 = *v95;
                          if ( v72 == *v95 )
                            break;
                          if ( v140 == -8 )
                          {
                            if ( v138 )
                              v95 = v138;
                            break;
                          }
                          if ( v138 || v140 != -16 )
                            v95 = v138;
                          v150 = v139 + k;
                          v138 = v95;
                          ++v139;
                        }
                      }
LABEL_130:
                      LODWORD(v194) = v194 + 1;
                      if ( *v95 != -8 )
                        --HIDWORD(v194);
                      *v95 = v72;
                      goto LABEL_77;
                    }
LABEL_228:
                    sub_1467110((__int64)&v192, 2 * v195);
                    if ( !(_DWORD)v195 )
                      goto LABEL_272;
                    k = 0;
                    v141 = 1;
                    for ( m = (v195 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4)); ; m = (v195 - 1) & v151 )
                    {
                      v95 = (__int64 *)(v193 + 8LL * m);
                      v143 = *v95;
                      if ( v72 == *v95 )
                        break;
                      if ( v143 == -8 )
                      {
                        if ( k )
                          v95 = (__int64 *)k;
                        goto LABEL_130;
                      }
                      if ( v143 != -16 || k )
                        v95 = (__int64 *)k;
                      v151 = v141 + m;
                      k = (__int64)v95;
                      ++v141;
                    }
                    goto LABEL_130;
                  }
                }
LABEL_77:
                v69 += 24;
                if ( v70 == v69 )
                {
                  v45 = v154;
                  goto LABEL_97;
                }
              }
            }
            goto LABEL_97;
          }
          v123 = 1;
          while ( v48 != -8 )
          {
            v49 = v123 + 1;
            v50 = k & (v123 + v50);
            v51 = (__int64 *)(v193 + 8LL * v50);
            v48 = *v51;
            if ( v46 == *v51 )
              goto LABEL_55;
            v123 = v49;
          }
        }
        if ( v47 )
          break;
LABEL_97:
        v168 = (_QWORD *)(*v168 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v159 == v168 )
        {
          v9 = (__int64)v174;
          a1 = v158;
          v4 = v161;
          a3 = v157;
LABEL_99:
          if ( v210 != &v210[8 * (unsigned int)v211] )
          {
            v182 = v4;
            v81 = (unsigned __int64)v210;
            v175 = a3;
            v82 = a1;
            v83 = &v210[8 * (unsigned int)v211];
            do
            {
              v84 = (_QWORD *)*((_QWORD *)v83 - 1);
              v83 -= 8;
              sub_15F22F0(v84, v164);
            }
            while ( (_BYTE *)v81 != v83 );
            a1 = v82;
            v4 = v182;
            a3 = v175;
          }
          v181 = v152;
LABEL_104:
          if ( (__m128i *)v213.m128i_i64[0] != &v214 )
            _libc_free(v213.m128i_u64[0]);
          if ( (__m128i *)v216.m128i_i64[0] != &v217 )
            _libc_free(v216.m128i_u64[0]);
          if ( v210 != v212 )
            _libc_free((unsigned __int64)v210);
          j___libc_free_0(v193);
          if ( v181 )
            goto LABEL_111;
LABEL_151:
          a2 = v9;
          v109 = sub_141C430(*(_QWORD *)a1, v9, 0);
          if ( (v109 & 7) != 1 )
            goto LABEL_27;
          v110 = v109 & 0xFFFFFFFFFFFFFFF8LL;
          v184 = v110;
          if ( *(_BYTE *)(v110 + 16) != 78 )
            goto LABEL_27;
          v111 = *(_QWORD *)(v110 - 24);
          if ( !*(_BYTE *)(v111 + 16) && (*(_BYTE *)(v111 + 33) & 0x20) != 0 && *(_DWORD *)(v111 + 36) == 133 )
            goto LABEL_27;
          v21 = *(_QWORD *)(v4 - 24);
          v162 = *(_BYTE *)(sub_1649C60(v21) + 16) == 53;
          if ( *(_QWORD *)(a1 + 32) )
          {
            v166 = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(a1 + 40))(a1 + 16);
            sub_141EDF0(&v213, v4);
            if ( (*(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL) != v184 + 24 )
            {
              v177 = a1;
              v113 = (_QWORD *)(*(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL);
              v114 = v4;
              do
              {
                v115 = 0;
                v116 = _mm_loadu_si128(&v213);
                if ( v113 )
                  v115 = (__int64)(v113 - 3);
                v117 = _mm_loadu_si128(&v214);
                LOBYTE(v219) = 1;
                v216 = v116;
                v218 = v215;
                v217 = v117;
                if ( (sub_13575E0(v166, v115, &v216, v112) & 3) != 0 || sub_15F3330(v115) == 1 && !v162 )
                {
                  a1 = v177;
                  v4 = v114;
                  goto LABEL_27;
                }
                v112 = *v113 & 0xFFFFFFFFFFFFFFF8LL;
                v113 = (_QWORD *)v112;
              }
              while ( v184 + 24 != v112 );
              a1 = v177;
              v4 = v114;
            }
            v118 = 1 << (*(unsigned __int16 *)(v4 + 18) >> 1) >> 1;
            if ( !v118 )
              v118 = sub_15A9FE0(v188, **(_QWORD **)(v4 - 48));
            v119 = 1 << (*(unsigned __int16 *)(v9 + 18) >> 1) >> 1;
            if ( !v119 )
            {
              v187 = v118;
              v122 = sub_15A9FE0(v188, *(_QWORD *)v9);
              v118 = v187;
              v119 = v122;
            }
            v120 = v118 | v119;
            v186 = sub_127FA20(v188, **(_QWORD **)(v4 - 48));
            v178 = sub_1649C60(*(_QWORD *)(v9 - 24));
            v121 = sub_1649C60(*(_QWORD *)(v4 - 24));
            if ( (unsigned __int8)sub_19D2360(
                                    a1,
                                    v9,
                                    v121,
                                    v178,
                                    (unsigned __int64)(v186 + 7) >> 3,
                                    v120 & (unsigned int)-v120,
                                    v184) )
            {
              sub_14191F0(*(_QWORD *)a1, v4);
              sub_15F20C0((_QWORD *)v4);
              sub_14191F0(*(_QWORD *)a1, v9);
              sub_15F20C0((_QWORD *)v9);
              return 1;
            }
            goto LABEL_27;
          }
          goto LABEL_255;
        }
      }
      v98 = v216.m128i_i64[0];
      v99 = 40LL * v216.m128i_u32[2];
      v155 = v216.m128i_i64[0] + v99;
      k = 0xCCCCCCCCCCCCCCCDLL;
      v100 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v99 >> 3)) >> 2;
      if ( v100 )
      {
        v101 = v216.m128i_i64[0] + 160 * v100;
        while ( 1 )
        {
          v209 = 1;
          v206 = _mm_loadu_si128((const __m128i *)v98);
          v207 = _mm_loadu_si128((const __m128i *)(v98 + 16));
          v208 = *(_QWORD *)(v98 + 32);
          if ( (sub_13575E0(v24, v46, v45, k) & 3) != 0 )
            break;
          v209 = 1;
          v206 = _mm_loadu_si128((const __m128i *)(v98 + 40));
          v207 = _mm_loadu_si128((const __m128i *)(v98 + 56));
          v208 = *(_QWORD *)(v98 + 72);
          if ( (sub_13575E0(v24, v46, v45, k) & 3) != 0 )
          {
            v98 += 40;
            break;
          }
          v209 = 1;
          v206 = _mm_loadu_si128((const __m128i *)(v98 + 80));
          v207 = _mm_loadu_si128((const __m128i *)(v98 + 96));
          v208 = *(_QWORD *)(v98 + 112);
          if ( (sub_13575E0(v24, v46, v45, k) & 3) != 0 )
          {
            v98 += 80;
            break;
          }
          v209 = 1;
          v206 = _mm_loadu_si128((const __m128i *)(v98 + 120));
          v207 = _mm_loadu_si128((const __m128i *)(v98 + 136));
          v208 = *(_QWORD *)(v98 + 152);
          if ( (sub_13575E0(v24, v46, v45, k) & 3) != 0 )
          {
            v98 += 120;
            break;
          }
          v98 += 160;
          if ( v101 == v98 )
            goto LABEL_191;
        }
LABEL_141:
        if ( v155 != v98 )
          goto LABEL_56;
LABEL_142:
        v102 = (__int64 *)v213.m128i_i64[0];
        v103 = 8LL * v213.m128i_u32[2];
        k = v213.m128i_i64[0] + v103;
        v104 = v103 >> 5;
        v156 = (__int64 *)k;
        if ( v104 )
        {
          v105 = v213.m128i_i64[0] + 32 * v104;
          while ( (sub_134F8C0(v24, v46, *v102, k, v48) & 3) == 0 )
          {
            if ( (sub_134F8C0(v24, v46, v102[1], k, v108) & 3) != 0 )
            {
              ++v102;
              break;
            }
            if ( (sub_134F8C0(v24, v46, v102[2], k, v106) & 3) != 0 )
            {
              v102 += 2;
              break;
            }
            if ( (sub_134F8C0(v24, v46, v102[3], k, v107) & 3) != 0 )
            {
              v102 += 3;
              break;
            }
            v102 += 4;
            if ( (__int64 *)v105 == v102 )
              goto LABEL_202;
          }
LABEL_149:
          if ( v156 != v102 )
            goto LABEL_56;
          goto LABEL_97;
        }
LABEL_202:
        v131 = (char *)v156 - (char *)v102;
        if ( (char *)v156 - (char *)v102 != 16 )
        {
          if ( v131 != 24 )
          {
            if ( v131 != 8 )
              goto LABEL_97;
            goto LABEL_205;
          }
          if ( (sub_134F8C0(v24, v46, *v102, k, v48) & 3) != 0 )
            goto LABEL_149;
          ++v102;
        }
        if ( (sub_134F8C0(v24, v46, *v102, k, v48) & 3) != 0 )
          goto LABEL_149;
        ++v102;
LABEL_205:
        if ( (sub_134F8C0(v24, v46, *v102, k, v48) & 3) == 0 )
          goto LABEL_97;
        goto LABEL_149;
      }
LABEL_191:
      v127 = v155 - v98;
      if ( v155 - v98 != 80 )
      {
        if ( v127 != 120 )
        {
          if ( v127 != 40 )
            goto LABEL_142;
          goto LABEL_194;
        }
        v132 = (__m128i *)v45;
        v209 = 1;
        v133 = 10;
        v134 = (__int32 *)v98;
        while ( v133 )
        {
          v132->m128i_i32[0] = *v134++;
          v132 = (__m128i *)((char *)v132 + 4);
          --v133;
        }
        v45 = &v206;
        if ( (sub_13575E0(v24, v46, &v206, 0) & 3) != 0 )
          goto LABEL_141;
        v98 += 40;
      }
      v135 = (__m128i *)v45;
      v209 = 1;
      v136 = 10;
      v137 = (__int32 *)v98;
      while ( v136 )
      {
        v135->m128i_i32[0] = *v137++;
        v135 = (__m128i *)((char *)v135 + 4);
        --v136;
      }
      v45 = &v206;
      if ( (sub_13575E0(v24, v46, &v206, 0) & 3) != 0 )
        goto LABEL_141;
      v98 += 40;
LABEL_194:
      v128 = (__m128i *)v45;
      v209 = 1;
      v129 = 10;
      v130 = (__int32 *)v98;
      while ( v129 )
      {
        v128->m128i_i32[0] = *v130++;
        v128 = (__m128i *)((char *)v128 + 4);
        --v129;
      }
      v45 = &v206;
      if ( (sub_13575E0(v24, v46, &v206, 0) & 3) == 0 )
        goto LABEL_142;
      goto LABEL_141;
    }
    v27 = *(_QWORD *)(v27 + 8);
  }
  while ( v26 != v27 );
  v9 = (__int64)v174;
  v4 = v163;
  a3 = v160;
LABEL_178:
  v164 = v4;
  v159 = (_QWORD *)(v4 + 24);
LABEL_111:
  sub_141EDF0(&v216, v4);
  v85 = sub_134CB50((__int64)v24, (__int64)&v216, (__int64)&v196);
  v183 = (unsigned __int64)(sub_127FA20(v188, v153) + 7) >> 3;
  v86 = sub_16498A0(v164);
  v216.m128i_i64[0] = 0;
  v217.m128i_i64[1] = v86;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v216.m128i_i64[1] = *(_QWORD *)(v164 + 40);
  v217.m128i_i64[0] = (__int64)v159;
  v87 = *(_QWORD *)(v164 + 48);
  v213.m128i_i64[0] = v87;
  if ( v87 )
  {
    sub_1623A60((__int64)&v213, v87, 2);
    if ( v216.m128i_i64[0] )
      sub_161E7C0((__int64)&v216, v216.m128i_i64[0]);
    v216.m128i_i64[0] = v213.m128i_i64[0];
    if ( v213.m128i_i64[0] )
      sub_1623210((__int64)&v213, (unsigned __int8 *)v213.m128i_i64[0], (__int64)&v216);
  }
  v176 = *(_WORD *)(v4 + 18) & 1;
  v88 = 1 << (*(unsigned __int16 *)(v9 + 18) >> 1) >> 1;
  if ( v85 )
  {
    if ( !v88 )
      v88 = sub_15A9FE0(v188, *(_QWORD *)v9);
    v165 = v88;
    v169 = *(_QWORD **)(v9 - 24);
    v89 = sub_19CEB00(v188, v4);
    v190 = *(_QWORD **)(v4 - 24);
    v90 = sub_1643360((_QWORD *)v217.m128i_i64[1]);
    v91 = (__int64 *)sub_159C470(v90, v183, 0);
    v92 = sub_15E7940(v216.m128i_i64, v190, v89, v169, v165, v91, v176, 0, 0, 0);
  }
  else
  {
    if ( !v88 )
      v88 = sub_15A9FE0(v188, *(_QWORD *)v9);
    v167 = v88;
    v170 = *(_QWORD **)(v9 - 24);
    v124 = sub_19CEB00(v188, v4);
    v191 = *(_QWORD **)(v4 - 24);
    v125 = sub_1643360((_QWORD *)v217.m128i_i64[1]);
    v126 = (__int64 *)sub_159C470(v125, v183, 0);
    v92 = sub_15E7430(v216.m128i_i64, v191, v124, v170, v167, v126, v176, 0, 0, 0, 0);
  }
  sub_14191F0(*(_QWORD *)a1, v4);
  sub_15F20C0((_QWORD *)v4);
  sub_14191F0(*(_QWORD *)a1, v9);
  sub_15F20C0((_QWORD *)v9);
  v93 = v216.m128i_i64[0];
  *a3 = v92 + 3;
  if ( v93 )
    sub_161E7C0((__int64)&v216, v93);
  return 1;
}
