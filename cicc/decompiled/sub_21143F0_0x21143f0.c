// Function: sub_21143F0
// Address: 0x21143f0
//
__int64 __fastcall sub_21143F0(
        const __m128i **a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        __m128i a5,
        __m128 a6,
        double a7,
        double a8,
        __m128 a9,
        __m128 a10)
{
  __int64 result; // rax
  const __m128i **v11; // rbx
  __m128i *v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // r12
  size_t v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rax
  const __m128i *v29; // r12
  __int64 v30; // r14
  const __m128i *v31; // rdi
  const __m128i *v32; // r13
  __m128i *v33; // rax
  const __m128i *v34; // r9
  unsigned __int64 v35; // r10
  __int64 v36; // rsi
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __m128 *v39; // rdx
  __m128i *v40; // rcx
  const __m128i *v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 **v48; // r14
  unsigned int v49; // r13d
  __int64 v50; // rdx
  unsigned int v51; // r12d
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 ***v54; // r15
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // r15
  __int64 v58; // rax
  unsigned __int64 v59; // rcx
  const __m128i *v60; // rax
  unsigned __int64 v61; // r12
  _QWORD *v62; // rax
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 **v65; // rdi
  __int64 v66; // rcx
  __int64 **v67; // rax
  __int64 v68; // r13
  __int128 v69; // rdi
  __int64 v70; // rcx
  __int8 *v71; // r13
  size_t v72; // r15
  __int64 **v73; // rax
  __m128i *v74; // rax
  __int64 v75; // rcx
  __int64 **v76; // r13
  __int64 *v77; // r13
  __int64 v78; // r15
  __int64 **v79; // rax
  _BYTE *v80; // rdx
  _QWORD *v81; // r12
  _QWORD *v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rax
  __int64 v86; // rdi
  __m128i *v87; // rax
  __int64 v88; // r13
  const __m128i *v89; // rdx
  _BYTE *v90; // rsi
  void *v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rax
  _QWORD *v94; // r12
  __int64 v95; // r13
  __int64 v96; // rax
  void *v97; // rsi
  unsigned int v98; // r15d
  _QWORD *v99; // rax
  __int64 *v100; // r15
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 v103; // r15
  __int64 v104; // rsi
  unsigned __int8 *v105; // rsi
  __int64 v106; // rax
  void *v107; // rsi
  __int64 v108; // rsi
  __int64 v109; // r15
  unsigned __int64 *v110; // r15
  unsigned __int64 v111; // rsi
  __int64 v112; // rax
  __int64 v113; // r15
  __int64 v114; // rsi
  unsigned __int8 *v115; // rsi
  __int64 v116; // rax
  _QWORD *v117; // rax
  _QWORD *v118; // r15
  unsigned __int64 *v119; // r13
  __int64 v120; // rax
  unsigned __int64 v121; // rcx
  __int64 v122; // rsi
  unsigned __int8 *v123; // rsi
  __int64 v124; // rax
  __int64 v125; // rbx
  __int64 v126; // r13
  __int64 v127; // rax
  __int64 v128; // r12
  double v129; // xmm4_8
  double v130; // xmm5_8
  __int64 v131; // rax
  void *v132; // rsi
  __int64 v133; // rsi
  __int64 v134; // rax
  __int64 v135; // r13
  _QWORD *v136; // rax
  _QWORD *v137; // r12
  unsigned __int64 *v138; // r15
  __int64 v139; // rax
  unsigned __int64 v140; // rcx
  __int64 v141; // rsi
  unsigned __int8 *v142; // rsi
  __int64 v143; // rdx
  _QWORD *v144; // rax
  _QWORD *v145; // r15
  unsigned __int64 *v146; // r12
  __int64 v147; // rax
  unsigned __int64 v148; // rcx
  __int64 v149; // rsi
  unsigned __int8 *v150; // rsi
  __int64 v151; // rax
  __int64 v152; // rax
  __int64 *v153; // rax
  __int64 *v154; // r12
  __int64 v155; // r15
  _QWORD *v156; // r13
  __int64 v157; // rdi
  unsigned __int64 *v158; // r15
  __int64 v159; // rax
  unsigned __int64 v160; // rsi
  __int64 *v161; // rsi
  __int64 v162; // rsi
  unsigned __int8 *v163; // rsi
  __int64 v164; // rdx
  _QWORD *v165; // rax
  _QWORD *v166; // r15
  __int64 v167; // rdi
  unsigned __int64 v168; // rsi
  __int64 v169; // rax
  __int64 v170; // rsi
  __int64 v171; // rsi
  unsigned __int8 *v172; // rsi
  const __m128i *v173; // rcx
  const __m128i *v174; // rax
  __int64 v175; // rdx
  unsigned __int64 v176; // r12
  __int64 v177; // r13
  unsigned __int64 v178; // rax
  char *v179; // rsi
  char *v180; // rax
  char *i; // rdx
  bool v182; // cf
  unsigned __int64 v183; // r10
  __int64 v184; // r15
  __m128 *v185; // rax
  const __m128i *v186; // rdi
  __m128 *v187; // rdx
  const __m128i *v188; // rsi
  __m128i *v189; // rcx
  __m128 *v190; // r15
  __m128i *v191; // r14
  __m128i *v192; // r12
  __int64 **v193; // rdi
  const __m128i *v194; // r8
  __m128 *v195; // rdi
  const __m128i *v196; // rdi
  __m128 *v197; // rcx
  const __m128i *v198; // rdi
  __m128 *v199; // rax
  __int64 v200; // rax
  _QWORD *v201; // [rsp+10h] [rbp-250h]
  _QWORD *v202; // [rsp+18h] [rbp-248h]
  __int64 v203; // [rsp+28h] [rbp-238h]
  __int64 v204; // [rsp+28h] [rbp-238h]
  __int64 v205; // [rsp+30h] [rbp-230h]
  const __m128i **v206; // [rsp+30h] [rbp-230h]
  __int64 v207; // [rsp+30h] [rbp-230h]
  __int64 v208; // [rsp+30h] [rbp-230h]
  __int64 v209; // [rsp+38h] [rbp-228h]
  __int64 v211; // [rsp+50h] [rbp-210h]
  unsigned __int64 *v212; // [rsp+50h] [rbp-210h]
  __int64 v214; // [rsp+58h] [rbp-208h]
  _QWORD *v215; // [rsp+58h] [rbp-208h]
  __m128 *v216; // [rsp+58h] [rbp-208h]
  _QWORD *v217; // [rsp+60h] [rbp-200h]
  char *v218; // [rsp+78h] [rbp-1E8h] BYREF
  __int64 v219[2]; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 **v220; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 *v221; // [rsp+98h] [rbp-1C8h]
  const char *v222; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v223; // [rsp+A8h] [rbp-1B8h]
  __int64 **v224; // [rsp+B0h] [rbp-1B0h] BYREF
  _BYTE *v225; // [rsp+B8h] [rbp-1A8h]
  _QWORD v226[2]; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v227; // [rsp+D0h] [rbp-190h] BYREF
  __m128 v228; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v229; // [rsp+F0h] [rbp-170h]
  int v230; // [rsp+F8h] [rbp-168h]
  __int64 v231; // [rsp+100h] [rbp-160h]
  __int64 v232; // [rsp+108h] [rbp-158h]
  void *s2; // [rsp+120h] [rbp-140h] BYREF
  size_t v234; // [rsp+128h] [rbp-138h]
  _QWORD v235[2]; // [rsp+130h] [rbp-130h] BYREF
  __int64 v236[5]; // [rsp+140h] [rbp-120h] BYREF
  int v237; // [rsp+168h] [rbp-F8h]
  __int64 v238; // [rsp+170h] [rbp-F0h]
  __int64 v239; // [rsp+178h] [rbp-E8h]
  __int16 v240; // [rsp+188h] [rbp-D8h]

  if ( (*(_BYTE *)(a2 + 19) & 0x40) == 0 )
    return 0;
  v11 = a1;
  v12 = (__m128i *)"shadow-stack";
  s2 = v235;
  sub_2113CA0((__int64 *)&s2, "shadow-stack", (__int64)"");
  v13 = sub_15E0FA0(a2);
  v14 = s2;
  v15 = *(_QWORD *)(v13 + 8);
  if ( v15 != v234 || v15 && (v12 = (__m128i *)s2, memcmp(*(const void **)v13, s2, v15)) )
  {
    if ( v14 != v235 )
      j_j___libc_free_0(v14, v235[0] + 1LL);
    return 0;
  }
  if ( v14 != v235 )
  {
    v12 = (__m128i *)(v235[0] + 1LL);
    j_j___libc_free_0(v14, v235[0] + 1LL);
  }
  v16 = sub_15E0530(a2);
  v17 = *(_QWORD *)(a2 + 80);
  v217 = (_QWORD *)v16;
  s2 = v235;
  v234 = 0x1000000000LL;
  if ( v17 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v17 )
        goto LABEL_259;
      v18 = *(_QWORD *)(v17 + 24);
      v19 = v17 + 16;
      if ( v18 != v17 + 16 )
        break;
LABEL_25:
      v17 = *(_QWORD *)(v17 + 8);
      if ( a2 + 72 == v17 )
      {
        v29 = (const __m128i *)s2;
        v30 = 16LL * (unsigned int)v234;
        v31 = (const __m128i *)((char *)s2 + v30);
        v32 = v11[23];
        if ( s2 == (char *)s2 + v30 )
          goto LABEL_38;
        v33 = (__m128i *)a1[24];
        v34 = a1[25];
        v35 = v30 >> 4;
        v36 = (char *)v33 - (char *)v32;
        v37 = v33 - v32;
        v38 = v37;
        if ( v30 <= (unsigned __int64)((char *)v34 - (char *)v33) )
        {
          if ( v30 >= (unsigned __int64)v36 )
          {
            v194 = (const __m128i *)((char *)s2 + v36);
            if ( v31 == (const __m128i *)((char *)s2 + v36) )
            {
              v196 = a1[24];
            }
            else
            {
              v195 = (__m128 *)a1[24];
              do
              {
                if ( v195 )
                {
                  a9 = (__m128)_mm_loadu_si128(v194);
                  *v195 = a9;
                }
                ++v195;
                ++v194;
              }
              while ( v195 != (__m128 *)&v32[(unsigned __int64)v30 / 0x10] );
              v196 = a1[24];
            }
            v197 = (__m128 *)&v196[v35 - v37];
            a1[24] = (const __m128i *)v197;
            if ( v32 != v33 )
            {
              v198 = v32;
              v199 = (__m128 *)((char *)v197 + (char *)v33 - (char *)v32);
              do
              {
                if ( v197 )
                {
                  a10 = (__m128)_mm_loadu_si128(v198);
                  *v197 = a10;
                }
                ++v197;
                ++v198;
              }
              while ( v197 != v199 );
              v197 = (__m128 *)a1[24];
            }
            a1[24] = (const __m128i *)((char *)v197 + v36);
            if ( v36 > 0 )
            {
              do
              {
                v200 = v29->m128i_i64[0];
                ++v32;
                ++v29;
                v32[-1].m128i_i64[0] = v200;
                v32[-1].m128i_i64[1] = v29[-1].m128i_i64[1];
                --v38;
              }
              while ( v38 );
            }
          }
          else
          {
            v39 = (__m128 *)a1[24];
            v40 = &v33[v30 / 0xFFFFFFFFFFFFFFF0LL];
            v41 = &v33[v30 / 0xFFFFFFFFFFFFFFF0LL];
            do
            {
              if ( v39 )
              {
                a4 = (__m128)_mm_loadu_si128(v41);
                *v39 = a4;
              }
              ++v39;
              ++v41;
            }
            while ( v39 != (__m128 *)&v33[(unsigned __int64)v30 / 0x10] );
            a1[24] = (const __m128i *)((char *)a1[24] + v30);
            v42 = v40 - v32;
            if ( (char *)v40 - (char *)v32 > 0 )
            {
              do
              {
                v43 = v40[-1].m128i_i64[0];
                --v40;
                --v33;
                v33->m128i_i64[0] = v43;
                v33->m128i_i64[1] = v40->m128i_i64[1];
                --v42;
              }
              while ( v42 );
            }
            if ( v30 )
            {
              do
              {
                v44 = v29->m128i_i64[0];
                ++v29;
                ++v32;
                v32[-1].m128i_i64[0] = v44;
                v32[-1].m128i_i64[1] = v29[-1].m128i_i64[1];
              }
              while ( v31 != v29 );
            }
          }
LABEL_37:
          v31 = (const __m128i *)s2;
LABEL_38:
          if ( v31 != (const __m128i *)v235 )
            _libc_free((unsigned __int64)v31);
          goto LABEL_40;
        }
        if ( v35 > 0x7FFFFFFFFFFFFFFLL - v37 )
          sub_4262D8((__int64)"vector::_M_range_insert");
        if ( v35 < v37 )
          v35 = a1[24] - v32;
        v182 = __CFADD__(v37, v35);
        v183 = v37 + v35;
        if ( v182 )
        {
          v184 = 0x7FFFFFFFFFFFFFFLL;
        }
        else
        {
          if ( !v183 )
          {
            v186 = a1[23];
            v190 = 0;
            v187 = 0;
            v189 = 0;
LABEL_213:
            v191 = &v189[(unsigned __int64)v30 / 0x10];
            do
            {
              if ( v189 )
                *v189 = _mm_loadu_si128(v29);
              ++v189;
              ++v29;
            }
            while ( v189 != v191 );
            if ( v33 == v32 )
            {
              v192 = v189;
            }
            else
            {
              v192 = (__m128i *)((char *)v191 + (char *)v33 - (char *)v32);
              do
              {
                if ( v191 )
                  *v191 = _mm_loadu_si128(v32);
                ++v191;
                ++v32;
              }
              while ( v191 != v192 );
            }
            if ( v186 )
            {
              v216 = v187;
              j_j___libc_free_0(v186, (char *)v34 - (char *)v186);
              v187 = v216;
            }
            v11[23] = (const __m128i *)v187;
            v11[24] = v192;
            v11[25] = (const __m128i *)v190;
            goto LABEL_37;
          }
          if ( v183 > 0x7FFFFFFFFFFFFFFLL )
            v183 = 0x7FFFFFFFFFFFFFFLL;
          v184 = v183;
        }
        v185 = (__m128 *)sub_22077B0(v184 * 16);
        v186 = a1[23];
        v187 = v185;
        if ( v32 == v186 )
        {
          v33 = (__m128i *)a1[24];
          v34 = a1[25];
          v190 = &v187[v184];
          v189 = (__m128i *)v187;
        }
        else
        {
          v188 = a1[23];
          v189 = (__m128i *)((char *)v185 + (char *)v32 - (char *)v186);
          do
          {
            if ( v185 )
            {
              a6 = (__m128)_mm_loadu_si128(v188);
              *v185 = a6;
            }
            ++v185;
            ++v188;
          }
          while ( v185 != (__m128 *)v189 );
          v33 = (__m128i *)a1[24];
          v34 = a1[25];
          v190 = &v187[v184];
        }
        goto LABEL_213;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = v18;
        v18 = *(_QWORD *)(v18 + 8);
        if ( *(_BYTE *)(v20 - 8) == 78 )
        {
          v21 = *(_QWORD *)(v20 - 48);
          if ( !*(_BYTE *)(v21 + 16) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 && *(_DWORD *)(v21 + 36) == 105 )
            break;
        }
LABEL_15:
        if ( v19 == v18 )
          goto LABEL_25;
      }
      v22 = sub_1649C60(*(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 - 4) & 0xFFFFFFF) - 24));
      v227.m128i_i64[0] = v20 - 24;
      v227.m128i_i64[1] = v22;
      v26 = 1LL - (*(_DWORD *)(v20 - 4) & 0xFFFFFFF);
      v27 = *(_QWORD *)(v20 - 24 + 24 * v26);
      if ( *(_BYTE *)(v27 + 16) <= 0x10u && sub_1593BB0(v27, (__int64)v12, v23, v26) )
      {
        v12 = (__m128i *)v11[24];
        if ( v12 == v11[25] )
        {
          sub_2114270(v11 + 23, v12, &v227);
        }
        else
        {
          if ( v12 )
          {
            a5 = _mm_load_si128(&v227);
            *v12 = a5;
            v12 = (__m128i *)a1[24];
          }
          v11[24] = ++v12;
        }
        goto LABEL_15;
      }
      v28 = (unsigned int)v234;
      if ( (unsigned int)v234 >= HIDWORD(v234) )
      {
        v12 = (__m128i *)v235;
        sub_16CD150((__int64)&s2, v235, 0, 16, v24, v25);
        v28 = (unsigned int)v234;
      }
      a3 = (__m128)_mm_load_si128(&v227);
      *((__m128 *)s2 + v28) = a3;
      LODWORD(v234) = v234 + 1;
      if ( v19 == v18 )
        goto LABEL_25;
    }
  }
LABEL_40:
  if ( v11[23] == v11[24] )
    return 0;
  v45 = (_QWORD *)sub_15E0530(a2);
  v46 = sub_16471D0(v45, 0);
  v47 = (__int64)v11[23];
  v48 = (__int64 **)v46;
  s2 = v235;
  v234 = 0x1000000000LL;
  if ( (const __m128i *)v47 == v11[24] )
  {
    v61 = 0;
  }
  else
  {
    v49 = 0;
    v50 = 0;
    v51 = 0;
    do
    {
      ++v51;
      v52 = *(_QWORD *)(v47 + 16 * v50);
      v53 = 1LL - (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
      v54 = *(__int64 ****)(v52 + 24 * v53);
      if ( !sub_1593BB0((__int64)v54, v47, v52, v53) )
        v49 = v51;
      v57 = sub_15A4510(v54, v48, 0);
      v58 = (unsigned int)v234;
      if ( (unsigned int)v234 >= HIDWORD(v234) )
      {
        sub_16CD150((__int64)&s2, v235, 0, 8, v55, v56);
        v58 = (unsigned int)v234;
      }
      *((_QWORD *)s2 + v58) = v57;
      v50 = v51;
      v47 = (__int64)v11[23];
      v59 = (unsigned int)(v234 + 1);
      v60 = v11[24];
      LODWORD(v234) = v234 + 1;
    }
    while ( v51 != ((__int64)v60->m128i_i64 - v47) >> 4 );
    v61 = v49;
    if ( v49 < v59 )
      goto LABEL_49;
    if ( v49 > v59 )
    {
      if ( v49 > (unsigned __int64)HIDWORD(v234) )
      {
        sub_16CD150((__int64)&s2, v235, v49, 8, v55, v56);
        v59 = (unsigned int)v234;
      }
      v180 = (char *)s2 + 8 * v59;
      for ( i = (char *)s2 + 8 * v49; i != v180; v180 += 8 )
      {
        if ( v180 )
          *(_QWORD *)v180 = 0;
      }
LABEL_49:
      LODWORD(v234) = v49;
    }
  }
  v62 = (_QWORD *)sub_15E0530(a2);
  v63 = sub_1643350(v62);
  v219[0] = sub_15A0680(v63, v11[24] - v11[23], 0);
  v64 = sub_15A0680(v63, v61, 0);
  v65 = (__int64 **)v11[22];
  v219[1] = v64;
  v67 = (__int64 **)sub_159F090(v65, v219, 2, v66);
  v68 = (unsigned int)v234;
  v220 = v67;
  *((_QWORD *)&v69 + 1) = s2;
  *(_QWORD *)&v69 = sub_1645D80((__int64 *)v48, v61);
  v221 = (__int64 *)sub_159DFD0(v69, v68, v70);
  v222 = (const char *)*v220;
  v223 = *v221;
  if ( !v61 )
  {
    v228.m128_i8[4] = 48;
    v71 = &v228.m128_i8[4];
    v224 = (__int64 **)v226;
LABEL_52:
    v72 = 1;
    LOBYTE(v226[0]) = *v71;
    v73 = (__int64 **)v226;
    goto LABEL_53;
  }
  v71 = &v228.m128_i8[5];
  do
  {
    *--v71 = v61 % 0xA + 48;
    v178 = v61;
    v61 /= 0xAu;
  }
  while ( v178 > 9 );
  v179 = (char *)((char *)&v228.m128_i32[1] + 1 - v71);
  v224 = (__int64 **)v226;
  v72 = (char *)&v228.m128_i32[1] + 1 - v71;
  v218 = (char *)((char *)&v228.m128_i32[1] + 1 - v71);
  if ( (unsigned __int64)((char *)&v228.m128_i32[1] + 1 - v71) > 0xF )
  {
    v224 = (__int64 **)sub_22409D0(&v224, &v218, 0);
    v193 = v224;
    v226[0] = v218;
  }
  else
  {
    if ( v179 == (char *)1 )
      goto LABEL_52;
    if ( !v179 )
    {
      v73 = (__int64 **)v226;
      goto LABEL_53;
    }
    v193 = (__int64 **)v226;
  }
  memcpy(v193, v71, v72);
  v72 = (size_t)v218;
  v73 = v224;
LABEL_53:
  v225 = (_BYTE *)v72;
  *((_BYTE *)v73 + v72) = 0;
  v74 = (__m128i *)sub_2241130(&v224, 0, 0, "gc_map.", 7);
  v227.m128i_i64[0] = (__int64)&v228;
  if ( (__m128i *)v74->m128i_i64[0] == &v74[1] )
  {
    a9 = (__m128)_mm_loadu_si128(v74 + 1);
    v228 = a9;
  }
  else
  {
    v227.m128i_i64[0] = v74->m128i_i64[0];
    v228.m128_u64[0] = v74[1].m128i_u64[0];
  }
  v227.m128i_i64[1] = v74->m128i_i64[1];
  v74->m128i_i64[0] = (__int64)v74[1].m128i_i64;
  v74->m128i_i64[1] = 0;
  v74[1].m128i_i8[0] = 0;
  v76 = (__int64 **)sub_1644140((__int64 **)&v222, 2, (const void *)v227.m128i_i64[0], v227.m128i_u64[1], 0);
  if ( (__m128 *)v227.m128i_i64[0] != &v228 )
    j_j___libc_free_0(v227.m128i_i64[0], v228.m128_u64[0] + 1);
  if ( v224 != v226 )
    j_j___libc_free_0(v224, v226[0] + 1LL);
  v77 = (__int64 *)sub_159F090(v76, (__int64 *)&v220, 2, v75);
  v78 = *(_QWORD *)(a2 + 40);
  v214 = *v77;
  v79 = (__int64 **)sub_1649960(a2);
  v228.m128_i16[0] = 1283;
  v224 = v79;
  v225 = v80;
  v227.m128i_i64[0] = (__int64)"__gc_";
  v227.m128i_i64[1] = (__int64)&v224;
  v81 = sub_1648A60(88, 1u);
  if ( v81 )
    sub_15E51E0((__int64)v81, v78, v214, 1, 7, (__int64)v77, (__int64)&v227, 0, 0, 0, 0);
  v82 = (_QWORD *)sub_15E0530(a2);
  v83 = sub_1643350(v82);
  v227.m128i_i64[0] = sub_159C470(v83, 0, 0);
  v84 = (_QWORD *)sub_15E0530(a2);
  v85 = sub_1643350(v84);
  v227.m128i_i64[1] = sub_159C470(v85, 0, 0);
  v86 = *v77;
  BYTE4(v224) = 0;
  v205 = sub_15A2E80(v86, (__int64)v81, (__int64 **)&v227, 2u, 0, (__int64)&v224, 0);
  if ( s2 != v235 )
    _libc_free((unsigned __int64)s2);
  v87 = (__m128i *)v11[21];
  v88 = 0;
  v224 = 0;
  v225 = 0;
  v226[0] = 0;
  s2 = v87;
  sub_1278040((__int64)&v224, 0, &s2);
  v89 = v11[23];
  if ( v89 != v11[24] )
  {
    do
    {
      v90 = v225;
      v91 = *(void **)(v89[v88].m128i_i64[1] + 56);
      s2 = v91;
      if ( v225 == (_BYTE *)v226[0] )
      {
        sub_1278040((__int64)&v224, v225, &s2);
        v89 = v11[23];
      }
      else
      {
        if ( v225 )
        {
          *(_QWORD *)v225 = v91;
          v90 = v225;
          v89 = v11[23];
        }
        v225 = v90 + 8;
      }
      ++v88;
    }
    while ( v11[24] - v89 != v88 );
  }
  v222 = sub_1649960(a2);
  v227.m128i_i64[0] = (__int64)"gc_stackentry.";
  v223 = v92;
  v228.m128_i16[0] = 1283;
  v227.m128i_i64[1] = (__int64)&v222;
  sub_16E2FC0((__int64 *)&s2, (__int64)&v227);
  v215 = (_QWORD *)sub_1644140(v224, (v225 - (_BYTE *)v224) >> 3, s2, v234, 0);
  if ( s2 != v235 )
    j_j___libc_free_0(s2, v235[0] + 1LL);
  if ( v224 )
    j_j___libc_free_0(v224, v226[0] - (_QWORD)v224);
  v93 = *(_QWORD *)(a2 + 80);
  if ( !v93 )
LABEL_259:
    BUG();
  v94 = *(_QWORD **)(v93 + 24);
  if ( !v94 )
    BUG();
  v95 = v94[2];
  v96 = sub_157E9C0(v95);
  v227.m128i_i64[1] = v95;
  v228.m128_u64[1] = v96;
  v227.m128i_i64[0] = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v228.m128_u64[0] = (unsigned __int64)v94;
  if ( v94 != (_QWORD *)(v95 + 40) )
  {
    v97 = (void *)v94[3];
    s2 = v97;
    if ( v97 )
    {
      sub_1623A60((__int64)&s2, (__int64)v97, 2);
      if ( v227.m128i_i64[0] )
        sub_161E7C0((__int64)&v227, v227.m128i_i64[0]);
      v227.m128i_i64[0] = (__int64)s2;
      if ( s2 )
        sub_1623210((__int64)&s2, (unsigned __int8 *)s2, (__int64)&v227);
    }
    v95 = v227.m128i_i64[1];
  }
  v224 = (__int64 **)"gc_frame";
  LOWORD(v226[0]) = 259;
  v98 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v95 + 56) + 40LL)) + 4);
  LOWORD(v235[0]) = 257;
  v99 = sub_1648A60(64, 1u);
  v209 = (__int64)v99;
  if ( v99 )
    sub_15F8BC0((__int64)v99, v215, v98, 0, (__int64)&s2, 0);
  if ( v227.m128i_i64[1] )
  {
    v100 = (__int64 *)v228.m128_u64[0];
    sub_157E9D0(v227.m128i_i64[1] + 40, v209);
    v101 = *v100;
    v102 = *(_QWORD *)(v209 + 24);
    *(_QWORD *)(v209 + 32) = v100;
    v101 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v209 + 24) = v101 | v102 & 7;
    *(_QWORD *)(v101 + 8) = v209 + 24;
    *v100 = *v100 & 7 | (v209 + 24);
  }
  sub_164B780(v209, (__int64 *)&v224);
  if ( v227.m128i_i64[0] )
  {
    v222 = (const char *)v227.m128i_i64[0];
    sub_1623A60((__int64)&v222, v227.m128i_i64[0], 2);
    v103 = v209 + 48;
    v104 = *(_QWORD *)(v209 + 48);
    if ( v104 )
      sub_161E7C0(v103, v104);
    v105 = (unsigned __int8 *)v222;
    *(_QWORD *)(v209 + 48) = v222;
    if ( v105 )
      sub_1623210((__int64)&v222, v105, v103);
  }
  while ( *((_BYTE *)v94 - 8) == 53 )
  {
    v94 = (_QWORD *)v94[1];
    if ( !v94 )
      goto LABEL_260;
  }
  v106 = v94[2];
  v228.m128_u64[0] = (unsigned __int64)v94;
  v227.m128i_i64[1] = v106;
  if ( v94 != (_QWORD *)(v106 + 40) )
  {
    v107 = (void *)v94[3];
    s2 = v107;
    if ( v107 )
    {
      sub_1623A60((__int64)&s2, (__int64)v107, 2);
      v108 = v227.m128i_i64[0];
      if ( !v227.m128i_i64[0] )
        goto LABEL_103;
    }
    else
    {
      v108 = v227.m128i_i64[0];
      if ( !v227.m128i_i64[0] )
        goto LABEL_105;
    }
    sub_161E7C0((__int64)&v227, v108);
LABEL_103:
    v227.m128i_i64[0] = (__int64)s2;
    if ( s2 )
      sub_1623210((__int64)&s2, (unsigned __int8 *)s2, (__int64)&v227);
  }
LABEL_105:
  s2 = "gc_currhead";
  v109 = (__int64)v11[20];
  LOWORD(v235[0]) = 259;
  v202 = sub_1648A60(64, 1u);
  if ( v202 )
    sub_15F9210((__int64)v202, *(_QWORD *)(*(_QWORD *)v109 + 24LL), v109, 0, 0, 0);
  if ( v227.m128i_i64[1] )
  {
    v110 = (unsigned __int64 *)v228.m128_u64[0];
    sub_157E9D0(v227.m128i_i64[1] + 40, (__int64)v202);
    v111 = *v110;
    v112 = v202[3];
    v202[4] = v110;
    v111 &= 0xFFFFFFFFFFFFFFF8LL;
    v202[3] = v111 | v112 & 7;
    *(_QWORD *)(v111 + 8) = v202 + 3;
    *v110 = *v110 & 7 | (unsigned __int64)(v202 + 3);
  }
  sub_164B780((__int64)v202, (__int64 *)&s2);
  if ( v227.m128i_i64[0] )
  {
    v224 = (__int64 **)v227.m128i_i64[0];
    sub_1623A60((__int64)&v224, v227.m128i_i64[0], 2);
    v113 = (__int64)(v202 + 6);
    v114 = v202[6];
    if ( v114 )
      sub_161E7C0(v113, v114);
    v115 = (unsigned __int8 *)v224;
    v202[6] = v224;
    if ( v115 )
      sub_1623210((__int64)&v224, v115, v113);
  }
  v116 = sub_21141B0(v217, v227.m128i_i64, (__int64)v215, (_BYTE *)v209, 1, "gc_frame.map");
  LOWORD(v235[0]) = 257;
  v203 = v116;
  v117 = sub_1648A60(64, 2u);
  v118 = v117;
  if ( v117 )
    sub_15F9650((__int64)v117, v205, v203, 0, 0);
  if ( v227.m128i_i64[1] )
  {
    v119 = (unsigned __int64 *)v228.m128_u64[0];
    sub_157E9D0(v227.m128i_i64[1] + 40, (__int64)v118);
    v120 = v118[3];
    v121 = *v119;
    v118[4] = v119;
    v121 &= 0xFFFFFFFFFFFFFFF8LL;
    v118[3] = v121 | v120 & 7;
    *(_QWORD *)(v121 + 8) = v118 + 3;
    *v119 = *v119 & 7 | (unsigned __int64)(v118 + 3);
  }
  sub_164B780((__int64)v118, (__int64 *)&s2);
  if ( v227.m128i_i64[0] )
  {
    v224 = (__int64 **)v227.m128i_i64[0];
    sub_1623A60((__int64)&v224, v227.m128i_i64[0], 2);
    v122 = v118[6];
    if ( v122 )
      sub_161E7C0((__int64)(v118 + 6), v122);
    v123 = (unsigned __int8 *)v224;
    v118[6] = v224;
    if ( v123 )
      sub_1623210((__int64)&v224, v123, (__int64)(v118 + 6));
  }
  v124 = v11[24] - v11[23];
  v204 = (unsigned int)v124;
  if ( !(_DWORD)v124 )
    goto LABEL_128;
  v201 = v94;
  v206 = v11;
  v125 = 0;
  do
  {
    v126 = sub_2114100(v217, v227.m128i_i64, (__int64)v215, (_BYTE *)v209, (int)v125 + 1, "gc_root");
    v127 = v125++;
    v128 = v206[23][v127].m128i_i64[1];
    sub_164B7C0(v126, v128);
    sub_164D160(
      v128,
      v126,
      a3,
      *(double *)a4.m128_u64,
      *(double *)a5.m128i_i64,
      *(double *)a6.m128_u64,
      v129,
      v130,
      *(double *)a9.m128_u64,
      a10);
  }
  while ( v204 != v125 );
  v94 = v201;
  v11 = v206;
  if ( *((_BYTE *)v201 - 8) == 55 )
  {
    while ( 1 )
    {
      v94 = (_QWORD *)v94[1];
      if ( !v94 )
        break;
LABEL_128:
      if ( *((_BYTE *)v94 - 8) != 55 )
        goto LABEL_129;
    }
LABEL_260:
    BUG();
  }
LABEL_129:
  v131 = v94[2];
  v228.m128_u64[0] = (unsigned __int64)v94;
  v227.m128i_i64[1] = v131;
  if ( v94 == (_QWORD *)(v131 + 40) )
    goto LABEL_135;
  v132 = (void *)v94[3];
  s2 = v132;
  if ( v132 )
  {
    sub_1623A60((__int64)&s2, (__int64)v132, 2);
    v133 = v227.m128i_i64[0];
    if ( !v227.m128i_i64[0] )
      goto LABEL_133;
    goto LABEL_132;
  }
  v133 = v227.m128i_i64[0];
  if ( v227.m128i_i64[0] )
  {
LABEL_132:
    sub_161E7C0((__int64)&v227, v133);
LABEL_133:
    v227.m128i_i64[0] = (__int64)s2;
    if ( s2 )
      sub_1623210((__int64)&s2, (unsigned __int8 *)s2, (__int64)&v227);
  }
LABEL_135:
  v207 = sub_21141B0(v217, v227.m128i_i64, (__int64)v215, (_BYTE *)v209, 0, "gc_frame.next");
  v134 = sub_2114100(v217, v227.m128i_i64, (__int64)v215, (_BYTE *)v209, 0, "gc_newhead");
  LOWORD(v235[0]) = 257;
  v135 = v134;
  v136 = sub_1648A60(64, 2u);
  v137 = v136;
  if ( v136 )
    sub_15F9650((__int64)v136, (__int64)v202, v207, 0, 0);
  if ( v227.m128i_i64[1] )
  {
    v138 = (unsigned __int64 *)v228.m128_u64[0];
    sub_157E9D0(v227.m128i_i64[1] + 40, (__int64)v137);
    v139 = v137[3];
    v140 = *v138;
    v137[4] = v138;
    v140 &= 0xFFFFFFFFFFFFFFF8LL;
    v137[3] = v140 | v139 & 7;
    *(_QWORD *)(v140 + 8) = v137 + 3;
    *v138 = *v138 & 7 | (unsigned __int64)(v137 + 3);
  }
  sub_164B780((__int64)v137, (__int64 *)&s2);
  if ( v227.m128i_i64[0] )
  {
    v224 = (__int64 **)v227.m128i_i64[0];
    sub_1623A60((__int64)&v224, v227.m128i_i64[0], 2);
    v141 = v137[6];
    if ( v141 )
      sub_161E7C0((__int64)(v137 + 6), v141);
    v142 = (unsigned __int8 *)v224;
    v137[6] = v224;
    if ( v142 )
      sub_1623210((__int64)&v224, v142, (__int64)(v137 + 6));
  }
  v143 = (__int64)v11[20];
  LOWORD(v235[0]) = 257;
  v208 = v143;
  v144 = sub_1648A60(64, 2u);
  v145 = v144;
  if ( v144 )
    sub_15F9650((__int64)v144, v135, v208, 0, 0);
  if ( v227.m128i_i64[1] )
  {
    v146 = (unsigned __int64 *)v228.m128_u64[0];
    sub_157E9D0(v227.m128i_i64[1] + 40, (__int64)v145);
    v147 = v145[3];
    v148 = *v146;
    v145[4] = v146;
    v148 &= 0xFFFFFFFFFFFFFFF8LL;
    v145[3] = v148 | v147 & 7;
    *(_QWORD *)(v148 + 8) = v145 + 3;
    *v146 = *v146 & 7 | (unsigned __int64)(v145 + 3);
  }
  sub_164B780((__int64)v145, (__int64 *)&s2);
  if ( v227.m128i_i64[0] )
  {
    v224 = (__int64 **)v227.m128i_i64[0];
    sub_1623A60((__int64)&v224, v227.m128i_i64[0], 2);
    v149 = v145[6];
    if ( v149 )
      sub_161E7C0((__int64)(v145 + 6), v149);
    v150 = (unsigned __int8 *)v224;
    v145[6] = v224;
    if ( v150 )
      sub_1623210((__int64)&v224, v150, (__int64)(v145 + 6));
  }
  v234 = (size_t)"gc_cleanup";
  v151 = *(_QWORD *)(a2 + 80);
  s2 = (void *)a2;
  v235[0] = v151;
  v235[1] = a2 + 72;
  v152 = sub_15E0530(a2);
  memset(v236, 0, 24);
  v236[3] = v152;
  v236[4] = 0;
  v237 = 0;
  v238 = 0;
  v239 = 0;
  v240 = 256;
  while ( 1 )
  {
    v153 = (__int64 *)sub_1AC5690((__int64)&s2);
    v154 = v153;
    if ( !v153 )
      break;
    v155 = sub_21141B0(v217, v153, (__int64)v215, (_BYTE *)v209, 0, "gc_frame.next");
    LOWORD(v226[0]) = 259;
    v224 = (__int64 **)"gc_savedhead";
    v156 = sub_1648A60(64, 1u);
    if ( v156 )
      sub_15F9210((__int64)v156, *(_QWORD *)(*(_QWORD *)v155 + 24LL), v155, 0, 0, 0);
    v157 = v154[1];
    if ( v157 )
    {
      v158 = (unsigned __int64 *)v154[2];
      sub_157E9D0(v157 + 40, (__int64)v156);
      v159 = v156[3];
      v160 = *v158;
      v156[4] = v158;
      v160 &= 0xFFFFFFFFFFFFFFF8LL;
      v156[3] = v160 | v159 & 7;
      *(_QWORD *)(v160 + 8) = v156 + 3;
      *v158 = *v158 & 7 | (unsigned __int64)(v156 + 3);
    }
    sub_164B780((__int64)v156, (__int64 *)&v224);
    v161 = (__int64 *)*v154;
    if ( *v154 )
    {
      v222 = (const char *)*v154;
      sub_1623A60((__int64)&v222, (__int64)v161, 2);
      v162 = v156[6];
      if ( v162 )
        sub_161E7C0((__int64)(v156 + 6), v162);
      v163 = (unsigned __int8 *)v222;
      v156[6] = v222;
      if ( v163 )
        sub_1623210((__int64)&v222, v163, (__int64)(v156 + 6));
    }
    v164 = (__int64)v11[20];
    LOWORD(v226[0]) = 257;
    v211 = v164;
    v165 = sub_1648A60(64, 2u);
    v166 = v165;
    if ( v165 )
      sub_15F9650((__int64)v165, (__int64)v156, v211, 0, 0);
    v167 = v154[1];
    if ( v167 )
    {
      v212 = (unsigned __int64 *)v154[2];
      sub_157E9D0(v167 + 40, (__int64)v166);
      v168 = *v212;
      v169 = v166[3] & 7LL;
      v166[4] = v212;
      v168 &= 0xFFFFFFFFFFFFFFF8LL;
      v166[3] = v168 | v169;
      *(_QWORD *)(v168 + 8) = v166 + 3;
      *v212 = *v212 & 7 | (unsigned __int64)(v166 + 3);
    }
    sub_164B780((__int64)v166, (__int64 *)&v224);
    v170 = *v154;
    if ( *v154 )
    {
      v222 = (const char *)*v154;
      sub_1623A60((__int64)&v222, v170, 2);
      v171 = v166[6];
      if ( v171 )
        sub_161E7C0((__int64)(v166 + 6), v171);
      v172 = (unsigned __int8 *)v222;
      v166[6] = v222;
      if ( v172 )
        sub_1623210((__int64)&v222, v172, (__int64)(v166 + 6));
    }
  }
  v173 = v11[24];
  v174 = v11[23];
  v175 = v173 - v174;
  if ( (_DWORD)v175 )
  {
    v176 = 0;
    v177 = 16LL * (unsigned int)(v175 - 1);
    while ( 1 )
    {
      sub_15F20C0((_QWORD *)v174[v176 / 0x10].m128i_i64[0]);
      sub_15F20C0((_QWORD *)v11[23][v176 / 0x10].m128i_i64[1]);
      v174 = v11[23];
      if ( v176 == v177 )
        break;
      v176 += 16LL;
    }
    v173 = v11[24];
  }
  if ( v174 != v173 )
    v11[24] = v174;
  if ( v236[0] )
    sub_161E7C0((__int64)v236, v236[0]);
  result = 1;
  if ( v227.m128i_i64[0] )
  {
    sub_161E7C0((__int64)&v227, v227.m128i_i64[0]);
    return 1;
  }
  return result;
}
