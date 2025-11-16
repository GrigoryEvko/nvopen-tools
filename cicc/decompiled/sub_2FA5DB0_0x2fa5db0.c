// Function: sub_2FA5DB0
// Address: 0x2fa5db0
//
__int64 __fastcall sub_2FA5DB0(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 *v3; // rbx
  __m128i *v4; // r15
  __m128i *v5; // r14
  __m128i *v6; // r12
  __m128i *v7; // r13
  __int64 v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __m128i v14; // xmm0
  unsigned __int64 v15; // rdx
  const __m128i *v16; // r12
  __int64 v17; // r14
  const __m128i *v18; // rdi
  const __m128i *v19; // r13
  __m128i *v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // r8
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __m128i *v25; // rdx
  __m128i *v26; // rcx
  const __m128i *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rsi
  unsigned int v34; // r15d
  __int64 v35; // rdx
  unsigned int v36; // r14d
  __int64 v37; // r13
  bool v38; // zf
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  unsigned __int64 v44; // r12
  const char *v45; // rax
  const char *i; // rdx
  _QWORD *v47; // rax
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 **v50; // rdi
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 v53; // r14
  __int64 **v54; // rax
  __int8 *v55; // r13
  size_t v56; // r14
  __int64 *v57; // rax
  unsigned __int64 *v58; // rdi
  __m128i *v59; // rax
  __int64 **v60; // r13
  __int64 v61; // rax
  _QWORD *v62; // r15
  __int64 v63; // r13
  __int64 v64; // r14
  __m128i v65; // rax
  unsigned __int8 *v66; // r12
  _QWORD *v67; // rax
  __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdi
  unsigned __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // r13
  __int64 *v75; // r14
  _QWORD *v76; // r15
  _BYTE *v77; // rsi
  unsigned __int64 v78; // rax
  const char *v79; // rax
  __int8 *v80; // rdx
  __int64 v81; // rax
  __int64 v82; // r12
  __int64 v83; // r13
  __int64 v84; // rax
  __int64 *v85; // r15
  __int64 v86; // r12
  unsigned __int8 v87; // al
  unsigned int v88; // r12d
  unsigned __int8 v89; // r13
  __int64 v90; // r13
  __int64 v91; // r12
  __int64 v92; // rdx
  unsigned int v93; // esi
  __int64 v94; // rdi
  __int64 v95; // rax
  __int16 v96; // dx
  __int64 v97; // r12
  __int64 v98; // rdx
  __int64 v99; // r15
  __int64 v100; // r14
  __int64 v101; // rax
  char v102; // al
  _QWORD *v103; // rax
  __int64 v104; // r9
  __int64 v105; // r13
  __int64 v106; // r15
  __int64 v107; // r14
  __int64 v108; // rdx
  unsigned int v109; // esi
  _BYTE *v110; // r14
  __int64 v111; // rax
  _QWORD *v112; // rax
  __int64 v113; // r9
  __int64 v114; // r15
  __int64 v115; // r14
  __int64 v116; // r14
  __int64 v117; // rbx
  __int64 v118; // rdx
  unsigned int v119; // esi
  __int64 v120; // r14
  __int64 v121; // rbx
  unsigned __int8 *v122; // r13
  __int64 v123; // rax
  unsigned __int8 *v124; // r12
  char v125; // al
  unsigned __int8 v126; // dl
  __int16 v127; // cx
  _QWORD *v128; // r12
  _BYTE *v129; // r15
  _BYTE *v130; // r14
  __int64 v131; // rax
  char v132; // al
  _QWORD *v133; // rax
  __int64 v134; // r9
  __int64 v135; // r12
  __int64 v136; // r15
  __int64 v137; // r13
  __int64 v138; // rdx
  unsigned int v139; // esi
  __int64 v140; // r15
  __int64 v141; // rax
  char v142; // r12
  _QWORD *v143; // rax
  __int64 v144; // r9
  __int64 v145; // r13
  __int64 v146; // r14
  __int64 v147; // r12
  __int64 v148; // rdx
  unsigned int v149; // esi
  const char *v150; // rax
  unsigned int **v151; // rax
  unsigned int **v152; // r12
  _BYTE *v153; // r14
  __int64 v154; // rax
  __int64 v155; // r13
  __int64 v156; // rax
  char v157; // al
  __int16 v158; // cx
  _QWORD *v159; // rax
  __int64 v160; // rbx
  unsigned int *v161; // r14
  __int64 v162; // r13
  __int64 v163; // rdx
  unsigned int v164; // esi
  __int64 v165; // rax
  char v166; // r15
  _QWORD *v167; // rax
  __int64 v168; // r9
  __int64 v169; // r13
  unsigned int *v170; // rbx
  __int64 v171; // r12
  __int64 v172; // rdx
  unsigned int v173; // esi
  __int64 *v174; // rax
  unsigned __int64 v175; // rax
  char *v176; // rsi
  __int64 *v177; // rdi
  bool v178; // cf
  unsigned __int64 v179; // rcx
  unsigned __int64 v180; // r15
  __m128i *v181; // rax
  unsigned __int64 v182; // rdi
  __int64 *v183; // rdx
  const __m128i *v184; // rsi
  __m128i *v185; // rcx
  __int64 *v186; // r15
  __m128i *v187; // r14
  __m128i *v188; // r12
  _QWORD **v189; // r13
  _QWORD **v190; // r12
  _QWORD *v191; // rdi
  __int64 v192; // rax
  const __m128i *v194; // r9
  __m128i *v195; // rdi
  __int64 v196; // rdi
  __m128i *v197; // rcx
  const __m128i *v198; // rsi
  __m128i *v199; // rax
  __int64 v200; // rax
  __int64 v201; // [rsp-10h] [rbp-320h]
  __int64 v202; // [rsp+0h] [rbp-310h]
  char v203; // [rsp+18h] [rbp-2F8h]
  __int64 v204; // [rsp+18h] [rbp-2F8h]
  char v205; // [rsp+20h] [rbp-2F0h]
  __int8 *v207; // [rsp+30h] [rbp-2E0h]
  char v208; // [rsp+48h] [rbp-2C8h]
  char v209; // [rsp+48h] [rbp-2C8h]
  __int64 v210; // [rsp+58h] [rbp-2B8h]
  __int64 *v211; // [rsp+58h] [rbp-2B8h]
  __int64 *v212; // [rsp+58h] [rbp-2B8h]
  __int64 *v213; // [rsp+58h] [rbp-2B8h]
  _QWORD *v214; // [rsp+60h] [rbp-2B0h]
  __int64 v215; // [rsp+68h] [rbp-2A8h]
  __int64 v216; // [rsp+68h] [rbp-2A8h]
  __m128i v217; // [rsp+70h] [rbp-2A0h] BYREF
  _QWORD *v218; // [rsp+80h] [rbp-290h]
  const char *v219; // [rsp+88h] [rbp-288h]
  unsigned __int64 *v220; // [rsp+90h] [rbp-280h]
  unsigned __int8 v221; // [rsp+9Dh] [rbp-273h]
  __int16 v222; // [rsp+9Eh] [rbp-272h]
  __int64 v223; // [rsp+A0h] [rbp-270h]
  __int64 v224; // [rsp+A8h] [rbp-268h]
  size_t v225; // [rsp+B8h] [rbp-258h] BYREF
  __int64 v226[2]; // [rsp+C0h] [rbp-250h] BYREF
  __int64 v227; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v228; // [rsp+D8h] [rbp-238h]
  __int64 v229[4]; // [rsp+E0h] [rbp-230h] BYREF
  char v230; // [rsp+100h] [rbp-210h]
  char v231; // [rsp+101h] [rbp-20Fh]
  char *v232; // [rsp+110h] [rbp-200h] BYREF
  _BYTE *v233; // [rsp+118h] [rbp-1F8h]
  _QWORD v234[2]; // [rsp+120h] [rbp-1F0h] BYREF
  __int16 v235; // [rsp+130h] [rbp-1E0h]
  __m128i v236; // [rsp+140h] [rbp-1D0h] BYREF
  __m128i v237; // [rsp+150h] [rbp-1C0h] BYREF
  __int16 v238; // [rsp+160h] [rbp-1B0h]
  __int64 v239; // [rsp+170h] [rbp-1A0h]
  __int64 v240; // [rsp+178h] [rbp-198h]
  __int64 v241; // [rsp+180h] [rbp-190h]
  __int64 *v242; // [rsp+188h] [rbp-188h]
  void **v243; // [rsp+190h] [rbp-180h]
  void **v244; // [rsp+198h] [rbp-178h]
  __int64 v245; // [rsp+1A0h] [rbp-170h]
  int v246; // [rsp+1A8h] [rbp-168h]
  __int16 v247; // [rsp+1ACh] [rbp-164h]
  char v248; // [rsp+1AEh] [rbp-162h]
  __int64 v249; // [rsp+1B0h] [rbp-160h]
  __int64 v250; // [rsp+1B8h] [rbp-158h]
  void *v251; // [rsp+1C0h] [rbp-150h] BYREF
  void *v252; // [rsp+1C8h] [rbp-148h] BYREF
  const char *v253; // [rsp+1D0h] [rbp-140h] BYREF
  __int64 v254; // [rsp+1D8h] [rbp-138h]
  const char *v255; // [rsp+1E0h] [rbp-130h] BYREF
  __int8 *v256; // [rsp+1E8h] [rbp-128h]
  _BYTE *v257; // [rsp+1F0h] [rbp-120h]
  __int64 v258; // [rsp+1F8h] [rbp-118h]
  _BYTE v259[32]; // [rsp+200h] [rbp-110h] BYREF
  __int64 v260; // [rsp+220h] [rbp-F0h]
  __int64 v261; // [rsp+228h] [rbp-E8h]
  __int16 v262; // [rsp+230h] [rbp-E0h]
  __int64 v263; // [rsp+238h] [rbp-D8h]
  void **v264; // [rsp+240h] [rbp-D0h]
  void **v265; // [rsp+248h] [rbp-C8h]
  __int64 v266; // [rsp+250h] [rbp-C0h]
  int v267; // [rsp+258h] [rbp-B8h]
  __int16 v268; // [rsp+25Ch] [rbp-B4h]
  char v269; // [rsp+25Eh] [rbp-B2h]
  __int64 v270; // [rsp+260h] [rbp-B0h]
  __int64 v271; // [rsp+268h] [rbp-A8h]
  void *v272; // [rsp+270h] [rbp-A0h] BYREF
  void *v273; // [rsp+278h] [rbp-98h] BYREF
  __int16 v274; // [rsp+280h] [rbp-90h]
  __int64 v275; // [rsp+288h] [rbp-88h]

  v3 = a1;
  v215 = (__int64)a2;
  v218 = (_QWORD *)sub_B2BE50((__int64)a2);
  v219 = (const char *)&v255;
  v253 = (const char *)&v255;
  v254 = 0x1000000000LL;
  v4 = (__m128i *)a2[5].m128i_i64[0];
  v207 = &a2[4].m128i_i8[8];
  if ( v4 == (__m128i *)&a2[4].m128i_u64[1] )
    goto LABEL_34;
  v220 = (unsigned __int64 *)a1;
  v5 = (__m128i *)((char *)a2 + 72);
  do
  {
    if ( !v4 )
      goto LABEL_206;
    v6 = (__m128i *)v4[2].m128i_i64[0];
    v7 = (__m128i *)((char *)v4 + 24);
    if ( v6 != (__m128i *)&v4[1].m128i_u64[1] )
    {
      while ( 1 )
      {
        if ( !v6 )
LABEL_205:
          BUG();
        if ( v6[-2].m128i_i8[8] != 85 )
          goto LABEL_6;
        v8 = v6[-4].m128i_i64[1];
        if ( !v8
          || *(_BYTE *)v8
          || *(_QWORD *)(v8 + 24) != v6[3].m128i_i64[1]
          || (*(_BYTE *)(v8 + 33) & 0x20) == 0
          || *(_DWORD *)(v8 + 36) != 183 )
        {
          goto LABEL_6;
        }
        v9 = sub_BD3990(*((unsigned __int8 **)&v6[-2 * (v6[-2].m128i_i32[3] & 0x7FFFFFF) - 1] - 1), (__int64)a2);
        v236.m128i_i64[0] = (__int64)&v6[-2].m128i_i64[1];
        v236.m128i_i64[1] = (__int64)v9;
        v12 = (_BYTE *)*((_QWORD *)&v6[2 * (1LL - (v6[-2].m128i_i32[3] & 0x7FFFFFF)) - 1] - 1);
        if ( *v12 <= 0x15u && sub_AC30F0((__int64)v12) )
        {
          v174 = (__int64 *)v220;
          a2 = (__m128i *)v220[4];
          if ( a2 == (__m128i *)v220[5] )
          {
            sub_2FA5C30(v220 + 3, a2, &v236);
          }
          else
          {
            if ( a2 )
            {
              *a2 = _mm_load_si128(&v236);
              a2 = (__m128i *)v174[4];
            }
            v220[4] = (unsigned __int64)++a2;
          }
LABEL_6:
          v6 = (__m128i *)v6->m128i_i64[1];
          if ( v7 == v6 )
            break;
        }
        else
        {
          v13 = (unsigned int)v254;
          v14 = _mm_load_si128(&v236);
          v15 = (unsigned int)v254 + 1LL;
          if ( v15 > HIDWORD(v254) )
          {
            a2 = (__m128i *)v219;
            v217 = v14;
            sub_C8D5F0((__int64)&v253, v219, v15, 0x10u, v10, v11);
            v13 = (unsigned int)v254;
            v14 = _mm_load_si128(&v217);
          }
          *(__m128i *)&v253[16 * v13] = v14;
          LODWORD(v254) = v254 + 1;
          v6 = (__m128i *)v6->m128i_i64[1];
          if ( v7 == v6 )
            break;
        }
      }
    }
    v4 = (__m128i *)v4->m128i_i64[1];
  }
  while ( v5 != v4 );
  v16 = (const __m128i *)v253;
  v3 = (__int64 *)v220;
  v17 = 16LL * (unsigned int)v254;
  v18 = (const __m128i *)&v253[v17];
  v19 = (const __m128i *)v220[3];
  if ( v253 == &v253[v17] )
    goto LABEL_32;
  v20 = (__m128i *)v220[4];
  v21 = v17 >> 4;
  v22 = (char *)v20 - (char *)v19;
  v23 = v20 - v19;
  v24 = v23;
  if ( v17 > v220[5] - (unsigned __int64)v20 )
  {
    if ( v21 > 0x7FFFFFFFFFFFFFFLL - v23 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v23 >= v21 )
      v21 = (__int64)(v220[4] - (_QWORD)v19) >> 4;
    v178 = __CFADD__(v21, v23);
    v179 = v21 + v23;
    if ( v178 )
    {
      v180 = 0xFFFFFFFFFFFFFFELL;
    }
    else
    {
      if ( !v179 )
      {
        v182 = v220[3];
        v186 = 0;
        v183 = 0;
        v185 = 0;
LABEL_145:
        v187 = &v185[(unsigned __int64)v17 / 0x10];
        do
        {
          if ( v185 )
            *v185 = _mm_loadu_si128(v16);
          ++v185;
          ++v16;
        }
        while ( v185 != v187 );
        if ( v19 == v20 )
        {
          v188 = v185;
        }
        else
        {
          v188 = (__m128i *)((char *)v187 + (char *)v20 - (char *)v19);
          do
          {
            if ( v187 )
              *v187 = _mm_loadu_si128(v19);
            ++v187;
            ++v19;
          }
          while ( v187 != v188 );
        }
        if ( v182 )
        {
          v220 = (unsigned __int64 *)v183;
          j_j___libc_free_0(v182);
          v183 = (__int64 *)v220;
        }
        v3[3] = (__int64)v183;
        v18 = (const __m128i *)v253;
        v3[4] = (__int64)v188;
        v3[5] = (__int64)v186;
        goto LABEL_32;
      }
      if ( v179 > 0x7FFFFFFFFFFFFFFLL )
        v179 = 0x7FFFFFFFFFFFFFFLL;
      v180 = 2 * v179;
    }
    v181 = (__m128i *)sub_22077B0(v180 * 8);
    v182 = v3[3];
    v183 = (__int64 *)v181;
    if ( v19 == (const __m128i *)v182 )
    {
      v20 = (__m128i *)v3[4];
      v186 = &v183[v180];
      v185 = (__m128i *)v183;
    }
    else
    {
      v184 = (const __m128i *)v3[3];
      v185 = (__m128i *)((char *)v19 + (_QWORD)v181 - v182);
      do
      {
        if ( v181 )
          *v181 = _mm_loadu_si128(v184);
        ++v181;
        ++v184;
      }
      while ( v181 != v185 );
      v20 = (__m128i *)v3[4];
      v186 = &v183[v180];
    }
    goto LABEL_145;
  }
  if ( v17 < (unsigned __int64)v22 )
  {
    v25 = (__m128i *)v220[4];
    v26 = &v20[v17 / 0xFFFFFFFFFFFFFFF0LL];
    v27 = &v20[v17 / 0xFFFFFFFFFFFFFFF0LL];
    do
    {
      if ( v25 )
        *v25 = _mm_loadu_si128(v27);
      ++v25;
      ++v27;
    }
    while ( v25 != &v20[(unsigned __int64)v17 / 0x10] );
    v3[4] += v17;
    v28 = v26 - v19;
    if ( (char *)v26 - (char *)v19 > 0 )
    {
      do
      {
        v29 = v26[-1].m128i_i64[0];
        --v26;
        --v20;
        v20->m128i_i64[0] = v29;
        v20->m128i_i64[1] = v26->m128i_i64[1];
        --v28;
      }
      while ( v28 );
    }
    if ( v17 )
    {
      do
      {
        v30 = v16->m128i_i64[0];
        ++v16;
        ++v19;
        v19[-1].m128i_i64[0] = v30;
        v19[-1].m128i_i64[1] = v16[-1].m128i_i64[1];
      }
      while ( v18 != v16 );
    }
LABEL_31:
    v18 = (const __m128i *)v253;
    goto LABEL_32;
  }
  v194 = (const __m128i *)&v253[v22];
  if ( v18 == (const __m128i *)&v253[v22] )
  {
    v196 = v220[4];
  }
  else
  {
    v195 = (__m128i *)v220[4];
    do
    {
      if ( v195 )
        *v195 = _mm_loadu_si128(v194);
      ++v195;
      ++v194;
    }
    while ( v195 != &v19[(unsigned __int64)v17 / 0x10] );
    v196 = v3[4];
  }
  v197 = (__m128i *)(v196 + 16 * (v21 - v23));
  v3[4] = (__int64)v197;
  if ( v19 != v20 )
  {
    v198 = v19;
    v199 = (__m128i *)((char *)v197 + (char *)v20 - (char *)v19);
    do
    {
      if ( v197 )
        *v197 = _mm_loadu_si128(v198);
      ++v197;
      ++v198;
    }
    while ( v197 != v199 );
    v197 = (__m128i *)v3[4];
  }
  v3[4] = (__int64)v197->m128i_i64 + v22;
  if ( v22 <= 0 )
    goto LABEL_31;
  do
  {
    v200 = v16->m128i_i64[0];
    ++v19;
    ++v16;
    v19[-1].m128i_i64[0] = v200;
    v19[-1].m128i_i64[1] = v16[-1].m128i_i64[1];
    --v24;
  }
  while ( v24 );
  v18 = (const __m128i *)v253;
LABEL_32:
  if ( v18 != (const __m128i *)v219 )
    _libc_free((unsigned __int64)v18);
LABEL_34:
  if ( v3[4] == v3[3] )
    return 0;
  v31 = (__int64 *)sub_B2BE50(v215);
  v32 = (__int64 *)sub_BCE3C0(v31, 0);
  v33 = v3[3];
  v220 = (unsigned __int64 *)v32;
  v253 = v219;
  v254 = 0x1000000000LL;
  if ( v3[4] == v33 )
  {
    v44 = 0;
    v217.m128i_i64[0] = (__int64)&v253;
  }
  else
  {
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v217.m128i_i64[0] = (__int64)&v253;
    do
    {
      ++v36;
      v37 = *(_QWORD *)(*(_QWORD *)(v33 + 16 * v35)
                      + 32 * (1LL - (*(_DWORD *)(*(_QWORD *)(v33 + 16 * v35) + 4LL) & 0x7FFFFFF)));
      v38 = !sub_AC30F0(v37);
      v41 = (unsigned int)v254;
      if ( v38 )
        v34 = v36;
      if ( (unsigned __int64)(unsigned int)v254 + 1 > HIDWORD(v254) )
      {
        sub_C8D5F0(v217.m128i_i64[0], v219, (unsigned int)v254 + 1LL, 8u, v39, v40);
        v41 = (unsigned int)v254;
      }
      *(_QWORD *)&v253[8 * v41] = v37;
      v35 = v36;
      v33 = v3[3];
      v42 = (unsigned int)(v254 + 1);
      v43 = v3[4];
      LODWORD(v254) = v254 + 1;
    }
    while ( v36 != (v43 - v33) >> 4 );
    v44 = v34;
    if ( v34 != v42 )
    {
      if ( v34 >= v42 )
      {
        if ( v34 > (unsigned __int64)HIDWORD(v254) )
        {
          sub_C8D5F0(v217.m128i_i64[0], v219, v34, 8u, v39, v40);
          v42 = (unsigned int)v254;
        }
        v45 = &v253[8 * v42];
        for ( i = &v253[8 * v34]; i != v45; v45 += 8 )
        {
          if ( v45 )
            *(_QWORD *)v45 = 0;
        }
      }
      LODWORD(v254) = v34;
    }
  }
  v47 = (_QWORD *)sub_B2BE50(v215);
  v48 = sub_BCB2D0(v47);
  v226[0] = sub_AD64C0(v48, (v3[4] - v3[3]) >> 4, 0);
  v49 = sub_AD64C0(v48, v44, 0);
  v50 = (__int64 **)v3[2];
  v226[1] = v49;
  v51 = sub_AD24A0(v50, v226, 2);
  v52 = (__int64 *)v253;
  v53 = (unsigned int)v254;
  v227 = v51;
  v54 = (__int64 **)sub_BCD420((__int64 *)v220, v44);
  v228 = sub_AD1300(v54, v52, v53);
  v229[0] = *(_QWORD *)(v227 + 8);
  v229[1] = *(_QWORD *)(v228 + 8);
  if ( !v44 )
  {
    v237.m128i_i8[4] = 48;
    v232 = (char *)v234;
    v55 = &v237.m128i_i8[4];
    v220 = (unsigned __int64 *)&v232;
LABEL_53:
    v56 = 1;
    LOBYTE(v234[0]) = *v55;
    v57 = v234;
    goto LABEL_54;
  }
  v55 = &v237.m128i_i8[5];
  do
  {
    *--v55 = v44 % 0xA + 48;
    v175 = v44;
    v44 /= 0xAu;
  }
  while ( v175 > 9 );
  v176 = (char *)(&v237.m128i_u8[5] - (unsigned __int8 *)v55);
  v220 = (unsigned __int64 *)&v232;
  v56 = &v237.m128i_u8[5] - (unsigned __int8 *)v55;
  v232 = (char *)v234;
  v225 = &v237.m128i_u8[5] - (unsigned __int8 *)v55;
  if ( (unsigned __int64)(&v237.m128i_u8[5] - (unsigned __int8 *)v55) > 0xF )
  {
    v232 = (char *)sub_22409D0((__int64)&v232, &v225, 0);
    v177 = (__int64 *)v232;
    v234[0] = v225;
LABEL_130:
    memcpy(v177, v55, v56);
    v56 = v225;
    v57 = (__int64 *)v232;
    goto LABEL_54;
  }
  if ( v176 == (char *)1 )
    goto LABEL_53;
  if ( v176 )
  {
    v177 = v234;
    goto LABEL_130;
  }
  v57 = v234;
LABEL_54:
  v233 = (_BYTE *)v56;
  v58 = v220;
  *((_BYTE *)v57 + v56) = 0;
  v59 = (__m128i *)sub_2241130(v58, 0, 0, "gc_map.", 7u);
  v236.m128i_i64[0] = (__int64)&v237;
  if ( (__m128i *)v59->m128i_i64[0] == &v59[1] )
  {
    v237 = _mm_loadu_si128(v59 + 1);
  }
  else
  {
    v236.m128i_i64[0] = v59->m128i_i64[0];
    v237.m128i_i64[0] = v59[1].m128i_i64[0];
  }
  v236.m128i_i64[1] = v59->m128i_i64[1];
  v59->m128i_i64[0] = (__int64)v59[1].m128i_i64;
  v59->m128i_i64[1] = 0;
  v59[1].m128i_i8[0] = 0;
  v60 = (__int64 **)sub_BD0EC0(v229, 2, (const void *)v236.m128i_i64[0], v236.m128i_u64[1], 0);
  if ( (__m128i *)v236.m128i_i64[0] != &v237 )
    j_j___libc_free_0(v236.m128i_u64[0]);
  if ( v232 != (char *)v234 )
    j_j___libc_free_0((unsigned __int64)v232);
  v61 = sub_AD24A0(v60, &v227, 2);
  v62 = *(_QWORD **)(v61 + 8);
  v63 = v61;
  v64 = *(_QWORD *)(v215 + 40);
  v65.m128i_i64[0] = (__int64)sub_BD5D20(v215);
  v237 = v65;
  v238 = 1283;
  v236.m128i_i64[0] = (__int64)"__gc_";
  BYTE4(v232) = 0;
  v66 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
  if ( v66 )
    sub_B30000((__int64)v66, v64, v62, 1, 7, v63, (__int64)&v236, 0, 0, (__int64)v232, 0);
  v67 = (_QWORD *)sub_B2BE50(v215);
  v68 = sub_BCB2D0(v67);
  v232 = (char *)sub_ACD640(v68, 0, 0);
  v69 = (_QWORD *)sub_B2BE50(v215);
  v70 = sub_BCB2D0(v69);
  v233 = (_BYTE *)sub_ACD640(v70, 0, 0);
  v71 = *(_QWORD *)(v63 + 8);
  LOBYTE(v238) = 0;
  v210 = sub_AD9FD0(v71, v66, (__int64 *)v220, 2, 0, (__int64)&v236, 0);
  if ( (_BYTE)v238 )
  {
    LOBYTE(v238) = 0;
    if ( v237.m128i_i32[2] > 0x40u && v237.m128i_i64[0] )
      j_j___libc_free_0_0(v237.m128i_u64[0]);
    if ( v236.m128i_i32[2] > 0x40u && v236.m128i_i64[0] )
      j_j___libc_free_0_0(v236.m128i_u64[0]);
  }
  if ( v253 != v219 )
    _libc_free((unsigned __int64)v253);
  v72 = v3[1];
  v232 = 0;
  v233 = 0;
  v234[0] = 0;
  v253 = (const char *)v72;
  sub_9183A0((__int64)v220, 0, v217.m128i_i64[0]);
  v73 = v3[3];
  v74 = v3[4];
  if ( v73 != v74 )
  {
    v75 = (__int64 *)v220;
    v76 = (_QWORD *)v217.m128i_i64[0];
    do
    {
      while ( 1 )
      {
        v77 = v233;
        v78 = *(_QWORD *)(*(_QWORD *)(v73 + 8) + 72LL);
        v253 = (const char *)v78;
        if ( v233 != (_BYTE *)v234[0] )
          break;
        v73 += 16;
        sub_9183A0((__int64)v75, v233, v76);
        if ( v74 == v73 )
          goto LABEL_72;
      }
      if ( v233 )
      {
        *(_QWORD *)v233 = v78;
        v77 = v233;
      }
      v73 += 16;
      v233 = v77 + 8;
    }
    while ( v74 != v73 );
  }
LABEL_72:
  v79 = sub_BD5D20(v215);
  LOWORD(v257) = 1283;
  v253 = "gc_stackentry.";
  v255 = v79;
  v256 = v80;
  sub_CA0F50(v236.m128i_i64, (void **)v217.m128i_i64[0]);
  v219 = (const char *)sub_BD0EC0(
                         (__int64 *)v232,
                         (v233 - v232) >> 3,
                         (const void *)v236.m128i_i64[0],
                         v236.m128i_u64[1],
                         0);
  if ( (__m128i *)v236.m128i_i64[0] != &v237 )
    j_j___libc_free_0(v236.m128i_u64[0]);
  if ( v232 )
    j_j___libc_free_0((unsigned __int64)v232);
  v81 = *(_QWORD *)(v215 + 80);
  if ( !v81 )
LABEL_206:
    BUG();
  v82 = *(_QWORD *)(v81 + 32);
  if ( !v82 )
    BUG();
  v83 = *(_QWORD *)(v82 + 16);
  v242 = (__int64 *)sub_AA48A0(v83);
  v243 = &v251;
  v244 = &v252;
  v236.m128i_i64[0] = (__int64)&v237;
  v236.m128i_i64[1] = 0x200000000LL;
  v251 = &unk_49DA100;
  LOWORD(v241) = 0;
  v247 = 512;
  v252 = &unk_49DA0B0;
  v245 = 0;
  v246 = 0;
  v248 = 7;
  v249 = 0;
  v250 = 0;
  v239 = 0;
  v240 = 0;
  sub_A88F30((__int64)&v236, v83, v82, 1);
  v232 = "gc_frame";
  v235 = 259;
  v84 = sub_AA4E30(v239);
  v85 = (__int64 *)v219;
  v86 = v84;
  v87 = sub_AE5260(v84, (__int64)v219);
  v88 = *(_DWORD *)(v86 + 4);
  v89 = v87;
  LOWORD(v257) = 257;
  v214 = sub_BD2C40(80, 1u);
  if ( v214 )
    sub_B4CCA0((__int64)v214, v85, v88, 0, v89, v217.m128i_i64[0], 0, 0);
  (*((void (__fastcall **)(void **, _QWORD *, unsigned __int64 *, __int64, __int64))*v244 + 2))(
    v244,
    v214,
    v220,
    v240,
    v241);
  v90 = v236.m128i_i64[0];
  v91 = v236.m128i_i64[0] + 16LL * v236.m128i_u32[2];
  if ( v236.m128i_i64[0] != v91 )
  {
    do
    {
      v92 = *(_QWORD *)(v90 + 8);
      v93 = *(_DWORD *)v90;
      v90 += 16;
      sub_B99FD0((__int64)v214, v93, v92);
    }
    while ( v91 != v90 );
  }
  v94 = *(_QWORD *)(v215 + 80);
  if ( v94 )
    v94 -= 24;
  v239 = v94;
  v95 = sub_AA5BA0(v94);
  v97 = v95;
  if ( v95 )
  {
    v221 = v96;
    v205 = HIBYTE(v96);
  }
  else
  {
    v205 = 0;
    v221 = 0;
  }
  v98 = v221;
  v223 = v95;
  BYTE1(v98) = v205;
  v240 = v95;
  v224 = v98;
  v99 = *v3;
  LOWORD(v241) = v98;
  v100 = sub_BCE3C0(v242, 0);
  v235 = 259;
  v232 = "gc_currhead";
  v101 = sub_AA4E30(v239);
  v102 = sub_AE5020(v101, v100);
  LOWORD(v257) = 257;
  v203 = v102;
  v103 = sub_BD2C40(80, 1u);
  v105 = (__int64)v103;
  if ( v103 )
  {
    sub_B4D190((__int64)v103, v100, v99, v217.m128i_i64[0], 0, v203, 0, 0);
    v104 = v201;
  }
  (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64, __int64))*v244 + 2))(
    v244,
    v105,
    v220,
    v240,
    v241,
    v104);
  v106 = v236.m128i_i64[0];
  v107 = v236.m128i_i64[0] + 16LL * v236.m128i_u32[2];
  if ( v236.m128i_i64[0] != v107 )
  {
    do
    {
      v108 = *(_QWORD *)(v106 + 8);
      v109 = *(_DWORD *)v106;
      v106 += 16;
      sub_B99FD0(v105, v109, v108);
    }
    while ( v107 != v106 );
  }
  v110 = sub_2FA5990(v218, (unsigned int **)&v236, (__int64)v219, (__int64)v214, 1, "gc_frame.map");
  v111 = sub_AA4E30(v239);
  v208 = sub_AE5020(v111, *(_QWORD *)(v210 + 8));
  LOWORD(v257) = 257;
  v112 = sub_BD2C40(80, unk_3F10A10);
  v114 = (__int64)v112;
  if ( v112 )
    sub_B4D3C0((__int64)v112, v210, (__int64)v110, 0, v208, v113, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64, __int64, __int64))*v244 + 2))(
    v244,
    v114,
    v217.m128i_i64[0],
    v240,
    v241);
  v115 = 16LL * v236.m128i_u32[2];
  if ( v236.m128i_i64[0] != v236.m128i_i64[0] + v115 )
  {
    v211 = v3;
    v116 = v236.m128i_i64[0] + v115;
    v117 = v236.m128i_i64[0];
    do
    {
      v118 = *(_QWORD *)(v117 + 8);
      v119 = *(_DWORD *)v117;
      v117 += 16;
      sub_B99FD0(v114, v119, v118);
    }
    while ( v116 != v117 );
    v3 = v211;
  }
  v120 = (v3[4] - v3[3]) >> 4;
  if ( (_DWORD)v120 )
  {
    v204 = v97;
    v202 = v105;
    v212 = v3;
    v121 = 0;
    do
    {
      v122 = sub_2FA58E0(v218, (unsigned int **)&v236, (__int64)v219, (__int64)v214, (int)v121 + 1, "gc_root");
      v123 = v121++;
      v124 = *(unsigned __int8 **)(v212[3] + 16 * v123 + 8);
      sub_BD6B90(v122, v124);
      sub_BD84D0((__int64)v124, (__int64)v122);
    }
    while ( (unsigned int)v120 != v121 );
    v97 = v204;
    v125 = v205;
    v126 = v221;
    v105 = v202;
    v3 = v212;
  }
  else
  {
    v125 = v205;
    v126 = v221;
  }
  while ( 1 )
  {
    if ( !v97 )
      goto LABEL_205;
    if ( *(_BYTE *)(v97 - 24) != 62 )
      break;
    v97 = *(_QWORD *)(v97 + 8);
    v125 = 0;
    v126 = 0;
  }
  LOBYTE(v127) = v126;
  HIBYTE(v127) = v125;
  sub_A88F30((__int64)&v236, *(_QWORD *)(v97 + 16), v97, v127);
  v128 = v218;
  v129 = sub_2FA5990(v218, (unsigned int **)&v236, (__int64)v219, (__int64)v214, 0, "gc_frame.next");
  v130 = sub_2FA58E0(v128, (unsigned int **)&v236, (__int64)v219, (__int64)v214, 0, "gc_newhead");
  v131 = sub_AA4E30(v239);
  v132 = sub_AE5020(v131, *(_QWORD *)(v105 + 8));
  LOWORD(v257) = 257;
  v209 = v132;
  v133 = sub_BD2C40(80, unk_3F10A10);
  v135 = (__int64)v133;
  if ( v133 )
    sub_B4D3C0((__int64)v133, v105, (__int64)v129, 0, v209, v134, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64, __int64, __int64))*v244 + 2))(
    v244,
    v135,
    v217.m128i_i64[0],
    v240,
    v241);
  v136 = v236.m128i_i64[0];
  v137 = v236.m128i_i64[0] + 16LL * v236.m128i_u32[2];
  if ( v236.m128i_i64[0] != v137 )
  {
    do
    {
      v138 = *(_QWORD *)(v136 + 8);
      v139 = *(_DWORD *)v136;
      v136 += 16;
      sub_B99FD0(v135, v139, v138);
    }
    while ( v137 != v136 );
  }
  v140 = *v3;
  v141 = sub_AA4E30(v239);
  v142 = sub_AE5020(v141, *((_QWORD *)v130 + 1));
  LOWORD(v257) = 257;
  v143 = sub_BD2C40(80, unk_3F10A10);
  v145 = (__int64)v143;
  if ( v143 )
    sub_B4D3C0((__int64)v143, (__int64)v130, v140, 0, v142, v144, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64, __int64, __int64))*v244 + 2))(
    v244,
    v145,
    v217.m128i_i64[0],
    v240,
    v241);
  v146 = v236.m128i_i64[0];
  v147 = v236.m128i_i64[0] + 16LL * v236.m128i_u32[2];
  if ( v236.m128i_i64[0] != v147 )
  {
    do
    {
      v148 = *(_QWORD *)(v146 + 8);
      v149 = *(_DWORD *)v146;
      v146 += 16;
      sub_B99FD0(v145, v149, v148);
    }
    while ( v147 != v146 );
  }
  v254 = (__int64)"gc_cleanup";
  v150 = *(const char **)(v215 + 80);
  v253 = (const char *)v215;
  v255 = v150;
  v256 = v207;
  v263 = sub_B2BE50(v215);
  v264 = &v272;
  v265 = &v273;
  v257 = v259;
  v272 = &unk_49DA100;
  v258 = 0x200000000LL;
  v268 = 512;
  v273 = &unk_49DA0B0;
  v266 = 0;
  v267 = 0;
  v269 = 7;
  v270 = 0;
  v271 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v274 = 256;
  v275 = a3;
  v213 = v3;
  while ( 1 )
  {
    v151 = (unsigned int **)sub_29CEC00(v217.m128i_i64[0]);
    v152 = v151;
    if ( !v151 )
      break;
    v153 = sub_2FA5990(v218, v151, (__int64)v219, (__int64)v214, 0, "gc_frame.next");
    v154 = sub_BCE3C0((__int64 *)v152[9], 0);
    v231 = 1;
    v155 = v154;
    v230 = 3;
    v229[0] = (__int64)"gc_savedhead";
    v156 = sub_AA4E30((__int64)v152[6]);
    v157 = sub_AE5020(v156, v155);
    HIBYTE(v158) = HIBYTE(v222);
    v235 = 257;
    LOBYTE(v158) = v157;
    v222 = v158;
    v159 = sub_BD2C40(80, 1u);
    v160 = (__int64)v159;
    if ( v159 )
      sub_B4D190((__int64)v159, v155, (__int64)v153, (__int64)v220, 0, v222, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)v152[11]
                                                                                               + 16LL))(
      v152[11],
      v160,
      v229,
      v152[7],
      v152[8]);
    v161 = *v152;
    v162 = (__int64)&(*v152)[4 * *((unsigned int *)v152 + 2)];
    if ( *v152 != (unsigned int *)v162 )
    {
      do
      {
        v163 = *((_QWORD *)v161 + 1);
        v164 = *v161;
        v161 += 4;
        sub_B99FD0(v160, v164, v163);
      }
      while ( (unsigned int *)v162 != v161 );
    }
    v216 = *v213;
    v165 = sub_AA4E30((__int64)v152[6]);
    v166 = sub_AE5020(v165, *(_QWORD *)(v160 + 8));
    v235 = 257;
    v167 = sub_BD2C40(80, unk_3F10A10);
    v169 = (__int64)v167;
    if ( v167 )
      sub_B4D3C0((__int64)v167, v160, v216, 0, v166, v168, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, unsigned __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)v152[11] + 16LL))(
      v152[11],
      v169,
      v220,
      v152[7],
      v152[8]);
    v170 = *v152;
    v171 = (__int64)&(*v152)[4 * *((unsigned int *)v152 + 2)];
    while ( (unsigned int *)v171 != v170 )
    {
      v172 = *((_QWORD *)v170 + 1);
      v173 = *v170;
      v170 += 4;
      sub_B99FD0(v169, v173, v172);
    }
  }
  v189 = (_QWORD **)v213[3];
  v190 = (_QWORD **)v213[4];
  if ( v189 != v190 )
  {
    do
    {
      v191 = *v189;
      v189 += 2;
      sub_B43D60(v191);
      sub_B43D60(*(v189 - 1));
    }
    while ( v190 != v189 );
    v192 = v213[3];
    if ( v213[4] != v192 )
      v213[4] = v192;
  }
  nullsub_61();
  v272 = &unk_49DA100;
  nullsub_63();
  if ( v257 != v259 )
    _libc_free((unsigned __int64)v257);
  nullsub_61();
  v251 = &unk_49DA100;
  nullsub_63();
  if ( (__m128i *)v236.m128i_i64[0] != &v237 )
    _libc_free(v236.m128i_u64[0]);
  return 1;
}
