// Function: sub_2404C10
// Address: 0x2404c10
//
__int64 *__fastcall sub_2404C10(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rbx
  __int64 v6; // rbx
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  _QWORD *v10; // r13
  const __m128i *v11; // rax
  unsigned __int64 v12; // rax
  __int8 *v13; // rax
  const __m128i *v14; // rax
  const __m128i *v15; // rsi
  const __m128i *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // r9
  const __m128i *v20; // rcx
  const __m128i *v21; // rax
  const __m128i *v22; // rdx
  __int64 v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rcx
  const __m128i *v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  __m128i *v30; // rax
  __m128i *v31; // rcx
  __m128i *v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 *result; // rax
  __int8 v38; // cl
  unsigned __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rcx
  __m128i *v49; // rdx
  const __m128i *v50; // rax
  __int64 v51; // r9
  unsigned __int64 v52; // rcx
  __int64 v53; // rax
  __m128i *v54; // rdx
  __m128i *v55; // rcx
  const __m128i *v56; // rax
  unsigned __int64 v57; // rax
  __int64 *v58; // r12
  _QWORD *v59; // rbx
  __int64 v60; // rcx
  __int64 v61; // rax
  unsigned __int64 v62; // r8
  __int64 v63; // rax
  __int64 v64; // r8
  _QWORD *v65; // r12
  __int64 v66; // rbx
  __int64 v67; // r13
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  char v70; // al
  __int64 v72; // r14
  unsigned __int64 v73; // rax
  void **v74; // r12
  unsigned int v75; // esi
  __int64 v76; // r11
  unsigned int v77; // ecx
  _QWORD *v78; // rax
  void *v79; // r9
  void *v80; // rdx
  int v81; // ecx
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  unsigned __int64 v90; // rsi
  __int64 v91; // rdx
  __int64 v92; // rax
  __int64 *v93; // rbx
  __int64 v94; // rdx
  __int64 v95; // r15
  __int64 *v96; // r12
  __int64 v97; // rax
  __m128i v98; // xmm1
  __m128i v99; // xmm0
  __m128i v100; // xmm7
  __int64 v101; // rdx
  int v102; // eax
  int v103; // eax
  __int64 *v104; // rcx
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 *v109; // rax
  __int64 v110; // rdx
  __int64 *v111; // rbx
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // r12
  __int64 *v117; // r14
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rax
  int v121; // r8d
  _QWORD *v122; // rdi
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 *v132; // r14
  int v133; // ecx
  int v134; // r8d
  __int64 *v135; // rcx
  __int64 *v136; // rsi
  __int64 *v137; // rdi
  __int64 *v138; // rax
  signed __int64 v139; // rbx
  __int64 v140; // rax
  _QWORD *v141; // rdx
  _QWORD *v142; // rax
  signed __int64 v143; // rdi
  unsigned int v144; // ecx
  unsigned int v145; // edx
  int v146; // ebx
  unsigned int v147; // eax
  __int64 v148; // rax
  __int64 v149; // rcx
  __int64 v150; // r9
  __m128i v151; // xmm7
  __m128i v152; // xmm0
  __m128i v153; // xmm7
  unsigned __int8 v154; // cl
  double v155; // xmm0_8
  unsigned __int64 v156; // rdi
  unsigned int v157; // eax
  __int64 *v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 *v161; // rax
  int v162; // edx
  _QWORD *v163; // [rsp+10h] [rbp-640h]
  __int64 *v164; // [rsp+18h] [rbp-638h]
  __int64 v165; // [rsp+18h] [rbp-638h]
  __int64 *v166; // [rsp+20h] [rbp-630h]
  unsigned __int8 v167; // [rsp+20h] [rbp-630h]
  unsigned __int64 v168; // [rsp+28h] [rbp-628h]
  __int64 v169; // [rsp+28h] [rbp-628h]
  __int64 v170; // [rsp+28h] [rbp-628h]
  unsigned __int64 v171; // [rsp+30h] [rbp-620h]
  const __m128i *v172; // [rsp+30h] [rbp-620h]
  void **v173; // [rsp+30h] [rbp-620h]
  __int64 v174; // [rsp+30h] [rbp-620h]
  __int64 v175; // [rsp+30h] [rbp-620h]
  unsigned int v176; // [rsp+30h] [rbp-620h]
  __int64 v178; // [rsp+40h] [rbp-610h]
  unsigned __int64 v179; // [rsp+48h] [rbp-608h]
  unsigned __int64 v180; // [rsp+48h] [rbp-608h]
  __int64 *v181; // [rsp+48h] [rbp-608h]
  const __m128i *v182; // [rsp+50h] [rbp-600h]
  signed __int64 v183; // [rsp+58h] [rbp-5F8h]
  unsigned __int64 v184; // [rsp+58h] [rbp-5F8h]
  __int64 v185; // [rsp+58h] [rbp-5F8h]
  __int64 *v186; // [rsp+58h] [rbp-5F8h]
  unsigned int v187; // [rsp+58h] [rbp-5F8h]
  __int64 *v188; // [rsp+68h] [rbp-5E8h] BYREF
  _BYTE *v189; // [rsp+70h] [rbp-5E0h] BYREF
  __int64 v190; // [rsp+78h] [rbp-5D8h]
  _BYTE v191[64]; // [rsp+80h] [rbp-5D0h] BYREF
  _BYTE v192[32]; // [rsp+C0h] [rbp-590h] BYREF
  _BYTE v193[64]; // [rsp+E0h] [rbp-570h] BYREF
  __m128i *v194; // [rsp+120h] [rbp-530h]
  __int64 v195; // [rsp+128h] [rbp-528h]
  __int8 *v196; // [rsp+130h] [rbp-520h]
  _DWORD v197[8]; // [rsp+140h] [rbp-510h] BYREF
  _BYTE v198[64]; // [rsp+160h] [rbp-4F0h] BYREF
  __m128i *v199; // [rsp+1A0h] [rbp-4B0h]
  __m128i *i; // [rsp+1A8h] [rbp-4A8h]
  __int8 *v201; // [rsp+1B0h] [rbp-4A0h]
  __int64 v202; // [rsp+1C0h] [rbp-490h] BYREF
  _QWORD *v203; // [rsp+1C8h] [rbp-488h]
  __int64 v204; // [rsp+1D0h] [rbp-480h]
  __int64 v205; // [rsp+1D8h] [rbp-478h]
  char v206[64]; // [rsp+1E0h] [rbp-470h] BYREF
  __int64 v207; // [rsp+220h] [rbp-430h]
  __int64 v208; // [rsp+228h] [rbp-428h]
  __int8 *v209; // [rsp+230h] [rbp-420h]
  unsigned __int64 *v210; // [rsp+240h] [rbp-410h] BYREF
  _QWORD *v211; // [rsp+248h] [rbp-408h]
  _BYTE *v212; // [rsp+250h] [rbp-400h] BYREF
  __int64 v213; // [rsp+258h] [rbp-3F8h]
  _BYTE v214[64]; // [rsp+260h] [rbp-3F0h] BYREF
  __m128i *v215; // [rsp+2A0h] [rbp-3B0h]
  unsigned __int64 v216; // [rsp+2A8h] [rbp-3A8h]
  __int8 *v217; // [rsp+2B0h] [rbp-3A0h]
  void *v218; // [rsp+2C0h] [rbp-390h] BYREF
  int v219; // [rsp+2C8h] [rbp-388h]
  char v220; // [rsp+2CCh] [rbp-384h]
  unsigned __int64 *v221; // [rsp+2D0h] [rbp-380h] BYREF
  __m128i v222; // [rsp+2D8h] [rbp-378h] BYREF
  __int64 v223; // [rsp+2E8h] [rbp-368h]
  __m128i v224; // [rsp+2F0h] [rbp-360h]
  __m128i v225; // [rsp+300h] [rbp-350h]
  const __m128i **v226; // [rsp+310h] [rbp-340h] BYREF
  __int64 v227; // [rsp+318h] [rbp-338h]
  const __m128i *v228; // [rsp+320h] [rbp-330h] BYREF
  const __m128i *v229; // [rsp+328h] [rbp-328h]
  __int8 *v230; // [rsp+330h] [rbp-320h]
  _BYTE v231[32]; // [rsp+338h] [rbp-318h] BYREF
  char v232[64]; // [rsp+358h] [rbp-2F8h] BYREF
  const __m128i *v233; // [rsp+398h] [rbp-2B8h]
  const __m128i *v234; // [rsp+3A0h] [rbp-2B0h]
  __int64 v235; // [rsp+3A8h] [rbp-2A8h]
  char v236; // [rsp+460h] [rbp-1F0h]
  int v237; // [rsp+464h] [rbp-1ECh]
  __int64 v238; // [rsp+468h] [rbp-1E8h]
  _BYTE v239[120]; // [rsp+470h] [rbp-1E0h] BYREF
  _BYTE v240[96]; // [rsp+4E8h] [rbp-168h] BYREF
  const __m128i *v241; // [rsp+548h] [rbp-108h]
  const __m128i *v242; // [rsp+550h] [rbp-100h]
  char v243; // [rsp+610h] [rbp-40h]
  int v244; // [rsp+614h] [rbp-3Ch]
  __int64 v245; // [rsp+618h] [rbp-38h]

  v3 = *a2;
  v4 = a2[4];
  v188 = 0;
  v171 = v4;
  v168 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 != (unsigned __int64 *)sub_22DBE80(a1[4], v3 & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_8;
    }
LABEL_6:
    if ( !(unsigned __int8)sub_22DB400(a2, *(_QWORD *)(v7 + 40)) )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_8;
        v7 = *(_QWORD *)(v6 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
          goto LABEL_6;
      }
    }
    return 0;
  }
LABEL_8:
  v8 = *a2;
  v9 = a2[4];
  memset(v239, 0, sizeof(v239));
  *(_QWORD *)&v239[8] = &v239[32];
  v10 = v239;
  *(_DWORD *)&v239[16] = 8;
  v239[28] = 1;
  sub_23FEE00((__int64)&v210, v8 & 0xFFFFFFFFFFFFFFF8LL, v9);
  sub_C8CF70((__int64)&v218, &v222.m128i_u64[1], 8, (__int64)v214, (__int64)&v210);
  v11 = v215;
  v215 = 0;
  v228 = v11;
  v12 = v216;
  v216 = 0;
  v229 = (const __m128i *)v12;
  v13 = v217;
  v217 = 0;
  v230 = v13;
  sub_C8CF70((__int64)v231, v232, 8, (__int64)&v239[32], (__int64)v239);
  v14 = *(const __m128i **)&v239[96];
  memset(&v239[96], 0, 24);
  v233 = v14;
  v234 = *(const __m128i **)&v239[104];
  v235 = *(_QWORD *)&v239[112];
  sub_23FD540((__int64)&v210);
  sub_23FD540((__int64)v239);
  v15 = (const __m128i *)v193;
  v16 = (const __m128i *)v192;
  sub_C8CD80((__int64)v192, (__int64)v193, (__int64)&v218, v17, (__int64)v192, v18);
  v20 = v229;
  v21 = v228;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v22 = (const __m128i *)((char *)v229 - (char *)v228);
  if ( v229 == v228 )
  {
    v25 = 0;
    v24 = 0;
  }
  else
  {
    if ( (unsigned __int64)v22 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_241;
    v183 = (char *)v229 - (char *)v228;
    v23 = sub_22077B0((char *)v229 - (char *)v228);
    v20 = v229;
    v24 = (__m128i *)v23;
    v21 = v228;
    v25 = v183;
  }
  v194 = v24;
  v195 = (__int64)v24;
  v196 = &v24->m128i_i8[v25];
  if ( v21 == v20 )
  {
    v26 = (__int64)v24;
  }
  else
  {
    v26 = (__int64)v24->m128i_i64 + (char *)v20 - (char *)v21;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v21);
        v24[1] = _mm_loadu_si128(v21 + 1);
      }
      v24 += 2;
      v21 += 2;
    }
    while ( v24 != (__m128i *)v26 );
  }
  v15 = (const __m128i *)v198;
  v195 = v26;
  sub_C8CD80((__int64)v197, (__int64)v198, (__int64)v231, v26, (__int64)v192, v19);
  v27 = v234;
  v22 = v233;
  v199 = 0;
  i = 0;
  v201 = 0;
  v16 = (const __m128i *)((char *)v234 - (char *)v233);
  if ( v234 == v233 )
  {
    v29 = 0;
  }
  else
  {
    if ( (unsigned __int64)v16 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_241;
    v182 = (const __m128i *)((char *)v234 - (char *)v233);
    v28 = sub_22077B0((unsigned __int64)v16);
    v27 = v234;
    v22 = v233;
    v16 = v182;
    v29 = v28;
  }
  v199 = (__m128i *)v29;
  i = (__m128i *)v29;
  v201 = &v16->m128i_i8[v29];
  if ( v22 == v27 )
  {
    v31 = (__m128i *)v29;
  }
  else
  {
    v30 = (__m128i *)v29;
    v31 = (__m128i *)(v29 + (char *)v27 - (char *)v22);
    do
    {
      if ( v30 )
      {
        *v30 = _mm_loadu_si128(v22);
        v30[1] = _mm_loadu_si128(v22 + 1);
      }
      v30 += 2;
      v22 += 2;
    }
    while ( v30 != v31 );
  }
  for ( i = v31; ; v31 = i )
  {
    v32 = v194;
    if ( (__m128i *)(v195 - (_QWORD)v194) != (__m128i *)((char *)v31 - v29) )
      goto LABEL_29;
    if ( v194 == (__m128i *)v195 )
      break;
    while ( v32->m128i_i64[0] == *(_QWORD *)v29 )
    {
      v38 = v32[1].m128i_i8[8];
      if ( v38 != *(_BYTE *)(v29 + 24) || v38 && v32[1].m128i_i32[0] != *(_DWORD *)(v29 + 16) )
        break;
      v32 += 2;
      v29 += 32;
      if ( (__m128i *)v195 == v32 )
        goto LABEL_51;
    }
LABEL_29:
    v33 = *(_QWORD *)(v195 - 32);
    if ( (*(_WORD *)(v33 + 2) & 0x7FFF) != 0 )
    {
LABEL_40:
      sub_23FD540((__int64)v197);
      sub_23FD540((__int64)v192);
      sub_23FD540((__int64)v231);
      sub_23FD540((__int64)&v218);
      return 0;
    }
    v34 = *(_QWORD *)(v33 + 56);
    v35 = v33 + 48;
    if ( v35 != v34 )
    {
      while ( v34 )
      {
        if ( *(_BYTE *)(v34 - 24) == 85 )
        {
          v36 = *(_QWORD *)(v34 - 56);
          if ( v36 )
          {
            if ( !*(_BYTE *)v36
              && *(_QWORD *)(v36 + 24) == *(_QWORD *)(v34 + 56)
              && (*(_BYTE *)(v36 + 33) & 0x20) != 0
              && *(_DWORD *)(v36 + 36) == 48 )
            {
              goto LABEL_40;
            }
          }
        }
        v34 = *(_QWORD *)(v34 + 8);
        if ( v35 == v34 )
          goto LABEL_44;
      }
LABEL_258:
      BUG();
    }
LABEL_44:
    sub_23EC7E0((__int64)v192);
    v29 = (__int64)v199;
  }
LABEL_51:
  sub_23FD540((__int64)v197);
  sub_23FD540((__int64)v192);
  sub_23FD540((__int64)v231);
  sub_23FD540((__int64)&v218);
  if ( v171 )
  {
    v39 = sub_986580(v168);
    v169 = v39;
    if ( *(_BYTE *)v39 == 31 && (*(_DWORD *)(v39 + 4) & 0x7FFFFFF) == 3 )
    {
      v82 = *(_QWORD *)(v39 - 32);
      v83 = *(_QWORD *)(v39 - 64);
      if ( v83 != v82 && (v171 == v82 || v171 == v83) )
      {
        v210 = a2;
        LOBYTE(v211) = 0;
        v212 = v214;
        v213 = 0x800000000LL;
        if ( (*(_DWORD *)(v169 + 4) & 0x7FFFFFF) != 3 )
          goto LABEL_122;
        v197[0] = -1;
        LODWORD(v202) = -1;
        v154 = sub_23FAB40(v169, v197, &v202);
        if ( !v154 )
          goto LABEL_122;
        if ( *(_QWORD *)(v169 - 32) == a2[4] )
        {
          v162 = v202;
          LODWORD(v202) = v197[0];
          v197[0] = v162;
        }
        v155 = 1000000.0 * *(double *)&qword_4FE2C28;
        v218 = a2;
        v176 = v197[0];
        v187 = v202;
        if ( 1000000.0 * *(double *)&qword_4FE2C28 >= 9.223372036854776e18 )
          v156 = (unsigned int)(int)(v155 - 9.223372036854776e18) ^ 0x8000000000000000LL;
        else
          v156 = (unsigned int)(int)v155;
        v167 = v154;
        v157 = sub_F02DD0(v156, 0xF4240u);
        if ( v157 > v176 )
        {
          if ( v157 <= v187 )
          {
            sub_23FF920((__int64)v239, (__int64)(a1 + 13), (__int64 *)&v218);
            v158 = sub_23FF430((__int64)(a1 + 25), (__int64 *)&v218);
            v84 = v167;
            *(_DWORD *)v158 = v187;
            goto LABEL_123;
          }
LABEL_122:
          v84 = 0;
          goto LABEL_123;
        }
        sub_23FF920((__int64)v239, (__int64)(a1 + 9), (__int64 *)&v218);
        v161 = sub_23FF430((__int64)(a1 + 25), (__int64 *)&v218);
        v84 = v167;
        *(_DWORD *)v161 = v176;
LABEL_123:
        LOBYTE(v211) = v84;
        v239[8] = v84;
        *(_QWORD *)v239 = v210;
        *(_QWORD *)&v239[16] = &v239[32];
        *(_QWORD *)&v239[24] = 0x800000000LL;
        if ( (_DWORD)v213 )
          sub_23FAD70((__int64)&v239[16], (__int64)&v212, v82, v84, v40, v41);
        v85 = sub_22077B0(0x718u);
        if ( v85 )
        {
          v185 = v85;
          sub_23FB0E0(v85, (__int64)v239, v86, v87, v88, v89);
          v85 = v185;
        }
        v188 = (__int64 *)v85;
        if ( *(_BYTE **)&v239[16] != &v239[32] )
          _libc_free(*(unsigned __int64 *)&v239[16]);
        sub_23FBD30((__int64)v239, (__int64)(a1 + 33), (__int64 *)&v188);
        ++a1[6];
        if ( !(_BYTE)v211 )
        {
          v186 = (__int64 *)a1[5];
          v175 = *v186;
          v148 = sub_B2BE50(*v186);
          if ( sub_B6EA50(v148)
            || (v159 = sub_B2BE50(v175),
                v160 = sub_B6F970(v159),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v160 + 48LL))(v160)) )
          {
            sub_B176B0((__int64)v239, (__int64)"chr", (__int64)"BranchNotBiased", 15, v169);
            sub_B18290((__int64)v239, "Branch not biased", 0x11u);
            v151 = _mm_loadu_si128((const __m128i *)&v239[24]);
            v152 = _mm_loadu_si128((const __m128i *)&v239[48]);
            v219 = *(_DWORD *)&v239[8];
            v222 = v151;
            v153 = _mm_loadu_si128((const __m128i *)&v239[64]);
            v220 = v239[12];
            v224 = v152;
            v221 = *(unsigned __int64 **)&v239[16];
            v218 = &unk_49D9D40;
            v225 = v153;
            v223 = *(_QWORD *)&v239[40];
            v226 = &v228;
            v227 = 0x400000000LL;
            if ( *(_DWORD *)&v239[88] )
              sub_23FE010((__int64)&v226, (__int64)&v239[80], (__int64)&unk_49D9D30, v149, (__int64)&v226, v150);
            v236 = v243;
            v237 = v244;
            v238 = v245;
            v218 = &unk_49D9DB0;
            *(_QWORD *)v239 = &unk_49D9D40;
            sub_23FD590((__int64)&v239[80]);
            sub_1049740(v186, (__int64)&v218);
            v218 = &unk_49D9D40;
            sub_23FD590((__int64)&v226);
          }
        }
        if ( v212 != v214 )
          _libc_free((unsigned __int64)v212);
      }
    }
  }
  v189 = v191;
  v190 = 0x800000000LL;
  sub_22DE850((__int64)&v218, a2);
  sub_22DE7C0((__int64)&v210, a2);
  sub_23FD870(v239, &v210, &v218);
  sub_23FD500((__int64)&v210);
  sub_23FD500((__int64)&v218);
  sub_C8CD80((__int64)&v202, (__int64)v206, (__int64)v239, v42, v43, v44);
  v15 = *(const __m128i **)&v239[104];
  v16 = *(const __m128i **)&v239[96];
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v22 = (const __m128i *)(*(_QWORD *)&v239[104] - *(_QWORD *)&v239[96]);
  if ( *(_QWORD *)&v239[104] != *(_QWORD *)&v239[96] )
  {
    if ( (unsigned __int64)v22 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v172 = (const __m128i *)(*(_QWORD *)&v239[104] - *(_QWORD *)&v239[96]);
      v47 = sub_22077B0(*(_QWORD *)&v239[104] - *(_QWORD *)&v239[96]);
      v15 = *(const __m128i **)&v239[104];
      v16 = *(const __m128i **)&v239[96];
      v22 = v172;
      v48 = v47;
      goto LABEL_57;
    }
LABEL_241:
    sub_4261EA(v16, v15, v22);
  }
  v48 = 0;
LABEL_57:
  v207 = v48;
  v208 = v48;
  v209 = &v22->m128i_i8[v48];
  if ( v15 != v16 )
  {
    v49 = (__m128i *)v48;
    v50 = v16;
    do
    {
      if ( v49 )
      {
        *v49 = _mm_loadu_si128(v50);
        v49[1] = _mm_loadu_si128(v50 + 1);
        v45 = v50[2].m128i_i64[0];
        v49[2].m128i_i64[0] = v45;
      }
      v50 = (const __m128i *)((char *)v50 + 40);
      v49 = (__m128i *)((char *)v49 + 40);
    }
    while ( v50 != v15 );
    v48 += 8 * ((unsigned __int64)((char *)&v50[-3].m128i_u64[1] - (char *)v16) >> 3) + 40;
  }
  v208 = v48;
  sub_C8CD80((__int64)&v210, (__int64)v214, (__int64)v240, v48, v45, v46);
  v15 = v242;
  v16 = v241;
  v215 = 0;
  v216 = 0;
  v217 = 0;
  v52 = (char *)v242 - (char *)v241;
  if ( v242 == v241 )
  {
    v54 = 0;
  }
  else
  {
    if ( v52 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v179 = (char *)v242 - (char *)v241;
    v53 = sub_22077B0((char *)v242 - (char *)v241);
    v15 = v242;
    v16 = v241;
    v52 = v179;
    v54 = (__m128i *)v53;
  }
  v215 = v54;
  v216 = (unsigned __int64)v54;
  v217 = &v54->m128i_i8[v52];
  if ( v15 == v16 )
  {
    v57 = (unsigned __int64)v54;
  }
  else
  {
    v55 = v54;
    v56 = v16;
    do
    {
      if ( v55 )
      {
        *v55 = _mm_loadu_si128(v56);
        v55[1] = _mm_loadu_si128(v56 + 1);
        v55[2].m128i_i64[0] = v56[2].m128i_i64[0];
      }
      v56 = (const __m128i *)((char *)v56 + 40);
      v55 = (__m128i *)((char *)v55 + 40);
    }
    while ( v15 != v56 );
    v57 = (unsigned __int64)&v54[2].m128i_u64[((unsigned __int64)((char *)&v15[-3].m128i_u64[1] - (char *)v16) >> 3) + 1];
  }
  v58 = &v202;
  v59 = a1;
  v216 = v57;
  while ( 2 )
  {
    v60 = v207;
    if ( v208 - v207 != v57 - (_QWORD)v54 )
    {
LABEL_75:
      v61 = **(_QWORD **)(v208 - 40);
      if ( (v61 & 4) == 0 )
      {
        v62 = v61 & 0xFFFFFFFFFFFFFFF8LL;
        v63 = *(_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 56);
        v64 = v62 + 48;
        if ( v64 != v63 )
        {
          v164 = v58;
          v65 = v59;
          v66 = v63;
          v163 = v10;
          v67 = v64;
          do
          {
            if ( !v66 )
              goto LABEL_258;
            if ( *(_BYTE *)(v66 - 24) == 86 )
            {
              v68 = (unsigned int)v190;
              v69 = (unsigned int)v190 + 1LL;
              if ( v69 > HIDWORD(v190) )
              {
                sub_C8D5F0((__int64)&v189, v191, v69, 8u, v64, v51);
                v68 = (unsigned int)v190;
              }
              *(_QWORD *)&v189[8 * v68] = v66 - 24;
              LODWORD(v190) = v190 + 1;
              ++v65[6];
            }
            v66 = *(_QWORD *)(v66 + 8);
          }
          while ( v67 != v66 );
          v59 = v65;
          v10 = v163;
          v58 = v164;
        }
      }
      sub_22DE410((__int64)v58);
      v54 = v215;
      v57 = v216;
      continue;
    }
    break;
  }
  if ( v207 != v208 )
  {
    while ( *(_QWORD *)v60 == v54->m128i_i64[0] )
    {
      v70 = *(_BYTE *)(v60 + 32);
      if ( v70 != v54[2].m128i_i8[0] )
        break;
      if ( v70 )
      {
        if ( !(((*(__int64 *)(v60 + 8) >> 1) & 3) != 0
             ? ((v54->m128i_i64[1] >> 1) & 3) == ((*(__int64 *)(v60 + 8) >> 1) & 3)
             : *(_DWORD *)(v60 + 24) == v54[1].m128i_i32[2]) )
          break;
      }
      v60 += 40;
      v54 = (__m128i *)((char *)v54 + 40);
      if ( v208 == v60 )
        goto LABEL_94;
    }
    goto LABEL_75;
  }
LABEL_94:
  sub_23FD500((__int64)&v210);
  sub_23FD500((__int64)&v202);
  sub_23FD500((__int64)v240);
  sub_23FD500((__int64)v10);
  if ( (_DWORD)v190 )
  {
    v210 = (unsigned __int64 *)&v189;
    v211 = a1;
    if ( v188 )
    {
      sub_2404660((__int64)&v210, *v188);
    }
    else
    {
      v218 = a2;
      LOBYTE(v219) = 0;
      v221 = &v222.m128i_u64[1];
      v222.m128i_i64[0] = 0x800000000LL;
      sub_2404660((__int64)&v210, (__int64)&v218);
      *(_QWORD *)&v239[16] = &v239[32];
      *(_QWORD *)&v239[24] = 0x800000000LL;
      *(_QWORD *)v239 = v218;
      v239[8] = v219;
      if ( v222.m128i_i32[0] )
        sub_23FAD70((__int64)&v239[16], (__int64)&v221, v123, v124, v125, v126);
      v127 = sub_22077B0(0x718u);
      v132 = (__int64 *)v127;
      if ( v127 )
        sub_23FB0E0(v127, (__int64)v10, v128, v129, v130, v131);
      v188 = v132;
      if ( *(_BYTE **)&v239[16] != &v239[32] )
        _libc_free(*(unsigned __int64 *)&v239[16]);
      sub_23FBD30((__int64)v10, (__int64)(a1 + 33), (__int64 *)&v188);
      if ( v221 != &v222.m128i_u64[1] )
        _libc_free((unsigned __int64)v221);
    }
  }
  if ( v189 != v191 )
    _libc_free((unsigned __int64)v189);
  result = v188;
  if ( v188 )
  {
    v72 = *v188;
    v184 = **(_QWORD **)*v188 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)(*v188 + 8) )
    {
      v180 = sub_986580(**(_QWORD **)*v188 & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_102;
    }
    if ( *(_DWORD *)(v72 + 24) )
    {
      v180 = 0;
LABEL_102:
      v73 = sub_23FB710(v72);
      v202 = 0;
      v203 = 0;
      v204 = 0;
      v205 = 0;
      v74 = *(void ***)(v72 + 16);
      v178 = v73;
      v173 = &v74[*(unsigned int *)(v72 + 24)];
      if ( v74 == v173 )
        goto LABEL_133;
      v75 = 0;
      v76 = 0;
      while ( 1 )
      {
        v80 = *v74;
        v218 = *v74;
        if ( v75 )
        {
          v77 = (v75 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
          v78 = (_QWORD *)(v76 + 8LL * v77);
          v79 = (void *)*v78;
          if ( v80 == (void *)*v78 )
            goto LABEL_105;
          v121 = 1;
          v122 = 0;
          while ( v79 != (void *)-4096LL )
          {
            if ( v79 == (void *)-8192LL && !v122 )
              v122 = v78;
            v77 = (v75 - 1) & (v121 + v77);
            v78 = (_QWORD *)(v76 + 8LL * v77);
            v79 = (void *)*v78;
            if ( v80 == (void *)*v78 )
              goto LABEL_105;
            ++v121;
          }
          if ( v122 )
            v78 = v122;
          ++v202;
          v81 = v204 + 1;
          *(_QWORD *)v239 = v78;
          if ( 4 * ((int)v204 + 1) < 3 * v75 )
          {
            if ( v75 - (v81 + HIDWORD(v204)) > v75 >> 3 )
              goto LABEL_111;
            goto LABEL_110;
          }
        }
        else
        {
          ++v202;
          *(_QWORD *)v239 = 0;
        }
        v75 *= 2;
LABEL_110:
        sub_CF4090((__int64)&v202, v75);
        sub_23FDF60((__int64)&v202, (__int64 *)&v218, v10);
        v80 = v218;
        v81 = v204 + 1;
        v78 = *(_QWORD **)v239;
LABEL_111:
        LODWORD(v204) = v81;
        if ( *v78 != -4096 )
          --HIDWORD(v204);
        *v78 = v80;
LABEL_105:
        if ( v173 == ++v74 )
        {
          v91 = *(unsigned int *)(v72 + 24);
          v92 = *(_QWORD *)(v72 + 16);
          if ( *(_DWORD *)(v72 + 24) )
          {
            v93 = *(__int64 **)(v72 + 16);
            do
            {
              v95 = *v93;
              v96 = v93 + 1;
              if ( v178 == *v93 )
              {
                ++v93;
                continue;
              }
              v210 = 0;
              v211 = 0;
              v212 = 0;
              v94 = a1[2];
              LODWORD(v213) = 0;
              if ( !(unsigned __int8)sub_24005F0(
                                       *(unsigned __int8 **)(v95 - 96),
                                       v178,
                                       v94,
                                       (__int64)&v202,
                                       0,
                                       (__int64)&v210) )
              {
                v166 = (__int64 *)a1[5];
                v165 = *v166;
                v97 = sub_B2BE50(*v166);
                if ( sub_B6EA50(v97)
                  || (v106 = sub_B2BE50(v165),
                      v107 = sub_B6F970(v106),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v107 + 48LL))(v107)) )
                {
                  sub_B176B0((__int64)v10, (__int64)"chr", (__int64)"DropUnhoistableSelect", 21, v95);
                  sub_B18290((__int64)v10, "Dropped unhoistable select", 0x1Au);
                  v98 = _mm_loadu_si128((const __m128i *)&v239[24]);
                  v99 = _mm_loadu_si128((const __m128i *)&v239[48]);
                  v100 = _mm_loadu_si128((const __m128i *)&v239[64]);
                  v227 = 0x400000000LL;
                  v219 = *(_DWORD *)&v239[8];
                  v222 = v98;
                  v220 = v239[12];
                  v224 = v99;
                  v221 = *(unsigned __int64 **)&v239[16];
                  v225 = v100;
                  v218 = &unk_49D9D40;
                  v223 = *(_QWORD *)&v239[40];
                  v226 = &v228;
                  if ( *(_DWORD *)&v239[88] )
                    sub_23FE010(
                      (__int64)&v226,
                      (__int64)&v239[80],
                      (__int64)&v228,
                      *(unsigned int *)&v239[88],
                      (__int64)&v226,
                      (__int64)&v239[80]);
                  *(_QWORD *)v239 = &unk_49D9D40;
                  v236 = v243;
                  v237 = v244;
                  v238 = v245;
                  v218 = &unk_49D9DB0;
                  sub_23FD590((__int64)&v239[80]);
                  sub_1049740(v166, (__int64)&v218);
                  v218 = &unk_49D9D40;
                  sub_23FD590((__int64)&v226);
                }
                v101 = *(_QWORD *)(v72 + 16) + 8LL * *(unsigned int *)(v72 + 24);
                v102 = *(_DWORD *)(v72 + 24);
                if ( (__int64 *)v101 != v96 )
                {
                  memmove(v93, v93 + 1, v101 - (_QWORD)v96);
                  v102 = *(_DWORD *)(v72 + 24);
                }
                *(_DWORD *)(v72 + 24) = v102 - 1;
                if ( !(_DWORD)v205 )
                  goto LABEL_156;
                v103 = (v205 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
                v104 = &v203[v103];
                v105 = *v104;
                if ( *v104 != v95 )
                {
                  v133 = 1;
                  while ( v105 != -4096 )
                  {
                    v134 = v133 + 1;
                    v103 = (v205 - 1) & (v133 + v103);
                    v104 = &v203[v103];
                    v105 = *v104;
                    if ( v95 == *v104 )
                      goto LABEL_153;
                    v133 = v134;
                  }
LABEL_156:
                  v96 = v93;
                  goto LABEL_141;
                }
LABEL_153:
                *v104 = -8192;
                v96 = v93;
                LODWORD(v204) = v204 - 1;
                ++HIDWORD(v204);
              }
LABEL_141:
              v93 = v96;
              sub_C7D6A0((__int64)v211, 16LL * (unsigned int)v213, 8);
              v92 = *(_QWORD *)(v72 + 16);
              v91 = *(unsigned int *)(v72 + 24);
            }
            while ( v96 != (__int64 *)(v92 + 8 * v91) );
          }
LABEL_133:
          v90 = sub_23FB710(v72);
          if ( !*(_BYTE *)(v72 + 8) || v180 == v90 )
          {
LABEL_135:
            sub_C7D6A0((__int64)v203, 8LL * (unsigned int)v205, 8);
            return v188;
          }
          v210 = 0;
          v211 = 0;
          v212 = 0;
          v108 = a1[2];
          LODWORD(v213) = 0;
          if ( (unsigned __int8)sub_24005F0(
                                  *(unsigned __int8 **)(v180 - 96),
                                  v90,
                                  v108,
                                  (__int64)&v202,
                                  0,
                                  (__int64)&v210) )
            goto LABEL_175;
          v109 = *(__int64 **)(v72 + 16);
          v110 = *(unsigned int *)(v72 + 24);
          v111 = v109;
          v181 = &v109[v110];
          if ( v109 != v181 )
          {
            v174 = v72;
            do
            {
              v116 = *v111;
              v117 = (__int64 *)a1[5];
              v170 = *v117;
              v118 = sub_B2BE50(*v117);
              if ( sub_B6EA50(v118)
                || (v119 = sub_B2BE50(v170),
                    v120 = sub_B6F970(v119),
                    (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v120 + 48LL))(v120)) )
              {
                sub_B176B0((__int64)v10, (__int64)"chr", (__int64)"DropSelectUnhoistableBranch", 27, v116);
                sub_B18290((__int64)v10, "Dropped select due to unhoistable branch", 0x28u);
                sub_23FE290((__int64)&v218, (__int64)v10, v112, v113, v114, v115);
                v238 = v245;
                v218 = &unk_49D9DB0;
                *(_QWORD *)v239 = &unk_49D9D40;
                sub_23FD590((__int64)&v239[80]);
                sub_1049740(v117, (__int64)&v218);
                v218 = &unk_49D9D40;
                sub_23FD590((__int64)&v226);
              }
              ++v111;
            }
            while ( v181 != v111 );
            v72 = v174;
            v109 = *(__int64 **)(v174 + 16);
            v110 = *(unsigned int *)(v174 + 24);
          }
          v135 = v109;
          v136 = &v109[v110];
          if ( (8 * v110) >> 5 )
          {
            while ( v184 != *(_QWORD *)(*v135 + 40) )
            {
              if ( v184 == *(_QWORD *)(v135[1] + 40) )
              {
                ++v135;
                break;
              }
              if ( v184 == *(_QWORD *)(v135[2] + 40) )
              {
                v135 += 2;
                break;
              }
              if ( v184 == *(_QWORD *)(v135[3] + 40) )
              {
                v135 += 3;
                break;
              }
              v135 += 4;
              if ( &v109[4 * ((8 * v110) >> 5)] == v135 )
                goto LABEL_218;
            }
LABEL_204:
            if ( v136 != v135 )
            {
              v137 = v135 + 1;
              if ( v136 != v135 + 1 )
              {
                do
                {
                  if ( v184 != *(_QWORD *)(*v137 + 40) )
                    *v135++ = *v137;
                  ++v137;
                }
                while ( v136 != v137 );
                v109 = *(__int64 **)(v72 + 16);
                v110 = *(unsigned int *)(v72 + 24);
              }
            }
LABEL_210:
            v138 = &v109[v110];
            v139 = (char *)v138 - (char *)v136;
            if ( v136 != v138 )
              v135 = (__int64 *)memmove(v135, v136, (char *)v138 - (char *)v136);
            *(_DWORD *)(v72 + 24) = ((__int64)v135 + v139 - *(_QWORD *)(v72 + 16)) >> 3;
            ++v202;
            if ( !(_DWORD)v204 )
            {
              if ( HIDWORD(v204) )
              {
                v140 = (unsigned int)v205;
                if ( (unsigned int)v205 <= 0x40 )
                  goto LABEL_215;
                sub_C7D6A0((__int64)v203, 8LL * (unsigned int)v205, 8);
                LODWORD(v205) = 0;
LABEL_173:
                v203 = 0;
LABEL_174:
                v204 = 0;
              }
LABEL_175:
              sub_C7D6A0((__int64)v211, 16LL * (unsigned int)v213, 8);
              goto LABEL_135;
            }
            v144 = 4 * v204;
            v140 = (unsigned int)v205;
            if ( (unsigned int)(4 * v204) < 0x40 )
              v144 = 64;
            if ( v144 >= (unsigned int)v205 )
            {
LABEL_215:
              v141 = v203;
              v142 = &v203[v140];
              if ( v203 != v142 )
              {
                do
                  *v141++ = -4096;
                while ( v142 != v141 );
              }
              goto LABEL_174;
            }
            if ( (_DWORD)v204 != 1 )
            {
              _BitScanReverse(&v145, v204 - 1);
              v146 = 1 << (33 - (v145 ^ 0x1F));
              if ( v146 < 64 )
                v146 = 64;
              if ( v146 != (_DWORD)v205 )
              {
LABEL_231:
                sub_C7D6A0((__int64)v203, 8LL * (unsigned int)v205, 8);
                v147 = sub_AF1560(4 * v146 / 3u + 1);
                LODWORD(v205) = v147;
                if ( !v147 )
                  goto LABEL_173;
                v203 = (_QWORD *)sub_C7D670(8LL * v147, 8);
              }
              sub_23FE770((__int64)&v202);
              goto LABEL_175;
            }
            v146 = 64;
            goto LABEL_231;
          }
LABEL_218:
          v143 = (char *)v136 - (char *)v135;
          if ( (char *)v136 - (char *)v135 != 16 )
          {
            if ( v143 != 24 )
            {
              if ( v143 != 8 )
              {
LABEL_221:
                v135 = &v109[v110];
                goto LABEL_210;
              }
LABEL_239:
              if ( v184 == *(_QWORD *)(*v135 + 40) )
                goto LABEL_204;
              goto LABEL_221;
            }
            if ( v184 == *(_QWORD *)(*v135 + 40) )
              goto LABEL_204;
            ++v135;
          }
          if ( v184 == *(_QWORD *)(*v135 + 40) )
            goto LABEL_204;
          ++v135;
          goto LABEL_239;
        }
        v76 = (__int64)v203;
        v75 = v205;
      }
    }
  }
  return result;
}
