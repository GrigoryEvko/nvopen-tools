// Function: sub_275FCE0
// Address: 0x275fce0
//
__int64 __fastcall sub_275FCE0(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4)
{
  unsigned __int8 *v4; // r13
  __int64 v6; // r8
  __int64 v8; // r12
  __int64 v9; // r9
  __m128i v10; // xmm5
  __m128i v11; // xmm6
  __m128i v12; // xmm7
  __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  __m128i *v16; // rax
  unsigned __int64 v17; // r8
  __m128i *v18; // rdx
  __int64 v19; // r9
  char v20; // al
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  __m128i v23; // xmm7
  __m128i *v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  __m128i *v28; // rax
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r12
  int v35; // r12d
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  char v42; // r15
  __int64 v43; // rax
  __int64 v44; // rdi
  char v45; // al
  __int64 v46; // r9
  __int64 v47; // rdx
  unsigned __int64 v48; // rcx
  unsigned __int64 v49; // rsi
  int v50; // eax
  unsigned __int64 v51; // rsi
  __m128i *v52; // rcx
  unsigned __int64 v53; // rdx
  __int64 v54; // r12
  __m128i *v55; // r15
  unsigned __int64 v56; // rdi
  __int64 v57; // rdx
  unsigned __int64 v58; // rcx
  __m128i *v59; // r15
  unsigned __int64 v60; // rsi
  int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rbx
  __m128i *v64; // r12
  unsigned __int64 v65; // rdi
  _BYTE *v66; // rbx
  unsigned __int64 v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // r15
  char v71; // dl
  __int64 v72; // r9
  __int64 v73; // rbx
  __int64 v74; // r14
  __int64 v75; // rax
  __int32 v76; // ebx
  __int64 *v77; // rax
  unsigned __int64 v78; // r14
  __int64 v79; // rax
  unsigned __int64 v80; // r13
  unsigned __int64 v81; // r12
  __int64 v82; // rbx
  unsigned __int64 v83; // r15
  unsigned __int64 v84; // rbx
  unsigned __int64 v85; // rdi
  const __m128i *v86; // rbx
  __m128i *v87; // r14
  __int64 v88; // r12
  __int64 v89; // rax
  unsigned __int64 v90; // rcx
  unsigned __int64 v91; // r9
  __m128i *v92; // rdx
  unsigned __int64 v93; // rsi
  __m128i v94; // xmm7
  __m128i *v95; // rax
  unsigned __int64 v96; // r15
  unsigned __int8 *v97; // r12
  __int64 v98; // rbx
  __m128i *v99; // r13
  unsigned __int64 v100; // rdi
  __int64 v101; // rbx
  __int64 v102; // r14
  __m128i *v103; // r15
  unsigned __int64 v104; // rdi
  __m128i *v105; // r13
  __int64 v106; // rdx
  __int64 v107; // rax
  unsigned int *v108; // rbx
  __m128i *v109; // r14
  __m128i *v110; // r13
  __int64 v111; // r12
  __m128i *v112; // rax
  __m128i *v113; // r14
  __int64 *v114; // r12
  __int64 v115; // rdx
  __int64 v116; // rax
  unsigned int v117; // esi
  unsigned __int64 v118; // rdx
  __int64 *v119; // rax
  unsigned int v120; // edi
  unsigned __int64 v121; // rax
  int v122; // ecx
  __int64 v123; // rdx
  __int64 v124; // rax
  __m128i v125; // xmm1
  unsigned __int64 v126; // rdx
  __int64 v127; // rdx
  const __m128i *v128; // rcx
  __m128i *v129; // rax
  __m128i *v130; // rbx
  unsigned __int64 v131; // rdi
  __m128i *v132; // rbx
  unsigned __int64 v133; // rdi
  __int8 *v134; // r15
  __int64 v135; // rax
  unsigned __int64 v136; // rbx
  __int64 v137; // rdx
  __int64 v138; // r15
  unsigned __int64 v139; // r14
  unsigned __int32 v140; // eax
  __m128i *v141; // rdx
  __int64 v142; // r13
  unsigned int v143; // eax
  unsigned __int64 v144; // rsi
  __int64 m128i_i64; // r13
  unsigned int v146; // eax
  unsigned int v147; // eax
  __int64 v148; // rbx
  __m128i *v149; // r13
  unsigned __int64 v150; // rdi
  __int32 v151; // esi
  unsigned __int64 v152; // r9
  __int64 v153; // rcx
  unsigned int v154; // eax
  __m128i *v155; // r13
  __int64 v156; // rax
  __int8 *v157; // r13
  __int64 v158; // rax
  __int64 v159; // r12
  __int64 v160; // rdx
  __int64 v161; // rbx
  __int64 v162; // r14
  unsigned __int64 v163; // r15
  unsigned __int64 v164; // r14
  unsigned __int64 v165; // rdi
  int v166; // edi
  __int64 v167; // rax
  unsigned int v168; // edx
  __int64 v169; // rax
  __int8 v170; // dl
  __int64 v171; // rbx
  unsigned __int64 v172; // r12
  unsigned __int64 v173; // rdi
  unsigned __int64 v174; // rbx
  __int8 *v175; // r13
  unsigned int v176; // esi
  __int64 v177; // rbx
  char v178; // al
  __m128i *v179; // rdi
  size_t v180; // rdx
  unsigned __int32 v181; // eax
  unsigned __int32 v182; // ecx
  unsigned int v183; // edx
  __int64 v184; // rdx
  int v185; // eax
  __int64 *v186; // rdx
  __int64 v187; // rbx
  __m128i *v188; // r12
  __int64 v189; // r14
  unsigned __int64 v190; // r15
  unsigned __int64 v191; // r14
  unsigned __int64 v192; // rdi
  __int8 *v193; // r12
  int v194; // edi
  __int8 *v195; // rbx
  __int64 v196; // rax
  __int64 v197; // rcx
  __m128i *v198; // r14
  __int64 v199; // rbx
  __int64 v200; // rax
  __m128i *v201; // r12
  __int64 v202; // rdx
  __m128i *v203; // r15
  __int64 v204; // rax
  __m128i *v205; // rax
  __m128i *v206; // r15
  __int64 v207; // rax
  unsigned __int64 v208; // r12
  unsigned __int64 v209; // rbx
  __int64 v210; // r13
  unsigned __int64 v211; // r14
  unsigned __int64 v212; // r13
  unsigned __int64 v213; // rdi
  int v214; // ebx
  __int32 v215; // ecx
  __m128i *v216; // rdi
  unsigned int v217; // edx
  __int64 v218; // rsi
  int v219; // eax
  __int32 v220; // esi
  __m128i *v221; // rdi
  unsigned int v222; // edx
  __int64 v223; // rcx
  int v224; // eax
  __m128i *v225; // [rsp+8h] [rbp-548h]
  __int64 v226; // [rsp+18h] [rbp-538h]
  const void *v227; // [rsp+20h] [rbp-530h]
  __int64 v229; // [rsp+50h] [rbp-500h]
  __int64 i; // [rsp+58h] [rbp-4F8h]
  _QWORD *v231; // [rsp+58h] [rbp-4F8h]
  __int8 v232; // [rsp+60h] [rbp-4F0h]
  __m128i *v233; // [rsp+60h] [rbp-4F0h]
  __m128i *v234; // [rsp+60h] [rbp-4F0h]
  __m128i *v236; // [rsp+78h] [rbp-4D8h]
  __int64 v237; // [rsp+80h] [rbp-4D0h]
  unsigned int *v238; // [rsp+80h] [rbp-4D0h]
  __m128i *v239; // [rsp+80h] [rbp-4D0h]
  __m128i *v240; // [rsp+88h] [rbp-4C8h]
  unsigned __int64 v241; // [rsp+88h] [rbp-4C8h]
  unsigned __int8 *v242; // [rsp+90h] [rbp-4C0h]
  char v243; // [rsp+90h] [rbp-4C0h]
  __m128i *v245; // [rsp+98h] [rbp-4B8h]
  __int64 v246; // [rsp+98h] [rbp-4B8h]
  unsigned __int64 v247; // [rsp+98h] [rbp-4B8h]
  __int64 v248; // [rsp+98h] [rbp-4B8h]
  __int64 v249; // [rsp+A0h] [rbp-4B0h] BYREF
  unsigned __int64 v250; // [rsp+A8h] [rbp-4A8h] BYREF
  _BYTE *v251; // [rsp+B0h] [rbp-4A0h] BYREF
  __int64 v252; // [rsp+B8h] [rbp-498h]
  _BYTE v253[64]; // [rsp+C0h] [rbp-490h] BYREF
  __m128i v254; // [rsp+100h] [rbp-450h] BYREF
  __m128i v255; // [rsp+110h] [rbp-440h] BYREF
  __m128i src; // [rsp+160h] [rbp-3F0h] BYREF
  __int64 v257; // [rsp+170h] [rbp-3E0h] BYREF
  _BYTE v258[72]; // [rsp+178h] [rbp-3D8h] BYREF
  __m128i *v259; // [rsp+1C0h] [rbp-390h] BYREF
  __int64 v260; // [rsp+1C8h] [rbp-388h] BYREF
  __m128i v261; // [rsp+1D0h] [rbp-380h] BYREF
  __m128i v262; // [rsp+1E0h] [rbp-370h]
  __m128i v263; // [rsp+280h] [rbp-2D0h] BYREF
  __m128i v264; // [rsp+290h] [rbp-2C0h] BYREF
  __m128i v265; // [rsp+2A0h] [rbp-2B0h] BYREF
  char v266; // [rsp+2B0h] [rbp-2A0h]
  __m128i v267; // [rsp+350h] [rbp-200h] BYREF
  __m128i v268; // [rsp+360h] [rbp-1F0h] BYREF
  __m128i v269; // [rsp+370h] [rbp-1E0h] BYREF
  __m128i *v270; // [rsp+380h] [rbp-1D0h] BYREF
  __int64 v271; // [rsp+388h] [rbp-1C8h]
  _BYTE v272[448]; // [rsp+390h] [rbp-1C0h] BYREF

  v4 = a3;
  *(_QWORD *)a1 = a1 + 16;
  v227 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  if ( (unsigned __int8)(*a3 - 34) > 0x33u || (v8 = 0x8000000000041LL, !_bittest64(&v8, (unsigned int)*a3 - 34)) )
  {
    if ( !(unsigned __int8)sub_B46490((__int64)a3) )
      goto LABEL_3;
    goto LABEL_13;
  }
  if ( (unsigned int)sub_B49240((__int64)a3) == 210 || sub_D5D560((__int64)v4, *(__int64 **)(a2 + 808)) )
  {
    sub_2753860((__int64)&v267, a2, v4);
    if ( (_BYTE)v271 )
    {
      v10 = _mm_loadu_si128(&v267);
      v266 = 0;
      v11 = _mm_loadu_si128(&v268);
      v12 = _mm_loadu_si128(&v269);
      v13 = *(unsigned int *)(a1 + 8);
      v14 = *(unsigned int *)(a1 + 12);
      v263 = v10;
      v15 = *(_QWORD *)a1;
      v264 = v11;
      v16 = &v263;
      v17 = v13 + 1;
      v265 = v12;
      if ( v13 + 1 > v14 )
      {
        if ( v15 > (unsigned __int64)&v263 || (unsigned __int64)&v263 >= v15 + 56 * v13 )
        {
          sub_C8D5F0(a1, v227, v17, 0x38u, v17, v9);
          v15 = *(_QWORD *)a1;
          v13 = *(unsigned int *)(a1 + 8);
          v16 = &v263;
        }
        else
        {
          v195 = &v263.m128i_i8[-v15];
          sub_C8D5F0(a1, v227, v17, 0x38u, v17, v9);
          v15 = *(_QWORD *)a1;
          v13 = *(unsigned int *)(a1 + 8);
          v16 = (__m128i *)&v195[*(_QWORD *)a1];
        }
      }
      v18 = (__m128i *)(v15 + 56 * v13);
      *v18 = _mm_loadu_si128(v16);
      v18[1] = _mm_loadu_si128(v16 + 1);
      v18[2] = _mm_loadu_si128(v16 + 2);
      v18[3].m128i_i64[0] = v16[3].m128i_i64[0];
      ++*(_DWORD *)(a1 + 8);
    }
    return a1;
  }
  if ( !(unsigned __int8)sub_B46490((__int64)v4) )
    goto LABEL_3;
  if ( (unsigned __int8)(*v4 - 34) > 0x33u || !_bittest64(&v8, (unsigned int)*v4 - 34) )
  {
LABEL_13:
    sub_D66840(&v263, v4);
    v20 = v266;
    goto LABEL_14;
  }
  sub_D67230(&v263, v4, *(__int64 **)(a2 + 808));
  v20 = v266;
LABEL_14:
  if ( !v20 )
  {
LABEL_3:
    if ( !a4 )
      return a1;
    goto LABEL_17;
  }
  v21 = _mm_loadu_si128(&v263);
  LOBYTE(v270) = 0;
  v22 = _mm_loadu_si128(&v264);
  v23 = _mm_loadu_si128(&v265);
  v24 = &v267;
  v25 = *(unsigned int *)(a1 + 8);
  v26 = *(unsigned int *)(a1 + 12);
  v267 = v21;
  v27 = *(_QWORD *)a1;
  v268 = v22;
  v6 = v25 + 1;
  v269 = v23;
  if ( v25 + 1 > v26 )
  {
    if ( v27 > (unsigned __int64)&v267 || (unsigned __int64)&v267 >= v27 + 56 * v25 )
    {
      v24 = &v267;
      sub_C8D5F0(a1, v227, v6, 0x38u, v6, v19);
      v27 = *(_QWORD *)a1;
      v25 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v193 = &v267.m128i_i8[-v27];
      sub_C8D5F0(a1, v227, v6, 0x38u, v6, v19);
      v27 = *(_QWORD *)a1;
      v25 = *(unsigned int *)(a1 + 8);
      v24 = (__m128i *)&v193[*(_QWORD *)a1];
    }
  }
  v28 = (__m128i *)(v27 + 56 * v25);
  *v28 = _mm_loadu_si128(v24);
  v28[1] = _mm_loadu_si128(v24 + 1);
  v28[2] = _mm_loadu_si128(v24 + 2);
  v28[3].m128i_i64[0] = v24[3].m128i_i64[0];
  ++*(_DWORD *)(a1 + 8);
  if ( !a4 )
    return a1;
LABEL_17:
  v29 = *v4;
  if ( (unsigned __int8)(v29 - 34) > 0x33u )
    return a1;
  v30 = 0x8000000000041LL;
  if ( !_bittest64(&v30, (unsigned int)(v29 - 34)) )
    return a1;
  v267.m128i_i64[0] = 0;
  v270 = (__m128i *)v272;
  v267.m128i_i64[1] = 1;
  v268.m128i_i64[0] = -4096;
  v269.m128i_i64[0] = -4096;
  v271 = 0x200000000LL;
  if ( v29 == 40 )
  {
    v31 = 32LL * (unsigned int)sub_B491D0((__int64)v4);
  }
  else
  {
    v31 = 0;
    if ( v29 != 85 )
    {
      v31 = 64;
      if ( v29 != 34 )
        BUG();
    }
  }
  if ( (v4[7] & 0x80u) != 0 )
  {
    v32 = sub_BD2BC0((__int64)v4);
    v34 = v32 + v33;
    if ( (v4[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v34 >> 4) )
        goto LABEL_441;
    }
    else if ( (unsigned int)((v34 - sub_BD2BC0((__int64)v4)) >> 4) )
    {
      if ( (v4[7] & 0x80u) != 0 )
      {
        v35 = *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8);
        if ( (v4[7] & 0x80u) == 0 )
          BUG();
        v36 = sub_BD2BC0((__int64)v4);
        v38 = 32LL * (unsigned int)(*(_DWORD *)(v36 + v37 - 4) - v35);
        goto LABEL_32;
      }
LABEL_441:
      BUG();
    }
  }
  v38 = 0;
LABEL_32:
  v39 = (32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF) - 32 - v31 - v38) >> 5;
  if ( !(_DWORD)v39 )
    goto LABEL_97;
  v229 = (__int64)v4;
  v226 = (unsigned int)v39;
  for ( i = 0; i != v226; ++i )
  {
    v251 = v253;
    v252 = 0x200000000LL;
    v263.m128i_i64[0] = *(_QWORD *)(v229 + 72);
    v40 = sub_A747F0(&v263, (int)i + 1, 98);
    if ( v40 )
    {
      v249 = v40;
    }
    else
    {
      v249 = sub_B49640(v229, i, 98);
      if ( !v249 )
        goto LABEL_37;
    }
    if ( !(unsigned __int8)sub_B49B80(v229, i, 81) )
    {
      v135 = sub_A72AC0(&v249);
      v263.m128i_i64[0] = (__int64)&v264;
      v136 = v135;
      v137 *= 32;
      v138 = v135 + v137;
      v263.m128i_i64[1] = 0x200000000LL;
      if ( v136 == v136 + v137 )
      {
LABEL_250:
        sub_27578C0((__int64)&v251, v263.m128i_i64);
        v148 = v263.m128i_i64[0];
        v149 = (__m128i *)(v263.m128i_i64[0] + 32LL * v263.m128i_u32[2]);
        if ( (__m128i *)v263.m128i_i64[0] != v149 )
        {
          do
          {
            v149 -= 2;
            if ( v149[1].m128i_i32[2] > 0x40u )
            {
              v150 = v149[1].m128i_u64[0];
              if ( v150 )
                j_j___libc_free_0_0(v150);
            }
            if ( v149->m128i_i32[2] > 0x40u && v149->m128i_i64[0] )
              j_j___libc_free_0_0(v149->m128i_i64[0]);
          }
          while ( (__m128i *)v148 != v149 );
          v149 = (__m128i *)v263.m128i_i64[0];
        }
        if ( v149 != &v264 )
          _libc_free((unsigned __int64)v149);
        goto LABEL_37;
      }
      v139 = v135 + 32;
      v140 = 0;
      v141 = &v264;
      v142 = 0;
      while ( 1 )
      {
        m128i_i64 = (__int64)v141[2 * v142].m128i_i64;
        if ( m128i_i64 )
        {
          v146 = *(_DWORD *)(v136 + 8);
          *(_DWORD *)(m128i_i64 + 8) = v146;
          if ( v146 <= 0x40 )
          {
            *(_QWORD *)m128i_i64 = *(_QWORD *)v136;
            v143 = *(_DWORD *)(v136 + 24);
            *(_DWORD *)(m128i_i64 + 24) = v143;
            if ( v143 <= 0x40 )
              goto LABEL_242;
          }
          else
          {
            sub_C43780(m128i_i64, (const void **)v136);
            v147 = *(_DWORD *)(v136 + 24);
            *(_DWORD *)(m128i_i64 + 24) = v147;
            if ( v147 <= 0x40 )
            {
LABEL_242:
              *(_QWORD *)(m128i_i64 + 16) = *(_QWORD *)(v136 + 16);
              v140 = v263.m128i_u32[2];
              goto LABEL_243;
            }
          }
          sub_C43780(m128i_i64 + 16, (const void **)(v136 + 16));
          v140 = v263.m128i_u32[2];
        }
LABEL_243:
        ++v140;
        v136 = v139;
        v263.m128i_i32[2] = v140;
        if ( v138 == v139 )
          goto LABEL_250;
        v142 = v140;
        v141 = (__m128i *)v263.m128i_i64[0];
        v144 = v140 + 1LL;
        if ( v144 > v263.m128i_u32[3] )
        {
          if ( v263.m128i_i64[0] > v139 || v263.m128i_i64[0] + 32 * (unsigned __int64)v140 <= v139 )
          {
            sub_9D5330((__int64)&v263, v144);
            v142 = v263.m128i_u32[2];
            v141 = (__m128i *)v263.m128i_i64[0];
            v140 = v263.m128i_u32[2];
          }
          else
          {
            v174 = v139 - v263.m128i_i64[0];
            sub_9D5330((__int64)&v263, v144);
            v141 = (__m128i *)v263.m128i_i64[0];
            v142 = v263.m128i_u32[2];
            v136 = v263.m128i_i64[0] + v174;
            v140 = v263.m128i_u32[2];
          }
        }
        v139 += 32LL;
      }
    }
LABEL_37:
    v242 = *(unsigned __int8 **)(v229 + 32 * (i - (*(_DWORD *)(v229 + 4) & 0x7FFFFFF)));
    if ( (_DWORD)v252 )
    {
      if ( !sub_B49EC0(v229) )
      {
        v97 = sub_98ACB0(v242, 6u);
        if ( !(unsigned __int8)sub_CF70D0(v97) || !(unsigned __int8)sub_D0C450(a2 + 16, v97, v229, 1) )
        {
          v263.m128i_i64[0] = (__int64)&v264;
          v263.m128i_i64[1] = 0x200000000LL;
          sub_27578C0((__int64)&v251, v263.m128i_i64);
          v98 = v263.m128i_i64[0];
          v99 = (__m128i *)(v263.m128i_i64[0] + 32LL * v263.m128i_u32[2]);
          if ( (__m128i *)v263.m128i_i64[0] != v99 )
          {
            do
            {
              v99 -= 2;
              if ( v99[1].m128i_i32[2] > 0x40u )
              {
                v100 = v99[1].m128i_u64[0];
                if ( v100 )
                  j_j___libc_free_0_0(v100);
              }
              if ( v99->m128i_i32[2] > 0x40u && v99->m128i_i64[0] )
                j_j___libc_free_0_0(v99->m128i_i64[0]);
            }
            while ( (__m128i *)v98 != v99 );
            v99 = (__m128i *)v263.m128i_i64[0];
          }
          if ( v99 != &v264 )
            _libc_free((unsigned __int64)v99);
        }
      }
    }
    v232 = sub_B49B80(v229, i, 9);
    if ( !v232 && *(_BYTE *)v229 == 85 )
    {
      v232 = sub_CF7590(v242, &src);
      if ( v232 )
      {
        if ( src.m128i_i8[0] )
        {
          v259 = (__m128i *)v242;
          LOBYTE(v260) = 1;
          sub_275ABE0((__int64)&v263, a2 + 1432, (__int64 *)&v259, &v260);
          v177 = v264.m128i_i64[0];
          if ( v265.m128i_i8[0] )
          {
            v178 = sub_D13FA0((__int64)v242, 0, 0);
            *(_BYTE *)(v177 + 8) = v178;
          }
          else
          {
            v178 = *(_BYTE *)(v264.m128i_i64[0] + 8);
          }
          v232 = v178 ^ 1;
        }
      }
    }
    v254.m128i_i32[0] = i;
    v254.m128i_i8[4] = v232;
    v254.m128i_i64[1] = (__int64)&v255.m128i_i64[1];
    v255.m128i_i64[0] = 0x200000000LL;
    if ( (_DWORD)v252 )
      sub_27583E0((__int64)&v254.m128i_i64[1], (__int64)&v251);
    v240 = (__m128i *)((char *)v270 + 200 * (unsigned int)v271);
    if ( v270 != v240 )
    {
      v41 = (__int64)&v270->m128i_i64[1];
      v42 = 0;
      while ( 1 )
      {
        v263.m128i_i64[1] = -1;
        v263.m128i_i64[0] = (__int64)v242;
        v264 = 0u;
        v265 = 0u;
        v43 = *(_QWORD *)(v41 - 8);
        v260 = -1;
        v259 = (__m128i *)v43;
        v261 = 0u;
        v44 = *(_QWORD *)(a2 + 104);
        v262 = 0u;
        v45 = sub_CF4D50(v44, (__int64)&v259, (__int64)&v263, a2 + 112, 0);
        if ( v45 )
        {
          if ( v45 == 3 )
          {
            v57 = *(unsigned int *)(v41 + 8);
            v58 = *(_QWORD *)v41;
            v59 = &v254;
            v60 = v57 + 1;
            v61 = *(_DWORD *)(v41 + 8);
            if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 12) )
            {
              if ( v58 > (unsigned __int64)&v254 || (unsigned __int64)&v254 >= v58 + 88 * v57 )
              {
                sub_2757D10(v41, v60, v57, v58, v6, v46);
                v57 = *(unsigned int *)(v41 + 8);
                v58 = *(_QWORD *)v41;
                v59 = &v254;
                v61 = *(_DWORD *)(v41 + 8);
              }
              else
              {
                v134 = &v254.m128i_i8[-v58];
                sub_2757D10(v41, v60, v57, v58, v6, v46);
                v58 = *(_QWORD *)v41;
                v57 = *(unsigned int *)(v41 + 8);
                v59 = (__m128i *)&v134[*(_QWORD *)v41];
                v61 = *(_DWORD *)(v41 + 8);
              }
            }
            v62 = v58 + 88 * v57;
            if ( v62 )
            {
              *(_DWORD *)v62 = v59->m128i_i32[0];
              *(_BYTE *)(v62 + 4) = v59->m128i_i8[4];
              *(_QWORD *)(v62 + 8) = v62 + 24;
              *(_QWORD *)(v62 + 16) = 0x200000000LL;
              if ( v59[1].m128i_i32[0] )
                sub_27583E0(v62 + 8, (__int64)&v59->m128i_i64[1]);
              v61 = *(_DWORD *)(v41 + 8);
            }
            v42 = a4;
            *(_DWORD *)(v41 + 8) = v61 + 1;
          }
          else
          {
            v263.m128i_i32[0] = i;
            v263.m128i_i8[4] = v232;
            v263.m128i_i64[1] = (__int64)&v264.m128i_i64[1];
            v264.m128i_i64[0] = 0x200000000LL;
            v47 = *(unsigned int *)(v41 + 8);
            v48 = *(unsigned int *)(v41 + 12);
            v49 = v47 + 1;
            v50 = *(_DWORD *)(v41 + 8);
            if ( v47 + 1 > v48 )
            {
              v96 = *(_QWORD *)v41;
              if ( *(_QWORD *)v41 > (unsigned __int64)&v263 || (unsigned __int64)&v263 >= v96 + 88 * v47 )
              {
                sub_2757D10(v41, v49, v47, v48, v6, v46);
                v47 = *(unsigned int *)(v41 + 8);
                v51 = *(_QWORD *)v41;
                v52 = &v263;
                v50 = *(_DWORD *)(v41 + 8);
              }
              else
              {
                sub_2757D10(v41, v49, v47, v48, v6, v46);
                v51 = *(_QWORD *)v41;
                v47 = *(unsigned int *)(v41 + 8);
                v52 = (__m128i *)((char *)&v263 + *(_QWORD *)v41 - v96);
                v50 = *(_DWORD *)(v41 + 8);
              }
            }
            else
            {
              v51 = *(_QWORD *)v41;
              v52 = &v263;
            }
            v53 = v51 + 88 * v47;
            if ( v53 )
            {
              *(_DWORD *)v53 = v52->m128i_i32[0];
              *(_BYTE *)(v53 + 4) = v52->m128i_i8[4];
              *(_QWORD *)(v53 + 8) = v53 + 24;
              *(_QWORD *)(v53 + 16) = 0x200000000LL;
              if ( v52[1].m128i_i32[0] )
                sub_27578C0(v53 + 8, &v52->m128i_i64[1]);
              v50 = *(_DWORD *)(v41 + 8);
            }
            *(_DWORD *)(v41 + 8) = v50 + 1;
            v54 = v263.m128i_i64[1];
            v55 = (__m128i *)(v263.m128i_i64[1] + 32LL * v264.m128i_u32[0]);
            if ( (__m128i *)v263.m128i_i64[1] != v55 )
            {
              do
              {
                v55 -= 2;
                if ( v55[1].m128i_i32[2] > 0x40u )
                {
                  v56 = v55[1].m128i_u64[0];
                  if ( v56 )
                    j_j___libc_free_0_0(v56);
                }
                if ( v55->m128i_i32[2] > 0x40u && v55->m128i_i64[0] )
                  j_j___libc_free_0_0(v55->m128i_i64[0]);
              }
              while ( (__m128i *)v54 != v55 );
              v55 = (__m128i *)v263.m128i_i64[1];
            }
            if ( v55 != (__m128i *)&v264.m128i_u64[1] )
              _libc_free((unsigned __int64)v55);
            v42 = a4;
          }
        }
        if ( v240 == (__m128i *)(v41 + 192) )
          break;
        v41 += 200;
      }
      if ( v42 )
        goto LABEL_73;
    }
    src.m128i_i32[0] = v254.m128i_i32[0];
    src.m128i_i8[4] = v254.m128i_i8[4];
    src.m128i_i64[1] = (__int64)v258;
    v257 = 0x200000000LL;
    if ( v255.m128i_i32[0] )
      sub_27583E0((__int64)&src.m128i_i64[1], (__int64)&v254.m128i_i64[1]);
    if ( (v267.m128i_i8[8] & 1) != 0 )
    {
      v151 = 1;
      v152 = (unsigned __int64)&v268;
    }
    else
    {
      v176 = v268.m128i_u32[2];
      v152 = v268.m128i_i64[0];
      if ( !v268.m128i_i32[2] )
      {
        v181 = v267.m128i_u32[2];
        ++v267.m128i_i64[0];
        v155 = 0;
        v182 = ((unsigned __int32)v267.m128i_i32[2] >> 1) + 1;
LABEL_323:
        v183 = 3 * v176;
        goto LABEL_324;
      }
      v151 = v268.m128i_i32[2] - 1;
    }
    v153 = (unsigned int)v242 >> 9;
    v154 = v151 & (v153 ^ ((unsigned int)v242 >> 4));
    v155 = (__m128i *)(v152 + 16LL * v154);
    v6 = v155->m128i_i64[0];
    if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
    {
LABEL_266:
      v156 = v155->m128i_u32[2];
      goto LABEL_267;
    }
    v194 = 1;
    v153 = 0;
    while ( v6 != -4096 )
    {
      if ( v6 == -8192 && !v153 )
        v153 = (__int64)v155;
      v154 = v151 & (v194 + v154);
      v155 = (__m128i *)(v152 + 16LL * v154);
      v6 = v155->m128i_i64[0];
      if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
        goto LABEL_266;
      ++v194;
    }
    v181 = v267.m128i_u32[2];
    if ( v153 )
      v155 = (__m128i *)v153;
    ++v267.m128i_i64[0];
    v182 = ((unsigned __int32)v267.m128i_i32[2] >> 1) + 1;
    if ( (v267.m128i_i8[8] & 1) == 0 )
    {
      v176 = v268.m128i_u32[2];
      goto LABEL_323;
    }
    v183 = 6;
    v176 = 2;
LABEL_324:
    if ( 4 * v182 >= v183 )
    {
      sub_275F8C0((__int64)&v267, 2 * v176);
      if ( (v267.m128i_i8[8] & 1) != 0 )
      {
        v220 = 1;
        v221 = &v268;
      }
      else
      {
        v221 = (__m128i *)v268.m128i_i64[0];
        if ( !v268.m128i_i32[2] )
          goto LABEL_439;
        v220 = v268.m128i_i32[2] - 1;
      }
      v181 = v267.m128i_u32[2];
      v222 = v220 & (((unsigned int)v242 >> 9) ^ ((unsigned int)v242 >> 4));
      v155 = &v221[v222];
      v223 = v155->m128i_i64[0];
      if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
        goto LABEL_326;
      v224 = 1;
      v6 = 0;
      while ( v223 != -4096 )
      {
        if ( !v6 && v223 == -8192 )
          v6 = (__int64)v155;
        v222 = v220 & (v224 + v222);
        v155 = &v221[v222];
        v223 = v155->m128i_i64[0];
        if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
          goto LABEL_409;
        ++v224;
      }
    }
    else
    {
      if ( v176 - v267.m128i_i32[3] - v182 > v176 >> 3 )
        goto LABEL_326;
      sub_275F8C0((__int64)&v267, v176);
      if ( (v267.m128i_i8[8] & 1) != 0 )
      {
        v215 = 1;
        v216 = &v268;
      }
      else
      {
        v216 = (__m128i *)v268.m128i_i64[0];
        if ( !v268.m128i_i32[2] )
        {
LABEL_439:
          v267.m128i_i32[2] = (2 * ((unsigned __int32)v267.m128i_i32[2] >> 1) + 2) | v267.m128i_i8[8] & 1;
          BUG();
        }
        v215 = v268.m128i_i32[2] - 1;
      }
      v181 = v267.m128i_u32[2];
      v217 = v215 & (((unsigned int)v242 >> 9) ^ ((unsigned int)v242 >> 4));
      v155 = &v216[v217];
      v218 = v155->m128i_i64[0];
      if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
        goto LABEL_326;
      v219 = 1;
      v6 = 0;
      while ( v218 != -4096 )
      {
        if ( !v6 && v218 == -8192 )
          v6 = (__int64)v155;
        v217 = v215 & (v219 + v217);
        v155 = &v216[v217];
        v218 = v155->m128i_i64[0];
        if ( v242 == (unsigned __int8 *)v155->m128i_i64[0] )
          goto LABEL_409;
        ++v219;
      }
    }
    if ( v6 )
      v155 = (__m128i *)v6;
LABEL_409:
    v181 = v267.m128i_u32[2];
LABEL_326:
    v267.m128i_i32[2] = (2 * (v181 >> 1) + 2) | v181 & 1;
    if ( v155->m128i_i64[0] != -4096 )
      --v267.m128i_i32[3];
    v155->m128i_i32[2] = 0;
    v155->m128i_i64[0] = (__int64)v242;
    v184 = (unsigned int)v271;
    v259 = &v261;
    v263.m128i_i64[0] = (__int64)v242;
    v152 = (unsigned int)v271 + 1LL;
    v260 = 0x200000000LL;
    v264.m128i_i64[0] = 0x200000000LL;
    v185 = v271;
    v263.m128i_i64[1] = (__int64)&v264.m128i_i64[1];
    if ( v152 > HIDWORD(v271) )
    {
      if ( v270 > &v263 || &v263 >= (__m128i *)((char *)v270 + 200 * (unsigned int)v271) )
      {
        v243 = 0;
        v241 = -1;
      }
      else
      {
        v241 = 0x8F5C28F5C28F5C29LL * (((char *)&v263 - (char *)v270) >> 3);
        v243 = a4;
      }
      v196 = sub_C8D7D0((__int64)&v270, (__int64)v272, (unsigned int)v271 + 1LL, 0xC8u, &v250, v152);
      v198 = v270;
      v239 = (__m128i *)v196;
      v199 = v196;
      v200 = (unsigned int)v271;
      v201 = v270;
      v202 = 25LL * (unsigned int)v271;
      v203 = (__m128i *)((char *)v270 + 200 * (unsigned int)v271);
      if ( v270 != v203 )
      {
        do
        {
          if ( v199 )
          {
            v204 = v201->m128i_i64[0];
            *(_DWORD *)(v199 + 16) = 0;
            *(_DWORD *)(v199 + 20) = 2;
            *(_QWORD *)v199 = v204;
            *(_QWORD *)(v199 + 8) = v199 + 24;
            v152 = v201[1].m128i_u32[0];
            if ( (_DWORD)v152 )
              sub_2757E90(v199 + 8, (__int64)&v201->m128i_i64[1], v202, v197, v6, v152);
          }
          v201 = (__m128i *)((char *)v201 + 200);
          v199 += 200;
        }
        while ( v203 != v201 );
        v198 = v270;
        v200 = (unsigned int)v271;
      }
      v205 = (__m128i *)((char *)v198 + 200 * v200);
      if ( v198 != v205 )
      {
        v234 = v155;
        v206 = v205;
        v225 = v198;
        do
        {
          v207 = v206[-12].m128i_u32[2];
          v208 = v206[-12].m128i_u64[0];
          v206 = (__m128i *)((char *)v206 - 200);
          v209 = v208 + 88 * v207;
          if ( v208 != v209 )
          {
            do
            {
              v210 = *(unsigned int *)(v209 - 72);
              v211 = *(_QWORD *)(v209 - 80);
              v209 -= 88LL;
              v212 = v211 + 32 * v210;
              if ( v211 != v212 )
              {
                do
                {
                  v212 -= 32LL;
                  if ( *(_DWORD *)(v212 + 24) > 0x40u )
                  {
                    v213 = *(_QWORD *)(v212 + 16);
                    if ( v213 )
                      j_j___libc_free_0_0(v213);
                  }
                  if ( *(_DWORD *)(v212 + 8) > 0x40u && *(_QWORD *)v212 )
                    j_j___libc_free_0_0(*(_QWORD *)v212);
                }
                while ( v211 != v212 );
                v211 = *(_QWORD *)(v209 + 8);
              }
              if ( v211 != v209 + 24 )
                _libc_free(v211);
            }
            while ( v208 != v209 );
            v208 = v206->m128i_u64[1];
          }
          if ( (unsigned __int64 *)v208 != &v206[1].m128i_u64[1] )
            _libc_free(v208);
        }
        while ( v225 != v206 );
        v155 = v234;
        v198 = v270;
      }
      v214 = v250;
      if ( v198 != (__m128i *)v272 )
        _libc_free((unsigned __int64)v198);
      v184 = (unsigned int)v271;
      HIDWORD(v271) = v214;
      v153 = (__int64)&v263;
      v270 = v239;
      v185 = v271;
      if ( v243 )
        v153 = (__int64)&v239->m128i_i64[25 * v241];
    }
    else
    {
      v153 = (__int64)&v263;
      v239 = v270;
    }
    v186 = &v239->m128i_i64[25 * v184];
    if ( v186 )
    {
      *v186 = *(_QWORD *)v153;
      v186[1] = (__int64)(v186 + 3);
      v186[2] = 0x200000000LL;
      v6 = *(unsigned int *)(v153 + 16);
      if ( (_DWORD)v6 )
        sub_2757E90((__int64)(v186 + 1), v153 + 8, (__int64)v186, v153, v6, v152);
      v185 = v271;
    }
    v187 = v263.m128i_i64[1];
    LODWORD(v271) = v185 + 1;
    v188 = (__m128i *)(v263.m128i_i64[1] + 88LL * v264.m128i_u32[0]);
    if ( (__m128i *)v263.m128i_i64[1] != v188 )
    {
      do
      {
        v189 = v188[-5].m128i_u32[2];
        v190 = v188[-5].m128i_u64[0];
        v188 = (__m128i *)((char *)v188 - 88);
        v191 = v190 + 32 * v189;
        if ( v190 != v191 )
        {
          do
          {
            v191 -= 32LL;
            if ( *(_DWORD *)(v191 + 24) > 0x40u )
            {
              v192 = *(_QWORD *)(v191 + 16);
              if ( v192 )
                j_j___libc_free_0_0(v192);
            }
            if ( *(_DWORD *)(v191 + 8) > 0x40u && *(_QWORD *)v191 )
              j_j___libc_free_0_0(*(_QWORD *)v191);
          }
          while ( v190 != v191 );
          v190 = v188->m128i_u64[1];
        }
        if ( (unsigned __int64 *)v190 != &v188[1].m128i_u64[1] )
          _libc_free(v190);
      }
      while ( (__m128i *)v187 != v188 );
      v188 = (__m128i *)v263.m128i_i64[1];
    }
    if ( v188 != (__m128i *)&v264.m128i_u64[1] )
      _libc_free((unsigned __int64)v188);
    v156 = (unsigned int)(v271 - 1);
    v155->m128i_i32[2] = v156;
LABEL_267:
    v157 = &v270->m128i_i8[200 * v156];
    v158 = *((unsigned int *)v157 + 4);
    v159 = *((_QWORD *)v157 + 1);
    v160 = 5 * v158;
    v161 = v159 + 88 * v158;
    while ( v159 != v161 )
    {
      while ( 1 )
      {
        v162 = *(unsigned int *)(v161 - 72);
        v163 = *(_QWORD *)(v161 - 80);
        v161 -= 88;
        v164 = v163 + 32 * v162;
        if ( v163 != v164 )
        {
          do
          {
            v164 -= 32LL;
            if ( *(_DWORD *)(v164 + 24) > 0x40u )
            {
              v165 = *(_QWORD *)(v164 + 16);
              if ( v165 )
                j_j___libc_free_0_0(v165);
            }
            if ( *(_DWORD *)(v164 + 8) > 0x40u && *(_QWORD *)v164 )
              j_j___libc_free_0_0(*(_QWORD *)v164);
          }
          while ( v163 != v164 );
          v163 = *(_QWORD *)(v161 + 8);
        }
        if ( v163 == v161 + 24 )
          break;
        _libc_free(v163);
        if ( v159 == v161 )
          goto LABEL_280;
      }
    }
LABEL_280:
    v166 = *((_DWORD *)v157 + 5);
    *((_DWORD *)v157 + 4) = 0;
    if ( v166 )
    {
      v167 = 0;
      v168 = 0;
    }
    else
    {
      sub_2757D10((__int64)(v157 + 8), 1u, v160, v153, v6, v152);
      v168 = *((_DWORD *)v157 + 4);
      v167 = 88LL * v168;
    }
    v169 = *((_QWORD *)v157 + 1) + v167;
    if ( v169 )
    {
      *(_DWORD *)v169 = src.m128i_i32[0];
      v170 = src.m128i_i8[4];
      *(_QWORD *)(v169 + 16) = 0x200000000LL;
      *(_BYTE *)(v169 + 4) = v170;
      *(_QWORD *)(v169 + 8) = v169 + 24;
      if ( (_DWORD)v257 )
        sub_27583E0(v169 + 8, (__int64)&src.m128i_i64[1]);
      v168 = *((_DWORD *)v157 + 4);
    }
    *((_DWORD *)v157 + 4) = v168 + 1;
    v171 = src.m128i_i64[1];
    v172 = src.m128i_i64[1] + 32LL * (unsigned int)v257;
    if ( src.m128i_i64[1] != v172 )
    {
      do
      {
        v172 -= 32LL;
        if ( *(_DWORD *)(v172 + 24) > 0x40u )
        {
          v173 = *(_QWORD *)(v172 + 16);
          if ( v173 )
            j_j___libc_free_0_0(v173);
        }
        if ( *(_DWORD *)(v172 + 8) > 0x40u && *(_QWORD *)v172 )
          j_j___libc_free_0_0(*(_QWORD *)v172);
      }
      while ( v171 != v172 );
      v172 = src.m128i_u64[1];
    }
    if ( (_BYTE *)v172 != v258 )
      _libc_free(v172);
LABEL_73:
    v63 = v254.m128i_i64[1];
    v64 = (__m128i *)(v254.m128i_i64[1] + 32LL * v255.m128i_u32[0]);
    if ( (__m128i *)v254.m128i_i64[1] != v64 )
    {
      do
      {
        v64 -= 2;
        if ( v64[1].m128i_i32[2] > 0x40u )
        {
          v65 = v64[1].m128i_u64[0];
          if ( v65 )
            j_j___libc_free_0_0(v65);
        }
        if ( v64->m128i_i32[2] > 0x40u && v64->m128i_i64[0] )
          j_j___libc_free_0_0(v64->m128i_i64[0]);
      }
      while ( (__m128i *)v63 != v64 );
      v64 = (__m128i *)v254.m128i_i64[1];
    }
    if ( v64 != (__m128i *)&v255.m128i_u64[1] )
      _libc_free((unsigned __int64)v64);
    v66 = v251;
    v67 = (unsigned __int64)&v251[32 * (unsigned int)v252];
    if ( v251 != (_BYTE *)v67 )
    {
      do
      {
        v67 -= 32LL;
        if ( *(_DWORD *)(v67 + 24) > 0x40u )
        {
          v68 = *(_QWORD *)(v67 + 16);
          if ( v68 )
            j_j___libc_free_0_0(v68);
        }
        if ( *(_DWORD *)(v67 + 8) > 0x40u && *(_QWORD *)v67 )
          j_j___libc_free_0_0(*(_QWORD *)v67);
      }
      while ( v66 != (_BYTE *)v67 );
      v67 = (unsigned __int64)v251;
    }
    if ( (_BYTE *)v67 != v253 )
      _libc_free(v67);
  }
  v4 = (unsigned __int8 *)v229;
LABEL_97:
  src.m128i_i64[0] = (__int64)&v257;
  v69 = (unsigned int)v271;
  src.m128i_i64[1] = 0x100000000LL;
  v245 = v270;
  v233 = (__m128i *)((char *)v270 + 200 * (unsigned int)v271);
  if ( v270 == v233 )
  {
    v263.m128i_i64[1] = 0x100000000LL;
    v263.m128i_i64[0] = (__int64)&v264;
    goto LABEL_112;
  }
  v70 = (__int64)v4;
  v231 = v4 + 72;
  v236 = v270;
  while ( 2 )
  {
    v71 = sub_A73ED0(v231, 41);
    if ( !v71 )
      v71 = sub_B49560(v70, 41);
    if ( v236[1].m128i_i32[0] )
    {
      v73 = v236->m128i_i64[1];
      v74 = v73 + 88LL * v236[1].m128i_u32[0];
      v75 = v73;
      do
      {
        if ( !v71 && !*(_BYTE *)(v75 + 4) || !*(_DWORD *)(v75 + 16) )
          goto LABEL_107;
        v75 += 88;
      }
      while ( v74 != v75 );
      v259 = &v261;
      v260 = 0x200000000LL;
      if ( *(_DWORD *)(v73 + 16) )
        sub_27583E0((__int64)&v259, v73 + 8);
      v101 = v73 + 88;
      if ( v74 != v101 )
      {
        v246 = v74;
        v237 = v70;
        do
        {
          sub_ABFB50(&v263, (__int64)&v259, v101 + 8);
          sub_27578C0((__int64)&v259, v263.m128i_i64);
          v102 = v263.m128i_i64[0];
          v6 = 32LL * v263.m128i_u32[2];
          v103 = (__m128i *)(v263.m128i_i64[0] + v6);
          if ( v263.m128i_i64[0] != v263.m128i_i64[0] + v6 )
          {
            do
            {
              v103 -= 2;
              if ( v103[1].m128i_i32[2] > 0x40u )
              {
                v104 = v103[1].m128i_u64[0];
                if ( v104 )
                  j_j___libc_free_0_0(v104);
              }
              if ( v103->m128i_i32[2] > 0x40u && v103->m128i_i64[0] )
                j_j___libc_free_0_0(v103->m128i_i64[0]);
            }
            while ( (__m128i *)v102 != v103 );
            v103 = (__m128i *)v263.m128i_i64[0];
          }
          if ( v103 != &v264 )
            _libc_free((unsigned __int64)v103);
          v101 += 88;
        }
        while ( v246 != v101 );
        v70 = v237;
      }
      v263.m128i_i64[0] = (__int64)&v264;
      v263.m128i_i64[1] = 0x200000000LL;
      if ( (_DWORD)v260 )
      {
        sub_27578C0((__int64)&v263, (__int64 *)&v259);
        v105 = v259;
        v132 = &v259[2 * (unsigned int)v260];
        if ( v259 != v132 )
        {
          do
          {
            v132 -= 2;
            if ( v132[1].m128i_i32[2] > 0x40u )
            {
              v133 = v132[1].m128i_u64[0];
              if ( v133 )
                j_j___libc_free_0_0(v133);
            }
            if ( v132->m128i_i32[2] > 0x40u && v132->m128i_i64[0] )
              j_j___libc_free_0_0(v132->m128i_i64[0]);
          }
          while ( v105 != v132 );
          goto LABEL_179;
        }
      }
      else
      {
LABEL_179:
        v105 = v259;
      }
      if ( v105 != &v261 )
        _libc_free((unsigned __int64)v105);
      v106 = v263.m128i_u32[2];
      v107 = v263.m128i_i64[0];
      if ( v263.m128i_i32[2] )
      {
        v108 = (unsigned int *)v236->m128i_i64[1];
        v238 = &v108[22 * v236[1].m128i_u32[0]];
        if ( v108 == v238 )
        {
          v110 = (__m128i *)(v263.m128i_i64[0] + 32LL * v263.m128i_u32[2]);
          goto LABEL_202;
        }
        v109 = &v254;
        while ( 1 )
        {
          v110 = (__m128i *)v107;
          v6 = 32 * v106;
          v111 = v107 + 32 * v106;
          if ( v111 == v107 )
            goto LABEL_201;
          v112 = v109;
          v113 = (__m128i *)v111;
          v114 = (__int64 *)v112;
          do
          {
            while ( 1 )
            {
              v117 = v110->m128i_u32[2];
              v118 = v110->m128i_i64[0];
              v119 = (__int64 *)v110[1].m128i_i64[0];
              v120 = v110[1].m128i_u32[2];
              if ( v117 <= 0x40 )
                break;
              v115 = *(_QWORD *)v118;
              if ( v120 <= 0x40 )
              {
LABEL_189:
                if ( v120 )
                {
                  v116 = (__int64)((_QWORD)v119 << (64 - (unsigned __int8)v120)) >> (64 - (unsigned __int8)v120);
                  if ( !v115 )
                    goto LABEL_195;
                }
                else if ( !v115 )
                {
LABEL_218:
                  sub_B91FC0(v114, v70);
                  v121 = 0;
                  goto LABEL_197;
                }
                goto LABEL_191;
              }
LABEL_194:
              v116 = *v119;
              if ( !v115 )
                goto LABEL_195;
LABEL_191:
              v110 += 2;
              if ( v113 == v110 )
                goto LABEL_200;
            }
            if ( v117 )
            {
              v115 = (__int64)(v118 << (64 - (unsigned __int8)v117)) >> (64 - (unsigned __int8)v117);
              if ( v120 <= 0x40 )
                goto LABEL_189;
              goto LABEL_194;
            }
            if ( v120 > 0x40 )
            {
              v116 = *v119;
            }
            else
            {
              if ( !v120 )
                goto LABEL_218;
              v116 = (__int64)((_QWORD)v119 << (64 - (unsigned __int8)v120)) >> (64 - (unsigned __int8)v120);
            }
LABEL_195:
            v247 = v116;
            sub_B91FC0(v114, v70);
            v121 = v247;
            if ( v247 > 0x3FFFFFFFFFFFFFFBLL )
              v121 = 0xBFFFFFFFFFFFFFFELL;
LABEL_197:
            v122 = *(_DWORD *)(v70 + 4);
            v123 = *v108;
            v260 = v121;
            v124 = src.m128i_u32[2];
            v125 = _mm_loadu_si128(&v255);
            v261 = _mm_loadu_si128(&v254);
            v262 = v125;
            v259 = *(__m128i **)(v70 + 32 * (v123 - (v122 & 0x7FFFFFF)));
            v126 = src.m128i_u32[2] + 1LL;
            if ( v126 > src.m128i_u32[3] )
            {
              if ( src.m128i_i64[0] > (unsigned __int64)&v259
                || (v248 = src.m128i_i64[0],
                    (unsigned __int64)&v259 >= src.m128i_i64[0] + 48 * (unsigned __int64)src.m128i_u32[2]) )
              {
                sub_C8D5F0((__int64)&src, &v257, v126, 0x30u, v6, src.m128i_i64[0]);
                v127 = src.m128i_i64[0];
                v124 = src.m128i_u32[2];
                v128 = (const __m128i *)&v259;
              }
              else
              {
                sub_C8D5F0((__int64)&src, &v257, v126, 0x30u, v6, src.m128i_i64[0]);
                v72 = v248;
                v127 = src.m128i_i64[0];
                v124 = src.m128i_u32[2];
                v128 = (const __m128i *)((char *)&v259 + src.m128i_i64[0] - v248);
              }
            }
            else
            {
              v127 = src.m128i_i64[0];
              v128 = (const __m128i *)&v259;
            }
            v110 += 2;
            v129 = (__m128i *)(v127 + 48 * v124);
            *v129 = _mm_loadu_si128(v128);
            v129[1] = _mm_loadu_si128(v128 + 1);
            v129[2] = _mm_loadu_si128(v128 + 2);
            ++src.m128i_i32[2];
          }
          while ( v113 != v110 );
LABEL_200:
          v106 = v263.m128i_u32[2];
          v107 = v263.m128i_i64[0];
          v109 = (__m128i *)v114;
          v110 = (__m128i *)(v263.m128i_i64[0] + 32LL * v263.m128i_u32[2]);
LABEL_201:
          v108 += 22;
          if ( v238 == v108 )
          {
LABEL_202:
            if ( (__m128i *)v107 != v110 )
            {
              v130 = (__m128i *)v107;
              do
              {
                v110 -= 2;
                if ( v110[1].m128i_i32[2] > 0x40u )
                {
                  v131 = v110[1].m128i_u64[0];
                  if ( v131 )
                    j_j___libc_free_0_0(v131);
                }
                if ( v110->m128i_i32[2] > 0x40u && v110->m128i_i64[0] )
                  j_j___libc_free_0_0(v110->m128i_i64[0]);
              }
              while ( v110 != v130 );
              v110 = (__m128i *)v263.m128i_i64[0];
            }
            if ( v110 != &v264 )
              _libc_free((unsigned __int64)v110);
            goto LABEL_107;
          }
        }
      }
      if ( (__m128i *)v263.m128i_i64[0] != &v264 )
        _libc_free(v263.m128i_u64[0]);
    }
LABEL_107:
    v236 = (__m128i *)((char *)v236 + 200);
    if ( v233 != v236 )
      continue;
    break;
  }
  v76 = src.m128i_i32[2];
  v77 = (__int64 *)src.m128i_i64[0];
  v263.m128i_i64[0] = (__int64)&v264;
  v263.m128i_i64[1] = 0x100000000LL;
  if ( !src.m128i_i32[2] )
    goto LABEL_109;
  if ( (__int64 *)src.m128i_i64[0] == &v257 )
  {
    v179 = &v264;
    v180 = 48;
    if ( src.m128i_i32[2] == 1 )
      goto LABEL_320;
    sub_C8D5F0((__int64)&v263, &v264, src.m128i_u32[2], 0x30u, src.m128i_u32[2], v72);
    v77 = (__int64 *)src.m128i_i64[0];
    v180 = 48LL * src.m128i_u32[2];
    if ( v180 )
    {
      v179 = (__m128i *)v263.m128i_i64[0];
LABEL_320:
      memcpy(v179, v77, v180);
      v77 = (__int64 *)src.m128i_i64[0];
    }
    v263.m128i_i32[2] = v76;
LABEL_109:
    if ( v77 != &v257 )
      _libc_free((unsigned __int64)v77);
    v245 = v270;
    v69 = (unsigned int)v271;
  }
  else
  {
    v263 = src;
    v245 = v270;
    v69 = (unsigned int)v271;
  }
LABEL_112:
  v78 = (unsigned __int64)v245 + 200 * v69;
  if ( v245 != (__m128i *)v78 )
  {
    do
    {
      v79 = *(unsigned int *)(v78 - 184);
      v80 = *(_QWORD *)(v78 - 192);
      v78 -= 200LL;
      v81 = v80 + 88 * v79;
      if ( v80 != v81 )
      {
        do
        {
          v82 = *(unsigned int *)(v81 - 72);
          v83 = *(_QWORD *)(v81 - 80);
          v81 -= 88LL;
          v84 = v83 + 32 * v82;
          if ( v83 != v84 )
          {
            do
            {
              v84 -= 32LL;
              if ( *(_DWORD *)(v84 + 24) > 0x40u )
              {
                v85 = *(_QWORD *)(v84 + 16);
                if ( v85 )
                  j_j___libc_free_0_0(v85);
              }
              if ( *(_DWORD *)(v84 + 8) > 0x40u && *(_QWORD *)v84 )
                j_j___libc_free_0_0(*(_QWORD *)v84);
            }
            while ( v83 != v84 );
            v83 = *(_QWORD *)(v81 + 8);
          }
          if ( v83 != v81 + 24 )
            _libc_free(v83);
        }
        while ( v80 != v81 );
        v80 = *(_QWORD *)(v78 + 8);
      }
      if ( v80 != v78 + 24 )
        _libc_free(v80);
    }
    while ( v245 != (__m128i *)v78 );
    v78 = (unsigned __int64)v270;
  }
  if ( (_BYTE *)v78 != v272 )
    _libc_free(v78);
  if ( (v267.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0(v268.m128i_i64[0], 16LL * v268.m128i_u32[2], 8);
  v86 = (const __m128i *)v263.m128i_i64[0];
  v87 = (__m128i *)(v263.m128i_i64[0] + 48LL * v263.m128i_u32[2]);
  if ( (__m128i *)v263.m128i_i64[0] != v87 )
  {
    v88 = v263.m128i_i64[0] + 48LL * v263.m128i_u32[2];
    v89 = *(unsigned int *)(a1 + 8);
    do
    {
      v90 = *(unsigned int *)(a1 + 12);
      v91 = v89 + 1;
      v92 = &v267;
      v93 = *(_QWORD *)a1;
      v267 = _mm_loadu_si128(v86);
      v268 = _mm_loadu_si128(v86 + 1);
      v94 = _mm_loadu_si128(v86 + 2);
      LOBYTE(v270) = 1;
      v269 = v94;
      if ( v89 + 1 > v90 )
      {
        if ( v93 > (unsigned __int64)&v267 || (unsigned __int64)&v267 >= v93 + 56 * v89 )
        {
          sub_C8D5F0(a1, v227, v91, 0x38u, v6, v91);
          v93 = *(_QWORD *)a1;
          v89 = *(unsigned int *)(a1 + 8);
          v92 = &v267;
        }
        else
        {
          v175 = &v267.m128i_i8[-v93];
          sub_C8D5F0(a1, v227, v91, 0x38u, v6, v91);
          v93 = *(_QWORD *)a1;
          v89 = *(unsigned int *)(a1 + 8);
          v92 = (__m128i *)&v175[*(_QWORD *)a1];
        }
      }
      v86 += 3;
      v95 = (__m128i *)(v93 + 56 * v89);
      *v95 = _mm_loadu_si128(v92);
      v95[1] = _mm_loadu_si128(v92 + 1);
      v95[2] = _mm_loadu_si128(v92 + 2);
      v95[3].m128i_i64[0] = v92[3].m128i_i64[0];
      v89 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v89;
    }
    while ( (const __m128i *)v88 != v86 );
    v87 = (__m128i *)v263.m128i_i64[0];
  }
  if ( v87 != &v264 )
    _libc_free((unsigned __int64)v87);
  return a1;
}
