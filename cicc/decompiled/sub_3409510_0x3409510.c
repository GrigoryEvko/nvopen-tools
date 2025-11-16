// Function: sub_3409510
// Address: 0x3409510
//
unsigned __int8 *__fastcall sub_3409510(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9,
        unsigned __int8 a10,
        char a11,
        char a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        const __m128i *a19,
        __int64 *a20)
{
  unsigned __int8 *v20; // r13
  __int64 v22; // r14
  _QWORD *v23; // r15
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 *v26; // rax
  __int64 v27; // rax
  __int32 v28; // eax
  char v29; // r8
  unsigned __int16 v30; // ax
  char v31; // r8
  char v32; // r9
  int v33; // edx
  unsigned __int8 (__fastcall *v34)(_DWORD *, unsigned __int64 *, _QWORD, unsigned __int64 *, _QWORD, _QWORD, _OWORD **); // r14
  unsigned int v35; // r13d
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __m128i v41; // xmm0
  __m128i v42; // xmm1
  unsigned __int64 v43; // rdx
  __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  __int8 v47; // cl
  unsigned __int64 v48; // rbx
  __int64 v49; // rdx
  unsigned __int64 v50; // rdx
  char v51; // al
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // r14
  unsigned __int64 v54; // rax
  unsigned __int16 v55; // dx
  __int64 (__fastcall *v56)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v57; // r13d
  unsigned __int64 v58; // rdx
  char v59; // al
  char v60; // dl
  __int16 v61; // r15
  unsigned __int64 v62; // rcx
  unsigned __int64 v63; // rax
  __int16 v64; // ax
  unsigned __int64 v65; // rdx
  unsigned __int8 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r9
  unsigned __int64 v69; // r15
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // r13
  unsigned __int64 v73; // r8
  unsigned __int64 v74; // rdx
  __m128i **v75; // rax
  unsigned __int64 v76; // rdx
  unsigned __int8 *v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // r8
  __m128i *v80; // r13
  __int64 v81; // rax
  unsigned int v82; // edx
  unsigned __int64 v83; // r12
  unsigned __int64 v84; // rdx
  __m128i **v85; // rax
  __int64 v86; // rax
  unsigned int v87; // r12d
  __int64 v88; // r13
  __int64 v89; // rax
  __int64 v90; // rdx
  unsigned int v91; // eax
  unsigned int v92; // ecx
  unsigned int v93; // r8d
  _BYTE *v94; // rax
  __int64 v95; // rdx
  __int64 v96; // r9
  __int64 v97; // rcx
  __int64 v98; // r8
  unsigned int v99; // ebx
  int v100; // r14d
  int v101; // r12d
  unsigned __int64 v102; // rax
  unsigned __int64 v103; // rax
  __int64 v104; // rax
  __int64 (*v105)(); // rcx
  unsigned int v106; // r12d
  unsigned __int8 *v107; // r13
  unsigned __int16 v108; // dx
  unsigned int v109; // edx
  unsigned __int64 v110; // rcx
  unsigned __int8 *v111; // rax
  unsigned __int64 v112; // rdx
  __m128i *v113; // r13
  __int64 v114; // rax
  unsigned int v115; // edx
  __int64 v116; // r8
  unsigned __int64 v117; // rdx
  __m128i **v118; // rax
  unsigned int v119; // r14d
  __int64 (__fastcall *v120)(__int64); // rax
  __int64 v121; // r8
  __int64 v122; // rax
  char v123; // bl
  unsigned int v124; // r14d
  int v125; // r12d
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rax
  __int16 v128; // ax
  __int64 v129; // rdx
  __int64 v130; // r8
  __int64 v131; // rbx
  __int64 v132; // r12
  __int64 v133; // rax
  __m128i v134; // xmm0
  __m128i v135; // xmm0
  __int64 v136; // rdx
  __int64 v137; // rdx
  int v138; // eax
  __int64 v139; // rdx
  int v140; // eax
  __int64 v141; // rsi
  unsigned __int8 v142; // r14
  __int64 v143; // r13
  int v144; // eax
  __int64 v145; // rdx
  __int64 v146; // rax
  char v147; // cl
  unsigned __int64 v148; // rax
  int v149; // edx
  __int64 v150; // rdi
  int v151; // r9d
  unsigned int v152; // edx
  int v153; // eax
  __int64 *v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  _QWORD *v157; // rcx
  unsigned int v158; // eax
  __int64 v159; // rdx
  unsigned int v160; // edx
  char v161; // al
  __int64 v162; // rdx
  int v163; // eax
  unsigned int v164; // r12d
  unsigned int v165; // ebx
  unsigned int v166; // ecx
  unsigned int v167; // r8d
  __int128 v168; // [rsp+0h] [rbp-6D0h]
  __int64 v169; // [rsp+8h] [rbp-6C8h]
  __int64 v170; // [rsp+10h] [rbp-6C0h]
  unsigned int v171; // [rsp+30h] [rbp-6A0h]
  unsigned int v172; // [rsp+38h] [rbp-698h]
  unsigned __int64 v173; // [rsp+38h] [rbp-698h]
  unsigned int v174; // [rsp+38h] [rbp-698h]
  __int64 v175; // [rsp+40h] [rbp-690h]
  __int64 v176; // [rsp+58h] [rbp-678h]
  unsigned int v177; // [rsp+58h] [rbp-678h]
  unsigned __int64 v178; // [rsp+58h] [rbp-678h]
  unsigned int v179; // [rsp+58h] [rbp-678h]
  __int64 v180; // [rsp+58h] [rbp-678h]
  unsigned __int64 v181; // [rsp+60h] [rbp-670h]
  __int64 v182; // [rsp+68h] [rbp-668h]
  __int64 *v183; // [rsp+70h] [rbp-660h]
  __int64 v185; // [rsp+88h] [rbp-648h]
  unsigned __int128 v187; // [rsp+A0h] [rbp-630h]
  __int64 v188; // [rsp+B0h] [rbp-620h]
  unsigned int v189; // [rsp+B0h] [rbp-620h]
  unsigned __int64 v190; // [rsp+B0h] [rbp-620h]
  __int64 v191; // [rsp+B8h] [rbp-618h]
  __int16 v192; // [rsp+B8h] [rbp-618h]
  __int64 v193; // [rsp+C0h] [rbp-610h]
  char v194; // [rsp+C8h] [rbp-608h]
  unsigned int v195; // [rsp+C8h] [rbp-608h]
  __int16 v196; // [rsp+C8h] [rbp-608h]
  bool v197; // [rsp+D2h] [rbp-5FEh]
  unsigned __int8 v198; // [rsp+D3h] [rbp-5FDh]
  __int16 v199; // [rsp+D4h] [rbp-5FCh]
  char v200; // [rsp+D6h] [rbp-5FAh]
  char v201; // [rsp+D7h] [rbp-5F9h]
  char v202; // [rsp+D7h] [rbp-5F9h]
  _DWORD *v203; // [rsp+D8h] [rbp-5F8h]
  unsigned int v204; // [rsp+D8h] [rbp-5F8h]
  bool v205; // [rsp+E0h] [rbp-5F0h]
  __int64 v206; // [rsp+E0h] [rbp-5F0h]
  unsigned int v207; // [rsp+E0h] [rbp-5F0h]
  __m128i v209; // [rsp+F0h] [rbp-5E0h] BYREF
  __m128i *v210; // [rsp+100h] [rbp-5D0h]
  __int64 v211; // [rsp+108h] [rbp-5C8h]
  unsigned __int64 v212; // [rsp+110h] [rbp-5C0h] BYREF
  char v213; // [rsp+118h] [rbp-5B8h]
  unsigned __int8 v214; // [rsp+119h] [rbp-5B7h]
  int v215; // [rsp+11Ah] [rbp-5B6h]
  __int8 v216; // [rsp+11Eh] [rbp-5B2h]
  __m128i v217; // [rsp+120h] [rbp-5B0h] BYREF
  unsigned __int64 v218; // [rsp+130h] [rbp-5A0h]
  __int64 v219; // [rsp+138h] [rbp-598h]
  unsigned __int64 v220; // [rsp+140h] [rbp-590h]
  __int64 v221; // [rsp+148h] [rbp-588h]
  unsigned __int64 v222; // [rsp+150h] [rbp-580h]
  __int64 v223; // [rsp+158h] [rbp-578h]
  __int64 v224; // [rsp+160h] [rbp-570h]
  __int64 v225; // [rsp+168h] [rbp-568h]
  __m128i v226; // [rsp+170h] [rbp-560h] BYREF
  __int64 v227; // [rsp+180h] [rbp-550h]
  __int64 v228; // [rsp+188h] [rbp-548h]
  unsigned __int64 v229; // [rsp+190h] [rbp-540h] BYREF
  __int64 v230; // [rsp+198h] [rbp-538h]
  __int64 v231; // [rsp+1A0h] [rbp-530h]
  __int64 v232; // [rsp+1B0h] [rbp-520h] BYREF
  __int64 v233; // [rsp+1B8h] [rbp-518h]
  unsigned __int64 v234; // [rsp+1C0h] [rbp-510h]
  __int128 v235; // [rsp+1D0h] [rbp-500h]
  __int64 v236; // [rsp+1E0h] [rbp-4F0h]
  __int128 v237; // [rsp+1F0h] [rbp-4E0h]
  __int64 v238; // [rsp+200h] [rbp-4D0h]
  __int128 v239; // [rsp+210h] [rbp-4C0h]
  __int64 v240; // [rsp+220h] [rbp-4B0h]
  unsigned __int64 v241; // [rsp+230h] [rbp-4A0h] BYREF
  __int64 v242; // [rsp+238h] [rbp-498h]
  __int64 v243; // [rsp+240h] [rbp-490h]
  _QWORD v244[2]; // [rsp+250h] [rbp-480h] BYREF
  __m128i v245; // [rsp+260h] [rbp-470h]
  _BYTE *v246; // [rsp+270h] [rbp-460h] BYREF
  __int64 v247; // [rsp+278h] [rbp-458h]
  _BYTE v248[256]; // [rsp+280h] [rbp-450h] BYREF
  _BYTE *v249; // [rsp+380h] [rbp-350h] BYREF
  __int64 v250; // [rsp+388h] [rbp-348h]
  _BYTE v251[256]; // [rsp+390h] [rbp-340h] BYREF
  _OWORD *v252; // [rsp+490h] [rbp-240h] BYREF
  __int64 v253; // [rsp+498h] [rbp-238h]
  _OWORD v254[35]; // [rsp+4A0h] [rbp-230h] BYREF

  v187 = __PAIR128__(a4, a3);
  if ( *(_DWORD *)(a7 + 24) == 51 )
    return (unsigned __int8 *)a3;
  v22 = a7;
  v23 = (_QWORD *)a1;
  v198 = a10;
  v203 = *(_DWORD **)(a1 + 16);
  v24 = sub_2E79000(*(__int64 **)(a1 + 40));
  v25 = *(_QWORD *)(a1 + 40);
  v229 = 0;
  v182 = v24;
  v26 = *(__int64 **)(a1 + 64);
  v230 = 0;
  v183 = v26;
  v27 = *(_QWORD *)(v25 + 48);
  v231 = 0;
  v188 = v27;
  v194 = sub_33CC5F0((__int64 *)v25, a1);
  v191 = a5;
  v205 = *(_DWORD *)(a5 + 24) == 15 || *(_DWORD *)(a5 + 24) == 39;
  if ( v205 )
  {
    v28 = *(_DWORD *)(a5 + 96);
    v29 = 1;
    if ( v28 < 0 )
    {
      v209.m128i_i32[0] = *(_DWORD *)(v188 + 32);
      v205 = v28 < -v209.m128i_i32[0];
      v29 = v205;
    }
  }
  else
  {
    v191 = 0;
    v29 = 0;
  }
  v201 = v29;
  v30 = sub_33E0440(a1, a7, a8);
  v31 = v201;
  v209.m128i_i8[0] = a10;
  if ( HIBYTE(v30) )
  {
    if ( a10 >= (unsigned __int8)v30 )
      LOBYTE(v30) = a10;
    v209.m128i_i8[0] = v30;
  }
  v32 = a11 ^ 1;
  if ( a11 )
    goto LABEL_12;
  v153 = *(_DWORD *)(a7 + 24);
  if ( v153 == 13 )
  {
    v157 = 0;
  }
  else
  {
    if ( v153 != 56 )
      goto LABEL_12;
    v154 = *(__int64 **)(a7 + 40);
    v22 = *v154;
    if ( *(_DWORD *)(*v154 + 24) != 13 )
      goto LABEL_12;
    v155 = v154[5];
    if ( *(_DWORD *)(v155 + 24) != 11 )
      goto LABEL_12;
    v156 = *(_QWORD *)(v155 + 96);
    v157 = *(_QWORD **)(v156 + 24);
    if ( *(_DWORD *)(v156 + 32) > 0x40u )
      v157 = (_QWORD *)*v157;
  }
  v161 = sub_98AE20(*(_QWORD *)(v22 + 96), &v232, 8u, (__int64)v157 + *(_QWORD *)(v22 + 104));
  v32 = 1;
  v31 = v201;
  v200 = v161;
  if ( !v161 )
  {
LABEL_12:
    v202 = 0;
    goto LABEL_13;
  }
  if ( v232 )
  {
    v202 = v161;
LABEL_13:
    v33 = -1;
    if ( !a12 )
    {
      v33 = v203[134244];
      if ( v194 )
        v33 = v203[134245];
    }
    v213 = v31;
    LOBYTE(v215) = v32;
    v212 = a9;
    *(_WORD *)((char *)&v215 + 1) = 0;
    v214 = a10;
    v200 = 0;
    HIBYTE(v215) = v202;
    v216 = v209.m128i_i8[0];
    goto LABEL_15;
  }
  v33 = -1;
  if ( !a12 )
  {
    v33 = v203[134245];
    if ( !v194 )
      v33 = v203[134244];
  }
  v213 = v201;
  v215 = 65793;
  v212 = a9;
  v216 = 0;
  v214 = a10;
  v202 = v161;
LABEL_15:
  v195 = v33;
  v34 = *(unsigned __int8 (__fastcall **)(_DWORD *, unsigned __int64 *, _QWORD, unsigned __int64 *, _QWORD, _QWORD, _OWORD **))(*(_QWORD *)v203 + 1984LL);
  v252 = *(_OWORD **)(*(_QWORD *)v25 + 120LL);
  v35 = sub_2EAC1E0((__int64)&a16);
  v36 = sub_2EAC1E0((__int64)&a13);
  if ( v34(v203, &v229, v195, &v212, v36, v35, &v252) )
  {
    if ( v205 )
    {
      v141 = sub_3007410(v229, v183, v37, v38, v39, v40);
      v142 = sub_AE5020(v182, v141);
      v143 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v25 + 16) + 200LL))(*(_QWORD *)(v25 + 16));
      if ( (!(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v143 + 544LL))(v143, v25)
         || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v143 + 536LL))(v143, v25))
        && *(_BYTE *)(v182 + 17)
        && v142 > *(_BYTE *)(v182 + 16) )
      {
        v142 = *(_BYTE *)(v182 + 16);
      }
      if ( v142 > a10 )
      {
        v144 = *(_DWORD *)(v191 + 96);
        v145 = *(_QWORD *)(v188 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v188 + 32) + v144);
        if ( v142 > *(_BYTE *)(v145 + 16) )
        {
          *(_BYTE *)(v145 + 16) = v142;
          if ( (*(_BYTE *)(*(_QWORD *)(v188 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v188 + 32) + v144) + 20) & 0xFD) == 0 )
            sub_2E76F70(v188, v142);
        }
        v198 = v142;
      }
    }
    v197 = 0;
    v244[1] = 0;
    v41 = _mm_loadu_si128(a19 + 1);
    v42 = _mm_loadu_si128(a19);
    v244[0] = 0;
    v245 = v41;
    if ( a16 )
    {
      if ( (a16 & 4) == 0 )
      {
        v197 = (a16 & 0xFFFFFFFFFFFFFFF8LL) != 0 && a20 != 0;
        if ( v197 )
        {
          v252 = (_OWORD *)(a16 & 0xFFFFFFFFFFFFFFF8LL);
          v43 = 0xBFFFFFFFFFFFFFFELL;
          v254[0] = v42;
          if ( a9 <= 0x3FFFFFFFFFFFFFFBLL )
            v43 = a9;
          v44 = *a20;
          v254[1] = v41;
          v253 = v43;
          v197 = (unsigned __int8)sub_CF4FA0(v44, (__int64)&v252, (__int64)(a20 + 1), 0) == 0;
        }
      }
    }
    v249 = v251;
    v192 = 4 * (a11 != 0);
    v246 = v248;
    v247 = 0x1000000000LL;
    v250 = 0x1000000000LL;
    v252 = v254;
    v253 = 0x2000000000LL;
    v45 = v229;
    v46 = (__int64)(v230 - v229) >> 4;
    if ( (_DWORD)v46 )
    {
      v47 = v209.m128i_i8[0];
      v48 = 0;
      v209.m128i_i64[0] = 0;
      v185 = 16LL * (unsigned int)(v46 - 1);
      v175 = 1LL << v47;
      v206 = (__int64)v23;
      while ( 1 )
      {
        v217 = _mm_loadu_si128((const __m128i *)(v45 + v209.m128i_i64[0]));
        if ( v217.m128i_i16[0] )
        {
          if ( v217.m128i_i16[0] == 1 || (unsigned __int16)(v217.m128i_i16[0] - 504) <= 7u )
LABEL_211:
            BUG();
          v86 = 16LL * (v217.m128i_u16[0] - 1);
          v50 = *(_QWORD *)&byte_444C4A0[v86];
          v51 = byte_444C4A0[v86 + 8];
        }
        else
        {
          v224 = sub_3007260((__int64)&v217);
          v225 = v49;
          v50 = v224;
          v51 = v225;
        }
        v241 = v50;
        LOBYTE(v242) = v51;
        v52 = (unsigned __int64)sub_CA1930(&v241) >> 3;
        v53 = (unsigned int)v52;
        v189 = v52;
        v54 = v48 + a9 - (unsigned int)v52;
        if ( v53 > a9 )
          v48 = v54;
        if ( !v202 )
        {
LABEL_41:
          v196 = 4 * (a11 != 0);
LABEL_42:
          v56 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v203 + 592LL);
          if ( v56 == sub_2D56A50 )
          {
            HIWORD(v57) = 0;
            sub_2FE6CC0((__int64)&v241, (__int64)v203, (__int64)v183, v217.m128i_i64[0], v217.m128i_i64[1]);
            LOWORD(v57) = v242;
            v176 = v243;
          }
          else
          {
            v158 = ((__int64 (__fastcall *)(_DWORD *, __int64 *, _QWORD, __int64, double, double))v56)(
                     v203,
                     v183,
                     v217.m128i_u32[0],
                     v217.m128i_i64[1],
                     *(double *)v41.m128i_i64,
                     *(double *)v42.m128i_i64);
            v176 = v159;
            v57 = v158;
          }
          v58 = a16 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (a16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (a16 & 4) != 0 )
            {
              v242 = v48 + a17;
              BYTE4(v243) = BYTE4(a18);
              v241 = v58 | 4;
              LODWORD(v243) = *(_DWORD *)(v58 + 12);
            }
            else
            {
              v241 = a16 & 0xFFFFFFFFFFFFFFF8LL;
              v242 = v48 + a17;
              BYTE4(v243) = BYTE4(a18);
              v136 = *(_QWORD *)(v58 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v136 + 8) - 17 <= 1 )
                v136 = **(_QWORD **)(v136 + 16);
              LODWORD(v243) = *(_DWORD *)(v136 + 8) >> 8;
            }
          }
          else
          {
            v242 = v48 + a17;
            v241 = 0;
            LODWORD(v243) = a18;
            BYTE4(v243) = 0;
          }
          v59 = sub_2EAC1F0((__int64 *)&v241, v189, (__int64)v183, v182);
          v60 = -1;
          v61 = v192 | 0x10;
          if ( !v59 )
            v61 = 4 * (a11 != 0);
          if ( v197 )
            v61 |= 0x20u;
          v62 = v48 | v175;
          if ( (v62 & -(__int64)v62) != 0 )
          {
            _BitScanReverse64(&v63, v62 & -(__int64)v62);
            v60 = 63 - (v63 ^ 0x3F);
          }
          LOBYTE(v64) = v60;
          HIBYTE(v64) = 1;
          v199 = v64;
          v65 = a16 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (a16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (a16 & 4) != 0 )
            {
              *((_QWORD *)&v237 + 1) = v48 + a17;
              BYTE4(v238) = BYTE4(a18);
              *(_QWORD *)&v237 = v65 | 4;
              LODWORD(v238) = *(_DWORD *)(v65 + 12);
            }
            else
            {
              *(_QWORD *)&v237 = a16 & 0xFFFFFFFFFFFFFFF8LL;
              v137 = *(_QWORD *)(v65 + 8);
              *((_QWORD *)&v237 + 1) = v48 + a17;
              v138 = *(unsigned __int8 *)(v137 + 8);
              BYTE4(v238) = BYTE4(a18);
              if ( (unsigned int)(v138 - 17) <= 1 )
                v137 = **(_QWORD **)(v137 + 16);
              LODWORD(v238) = *(_DWORD *)(v137 + 8) >> 8;
            }
          }
          else
          {
            *((_QWORD *)&v237 + 1) = v48 + a17;
            *(_QWORD *)&v237 = 0;
            LODWORD(v238) = a18;
            BYTE4(v238) = 0;
          }
          LOBYTE(v221) = 0;
          v220 = v48;
          v66 = sub_3409320((_QWORD *)v206, a7, a8, v48, 0, a2, v41, 0);
          v210 = sub_33F1DB0(
                   (__int64 *)v206,
                   1,
                   a2,
                   v57,
                   v176,
                   v199,
                   v187,
                   (__int64)v66,
                   v67,
                   v237,
                   v238,
                   v217.m128i_i64[0],
                   v217.m128i_i64[1],
                   v61,
                   (__int64)v244);
          v69 = (unsigned __int64)v210;
          v70 = (unsigned int)v247;
          v72 = (unsigned int)v71;
          v211 = v71;
          v73 = v181 & 0xFFFFFFFF00000000LL | 1;
          v74 = (unsigned int)v247 + 1LL;
          v181 = v73;
          if ( v74 > HIDWORD(v247) )
          {
            v190 = v73;
            sub_C8D5F0((__int64)&v246, v248, v74, 0x10u, v73, v68);
            v70 = (unsigned int)v247;
            v73 = v190;
          }
          v75 = (__m128i **)&v246[16 * v70];
          *v75 = v210;
          v75[1] = (__m128i *)v73;
          LODWORD(v247) = v247 + 1;
          v76 = a13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (a13 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (a13 & 4) != 0 )
            {
              *((_QWORD *)&v239 + 1) = v48 + a14;
              BYTE4(v240) = BYTE4(a15);
              *(_QWORD *)&v239 = v76 | 4;
              LODWORD(v240) = *(_DWORD *)(v76 + 12);
            }
            else
            {
              *(_QWORD *)&v239 = a13 & 0xFFFFFFFFFFFFFFF8LL;
              v139 = *(_QWORD *)(v76 + 8);
              *((_QWORD *)&v239 + 1) = v48 + a14;
              v140 = *(unsigned __int8 *)(v139 + 8);
              BYTE4(v240) = BYTE4(a15);
              if ( (unsigned int)(v140 - 17) <= 1 )
                v139 = **(_QWORD **)(v139 + 16);
              LODWORD(v240) = *(_DWORD *)(v139 + 8) >> 8;
            }
          }
          else
          {
            BYTE4(v240) = 0;
            *(_QWORD *)&v239 = 0;
            *((_QWORD *)&v239 + 1) = v48 + a14;
            LODWORD(v240) = a15;
          }
          LOBYTE(v223) = 0;
          v222 = v48;
          v77 = sub_3409320((_QWORD *)v206, a5, a6, v48, 0, a2, v41, 0);
          v80 = sub_33F5040(
                  (_QWORD *)v206,
                  v187,
                  *((unsigned __int64 *)&v187 + 1),
                  a2,
                  v69,
                  v72,
                  (unsigned __int64)v77,
                  v78,
                  v239,
                  v240,
                  v217.m128i_i64[0],
                  v217.m128i_u64[1],
                  v198,
                  v196,
                  (__int64)v244);
          v81 = (unsigned int)v250;
          v83 = v82 | v193 & 0xFFFFFFFF00000000LL;
          v84 = (unsigned int)v250 + 1LL;
          if ( v84 > HIDWORD(v250) )
          {
            sub_C8D5F0((__int64)&v249, v251, v84, 0x10u, v79, v40);
            v81 = (unsigned int)v250;
          }
          v85 = (__m128i **)&v249[16 * v81];
          *v85 = v80;
          v85[1] = (__m128i *)v83;
          LODWORD(v250) = v250 + 1;
          goto LABEL_61;
        }
        if ( !v200 )
        {
          if ( v217.m128i_i16[0] )
          {
            v55 = v217.m128i_i16[0] - 17;
            if ( (unsigned __int16)(v217.m128i_i16[0] - 2) > 7u
              && v55 > 0x6Cu
              && (unsigned __int16)(v217.m128i_i16[0] - 176) > 0x1Fu
              || v55 <= 0xD3u )
            {
              goto LABEL_41;
            }
          }
          else if ( !sub_3007070((__int64)&v217) || sub_30070B0((__int64)&v217) )
          {
            goto LABEL_41;
          }
        }
        if ( v234 <= v48 )
        {
          v226 = _mm_load_si128(&v217);
        }
        else
        {
          v87 = v234 - v48;
          v88 = v232;
          v226 = _mm_load_si128(&v217);
          v171 = v48 + v233;
          if ( v232 )
          {
            if ( v226.m128i_i16[0] )
            {
              if ( v226.m128i_i16[0] == 1 || (unsigned __int16)(v226.m128i_i16[0] - 504) <= 7u )
                goto LABEL_211;
              v90 = 16LL * (v226.m128i_u16[0] - 1);
              v89 = *(_QWORD *)&byte_444C4A0[v90];
              LOBYTE(v90) = byte_444C4A0[v90 + 8];
            }
            else
            {
              v89 = sub_3007260((__int64)&v226);
              v227 = v89;
              v228 = v90;
            }
            v241 = v89;
            LOBYTE(v242) = v90;
            v91 = sub_CA1930(&v241);
            v92 = v87;
            LODWORD(v242) = v91;
            v93 = v91 >> 3;
            if ( v91 >> 3 < v87 )
              v92 = v91 >> 3;
            if ( v91 > 0x40 )
            {
              v174 = v91 >> 3;
              v179 = v92;
              sub_C43690((__int64)&v241, 0, 0);
              v93 = v174;
              v92 = v179;
            }
            else
            {
              v241 = 0;
            }
            v172 = v93;
            v177 = v92;
            v94 = (_BYTE *)sub_2E79000(*(__int64 **)(v206 + 40));
            v97 = v177;
            v98 = v172;
            if ( *v94 )
            {
              if ( v177 )
              {
                v178 = v48;
                v173 = v53;
                v123 = 8 * v98 - 8;
                v124 = v171;
                v125 = v171 + v97;
                do
                {
                  v126 = (unsigned __int64)(unsigned __int8)sub_AC5320(v88, v124) << v123;
                  if ( (unsigned int)v242 > 0x40 )
                  {
                    v97 = v241;
                    *(_QWORD *)v241 |= v126;
                  }
                  else
                  {
                    v97 = 0;
                    v127 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v242) & (v241 | v126);
                    if ( !(_DWORD)v242 )
                      v127 = 0;
                    v241 = v127;
                  }
                  ++v124;
                  v123 -= 8;
                }
                while ( v125 != v124 );
                goto LABEL_84;
              }
            }
            else if ( v177 )
            {
              v178 = v48;
              v99 = v171;
              v173 = v53;
              v100 = 0;
              v101 = v97 + v171;
              do
              {
                while ( 1 )
                {
                  v102 = (unsigned __int64)(unsigned __int8)sub_AC5320(v88, v99) << v100;
                  if ( (unsigned int)v242 <= 0x40 )
                    break;
                  v97 = v241;
                  ++v99;
                  v100 += 8;
                  *(_QWORD *)v241 |= v102;
                  if ( v99 == v101 )
                    goto LABEL_84;
                }
                v97 = 0;
                v103 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v242) & (v241 | v102);
                if ( !(_DWORD)v242 )
                  v103 = 0;
                ++v99;
                v100 += 8;
                v241 = v103;
              }
              while ( v99 != v101 );
LABEL_84:
              v48 = v178;
              v53 = v173;
            }
            v104 = sub_3007410((__int64)&v226, *(__int64 **)(v206 + 64), v95, v97, v98, v96);
            v105 = *(__int64 (**)())(*(_QWORD *)v203 + 1672LL);
            if ( v105 == sub_2FE35A0
              || !((unsigned __int8 (__fastcall *)(_DWORD *, unsigned __int64 *, __int64, double, double))v105)(
                    v203,
                    &v241,
                    v104,
                    *(double *)v41.m128i_i64,
                    *(double *)v42.m128i_i64) )
            {
              v106 = 0;
              v107 = 0;
            }
            else
            {
              v107 = sub_34007B0(v206, (__int64)&v241, a2, v226.m128i_u32[0], v226.m128i_i64[1], 0, v41, 0);
              v106 = v160;
            }
            if ( (unsigned int)v242 > 0x40 && v241 )
              j_j___libc_free_0_0(v241);
            goto LABEL_96;
          }
        }
        if ( v226.m128i_i16[0] )
        {
          v108 = v226.m128i_i16[0] - 17;
          if ( (unsigned __int16)(v226.m128i_i16[0] - 2) > 7u
            && v108 > 0x6Cu
            && (unsigned __int16)(v226.m128i_i16[0] - 176) > 0x1Fu )
          {
            if ( v108 <= 0xD3u )
            {
              LOWORD(v241) = v226.m128i_i16[0];
              v128 = sub_30369B0((unsigned __int16 *)&v241);
              v130 = 0;
            }
            else
            {
              if ( v226.m128i_i16[0] == 1 || (unsigned __int16)(v226.m128i_i16[0] - 504) <= 7u )
                goto LABEL_211;
              v146 = 16LL * (v226.m128i_u16[0] - 1);
              v147 = byte_444C4A0[v146 + 8];
              v148 = *(_QWORD *)&byte_444C4A0[v146];
              LOBYTE(v242) = v147;
              v241 = v148;
              v149 = sub_CA1930(&v241);
              v128 = 2;
              if ( v149 != 1 )
              {
                v128 = 3;
                if ( v149 != 2 )
                {
                  v128 = 4;
                  if ( v149 != 4 )
                  {
                    v128 = 5;
                    if ( v149 != 8 )
                    {
                      v128 = 6;
                      if ( v149 != 16 )
                      {
                        v128 = 7;
                        if ( v149 != 32 )
                        {
                          v128 = 8;
                          if ( v149 != 64 )
                            v128 = 9 * (v149 == 128);
                        }
                      }
                    }
                  }
                }
              }
              v130 = 0;
            }
            goto LABEL_172;
          }
        }
        else if ( !sub_3007070((__int64)&v226) )
        {
          if ( sub_30070B0((__int64)&v226) )
            v128 = sub_300A990((unsigned __int16 *)&v226, v209.m128i_i64[0]);
          else
            v128 = sub_30072B0((__int64)&v226);
          v130 = v129;
LABEL_172:
          v150 = v170;
          LOWORD(v150) = v128;
          v170 = v150;
          sub_3400BD0(v206, 0, a2, (unsigned int)v150, v130, 0, v41, 0);
          v107 = sub_33FAF80(v206, 234, a2, v226.m128i_u32[0], v226.m128i_i64[1], v151, v41);
          v106 = v152;
          goto LABEL_96;
        }
        v107 = sub_3400BD0(v206, 0, a2, v226.m128i_u32[0], v226.m128i_i64[1], 0, v41, 0);
        v106 = v109;
LABEL_96:
        v196 = 4 * (a11 != 0);
        if ( !v107 )
          goto LABEL_42;
        v110 = a13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (a13 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (a13 & 4) != 0 )
          {
            *((_QWORD *)&v235 + 1) = v48 + a14;
            BYTE4(v236) = BYTE4(a15);
            *(_QWORD *)&v235 = v110 | 4;
            LODWORD(v236) = *(_DWORD *)(v110 + 12);
          }
          else
          {
            *((_QWORD *)&v235 + 1) = v48 + a14;
            v162 = *(_QWORD *)(v110 + 8);
            *(_QWORD *)&v235 = a13 & 0xFFFFFFFFFFFFFFF8LL;
            v163 = *(unsigned __int8 *)(v162 + 8);
            BYTE4(v236) = BYTE4(a15);
            if ( (unsigned int)(v163 - 17) <= 1 )
              v162 = **(_QWORD **)(v162 + 16);
            LODWORD(v236) = *(_DWORD *)(v162 + 8) >> 8;
          }
        }
        else
        {
          *((_QWORD *)&v235 + 1) = v48 + a14;
          *(_QWORD *)&v235 = 0;
          LODWORD(v236) = a15;
          BYTE4(v236) = 0;
        }
        LOBYTE(v219) = 0;
        v218 = v48;
        v111 = sub_3409320((_QWORD *)v206, a5, a6, v48, 0, a2, v41, 0);
        v113 = sub_33F4560(
                 (_QWORD *)v206,
                 v187,
                 *((unsigned __int64 *)&v187 + 1),
                 a2,
                 (unsigned __int64)v107,
                 v106,
                 (unsigned __int64)v111,
                 v112,
                 v235,
                 v236,
                 v198,
                 v192,
                 (__int64)v244);
        v114 = (unsigned int)v253;
        v116 = v115;
        v193 = v115;
        v117 = (unsigned int)v253 + 1LL;
        if ( v117 > HIDWORD(v253) )
        {
          v180 = v116;
          sub_C8D5F0((__int64)&v252, v254, v117, 0x10u, v116, v40);
          v114 = (unsigned int)v253;
          v116 = v180;
        }
        v118 = (__m128i **)&v252[v114];
        *v118 = v113;
        v118[1] = (__m128i *)v116;
        LODWORD(v253) = v253 + 1;
        if ( !v113 )
          goto LABEL_42;
LABEL_61:
        a9 -= v53;
        v48 += v53;
        if ( v185 == v209.m128i_i64[0] )
        {
          v23 = (_QWORD *)v206;
          break;
        }
        v45 = v229;
        v209.m128i_i64[0] += 16;
      }
    }
    v119 = qword_50395A8;
    if ( !(_DWORD)qword_50395A8 )
    {
      v120 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v203 + 800LL);
      if ( v120 == sub_2FE32A0 )
        v119 = v203[134246];
      else
        v119 = ((__int64 (__fastcall *)(_DWORD *, double, double))v120)(
                 v203,
                 *(double *)v41.m128i_i64,
                 *(double *)v42.m128i_i64);
    }
    v121 = (unsigned int)v250;
    if ( !(_DWORD)v250 )
      goto LABEL_114;
    if ( v119 <= 1 || !(_BYTE)qword_5039688 )
    {
      v122 = (unsigned int)v253;
      v131 = 0;
      v132 = 16LL * (unsigned int)v250;
      do
      {
        v135 = _mm_loadu_si128((const __m128i *)&v246[v131]);
        if ( v122 + 1 > (unsigned __int64)HIDWORD(v253) )
        {
          v209 = v135;
          sub_C8D5F0((__int64)&v252, v254, v122 + 1, 0x10u, v121, v40);
          v122 = (unsigned int)v253;
          v135 = _mm_load_si128(&v209);
        }
        v252[v122] = v135;
        LODWORD(v253) = v253 + 1;
        v133 = (unsigned int)v253;
        v134 = _mm_loadu_si128((const __m128i *)&v249[v131]);
        if ( (unsigned __int64)(unsigned int)v253 + 1 > HIDWORD(v253) )
        {
          v209 = v134;
          sub_C8D5F0((__int64)&v252, v254, (unsigned int)v253 + 1LL, 0x10u, v121, v40);
          v133 = (unsigned int)v253;
          v134 = _mm_load_si128(&v209);
        }
        v131 += 16;
        v252[v133] = v134;
        v122 = (unsigned int)(v253 + 1);
        LODWORD(v253) = v253 + 1;
      }
      while ( v132 != v131 );
      goto LABEL_115;
    }
    if ( v119 >= (unsigned int)v250 )
    {
      sub_3402B60(v23, a2, (__int64)&v252, 0, v250, &v246, &v249);
    }
    else
    {
      v164 = v250 - v119;
      v165 = 0;
      v207 = (unsigned int)v250 / v119;
      v204 = (unsigned int)v250 % v119;
      v209.m128i_i64[0] = (__int64)&v249;
      do
      {
        v166 = v164;
        v167 = v164 + v119;
        ++v165;
        v164 -= v119;
        sub_3402B60(v23, a2, (__int64)&v252, v166, v167, &v246, v209.m128i_i64[0]);
        v40 = v169;
      }
      while ( v207 > v165 );
      if ( !v204 )
      {
LABEL_114:
        LODWORD(v122) = v253;
LABEL_115:
        *((_QWORD *)&v168 + 1) = (unsigned int)v122;
        *(_QWORD *)&v168 = v252;
        v20 = sub_33FC220(v23, 2, a2, 1, 0, v40, v168);
        if ( v252 != v254 )
          _libc_free((unsigned __int64)v252);
        if ( v249 != v251 )
          _libc_free((unsigned __int64)v249);
        if ( v246 != v248 )
          _libc_free((unsigned __int64)v246);
        goto LABEL_17;
      }
      sub_3402B60(v23, a2, (__int64)&v252, 0, v204, &v246, v209.m128i_i64[0]);
    }
    LODWORD(v122) = v253;
    goto LABEL_115;
  }
  v20 = 0;
LABEL_17:
  if ( v229 )
    j_j___libc_free_0(v229);
  return v20;
}
