// Function: sub_93CB50
// Address: 0x93cb50
//
__int64 __fastcall sub_93CB50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  char i; // al
  unsigned __int64 v10; // r12
  char j; // al
  unsigned __int64 v12; // rax
  __m128i *v13; // rax
  __int64 *v14; // r14
  __int64 *v15; // r13
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __m128i *v18; // rcx
  __m128i *v19; // rdx
  __m128i *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r12
  int v23; // ebx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r14
  char v30; // al
  unsigned __int16 v31; // dx
  __int64 k; // rax
  __int64 *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rbx
  __m128i *v40; // r13
  unsigned int v41; // eax
  unsigned __int64 v42; // r14
  __int32 v43; // edx
  unsigned __int32 v44; // r12d
  int v45; // r8d
  __int64 v46; // rax
  char v47; // al
  __int16 v48; // cx
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 v51; // r12
  unsigned int *v52; // r14
  unsigned int *v53; // r15
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rsi
  _QWORD *v61; // rdx
  _QWORD *v62; // rcx
  __int64 v63; // rcx
  int *v64; // rax
  unsigned int **v65; // r15
  int v66; // r13d
  int v67; // r13d
  int v68; // r13d
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r14
  __int64 v72; // rdi
  __int64 (__fastcall *v73)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v74; // r9
  __int64 v75; // rax
  unsigned __int64 v76; // rcx
  unsigned __int64 v77; // rdx
  __int64 v78; // r14
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // rax
  unsigned __int64 v82; // rsi
  __int64 v83; // rax
  _BYTE *v84; // r14
  __int64 v85; // r10
  __int64 v86; // rdi
  __int64 (__fastcall *v87)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v88; // rax
  unsigned __int64 v89; // rax
  __int64 v90; // r12
  __int64 v91; // rcx
  __int32 v92; // r13d
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rbx
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  __int64 v100; // rbx
  __int64 v101; // r15
  __int64 v102; // rdx
  __int64 v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rdx
  unsigned int v106; // eax
  __int64 v107; // rcx
  unsigned __int64 v108; // r15
  int v109; // ecx
  __int64 v110; // rax
  unsigned __int8 v111; // al
  int v112; // r8d
  __int64 v113; // rax
  int v114; // r9d
  __int64 v115; // r13
  unsigned int *v116; // r12
  unsigned int *v117; // rbx
  __int64 v118; // rdx
  __int64 v119; // rsi
  __int64 v120; // rax
  unsigned __int64 v121; // rdx
  unsigned __int64 v122; // rax
  __int64 v123; // rbx
  __m128i *v124; // r15
  __int64 v125; // rax
  unsigned __int64 v126; // r15
  __int64 v127; // rax
  __int8 *v128; // rax
  __int8 *m; // rdx
  __int64 v130; // r12
  __int64 v131; // r13
  __m128i *v132; // rdx
  __m128i *v133; // rbx
  __m128i *v134; // rdi
  __int64 v135; // r14
  __int64 v136; // rax
  unsigned __int64 v137; // rdx
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rcx
  __int64 v142; // r13
  __int64 v143; // rax
  unsigned __int64 v144; // rdx
  __int64 v145; // r14
  __int64 v146; // rax
  unsigned __int64 v147; // rdx
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // r13
  __int64 v153; // rax
  unsigned __int64 v154; // rdx
  __int64 v155; // r14
  __int64 v156; // rax
  unsigned __int64 v157; // rdx
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 v162; // r13
  __int64 v163; // rax
  unsigned __int64 v164; // rdx
  __int64 v165; // rcx
  __m128i v166; // xmm2
  char v168; // al
  const char *v169; // rsi
  __int64 v170; // rcx
  unsigned __int64 v171; // rax
  __int64 v172; // r15
  __int64 v173; // r13
  __int64 v174; // rdx
  unsigned int v175; // r14d
  __int64 v176; // rax
  __int64 v177; // r14
  unsigned int *v178; // r14
  unsigned int *v179; // rbx
  __int64 v180; // rdx
  char v181; // al
  __int64 v182; // r9
  __int64 v183; // rdx
  unsigned int v184; // r14d
  __int64 v185; // rax
  __int64 v186; // r14
  unsigned int *v187; // r14
  unsigned int *v188; // rbx
  __int64 v189; // rdx
  __int64 v190; // rsi
  _DWORD *v191; // rdi
  size_t v192; // rdx
  __int64 v193; // r15
  unsigned __int64 v194; // r14
  int v195; // ecx
  __int64 v196; // rax
  unsigned __int8 v197; // al
  int v198; // r8d
  __int64 v199; // rax
  int v200; // r9d
  __int64 v201; // r13
  unsigned int *v202; // r12
  unsigned int *v203; // rbx
  __int64 v204; // rdx
  __int64 v205; // rsi
  int v206; // esi
  int v207; // esi
  int v208; // eax
  __int64 v209; // rax
  __int64 v210; // rax
  __int64 v211; // rax
  unsigned __int64 v212; // rax
  unsigned __int64 v213; // rax
  __int64 v214; // [rsp-8h] [rbp-558h]
  unsigned __int64 v215; // [rsp+8h] [rbp-548h]
  __int64 v216; // [rsp+10h] [rbp-540h]
  int v217; // [rsp+20h] [rbp-530h]
  int v218; // [rsp+20h] [rbp-530h]
  __int64 v219; // [rsp+20h] [rbp-530h]
  __int64 v220; // [rsp+20h] [rbp-530h]
  __int64 v221; // [rsp+20h] [rbp-530h]
  __int64 v222; // [rsp+20h] [rbp-530h]
  unsigned int v223; // [rsp+28h] [rbp-528h]
  unsigned __int64 v225; // [rsp+30h] [rbp-520h]
  __int64 v226; // [rsp+30h] [rbp-520h]
  __int64 v227; // [rsp+30h] [rbp-520h]
  __int64 v228; // [rsp+30h] [rbp-520h]
  __int64 v230; // [rsp+40h] [rbp-510h]
  unsigned __int64 v231; // [rsp+40h] [rbp-510h]
  __int16 dest; // [rsp+48h] [rbp-508h]
  unsigned int *v233; // [rsp+50h] [rbp-500h]
  __int64 *v234; // [rsp+50h] [rbp-500h]
  __m128i *v235; // [rsp+50h] [rbp-500h]
  unsigned __int64 v237; // [rsp+90h] [rbp-4C0h]
  __int64 v239; // [rsp+A0h] [rbp-4B0h]
  __int64 v240; // [rsp+A0h] [rbp-4B0h]
  __m128i *v241; // [rsp+A8h] [rbp-4A8h]
  int v242; // [rsp+A8h] [rbp-4A8h]
  int v243; // [rsp+A8h] [rbp-4A8h]
  __int64 v244; // [rsp+B0h] [rbp-4A0h]
  int v245; // [rsp+B0h] [rbp-4A0h]
  int v246; // [rsp+B0h] [rbp-4A0h]
  __int64 v247; // [rsp+B0h] [rbp-4A0h]
  int v248; // [rsp+B0h] [rbp-4A0h]
  int v249; // [rsp+B0h] [rbp-4A0h]
  __int64 v250; // [rsp+B8h] [rbp-498h]
  char v251[4]; // [rsp+CCh] [rbp-484h] BYREF
  _BYTE *v252; // [rsp+D0h] [rbp-480h] BYREF
  __int64 v253; // [rsp+D8h] [rbp-478h]
  _BYTE v254[16]; // [rsp+E0h] [rbp-470h] BYREF
  unsigned __int8 *v255; // [rsp+F0h] [rbp-460h] BYREF
  __int64 v256; // [rsp+F8h] [rbp-458h]
  unsigned __int64 v257; // [rsp+100h] [rbp-450h]
  _BYTE v258[24]; // [rsp+108h] [rbp-448h] BYREF
  unsigned __int8 *v259; // [rsp+120h] [rbp-430h] BYREF
  __int64 v260; // [rsp+128h] [rbp-428h]
  unsigned __int64 v261; // [rsp+130h] [rbp-420h]
  _BYTE v262[24]; // [rsp+138h] [rbp-418h] BYREF
  unsigned __int8 *v263; // [rsp+150h] [rbp-400h] BYREF
  __int64 v264; // [rsp+158h] [rbp-3F8h]
  unsigned __int64 v265; // [rsp+160h] [rbp-3F0h]
  _BYTE v266[24]; // [rsp+168h] [rbp-3E8h] BYREF
  _BYTE v267[32]; // [rsp+180h] [rbp-3D0h] BYREF
  __int16 v268; // [rsp+1A0h] [rbp-3B0h]
  __m128i v269; // [rsp+1B0h] [rbp-3A0h] BYREF
  _QWORD v270[2]; // [rsp+1C0h] [rbp-390h] BYREF
  __int16 v271; // [rsp+1D0h] [rbp-380h]
  void *src; // [rsp+1E0h] [rbp-370h] BYREF
  __int64 v273; // [rsp+1E8h] [rbp-368h]
  _BYTE v274[128]; // [rsp+1F0h] [rbp-360h] BYREF
  unsigned __int64 *v275; // [rsp+270h] [rbp-2E0h] BYREF
  __int64 v276; // [rsp+278h] [rbp-2D8h]
  _BYTE v277[128]; // [rsp+280h] [rbp-2D0h] BYREF
  __m128i v278; // [rsp+300h] [rbp-250h] BYREF
  _DWORD v279[4]; // [rsp+310h] [rbp-240h] BYREF
  __int16 v280; // [rsp+320h] [rbp-230h]
  __m128i *v281; // [rsp+390h] [rbp-1C0h] BYREF
  __int64 v282; // [rsp+398h] [rbp-1B8h]
  _BYTE v283[432]; // [rsp+3A0h] [rbp-1B0h] BYREF

  v250 = a2;
  v255 = v258;
  src = v274;
  v259 = v262;
  v263 = v266;
  v273 = 0x1000000000LL;
  v281 = (__m128i *)v283;
  v282 = 0x1000000000LL;
  v275 = (unsigned __int64 *)v277;
  v256 = 0;
  v257 = 16;
  v260 = 0;
  v261 = 16;
  v264 = 0;
  v265 = 16;
  v276 = 0x1000000000LL;
  v5 = *(__int64 **)(a3 + 72);
  v244 = (__int64)v5;
  v6 = sub_72B0F0((__int64)v5, 0);
  v7 = *v5;
  v8 = v6;
  for ( i = *(_BYTE *)(v7 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( i != 6 )
    sub_91B8A0("Expected pointer to function!", (_DWORD *)(a3 + 36), 1);
  v10 = *(_QWORD *)(v7 + 160);
  for ( j = *(_BYTE *)(v10 + 140); j == 12; j = *(_BYTE *)(v10 + 140) )
    v10 = *(_QWORD *)(v10 + 160);
  if ( j != 7 )
    sub_91B8A0("unexpected: Callee does not have routine type!", (_DWORD *)(a3 + 36), 1);
  v12 = *(_QWORD *)(v10 + 160);
  v237 = v12;
  if ( *(char *)(v12 + 142) >= 0 && (v12 = *(_QWORD *)(v10 + 160), *(_BYTE *)(v237 + 140) == 12) )
    v223 = sub_8D4AB0(v237);
  else
    v223 = *(_DWORD *)(v12 + 136);
  if ( *(_BYTE *)(v244 + 24) != 20 )
  {
    if ( !v8 )
      goto LABEL_14;
    goto LABEL_37;
  }
  if ( (*(_BYTE *)(v8 + 199) & 2) != 0 )
  {
    sub_955A70(&v269, a2, a3);
    if ( !sub_91B770(v237) )
    {
      v166 = _mm_loadu_si128(&v269);
      *(_QWORD *)(a1 + 16) = v270[0];
      *(__m128i *)a1 = v166;
LABEL_183:
      v33 = (__int64 *)v275;
      goto LABEL_184;
    }
    v193 = v269.m128i_i64[0];
    v194 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v195 = 1;
      if ( (a4 & 4) != 0 )
      {
LABEL_239:
        if ( a5 )
        {
          _BitScanReverse64(&v213, a5);
          v198 = (unsigned __int8)(63 - (v213 ^ 0x3F));
        }
        else
        {
          v248 = v195;
          v196 = sub_AA4E30(*(_QWORD *)(a2 + 96));
          v197 = sub_AE5020(v196, *(_QWORD *)(v193 + 8));
          v195 = v248;
          v198 = v197;
        }
        v243 = v198;
        v280 = 257;
        v249 = v195;
        v199 = sub_BD2C40(80, unk_3F10A10);
        v201 = v199;
        if ( v199 )
          sub_B4D3C0(v199, v193, v194, v249, v243, v200, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v201,
          &v278,
          *(_QWORD *)(a2 + 104),
          *(_QWORD *)(a2 + 112));
        v202 = *(unsigned int **)(a2 + 48);
        v203 = &v202[4 * *(unsigned int *)(a2 + 56)];
        while ( v203 != v202 )
        {
          v204 = *((_QWORD *)v202 + 1);
          v205 = *v202;
          v202 += 4;
          sub_B99FD0(v201, v205, v204);
        }
        a2 = a5;
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v194;
        *(_DWORD *)(a1 + 8) = 1;
        *(_DWORD *)(a1 + 16) = a5;
        goto LABEL_183;
      }
    }
    else
    {
      v278.m128i_i64[0] = (__int64)"agg.tmp";
      v280 = 259;
      v194 = sub_921D70(a2, v237, (__int64)&v278, v165);
      a5 = v223;
    }
    v195 = unk_4D0463C;
    if ( unk_4D0463C )
      v195 = sub_90AA40(*(_QWORD *)(a2 + 32), v194);
    goto LABEL_239;
  }
LABEL_37:
  sub_5E3780(v8, (__int64)v251);
  if ( !*(_BYTE *)(v8 + 174) )
  {
    v31 = *(_WORD *)(v8 + 176);
    if ( v31 )
    {
      for ( k = *(_QWORD *)(v8 + 152); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(k + 168) + 16LL) & 1) != 0
        && (((v31 - 10219) & 0xFFF7) == 0 || v31 == 10214 || v31 == 15752) )
      {
        sub_939370(a1, a2, v31, (const char *)v244);
        v33 = (__int64 *)v275;
        goto LABEL_184;
      }
    }
  }
LABEL_14:
  v13 = sub_92F410(a2, v244);
  v14 = *(__int64 **)(v244 + 16);
  v239 = (__int64)v13;
  if ( v14 )
  {
    v15 = **(__int64 ***)(v10 + 168);
    if ( (*(_BYTE *)(a3 + 60) & 2) != 0 )
    {
      v123 = 0;
      v270[0] = 0;
      v124 = &v269;
      v269.m128i_i64[1] = (__int64)&v269;
      v269.m128i_i64[0] = (__int64)&v269;
      while ( 1 )
      {
        v125 = sub_22077B0(24);
        *(_QWORD *)(v125 + 16) = v14;
        sub_2208C80(v125, v124);
        ++v270[0];
        v14 = (__int64 *)v14[2];
        if ( !v14 )
          break;
        v124 = (__m128i *)v269.m128i_i64[0];
        ++v123;
      }
      v126 = v123 + 1;
      v127 = (unsigned int)v282;
      if ( v123 + 1 != (unsigned int)v282 )
      {
        if ( v126 >= (unsigned int)v282 )
        {
          if ( v126 > HIDWORD(v282) )
          {
            sub_C8D5F0(&v281, v283, v126, 24);
            v127 = (unsigned int)v282;
          }
          v128 = &v281->m128i_i8[24 * v127];
          for ( m = &v281->m128i_i8[24 * v126]; m != v128; v128 += 24 )
          {
            if ( v128 )
            {
              v128[12] &= ~1u;
              *(_QWORD *)v128 = 0;
              *((_DWORD *)v128 + 2) = 0;
              *((_DWORD *)v128 + 4) = 0;
            }
          }
        }
        LODWORD(v282) = v123 + 1;
      }
      if ( (__m128i *)v269.m128i_i64[0] != &v269 )
      {
        v234 = v15;
        v231 = v10;
        v130 = v269.m128i_i64[0];
        v131 = 6 * v123;
        do
        {
          sub_921F50((__int64)&v278, a2, *(__int64 **)(v130 + 16), 0);
          v132 = v281;
          *(__m128i *)((char *)v281 + v131 * 4) = _mm_loadu_si128(&v278);
          v132[1].m128i_i32[v131] = v279[0];
          v130 = *(_QWORD *)v130;
          v131 -= 6;
        }
        while ( (__m128i *)v130 != &v269 );
        v133 = (__m128i *)v269.m128i_i64[0];
        v15 = v234;
        v10 = v231;
        while ( v133 != &v269 )
        {
          v134 = v133;
          v133 = (__m128i *)v133->m128i_i64[0];
          j_j___libc_free_0(v134, 24);
        }
      }
    }
    else
    {
      do
      {
        sub_921F50((__int64)&v278, a2, v14, 0);
        v16 = (unsigned int)v282;
        v17 = (unsigned int)v282 + 1LL;
        if ( v17 > HIDWORD(v282) )
        {
          if ( v281 > &v278 || (v235 = v281, &v278 >= (__m128i *)((char *)v281 + 24 * (unsigned int)v282)) )
          {
            sub_C8D5F0(&v281, v283, v17, 24);
            v18 = v281;
            v16 = (unsigned int)v282;
            v19 = &v278;
          }
          else
          {
            sub_C8D5F0(&v281, v283, v17, 24);
            v18 = v281;
            v16 = (unsigned int)v282;
            v19 = (__m128i *)((char *)v281 + (char *)&v278 - (char *)v235);
          }
        }
        else
        {
          v18 = v281;
          v19 = &v278;
        }
        v20 = (__m128i *)((char *)v18 + 24 * v16);
        *v20 = _mm_loadu_si128(v19);
        v21 = v19[1].m128i_i64[0];
        LODWORD(v282) = v282 + 1;
        v20[1].m128i_i64[0] = v21;
        v14 = (__int64 *)v14[2];
      }
      while ( v14 );
    }
    if ( *(_QWORD *)(v244 + 16) )
    {
      v225 = v10;
      v22 = *(__int64 **)(v244 + 16);
      do
      {
        if ( v15 )
        {
          v29 = v15[1];
          if ( dword_4F0690C )
          {
            v23 = (*((_DWORD *)v15 + 8) >> 13) & 1;
          }
          else
          {
            LOBYTE(v23) = 0;
            if ( (*(_BYTE *)(v29 + 140) & 0xFB) == 8 )
            {
              v30 = sub_8D4C10(v15[1], dword_4F077C4 != 2);
              v29 = v15[1];
              LOBYTE(v23) = (v30 & 4) != 0;
            }
          }
          goto LABEL_22;
        }
        while ( 1 )
        {
          v29 = *v22;
          LOBYTE(v23) = 0;
          if ( (*(_BYTE *)(*v22 + 140) & 0xFB) == 8 )
          {
            v168 = sub_8D4C10(*v22, dword_4F077C4 != 2);
            v29 = *v22;
            LOBYTE(v23) = (v168 & 4) != 0;
          }
          v15 = 0;
LABEL_22:
          v24 = v256;
          if ( v256 + 1 > v257 )
          {
            sub_C8D290(&v255, v258, v256 + 1, 1);
            v24 = v256;
          }
          v255[v24] = v23;
          v25 = (unsigned int)v273;
          ++v256;
          v26 = (unsigned int)v273 + 1LL;
          if ( v26 > HIDWORD(v273) )
          {
            sub_C8D5F0(&src, v274, v26, 8);
            v25 = (unsigned int)v273;
          }
          *((_QWORD *)src + v25) = v29;
          v27 = v260;
          LODWORD(v273) = v273 + 1;
          if ( v260 + 1 > v261 )
          {
            sub_C8D290(&v259, v262, v260 + 1, 1);
            v27 = v260;
          }
          v259[v27] = 0;
          v28 = v264;
          ++v260;
          if ( v264 + 1 > v265 )
          {
            sub_C8D290(&v263, v266, v264 + 1, 1);
            v28 = v264;
          }
          v263[v28] = 0;
          v22 = (__int64 *)v22[2];
          ++v264;
          if ( v15 )
            break;
          if ( !v22 )
            goto LABEL_203;
        }
        v15 = (__int64 *)*v15;
      }
      while ( v22 );
LABEL_203:
      v10 = v225;
    }
  }
  v230 = sub_9378E0((_QWORD **)(*(_QWORD *)(a2 + 32) + 8LL), v237, (__int64)&src, &v255, &v259, &v263);
  v216 = *(_QWORD *)(v230 + 16);
  if ( sub_938130(*(_QWORD *)(a2 + 32), v230) )
  {
    v35 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v278.m128i_i64[0] = (__int64)"tmp";
      v280 = 259;
      v35 = sub_921D70(a2, v237, (__int64)&v278, v34);
    }
    v36 = (unsigned int)v276;
    v37 = (unsigned int)v276 + 1LL;
    if ( v37 > HIDWORD(v276) )
    {
      sub_C8D5F0(&v275, v277, v37, 8);
      v36 = (unsigned int)v276;
    }
    v275[v36] = v35;
    LODWORD(v276) = v276 + 1;
  }
  v38 = (__int64)&v269;
  v39 = *(_QWORD *)(v230 + 16) + 40LL;
  v233 = (unsigned int *)(a3 + 36);
  v241 = (__m128i *)((char *)v281 + 24 * (unsigned int)v282);
  if ( v281 != v241 )
  {
    v215 = v10;
    v40 = v281;
    while ( 1 )
    {
      v41 = *(_DWORD *)(v39 + 12);
      v42 = v40->m128i_i64[0];
      v43 = v40->m128i_i32[2];
      if ( v41 > 1 )
      {
        if ( v41 == 2 )
        {
          if ( !v43 )
            sub_91B8A0("passing scalars indirectly is not supported!", v233, 1);
LABEL_141:
          v120 = (unsigned int)v276;
          v38 = HIDWORD(v276);
          v121 = (unsigned int)v276 + 1LL;
          if ( v121 > HIDWORD(v276) )
          {
            sub_C8D5F0(&v275, v277, v121, 8);
            v120 = (unsigned int)v276;
          }
          v275[v120] = v42;
          LODWORD(v276) = v276 + 1;
        }
      }
      else
      {
        if ( !v43 )
          goto LABEL_141;
        v44 = v40[1].m128i_u32[0];
        v271 = 257;
        v45 = unk_4D0463C;
        if ( unk_4D0463C )
          v45 = sub_90AA40(*(_QWORD *)(v250 + 32), v42);
        v226 = *(_QWORD *)(v42 + 72);
        if ( v44 )
        {
          _BitScanReverse64(&v122, v44);
          BYTE1(v122) = HIBYTE(dest);
          LOBYTE(v122) = 63 - (v122 ^ 0x3F);
          dest = v122;
        }
        else
        {
          v217 = v45;
          v46 = sub_AA4E30(*(_QWORD *)(v250 + 96));
          v47 = sub_AE5020(v46, v226);
          HIBYTE(v48) = HIBYTE(dest);
          v45 = v217;
          LOBYTE(v48) = v47;
          dest = v48;
        }
        v218 = v45;
        v280 = 257;
        v49 = sub_BD2C40(80, unk_3F10A14);
        v51 = v49;
        if ( v49 )
        {
          sub_B4D190(v49, v226, v42, (unsigned int)&v278, v218, (unsigned __int8)dest, 0, 0);
          v50 = v214;
        }
        (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v250 + 136) + 16LL))(
          *(_QWORD *)(v250 + 136),
          v51,
          &v269,
          *(_QWORD *)(v250 + 104),
          *(_QWORD *)(v250 + 112),
          v50);
        v52 = *(unsigned int **)(v250 + 48);
        v53 = &v52[4 * *(unsigned int *)(v250 + 56)];
        while ( v53 != v52 )
        {
          v54 = *((_QWORD *)v52 + 1);
          v55 = *v52;
          v52 += 4;
          sub_B99FD0(v51, v55, v54);
        }
        v56 = (unsigned int)v276;
        v38 = HIDWORD(v276);
        v57 = (unsigned int)v276 + 1LL;
        if ( v57 > HIDWORD(v276) )
        {
          sub_C8D5F0(&v275, v277, v57, 8);
          v56 = (unsigned int)v276;
        }
        v275[v56] = v51;
        LODWORD(v276) = v276 + 1;
      }
      v40 = (__m128i *)((char *)v40 + 24);
      v39 += 40;
      if ( v241 == v40 )
      {
        v10 = v215;
        break;
      }
    }
  }
  if ( *(_BYTE *)v239 == 5 )
  {
    v58 = *(_QWORD *)(v239 - 32LL * (*(_DWORD *)(v239 + 4) & 0x7FFFFFF));
    if ( !*(_BYTE *)v58 )
    {
      v59 = sub_91A390(*(_QWORD *)(v250 + 32) + 8LL, v10, 0, v38);
      if ( *(_WORD *)(v239 + 2) == 49 )
      {
        v60 = *(_QWORD *)(v58 + 24);
        v61 = *(_QWORD **)(v59 + 16);
        v62 = *(_QWORD **)(v60 + 16);
        if ( *v61 == *v62 )
        {
          v206 = *(_DWORD *)(v60 + 12);
          if ( *(_DWORD *)(v59 + 12) == v206 )
          {
            v207 = v206 - 1;
            if ( v207 == (_DWORD)v276 )
            {
              v208 = 0;
              while ( v208 != v207 )
              {
                ++v208;
                if ( v61[v208] != v62[v208] )
                  goto LABEL_73;
              }
              v239 = v58;
            }
          }
        }
      }
    }
  }
LABEL_73:
  sub_92FD10(v250, v233);
  sub_91CAC0(v233);
  if ( *(_BYTE *)(v244 + 24) == 20 )
  {
    v169 = *(const char **)(*(_QWORD *)(v244 + 56) + 8LL);
    if ( v169 )
    {
      if ( !strcmp(v169, "printf") )
      {
        a2 = v230;
        v90 = sub_939F40(v250, v230, (__int64 **)&v275, v233);
        goto LABEL_105;
      }
    }
  }
  v63 = v250;
  v64 = *(int **)(a3 + 64);
  v65 = (unsigned int **)(v250 + 48);
  if ( v64 )
  {
    v278.m128i_i64[0] = (__int64)v279;
    v278.m128i_i64[1] = 0x400000000LL;
    v66 = *v64;
    if ( *v64 >= 0 )
    {
      v155 = sub_B9B140(*(_QWORD *)(v250 + 40), "preserve_n_data", 15);
      v156 = v278.m128i_u32[2];
      v157 = v278.m128i_u32[2] + 1LL;
      if ( v157 > v278.m128i_u32[3] )
      {
        sub_C8D5F0(&v278, v279, v157, 8);
        v156 = v278.m128i_u32[2];
      }
      *(_QWORD *)(v278.m128i_i64[0] + 8 * v156) = v155;
      ++v278.m128i_i32[2];
      v158 = sub_BCB2D0(*(_QWORD *)(v250 + 40));
      v159 = sub_ACD640(v158, v66, 0);
      v162 = sub_B98A20(v159, v66, v160, v161);
      v163 = v278.m128i_u32[2];
      v164 = v278.m128i_u32[2] + 1LL;
      if ( v164 > v278.m128i_u32[3] )
      {
        sub_C8D5F0(&v278, v279, v164, 8);
        v163 = v278.m128i_u32[2];
      }
      *(_QWORD *)(v278.m128i_i64[0] + 8 * v163) = v162;
      ++v278.m128i_i32[2];
      v64 = *(int **)(a3 + 64);
      v67 = v64[1];
      if ( v67 < 0 )
      {
LABEL_77:
        v68 = v64[2];
        if ( v68 < 0 )
        {
LABEL_78:
          v69 = v278.m128i_u32[2];
          goto LABEL_79;
        }
LABEL_164:
        v135 = sub_B9B140(*(_QWORD *)(v250 + 40), "preserve_n_after", 16);
        v136 = v278.m128i_u32[2];
        v137 = v278.m128i_u32[2] + 1LL;
        if ( v137 > v278.m128i_u32[3] )
        {
          sub_C8D5F0(&v278, v279, v137, 8);
          v136 = v278.m128i_u32[2];
        }
        *(_QWORD *)(v278.m128i_i64[0] + 8 * v136) = v135;
        ++v278.m128i_i32[2];
        v138 = sub_BCB2D0(*(_QWORD *)(v250 + 40));
        v139 = sub_ACD640(v138, v68, 0);
        v142 = sub_B98A20(v139, v68, v140, v141);
        v143 = v278.m128i_u32[2];
        v144 = v278.m128i_u32[2] + 1LL;
        if ( v144 > v278.m128i_u32[3] )
        {
          sub_C8D5F0(&v278, v279, v144, 8);
          v143 = v278.m128i_u32[2];
        }
        *(_QWORD *)(v278.m128i_i64[0] + 8 * v143) = v142;
        v69 = (unsigned int)++v278.m128i_i32[2];
LABEL_79:
        v227 = sub_B9C770(*(_QWORD *)(v250 + 40), v278.m128i_i64[0], v69, 0, 1);
        v252 = v254;
        v253 = 0x200000000LL;
        v70 = *(_QWORD *)(v250 + 32);
        v268 = 257;
        v71 = *(_QWORD *)(v70 + 696);
        if ( v71 == *(_QWORD *)(v239 + 8) )
        {
          v74 = v239;
          v75 = 0;
          goto LABEL_88;
        }
        v72 = *(_QWORD *)(v250 + 128);
        v73 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v72 + 120LL);
        if ( v73 == sub_920130 )
        {
          if ( *(_BYTE *)v239 > 0x15u )
            goto LABEL_223;
          if ( (unsigned __int8)sub_AC4810(49) )
            v74 = sub_ADAB70(49, v239, v71, 0);
          else
            v74 = sub_AA93C0(49, v239, v71);
        }
        else
        {
          v74 = v73(v72, 49u, (_BYTE *)v239, v71);
        }
        if ( v74 )
        {
          v75 = (unsigned int)v253;
          v76 = HIDWORD(v253);
          v77 = (unsigned int)v253 + 1LL;
LABEL_86:
          if ( v77 > v76 )
          {
            v222 = v74;
            sub_C8D5F0(&v252, v254, v77, 8);
            v75 = (unsigned int)v253;
            v74 = v222;
          }
LABEL_88:
          *(_QWORD *)&v252[8 * v75] = v74;
          LODWORD(v253) = v253 + 1;
          v78 = sub_B9F6F0(*(_QWORD *)(v250 + 40), v227);
          v79 = (unsigned int)v253;
          v80 = (unsigned int)v253 + 1LL;
          if ( v80 > HIDWORD(v253) )
          {
            sub_C8D5F0(&v252, v254, v80, 8);
            v79 = (unsigned int)v253;
          }
          *(_QWORD *)&v252[8 * v79] = v78;
          LODWORD(v253) = v253 + 1;
          v81 = sub_B6E160(**(_QWORD **)(v250 + 32), 8285, 0, 0);
          v82 = 0;
          v271 = 257;
          if ( v81 )
            v82 = *(_QWORD *)(v81 + 24);
          v83 = sub_921880(v65, v82, v81, (int)v252, v253, (__int64)&v269, 0);
          v268 = 257;
          v84 = (_BYTE *)v83;
          v85 = *(_QWORD *)(v239 + 8);
          if ( v85 == *(_QWORD *)(v83 + 8) )
          {
            LODWORD(v239) = v83;
            goto LABEL_99;
          }
          v86 = *(_QWORD *)(v250 + 128);
          v87 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v86 + 120LL);
          if ( v87 == sub_920130 )
          {
            if ( *v84 > 0x15u )
              goto LABEL_215;
            v240 = *(_QWORD *)(v239 + 8);
            v82 = (unsigned __int64)v84;
            if ( (unsigned __int8)sub_AC4810(49) )
              v88 = sub_ADAB70(49, v84, v240, 0);
            else
              v88 = sub_AA93C0(49, v84, v240);
            v85 = v240;
            v239 = v88;
          }
          else
          {
            v228 = *(_QWORD *)(v239 + 8);
            v82 = 49;
            v209 = v87(v86, 49u, v84, v85);
            v85 = v228;
            v239 = v209;
          }
          if ( v239 )
          {
LABEL_99:
            if ( v252 != v254 )
              _libc_free(v252, v82);
            if ( (_DWORD *)v278.m128i_i64[0] != v279 )
              _libc_free(v278.m128i_i64[0], v82);
            goto LABEL_103;
          }
LABEL_215:
          v271 = 257;
          v239 = sub_B51D30(49, v84, v85, &v269, 0, 0);
          if ( (unsigned __int8)sub_920620(v239) )
          {
            v174 = *(_QWORD *)(v250 + 144);
            v175 = *(_DWORD *)(v250 + 152);
            if ( v174 )
              sub_B99FD0(v239, 3, v174);
            sub_B45150(v239, v175);
          }
          v82 = v239;
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v250 + 136) + 16LL))(
            *(_QWORD *)(v250 + 136),
            v239,
            v267,
            *(_QWORD *)(v250 + 104),
            *(_QWORD *)(v250 + 112));
          v176 = *(_QWORD *)(v250 + 48);
          v177 = 16LL * *(unsigned int *)(v250 + 56);
          if ( v176 != v176 + v177 )
          {
            v178 = (unsigned int *)(v176 + v177);
            v179 = *(unsigned int **)(v250 + 48);
            do
            {
              v180 = *((_QWORD *)v179 + 1);
              v82 = *v179;
              v179 += 4;
              sub_B99FD0(v239, v82, v180);
            }
            while ( v178 != v179 );
          }
          goto LABEL_99;
        }
LABEL_223:
        v271 = 257;
        v219 = sub_B51D30(49, v239, v71, &v269, 0, 0);
        v181 = sub_920620(v219);
        v182 = v219;
        if ( v181 )
        {
          v183 = *(_QWORD *)(v250 + 144);
          v184 = *(_DWORD *)(v250 + 152);
          if ( v183 )
          {
            sub_B99FD0(v219, 3, v183);
            v182 = v219;
          }
          v220 = v182;
          sub_B45150(v182, v184);
          v182 = v220;
        }
        v221 = v182;
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v250 + 136) + 16LL))(
          *(_QWORD *)(v250 + 136),
          v182,
          v267,
          *(_QWORD *)(v250 + 104),
          *(_QWORD *)(v250 + 112));
        v185 = *(_QWORD *)(v250 + 48);
        v74 = v221;
        v186 = 16LL * *(unsigned int *)(v250 + 56);
        if ( v185 != v185 + v186 )
        {
          v187 = (unsigned int *)(v185 + v186);
          v188 = *(unsigned int **)(v250 + 48);
          do
          {
            v189 = *((_QWORD *)v188 + 1);
            v190 = *v188;
            v188 += 4;
            sub_B99FD0(v221, v190, v189);
          }
          while ( v187 != v188 );
          v74 = v221;
        }
        v75 = (unsigned int)v253;
        v76 = HIDWORD(v253);
        v77 = (unsigned int)v253 + 1LL;
        goto LABEL_86;
      }
    }
    else
    {
      v67 = v64[1];
      if ( v67 < 0 )
        goto LABEL_77;
    }
    v145 = sub_B9B140(*(_QWORD *)(v250 + 40), "preserve_n_control", 18);
    v146 = v278.m128i_u32[2];
    v147 = v278.m128i_u32[2] + 1LL;
    if ( v147 > v278.m128i_u32[3] )
    {
      sub_C8D5F0(&v278, v279, v147, 8);
      v146 = v278.m128i_u32[2];
    }
    *(_QWORD *)(v278.m128i_i64[0] + 8 * v146) = v145;
    ++v278.m128i_i32[2];
    v148 = sub_BCB2D0(*(_QWORD *)(v250 + 40));
    v149 = sub_ACD640(v148, v67, 0);
    v152 = sub_B98A20(v149, v67, v150, v151);
    v153 = v278.m128i_u32[2];
    v154 = v278.m128i_u32[2] + 1LL;
    if ( v154 > v278.m128i_u32[3] )
    {
      sub_C8D5F0(&v278, v279, v154, 8);
      v153 = v278.m128i_u32[2];
    }
    *(_QWORD *)(v278.m128i_i64[0] + 8 * v153) = v152;
    ++v278.m128i_i32[2];
    v68 = *(_DWORD *)(*(_QWORD *)(a3 + 64) + 8LL);
    if ( v68 < 0 )
      goto LABEL_78;
    goto LABEL_164;
  }
LABEL_103:
  v269 = 0u;
  v270[0] = 0;
  v89 = sub_91A390(*(_QWORD *)(v250 + 32) + 8LL, v10, 0, v63);
  v280 = 257;
  v90 = sub_921880(v65, v89, v239, (int)v275, v276, (__int64)&v278, 0);
  sub_93AE30(*(_QWORD *)(v250 + 32), v230, 0, (__int64)&v269);
  a2 = v269.m128i_i64[0];
  *(_QWORD *)(v90 + 72) = sub_A7B050(
                            *(_QWORD *)(v250 + 40),
                            v269.m128i_i64[0],
                            (v269.m128i_i64[1] - v269.m128i_i64[0]) >> 3);
  if ( v269.m128i_i64[0] )
  {
    a2 = v270[0] - v269.m128i_i64[0];
    j_j___libc_free_0(v269.m128i_i64[0], v270[0] - v269.m128i_i64[0]);
  }
LABEL_105:
  if ( *(_BYTE *)(v244 + 24) != 20 )
  {
    v92 = v273;
    v278.m128i_i64[0] = (__int64)v279;
    v278.m128i_i64[1] = 0x1000000000LL;
    if ( (_DWORD)v273 )
    {
      v191 = v279;
      v192 = 8LL * (unsigned int)v273;
      if ( (unsigned int)v273 <= 0x10
        || (sub_C8D5F0(&v278, v279, (unsigned int)v273, 8),
            v191 = (_DWORD *)v278.m128i_i64[0],
            (v192 = 8LL * (unsigned int)v273) != 0) )
      {
        memcpy(v191, src, v192);
      }
      v278.m128i_i32[2] = v92;
    }
    a2 = v237;
    v269.m128i_i64[0] = (__int64)v270;
    v269.m128i_i64[1] = 0x400000000LL;
    v95 = sub_937280(v250, v237, 0, v91);
    if ( v95 )
    {
      v96 = sub_B98A20(v95, v237, v93, v94);
      v94 = v269.m128i_u32[3];
      v97 = v96;
      v98 = v269.m128i_u32[2];
      v99 = v269.m128i_u32[2] + 1LL;
      if ( v99 > v269.m128i_u32[3] )
      {
        a2 = (__int64)v270;
        sub_C8D5F0(&v269, v270, v99, 8);
        v98 = v269.m128i_u32[2];
      }
      *(_QWORD *)(v269.m128i_i64[0] + 8 * v98) = v97;
      ++v269.m128i_i32[2];
    }
    v100 = v278.m128i_u32[2];
    if ( v278.m128i_i32[2] )
    {
      v101 = 0;
      do
      {
        a2 = *(_QWORD *)(v278.m128i_i64[0] + 8 * v101);
        v103 = sub_937280(v250, a2, (unsigned int)(v101 + 1), v94);
        if ( v103 )
        {
          v104 = sub_B98A20(v103, a2, v102, v94);
          v105 = v269.m128i_u32[2];
          if ( (unsigned __int64)v269.m128i_u32[2] + 1 > v269.m128i_u32[3] )
          {
            a2 = (__int64)v270;
            v247 = v104;
            sub_C8D5F0(&v269, v270, v269.m128i_u32[2] + 1LL, 8);
            v105 = v269.m128i_u32[2];
            v104 = v247;
          }
          v94 = v269.m128i_i64[0];
          *(_QWORD *)(v269.m128i_i64[0] + 8 * v105) = v104;
          ++v269.m128i_i32[2];
        }
        ++v101;
      }
      while ( v100 != v101 );
    }
    if ( v269.m128i_i32[2] )
    {
      v172 = *(_QWORD *)(v250 + 40);
      v173 = sub_B9C770(v172, v269.m128i_i64[0], v269.m128i_u32[2], 0, 1);
      a2 = (unsigned int)sub_B6ED60(v172, "callalign", 9);
      sub_B99FD0(v90, a2, v173);
    }
    if ( (_QWORD *)v269.m128i_i64[0] != v270 )
      _libc_free(v269.m128i_i64[0], a2);
    if ( (_DWORD *)v278.m128i_i64[0] != v279 )
      _libc_free(v278.m128i_i64[0], a2);
  }
  if ( *(_BYTE *)(*(_QWORD *)(v90 + 8) + 8LL) != 7 )
  {
    a2 = (__int64)&v278;
    v278.m128i_i64[0] = (__int64)"call";
    v280 = 259;
    sub_BD6B50(v90, &v278);
  }
  v106 = *(_DWORD *)(v216 + 12);
  if ( v106 == 2 )
  {
    if ( sub_91B770(v237) )
    {
      v33 = (__int64 *)v275;
      v171 = *v275;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_DWORD *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v171;
      *(_DWORD *)(a1 + 16) = v223;
    }
    else
    {
      v210 = sub_91A390(*(_QWORD *)(v250 + 32) + 8LL, v237, 0, v170);
      a2 = *v275;
      v211 = sub_926480(v250, *v275, v210, v223, 0);
      v33 = (__int64 *)v275;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v211;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
    }
    goto LABEL_184;
  }
  if ( v106 <= 2 )
  {
    if ( !sub_91B770(v237) )
    {
      v33 = (__int64 *)v275;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v90;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      goto LABEL_184;
    }
    v108 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v109 = 1;
      if ( (a4 & 4) != 0 )
      {
LABEL_131:
        if ( a5 )
        {
          _BitScanReverse64(&v212, a5);
          v112 = (unsigned __int8)(63 - (v212 ^ 0x3F));
        }
        else
        {
          v245 = v109;
          v110 = sub_AA4E30(*(_QWORD *)(v250 + 96));
          v111 = sub_AE5020(v110, *(_QWORD *)(v90 + 8));
          v109 = v245;
          v112 = v111;
        }
        v246 = v109;
        v242 = v112;
        v280 = 257;
        v113 = sub_BD2C40(80, unk_3F10A10);
        v115 = v113;
        if ( v113 )
          sub_B4D3C0(v113, v90, v108, v246, v242, v114, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v250 + 136) + 16LL))(
          *(_QWORD *)(v250 + 136),
          v115,
          &v278,
          *(_QWORD *)(v250 + 104),
          *(_QWORD *)(v250 + 112));
        v116 = *(unsigned int **)(v250 + 48);
        v117 = &v116[4 * *(unsigned int *)(v250 + 56)];
        while ( v117 != v116 )
        {
          v118 = *((_QWORD *)v116 + 1);
          v119 = *v116;
          v116 += 4;
          sub_B99FD0(v115, v119, v118);
        }
        a2 = a5;
        v33 = (__int64 *)v275;
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v108;
        *(_DWORD *)(a1 + 8) = 1;
        *(_DWORD *)(a1 + 16) = a5;
        goto LABEL_184;
      }
    }
    else
    {
      v278.m128i_i64[0] = (__int64)"agg.tmp";
      v280 = 259;
      v108 = sub_921D70(v250, v237, (__int64)&v278, v107);
      a5 = v223;
    }
    v109 = unk_4D0463C;
    if ( unk_4D0463C )
      v109 = sub_90AA40(*(_QWORD *)(v250 + 32), v108);
    goto LABEL_131;
  }
  if ( v106 != 3 )
    sub_91B8A0("unexpected ABI kind for return value!", v233, 1);
  a2 = v250;
  sub_923000(a1, v250, v237);
  v33 = (__int64 *)v275;
LABEL_184:
  if ( v33 != (__int64 *)v277 )
    _libc_free(v33, a2);
  if ( v281 != (__m128i *)v283 )
    _libc_free(v281, a2);
  if ( v263 != v266 )
    _libc_free(v263, a2);
  if ( v259 != v262 )
    _libc_free(v259, a2);
  if ( v255 != v258 )
    _libc_free(v255, a2);
  if ( src != v274 )
    _libc_free(src, a2);
  return a1;
}
