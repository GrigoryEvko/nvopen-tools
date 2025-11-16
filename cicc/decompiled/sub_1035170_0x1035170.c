// Function: sub_1035170
// Address: 0x1035170
//
__int64 __fastcall sub_1035170(
        __int64 a1,
        _BYTE *a2,
        __int64 *a3,
        __m128i *a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int8 a9,
        char a10,
        unsigned __int8 a11)
{
  __int64 v11; // r10
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r14
  char v15; // r11
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  __int64 v21; // rcx
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // rcx
  int v27; // r9d
  __m128i *v28; // r8
  unsigned int v29; // eax
  __m128i *v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rdi
  const char *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rbx
  _QWORD *v36; // r12
  unsigned __int64 v37; // rsi
  _QWORD *v38; // rax
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // r8
  __int64 v44; // rcx
  __int64 v45; // rdx
  _DWORD *v46; // rax
  int v47; // eax
  __int64 v48; // r12
  __int64 v49; // rbx
  __int64 v50; // r13
  unsigned __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // r13
  unsigned __int64 v59; // rsi
  __int64 v60; // rdx
  __int64 *v61; // rax
  __int64 *v62; // rax
  unsigned int v63; // eax
  _DWORD *v64; // rbx
  _QWORD *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 v68; // rcx
  __int64 v69; // rdx
  _QWORD *v70; // rdx
  __int64 v71; // r9
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // rax
  const __m128i *v75; // rbx
  unsigned __int64 v76; // r8
  __m128i *v77; // rax
  unsigned int v78; // r12d
  _QWORD *v79; // rbx
  _QWORD *v80; // r13
  _QWORD *v81; // rdi
  unsigned __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned int v86; // edx
  __int64 v87; // rdx
  unsigned __int64 v88; // rcx
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rsi
  __int64 *v92; // rbx
  __int64 *v93; // r15
  __int64 v94; // r12
  __int64 v95; // r8
  __int64 v96; // rdx
  __m128i *v97; // r13
  signed __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // rbx
  unsigned __int64 v101; // rdx
  __int64 v102; // r13
  __int64 v103; // r12
  __int64 v104; // rdi
  __int64 v105; // rax
  __int64 v106; // r8
  __int64 v107; // r9
  __int64 v108; // rdx
  void **v109; // r12
  void *v110; // r14
  unsigned int v111; // r13d
  __int64 v112; // rax
  __int64 v113; // rdx
  void **v114; // r13
  __int64 v115; // rcx
  unsigned __int64 v116; // rsi
  __int32 v117; // eax
  __int64 v118; // rsi
  void **v119; // rdi
  __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // r13
  __int64 v124; // r13
  void *v125; // rbx
  void *v126; // r9
  __m128i v127; // xmm2
  __m128i v128; // xmm3
  char v129; // al
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rdi
  _QWORD *v133; // rcx
  _QWORD *v134; // r12
  __int64 v135; // r8
  int v136; // esi
  unsigned int v137; // edx
  __int64 *v138; // rax
  __int64 v139; // r10
  unsigned int v140; // eax
  int v141; // esi
  _BYTE *v142; // rdi
  size_t v143; // rdx
  __int64 v144; // rax
  __int64 v145; // rcx
  unsigned __int64 v146; // rcx
  const __m128i *v147; // rbx
  __int64 v148; // rax
  unsigned __int64 v149; // rdx
  unsigned __int64 v150; // r8
  __m128i *v151; // rax
  int v152; // eax
  int v153; // r9d
  char *v154; // r13
  char *v155; // r12
  __int64 *v156; // rax
  __int64 v157; // r8
  unsigned int v158; // ecx
  __int64 *v159; // rdx
  __int64 v160; // r10
  unsigned int v161; // edx
  __int64 v162; // rdi
  int v163; // edx
  const void *v164; // rsi
  char *v165; // rbx
  char *v166; // r8
  __int64 *v167; // rax
  unsigned int v168; // ecx
  __int64 *v169; // rdx
  __int64 v170; // r12
  unsigned int v171; // edx
  __int64 v172; // rdi
  int v173; // edx
  int v174; // r11d
  int v175; // ecx
  const char **v176; // r8
  unsigned __int64 v177; // rdx
  const char **v178; // rax
  const char **v179; // rbx
  char v180; // di
  const char *v181; // r12
  __int64 v182; // r10
  int v183; // esi
  unsigned int v184; // ecx
  __int64 *v185; // rdx
  __int64 v186; // r13
  __int64 v187; // rcx
  __int64 v188; // rdx
  const char **v189; // rbx
  __int64 v190; // rsi
  __int64 v191; // rdi
  unsigned int v192; // edx
  __int64 v193; // rdx
  const __m128i *v194; // rcx
  __int64 v195; // rdx
  unsigned __int64 v196; // rsi
  unsigned __int64 v197; // r8
  __m128i *v198; // rdx
  __int64 v199; // r9
  __int64 v200; // rax
  __m128i v201; // xmm5
  const void *v202; // rsi
  char *v203; // rbx
  int v204; // r12d
  int v205; // r12d
  __int64 v206; // r9
  unsigned int v207; // edi
  __int64 v208; // rsi
  int v209; // r8d
  __m128i *v210; // rax
  int v211; // r9d
  int v212; // r9d
  __int64 v213; // r8
  int v214; // edi
  unsigned int v215; // r12d
  __m128i *v216; // rsi
  __int64 v217; // rax
  __int64 v218; // rdx
  int v219; // edx
  int v220; // r15d
  unsigned int v221; // [rsp+14h] [rbp-ADCh]
  _DWORD *v222; // [rsp+20h] [rbp-AD0h]
  __int64 v223; // [rsp+30h] [rbp-AC0h]
  unsigned int v227; // [rsp+48h] [rbp-AA8h]
  void **v228; // [rsp+48h] [rbp-AA8h]
  int v229; // [rsp+50h] [rbp-AA0h]
  bool v230; // [rsp+57h] [rbp-A99h]
  __int64 v231; // [rsp+58h] [rbp-A98h]
  __int64 v232; // [rsp+60h] [rbp-A90h]
  __int64 m128i_i64; // [rsp+60h] [rbp-A90h]
  __int64 v234; // [rsp+60h] [rbp-A90h]
  __int64 v235; // [rsp+60h] [rbp-A90h]
  char v236; // [rsp+60h] [rbp-A90h]
  char v237; // [rsp+60h] [rbp-A90h]
  __int64 v238; // [rsp+68h] [rbp-A88h]
  __m128i *v239; // [rsp+68h] [rbp-A88h]
  __m128i *v240; // [rsp+68h] [rbp-A88h]
  __int64 *v241; // [rsp+68h] [rbp-A88h]
  __int64 v242; // [rsp+68h] [rbp-A88h]
  __int64 v243; // [rsp+68h] [rbp-A88h]
  __int8 *v244; // [rsp+68h] [rbp-A88h]
  __int64 *v245; // [rsp+70h] [rbp-A80h]
  const void *v246; // [rsp+70h] [rbp-A80h]
  unsigned __int8 v247; // [rsp+78h] [rbp-A78h]
  __int64 v248; // [rsp+78h] [rbp-A78h]
  const char **v249; // [rsp+78h] [rbp-A78h]
  bool dest; // [rsp+80h] [rbp-A70h]
  void *destb; // [rsp+80h] [rbp-A70h]
  void *desta; // [rsp+80h] [rbp-A70h]
  unsigned __int64 v254; // [rsp+98h] [rbp-A58h] BYREF
  __int64 v255; // [rsp+A0h] [rbp-A50h] BYREF
  __int64 v256; // [rsp+A8h] [rbp-A48h] BYREF
  void *v257; // [rsp+B0h] [rbp-A40h] BYREF
  __int64 v258[3]; // [rsp+B8h] [rbp-A38h] BYREF
  char v259; // [rsp+D0h] [rbp-A20h]
  void *src[2]; // [rsp+E0h] [rbp-A10h] BYREF
  __m128i v261; // [rsp+F0h] [rbp-A00h] BYREF
  __m128i v262; // [rsp+100h] [rbp-9F0h] BYREF
  __int64 v263; // [rsp+110h] [rbp-9E0h]
  _BYTE v264[88]; // [rsp+118h] [rbp-9D8h] BYREF
  _QWORD *v265; // [rsp+170h] [rbp-980h] BYREF
  __int64 v266; // [rsp+178h] [rbp-978h]
  _QWORD v267[32]; // [rsp+180h] [rbp-970h] BYREF
  const char *v268; // [rsp+280h] [rbp-870h] BYREF
  __int64 v269[2]; // [rsp+288h] [rbp-868h] BYREF
  __int64 v270; // [rsp+298h] [rbp-858h]
  __int64 v271; // [rsp+2A0h] [rbp-850h] BYREF
  unsigned int v272; // [rsp+2A8h] [rbp-848h]
  _QWORD v273[2]; // [rsp+3E0h] [rbp-710h] BYREF
  char v274; // [rsp+3F0h] [rbp-700h]
  _BYTE *v275; // [rsp+3F8h] [rbp-6F8h]
  __int64 v276; // [rsp+400h] [rbp-6F0h]
  _BYTE v277[128]; // [rsp+408h] [rbp-6E8h] BYREF
  __int16 v278; // [rsp+488h] [rbp-668h]
  void *v279; // [rsp+490h] [rbp-660h]
  __int64 v280; // [rsp+498h] [rbp-658h]
  __int64 v281; // [rsp+4A0h] [rbp-650h]
  __int64 v282; // [rsp+4A8h] [rbp-648h] BYREF
  unsigned int v283; // [rsp+4B0h] [rbp-640h]
  char v284; // [rsp+528h] [rbp-5C8h] BYREF
  __m128i v285; // [rsp+530h] [rbp-5C0h] BYREF
  _QWORD v286[3]; // [rsp+540h] [rbp-5B0h] BYREF
  __int64 v287; // [rsp+558h] [rbp-598h]
  __m128i v288; // [rsp+560h] [rbp-590h] BYREF
  __m128i v289; // [rsp+570h] [rbp-580h] BYREF
  __int64 v290; // [rsp+B08h] [rbp+18h]

  v11 = a6;
  v12 = a3;
  v13 = 4LL * a5;
  v14 = a8;
  v247 = a9;
  v15 = a10;
  v229 = a5;
  v16 = *a3;
  v17 = a4[1].m128i_i64[0];
  v18 = a4->m128i_i64[1];
  v19 = v16 & 0xFFFFFFFFFFFFFFFBLL | v13;
  v21 = a4[1].m128i_i64[1];
  v22 = a4[2].m128i_i64[0];
  v23 = a4[2].m128i_i64[1];
  v254 = v19;
  v230 = 0;
  if ( a2 && *a2 == 61 && (a2[7] & 0x20) != 0 )
  {
    v231 = v18;
    v232 = v17;
    v238 = v21;
    v24 = sub_B91C10((__int64)a2, 6);
    v19 = v254;
    v21 = v238;
    v17 = v232;
    v18 = v231;
    v11 = a6;
    v15 = a10;
    v230 = v24 != 0;
  }
  v287 = v18;
  v288.m128i_i64[0] = v17;
  v288.m128i_i64[1] = v21;
  v223 = a1 + 96;
  v25 = *(_DWORD *)(a1 + 120);
  v289.m128i_i64[0] = v22;
  v289.m128i_i64[1] = v23;
  if ( !v25 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_294;
  }
  v26 = *(_QWORD *)(a1 + 104);
  v27 = 1;
  v28 = 0;
  v29 = (v25 - 1) & (v19 ^ (v19 >> 9));
  v30 = (__m128i *)(v26 + 80LL * v29);
  v31 = v30->m128i_i64[0];
  if ( v19 == v30->m128i_i64[0] )
    goto LABEL_7;
  while ( 1 )
  {
    if ( v31 == -4 )
    {
      if ( v28 )
        v30 = v28;
      ++*(_QWORD *)(a1 + 96);
      v175 = *(_DWORD *)(a1 + 112) + 1;
      if ( 4 * v175 < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(a1 + 116) - v175 > v25 >> 3 )
          goto LABEL_237;
        v237 = v15;
        v243 = v11;
        sub_1031120(v223, v25);
        v211 = *(_DWORD *)(a1 + 120);
        if ( v211 )
        {
          v212 = v211 - 1;
          v213 = *(_QWORD *)(a1 + 104);
          v214 = 1;
          v215 = v212 & (v19 ^ (v19 >> 9));
          v11 = v243;
          v15 = v237;
          v175 = *(_DWORD *)(a1 + 112) + 1;
          v216 = 0;
          v30 = (__m128i *)(v213 + 80LL * v215);
          v217 = v30->m128i_i64[0];
          if ( v30->m128i_i64[0] != v19 )
          {
            while ( v217 != -4 )
            {
              if ( v217 == -16 && !v216 )
                v216 = v30;
              v215 = v212 & (v214 + v215);
              v30 = (__m128i *)(v213 + 80LL * v215);
              v217 = v30->m128i_i64[0];
              if ( v19 == v30->m128i_i64[0] )
                goto LABEL_237;
              ++v214;
            }
            if ( v216 )
              v30 = v216;
          }
          goto LABEL_237;
        }
        goto LABEL_337;
      }
LABEL_294:
      v236 = v15;
      v242 = v11;
      sub_1031120(v223, 2 * v25);
      v204 = *(_DWORD *)(a1 + 120);
      if ( v204 )
      {
        v205 = v204 - 1;
        v206 = *(_QWORD *)(a1 + 104);
        v11 = v242;
        v15 = v236;
        v207 = v205 & (v19 ^ (v19 >> 9));
        v30 = (__m128i *)(v206 + 80LL * v207);
        v175 = *(_DWORD *)(a1 + 112) + 1;
        v208 = v30->m128i_i64[0];
        if ( v19 != v30->m128i_i64[0] )
        {
          v209 = 1;
          v210 = 0;
          while ( v208 != -4 )
          {
            if ( !v210 && v208 == -16 )
              v210 = v30;
            v207 = v205 & (v209 + v207);
            v30 = (__m128i *)(v206 + 80LL * v207);
            v208 = v30->m128i_i64[0];
            if ( v30->m128i_i64[0] == v19 )
              goto LABEL_237;
            ++v209;
          }
          if ( v210 )
            v30 = v210;
        }
LABEL_237:
        *(_DWORD *)(a1 + 112) = v175;
        if ( v30->m128i_i64[0] != -4 )
          --*(_DWORD *)(a1 + 116);
        v30->m128i_i64[0] = v19;
        v30->m128i_i64[1] = 0;
        v30[1].m128i_i64[0] = 0;
        v30[1].m128i_i64[1] = 0;
        v30[2].m128i_i64[0] = 0;
        v30[2].m128i_i64[1] = v287;
        v30[3] = _mm_loadu_si128(&v288);
        v30[4] = _mm_loadu_si128(&v289);
        if ( v230 )
          goto LABEL_8;
        goto LABEL_240;
      }
LABEL_337:
      ++*(_DWORD *)(a1 + 112);
      BUG();
    }
    if ( v31 != -16 || v28 )
      v30 = v28;
    v29 = (v25 - 1) & (v27 + v29);
    v31 = *(_QWORD *)(v26 + 80LL * v29);
    if ( v31 == v19 )
      break;
    ++v27;
    v28 = v30;
    v30 = (__m128i *)(v26 + 80LL * v29);
  }
  v30 = (__m128i *)(v26 + 80LL * v29);
LABEL_7:
  if ( v230 )
  {
LABEL_8:
    m128i_i64 = (__int64)v30[1].m128i_i64;
    goto LABEL_9;
  }
  if ( a4->m128i_i64[1] != v30[2].m128i_i64[1] )
  {
    v48 = v30[1].m128i_i64[0];
    v49 = v30[1].m128i_i64[1];
    v30->m128i_i64[1] = 0;
    v30[2].m128i_i64[1] = a4->m128i_i64[1];
    if ( v48 == v49 )
    {
      v15 = 1;
    }
    else
    {
      v239 = v30;
      v50 = v48;
      v234 = v11;
      do
      {
        v52 = *(_QWORD *)(v50 + 8) & 7LL;
        if ( (unsigned int)v52 <= 2 )
        {
          v51 = *(_QWORD *)(v50 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v51 )
            sub_1029EF0(a1 + 128, v51, v254);
        }
        else if ( (_DWORD)v52 != 3 )
        {
LABEL_338:
          BUG();
        }
        v50 += 16;
      }
      while ( v50 != v49 );
      v30 = v239;
      v11 = v234;
      v15 = 1;
      v53 = v239[1].m128i_i64[0];
      if ( v53 != v239[1].m128i_i64[1] )
        v239[1].m128i_i64[1] = v53;
    }
  }
  v54 = v30[3].m128i_i64[0];
  v55 = a4[1].m128i_i64[0];
  if ( v54 != v55
    || v30[3].m128i_i64[1] != a4[1].m128i_i64[1]
    || v30[4].m128i_i64[0] != a4[2].m128i_i64[0]
    || v30[4].m128i_i64[1] != a4[2].m128i_i64[1] )
  {
    if ( v54 || v30[3].m128i_i64[1] || v30[4].m128i_i64[0] || v30[4].m128i_i64[1] )
    {
      v56 = v30[1].m128i_i64[0];
      v57 = v30[1].m128i_i64[1];
      v30->m128i_i64[1] = 0;
      v30[3].m128i_i64[0] = 0;
      v30[3].m128i_i64[1] = 0;
      v30[4].m128i_i64[0] = 0;
      v30[4].m128i_i64[1] = 0;
      if ( v56 != v57 )
      {
        v240 = v30;
        v58 = v56;
        v235 = v11;
        do
        {
          v60 = *(_QWORD *)(v58 + 8) & 7LL;
          if ( (unsigned int)v60 <= 2 )
          {
            v59 = *(_QWORD *)(v58 + 8) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v59 )
              sub_1029EF0(a1 + 128, v59, v254);
          }
          else if ( (_DWORD)v60 != 3 )
          {
            goto LABEL_338;
          }
          v58 += 16;
        }
        while ( v57 != v58 );
        v30 = v240;
        v11 = v235;
        v200 = v240[1].m128i_i64[0];
        if ( v200 != v240[1].m128i_i64[1] )
          v240[1].m128i_i64[1] = v200;
      }
      v15 = 1;
      v55 = a4[1].m128i_i64[0];
    }
    if ( v55 || a4[1].m128i_i64[1] || a4[2].m128i_i64[0] || a4[2].m128i_i64[1] )
    {
      memset(v286, 0, sizeof(v286));
      v201 = _mm_loadu_si128(a4);
      v287 = 0;
      v285 = v201;
      return (unsigned int)sub_1035170(
                             a1,
                             (_DWORD)a2,
                             (_DWORD)v12,
                             (unsigned int)&v285,
                             v229,
                             v11,
                             a7,
                             a8,
                             a9,
                             v15,
                             a11);
    }
  }
LABEL_240:
  m128i_i64 = (__int64)v30[1].m128i_i64;
  if ( v15 )
  {
LABEL_243:
    v30->m128i_i64[1] = 0;
    goto LABEL_9;
  }
  v176 = (const char **)v30[1].m128i_i64[1];
  v177 = v11 & 0xFFFFFFFFFFFFFFFBLL | (4LL * a9);
  v178 = (const char **)v30[1].m128i_i64[0];
  if ( v177 == v30->m128i_i64[1] )
  {
    if ( *(_DWORD *)(a8 + 8) >> 1 )
    {
      if ( v176 != v178 )
      {
        v179 = (const char **)v30[1].m128i_i64[0];
        desta = (void *)*v12;
        v180 = *(_BYTE *)(a8 + 8) & 1;
        while ( 1 )
        {
          v181 = *v179;
          if ( v180 )
          {
            v182 = a8 + 16;
            v183 = 15;
          }
          else
          {
            v188 = *(unsigned int *)(a8 + 24);
            v182 = *(_QWORD *)(a8 + 16);
            if ( !(_DWORD)v188 )
              goto LABEL_308;
            v183 = v188 - 1;
          }
          v184 = v183 & (((unsigned int)v181 >> 9) ^ ((unsigned int)v181 >> 4));
          v185 = (__int64 *)(v182 + 16LL * v184);
          v186 = *v185;
          if ( v181 == (const char *)*v185 )
            goto LABEL_250;
          v219 = 1;
          while ( v186 != -4096 )
          {
            v220 = v219 + 1;
            v184 = v183 & (v219 + v184);
            v185 = (__int64 *)(v182 + 16LL * v184);
            v186 = *v185;
            if ( v181 == (const char *)*v185 )
              goto LABEL_250;
            v219 = v220;
          }
          if ( v180 )
          {
            v218 = 256;
            goto LABEL_309;
          }
          v188 = *(unsigned int *)(a8 + 24);
LABEL_308:
          v218 = 16 * v188;
LABEL_309:
          v185 = (__int64 *)(v182 + v218);
LABEL_250:
          v187 = 256;
          if ( !v180 )
            v187 = 16LL * *(unsigned int *)(a8 + 24);
          if ( v185 != (__int64 *)(v182 + v187) && desta != (void *)v185[1] )
            return 0;
          v179 += 2;
          if ( v176 == v179 )
            goto LABEL_258;
        }
      }
    }
    else
    {
      desta = (void *)*v12;
      if ( v176 != v178 )
      {
LABEL_258:
        v249 = v176;
        v246 = (const void *)(a7 + 16);
        v189 = v178;
        do
        {
          v268 = *v189;
          v269[0] = (__int64)desta;
          sub_F5EC60((__int64)&v285, a8, (__int64 *)&v268, v269);
          if ( ((_DWORD)v189[1] & 7) != 3 || (unsigned __int64)v189[1] >> 61 != 1 )
          {
            v190 = *(_QWORD *)(a1 + 280);
            if ( *v189 )
            {
              v191 = (unsigned int)(*((_DWORD *)*v189 + 11) + 1);
              v192 = *((_DWORD *)*v189 + 11) + 1;
            }
            else
            {
              v191 = 0;
              v192 = 0;
            }
            if ( v192 < *(_DWORD *)(v190 + 32) && *(_QWORD *)(*(_QWORD *)(v190 + 24) + 8 * v191) )
            {
              v193 = (__int64)v189[1];
              v285.m128i_i64[0] = (__int64)*v189;
              v194 = &v285;
              v286[0] = desta;
              v285.m128i_i64[1] = v193;
              v195 = *(unsigned int *)(a7 + 8);
              v196 = *(_QWORD *)a7;
              v197 = v195 + 1;
              if ( v195 + 1 > (unsigned __int64)*(unsigned int *)(a7 + 12) )
              {
                if ( v196 > (unsigned __int64)&v285 || (unsigned __int64)&v285 >= v196 + 24 * v195 )
                {
                  sub_C8D5F0(a7, v246, v197, 0x18u, v197, v199);
                  v194 = &v285;
                  v196 = *(_QWORD *)a7;
                  v195 = *(unsigned int *)(a7 + 8);
                }
                else
                {
                  v244 = &v285.m128i_i8[-v196];
                  sub_C8D5F0(a7, v246, v197, 0x18u, v197, v199);
                  v196 = *(_QWORD *)a7;
                  v194 = (const __m128i *)&v244[*(_QWORD *)a7];
                  v195 = *(unsigned int *)(a7 + 8);
                }
              }
              v198 = (__m128i *)(v196 + 24 * v195);
              *v198 = _mm_loadu_si128(v194);
              v198[1].m128i_i64[0] = v194[1].m128i_i64[0];
              ++*(_DWORD *)(a7 + 8);
            }
          }
          v189 += 2;
        }
        while ( v249 != v189 );
      }
    }
    return 1;
  }
  if ( v176 != v178 )
    goto LABEL_243;
  v30->m128i_i64[1] = v177;
LABEL_9:
  v267[0] = v11;
  v32 = *(_QWORD *)(v11 + 72);
  v265 = v267;
  v266 = 0x2000000001LL;
  v285.m128i_i64[0] = (__int64)v286;
  v285.m128i_i64[1] = 0x1000000000LL;
  dest = 0;
  v227 = (v30[1].m128i_i64[1] - v30[1].m128i_i64[0]) >> 4;
  if ( v32 )
  {
    v33 = sub_BD5D20(v32);
    v269[0] = v34;
    v268 = v33;
    dest = sub_C931B0((__int64 *)&v268, "cutlass", 7u, 0) != -1;
  }
  v35 = sub_C52410();
  v36 = v35 + 1;
  v37 = sub_C959E0();
  v38 = (_QWORD *)v35[2];
  if ( v38 )
  {
    v39 = v35 + 1;
    do
    {
      while ( 1 )
      {
        v40 = v38[2];
        v41 = v38[3];
        if ( v37 <= v38[4] )
          break;
        v38 = (_QWORD *)v38[3];
        if ( !v41 )
          goto LABEL_16;
      }
      v39 = v38;
      v38 = (_QWORD *)v38[2];
    }
    while ( v40 );
LABEL_16:
    if ( v36 != v39 && v37 >= v39[4] )
      v36 = v39;
  }
  if ( v36 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v42 = v36[7];
    if ( v42 )
    {
      v43 = v36 + 6;
      do
      {
        while ( 1 )
        {
          v44 = *(_QWORD *)(v42 + 16);
          v45 = *(_QWORD *)(v42 + 24);
          if ( *(_DWORD *)(v42 + 32) >= dword_4F8F128 )
            break;
          v42 = *(_QWORD *)(v42 + 24);
          if ( !v45 )
            goto LABEL_25;
        }
        v43 = (_QWORD *)v42;
        v42 = *(_QWORD *)(v42 + 16);
      }
      while ( v44 );
LABEL_25:
      if ( v43 != v36 + 6 && dword_4F8F128 >= *((_DWORD *)v43 + 8) && *((_DWORD *)v43 + 9) )
        goto LABEL_53;
    }
  }
  v46 = sub_C94E20((__int64)qword_4F86370);
  v47 = v46 ? *v46 : LODWORD(qword_4F86370[2]);
  if ( v47 <= 2 )
LABEL_53:
    v221 = qword_4F8F1A8;
  else
    v221 = !dest ? 400 : 2000;
  v269[1] = 0;
  v270 = 1;
  v268 = *(const char **)(a1 + 256);
  v269[0] = (__int64)v268;
  v61 = &v271;
  do
  {
    *v61 = -4;
    v61 += 5;
    *(v61 - 4) = -3;
    *(v61 - 3) = -4;
    *(v61 - 2) = -3;
  }
  while ( v61 != v273 );
  v273[0] = a1 + 416;
  v275 = v277;
  v276 = 0x400000000LL;
  v273[1] = 0;
  v274 = 0;
  v278 = 256;
  v280 = 0;
  v281 = 1;
  v279 = &unk_49DDBE8;
  v62 = &v282;
  do
  {
    *v62 = -4096;
    v62 += 2;
  }
  while ( v62 != (__int64 *)&v284 );
  v63 = v266;
  if ( !(_DWORD)v266 )
    goto LABEL_76;
  v241 = &v30->m128i_i64[1];
LABEL_60:
  v64 = (_DWORD *)v265[v63 - 1];
  LODWORD(v266) = v63 - 1;
  if ( *(_DWORD *)(a7 + 8) > 0x64u )
  {
    v67 = v227;
    if ( v227 != (__int64)(*(_QWORD *)(m128i_i64 + 8) - *(_QWORD *)m128i_i64) >> 4 )
      sub_102D580((__m128i **)m128i_i64, v227);
    v78 = 0;
    *v241 = 0;
    goto LABEL_77;
  }
  if ( !v247 )
  {
    v83 = sub_1034C30(a1, a2, a4, v229, (__int64)v64, m128i_i64, v227, a11, (__int64 *)&v268);
    if ( (v83 & 7) != 3 || v83 >> 61 != 1 )
    {
      v84 = *(_QWORD *)(a1 + 280);
      if ( v64 )
      {
        v85 = (unsigned int)(v64[11] + 1);
        v86 = v64[11] + 1;
      }
      else
      {
        v85 = 0;
        v86 = 0;
      }
      if ( v86 < *(_DWORD *)(v84 + 32) && *(_QWORD *)(*(_QWORD *)(v84 + 24) + 8 * v85) )
      {
        src[1] = (void *)v83;
        v87 = *v12;
        src[0] = v64;
        v88 = *(unsigned int *)(a7 + 12);
        v75 = (const __m128i *)src;
        v74 = *(unsigned int *)(a7 + 8);
        v261.m128i_i64[0] = v87;
        v73 = *(_QWORD *)a7;
        v76 = v74 + 1;
        if ( v74 + 1 <= v88 )
          goto LABEL_74;
        v164 = (const void *)(a7 + 16);
        if ( v73 <= (unsigned __int64)src && (unsigned __int64)src < v73 + 24 * v74 )
        {
LABEL_210:
          v165 = (char *)src - v73;
          sub_C8D5F0(a7, v164, v76, 0x18u, v76, v71);
          v73 = *(_QWORD *)a7;
          v74 = *(unsigned int *)(a7 + 8);
          v75 = (const __m128i *)&v165[*(_QWORD *)a7];
LABEL_74:
          v77 = (__m128i *)(v73 + 24 * v74);
          *v77 = _mm_loadu_si128(v75);
          v77[1].m128i_i64[0] = v75[1].m128i_i64[0];
          ++*(_DWORD *)(a7 + 8);
          v63 = v266;
          goto LABEL_75;
        }
LABEL_211:
        sub_C8D5F0(a7, v164, v76, 0x18u, v76, v71);
        v75 = (const __m128i *)src;
        v73 = *(_QWORD *)a7;
        v74 = *(unsigned int *)(a7 + 8);
        goto LABEL_74;
      }
    }
  }
  v65 = (_QWORD *)v12[4];
  v66 = 8LL * *((unsigned int *)v12 + 10);
  v67 = (__int64)&v65[(unsigned __int64)v66 / 8];
  v68 = v66 >> 3;
  v69 = v66 >> 5;
  if ( !v69 )
    goto LABEL_101;
  v70 = &v65[4 * v69];
  while ( v64 != *(_DWORD **)(*v65 + 40LL) )
  {
    if ( v64 == *(_DWORD **)(v65[1] + 40LL) )
    {
      ++v65;
    }
    else if ( v64 == *(_DWORD **)(v65[2] + 40LL) )
    {
      v65 += 2;
    }
    else if ( v64 == *(_DWORD **)(v65[3] + 40LL) )
    {
      v65 += 3;
    }
    else
    {
      v65 += 4;
      if ( v70 != v65 )
        continue;
      v68 = (v67 - (__int64)v65) >> 3;
LABEL_101:
      if ( v68 == 2 )
        goto LABEL_198;
      if ( v68 != 3 )
      {
        if ( v68 == 1 )
          goto LABEL_104;
        goto LABEL_105;
      }
      v70 = (_QWORD *)*v65;
      if ( v64 != *(_DWORD **)(*v65 + 40LL) )
      {
        ++v65;
LABEL_198:
        v70 = (_QWORD *)*v65;
        if ( v64 != *(_DWORD **)(*v65 + 40LL) )
        {
          ++v65;
LABEL_104:
          v70 = (_QWORD *)*v65;
          if ( v64 != *(_DWORD **)(*v65 + 40LL) )
          {
LABEL_105:
            src[0] = &v261;
            src[1] = (void *)0x1000000000LL;
            v89 = sub_102DBD0(a1 + 288, (__int64)v64);
            v91 = v89 + 8 * v90;
            v245 = (__int64 *)v91;
            if ( v91 == v89 )
              goto LABEL_114;
            v222 = v64;
            v92 = v12;
            v93 = (__int64 *)v89;
            while ( 2 )
            {
              while ( 2 )
              {
                v94 = *v93;
                v91 = v14;
                v256 = *v92;
                v255 = v94;
                sub_F5EC60((__int64)&v257, v14, &v255, &v256);
                if ( v259 )
                {
                  v96 = LODWORD(src[1]);
                  if ( (unsigned __int64)LODWORD(src[1]) + 1 > HIDWORD(src[1]) )
                  {
                    v91 = (__int64)&v261;
                    sub_C8D5F0((__int64)src, &v261, LODWORD(src[1]) + 1LL, 8u, v95, v71);
                    v96 = LODWORD(src[1]);
                  }
                  ++v93;
                  *((_QWORD *)src[0] + v96) = v94;
                  ++LODWORD(src[1]);
                  if ( v245 != v93 )
                    continue;
LABEL_113:
                  v12 = v92;
                  v64 = v222;
LABEL_114:
                  v97 = (__m128i *)src[0];
                  v98 = 8LL * LODWORD(src[1]);
                  if ( LODWORD(src[1]) <= v221 )
                  {
                    v99 = (unsigned int)v266;
                    v100 = v98 >> 3;
                    v221 -= LODWORD(src[1]);
                    v101 = (v98 >> 3) + (unsigned int)v266;
                    if ( v101 > HIDWORD(v266) )
                    {
                      v91 = (__int64)v267;
                      sub_C8D5F0((__int64)&v265, v267, v101, 8u, LODWORD(src[1]), v71);
                      v99 = (unsigned int)v266;
                    }
                    if ( v98 )
                    {
                      v91 = (__int64)v97;
                      memcpy(&v265[v99], v97, v98);
                      LODWORD(v99) = v266;
                    }
                    LODWORD(v266) = v99 + v100;
                    v63 = v99 + v100;
                    if ( src[0] != &v261 )
                    {
                      _libc_free(src[0], v91);
                      v63 = v266;
                    }
                    goto LABEL_75;
                  }
                  v166 = (char *)src[0] + v98;
                  if ( (char *)src[0] + v98 != src[0] )
                  {
                    v167 = (__int64 *)src[0];
                    while ( 1 )
                    {
                      v172 = *v167;
                      if ( (*(_BYTE *)(v14 + 8) & 1) != 0 )
                        break;
                      v91 = *(unsigned int *)(v14 + 24);
                      v71 = *(_QWORD *)(v14 + 16);
                      if ( (_DWORD)v91 )
                      {
                        v91 = (unsigned int)(v91 - 1);
LABEL_215:
                        v168 = v91 & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4));
                        v169 = (__int64 *)(v71 + 16LL * v168);
                        v170 = *v169;
                        if ( v172 == *v169 )
                        {
LABEL_216:
                          *v169 = -8192;
                          v171 = *(_DWORD *)(v14 + 8);
                          ++*(_DWORD *)(v14 + 12);
                          *(_DWORD *)(v14 + 8) = (2 * (v171 >> 1) - 2) | v171 & 1;
                        }
                        else
                        {
                          v173 = 1;
                          while ( v170 != -4096 )
                          {
                            v174 = v173 + 1;
                            v168 = v91 & (v173 + v168);
                            v169 = (__int64 *)(v71 + 16LL * v168);
                            v170 = *v169;
                            if ( v172 == *v169 )
                              goto LABEL_216;
                            v173 = v174;
                          }
                        }
                      }
                      if ( v166 == (char *)++v167 )
                        goto LABEL_190;
                    }
                    v71 = v14 + 16;
                    v91 = 15;
                    goto LABEL_215;
                  }
LABEL_191:
                  if ( v97 != &v261 )
                    _libc_free(v97, v91);
                  *v241 = 0;
LABEL_72:
                  if ( !v230 )
                  {
                    v144 = *(_QWORD *)(m128i_i64 + 8);
                    if ( v144 != *(_QWORD *)m128i_i64 )
                    {
                      v145 = v144 - 16;
                      if ( v64 == *(_DWORD **)(v144 - 16) )
                      {
LABEL_165:
                        *(_QWORD *)(v144 - 8) = 0x6000000000000003LL;
                      }
                      else
                      {
                        while ( 1 )
                        {
                          v144 = v145;
                          if ( v145 == *(_QWORD *)m128i_i64 )
                            break;
                          v145 -= 16;
                          if ( v64 == *(_DWORD **)(v144 - 16) )
                            goto LABEL_165;
                        }
                      }
                    }
                  }
                  v72 = *v12;
                  src[0] = v64;
                  src[1] = (void *)0x6000000000000003LL;
                  v261.m128i_i64[0] = v72;
                  v73 = *(_QWORD *)a7;
                  v74 = *(unsigned int *)(a7 + 8);
                  v75 = (const __m128i *)src;
                  v76 = v74 + 1;
                  if ( v74 + 1 <= (unsigned __int64)*(unsigned int *)(a7 + 12) )
                    goto LABEL_74;
                  v164 = (const void *)(a7 + 16);
                  if ( v73 <= (unsigned __int64)src && (unsigned __int64)src < v73 + 24 * v74 )
                    goto LABEL_210;
                  goto LABEL_211;
                }
                break;
              }
              if ( *(_QWORD *)(v258[1] + 8) == *v92 )
              {
                if ( v245 == ++v93 )
                  goto LABEL_113;
                continue;
              }
              break;
            }
            v97 = (__m128i *)src[0];
            v12 = v92;
            v64 = v222;
            v155 = (char *)src[0] + 8 * LODWORD(src[1]);
            if ( v155 == src[0] )
              goto LABEL_191;
            v156 = (__int64 *)src[0];
            while ( 1 )
            {
              v162 = *v156;
              if ( (*(_BYTE *)(v14 + 8) & 1) != 0 )
                break;
              v91 = *(unsigned int *)(v14 + 24);
              v157 = *(_QWORD *)(v14 + 16);
              if ( (_DWORD)v91 )
              {
                v91 = (unsigned int)(v91 - 1);
LABEL_184:
                v158 = v91 & (((unsigned int)v162 >> 9) ^ ((unsigned int)v162 >> 4));
                v159 = (__int64 *)(v157 + 16LL * v158);
                v160 = *v159;
                if ( v162 == *v159 )
                {
LABEL_185:
                  *v159 = -8192;
                  v161 = *(_DWORD *)(v14 + 8);
                  ++*(_DWORD *)(v14 + 12);
                  *(_DWORD *)(v14 + 8) = (2 * (v161 >> 1) - 2) | v161 & 1;
                }
                else
                {
                  v163 = 1;
                  while ( v160 != -4096 )
                  {
                    v71 = (unsigned int)(v163 + 1);
                    v158 = v91 & (v163 + v158);
                    v159 = (__int64 *)(v157 + 16LL * v158);
                    v160 = *v159;
                    if ( v162 == *v159 )
                      goto LABEL_185;
                    v163 = v71;
                  }
                }
              }
              if ( v155 == (char *)++v156 )
              {
LABEL_190:
                v97 = (__m128i *)src[0];
                goto LABEL_191;
              }
            }
            v157 = v14 + 16;
            v91 = 15;
            goto LABEL_184;
          }
        }
      }
    }
    break;
  }
  if ( (_QWORD *)v67 == v65 )
    goto LABEL_105;
  if ( !(unsigned __int8)sub_104A900(v12, v67, v70) )
    goto LABEL_71;
  if ( v227 != (__int64)(*(_QWORD *)(m128i_i64 + 8) - *(_QWORD *)m128i_i64) >> 4 )
    sub_102D580((__m128i **)m128i_i64, v227);
  v102 = v285.m128i_i64[0];
  v103 = v285.m128i_i64[0] + 88LL * v285.m128i_u32[2];
  while ( v102 != v103 )
  {
    while ( 1 )
    {
      v103 -= 88;
      v104 = *(_QWORD *)(v103 + 40);
      if ( v104 == v103 + 56 )
        break;
      _libc_free(v104, v227);
      if ( v102 == v103 )
        goto LABEL_127;
    }
  }
LABEL_127:
  v285.m128i_i32[2] = 0;
  v105 = sub_102DBD0(a1 + 288, (__int64)v64);
  v228 = (void **)(v105 + 8 * v108);
  if ( (void **)v105 == v228 )
    goto LABEL_140;
  v290 = v14;
  v109 = (void **)v105;
  while ( 1 )
  {
    v110 = *v109;
    v111 = *((_DWORD *)v12 + 10);
    src[1] = (void *)*v12;
    v112 = v12[1];
    src[0] = v110;
    v261.m128i_i64[0] = v112;
    v261.m128i_i64[1] = v12[2];
    v262.m128i_i64[0] = v12[3];
    v262.m128i_i64[1] = (__int64)v264;
    v263 = 0x400000000LL;
    if ( v111 )
    {
      v142 = v264;
      v143 = 8LL * v111;
      if ( v111 <= 4
        || (sub_C8D5F0((__int64)&v262.m128i_i64[1], v264, v111, 8u, v111, v107),
            v142 = (_BYTE *)v262.m128i_i64[1],
            (v143 = 8LL * *((unsigned int *)v12 + 10)) != 0) )
      {
        memcpy(v142, (const void *)v12[4], v143);
      }
      LODWORD(v263) = v111;
    }
    v113 = v285.m128i_u32[2];
    v114 = src;
    v115 = v285.m128i_i64[0];
    v116 = v285.m128i_u32[2] + 1LL;
    v117 = v285.m128i_i32[2];
    if ( v116 > v285.m128i_u32[3] )
    {
      if ( v285.m128i_i64[0] > (unsigned __int64)src
        || (unsigned __int64)src >= v285.m128i_i64[0] + 88 * (unsigned __int64)v285.m128i_u32[2] )
      {
        v114 = src;
        sub_102D890((__int64)&v285, v116, v285.m128i_u32[2], v285.m128i_i64[0], v106, v107);
        v113 = v285.m128i_u32[2];
        v115 = v285.m128i_i64[0];
        v117 = v285.m128i_i32[2];
      }
      else
      {
        v154 = (char *)src - v285.m128i_i64[0];
        sub_102D890((__int64)&v285, v116, v285.m128i_u32[2], v285.m128i_i64[0], v106, v107);
        v115 = v285.m128i_i64[0];
        v113 = v285.m128i_u32[2];
        v114 = (void **)&v154[v285.m128i_i64[0]];
        v117 = v285.m128i_i32[2];
      }
    }
    v118 = 5 * v113;
    v119 = (void **)(v115 + 88 * v113);
    if ( v119 )
    {
      *v119 = *v114;
      v119[1] = v114[1];
      v119[2] = v114[2];
      v119[3] = v114[3];
      v119[4] = v114[4];
      v119[5] = v119 + 7;
      v119[6] = (void *)0x400000000LL;
      v120 = *((unsigned int *)v114 + 12);
      if ( (_DWORD)v120 )
      {
        v118 = (__int64)(v114 + 5);
        sub_1029680((__int64)(v119 + 5), (char **)v114 + 5, v120, v115, v106, v107);
      }
      v117 = v285.m128i_i32[2];
    }
    v121 = (unsigned int)(v117 + 1);
    v285.m128i_i32[2] = v121;
    if ( (_BYTE *)v262.m128i_i64[1] != v264 )
    {
      _libc_free(v262.m128i_i64[1], v118);
      v121 = v285.m128i_u32[2];
    }
    v122 = sub_104B4A0(v285.m128i_i64[0] + 88 * v121 - 80, v64, v110, *(_QWORD *)(a1 + 280), 0);
    v257 = v110;
    v123 = v122;
    v258[0] = v122;
    sub_F5EC60((__int64)src, v290, (__int64 *)&v257, v258);
    if ( !v262.m128i_i8[0] )
    {
      --v285.m128i_i32[2];
      v131 = v285.m128i_i64[0] + 88LL * v285.m128i_u32[2];
      v132 = *(_QWORD *)(v131 + 40);
      if ( v132 != v131 + 56 )
        _libc_free(v132, v290);
      if ( v123 != *(_QWORD *)(v261.m128i_i64[0] + 8) )
        break;
    }
    if ( v228 == ++v109 )
    {
      v14 = v290;
LABEL_140:
      v124 = v285.m128i_i64[0];
      v248 = v285.m128i_i64[0] + 88LL * v285.m128i_u32[2];
      if ( v248 != v285.m128i_i64[0] )
      {
        do
        {
          v125 = *(void **)(v124 + 8);
          v126 = *(void **)v124;
          if ( !v125 )
            goto LABEL_169;
          v127 = _mm_loadu_si128(a4 + 1);
          v128 = _mm_loadu_si128(a4 + 2);
          destb = *(void **)v124;
          src[1] = (void *)_mm_loadu_si128(a4).m128i_i64[1];
          v261 = v127;
          src[0] = v125;
          v262 = v128;
          v129 = sub_1035170(a1, (_DWORD)a2, (int)v124 + 8, (unsigned int)src, v229, (_DWORD)v126, a7, v14, 0, 0, 1);
          v126 = destb;
          if ( !v129 )
          {
LABEL_169:
            v146 = *(unsigned int *)(a7 + 12);
            v261.m128i_i64[0] = (__int64)v125;
            v147 = (const __m128i *)src;
            src[1] = (void *)0x6000000000000003LL;
            v148 = *(unsigned int *)(a7 + 8);
            src[0] = v126;
            v149 = *(_QWORD *)a7;
            v150 = v148 + 1;
            if ( v148 + 1 > v146 )
            {
              v202 = (const void *)(a7 + 16);
              if ( v149 > (unsigned __int64)src || (unsigned __int64)src >= v149 + 24 * v148 )
              {
                v147 = (const __m128i *)src;
                sub_C8D5F0(a7, v202, v150, 0x18u, v150, (__int64)v126);
                v149 = *(_QWORD *)a7;
                v148 = *(unsigned int *)(a7 + 8);
              }
              else
              {
                v203 = (char *)src - v149;
                sub_C8D5F0(a7, v202, v150, 0x18u, v150, (__int64)v126);
                v149 = *(_QWORD *)a7;
                v148 = *(unsigned int *)(a7 + 8);
                v147 = (const __m128i *)&v203[*(_QWORD *)a7];
              }
            }
            v151 = (__m128i *)(v149 + 24 * v148);
            *v151 = _mm_loadu_si128(v147);
            v151[1].m128i_i64[0] = v147[1].m128i_i64[0];
            ++*(_DWORD *)(a7 + 8);
            *sub_1031390(v223, (__int64 *)&v254) = 0;
          }
          v124 += 88;
        }
        while ( v248 != v124 );
      }
      v241 = sub_1031390(v223, (__int64 *)&v254);
      m128i_i64 = (__int64)(v241 + 1);
      v130 = v241[2] - v241[1];
      *v241 = 0;
      v227 = v130 >> 4;
      v63 = v266;
LABEL_75:
      v247 = 0;
      if ( !v63 )
      {
LABEL_76:
        v67 = v227;
        v78 = 1;
        sub_102D580((__m128i **)m128i_i64, v227);
        goto LABEL_77;
      }
      goto LABEL_60;
    }
  }
  v133 = (_QWORD *)v285.m128i_i64[0];
  v14 = v290;
  v134 = (_QWORD *)(v285.m128i_i64[0] + 88LL * v285.m128i_u32[2]);
  if ( v134 != (_QWORD *)v285.m128i_i64[0] )
  {
    while ( (*(_BYTE *)(v290 + 8) & 1) == 0 )
    {
      v141 = *(_DWORD *)(v290 + 24);
      v135 = *(_QWORD *)(v290 + 16);
      if ( v141 )
      {
        v136 = v141 - 1;
LABEL_151:
        v137 = v136 & (((unsigned int)*v133 >> 9) ^ ((unsigned int)*v133 >> 4));
        v138 = (__int64 *)(v135 + 16LL * v137);
        v139 = *v138;
        if ( *v138 == *v133 )
        {
LABEL_152:
          *v138 = -8192;
          v140 = *(_DWORD *)(v290 + 8);
          ++*(_DWORD *)(v290 + 12);
          *(_DWORD *)(v290 + 8) = (2 * (v140 >> 1) - 2) | v140 & 1;
        }
        else
        {
          v152 = 1;
          while ( v139 != -4096 )
          {
            v153 = v152 + 1;
            v137 = v136 & (v152 + v137);
            v138 = (__int64 *)(v135 + 16LL * v137);
            v139 = *v138;
            if ( *v133 == *v138 )
              goto LABEL_152;
            v152 = v153;
          }
        }
      }
      v133 += 11;
      if ( v134 == v133 )
        goto LABEL_160;
    }
    v135 = v290 + 16;
    v136 = 15;
    goto LABEL_151;
  }
LABEL_160:
  v67 = (__int64)sub_1031390(v223, (__int64 *)&v254);
  v241 = (__int64 *)v67;
  m128i_i64 = v67 + 8;
  v227 = (__int64)(*(_QWORD *)(v67 + 16) - *(_QWORD *)(v67 + 8)) >> 4;
LABEL_71:
  *v241 = 0;
  if ( !v247 )
    goto LABEL_72;
  v78 = 0;
LABEL_77:
  v279 = &unk_49DDBE8;
  if ( (v281 & 1) == 0 )
  {
    v67 = 16LL * v283;
    sub_C7D6A0(v282, v67, 8);
  }
  nullsub_184();
  if ( v275 != v277 )
    _libc_free(v275, v67);
  if ( (v270 & 1) == 0 )
  {
    v67 = 40LL * v272;
    sub_C7D6A0(v271, v67, 8);
  }
  v79 = (_QWORD *)v285.m128i_i64[0];
  v80 = (_QWORD *)(v285.m128i_i64[0] + 88LL * v285.m128i_u32[2]);
  if ( (_QWORD *)v285.m128i_i64[0] != v80 )
  {
    do
    {
      v80 -= 11;
      v81 = (_QWORD *)v80[5];
      if ( v81 != v80 + 7 )
        _libc_free(v81, v67);
    }
    while ( v79 != v80 );
    v80 = (_QWORD *)v285.m128i_i64[0];
  }
  if ( v80 != v286 )
    _libc_free(v80, v67);
  if ( v265 != v267 )
    _libc_free(v265, v67);
  return v78;
}
