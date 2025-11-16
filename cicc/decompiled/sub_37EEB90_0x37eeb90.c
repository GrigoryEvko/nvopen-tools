// Function: sub_37EEB90
// Address: 0x37eeb90
//
__int64 __fastcall sub_37EEB90(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  const __m128i *v20; // rax
  __int64 v21; // rax
  const __m128i *v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __m128i *v27; // rdi
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  const __m128i *v35; // rcx
  unsigned __int64 v36; // rbx
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  __m128i *v39; // rdx
  const __m128i *v40; // rax
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // r15
  __int64 *v49; // r12
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rdi
  int v53; // ecx
  __int64 v54; // rax
  __int64 k; // rdi
  __int64 v56; // rax
  _QWORD *v57; // rax
  __m128i *v58; // rdx
  __m128i si128; // xmm0
  _QWORD *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rbx
  const char *v63; // rax
  size_t v64; // rdx
  _WORD *v65; // rdi
  unsigned __int8 *v66; // rsi
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  _DWORD *v69; // rdx
  __int64 v70; // rbx
  const char *v71; // rax
  size_t v72; // rdx
  __m128i *v73; // rdi
  unsigned __int8 *v74; // rsi
  unsigned __int64 v75; // rax
  __m128i v76; // xmm0
  __int64 v77; // rdi
  _BYTE *v78; // rax
  _QWORD *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rbx
  const char *v82; // rax
  size_t v83; // rdx
  _WORD *v84; // rdi
  unsigned __int8 *v85; // rsi
  unsigned __int64 v86; // rax
  __int64 v87; // rax
  _DWORD *v88; // rdx
  __int64 v89; // rbx
  const char *v90; // rax
  size_t v91; // rdx
  __m128i *v92; // rdi
  unsigned __int8 *v93; // rsi
  unsigned __int64 v94; // rax
  __m128i v95; // xmm0
  __int64 v96; // rdi
  _BYTE *v97; // rax
  _QWORD *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rbx
  const char *v101; // rax
  size_t v102; // rdx
  _WORD *v103; // rdi
  unsigned __int8 *v104; // rsi
  unsigned __int64 v105; // rax
  __int64 v106; // rax
  __m128i *v107; // rdx
  __int64 v108; // rdi
  __m128i v109; // xmm0
  __int64 v110; // rdi
  _BYTE *v111; // rax
  _QWORD *v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rbx
  const char *v115; // rax
  size_t v116; // rdx
  _WORD *v117; // rdi
  unsigned __int8 *v118; // rsi
  unsigned __int64 v119; // rax
  __int64 v120; // rax
  __m128i *v121; // rdx
  __int64 v122; // rdi
  __m128i v123; // xmm0
  __int64 v124; // rdi
  _BYTE *v125; // rax
  _QWORD *v126; // rax
  __m128i *v127; // rdx
  __int64 v128; // rbx
  __m128i v129; // xmm0
  const char *v130; // rax
  size_t v131; // rdx
  _BYTE *v132; // rdi
  unsigned __int8 *v133; // rsi
  unsigned __int64 v134; // rax
  _QWORD *v135; // rax
  __int64 v136; // rdx
  __int64 v137; // rbx
  const char *v138; // rax
  size_t v139; // rdx
  _WORD *v140; // rdi
  unsigned __int8 *v141; // rsi
  unsigned __int64 v142; // rax
  __int64 v143; // rax
  __m128i *v144; // rdx
  __m128i v145; // xmm0
  int v146; // ecx
  __int64 v147; // r8
  unsigned int v148; // esi
  __int64 v149; // rax
  _QWORD *v151; // rax
  _BYTE *v152; // rdx
  _QWORD *v153; // rax
  __int64 v154; // rdx
  __int64 v155; // rbx
  const char *v156; // rax
  size_t v157; // rdx
  _WORD *v158; // rdi
  unsigned __int8 *v159; // rsi
  unsigned __int64 v160; // rax
  __int64 v161; // rax
  __m128i *v162; // rdx
  __m128i v163; // xmm0
  int v164; // ecx
  __int64 v165; // r8
  unsigned int v166; // esi
  __int64 v167; // rax
  _QWORD *v169; // rax
  _BYTE *v170; // rdx
  unsigned int v171; // eax
  __int64 v172; // r14
  __int64 v173; // r13
  __int64 v174; // rbx
  unsigned __int64 v175; // rdi
  unsigned __int64 v176; // rdi
  size_t v177; // rdx
  __int64 *v178; // rcx
  __int64 *v179; // rax
  __int64 *v180; // rdx
  __int64 v181; // rcx
  __int64 *v182; // rdx
  __int64 v183; // rbx
  _QWORD *v184; // rax
  int i; // ebx
  void *v187; // rax
  __int64 v188; // rdi
  _BYTE *v189; // rax
  int v190; // edx
  unsigned int v191; // eax
  unsigned int v192; // edi
  unsigned int v193; // esi
  __int64 v194; // r8
  int v195; // ecx
  unsigned __int64 v196; // rax
  __int64 v197; // rdx
  unsigned __int64 v198; // r9
  int j; // ebx
  void *v203; // rax
  __int64 v204; // rdi
  _BYTE *v205; // rax
  int v206; // edx
  unsigned int v207; // eax
  unsigned int v208; // edi
  unsigned int v209; // esi
  __int64 v210; // r8
  int v211; // ecx
  unsigned __int64 v212; // rax
  __int64 v213; // rdx
  unsigned __int64 v214; // r9
  __int64 v217; // rax
  __int64 v218; // rax
  __int64 v219; // rax
  __int64 v220; // rax
  __int64 v221; // rax
  __int64 v222; // rax
  __int64 v223; // rax
  __int64 v224; // rax
  __int64 v225; // rax
  __int64 v226; // rax
  __int64 v227; // rax
  __int64 v228; // rax
  char v229; // dl
  unsigned __int64 v230; // rdx
  __int64 v231; // rcx
  __int64 v232; // r8
  __int64 v233; // r9
  __int64 v234; // rax
  __int64 v235; // rax
  __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rax
  __int64 v239; // rax
  unsigned __int64 v240; // r15
  __int64 v241; // rdx
  __int64 v242; // r13
  __int64 v243; // r12
  unsigned __int64 v244; // rdi
  unsigned __int64 v245; // rdi
  unsigned __int64 v246; // rdx
  bool v247; // cf
  unsigned __int64 v248; // rax
  unsigned __int64 v249; // rbx
  __int64 v250; // rax
  __int64 v251; // rdx
  __int64 v252; // rcx
  __int64 v253; // r8
  __int64 v254; // r14
  __int64 v255; // r13
  unsigned __int64 v256; // rsi
  unsigned __int64 v257; // r15
  __int64 v258; // r14
  int v259; // eax
  int v260; // eax
  __int64 v261; // r9
  __int64 v262; // r14
  unsigned __int64 v263; // rdi
  unsigned __int64 v264; // rdi
  size_t v266; // [rsp+30h] [rbp-340h]
  size_t v267; // [rsp+30h] [rbp-340h]
  size_t v268; // [rsp+30h] [rbp-340h]
  size_t v269; // [rsp+30h] [rbp-340h]
  size_t v270; // [rsp+30h] [rbp-340h]
  size_t v271; // [rsp+30h] [rbp-340h]
  size_t v272; // [rsp+30h] [rbp-340h]
  size_t v273; // [rsp+30h] [rbp-340h]
  size_t v274; // [rsp+30h] [rbp-340h]
  __int32 v275; // [rsp+3Ch] [rbp-334h]
  __int64 v276; // [rsp+40h] [rbp-330h]
  __int64 v277; // [rsp+40h] [rbp-330h]
  __m128i v279[8]; // [rsp+50h] [rbp-320h] BYREF
  __m128i v280; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v281; // [rsp+E0h] [rbp-290h]
  int v282; // [rsp+E8h] [rbp-288h]
  char v283; // [rsp+ECh] [rbp-284h]
  _QWORD v284[8]; // [rsp+F0h] [rbp-280h] BYREF
  unsigned __int64 v285; // [rsp+130h] [rbp-240h] BYREF
  unsigned __int64 v286; // [rsp+138h] [rbp-238h]
  unsigned __int64 v287; // [rsp+140h] [rbp-230h]
  __m128i v288; // [rsp+150h] [rbp-220h] BYREF
  unsigned int v289; // [rsp+160h] [rbp-210h]
  unsigned int v290; // [rsp+164h] [rbp-20Ch]
  char v291; // [rsp+16Ch] [rbp-204h]
  _BYTE v292[64]; // [rsp+170h] [rbp-200h] BYREF
  unsigned __int64 v293; // [rsp+1B0h] [rbp-1C0h] BYREF
  unsigned __int64 v294; // [rsp+1B8h] [rbp-1B8h]
  unsigned __int64 v295; // [rsp+1C0h] [rbp-1B0h]
  __m128i v296; // [rsp+1D0h] [rbp-1A0h] BYREF
  char v297; // [rsp+1ECh] [rbp-184h]
  _WORD v298[32]; // [rsp+1F0h] [rbp-180h] BYREF
  unsigned __int128 v299; // [rsp+230h] [rbp-140h]
  __int64 v300; // [rsp+240h] [rbp-130h]
  __m128i v301; // [rsp+250h] [rbp-120h] BYREF
  char v302; // [rsp+260h] [rbp-110h]
  char v303; // [rsp+26Ch] [rbp-104h]
  _BYTE v304[64]; // [rsp+270h] [rbp-100h] BYREF
  __m128i *v305; // [rsp+2B0h] [rbp-C0h]
  __int64 v306; // [rsp+2B8h] [rbp-B8h]
  unsigned __int64 v307; // [rsp+2C0h] [rbp-B0h]
  char v308[8]; // [rsp+2C8h] [rbp-A8h] BYREF
  unsigned __int64 v309; // [rsp+2D0h] [rbp-A0h]
  char v310; // [rsp+2E4h] [rbp-8Ch]
  char v311[64]; // [rsp+2E8h] [rbp-88h] BYREF
  const __m128i *v312; // [rsp+328h] [rbp-48h]
  const __m128i *v313; // [rsp+330h] [rbp-40h]
  __int64 v314; // [rsp+338h] [rbp-38h]

  LOBYTE(v2) = sub_2E791F0(a2);
  v3 = v2;
  if ( !(_BYTE)v2 )
    return v3;
  v5 = a1[26];
  v6 = a1[25];
  v7 = (unsigned int)((a2[13] - a2[12]) >> 3);
  v8 = v5 - v6;
  v9 = 0xD37A6F4DE9BD37A7LL * ((v5 - v6) >> 3);
  if ( v7 > v9 )
  {
    v240 = v7 - v9;
    if ( v7 - v9 > 0xD37A6F4DE9BD37A7LL * ((a1[27] - v5) >> 3) )
    {
      if ( v240 > 0xB21642C8590B21LL - v9 )
        sub_4262D8((__int64)"vector::_M_default_append");
      v246 = v7 - v9;
      if ( v9 >= v240 )
        v246 = 0xD37A6F4DE9BD37A7LL * (v8 >> 3);
      v247 = __CFADD__(v246, v9);
      v248 = v246 - 0x2C8590B21642C859LL * (v8 >> 3);
      if ( v247 )
      {
        v249 = 0x7FFFFFFFFFFFFFB8LL;
      }
      else
      {
        if ( v248 > 0xB21642C8590B21LL )
          v248 = 0xB21642C8590B21LL;
        v249 = 184 * v248;
      }
      v250 = sub_22077B0(v249);
      v254 = v250 + v8;
      v255 = v250;
      v256 = v254 + 184 * v240;
      do
      {
        if ( v254 )
        {
          memset((void *)v254, 0, 0xB8u);
          v252 = 0;
          *(_QWORD *)(v254 + 32) = v254 + 48;
          v251 = v254 + 120;
          *(_QWORD *)(v254 + 8) = -1;
          *(_QWORD *)(v254 + 16) = -1;
          *(_DWORD *)(v254 + 44) = 6;
          *(_QWORD *)(v254 + 104) = v254 + 120;
          *(_DWORD *)(v254 + 116) = 6;
        }
        v254 += 184;
      }
      while ( v254 != v256 );
      v257 = a1[25];
      v277 = a1[26];
      if ( v277 != v257 )
      {
        v258 = v250;
        do
        {
          if ( v258 )
          {
            *(_QWORD *)v258 = *(_QWORD *)v257;
            *(_QWORD *)(v258 + 8) = *(_QWORD *)(v257 + 8);
            *(_QWORD *)(v258 + 16) = *(_QWORD *)(v257 + 16);
            *(_DWORD *)(v258 + 24) = *(_DWORD *)(v257 + 24);
            v260 = *(_DWORD *)(v257 + 28);
            *(_DWORD *)(v258 + 40) = 0;
            *(_DWORD *)(v258 + 28) = v260;
            *(_QWORD *)(v258 + 32) = v258 + 48;
            *(_DWORD *)(v258 + 44) = 6;
            v261 = *(unsigned int *)(v257 + 40);
            if ( (_DWORD)v261 )
              sub_37EBA00(v258 + 32, v257 + 32, v251, v252, v253, v261);
            v259 = *(_DWORD *)(v257 + 96);
            *(_DWORD *)(v258 + 112) = 0;
            *(_DWORD *)(v258 + 116) = 6;
            *(_DWORD *)(v258 + 96) = v259;
            *(_QWORD *)(v258 + 104) = v258 + 120;
            v253 = *(unsigned int *)(v257 + 112);
            if ( (_DWORD)v253 )
              sub_37EBA00(v258 + 104, v257 + 104, v251, v252, v253, v261);
            *(_DWORD *)(v258 + 168) = *(_DWORD *)(v257 + 168);
            *(_BYTE *)(v258 + 176) = *(_BYTE *)(v257 + 176);
          }
          v257 += 184LL;
          v258 += 184;
        }
        while ( v277 != v257 );
        v262 = a1[26];
        v257 = a1[25];
        if ( v262 != v257 )
        {
          do
          {
            v263 = *(_QWORD *)(v257 + 104);
            if ( v263 != v257 + 120 )
              _libc_free(v263);
            v264 = *(_QWORD *)(v257 + 32);
            if ( v264 != v257 + 48 )
              _libc_free(v264);
            v257 += 184LL;
          }
          while ( v262 != v257 );
          v257 = a1[25];
        }
      }
      if ( v257 )
        j_j___libc_free_0(v257);
      a1[25] = v255;
      a1[27] = v255 + v249;
      a1[26] = v255 + 184 * v7;
    }
    else
    {
      v241 = v5 + 184 * v240;
      do
      {
        if ( v5 )
        {
          memset((void *)v5, 0, 0xB8u);
          *(_QWORD *)(v5 + 8) = -1;
          *(_QWORD *)(v5 + 32) = v5 + 48;
          *(_QWORD *)(v5 + 16) = -1;
          *(_DWORD *)(v5 + 44) = 6;
          *(_QWORD *)(v5 + 104) = v5 + 120;
          *(_DWORD *)(v5 + 116) = 6;
        }
        v5 += 184;
      }
      while ( v5 != v241 );
      a1[26] = v5;
    }
  }
  else if ( v7 < v9 )
  {
    v242 = 184 * v7 + v6;
    if ( v5 != v242 )
    {
      v243 = v242;
      do
      {
        v244 = *(_QWORD *)(v243 + 104);
        if ( v244 != v243 + 120 )
          _libc_free(v244);
        v245 = *(_QWORD *)(v243 + 32);
        if ( v245 != v243 + 48 )
          _libc_free(v245);
        v243 += 184;
      }
      while ( v5 != v243 );
      a1[26] = v242;
    }
  }
  sub_37ED710((size_t)a1, (__int64)a2);
  if ( !(_BYTE)qword_50513E8 )
    goto LABEL_157;
  memset(v279, 0, 0x78u);
  v281 = 0x100000008LL;
  v284[0] = a2[41];
  v301.m128i_i64[0] = v284[0];
  v279[0].m128i_i64[1] = (__int64)v279[2].m128i_i64;
  v280.m128i_i64[1] = (__int64)v284;
  v279[1].m128i_i32[0] = 8;
  v279[1].m128i_i8[12] = 1;
  v285 = 0;
  v286 = 0;
  v287 = 0;
  v282 = 0;
  v283 = 1;
  v280.m128i_i64[0] = 1;
  v302 = 0;
  sub_37EEB50(&v285, &v301);
  sub_C8CF70((__int64)&v296, v298, 8, (__int64)v279[2].m128i_i64, (__int64)v279);
  v10 = v279[6].m128i_i64[0];
  memset(&v279[6], 0, 24);
  v299 = __PAIR128__(v279[6].m128i_u64[1], v10);
  v300 = v279[7].m128i_i64[0];
  sub_C8CF70((__int64)&v288, v292, 8, (__int64)v284, (__int64)&v280);
  v11 = v285;
  v285 = 0;
  v293 = v11;
  v12 = v286;
  v286 = 0;
  v294 = v12;
  v13 = v287;
  v287 = 0;
  v295 = v13;
  sub_C8CF70((__int64)&v301, v304, 8, (__int64)v292, (__int64)&v288);
  v14 = v293;
  v293 = 0;
  v305 = (__m128i *)v14;
  v15 = v294;
  v294 = 0;
  v306 = v15;
  v16 = v295;
  v295 = 0;
  v307 = v16;
  sub_C8CF70((__int64)v308, v311, 8, (__int64)v298, (__int64)&v296);
  v20 = (const __m128i *)*((_QWORD *)&v299 + 1);
  v312 = (const __m128i *)v299;
  v299 = 0u;
  v313 = v20;
  v21 = v300;
  v300 = 0;
  v314 = v21;
  if ( v293 )
    j_j___libc_free_0(v293);
  if ( !v291 )
    _libc_free(v288.m128i_u64[1]);
  if ( (_QWORD)v299 )
    j_j___libc_free_0(v299);
  if ( !v297 )
    _libc_free(v296.m128i_u64[1]);
  if ( v285 )
    j_j___libc_free_0(v285);
  if ( !v283 )
    _libc_free(v280.m128i_u64[1]);
  if ( v279[6].m128i_i64[0] )
    j_j___libc_free_0(v279[6].m128i_u64[0]);
  if ( !v279[1].m128i_i8[12] )
    _libc_free(v279[0].m128i_u64[1]);
  v22 = (const __m128i *)v292;
  sub_C8CD80((__int64)&v288, (__int64)v292, (__int64)&v301, v17, v18, v19);
  v26 = v306;
  v27 = v305;
  v293 = 0;
  v294 = 0;
  v295 = 0;
  v28 = v306 - (_QWORD)v305;
  if ( (__m128i *)v306 == v305 )
  {
    v28 = 0;
    v30 = 0;
  }
  else
  {
    if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_336;
    v29 = sub_22077B0(v306 - (_QWORD)v305);
    v26 = v306;
    v27 = v305;
    v30 = v29;
  }
  v293 = v30;
  v294 = v30;
  v295 = v30 + v28;
  if ( (__m128i *)v26 != v27 )
  {
    v31 = (__m128i *)v30;
    v32 = v27;
    do
    {
      if ( v31 )
      {
        *v31 = _mm_loadu_si128(v32);
        v24 = v32[1].m128i_i64[0];
        v31[1].m128i_i64[0] = v24;
      }
      v32 = (const __m128i *)((char *)v32 + 24);
      v31 = (__m128i *)((char *)v31 + 24);
    }
    while ( v32 != (const __m128i *)v26 );
    v30 += 8 * ((unsigned __int64)((char *)&v32[-2].m128i_u64[1] - (char *)v27) >> 3) + 24;
  }
  v27 = &v296;
  v294 = v30;
  sub_C8CD80((__int64)&v296, (__int64)v298, (__int64)v308, v26, v24, v25);
  v35 = v313;
  v22 = v312;
  v299 = 0u;
  v300 = 0;
  v36 = (char *)v313 - (char *)v312;
  if ( v313 != v312 )
  {
    if ( v36 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v37 = sub_22077B0((char *)v313 - (char *)v312);
      v35 = v313;
      v22 = v312;
      v38 = v37;
      goto LABEL_34;
    }
LABEL_336:
    sub_4261EA(v27, v22, v23);
  }
  v36 = 0;
  v38 = 0;
LABEL_34:
  *(_QWORD *)&v299 = v38;
  v39 = (__m128i *)v38;
  *((_QWORD *)&v299 + 1) = v38;
  v300 = v38 + v36;
  if ( v35 != v22 )
  {
    v40 = v22;
    do
    {
      if ( v39 )
      {
        *v39 = _mm_loadu_si128(v40);
        v33 = v40[1].m128i_i64[0];
        v39[1].m128i_i64[0] = v33;
      }
      v40 = (const __m128i *)((char *)v40 + 24);
      v39 = (__m128i *)((char *)v39 + 24);
    }
    while ( v40 != v35 );
    v39 = (__m128i *)(v38 + 8 * ((unsigned __int64)((char *)&v40[-2].m128i_u64[1] - (char *)v22) >> 3) + 24);
  }
  v41 = v294;
  v42 = v293;
  *((_QWORD *)&v299 + 1) = v39;
  v275 = 0;
  v43 = v294 - v293;
  if ( (__m128i *)(v294 - v293) == (__m128i *)((char *)v39 - v38) )
    goto LABEL_248;
  do
  {
LABEL_41:
    v44 = *(_QWORD *)(v41 - 24);
    v45 = a1[25];
    v46 = *(_QWORD *)(v44 + 112);
    v276 = v46 + 8LL * *(unsigned int *)(v44 + 120);
    v47 = v45 + 184LL * *(int *)(v44 + 24);
    if ( v46 != v276 )
    {
      v48 = *(_QWORD *)(v44 + 112);
      while ( 1 )
      {
        v49 = (__int64 *)(184LL * *(int *)(*(_QWORD *)v48 + 24LL) + v45);
        if ( v49[1] == *(_QWORD *)(v47 + 16) && *((_DWORD *)v49 + 6) == *(_DWORD *)(v47 + 28) )
          goto LABEL_112;
        v50 = *v49;
        if ( !*(_DWORD *)(*v49 + 120) )
          break;
LABEL_55:
        v57 = sub_CB72A0();
        v58 = (__m128i *)v57[4];
        if ( v57[3] - (_QWORD)v58 <= 0x45u )
        {
          sub_CB6200((__int64)v57, "*** Inconsistent CFA register and/or offset between pred and succ ***\n", 0x46u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_430AEC0);
          v58[4].m128i_i32[0] = 707403875;
          v58[4].m128i_i16[2] = 2602;
          *v58 = si128;
          v58[1] = _mm_load_si128((const __m128i *)&xmmword_430AED0);
          v58[2] = _mm_load_si128((const __m128i *)&xmmword_430AEE0);
          v58[3] = _mm_load_si128((const __m128i *)&xmmword_430AEF0);
          v57[4] += 70LL;
        }
        v60 = sub_CB72A0();
        v61 = v60[4];
        v62 = (__int64)v60;
        if ( (unsigned __int64)(v60[3] - v61) <= 5 )
        {
          v62 = sub_CB6200((__int64)v60, "Pred: ", 6u);
        }
        else
        {
          *(_DWORD *)v61 = 1684370000;
          *(_WORD *)(v61 + 4) = 8250;
          v60[4] += 6LL;
        }
        v63 = sub_2E31BC0(*(_QWORD *)v47);
        v65 = *(_WORD **)(v62 + 32);
        v66 = (unsigned __int8 *)v63;
        v67 = *(_QWORD *)(v62 + 24) - (_QWORD)v65;
        if ( v67 < v64 )
        {
          v217 = sub_CB6200(v62, v66, v64);
          v65 = *(_WORD **)(v217 + 32);
          v62 = v217;
          if ( *(_QWORD *)(v217 + 24) - (_QWORD)v65 > 1u )
            goto LABEL_63;
        }
        else
        {
          if ( v64 )
          {
            v269 = v64;
            memcpy(v65, v66, v64);
            v226 = *(_QWORD *)(v62 + 24);
            v65 = (_WORD *)(v269 + *(_QWORD *)(v62 + 32));
            *(_QWORD *)(v62 + 32) = v65;
            v67 = v226 - (_QWORD)v65;
          }
          if ( v67 > 1 )
          {
LABEL_63:
            *v65 = 8992;
            *(_QWORD *)(v62 + 32) += 2LL;
            goto LABEL_64;
          }
        }
        v62 = sub_CB6200(v62, (unsigned __int8 *)" #", 2u);
LABEL_64:
        v68 = sub_CB59F0(v62, *(int *)(*(_QWORD *)v47 + 24LL));
        v69 = *(_DWORD **)(v68 + 32);
        v70 = v68;
        if ( *(_QWORD *)(v68 + 24) - (_QWORD)v69 <= 3u )
        {
          v70 = sub_CB6200(v68, (unsigned __int8 *)" in ", 4u);
        }
        else
        {
          *v69 = 544106784;
          *(_QWORD *)(v68 + 32) += 4LL;
        }
        v71 = sub_2E791E0(*(__int64 **)(*(_QWORD *)v47 + 32LL));
        v73 = *(__m128i **)(v70 + 32);
        v74 = (unsigned __int8 *)v71;
        v75 = *(_QWORD *)(v70 + 24) - (_QWORD)v73;
        if ( v75 < v72 )
        {
          v219 = sub_CB6200(v70, v74, v72);
          v73 = *(__m128i **)(v219 + 32);
          v70 = v219;
          v75 = *(_QWORD *)(v219 + 24) - (_QWORD)v73;
        }
        else if ( v72 )
        {
          v268 = v72;
          memcpy(v73, v74, v72);
          v225 = *(_QWORD *)(v70 + 24);
          v73 = (__m128i *)(v268 + *(_QWORD *)(v70 + 32));
          *(_QWORD *)(v70 + 32) = v73;
          v75 = v225 - (_QWORD)v73;
        }
        if ( v75 <= 0x11 )
        {
          v70 = sub_CB6200(v70, " outgoing CFA Reg:", 0x12u);
        }
        else
        {
          v76 = _mm_load_si128((const __m128i *)&xmmword_430AF00);
          v73[1].m128i_i16[0] = 14951;
          *v73 = v76;
          *(_QWORD *)(v70 + 32) += 18LL;
        }
        v77 = sub_CB59D0(v70, *(unsigned int *)(v47 + 28));
        v78 = *(_BYTE **)(v77 + 32);
        if ( *(_BYTE **)(v77 + 24) == v78 )
        {
          sub_CB6200(v77, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v78 = 10;
          ++*(_QWORD *)(v77 + 32);
        }
        v79 = sub_CB72A0();
        v80 = v79[4];
        v81 = (__int64)v79;
        if ( (unsigned __int64)(v79[3] - v80) <= 5 )
        {
          v81 = sub_CB6200((__int64)v79, "Pred: ", 6u);
        }
        else
        {
          *(_DWORD *)v80 = 1684370000;
          *(_WORD *)(v80 + 4) = 8250;
          v79[4] += 6LL;
        }
        v82 = sub_2E31BC0(*(_QWORD *)v47);
        v84 = *(_WORD **)(v81 + 32);
        v85 = (unsigned __int8 *)v82;
        v86 = *(_QWORD *)(v81 + 24) - (_QWORD)v84;
        if ( v86 < v83 )
        {
          v218 = sub_CB6200(v81, v85, v83);
          v84 = *(_WORD **)(v218 + 32);
          v81 = v218;
          if ( *(_QWORD *)(v218 + 24) - (_QWORD)v84 <= 1u )
            goto LABEL_223;
        }
        else
        {
          if ( v83 )
          {
            v271 = v83;
            memcpy(v84, v85, v83);
            v228 = *(_QWORD *)(v81 + 24);
            v84 = (_WORD *)(v271 + *(_QWORD *)(v81 + 32));
            *(_QWORD *)(v81 + 32) = v84;
            v86 = v228 - (_QWORD)v84;
          }
          if ( v86 <= 1 )
          {
LABEL_223:
            v81 = sub_CB6200(v81, (unsigned __int8 *)" #", 2u);
            goto LABEL_80;
          }
        }
        *v84 = 8992;
        *(_QWORD *)(v81 + 32) += 2LL;
LABEL_80:
        v87 = sub_CB59F0(v81, *(int *)(*(_QWORD *)v47 + 24LL));
        v88 = *(_DWORD **)(v87 + 32);
        v89 = v87;
        if ( *(_QWORD *)(v87 + 24) - (_QWORD)v88 <= 3u )
        {
          v89 = sub_CB6200(v87, (unsigned __int8 *)" in ", 4u);
        }
        else
        {
          *v88 = 544106784;
          *(_QWORD *)(v87 + 32) += 4LL;
        }
        v90 = sub_2E791E0(*(__int64 **)(*(_QWORD *)v47 + 32LL));
        v92 = *(__m128i **)(v89 + 32);
        v93 = (unsigned __int8 *)v90;
        v94 = *(_QWORD *)(v89 + 24) - (_QWORD)v92;
        if ( v94 < v91 )
        {
          v222 = sub_CB6200(v89, v93, v91);
          v92 = *(__m128i **)(v222 + 32);
          v89 = v222;
          v94 = *(_QWORD *)(v222 + 24) - (_QWORD)v92;
        }
        else if ( v91 )
        {
          v270 = v91;
          memcpy(v92, v93, v91);
          v227 = *(_QWORD *)(v89 + 24);
          v92 = (__m128i *)(v270 + *(_QWORD *)(v89 + 32));
          *(_QWORD *)(v89 + 32) = v92;
          v94 = v227 - (_QWORD)v92;
        }
        if ( v94 <= 0x14 )
        {
          v89 = sub_CB6200(v89, " outgoing CFA Offset:", 0x15u);
        }
        else
        {
          v95 = _mm_load_si128((const __m128i *)&xmmword_430AF10);
          v92[1].m128i_i32[0] = 1952805734;
          v92[1].m128i_i8[4] = 58;
          *v92 = v95;
          *(_QWORD *)(v89 + 32) += 21LL;
        }
        v96 = sub_CB59F0(v89, *(_QWORD *)(v47 + 16));
        v97 = *(_BYTE **)(v96 + 32);
        if ( *(_BYTE **)(v96 + 24) == v97 )
        {
          sub_CB6200(v96, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v97 = 10;
          ++*(_QWORD *)(v96 + 32);
        }
        v98 = sub_CB72A0();
        v99 = v98[4];
        v100 = (__int64)v98;
        if ( (unsigned __int64)(v98[3] - v99) <= 5 )
        {
          v100 = sub_CB6200((__int64)v98, "Succ: ", 6u);
        }
        else
        {
          *(_DWORD *)v99 = 1667462483;
          *(_WORD *)(v99 + 4) = 8250;
          v98[4] += 6LL;
        }
        v101 = sub_2E31BC0(*v49);
        v103 = *(_WORD **)(v100 + 32);
        v104 = (unsigned __int8 *)v101;
        v105 = *(_QWORD *)(v100 + 24) - (_QWORD)v103;
        if ( v102 > v105 )
        {
          v221 = sub_CB6200(v100, v104, v102);
          v103 = *(_WORD **)(v221 + 32);
          v100 = v221;
          v105 = *(_QWORD *)(v221 + 24) - (_QWORD)v103;
        }
        else if ( v102 )
        {
          v267 = v102;
          memcpy(v103, v104, v102);
          v224 = *(_QWORD *)(v100 + 24);
          v103 = (_WORD *)(v267 + *(_QWORD *)(v100 + 32));
          *(_QWORD *)(v100 + 32) = v103;
          v105 = v224 - (_QWORD)v103;
        }
        if ( v105 <= 1 )
        {
          v100 = sub_CB6200(v100, (unsigned __int8 *)" #", 2u);
        }
        else
        {
          *v103 = 8992;
          *(_QWORD *)(v100 + 32) += 2LL;
        }
        v106 = sub_CB59F0(v100, *(int *)(*v49 + 24));
        v107 = *(__m128i **)(v106 + 32);
        v108 = v106;
        if ( *(_QWORD *)(v106 + 24) - (_QWORD)v107 <= 0x11u )
        {
          v108 = sub_CB6200(v106, " incoming CFA Reg:", 0x12u);
        }
        else
        {
          v109 = _mm_load_si128((const __m128i *)&xmmword_430AF20);
          v107[1].m128i_i16[0] = 14951;
          *v107 = v109;
          *(_QWORD *)(v106 + 32) += 18LL;
        }
        v110 = sub_CB59D0(v108, *((unsigned int *)v49 + 6));
        v111 = *(_BYTE **)(v110 + 32);
        if ( *(_BYTE **)(v110 + 24) == v111 )
        {
          sub_CB6200(v110, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v111 = 10;
          ++*(_QWORD *)(v110 + 32);
        }
        v112 = sub_CB72A0();
        v113 = v112[4];
        v114 = (__int64)v112;
        if ( (unsigned __int64)(v112[3] - v113) <= 5 )
        {
          v114 = sub_CB6200((__int64)v112, "Succ: ", 6u);
        }
        else
        {
          *(_DWORD *)v113 = 1667462483;
          *(_WORD *)(v113 + 4) = 8250;
          v112[4] += 6LL;
        }
        v115 = sub_2E31BC0(*v49);
        v117 = *(_WORD **)(v114 + 32);
        v118 = (unsigned __int8 *)v115;
        v119 = *(_QWORD *)(v114 + 24) - (_QWORD)v117;
        if ( v116 > v119 )
        {
          v220 = sub_CB6200(v114, v118, v116);
          v117 = *(_WORD **)(v220 + 32);
          v114 = v220;
          v119 = *(_QWORD *)(v220 + 24) - (_QWORD)v117;
        }
        else if ( v116 )
        {
          v266 = v116;
          memcpy(v117, v118, v116);
          v223 = *(_QWORD *)(v114 + 24);
          v117 = (_WORD *)(v266 + *(_QWORD *)(v114 + 32));
          *(_QWORD *)(v114 + 32) = v117;
          v119 = v223 - (_QWORD)v117;
        }
        if ( v119 <= 1 )
        {
          v114 = sub_CB6200(v114, (unsigned __int8 *)" #", 2u);
        }
        else
        {
          *v117 = 8992;
          *(_QWORD *)(v114 + 32) += 2LL;
        }
        v120 = sub_CB59F0(v114, *(int *)(*v49 + 24));
        v121 = *(__m128i **)(v120 + 32);
        v122 = v120;
        if ( *(_QWORD *)(v120 + 24) - (_QWORD)v121 <= 0x14u )
        {
          v122 = sub_CB6200(v120, " incoming CFA Offset:", 0x15u);
        }
        else
        {
          v123 = _mm_load_si128((const __m128i *)&xmmword_430AF30);
          v121[1].m128i_i32[0] = 1952805734;
          v121[1].m128i_i8[4] = 58;
          *v121 = v123;
          *(_QWORD *)(v120 + 32) += 21LL;
        }
        v124 = sub_CB59F0(v122, v49[1]);
        v125 = *(_BYTE **)(v124 + 32);
        if ( *(_BYTE **)(v124 + 24) == v125 )
        {
          sub_CB6200(v124, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v125 = 10;
          ++*(_QWORD *)(v124 + 32);
        }
        ++v275;
LABEL_112:
        if ( *((_DWORD *)v49 + 24) != *(_DWORD *)(v47 + 168) )
          goto LABEL_113;
        v177 = 8LL * *((unsigned int *)v49 + 10);
        if ( !v177 )
          goto LABEL_155;
        if ( memcmp((const void *)v49[4], *(const void **)(v47 + 104), v177) )
        {
LABEL_113:
          v126 = sub_CB72A0();
          v127 = (__m128i *)v126[4];
          v128 = (__int64)v126;
          if ( v126[3] - (_QWORD)v127 <= 0x3Cu )
          {
            v128 = sub_CB6200((__int64)v126, "*** Inconsistent CSR Saved between pred and succ in function ", 0x3Du);
          }
          else
          {
            v129 = _mm_load_si128((const __m128i *)&xmmword_430AEC0);
            qmemcpy(&v127[3], " in function ", 13);
            *v127 = v129;
            v127[1] = _mm_load_si128((const __m128i *)&xmmword_4527740);
            v127[2] = _mm_load_si128((const __m128i *)&xmmword_4527750);
            v126[4] += 61LL;
          }
          v130 = sub_2E791E0(*(__int64 **)(*(_QWORD *)v47 + 32LL));
          v132 = *(_BYTE **)(v128 + 32);
          v133 = (unsigned __int8 *)v130;
          v134 = *(_QWORD *)(v128 + 24) - (_QWORD)v132;
          if ( v134 < v131 )
          {
            v236 = sub_CB6200(v128, v133, v131);
            v132 = *(_BYTE **)(v236 + 32);
            v128 = v236;
            v134 = *(_QWORD *)(v236 + 24) - (_QWORD)v132;
          }
          else if ( v131 )
          {
            v274 = v131;
            memcpy(v132, v133, v131);
            v239 = *(_QWORD *)(v128 + 24);
            v132 = (_BYTE *)(v274 + *(_QWORD *)(v128 + 32));
            *(_QWORD *)(v128 + 32) = v132;
            v134 = v239 - (_QWORD)v132;
          }
          if ( v134 <= 4 )
          {
            sub_CB6200(v128, (unsigned __int8 *)" ***\n", 5u);
          }
          else
          {
            *(_DWORD *)v132 = 707406368;
            v132[4] = 10;
            *(_QWORD *)(v128 + 32) += 5LL;
          }
          v135 = sub_CB72A0();
          v136 = v135[4];
          v137 = (__int64)v135;
          if ( (unsigned __int64)(v135[3] - v136) <= 5 )
          {
            v137 = sub_CB6200((__int64)v135, "Pred: ", 6u);
          }
          else
          {
            *(_DWORD *)v136 = 1684370000;
            *(_WORD *)(v136 + 4) = 8250;
            v135[4] += 6LL;
          }
          v138 = sub_2E31BC0(*(_QWORD *)v47);
          v140 = *(_WORD **)(v137 + 32);
          v141 = (unsigned __int8 *)v138;
          v142 = *(_QWORD *)(v137 + 24) - (_QWORD)v140;
          if ( v139 > v142 )
          {
            v235 = sub_CB6200(v137, v141, v139);
            v140 = *(_WORD **)(v235 + 32);
            v137 = v235;
            v142 = *(_QWORD *)(v235 + 24) - (_QWORD)v140;
          }
          else if ( v139 )
          {
            v273 = v139;
            memcpy(v140, v141, v139);
            v238 = *(_QWORD *)(v137 + 24);
            v140 = (_WORD *)(v273 + *(_QWORD *)(v137 + 32));
            *(_QWORD *)(v137 + 32) = v140;
            v142 = v238 - (_QWORD)v140;
          }
          if ( v142 <= 1 )
          {
            v137 = sub_CB6200(v137, (unsigned __int8 *)" #", 2u);
          }
          else
          {
            *v140 = 8992;
            *(_QWORD *)(v137 + 32) += 2LL;
          }
          v143 = sub_CB59F0(v137, *(int *)(*(_QWORD *)v47 + 24LL));
          v144 = *(__m128i **)(v143 + 32);
          if ( *(_QWORD *)(v143 + 24) - (_QWORD)v144 <= 0x14u )
          {
            sub_CB6200(v143, " outgoing CSR Saved: ", 0x15u);
          }
          else
          {
            v145 = _mm_load_si128((const __m128i *)&xmmword_4527760);
            v144[1].m128i_i32[0] = 979658102;
            v144[1].m128i_i8[4] = 32;
            *v144 = v145;
            *(_QWORD *)(v143 + 32) += 21LL;
          }
          v146 = *(_DWORD *)(v47 + 168);
          if ( v146 )
          {
            v147 = *(_QWORD *)(v47 + 104);
            v148 = (unsigned int)(v146 - 1) >> 6;
            v149 = 0;
            while ( 1 )
            {
              _RDX = *(_QWORD *)(v147 + 8 * v149);
              if ( v148 == (_DWORD)v149 )
                _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v146) & *(_QWORD *)(v147 + 8 * v149);
              if ( _RDX )
                break;
              if ( ++v149 == v148 + 1 )
                goto LABEL_135;
            }
            __asm { tzcnt   rdx, rdx }
            for ( i = _RDX + ((_DWORD)v149 << 6); i != -1; i = ((_DWORD)v197 << 6) + _RAX )
            {
              v187 = sub_CB72A0();
              v188 = sub_CB59F0((__int64)v187, i);
              v189 = *(_BYTE **)(v188 + 32);
              if ( *(_BYTE **)(v188 + 24) == v189 )
              {
                sub_CB6200(v188, (unsigned __int8 *)" ", 1u);
              }
              else
              {
                *v189 = 32;
                ++*(_QWORD *)(v188 + 32);
              }
              v190 = *(_DWORD *)(v47 + 168);
              v191 = i + 1;
              if ( v190 == i + 1 )
                break;
              v192 = v191 >> 6;
              v193 = (unsigned int)(v190 - 1) >> 6;
              if ( v191 >> 6 > v193 )
                break;
              v194 = *(_QWORD *)(v47 + 104);
              v195 = 64 - (v191 & 0x3F);
              v196 = 0xFFFFFFFFFFFFFFFFLL >> v195;
              if ( v195 == 64 )
                v196 = 0;
              v197 = v192;
              v198 = ~v196;
              while ( 1 )
              {
                _RAX = *(_QWORD *)(v194 + 8 * v197);
                if ( v192 == (_DWORD)v197 )
                  _RAX = v198 & *(_QWORD *)(v194 + 8 * v197);
                if ( v193 == (_DWORD)v197 )
                  _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v47 + 168);
                if ( _RAX )
                  break;
                if ( v193 < (unsigned int)++v197 )
                  goto LABEL_135;
              }
              __asm { tzcnt   rax, rax }
            }
          }
LABEL_135:
          v151 = sub_CB72A0();
          v152 = (_BYTE *)v151[4];
          if ( (_BYTE *)v151[3] == v152 )
          {
            sub_CB6200((__int64)v151, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v152 = 10;
            ++v151[4];
          }
          v153 = sub_CB72A0();
          v154 = v153[4];
          v155 = (__int64)v153;
          if ( (unsigned __int64)(v153[3] - v154) <= 5 )
          {
            v155 = sub_CB6200((__int64)v153, "Succ: ", 6u);
          }
          else
          {
            *(_DWORD *)v154 = 1667462483;
            *(_WORD *)(v154 + 4) = 8250;
            v153[4] += 6LL;
          }
          v156 = sub_2E31BC0(*v49);
          v158 = *(_WORD **)(v155 + 32);
          v159 = (unsigned __int8 *)v156;
          v160 = *(_QWORD *)(v155 + 24) - (_QWORD)v158;
          if ( v157 > v160 )
          {
            v234 = sub_CB6200(v155, v159, v157);
            v158 = *(_WORD **)(v234 + 32);
            v155 = v234;
            v160 = *(_QWORD *)(v234 + 24) - (_QWORD)v158;
          }
          else if ( v157 )
          {
            v272 = v157;
            memcpy(v158, v159, v157);
            v237 = *(_QWORD *)(v155 + 24);
            v158 = (_WORD *)(v272 + *(_QWORD *)(v155 + 32));
            *(_QWORD *)(v155 + 32) = v158;
            v160 = v237 - (_QWORD)v158;
          }
          if ( v160 <= 1 )
          {
            v155 = sub_CB6200(v155, (unsigned __int8 *)" #", 2u);
          }
          else
          {
            *v158 = 8992;
            *(_QWORD *)(v155 + 32) += 2LL;
          }
          v161 = sub_CB59F0(v155, *(int *)(*v49 + 24));
          v162 = *(__m128i **)(v161 + 32);
          if ( *(_QWORD *)(v161 + 24) - (_QWORD)v162 <= 0x14u )
          {
            sub_CB6200(v161, " incoming CSR Saved: ", 0x15u);
          }
          else
          {
            v163 = _mm_load_si128((const __m128i *)&xmmword_4527770);
            v162[1].m128i_i32[0] = 979658102;
            v162[1].m128i_i8[4] = 32;
            *v162 = v163;
            *(_QWORD *)(v161 + 32) += 21LL;
          }
          v164 = *((_DWORD *)v49 + 24);
          if ( v164 )
          {
            v165 = v49[4];
            v166 = (unsigned int)(v164 - 1) >> 6;
            v167 = 0;
            while ( 1 )
            {
              _RDX = *(_QWORD *)(v165 + 8 * v167);
              if ( v166 == (_DWORD)v167 )
                _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v164) & *(_QWORD *)(v165 + 8 * v167);
              if ( _RDX )
                break;
              if ( v166 + 1 == ++v167 )
                goto LABEL_152;
            }
            __asm { tzcnt   rdx, rdx }
            for ( j = _RDX + ((_DWORD)v167 << 6); j != -1; j = ((_DWORD)v213 << 6) + _RAX )
            {
              v203 = sub_CB72A0();
              v204 = sub_CB59F0((__int64)v203, j);
              v205 = *(_BYTE **)(v204 + 32);
              if ( *(_BYTE **)(v204 + 24) == v205 )
              {
                sub_CB6200(v204, (unsigned __int8 *)" ", 1u);
              }
              else
              {
                *v205 = 32;
                ++*(_QWORD *)(v204 + 32);
              }
              v206 = *((_DWORD *)v49 + 24);
              v207 = j + 1;
              if ( v206 == j + 1 )
                break;
              v208 = v207 >> 6;
              v209 = (unsigned int)(v206 - 1) >> 6;
              if ( v207 >> 6 > v209 )
                break;
              v210 = v49[4];
              v211 = 64 - (v207 & 0x3F);
              v212 = 0xFFFFFFFFFFFFFFFFLL >> v211;
              if ( v211 == 64 )
                v212 = 0;
              v213 = v208;
              v214 = ~v212;
              while ( 1 )
              {
                _RAX = *(_QWORD *)(v210 + 8 * v213);
                if ( v208 == (_DWORD)v213 )
                  _RAX = v214 & *(_QWORD *)(v210 + 8 * v213);
                if ( v209 == (_DWORD)v213 )
                  _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*((_DWORD *)v49 + 24);
                if ( _RAX )
                  break;
                if ( v209 < (unsigned int)++v213 )
                  goto LABEL_152;
              }
              __asm { tzcnt   rax, rax }
            }
          }
LABEL_152:
          v169 = sub_CB72A0();
          v170 = (_BYTE *)v169[4];
          if ( (_BYTE *)v169[3] == v170 )
          {
            sub_CB6200((__int64)v169, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v170 = 10;
            ++v169[4];
          }
          ++v275;
LABEL_155:
          v48 += 8;
          if ( v276 == v48 )
            goto LABEL_169;
          goto LABEL_156;
        }
        v48 += 8;
        if ( v276 == v48 )
        {
LABEL_169:
          v41 = v294;
          v44 = *(_QWORD *)(v294 - 24);
          v178 = *(__int64 **)(v44 + 112);
          goto LABEL_170;
        }
LABEL_156:
        v45 = a1[25];
      }
      v51 = *(_QWORD *)(v50 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v52 = v51;
      if ( v51 == v50 + 48 )
        goto LABEL_155;
      if ( !v51 )
        BUG();
      v53 = *(_DWORD *)(v51 + 44);
      v54 = *(_QWORD *)v51;
      LOBYTE(v51) = v53;
      if ( (v54 & 4) != 0 )
      {
        if ( (v53 & 4) != 0 )
          goto LABEL_244;
      }
      else if ( (v53 & 4) != 0 )
      {
        for ( k = v54; ; k = *(_QWORD *)v52 )
        {
          v52 = k & 0xFFFFFFFFFFFFFFF8LL;
          LODWORD(v51) = *(_DWORD *)(v52 + 44) & 0xFFFFFF;
          if ( (*(_DWORD *)(v52 + 44) & 4) == 0 )
            break;
        }
      }
      if ( (v51 & 8) != 0 )
      {
        LOBYTE(v56) = sub_2E88A90(v52, 32, 1);
        goto LABEL_54;
      }
LABEL_244:
      v56 = (*(_QWORD *)(*(_QWORD *)(v52 + 16) + 24LL) >> 5) & 1LL;
LABEL_54:
      if ( !(_BYTE)v56 )
        goto LABEL_155;
      goto LABEL_55;
    }
    v178 = (__int64 *)(v46 + 8LL * *(unsigned int *)(v44 + 120));
LABEL_170:
    if ( *(_BYTE *)(v41 - 8) )
    {
LABEL_171:
      v179 = *(__int64 **)(v41 - 16);
      v180 = v178;
      goto LABEL_172;
    }
    while ( 1 )
    {
      *(_QWORD *)(v41 - 16) = v178;
      v179 = v178;
      *(_BYTE *)(v41 - 8) = 1;
      v180 = *(__int64 **)(v44 + 112);
LABEL_172:
      v181 = *(unsigned int *)(v44 + 120);
      if ( v179 != &v180[v181] )
        break;
LABEL_179:
      v294 -= 24LL;
      v42 = v293;
      v41 = v294;
      if ( v294 == v293 )
        goto LABEL_247;
      v44 = *(_QWORD *)(v294 - 24);
      v178 = *(__int64 **)(v44 + 112);
      if ( *(_BYTE *)(v294 - 8) )
        goto LABEL_171;
    }
    while ( 1 )
    {
      v182 = v179 + 1;
      *(_QWORD *)(v41 - 16) = v179 + 1;
      v183 = *v179;
      if ( v291 )
      {
        v184 = (_QWORD *)v288.m128i_i64[1];
        v181 = v290;
        v182 = (__int64 *)(v288.m128i_i64[1] + 8LL * v290);
        if ( (__int64 *)v288.m128i_i64[1] != v182 )
        {
          while ( v183 != *v184 )
          {
            if ( v182 == ++v184 )
              goto LABEL_273;
          }
          goto LABEL_178;
        }
LABEL_273:
        if ( v290 < v289 )
          break;
      }
      sub_C8CC70((__int64)&v288, v183, (__int64)v182, v181, v33, v34);
      if ( v229 )
        goto LABEL_246;
LABEL_178:
      v181 = *(unsigned int *)(v44 + 120);
      v179 = *(__int64 **)(v41 - 16);
      if ( v179 == (__int64 *)(*(_QWORD *)(v44 + 112) + 8 * v181) )
        goto LABEL_179;
    }
    ++v290;
    *v182 = v183;
    ++v288.m128i_i64[0];
LABEL_246:
    v280.m128i_i64[0] = v183;
    LOBYTE(v281) = 0;
    sub_37EEB50(&v293, &v280);
    v42 = v293;
    v41 = v294;
LABEL_247:
    v38 = v299;
    v43 = v41 - v42;
  }
  while ( v41 - v42 != *((_QWORD *)&v299 + 1) - (_QWORD)v299 );
LABEL_248:
  if ( v42 != v41 )
  {
    v230 = v38;
    while ( *(_QWORD *)v42 == *(_QWORD *)v230 )
    {
      v43 = *(unsigned __int8 *)(v42 + 16);
      if ( (_BYTE)v43 != *(_BYTE *)(v230 + 16) || (_BYTE)v43 && *(_QWORD *)(v42 + 8) != *(_QWORD *)(v230 + 8) )
        break;
      v42 += 24LL;
      v230 += 24LL;
      if ( v42 == v41 )
        goto LABEL_255;
    }
    goto LABEL_41;
  }
LABEL_255:
  if ( v38 )
    j_j___libc_free_0(v38);
  if ( !v297 )
    _libc_free(v296.m128i_u64[1]);
  if ( v293 )
    j_j___libc_free_0(v293);
  if ( !v291 )
    _libc_free(v288.m128i_u64[1]);
  if ( v312 )
    j_j___libc_free_0((unsigned __int64)v312);
  if ( !v310 )
    _libc_free(v309);
  if ( v305 )
    j_j___libc_free_0((unsigned __int64)v305);
  if ( !v303 )
    _libc_free(v301.m128i_u64[1]);
  if ( v275 )
  {
    v280.m128i_i64[0] = (__int64)" in/out CFI information errors.";
    LOWORD(v284[0]) = 259;
    v296.m128i_i32[0] = v275;
    v301.m128i_i64[0] = (__int64)"Found ";
    v298[0] = 265;
    v304[1] = 1;
    v304[0] = 3;
    sub_9C6370(&v288, &v301, &v296, v43, v33, v34);
    sub_9C6370(v279, &v288, &v280, v231, v232, v233);
    sub_C64D30((__int64)v279, 1u);
  }
LABEL_157:
  v171 = sub_37EC0D0((__int64)a1, (__int64)a2);
  v172 = a1[25];
  v173 = a1[26];
  v3 = v171;
  if ( v172 != v173 )
  {
    v174 = a1[25];
    do
    {
      v175 = *(_QWORD *)(v174 + 104);
      if ( v175 != v174 + 120 )
        _libc_free(v175);
      v176 = *(_QWORD *)(v174 + 32);
      if ( v176 != v174 + 48 )
        _libc_free(v176);
      v174 += 184;
    }
    while ( v173 != v174 );
    a1[26] = v172;
  }
  return v3;
}
