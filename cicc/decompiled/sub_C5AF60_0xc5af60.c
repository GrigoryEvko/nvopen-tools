// Function: sub_C5AF60
// Address: 0xc5af60
//
__int64 __fastcall sub_C5AF60(
        int a1,
        __int64 (*a2)(),
        __int64 *a3,
        unsigned int *a4,
        __int64 a5,
        const char *a6,
        int a7)
{
  __int64 (*v9)(); // r12
  __int64 v10; // rax
  size_t v11; // rax
  int v12; // r15d
  __int64 *v13; // rbx
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rax
  void *v18; // r14
  unsigned int *v19; // rcx
  unsigned __int64 v20; // r13
  __int64 v21; // r13
  __m128i v22; // rdi
  _BYTE *v23; // rax
  unsigned int v24; // r13d
  __int64 v25; // rcx
  __int64 v26; // r8
  size_t v27; // rsi
  const char *v28; // r12
  __int64 v29; // rdx
  const char **v30; // rcx
  const char **v31; // rcx
  const char *v32; // rdx
  unsigned int *v33; // rax
  const char *v34; // r13
  int v35; // r14d
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  char *v39; // rdi
  char *v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 *v43; // rbx
  __int64 v44; // rbx
  char v45; // r15
  unsigned int v46; // r14d
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rax
  const char **v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // eax
  unsigned int *v56; // rax
  unsigned int v57; // ebx
  __int64 v58; // r12
  char v59; // al
  size_t v60; // rax
  char v61; // bl
  char v62; // r14
  size_t v63; // r15
  char v64; // al
  size_t v65; // rax
  size_t v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  const __m128i *v69; // rbx
  __m128i *v70; // rax
  unsigned int v71; // r12d
  int v72; // ebx
  __int64 *v73; // r13
  int v74; // r14d
  __int64 v75; // r15
  size_t v76; // rax
  __int64 v77; // rax
  const __m128i *v78; // rdx
  __int64 v79; // rcx
  __m128i *v80; // rax
  unsigned int v81; // r12d
  unsigned int v82; // eax
  __int64 *v83; // rbx
  __int64 *v84; // r13
  int v85; // r14d
  unsigned int v86; // r12d
  __int64 v87; // rax
  _BYTE *v88; // rax
  bool v89; // dl
  __int64 v90; // rax
  unsigned int *v91; // r13
  int v92; // ebx
  _BYTE *v93; // rax
  unsigned int v94; // edx
  _QWORD *v95; // rcx
  __int64 v96; // rax
  _QWORD *v97; // r13
  _QWORD *v98; // rbx
  __int64 v99; // r15
  __int64 v100; // r14
  __int64 v101; // rdx
  _QWORD *v102; // rax
  __int64 v103; // rdx
  _QWORD *v104; // rax
  _QWORD *v105; // r12
  __int64 v106; // rcx
  __int64 v107; // rdx
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // r8
  _QWORD *v111; // r9
  __int64 v112; // rax
  __int64 v113; // r14
  __int64 v114; // rax
  const char *v115; // rax
  __int64 *v116; // r14
  __int64 *v117; // rbx
  __int64 *i; // rax
  unsigned int v119; // ecx
  __m128i *v120; // rbx
  __m128i *v121; // r12
  __int64 *v122; // r14
  __int64 *v123; // rbx
  __int64 *j; // rax
  unsigned int v125; // ecx
  __m128i *v126; // rbx
  __m128i *v127; // r12
  __int64 v129; // r12
  __int64 *v130; // rbx
  __int64 v131; // r14
  size_t v132; // r9
  void (__fastcall *v133)(__int64, __int64, const char *, _QWORD, const char *, size_t, _QWORD); // r15
  const char *v134; // r13
  size_t v135; // rax
  char *v136; // rbx
  __int64 v137; // rax
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  const char *v141; // rsi
  __int64 v142; // rdi
  __int64 v143; // rax
  const char *v144; // rsi
  __int64 v145; // rdi
  __int64 v146; // rax
  size_t v147; // rdx
  const char *v148; // rcx
  __int64 v149; // rax
  __int64 *v150; // r12
  __int64 *v151; // r15
  __int64 *v152; // rax
  __int64 *v153; // rdx
  __int64 *v154; // rax
  char v155; // r8
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  unsigned int v161; // r15d
  unsigned int v162; // r14d
  __int64 v163; // rax
  _BYTE *v164; // rax
  __int64 v165; // r13
  unsigned __int8 v166; // al
  __int64 v167; // rax
  _BYTE *v168; // rax
  __int64 v169; // rax
  _BYTE *v170; // rax
  size_t v171; // r14
  void ***v172; // rbx
  __int8 *v173; // rbx
  char v174; // al
  __int64 v175; // r12
  int v176; // r13d
  size_t v177; // rdx
  void ***v178; // r13
  void **v179; // r12
  _QWORD *v180; // r9
  size_t v181; // r15
  size_t v182; // r14
  const char *v183; // rcx
  int v184; // eax
  void ***v185; // rax
  unsigned int v186; // eax
  __int64 v187; // rax
  __int64 v188; // rax
  unsigned __int64 v189; // rax
  unsigned __int64 v190; // rdx
  unsigned __int64 v191; // rcx
  unsigned __int64 v192; // rcx
  __int64 v193; // rdx
  __int64 *v194; // rax
  __int64 v195; // rcx
  __int64 *v196; // r14
  unsigned int v197; // r15d
  _BYTE *v198; // r12
  __int64 *v199; // rsi
  __int64 v200; // rdi
  unsigned __int8 v201; // al
  unsigned __int128 v202; // kr00_16
  const __m128i *v203; // r12
  _BYTE *v204; // r14
  unsigned int v205; // eax
  unsigned int v206; // r13d
  __int64 v207; // rax
  _BYTE *v208; // rdi
  __int64 v209; // r8
  _BYTE *v210; // rdi
  __int64 v211; // r8
  __int64 v212; // rax
  __int64 *v213; // rax
  __int64 v214; // rdx
  size_t v215; // rdx
  size_t v216; // rdx
  __int64 v217; // rax
  __m128i v218; // xmm0
  unsigned __int64 v219; // rdx
  __int64 v220; // rcx
  __int64 v221; // r8
  __int64 v222; // r9
  _WORD *v223; // r14
  __int64 v224; // r13
  unsigned __int64 v225; // r13
  unsigned __int8 *v226; // r12
  int v227; // eax
  __int64 v228; // rdx
  unsigned __int8 v229; // al
  int v230; // eax
  char v231; // al
  __int64 v232; // rax
  __int64 v233; // rax
  char v234; // al
  __int64 v235; // [rsp-8h] [rbp-638h]
  __int64 *v236; // [rsp+0h] [rbp-630h]
  __int64 *v237; // [rsp+8h] [rbp-628h]
  _BYTE *v238; // [rsp+10h] [rbp-620h]
  __int64 v239; // [rsp+18h] [rbp-618h]
  unsigned __int64 v240; // [rsp+20h] [rbp-610h]
  _BYTE *v241; // [rsp+28h] [rbp-608h]
  __int64 v242; // [rsp+30h] [rbp-600h]
  unsigned __int64 v243; // [rsp+38h] [rbp-5F8h]
  int v244; // [rsp+4Ch] [rbp-5E4h]
  __int64 v245; // [rsp+60h] [rbp-5D0h]
  __int64 v246; // [rsp+68h] [rbp-5C8h]
  __m128i v248; // [rsp+C0h] [rbp-570h] BYREF
  char v249; // [rsp+D0h] [rbp-560h]
  bool v250; // [rsp+D1h] [rbp-55Fh]
  char v251; // [rsp+D2h] [rbp-55Eh]
  char v252; // [rsp+D3h] [rbp-55Dh]
  unsigned int v253; // [rsp+D4h] [rbp-55Ch]
  __int64 v254; // [rsp+D8h] [rbp-558h]
  int v255; // [rsp+E0h] [rbp-550h]
  int v256; // [rsp+E4h] [rbp-54Ch]
  __int64 v257; // [rsp+E8h] [rbp-548h]
  const char **v258; // [rsp+F0h] [rbp-540h]
  __m128i *v259; // [rsp+F8h] [rbp-538h]
  __int64 *v260; // [rsp+100h] [rbp-530h]
  unsigned int *v261; // [rsp+108h] [rbp-528h]
  __int32 v262; // [rsp+114h] [rbp-51Ch] BYREF
  _QWORD *v263; // [rsp+118h] [rbp-518h] BYREF
  const char **v264; // [rsp+120h] [rbp-510h] BYREF
  __int64 v265; // [rsp+128h] [rbp-508h] BYREF
  unsigned __int8 *v266; // [rsp+130h] [rbp-500h] BYREF
  size_t v267; // [rsp+138h] [rbp-4F8h]
  __m128i v268; // [rsp+140h] [rbp-4F0h] BYREF
  __m128i v269; // [rsp+150h] [rbp-4E0h] BYREF
  char *v270; // [rsp+160h] [rbp-4D0h] BYREF
  size_t v271; // [rsp+168h] [rbp-4C8h]
  char v272; // [rsp+170h] [rbp-4C0h] BYREF
  void *v273; // [rsp+180h] [rbp-4B0h] BYREF
  size_t v274; // [rsp+188h] [rbp-4A8h]
  __int64 v275; // [rsp+190h] [rbp-4A0h] BYREF
  _QWORD *v276; // [rsp+1A0h] [rbp-490h] BYREF
  size_t v277; // [rsp+1A8h] [rbp-488h]
  _QWORD v278[2]; // [rsp+1B0h] [rbp-480h] BYREF
  _QWORD v279[4]; // [rsp+1C0h] [rbp-470h] BYREF
  __int16 v280; // [rsp+1E0h] [rbp-450h]
  size_t v281[2]; // [rsp+1F0h] [rbp-440h] BYREF
  _QWORD v282[2]; // [rsp+200h] [rbp-430h] BYREF
  __int16 v283; // [rsp+210h] [rbp-420h]
  _BYTE v284[64]; // [rsp+220h] [rbp-410h] BYREF
  _QWORD v285[2]; // [rsp+260h] [rbp-3D0h] BYREF
  __int64 *v286; // [rsp+270h] [rbp-3C0h]
  __int64 v287; // [rsp+278h] [rbp-3B8h]
  _BYTE v288[32]; // [rsp+280h] [rbp-3B0h] BYREF
  __m128i *v289; // [rsp+2A0h] [rbp-390h]
  __int64 v290; // [rsp+2A8h] [rbp-388h]
  _QWORD v291[2]; // [rsp+2B0h] [rbp-380h] BYREF
  _QWORD v292[2]; // [rsp+2C0h] [rbp-370h] BYREF
  __int64 *v293; // [rsp+2D0h] [rbp-360h]
  __int64 v294; // [rsp+2D8h] [rbp-358h]
  _BYTE v295[32]; // [rsp+2E0h] [rbp-350h] BYREF
  __m128i *v296; // [rsp+300h] [rbp-330h]
  __int64 v297; // [rsp+308h] [rbp-328h]
  _QWORD v298[2]; // [rsp+310h] [rbp-320h] BYREF
  size_t v299; // [rsp+320h] [rbp-310h] BYREF
  __int64 v300; // [rsp+328h] [rbp-308h]
  _BYTE v301[96]; // [rsp+330h] [rbp-300h] BYREF
  void *src; // [rsp+390h] [rbp-2A0h] BYREF
  __int64 v303; // [rsp+398h] [rbp-298h]
  _QWORD v304[20]; // [rsp+3A0h] [rbp-290h] BYREF
  const char **v305; // [rsp+440h] [rbp-1F0h] BYREF
  __int64 v306; // [rsp+448h] [rbp-1E8h]
  _BYTE v307[160]; // [rsp+450h] [rbp-1E0h] BYREF
  const char *v308; // [rsp+4F0h] [rbp-140h] BYREF
  size_t n; // [rsp+4F8h] [rbp-138h]
  _QWORD v310[2]; // [rsp+500h] [rbp-130h] BYREF
  char v311; // [rsp+510h] [rbp-120h]
  char v312; // [rsp+511h] [rbp-11Fh]

  v9 = a2;
  v261 = a4;
  LODWORD(v257) = a7;
  v249 = a7;
  if ( !qword_4F83C80 )
  {
    a2 = (__int64 (*)())sub_C58C10;
    sub_C7D570(&qword_4F83C80, sub_C58C10, sub_C51550);
  }
  sub_C61300();
  nullsub_155();
  sub_C8C3C0();
  sub_C91D20();
  sub_C9F380();
  sub_CA1770();
  sub_CA5730();
  nullsub_148();
  sub_C88600();
  nullsub_180();
  v285[0] = 0;
  src = v304;
  v263 = v285;
  v10 = *(_QWORD *)v9;
  v286 = (__int64 *)v288;
  v287 = 0x400000000LL;
  v304[0] = v10;
  v285[1] = 0;
  v289 = (__m128i *)v291;
  v290 = 0;
  v291[0] = 0;
  v291[1] = 1;
  v303 = 0x1400000001LL;
  if ( a6 )
  {
    v11 = strlen(a6);
    a2 = (__int64 (*)())a6;
    v259 = (__m128i *)&v308;
    sub_C86120(&v308, a6, v11);
    if ( v311 )
    {
      a2 = (__int64 (*)())n;
      sub_C50FA0((__int64)v308, (_BYTE *)n, (__int64)&v263, (__int64)&src, 0);
      if ( v311 )
      {
        v311 = 0;
        sub_2240A30(v259);
      }
    }
    v12 = v303;
  }
  else
  {
    v12 = 1;
    v259 = (__m128i *)&v308;
  }
  if ( a1 > 1 )
  {
    v260 = a3;
    v13 = (__int64 *)((char *)v9 + 8);
    v14 = (__int64)v9 + 8 * (unsigned int)(a1 - 2) + 16;
    do
    {
      v15 = (unsigned int)v12;
      v16 = *v13;
      if ( (unsigned __int64)(unsigned int)v12 + 1 > HIDWORD(v303) )
      {
        a2 = (__int64 (*)())v304;
        sub_C8D5F0(&src, v304, (unsigned int)v12 + 1LL, 8);
        v15 = (unsigned int)v303;
      }
      ++v13;
      *((_QWORD *)src + v15) = v16;
      v12 = v303 + 1;
      LODWORD(v303) = v303 + 1;
    }
    while ( (__int64 *)v14 != v13 );
    a3 = v260;
  }
  if ( !qword_4F83CE0 )
  {
    a2 = sub_C53DA0;
    sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
  }
  v17 = qword_4F83CE0;
  v18 = src;
  v19 = v261;
  v258 = (const char **)qword_4F83CE0;
  v264 = (const char **)src;
  v265 = a5;
  *(_QWORD *)(qword_4F83CE0 + 32) = a3;
  *(_QWORD *)(v17 + 40) = v19;
  if ( !a5 )
  {
    v160 = sub_CB72A0(0, a2);
    v18 = v264;
    v265 = v160;
  }
  v20 = 8LL * v12;
  v305 = (const char **)v307;
  v306 = 0x1400000000LL;
  if ( v20 > 0xA0 )
  {
    sub_C8D5F0(&v305, v307, v12, 8);
    v51 = &v305[(unsigned int)v306];
  }
  else
  {
    if ( !v20 )
      goto LABEL_18;
    v51 = (const char **)v307;
  }
  memcpy(v51, v18, 8LL * v12);
  LODWORD(v20) = v306;
LABEL_18:
  v292[0] = 0;
  v293 = (__int64 *)v295;
  v294 = 0x400000000LL;
  LODWORD(v306) = v12 + v20;
  v296 = (__m128i *)v298;
  v292[1] = 0;
  v297 = 0;
  v298[0] = 0;
  v298[1] = 1;
  sub_C53200((__int64)v284, (__int64)v292, (__int64)sub_C50FA0);
  sub_C59DA0(v281, (__int64)v284, (__int64 *)&v305);
  if ( (v281[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v21 = v265;
    v299 = v281[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    v281[0] = 0;
    sub_C64870(v259, &v299);
    v22.m128i_i64[1] = (__int64)v308;
    v22.m128i_i64[0] = sub_CB6200(v21, v308, n);
    v23 = *(_BYTE **)(v22.m128i_i64[0] + 32);
    if ( (unsigned __int64)v23 >= *(_QWORD *)(v22.m128i_i64[0] + 24) )
    {
      v22.m128i_i64[1] = 10;
      sub_CB5D20(v22.m128i_i64[0], 10);
    }
    else
    {
      *(_QWORD *)(v22.m128i_i64[0] + 32) = v23 + 1;
      *v23 = 10;
    }
    v24 = 0;
    sub_2240A30(v259);
    sub_9C66B0((__int64 *)&v299);
    sub_9C66B0((__int64 *)v281);
    goto LABEL_141;
  }
  v281[0] = 0;
  sub_9C66B0((__int64 *)v281);
  v27 = 0;
  v264 = v305;
  v28 = *v305;
  v256 = v306;
  if ( v28 )
    v27 = strlen(v28);
  v22.m128i_i64[1] = sub_C80C60(v28, v27, 0, v25, v26);
  v308 = (const char *)v310;
  sub_C4FB50(v259->m128i_i64, (_BYTE *)v22.m128i_i64[1], v22.m128i_i64[1] + v29);
  v22.m128i_i64[0] = (__int64)*v258;
  if ( v308 == (const char *)v310 )
  {
    v147 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v22.m128i_i64[0] = v310[0];
      }
      else
      {
        v22.m128i_i64[1] = (__int64)v310;
        memcpy((void *)v22.m128i_i64[0], v310, n);
      }
      v147 = n;
      v22.m128i_i64[0] = (__int64)*v258;
    }
    v258[1] = (const char *)v147;
    *(_BYTE *)(v22.m128i_i64[0] + v147) = 0;
    v22.m128i_i64[0] = (__int64)v308;
    goto LABEL_28;
  }
  v30 = v258;
  if ( (const char **)v22.m128i_i64[0] == v258 + 2 )
  {
    *v258 = v308;
    v30[1] = (const char *)n;
    v30[2] = (const char *)v310[0];
    goto LABEL_222;
  }
  v31 = v258;
  *v258 = v308;
  v32 = v31[2];
  v31[1] = (const char *)n;
  v31[2] = (const char *)v310[0];
  if ( !v22.m128i_i64[0] )
  {
LABEL_222:
    v308 = (const char *)v310;
    v22.m128i_i64[0] = (__int64)v310;
    goto LABEL_28;
  }
  v308 = (const char *)v22.m128i_i64[0];
  v310[0] = v32;
LABEL_28:
  n = 0;
  *(_BYTE *)v22.m128i_i64[0] = 0;
  v22.m128i_i64[0] = (__int64)v259;
  sub_2240A30(v259);
  v33 = (unsigned int *)sub_C52570();
  v271 = 0;
  v261 = v33;
  v270 = &v272;
  v272 = 0;
  if ( v256 > 1 )
  {
    v34 = v264[1];
    if ( *v34 != 45 )
    {
      v148 = v258[36];
      v149 = *((_BYTE *)v258 + 308) ? *((unsigned int *)v258 + 75) : *((unsigned int *)v258 + 74);
      v150 = (__int64 *)&v148[8 * v149];
      v151 = (__int64 *)v258[36];
      v152 = v151;
      if ( v148 != (const char *)v150 )
      {
        while ( 1 )
        {
          v22.m128i_i64[1] = *v152;
          v153 = v152;
          if ( (unsigned __int64)*v152 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v150 == ++v152 )
            goto LABEL_30;
        }
        while ( v150 != v153 )
        {
          if ( *(_QWORD *)(v22.m128i_i64[1] + 8) )
          {
            v22.m128i_i64[0] = (__int64)v264[1];
            v171 = strlen((const char *)v22.m128i_i64[0]);
            if ( v171 )
            {
              while ( 1 )
              {
                v172 = (void ***)v151;
                if ( (unsigned __int64)*v151 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v150 == ++v151 )
                  goto LABEL_241;
              }
              v261 = (unsigned int *)*v151;
              if ( v150 != v151 )
              {
                v22.m128i_i64[1] = (__int64)v34;
                v178 = (void ***)v150;
                v179 = (void **)v261;
                v180 = 0;
                v181 = v171;
                do
                {
                  v182 = (size_t)v179[1];
                  if ( v182 )
                  {
                    v183 = (const char *)*v179;
                    if ( v181 == v182 )
                    {
                      v22.m128i_i64[0] = (__int64)*v179;
                      v254 = (__int64)v180;
                      v260 = (__int64 *)v22.m128i_i64[1];
                      v261 = (unsigned int *)v183;
                      v184 = memcmp((const void *)v22.m128i_i64[0], (const void *)v22.m128i_i64[1], v181);
                      v183 = (const char *)v261;
                      v180 = (_QWORD *)v254;
                      if ( !v184 )
                      {
                        v261 = (unsigned int *)v179;
                        goto LABEL_242;
                      }
                    }
                    if ( !v180 )
                    {
                      v22.m128i_i64[0] = (__int64)v259;
                      v308 = v183;
                      v260 = 0;
                      v261 = (unsigned int *)v22.m128i_i64[1];
                      n = v182;
                      v186 = sub_C93110(v259, v22.m128i_i64[1], v181, 1, 0);
                      v180 = 0;
                      if ( v186 <= 1 )
                        v180 = v179;
                    }
                  }
                  v185 = v172 + 1;
                  if ( v178 == v172 + 1 )
                    break;
                  while ( 1 )
                  {
                    v179 = *v185;
                    v172 = v185;
                    if ( (unsigned __int64)*v185 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v178 == ++v185 )
                      goto LABEL_267;
                  }
                }
                while ( v178 != v185 );
LABEL_267:
                if ( v180 )
                {
                  v22.m128i_i64[0] = (__int64)&v270;
                  v22.m128i_i64[1] = 0;
                  sub_2241130(&v270, 0, v271, *v180, v180[1]);
                }
              }
            }
LABEL_241:
            v261 = (unsigned int *)sub_C52570();
LABEL_242:
            v252 = 1;
            v35 = (v261 != (unsigned int *)sub_C52570()) + 1;
            goto LABEL_31;
          }
          v154 = v153 + 1;
          if ( v153 + 1 == v150 )
            break;
          while ( 1 )
          {
            v22.m128i_i64[1] = *v154;
            v153 = v154;
            if ( (unsigned __int64)*v154 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v150 == ++v154 )
              goto LABEL_30;
          }
        }
      }
    }
  }
LABEL_30:
  v252 = 0;
  v35 = 1;
LABEL_31:
  v36 = sub_C4F9D0(v22.m128i_i64[0], v22.m128i_i64[1]);
  v39 = (char *)v261;
  *(_QWORD *)(v36 + 344) = v261;
  v40 = v39 + 128;
  v22.m128i_i64[0] = (__int64)v258;
  v246 = (__int64)v40;
  v41 = *((unsigned int *)v258 + 20);
  v42 = (__int64 *)v258[9];
  if ( v42 != &v42[v41] )
  {
    LODWORD(v254) = v35;
    v260 = &v42[v41];
    v43 = v42;
    do
    {
      v22.m128i_i64[1] = *v43;
      v22.m128i_i64[0] = (__int64)v258;
      v308 = (const char *)&v299;
      ++v43;
      v299 = v22.m128i_u64[1];
      n = (size_t)v258;
      sub_C52DD0((__int64)v258, v22.m128i_i64[1], (__int64 (__fastcall *)(__int64, __int64))sub_C53EC0, (__int64)v259);
    }
    while ( v260 != v43 );
    v35 = v254;
  }
  v253 = v261[10];
  if ( v253 )
  {
    v44 = 0;
    v45 = 0;
    LODWORD(v260) = v35;
    v46 = 0;
    v47 = v253;
    LOBYTE(v254) = 0;
    do
    {
      v48 = *(_QWORD *)(*((_QWORD *)v261 + 4) + 8 * v44);
      v49 = *(_BYTE *)(v48 + 12) & 7;
      if ( (*(_BYTE *)(v48 + 12) & 7u) - 2 <= 1 )
      {
        ++v46;
      }
      else if ( *((_QWORD *)v261 + 19) )
      {
        if ( v261[10] > 1 )
        {
          LOBYTE(v254) = 1;
          if ( !a5 )
          {
            v248.m128i_i64[0] = v47;
            v50 = sub_CEADF0(v22.m128i_i64[0], v22.m128i_i64[1], v49, v37, v38, v47);
            v22.m128i_i64[1] = (__int64)v259;
            v22.m128i_i64[0] = v48;
            v312 = 1;
            v308 = "error - this positional option will never be matched, because it does not Require a value, and a cl::"
                   "ConsumeAfter option is active!";
            v311 = 3;
            sub_C53280(v48, (__int64)v259, 0, 0, v50);
            v47 = v248.m128i_i64[0];
            LOBYTE(v49) = *(_BYTE *)(v48 + 12) & 7;
          }
        }
      }
      else if ( v45 && !*(_QWORD *)(v48 + 32) )
      {
        if ( !a5 )
        {
          v254 = v47;
          v232 = sub_CEADF0(v22.m128i_i64[0], v22.m128i_i64[1], v49, v37, v38, v47);
          v312 = 1;
          v308 = "error - option can never match, because another positional argument will match an unbounded number of v"
                 "alues, and this option does not require a value!";
          v311 = 3;
          sub_C53280(v48, (__int64)v259, 0, 0, v232);
          v47 = v254;
        }
        v248.m128i_i64[0] = v47;
        v52 = sub_CB6200(v265, *v258, v258[1]);
        v53 = sub_904010(v52, ": CommandLine Error: Option '");
        v54 = sub_A51340(v53, *(const void **)(v48 + 24), *(_QWORD *)(v48 + 32));
        sub_904010(v54, "' is all messed up!\n");
        v22.m128i_i64[1] = v261[10];
        v22.m128i_i64[0] = v265;
        sub_CB59D0(v265, v22.m128i_i64[1]);
        LOBYTE(v49) = *(_BYTE *)(v48 + 12);
        LOBYTE(v254) = v45;
        v47 = v248.m128i_i64[0];
        LOBYTE(v49) = v49 & 7;
      }
      ++v44;
      v45 |= (v49 & 0xFD) == 1;
    }
    while ( v47 != v44 );
    v253 = v46;
    v35 = (int)v260;
    v250 = v45;
    if ( !v45 )
      v250 = *((_QWORD *)v261 + 19) != 0;
    v262 = (int)v260;
    v299 = (size_t)v301;
    v300 = 0x400000000LL;
    v245 = v253;
    if ( v256 <= (int)v260 )
    {
      v81 = 0;
      goto LABEL_85;
    }
  }
  else
  {
    v262 = v35;
    v299 = (size_t)v301;
    v300 = 0x400000000LL;
    if ( v256 <= v35 )
    {
      LOBYTE(v254) = 0;
      v81 = 0;
      v82 = v261[10];
      goto LABEL_88;
    }
    v245 = 0;
    LOBYTE(v254) = 0;
    v250 = 0;
  }
  v55 = (unsigned __int8)v257;
  v251 = 0;
  v257 = 0;
  v255 = v55;
  v260 = &v275;
  while ( 1 )
  {
    v274 = 0;
    LOBYTE(v275) = 0;
    v273 = v260;
    v268.m128i_i64[0] = (__int64)byte_3F871B3;
    v266 = 0;
    v267 = 0;
    v268.m128i_i64[1] = 0;
    v58 = (__int64)v264[v35];
    if ( *(_BYTE *)v58 != 45 )
      break;
    v59 = *(_BYTE *)(v58 + 1);
    if ( (unsigned __int8)v251 | (v59 == 0) )
      break;
    if ( v59 == 45 && !*(_BYTE *)(v58 + 2) )
    {
      v251 = 1;
      goto LABEL_62;
    }
    v60 = strlen((const char *)(v58 + 1));
    if ( v257 && (*(_BYTE *)(v257 + 13) & 4) != 0 )
    {
      v268.m128i_i64[0] = v58 + 1;
      v268.m128i_i64[1] = v60;
      if ( v60 && *(_BYTE *)(v58 + 1) == 45 )
      {
        v155 = 1;
        v268.m128i_i64[0] = v58 + 2;
        v268.m128i_i64[1] = v60 - 1;
      }
      else
      {
        v155 = 0;
      }
      v22.m128i_i64[1] = (__int64)&v268;
      v156 = sub_C51940((__int64)v261, (__int64)&v268, &v266, v255, v155);
      if ( !v156 || ((*(_WORD *)(v156 + 12) >> 7) & 3) != 1 )
      {
        v175 = (__int64)v264[v262];
        v176 = v262;
        v177 = 0;
        if ( v175 )
          v177 = strlen(v264[v262]);
        v22.m128i_i64[1] = v175;
        sub_C53770(v257, v175, v177, v176);
        v251 = 0;
        goto LABEL_60;
      }
      v257 = v156;
      goto LABEL_248;
    }
    v268.m128i_i64[0] = v58 + 1;
    v268.m128i_i64[1] = v60;
    if ( v60 && *(_BYTE *)(v58 + 1) == 45 )
    {
      v62 = 1;
      v61 = 1;
      v268.m128i_i64[0] = v58 + 2;
      v268.m128i_i64[1] = v60 - 1;
    }
    else
    {
      v61 = 0;
      v62 = 0;
    }
    v22.m128i_i64[1] = (__int64)&v268;
    v63 = sub_C51940((__int64)v261, (__int64)&v268, &v266, v255, v62);
    if ( !v63 )
    {
      if ( v261 != (unsigned int *)sub_C52570() )
      {
        v187 = sub_C52570();
        v22.m128i_i64[1] = (__int64)&v268;
        v188 = sub_C51940(v187, (__int64)&v268, &v266, v255, v62);
        if ( v188 )
        {
          v63 = v188;
          goto LABEL_71;
        }
      }
      if ( (!v249 || !v61) && v268.m128i_i64[1] != 1 )
      {
        v281[0] = 0;
        v22 = v268;
        v223 = (_WORD *)sub_C50DF0(
                          v268.m128i_i64[0],
                          v268.m128i_u64[1],
                          v281,
                          (unsigned __int8 (__fastcall *)(_QWORD))sub_C4F870,
                          v246);
        if ( v223 )
        {
          while ( 1 )
          {
            v224 = v268.m128i_i64[1];
            if ( v268.m128i_i64[1] <= v281[0] )
              break;
            v268.m128i_i64[1] = v281[0];
            v225 = v224 - v281[0];
            v226 = (unsigned __int8 *)(v268.m128i_i64[0] + v281[0]);
            v227 = (v223[6] >> 7) & 3;
            if ( v227 == 3 )
              goto LABEL_367;
            v228 = *v226;
            if ( v227 == 2 )
            {
              if ( (_BYTE)v228 == 61 )
              {
LABEL_366:
                v63 = (size_t)v223;
                v266 = v226 + 1;
                v267 = v225 - 1;
                goto LABEL_71;
              }
LABEL_367:
              v63 = v268.m128i_i64[0] + v281[0];
LABEL_355:
              v266 = (unsigned __int8 *)v63;
              v63 = (size_t)v223;
              v267 = v225;
              goto LABEL_71;
            }
            if ( (_BYTE)v228 == 61 )
              goto LABEL_366;
            v229 = *((_BYTE *)v223 + 12);
            if ( (v229 & 0x18) != 0 )
            {
              v230 = (v229 >> 3) & 3;
            }
            else
            {
              v22.m128i_i64[0] = (__int64)v223;
              v230 = (*(__int64 (__fastcall **)(_WORD *))(*(_QWORD *)v223 + 8LL))(v223);
            }
            if ( v230 == 2 )
            {
              v233 = sub_CEADF0(v22.m128i_i64[0], v22.m128i_i64[1], v228, v220, v221, v222);
              v312 = 1;
              v308 = "may not occur within a group!";
              v311 = 3;
              v234 = sub_C53280((__int64)v223, (__int64)v259, 0, 0, v233);
              LOBYTE(v254) = v234 | v254;
              goto LABEL_278;
            }
            LODWORD(v308) = 0;
            v231 = sub_C533F0((__int64)v223, v268.m128i_i64[0], v268.m128i_i64[1], 0, 0, 0, 0, v259);
            v22.m128i_i64[0] = (__int64)v226;
            v22.m128i_i64[1] = v225;
            v268.m128i_i64[0] = (__int64)v226;
            LOBYTE(v254) = v231 | v254;
            v268.m128i_i64[1] = v225;
            v223 = (_WORD *)sub_C50DF0(
                              (__int64)v226,
                              v225,
                              v281,
                              (unsigned __int8 (__fastcall *)(_QWORD))sub_C4F5C0,
                              v246);
            if ( !v223 )
              goto LABEL_278;
          }
          v225 = 0;
          goto LABEL_355;
        }
      }
LABEL_278:
      v57 = v261[22];
      if ( v57 )
        goto LABEL_172;
      v269 = _mm_load_si128(&v268);
      if ( !v269.m128i_i64[1] )
        goto LABEL_58;
      LOBYTE(v308) = 61;
      v189 = sub_C931B0(&v269, v259, 1, 0);
      if ( v189 == -1 )
      {
        v243 = 0;
        v240 = 0;
        v239 = v269.m128i_i64[1];
        v242 = v269.m128i_i64[0];
      }
      else
      {
        v190 = v269.m128i_u64[1];
        v191 = v189 + 1;
        v242 = v269.m128i_i64[0];
        if ( v189 + 1 > v269.m128i_i64[1] )
        {
          v243 = 0;
          v191 = v269.m128i_u64[1];
        }
        else
        {
          v243 = v269.m128i_i64[1] - v191;
        }
        v192 = v269.m128i_i64[0] + v191;
        if ( v189 <= v269.m128i_i64[1] )
          v190 = v189;
        v240 = v192;
        v239 = v190;
      }
      v193 = v261[34];
      if ( (_DWORD)v193 )
      {
        v194 = (__int64 *)*((_QWORD *)v261 + 16);
        v195 = *v194;
        v196 = v194;
        if ( *v194 )
          goto LABEL_289;
        do
        {
          do
          {
            v195 = v196[1];
            ++v196;
          }
          while ( !v195 );
LABEL_289:
          ;
        }
        while ( v195 == -8 );
        v237 = &v194[v193];
        if ( v237 != v196 )
        {
          v238 = 0;
          v197 = 0;
          do
          {
            v198 = *(_BYTE **)(*v196 + 8);
            if ( ((v198[12] >> 5) & 3) != 2 )
            {
              v199 = (__int64 *)v259;
              v200 = *(_QWORD *)(*v196 + 8);
              v308 = (const char *)v310;
              n = 0x1000000000LL;
              (*(void (__fastcall **)(__int64, __m128i *))(*(_QWORD *)v198 + 72LL))(v200, v259);
              if ( *((_QWORD *)v198 + 4) )
              {
                v217 = (unsigned int)n;
                v218 = _mm_loadu_si128((const __m128i *)(v198 + 24));
                v219 = (unsigned int)n + 1LL;
                if ( v219 > HIDWORD(n) )
                {
                  v199 = v310;
                  v248 = v218;
                  sub_C8D5F0(v259, v310, v219, 16);
                  v217 = (unsigned int)n;
                  v218 = _mm_load_si128(&v248);
                }
                *(__m128i *)&v308[16 * v217] = v218;
                LODWORD(n) = n + 1;
              }
              v201 = v198[12];
              if ( (v201 & 0x18) != 0 )
                v244 = (v201 >> 3) & 3;
              else
                v244 = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)v198 + 8LL))(v198);
              if ( v244 == 3 )
                v202 = (unsigned __int128)v269;
              else
                v202 = __PAIR128__(v239, v242);
              v248.m128i_i64[0] = (__int64)&v308[16 * (unsigned int)n];
              if ( v308 != (const char *)v248.m128i_i64[0] )
              {
                v241 = v198;
                v203 = (const __m128i *)v308;
                v236 = v196;
                v204 = v238;
                while ( 2 )
                {
                  v199 = (__int64 *)v202;
                  *(__m128i *)v281 = _mm_loadu_si128(v203);
                  v205 = sub_C93110(v281, v202, *((_QWORD *)&v202 + 1), 1, v197);
                  v206 = v205;
                  if ( v204 && v205 >= v197 )
                    goto LABEL_301;
                  v199 = (__int64 *)v203->m128i_i64[0];
                  v207 = v203->m128i_i64[1];
                  if ( v244 != 3 && v243 )
                  {
                    v281[1] = v203->m128i_u64[1];
                    v282[0] = "=";
                    v281[0] = (size_t)v199;
                    v199 = v279;
                    v279[3] = v243;
                    v279[2] = v240;
                    v283 = 773;
                    v279[0] = v281;
                    v280 = 1282;
                    sub_CA0F50(&v276, v279);
                    v208 = v273;
                    if ( v276 != v278 )
                    {
                      v199 = (__int64 *)v277;
                      if ( v273 == v260 )
                      {
                        v273 = v276;
                        v274 = v277;
                        v275 = v278[0];
                      }
                      else
                      {
                        v209 = v275;
                        v273 = v276;
                        v274 = v277;
                        v275 = v278[0];
                        if ( v208 )
                        {
                          v276 = v208;
                          v278[0] = v209;
                          goto LABEL_309;
                        }
                      }
                      v276 = v278;
                      v208 = v278;
                      goto LABEL_309;
                    }
                    v216 = v277;
                    if ( v277 )
                    {
                      if ( v277 == 1 )
                      {
                        *(_BYTE *)v273 = v278[0];
                      }
                      else
                      {
                        v199 = v278;
                        memcpy(v273, v278, v277);
                      }
                      v216 = v277;
                      v208 = v273;
                    }
                    v274 = v216;
                    v208[v216] = 0;
                    v208 = v276;
LABEL_309:
                    v277 = 0;
                    *v208 = 0;
                    if ( v276 != v278 )
                    {
                      v199 = (__int64 *)(v278[0] + 1LL);
                      j_j___libc_free_0(v276, v278[0] + 1LL);
                    }
LABEL_311:
                    v204 = v241;
                    v197 = v206;
LABEL_301:
                    if ( (const __m128i *)v248.m128i_i64[0] == ++v203 )
                    {
                      v238 = v204;
                      v196 = v236;
                      v248.m128i_i64[0] = (__int64)v308;
                      goto LABEL_319;
                    }
                    continue;
                  }
                  break;
                }
                v281[0] = (size_t)v282;
                sub_C4FB50((__int64 *)v281, v199, (__int64)v199 + v207);
                v210 = v273;
                if ( (_QWORD *)v281[0] == v282 )
                {
                  v215 = v281[1];
                  if ( v281[1] )
                  {
                    if ( v281[1] == 1 )
                    {
                      *(_BYTE *)v273 = v282[0];
                    }
                    else
                    {
                      v199 = v282;
                      memcpy(v273, v282, v281[1]);
                    }
                    v215 = v281[1];
                    v210 = v273;
                  }
                  v274 = v215;
                  v210[v215] = 0;
                  v210 = (_BYTE *)v281[0];
                  goto LABEL_316;
                }
                v199 = (__int64 *)v281[1];
                if ( v273 == v260 )
                {
                  v273 = (void *)v281[0];
                  v274 = v281[1];
                  v275 = v282[0];
                }
                else
                {
                  v211 = v275;
                  v273 = (void *)v281[0];
                  v274 = v281[1];
                  v275 = v282[0];
                  if ( v210 )
                  {
                    v281[0] = (size_t)v210;
                    v282[0] = v211;
                    goto LABEL_316;
                  }
                }
                v281[0] = (size_t)v282;
                v210 = v282;
LABEL_316:
                v281[1] = 0;
                *v210 = 0;
                if ( (_QWORD *)v281[0] != v282 )
                {
                  v199 = (__int64 *)(v282[0] + 1LL);
                  j_j___libc_free_0(v281[0], v282[0] + 1LL);
                }
                goto LABEL_311;
              }
LABEL_319:
              if ( (_QWORD *)v248.m128i_i64[0] != v310 )
                _libc_free(v248.m128i_i64[0], v199);
            }
            v212 = v196[1];
            if ( v212 != -8 && v212 )
            {
              ++v196;
            }
            else
            {
              v213 = v196 + 2;
              do
              {
                do
                {
                  v214 = *v213;
                  v196 = v213++;
                }
                while ( v214 == -8 );
              }
              while ( !v214 );
            }
          }
          while ( v237 != v196 );
        }
      }
      v56 = v261;
LABEL_57:
      v57 = v56[22];
      if ( v57 )
      {
LABEL_172:
        v129 = *((_QWORD *)v261 + 10) + 8LL * v57;
        v130 = (__int64 *)*((_QWORD *)v261 + 10);
        do
        {
          v131 = *v130;
          v132 = 0;
          v22.m128i_i64[1] = v262;
          v133 = *(void (__fastcall **)(__int64, __int64, const char *, _QWORD, const char *, size_t, _QWORD))(*(_QWORD *)*v130 + 80LL);
          v134 = v264[v262];
          if ( v134 )
          {
            v22.m128i_i64[0] = (__int64)v264[v262];
            v248.m128i_i32[0] = v262;
            v135 = strlen((const char *)v22.m128i_i64[0]);
            v22.m128i_i64[1] = v248.m128i_u32[0];
            v132 = v135;
          }
          ++v130;
          v133(v131, v22.m128i_i64[1], byte_3F871B3, 0, v134, v132, 0);
        }
        while ( (__int64 *)v129 != v130 );
      }
      else
      {
LABEL_58:
        v308 = (const char *)&v265;
        n = (size_t)v258;
        v310[0] = &v264;
        v310[1] = &v262;
        LOBYTE(v254) = v252 & (v262 <= 1);
        if ( (_BYTE)v254 )
        {
          v22.m128i_i64[1] = 0;
          sub_C51E50((__int64 **)v259, 0, v270, v271);
        }
        else
        {
          v22.m128i_i64[1] = 1;
          sub_C51E50((__int64 **)v259, 1, v273, v274);
          LOBYTE(v254) = 1;
        }
      }
LABEL_60:
      if ( v273 != v260 )
      {
        v22.m128i_i64[1] = v275 + 1;
        j_j___libc_free_0(v273, v275 + 1);
      }
      goto LABEL_62;
    }
LABEL_71:
    if ( ((*(_WORD *)(v63 + 12) >> 7) & 3) == 1 )
    {
      v257 = v63;
LABEL_248:
      if ( (*(_BYTE *)(v257 + 13) & 4) != 0 && v267 )
      {
        v22.m128i_i64[1] = (__int64)v259;
        v312 = 1;
        v308 = "This argument does not take a value.\n"
               "\tInstead, it consumes any positional arguments until the next recognized option.";
        v311 = 3;
        sub_C53280(v257, (__int64)v259, 0, 0, v265);
        LOBYTE(v254) = 1;
      }
      goto LABEL_73;
    }
    v64 = sub_C533F0(v63, v268.m128i_i64[0], v268.m128i_i64[1], (__int64)v266, v267, v256, (__int64)v264, &v262);
    LOBYTE(v254) = v64 | v254;
    v22.m128i_i64[1] = v235;
LABEL_73:
    sub_2240A30(&v273);
    v251 = 0;
LABEL_62:
    v35 = v262 + 1;
    v262 = v35;
    if ( v256 <= v35 )
    {
      v81 = v300;
      goto LABEL_85;
    }
  }
  if ( v257 )
  {
    v65 = strlen(v264[v35]);
    v22.m128i_i64[1] = v58;
    sub_C53770(v257, v58, v65, v35);
    goto LABEL_60;
  }
  v56 = v261;
  if ( !v261[10] )
    goto LABEL_57;
  v66 = strlen(v264[v35]);
  v308 = (const char *)v58;
  n = v66;
  v67 = (unsigned int)v300;
  LODWORD(v310[0]) = v35;
  v68 = v299;
  v69 = v259;
  if ( (unsigned __int64)(unsigned int)v300 + 1 > HIDWORD(v300) )
  {
    if ( v299 > (unsigned __int64)v259 || (unsigned __int64)v259 >= v299 + 24LL * (unsigned int)v300 )
    {
      v22.m128i_i64[1] = (__int64)v301;
      sub_C8D5F0(&v299, v301, (unsigned int)v300 + 1LL, 24);
      v68 = v299;
      v67 = (unsigned int)v300;
      v69 = v259;
    }
    else
    {
      v22.m128i_i64[1] = (__int64)v301;
      v173 = &v259->m128i_i8[-v299];
      sub_C8D5F0(&v299, v301, (unsigned int)v300 + 1LL, 24);
      v68 = v299;
      v67 = (unsigned int)v300;
      v69 = (const __m128i *)&v173[v299];
    }
  }
  v70 = (__m128i *)(v68 + 24 * v67);
  *v70 = _mm_loadu_si128(v69);
  v70[1].m128i_i64[0] = v69[1].m128i_i64[0];
  v71 = v300 + 1;
  LODWORD(v300) = v71;
  if ( v71 < v253 || !*((_QWORD *)v261 + 19) )
    goto LABEL_60;
  v72 = v262 + 1;
  v262 = v72;
  if ( v256 > v72 )
  {
    v73 = (__int64 *)v259;
    v74 = v256;
    do
    {
      v75 = (__int64)v264[v72];
      v76 = 0;
      if ( v75 )
        v76 = strlen(v264[v72]);
      n = v76;
      v22.m128i_i64[1] = HIDWORD(v300);
      v77 = v71;
      v78 = (const __m128i *)v73;
      v308 = (const char *)v75;
      v79 = v299;
      LODWORD(v310[0]) = v72;
      if ( (unsigned __int64)v71 + 1 > HIDWORD(v300) )
      {
        if ( v299 > (unsigned __int64)v73 || (unsigned __int64)v73 >= v299 + 24LL * v71 )
        {
          v22.m128i_i64[1] = (__int64)v301;
          sub_C8D5F0(&v299, v301, v71 + 1LL, 24);
          v79 = v299;
          v77 = (unsigned int)v300;
          v78 = (const __m128i *)v73;
        }
        else
        {
          v22.m128i_i64[1] = (__int64)v301;
          v136 = (char *)v73 - v299;
          sub_C8D5F0(&v299, v301, v71 + 1LL, 24);
          v79 = v299;
          v77 = (unsigned int)v300;
          v78 = (const __m128i *)&v136[v299];
        }
      }
      v80 = (__m128i *)(v79 + 24 * v77);
      *v80 = _mm_loadu_si128(v78);
      v80[1].m128i_i64[0] = v78[1].m128i_i64[0];
      v71 = v300 + 1;
      LODWORD(v300) = v300 + 1;
      v72 = v262 + 1;
      v262 = v72;
    }
    while ( v74 > v72 );
  }
  sub_2240A30(&v273);
  v81 = v300;
LABEL_85:
  if ( v81 < v253 )
  {
    v137 = sub_CB6200(v265, *v258, v258[1]);
    v138 = sub_904010(v137, ": Not enough positional command line arguments specified!\n");
    v139 = sub_904010(v138, "Must specify at least ");
    v140 = sub_CB59D0(v139, v245);
    v141 = "s";
    v142 = sub_904010(v140, " positional argument");
    if ( v253 <= 1 )
      v141 = byte_3F871B3;
    v143 = sub_904010(v142, v141);
    v144 = ": See: ";
    v145 = v143;
    goto LABEL_191;
  }
  v82 = v261[10];
  if ( !v250 && v82 < v81 )
  {
    v157 = sub_CB6200(v265, *v258, v258[1]);
    v158 = sub_904010(v157, ": Too many positional arguments specified!\n");
    v159 = sub_904010(v158, "Can specify at most ");
    v144 = " positional arguments: See: ";
    v145 = sub_CB59D0(v159, v261[10]);
LABEL_191:
    v146 = sub_904010(v145, v144);
    v22.m128i_i64[1] = (__int64)" --help\n";
    v22.m128i_i64[0] = sub_904010(v146, *v264);
    sub_904010(v22.m128i_i64[0], " --help\n");
    LOBYTE(v254) = 1;
    goto LABEL_102;
  }
LABEL_88:
  v22.m128i_i64[0] = (__int64)v261;
  v83 = (__int64 *)*((_QWORD *)v261 + 4);
  v84 = &v83[v82];
  if ( *((_QWORD *)v261 + 19) )
  {
    if ( v84 == v83 )
    {
      v89 = 1;
      v86 = 0;
    }
    else
    {
      v85 = (unsigned __int8)v254;
      v86 = 0;
      do
      {
        v22.m128i_i64[0] = *v83;
        if ( (*(_BYTE *)(*v83 + 12) & 7u) - 2 <= 1 )
        {
          v87 = v86++;
          v88 = (_BYTE *)(v299 + 24 * v87);
          v22.m128i_i64[1] = *(_QWORD *)v88;
          v85 |= sub_C53770(v22.m128i_i64[0], *(_QWORD *)v88, *((_QWORD *)v88 + 1), *((_DWORD *)v88 + 4));
        }
        ++v83;
      }
      while ( v84 != v83 );
      LOBYTE(v254) = v85;
      v89 = v86 == 0;
      v82 = v261[10];
    }
    if ( v82 == 1 && v89 )
    {
      if ( (_DWORD)v300 )
      {
        v86 = 1;
        v22.m128i_i64[1] = *(_QWORD *)v299;
        v22.m128i_i64[0] = **((_QWORD **)v261 + 4);
        v174 = sub_C53770(v22.m128i_i64[0], *(_QWORD *)v299, *(_QWORD *)(v299 + 8), *(_DWORD *)(v299 + 16));
        LOBYTE(v254) = v174 | v254;
        v90 = 1;
        goto LABEL_98;
      }
    }
    else
    {
      v90 = v86;
LABEL_98:
      if ( v86 != (_DWORD)v300 )
      {
        v91 = v261;
        v92 = (unsigned __int8)v254;
        do
        {
          v22.m128i_i64[0] = *((_QWORD *)v91 + 19);
          v93 = (_BYTE *)(v299 + 24 * v90);
          v22.m128i_i64[1] = *(_QWORD *)v93;
          v92 |= sub_C53770(v22.m128i_i64[0], *(_QWORD *)v93, *((_QWORD *)v93 + 1), *((_DWORD *)v93 + 4));
          v90 = v86 + 1;
          v86 = v90;
        }
        while ( (_DWORD)v300 != (_DWORD)v90 );
        LOBYTE(v254) = v92;
      }
    }
  }
  else if ( v84 != v83 )
  {
    v260 = &v83[v82];
    v161 = v253;
    v162 = 0;
    do
    {
      v165 = *v83;
      v166 = *(_BYTE *)(*v83 + 12) & 7;
      if ( (unsigned int)v166 - 2 <= 1 )
      {
        v167 = v162;
        v22.m128i_i64[0] = *v83;
        ++v162;
        --v161;
        v168 = (_BYTE *)(v299 + 24 * v167);
        v22.m128i_i64[1] = *(_QWORD *)v168;
        sub_C53770(*v83, *(_QWORD *)v168, *((_QWORD *)v168 + 1), *((_DWORD *)v168 + 4));
        v166 = *(_BYTE *)(v165 + 12) & 7;
      }
      if ( v161 < v81 - v162 && v166 != 2 )
      {
        while ( v166 )
        {
          if ( (v166 & 0xFD) != 1 )
            BUG();
          v169 = v162;
          v22.m128i_i64[0] = v165;
          ++v162;
          v170 = (_BYTE *)(v299 + 24 * v169);
          v22.m128i_i64[1] = *(_QWORD *)v170;
          sub_C53770(v165, *(_QWORD *)v170, *((_QWORD *)v170 + 1), *((_DWORD *)v170 + 4));
          if ( v161 >= v81 - v162 )
            goto LABEL_228;
          v166 = *(_BYTE *)(v165 + 12) & 7;
        }
        v163 = v162;
        v22.m128i_i64[0] = v165;
        ++v162;
        v164 = (_BYTE *)(v299 + 24 * v163);
        v22.m128i_i64[1] = *(_QWORD *)v164;
        sub_C53770(v165, *(_QWORD *)v164, *((_QWORD *)v164 + 1), *((_DWORD *)v164 + 4));
      }
LABEL_228:
      ++v83;
    }
    while ( v260 != v83 );
  }
LABEL_102:
  v94 = v261[34];
  if ( v94 )
  {
    v95 = (_QWORD *)*((_QWORD *)v261 + 16);
    v96 = *v95;
    v97 = v95;
    if ( *v95 != -8 )
      goto LABEL_105;
    do
    {
      do
      {
        v96 = v97[1];
        ++v97;
      }
      while ( v96 == -8 );
LABEL_105:
      ;
    }
    while ( !v96 );
    v98 = &v95[v94];
    if ( v98 != v97 )
    {
      while ( 1 )
      {
        v99 = *v97;
        v100 = *(_QWORD *)(*v97 + 8LL);
        if ( (*(_BYTE *)(v100 + 12) & 7u) - 2 <= 1 )
        {
          v261 = (unsigned int *)sub_C52410();
          v22.m128i_i64[1] = sub_C959E0();
          v104 = (_QWORD *)*((_QWORD *)v261 + 2);
          v105 = v261 + 2;
          if ( v104 )
          {
            v22.m128i_i64[0] = (__int64)(v261 + 2);
            do
            {
              while ( 1 )
              {
                v106 = v104[2];
                v107 = v104[3];
                if ( v22.m128i_i64[1] <= v104[4] )
                  break;
                v104 = (_QWORD *)v104[3];
                if ( !v107 )
                  goto LABEL_119;
              }
              v22.m128i_i64[0] = (__int64)v104;
              v104 = (_QWORD *)v104[2];
            }
            while ( v106 );
LABEL_119:
            if ( (_QWORD *)v22.m128i_i64[0] != v105 && v22.m128i_i64[1] >= *(_QWORD *)(v22.m128i_i64[0] + 32) )
              v105 = (_QWORD *)v22.m128i_i64[0];
          }
          if ( v105 == (_QWORD *)((char *)sub_C52410() + 8) )
            goto LABEL_130;
          v112 = v105[7];
          v111 = v105 + 6;
          if ( !v112 )
            goto LABEL_130;
          v22.m128i_i64[1] = *(unsigned int *)(v100 + 8);
          v22.m128i_i64[0] = (__int64)(v105 + 6);
          do
          {
            while ( 1 )
            {
              v109 = *(_QWORD *)(v112 + 16);
              v108 = *(_QWORD *)(v112 + 24);
              if ( *(_DWORD *)(v112 + 32) >= v22.m128i_i32[2] )
                break;
              v112 = *(_QWORD *)(v112 + 24);
              if ( !v108 )
                goto LABEL_128;
            }
            v22.m128i_i64[0] = v112;
            v112 = *(_QWORD *)(v112 + 16);
          }
          while ( v109 );
LABEL_128:
          if ( (_QWORD *)v22.m128i_i64[0] == v111
            || v22.m128i_i32[2] < *(_DWORD *)(v22.m128i_i64[0] + 32)
            || !*(_DWORD *)(v22.m128i_i64[0] + 36) )
          {
LABEL_130:
            v113 = *(_QWORD *)(v99 + 8);
            v114 = sub_CEADF0(v22.m128i_i64[0], v22.m128i_i64[1], v108, v109, v110, v111);
            v22.m128i_i64[1] = (__int64)v259;
            v22.m128i_i64[0] = v113;
            v312 = 1;
            v308 = "must be specified at least once!";
            v311 = 3;
            sub_C53280(v113, (__int64)v259, 0, 0, v114);
            LOBYTE(v254) = 1;
          }
        }
        v101 = v97[1];
        v102 = v97 + 1;
        if ( v101 == -8 || !v101 )
        {
          do
          {
            do
            {
              v103 = v102[1];
              ++v102;
            }
            while ( !v103 );
          }
          while ( v103 == -8 );
        }
        if ( v98 == v102 )
          break;
        v97 = v102;
      }
    }
  }
  v115 = v258[6];
  if ( v115 != v258[7] )
    v258[7] = v115;
  if ( (_BYTE)v254 )
  {
    if ( !a5 )
      exit(1);
    v24 = 0;
  }
  else
  {
    v24 = 1;
  }
  if ( (_BYTE *)v299 != v301 )
    _libc_free(v299, v22.m128i_i64[1]);
  sub_2240A30(&v270);
LABEL_141:
  v116 = v293;
  v117 = &v293[(unsigned int)v294];
  if ( v293 != v117 )
  {
    for ( i = v293; ; i = v293 )
    {
      v22.m128i_i64[0] = *v116;
      v119 = (unsigned int)(v116 - i) >> 7;
      v22.m128i_i64[1] = 4096LL << v119;
      if ( v119 >= 0x1E )
        v22.m128i_i64[1] = 0x40000000000LL;
      ++v116;
      sub_C7D6A0(v22.m128i_i64[0], v22.m128i_i64[1], 16);
      if ( v117 == v116 )
        break;
    }
  }
  v120 = v296;
  v121 = &v296[(unsigned int)v297];
  if ( v296 != v121 )
  {
    do
    {
      v22 = *v120++;
      sub_C7D6A0(v22.m128i_i64[0], v22.m128i_i64[1], 16);
    }
    while ( v121 != v120 );
    v121 = v296;
  }
  if ( v121 != (__m128i *)v298 )
    _libc_free(v121, v22.m128i_i64[1]);
  if ( v293 != (__int64 *)v295 )
    _libc_free(v293, v22.m128i_i64[1]);
  if ( v305 != (const char **)v307 )
    _libc_free(v305, v22.m128i_i64[1]);
  v122 = v286;
  v123 = &v286[(unsigned int)v287];
  if ( v286 != v123 )
  {
    for ( j = v286; ; j = v286 )
    {
      v22.m128i_i64[0] = *v122;
      v125 = (unsigned int)(v122 - j) >> 7;
      v22.m128i_i64[1] = 4096LL << v125;
      if ( v125 >= 0x1E )
        v22.m128i_i64[1] = 0x40000000000LL;
      ++v122;
      sub_C7D6A0(v22.m128i_i64[0], v22.m128i_i64[1], 16);
      if ( v123 == v122 )
        break;
    }
  }
  v126 = v289;
  v127 = &v289[(unsigned int)v290];
  if ( v289 != v127 )
  {
    do
    {
      v22 = *v126++;
      sub_C7D6A0(v22.m128i_i64[0], v22.m128i_i64[1], 16);
    }
    while ( v127 != v126 );
    v127 = v289;
  }
  if ( v127 != (__m128i *)v291 )
    _libc_free(v127, v22.m128i_i64[1]);
  if ( v286 != (__int64 *)v288 )
    _libc_free(v286, v22.m128i_i64[1]);
  if ( src != v304 )
    _libc_free(src, v22.m128i_i64[1]);
  return v24;
}
