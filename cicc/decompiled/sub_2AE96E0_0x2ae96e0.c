// Function: sub_2AE96E0
// Address: 0x2ae96e0
//
__int64 __fastcall sub_2AE96E0(__int64 a1, __int64 a2, unsigned int *a3, unsigned __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 *v7; // rax
  __int64 *v8; // r13
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 *v11; // r9
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // r14
  __m128i v16; // xmm3
  char *v17; // r13
  char *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  char *v23; // r12
  char *v24; // rbx
  _BYTE *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  _QWORD *v28; // rax
  __int64 v29; // rdx
  int v30; // r13d
  int v31; // esi
  __int64 *v32; // rax
  __int64 v33; // r10
  _DWORD *v34; // rax
  char *v35; // rax
  char *v36; // rsi
  __m128i *v37; // rax
  char v38; // cl
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 *v41; // rbx
  __int64 *v42; // r12
  __int64 v43; // rax
  __int64 *v44; // r14
  __m128i *v45; // r11
  int v46; // edx
  int v47; // esi
  __int64 v48; // rdi
  unsigned int *v49; // rax
  __int64 v50; // r12
  __int64 v51; // rbx
  __int64 v52; // rbx
  __int64 v53; // r8
  _DWORD *v54; // rsi
  _DWORD *v55; // rdx
  _DWORD *v56; // rdx
  __int64 v57; // rcx
  __m128i *v58; // rax
  __m128i *v59; // r9
  __m128i *v60; // rsi
  __m128i *i; // r8
  __m128i *v62; // rdx
  __int64 v63; // r8
  __int64 v64; // r9
  unsigned int v65; // eax
  __m128i *v66; // r11
  __int64 v67; // rdx
  unsigned int v68; // edi
  __int64 v69; // rcx
  __int64 *v70; // r12
  __int64 *j; // r13
  __int64 v72; // rsi
  __int64 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned int *v76; // r13
  unsigned __int64 *v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // r8
  int v82; // esi
  __int64 v83; // rdi
  __int64 v84; // r10
  int v85; // esi
  unsigned int v86; // edx
  __int64 *v87; // rax
  __int64 v88; // r12
  __int64 v89; // rax
  char v90; // r14
  unsigned int v91; // r12d
  __int64 v92; // rsi
  __int64 v93; // rdx
  int v94; // eax
  int v95; // r10d
  unsigned int m; // eax
  __int64 v97; // r11
  unsigned int v98; // eax
  unsigned int v99; // esi
  unsigned int v100; // eax
  unsigned int v101; // edx
  int v102; // r11d
  unsigned int v103; // ecx
  int v104; // r10d
  __int64 v105; // rax
  int v106; // esi
  unsigned int v107; // ecx
  int v108; // r10d
  __int64 *v109; // r11
  int v110; // eax
  _QWORD *v111; // rax
  _QWORD *v112; // rdx
  __int64 *v113; // rdi
  unsigned __int8 v114; // si
  int v115; // eax
  __int64 v116; // rdx
  __int64 v117; // r8
  __int64 v118; // r9
  __int64 v119; // rcx
  __int64 v120; // rbx
  int v121; // ebx
  _DWORD *v122; // rax
  char v123; // al
  unsigned int v124; // ebx
  void *v125; // rax
  void *v126; // rdi
  char v127; // dl
  void **v128; // rsi
  size_t v129; // rdx
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // rbx
  char v139; // al
  __int64 v140; // r12
  __int64 v141; // rdi
  void *v142; // rax
  void *p_dest; // rdi
  const void *v144; // rsi
  size_t v145; // rdx
  __int64 v146; // rdx
  __int64 v147; // r8
  __int64 v148; // r9
  __int64 v149; // rcx
  __int64 v150; // rbx
  char v151; // al
  int v152; // r12d
  __int64 v153; // rdi
  __int64 v154; // rax
  void *v155; // rdi
  const void *v156; // rsi
  size_t v157; // rdx
  __int64 v158; // rdx
  __int64 v159; // rcx
  __int64 v160; // r8
  __int64 v161; // r9
  char v162; // al
  int v163; // r12d
  __int64 v164; // rdi
  __int64 v165; // rax
  void *v166; // rdi
  const void *v167; // rsi
  size_t v168; // rdx
  __int64 v169; // rbx
  __m128i *v170; // r12
  unsigned __int64 v171; // rdi
  __m128i *v172; // r12
  __int64 v173; // rsi
  __m128i *v174; // rbx
  unsigned __int64 v175; // rdi
  int v177; // r11d
  __int64 *v178; // rax
  __int64 *v179; // rax
  __int64 v180; // rdx
  unsigned int v181; // esi
  unsigned __int32 v182; // eax
  __int64 v183; // rdx
  unsigned int v184; // edi
  __int64 v185; // rcx
  unsigned int v186; // edx
  unsigned __int32 v187; // eax
  __int64 v188; // rcx
  unsigned int v189; // edi
  unsigned int v190; // esi
  __int64 v191; // rax
  int v192; // edx
  unsigned int *v193; // r12
  __int64 v194; // r13
  unsigned __int64 *v195; // rax
  char v196; // si
  int v197; // eax
  unsigned int v198; // edx
  __int64 *v199; // rdx
  __int64 v200; // rax
  unsigned __int64 v201; // rdx
  __int64 v202; // r8
  __int64 v203; // r9
  __int64 *k; // rcx
  __int64 v205; // rbx
  _QWORD *v206; // rax
  __int64 *v207; // rax
  _BYTE *v208; // rdi
  _BYTE *v209; // rbx
  unsigned __int64 v210; // r14
  _DWORD *v211; // rax
  char v212; // di
  __int64 v213; // rdx
  __int64 v214; // rsi
  __int64 v215; // rax
  unsigned int v216; // ecx
  int v217; // r11d
  __int64 v218; // r9
  int v219; // ecx
  int v220; // eax
  __int64 v221; // rdx
  __int64 v222; // rcx
  __int64 v223; // r8
  __int64 v224; // r9
  __int64 v225; // rbx
  int v226; // ebx
  _DWORD *v227; // rax
  __int64 v228; // rdx
  __int64 v229; // rcx
  __int64 v230; // r8
  __int64 v231; // r9
  _DWORD *v232; // rax
  __int64 v233; // r8
  int v234; // ecx
  int v235; // eax
  int v236; // r10d
  unsigned int v237; // ecx
  int *v238; // rdi
  int v239; // r9d
  unsigned int v240; // ecx
  __int64 *v241; // rdx
  __int64 v242; // rax
  __int64 v243; // rdx
  __int64 v244; // rcx
  __int64 v245; // r8
  __int64 v246; // r9
  _DWORD *v247; // rax
  __int64 *v248; // rax
  int v249; // r10d
  int v250; // r10d
  unsigned int v251; // esi
  __int64 *v252; // rbx
  __int64 v253; // rax
  __int64 v254; // r13
  __int32 v255; // r14d
  unsigned int v257; // [rsp+44h] [rbp-C5Ch]
  __int64 v258; // [rsp+90h] [rbp-C10h]
  char v259; // [rsp+90h] [rbp-C10h]
  __int64 v261; // [rsp+A0h] [rbp-C00h]
  unsigned __int64 v262; // [rsp+A8h] [rbp-BF8h]
  __int32 v263; // [rsp+A8h] [rbp-BF8h]
  __int64 v265; // [rsp+B0h] [rbp-BF0h]
  __int64 v266; // [rsp+B8h] [rbp-BE8h]
  _BYTE *v267; // [rsp+B8h] [rbp-BE8h]
  __int64 v268; // [rsp+C0h] [rbp-BE0h]
  __int64 v269; // [rsp+C0h] [rbp-BE0h]
  unsigned __int64 v270; // [rsp+C8h] [rbp-BD8h]
  unsigned int v271; // [rsp+D0h] [rbp-BD0h] BYREF
  int v272; // [rsp+D4h] [rbp-BCCh] BYREF
  __int64 *v273; // [rsp+D8h] [rbp-BC8h] BYREF
  unsigned __int64 v274; // [rsp+E0h] [rbp-BC0h] BYREF
  __int64 *v275; // [rsp+E8h] [rbp-BB8h]
  __int64 *v276; // [rsp+F0h] [rbp-BB0h]
  __int64 v277; // [rsp+F8h] [rbp-BA8h]
  __int64 *v278; // [rsp+100h] [rbp-BA0h] BYREF
  __int64 v279; // [rsp+108h] [rbp-B98h]
  __int64 *v280; // [rsp+110h] [rbp-B90h]
  __int64 v281; // [rsp+118h] [rbp-B88h]
  __int64 v282[4]; // [rsp+120h] [rbp-B80h] BYREF
  unsigned int v283; // [rsp+140h] [rbp-B60h]
  unsigned __int64 v284; // [rsp+148h] [rbp-B58h]
  __int64 v285; // [rsp+150h] [rbp-B50h]
  __int64 v286; // [rsp+160h] [rbp-B40h] BYREF
  char *v287; // [rsp+168h] [rbp-B38h]
  __int64 v288; // [rsp+170h] [rbp-B30h]
  int v289; // [rsp+178h] [rbp-B28h]
  char v290; // [rsp+17Ch] [rbp-B24h]
  char v291; // [rsp+180h] [rbp-B20h] BYREF
  _BYTE *v292; // [rsp+1C0h] [rbp-AE0h] BYREF
  __int64 *v293; // [rsp+1C8h] [rbp-AD8h]
  __int64 v294; // [rsp+1D0h] [rbp-AD0h]
  int v295; // [rsp+1D8h] [rbp-AC8h]
  char v296; // [rsp+1DCh] [rbp-AC4h]
  char v297; // [rsp+1E0h] [rbp-AC0h] BYREF
  __m128i v298; // [rsp+220h] [rbp-A80h] BYREF
  void *src[2]; // [rsp+230h] [rbp-A70h] BYREF
  _BYTE v300[16]; // [rsp+240h] [rbp-A60h] BYREF
  _BYTE *v301; // [rsp+250h] [rbp-A50h] BYREF
  __int64 v302; // [rsp+258h] [rbp-A48h]
  _BYTE v303[32]; // [rsp+260h] [rbp-A40h] BYREF
  __int64 v304; // [rsp+280h] [rbp-A20h] BYREF
  __int64 v305; // [rsp+288h] [rbp-A18h]
  __int64 v306; // [rsp+290h] [rbp-A10h]
  __int64 v307; // [rsp+298h] [rbp-A08h]
  _BYTE *v308; // [rsp+2A0h] [rbp-A00h]
  __int64 v309; // [rsp+2A8h] [rbp-9F8h]
  _BYTE v310[64]; // [rsp+2B0h] [rbp-9F0h] BYREF
  __int64 v311; // [rsp+2F0h] [rbp-9B0h] BYREF
  unsigned __int64 v312; // [rsp+2F8h] [rbp-9A8h]
  void *v313; // [rsp+300h] [rbp-9A0h] BYREF
  unsigned int v314; // [rsp+308h] [rbp-998h]
  _QWORD v315[2]; // [rsp+320h] [rbp-980h] BYREF
  char v316; // [rsp+330h] [rbp-970h] BYREF
  __int64 v317; // [rsp+350h] [rbp-950h] BYREF
  unsigned int v318; // [rsp+358h] [rbp-948h]
  int v319; // [rsp+35Ch] [rbp-944h]
  void *dest; // [rsp+360h] [rbp-940h] BYREF
  unsigned int v321; // [rsp+368h] [rbp-938h]
  _QWORD v322[2]; // [rsp+380h] [rbp-920h] BYREF
  char v323; // [rsp+390h] [rbp-910h] BYREF
  __int64 v324; // [rsp+3B0h] [rbp-8F0h] BYREF
  __int64 v325; // [rsp+3B8h] [rbp-8E8h]
  __int64 *v326; // [rsp+3C0h] [rbp-8E0h] BYREF
  unsigned int v327; // [rsp+3C8h] [rbp-8D8h]
  _BYTE *v328; // [rsp+4C0h] [rbp-7E0h] BYREF
  __int64 v329; // [rsp+4C8h] [rbp-7D8h]
  _BYTE v330[512]; // [rsp+4D0h] [rbp-7D0h] BYREF
  __m128i v331; // [rsp+6D0h] [rbp-5D0h] BYREF
  __m128i v332; // [rsp+6E0h] [rbp-5C0h] BYREF
  _BYTE v333[16]; // [rsp+6F0h] [rbp-5B0h] BYREF
  void (__fastcall *v334)(_QWORD, _QWORD, _QWORD); // [rsp+700h] [rbp-5A0h]
  __int64 v335; // [rsp+708h] [rbp-598h]
  __m128i v336; // [rsp+960h] [rbp-340h] BYREF
  __m128i v337; // [rsp+970h] [rbp-330h] BYREF
  _BYTE v338[16]; // [rsp+980h] [rbp-320h] BYREF
  _BYTE *v339; // [rsp+990h] [rbp-310h]
  __int64 v340; // [rsp+998h] [rbp-308h]
  __m128i v341; // [rsp+9A0h] [rbp-300h] BYREF
  __m128i v342; // [rsp+9B0h] [rbp-2F0h] BYREF
  _BYTE v343[16]; // [rsp+9C0h] [rbp-2E0h] BYREF
  void (__fastcall *v344)(_BYTE *, _BYTE *, __int64); // [rsp+9D0h] [rbp-2D0h]
  __int64 v345; // [rsp+9D8h] [rbp-2C8h]

  sub_D33BC0((__int64)v282, *(_QWORD *)(a2 + 416));
  sub_D4E470(v282, *(_QWORD *)(a2 + 432));
  LOBYTE(v312) = v312 | 1;
  v311 = 0;
  sub_2AC3540((__int64)&v311);
  LOBYTE(v318) = v318 | 1;
  v315[0] = &v316;
  v315[1] = 0x400000000LL;
  v317 = 0;
  sub_2AC3540((__int64)&v317);
  v322[1] = 0x400000000LL;
  v322[0] = &v323;
  v328 = v330;
  v329 = 0x4000000000LL;
  v7 = (unsigned __int64 *)&v326;
  v324 = 0;
  v325 = 1;
  do
  {
    *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != (unsigned __int64 *)&v328 );
  v286 = 0;
  v287 = &v291;
  v8 = &v286;
  v308 = v310;
  v309 = 0x800000000LL;
  v288 = 8;
  v289 = 0;
  v290 = 1;
  v304 = 0;
  v305 = 0;
  v306 = 0;
  v307 = 0;
  v262 = v284;
  v266 = v285;
  if ( v284 != v285 )
  {
    while ( 1 )
    {
      v9 = (__int64)&v336;
      sub_AA72C0(&v336, *(_QWORD *)(v266 - 8), 1);
      v12 = _mm_loadu_si128(&v336);
      v13 = _mm_loadu_si128(&v337);
      v301 = 0;
      v298 = v12;
      *(__m128i *)src = v13;
      if ( v339 )
      {
        v9 = (__int64)v300;
        ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v339)(v300, v338, 2);
        v302 = v340;
        v301 = v339;
      }
      v14 = _mm_loadu_si128(&v341);
      v15 = (__int64)v8;
      v16 = _mm_loadu_si128(&v342);
      v334 = 0;
      v331 = v14;
      v332 = v16;
      if ( v344 )
      {
        v9 = (__int64)v333;
        v344(v333, v343, 2);
        v335 = v345;
        v334 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v344;
      }
LABEL_8:
      v17 = (char *)v298.m128i_i64[0];
      if ( v298.m128i_i64[0] != v331.m128i_i64[0] )
        break;
LABEL_40:
      v8 = (__int64 *)v15;
      sub_A17130((__int64)v333);
      sub_A17130((__int64)v300);
      sub_A17130((__int64)v343);
      sub_A17130((__int64)v338);
      v266 -= 8;
      if ( v262 == v266 )
        goto LABEL_41;
    }
    while ( 1 )
    {
      if ( v17 )
      {
        v17 -= 24;
        v18 = v17;
      }
      else
      {
        v18 = 0;
      }
      v19 = (unsigned int)v329;
      v20 = HIDWORD(v329);
      v21 = (unsigned int)v329 + 1LL;
      if ( v21 > HIDWORD(v329) )
      {
        v9 = (__int64)&v328;
        sub_C8D5F0((__int64)&v328, v330, v21, 8u, v10, (__int64)v11);
        v19 = (unsigned int)v329;
      }
      v22 = (__int64)v328;
      *(_QWORD *)&v328[8 * v19] = v18;
      LODWORD(v329) = v329 + 1;
      if ( (v17[7] & 0x40) != 0 )
      {
        v23 = (char *)*((_QWORD *)v17 - 1);
        v24 = &v23[32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
      }
      else
      {
        v24 = v17;
        v23 = &v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
      }
      if ( v23 != v24 )
        break;
LABEL_32:
      v17 = *(char **)(v298.m128i_i64[0] + 8);
      v298.m128i_i16[4] = 0;
      v298.m128i_i64[0] = (__int64)v17;
      v36 = v17;
      if ( v17 != src[0] )
      {
        do
        {
          if ( v36 )
            v36 -= 24;
          if ( !v301 )
            sub_4263D6(v9, v36, v22);
          v9 = (__int64)v300;
          if ( ((unsigned __int8 (__fastcall *)(_BYTE *, char *))v302)(v300, v36) )
            goto LABEL_8;
          v22 = 0;
          v36 = *(char **)(v298.m128i_i64[0] + 8);
          v298.m128i_i16[4] = 0;
          v298.m128i_i64[0] = (__int64)v36;
        }
        while ( src[0] != v36 );
        v17 = v36;
      }
      if ( (char *)v331.m128i_i64[0] == v17 )
        goto LABEL_40;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v25 = *(_BYTE **)v23;
        if ( **(_BYTE **)v23 > 0x1Cu )
          break;
LABEL_31:
        v23 += 32;
        if ( v24 == v23 )
          goto LABEL_32;
      }
      v26 = *(_QWORD *)(a2 + 416);
      v292 = *(_BYTE **)v23;
      v27 = *((_QWORD *)v25 + 5);
      if ( *(_BYTE *)(v26 + 84) )
        break;
      if ( sub_C8CA60(v26 + 56, v27) )
      {
LABEL_22:
        v30 = v329;
        v20 = v325 & 1;
        if ( (v325 & 1) != 0 )
        {
          v11 = (__int64 *)&v326;
          v31 = 15;
        }
        else
        {
          v99 = v327;
          v11 = v326;
          if ( !v327 )
          {
            v100 = v325;
            ++v324;
            v10 = 0;
            v101 = ((unsigned int)v325 >> 1) + 1;
            goto LABEL_132;
          }
          v31 = v327 - 1;
        }
        v9 = (__int64)v292;
        v22 = v31 & (((unsigned int)v292 >> 9) ^ ((unsigned int)v292 >> 4));
        v32 = &v11[2 * v22];
        v33 = *v32;
        if ( v292 == (_BYTE *)*v32 )
          goto LABEL_25;
        v102 = 1;
        v10 = 0;
        while ( 1 )
        {
          if ( v33 == -4096 )
          {
            v9 = 48;
            v99 = 16;
            if ( !v10 )
              v10 = (__int64)v32;
            v100 = v325;
            ++v324;
            v101 = ((unsigned int)v325 >> 1) + 1;
            if ( !(_BYTE)v20 )
            {
              v99 = v327;
LABEL_132:
              v9 = 3 * v99;
            }
            if ( 4 * v101 < (unsigned int)v9 )
            {
              if ( v99 - HIDWORD(v325) - v101 > v99 >> 3 )
              {
                v22 = (__int64)v292;
LABEL_136:
                v20 = 2 * (v100 >> 1) + 2;
                LODWORD(v325) = v20 | v100 & 1;
                if ( *(_QWORD *)v10 != -4096 )
                  --HIDWORD(v325);
                *(_QWORD *)v10 = v22;
                v34 = (_DWORD *)(v10 + 8);
                *(_DWORD *)(v10 + 8) = 0;
                goto LABEL_26;
              }
              sub_2ACEE70((__int64)&v324, v99);
              if ( (v325 & 1) != 0 )
              {
                v9 = (__int64)&v326;
                v106 = 15;
                goto LABEL_156;
              }
              v9 = (__int64)v326;
              if ( v327 )
              {
                v106 = v327 - 1;
LABEL_156:
                v22 = (__int64)v292;
                v100 = v325;
                v107 = v106 & (((unsigned int)v292 >> 9) ^ ((unsigned int)v292 >> 4));
                v10 = v9 + 16LL * v107;
                v11 = *(__int64 **)v10;
                if ( *(_BYTE **)v10 == v292 )
                  goto LABEL_136;
                v108 = 1;
                v105 = 0;
                while ( v11 != (__int64 *)-4096LL )
                {
                  if ( v11 == (__int64 *)-8192LL && !v105 )
                    v105 = v10;
                  v107 = v106 & (v108 + v107);
                  v10 = v9 + 16LL * v107;
                  v11 = *(__int64 **)v10;
                  if ( v292 == *(_BYTE **)v10 )
                    goto LABEL_153;
                  ++v108;
                }
                if ( !v105 )
                {
LABEL_153:
                  v100 = v325;
                  goto LABEL_136;
                }
LABEL_152:
                v10 = v105;
                goto LABEL_153;
              }
LABEL_464:
              LODWORD(v325) = (2 * ((unsigned int)v325 >> 1) + 2) | v325 & 1;
              BUG();
            }
            sub_2ACEE70((__int64)&v324, 2 * v99);
            if ( (v325 & 1) != 0 )
            {
              v11 = (__int64 *)&v326;
              v9 = 15;
            }
            else
            {
              v11 = v326;
              if ( !v327 )
                goto LABEL_464;
              v9 = v327 - 1;
            }
            v100 = v325;
            v103 = v9 & (((unsigned int)v292 >> 9) ^ ((unsigned int)v292 >> 4));
            v10 = (__int64)&v11[2 * v103];
            v22 = *(_QWORD *)v10;
            if ( v292 == *(_BYTE **)v10 )
              goto LABEL_136;
            v104 = 1;
            v105 = 0;
            while ( v22 != -4096 )
            {
              if ( !v105 && v22 == -8192 )
                v105 = v10;
              v103 = v9 & (v104 + v103);
              v10 = (__int64)&v11[2 * v103];
              v22 = *(_QWORD *)v10;
              if ( v292 == *(_BYTE **)v10 )
                goto LABEL_153;
              ++v104;
            }
            v22 = (__int64)v292;
            if ( !v105 )
              goto LABEL_153;
            goto LABEL_152;
          }
          if ( v33 != -8192 || v10 )
            v32 = (__int64 *)v10;
          v10 = (unsigned int)(v102 + 1);
          v22 = v31 & (unsigned int)(v102 + v22);
          v109 = &v11[2 * (unsigned int)v22];
          v33 = *v109;
          if ( v292 == (_BYTE *)*v109 )
            break;
          v102 = v10;
          v10 = (__int64)v32;
          v32 = &v11[2 * (unsigned int)v22];
        }
        v32 = &v11[2 * (unsigned int)v22];
LABEL_25:
        v34 = v32 + 1;
LABEL_26:
        *v34 = v30;
        if ( v290 )
        {
          v35 = v287;
          v20 = HIDWORD(v288);
          v22 = (__int64)&v287[8 * HIDWORD(v288)];
          if ( v287 != (char *)v22 )
          {
            while ( v292 != *(_BYTE **)v35 )
            {
              v35 += 8;
              if ( (char *)v22 == v35 )
                goto LABEL_128;
            }
            goto LABEL_31;
          }
LABEL_128:
          if ( HIDWORD(v288) >= (unsigned int)v288 )
            goto LABEL_126;
          v20 = (unsigned int)(HIDWORD(v288) + 1);
          v23 += 32;
          ++HIDWORD(v288);
          *(_QWORD *)v22 = v292;
          ++v286;
          if ( v24 == v23 )
            goto LABEL_32;
        }
        else
        {
LABEL_126:
          v9 = v15;
          v23 += 32;
          sub_C8CC70(v15, (__int64)v292, v22, v20, v10, (__int64)v11);
          if ( v24 == v23 )
            goto LABEL_32;
        }
      }
      else
      {
LABEL_124:
        v9 = (__int64)&v304;
        v23 += 32;
        sub_2ADDD60((__int64)&v304, (__int64 *)&v292, v29, v20, v10, (__int64)v11);
        if ( v24 == v23 )
          goto LABEL_32;
      }
    }
    v28 = *(_QWORD **)(v26 + 64);
    v29 = (__int64)&v28[*(unsigned int *)(v26 + 76)];
    if ( v28 == (_QWORD *)v29 )
      goto LABEL_124;
    while ( v27 != *v28 )
    {
      if ( (_QWORD *)v29 == ++v28 )
        goto LABEL_124;
    }
    goto LABEL_22;
  }
LABEL_41:
  v37 = &v332;
  v331.m128i_i64[0] = 0;
  v331.m128i_i64[1] = 1;
  do
  {
    v37->m128i_i32[0] = -1;
    v37 = (__m128i *)((char *)v37 + 40);
  }
  while ( v37 != &v336 );
  v38 = v325 & 1;
  if ( (unsigned int)v325 >> 1 )
  {
    if ( v38 )
    {
      v42 = (__int64 *)&v328;
      v41 = (__int64 *)&v326;
    }
    else
    {
      v39 = v327;
      v40 = v326;
      v41 = v326;
      v42 = &v326[2 * v327];
      if ( v42 == v326 )
      {
LABEL_51:
        v43 = 2 * v39;
        goto LABEL_52;
      }
    }
    do
    {
      if ( *v41 != -4096 && *v41 != -8192 )
        break;
      v41 += 2;
    }
    while ( v41 != v42 );
  }
  else
  {
    if ( v38 )
    {
      v252 = (__int64 *)&v326;
      v253 = 32;
    }
    else
    {
      v252 = v326;
      v253 = 2LL * v327;
    }
    v41 = &v252[v253];
    v42 = v41;
  }
  if ( !v38 )
  {
    v40 = v326;
    v39 = v327;
    goto LABEL_51;
  }
  v40 = (__int64 *)&v326;
  v43 = 32;
LABEL_52:
  v44 = &v40[v43];
  if ( &v40[v43] != v41 )
  {
    while ( 1 )
    {
      if ( (v331.m128i_i8[8] & 1) != 0 )
      {
        v45 = &v332;
        v46 = 15;
      }
      else
      {
        v181 = v332.m128i_u32[2];
        v45 = (__m128i *)v332.m128i_i64[0];
        v46 = v332.m128i_i32[2] - 1;
        if ( !v332.m128i_i32[2] )
        {
          v182 = v331.m128i_u32[2];
          ++v331.m128i_i64[0];
          v336.m128i_i64[0] = 0;
          v183 = ((unsigned __int32)v331.m128i_i32[2] >> 1) + 1;
          goto LABEL_286;
        }
      }
      v47 = *((_DWORD *)v41 + 2);
      LODWORD(v48) = v46 & (37 * v47);
      v49 = (unsigned int *)v45 + 10 * (unsigned int)v48;
      v5 = *v49;
      if ( v47 != (_DWORD)v5 )
        break;
LABEL_56:
      sub_9C95B0((__int64)(v49 + 2), *v41);
      do
        v41 += 2;
      while ( v41 != v42 && (*v41 == -8192 || *v41 == -4096) );
      if ( v41 == v44 )
        goto LABEL_61;
    }
    v249 = 1;
    v6 = 0;
    while ( (_DWORD)v5 != -1 )
    {
      if ( !v6 && (_DWORD)v5 == -2 )
        v6 = (__int64)v49;
      v48 = v46 & (unsigned int)(v48 + v249);
      v49 = (unsigned int *)v45 + 10 * v48;
      v5 = *v49;
      if ( v47 == (_DWORD)v5 )
        goto LABEL_56;
      ++v249;
    }
    v184 = 48;
    v181 = 16;
    if ( v6 )
      v49 = (unsigned int *)v6;
    ++v331.m128i_i64[0];
    v336.m128i_i64[0] = (__int64)v49;
    v182 = v331.m128i_u32[2];
    v183 = ((unsigned __int32)v331.m128i_i32[2] >> 1) + 1;
    if ( (v331.m128i_i8[8] & 1) == 0 )
    {
      v181 = v332.m128i_u32[2];
LABEL_286:
      v184 = 3 * v181;
    }
    v185 = (unsigned int)(4 * v183);
    if ( (unsigned int)v185 >= v184 )
    {
      v181 *= 2;
    }
    else
    {
      v185 = v181 - v331.m128i_i32[3] - (unsigned int)v183;
      v183 = v181 >> 3;
      if ( (unsigned int)v185 > (unsigned int)v183 )
      {
LABEL_289:
        v331.m128i_i32[2] = (2 * (v182 >> 1) + 2) | v182 & 1;
        v49 = (unsigned int *)v336.m128i_i64[0];
        if ( *(_DWORD *)v336.m128i_i64[0] != -1 )
          --v331.m128i_i32[3];
        v186 = *((_DWORD *)v41 + 2);
        *(_QWORD *)(v336.m128i_i64[0] + 16) = 0x200000000LL;
        *v49 = v186;
        *((_QWORD *)v49 + 1) = v49 + 6;
        goto LABEL_56;
      }
    }
    sub_2ACF290((__int64)&v331, v181, v183, v185, v5, v6);
    sub_2AC1640((__int64)&v331, (int *)v41 + 2, &v336);
    v182 = v331.m128i_u32[2];
    goto LABEL_289;
  }
LABEL_61:
  v292 = 0;
  v293 = (__int64 *)&v297;
  v50 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  *(_QWORD *)a1 = a1 + 16;
  v294 = 8;
  v295 = 0;
  v296 = 1;
  if ( !a4 )
  {
    v336.m128i_i64[1] = 0x800000000LL;
    v261 = *(_QWORD *)(a2 + 448);
    v336.m128i_i64[0] = (__int64)&v337;
    sub_2AD3790(a2);
    v271 = 0;
    v257 = v329;
    if ( !(_DWORD)v329 )
      goto LABEL_240;
    v263 = 0;
    goto LABEL_86;
  }
  v51 = 192 * a4;
  if ( a4 <= 8 )
  {
    v52 = v50 + v51;
    if ( v50 == v52 )
    {
      v59 = &v337;
      v336.m128i_i64[1] = 0x800000000LL;
      v336.m128i_i64[0] = (__int64)&v337;
      v58 = &v337;
      v263 = a4;
      *(_DWORD *)(a1 + 8) = a4;
      goto LABEL_77;
    }
    goto LABEL_64;
  }
  v254 = sub_C8D7D0(a1, v50, a4, 0xC0u, (unsigned __int64 *)&v336, v6);
  sub_2AE94D0((__int64 *)a1, v254);
  v255 = v336.m128i_i32[0];
  if ( v50 != *(_QWORD *)a1 )
    _libc_free(*(_QWORD *)a1);
  v52 = v254 + v51;
  *(_QWORD *)a1 = v254;
  *(_DWORD *)(a1 + 12) = v255;
  v50 = v254 + 192LL * *(unsigned int *)(a1 + 8);
  if ( v50 != v52 )
  {
LABEL_64:
    v53 = v50 + 144;
    v54 = (_DWORD *)(v50 + 48);
    do
    {
      if ( v50 )
      {
        v55 = (_DWORD *)(v50 + 16);
        memset((void *)v50, 0, 0xC0u);
        *(_DWORD *)(v50 + 8) = 1;
        do
        {
          if ( v55 )
            *v55 = -1;
          v55 += 2;
        }
        while ( v55 != v54 );
        *(_DWORD *)(v50 + 56) = 0;
        *(_DWORD *)(v50 + 60) = 4;
        *(_QWORD *)(v50 + 96) = 0;
        *(_DWORD *)(v50 + 104) = 1;
        *(_DWORD *)(v50 + 108) = 0;
        *(_QWORD *)(v50 + 48) = v50 + 64;
        v56 = (_DWORD *)(v50 + 112);
        do
        {
          if ( v56 )
            *v56 = -1;
          v56 += 2;
        }
        while ( v56 != (_DWORD *)v53 );
        v40 = (__int64 *)(v50 + 160);
        *(_DWORD *)(v50 + 152) = 0;
        *(_QWORD *)(v50 + 144) = v50 + 160;
        *(_DWORD *)(v50 + 156) = 4;
      }
      v50 += 192;
      v53 += 192;
      v54 += 48;
    }
    while ( v50 != v52 );
    v57 = a4;
    v336.m128i_i64[1] = 0x800000000LL;
    *(_DWORD *)(a1 + 8) = a4;
    v58 = &v337;
    v263 = a4;
    v59 = &v337;
    v336.m128i_i64[0] = (__int64)&v337;
    if ( a4 <= 8 )
      goto LABEL_77;
    goto LABEL_440;
  }
  v57 = a1;
  v263 = a4;
  *(_DWORD *)(a1 + 8) = a4;
  v336.m128i_i64[0] = (__int64)&v337;
  v336.m128i_i64[1] = 0x800000000LL;
LABEL_440:
  sub_2AE9350((__int64)&v336, a4, (__int64)v40, v57, v53, (__int64)v59);
  v58 = (__m128i *)v336.m128i_i64[0];
  v59 = (__m128i *)(v336.m128i_i64[0] + 96LL * v336.m128i_u32[2]);
LABEL_77:
  v60 = v59 + 3;
  for ( i = &v58[6 * a4]; i != v59; v60 += 6 )
  {
    if ( v59 )
    {
      v62 = v59 + 1;
      memset(v59, 0, 0x60u);
      v59->m128i_i32[2] = 1;
      do
      {
        if ( v62 )
          v62->m128i_i32[0] = -1;
        v62 = (__m128i *)((char *)v62 + 8);
      }
      while ( v62 != v60 );
      v59[3].m128i_i32[2] = 0;
      v59[3].m128i_i64[0] = (__int64)v59[4].m128i_i64;
      v59[3].m128i_i32[3] = 4;
    }
    v59 += 6;
  }
  v336.m128i_i32[2] = v263;
  v261 = *(_QWORD *)(a2 + 448);
  sub_2AD3790(a2);
  v271 = 0;
  v257 = v329;
  if ( (_DWORD)v329 )
  {
LABEL_86:
    v65 = 0;
    while ( 1 )
    {
      v265 = *(_QWORD *)&v328[8 * v65];
      if ( (v331.m128i_i8[8] & 1) != 0 )
      {
        v66 = &v332;
        v67 = 15;
      }
      else
      {
        v180 = v332.m128i_u32[2];
        v66 = (__m128i *)v332.m128i_i64[0];
        if ( !v332.m128i_i32[2] )
        {
          v187 = v331.m128i_u32[2];
          ++v331.m128i_i64[0];
          v298.m128i_i64[0] = 0;
          v188 = ((unsigned __int32)v331.m128i_i32[2] >> 1) + 1;
          goto LABEL_301;
        }
        v67 = (unsigned int)(v332.m128i_i32[2] - 1);
      }
      v68 = v67 & (37 * v65);
      v69 = (__int64)&v66->m128i_i64[5 * v68];
      v63 = *(unsigned int *)v69;
      if ( (_DWORD)v63 != v65 )
        break;
LABEL_90:
      v70 = *(__int64 **)(v69 + 8);
      for ( j = &v70[*(unsigned int *)(v69 + 16)]; j != v70; ++v70 )
      {
        v72 = *v70;
        if ( v296 )
        {
          v67 = (__int64)&v293[HIDWORD(v294)];
          v69 = HIDWORD(v294);
          v73 = v293;
          if ( v293 != (__int64 *)v67 )
          {
            while ( v72 != *v73 )
            {
              if ( (__int64 *)v67 == ++v73 )
                goto LABEL_97;
            }
            --HIDWORD(v294);
            v69 = HIDWORD(v294);
            v67 = v293[HIDWORD(v294)];
            *v73 = v67;
            ++v292;
          }
        }
        else
        {
          v179 = sub_C8CA60((__int64)&v292, v72);
          if ( v179 )
          {
            *v179 = -2;
            ++v295;
            ++v292;
          }
        }
LABEL_97:
        ;
      }
LABEL_98:
      if ( !(unsigned __int8)sub_B19060((__int64)&v286, v265, v67, v69)
        || (unsigned __int8)sub_B19060(a2 + 512, v265, v74, v75) )
      {
        goto LABEL_100;
      }
      if ( v263 )
      {
        v193 = a3;
        v194 = 0;
        do
        {
          v195 = (unsigned __int64 *)src;
          v298.m128i_i64[0] = 0;
          v298.m128i_i64[1] = 1;
          do
            *(_DWORD *)v195++ = -1;
          while ( v195 != (unsigned __int64 *)&v301 );
          v301 = v303;
          v302 = 0x400000000LL;
          v196 = *((_BYTE *)v193 + 4);
          if ( v196 )
          {
            v278 = *(__int64 **)v193;
            v197 = *(_DWORD *)(a2 + 184);
            v198 = *v193;
            if ( v197 )
            {
              v233 = *(_QWORD *)(a2 + 168);
              v234 = 37 * v198 - 1;
LABEL_367:
              v235 = v197 - 1;
              v236 = 1;
              v237 = v235 & v234;
              while ( 2 )
              {
                v238 = (int *)(v233 + 72LL * v237);
                v239 = *v238;
                if ( *v238 == v198 )
                {
                  if ( v196 == *((_BYTE *)v238 + 4) )
                    goto LABEL_314;
                  if ( v239 != -1 )
                    goto LABEL_370;
                }
                else if ( v239 != -1 )
                {
LABEL_370:
                  v240 = v236 + v237;
                  ++v236;
                  v237 = v235 & v240;
                  continue;
                }
                break;
              }
              if ( *((_BYTE *)v238 + 4) )
                goto LABEL_313;
              goto LABEL_370;
            }
          }
          else
          {
            v198 = *v193;
            if ( *v193 == 1 )
            {
              if ( v296 )
                v241 = &v293[HIDWORD(v294)];
              else
                v241 = &v293[(unsigned int)v294];
              v274 = (unsigned __int64)v293;
              v275 = v241;
              sub_254BBF0((__int64)&v274);
              v276 = (__int64 *)&v292;
              v277 = (__int64)v292;
              if ( v296 )
                v242 = (__int64)&v293[HIDWORD(v294)];
              else
                v242 = (__int64)&v293[(unsigned int)v294];
              v278 = (__int64 *)v242;
              v279 = v242;
              sub_254BBF0((__int64)&v278);
              v280 = (__int64 *)&v292;
              v281 = (__int64)v292;
              while ( v278 != (__int64 *)v274 )
              {
                while ( 1 )
                {
                  LODWORD(v273) = sub_DFB180(*(__int64 **)(a2 + 448), 0);
                  v247 = (_DWORD *)sub_2AE5BD0((__int64)&v298, (int *)&v273, v243, v244, v245, v246);
                  ++*v247;
                  k = v275;
                  v248 = (__int64 *)(v274 + 8);
                  v274 = (unsigned __int64)v248;
                  if ( v248 != v275 )
                    break;
LABEL_386:
                  if ( v278 == v248 )
                    goto LABEL_328;
                }
                while ( 1 )
                {
                  v201 = *v248 + 2;
                  if ( v201 > 1 )
                    break;
                  v274 = (unsigned __int64)++v248;
                  if ( v275 == v248 )
                    goto LABEL_386;
                }
              }
              goto LABEL_328;
            }
            v278 = *(__int64 **)v193;
            v197 = *(_DWORD *)(a2 + 184);
            if ( v197 )
            {
              v233 = *(_QWORD *)(a2 + 168);
              v234 = 37 * v198;
              goto LABEL_367;
            }
          }
LABEL_313:
          sub_2ACAC50(a2, (__int64)v278);
          sub_2AE4570(a2, (__int64)v278);
          sub_2AC7F80(a2, (__int64)v278);
          sub_2ADE2D0(a2, (__int64)v278);
LABEL_314:
          if ( v296 )
            v199 = &v293[HIDWORD(v294)];
          else
            v199 = &v293[(unsigned int)v294];
          v274 = (unsigned __int64)v293;
          v275 = v199;
          sub_254BBF0((__int64)&v274);
          v276 = (__int64 *)&v292;
          v277 = (__int64)v292;
          if ( v296 )
            v200 = (__int64)&v293[HIDWORD(v294)];
          else
            v200 = (__int64)&v293[(unsigned int)v294];
          v278 = (__int64 *)v200;
          v279 = v200;
          sub_254BBF0((__int64)&v278);
          k = (__int64 *)v274;
          v280 = (__int64 *)&v292;
          v281 = (__int64)v292;
          if ( v278 != (__int64 *)v274 )
          {
            while ( 1 )
            {
              v205 = *k;
              if ( *(_BYTE *)(a2 + 700) )
                break;
              if ( !sub_C8CA60(a2 + 672, *k) )
                goto LABEL_341;
LABEL_324:
              k = v275;
              v207 = (__int64 *)(v274 + 8);
              v274 = (unsigned __int64)v207;
              if ( v207 == v275 )
              {
LABEL_327:
                if ( v278 == v275 )
                  goto LABEL_328;
              }
              else
              {
                while ( 1 )
                {
                  v201 = *v207 + 2;
                  if ( v201 > 1 )
                    break;
                  v274 = (unsigned __int64)++v207;
                  if ( v207 == v275 )
                    goto LABEL_327;
                }
                k = (__int64 *)v274;
                if ( v278 == (__int64 *)v274 )
                  goto LABEL_328;
              }
            }
            v206 = *(_QWORD **)(a2 + 680);
            v201 = (unsigned __int64)&v206[*(unsigned int *)(a2 + 692)];
            if ( v206 != (_QWORD *)v201 )
            {
              while ( v205 != *v206 )
              {
                if ( (_QWORD *)v201 == ++v206 )
                  goto LABEL_341;
              }
              goto LABEL_324;
            }
LABEL_341:
            v212 = *((_BYTE *)v193 + 4);
            v213 = *v193;
            if ( v212 )
            {
              v214 = *(unsigned int *)(a2 + 216);
              v215 = *(_QWORD *)(a2 + 200);
              if ( (_DWORD)v214 )
              {
                v216 = 37 * v213 - 1;
                goto LABEL_345;
              }
LABEL_351:
              if ( !(unsigned __int8)sub_B19060(v215 + 8, v205, v213, (__int64)k) )
              {
                v220 = sub_DFB180(*(__int64 **)(a2 + 448), 1u);
                v225 = *(_QWORD *)(v205 + 8);
                v272 = v220;
                v273 = *(__int64 **)v193;
                if ( *(_BYTE *)(v225 + 8) == 11
                  || (v259 = *((_BYTE *)v193 + 4), !(unsigned __int8)sub_BCBCB0(v225))
                  || v259 && !(unsigned __int8)sub_DFE280(v261) )
                {
                  v226 = 0;
                }
                else
                {
                  sub_BCE1B0((__int64 *)v225, (__int64)v273);
                  v226 = sub_DFA920(v261);
                }
                v227 = (_DWORD *)sub_2AE5BD0((__int64)&v298, &v272, v221, v222, v223, v224);
                *v227 += v226;
                goto LABEL_324;
              }
            }
            else if ( (_DWORD)v213 != 1 )
            {
              v214 = *(unsigned int *)(a2 + 216);
              v215 = *(_QWORD *)(a2 + 200);
              if ( (_DWORD)v214 )
              {
                v216 = 37 * v213;
LABEL_345:
                v217 = 1;
                for ( k = (__int64 *)(((_DWORD)v214 - 1) & v216); ; k = (__int64 *)(((_DWORD)v214 - 1)
                                                                                  & (unsigned int)v219) )
                {
                  v218 = v215 + 72LL * (unsigned int)k;
                  if ( *(_DWORD *)v218 == (_DWORD)v213 && v212 == *(_BYTE *)(v218 + 4) )
                  {
                    v215 += 72LL * (unsigned int)k;
                    goto LABEL_351;
                  }
                  if ( *(_DWORD *)v218 == -1 && *(_BYTE *)(v218 + 4) )
                    break;
                  v219 = v217 + (_DWORD)k;
                  ++v217;
                }
                v213 = 9 * v214;
                v215 += 72 * v214;
              }
              goto LABEL_351;
            }
            LODWORD(v273) = sub_DFB180(*(__int64 **)(a2 + 448), 0);
            v232 = (_DWORD *)sub_2AE5BD0((__int64)&v298, (int *)&v273, v228, v229, v230, v231);
            ++*v232;
            goto LABEL_324;
          }
LABEL_328:
          v208 = v301;
          v209 = &v301[8 * (unsigned int)v302];
          if ( v209 != v301 )
          {
            v210 = (unsigned __int64)v301;
            do
            {
              v211 = (_DWORD *)sub_2AE5BD0(v194 + v336.m128i_i64[0], (int *)v210, v201, (__int64)k, v202, v203);
              v201 = (unsigned int)*v211;
              if ( *(_DWORD *)(v210 + 4) >= (unsigned int)v201 )
                v201 = *(unsigned int *)(v210 + 4);
              v210 += 8LL;
              *v211 = v201;
            }
            while ( v209 != (_BYTE *)v210 );
            v208 = v301;
          }
          if ( v208 != v303 )
            _libc_free((unsigned __int64)v208);
          if ( (v298.m128i_i8[8] & 1) == 0 )
            sub_C7D6A0((__int64)src[0], 8LL * LODWORD(src[1]), 4);
          v193 += 2;
          v194 += 96;
        }
        while ( v193 != &a3[2 * (v263 - 1) + 2] );
      }
      sub_BED950((__int64)&v298, (__int64)&v292, v265);
LABEL_100:
      v65 = v271 + 1;
      v271 = v65;
      if ( v65 >= v257 )
        goto LABEL_101;
    }
    v250 = 1;
    v64 = 0;
    while ( (_DWORD)v63 != -1 )
    {
      if ( (_DWORD)v63 == -2 && !v64 )
        v64 = v69;
      v68 = v67 & (v250 + v68);
      v69 = (__int64)&v66->m128i_i64[5 * v68];
      v63 = *(unsigned int *)v69;
      if ( (_DWORD)v63 == v65 )
        goto LABEL_90;
      ++v250;
    }
    v187 = v331.m128i_u32[2];
    v189 = 48;
    v180 = 16;
    if ( v64 )
      v69 = v64;
    ++v331.m128i_i64[0];
    v298.m128i_i64[0] = v69;
    v188 = ((unsigned __int32)v331.m128i_i32[2] >> 1) + 1;
    if ( (v331.m128i_i8[8] & 1) == 0 )
    {
      v180 = v332.m128i_u32[2];
LABEL_301:
      v189 = 3 * v180;
    }
    if ( v189 <= 4 * (int)v188 )
    {
      v251 = 2 * v180;
    }
    else
    {
      v190 = v180 - v331.m128i_i32[3] - v188;
      v188 = (unsigned int)v180 >> 3;
      if ( v190 > (unsigned int)v188 )
      {
LABEL_304:
        v331.m128i_i32[2] = (2 * (v187 >> 1) + 2) | v187 & 1;
        v191 = v298.m128i_i64[0];
        if ( *(_DWORD *)v298.m128i_i64[0] != -1 )
          --v331.m128i_i32[3];
        v69 = 0x200000000LL;
        v192 = v271;
        *(_QWORD *)(v298.m128i_i64[0] + 16) = 0x200000000LL;
        *(_DWORD *)v191 = v192;
        v67 = v191 + 24;
        *(_QWORD *)(v191 + 8) = v191 + 24;
        goto LABEL_98;
      }
      v251 = v180;
    }
    sub_2ACF290((__int64)&v331, v251, v180, v188, v63, v64);
    sub_2AC1640((__int64)&v331, (int *)&v271, &v298);
    v187 = v331.m128i_u32[2];
    goto LABEL_304;
  }
LABEL_101:
  if ( !v263 )
    goto LABEL_240;
  v76 = a3;
  v258 = 0;
  do
  {
    v77 = (unsigned __int64 *)src;
    v298.m128i_i64[0] = 0;
    v298.m128i_i64[1] = 1;
    do
      *(_DWORD *)v77++ = -1;
    while ( v77 != (unsigned __int64 *)&v301 );
    v301 = v303;
    v302 = 0x400000000LL;
    v270 = (unsigned __int64)v308;
    v267 = &v308[8 * (unsigned int)v309];
    if ( v267 == v308 )
      goto LABEL_188;
    do
    {
      v78 = *(_QWORD *)v270;
      v79 = *(_QWORD *)(*(_QWORD *)v270 + 16LL);
      if ( !v79 )
      {
LABEL_276:
        v113 = *(__int64 **)(a2 + 448);
        v114 = 0;
        v90 = 0;
        v91 = 1;
        goto LABEL_182;
      }
      while ( 1 )
      {
        v80 = *(_QWORD *)(a2 + 432);
        v81 = *(_QWORD *)(v79 + 24);
        v82 = *(_DWORD *)(v80 + 24);
        v83 = *(_QWORD *)(v81 + 40);
        v84 = *(_QWORD *)(v80 + 8);
        if ( v82 )
        {
          v85 = v82 - 1;
          v86 = v85 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
          v87 = (__int64 *)(v84 + 16LL * v86);
          v88 = *v87;
          if ( v83 == *v87 )
          {
LABEL_111:
            v89 = v87[1];
            goto LABEL_112;
          }
          v110 = 1;
          while ( v88 != -4096 )
          {
            v177 = v110 + 1;
            v86 = v85 & (v110 + v86);
            v87 = (__int64 *)(v84 + 16LL * v86);
            v88 = *v87;
            if ( v83 == *v87 )
              goto LABEL_111;
            v110 = v177;
          }
        }
        v89 = 0;
LABEL_112:
        if ( *(_QWORD *)(a2 + 416) != v89 )
          goto LABEL_108;
        v90 = *((_BYTE *)v76 + 4);
        v91 = *v76;
        if ( v90 )
        {
          v92 = *(unsigned int *)(a2 + 216);
          v93 = *(_QWORD *)(a2 + 200);
          if ( !(_DWORD)v92 )
            goto LABEL_176;
          v94 = 37 * v91 - 1;
          goto LABEL_117;
        }
        if ( v91 == 1 )
          goto LABEL_108;
        v92 = *(unsigned int *)(a2 + 216);
        v93 = *(_QWORD *)(a2 + 200);
        if ( (_DWORD)v92 )
        {
          v94 = 37 * v91;
LABEL_117:
          v95 = 1;
          for ( m = (v92 - 1) & v94; ; m = (v92 - 1) & v98 )
          {
            v97 = v93 + 72LL * m;
            if ( *(_DWORD *)v97 == v91 && v90 == *(_BYTE *)(v97 + 4) )
            {
              v93 += 72LL * m;
              goto LABEL_176;
            }
            if ( *(_DWORD *)v97 == -1 && *(_BYTE *)(v97 + 4) )
              break;
            v98 = v95 + m;
            ++v95;
          }
          v93 += 72 * v92;
        }
LABEL_176:
        if ( !*(_BYTE *)(v93 + 36) )
          break;
        v111 = *(_QWORD **)(v93 + 16);
        v112 = &v111[*(unsigned int *)(v93 + 28)];
        if ( v111 == v112 )
          goto LABEL_180;
        while ( v81 != *v111 )
        {
          if ( v112 == ++v111 )
            goto LABEL_180;
        }
LABEL_108:
        v79 = *(_QWORD *)(v79 + 8);
        if ( !v79 )
          goto LABEL_276;
      }
      v269 = v78;
      v178 = sub_C8CA60(v93 + 8, *(_QWORD *)(v79 + 24));
      v78 = v269;
      if ( v178 )
        goto LABEL_108;
      v90 = *((_BYTE *)v76 + 4);
      v91 = *v76;
LABEL_180:
      v113 = *(__int64 **)(a2 + 448);
      if ( v90 )
        v114 = v91 != 0;
      else
        v114 = v91 > 1;
LABEL_182:
      v268 = v78;
      v115 = sub_DFB180(v113, v114);
      v119 = v268;
      LODWORD(v274) = v115;
      v120 = *(_QWORD *)(v268 + 8);
      LODWORD(v278) = v91;
      BYTE4(v278) = v90;
      if ( *(_BYTE *)(v120 + 8) == 11 || !(unsigned __int8)sub_BCBCB0(v120) || v90 && !(unsigned __int8)sub_DFE280(v261) )
      {
        v121 = 0;
      }
      else
      {
        sub_BCE1B0((__int64 *)v120, (__int64)v278);
        v121 = sub_DFA920(v261);
      }
      v122 = (_DWORD *)sub_2AE5BD0((__int64)&v298, (int *)&v274, v116, v119, v117, v118);
      v270 += 8LL;
      *v122 += v121;
    }
    while ( v267 != (_BYTE *)v270 );
LABEL_188:
    if ( (v312 & 1) == 0 )
      sub_C7D6A0((__int64)v313, 8LL * v314, 4);
    v123 = v312 | 1;
    LOBYTE(v312) = v312 | 1;
    if ( (v298.m128i_i8[8] & 1) != 0 )
    {
      v312 = v298.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL | v312 & 1;
      if ( (v312 & 1) == 0 )
      {
        v126 = v313;
        v128 = src;
        goto LABEL_195;
      }
      goto LABEL_400;
    }
    v124 = (unsigned int)src[1];
    if ( LODWORD(src[1]) > 4 )
    {
      LOBYTE(v312) = v123 & 0xFE;
      v125 = (void *)sub_C7D670(8LL * LODWORD(src[1]), 4);
      v314 = v124;
      v313 = v125;
      v126 = v125;
      v127 = v298.m128i_i8[8] & 1;
      v312 = v298.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL | v312 & 1;
      if ( (v312 & 1) == 0 )
      {
        if ( !v127 )
          goto LABEL_194;
        v128 = src;
        goto LABEL_195;
      }
      if ( v127 )
      {
LABEL_400:
        v128 = src;
        v129 = 32;
        v126 = &v313;
        goto LABEL_196;
      }
LABEL_395:
      v128 = (void **)src[0];
      v129 = 32;
      v126 = &v313;
      goto LABEL_196;
    }
    v312 = v298.m128i_i64[1] & 0xFFFFFFFFFFFFFFFELL | v312 & 1;
    if ( (v312 & 1) != 0 )
      goto LABEL_395;
    v126 = v313;
LABEL_194:
    v128 = (void **)src[0];
LABEL_195:
    v129 = 8LL * v314;
LABEL_196:
    memcpy(v126, v128, v129);
    sub_2AA8A70((__int64)v315, (__int64)&v301, v130, v131, v132, v133);
    v138 = v336.m128i_i64[0] + v258;
    if ( (__int64 *)(v336.m128i_i64[0] + v258) != &v317 )
    {
      if ( (v318 & 1) == 0 )
        sub_C7D6A0((__int64)dest, 8LL * v321, 4);
      v139 = v318 | 1;
      LOBYTE(v318) = v318 | 1;
      if ( (*(_BYTE *)(v138 + 8) & 1) == 0 && *(_DWORD *)(v138 + 24) > 4u )
      {
        LOBYTE(v318) = v139 & 0xFE;
        if ( (*(_BYTE *)(v138 + 8) & 1) != 0 )
        {
          v141 = 32;
          LODWORD(v140) = 4;
        }
        else
        {
          v140 = *(unsigned int *)(v138 + 24);
          v141 = 8 * v140;
        }
        v142 = (void *)sub_C7D670(v141, 4);
        v321 = v140;
        dest = v142;
      }
      p_dest = &dest;
      v144 = (const void *)(v138 + 16);
      v318 = *(_DWORD *)(v138 + 8) & 0xFFFFFFFE | v318 & 1;
      v319 = *(_DWORD *)(v138 + 12);
      if ( (v318 & 1) == 0 )
        p_dest = dest;
      if ( (*(_BYTE *)(v138 + 8) & 1) == 0 )
        v144 = *(const void **)(v138 + 16);
      v145 = 32;
      if ( (v318 & 1) == 0 )
        v145 = 8LL * v321;
      memcpy(p_dest, v144, v145);
    }
    sub_2AA8A70((__int64)v322, v138 + 48, v134, v135, v136, v137);
    v149 = v258;
    v150 = *(_QWORD *)a1 + 2 * v258;
    if ( (__int64 *)v150 != &v311 )
    {
      if ( (*(_BYTE *)(v150 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v150 + 16), 8LL * *(unsigned int *)(v150 + 24), 4);
      v151 = *(_BYTE *)(v150 + 8) | 1;
      *(_BYTE *)(v150 + 8) = v151;
      if ( (v312 & 1) == 0 && v314 > 4 )
      {
        *(_BYTE *)(v150 + 8) = v151 & 0xFE;
        if ( (v312 & 1) != 0 )
        {
          v153 = 32;
          v152 = 4;
        }
        else
        {
          v152 = v314;
          v153 = 8LL * v314;
        }
        v154 = sub_C7D670(v153, 4);
        *(_DWORD *)(v150 + 24) = v152;
        *(_QWORD *)(v150 + 16) = v154;
      }
      *(_DWORD *)(v150 + 8) = v312 & 0xFFFFFFFE | *(_DWORD *)(v150 + 8) & 1;
      *(_DWORD *)(v150 + 12) = HIDWORD(v312);
      if ( (*(_BYTE *)(v150 + 8) & 1) != 0 )
      {
        v155 = (void *)(v150 + 16);
        if ( (v312 & 1) != 0 )
        {
          v157 = 32;
          v156 = &v313;
        }
        else
        {
          v156 = v313;
          v157 = 32;
        }
      }
      else
      {
        v156 = &v313;
        v155 = *(void **)(v150 + 16);
        if ( (v312 & 1) == 0 )
          v156 = v313;
        v157 = 8LL * *(unsigned int *)(v150 + 24);
      }
      memcpy(v155, v156, v157);
    }
    sub_2AA8A70(v150 + 48, (__int64)v315, v146, v149, v147, v148);
    if ( &v317 != (__int64 *)(v150 + 96) )
    {
      if ( (*(_BYTE *)(v150 + 104) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v150 + 112), 8LL * *(unsigned int *)(v150 + 120), 4);
      v162 = *(_BYTE *)(v150 + 104) | 1;
      *(_BYTE *)(v150 + 104) = v162;
      if ( (v318 & 1) == 0 && v321 > 4 )
      {
        *(_BYTE *)(v150 + 104) = v162 & 0xFE;
        if ( (v318 & 1) != 0 )
        {
          v164 = 32;
          v163 = 4;
        }
        else
        {
          v163 = v321;
          v164 = 8LL * v321;
        }
        v165 = sub_C7D670(v164, 4);
        *(_DWORD *)(v150 + 120) = v163;
        *(_QWORD *)(v150 + 112) = v165;
      }
      *(_DWORD *)(v150 + 104) = v318 & 0xFFFFFFFE | *(_DWORD *)(v150 + 104) & 1;
      *(_DWORD *)(v150 + 108) = v319;
      if ( (*(_BYTE *)(v150 + 104) & 1) != 0 )
      {
        v166 = (void *)(v150 + 112);
        if ( (v318 & 1) != 0 )
        {
          v168 = 32;
          v167 = &dest;
        }
        else
        {
          v167 = dest;
          v168 = 32;
        }
      }
      else
      {
        v167 = &dest;
        v166 = *(void **)(v150 + 112);
        if ( (v318 & 1) == 0 )
          v167 = dest;
        v168 = 8LL * *(unsigned int *)(v150 + 120);
      }
      memcpy(v166, v167, v168);
    }
    sub_2AA8A70(v150 + 144, (__int64)v322, v158, v159, v160, v161);
    if ( v301 != v303 )
      _libc_free((unsigned __int64)v301);
    if ( (v298.m128i_i8[8] & 1) == 0 )
      sub_C7D6A0((__int64)src[0], 8LL * LODWORD(src[1]), 4);
    v258 += 96;
    v76 += 2;
  }
  while ( v76 != &a3[2 * (v263 - 1) + 2] );
LABEL_240:
  v169 = v336.m128i_i64[0];
  v170 = (__m128i *)(v336.m128i_i64[0] + 96LL * v336.m128i_u32[2]);
  if ( (__m128i *)v336.m128i_i64[0] != v170 )
  {
    do
    {
      v170 -= 6;
      v171 = v170[3].m128i_u64[0];
      if ( (__m128i *)v171 != &v170[4] )
        _libc_free(v171);
      if ( (v170->m128i_i8[8] & 1) == 0 )
        sub_C7D6A0(v170[1].m128i_i64[0], 8LL * v170[1].m128i_u32[2], 4);
    }
    while ( (__m128i *)v169 != v170 );
    v170 = (__m128i *)v336.m128i_i64[0];
  }
  if ( v170 != &v337 )
    _libc_free((unsigned __int64)v170);
  if ( !v296 )
    _libc_free((unsigned __int64)v293);
  if ( (v331.m128i_i8[8] & 1) != 0 )
  {
    v174 = &v336;
    v172 = &v332;
    goto LABEL_256;
  }
  v172 = (__m128i *)v332.m128i_i64[0];
  v173 = 40LL * v332.m128i_u32[2];
  if ( v332.m128i_i32[2]
    && (v174 = (__m128i *)(v332.m128i_i64[0] + v173), v332.m128i_i64[0] + v173 != v332.m128i_i64[0]) )
  {
    do
    {
LABEL_256:
      while ( 1 )
      {
        if ( v172->m128i_i32[0] <= 0xFFFFFFFD )
        {
          v175 = v172->m128i_u64[1];
          if ( (unsigned __int64 *)v175 != &v172[1].m128i_u64[1] )
            break;
        }
        v172 = (__m128i *)((char *)v172 + 40);
        if ( v172 == v174 )
          goto LABEL_259;
      }
      _libc_free(v175);
      v172 = (__m128i *)((char *)v172 + 40);
    }
    while ( v172 != v174 );
LABEL_259:
    if ( (v331.m128i_i8[8] & 1) == 0 )
    {
      v172 = (__m128i *)v332.m128i_i64[0];
      v173 = 40LL * v332.m128i_u32[2];
      goto LABEL_407;
    }
  }
  else
  {
LABEL_407:
    sub_C7D6A0((__int64)v172, v173, 8);
  }
  if ( v308 != v310 )
    _libc_free((unsigned __int64)v308);
  sub_C7D6A0(v305, 8LL * (unsigned int)v307, 8);
  if ( !v290 )
    _libc_free((unsigned __int64)v287);
  if ( (v325 & 1) == 0 )
    sub_C7D6A0((__int64)v326, 16LL * v327, 8);
  if ( v328 != v330 )
    _libc_free((unsigned __int64)v328);
  sub_2AB4A90((__int64)&v317);
  sub_2AB4A90((__int64)&v311);
  if ( v284 )
    j_j___libc_free_0(v284);
  sub_C7D6A0(v282[2], 16LL * v283, 8);
  return a1;
}
