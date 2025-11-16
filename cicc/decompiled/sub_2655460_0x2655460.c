// Function: sub_2655460
// Address: 0x2655460
//
__int64 __fastcall sub_2655460(__int64 a1, __m128i a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // rsi
  __int32 v9; // ecx
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdi
  int v15; // r10d
  __int32 v16; // ecx
  unsigned int j; // eax
  __int64 v18; // r8
  unsigned int v19; // eax
  unsigned __int64 v20; // r12
  __int64 *v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 *v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // r12
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rsi
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rax
  const __m128i *v38; // rdx
  __int64 v39; // r9
  char *v40; // r8
  __int64 v41; // r12
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  _QWORD *v44; // rax
  __int64 v45; // rbx
  int v46; // esi
  __int64 *v47; // r12
  __int64 v48; // rcx
  __m128i *v49; // r11
  int v50; // r13d
  unsigned int v51; // edx
  __m128i *v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rsi
  unsigned __int64 v57; // rcx
  __int64 v58; // r8
  unsigned __int64 v59; // r14
  __int64 v60; // r15
  __int64 v61; // r13
  __int64 v62; // r14
  unsigned __int64 v63; // rdi
  int v64; // r9d
  __int64 v65; // r10
  signed __int32 v66; // eax
  volatile signed __int32 *v67; // r12
  _QWORD *v68; // rbx
  __int64 v69; // rax
  int v70; // r10d
  __int64 *v71; // r8
  int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rdx
  int v75; // r13d
  __int64 v76; // rcx
  __int64 *v77; // rdi
  __int64 *v78; // rax
  __int64 v79; // r8
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 *v82; // r13
  volatile signed __int32 *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rbx
  unsigned int v86; // esi
  __int64 v87; // rax
  __int64 v88; // rdi
  int v89; // r10d
  unsigned int v90; // ecx
  unsigned int v91; // r12d
  __int64 v92; // rdx
  __int64 v93; // r9
  __int64 v94; // r11
  __int64 v95; // r14
  int v96; // r11d
  __int64 v97; // r10
  unsigned int v98; // edx
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rdx
  int *v103; // rax
  int *v104; // r14
  __int64 v105; // rax
  int v106; // eax
  const __m128i *v107; // r9
  const __m128i *v108; // rbx
  const __m128i *v109; // r13
  _QWORD *v110; // rax
  __int64 v111; // rdx
  __int32 v112; // eax
  signed __int32 v113; // eax
  __int64 v114; // r13
  volatile signed __int32 *v115; // r12
  signed __int32 v116; // eax
  signed __int32 v117; // eax
  __int64 *v118; // rbx
  unsigned __int64 v119; // rdi
  __int64 *v120; // r15
  __int64 v121; // r13
  __int64 v122; // rbx
  __int64 v123; // r12
  __int64 v124; // r8
  unsigned int v125; // esi
  unsigned int v126; // r14d
  __int64 v127; // rcx
  _QWORD *v128; // rdx
  __int64 v129; // rax
  const __m128i *v130; // rdx
  __m128i *v131; // rsi
  __m128i *v132; // rax
  __m128i *v133; // rax
  __m128i *v134; // rsi
  __int64 v135; // r12
  __int64 *v136; // r14
  volatile signed __int32 *v137; // r13
  signed __int32 v138; // eax
  signed __int32 v139; // eax
  unsigned int v140; // edx
  _QWORD *v141; // rdi
  __int64 v142; // r8
  int v143; // eax
  _QWORD *v144; // rdi
  __m128i *v145; // rax
  __int64 v146; // rbx
  __m128i *v147; // rsi
  __m128i *v148; // rax
  __int64 v149; // rdi
  unsigned int v150; // r10d
  int v151; // r9d
  int v152; // r10d
  int v153; // ecx
  _QWORD *v154; // rdx
  unsigned int v155; // r14d
  __int64 v156; // rsi
  __int64 v157; // r14
  __int64 v158; // rbx
  unsigned int v159; // r9d
  int v160; // r10d
  unsigned int v161; // edi
  _QWORD *v162; // rdx
  _QWORD *v163; // rax
  __int64 v164; // rcx
  _QWORD *v165; // rax
  __int64 v166; // r12
  unsigned int v167; // edx
  int v168; // ecx
  __int64 v169; // r8
  __int64 *v170; // rbx
  unsigned __int64 v171; // rdi
  __int64 *v172; // rbx
  volatile signed __int32 *v173; // rdi
  __int64 v174; // r9
  __int64 v175; // r9
  __int64 *v176; // r9
  int v177; // edx
  _QWORD *v178; // r8
  unsigned int v179; // r13d
  __int64 v180; // rsi
  unsigned int v181; // eax
  _QWORD *v182; // rax
  __int64 v183; // r13
  _QWORD *v184; // rax
  __int64 v185; // rax
  __int64 v186; // rdx
  unsigned int v187; // ecx
  _QWORD *v188; // rdi
  __int64 v189; // r8
  unsigned int v190; // r11d
  unsigned int v191; // r11d
  __int64 v192; // rdi
  int v193; // r13d
  __int64 *v194; // r10
  unsigned int v195; // eax
  _QWORD *v196; // rax
  __int64 v197; // r13
  _QWORD *v198; // rax
  __int64 v199; // rax
  __int64 v200; // rdx
  unsigned int v201; // ecx
  _QWORD *v202; // rdi
  __int64 v203; // r8
  unsigned int v204; // r13d
  __int64 v205; // rdi
  unsigned int v206; // r13d
  int v207; // r11d
  __int64 v208; // r10
  int *v209; // rax
  int *v210; // rsi
  __int64 v211; // rbx
  __int64 v212; // r13
  int v213; // r11d
  __m128i *v214; // r10
  __int64 v215; // rcx
  __m128i *v216; // rax
  __int64 v217; // rdi
  __int64 *v218; // rax
  __int64 v219; // rdx
  int v220; // esi
  int v221; // edi
  __m128i *v222; // rdx
  _QWORD *v223; // rdi
  __int64 **v224; // r12
  __int64 **v225; // rbx
  __int64 v226; // rdi
  __int64 v227; // r12
  __int64 v228; // rbx
  __int64 v229; // rdi
  _QWORD **v230; // r13
  _QWORD **jj; // r14
  _QWORD *v232; // r15
  __int64 **v233; // r12
  __int64 **kk; // rbx
  __int64 v235; // rdi
  __int64 v236; // r12
  __int64 mm; // rbx
  __int64 v238; // rdi
  int v239; // r14d
  __int64 v240; // r10
  int v241; // r14d
  __int64 v242; // r11
  int v243; // ecx
  __int64 v244; // rcx
  int v245; // edx
  int v246; // r10d
  _QWORD *v247; // rdi
  int v248; // edx
  __int64 v249; // rax
  unsigned int v250; // eax
  _QWORD *v251; // rax
  __int64 v252; // r14
  __int64 v253; // rsi
  _QWORD *ii; // rdx
  __int64 v255; // rax
  __int64 v256; // r12
  int v257; // ebx
  _QWORD *v258; // r11
  unsigned int v259; // edx
  _QWORD *v260; // rdi
  __int64 v261; // r9
  unsigned int v262; // r11d
  __int64 v263; // rsi
  unsigned int v264; // r11d
  unsigned int v265; // ecx
  __int64 v266; // r8
  int v267; // ebx
  _QWORD *v268; // r9
  unsigned int v269; // eax
  _QWORD *v270; // rax
  __int64 v271; // rbx
  __int64 v272; // r14
  _QWORD *n; // rdx
  __int64 v274; // rax
  __int64 v275; // r12
  int v276; // r11d
  _QWORD *v277; // r10
  unsigned int v278; // edx
  _QWORD *v279; // rsi
  __int64 v280; // rdi
  unsigned int v281; // r11d
  __int64 v282; // rsi
  unsigned int v283; // r11d
  int v284; // ebx
  unsigned int v285; // ecx
  __int64 v286; // r8
  int v287; // ecx
  _QWORD *v288; // rcx
  _QWORD *v289; // rdx
  _QWORD *v290; // rcx
  _QWORD *v291; // rdx
  _QWORD *k; // rcx
  _QWORD *v293; // r10
  _QWORD *v294; // r10
  _QWORD *m; // rcx
  int v296; // ecx
  int v297; // eax
  unsigned int v298; // r10d
  _QWORD *v299; // r11
  int v300; // edi
  _QWORD *v301; // rsi
  int v302; // esi
  _QWORD *v303; // rcx
  __int64 *v304; // [rsp+8h] [rbp-308h]
  __int64 *v305; // [rsp+10h] [rbp-300h]
  volatile signed __int32 *v306; // [rsp+20h] [rbp-2F0h]
  __m128i *v307; // [rsp+28h] [rbp-2E8h]
  __int64 *v308; // [rsp+40h] [rbp-2D0h]
  _QWORD *v309; // [rsp+50h] [rbp-2C0h]
  _QWORD *v310; // [rsp+70h] [rbp-2A0h]
  __int32 v311; // [rsp+7Ch] [rbp-294h]
  __int64 *v312; // [rsp+88h] [rbp-288h]
  int v313; // [rsp+88h] [rbp-288h]
  int v314; // [rsp+88h] [rbp-288h]
  __m128i *v315; // [rsp+90h] [rbp-280h]
  __int64 v316; // [rsp+A0h] [rbp-270h]
  __int64 v317; // [rsp+A0h] [rbp-270h]
  __int64 v318; // [rsp+A0h] [rbp-270h]
  unsigned __int8 v319; // [rsp+ABh] [rbp-265h]
  unsigned int v320; // [rsp+ACh] [rbp-264h]
  __int64 v322; // [rsp+D8h] [rbp-238h]
  __int64 *v323; // [rsp+D8h] [rbp-238h]
  __int64 v324; // [rsp+E8h] [rbp-228h]
  _QWORD *v325; // [rsp+F8h] [rbp-218h] BYREF
  __int64 v326; // [rsp+100h] [rbp-210h] BYREF
  const __m128i *v327; // [rsp+108h] [rbp-208h] BYREF
  __m128i v328; // [rsp+110h] [rbp-200h] BYREF
  __m128i v329; // [rsp+120h] [rbp-1F0h] BYREF
  __int64 v330; // [rsp+130h] [rbp-1E0h] BYREF
  volatile signed __int32 *v331; // [rsp+138h] [rbp-1D8h]
  __m128i v332; // [rsp+140h] [rbp-1D0h] BYREF
  unsigned __int8 *v333; // [rsp+150h] [rbp-1C0h]
  __int64 v334; // [rsp+158h] [rbp-1B8h]
  _QWORD v335[4]; // [rsp+160h] [rbp-1B0h] BYREF
  __m128i v336; // [rsp+180h] [rbp-190h] BYREF
  __int64 *v337; // [rsp+1A0h] [rbp-170h] BYREF
  __int64 *v338; // [rsp+1A8h] [rbp-168h]
  __int64 v339; // [rsp+1B0h] [rbp-160h]
  __int64 v340; // [rsp+1C0h] [rbp-150h] BYREF
  _QWORD *v341; // [rsp+1C8h] [rbp-148h]
  __int64 v342; // [rsp+1D0h] [rbp-140h]
  unsigned int v343; // [rsp+1D8h] [rbp-138h]
  __m128i v344; // [rsp+1E0h] [rbp-130h] BYREF
  __int64 v345; // [rsp+1F0h] [rbp-120h]
  __int64 v346; // [rsp+1F8h] [rbp-118h]
  __int64 v347; // [rsp+200h] [rbp-110h] BYREF
  int v348; // [rsp+208h] [rbp-108h] BYREF
  _QWORD *v349; // [rsp+210h] [rbp-100h]
  int *v350; // [rsp+218h] [rbp-F8h]
  int *v351; // [rsp+220h] [rbp-F0h]
  unsigned __int64 v352; // [rsp+228h] [rbp-E8h]
  __int64 v353; // [rsp+230h] [rbp-E0h] BYREF
  int v354; // [rsp+238h] [rbp-D8h] BYREF
  __m128i *v355; // [rsp+240h] [rbp-D0h]
  int *v356; // [rsp+248h] [rbp-C8h]
  int *v357; // [rsp+250h] [rbp-C0h]
  __int64 v358; // [rsp+258h] [rbp-B8h]
  __m128i *v359; // [rsp+260h] [rbp-B0h] BYREF
  unsigned __int64 v360; // [rsp+268h] [rbp-A8h] BYREF
  __m128i *v361; // [rsp+270h] [rbp-A0h]
  unsigned __int64 *v362; // [rsp+278h] [rbp-98h]
  unsigned __int64 *v363; // [rsp+280h] [rbp-90h]
  __int64 v364; // [rsp+288h] [rbp-88h]
  unsigned __int64 v365; // [rsp+290h] [rbp-80h] BYREF
  __int64 v366; // [rsp+298h] [rbp-78h]
  const __m128i *v367; // [rsp+2A0h] [rbp-70h]
  const __m128i *v368; // [rsp+2A8h] [rbp-68h]
  char *v369; // [rsp+2B0h] [rbp-60h]
  unsigned __int64 *v370; // [rsp+2B8h] [rbp-58h]
  __m128i *v371; // [rsp+2C0h] [rbp-50h]
  __int64 v372; // [rsp+2C8h] [rbp-48h]
  __int64 v373; // [rsp+2D0h] [rbp-40h]
  unsigned __int64 *v374; // [rsp+2D8h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 40);
  v3 = *(__int64 **)(a1 + 32);
  v340 = 0;
  v341 = 0;
  v2 *= 32;
  v342 = 0;
  v343 = 0;
  v304 = (__int64 *)((char *)v3 + v2);
  if ( (__int64 *)((char *)v3 + v2) != v3 )
  {
    v305 = v3;
    v319 = 0;
    while ( 1 )
    {
      v4 = *v305;
      v328.m128i_i32[2] = 0;
      v328.m128i_i64[0] = v4;
      v348 = 0;
      v349 = 0;
      v350 = &v348;
      v351 = &v348;
      v352 = 0;
      v307 = (__m128i *)v305[2];
      if ( (__m128i *)v305[1] != v307 )
        break;
      v30 = 0;
LABEL_32:
      sub_2642590(v30);
      v305 += 4;
      if ( v304 == v305 )
        goto LABEL_33;
    }
    v315 = (__m128i *)v305[1];
    while ( 1 )
    {
      v5 = *(unsigned int *)(a1 + 248);
      v6 = *(_QWORD *)(a1 + 232);
      if ( (_DWORD)v5 )
      {
        v7 = 1;
        v8 = v315->m128i_i64[0];
        v9 = v315->m128i_i32[2];
        for ( i = (v5 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v9)
                    | ((unsigned __int64)(((unsigned int)v315->m128i_i64[0] >> 9)
                                        ^ ((unsigned int)v315->m128i_i64[0] >> 4)) << 32))) >> 31)
                 ^ (756364221 * v9)); ; i = (v5 - 1) & v12 )
        {
          v11 = v6 + 24LL * i;
          if ( *(_QWORD *)v11 == v8 && *(_DWORD *)(v11 + 8) == v9 )
            break;
          if ( *(_QWORD *)v11 == -4096 && *(_DWORD *)(v11 + 8) == -1 )
            goto LABEL_11;
          v12 = v7 + i;
          ++v7;
        }
        if ( v11 != v6 + 24 * v5 )
        {
          v35 = *(_QWORD **)(*(_QWORD *)(a1 + 256) + 24LL * *(unsigned int *)(v11 + 16) + 16);
          if ( v35 )
            break;
        }
      }
LABEL_11:
      v13 = *(unsigned int *)(a1 + 296);
      v14 = *(_QWORD *)(a1 + 280);
      if ( !(_DWORD)v13 )
        goto LABEL_30;
      v15 = 1;
      v8 = v315->m128i_i64[0];
      v16 = v315->m128i_i32[2];
      for ( j = (v13 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * v16)
                  | ((unsigned __int64)(((unsigned int)v315->m128i_i64[0] >> 9) ^ ((unsigned int)v315->m128i_i64[0] >> 4)) << 32))) >> 31)
               ^ (756364221 * v16)); ; j = (v13 - 1) & v19 )
      {
        v18 = v14 + 24LL * j;
        if ( *(_QWORD *)v18 == v8 && *(_DWORD *)(v18 + 8) == v16 )
          break;
        if ( *(_QWORD *)v18 == -4096 && *(_DWORD *)(v18 + 8) == -1 )
          goto LABEL_30;
        v19 = v15 + j;
        ++v15;
      }
      if ( v18 != v14 + 24 * v13 )
      {
        v35 = *(_QWORD **)(*(_QWORD *)(a1 + 304) + 24LL * *(unsigned int *)(v18 + 16) + 16);
        v325 = v35;
        if ( v35 )
          goto LABEL_40;
      }
LABEL_30:
      if ( v307 == ++v315 )
      {
        v30 = v349;
        goto LABEL_32;
      }
    }
    v325 = *(_QWORD **)(*(_QWORD *)(a1 + 256) + 24LL * *(unsigned int *)(v11 + 16) + 16);
LABEL_40:
    if ( v35[12] == v35[13] )
      goto LABEL_30;
    v354 = 0;
    v356 = &v354;
    v357 = &v354;
    v355 = 0;
    v335[0] = &v353;
    v358 = 0;
    v335[1] = &v347;
    v335[2] = &v325;
    v365 = 0;
    v366 = 0;
    v367 = 0;
    v368 = 0;
    v369 = 0;
    v370 = 0;
    v371 = 0;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    sub_263FB70((__int64 *)&v365);
    if ( !(unsigned __int8)sub_2647CD0((__int64)v325, v8, v36) )
    {
      v222 = v371;
      if ( v371 != (__m128i *)(v373 - 8) )
      {
        v37 = v325;
        if ( v371 )
        {
          v371->m128i_i64[0] = (__int64)v325;
          v222 = v371;
        }
        v38 = (const __m128i *)&v222->m128i_u64[1];
        v371 = (__m128i *)v38;
LABEL_43:
        v39 = v37[13];
        v40 = (char *)v37[12];
        v359 = (__m128i *)v38;
        v41 = (((__int64)v38->m128i_i64 - v372) >> 3) + ((v374 - v370 - 1) << 6) + ((v369 - (char *)v367) >> 3);
        v42 = *v374;
        v362 = v374;
        v360 = v42;
        v361 = (__m128i *)(v42 + 512);
        sub_26431D0(&v365, (__int64 *)&v359, v40, v39);
        v359 = (__m128i *)v367;
        v360 = (unsigned __int64)v368;
        v361 = (__m128i *)v369;
        v362 = v370;
        sub_263F500((__int64 *)&v359, v41);
        v23 = (__int64 *)v367;
        v320 = 0;
        if ( v371 == v367 )
          goto LABEL_22;
        while ( 1 )
        {
          v324 = *v23;
          v43 = (__int64)(v369 - 8);
          if ( v23 == (__int64 *)(v369 - 8) )
          {
            j_j___libc_free_0((unsigned __int64)v368);
            v43 = *++v370 + 512;
            v368 = (const __m128i *)*v370;
            v369 = (char *)v43;
            v367 = v368;
          }
          else
          {
            v367 = (const __m128i *)(v23 + 1);
          }
          ++v320;
          if ( (_BYTE)qword_4FF39C8 && *(_BYTE *)(v324 + 2) )
            sub_264C780((_QWORD *)v324);
          if ( v320 <= v352 )
          {
            v310 = (_QWORD *)(v324 + 72);
            goto LABEL_169;
          }
          if ( v320 == 1 )
          {
            v209 = (int *)v349;
            if ( !v349 )
            {
              v210 = &v348;
              goto LABEL_382;
            }
            v210 = &v348;
            do
            {
              while ( *((_QWORD *)v209 + 4) >= v328.m128i_i64[0]
                   && (*((_QWORD *)v209 + 4) != v328.m128i_i64[0] || (unsigned int)v209[10] >= v328.m128i_i32[2]) )
              {
                v210 = v209;
                v209 = (int *)*((_QWORD *)v209 + 2);
                if ( !v209 )
                  goto LABEL_378;
              }
              v209 = (int *)*((_QWORD *)v209 + 3);
            }
            while ( v209 );
LABEL_378:
            if ( v210 == &v348
              || *((_QWORD *)v210 + 4) > v328.m128i_i64[0]
              || *((_QWORD *)v210 + 4) == v328.m128i_i64[0] && (unsigned int)v210[10] > v328.m128i_i32[2] )
            {
LABEL_382:
              v359 = &v328;
              v210 = (int *)sub_2642040(&v347, (__int64)v210, (const __m128i **)&v359);
            }
            sub_2642000((_QWORD *)v210 + 6);
            sub_26422D0((__int64)v335, &v328, v315, v324);
            v211 = *(_QWORD *)(v324 + 72);
            v212 = *(_QWORD *)(v324 + 80);
            if ( v211 == v212 )
              goto LABEL_21;
            while ( 2 )
            {
              v219 = *(_QWORD *)(*(_QWORD *)v211 + 8LL);
              if ( !*(_QWORD *)(v219 + 8) )
              {
LABEL_387:
                v211 += 16;
                if ( v212 == v211 )
                  goto LABEL_21;
                continue;
              }
              break;
            }
            v220 = v343;
            v344.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)v211 + 8LL);
            if ( !v343 )
            {
              ++v340;
              v359 = 0;
              goto LABEL_391;
            }
            v213 = 1;
            v214 = 0;
            LODWORD(v215) = (v343 - 1) & (((unsigned int)v219 >> 9) ^ ((unsigned int)v219 >> 4));
            v216 = (__m128i *)&v341[3 * (unsigned int)v215];
            v217 = v216->m128i_i64[0];
            if ( v219 == v216->m128i_i64[0] )
            {
LABEL_386:
              v218 = &v216->m128i_i64[1];
              *v218 = v328.m128i_i64[0];
              *((_DWORD *)v218 + 2) = v328.m128i_i32[2];
              goto LABEL_387;
            }
            while ( v217 != -4096 )
            {
              if ( !v214 && v217 == -8192 )
                v214 = v216;
              v215 = (v343 - 1) & ((_DWORD)v215 + v213);
              v216 = (__m128i *)&v341[3 * v215];
              v217 = v216->m128i_i64[0];
              if ( v219 == v216->m128i_i64[0] )
                goto LABEL_386;
              ++v213;
            }
            if ( v214 )
              v216 = v214;
            ++v340;
            v221 = v342 + 1;
            v359 = v216;
            if ( 4 * ((int)v342 + 1) < 3 * v343 )
            {
              if ( v343 - HIDWORD(v342) - v221 <= v343 >> 3 )
              {
LABEL_392:
                sub_2644FF0((__int64)&v340, v220);
                sub_263DE10((__int64)&v340, v344.m128i_i64, &v359);
                v219 = v344.m128i_i64[0];
                v221 = v342 + 1;
                v216 = v359;
              }
              LODWORD(v342) = v221;
              if ( v216->m128i_i64[0] != -4096 )
                --HIDWORD(v342);
              v216->m128i_i64[0] = v219;
              v216->m128i_i64[1] = 0;
              v216[1].m128i_i32[0] = 0;
              goto LABEL_386;
            }
LABEL_391:
            v220 = 2 * v343;
            goto LABEL_392;
          }
          v44 = sub_2644120(*(_QWORD **)(v324 + 72), *(_QWORD *)(v324 + 80), (__int64)&v340);
          if ( v44 == *(_QWORD **)(v324 + 80) )
          {
            v311 = 0;
            v316 = 0;
            v319 = 0;
            goto LABEL_56;
          }
          v45 = *v44;
          v46 = v343;
          v47 = (__int64 *)(*v44 + 8LL);
          if ( !v343 )
            break;
          v48 = *(_QWORD *)(v45 + 8);
          v49 = 0;
          v50 = 1;
          v51 = (v343 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v52 = (__m128i *)&v341[3 * v51];
          v53 = v52->m128i_i64[0];
          if ( v48 != v52->m128i_i64[0] )
          {
            while ( v53 != -4096 )
            {
              if ( v53 == -8192 && !v49 )
                v49 = v52;
              v51 = (v343 - 1) & (v50 + v51);
              v52 = (__m128i *)&v341[3 * v51];
              v53 = v52->m128i_i64[0];
              if ( v48 == v52->m128i_i64[0] )
                goto LABEL_54;
              ++v50;
            }
            if ( v49 )
              v52 = v49;
            ++v340;
            v287 = v342 + 1;
            v359 = v52;
            if ( 4 * ((int)v342 + 1) < 3 * v343 )
            {
              if ( v343 - HIDWORD(v342) - v287 > v343 >> 3 )
              {
LABEL_524:
                LODWORD(v342) = v287;
                if ( v52->m128i_i64[0] != -4096 )
                  --HIDWORD(v342);
                v316 = 0;
                v311 = 0;
                v52->m128i_i64[0] = *(_QWORD *)(v45 + 8);
                v52->m128i_i64[1] = 0;
                v52[1].m128i_i32[0] = 0;
                goto LABEL_55;
              }
LABEL_529:
              sub_2644FF0((__int64)&v340, v46);
              sub_263DE10((__int64)&v340, v47, &v359);
              v287 = v342 + 1;
              v52 = v359;
              goto LABEL_524;
            }
LABEL_528:
            v46 = 2 * v343;
            goto LABEL_529;
          }
LABEL_54:
          v311 = v52[1].m128i_i32[0];
          v316 = v52->m128i_i64[1];
LABEL_55:
          v319 = 1;
LABEL_56:
          v362 = &v360;
          v363 = &v360;
          LODWORD(v360) = 0;
          v361 = 0;
          v364 = 0;
          v333 = sub_2654C20(a1, v328.m128i_i64, &v359, v305 + 1, v352, a2);
          v334 = v54;
          v329.m128i_i64[0] = (__int64)v333;
          v329.m128i_i32[2] = v54;
          sub_2641EB0((__int64)&v347, &v329, (__int64)&v359);
          if ( v319 )
          {
            v56 = v324 + 72;
            v310 = (_QWORD *)(v324 + 72);
            sub_2640E50(&v336, (_QWORD *)(v324 + 72), v55);
            v59 = v336.m128i_u64[1];
            if ( v336.m128i_i64[0] == v336.m128i_i64[1] )
              goto LABEL_165;
            v322 = v336.m128i_i64[1];
            v60 = v336.m128i_i64[0];
            while ( 1 )
            {
LABEL_68:
              v67 = *(volatile signed __int32 **)(v60 + 8);
              v68 = *(_QWORD **)v60;
              if ( v67 )
              {
                if ( &_pthread_key_create )
                  _InterlockedAdd(v67 + 2, 1u);
                else
                  ++*((_DWORD *)v67 + 2);
              }
              v69 = v68[1];
              if ( !*v68 && !v69 )
                goto LABEL_63;
              if ( !*(_QWORD *)(v69 + 8) )
                goto LABEL_63;
              v61 = v343;
              v62 = (__int64)v341;
              if ( !v343 )
                goto LABEL_63;
              v63 = v343 - 1;
              v64 = 1;
              v43 = (unsigned int)v63 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
              LODWORD(v65) = v63 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
              v57 = (unsigned __int64)&v341[3 * (unsigned int)v43];
              v56 = *(_QWORD *)v57;
              v58 = *(_QWORD *)v57;
              if ( *(_QWORD *)v57 == v69 )
                break;
              while ( 1 )
              {
                if ( v58 == -4096 )
                  goto LABEL_63;
                v65 = (unsigned int)v63 & ((_DWORD)v65 + v64);
                v58 = v341[3 * v65];
                if ( v58 == v69 )
                  break;
                ++v64;
              }
              v70 = 1;
              v71 = 0;
              while ( v56 != -4096 )
              {
                if ( !v71 && v56 == -8192 )
                  v71 = (__int64 *)v57;
                v43 = (unsigned int)v63 & ((_DWORD)v43 + v70);
                v57 = (unsigned __int64)&v341[3 * v43];
                v56 = *(_QWORD *)v57;
                if ( *(_QWORD *)v57 == v69 )
                  goto LABEL_62;
                ++v70;
              }
              if ( !v71 )
                v71 = (__int64 *)v57;
              ++v340;
              v72 = v342 + 1;
              v56 = (unsigned int)(4 * (v342 + 1));
              if ( (unsigned int)v56 >= 3 * v343 )
              {
                v181 = sub_AF1560(2 * v343 - 1);
                if ( v181 < 0x40 )
                  v181 = 64;
                v343 = v181;
                v182 = (_QWORD *)sub_C7D670(24LL * v181, 8);
                v341 = v182;
                v43 = (__int64)v182;
                if ( v62 )
                {
                  v342 = 0;
                  v183 = 24 * v61;
                  v184 = &v182[3 * v343];
                  if ( (_QWORD *)v43 != v184 )
                  {
                    do
                    {
                      if ( v43 )
                        *(_QWORD *)v43 = -4096;
                      v43 += 24;
                    }
                    while ( v184 != (_QWORD *)v43 );
                  }
                  v185 = v62;
                  do
                  {
                    v186 = *(_QWORD *)v185;
                    if ( *(_QWORD *)v185 != -8192 && v186 != -4096 )
                    {
                      if ( !v343 )
                      {
                        MEMORY[0] = *(_QWORD *)v185;
                        BUG();
                      }
                      v187 = (v343 - 1) & (((unsigned int)v186 >> 9) ^ ((unsigned int)v186 >> 4));
                      v188 = &v341[3 * v187];
                      v189 = *v188;
                      if ( v186 != *v188 )
                      {
                        v313 = 1;
                        v293 = 0;
                        while ( v189 != -4096 )
                        {
                          if ( !v293 && v189 == -8192 )
                            v293 = v188;
                          v187 = (v343 - 1) & (v313 + v187);
                          v188 = &v341[3 * v187];
                          v189 = *v188;
                          if ( v186 == *v188 )
                            goto LABEL_340;
                          ++v313;
                        }
                        if ( v293 )
                          v188 = v293;
                      }
LABEL_340:
                      *v188 = v186;
                      *(__m128i *)(v188 + 1) = _mm_loadu_si128((const __m128i *)(v185 + 8));
                      LODWORD(v342) = v342 + 1;
                    }
                    v185 += 24;
                  }
                  while ( v62 + v183 != v185 );
                  sub_C7D6A0(v62, v183, 8);
                  v43 = (__int64)v341;
                  v190 = v343;
                  v72 = v342 + 1;
                }
                else
                {
                  v342 = 0;
                  v190 = v343;
                  for ( k = &v182[3 * v343]; k != v182; v182 += 3 )
                  {
                    if ( v182 )
                      *v182 = -4096;
                  }
                  v72 = 1;
                }
                if ( !v190 )
                {
LABEL_705:
                  LODWORD(v342) = v342 + 1;
                  BUG();
                }
                v56 = v68[1];
                v191 = v190 - 1;
                v57 = v191 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v71 = (__int64 *)(v43 + 24 * v57);
                v192 = *v71;
                if ( *v71 != v56 )
                {
                  v193 = 1;
                  v194 = 0;
                  while ( v192 != -4096 )
                  {
                    if ( v192 == -8192 && !v194 )
                      v194 = v71;
                    v57 = v191 & ((_DWORD)v57 + v193);
                    v71 = (__int64 *)(v43 + 24 * v57);
                    v192 = *v71;
                    if ( v56 == *v71 )
                      goto LABEL_83;
                    ++v193;
                  }
                  if ( v194 )
                    v71 = v194;
                }
              }
              else
              {
                v43 = v343 - HIDWORD(v342) - v72;
                v57 = v343 >> 3;
                if ( (unsigned int)v43 <= (unsigned int)v57 )
                {
                  v195 = sub_AF1560(v63);
                  if ( v195 < 0x40 )
                    v195 = 64;
                  v343 = v195;
                  v196 = (_QWORD *)sub_C7D670(24LL * v195, 8);
                  v341 = v196;
                  v43 = (__int64)v196;
                  if ( v62 )
                  {
                    v342 = 0;
                    v197 = 24 * v61;
                    v198 = &v196[3 * v343];
                    if ( (_QWORD *)v43 != v198 )
                    {
                      do
                      {
                        if ( v43 )
                          *(_QWORD *)v43 = -4096;
                        v43 += 24;
                      }
                      while ( v198 != (_QWORD *)v43 );
                    }
                    v199 = v62;
                    do
                    {
                      v200 = *(_QWORD *)v199;
                      if ( *(_QWORD *)v199 != -4096 && v200 != -8192 )
                      {
                        if ( !v343 )
                        {
                          MEMORY[0] = *(_QWORD *)v199;
                          BUG();
                        }
                        v201 = (v343 - 1) & (((unsigned int)v200 >> 9) ^ ((unsigned int)v200 >> 4));
                        v202 = &v341[3 * v201];
                        v203 = *v202;
                        if ( v200 != *v202 )
                        {
                          v314 = 1;
                          v294 = 0;
                          while ( v203 != -4096 )
                          {
                            if ( !v294 && v203 == -8192 )
                              v294 = v202;
                            v201 = (v343 - 1) & (v314 + v201);
                            v202 = &v341[3 * v201];
                            v203 = *v202;
                            if ( v200 == *v202 )
                              goto LABEL_362;
                            ++v314;
                          }
                          if ( v294 )
                            v202 = v294;
                        }
LABEL_362:
                        *v202 = v200;
                        *(__m128i *)(v202 + 1) = _mm_loadu_si128((const __m128i *)(v199 + 8));
                        LODWORD(v342) = v342 + 1;
                      }
                      v199 += 24;
                    }
                    while ( v62 + v197 != v199 );
                    sub_C7D6A0(v62, v197, 8);
                    v43 = (__int64)v341;
                    v204 = v343;
                    v72 = v342 + 1;
                  }
                  else
                  {
                    v342 = 0;
                    v204 = v343;
                    for ( m = &v196[3 * v343]; m != v196; v196 += 3 )
                    {
                      if ( v196 )
                        *v196 = -4096;
                    }
                    v72 = 1;
                  }
                  if ( !v204 )
                    goto LABEL_705;
                  v205 = v68[1];
                  v206 = v204 - 1;
                  v207 = 1;
                  LODWORD(v208) = v206 & (((unsigned int)v205 >> 9) ^ ((unsigned int)v205 >> 4));
                  v71 = (__int64 *)(v43 + 24LL * (unsigned int)v208);
                  v57 = 0;
                  v56 = *v71;
                  if ( *v71 != v205 )
                  {
                    while ( v56 != -4096 )
                    {
                      if ( !v57 && v56 == -8192 )
                        v57 = (unsigned __int64)v71;
                      v208 = v206 & ((_DWORD)v208 + v207);
                      v71 = (__int64 *)(v43 + 24 * v208);
                      v56 = *v71;
                      if ( v205 == *v71 )
                        goto LABEL_83;
                      ++v207;
                    }
                    if ( v57 )
                      v71 = (__int64 *)v57;
                  }
                }
              }
LABEL_83:
              LODWORD(v342) = v72;
              if ( *v71 != -4096 )
                --HIDWORD(v342);
              v73 = v68[1];
              v58 = (__int64)(v71 + 1);
              *(_QWORD *)v58 = 0;
              *(_DWORD *)(v58 + 8) = 0;
              *(_QWORD *)(v58 - 8) = v73;
              if ( !v316 )
                goto LABEL_86;
LABEL_63:
              if ( !v67 )
                goto LABEL_67;
              if ( &_pthread_key_create )
              {
                v66 = _InterlockedExchangeAdd(v67 + 2, 0xFFFFFFFF);
              }
              else
              {
                v66 = *((_DWORD *)v67 + 2);
                v43 = (unsigned int)(v66 - 1);
                *((_DWORD *)v67 + 2) = v43;
              }
              if ( v66 != 1 )
                goto LABEL_67;
              (*(void (__fastcall **)(volatile signed __int32 *, __int64, __int64, unsigned __int64, __int64))(*(_QWORD *)v67 + 16LL))(
                v67,
                v56,
                v43,
                v57,
                v58);
              if ( &_pthread_key_create )
              {
                v113 = _InterlockedExchangeAdd(v67 + 3, 0xFFFFFFFF);
              }
              else
              {
                v113 = *((_DWORD *)v67 + 3);
                v43 = (unsigned int)(v113 - 1);
                *((_DWORD *)v67 + 3) = v43;
              }
              if ( v113 != 1 )
                goto LABEL_67;
              v60 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v67 + 24LL))(v67);
              if ( v322 == v60 )
                goto LABEL_153;
            }
LABEL_62:
            v58 = v57 + 8;
            if ( v316 != *(_QWORD *)(v57 + 8) )
              goto LABEL_63;
LABEL_86:
            if ( v311 != *(_DWORD *)(v58 + 8) )
              goto LABEL_63;
            v74 = v68[1];
            v337 = (__int64 *)v74;
            if ( !v343 )
            {
              ++v340;
              v344.m128i_i64[0] = 0;
LABEL_620:
              sub_2644FF0((__int64)&v340, 2 * v343);
LABEL_621:
              sub_263DE10((__int64)&v340, (__int64 *)&v337, &v344);
              v74 = (__int64)v337;
              v296 = v342 + 1;
              v78 = (__int64 *)v344.m128i_i64[0];
              goto LABEL_611;
            }
            v75 = 1;
            LODWORD(v76) = (v343 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
            v77 = &v341[3 * (unsigned int)v76];
            v78 = 0;
            v79 = *v77;
            if ( v74 == *v77 )
            {
LABEL_89:
              v80 = v77 + 1;
              goto LABEL_90;
            }
            while ( v79 != -4096 )
            {
              if ( v79 == -8192 && !v78 )
                v78 = v77;
              v76 = (v343 - 1) & ((_DWORD)v76 + v75);
              v77 = &v341[3 * v76];
              v79 = *v77;
              if ( v74 == *v77 )
                goto LABEL_89;
              ++v75;
            }
            if ( !v78 )
              v78 = v77;
            ++v340;
            v296 = v342 + 1;
            v344.m128i_i64[0] = (__int64)v78;
            if ( 4 * ((int)v342 + 1) >= 3 * v343 )
              goto LABEL_620;
            if ( v343 - HIDWORD(v342) - v296 <= v343 >> 3 )
            {
              sub_2644FF0((__int64)&v340, v343);
              goto LABEL_621;
            }
LABEL_611:
            LODWORD(v342) = v296;
            if ( *v78 != -4096 )
              --HIDWORD(v342);
            *v78 = v74;
            v80 = v78 + 1;
            *v80 = 0;
            *((_DWORD *)v80 + 2) = 0;
LABEL_90:
            *v80 = v329.m128i_i64[0];
            v81 = v329.m128i_u32[2];
            *((_DWORD *)v80 + 2) = v329.m128i_i32[2];
            v56 = v68[1] + 48LL;
            sub_2640E50(&v337, (_QWORD *)v56, v81);
            v82 = v337;
            v312 = v338;
            if ( v338 == v337 )
              goto LABEL_297;
            v306 = v67;
            while ( 2 )
            {
              v43 = *v82;
              v330 = *v82;
              v83 = (volatile signed __int32 *)v82[1];
              v331 = v83;
              if ( v83 )
              {
                if ( &_pthread_key_create )
                {
                  _InterlockedAdd(v83 + 2, 1u);
                  v43 = v330;
                }
                else
                {
                  ++*((_DWORD *)v83 + 2);
                }
              }
              v84 = *(_QWORD *)v43;
              if ( *(_QWORD *)v43 )
              {
                v326 = *(_QWORD *)v43;
                if ( v324 == v84 || !*(_QWORD *)(v84 + 8) || *(_QWORD *)(v43 + 8) == v84 )
                  goto LABEL_95;
LABEL_104:
                v344 = 0u;
                v345 = 0;
                v346 = 0;
                v85 = sub_2650560((_QWORD *)a1, &v330, (__int64)&v344);
                sub_C7D6A0(v344.m128i_i64[1], 4LL * (unsigned int)v346, 4);
                sub_264F9A0(v85);
                sub_264F9A0(v326);
                v86 = v343;
                v87 = v326;
                v88 = (__int64)v341;
                if ( v343 )
                {
                  v58 = v343 - 1;
                  v89 = 1;
                  v90 = v58 & (((unsigned int)v326 >> 9) ^ ((unsigned int)v326 >> 4));
                  v91 = v90;
                  v92 = (__int64)&v341[3 * v90];
                  v93 = *(_QWORD *)v92;
                  v94 = *(_QWORD *)v92;
                  if ( v326 == *(_QWORD *)v92 )
                  {
LABEL_106:
                    v332.m128i_i64[0] = v85;
                    v95 = v92 + 8;
                    goto LABEL_107;
                  }
                  while ( 1 )
                  {
                    if ( v94 == -4096 )
                      goto LABEL_109;
                    v239 = v89 + 1;
                    v240 = (unsigned int)v58 & (v91 + v89);
                    v91 = v240;
                    v94 = v341[3 * v240];
                    if ( v326 == v94 )
                      break;
                    v89 = v239;
                  }
                  v241 = 1;
                  v242 = 0;
                  while ( v93 != -4096 )
                  {
                    if ( v93 == -8192 && !v242 )
                      v242 = v92;
                    v90 = v58 & (v241 + v90);
                    v92 = (__int64)&v341[3 * v90];
                    v93 = *(_QWORD *)v92;
                    if ( v326 == *(_QWORD *)v92 )
                      goto LABEL_106;
                    ++v241;
                  }
                  if ( v242 )
                    v92 = v242;
                  ++v340;
                  v243 = v342 + 1;
                  v344.m128i_i64[0] = v92;
                  if ( 4 * ((int)v342 + 1) >= 3 * v343 )
                  {
                    v86 = 2 * v343;
                  }
                  else if ( v343 - HIDWORD(v342) - v243 > v343 >> 3 )
                  {
                    goto LABEL_437;
                  }
                  sub_2644FF0((__int64)&v340, v86);
                  sub_263DE10((__int64)&v340, &v326, &v344);
                  v87 = v326;
                  v92 = v344.m128i_i64[0];
                  v243 = v342 + 1;
LABEL_437:
                  LODWORD(v342) = v243;
                  if ( *(_QWORD *)v92 != -4096 )
                    --HIDWORD(v342);
                  *(_QWORD *)v92 = v87;
                  v95 = v92 + 8;
                  *(_QWORD *)(v92 + 8) = 0;
                  v88 = (__int64)v341;
                  *(_DWORD *)(v92 + 16) = 0;
                  v86 = v343;
                  v332.m128i_i64[0] = v85;
                  if ( !v343 )
                  {
                    ++v340;
                    v344.m128i_i64[0] = 0;
                    goto LABEL_441;
                  }
                  v58 = v343 - 1;
LABEL_107:
                  v96 = 1;
                  v97 = 0;
                  v98 = v58 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                  v99 = v88 + 24LL * v98;
                  v100 = *(_QWORD *)v99;
                  if ( v85 == *(_QWORD *)v99 )
                  {
LABEL_108:
                    v101 = v99 + 8;
                    *(_QWORD *)v101 = *(_QWORD *)v95;
                    *(_DWORD *)(v101 + 8) = *(_DWORD *)(v95 + 8);
                    v87 = v326;
                    goto LABEL_109;
                  }
                  while ( v100 != -4096 )
                  {
                    if ( v100 == -8192 && !v97 )
                      v97 = v99;
                    v98 = v58 & (v96 + v98);
                    v99 = v88 + 24LL * v98;
                    v100 = *(_QWORD *)v99;
                    if ( v85 == *(_QWORD *)v99 )
                      goto LABEL_108;
                    ++v96;
                  }
                  if ( v97 )
                    v99 = v97;
                  ++v340;
                  v245 = v342 + 1;
                  v344.m128i_i64[0] = v99;
                  if ( 4 * ((int)v342 + 1) < 3 * v86 )
                  {
                    v58 = v86 - (v245 + HIDWORD(v342));
                    v244 = v85;
                    if ( (unsigned int)v58 <= v86 >> 3 )
                    {
LABEL_442:
                      sub_2644FF0((__int64)&v340, v86);
                      sub_263DE10((__int64)&v340, v332.m128i_i64, &v344);
                      v244 = v332.m128i_i64[0];
                      v245 = v342 + 1;
                      v99 = v344.m128i_i64[0];
                    }
                    LODWORD(v342) = v245;
                    if ( *(_QWORD *)v99 != -4096 )
                      --HIDWORD(v342);
                    *(_QWORD *)v99 = v244;
                    *(_QWORD *)(v99 + 8) = 0;
                    *(_DWORD *)(v99 + 16) = 0;
                    goto LABEL_108;
                  }
LABEL_441:
                  v86 *= 2;
                  goto LABEL_442;
                }
LABEL_109:
                v102 = *(_QWORD *)(v87 + 120);
                if ( !v102 )
                  v102 = v87;
                v103 = (int *)v349;
                v332 = _mm_loadu_si128((const __m128i *)(v102 + 8));
                v332.m128i_i32[2] = 0;
                if ( !v349 )
                {
                  v104 = &v348;
                  goto LABEL_122;
                }
                v57 = v329.m128i_u32[2];
                v104 = &v348;
                do
                {
                  while ( *((_QWORD *)v103 + 4) >= v329.m128i_i64[0]
                       && (*((_QWORD *)v103 + 4) != v329.m128i_i64[0] || (unsigned int)v103[10] >= v329.m128i_i32[2]) )
                  {
                    v104 = v103;
                    v103 = (int *)*((_QWORD *)v103 + 2);
                    if ( !v103 )
                      goto LABEL_118;
                  }
                  v103 = (int *)*((_QWORD *)v103 + 3);
                }
                while ( v103 );
LABEL_118:
                if ( v104 == &v348
                  || *((_QWORD *)v104 + 4) > v329.m128i_i64[0]
                  || *((_QWORD *)v104 + 4) == v329.m128i_i64[0] && (unsigned int)v104[10] > v329.m128i_i32[2] )
                {
LABEL_122:
                  v344.m128i_i64[0] = (__int64)&v329;
                  v104 = (int *)sub_2642040(&v347, (__int64)v104, (const __m128i **)&v344);
                }
                v309 = v104 + 12;
                v105 = *((_QWORD *)v104 + 8);
                if ( !v105 )
                {
                  v56 = (__int64)(v104 + 14);
                  goto LABEL_134;
                }
                v56 = (__int64)(v104 + 14);
                do
                {
                  while ( *(_QWORD *)(v105 + 32) >= v332.m128i_i64[0]
                       && (*(_QWORD *)(v105 + 32) != v332.m128i_i64[0] || *(_DWORD *)(v105 + 40) >= v332.m128i_i32[2]) )
                  {
                    v56 = v105;
                    v105 = *(_QWORD *)(v105 + 16);
                    if ( !v105 )
                      goto LABEL_130;
                  }
                  v105 = *(_QWORD *)(v105 + 24);
                }
                while ( v105 );
LABEL_130:
                if ( (int *)v56 == v104 + 14
                  || *(_QWORD *)(v56 + 32) > v332.m128i_i64[0]
                  || *(_QWORD *)(v56 + 32) == v332.m128i_i64[0] && *(_DWORD *)(v56 + 40) > v332.m128i_i32[2] )
                {
LABEL_134:
                  v344.m128i_i64[0] = (__int64)&v332;
                  v56 = sub_263EBB0(v309, v56, (const __m128i **)&v344);
                }
                v43 = *(_QWORD *)(v56 + 48);
                v106 = *(_DWORD *)(v56 + 56);
                v107 = *(const __m128i **)(v85 + 24);
                *(_QWORD *)(v85 + 8) = v43;
                *(_DWORD *)(v85 + 16) = v106;
                v108 = &v107[*(unsigned int *)(v85 + 32)];
                if ( v108 == v107 )
                  goto LABEL_95;
                v308 = v82;
                v109 = v107;
                while ( 2 )
                {
                  a2 = _mm_loadu_si128(v109);
                  v344 = a2;
                  v344.m128i_i32[2] = 0;
                  v110 = (_QWORD *)*((_QWORD *)v104 + 8);
                  if ( v110 )
                  {
                    v56 = (__int64)(v104 + 14);
                    do
                    {
                      while ( 1 )
                      {
                        v111 = v110[2];
                        v57 = v110[3];
                        if ( v110[4] >= v344.m128i_i64[0] )
                          break;
                        v110 = (_QWORD *)v110[3];
                        if ( !v57 )
                          goto LABEL_142;
                      }
                      v56 = (__int64)v110;
                      v110 = (_QWORD *)v110[2];
                    }
                    while ( v111 );
LABEL_142:
                    if ( (int *)v56 == v104 + 14
                      || *(_QWORD *)(v56 + 32) > v344.m128i_i64[0]
                      || *(_QWORD *)(v56 + 32) == v344.m128i_i64[0] && *(_DWORD *)(v56 + 40) )
                    {
LABEL_146:
                      v327 = &v344;
                      v56 = sub_263EBB0(v309, v56, &v327);
                    }
                    v43 = *(_QWORD *)(v56 + 48);
                    v112 = *(_DWORD *)(v56 + 56);
                    ++v109;
                    v109[-1].m128i_i64[0] = v43;
                    v109[-1].m128i_i32[2] = v112;
                    if ( v108 == v109 )
                    {
                      v82 = v308;
                      goto LABEL_95;
                    }
                    continue;
                  }
                  break;
                }
                v56 = (__int64)(v104 + 14);
                goto LABEL_146;
              }
              if ( *(_QWORD *)(v43 + 8) )
              {
                v326 = 0;
                if ( MEMORY[8] )
                  goto LABEL_104;
              }
LABEL_95:
              if ( v331 )
                sub_A191D0(v331);
              v82 += 2;
              if ( v312 != v82 )
                continue;
              break;
            }
            v172 = v338;
            v82 = v337;
            v67 = v306;
            if ( v337 != v338 )
            {
              do
              {
                v173 = (volatile signed __int32 *)v82[1];
                if ( v173 )
                  sub_A191D0(v173);
                v82 += 2;
              }
              while ( v172 != v82 );
              v82 = v337;
            }
LABEL_297:
            if ( v82 )
            {
              v56 = v339 - (_QWORD)v82;
              j_j___libc_free_0((unsigned __int64)v82);
            }
            if ( v67 )
              sub_A191D0(v67);
LABEL_67:
            v60 += 16;
            if ( v322 != v60 )
              goto LABEL_68;
LABEL_153:
            v114 = v336.m128i_i64[1];
            v59 = v336.m128i_i64[0];
            if ( v336.m128i_i64[1] != v336.m128i_i64[0] )
            {
              do
              {
                while ( 1 )
                {
                  v115 = *(volatile signed __int32 **)(v59 + 8);
                  if ( v115 )
                  {
                    if ( &_pthread_key_create )
                    {
                      v116 = _InterlockedExchangeAdd(v115 + 2, 0xFFFFFFFF);
                    }
                    else
                    {
                      v116 = *((_DWORD *)v115 + 2);
                      v43 = (unsigned int)(v116 - 1);
                      *((_DWORD *)v115 + 2) = v43;
                    }
                    if ( v116 == 1 )
                    {
                      (*(void (__fastcall **)(volatile signed __int32 *, __int64, __int64, unsigned __int64, __int64))(*(_QWORD *)v115 + 16LL))(
                        v115,
                        v56,
                        v43,
                        v57,
                        v58);
                      if ( &_pthread_key_create )
                      {
                        v117 = _InterlockedExchangeAdd(v115 + 3, 0xFFFFFFFF);
                      }
                      else
                      {
                        v117 = *((_DWORD *)v115 + 3);
                        v43 = (unsigned int)(v117 - 1);
                        *((_DWORD *)v115 + 3) = v43;
                      }
                      if ( v117 == 1 )
                        break;
                    }
                  }
                  v59 += 16LL;
                  if ( v114 == v59 )
                    goto LABEL_164;
                }
                v59 += 16LL;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v115 + 24LL))(v115);
              }
              while ( v114 != v59 );
LABEL_164:
              v59 = v336.m128i_i64[0];
            }
LABEL_165:
            if ( v59 )
              j_j___libc_free_0(v59);
            v118 = (__int64 *)v361;
            if ( v361 )
            {
              do
              {
                sub_2641CE0(v118[3]);
                v119 = (unsigned __int64)v118;
                v118 = (__int64 *)v118[2];
                j_j___libc_free_0(v119);
              }
              while ( v118 );
            }
LABEL_169:
            LODWORD(v360) = 0;
            v361 = 0;
            v362 = &v360;
            v363 = &v360;
            v364 = 0;
            v332.m128i_i64[0] = 0;
            v332.m128i_i32[2] = 0;
            sub_2640E50(&v337, v310, v43);
            v20 = (unsigned __int64)v337;
            v323 = v338;
            if ( v338 == v337 )
              goto LABEL_17;
            v120 = v337;
            while ( 2 )
            {
              while ( 2 )
              {
                while ( 2 )
                {
                  while ( 2 )
                  {
                    v121 = *v120;
                    v122 = *(_QWORD *)(*v120 + 8);
                    if ( !*(_QWORD *)*v120 && !v122 )
                    {
                      v120 += 2;
                      if ( v323 != v120 )
                        continue;
                      goto LABEL_208;
                    }
                    break;
                  }
                  if ( !*(_QWORD *)(v122 + 8) )
                    goto LABEL_207;
                  v123 = v343;
                  v124 = (__int64)v341;
                  if ( !v343 )
                  {
                    if ( v332.m128i_i64[0] )
                      goto LABEL_225;
LABEL_306:
                    v174 = (__int64)v350;
                    if ( v350 != &v348 )
                    {
                      while ( &v354 != (int *)sub_263E860((__int64)&v353, (unsigned __int64 *)(v174 + 32)) )
                      {
                        v174 = sub_220EEE0(v175);
                        if ( (int *)v174 == &v348 )
                          goto LABEL_309;
                      }
                      v332.m128i_i64[0] = *(_QWORD *)(v175 + 32);
                      v332.m128i_i32[2] = *(_DWORD *)(v175 + 40);
                    }
LABEL_309:
                    sub_26422D0((__int64)v335, &v332, v315, v324);
                    LODWORD(v123) = v343;
                    v124 = (__int64)v341;
                    v122 = *(_QWORD *)(*v120 + 8);
                    if ( v343 )
                    {
                      v125 = v343 - 1;
                      v126 = ((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4);
                      LODWORD(v127) = (v343 - 1) & v126;
                      v176 = &v341[3 * (unsigned int)v127];
                      v129 = *v176;
                      if ( *v176 == v122 )
                      {
LABEL_311:
                        v144 = v176 + 1;
LABEL_231:
                        v120 += 2;
                        *v144 = v332.m128i_i64[0];
                        *((_DWORD *)v144 + 2) = v332.m128i_i32[2];
                        if ( v323 != v120 )
                          continue;
                        goto LABEL_208;
                      }
                      v128 = &v341[3 * (v125 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4)))];
LABEL_260:
                      v152 = 1;
                      v141 = 0;
                      while ( v129 != -4096 )
                      {
                        if ( v141 || v129 != -8192 )
                          v128 = v141;
                        v127 = v125 & ((_DWORD)v127 + v152);
                        v176 = (__int64 *)(v124 + 24 * v127);
                        v129 = *v176;
                        if ( v122 == *v176 )
                          goto LABEL_311;
                        ++v152;
                        v141 = v128;
                        v128 = (_QWORD *)(v124 + 24 * v127);
                      }
                      if ( !v141 )
                        v141 = v128;
                      ++v340;
                      v143 = v342 + 1;
                      if ( 4 * ((int)v342 + 1) < (unsigned int)(3 * v123) )
                      {
                        if ( (int)v123 - (v143 + HIDWORD(v342)) > (unsigned int)v123 >> 3 )
                        {
LABEL_228:
                          LODWORD(v342) = v143;
                          if ( *v141 != -4096 )
                            --HIDWORD(v342);
                          *v141 = v122;
                          v144 = v141 + 1;
                          *v144 = 0;
                          *((_DWORD *)v144 + 2) = 0;
                          goto LABEL_231;
                        }
                        sub_2644FF0((__int64)&v340, v123);
                        if ( v343 )
                        {
                          v153 = 1;
                          v154 = 0;
                          v155 = (v343 - 1) & v126;
                          v141 = &v341[3 * v155];
                          v156 = *v141;
                          v143 = v342 + 1;
                          if ( v122 != *v141 )
                          {
                            while ( v156 != -4096 )
                            {
                              if ( !v154 && v156 == -8192 )
                                v154 = v141;
                              v155 = (v343 - 1) & (v153 + v155);
                              v141 = &v341[3 * v155];
                              v156 = *v141;
                              if ( v122 == *v141 )
                                goto LABEL_228;
                              ++v153;
                            }
                            if ( v154 )
                              v141 = v154;
                          }
                          goto LABEL_228;
                        }
LABEL_706:
                        LODWORD(v342) = v342 + 1;
                        BUG();
                      }
                    }
                    else
                    {
LABEL_225:
                      ++v340;
                      LODWORD(v123) = 0;
                    }
                    sub_2644FF0((__int64)&v340, 2 * v123);
                    if ( v343 )
                    {
                      v140 = (v343 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
                      v141 = &v341[3 * v140];
                      v142 = *v141;
                      v143 = v342 + 1;
                      if ( *v141 != v122 )
                      {
                        v302 = 1;
                        v303 = 0;
                        while ( v142 != -4096 )
                        {
                          if ( !v303 && v142 == -8192 )
                            v303 = v141;
                          v140 = (v343 - 1) & (v302 + v140);
                          v141 = &v341[3 * v140];
                          v142 = *v141;
                          if ( *v141 == v122 )
                            goto LABEL_228;
                          ++v302;
                        }
                        if ( v303 )
                          v141 = v303;
                      }
                      goto LABEL_228;
                    }
                    goto LABEL_706;
                  }
                  break;
                }
                v125 = v343 - 1;
                v126 = ((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4);
                LODWORD(v127) = (v343 - 1) & v126;
                v128 = &v341[3 * (unsigned int)v127];
                v129 = *v128;
                if ( *v128 == v122 )
                {
LABEL_175:
                  v130 = (const __m128i *)(v128 + 1);
                  goto LABEL_176;
                }
                v149 = *v128;
                v150 = (v343 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
                v151 = 1;
                while ( 2 )
                {
                  if ( v149 == -4096 )
                  {
                    if ( !v332.m128i_i64[0] )
                      goto LABEL_306;
                    goto LABEL_260;
                  }
                  v150 = v125 & (v151 + v150);
                  v149 = v341[3 * v150];
                  if ( v149 != v122 )
                  {
                    ++v151;
                    continue;
                  }
                  break;
                }
                v246 = 1;
                v247 = 0;
                while ( v129 != -4096 )
                {
                  if ( v129 == -8192 && !v247 )
                    v247 = v128;
                  LODWORD(v127) = v125 & (v246 + v127);
                  v128 = &v341[3 * (unsigned int)v127];
                  v129 = *v128;
                  if ( *v128 == v122 )
                    goto LABEL_175;
                  ++v246;
                }
                if ( !v247 )
                  v247 = v128;
                ++v340;
                v248 = v342 + 1;
                if ( 4 * ((int)v342 + 1) < 3 * v343 )
                {
                  if ( v343 - HIDWORD(v342) - v248 > v343 >> 3 )
                    goto LABEL_453;
                  v318 = (__int64)v341;
                  v269 = sub_AF1560(v125);
                  if ( v269 < 0x40 )
                    v269 = 64;
                  v343 = v269;
                  v270 = (_QWORD *)sub_C7D670(24LL * v269, 8);
                  v341 = v270;
                  if ( v318 )
                  {
                    v342 = 0;
                    v271 = 24 * v123;
                    v272 = v318 + 24 * v123;
                    for ( n = &v270[3 * v343]; n != v270; v270 += 3 )
                    {
                      if ( v270 )
                        *v270 = -4096;
                    }
                    v274 = v318;
                    do
                    {
                      v275 = *(_QWORD *)v274;
                      if ( *(_QWORD *)v274 != -4096 && v275 != -8192 )
                      {
                        if ( !v343 )
                        {
                          MEMORY[0] = *(_QWORD *)v274;
                          BUG();
                        }
                        v276 = 1;
                        v277 = 0;
                        v278 = (v343 - 1) & (((unsigned int)v275 >> 9) ^ ((unsigned int)v275 >> 4));
                        v279 = &v341[3 * v278];
                        v280 = *v279;
                        if ( *v279 != v275 )
                        {
                          while ( v280 != -4096 )
                          {
                            if ( !v277 && v280 == -8192 )
                              v277 = v279;
                            v278 = (v343 - 1) & (v276 + v278);
                            v279 = &v341[3 * v278];
                            v280 = *v279;
                            if ( v275 == *v279 )
                              goto LABEL_490;
                            ++v276;
                          }
                          if ( v277 )
                            v279 = v277;
                        }
LABEL_490:
                        *v279 = v275;
                        *(__m128i *)(v279 + 1) = _mm_loadu_si128((const __m128i *)(v274 + 8));
                        LODWORD(v342) = v342 + 1;
                      }
                      v274 += 24;
                    }
                    while ( v272 != v274 );
                    sub_C7D6A0(v318, v271, 8);
                    v270 = v341;
                    v281 = v343;
                    v248 = v342 + 1;
                  }
                  else
                  {
                    v342 = 0;
                    v281 = v343;
                    v290 = &v270[3 * v343];
                    if ( v270 != v290 )
                    {
                      v291 = v270;
                      do
                      {
                        if ( v291 )
                          *v291 = -4096;
                        v291 += 3;
                      }
                      while ( v290 != v291 );
                    }
                    v248 = 1;
                  }
                  if ( v281 )
                  {
                    v282 = *(_QWORD *)(v121 + 8);
                    v283 = v281 - 1;
                    v284 = 1;
                    v268 = 0;
                    v285 = v283 & (((unsigned int)v282 >> 9) ^ ((unsigned int)v282 >> 4));
                    v247 = &v270[3 * v285];
                    v286 = *v247;
                    if ( *v247 != v282 )
                    {
                      while ( v286 != -4096 )
                      {
                        if ( v286 == -8192 && !v268 )
                          v268 = v247;
                        v285 = v283 & (v284 + v285);
                        v247 = &v270[3 * v285];
                        v286 = *v247;
                        if ( v282 == *v247 )
                          goto LABEL_453;
                        ++v284;
                      }
                      goto LABEL_475;
                    }
                    goto LABEL_453;
                  }
LABEL_703:
                  LODWORD(v342) = v342 + 1;
                  BUG();
                }
                v317 = (__int64)v341;
                v250 = sub_AF1560(2 * v343 - 1);
                if ( v250 < 0x40 )
                  v250 = 64;
                v343 = v250;
                v251 = (_QWORD *)sub_C7D670(24LL * v250, 8);
                v341 = v251;
                if ( v317 )
                {
                  v342 = 0;
                  v252 = 24 * v123;
                  v253 = v317 + 24 * v123;
                  for ( ii = &v251[3 * v343]; ii != v251; v251 += 3 )
                  {
                    if ( v251 )
                      *v251 = -4096;
                  }
                  v255 = v317;
                  do
                  {
                    v256 = *(_QWORD *)v255;
                    if ( *(_QWORD *)v255 != -4096 && v256 != -8192 )
                    {
                      if ( !v343 )
                      {
                        MEMORY[0] = *(_QWORD *)v255;
                        BUG();
                      }
                      v257 = 1;
                      v258 = 0;
                      v259 = (v343 - 1) & (((unsigned int)v256 >> 9) ^ ((unsigned int)v256 >> 4));
                      v260 = &v341[3 * v259];
                      v261 = *v260;
                      if ( v256 != *v260 )
                      {
                        while ( v261 != -4096 )
                        {
                          if ( v261 == -8192 && !v258 )
                            v258 = v260;
                          v259 = (v343 - 1) & (v257 + v259);
                          v260 = &v341[3 * v259];
                          v261 = *v260;
                          if ( v256 == *v260 )
                            goto LABEL_468;
                          ++v257;
                        }
                        if ( v258 )
                          v260 = v258;
                      }
LABEL_468:
                      *v260 = v256;
                      *(__m128i *)(v260 + 1) = _mm_loadu_si128((const __m128i *)(v255 + 8));
                      LODWORD(v342) = v342 + 1;
                    }
                    v255 += 24;
                  }
                  while ( v253 != v255 );
                  sub_C7D6A0(v317, v252, 8);
                  v251 = v341;
                  v262 = v343;
                  v248 = v342 + 1;
                }
                else
                {
                  v342 = 0;
                  v262 = v343;
                  v288 = &v251[3 * v343];
                  if ( v251 != v288 )
                  {
                    v289 = v251;
                    do
                    {
                      if ( v289 )
                        *v289 = -4096;
                      v289 += 3;
                    }
                    while ( v288 != v289 );
                  }
                  v248 = 1;
                }
                if ( !v262 )
                  goto LABEL_703;
                v263 = *(_QWORD *)(v121 + 8);
                v264 = v262 - 1;
                v265 = v264 & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
                v247 = &v251[3 * v265];
                v266 = *v247;
                if ( v263 != *v247 )
                {
                  v267 = 1;
                  v268 = 0;
                  while ( v266 != -4096 )
                  {
                    if ( v266 == -8192 && !v268 )
                      v268 = v247;
                    v265 = v264 & (v267 + v265);
                    v247 = &v251[3 * v265];
                    v266 = *v247;
                    if ( v263 == *v247 )
                      goto LABEL_453;
                    ++v267;
                  }
LABEL_475:
                  if ( v268 )
                    v247 = v268;
                }
LABEL_453:
                LODWORD(v342) = v248;
                if ( *v247 != -4096 )
                  --HIDWORD(v342);
                v249 = *(_QWORD *)(v121 + 8);
                v130 = (const __m128i *)(v247 + 1);
                v247[1] = 0;
                *((_DWORD *)v247 + 4) = 0;
                *v247 = v249;
LABEL_176:
                v336 = _mm_loadu_si128(v130);
                v131 = (__m128i *)&v354;
                if ( &v354 == (int *)sub_263E860((__int64)&v353, (unsigned __int64 *)&v336) )
                  goto LABEL_190;
                v132 = v355;
                if ( !v355 )
                {
                  v131 = (__m128i *)&v354;
                  goto LABEL_188;
                }
                do
                {
                  while ( v132[2].m128i_i64[0] >= (unsigned __int64)v336.m128i_i64[0]
                       && (v132[2].m128i_i64[0] != v336.m128i_i64[0]
                        || v132[2].m128i_i32[2] >= (unsigned __int32)v336.m128i_i32[2]) )
                  {
                    v131 = v132;
                    v132 = (__m128i *)v132[1].m128i_i64[0];
                    if ( !v132 )
                      goto LABEL_184;
                  }
                  v132 = (__m128i *)v132[1].m128i_i64[1];
                }
                while ( v132 );
LABEL_184:
                if ( v131 == (__m128i *)&v354
                  || v131[2].m128i_i64[0] > (unsigned __int64)v336.m128i_i64[0]
                  || v131[2].m128i_i64[0] == v336.m128i_i64[0]
                  && v131[2].m128i_i32[2] > (unsigned __int32)v336.m128i_i32[2] )
                {
LABEL_188:
                  v344.m128i_i64[0] = (__int64)&v336;
                  v131 = sub_263E8D0(&v353, (__int64)v131, (const __m128i **)&v344);
                }
                if ( v131[3].m128i_i64[0] == v324 )
                {
LABEL_190:
                  if ( v332.m128i_i64[0] )
                  {
                    if ( v332.m128i_i64[0] != v336.m128i_i64[0] || v332.m128i_i32[2] != v336.m128i_i32[2] )
                      break;
                    v120 += 2;
                    if ( v323 == v120 )
                      goto LABEL_208;
                  }
                  else
                  {
                    v332.m128i_i64[0] = v336.m128i_i64[0];
                    v120 += 2;
                    v332.m128i_i32[2] = v336.m128i_i32[2];
                    sub_26422D0((__int64)v335, &v336, v315, v324);
                    if ( v323 == v120 )
                      goto LABEL_208;
                  }
                  continue;
                }
                break;
              }
              if ( &v360 == (unsigned __int64 *)sub_263E860((__int64)&v359, (unsigned __int64 *)&v336) )
              {
                v344 = 0u;
                v345 = 0;
                v346 = 0;
                v330 = sub_2650560((_QWORD *)a1, v120, (__int64)&v344);
                sub_C7D6A0(v344.m128i_i64[1], 4LL * (unsigned int)v346, 4);
                sub_264F9A0(v330);
                v145 = v361;
                v146 = v330;
                if ( v361 )
                {
                  v147 = (__m128i *)&v360;
                  do
                  {
                    while ( v145[2].m128i_i64[0] >= (unsigned __int64)v336.m128i_i64[0]
                         && (v145[2].m128i_i64[0] != v336.m128i_i64[0]
                          || v145[2].m128i_i32[2] >= (unsigned __int32)v336.m128i_i32[2]) )
                    {
                      v147 = v145;
                      v145 = (__m128i *)v145[1].m128i_i64[0];
                      if ( !v145 )
                        goto LABEL_240;
                    }
                    v145 = (__m128i *)v145[1].m128i_i64[1];
                  }
                  while ( v145 );
LABEL_240:
                  if ( v147 == (__m128i *)&v360
                    || v147[2].m128i_i64[0] > (unsigned __int64)v336.m128i_i64[0]
                    || v147[2].m128i_i64[0] == v336.m128i_i64[0]
                    && v147[2].m128i_i32[2] > (unsigned __int32)v336.m128i_i32[2] )
                  {
LABEL_244:
                    v344.m128i_i64[0] = (__int64)&v336;
                    v147 = sub_263E8D0(&v359, (__int64)v147, (const __m128i **)&v344);
                  }
                  v147[3].m128i_i64[0] = v146;
                  v148 = v371;
                  if ( v371 == (__m128i *)(v373 - 8) )
                  {
                    sub_26408A0(&v365, &v330);
                  }
                  else
                  {
                    if ( v371 )
                    {
                      v371->m128i_i64[0] = v330;
                      v148 = v371;
                    }
                    v371 = (__m128i *)&v148->m128i_u64[1];
                  }
                  goto LABEL_206;
                }
                v147 = (__m128i *)&v360;
                goto LABEL_244;
              }
              v133 = v361;
              if ( !v361 )
              {
                v134 = (__m128i *)&v360;
                goto LABEL_204;
              }
              v134 = (__m128i *)&v360;
              do
              {
                while ( v133[2].m128i_i64[0] >= (unsigned __int64)v336.m128i_i64[0]
                     && (v133[2].m128i_i64[0] != v336.m128i_i64[0]
                      || v133[2].m128i_i32[2] >= (unsigned __int32)v336.m128i_i32[2]) )
                {
                  v134 = v133;
                  v133 = (__m128i *)v133[1].m128i_i64[0];
                  if ( !v133 )
                    goto LABEL_200;
                }
                v133 = (__m128i *)v133[1].m128i_i64[1];
              }
              while ( v133 );
LABEL_200:
              if ( v134 == (__m128i *)&v360
                || v134[2].m128i_i64[0] > (unsigned __int64)v336.m128i_i64[0]
                || v134[2].m128i_i64[0] == v336.m128i_i64[0]
                && v134[2].m128i_i32[2] > (unsigned __int32)v336.m128i_i32[2] )
              {
LABEL_204:
                v344.m128i_i64[0] = (__int64)&v336;
                v134 = sub_263E8D0(&v359, (__int64)v134, (const __m128i **)&v344);
              }
              v135 = v134[3].m128i_i64[0];
              v344 = 0u;
              v345 = 0;
              v346 = 0;
              sub_264FE30(a1, v120, v135, 0, (__int64)&v344);
              sub_C7D6A0(v344.m128i_i64[1], 4LL * (unsigned int)v346, 4);
              sub_264F9A0(v135);
LABEL_206:
              sub_264F9A0(v324);
LABEL_207:
              v120 += 2;
              if ( v323 != v120 )
                continue;
              break;
            }
LABEL_208:
            v136 = v338;
            v20 = (unsigned __int64)v337;
            if ( v338 != v337 )
            {
              do
              {
                v137 = *(volatile signed __int32 **)(v20 + 8);
                if ( v137 )
                {
                  if ( &_pthread_key_create )
                  {
                    v138 = _InterlockedExchangeAdd(v137 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v138 = *((_DWORD *)v137 + 2);
                    *((_DWORD *)v137 + 2) = v138 - 1;
                  }
                  if ( v138 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v137 + 16LL))(v137);
                    if ( &_pthread_key_create )
                    {
                      v139 = _InterlockedExchangeAdd(v137 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v139 = *((_DWORD *)v137 + 3);
                      *((_DWORD *)v137 + 3) = v139 - 1;
                    }
                    if ( v139 == 1 )
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v137 + 24LL))(v137);
                  }
                }
                v20 += 16LL;
              }
              while ( v136 != (__int64 *)v20 );
              v20 = (unsigned __int64)v337;
            }
LABEL_17:
            if ( v20 )
              j_j___libc_free_0(v20);
            v21 = (__int64 *)v361;
            if ( v361 )
            {
              do
              {
                sub_26416A0(v21[3]);
                v22 = (unsigned __int64)v21;
                v21 = (__int64 *)v21[2];
                j_j___libc_free_0(v22);
              }
              while ( v21 );
            }
LABEL_21:
            v23 = (__int64 *)v367;
            if ( v367 == v371 )
            {
LABEL_22:
              if ( (_BYTE)qword_4FF3AA8 )
              {
                v223 = v325;
                if ( *((_BYTE *)v325 + 2) )
                {
                  sub_264C780(v325);
                  v223 = v325;
                }
                v224 = (__int64 **)v223[6];
                v225 = (__int64 **)v223[7];
                if ( v225 != v224 )
                {
                  do
                  {
                    v226 = **v224;
                    if ( *(_BYTE *)(v226 + 2) )
                      sub_264C780((_QWORD *)v226);
                    v224 += 2;
                  }
                  while ( v225 != v224 );
                  v223 = v325;
                }
                v227 = v223[9];
                v228 = v223[10];
                if ( v228 != v227 )
                {
                  do
                  {
                    v229 = *(_QWORD *)(*(_QWORD *)v227 + 8LL);
                    if ( *(_BYTE *)(v229 + 2) )
                      sub_264C780((_QWORD *)v229);
                    v227 += 16;
                  }
                  while ( v228 != v227 );
                  v223 = v325;
                }
                v230 = (_QWORD **)v223[12];
                for ( jj = (_QWORD **)v223[13]; jj != v230; ++v230 )
                {
                  v232 = *v230;
                  if ( *((_BYTE *)*v230 + 2) )
                    sub_264C780(*v230);
                  v233 = (__int64 **)v232[7];
                  for ( kk = (__int64 **)v232[6]; v233 != kk; kk += 2 )
                  {
                    v235 = **kk;
                    if ( *(_BYTE *)(v235 + 2) )
                      sub_264C780((_QWORD *)v235);
                  }
                  v236 = v232[10];
                  for ( mm = v232[9]; v236 != mm; mm += 16 )
                  {
                    v238 = *(_QWORD *)(*(_QWORD *)mm + 8LL);
                    if ( *(_BYTE *)(v238 + 2) )
                      sub_264C780((_QWORD *)v238);
                  }
                }
              }
              v24 = v365;
              if ( v365 )
              {
                v25 = v370;
                v26 = v374 + 1;
                if ( v374 + 1 > v370 )
                {
                  do
                  {
                    v27 = *v25++;
                    j_j___libc_free_0(v27);
                  }
                  while ( v26 > v25 );
                  v24 = v365;
                }
                j_j___libc_free_0(v24);
              }
              v28 = (unsigned __int64)v355;
              if ( v355 )
              {
                do
                {
                  sub_26416A0(*(_QWORD *)(v28 + 24));
                  v29 = v28;
                  v28 = *(_QWORD *)(v28 + 16);
                  j_j___libc_free_0(v29);
                }
                while ( v28 );
              }
              goto LABEL_30;
            }
          }
          else
          {
            sub_26422D0((__int64)v335, &v329, v315, v324);
            v157 = *(_QWORD *)(v324 + 80);
            v158 = *(_QWORD *)(v324 + 72);
            if ( v158 == v157 )
              goto LABEL_286;
            while ( 2 )
            {
              v166 = *(_QWORD *)(*(_QWORD *)v158 + 8LL);
              if ( !*(_QWORD *)(v166 + 8) )
                goto LABEL_277;
              if ( !v343 )
              {
                ++v340;
                goto LABEL_281;
              }
              v159 = v343 - 1;
              v160 = 1;
              v161 = (v343 - 1) & (((unsigned int)v166 >> 9) ^ ((unsigned int)v166 >> 4));
              v162 = &v341[3 * v161];
              v163 = 0;
              v164 = *v162;
              if ( v166 == *v162 )
                goto LABEL_275;
              while ( 1 )
              {
                if ( v164 == -4096 )
                {
                  if ( !v163 )
                    v163 = v162;
                  ++v340;
                  v168 = v342 + 1;
                  if ( 4 * ((int)v342 + 1) >= 3 * v343 )
                  {
LABEL_281:
                    sub_2644FF0((__int64)&v340, 2 * v343);
                    if ( !v343 )
                      goto LABEL_708;
                    v167 = (v343 - 1) & (((unsigned int)v166 >> 9) ^ ((unsigned int)v166 >> 4));
                    v168 = v342 + 1;
                    v163 = &v341[3 * v167];
                    v169 = *v163;
                    if ( v166 != *v163 )
                    {
                      v300 = 1;
                      v301 = 0;
                      while ( v169 != -4096 )
                      {
                        if ( v169 == -8192 && !v301 )
                          v301 = v163;
                        v167 = (v343 - 1) & (v300 + v167);
                        v163 = &v341[3 * v167];
                        v169 = *v163;
                        if ( v166 == *v163 )
                          goto LABEL_283;
                        ++v300;
                      }
                      if ( v301 )
                        v163 = v301;
                    }
                  }
                  else if ( v343 - HIDWORD(v342) - v168 <= v343 >> 3 )
                  {
                    sub_2644FF0((__int64)&v340, v343);
                    if ( v343 )
                    {
                      v177 = 1;
                      v178 = 0;
                      v179 = (v343 - 1) & (((unsigned int)v166 >> 9) ^ ((unsigned int)v166 >> 4));
                      v168 = v342 + 1;
                      v163 = &v341[3 * v179];
                      v180 = *v163;
                      if ( v166 != *v163 )
                      {
                        while ( v180 != -4096 )
                        {
                          if ( v180 == -8192 && !v178 )
                            v178 = v163;
                          v179 = (v343 - 1) & (v177 + v179);
                          v163 = &v341[3 * v179];
                          v180 = *v163;
                          if ( v166 == *v163 )
                            goto LABEL_283;
                          ++v177;
                        }
                        if ( v178 )
                          v163 = v178;
                      }
                      goto LABEL_283;
                    }
LABEL_708:
                    LODWORD(v342) = v342 + 1;
                    BUG();
                  }
LABEL_283:
                  LODWORD(v342) = v168;
                  if ( *v163 != -4096 )
                    --HIDWORD(v342);
                  *v163 = v166;
                  v165 = v163 + 1;
                  *v165 = 0;
                  *((_DWORD *)v165 + 2) = 0;
                  goto LABEL_276;
                }
                if ( v164 != -8192 || v163 )
                  v162 = v163;
                v297 = v160 + 1;
                v298 = v161 + v160;
                v161 = v159 & v298;
                v299 = &v341[3 * (v159 & v298)];
                v164 = *v299;
                if ( v166 == *v299 )
                  break;
                v160 = v297;
                v163 = v162;
                v162 = v299;
              }
              v162 = &v341[3 * (v159 & v298)];
LABEL_275:
              v165 = v162 + 1;
LABEL_276:
              *v165 = v329.m128i_i64[0];
              *((_DWORD *)v165 + 2) = v329.m128i_i32[2];
LABEL_277:
              v158 += 16;
              if ( v157 != v158 )
                continue;
              break;
            }
LABEL_286:
            v170 = (__int64 *)v361;
            if ( v361 )
            {
              do
              {
                sub_2641CE0(v170[3]);
                v171 = (unsigned __int64)v170;
                v170 = (__int64 *)v170[2];
                j_j___libc_free_0(v171);
              }
              while ( v170 );
            }
            v319 = 1;
            v23 = (__int64 *)v367;
            if ( v367 == v371 )
              goto LABEL_22;
          }
        }
        ++v340;
        v359 = 0;
        goto LABEL_528;
      }
      sub_26408A0(&v365, &v325);
    }
    v37 = v325;
    v38 = v371;
    goto LABEL_43;
  }
  v319 = 0;
LABEL_33:
  LOBYTE(v353) = 3;
  v365 = 0;
  v359 = (__m128i *)&v353;
  v360 = a1;
  v361 = (__m128i *)&v340;
  sub_2646720((__int64)&v365, 0);
  v31 = *(_QWORD *)(a1 + 256);
  v32 = v31 + 24LL * *(unsigned int *)(a1 + 264);
  while ( v31 != v32 )
  {
    v33 = *(_QWORD *)(v31 + 16);
    v31 += 24;
    sub_264DE20((__int64 *)&v359, v33, (__int64)&v365, (__int64)&v359);
  }
  sub_C7D6A0(v366, 8LL * (unsigned int)v368, 8);
  sub_C7D6A0((__int64)v341, 24LL * v343, 8);
  return v319;
}
