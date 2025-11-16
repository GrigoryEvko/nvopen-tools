// Function: sub_209C180
// Address: 0x209c180
//
__int64 __fastcall sub_209C180(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int64 v9; // r14
  __int64 v11; // rbx
  unsigned int v12; // esi
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // r11
  __int64 k; // rcx
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // r14
  unsigned __int64 v23; // rax
  int v24; // edx
  int v25; // r9d
  __int64 v26; // r8
  unsigned __int64 v27; // rbx
  int v28; // r12d
  int v29; // r11d
  int v30; // r10d
  unsigned int i; // ecx
  __int64 v32; // rdx
  unsigned int v33; // ecx
  __int64 v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *v41; // rax
  unsigned int v42; // esi
  __int64 *v43; // r8
  __int64 v44; // rax
  int v45; // ecx
  unsigned int v46; // esi
  __int64 *v47; // rbx
  __int64 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // r8d
  int v52; // r9d
  int v53; // r8d
  int v54; // r9d
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  __int32 v57; // ebx
  __int64 v58; // r15
  _QWORD *v59; // rdi
  __int64 v60; // rdx
  __int64 **v61; // r15
  __int64 *v62; // r8
  _QWORD *v63; // rdi
  __int64 v64; // rax
  _QWORD *v65; // rsi
  __int64 v66; // r15
  unsigned __int32 v67; // ebx
  __int64 *v68; // rdi
  __int64 **v69; // rbx
  __int64 *v70; // r15
  __int64 *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r8
  int v74; // edx
  unsigned __int64 v75; // r9
  _QWORD *v76; // rdi
  __int64 v77; // rax
  __int64 *v78; // r10
  __int64 v79; // rsi
  _QWORD *v80; // rax
  __int64 v81; // r15
  unsigned __int32 v82; // ebx
  __int64 *v83; // rax
  __int64 v84; // rdx
  __int64 *v85; // rsi
  __int64 *v86; // rax
  __int64 v87; // rdx
  __int64 **v88; // rbx
  unsigned int v89; // r12d
  __int64 *v90; // r15
  int v91; // eax
  _QWORD *v92; // r13
  __int64 v93; // rax
  unsigned int v94; // ecx
  char v95; // al
  unsigned int v96; // edx
  _QWORD *v97; // r8
  __int64 v98; // r9
  __int64 v99; // rax
  _QWORD *v100; // rax
  __int64 v101; // r12
  unsigned int v102; // esi
  __int64 v103; // rcx
  unsigned int v104; // r15d
  unsigned int v105; // edx
  __int64 *v106; // rbx
  _QWORD *v107; // rax
  _QWORD *v108; // r15
  __int64 v109; // r13
  __int64 v110; // rax
  unsigned __int64 v111; // rcx
  unsigned __int8 v112; // dl
  unsigned __int64 v113; // rsi
  unsigned __int64 v114; // rcx
  __int64 v115; // rdx
  _QWORD *v116; // rax
  __int64 *v117; // r12
  int v118; // edx
  __int64 *v119; // rsi
  __int64 v120; // rax
  int v121; // r10d
  __int64 v122; // r8
  int v123; // r11d
  unsigned int ii; // ecx
  __int64 v125; // rdx
  unsigned int v126; // ecx
  unsigned int v127; // esi
  __int64 v128; // r9
  unsigned int v129; // r8d
  __int64 v130; // rax
  __int64 *v131; // rdi
  __int64 v132; // rcx
  __int32 v133; // edx
  __int64 v134; // rax
  __int16 v135; // cx
  __int64 v136; // r12
  __int64 **v137; // rax
  int v138; // edx
  __int64 *v139; // rbx
  __int64 *v140; // rcx
  unsigned int *v141; // rax
  unsigned __int64 v142; // rdx
  int v143; // edx
  __int64 v144; // r10
  __int64 v145; // rax
  __int64 v146; // rsi
  unsigned int v147; // edx
  __int64 v148; // r8
  __int64 v149; // r9
  __int64 v150; // rax
  __int64 *v151; // rax
  int v152; // edx
  __int64 v153; // rax
  __int64 v154; // r10
  __int64 v155; // rsi
  unsigned int v156; // edx
  __int64 v157; // r8
  __int64 v158; // r9
  __int64 v159; // rax
  __int64 *v160; // rax
  int v161; // ecx
  int v162; // edx
  __int64 v163; // rax
  __int64 v164; // r10
  unsigned int v165; // ecx
  __int64 v166; // rsi
  unsigned int v167; // edx
  __int64 v168; // r8
  __int64 v169; // r9
  __int64 v170; // rax
  __int64 *v171; // rax
  __int64 v172; // rax
  __int64 v173; // r8
  _QWORD *v174; // rax
  __int64 v175; // rdx
  const __m128i *v176; // rcx
  __int64 v177; // r8
  const __m128i *v178; // r9
  unsigned __int64 v179; // rax
  __m128 *v180; // rcx
  __int32 v181; // eax
  __int64 v182; // rdx
  __int64 v183; // rdi
  void *v184; // r10
  const __m128i *v185; // r8
  int v186; // r11d
  __int64 v187; // r9
  __int64 v188; // rax
  __m128i v189; // xmm0
  __int64 v190; // rax
  __int64 **v191; // rax
  __int64 v192; // rax
  __int64 **v193; // rax
  __int64 v194; // rax
  __int64 v195; // r9
  _QWORD *v196; // r15
  __int64 v197; // rcx
  int v198; // edx
  int v199; // ebx
  __int64 v200; // rax
  __int64 *v201; // r10
  __int64 v202; // r11
  __int64 v203; // rsi
  unsigned int v204; // edx
  unsigned __int64 v206; // rdx
  __int64 v207; // rdi
  __int64 v208; // rax
  int v209; // ecx
  unsigned int v210; // r9d
  __int64 v211; // rax
  __int64 *v212; // r8
  __int64 v213; // rax
  __int64 *v214; // r12
  int v215; // ecx
  int v216; // ecx
  int v217; // r11d
  __int64 v218; // rdx
  int v219; // eax
  int v220; // eax
  __int64 **v221; // rdx
  __int64 **v222; // rbx
  __int64 **v223; // r13
  __int64 *v224; // r15
  __int64 *v225; // rax
  __int64 v226; // rcx
  __int64 v227; // rdx
  __int64 v228; // r9
  __int64 *v229; // r8
  __int64 v230; // rdx
  __int64 **v231; // rdx
  __int64 v232; // rdx
  __int64 v233; // rax
  __int64 **v234; // rax
  __int64 v235; // rax
  __int64 *v236; // rax
  __int64 v237; // rdi
  __int64 v238; // rax
  __int64 v239; // r9
  __int64 *v240; // r15
  const void ***v241; // rcx
  int v242; // edx
  int v243; // ebx
  __int64 v244; // rax
  __int64 v245; // r10
  __int64 v246; // r11
  __int64 v247; // rsi
  __int64 *v248; // rbx
  __int64 **v249; // r8
  __int64 **v250; // rbx
  __int64 **v251; // r12
  __int64 *v252; // r13
  __int64 v253; // rdx
  __int64 v254; // rcx
  __int64 v255; // r8
  __int64 v256; // r9
  __int64 *v257; // r10
  __int64 v258; // r11
  __int64 v259; // rax
  __int64 **v260; // rax
  __int64 v261; // rdx
  _QWORD *v262; // r10
  __int64 v263; // r11
  __int64 v264; // rax
  _QWORD *v265; // rax
  _QWORD *v266; // rdx
  __int64 v267; // rax
  __int64 v268; // rax
  int v269; // edx
  __int64 *v270; // r10
  const void ***v271; // r9
  int v272; // r8d
  __int64 v273; // rax
  __int128 v274; // rcx
  __int64 v275; // rsi
  int v276; // edi
  int v277; // edi
  __int64 v278; // r9
  unsigned int v279; // esi
  __int64 *v280; // r8
  int v281; // r13d
  __int64 v282; // r10
  int v283; // r11d
  int v284; // r11d
  unsigned int v285; // esi
  int v286; // r12d
  __int64 *v287; // r10
  int v288; // r11d
  int v289; // r11d
  __int64 *v290; // r12
  int v291; // r10d
  unsigned int v292; // esi
  int v293; // esi
  int v294; // esi
  __int64 v295; // r8
  unsigned int v296; // r13d
  int v297; // r11d
  __int64 *v298; // rdi
  int v299; // r11d
  __int64 v300; // rcx
  int v301; // eax
  int v302; // eax
  int v303; // esi
  int v304; // esi
  __int64 v305; // r8
  unsigned int v306; // edx
  __int64 *v307; // rdi
  int v308; // r11d
  __int64 v309; // r10
  unsigned int j; // edx
  int v311; // edx
  int v312; // esi
  int v313; // esi
  __int64 v314; // r8
  int v315; // r11d
  unsigned int v316; // edx
  __int64 *v317; // rdi
  int v318; // r11d
  __int64 *v319; // r10
  int v320; // eax
  int v321; // eax
  int v322; // r15d
  __int64 v323; // r10
  int v324; // eax
  _QWORD *v325; // rax
  int v326; // esi
  int v327; // ecx
  int v328; // ecx
  __int64 v329; // rdi
  unsigned int v330; // edx
  _QWORD *v331; // rsi
  int v332; // r10d
  __int64 *v333; // r8
  int v334; // eax
  _QWORD *v335; // rsi
  __int64 v336; // rax
  __int64 v337; // rdi
  int v338; // r10d
  int v339; // esi
  int v340; // edx
  int v341; // edx
  __int64 v342; // rsi
  __int64 *v343; // rdi
  unsigned int v344; // r10d
  int v345; // r9d
  _QWORD *v346; // rcx
  int v347; // eax
  _QWORD *v348; // rsi
  int v349; // r10d
  __int64 v350; // rax
  __int64 v351; // rdi
  int v352; // edi
  __int64 v353; // rdx
  int v354; // r8d
  int v355; // edi
  __int128 v356; // [rsp-10h] [rbp-7C0h]
  unsigned int v357; // [rsp+0h] [rbp-7B0h]
  int v358; // [rsp+0h] [rbp-7B0h]
  __int64 *v359; // [rsp+0h] [rbp-7B0h]
  __int64 v360; // [rsp+0h] [rbp-7B0h]
  __int64 v361; // [rsp+0h] [rbp-7B0h]
  __int64 v362; // [rsp+0h] [rbp-7B0h]
  __int64 v363; // [rsp+0h] [rbp-7B0h]
  const __m128i *v364; // [rsp+0h] [rbp-7B0h]
  _QWORD *v365; // [rsp+0h] [rbp-7B0h]
  __int64 v366; // [rsp+8h] [rbp-7A8h]
  __int64 v367; // [rsp+8h] [rbp-7A8h]
  void *v368; // [rsp+10h] [rbp-7A0h]
  __int64 v369; // [rsp+18h] [rbp-798h]
  __int64 v370; // [rsp+18h] [rbp-798h]
  __int64 v371; // [rsp+18h] [rbp-798h]
  int v372; // [rsp+18h] [rbp-798h]
  __m128i *v373; // [rsp+18h] [rbp-798h]
  const void ***v374; // [rsp+18h] [rbp-798h]
  __int64 v375; // [rsp+18h] [rbp-798h]
  __int64 v376; // [rsp+18h] [rbp-798h]
  __int64 v377; // [rsp+18h] [rbp-798h]
  __int64 v378; // [rsp+18h] [rbp-798h]
  const __m128i *v379; // [rsp+18h] [rbp-798h]
  __int64 v380; // [rsp+18h] [rbp-798h]
  int v381; // [rsp+20h] [rbp-790h]
  unsigned int v382; // [rsp+28h] [rbp-788h]
  __int64 v383; // [rsp+28h] [rbp-788h]
  __int64 *v384; // [rsp+30h] [rbp-780h]
  __int64 v385; // [rsp+30h] [rbp-780h]
  __int128 v386; // [rsp+30h] [rbp-780h]
  __int64 *v387; // [rsp+40h] [rbp-770h]
  _QWORD *v388; // [rsp+40h] [rbp-770h]
  int v389; // [rsp+40h] [rbp-770h]
  int v390; // [rsp+40h] [rbp-770h]
  __int64 v391; // [rsp+40h] [rbp-770h]
  __int64 v392; // [rsp+40h] [rbp-770h]
  unsigned int v393; // [rsp+40h] [rbp-770h]
  int v394; // [rsp+48h] [rbp-768h]
  __int64 v395; // [rsp+48h] [rbp-768h]
  __int64 v396; // [rsp+48h] [rbp-768h]
  __int64 v397; // [rsp+48h] [rbp-768h]
  int v398; // [rsp+48h] [rbp-768h]
  int v399; // [rsp+48h] [rbp-768h]
  int v400; // [rsp+48h] [rbp-768h]
  int v401; // [rsp+48h] [rbp-768h]
  __m128i *v402; // [rsp+50h] [rbp-760h]
  unsigned __int64 v403; // [rsp+50h] [rbp-760h]
  __int64 v404; // [rsp+70h] [rbp-740h]
  __int64 **m; // [rsp+70h] [rbp-740h]
  __int64 **n; // [rsp+70h] [rbp-740h]
  __m128i *v407; // [rsp+70h] [rbp-740h]
  _QWORD *v408; // [rsp+70h] [rbp-740h]
  __int64 *v409; // [rsp+70h] [rbp-740h]
  __int64 v410; // [rsp+70h] [rbp-740h]
  __int64 *v411; // [rsp+70h] [rbp-740h]
  __int64 *v412; // [rsp+70h] [rbp-740h]
  __int64 *v413; // [rsp+70h] [rbp-740h]
  __int64 v414; // [rsp+78h] [rbp-738h]
  __int64 v415; // [rsp+78h] [rbp-738h]
  __int64 v416; // [rsp+80h] [rbp-730h]
  __int64 v417; // [rsp+80h] [rbp-730h]
  __int64 **v418; // [rsp+80h] [rbp-730h]
  _QWORD *v419; // [rsp+80h] [rbp-730h]
  __int64 *v420; // [rsp+80h] [rbp-730h]
  const __m128i *v421; // [rsp+80h] [rbp-730h]
  const __m128i *v422; // [rsp+80h] [rbp-730h]
  __int64 v423; // [rsp+80h] [rbp-730h]
  __int64 v424; // [rsp+80h] [rbp-730h]
  const void ***v425; // [rsp+80h] [rbp-730h]
  const __m128i *v426; // [rsp+80h] [rbp-730h]
  int v427; // [rsp+80h] [rbp-730h]
  int v428; // [rsp+80h] [rbp-730h]
  __int64 v429; // [rsp+88h] [rbp-728h]
  __int64 v430; // [rsp+90h] [rbp-720h]
  __int64 *v431; // [rsp+90h] [rbp-720h]
  int v432; // [rsp+90h] [rbp-720h]
  __int64 v433; // [rsp+90h] [rbp-720h]
  __int64 v434; // [rsp+90h] [rbp-720h]
  __int64 v435; // [rsp+90h] [rbp-720h]
  _QWORD *v436; // [rsp+90h] [rbp-720h]
  _QWORD *v437; // [rsp+90h] [rbp-720h]
  _QWORD *v438; // [rsp+90h] [rbp-720h]
  __int64 v439; // [rsp+98h] [rbp-718h]
  __int64 *v440; // [rsp+A0h] [rbp-710h]
  __int64 v441; // [rsp+B0h] [rbp-700h] BYREF
  __int64 v442; // [rsp+B8h] [rbp-6F8h]
  __int64 v443; // [rsp+C0h] [rbp-6F0h]
  unsigned int v444; // [rsp+C8h] [rbp-6E8h]
  _QWORD *v445; // [rsp+D0h] [rbp-6E0h] BYREF
  __int64 v446; // [rsp+D8h] [rbp-6D8h]
  _QWORD v447[64]; // [rsp+E0h] [rbp-6D0h] BYREF
  void *src; // [rsp+2E0h] [rbp-4D0h] BYREF
  __int64 v449; // [rsp+2E8h] [rbp-4C8h]
  _BYTE v450[512]; // [rsp+2F0h] [rbp-4C0h] BYREF
  __m128i v451; // [rsp+4F0h] [rbp-2C0h] BYREF
  _QWORD v452[86]; // [rsp+500h] [rbp-2B0h] BYREF

  v9 = a1;
  sub_2098560(a1 + 248, a1, a3, a4, a5, a6);
  v11 = *(_QWORD *)(a1 + 712);
  v12 = *(_DWORD *)(v11 + 328);
  LODWORD(v13) = v11 + 304;
  if ( !v12 )
  {
    ++*(_QWORD *)(v11 + 304);
    goto LABEL_347;
  }
  v14 = a2[28].m128i_i64[0];
  LODWORD(v15) = v12 - 1;
  v16 = *(_QWORD *)(v11 + 312);
  v17 = (v12 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v18 = v16 + 72LL * v17;
  k = *(_QWORD *)v18;
  if ( v14 != *(_QWORD *)v18 )
  {
    v322 = 1;
    v323 = 0;
    while ( k != -8 )
    {
      if ( k == -16 && !v323 )
        v323 = v18;
      v17 = v15 & (v322 + v17);
      v18 = v16 + 72LL * v17;
      k = *(_QWORD *)v18;
      if ( v14 == *(_QWORD *)v18 )
        goto LABEL_3;
      ++v322;
    }
    v324 = *(_DWORD *)(v11 + 320);
    if ( v323 )
      v18 = v323;
    ++*(_QWORD *)(v11 + 304);
    v14 = (unsigned int)(v324 + 1);
    if ( 4 * (int)v14 < 3 * v12 )
    {
      k = v12 >> 3;
      if ( v12 - *(_DWORD *)(v11 + 324) - (unsigned int)v14 > (unsigned int)k )
      {
LABEL_329:
        *(_DWORD *)(v11 + 320) = v14;
        if ( *(_QWORD *)v18 != -8 )
          --*(_DWORD *)(v11 + 324);
        v325 = (_QWORD *)a2[28].m128i_i64[0];
        a7 = 0;
        *(_OWORD *)(v18 + 8) = 0;
        *(_QWORD *)v18 = v325;
        *(_OWORD *)(v18 + 24) = 0;
        *(_OWORD *)(v18 + 40) = 0;
        *(_OWORD *)(v18 + 56) = 0;
        goto LABEL_3;
      }
      sub_209A280(v11 + 304, v12);
      v347 = *(_DWORD *)(v11 + 328);
      if ( v347 )
      {
        v348 = (_QWORD *)a2[28].m128i_i64[0];
        k = (unsigned int)(v347 - 1);
        v15 = *(_QWORD *)(v11 + 312);
        v13 = 0;
        v349 = 1;
        LODWORD(v350) = k & (((unsigned int)v348 >> 9) ^ ((unsigned int)v348 >> 4));
        v18 = v15 + 72LL * (unsigned int)v350;
        v351 = *(_QWORD *)v18;
        v14 = (unsigned int)(*(_DWORD *)(v11 + 320) + 1);
        if ( v348 == *(_QWORD **)v18 )
          goto LABEL_329;
        while ( v351 != -8 )
        {
          if ( v351 == -16 && !v13 )
            v13 = v18;
          v350 = (unsigned int)k & ((_DWORD)v350 + v349);
          v18 = v15 + 72 * v350;
          v351 = *(_QWORD *)v18;
          if ( v348 == *(_QWORD **)v18 )
            goto LABEL_329;
          ++v349;
        }
        goto LABEL_351;
      }
      goto LABEL_470;
    }
LABEL_347:
    sub_209A280(v11 + 304, 2 * v12);
    v334 = *(_DWORD *)(v11 + 328);
    if ( v334 )
    {
      v335 = (_QWORD *)a2[28].m128i_i64[0];
      k = (unsigned int)(v334 - 1);
      v15 = *(_QWORD *)(v11 + 312);
      LODWORD(v336) = k & (((unsigned int)v335 >> 9) ^ ((unsigned int)v335 >> 4));
      v18 = v15 + 72LL * (unsigned int)v336;
      v337 = *(_QWORD *)v18;
      v14 = (unsigned int)(*(_DWORD *)(v11 + 320) + 1);
      if ( v335 == *(_QWORD **)v18 )
        goto LABEL_329;
      v338 = 1;
      v13 = 0;
      while ( v337 != -8 )
      {
        if ( !v13 && v337 == -16 )
          v13 = v18;
        v336 = (unsigned int)k & ((_DWORD)v336 + v338);
        v18 = v15 + 72 * v336;
        v337 = *(_QWORD *)v18;
        if ( v335 == *(_QWORD **)v18 )
          goto LABEL_329;
        ++v338;
      }
LABEL_351:
      if ( v13 )
        v18 = v13;
      goto LABEL_329;
    }
LABEL_470:
    ++*(_DWORD *)(v11 + 320);
    BUG();
  }
LABEL_3:
  v441 = 0;
  v445 = v447;
  v446 = 0x4000000000LL;
  src = v450;
  v449 = 0x4000000000LL;
  v451.m128i_i64[1] = 0x4000000000LL;
  v20 = a2[9].m128i_u32[2];
  v451.m128i_i64[0] = (__int64)v452;
  v442 = 0;
  v443 = 0;
  v444 = 0;
  v404 = v20;
  if ( !v20 )
    goto LABEL_29;
  v416 = v9;
  v21 = v18;
  v22 = 0;
  do
  {
    v430 = 8 * v22;
    v23 = (unsigned __int64)sub_20685E0(v416, *(__int64 **)(a2[9].m128i_i64[0] + 8 * v22), a7, a8, a9);
    v26 = 8 * v22;
    v27 = v23;
    v28 = v24;
    v29 = v24;
    if ( v444 )
    {
      v25 = v444 - 1;
      v30 = 1;
      for ( i = (v444 - 1) & (v24 + ((v23 >> 9) ^ (v23 >> 4))); ; i = v25 & v33 )
      {
        v32 = v442 + 24LL * i;
        if ( *(_QWORD *)v32 == v23 && *(_DWORD *)(v32 + 8) == v29 )
          break;
        if ( !*(_QWORD *)v32 && *(_DWORD *)(v32 + 8) == -1 )
          goto LABEL_11;
        v33 = v30 + i;
        ++v30;
      }
      if ( v32 != v442 + 24LL * v444 )
      {
        v46 = *(_DWORD *)(v21 + 64);
        v47 = (__int64 *)(v430 + a2[9].m128i_i64[0]);
        if ( v46 )
        {
          v13 = *(_QWORD *)(v21 + 48);
          k = (v46 - 1) & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
          v48 = (__int64 *)(v13 + 16 * k);
          v15 = *v48;
          if ( *v47 == *v48 )
          {
LABEL_26:
            v14 = *(_QWORD *)(v32 + 16);
            v48[1] = v14;
            goto LABEL_27;
          }
          v432 = 1;
          v214 = 0;
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v214 )
              v214 = v48;
            k = (v46 - 1) & (v432 + (_DWORD)k);
            v48 = (__int64 *)(v13 + 16LL * (unsigned int)k);
            v15 = *v48;
            if ( *v47 == *v48 )
              goto LABEL_26;
            ++v432;
          }
          v215 = *(_DWORD *)(v21 + 56);
          if ( v214 )
            v48 = v214;
          ++*(_QWORD *)(v21 + 40);
          v216 = v215 + 1;
          if ( 4 * v216 < 3 * v46 )
          {
            LODWORD(v15) = v46 >> 3;
            if ( v46 - *(_DWORD *)(v21 + 60) - v216 > v46 >> 3 )
            {
LABEL_194:
              *(_DWORD *)(v21 + 56) = v216;
              if ( *v48 != -8 )
                --*(_DWORD *)(v21 + 60);
              k = *v47;
              v48[1] = 0;
              *v48 = k;
              goto LABEL_26;
            }
            v435 = v32;
            sub_209BDF0(v21 + 40, v46);
            v288 = *(_DWORD *)(v21 + 64);
            if ( v288 )
            {
              v289 = v288 - 1;
              v13 = *(_QWORD *)(v21 + 48);
              v290 = 0;
              v32 = v435;
              v291 = 1;
              v216 = *(_DWORD *)(v21 + 56) + 1;
              v292 = v289 & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
              v48 = (__int64 *)(v13 + 16LL * v292);
              v15 = *v48;
              if ( *v47 != *v48 )
              {
                while ( v15 != -8 )
                {
                  if ( !v290 && v15 == -16 )
                    v290 = v48;
                  v292 = v289 & (v291 + v292);
                  v48 = (__int64 *)(v13 + 16LL * v292);
                  v15 = *v48;
                  if ( *v47 == *v48 )
                    goto LABEL_194;
                  ++v291;
                }
                if ( v290 )
                  v48 = v290;
              }
              goto LABEL_194;
            }
LABEL_472:
            ++*(_DWORD *)(v21 + 56);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(v21 + 40);
        }
        v434 = v32;
        sub_209BDF0(v21 + 40, 2 * v46);
        v283 = *(_DWORD *)(v21 + 64);
        if ( v283 )
        {
          v284 = v283 - 1;
          v13 = *(_QWORD *)(v21 + 48);
          v32 = v434;
          v216 = *(_DWORD *)(v21 + 56) + 1;
          v285 = v284 & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
          v48 = (__int64 *)(v13 + 16LL * v285);
          v15 = *v48;
          if ( *v48 != *v47 )
          {
            v286 = 1;
            v287 = 0;
            while ( v15 != -8 )
            {
              if ( v15 != -16 || v287 )
                v48 = v287;
              v285 = v284 & (v286 + v285);
              v15 = *(_QWORD *)(v13 + 16LL * v285);
              if ( *v47 == v15 )
              {
                v48 = (__int64 *)(v13 + 16LL * v285);
                goto LABEL_194;
              }
              ++v286;
              v287 = v48;
              v48 = (__int64 *)(v13 + 16LL * v285);
            }
            if ( v287 )
              v48 = v287;
          }
          goto LABEL_194;
        }
        goto LABEL_472;
      }
    }
LABEL_11:
    v34 = (unsigned int)v446;
    v35 = (_QWORD *)(v430 + a2->m128i_i64[0]);
    if ( (unsigned int)v446 >= HIDWORD(v446) )
    {
      v400 = v29;
      v438 = (_QWORD *)(v430 + a2->m128i_i64[0]);
      sub_16CD150((__int64)&v445, v447, 0, 8, v26, v25);
      v34 = (unsigned int)v446;
      v26 = 8 * v22;
      v29 = v400;
      v35 = v438;
    }
    v445[v34] = *v35;
    v36 = (_QWORD *)a2[9].m128i_i64[0];
    LODWORD(v446) = v446 + 1;
    v37 = (unsigned int)v449;
    v38 = (_QWORD *)((char *)v36 + v26);
    if ( (unsigned int)v449 >= HIDWORD(v449) )
    {
      v392 = v26;
      v399 = v29;
      v437 = v38;
      sub_16CD150((__int64)&src, v450, 0, 8, v26, v25);
      v37 = (unsigned int)v449;
      v26 = v392;
      v29 = v399;
      v38 = v437;
    }
    *((_QWORD *)src + v37) = *v38;
    v39 = (_QWORD *)a2[18].m128i_i64[0];
    LODWORD(v449) = v449 + 1;
    v40 = v451.m128i_u32[2];
    v41 = (_QWORD *)((char *)v39 + v26);
    if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
    {
      v391 = v26;
      v398 = v29;
      v436 = v41;
      sub_16CD150((__int64)&v451, v452, 0, 8, v26, v25);
      v40 = v451.m128i_u32[2];
      v26 = v391;
      v29 = v398;
      v41 = v436;
    }
    *(_QWORD *)(v451.m128i_i64[0] + 8 * v40) = *v41;
    v42 = v444;
    v43 = (__int64 *)(a2[9].m128i_i64[0] + v26);
    ++v451.m128i_i32[2];
    v431 = v43;
    if ( !v444 )
    {
      ++v441;
LABEL_301:
      v401 = v29;
      sub_209BBE0((__int64)&v441, 2 * v42);
      if ( !v444 )
        goto LABEL_471;
      v15 = 0;
      LODWORD(v13) = 1;
      for ( j = (v444 - 1) & (v28 + ((v27 >> 9) ^ (v27 >> 4))); ; j = (v444 - 1) & v311 )
      {
        v44 = v442 + 24LL * j;
        if ( *(_QWORD *)v44 == v27 && *(_DWORD *)(v44 + 8) == v401 )
          break;
        if ( !*(_QWORD *)v44 )
        {
          v326 = *(_DWORD *)(v44 + 8);
          if ( v326 == -1 )
          {
            k = (unsigned int)v443;
            if ( v15 )
              v44 = v15;
            v352 = v443 + 1;
            goto LABEL_388;
          }
          if ( v326 == -2 && !v15 )
            v15 = v442 + 24LL * j;
        }
        v311 = v13 + j;
        LODWORD(v13) = v13 + 1;
      }
LABEL_387:
      k = (unsigned int)v443;
      v352 = v443 + 1;
      goto LABEL_388;
    }
    LODWORD(v15) = v444 - 1;
    v13 = 0;
    v394 = 1;
    for ( k = (v444 - 1) & (((unsigned int)(v27 >> 9) ^ (unsigned int)(v27 >> 4)) + v28); ; k = (unsigned int)v15 & v45 )
    {
      v44 = v442 + 24LL * (unsigned int)k;
      if ( *(_QWORD *)v44 == v27 && *(_DWORD *)(v44 + 8) == v29 )
      {
        v14 = *v431;
        *(_QWORD *)(v44 + 16) = *v431;
        goto LABEL_27;
      }
      if ( !*(_QWORD *)v44 )
        break;
LABEL_21:
      v45 = v394 + k;
      ++v394;
    }
    v339 = *(_DWORD *)(v44 + 8);
    if ( v339 != -1 )
    {
      if ( !v13 && v339 == -2 )
        v13 = v442 + 24LL * (unsigned int)k;
      goto LABEL_21;
    }
    v42 = v444;
    if ( v13 )
      v44 = v13;
    ++v441;
    v352 = v443 + 1;
    if ( 4 * ((int)v443 + 1) >= 3 * v444 )
      goto LABEL_301;
    k = v444 - HIDWORD(v443) - v352;
    LODWORD(v15) = v444 >> 3;
    if ( (unsigned int)k <= v444 >> 3 )
    {
      v381 = v29;
      sub_209BBE0((__int64)&v441, v444);
      if ( v444 )
      {
        LODWORD(v13) = 1;
        v353 = 0;
        for ( LODWORD(v15) = (v444 - 1) & (((v27 >> 9) ^ (v27 >> 4)) + v28); ; LODWORD(v15) = (v444 - 1) & v354 )
        {
          v44 = v442 + 24LL * (unsigned int)v15;
          if ( *(_QWORD *)v44 == v27 && *(_DWORD *)(v44 + 8) == v381 )
            break;
          if ( !*(_QWORD *)v44 )
          {
            v355 = *(_DWORD *)(v44 + 8);
            if ( v355 == -1 )
            {
              k = (unsigned int)v443;
              if ( v353 )
                v44 = v353;
              v352 = v443 + 1;
              goto LABEL_388;
            }
            if ( v355 == -2 && !v353 )
              v353 = v442 + 24LL * (unsigned int)v15;
          }
          v354 = v13 + v15;
          LODWORD(v13) = v13 + 1;
        }
        goto LABEL_387;
      }
LABEL_471:
      LODWORD(v443) = v443 + 1;
      BUG();
    }
LABEL_388:
    LODWORD(v443) = v352;
    if ( *(_QWORD *)v44 || *(_DWORD *)(v44 + 8) != -1 )
      --HIDWORD(v443);
    *(_QWORD *)v44 = v27;
    *(_QWORD *)(v44 + 16) = 0;
    *(_DWORD *)(v44 + 8) = v28;
    v14 = *v431;
    *(_QWORD *)(v44 + 16) = *v431;
LABEL_27:
    ++v22;
  }
  while ( v404 != v22 );
  v9 = v416;
LABEL_29:
  sub_20982A0((__int64)a2, (__int64)&v445, v14, k, v15, v13);
  sub_20982A0((__int64)a2[9].m128i_i64, (__int64)&src, v49, v50, v51, v52);
  v55 = v451.m128i_u32[2];
  v56 = a2[18].m128i_u32[2];
  v57 = v451.m128i_i32[2];
  if ( v451.m128i_u32[2] <= v56 )
  {
    if ( v451.m128i_i32[2] )
      memmove((void *)a2[18].m128i_i64[0], (const void *)v451.m128i_i64[0], 8LL * v451.m128i_u32[2]);
    a2[18].m128i_i32[2] = v57;
    v59 = (_QWORD *)v451.m128i_i64[0];
  }
  else
  {
    if ( v451.m128i_u32[2] > (unsigned __int64)a2[18].m128i_u32[3] )
    {
      v58 = 0;
      a2[18].m128i_i32[2] = 0;
      sub_16CD150((__int64)a2[18].m128i_i64, &a2[19], v55, 8, v53, v54);
      v55 = v451.m128i_u32[2];
    }
    else
    {
      v58 = 8 * v56;
      if ( a2[18].m128i_i32[2] )
      {
        memmove((void *)a2[18].m128i_i64[0], (const void *)v451.m128i_i64[0], 8 * v56);
        v55 = v451.m128i_u32[2];
      }
    }
    v59 = (_QWORD *)v451.m128i_i64[0];
    v60 = 8 * v55;
    if ( v451.m128i_i64[0] + v58 != v451.m128i_i64[0] + v60 )
    {
      memcpy((void *)(v58 + a2[18].m128i_i64[0]), (const void *)(v451.m128i_i64[0] + v58), v60 - v58);
      v59 = (_QWORD *)v451.m128i_i64[0];
    }
    a2[18].m128i_i32[2] = v57;
  }
  if ( v59 != v452 )
    _libc_free((unsigned __int64)v59);
  if ( src != v450 )
    _libc_free((unsigned __int64)src);
  if ( v445 != v447 )
    _libc_free((unsigned __int64)v445);
  j___libc_free_0(v442);
  v61 = (__int64 **)a2[269].m128i_i64[0];
  src = v450;
  v449 = 0xA00000000LL;
  v417 = a2[270].m128i_i64[0] & 2;
  for ( m = &v61[3 * a2[269].m128i_i64[1]]; m != v61; v61 += 3 )
  {
    v62 = *v61;
    if ( v417 )
    {
      v63 = (_QWORD *)a2[9].m128i_i64[0];
      v64 = a2[9].m128i_u32[2];
      v451.m128i_i64[0] = (__int64)*v61;
      if ( &v63[v64] == sub_20981E0(v63, (__int64)&v63[v64], v451.m128i_i64) )
      {
        v65 = (_QWORD *)(a2->m128i_i64[0] + 8LL * a2->m128i_u32[2]);
        if ( v65 == sub_20981E0(a2->m128i_i64[0], (__int64)v65, v451.m128i_i64) )
          continue;
      }
    }
    sub_209AB70(v62, v9, a7, a8, a9);
  }
  v66 = 0;
  v67 = 0;
  while ( a2->m128i_i32[2] > v67 )
  {
    sub_209AB70(*(__int64 **)(a2->m128i_i64[0] + 8 * v66), v9, a7, a8, a9);
    v68 = *(__int64 **)(a2[9].m128i_i64[0] + 8 * v66);
    v66 = ++v67;
    sub_209AB70(v68, v9, a7, a8, a9);
  }
  sub_2098400((__int64)&src, (__int64 *)v9, a2[269].m128i_i32[2], a7, *(double *)a8.m128i_i64, a9);
  v69 = (__int64 **)a2[269].m128i_i64[0];
  for ( n = &v69[3 * a2[269].m128i_i64[1]]; n != v69; v69 += 3 )
  {
    v70 = *v69;
    v71 = sub_20685E0(v9, *v69, a7, a8, a9);
    v73 = v72;
    v74 = 0;
    v75 = (unsigned __int64)v71;
    if ( v417 )
    {
      v76 = (_QWORD *)a2[9].m128i_i64[0];
      v77 = a2[9].m128i_u32[2];
      v451.m128i_i64[0] = (__int64)v70;
      if ( &v76[v77] != sub_20981E0(v76, (__int64)&v76[v77], v451.m128i_i64)
        || (v79 = a2->m128i_i64[0] + 8LL * a2->m128i_u32[2],
            v80 = sub_20981E0(a2->m128i_i64[0], v79, v78),
            v74 = 1,
            (_QWORD *)v79 != v80) )
      {
        v74 = 0;
      }
    }
    sub_20993A0(v75, v73, v74, (__int64)&src, v9, a7, *(double *)a8.m128i_i64, a9);
  }
  v81 = 0;
  v82 = 0;
  while ( a2->m128i_i32[2] > v82 )
  {
    v83 = sub_20685E0(v9, *(__int64 **)(a2->m128i_i64[0] + 8 * v81), a7, a8, a9);
    sub_20993A0((unsigned __int64)v83, v84, 0, (__int64)&src, v9, a7, *(double *)a8.m128i_i64, a9);
    v85 = *(__int64 **)(a2[9].m128i_i64[0] + 8 * v81);
    v81 = ++v82;
    v86 = sub_20685E0(v9, v85, a7, a8, a9);
    sub_20993A0((unsigned __int64)v86, v87, 0, (__int64)&src, v9, a7, *(double *)a8.m128i_i64, a9);
  }
  v88 = (__int64 **)a2[27].m128i_i64[0];
  v418 = &v88[3 * a2[27].m128i_i64[1]];
  if ( v88 != v418 )
  {
    v89 = v382;
    v407 = a2;
    do
    {
      while ( 1 )
      {
        v90 = sub_20685E0(v9, *v88, a7, a8, a9);
        v91 = *((unsigned __int16 *)v90 + 12);
        if ( v91 == 14 || v91 == 36 )
          break;
        v88 += 3;
        if ( v418 == v88 )
          goto LABEL_70;
      }
      v92 = *(_QWORD **)(v9 + 552);
      v93 = sub_1E0A0C0(v92[4]);
      v94 = 8 * sub_15A9520(v93, *(_DWORD *)(v93 + 4));
      if ( v94 == 32 )
      {
        v95 = 5;
      }
      else if ( v94 > 0x20 )
      {
        v95 = 6;
        if ( v94 != 64 )
        {
          v95 = 0;
          if ( v94 == 128 )
            v95 = 7;
        }
      }
      else
      {
        v95 = 3;
        if ( v94 != 8 )
          v95 = 4 * (v94 == 16);
      }
      LOBYTE(v89) = v95;
      v97 = sub_1D299D0(v92, *((_DWORD *)v90 + 21), v89, 0, 1);
      v98 = v96;
      v99 = (unsigned int)v449;
      if ( (unsigned int)v449 >= HIDWORD(v449) )
      {
        v388 = v97;
        v397 = v96;
        sub_16CD150((__int64)&src, v450, 0, 16, (int)v97, v96);
        v99 = (unsigned int)v449;
        v97 = v388;
        v98 = v397;
      }
      v100 = (char *)src + 16 * v99;
      v88 += 3;
      *v100 = v97;
      v100[1] = v98;
      LODWORD(v449) = v449 + 1;
    }
    while ( v418 != v88 );
LABEL_70:
    a2 = v407;
  }
  v101 = *(_QWORD *)(v9 + 712);
  v102 = *(_DWORD *)(v101 + 328);
  v408 = (_QWORD *)a2[28].m128i_i64[0];
  if ( !v102 )
  {
    ++*(_QWORD *)(v101 + 304);
    goto LABEL_339;
  }
  v103 = *(_QWORD *)(v101 + 312);
  v104 = ((unsigned int)v408 >> 9) ^ ((unsigned int)v408 >> 4);
  v105 = (v102 - 1) & v104;
  v106 = (__int64 *)(v103 + 72LL * v105);
  v107 = (_QWORD *)*v106;
  if ( v408 == (_QWORD *)*v106 )
    goto LABEL_73;
  v318 = 1;
  v319 = 0;
  while ( v107 != (_QWORD *)-8LL )
  {
    if ( !v319 && v107 == (_QWORD *)-16LL )
      v319 = v106;
    v105 = (v102 - 1) & (v318 + v105);
    v106 = (__int64 *)(v103 + 72LL * v105);
    v107 = (_QWORD *)*v106;
    if ( v408 == (_QWORD *)*v106 )
      goto LABEL_73;
    ++v318;
  }
  v320 = *(_DWORD *)(v101 + 320);
  if ( v319 )
    v106 = v319;
  ++*(_QWORD *)(v101 + 304);
  v321 = v320 + 1;
  if ( 4 * v321 >= 3 * v102 )
  {
LABEL_339:
    sub_209A280(v101 + 304, 2 * v102);
    v327 = *(_DWORD *)(v101 + 328);
    if ( v327 )
    {
      v328 = v327 - 1;
      v329 = *(_QWORD *)(v101 + 312);
      v330 = v328 & (((unsigned int)v408 >> 9) ^ ((unsigned int)v408 >> 4));
      v106 = (__int64 *)(v329 + 72LL * v330);
      v331 = (_QWORD *)*v106;
      v321 = *(_DWORD *)(v101 + 320) + 1;
      if ( v408 != (_QWORD *)*v106 )
      {
        v332 = 1;
        v333 = 0;
        while ( v331 != (_QWORD *)-8LL )
        {
          if ( v331 == (_QWORD *)-16LL && !v333 )
            v333 = v106;
          v330 = v328 & (v332 + v330);
          v106 = (__int64 *)(v329 + 72LL * v330);
          v331 = (_QWORD *)*v106;
          if ( v408 == (_QWORD *)*v106 )
            goto LABEL_320;
          ++v332;
        }
        if ( v333 )
          v106 = v333;
      }
      goto LABEL_320;
    }
    goto LABEL_467;
  }
  if ( v102 - *(_DWORD *)(v101 + 324) - v321 <= v102 >> 3 )
  {
    sub_209A280(v101 + 304, v102);
    v340 = *(_DWORD *)(v101 + 328);
    if ( v340 )
    {
      v341 = v340 - 1;
      v342 = *(_QWORD *)(v101 + 312);
      v343 = 0;
      v344 = v341 & v104;
      v345 = 1;
      v106 = (__int64 *)(v342 + 72LL * (v341 & v104));
      v346 = (_QWORD *)*v106;
      v321 = *(_DWORD *)(v101 + 320) + 1;
      if ( v408 != (_QWORD *)*v106 )
      {
        while ( v346 != (_QWORD *)-8LL )
        {
          if ( v346 == (_QWORD *)-16LL && !v343 )
            v343 = v106;
          v344 = v341 & (v345 + v344);
          v106 = (__int64 *)(v342 + 72LL * v344);
          v346 = (_QWORD *)*v106;
          if ( v408 == (_QWORD *)*v106 )
            goto LABEL_320;
          ++v345;
        }
        if ( v343 )
          v106 = v343;
      }
      goto LABEL_320;
    }
LABEL_467:
    ++*(_DWORD *)(v101 + 320);
    BUG();
  }
LABEL_320:
  *(_DWORD *)(v101 + 320) = v321;
  if ( *v106 != -8 )
    --*(_DWORD *)(v101 + 324);
  a7 = 0;
  *(_OWORD *)(v106 + 1) = 0;
  *v106 = (__int64)v408;
  *(_OWORD *)(v106 + 3) = 0;
  *(_OWORD *)(v106 + 5) = 0;
  *(_OWORD *)(v106 + 7) = 0;
LABEL_73:
  v108 = (_QWORD *)a2[18].m128i_i64[0];
  v395 = (__int64)(v106 + 1);
  v419 = &v108[a2[18].m128i_u32[2]];
  if ( v108 == v419 )
    goto LABEL_103;
  v402 = a2;
  while ( 2 )
  {
    v109 = *v108;
    v110 = *(_DWORD *)(*v108 + 20LL) & 0xFFFFFFF;
    v111 = *(_QWORD *)(*v108 - 24 * v110);
    v112 = *(_BYTE *)(v111 + 16);
    if ( v112 == 88 )
    {
      v213 = sub_157F120(*(_QWORD *)(v111 + 40));
      v111 = sub_157EBA0(v213);
      v112 = *(_BYTE *)(v111 + 16);
      v110 = *(_DWORD *)(v109 + 20) & 0xFFFFFFF;
    }
    if ( v112 <= 0x17u )
    {
      v113 = 0;
      goto LABEL_80;
    }
    if ( v112 == 78 )
    {
      v206 = v111 | 4;
    }
    else
    {
      v113 = 0;
      if ( v112 != 29 )
        goto LABEL_80;
      v206 = v111 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v113 = v206 & 0xFFFFFFFFFFFFFFF8LL;
    v114 = (v206 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v206 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
    if ( (v206 & 4) == 0 )
LABEL_80:
      v114 = v113 - 24LL * (*(_DWORD *)(v113 + 20) & 0xFFFFFFF);
    v115 = *(_QWORD *)(v109 + 24 * (2 - v110));
    v116 = *(_QWORD **)(v115 + 24);
    if ( *(_DWORD *)(v115 + 32) > 0x40u )
      v116 = (_QWORD *)*v116;
    v117 = *(__int64 **)(v114 + 24LL * (unsigned int)v116);
    v119 = sub_20685E0(v9, v117, a7, a8, a9);
    v120 = *(unsigned int *)(v9 + 272);
    if ( !(_DWORD)v120 )
    {
LABEL_94:
      v127 = *((_DWORD *)v106 + 8);
      goto LABEL_95;
    }
    v121 = v118;
    v122 = *(_QWORD *)(v9 + 256);
    v123 = 1;
    for ( ii = (v120 - 1) & (v118 + (((unsigned __int64)v119 >> 9) ^ ((unsigned __int64)v119 >> 4)));
          ;
          ii = (v120 - 1) & v126 )
    {
      v125 = v122 + 32LL * ii;
      if ( *(__int64 **)v125 == v119 && *(_DWORD *)(v125 + 8) == v121 )
        break;
      if ( !*(_QWORD *)v125 && *(_DWORD *)(v125 + 8) == -1 )
        goto LABEL_94;
      v126 = v123 + ii;
      ++v123;
    }
    v207 = v106[2];
    v127 = *((_DWORD *)v106 + 8);
    if ( v125 == v122 + 32 * v120 || (v208 = *(_QWORD *)(v125 + 16)) == 0 )
    {
LABEL_95:
      if ( v127 )
      {
        v128 = v106[2];
        v129 = (v127 - 1) & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
        v130 = v128 + 16LL * v129;
        v131 = *(__int64 **)v130;
        if ( v117 == *(__int64 **)v130 )
        {
LABEL_97:
          if ( *(_BYTE *)(v130 + 12) )
            *(_BYTE *)(v130 + 12) = 0;
LABEL_99:
          if ( *(_QWORD *)(v109 + 40) != v408[5] )
            sub_2090460(v9, (__int64)v117, a7, a8, a9);
          goto LABEL_101;
        }
        v299 = 1;
        v300 = 0;
        while ( v131 != (__int64 *)-8LL )
        {
          if ( !v300 && v131 == (__int64 *)-16LL )
            v300 = v130;
          v129 = (v127 - 1) & (v299 + v129);
          v130 = v128 + 16LL * v129;
          v131 = *(__int64 **)v130;
          if ( v117 == *(__int64 **)v130 )
            goto LABEL_97;
          ++v299;
        }
        if ( !v300 )
          v300 = v130;
        v301 = *((_DWORD *)v106 + 6);
        ++v106[1];
        v302 = v301 + 1;
        if ( 4 * v302 < 3 * v127 )
        {
          if ( v127 - *((_DWORD *)v106 + 7) - v302 > v127 >> 3 )
            goto LABEL_289;
          v393 = ((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4);
          sub_209BFB0(v395, v127);
          v312 = *((_DWORD *)v106 + 8);
          if ( !v312 )
          {
LABEL_469:
            ++*((_DWORD *)v106 + 6);
            BUG();
          }
          v313 = v312 - 1;
          v314 = v106[2];
          v309 = 0;
          v315 = 1;
          v316 = v313 & v393;
          v302 = *((_DWORD *)v106 + 6) + 1;
          v300 = v314 + 16LL * (v313 & v393);
          v317 = *(__int64 **)v300;
          if ( *(__int64 **)v300 == v117 )
            goto LABEL_289;
          while ( v317 != (__int64 *)-8LL )
          {
            if ( !v309 && v317 == (__int64 *)-16LL )
              v309 = v300;
            v316 = v313 & (v315 + v316);
            v300 = v314 + 16LL * v316;
            v317 = *(__int64 **)v300;
            if ( v117 == *(__int64 **)v300 )
              goto LABEL_289;
            ++v315;
          }
          goto LABEL_297;
        }
      }
      else
      {
        ++v106[1];
      }
      sub_209BFB0(v395, 2 * v127);
      v303 = *((_DWORD *)v106 + 8);
      if ( !v303 )
        goto LABEL_469;
      v304 = v303 - 1;
      v305 = v106[2];
      v306 = v304 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
      v302 = *((_DWORD *)v106 + 6) + 1;
      v300 = v305 + 16LL * v306;
      v307 = *(__int64 **)v300;
      if ( *(__int64 **)v300 == v117 )
        goto LABEL_289;
      v308 = 1;
      v309 = 0;
      while ( v307 != (__int64 *)-8LL )
      {
        if ( !v309 && v307 == (__int64 *)-16LL )
          v309 = v300;
        v306 = v304 & (v308 + v306);
        v300 = v305 + 16LL * v306;
        v307 = *(__int64 **)v300;
        if ( v117 == *(__int64 **)v300 )
          goto LABEL_289;
        ++v308;
      }
LABEL_297:
      if ( v309 )
        v300 = v309;
LABEL_289:
      *((_DWORD *)v106 + 6) = v302;
      if ( *(_QWORD *)v300 != -8 )
        --*((_DWORD *)v106 + 7);
      *(_QWORD *)v300 = v117;
      *(_BYTE *)(v300 + 12) = 0;
      goto LABEL_99;
    }
    v209 = *(_DWORD *)(v208 + 84);
    if ( !v127 )
    {
      ++v106[1];
      goto LABEL_253;
    }
    v210 = (v127 - 1) & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
    v211 = v207 + 16LL * v210;
    v212 = *(__int64 **)v211;
    if ( *(__int64 **)v211 != v117 )
    {
      v217 = 1;
      v218 = 0;
      while ( v212 != (__int64 *)-8LL )
      {
        if ( !v218 && v212 == (__int64 *)-16LL )
          v218 = v211;
        v210 = (v127 - 1) & (v217 + v210);
        v211 = v207 + 16LL * v210;
        v212 = *(__int64 **)v211;
        if ( v117 == *(__int64 **)v211 )
          goto LABEL_182;
        ++v217;
      }
      if ( !v218 )
        v218 = v211;
      v219 = *((_DWORD *)v106 + 6);
      ++v106[1];
      v220 = v219 + 1;
      if ( 4 * v220 < 3 * v127 )
      {
        if ( v127 - *((_DWORD *)v106 + 7) - v220 > v127 >> 3 )
          goto LABEL_203;
        v390 = v209;
        sub_209BFB0(v395, v127);
        v293 = *((_DWORD *)v106 + 8);
        if ( !v293 )
        {
LABEL_468:
          ++*((_DWORD *)v106 + 6);
          BUG();
        }
        v294 = v293 - 1;
        v295 = v106[2];
        v282 = 0;
        v296 = v294 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
        v209 = v390;
        v297 = 1;
        v220 = *((_DWORD *)v106 + 6) + 1;
        v218 = v295 + 16LL * v296;
        v298 = *(__int64 **)v218;
        if ( *(__int64 **)v218 == v117 )
          goto LABEL_203;
        while ( v298 != (__int64 *)-8LL )
        {
          if ( v298 == (__int64 *)-16LL && !v282 )
            v282 = v218;
          v296 = v294 & (v297 + v296);
          v218 = v295 + 16LL * v296;
          v298 = *(__int64 **)v218;
          if ( v117 == *(__int64 **)v218 )
            goto LABEL_203;
          ++v297;
        }
        goto LABEL_257;
      }
LABEL_253:
      v389 = v209;
      sub_209BFB0(v395, 2 * v127);
      v276 = *((_DWORD *)v106 + 8);
      if ( !v276 )
        goto LABEL_468;
      v277 = v276 - 1;
      v278 = v106[2];
      v209 = v389;
      v279 = v277 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
      v220 = *((_DWORD *)v106 + 6) + 1;
      v218 = v278 + 16LL * v279;
      v280 = *(__int64 **)v218;
      if ( v117 == *(__int64 **)v218 )
        goto LABEL_203;
      v281 = 1;
      v282 = 0;
      while ( v280 != (__int64 *)-8LL )
      {
        if ( v280 != (__int64 *)-16LL || v282 )
          v218 = v282;
        v279 = v277 & (v281 + v279);
        v280 = *(__int64 **)(v278 + 16LL * v279);
        if ( v117 == v280 )
        {
          v218 = v278 + 16LL * v279;
          goto LABEL_203;
        }
        ++v281;
        v282 = v218;
        v218 = v278 + 16LL * v279;
      }
LABEL_257:
      if ( v282 )
        v218 = v282;
LABEL_203:
      *((_DWORD *)v106 + 6) = v220;
      if ( *(_QWORD *)v218 != -8 )
        --*((_DWORD *)v106 + 7);
      *(_QWORD *)v218 = v117;
      *(_BYTE *)(v218 + 12) = 0;
      goto LABEL_206;
    }
LABEL_182:
    if ( *(_BYTE *)(v211 + 12) )
    {
      *(_DWORD *)(v211 + 8) = v209;
      goto LABEL_101;
    }
    v218 = v211;
LABEL_206:
    *(_DWORD *)(v218 + 8) = v209;
    *(_BYTE *)(v218 + 12) = 1;
LABEL_101:
    if ( v419 != ++v108 )
      continue;
    break;
  }
  a2 = v402;
LABEL_103:
  v440 = sub_2051C20((__int64 *)v9, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
  v132 = a2[271].m128i_i64[0];
  a2[30].m128i_i64[0] = (__int64)v440;
  a2[30].m128i_i32[2] = v133;
  sub_2061DC0(&v451, v9, a2 + 30, v132, a7, a8, a9);
  v396 = v451.m128i_i64[0];
  v134 = v452[0];
  if ( *(_BYTE *)(a2[31].m128i_i64[0] + 8) )
  {
    v135 = *(_WORD *)(v452[0] + 24LL);
    if ( v135 == 185 )
    {
      v134 = **(_QWORD **)(v452[0] + 32LL);
    }
    else if ( v135 == 47 )
    {
      do
        v134 = **(_QWORD **)(v134 + 32);
      while ( *(_WORD *)(v134 + 24) == 47 );
    }
  }
  v403 = 0;
  v420 = 0;
  v136 = **(_QWORD **)(v134 + 32);
  v451.m128i_i64[1] = v451.m128i_u32[2];
  v137 = *(__int64 ***)(v136 + 32);
  v138 = *(_DWORD *)(v136 + 56);
  v139 = *v137;
  v387 = v137[1];
  v140 = 0;
  v409 = 0;
  if ( v138 )
  {
    v141 = (unsigned int *)&v137[5 * (unsigned int)(v138 - 1)];
    v142 = v141[2];
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v141 + 40LL) + 16 * v142) == 111 )
      v140 = *(__int64 **)v141;
    else
      v142 = 0;
    v409 = v140;
    v403 = v142;
    v420 = v140;
  }
  v383 = a2[270].m128i_i64[0] & 1;
  if ( (a2[270].m128i_i64[0] & 1) == 0 )
  {
    v384 = v409;
    goto LABEL_114;
  }
  v249 = (__int64 **)a2[28].m128i_i64[1];
  v452[0] = v139;
  v451.m128i_i64[0] = (__int64)v452;
  v452[1] = v387;
  v451.m128i_i64[1] = 0x800000001LL;
  if ( &v249[3 * a2[29].m128i_i64[0]] == v249 )
  {
    if ( v420 )
    {
      v266 = v452;
      v267 = 2;
      goto LABEL_242;
    }
  }
  else
  {
    v385 = v136;
    v250 = &v249[3 * a2[29].m128i_i64[0]];
    v251 = v249;
    v373 = a2;
    do
    {
      while ( 1 )
      {
        v252 = *v251;
        v257 = sub_20685E0(v9, *v251, a7, a8, a9);
        v258 = v253;
        v259 = v451.m128i_u32[2];
        if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
        {
          v366 = v253;
          v359 = v257;
          sub_16CD150((__int64)&v451, v452, 0, 16, v255, v256);
          v259 = v451.m128i_u32[2];
          v257 = v359;
          v258 = v366;
        }
        v260 = (__int64 **)(v451.m128i_i64[0] + 16 * v259);
        v260[1] = (__int64 *)v258;
        *v260 = v257;
        ++v451.m128i_i32[2];
        if ( *(_BYTE *)(*v252 + 8) == 15 )
          break;
        v251 += 3;
        if ( v250 == v251 )
          goto LABEL_238;
      }
      v262 = sub_1D2AD90(*(_QWORD **)(v9 + 552), (__int64)v252, v253, v254, v255, v256);
      v263 = v261;
      v264 = v451.m128i_u32[2];
      if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
      {
        v367 = v261;
        v365 = v262;
        sub_16CD150((__int64)&v451, v452, 0, 16, v255, v256);
        v264 = v451.m128i_u32[2];
        v262 = v365;
        v263 = v367;
      }
      v251 += 3;
      v265 = (_QWORD *)(v451.m128i_i64[0] + 16 * v264);
      *v265 = v262;
      v265[1] = v263;
      ++v451.m128i_i32[2];
    }
    while ( v250 != v251 );
LABEL_238:
    v136 = v385;
    a2 = v373;
    if ( v420 )
    {
      if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
        sub_16CD150((__int64)&v451, v452, 0, 16, v255, v256);
      v266 = (_QWORD *)v451.m128i_i64[0];
      v267 = 2LL * v451.m128i_u32[2];
LABEL_242:
      v266[v267] = v409;
      v266[v267 + 1] = v403;
      ++v451.m128i_i32[2];
    }
  }
  v268 = sub_1D252B0(*(_QWORD *)(v9 + 552), 1, 0, 111, 0);
  v270 = *(__int64 **)(v9 + 552);
  v445 = 0;
  v271 = (const void ***)v268;
  v272 = v269;
  v273 = *(_QWORD *)v9;
  *(_QWORD *)&v274 = v451.m128i_i64[0];
  *((_QWORD *)&v274 + 1) = v451.m128i_u32[2];
  LODWORD(v446) = *(_DWORD *)(v9 + 536);
  if ( v273 )
  {
    if ( &v445 != (_QWORD **)(v273 + 48) )
    {
      v275 = *(_QWORD *)(v273 + 48);
      v445 = (_QWORD *)v275;
      if ( v275 )
      {
        v358 = v269;
        v374 = v271;
        *(_QWORD *)&v386 = v451.m128i_i64[0];
        *((_QWORD *)&v386 + 1) = v451.m128i_u32[2];
        v411 = v270;
        sub_1623A60((__int64)&v445, v275, 2);
        v272 = v358;
        v271 = v374;
        v274 = v386;
        v270 = v411;
      }
    }
  }
  v384 = sub_1D36D80(
           v270,
           241,
           (__int64)&v445,
           v271,
           v272,
           *(double *)a7.m128i_i64,
           *(double *)a8.m128i_i64,
           a9,
           (__int64)v271,
           v274);
  if ( v445 )
    sub_161E7C0((__int64)&v445, (__int64)v445);
  v387 = (__int64 *)((unsigned __int64)v387 & 0xFFFFFFFF00000000LL);
  v409 = v384;
  v139 = v384;
  v403 = v403 & 0xFFFFFFFF00000000LL | 1;
  if ( (_QWORD *)v451.m128i_i64[0] != v452 )
    _libc_free(v451.m128i_u64[0]);
LABEL_114:
  v143 = *(_DWORD *)(v9 + 536);
  v445 = 0;
  v144 = *(_QWORD *)(v9 + 552);
  v451.m128i_i64[0] = (__int64)v452;
  v451.m128i_i64[1] = 0x2800000000LL;
  v145 = *(_QWORD *)v9;
  LODWORD(v446) = v143;
  if ( v145 )
  {
    if ( &v445 != (_QWORD **)(v145 + 48) )
    {
      v146 = *(_QWORD *)(v145 + 48);
      v445 = (_QWORD *)v146;
      if ( v146 )
      {
        v369 = v144;
        sub_1623A60((__int64)&v445, v146, 2);
        v144 = v369;
      }
    }
  }
  v148 = sub_1D38BB0(v144, a2[29].m128i_u32[2], (__int64)&v445, 6, 0, 1, a7, *(double *)a8.m128i_i64, a9, 0);
  v149 = v147;
  v150 = v451.m128i_u32[2];
  if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
  {
    v362 = v148;
    v378 = v147;
    sub_16CD150((__int64)&v451, v452, 0, 16, v148, v147);
    v150 = v451.m128i_u32[2];
    v148 = v362;
    v149 = v378;
  }
  v151 = (__int64 *)(v451.m128i_i64[0] + 16 * v150);
  *v151 = v148;
  v151[1] = v149;
  ++v451.m128i_i32[2];
  if ( v445 )
    sub_161E7C0((__int64)&v445, (__int64)v445);
  v152 = *(_DWORD *)(v9 + 536);
  v153 = *(_QWORD *)v9;
  v445 = 0;
  v154 = *(_QWORD *)(v9 + 552);
  LODWORD(v446) = v152;
  if ( v153 )
  {
    if ( &v445 != (_QWORD **)(v153 + 48) )
    {
      v155 = *(_QWORD *)(v153 + 48);
      v445 = (_QWORD *)v155;
      if ( v155 )
      {
        v370 = v154;
        sub_1623A60((__int64)&v445, v155, 2);
        v154 = v370;
      }
    }
  }
  v157 = sub_1D38BB0(v154, a2[270].m128i_u32[2], (__int64)&v445, 5, 0, 1, a7, *(double *)a8.m128i_i64, a9, 0);
  v158 = v156;
  v159 = v451.m128i_u32[2];
  if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
  {
    v361 = v157;
    v377 = v156;
    sub_16CD150((__int64)&v451, v452, 0, 16, v157, v156);
    v159 = v451.m128i_u32[2];
    v157 = v361;
    v158 = v377;
  }
  v160 = (__int64 *)(v451.m128i_i64[0] + 16 * v159);
  *v160 = v157;
  v160[1] = v158;
  ++v451.m128i_i32[2];
  if ( v445 )
    sub_161E7C0((__int64)&v445, (__int64)v445);
  v161 = *(_DWORD *)(v136 + 56);
  v162 = *(_DWORD *)(v9 + 536);
  v445 = 0;
  v163 = *(_QWORD *)v9;
  LODWORD(v446) = v162;
  v164 = *(_QWORD *)(v9 + 552);
  v165 = (v420 == 0) + v161 - 4;
  if ( v163 )
  {
    if ( &v445 != (_QWORD **)(v163 + 48) )
    {
      v166 = *(_QWORD *)(v163 + 48);
      v445 = (_QWORD *)v166;
      if ( v166 )
      {
        v357 = v165;
        v371 = v164;
        sub_1623A60((__int64)&v445, v166, 2);
        v165 = v357;
        v164 = v371;
      }
    }
  }
  v168 = sub_1D38BB0(v164, v165, (__int64)&v445, 5, 0, 1, a7, *(double *)a8.m128i_i64, a9, 0);
  v169 = v167;
  v170 = v451.m128i_u32[2];
  if ( v451.m128i_i32[2] >= (unsigned __int32)v451.m128i_i32[3] )
  {
    v360 = v168;
    v376 = v167;
    sub_16CD150((__int64)&v451, v452, 0, 16, v168, v167);
    v170 = v451.m128i_u32[2];
    v168 = v360;
    v169 = v376;
  }
  v171 = (__int64 *)(v451.m128i_i64[0] + 16 * v170);
  *v171 = v168;
  v171[1] = v169;
  v172 = (unsigned int)++v451.m128i_i32[2];
  if ( v445 )
  {
    sub_161E7C0((__int64)&v445, (__int64)v445);
    v172 = v451.m128i_u32[2];
  }
  v173 = *(_QWORD *)(*(_QWORD *)(v136 + 32) + 40LL);
  if ( v451.m128i_i32[3] <= (unsigned int)v172 )
  {
    v375 = *(_QWORD *)(*(_QWORD *)(v136 + 32) + 40LL);
    sub_16CD150((__int64)&v451, v452, 0, 16, v173, v169);
    v172 = v451.m128i_u32[2];
    v173 = v375;
  }
  v174 = (_QWORD *)(v451.m128i_i64[0] + 16 * v172);
  *v174 = v173;
  v174[1] = 0;
  v175 = (unsigned int)++v451.m128i_i32[2];
  v176 = *(const __m128i **)(v136 + 32);
  if ( v420 )
    v177 = (__int64)&v176[-5].m128i_i64[5 * *(unsigned int *)(v136 + 56)];
  else
    v177 = (__int64)&v176[-2] + 40 * *(unsigned int *)(v136 + 56) - 8;
  v178 = v176 + 5;
  v179 = 0xCCCCCCCCCCCCCCCDLL * ((v177 - (__int64)v176[5].m128i_i64) >> 3);
  if ( v179 > (unsigned __int64)v451.m128i_u32[3] - v175 )
  {
    v364 = v176 + 5;
    v380 = v177;
    v428 = -858993459 * ((v177 - (__int64)v176[5].m128i_i64) >> 3);
    sub_16CD150((__int64)&v451, v452, v179 + v175, 16, v177, (int)v178);
    v175 = v451.m128i_u32[2];
    v178 = v364;
    v177 = v380;
    LODWORD(v179) = v428;
  }
  v180 = (__m128 *)(v451.m128i_i64[0] + 16 * v175);
  if ( v178 != (const __m128i *)v177 )
  {
    do
    {
      if ( v180 )
      {
        a8 = _mm_loadu_si128(v178);
        *v180 = (__m128)a8;
      }
      v178 = (const __m128i *)((char *)v178 + 40);
      ++v180;
    }
    while ( (const __m128i *)v177 != v178 );
    LODWORD(v175) = v451.m128i_i32[2];
  }
  v181 = v175 + v179;
  v182 = a2[32].m128i_u32[0];
  v421 = (const __m128i *)v177;
  v451.m128i_i32[2] = v181;
  sub_2098400((__int64)&v451, (__int64 *)v9, v182, a7, *(double *)a8.m128i_i64, a9);
  sub_2098400((__int64)&v451, (__int64 *)v9, a2[270].m128i_i64[0], a7, *(double *)a8.m128i_i64, a9);
  v183 = v451.m128i_u32[2];
  v184 = src;
  v185 = v421;
  v186 = v449;
  v187 = 16LL * (unsigned int)v449;
  if ( (unsigned int)v449 > v451.m128i_u32[3] - (unsigned __int64)v451.m128i_u32[2] )
  {
    v368 = src;
    v363 = 16LL * (unsigned int)v449;
    v379 = v421;
    v427 = v449;
    sub_16CD150((__int64)&v451, v452, v451.m128i_u32[2] + (unsigned __int64)(unsigned int)v449, 16, (int)v185, v187);
    v183 = v451.m128i_u32[2];
    v184 = v368;
    v187 = v363;
    v185 = v379;
    v186 = v427;
  }
  if ( v187 )
  {
    v372 = v186;
    v422 = v185;
    memcpy((void *)(v451.m128i_i64[0] + 16 * v183), v184, v187);
    LODWORD(v183) = v451.m128i_i32[2];
    v186 = v372;
    v185 = v422;
  }
  v451.m128i_i32[2] = v183 + v186;
  v188 = (unsigned int)(v183 + v186);
  if ( v451.m128i_i32[3] <= (unsigned int)(v183 + v186) )
  {
    v426 = v185;
    sub_16CD150((__int64)&v451, v452, 0, 16, (int)v185, v187);
    v188 = v451.m128i_u32[2];
    v185 = v426;
  }
  v189 = _mm_loadu_si128(v185);
  *(__m128i *)(v451.m128i_i64[0] + 16 * v188) = v189;
  v190 = (unsigned int)(v451.m128i_i32[2] + 1);
  v451.m128i_i32[2] = v190;
  if ( v451.m128i_i32[3] <= (unsigned int)v190 )
  {
    sub_16CD150((__int64)&v451, v452, 0, 16, (int)v185, v187);
    v190 = v451.m128i_u32[2];
  }
  v191 = (__int64 **)(v451.m128i_i64[0] + 16 * v190);
  *v191 = v139;
  v191[1] = v387;
  v192 = (unsigned int)++v451.m128i_i32[2];
  if ( v384 )
  {
    if ( (unsigned int)v192 >= v451.m128i_i32[3] )
    {
      sub_16CD150((__int64)&v451, v452, 0, 16, (int)v185, v187);
      v192 = v451.m128i_u32[2];
    }
    v193 = (__int64 **)(v451.m128i_i64[0] + 16 * v192);
    *v193 = v409;
    v193[1] = (__int64 *)v403;
    ++v451.m128i_i32[2];
  }
  v194 = sub_1D252B0(*(_QWORD *)(v9 + 552), 1, 0, 111, 0);
  v196 = *(_QWORD **)(v9 + 552);
  v445 = 0;
  v197 = v194;
  v199 = v198;
  v200 = *(_QWORD *)v9;
  v201 = (__int64 *)v451.m128i_i64[0];
  v202 = v451.m128i_u32[2];
  LODWORD(v446) = *(_DWORD *)(v9 + 536);
  if ( v200 )
  {
    if ( &v445 != (_QWORD **)(v200 + 48) )
    {
      v203 = *(_QWORD *)(v200 + 48);
      v445 = (_QWORD *)v203;
      if ( v203 )
      {
        v410 = v197;
        v423 = v451.m128i_i64[0];
        v429 = v451.m128i_u32[2];
        sub_1623A60((__int64)&v445, v203, 2);
        v197 = v410;
        v201 = (__int64 *)v423;
        v202 = v429;
      }
    }
  }
  v424 = sub_1D23DE0(v196, 23, (__int64)&v445, v197, v199, v195, v201, v202);
  if ( v445 )
    sub_161E7C0((__int64)&v445, (__int64)v445);
  if ( v383 )
  {
    v221 = (__int64 **)a2[28].m128i_i64[1];
    v447[1] = 0;
    v445 = v447;
    v447[0] = v424;
    v446 = 0x800000001LL;
    v222 = &v221[3 * a2[29].m128i_i64[0]];
    if ( v222 == v221 )
    {
      v235 = 1;
    }
    else
    {
      v223 = v221;
      do
      {
        while ( 1 )
        {
          v224 = *v223;
          v225 = sub_20685E0(v9, *v223, v189, a8, a9);
          v228 = v227;
          v229 = v225;
          v230 = (unsigned int)v446;
          if ( (unsigned int)v446 >= HIDWORD(v446) )
          {
            v412 = v225;
            v414 = v228;
            sub_16CD150((__int64)&v445, v447, 0, 16, (int)v225, v228);
            v230 = (unsigned int)v446;
            v229 = v412;
            v228 = v414;
          }
          v231 = (__int64 **)&v445[2 * v230];
          *v231 = v229;
          v231[1] = (__int64 *)v228;
          LODWORD(v446) = v446 + 1;
          if ( *(_BYTE *)(*v224 + 8) == 15 )
            break;
          v223 += 3;
          if ( v222 == v223 )
            goto LABEL_219;
        }
        v229 = sub_1D2AD90(*(_QWORD **)(v9 + 552), (__int64)v224, (__int64)v231, v226, (__int64)v229, v228);
        v228 = v232;
        v233 = (unsigned int)v446;
        if ( (unsigned int)v446 >= HIDWORD(v446) )
        {
          v415 = v232;
          v413 = v229;
          sub_16CD150((__int64)&v445, v447, 0, 16, (int)v229, v232);
          v233 = (unsigned int)v446;
          v229 = v413;
          v228 = v415;
        }
        v223 += 3;
        v234 = (__int64 **)&v445[2 * v233];
        *v234 = v229;
        v234[1] = (__int64 *)v228;
        LODWORD(v446) = v446 + 1;
      }
      while ( v222 != v223 );
LABEL_219:
      v235 = (unsigned int)v446;
      if ( (unsigned int)v446 >= HIDWORD(v446) )
      {
        sub_16CD150((__int64)&v445, v447, 0, 16, (int)v229, v228);
        v235 = (unsigned int)v446;
      }
    }
    v236 = &v445[2 * v235];
    *v236 = v424;
    v236[1] = 1;
    v237 = *(_QWORD *)(v9 + 552);
    LODWORD(v446) = v446 + 1;
    v238 = sub_1D252B0(v237, 1, 0, 111, 0);
    v240 = *(__int64 **)(v9 + 552);
    v441 = 0;
    v241 = (const void ***)v238;
    v243 = v242;
    v244 = *(_QWORD *)v9;
    v245 = (__int64)v445;
    v246 = (unsigned int)v446;
    LODWORD(v442) = *(_DWORD *)(v9 + 536);
    if ( v244 )
    {
      if ( &v441 != (__int64 *)(v244 + 48) )
      {
        v247 = *(_QWORD *)(v244 + 48);
        v441 = v247;
        if ( v247 )
        {
          v425 = v241;
          v433 = (__int64)v445;
          v439 = (unsigned int)v446;
          sub_1623A60((__int64)&v441, v247, 2);
          v241 = v425;
          v245 = v433;
          v246 = v439;
        }
      }
    }
    *((_QWORD *)&v356 + 1) = v246;
    *(_QWORD *)&v356 = v245;
    v248 = sub_1D36D80(
             v240,
             242,
             (__int64)&v441,
             v241,
             v243,
             *(double *)v189.m128i_i64,
             *(double *)a8.m128i_i64,
             a9,
             v239,
             v356);
    if ( v441 )
      sub_161E7C0((__int64)&v441, v441);
    v424 = (__int64)v248;
    if ( v445 != v447 )
      _libc_free((unsigned __int64)v445);
  }
  sub_1D444E0(*(_QWORD *)(v9 + 552), v136, v424);
  sub_1D2DE10(*(_QWORD *)(v9 + 552), v136, v204);
  if ( (_QWORD *)v451.m128i_i64[0] != v452 )
    _libc_free(v451.m128i_u64[0]);
  if ( src != v450 )
    _libc_free((unsigned __int64)src);
  return v396;
}
