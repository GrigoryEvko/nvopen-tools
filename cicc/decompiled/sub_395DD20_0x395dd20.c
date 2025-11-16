// Function: sub_395DD20
// Address: 0x395dd20
//
__int64 __fastcall sub_395DD20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128i a5,
        __m128 a6,
        __m128 a7,
        double a8,
        double a9,
        __m128 a10,
        __m128 a11)
{
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rax
  _QWORD *v15; // rdi
  unsigned __int64 *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rsi
  int *v20; // rax
  int *v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // rax
  int *v25; // r15
  __int64 v26; // rdi
  _QWORD *v27; // r12
  int v28; // eax
  __int64 v29; // rbx
  int v30; // eax
  __int64 *v31; // rsi
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 *v34; // rcx
  __int64 v35; // r8
  int v36; // r8d
  int v37; // r9d
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  __int64 v40; // rax
  _QWORD *v41; // rdi
  int v42; // ecx
  _QWORD **v43; // rbx
  _QWORD *v44; // r12
  unsigned int v45; // edx
  __int64 v46; // rax
  _QWORD *n; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned int v50; // ecx
  __int64 *v51; // rdx
  __int64 v52; // r8
  __int64 v53; // rax
  unsigned __int64 **v54; // r12
  unsigned __int64 **v55; // rbx
  int *v56; // r13
  unsigned __int64 v57; // rsi
  int *v58; // rax
  int *v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rdx
  unsigned __int64 *v62; // rax
  unsigned __int64 v63; // rdx
  int *v64; // rax
  char v65; // r14
  __int64 v66; // rax
  __int64 ii; // r12
  unsigned __int64 v68; // rdi
  unsigned __int64 *v69; // r12
  unsigned __int64 v70; // rbx
  unsigned __int64 v71; // rdi
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  _QWORD *v75; // r8
  __int64 v76; // rcx
  _QWORD *v77; // rax
  __int64 v78; // rsi
  unsigned __int64 v79; // rcx
  __int64 v80; // rdx
  int v81; // esi
  int v82; // r9d
  int v83; // edx
  int v84; // r9d
  _QWORD *v85; // r8
  unsigned int v86; // ecx
  unsigned int v87; // edx
  __int64 v88; // rdx
  unsigned __int64 v89; // rax
  unsigned __int64 v90; // rax
  int v91; // ebx
  unsigned __int64 v92; // r12
  _QWORD *v93; // rax
  _QWORD *m; // rdx
  const __m128i *v95; // r14
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 v98; // r13
  __m128i *v99; // rax
  __m128i *v100; // rdx
  __int64 v101; // rdi
  __int64 v102; // rsi
  int v103; // r8d
  int v104; // r9d
  __m128i *v105; // rax
  __int64 v106; // rbx
  __int64 v107; // r13
  __int64 v108; // rax
  __int64 v109; // rsi
  _QWORD *v110; // rdx
  __int64 v111; // rcx
  _QWORD *v112; // rax
  unsigned __int64 *v113; // r13
  __m128i *v114; // rax
  unsigned __int64 v115; // rcx
  __m128i *v116; // rdx
  __int64 v117; // rdi
  __int64 v118; // rsi
  unsigned __int64 v119; // r14
  __int64 v120; // rbx
  __int64 i; // r8
  const __m128i *v122; // r9
  __m128 *v123; // rax
  unsigned __int64 v124; // r12
  unsigned __int64 v125; // rdi
  int v126; // r8d
  int v127; // r9d
  const __m128i *v128; // r13
  const __m128i *v129; // r14
  __int64 v130; // rax
  __m128 *v131; // rax
  unsigned __int64 v132; // rdi
  unsigned __int64 v133; // r13
  unsigned __int64 *v134; // rcx
  const __m128i *v135; // rdx
  __int64 v136; // rax
  const __m128i *v137; // r14
  __m128i *v138; // rax
  __int64 v139; // r14
  unsigned __int64 *v140; // r13
  __int64 v141; // rsi
  int v142; // r15d
  __int64 v143; // rax
  __m128i *v144; // rdx
  unsigned __int64 *v145; // rax
  int v146; // edx
  int v147; // ecx
  int v148; // r9d
  __int64 v149; // rax
  int v150; // r8d
  int v151; // r9d
  __m128i *v152; // rax
  __m128i *v153; // r12
  __int64 v154; // rax
  char *v155; // rcx
  __m128i *v156; // rax
  __m128i *v157; // rdx
  char v158; // di
  __int64 v159; // rax
  __int64 v160; // r14
  int v161; // edx
  __int64 v162; // rax
  __int64 v163; // r13
  __int64 v164; // rsi
  bool v165; // al
  __int64 v166; // rax
  __m128 *v167; // rdx
  __int64 v168; // rax
  __int64 *v169; // rax
  __int64 v170; // rsi
  unsigned __int64 v171; // rcx
  __int64 v172; // rax
  __int64 v173; // rax
  __m128 *v174; // rax
  __int64 *v175; // r15
  unsigned int v176; // r12d
  const __m128i *v177; // rax
  __int64 v178; // r15
  const __m128i *v179; // rbx
  int v180; // r8d
  int v181; // r9d
  __int64 v182; // rdx
  __m128 *v183; // rdx
  const __m128i *v184; // rbx
  const __m128i *v185; // rsi
  const __m128i *v186; // rdi
  int v187; // edx
  __int64 v188; // rax
  __int64 v189; // rdx
  const __m128i *v190; // rax
  const __m128i *v191; // r14
  __int64 v192; // rax
  int v193; // r8d
  int v194; // r9d
  __m128i *v195; // r12
  __m128 *v196; // rcx
  signed __int64 v197; // r12
  __int64 v198; // rax
  __int64 v199; // rdx
  int v200; // r8d
  int v201; // r9d
  __int64 v202; // rax
  __int8 *v203; // rax
  __int64 v204; // r13
  unsigned __int32 v205; // r14d
  __int64 v206; // rax
  __int64 v207; // rax
  __int64 v208; // r14
  unsigned __int64 v209; // r14
  unsigned __int64 v210; // r15
  unsigned __int64 v211; // rdi
  unsigned __int64 v212; // rdi
  int v213; // esi
  __int64 v214; // rcx
  unsigned int v215; // edx
  __int64 *v216; // rax
  _BYTE *v217; // r9
  int v218; // r8d
  __int64 *v219; // rax
  __int64 v220; // rdi
  __int64 *v221; // r15
  __int64 v222; // r11
  _QWORD *v223; // rax
  __int64 v224; // rcx
  __int64 v225; // rbx
  int v226; // ecx
  int v227; // edx
  __int64 v228; // r8
  __m128i *v229; // r14
  __m128i *v230; // r12
  __int64 v231; // rax
  __int64 v232; // rcx
  __int64 v233; // rax
  int v234; // r8d
  int v235; // r9d
  int v236; // edx
  __int64 v237; // rcx
  __int8 *v238; // rax
  __int32 v239; // edx
  __m128i *v240; // r14
  __m128i *v241; // r12
  __int64 v242; // rax
  __int64 v243; // rcx
  __int64 v244; // rax
  __int64 v245; // r13
  __int64 v246; // rax
  __int64 v247; // rax
  __m128i *v248; // r14
  __m128i *v249; // r12
  __int64 v250; // rax
  __int64 v251; // rcx
  __int64 v252; // r14
  __m128i *v253; // rax
  __int64 v254; // rcx
  __int64 v255; // r8
  int v256; // r9d
  char v257; // di
  __int64 v258; // r13
  int v259; // edx
  _QWORD *v260; // rdi
  int v261; // ecx
  int v262; // r13d
  _QWORD *v263; // rdi
  __int64 v264; // rsi
  int v265; // edi
  _QWORD *v266; // rsi
  __int64 v267; // rax
  __m128i *v268; // rax
  int v269; // ecx
  __int64 *v270; // rdx
  int v271; // edx
  int v272; // edx
  int j; // r14d
  __int64 *v274; // rcx
  __int64 v275; // r8
  int v276; // r8d
  __int64 *v277; // rcx
  int k; // edx
  __int64 v279; // r9
  int v280; // r8d
  __int64 *v281; // rdi
  int v282; // edx
  int v283; // r14d
  int v284; // edx
  __m128i *v285; // r14
  __m128i *v286; // r12
  __int64 v287; // rax
  __int64 v288; // rcx
  int v289; // [rsp-10h] [rbp-8C0h]
  int v290; // [rsp-8h] [rbp-8B8h]
  _QWORD *v291; // [rsp+8h] [rbp-8A8h]
  __int64 v292; // [rsp+40h] [rbp-870h]
  char *v293; // [rsp+48h] [rbp-868h]
  char *v294; // [rsp+48h] [rbp-868h]
  __int64 v295; // [rsp+48h] [rbp-868h]
  __m128i *v296; // [rsp+48h] [rbp-868h]
  _QWORD *v297; // [rsp+50h] [rbp-860h]
  const __m128i *v298; // [rsp+50h] [rbp-860h]
  __int64 v299; // [rsp+50h] [rbp-860h]
  _QWORD *v300; // [rsp+58h] [rbp-858h]
  __int64 v301; // [rsp+58h] [rbp-858h]
  __int64 v302; // [rsp+58h] [rbp-858h]
  __int8 *v303; // [rsp+58h] [rbp-858h]
  __int8 *v304; // [rsp+58h] [rbp-858h]
  unsigned __int8 v305; // [rsp+6Eh] [rbp-842h]
  unsigned __int8 v306; // [rsp+6Fh] [rbp-841h]
  const __m128i *v307; // [rsp+70h] [rbp-840h]
  _QWORD *v308; // [rsp+70h] [rbp-840h]
  _QWORD *v309; // [rsp+88h] [rbp-828h]
  _QWORD **v310; // [rsp+90h] [rbp-820h]
  unsigned __int64 *v312; // [rsp+A8h] [rbp-808h]
  _QWORD *v313; // [rsp+A8h] [rbp-808h]
  unsigned __int64 v314; // [rsp+A8h] [rbp-808h]
  __int8 *v316; // [rsp+B8h] [rbp-7F8h]
  __int64 v317; // [rsp+C0h] [rbp-7F0h]
  __int64 *v318; // [rsp+C8h] [rbp-7E8h]
  int *v320; // [rsp+D8h] [rbp-7D8h]
  __int64 v321; // [rsp+E0h] [rbp-7D0h]
  __m128i v322; // [rsp+F0h] [rbp-7C0h] BYREF
  __int64 v323; // [rsp+100h] [rbp-7B0h]
  unsigned int v324; // [rsp+114h] [rbp-79Ch] BYREF
  __int64 v325; // [rsp+118h] [rbp-798h] BYREF
  __m128 v326; // [rsp+120h] [rbp-790h] BYREF
  __int64 v327; // [rsp+130h] [rbp-780h]
  __int64 v328[2]; // [rsp+140h] [rbp-770h] BYREF
  _QWORD v329[2]; // [rsp+150h] [rbp-760h] BYREF
  __int64 v330; // [rsp+160h] [rbp-750h] BYREF
  _QWORD *v331; // [rsp+168h] [rbp-748h]
  __int64 v332; // [rsp+170h] [rbp-740h]
  __int64 v333; // [rsp+178h] [rbp-738h]
  __int64 v334; // [rsp+180h] [rbp-730h] BYREF
  unsigned __int64 v335; // [rsp+188h] [rbp-728h]
  __int64 v336; // [rsp+190h] [rbp-720h]
  __int64 v337; // [rsp+198h] [rbp-718h]
  __int64 v338; // [rsp+1A0h] [rbp-710h] BYREF
  int v339; // [rsp+1A8h] [rbp-708h] BYREF
  unsigned __int64 v340; // [rsp+1B0h] [rbp-700h]
  int *v341; // [rsp+1B8h] [rbp-6F8h]
  int *v342; // [rsp+1C0h] [rbp-6F0h]
  __int64 v343; // [rsp+1C8h] [rbp-6E8h]
  int v344; // [rsp+1D8h] [rbp-6D8h] BYREF
  int *v345; // [rsp+1E0h] [rbp-6D0h]
  int *v346; // [rsp+1E8h] [rbp-6C8h]
  int *v347; // [rsp+1F0h] [rbp-6C0h]
  __int64 v348; // [rsp+1F8h] [rbp-6B8h]
  char *v349; // [rsp+200h] [rbp-6B0h] BYREF
  __int64 v350; // [rsp+208h] [rbp-6A8h]
  _BYTE v351[32]; // [rsp+210h] [rbp-6A0h] BYREF
  __m128i *v352; // [rsp+230h] [rbp-680h] BYREF
  __int64 v353; // [rsp+238h] [rbp-678h]
  _BYTE v354[32]; // [rsp+240h] [rbp-670h] BYREF
  char *v355; // [rsp+260h] [rbp-650h] BYREF
  __int64 v356; // [rsp+268h] [rbp-648h]
  _BYTE v357[32]; // [rsp+270h] [rbp-640h] BYREF
  __int64 *v358; // [rsp+290h] [rbp-620h] BYREF
  __int64 v359; // [rsp+298h] [rbp-618h]
  _BYTE v360[64]; // [rsp+2A0h] [rbp-610h] BYREF
  __int64 v361; // [rsp+2E0h] [rbp-5D0h] BYREF
  __int64 v362; // [rsp+2E8h] [rbp-5C8h]
  unsigned __int64 v363; // [rsp+2F0h] [rbp-5C0h]
  __int64 v364; // [rsp+2F8h] [rbp-5B8h]
  __int64 v365; // [rsp+300h] [rbp-5B0h]
  unsigned __int64 *v366; // [rsp+308h] [rbp-5A8h]
  unsigned __int64 *v367; // [rsp+310h] [rbp-5A0h]
  _QWORD *v368; // [rsp+318h] [rbp-598h]
  __int64 v369; // [rsp+320h] [rbp-590h]
  __int64 *v370; // [rsp+328h] [rbp-588h]
  __m128i *v371; // [rsp+330h] [rbp-580h] BYREF
  __m128i v372; // [rsp+338h] [rbp-578h] BYREF
  __int64 v373; // [rsp+348h] [rbp-568h]
  __m128i v374; // [rsp+3A0h] [rbp-510h] BYREF
  unsigned __int64 v375; // [rsp+3B0h] [rbp-500h] BYREF
  __int64 *v376; // [rsp+3B8h] [rbp-4F8h] BYREF
  __int64 *v377; // [rsp+3C0h] [rbp-4F0h]
  __int64 v378; // [rsp+3C8h] [rbp-4E8h]
  char *v379; // [rsp+3D8h] [rbp-4D8h] BYREF
  __int64 v380; // [rsp+3E0h] [rbp-4D0h]
  _BYTE v381[32]; // [rsp+3E8h] [rbp-4C8h] BYREF
  __int64 v382; // [rsp+408h] [rbp-4A8h]
  int v383; // [rsp+410h] [rbp-4A0h]
  const __m128i *v384; // [rsp+420h] [rbp-490h] BYREF
  __int64 v385; // [rsp+428h] [rbp-488h]
  _BYTE v386[192]; // [rsp+430h] [rbp-480h] BYREF
  const __m128i *v387; // [rsp+4F0h] [rbp-3C0h] BYREF
  __int64 v388; // [rsp+4F8h] [rbp-3B8h]
  _BYTE v389[192]; // [rsp+500h] [rbp-3B0h] BYREF
  unsigned __int64 *v390; // [rsp+5C0h] [rbp-2F0h] BYREF
  __int64 v391; // [rsp+5C8h] [rbp-2E8h]
  _BYTE v392[192]; // [rsp+5D0h] [rbp-2E0h] BYREF
  const __m128i *v393; // [rsp+690h] [rbp-220h] BYREF
  __int64 v394; // [rsp+698h] [rbp-218h]
  _BYTE v395[528]; // [rsp+6A0h] [rbp-210h] BYREF

  v12 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v330 = 0;
  v317 = v12;
  v384 = (const __m128i *)v386;
  v387 = (const __m128i *)v389;
  v385 = 0x800000000LL;
  v388 = 0x800000000LL;
  v358 = (__int64 *)v360;
  v359 = 0x800000000LL;
  v341 = &v339;
  v342 = &v339;
  v331 = 0;
  v332 = 0;
  v333 = 0;
  v334 = 0;
  v335 = 0;
  v336 = 0;
  v337 = 0;
  v339 = 0;
  v340 = 0;
  v343 = 0;
  v344 = 0;
  v345 = 0;
  v346 = &v344;
  v347 = &v344;
  v348 = 0;
  v361 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  v369 = 0;
  v370 = 0;
  v362 = 8;
  v361 = sub_22077B0(0x40u);
  v13 = (__int64 *)(((4 * v362 - 4) & 0xFFFFFFFFFFFFFFF8LL) + v361);
  v14 = sub_22077B0(0x200u);
  v366 = (unsigned __int64 *)v13;
  v15 = (_QWORD *)v14;
  *v13 = v14;
  v364 = v14;
  v365 = v14 + 512;
  v370 = v13;
  v16 = *(unsigned __int64 **)a3;
  v369 = v14 + 512;
  v17 = *(unsigned int *)(a3 + 8);
  v368 = v15;
  v18 = &v16[v17];
  v363 = (unsigned __int64)v15;
  v367 = v15;
  if ( v16 == v18 )
  {
    v306 = 0;
    goto LABEL_69;
  }
  do
  {
    v19 = *v16;
    v20 = v345;
    v393 = (const __m128i *)*v16;
    if ( !v345 )
      goto LABEL_9;
    v21 = &v344;
    do
    {
      while ( 1 )
      {
        v22 = *((_QWORD *)v20 + 2);
        v23 = *((_QWORD *)v20 + 3);
        if ( *((_QWORD *)v20 + 4) >= v19 )
          break;
        v20 = (int *)*((_QWORD *)v20 + 3);
        if ( !v23 )
          goto LABEL_7;
      }
      v21 = v20;
      v20 = (int *)*((_QWORD *)v20 + 2);
    }
    while ( v22 );
LABEL_7:
    if ( v21 == &v344 || *((_QWORD *)v21 + 4) > v19 )
    {
LABEL_9:
      if ( v15 == (_QWORD *)(v369 - 8) )
      {
        sub_1B4ECC0(&v361, &v393);
        v15 = v367;
      }
      else
      {
        if ( v15 )
        {
          *v15 = v19;
          v15 = v367;
        }
        v367 = ++v15;
      }
    }
    ++v16;
  }
  while ( v18 != v16 );
  v24 = (__int64 *)v363;
  v306 = 0;
  if ( v15 == (_QWORD *)v363 )
    goto LABEL_69;
  v25 = &v344;
  do
  {
    v321 = *v24;
    if ( v24 == (__int64 *)(v365 - 8) )
    {
      j_j___libc_free_0(v364);
      v80 = *++v366 + 512;
      v364 = *v366;
      v365 = v80;
      v363 = v364;
    }
    else
    {
      v363 = (unsigned __int64)(v24 + 1);
    }
    v26 = *(_QWORD *)(v321 + 40);
    if ( v321 + 40 == (v26 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_34;
    v320 = v25;
    v27 = (_QWORD *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
    while ( 2 )
    {
      if ( !v27 )
        BUG();
      v28 = *((unsigned __int8 *)v27 - 8);
      if ( (unsigned int)(v28 - 35) > 0x11 || (((_BYTE)v28 - 35) & 0xFD) != 0 )
        goto LABEL_20;
      v29 = (__int64)(v27 - 3);
      if ( (_DWORD)v333 )
      {
        v30 = (v333 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v31 = &v331[v30];
        v32 = *v31;
        if ( v29 == *v31 )
        {
LABEL_26:
          if ( &v331[(unsigned int)v333] != v31 )
            goto LABEL_20;
        }
        else
        {
          v81 = 1;
          while ( v32 != -8 )
          {
            v82 = v81 + 1;
            v30 = (v333 - 1) & (v81 + v30);
            v31 = &v331[v30];
            v32 = *v31;
            if ( v29 == *v31 )
              goto LABEL_26;
            v81 = v82;
          }
        }
      }
      if ( (_DWORD)v337 )
      {
        v33 = (v337 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v34 = (__int64 *)(v335 + 8LL * v33);
        v35 = *v34;
        if ( v29 == *v34 )
        {
LABEL_29:
          if ( (__int64 *)(v335 + 8LL * (unsigned int)v337) != v34 )
            goto LABEL_20;
        }
        else
        {
          v147 = 1;
          while ( v35 != -8 )
          {
            v148 = v147 + 1;
            v33 = (v337 - 1) & (v147 + v33);
            v34 = (__int64 *)(v335 + 8LL * v33);
            v35 = *v34;
            if ( v29 == *v34 )
              goto LABEL_29;
            v147 = v148;
          }
        }
      }
      v324 = 0;
      v305 = sub_395C630(
               v317,
               v27 - 3,
               (__int64)(v27 - 3),
               (__int64)&v384,
               (__int64)&v387,
               (unsigned int *)&v358,
               &v324);
      if ( !v305 )
        goto LABEL_32;
      v40 = (unsigned int)v385;
      if ( (unsigned int)v385 <= 1 )
        goto LABEL_32;
      v310 = *(_QWORD ***)(a1 + 40);
      v309 = *v310;
      if ( !v317 )
        goto LABEL_209;
      v95 = v384;
      v374.m128i_i32[2] = 0;
      v390 = (unsigned __int64 *)v392;
      v307 = (const __m128i *)((char *)v384 + 24 * (unsigned int)v385);
      v393 = (const __m128i *)v395;
      v394 = 0x800000000LL;
      v391 = 0x800000000LL;
      v375 = 0;
      v376 = &v374.m128i_i64[1];
      v377 = &v374.m128i_i64[1];
      v378 = 0;
      v300 = v27;
      v297 = v27 - 3;
      do
      {
        v96 = v95->m128i_i64[0];
        if ( (unsigned int)*(unsigned __int8 *)(v95->m128i_i64[0] + 16) - 35 > 0x11 )
          BUG();
        v97 = *(_QWORD *)(v96 - 48);
        v98 = *(_QWORD *)(v96 - 24);
        v349 = 0;
        v352 = 0;
        sub_395D8D0(v317, v97, (__int64 *)&v349, (__int64 *)&v352, &v338);
        if ( v349 )
        {
          v99 = (__m128i *)v375;
          if ( v375 )
          {
            v100 = (__m128i *)&v374.m128i_u64[1];
            do
            {
              while ( 1 )
              {
                v101 = v99[1].m128i_i64[0];
                v102 = v99[1].m128i_i64[1];
                if ( v99[2].m128i_i64[0] >= (unsigned __int64)v349 )
                  break;
                v99 = (__m128i *)v99[1].m128i_i64[1];
                if ( !v102 )
                  goto LABEL_159;
              }
              v100 = v99;
              v99 = (__m128i *)v99[1].m128i_i64[0];
            }
            while ( v101 );
LABEL_159:
            if ( v100 != (__m128i *)&v374.m128i_u64[1] && v100[2].m128i_i64[0] <= (unsigned __int64)v349 )
              goto LABEL_170;
          }
        }
        v355 = 0;
        v371 = 0;
        sub_395D8D0(v317, v98, (__int64 *)&v355, (__int64 *)&v371, &v338);
        if ( v355 )
        {
          v349 = v355;
          v352 = v371;
          v105 = (__m128i *)v375;
          if ( v375 )
          {
            v100 = (__m128i *)&v374.m128i_u64[1];
            do
            {
              if ( v105[2].m128i_i64[0] < (unsigned __int64)v355 )
              {
                v105 = (__m128i *)v105[1].m128i_i64[1];
              }
              else
              {
                v100 = v105;
                v105 = (__m128i *)v105[1].m128i_i64[0];
              }
            }
            while ( v105 );
            if ( v100 != (__m128i *)&v374.m128i_u64[1] && v100[2].m128i_i64[0] <= (unsigned __int64)v355 )
            {
LABEL_170:
              v106 = v100[2].m128i_i64[1];
              v107 = v106 + 8;
LABEL_171:
              v372 = _mm_loadu_si128(v95);
              v293 = (char *)v352;
              v373 = v95[1].m128i_i64[0];
              v108 = sub_22077B0(0x40u);
              a10 = (__m128)_mm_loadu_si128(&v372);
              v109 = v108;
              *(_QWORD *)(v108 + 32) = v293;
              *(__m128 *)(v108 + 40) = a10;
              *(_QWORD *)(v108 + 56) = v373;
              v110 = *(_QWORD **)(v106 + 16);
              if ( v110 )
              {
                while ( 1 )
                {
                  v111 = v110[4];
                  v112 = (_QWORD *)v110[3];
                  if ( (__int64)v293 < v111 )
                    v112 = (_QWORD *)v110[2];
                  if ( !v112 )
                    break;
                  v110 = v112;
                }
                v257 = 1;
                if ( (_QWORD *)v107 != v110 )
                  v257 = (__int64)v293 < v111;
              }
              else
              {
                v110 = (_QWORD *)v107;
                v257 = 1;
              }
              sub_220F040(v257, v109, v110, (_QWORD *)v107);
              ++*(_QWORD *)(v106 + 40);
              goto LABEL_180;
            }
          }
LABEL_236:
          v149 = sub_22077B0(0x30u);
          v106 = v149;
          if ( v149 )
          {
            v107 = v149 + 8;
            *(_DWORD *)(v149 + 8) = 0;
            *(_QWORD *)(v149 + 16) = 0;
            *(_QWORD *)(v149 + 24) = v149 + 8;
            *(_QWORD *)(v149 + 32) = v149 + 8;
            *(_QWORD *)(v149 + 40) = 0;
          }
          else
          {
            v107 = 8;
          }
          v152 = (__m128i *)v375;
          if ( v375 )
          {
            v153 = (__m128i *)&v374.m128i_u64[1];
            do
            {
              if ( v152[2].m128i_i64[0] < (unsigned __int64)v349 )
              {
                v152 = (__m128i *)v152[1].m128i_i64[1];
              }
              else
              {
                v153 = v152;
                v152 = (__m128i *)v152[1].m128i_i64[0];
              }
            }
            while ( v152 );
            if ( v153 == (__m128i *)&v374.m128i_u64[1] || v153[2].m128i_i64[0] > (unsigned __int64)v349 )
            {
LABEL_246:
              v292 = (__int64)v153;
              v154 = sub_22077B0(0x30u);
              v155 = v349;
              *(_QWORD *)(v154 + 40) = 0;
              v153 = (__m128i *)v154;
              *(_QWORD *)(v154 + 32) = v155;
              v294 = v155;
              v156 = (__m128i *)sub_395DC20(&v374, v292, (unsigned __int64 *)(v154 + 32));
              if ( v157 )
              {
                v158 = v156
                    || &v374.m128i_u64[1] == (unsigned __int64 *)v157
                    || (unsigned __int64)v294 < v157[2].m128i_i64[0];
                sub_220F040(v158, (__int64)v153, v157, &v374.m128i_i64[1]);
                ++v378;
              }
              else
              {
                v296 = v156;
                j_j___libc_free_0((unsigned __int64)v153);
                v153 = v296;
              }
            }
            v153[2].m128i_i64[1] = v106;
            v159 = (unsigned int)v391;
            if ( (unsigned int)v391 >= HIDWORD(v391) )
            {
              sub_16CD150((__int64)&v390, v392, 0, 8, v150, v151);
              v159 = (unsigned int)v391;
            }
            v390[v159] = (unsigned __int64)v349;
            LODWORD(v391) = v391 + 1;
            goto LABEL_171;
          }
          v153 = (__m128i *)&v374.m128i_u64[1];
          goto LABEL_246;
        }
        if ( v349 )
          goto LABEL_236;
        v267 = (unsigned int)v394;
        if ( (unsigned int)v394 >= HIDWORD(v394) )
        {
          sub_16CD150((__int64)&v393, v395, 0, 24, v103, v104);
          v267 = (unsigned int)v394;
        }
        v268 = (__m128i *)((char *)v393 + 24 * v267);
        *v268 = _mm_loadu_si128(v95);
        v268[1].m128i_i64[0] = v95[1].m128i_i64[0];
        LODWORD(v394) = v394 + 1;
LABEL_180:
        v95 = (const __m128i *)((char *)v95 + 24);
      }
      while ( v307 != v95 );
      v113 = v390;
      LODWORD(v385) = 0;
      v27 = v300;
      v29 = (__int64)v297;
      v312 = &v390[(unsigned int)v391];
      if ( v390 != v312 )
      {
        v308 = v300;
        v301 = (__int64)v297;
        do
        {
          v114 = (__m128i *)v375;
          v115 = *v113;
          v116 = (__m128i *)&v374.m128i_u64[1];
          if ( v375 )
          {
            do
            {
              while ( 1 )
              {
                v117 = v114[1].m128i_i64[0];
                v118 = v114[1].m128i_i64[1];
                if ( v114[2].m128i_i64[0] >= v115 )
                  break;
                v114 = (__m128i *)v114[1].m128i_i64[1];
                if ( !v118 )
                  goto LABEL_188;
              }
              v116 = v114;
              v114 = (__m128i *)v114[1].m128i_i64[0];
            }
            while ( v117 );
LABEL_188:
            if ( v116 != (__m128i *)&v374.m128i_u64[1] && v116[2].m128i_i64[0] > v115 )
              v116 = (__m128i *)&v374.m128i_u64[1];
          }
          v119 = v116[2].m128i_u64[1];
          v120 = (unsigned int)v385;
          for ( i = *(_QWORD *)(v119 + 24); v119 + 8 != i; i = sub_220EEE0(i) )
          {
            v122 = (const __m128i *)(i + 40);
            if ( HIDWORD(v385) <= (unsigned int)v120 )
            {
              v295 = i;
              v298 = (const __m128i *)(i + 40);
              sub_16CD150((__int64)&v384, v386, 0, 24, i, (int)v122);
              v120 = (unsigned int)v385;
              i = v295;
              v122 = v298;
            }
            a4 = (__m128)_mm_loadu_si128(v122);
            v123 = (__m128 *)((char *)v384 + 24 * v120);
            *v123 = a4;
            v123[1].m128_u64[0] = v122[1].m128i_u64[0];
            v120 = (unsigned int)(v385 + 1);
            LODWORD(v385) = v385 + 1;
          }
          v124 = *(_QWORD *)(v119 + 16);
          while ( v124 )
          {
            sub_3959C60(*(_QWORD *)(v124 + 24));
            v125 = v124;
            v124 = *(_QWORD *)(v124 + 16);
            j_j___libc_free_0(v125);
          }
          ++v113;
          j_j___libc_free_0(v119);
        }
        while ( v312 != v113 );
        v27 = v308;
        v29 = v301;
      }
      sub_3959E30(v375);
      v128 = v393;
      v375 = 0;
      v376 = &v374.m128i_i64[1];
      v377 = &v374.m128i_i64[1];
      v378 = 0;
      v129 = (const __m128i *)((char *)v393 + 24 * (unsigned int)v394);
      v130 = (unsigned int)v385;
      LODWORD(v391) = 0;
      if ( v393 == v129 )
      {
        v132 = 0;
      }
      else
      {
        do
        {
          if ( (unsigned int)v130 >= HIDWORD(v385) )
          {
            sub_16CD150((__int64)&v384, v386, 0, 24, v126, v127);
            v130 = (unsigned int)v385;
          }
          a11 = (__m128)_mm_loadu_si128(v128);
          v128 = (const __m128i *)((char *)v128 + 24);
          v131 = (__m128 *)((char *)v384 + 24 * v130);
          *v131 = a11;
          v131[1].m128_u64[0] = v128[-1].m128i_u64[1];
          v130 = (unsigned int)(v385 + 1);
          LODWORD(v385) = v385 + 1;
        }
        while ( v129 != v128 );
        v132 = v375;
      }
      sub_3959E30(v132);
      if ( v390 != (unsigned __int64 *)v392 )
        _libc_free((unsigned __int64)v390);
      if ( v393 != (const __m128i *)v395 )
        _libc_free((unsigned __int64)v393);
      v40 = (unsigned int)v385;
LABEL_209:
      v133 = (unsigned __int64)v384;
      v134 = (unsigned __int64 *)v392;
      v352 = (__m128i *)v354;
      v135 = (const __m128i *)((char *)v384 + 24 * v40);
      v136 = 0;
      v137 = (const __m128i *)((char *)v135 - 24);
      v371 = (__m128i *)&v372.m128i_u64[1];
      v349 = v351;
      v350 = 0x400000000LL;
      v353 = 0x400000000LL;
      v372.m128i_i64[0] = 0x400000000LL;
      v393 = (const __m128i *)v395;
      v394 = 0x400000000LL;
      v356 = 0x400000000LL;
      v355 = v357;
      v390 = (unsigned __int64 *)v392;
      v391 = 0x800000000LL;
      if ( v384 == v135 )
      {
        LODWORD(v385) = 0;
        v258 = 0;
        goto LABEL_321;
      }
      while ( 1 )
      {
        v38 = _mm_loadu_si128(v137);
        v138 = (__m128i *)&v134[3 * v136];
        *v138 = v38;
        v138[1].m128i_i64[0] = v137[1].m128i_i64[0];
        v136 = (unsigned int)(v391 + 1);
        LODWORD(v391) = v391 + 1;
        if ( (const __m128i *)v133 == v137 )
          break;
        v137 = (const __m128i *)((char *)v137 - 24);
        if ( HIDWORD(v391) <= (unsigned int)v136 )
        {
          sub_16CD150((__int64)&v390, v392, 0, 24, v36, v37);
          v136 = (unsigned int)v391;
        }
        v134 = v390;
      }
      LODWORD(v385) = 0;
      if ( (_DWORD)v136 )
      {
        v313 = v27;
        while ( 1 )
        {
          v139 = (__int64)v390;
          v326.m128_i32[0] = 0;
          LODWORD(v328[0]) = 0;
          LODWORD(v350) = 0;
          v140 = &v390[3 * v136];
          LODWORD(v353) = 0;
          v372.m128i_i32[0] = 0;
LABEL_218:
          v141 = *(v140 - 3);
          v142 = *((_DWORD *)v140 - 3);
          if ( *(_BYTE *)(v141 + 16) == 39 )
          {
LABEL_219:
            if ( sub_395A9E0(v317, v141, (__int64)&v349, (int *)&v326, 8u, (__int64)&v352, (int *)v328) )
            {
              v143 = v372.m128i_u32[0];
              if ( v372.m128i_i32[0] >= (unsigned __int32)v372.m128i_i32[1] )
              {
                sub_16CD150((__int64)&v371, &v372.m128i_u64[1], 0, 24, v36, v37);
                v143 = v372.m128i_u32[0];
              }
              v39 = _mm_loadu_si128((const __m128i *)(v140 - 3));
              v144 = (__m128i *)((char *)v371 + 24 * v143);
              v145 = v140 - 3;
              *v144 = v39;
              v144[1].m128i_i64[0] = *(v140 - 1);
              ++v372.m128i_i32[0];
              goto LABEL_223;
            }
          }
LABEL_229:
          v145 = v140 - 3;
LABEL_223:
          v146 = v350;
          while ( 1 )
          {
            v140 = v145;
            if ( (unsigned __int64 *)v139 == v145 || (unsigned int)v350 > 3 )
              break;
            if ( v142 == 29 )
              goto LABEL_218;
            v145 -= 3;
            if ( *((_DWORD *)v145 + 3) == v142 )
            {
              v141 = *(v140 - 3);
              v142 = *((_DWORD *)v140 - 3);
              if ( *(_BYTE *)(v141 + 16) == 39 )
                goto LABEL_219;
              goto LABEL_229;
            }
          }
          if ( v326.m128_i32[0] )
          {
            v37 = v328[0];
            if ( LODWORD(v328[0]) )
            {
              if ( (_DWORD)v350 == 3 )
              {
                v244 = sub_1644900(*v310, 8u);
                v245 = sub_159C470(v244, 0, 0);
                v246 = (unsigned int)v350;
                if ( (unsigned int)v350 >= HIDWORD(v350) )
                {
                  sub_16CD150((__int64)&v349, v351, 0, 8, v36, v37);
                  v246 = (unsigned int)v350;
                }
                *(_QWORD *)&v349[8 * v246] = v245;
                v247 = (unsigned int)v353;
                LODWORD(v350) = v350 + 1;
                if ( (unsigned int)v353 >= HIDWORD(v353) )
                {
                  sub_16CD150((__int64)&v352, v354, 0, 8, v36, v37);
                  v247 = (unsigned int)v353;
                }
                v352->m128i_i64[v247] = v245;
                v146 = v350;
                LODWORD(v353) = v353 + 1;
              }
              if ( v146 == 4 )
              {
                if ( !byte_5054A00 )
                  goto LABEL_357;
                v240 = v371;
                LODWORD(v356) = 0;
                v241 = (__m128i *)((char *)v371 + 24 * v372.m128i_u32[0]);
                v242 = 0;
                if ( v371 != v241 )
                {
                  do
                  {
                    if ( HIDWORD(v356) <= (unsigned int)v242 )
                    {
                      sub_16CD150((__int64)&v355, v357, 0, 8, v36, v37);
                      v242 = (unsigned int)v356;
                    }
                    v243 = v240->m128i_i64[0];
                    v240 = (__m128i *)((char *)v240 + 24);
                    *(_QWORD *)&v355[8 * v242] = v243;
                    v242 = (unsigned int)(v356 + 1);
                    LODWORD(v356) = v356 + 1;
                  }
                  while ( v241 != v240 );
                }
                v233 = sub_3958EE0(a2, (__int64 *)&v355, 0);
                if ( !v233 )
                {
LABEL_357:
                  v229 = v371;
                  LODWORD(v356) = 0;
                  v230 = (__m128i *)((char *)v371 + 24 * v372.m128i_u32[0]);
                  if ( v371 != v230 )
                  {
                    v231 = 0;
                    do
                    {
                      if ( HIDWORD(v356) <= (unsigned int)v231 )
                      {
                        sub_16CD150((__int64)&v355, v357, 0, 8, v36, v37);
                        v231 = (unsigned int)v356;
                      }
                      v232 = v229[1].m128i_i64[0];
                      v229 = (__m128i *)((char *)v229 + 24);
                      *(_QWORD *)&v355[8 * v231] = v232;
                      v231 = (unsigned int)(v356 + 1);
                      LODWORD(v356) = v356 + 1;
                    }
                    while ( v230 != v229 );
                  }
LABEL_362:
                  v233 = sub_3958EE0(a2, (__int64 *)&v355, v29);
                }
LABEL_363:
                v382 = v233;
                v375 = 0x400000000LL;
                v374.m128i_i64[0] = __PAIR64__(v328[0], v326.m128_u32[0]);
                v380 = 0x400000000LL;
                v383 = v142;
                v374.m128i_i64[1] = (__int64)&v376;
                v379 = v381;
                sub_3959440((__int64)&v374.m128i_i64[1], (char *)&v376, v349, &v349[8 * (unsigned int)v350]);
                sub_3959440((__int64)&v379, v379, v352->m128i_i8, &v352->m128i_i8[8 * (unsigned int)v353]);
                v236 = v394;
                if ( (unsigned int)v394 >= HIDWORD(v394) )
                {
                  sub_395D970((unsigned __int64 *)&v393, 0);
                  v236 = v394;
                }
                v237 = (__int64)v393;
                v238 = &v393->m128i_i8[120 * v236];
                if ( v238 )
                {
                  *(_DWORD *)v238 = v374.m128i_i32[0];
                  v239 = v374.m128i_i32[1];
                  *((_QWORD *)v238 + 2) = 0x400000000LL;
                  *((_DWORD *)v238 + 1) = v239;
                  *((_QWORD *)v238 + 1) = v238 + 24;
                  if ( (_DWORD)v375 )
                  {
                    v304 = v238;
                    sub_39592B0((__int64)(v238 + 8), (__int64)&v374.m128i_i64[1], (unsigned int)v375, v237, v234, v235);
                    v238 = v304;
                  }
                  *((_QWORD *)v238 + 7) = v238 + 72;
                  *((_QWORD *)v238 + 8) = 0x400000000LL;
                  if ( (_DWORD)v380 )
                  {
                    v303 = v238;
                    sub_39592B0((__int64)(v238 + 56), (__int64)&v379, (__int64)(v238 + 72), v237, v234, v235);
                    v238 = v303;
                  }
                  *((_QWORD *)v238 + 13) = v382;
                  *((_DWORD *)v238 + 28) = v383;
                  v236 = v394;
                }
                LODWORD(v394) = v236 + 1;
                sub_3959390((__int64)&v390, (__int64)&v371);
                if ( v379 != v381 )
                  _libc_free((unsigned __int64)v379);
                if ( (__int64 **)v374.m128i_i64[1] != &v376 )
                  _libc_free(v374.m128i_u64[1]);
                goto LABEL_276;
              }
            }
          }
          v160 = (__int64)v390;
          v372.m128i_i32[0] = 0;
          LODWORD(v350) = 0;
          LODWORD(v353) = 0;
          v161 = v391;
          v326.m128_i32[0] = 0;
          v162 = 3LL * (unsigned int)v391;
          LODWORD(v328[0]) = 0;
          v163 = (__int64)&v390[v162];
          if ( v390 != &v390[v162] )
          {
LABEL_259:
            v164 = *(_QWORD *)(v163 - 24);
            v142 = *(_DWORD *)(v163 - 12);
            if ( *(_BYTE *)(v164 + 16) == 39
              && (v165 = sub_395A9E0(v317, v164, (__int64)&v349, (int *)&v326, 0x10u, (__int64)&v352, (int *)v328),
                  v36 = v290,
                  v165) )
            {
              v166 = v372.m128i_u32[0];
              if ( v372.m128i_i32[0] >= (unsigned __int32)v372.m128i_i32[1] )
              {
                sub_16CD150((__int64)&v371, &v372.m128i_u64[1], 0, 24, v290, v37);
                v166 = v372.m128i_u32[0];
              }
              a10 = (__m128)_mm_loadu_si128((const __m128i *)(v163 - 24));
              v167 = (__m128 *)((char *)v371 + 24 * v166);
              v168 = v163 - 24;
              *v167 = a10;
              v167[1].m128_u64[0] = *(_QWORD *)(v163 - 24 + 16);
              ++v372.m128i_i32[0];
            }
            else
            {
              v168 = v163 - 24;
            }
            v163 = v168;
            if ( v160 != v168 )
            {
              while ( (unsigned int)v350 <= 1 )
              {
                if ( v142 == 29 || *(_DWORD *)(v168 - 12) == v142 )
                  goto LABEL_259;
                v168 -= 24;
                v163 = v168;
                if ( v160 == v168 )
                  break;
              }
            }
            if ( (_DWORD)v350 == 2 && v326.m128_i32[0] && LODWORD(v328[0]) )
            {
              if ( !byte_5054A00 )
                goto LABEL_387;
              v285 = v371;
              LODWORD(v356) = 0;
              v286 = (__m128i *)((char *)v371 + 24 * v372.m128i_u32[0]);
              v287 = 0;
              if ( v371 != v286 )
              {
                do
                {
                  if ( HIDWORD(v356) <= (unsigned int)v287 )
                  {
                    sub_16CD150((__int64)&v355, v357, 0, 8, v36, v37);
                    v287 = (unsigned int)v356;
                  }
                  v288 = v285->m128i_i64[0];
                  v285 = (__m128i *)((char *)v285 + 24);
                  *(_QWORD *)&v355[8 * v287] = v288;
                  v287 = (unsigned int)(v356 + 1);
                  LODWORD(v356) = v356 + 1;
                }
                while ( v286 != v285 );
              }
              v233 = sub_3958EE0(a2, (__int64 *)&v355, 0);
              if ( !v233 )
              {
LABEL_387:
                v248 = v371;
                LODWORD(v356) = 0;
                v249 = (__m128i *)((char *)v371 + 24 * v372.m128i_u32[0]);
                if ( v371 != v249 )
                {
                  v250 = 0;
                  do
                  {
                    if ( (unsigned int)v250 >= HIDWORD(v356) )
                    {
                      sub_16CD150((__int64)&v355, v357, 0, 8, v36, v37);
                      v250 = (unsigned int)v356;
                    }
                    v251 = v248[1].m128i_i64[0];
                    v248 = (__m128i *)((char *)v248 + 24);
                    *(_QWORD *)&v355[8 * v250] = v251;
                    v250 = (unsigned int)(v356 + 1);
                    LODWORD(v356) = v356 + 1;
                  }
                  while ( v249 != v248 );
                }
                goto LABEL_362;
              }
              goto LABEL_363;
            }
            v163 = (__int64)v390;
            v161 = v391;
            v162 = 3LL * (unsigned int)v391;
          }
          v169 = (__int64 *)(v163 + v162 * 8 - 24);
          v170 = *v169;
          v171 = v169[2];
          v172 = v169[1];
          LODWORD(v391) = v161 - 1;
          v374.m128i_i64[0] = v170;
          v374.m128i_i64[1] = v172;
          v173 = (unsigned int)v388;
          v375 = v171;
          if ( (unsigned int)v388 >= HIDWORD(v388) )
          {
            sub_16CD150((__int64)&v387, v389, 0, 24, v36, v37);
            v173 = (unsigned int)v388;
          }
          a11 = (__m128)_mm_loadu_si128(&v374);
          v174 = (__m128 *)((char *)v387 + 24 * v173);
          *v174 = a11;
          LODWORD(v388) = v388 + 1;
          v174[1].m128_u64[0] = v375;
LABEL_276:
          v136 = (unsigned int)v391;
          if ( !(_DWORD)v391 )
          {
            v27 = v313;
            break;
          }
        }
      }
      if ( !(_DWORD)v394 )
      {
        v258 = 0;
        goto LABEL_321;
      }
      v314 = (unsigned __int64)v393;
      v316 = &v393->m128i_i8[120 * (unsigned int)v394];
      if ( !byte_5054CA0 )
      {
        v252 = (__int64)v393;
        do
        {
          sub_395A530((__int64)v328, a2, v309, (__int64)&v387, v252, v317);
          if ( *(_DWORD *)(v252 + 16) == 4 )
            sub_395B9D0((__int64)&v374, v317, a2, v29, a1, v252, (__int64)v328);
          else
            sub_395BE60((__int64)&v374, v317, a2, v29, a1, v252, (__int64)v328);
          v252 += 120;
          v253 = (__m128i *)sub_39596E0(
                              (__int64)v387,
                              (__int64)&v387->m128i_i64[3 * (unsigned int)v388],
                              (__int64)&v374,
                              a2);
          sub_395A000((__int64)&v387, v253, &v374, v254, v255, v256);
        }
        while ( v316 != (__int8 *)v252 );
        goto LABEL_312;
      }
      v291 = v27;
      v175 = v328;
      v374.m128i_i64[0] = (__int64)&v375;
      v374.m128i_i64[1] = 0x400000000LL;
      v302 = v29;
      while ( 2 )
      {
        if ( dword_5054BC0 )
        {
          v176 = 0;
          if ( v316 != (__int8 *)v314 )
          {
            v177 = (const __m128i *)v175;
            v178 = (__int64)(v316 - 120);
            v179 = v177;
            while ( 1 )
            {
              v316 = (__int8 *)v178;
              sub_395A530((__int64)&v326, a2, v309, (__int64)&v387, v178, v317);
              if ( *(_DWORD *)(v178 + 16) == 4 )
              {
                sub_395B9D0((__int64)v179, v317, a2, v302, a1, v178, (__int64)&v326);
                v180 = v289;
                v182 = v374.m128i_u32[2];
                v181 = v290;
                if ( v374.m128i_i32[2] < (unsigned __int32)v374.m128i_i32[3] )
                  goto LABEL_285;
              }
              else
              {
                sub_395BE60((__int64)v179, v317, a2, v302, a1, v178, (__int64)&v326);
                v182 = v374.m128i_u32[2];
                if ( v374.m128i_i32[2] < (unsigned __int32)v374.m128i_i32[3] )
                  goto LABEL_285;
              }
              sub_16CD150((__int64)&v374, &v375, 0, 24, v180, v181);
              v182 = v374.m128i_u32[2];
LABEL_285:
              a6 = (__m128)_mm_loadu_si128(v179);
              ++v176;
              v178 -= 120;
              v183 = (__m128 *)(v374.m128i_i64[0] + 24 * v182);
              *v183 = a6;
              v183[1].m128_u64[0] = v179[1].m128i_u64[0];
              ++v374.m128i_i32[2];
              if ( dword_5054BC0 <= v176 || v316 == (__int8 *)v314 )
              {
                v175 = (__int64 *)v179;
                break;
              }
            }
          }
        }
        v184 = (const __m128i *)v374.m128i_i64[0];
        v299 = v374.m128i_i64[0] + 24LL * v374.m128i_u32[2];
        if ( v374.m128i_i64[0] != v299 )
        {
          do
          {
            while ( 1 )
            {
              v191 = v184;
              v192 = sub_39596E0((__int64)v387, (__int64)&v387->m128i_i64[3 * (unsigned int)v388], (__int64)v184, a2);
              v186 = v387;
              v195 = (__m128i *)v192;
              v187 = v388;
              v188 = 24LL * (unsigned int)v388;
              v196 = (__m128 *)&v387->m128i_i8[v188];
              if ( v195 == (__m128i *)&v387->m128i_i8[v188] )
                break;
              if ( (unsigned int)v388 >= (unsigned __int64)HIDWORD(v388) )
              {
                v197 = (char *)v195 - (char *)v387;
                sub_16CD150((__int64)&v387, v389, 0, 24, v193, v194);
                v186 = v387;
                v187 = v388;
                v195 = (__m128i *)((char *)v387 + v197);
                v188 = 24LL * (unsigned int)v388;
                v196 = (__m128 *)&v387->m128i_i8[v188];
              }
              v185 = (const __m128i *)((char *)v186 + v188 - 24);
              if ( v196 )
              {
                a7 = (__m128)_mm_loadu_si128(v185);
                *v196 = a7;
                v186 = v387;
                v196[1].m128_u64[0] = v185[1].m128i_u64[0];
                v187 = v388;
                v188 = 24LL * (unsigned int)v388;
                v185 = (const __m128i *)((char *)v186 + v188 - 24);
              }
              if ( v185 != v195 )
              {
                memmove(&v186->m128i_i8[v188 - ((char *)v185 - (char *)v195)], v195, (char *)v185 - (char *)v195);
                v187 = v388;
              }
              v189 = (unsigned int)(v187 + 1);
              v190 = (const __m128i *)((char *)v184 + 24);
              LODWORD(v388) = v189;
              if ( v195 <= v184 )
              {
                if ( v184 < (const __m128i *)((char *)v387 + 24 * v189) )
                  v184 = (const __m128i *)((char *)v184 + 24);
                v191 = v184;
              }
              a5 = _mm_loadu_si128(v191);
              v184 = v190;
              *v195 = a5;
              v195[1].m128i_i64[0] = v191[1].m128i_i64[0];
              if ( v190 == (const __m128i *)v299 )
                goto LABEL_309;
            }
            if ( (unsigned int)v388 >= HIDWORD(v388) )
            {
              sub_16CD150((__int64)&v387, v389, 0, 24, v193, v194);
              v196 = (__m128 *)((char *)v387 + 24 * (unsigned int)v388);
            }
            a7 = (__m128)_mm_loadu_si128(v184);
            *v196 = a7;
            v198 = v184[1].m128i_i64[0];
            LODWORD(v388) = v388 + 1;
            v196[1].m128_u64[0] = v198;
            v184 = (const __m128i *)((char *)v184 + 24);
          }
          while ( v184 != (const __m128i *)v299 );
        }
LABEL_309:
        v374.m128i_i32[2] = 0;
        if ( v316 != (__int8 *)v314 )
          continue;
        break;
      }
      v27 = v291;
      v29 = v302;
      if ( (unsigned __int64 *)v374.m128i_i64[0] != &v375 )
        _libc_free(v374.m128i_u64[0]);
LABEL_312:
      v199 = (unsigned int)v388;
      if ( (unsigned int)v388 > 1uLL )
      {
        do
        {
          sub_395A2F0((__int64)&v374, a2, v29, (_DWORD *)&v387[-1] + 6 * v199 - 2, &v387[-3].m128i_i32[6 * v199]);
          v202 = (unsigned int)(v388 - 2);
          LODWORD(v388) = v202;
          if ( (unsigned int)v202 >= HIDWORD(v388) )
          {
            sub_16CD150((__int64)&v387, v389, 0, 24, v200, v201);
            v202 = (unsigned int)v388;
          }
          v39 = _mm_loadu_si128(&v374);
          v203 = &v387->m128i_i8[24 * v202];
          *(__m128i *)v203 = v39;
          *((_QWORD *)v203 + 2) = v375;
          v199 = (unsigned int)(v388 + 1);
          LODWORD(v388) = v199;
        }
        while ( (unsigned int)v199 > 1 );
      }
      v38 = _mm_loadu_si128(v387);
      v326 = (__m128)v38;
      v327 = v387[1].m128i_i64[0];
      if ( v38.m128i_i32[3] != 11 )
      {
        v204 = v387[1].m128i_i64[0];
        v205 = v387->m128i_u32[2];
        v206 = sub_1644900(v309, 0x20u);
        v207 = sub_159C470(v206, 0, 0);
        v375 = v204;
        v374.m128i_i64[0] = v207;
        v374.m128i_i64[1] = v205 | 0xB00000000LL;
        sub_395A2F0((__int64)&v322, a2, v29, &v374, &v326);
        a10 = (__m128)_mm_loadu_si128(&v322);
        v327 = v323;
        v326 = a10;
      }
      v258 = v326.m128_u64[0];
      v208 = *(v27 - 3);
      if ( v208 != *(_QWORD *)v326.m128_u64[0] )
      {
        v329[0] = v327;
        v328[0] = (__int64)v329;
        v328[1] = 0x100000001LL;
        v325 = sub_3958EE0(a2, v328, v29);
        v374.m128i_i64[0] = (__int64)"dot.res";
        LOWORD(v375) = 259;
        v326.m128_u64[0] = sub_3958FF0(v326.m128_i64[0], v326.m128_i32[2], v208, &v325, (__int64)&v374);
        v258 = v326.m128_u64[0];
        v327 = v325;
        if ( (_QWORD *)v328[0] != v329 )
        {
          _libc_free(v328[0]);
          v258 = v326.m128_u64[0];
        }
      }
LABEL_321:
      if ( v390 != (unsigned __int64 *)v392 )
        _libc_free((unsigned __int64)v390);
      if ( v355 != v357 )
        _libc_free((unsigned __int64)v355);
      v209 = (unsigned __int64)v393;
      v210 = (unsigned __int64)v393 + 120 * (unsigned int)v394;
      if ( v393 != (const __m128i *)v210 )
      {
        do
        {
          v210 -= 120LL;
          v211 = *(_QWORD *)(v210 + 56);
          if ( v211 != v210 + 72 )
            _libc_free(v211);
          v212 = *(_QWORD *)(v210 + 8);
          if ( v212 != v210 + 24 )
            _libc_free(v212);
        }
        while ( v209 != v210 );
        v210 = (unsigned __int64)v393;
      }
      if ( (_BYTE *)v210 != v395 )
        _libc_free(v210);
      if ( v371 != (__m128i *)&v372.m128i_u64[1] )
        _libc_free((unsigned __int64)v371);
      if ( v352 != (__m128i *)v354 )
        _libc_free((unsigned __int64)v352);
      if ( v349 != v351 )
        _libc_free((unsigned __int64)v349);
      v390 = (unsigned __int64 *)v258;
      if ( v258 )
      {
        sub_164D160(
          v29,
          v258,
          a4,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128_u64,
          *(double *)a7.m128_u64,
          *(double *)v38.m128i_i64,
          *(double *)v39.m128i_i64,
          *(double *)a10.m128_u64,
          a11);
        v213 = v337;
        if ( (_DWORD)v337 )
        {
          v214 = (__int64)v390;
          v215 = (v337 - 1) & (((unsigned int)v390 >> 9) ^ ((unsigned int)v390 >> 4));
          v216 = (__int64 *)(v335 + 8LL * v215);
          v217 = (_BYTE *)*v216;
          if ( (unsigned __int64 *)*v216 == v390 )
            goto LABEL_343;
          v280 = 1;
          v281 = 0;
          while ( v217 != (_BYTE *)-8LL )
          {
            if ( !v281 && v217 == (_BYTE *)-16LL )
              v281 = v216;
            v215 = (v337 - 1) & (v280 + v215);
            v216 = (__int64 *)(v335 + 8LL * v215);
            v217 = (_BYTE *)*v216;
            if ( v390 == (unsigned __int64 *)*v216 )
              goto LABEL_343;
            ++v280;
          }
          if ( v281 )
            v216 = v281;
          ++v334;
          v282 = v336 + 1;
          if ( 4 * ((int)v336 + 1) < (unsigned int)(3 * v337) )
          {
            if ( (int)v337 - HIDWORD(v336) - v282 > (unsigned int)v337 >> 3 )
            {
LABEL_467:
              LODWORD(v336) = v282;
              if ( *v216 != -8 )
                --HIDWORD(v336);
              *v216 = v214;
LABEL_343:
              if ( (_DWORD)v333 )
              {
                v218 = (v333 - 1) & (((unsigned int)v29 >> 4) ^ ((unsigned int)v29 >> 9));
                v219 = &v331[v218];
                v220 = *v219;
                if ( v29 == *v219 )
                  goto LABEL_345;
                v269 = 1;
                v270 = 0;
                while ( v220 != -8 )
                {
                  if ( v220 == -16 && !v270 )
                    v270 = v219;
                  v218 = (v333 - 1) & (v269 + v218);
                  v219 = &v331[v218];
                  v220 = *v219;
                  if ( v29 == *v219 )
                    goto LABEL_345;
                  ++v269;
                }
                if ( v270 )
                  v219 = v270;
                ++v330;
                v271 = v332 + 1;
                if ( 4 * ((int)v332 + 1) < (unsigned int)(3 * v333) )
                {
                  if ( (int)v333 - HIDWORD(v332) - v271 > (unsigned int)v333 >> 3 )
                  {
LABEL_445:
                    LODWORD(v332) = v271;
                    if ( *v219 != -8 )
                      --HIDWORD(v332);
                    *v219 = v29;
LABEL_345:
                    v221 = v358;
                    v318 = &v358[(unsigned int)v359];
                    if ( v358 == v318 )
                    {
LABEL_505:
                      v306 = v305;
                      goto LABEL_32;
                    }
                    while ( 2 )
                    {
                      v225 = *v221;
                      if ( !(_DWORD)v333 )
                      {
                        ++v330;
                        goto LABEL_351;
                      }
                      LODWORD(v222) = (v333 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
                      v223 = &v331[(unsigned int)v222];
                      v224 = *v223;
                      if ( v225 != *v223 )
                      {
                        v259 = 1;
                        v260 = 0;
                        while ( v224 != -8 )
                        {
                          if ( v224 == -16 && !v260 )
                            v260 = v223;
                          v222 = ((_DWORD)v333 - 1) & (unsigned int)(v222 + v259);
                          v223 = &v331[v222];
                          v224 = *v223;
                          if ( v225 == *v223 )
                            goto LABEL_348;
                          ++v259;
                        }
                        if ( v260 )
                          v223 = v260;
                        ++v330;
                        v227 = v332 + 1;
                        if ( 4 * ((int)v332 + 1) >= (unsigned int)(3 * v333) )
                        {
LABEL_351:
                          sub_1467110((__int64)&v330, 2 * v333);
                          if ( !(_DWORD)v333 )
                            goto LABEL_506;
                          v226 = (v333 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
                          v227 = v332 + 1;
                          v223 = &v331[v226];
                          v228 = *v223;
                          if ( v225 != *v223 )
                          {
                            v265 = 1;
                            v266 = 0;
                            while ( v228 != -8 )
                            {
                              if ( !v266 && v228 == -16 )
                                v266 = v223;
                              v226 = (v333 - 1) & (v265 + v226);
                              v223 = &v331[v226];
                              v228 = *v223;
                              if ( v225 == *v223 )
                                goto LABEL_353;
                              ++v265;
                            }
                            if ( v266 )
                              v223 = v266;
                          }
                        }
                        else if ( (int)v333 - HIDWORD(v332) - v227 <= (unsigned int)v333 >> 3 )
                        {
                          sub_1467110((__int64)&v330, v333);
                          if ( !(_DWORD)v333 )
                          {
LABEL_506:
                            LODWORD(v332) = v332 + 1;
                            BUG();
                          }
                          v261 = 1;
                          v262 = (v333 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
                          v227 = v332 + 1;
                          v263 = 0;
                          v223 = &v331[v262];
                          v264 = *v223;
                          if ( v225 != *v223 )
                          {
                            while ( v264 != -8 )
                            {
                              if ( !v263 && v264 == -16 )
                                v263 = v223;
                              v262 = (v333 - 1) & (v261 + v262);
                              v223 = &v331[v262];
                              v264 = *v223;
                              if ( v225 == *v223 )
                                goto LABEL_353;
                              ++v261;
                            }
                            if ( v263 )
                              v223 = v263;
                          }
                        }
LABEL_353:
                        LODWORD(v332) = v227;
                        if ( *v223 != -8 )
                          --HIDWORD(v332);
                        *v223 = v225;
                      }
LABEL_348:
                      if ( v318 == ++v221 )
                        goto LABEL_505;
                      continue;
                    }
                  }
                  sub_1467110((__int64)&v330, v333);
                  if ( (_DWORD)v333 )
                  {
                    v272 = 1;
                    v219 = 0;
                    for ( j = (v333 - 1) & (((unsigned int)v29 >> 4) ^ ((unsigned int)v29 >> 9)); ; j = (v333 - 1) & v283 )
                    {
                      v274 = &v331[j];
                      v275 = *v274;
                      if ( v29 == *v274 )
                      {
                        v271 = v332 + 1;
                        v219 = &v331[j];
                        goto LABEL_445;
                      }
                      if ( v275 == -8 )
                        break;
                      if ( v219 || v275 != -16 )
                        v274 = v219;
                      v283 = v272 + j;
                      v219 = v274;
                      ++v272;
                    }
                    if ( !v219 )
                      v219 = &v331[j];
                    v271 = v332 + 1;
                    goto LABEL_445;
                  }
LABEL_508:
                  LODWORD(v332) = v332 + 1;
                  BUG();
                }
              }
              else
              {
                ++v330;
              }
              sub_1467110((__int64)&v330, 2 * v333);
              if ( (_DWORD)v333 )
              {
                v276 = 1;
                v277 = 0;
                for ( k = (v333 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4)); ; k = (v333 - 1) & v284 )
                {
                  v219 = &v331[k];
                  v279 = *v219;
                  if ( v29 == *v219 )
                  {
                    v271 = v332 + 1;
                    goto LABEL_445;
                  }
                  if ( v279 == -8 )
                    break;
                  if ( v277 || v279 != -16 )
                    v219 = v277;
                  v284 = v276 + k;
                  v277 = v219;
                  ++v276;
                }
                if ( v277 )
                  v219 = v277;
                v271 = v332 + 1;
                goto LABEL_445;
              }
              goto LABEL_508;
            }
LABEL_472:
            sub_1353F00((__int64)&v334, v213);
            sub_1A97120((__int64)&v334, (__int64 *)&v390, &v393);
            v216 = (__int64 *)v393;
            v214 = (__int64)v390;
            v282 = v336 + 1;
            goto LABEL_467;
          }
        }
        else
        {
          ++v334;
        }
        v213 = 2 * v337;
        goto LABEL_472;
      }
LABEL_32:
      LODWORD(v385) = 0;
      LODWORD(v359) = 0;
      LODWORD(v388) = 0;
LABEL_20:
      v27 = (_QWORD *)(*v27 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)(v321 + 40) != v27 )
        continue;
      break;
    }
    v25 = v320;
LABEL_34:
    v41 = v331;
    v42 = v332;
    v43 = (_QWORD **)v331;
    v44 = &v331[(unsigned int)v333];
    if ( (_DWORD)v332 )
    {
      if ( v331 == v44 )
      {
LABEL_95:
        if ( !v42 )
          goto LABEL_35;
      }
      else
      {
        v74 = (unsigned __int64)v331;
        while ( *(_QWORD *)v74 == -16 || *(_QWORD *)v74 == -8 )
        {
          v74 += 8LL;
          if ( (_QWORD *)v74 == v44 )
            goto LABEL_95;
        }
        if ( (_QWORD *)v74 != v44 )
        {
LABEL_109:
          v75 = *(_QWORD **)v74;
          v76 = 24LL * (*(_DWORD *)(*(_QWORD *)v74 + 20LL) & 0xFFFFFFF);
          if ( (*(_BYTE *)(*(_QWORD *)v74 + 23LL) & 0x40) != 0 )
          {
            v77 = (_QWORD *)*(v75 - 1);
            v75 = &v77[(unsigned __int64)v76 / 8];
          }
          else
          {
            v77 = &v75[v76 / 0xFFFFFFFFFFFFFFF8LL];
          }
          for ( ; v75 != v77; v77 += 3 )
          {
            if ( *v77 )
            {
              v78 = v77[1];
              v79 = v77[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v79 = v78;
              if ( v78 )
                *(_QWORD *)(v78 + 16) = *(_QWORD *)(v78 + 16) & 3LL | v79;
            }
            *v77 = 0;
          }
          while ( 1 )
          {
            v74 += 8LL;
            if ( (_QWORD *)v74 == v44 )
              break;
            if ( *(_QWORD *)v74 != -8 && *(_QWORD *)v74 != -16 )
            {
              if ( (_QWORD *)v74 != v44 )
                goto LABEL_109;
              break;
            }
          }
          v41 = v331;
          v42 = v332;
          v43 = (_QWORD **)v331;
          v44 = &v331[(unsigned int)v333];
          goto LABEL_95;
        }
      }
      if ( v41 == v44 )
        goto LABEL_35;
      while ( *v43 == (_QWORD *)-16LL || *v43 == (_QWORD *)-8LL )
      {
        if ( ++v43 == v44 )
          goto LABEL_35;
      }
      if ( v44 != v43 )
      {
LABEL_102:
        sub_15F20C0(*v43);
        while ( ++v43 != v44 )
        {
          if ( *v43 != (_QWORD *)-16LL && *v43 != (_QWORD *)-8LL )
          {
            if ( v43 != v44 )
              goto LABEL_102;
            break;
          }
        }
        v42 = v332;
        goto LABEL_35;
      }
      ++v330;
LABEL_37:
      v45 = 4 * v42;
      v46 = (unsigned int)v333;
      if ( (unsigned int)(4 * v42) < 0x40 )
        v45 = 64;
      if ( v45 >= (unsigned int)v333 )
        goto LABEL_40;
      v85 = v41;
      v86 = v42 - 1;
      if ( v86 )
      {
        _BitScanReverse(&v87, v86);
        v88 = (unsigned int)(1 << (33 - (v87 ^ 0x1F)));
        if ( (int)v88 < 64 )
          v88 = 64;
        if ( (_DWORD)v88 == (_DWORD)v333 )
        {
          v332 = 0;
          do
          {
            if ( v85 )
              *v85 = -8;
            ++v85;
          }
          while ( &v41[v88] != v85 );
          goto LABEL_43;
        }
        v89 = (4 * (int)v88 / 3u + 1) | ((unsigned __int64)(4 * (int)v88 / 3u + 1) >> 1);
        v90 = ((v89 | (v89 >> 2)) >> 4) | v89 | (v89 >> 2) | ((((v89 | (v89 >> 2)) >> 4) | v89 | (v89 >> 2)) >> 8);
        v91 = (v90 | (v90 >> 16)) + 1;
        v92 = 8 * ((v90 | (v90 >> 16)) + 1);
      }
      else
      {
        v92 = 1024;
        v91 = 128;
      }
      j___libc_free_0((unsigned __int64)v41);
      LODWORD(v333) = v91;
      v93 = (_QWORD *)sub_22077B0(v92);
      v332 = 0;
      v331 = v93;
      for ( m = &v93[(unsigned int)v333]; m != v93; ++v93 )
      {
        if ( v93 )
          *v93 = -8;
      }
      goto LABEL_43;
    }
LABEL_35:
    ++v330;
    if ( v42 )
    {
      v41 = v331;
      goto LABEL_37;
    }
    if ( HIDWORD(v332) )
    {
      v46 = (unsigned int)v333;
      v41 = v331;
      if ( (unsigned int)v333 > 0x40 )
      {
        j___libc_free_0((unsigned __int64)v331);
        v331 = 0;
        v332 = 0;
        LODWORD(v333) = 0;
        goto LABEL_43;
      }
LABEL_40:
      for ( n = &v41[v46]; n != v41; ++v41 )
        *v41 = -8;
      v332 = 0;
    }
LABEL_43:
    v48 = *(unsigned int *)(a3 + 72);
    if ( !(_DWORD)v48 )
      goto LABEL_509;
    v49 = *(_QWORD *)(a3 + 56);
    v50 = (v48 - 1) & (((unsigned int)v321 >> 9) ^ ((unsigned int)v321 >> 4));
    v51 = (__int64 *)(v49 + 16LL * v50);
    v52 = *v51;
    if ( v321 != *v51 )
    {
      v83 = 1;
      while ( v52 != -8 )
      {
        v84 = v83 + 1;
        v50 = (v48 - 1) & (v83 + v50);
        v51 = (__int64 *)(v49 + 16LL * v50);
        v52 = *v51;
        if ( v321 == *v51 )
          goto LABEL_45;
        v83 = v84;
      }
LABEL_509:
      BUG();
    }
LABEL_45:
    if ( v51 == (__int64 *)(v49 + 16 * v48) )
      goto LABEL_509;
    v53 = v51[1];
    v54 = *(unsigned __int64 ***)(v53 + 32);
    v55 = *(unsigned __int64 ***)(v53 + 24);
    if ( v54 != v55 )
    {
      while ( 2 )
      {
        v56 = v345;
        v57 = **v55;
        v393 = (const __m128i *)v57;
        if ( v345 )
        {
          v58 = v345;
          v59 = v25;
          do
          {
            while ( 1 )
            {
              v60 = *((_QWORD *)v58 + 2);
              v61 = *((_QWORD *)v58 + 3);
              if ( *((_QWORD *)v58 + 4) >= v57 )
                break;
              v58 = (int *)*((_QWORD *)v58 + 3);
              if ( !v61 )
                goto LABEL_52;
            }
            v59 = v58;
            v58 = (int *)*((_QWORD *)v58 + 2);
          }
          while ( v60 );
LABEL_52:
          if ( v59 != v25 && *((_QWORD *)v59 + 4) <= v57 )
            goto LABEL_67;
        }
        v62 = v367;
        if ( v367 == (unsigned __int64 *)(v369 - 8) )
        {
          sub_1B4ECC0(&v361, &v393);
          v56 = v345;
          if ( v345 )
            goto LABEL_60;
        }
        else
        {
          if ( v367 )
          {
            *v367 = v57;
            v62 = v367;
            v56 = v345;
          }
          v367 = v62 + 1;
          if ( v56 )
          {
            while ( 1 )
            {
LABEL_60:
              v63 = *((_QWORD *)v56 + 4);
              v64 = (int *)*((_QWORD *)v56 + 3);
              if ( (unsigned __int64)v393 < v63 )
                v64 = (int *)*((_QWORD *)v56 + 2);
              if ( !v64 )
                break;
              v56 = v64;
            }
            if ( (unsigned __int64)v393 >= v63 )
            {
              if ( (unsigned __int64)v393 > v63 )
                goto LABEL_65;
LABEL_67:
              if ( v54 == ++v55 )
                goto LABEL_68;
              continue;
            }
            if ( v346 == v56 )
            {
LABEL_65:
              v65 = 1;
              if ( v56 != v25 )
                goto LABEL_86;
            }
            else
            {
LABEL_84:
              v73 = sub_220EF80((__int64)v56);
              if ( *(_QWORD *)(v73 + 32) >= (unsigned __int64)v393 )
                goto LABEL_67;
              v65 = 1;
              if ( v56 != v25 )
LABEL_86:
                v65 = (unsigned __int64)v393 < *((_QWORD *)v56 + 4);
            }
LABEL_66:
            v66 = sub_22077B0(0x28u);
            *(_QWORD *)(v66 + 32) = v393;
            sub_220F040(v65, v66, v56, v25);
            ++v348;
            goto LABEL_67;
          }
        }
        break;
      }
      v56 = v25;
      if ( v346 != v25 )
        goto LABEL_84;
      v65 = 1;
      goto LABEL_66;
    }
LABEL_68:
    v24 = (__int64 *)v363;
  }
  while ( v367 != (unsigned __int64 *)v363 );
LABEL_69:
  for ( ii = (__int64)v341; (int *)ii != &v339; ii = sub_220EEE0(ii) )
    sub_15F20C0(*(_QWORD **)(ii + 40));
  v68 = v361;
  if ( v361 )
  {
    v69 = v366;
    v70 = (unsigned __int64)(v370 + 1);
    if ( v370 + 1 > (__int64 *)v366 )
    {
      do
      {
        v71 = *v69++;
        j_j___libc_free_0(v71);
      }
      while ( v70 > (unsigned __int64)v69 );
      v68 = v361;
    }
    j_j___libc_free_0(v68);
  }
  sub_39598C0((unsigned __int64)v345);
  sub_3959A90(v340);
  j___libc_free_0(v335);
  if ( v358 != (__int64 *)v360 )
    _libc_free((unsigned __int64)v358);
  j___libc_free_0((unsigned __int64)v331);
  if ( v387 != (const __m128i *)v389 )
    _libc_free((unsigned __int64)v387);
  if ( v384 != (const __m128i *)v386 )
    _libc_free((unsigned __int64)v384);
  return v306;
}
