// Function: sub_2714C60
// Address: 0x2714c60
//
__int64 __fastcall sub_2714C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  const __m128i *v13; // rbx
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 *v19; // rax
  char *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // rax
  int v27; // eax
  int v28; // ebx
  unsigned int v29; // esi
  __int64 v30; // rdi
  __int64 *v31; // rdx
  __int64 v32; // rcx
  __int64 *v33; // rax
  __int64 v34; // rsi
  char *v35; // rax
  char *v36; // rdx
  __int64 v37; // rsi
  char *v38; // rdx
  int v39; // ecx
  char *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // r13
  __int64 v45; // r14
  __int64 *v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rcx
  _QWORD *v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r14
  __int64 *v58; // rdx
  __int64 *v59; // rax
  __int64 v60; // rcx
  _QWORD *v61; // rax
  __int64 *v62; // rax
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 v65; // r14
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int64 *v68; // rax
  char v69; // dl
  unsigned __int64 v70; // rax
  int v71; // edx
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int64 v74; // r8
  unsigned __int64 v75; // rcx
  const __m128i *v76; // rdx
  __m128i *v77; // rax
  __int64 *v78; // rax
  __int64 v79; // r9
  __int64 v80; // r8
  __int64 *v81; // rbx
  __int64 v82; // rax
  __int64 *v83; // rax
  __int64 v84; // r8
  __int64 *v85; // rbx
  __int64 v86; // rax
  unsigned __int64 v87; // rcx
  char *v88; // rdx
  char *v89; // rax
  __int64 *v90; // rax
  __int64 v91; // r9
  __int64 v92; // rdx
  __int64 *v93; // rax
  __int64 v94; // r8
  __int64 v95; // rax
  __int64 *v96; // rax
  char v97; // dl
  __int64 v98; // rax
  __int64 v99; // r14
  unsigned __int64 v100; // rdx
  unsigned __int64 *v101; // r15
  __int64 *v102; // rax
  __int64 *v103; // rbx
  __int64 v104; // r12
  unsigned __int64 v105; // rdx
  __int64 v106; // r8
  unsigned int v107; // eax
  __int64 v108; // rbx
  __int64 v109; // rdi
  __int64 v110; // rax
  void *v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rdi
  __int64 v114; // r9
  __int64 v115; // rcx
  __int64 v116; // r14
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // r8
  __int64 v121; // rcx
  __int64 v122; // r9
  __int64 v123; // r12
  __int64 v124; // r13
  __int64 v125; // rbx
  char v126; // al
  __int64 v127; // rbx
  unsigned __int64 v128; // rdi
  int v129; // ecx
  __int64 v130; // rdi
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 *v133; // rbx
  __int64 v134; // r9
  unsigned int v135; // eax
  __int64 k; // r13
  __int64 v137; // rcx
  __int64 v138; // rcx
  __int64 v139; // r8
  __int64 v140; // r9
  __int64 v141; // r8
  __int64 v142; // rdx
  __int64 v143; // r9
  __int64 v144; // r8
  __int64 v145; // rcx
  __int64 v146; // r9
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r8
  int v150; // ebx
  _QWORD *v151; // r13
  unsigned __int64 v152; // rax
  __int64 v153; // r14
  __int64 v154; // r13
  __int64 v155; // r15
  __int64 v156; // r12
  unsigned __int64 v157; // rax
  char v158; // al
  __int64 *v159; // rdi
  __int64 *v160; // rax
  __int64 v161; // rax
  unsigned __int8 **v162; // rax
  unsigned __int8 *i; // rdi
  __int64 v164; // rdi
  unsigned __int8 *v165; // rax
  __int64 v166; // rdx
  __int64 v167; // r14
  unsigned __int8 v168; // al
  __int64 v169; // r8
  unsigned int v170; // eax
  _QWORD *v171; // rax
  __int64 v172; // rdx
  _QWORD *v173; // r13
  __int64 v174; // r12
  _QWORD *v175; // rbx
  __int64 *v176; // rax
  _QWORD *v177; // rbx
  __int64 v178; // rax
  __int64 v179; // rdx
  __int64 v180; // rdi
  __int64 v181; // r8
  unsigned int v182; // eax
  __int64 v183; // rbx
  __int64 v184; // rax
  void *v185; // rax
  __int64 v186; // rdx
  __int64 v187; // r9
  __int64 v188; // rcx
  __int64 v189; // r15
  signed __int64 v190; // r12
  __int64 v191; // r13
  __int64 v192; // rdx
  __int64 v193; // rsi
  __int64 v194; // rcx
  __int64 v195; // rbx
  __int64 v196; // r14
  char v197; // al
  signed __int64 v198; // r12
  __int64 *v199; // r15
  unsigned int v200; // edx
  __int64 *v201; // rax
  __int64 v202; // r9
  __int64 v203; // rcx
  __int64 v204; // rdi
  __int64 v205; // rsi
  __int64 *v206; // r14
  __int64 *m; // r13
  unsigned int v208; // ecx
  __int64 *v209; // rax
  __int64 v210; // r10
  __int64 v211; // rdx
  __int64 v212; // r8
  int v213; // eax
  int v214; // r11d
  int v215; // r10d
  __int64 v216; // r13
  __int64 v217; // rbx
  __int64 v218; // r14
  unsigned __int64 v219; // rdi
  __int64 v220; // r8
  int v221; // r10d
  __int64 v222; // r9
  __int64 v223; // rcx
  __int64 v224; // rax
  __int64 *v225; // rdx
  __int64 v226; // rdi
  _QWORD *v227; // rax
  int v228; // edx
  __int64 *v229; // rax
  __int64 v230; // rbx
  __int64 v231; // r14
  unsigned int v232; // r12d
  unsigned int v234; // ecx
  int v235; // edi
  int v236; // ecx
  unsigned int v237; // r15d
  int v238; // ecx
  __int64 v239; // rax
  __int64 v240; // r14
  __int64 v241; // rax
  __int64 v242; // r14
  __int64 v243; // r12
  __int64 v244; // rbx
  __int64 v245; // r15
  __int64 v246; // r13
  __int64 v247; // rcx
  __int64 v248; // r8
  __int64 v249; // r9
  unsigned __int64 v250; // r8
  __int64 v251; // rbx
  unsigned __int64 v252; // rdi
  __int64 v253; // r13
  __int64 v254; // rcx
  __int64 v255; // r14
  __int64 v256; // r12
  unsigned __int64 v257; // rbx
  char v258; // al
  __int64 v259; // rcx
  __int64 v260; // r14
  __int64 v261; // rbx
  __int64 v262; // r15
  __int64 v263; // r13
  __int64 v264; // r12
  __int64 v265; // rcx
  __int64 v266; // r8
  __int64 v267; // r9
  __int64 v268; // r13
  char *v269; // rbx
  __int64 v270; // rax
  __int64 j; // rdx
  __int64 v272; // rdi
  __int64 v273; // rbx
  unsigned int v274; // eax
  _DWORD **v275; // rsi
  unsigned int v276; // r12d
  unsigned int v277; // edx
  __int64 *v278; // rcx
  __int64 v279; // r8
  __int64 *v280; // rax
  __int64 v281; // rdx
  __int64 *v282; // r15
  __int64 v283; // rcx
  __int64 v284; // r13
  _QWORD *v285; // r15
  __int64 v286; // r12
  __int64 v287; // rax
  __int64 v288; // rax
  __int64 v289; // rbx
  __int64 v290; // r13
  unsigned __int64 v291; // rdi
  unsigned __int8 *v292; // rdi
  __int64 v293; // rdi
  unsigned __int8 *v294; // rax
  __int64 v295; // rdx
  unsigned __int8 v296; // al
  __int64 v297; // r8
  unsigned int v298; // eax
  __int64 v299; // rax
  unsigned __int8 *v300; // rdi
  __int64 v301; // rdi
  unsigned __int8 *v302; // rax
  __int64 v303; // rdx
  unsigned __int8 v304; // al
  __int64 v305; // r8
  unsigned int v306; // eax
  __int64 v307; // r9
  unsigned int v308; // esi
  __int64 v309; // r11
  __int64 v310; // rcx
  unsigned int v311; // r15d
  __int64 v312; // rax
  __int64 v313; // r8
  __int64 v314; // rdx
  __int64 v315; // rdi
  __int64 v316; // r15
  __int64 v317; // rbx
  __int64 v318; // r13
  unsigned __int64 v319; // rdi
  __int64 *v320; // r12
  __int64 v321; // rax
  __int64 *v322; // rax
  int v323; // ecx
  int v324; // r9d
  int v325; // eax
  int v326; // r10d
  __int64 v327; // rax
  int v328; // ecx
  unsigned __int64 v329; // r14
  __int64 v330; // r13
  __int64 v331; // r15
  char v332; // al
  __int64 v333; // r14
  __int64 v334; // r15
  __int64 v335; // rcx
  __int64 v336; // r8
  __int64 v337; // r9
  __int64 v338; // rax
  __int64 v339; // r14
  __int64 v340; // r13
  __int64 v341; // r15
  signed __int64 v342; // r14
  __int64 v343; // r12
  __int64 v344; // rbx
  __int64 v345; // rcx
  __int64 v346; // r8
  __int64 v347; // r9
  __int64 v348; // r15
  unsigned __int64 v349; // r13
  unsigned __int64 v350; // rdi
  int v351; // eax
  int v352; // eax
  int v353; // eax
  int v354; // r10d
  __int64 v355; // r11
  __int64 v356; // rcx
  int v357; // edi
  __int64 v358; // rsi
  int v359; // eax
  int v360; // r10d
  __int64 v361; // r11
  __int64 v362; // rcx
  int v363; // edi
  char *v364; // rbx
  int v365; // r10d
  __int64 v366; // rcx
  int v367; // r11d
  __int64 v368; // rdi
  __int64 v369; // [rsp+0h] [rbp-5F0h]
  unsigned __int64 v370; // [rsp+18h] [rbp-5D8h]
  __int64 *v371; // [rsp+20h] [rbp-5D0h]
  unsigned __int64 v372; // [rsp+20h] [rbp-5D0h]
  char v374; // [rsp+28h] [rbp-5C8h]
  unsigned __int8 v375; // [rsp+37h] [rbp-5B9h]
  _BYTE *v376; // [rsp+38h] [rbp-5B8h]
  __int64 v377; // [rsp+40h] [rbp-5B0h]
  char v378; // [rsp+40h] [rbp-5B0h]
  __int64 v380; // [rsp+50h] [rbp-5A0h]
  __int64 v381; // [rsp+50h] [rbp-5A0h]
  _BYTE *v382; // [rsp+50h] [rbp-5A0h]
  unsigned __int64 *v383; // [rsp+50h] [rbp-5A0h]
  __int64 *v384; // [rsp+58h] [rbp-598h]
  __int64 v385; // [rsp+58h] [rbp-598h]
  __int64 v387; // [rsp+68h] [rbp-588h]
  __int64 *v388; // [rsp+68h] [rbp-588h]
  __int64 v389; // [rsp+68h] [rbp-588h]
  __int64 v390; // [rsp+68h] [rbp-588h]
  __int64 v391; // [rsp+68h] [rbp-588h]
  __int64 v392; // [rsp+68h] [rbp-588h]
  __int64 v393; // [rsp+68h] [rbp-588h]
  __int64 v394; // [rsp+68h] [rbp-588h]
  unsigned __int64 *v395; // [rsp+68h] [rbp-588h]
  __int64 v396; // [rsp+68h] [rbp-588h]
  __int64 v397; // [rsp+68h] [rbp-588h]
  __int64 v398; // [rsp+68h] [rbp-588h]
  __int64 v399; // [rsp+70h] [rbp-580h]
  __int64 v402; // [rsp+80h] [rbp-570h]
  unsigned __int64 v403; // [rsp+80h] [rbp-570h]
  __int64 v404; // [rsp+80h] [rbp-570h]
  __int64 v405; // [rsp+80h] [rbp-570h]
  __int64 v406; // [rsp+80h] [rbp-570h]
  __int64 v407; // [rsp+88h] [rbp-568h]
  __int64 v408; // [rsp+88h] [rbp-568h]
  __int64 v409; // [rsp+88h] [rbp-568h]
  __int64 v410; // [rsp+88h] [rbp-568h]
  __int64 v411; // [rsp+88h] [rbp-568h]
  __int64 v412; // [rsp+88h] [rbp-568h]
  __int64 v413; // [rsp+88h] [rbp-568h]
  __int64 *v414; // [rsp+88h] [rbp-568h]
  _QWORD *v415; // [rsp+88h] [rbp-568h]
  __int64 v416; // [rsp+88h] [rbp-568h]
  __int64 v417; // [rsp+88h] [rbp-568h]
  __int64 v418; // [rsp+88h] [rbp-568h]
  __int64 v419; // [rsp+88h] [rbp-568h]
  __int64 v420; // [rsp+88h] [rbp-568h]
  unsigned int v421; // [rsp+88h] [rbp-568h]
  unsigned __int64 v422; // [rsp+90h] [rbp-560h]
  __int64 v423; // [rsp+90h] [rbp-560h]
  __int64 *v424; // [rsp+90h] [rbp-560h]
  __int64 v425; // [rsp+90h] [rbp-560h]
  __int64 v426; // [rsp+90h] [rbp-560h]
  __int64 v427; // [rsp+90h] [rbp-560h]
  __int64 *v428; // [rsp+90h] [rbp-560h]
  __int64 v429; // [rsp+90h] [rbp-560h]
  __int64 v430; // [rsp+90h] [rbp-560h]
  _QWORD *v431; // [rsp+90h] [rbp-560h]
  unsigned __int64 *v432; // [rsp+90h] [rbp-560h]
  __int64 *v433; // [rsp+90h] [rbp-560h]
  __int64 *v434; // [rsp+90h] [rbp-560h]
  __int64 v435; // [rsp+90h] [rbp-560h]
  _QWORD *v436; // [rsp+90h] [rbp-560h]
  __int64 v437; // [rsp+90h] [rbp-560h]
  __int64 v438; // [rsp+A8h] [rbp-548h] BYREF
  __int64 v439; // [rsp+B0h] [rbp-540h] BYREF
  __int64 v440; // [rsp+B8h] [rbp-538h] BYREF
  _BYTE *v441; // [rsp+C0h] [rbp-530h] BYREF
  __int64 v442; // [rsp+C8h] [rbp-528h]
  _BYTE v443[128]; // [rsp+D0h] [rbp-520h] BYREF
  _BYTE *v444; // [rsp+150h] [rbp-4A0h] BYREF
  __int64 v445; // [rsp+158h] [rbp-498h]
  _BYTE v446[128]; // [rsp+160h] [rbp-490h] BYREF
  __int64 v447; // [rsp+1E0h] [rbp-410h] BYREF
  void *s; // [rsp+1E8h] [rbp-408h]
  _BYTE v449[12]; // [rsp+1F0h] [rbp-400h]
  char v450; // [rsp+1FCh] [rbp-3F4h]
  char v451; // [rsp+200h] [rbp-3F0h] BYREF
  __int64 v452; // [rsp+280h] [rbp-370h] BYREF
  char *v453; // [rsp+288h] [rbp-368h]
  __int64 v454; // [rsp+290h] [rbp-360h]
  int v455; // [rsp+298h] [rbp-358h]
  char v456; // [rsp+29Ch] [rbp-354h]
  char v457; // [rsp+2A0h] [rbp-350h] BYREF
  _DWORD *v458; // [rsp+320h] [rbp-2D0h] BYREF
  __int64 v459; // [rsp+328h] [rbp-2C8h] BYREF
  _DWORD v460[64]; // [rsp+330h] [rbp-2C0h] BYREF
  unsigned __int64 v461; // [rsp+430h] [rbp-1C0h] BYREF
  __int64 v462; // [rsp+438h] [rbp-1B8h]
  __int64 v463; // [rsp+440h] [rbp-1B0h] BYREF
  __int64 v464; // [rsp+448h] [rbp-1A8h] BYREF
  _BYTE *v465; // [rsp+450h] [rbp-1A0h]
  __int64 v466; // [rsp+458h] [rbp-198h]
  int v467; // [rsp+460h] [rbp-190h]
  char v468; // [rsp+464h] [rbp-18Ch]
  _BYTE v469[16]; // [rsp+468h] [rbp-188h] BYREF
  __int64 v470; // [rsp+478h] [rbp-178h] BYREF
  _BYTE *v471; // [rsp+480h] [rbp-170h]
  __int64 v472; // [rsp+488h] [rbp-168h]
  int v473; // [rsp+490h] [rbp-160h]
  char v474; // [rsp+494h] [rbp-15Ch]
  _BYTE v475[16]; // [rsp+498h] [rbp-158h] BYREF
  char v476; // [rsp+4A8h] [rbp-148h]

  v5 = *(_BYTE *)(a1 + 196) == 0;
  v441 = v443;
  v442 = 0x1000000000LL;
  v444 = v446;
  v445 = 0x1000000000LL;
  if ( v5 )
  {
    v351 = sub_B6ED60(**(__int64 ***)(a1 + 168), "clang.arc.no_objc_arc_exceptions", 0x20u);
    *(_BYTE *)(a1 + 196) = 1;
    *(_DWORD *)(a1 + 192) = v351;
  }
  v447 = 0;
  s = &v451;
  v453 = &v457;
  v462 = 0x1000000000LL;
  *(_QWORD *)v449 = 16;
  v6 = *(_QWORD *)(a2 + 80);
  *(_DWORD *)&v449[8] = 0;
  v450 = 1;
  v452 = 0;
  if ( v6 )
    v6 -= 24;
  v456 = 1;
  v454 = 16;
  v455 = 0;
  v461 = (unsigned __int64)&v463;
  v438 = v6;
  *(_DWORD *)sub_270EC80(a3, &v438) = 1;
  v8 = *(_QWORD *)(v438 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == v438 + 48 )
  {
    v10 = 0;
  }
  else
  {
    if ( !v8 )
LABEL_578:
      BUG();
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = v8 - 24;
    if ( (unsigned int)(v9 - 30) >= 0xB )
      v10 = 0;
  }
  v458 = (_DWORD *)v438;
  v11 = (unsigned int)v462;
  v12 = HIDWORD(v462);
  v459 = v10;
  v13 = (const __m128i *)&v458;
  v14 = (unsigned int)v462 + 1LL;
  v15 = v461;
  v460[0] = 0;
  if ( v14 > HIDWORD(v462) )
  {
    if ( v461 > (unsigned __int64)&v458 || (unsigned __int64)&v458 >= v461 + 24LL * (unsigned int)v462 )
    {
      sub_C8D5F0((__int64)&v461, &v463, (unsigned int)v462 + 1LL, 0x18u, v14, v7);
      v15 = v461;
      v11 = (unsigned int)v462;
    }
    else
    {
      v364 = (char *)&v458 - v461;
      sub_C8D5F0((__int64)&v461, &v463, (unsigned int)v462 + 1LL, 0x18u, v14, v7);
      v15 = v461;
      v11 = (unsigned int)v462;
      v13 = (const __m128i *)&v364[v461];
    }
  }
  v16 = (__m128i *)(v15 + 24 * v11);
  *v16 = _mm_loadu_si128(v13);
  v17 = v13[1].m128i_i64[0];
  v18 = v438;
  v16[1].m128i_i64[0] = v17;
  LODWORD(v462) = v462 + 1;
  if ( !v450 )
    goto LABEL_368;
  v19 = (unsigned __int64 *)s;
  v17 = *(unsigned int *)&v449[4];
  v12 = (__int64)s + 8 * *(unsigned int *)&v449[4];
  if ( s != (void *)v12 )
  {
    do
    {
      v14 = *v19;
      if ( v18 == *v19 )
        goto LABEL_15;
      ++v19;
    }
    while ( (unsigned __int64 *)v12 != v19 );
  }
  if ( *(_DWORD *)&v449[4] < *(_DWORD *)v449 )
  {
    v17 = (unsigned int)++*(_DWORD *)&v449[4];
    *(_QWORD *)v12 = v18;
    v14 = v438;
    ++v447;
  }
  else
  {
LABEL_368:
    sub_C8CC70((__int64)&v447, v18, v17, v12, v14, v7);
    v14 = v438;
  }
LABEL_15:
  if ( !v456 )
    goto LABEL_366;
  v20 = v453;
  v17 = HIDWORD(v454);
  v12 = (__int64)&v453[8 * HIDWORD(v454)];
  if ( v453 != (char *)v12 )
  {
    while ( *(_QWORD *)v20 != v14 )
    {
      v20 += 8;
      if ( (char *)v12 == v20 )
        goto LABEL_365;
    }
    goto LABEL_20;
  }
LABEL_365:
  if ( HIDWORD(v454) < (unsigned int)v454 )
  {
    ++HIDWORD(v454);
    *(_QWORD *)v12 = v14;
    ++v452;
  }
  else
  {
LABEL_366:
    sub_C8CC70((__int64)&v452, v14, v17, v12, v14, v7);
  }
LABEL_20:
  v21 = (unsigned int)v462;
  do
  {
    while ( 1 )
    {
      v22 = v461 + 24 * v21 - 24;
      v23 = *(_QWORD *)v22;
      v24 = *(_QWORD *)(*(_QWORD *)v22 + 48LL);
      v25 = *(_QWORD *)v22 + 48LL;
      v439 = *(_QWORD *)v22;
      v26 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v26 == v25 )
        goto LABEL_103;
      if ( !v26 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 > 0xA )
      {
LABEL_103:
        v28 = 0;
      }
      else
      {
        v407 = v23;
        v422 = v22;
        v27 = sub_B46E30(v26 - 24);
        v22 = v422;
        v23 = v407;
        v28 = v27;
      }
      v29 = *(_DWORD *)(v22 + 16);
      if ( v28 == v29 )
        break;
      while ( 1 )
      {
        v30 = *(_QWORD *)(v22 + 8);
        *(_DWORD *)(v22 + 16) = v29 + 1;
        v440 = sub_B46EC0(v30, v29);
        if ( v450 )
        {
          v33 = (__int64 *)s;
          v32 = *(unsigned int *)&v449[4];
          v31 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v449[4]);
          if ( s != v31 )
          {
            do
            {
              v34 = *v33;
              if ( v440 == *v33 )
                goto LABEL_31;
              ++v33;
            }
            while ( v31 != v33 );
          }
          if ( *(_DWORD *)&v449[4] < *(_DWORD *)v449 )
            break;
        }
        sub_C8CC70((__int64)&v447, v440, (__int64)v31, v32, v440, v7);
        v34 = v440;
        if ( v69 )
          goto LABEL_79;
LABEL_31:
        if ( v456 )
        {
          v35 = v453;
          v36 = &v453[8 * HIDWORD(v454)];
          if ( v453 != v36 )
          {
            while ( *(_QWORD *)v35 != v34 )
            {
              v35 += 8;
              if ( v36 == v35 )
                goto LABEL_97;
            }
            goto LABEL_36;
          }
LABEL_97:
          v90 = sub_270EC80(a3, &v439);
          v91 = v440;
          v92 = *((unsigned int *)v90 + 40);
          if ( v92 + 1 > (unsigned __int64)*((unsigned int *)v90 + 41) )
          {
            v410 = v440;
            v424 = v90;
            sub_C8D5F0((__int64)(v90 + 19), v90 + 21, v92 + 1, 8u, v92 + 1, v440);
            v90 = v424;
            v91 = v410;
            v92 = *((unsigned int *)v424 + 40);
          }
          *(_QWORD *)(v90[19] + 8 * v92) = v91;
          ++*((_DWORD *)v90 + 40);
          v93 = sub_270EC80(a3, &v440);
          v7 = v439;
          v94 = (__int64)v93;
          v95 = *((unsigned int *)v93 + 32);
          if ( v95 + 1 > (unsigned __int64)*(unsigned int *)(v94 + 132) )
          {
            v409 = v439;
            v423 = v94;
            sub_C8D5F0(v94 + 120, (const void *)(v94 + 136), v95 + 1, 8u, v94, v439);
            v94 = v423;
            v7 = v409;
            v95 = *(unsigned int *)(v423 + 128);
          }
          *(_QWORD *)(*(_QWORD *)(v94 + 120) + 8 * v95) = v7;
          ++*(_DWORD *)(v94 + 128);
          goto LABEL_36;
        }
        if ( !sub_C8CA60((__int64)&v452, v34) )
          goto LABEL_97;
LABEL_36:
        v22 = v461 + 24LL * (unsigned int)v462 - 24;
        v29 = *(_DWORD *)(v22 + 16);
        if ( v29 == v28 )
        {
          v23 = v439;
          goto LABEL_38;
        }
      }
      ++*(_DWORD *)&v449[4];
      *v31 = v440;
      v34 = v440;
      ++v447;
LABEL_79:
      v70 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v70 == v34 + 48 )
      {
        v72 = 0;
        goto LABEL_83;
      }
      if ( !v70 )
        goto LABEL_578;
      v71 = *(unsigned __int8 *)(v70 - 24);
      v72 = v70 - 24;
      if ( (unsigned int)(v71 - 30) >= 0xB )
        v72 = 0;
LABEL_83:
      v459 = v72;
      v73 = (unsigned int)v462;
      v458 = (_DWORD *)v34;
      v74 = (unsigned int)v462 + 1LL;
      v75 = v461;
      v460[0] = 0;
      v76 = (const __m128i *)&v458;
      if ( v74 > HIDWORD(v462) )
      {
        if ( v461 > (unsigned __int64)&v458 || (unsigned __int64)&v458 >= v461 + 24LL * (unsigned int)v462 )
        {
          sub_C8D5F0((__int64)&v461, &v463, (unsigned int)v462 + 1LL, 0x18u, v74, v7);
          v75 = v461;
          v73 = (unsigned int)v462;
          v76 = (const __m128i *)&v458;
        }
        else
        {
          v269 = (char *)&v458 - v461;
          sub_C8D5F0((__int64)&v461, &v463, (unsigned int)v462 + 1LL, 0x18u, v74, v7);
          v75 = v461;
          v73 = (unsigned int)v462;
          v76 = (const __m128i *)&v269[v461];
        }
      }
      v77 = (__m128i *)(v75 + 24 * v73);
      *v77 = _mm_loadu_si128(v76);
      v77[1].m128i_i64[0] = v76[1].m128i_i64[0];
      LODWORD(v462) = v462 + 1;
      v78 = sub_270EC80(a3, &v439);
      v80 = v440;
      v81 = v78;
      v82 = *((unsigned int *)v78 + 40);
      if ( v82 + 1 > (unsigned __int64)*((unsigned int *)v81 + 41) )
      {
        v426 = v440;
        sub_C8D5F0((__int64)(v81 + 19), v81 + 21, v82 + 1, 8u, v440, v79);
        v82 = *((unsigned int *)v81 + 40);
        v80 = v426;
      }
      *(_QWORD *)(v81[19] + 8 * v82) = v80;
      ++*((_DWORD *)v81 + 40);
      v83 = sub_270EC80(a3, &v440);
      v84 = v439;
      v85 = v83;
      v86 = *((unsigned int *)v83 + 32);
      v87 = *((unsigned int *)v85 + 33);
      if ( v86 + 1 > v87 )
      {
        v425 = v439;
        sub_C8D5F0((__int64)(v85 + 15), v85 + 17, v86 + 1, 8u, v439, v7);
        v86 = *((unsigned int *)v85 + 32);
        v84 = v425;
      }
      v88 = (char *)v85[15];
      *(_QWORD *)&v88[8 * v86] = v84;
      ++*((_DWORD *)v85 + 32);
      if ( !v456 )
        goto LABEL_102;
      v89 = v453;
      v87 = HIDWORD(v454);
      v88 = &v453[8 * HIDWORD(v454)];
      if ( v453 == v88 )
      {
LABEL_107:
        if ( HIDWORD(v454) < (unsigned int)v454 )
        {
          ++HIDWORD(v454);
          *(_QWORD *)v88 = v440;
          v21 = (unsigned int)v462;
          ++v452;
          continue;
        }
LABEL_102:
        sub_C8CC70((__int64)&v452, v440, (__int64)v88, v87, v84, v7);
        v21 = (unsigned int)v462;
      }
      else
      {
        while ( v440 != *(_QWORD *)v89 )
        {
          v89 += 8;
          if ( v88 == v89 )
            goto LABEL_107;
        }
        v21 = (unsigned int)v462;
      }
    }
LABEL_38:
    if ( v456 )
    {
      v37 = (__int64)v453;
      v38 = &v453[8 * HIDWORD(v454)];
      v39 = HIDWORD(v454);
      if ( v453 != v38 )
      {
        v40 = v453;
        while ( *(_QWORD *)v40 != v23 )
        {
          v40 += 8;
          if ( v38 == v40 )
            goto LABEL_44;
        }
        --HIDWORD(v454);
        *(_QWORD *)v40 = *(_QWORD *)&v453[8 * (v39 - 1)];
        v23 = v439;
        ++v452;
      }
    }
    else
    {
      v37 = v23;
      v96 = sub_C8CA60((__int64)&v452, v23);
      if ( v96 )
      {
        *v96 = -2;
        ++v455;
        ++v452;
      }
      v23 = v439;
    }
LABEL_44:
    v41 = (unsigned int)v442;
    v42 = (unsigned int)v442 + 1LL;
    if ( v42 > HIDWORD(v442) )
    {
      v37 = (__int64)v443;
      v427 = v23;
      sub_C8D5F0((__int64)&v441, v443, v42, 8u, v23, v7);
      v41 = (unsigned int)v442;
      v23 = v427;
    }
    *(_QWORD *)&v441[8 * v41] = v23;
    LODWORD(v442) = v442 + 1;
    v5 = (_DWORD)v462 == 1;
    v21 = (unsigned int)(v462 - 1);
    LODWORD(v462) = v462 - 1;
  }
  while ( !v5 );
  ++v447;
  if ( v450 )
    goto LABEL_52;
  v43 = 4 * (*(_DWORD *)&v449[4] - *(_DWORD *)&v449[8]);
  if ( v43 < 0x20 )
    v43 = 32;
  if ( *(_DWORD *)v449 > v43 )
  {
    sub_C8C990((__int64)&v447, v37);
  }
  else
  {
    v37 = 0xFFFFFFFFLL;
    memset(s, -1, 8LL * *(unsigned int *)v449);
LABEL_52:
    *(_QWORD *)&v449[4] = 0;
  }
  v458 = v460;
  v459 = 0x1000000000LL;
  v44 = *(_QWORD *)(a2 + 80);
  v408 = a2 + 72;
  if ( v44 == a2 + 72 )
    goto LABEL_118;
LABEL_56:
  while ( 2 )
  {
    v45 = 0;
    v37 = (__int64)&v440;
    if ( v44 )
      v45 = v44 - 24;
    v440 = v45;
    v46 = sub_270EC80(a3, &v440);
    if ( !*((_DWORD *)v46 + 40) )
    {
      *((_DWORD *)v46 + 1) = 1;
      v48 = v46[15];
      v49 = (unsigned int)v459;
      v50 = HIDWORD(v459);
      v51 = (_QWORD *)((unsigned int)v459 + 1LL);
      if ( (unsigned __int64)v51 > HIDWORD(v459) )
      {
        v398 = v48;
        sub_C8D5F0((__int64)&v458, v460, (unsigned __int64)v51, 0x10u, v47, v48);
        v49 = (unsigned int)v459;
        v48 = v398;
      }
      v52 = (__int64 *)&v458[4 * v49];
      *v52 = v45;
      v52[1] = v48;
      v53 = (unsigned int)(v459 + 1);
      LODWORD(v459) = v459 + 1;
      if ( v450 )
      {
        v51 = s;
        v37 = *(unsigned int *)&v449[4];
        v50 = (__int64)s + 8 * *(unsigned int *)&v449[4];
        if ( s == (void *)v50 )
        {
LABEL_170:
          if ( *(_DWORD *)&v449[4] >= *(_DWORD *)v449 )
            goto LABEL_171;
          v37 = (unsigned int)++*(_DWORD *)&v449[4];
          *(_QWORD *)v50 = v45;
          v53 = (unsigned int)v459;
          ++v447;
        }
        else
        {
          while ( v45 != *v51 )
          {
            if ( (_QWORD *)v50 == ++v51 )
              goto LABEL_170;
          }
        }
      }
      else
      {
LABEL_171:
        v37 = v45;
        sub_C8CC70((__int64)&v447, v45, (__int64)v51, v50, v47, v48);
        v53 = (unsigned int)v459;
      }
      if ( !(_DWORD)v53 )
        goto LABEL_55;
LABEL_67:
      v37 = (__int64)&v458[4 * v53 - 4];
      v54 = sub_270EC80(a3, (__int64 *)v37);
      v57 = v54[15] + 8LL * *((unsigned int *)v54 + 32);
      while ( 1 )
      {
        while ( 1 )
        {
          v58 = (__int64 *)&v458[4 * (unsigned int)v459 - 4];
          v59 = (__int64 *)v58[1];
          if ( (__int64 *)v57 == v59 )
          {
            v98 = (unsigned int)v445;
            v99 = *v58;
            LODWORD(v459) = v459 - 1;
            v100 = (unsigned int)v445 + 1LL;
            if ( v100 > HIDWORD(v445) )
            {
              v37 = (__int64)v446;
              sub_C8D5F0((__int64)&v444, v446, v100, 8u, v55, v56);
              v98 = (unsigned int)v445;
            }
            *(_QWORD *)&v444[8 * v98] = v99;
            v53 = (unsigned int)v459;
            LODWORD(v445) = v445 + 1;
            if ( !(_DWORD)v459 )
            {
              v44 = *(_QWORD *)(v44 + 8);
              if ( v408 == v44 )
                goto LABEL_116;
              goto LABEL_56;
            }
            goto LABEL_67;
          }
          v60 = (__int64)(v59 + 1);
          v58[1] = (__int64)(v59 + 1);
          v37 = *v59;
          v440 = *v59;
          if ( v450 )
            break;
LABEL_110:
          sub_C8CC70((__int64)&v447, v37, (__int64)v58, v60, v55, v56);
          if ( v97 )
            goto LABEL_75;
        }
        v61 = s;
        v60 = *(unsigned int *)&v449[4];
        v58 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v449[4]);
        if ( s == v58 )
        {
LABEL_73:
          if ( *(_DWORD *)&v449[4] < *(_DWORD *)v449 )
          {
            ++*(_DWORD *)&v449[4];
            *v58 = v37;
            ++v447;
LABEL_75:
            v62 = sub_270EC80(a3, &v440);
            v64 = v440;
            v65 = v62[15];
            v66 = (unsigned int)v459;
            v67 = (unsigned int)v459 + 1LL;
            if ( v67 > HIDWORD(v459) )
            {
              v389 = v440;
              sub_C8D5F0((__int64)&v458, v460, v67, 0x10u, v440, v63);
              v66 = (unsigned int)v459;
              v64 = v389;
            }
            v68 = (__int64 *)&v458[4 * v66];
            *v68 = v64;
            v68[1] = v65;
            v53 = (unsigned int)(v459 + 1);
            LODWORD(v459) = v459 + 1;
            goto LABEL_67;
          }
          goto LABEL_110;
        }
        while ( v37 != *v61 )
        {
          if ( v58 == ++v61 )
            goto LABEL_73;
        }
      }
    }
LABEL_55:
    v44 = *(_QWORD *)(v44 + 8);
    if ( v408 != v44 )
      continue;
    break;
  }
LABEL_116:
  if ( v458 != v460 )
    _libc_free((unsigned __int64)v458);
LABEL_118:
  if ( (__int64 *)v461 != &v463 )
    _libc_free(v461);
  if ( !v456 )
    _libc_free((unsigned __int64)v453);
  if ( !v450 )
    _libc_free((unsigned __int64)s);
  v375 = 0;
  v370 = (unsigned __int64)v444;
  v376 = &v444[8 * (unsigned int)v445];
  if ( v444 == v376 )
  {
LABEL_188:
    v461 = 0;
    v462 = 0;
    v159 = *(__int64 **)(a4 + 40);
    v160 = *(__int64 **)(a4 + 32);
    v463 = 0;
    LODWORD(v464) = 0;
    v414 = v159;
    v433 = v160;
    if ( v160 == v159 )
      goto LABEL_207;
    while ( 1 )
    {
      v161 = *v433;
      if ( (*(_BYTE *)(*v433 + 7) & 0x40) != 0 )
        v162 = *(unsigned __int8 ***)(v161 - 8);
      else
        v162 = (unsigned __int8 **)(v161 - 32LL * (*(_DWORD *)(v161 + 4) & 0x7FFFFFF));
      for ( i = *v162; ; i = *(unsigned __int8 **)(v167 - 32LL * (*(_DWORD *)(v167 + 4) & 0x7FFFFFF)) )
      {
        v165 = sub_BD3990(i, v37);
        v164 = 23;
        v167 = (__int64)v165;
        v168 = *v165;
        if ( v168 > 0x1Cu )
        {
          if ( v168 != 85 )
          {
            v164 = 2 * (unsigned int)(v168 != 34) + 21;
            goto LABEL_193;
          }
          v169 = *(_QWORD *)(v167 - 32);
          v164 = 21;
          if ( v169 )
          {
            if ( !*(_BYTE *)v169 && *(_QWORD *)(v169 + 24) == *(_QWORD *)(v167 + 80) )
              break;
          }
        }
LABEL_193:
        if ( !(unsigned __int8)sub_3108CA0(v164) )
          goto LABEL_201;
LABEL_194:
        ;
      }
      v170 = sub_3108960(*(_QWORD *)(v167 - 32), v37, v166);
      if ( (unsigned __int8)sub_3108CA0(v170) )
        goto LABEL_194;
LABEL_201:
      v171 = (_QWORD *)v433[10];
      if ( *((_BYTE *)v433 + 100) )
        v172 = *((unsigned int *)v433 + 23);
      else
        v172 = *((unsigned int *)v433 + 22);
      v173 = &v171[v172];
      if ( v171 != v173 )
      {
        while ( 1 )
        {
          v174 = *v171;
          v175 = v171;
          if ( *v171 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v173 == ++v171 )
            goto LABEL_206;
        }
LABEL_264:
        if ( v173 == v175 )
          goto LABEL_206;
        v37 = (unsigned int)v464;
        if ( (_DWORD)v464 )
        {
          v220 = (unsigned int)(v464 - 1);
          v221 = 1;
          v222 = 0;
          v223 = (unsigned int)v220 & (((unsigned int)v174 >> 9) ^ ((unsigned int)v174 >> 4));
          v224 = v462 + 56 * v223;
          v225 = *(__int64 **)v224;
          if ( *(_QWORD *)v224 == v174 )
          {
LABEL_267:
            v226 = v224 + 8;
            if ( !*(_BYTE *)(v224 + 36) )
              goto LABEL_268;
LABEL_286:
            v229 = *(__int64 **)(v226 + 8);
            v223 = *(unsigned int *)(v226 + 20);
            v225 = &v229[v223];
            if ( v229 == v225 )
            {
LABEL_289:
              if ( (unsigned int)v223 >= *(_DWORD *)(v226 + 16) )
              {
LABEL_268:
                v37 = v167;
                sub_C8CC70(v226, v167, (__int64)v225, v223, v220, v222);
              }
              else
              {
                *(_DWORD *)(v226 + 20) = v223 + 1;
                *v225 = v167;
                ++*(_QWORD *)v226;
              }
            }
            else
            {
              while ( v167 != *v229 )
              {
                if ( v225 == ++v229 )
                  goto LABEL_289;
              }
            }
            v227 = v175 + 1;
            if ( v175 + 1 == v173 )
              goto LABEL_206;
            while ( 1 )
            {
              v174 = *v227;
              v175 = v227;
              if ( *v227 < 0xFFFFFFFFFFFFFFFELL )
                goto LABEL_264;
              if ( v173 == ++v227 )
                goto LABEL_206;
            }
          }
          while ( v225 != (__int64 *)-4096LL )
          {
            if ( !v222 && v225 == (__int64 *)-8192LL )
              v222 = v224;
            v223 = (unsigned int)v220 & ((_DWORD)v223 + v221);
            v224 = v462 + 56 * v223;
            v225 = *(__int64 **)v224;
            if ( *(_QWORD *)v224 == v174 )
              goto LABEL_267;
            ++v221;
          }
          if ( v222 )
            v224 = v222;
          ++v461;
          v228 = v463 + 1;
          if ( 4 * ((int)v463 + 1) < (unsigned int)(3 * v464) )
          {
            if ( (int)v464 - HIDWORD(v463) - v228 <= (unsigned int)v464 >> 3 )
            {
              sub_2712830((__int64)&v461, v464);
              if ( !(_DWORD)v464 )
              {
LABEL_577:
                LODWORD(v463) = v463 + 1;
                BUG();
              }
              v220 = v462;
              v222 = 0;
              v237 = (v464 - 1) & (((unsigned int)v174 >> 9) ^ ((unsigned int)v174 >> 4));
              v228 = v463 + 1;
              v238 = 1;
              v224 = v462 + 56LL * v237;
              v37 = *(_QWORD *)v224;
              if ( *(_QWORD *)v224 != v174 )
              {
                while ( v37 != -4096 )
                {
                  if ( v37 == -8192 && !v222 )
                    v222 = v224;
                  v365 = v238 + 1;
                  v366 = ((_DWORD)v464 - 1) & (v237 + v238);
                  v237 = v366;
                  v224 = v462 + 56 * v366;
                  v37 = *(_QWORD *)v224;
                  if ( *(_QWORD *)v224 == v174 )
                    goto LABEL_283;
                  v238 = v365;
                }
                if ( v222 )
                  v224 = v222;
              }
            }
            goto LABEL_283;
          }
        }
        else
        {
          ++v461;
        }
        v37 = (unsigned int)(2 * v464);
        sub_2712830((__int64)&v461, v37);
        if ( !(_DWORD)v464 )
          goto LABEL_577;
        v222 = (unsigned int)(v464 - 1);
        v234 = v222 & (((unsigned int)v174 >> 9) ^ ((unsigned int)v174 >> 4));
        v228 = v463 + 1;
        v224 = v462 + 56LL * v234;
        v220 = *(_QWORD *)v224;
        if ( *(_QWORD *)v224 != v174 )
        {
          v235 = 1;
          v37 = 0;
          while ( v220 != -4096 )
          {
            if ( v220 == -8192 && !v37 )
              v37 = v224;
            v367 = v235 + 1;
            v368 = (unsigned int)v222 & (v234 + v235);
            v234 = v368;
            v224 = v462 + 56 * v368;
            v220 = *(_QWORD *)v224;
            if ( *(_QWORD *)v224 == v174 )
              goto LABEL_283;
            v235 = v367;
          }
          if ( v37 )
            v224 = v37;
        }
LABEL_283:
        LODWORD(v463) = v228;
        if ( *(_QWORD *)v224 != -4096 )
          --HIDWORD(v463);
        *(_QWORD *)v224 = v174;
        v226 = v224 + 8;
        *(_QWORD *)(v224 + 8) = 0;
        *(_QWORD *)(v224 + 16) = v224 + 40;
        *(_QWORD *)(v224 + 24) = 2;
        *(_DWORD *)(v224 + 32) = 0;
        *(_BYTE *)(v224 + 36) = 1;
        goto LABEL_286;
      }
LABEL_206:
      v433 += 16;
      if ( v414 == v433 )
      {
LABEL_207:
        v374 = 0;
        v372 = (unsigned __int64)v441;
        v382 = &v441[8 * (unsigned int)v442];
        if ( v441 == v382 )
        {
LABEL_553:
          v232 = 0;
LABEL_408:
          v288 = (unsigned int)v464;
          if ( (_DWORD)v464 )
          {
            v289 = v462;
            v290 = v462 + 56LL * (unsigned int)v464;
            do
            {
              while ( *(_QWORD *)v289 == -4096 || *(_QWORD *)v289 == -8192 || *(_BYTE *)(v289 + 36) )
              {
                v289 += 56;
                if ( v290 == v289 )
                  goto LABEL_415;
              }
              v291 = *(_QWORD *)(v289 + 16);
              v289 += 56;
              _libc_free(v291);
            }
            while ( v290 != v289 );
LABEL_415:
            v288 = (unsigned int)v464;
          }
          sub_C7D6A0(v462, 56 * v288, 8);
          goto LABEL_299;
        }
        v369 = a1 + 8;
        while ( 2 )
        {
          v37 = (__int64)&v452;
          v452 = *((_QWORD *)v382 - 1);
          v176 = sub_270EC80(a3, &v452);
          v177 = (_QWORD *)v176[15];
          v399 = (__int64)v176;
          v415 = v177;
          v178 = *((unsigned int *)v176 + 32);
          v434 = &v177[v178];
          if ( v177 == &v177[v178] )
            goto LABEL_379;
          v179 = *(unsigned int *)(a3 + 24);
          v180 = *v177;
          v181 = *(_QWORD *)(a3 + 8);
          if ( (_DWORD)v179 )
          {
            v182 = (v179 - 1) & (((unsigned int)v180 >> 9) ^ ((unsigned int)v180 >> 4));
            v183 = v181 + 192LL * v182;
            v37 = *(_QWORD *)v183;
            if ( *(_QWORD *)v183 == v180 )
              goto LABEL_212;
            v328 = 1;
            while ( v37 != -4096 )
            {
              v182 = (v179 - 1) & (v328 + v182);
              v183 = v181 + 192LL * v182;
              v37 = *(_QWORD *)v183;
              if ( v180 == *(_QWORD *)v183 )
                goto LABEL_212;
              ++v328;
            }
          }
          v183 = v181 + 192 * v179;
LABEL_212:
          if ( v183 + 16 != v399 + 8 )
          {
            v37 = 16LL * *(unsigned int *)(v399 + 32);
            sub_C7D6A0(*(_QWORD *)(v399 + 16), v37, 8);
            v184 = *(unsigned int *)(v183 + 40);
            *(_DWORD *)(v399 + 32) = v184;
            if ( (_DWORD)v184 )
            {
              v185 = (void *)sub_C7D670(16 * v184, 8);
              *(_QWORD *)(v399 + 16) = v185;
              v186 = *(unsigned int *)(v399 + 32);
              *(_DWORD *)(v399 + 24) = *(_DWORD *)(v183 + 32);
              *(_DWORD *)(v399 + 28) = *(_DWORD *)(v183 + 36);
              v37 = *(_QWORD *)(v183 + 24);
              memcpy(v185, (const void *)v37, 16 * v186);
            }
            else
            {
              *(_QWORD *)(v399 + 16) = 0;
              *(_QWORD *)(v399 + 24) = 0;
            }
          }
          v113 = v399;
          if ( v183 + 48 == v399 + 40 )
            goto LABEL_228;
          v187 = *(_QWORD *)(v183 + 56);
          v188 = *(_QWORD *)(v183 + 48);
          v189 = *(_QWORD *)(v399 + 40);
          v190 = v187 - v188;
          v105 = *(_QWORD *)(v399 + 56) - v189;
          if ( v105 >= v187 - v188 )
          {
            v191 = *(_QWORD *)(v399 + 48);
            v192 = v191 - v189;
            v193 = v191 - v189;
            if ( v190 <= (unsigned __int64)(v191 - v189) )
            {
              if ( v190 <= 0 )
                goto LABEL_374;
              v194 = v188 + 32;
              v390 = v183;
              v195 = v194;
              v403 = 0xF0F0F0F0F0F0F0F1LL * (v190 >> 3);
              v196 = v189 + 32;
              do
              {
                *(_QWORD *)(v196 - 32) = *(_QWORD *)(v195 - 32);
                *(_BYTE *)(v196 - 24) = *(_BYTE *)(v195 - 24);
                *(_BYTE *)(v196 - 23) = *(_BYTE *)(v195 - 23);
                *(_BYTE *)(v196 - 22) = *(_BYTE *)(v195 - 22);
                *(_BYTE *)(v196 - 16) = *(_BYTE *)(v195 - 16);
                *(_BYTE *)(v196 - 15) = *(_BYTE *)(v195 - 15);
                *(_QWORD *)(v196 - 8) = *(_QWORD *)(v195 - 8);
                if ( v195 != v196 )
                  sub_C8CE00(v196, v196 + 32, v195, v194, v181, v187);
                if ( v196 + 48 != v195 + 48 )
                  sub_C8CE00(v196 + 48, v196 + 80, v195 + 48, v194, v181, v187);
                v197 = *(_BYTE *)(v195 + 96);
                v196 += 136;
                v195 += 136;
                *(_BYTE *)(v196 - 40) = v197;
                --v403;
              }
              while ( v403 );
              v183 = v390;
              v189 += v190;
              if ( v191 == v189 )
              {
LABEL_226:
                v198 = *(_QWORD *)(v399 + 40) + v190;
                goto LABEL_227;
              }
              while ( 2 )
              {
                if ( *(_BYTE *)(v189 + 108) )
                {
                  if ( !*(_BYTE *)(v189 + 60) )
LABEL_377:
                    _libc_free(*(_QWORD *)(v189 + 40));
                }
                else
                {
                  _libc_free(*(_QWORD *)(v189 + 88));
                  if ( !*(_BYTE *)(v189 + 60) )
                    goto LABEL_377;
                }
                v189 += 136;
LABEL_374:
                if ( v191 == v189 )
                  goto LABEL_226;
                continue;
              }
            }
            v329 = 0xF0F0F0F0F0F0F0F1LL * (v192 >> 3);
            if ( v192 > 0 )
            {
              v330 = v188 + 32;
              v331 = v189 + 32;
              do
              {
                *(_QWORD *)(v331 - 32) = *(_QWORD *)(v330 - 32);
                *(_BYTE *)(v331 - 24) = *(_BYTE *)(v330 - 24);
                *(_BYTE *)(v331 - 23) = *(_BYTE *)(v330 - 23);
                *(_BYTE *)(v331 - 22) = *(_BYTE *)(v330 - 22);
                *(_BYTE *)(v331 - 16) = *(_BYTE *)(v330 - 16);
                *(_BYTE *)(v331 - 15) = *(_BYTE *)(v330 - 15);
                *(_QWORD *)(v331 - 8) = *(_QWORD *)(v330 - 8);
                if ( v330 != v331 )
                  sub_C8CE00(v331, v331 + 32, v330, v188, v181, v187);
                if ( v331 + 48 != v330 + 48 )
                  sub_C8CE00(v331 + 48, v331 + 80, v330 + 48, v188, v181, v187);
                v332 = *(_BYTE *)(v330 + 96);
                v331 += 136;
                v330 += 136;
                *(_BYTE *)(v331 - 40) = v332;
                --v329;
              }
              while ( v329 );
              v187 = *(_QWORD *)(v183 + 56);
              v188 = *(_QWORD *)(v183 + 48);
              v191 = *(_QWORD *)(v399 + 48);
              v189 = *(_QWORD *)(v399 + 40);
              v193 = v191 - v189;
            }
            v333 = v188 + v193;
            if ( v188 + v193 != v187 )
            {
              v334 = v187;
              do
              {
                if ( v191 )
                {
                  *(_QWORD *)v191 = *(_QWORD *)v333;
                  *(_BYTE *)(v191 + 8) = *(_BYTE *)(v333 + 8);
                  *(_BYTE *)(v191 + 9) = *(_BYTE *)(v333 + 9);
                  *(_BYTE *)(v191 + 10) = *(_BYTE *)(v333 + 10);
                  *(_BYTE *)(v191 + 16) = *(_BYTE *)(v333 + 16);
                  *(_BYTE *)(v191 + 17) = *(_BYTE *)(v333 + 17);
                  *(_QWORD *)(v191 + 24) = *(_QWORD *)(v333 + 24);
                  sub_C8CD80(v191 + 32, v191 + 64, v333 + 32, v188, v181, v187);
                  sub_C8CD80(v191 + 80, v191 + 112, v333 + 80, v335, v336, v337);
                  *(_BYTE *)(v191 + 128) = *(_BYTE *)(v333 + 128);
                }
                v333 += 136;
                v191 += 136;
              }
              while ( v333 != v334 );
              goto LABEL_226;
            }
            v198 = v189 + v190;
LABEL_227:
            *(_QWORD *)(v399 + 48) = v198;
LABEL_228:
            *(_DWORD *)v399 = *(_DWORD *)(v183 + 8);
            v199 = v415 + 1;
            if ( v434 != v415 + 1 )
            {
              while ( 1 )
              {
                v203 = *(unsigned int *)(a3 + 24);
                v204 = *v199;
                v205 = *(_QWORD *)(a3 + 8);
                if ( !(_DWORD)v203 )
                  goto LABEL_233;
                v200 = (v203 - 1) & (((unsigned int)v204 >> 9) ^ ((unsigned int)v204 >> 4));
                v201 = (__int64 *)(v205 + 192LL * v200);
                v202 = *v201;
                if ( v204 != *v201 )
                  break;
LABEL_231:
                ++v199;
                sub_2713180(v399, (__int64)(v201 + 1));
                if ( v434 == v199 )
                  goto LABEL_378;
              }
              v325 = 1;
              while ( v202 != -4096 )
              {
                v326 = v325 + 1;
                v327 = ((_DWORD)v203 - 1) & (v200 + v325);
                v200 = v327;
                v201 = (__int64 *)(v205 + 192 * v327);
                v202 = *v201;
                if ( v204 == *v201 )
                  goto LABEL_231;
                v325 = v326;
              }
LABEL_233:
              v201 = (__int64 *)(v205 + 192 * v203);
              goto LABEL_231;
            }
LABEL_378:
            v178 = *(unsigned int *)(v399 + 128);
LABEL_379:
            if ( !(unsigned __int8)sub_AA5590(v452, (v178 * 8) >> 3) )
            {
              v270 = *(_QWORD *)(v399 + 40);
              for ( j = *(_QWORD *)(v399 + 48); j != v270; v270 += 136 )
                *(_BYTE *)(v270 + 128) = 1;
            }
            v272 = v452;
            v378 = 0;
            v385 = v452 + 48;
            v396 = *(_QWORD *)(v452 + 56);
            if ( v396 == v452 + 48 )
            {
LABEL_405:
              sub_2713CF0(v272, a3, v399);
              if ( *(_BYTE *)(a1 + 208) )
                goto LABEL_553;
              v382 -= 8;
              if ( (_BYTE *)v372 == v382 )
              {
                v232 = v375;
                LOBYTE(v232) = v374 & v375;
                goto LABEL_408;
              }
              continue;
            }
            while ( 2 )
            {
              v273 = v396 - 24;
              if ( !v396 )
                v273 = 0;
              v274 = sub_3108990(v273);
              v275 = (_DWORD **)v462;
              v276 = v274;
              if ( (_DWORD)v464 )
              {
                v277 = (v464 - 1) & (((unsigned int)v273 >> 9) ^ ((unsigned int)v273 >> 4));
                v278 = (__int64 *)(v462 + 56LL * v277);
                v279 = *v278;
                if ( *v278 == v273 )
                {
LABEL_387:
                  if ( v278 != (__int64 *)(v462 + 56LL * (unsigned int)v464) )
                  {
                    v280 = (__int64 *)v278[2];
                    v281 = *((_BYTE *)v278 + 36) ? *((unsigned int *)v278 + 7) : *((unsigned int *)v278 + 6);
                    v282 = &v280[v281];
                    if ( v280 != v282 )
                    {
                      while ( 1 )
                      {
                        v283 = *v280;
                        if ( (unsigned __int64)*v280 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( v282 == ++v280 )
                          goto LABEL_393;
                      }
                      if ( v280 != v282 )
                      {
                        v421 = v276;
                        v320 = v280;
                        do
                        {
                          v275 = &v458;
                          v458 = (_DWORD *)v283;
                          v321 = sub_2713760((unsigned __int64 *)(v399 + 8), (__int64 *)&v458);
                          if ( *(_BYTE *)(v321 + 2) == 1 )
                            *(_BYTE *)(v321 + 120) = 1;
                          v322 = v320 + 1;
                          if ( v320 + 1 == v282 )
                            break;
                          while ( 1 )
                          {
                            v283 = *v322;
                            v320 = v322;
                            if ( (unsigned __int64)*v322 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            if ( v282 == ++v322 )
                              goto LABEL_461;
                          }
                        }
                        while ( v282 != v322 );
LABEL_461:
                        v276 = v421;
                      }
                    }
                  }
                }
                else
                {
                  v323 = 1;
                  while ( v279 != -4096 )
                  {
                    v324 = v323 + 1;
                    v277 = (v464 - 1) & (v323 + v277);
                    v278 = (__int64 *)(v462 + 56LL * v277);
                    v279 = *v278;
                    if ( v273 == *v278 )
                      goto LABEL_387;
                    v323 = v324;
                  }
                }
              }
LABEL_393:
              switch ( v276 )
              {
                case 0u:
                case 1u:
                  v292 = *(unsigned __int8 **)(v273 - 32LL * (*(_DWORD *)(v273 + 4) & 0x7FFFFFF));
                  while ( 2 )
                  {
                    v294 = sub_BD3990(v292, (__int64)v275);
                    v293 = 23;
                    v284 = (__int64)v294;
                    v296 = *v294;
                    if ( v296 <= 0x1Cu )
                      goto LABEL_419;
                    if ( v296 != 85 )
                    {
                      v293 = 2 * (unsigned int)(v296 != 34) + 21;
LABEL_419:
                      if ( (unsigned __int8)sub_3108CA0(v293) )
                        goto LABEL_420;
                      break;
                    }
                    v297 = *(_QWORD *)(v284 - 32);
                    v293 = 21;
                    if ( !v297 || *(_BYTE *)v297 || *(_QWORD *)(v297 + 24) != *(_QWORD *)(v284 + 80) )
                      goto LABEL_419;
                    v298 = sub_3108960(*(_QWORD *)(v284 - 32), v275, v295);
                    if ( (unsigned __int8)sub_3108CA0(v298) )
                    {
LABEL_420:
                      v292 = *(unsigned __int8 **)(v284 - 32LL * (*(_DWORD *)(v284 + 4) & 0x7FFFFFF));
                      continue;
                    }
                    break;
                  }
                  v458 = (_DWORD *)v284;
                  v299 = sub_2713760((unsigned __int64 *)(v399 + 8), (__int64 *)&v458);
                  v378 |= sub_271DB10(v299, v276, v273);
                  goto LABEL_395;
                case 4u:
                  v300 = *(unsigned __int8 **)(v273 - 32LL * (*(_DWORD *)(v273 + 4) & 0x7FFFFFF));
                  while ( 2 )
                  {
                    v302 = sub_BD3990(v300, (__int64)v275);
                    v301 = 23;
                    v284 = (__int64)v302;
                    v304 = *v302;
                    if ( v304 <= 0x1Cu )
                      goto LABEL_430;
                    if ( v304 != 85 )
                    {
                      v301 = 2 * (unsigned int)(v304 != 34) + 21;
LABEL_430:
                      if ( (unsigned __int8)sub_3108CA0(v301) )
                        goto LABEL_431;
                      break;
                    }
                    v305 = *(_QWORD *)(v284 - 32);
                    v301 = 21;
                    if ( !v305 || *(_BYTE *)v305 || *(_QWORD *)(v305 + 24) != *(_QWORD *)(v284 + 80) )
                      goto LABEL_430;
                    v306 = sub_3108960(*(_QWORD *)(v284 - 32), v275, v303);
                    if ( (unsigned __int8)sub_3108CA0(v306) )
                    {
LABEL_431:
                      v300 = *(unsigned __int8 **)(v284 - 32LL * (*(_DWORD *)(v284 + 4) & 0x7FFFFFF));
                      continue;
                    }
                    break;
                  }
                  v458 = (_DWORD *)v284;
                  v437 = sub_2713760((unsigned __int64 *)(v399 + 8), (__int64 *)&v458);
                  if ( !(unsigned __int8)sub_271DBD0(v437, a1 + 168, v273) )
                    goto LABEL_395;
                  v308 = *(_DWORD *)(a5 + 24);
                  if ( v308 )
                  {
                    v309 = *(_QWORD *)(a5 + 8);
                    v310 = 1;
                    v311 = ((unsigned int)v273 >> 9) ^ ((unsigned int)v273 >> 4);
                    v312 = 0;
                    v313 = (v308 - 1) & v311;
                    v314 = v309 + (v313 << 7);
                    v315 = *(_QWORD *)v314;
                    if ( v273 == *(_QWORD *)v314 )
                      goto LABEL_441;
                    while ( v315 != -4096 )
                    {
                      if ( v315 == -8192 && !v312 )
                        v312 = v314;
                      v307 = (unsigned int)(v310 + 1);
                      v313 = (v308 - 1) & ((_DWORD)v310 + (_DWORD)v313);
                      v314 = v309 + ((unsigned __int64)(unsigned int)v313 << 7);
                      v315 = *(_QWORD *)v314;
                      if ( v273 == *(_QWORD *)v314 )
                        goto LABEL_441;
                      v310 = (unsigned int)v307;
                    }
                    if ( v312 )
                      v314 = v312;
                    ++*(_QWORD *)a5;
                    v352 = *(_DWORD *)(a5 + 16) + 1;
                    if ( 4 * v352 < 3 * v308 )
                    {
                      if ( v308 - *(_DWORD *)(a5 + 20) - v352 <= v308 >> 3 )
                      {
                        sub_2712A80(a5, v308);
                        v353 = *(_DWORD *)(a5 + 24);
                        if ( !v353 )
                          goto LABEL_576;
                        v354 = v353 - 1;
                        v355 = *(_QWORD *)(a5 + 8);
                        LODWORD(v356) = v354 & v311;
                        v357 = 1;
                        v358 = 0;
                        v352 = *(_DWORD *)(a5 + 16) + 1;
                        v314 = v355 + ((unsigned __int64)(v354 & v311) << 7);
                        v313 = *(_QWORD *)v314;
                        if ( v273 != *(_QWORD *)v314 )
                        {
                          while ( v313 != -4096 )
                          {
                            if ( v313 == -8192 && !v358 )
                              v358 = v314;
                            v307 = (unsigned int)(v357 + 1);
                            v356 = v354 & (unsigned int)(v356 + v357);
                            v314 = v355 + (v356 << 7);
                            v313 = *(_QWORD *)v314;
                            if ( v273 == *(_QWORD *)v314 )
                              goto LABEL_526;
                            ++v357;
                          }
LABEL_542:
                          if ( v358 )
                            v314 = v358;
                        }
                      }
LABEL_526:
                      *(_DWORD *)(a5 + 16) = v352;
                      if ( *(_QWORD *)v314 != -4096 )
                        --*(_DWORD *)(a5 + 20);
                      *(_QWORD *)v314 = v273;
                      memset((void *)(v314 + 8), 0, 0x78u);
                      v310 = 0;
                      *(_BYTE *)(v314 + 52) = 1;
                      *(_QWORD *)(v314 + 32) = v314 + 56;
                      *(_QWORD *)(v314 + 40) = 2;
                      *(_QWORD *)(v314 + 80) = v314 + 104;
                      *(_QWORD *)(v314 + 88) = 2;
                      *(_BYTE *)(v314 + 100) = 1;
LABEL_441:
                      v316 = v314 + 8;
                      *(_BYTE *)(v314 + 8) = *(_BYTE *)(v437 + 8);
                      *(_BYTE *)(v314 + 9) = *(_BYTE *)(v437 + 9);
                      *(_QWORD *)(v314 + 16) = *(_QWORD *)(v437 + 16);
                      if ( v437 + 24 != v314 + 24 )
                        sub_C8CE00(v314 + 24, v314 + 56, v437 + 24, v310, v313, v307);
                      if ( v437 + 72 != v316 + 64 )
                        sub_C8CE00(v316 + 64, v316 + 96, v437 + 72, v310, v313, v307);
                      *(_BYTE *)(v316 + 112) = *(_BYTE *)(v437 + 120);
                      sub_271D520(v437, 0);
LABEL_395:
                      v436 = *(_QWORD **)(v399 + 48);
                      if ( *(_QWORD **)(v399 + 40) != v436 )
                      {
                        v285 = *(_QWORD **)(v399 + 40);
                        do
                        {
                          if ( *v285 != v284 )
                          {
                            v404 = *v285;
                            if ( !(unsigned __int8)sub_271DD10(v285 + 1, v273, *v285, v369, v276, *(_QWORD *)(a1 + 200)) )
                              sub_271DE80(v285 + 1, v273, v404, v369, v276);
                          }
                          v285 += 17;
                        }
                        while ( v436 != v285 );
LABEL_401:
                        v286 = *(_QWORD *)(v399 + 48);
                        v287 = *(_QWORD *)(v399 + 40);
LABEL_402:
                        if ( (unsigned int)qword_4FF9AC8 < -252645135 * (unsigned int)((v286 - v287) >> 3) )
                        {
                          v232 = 0;
                          *(_BYTE *)(a1 + 208) = 1;
                          goto LABEL_408;
                        }
                      }
LABEL_403:
                      v396 = *(_QWORD *)(v396 + 8);
                      if ( v385 == v396 )
                      {
                        v374 |= v378;
                        v272 = v452;
                        goto LABEL_405;
                      }
                      continue;
                    }
                  }
                  else
                  {
                    ++*(_QWORD *)a5;
                  }
                  sub_2712A80(a5, 2 * v308);
                  v359 = *(_DWORD *)(a5 + 24);
                  if ( !v359 )
                  {
LABEL_576:
                    ++*(_DWORD *)(a5 + 16);
                    BUG();
                  }
                  v360 = v359 - 1;
                  v361 = *(_QWORD *)(a5 + 8);
                  LODWORD(v362) = (v359 - 1) & (((unsigned int)v273 >> 9) ^ ((unsigned int)v273 >> 4));
                  v352 = *(_DWORD *)(a5 + 16) + 1;
                  v314 = v361 + ((unsigned __int64)(unsigned int)v362 << 7);
                  v313 = *(_QWORD *)v314;
                  if ( v273 != *(_QWORD *)v314 )
                  {
                    v363 = 1;
                    v358 = 0;
                    while ( v313 != -4096 )
                    {
                      if ( !v358 && v313 == -8192 )
                        v358 = v314;
                      v307 = (unsigned int)(v363 + 1);
                      v362 = v360 & (unsigned int)(v362 + v363);
                      v314 = v361 + (v362 << 7);
                      v313 = *(_QWORD *)v314;
                      if ( v273 == *(_QWORD *)v314 )
                        goto LABEL_526;
                      ++v363;
                    }
                    goto LABEL_542;
                  }
                  goto LABEL_526;
                case 7u:
                case 0x18u:
                  goto LABEL_401;
                case 8u:
                  sub_270F800(v399 + 8);
                  v286 = *(_QWORD *)(v399 + 40);
                  v317 = *(_QWORD *)(v399 + 48);
                  if ( v286 == v317 )
                    goto LABEL_403;
                  v318 = *(_QWORD *)(v399 + 40);
                  do
                  {
LABEL_450:
                    if ( *(_BYTE *)(v318 + 108) )
                    {
                      if ( *(_BYTE *)(v318 + 60) )
                        goto LABEL_449;
                    }
                    else
                    {
                      _libc_free(*(_QWORD *)(v318 + 88));
                      if ( *(_BYTE *)(v318 + 60) )
                      {
LABEL_449:
                        v318 += 136;
                        if ( v317 == v318 )
                          break;
                        goto LABEL_450;
                      }
                    }
                    v319 = *(_QWORD *)(v318 + 40);
                    v318 += 136;
                    _libc_free(v319);
                  }
                  while ( v317 != v318 );
                  *(_QWORD *)(v399 + 48) = v286;
                  v287 = *(_QWORD *)(v399 + 40);
                  goto LABEL_402;
                default:
                  v284 = 0;
                  goto LABEL_395;
              }
            }
          }
          break;
        }
        if ( v190 )
        {
          v397 = *(_QWORD *)(v183 + 48);
          v405 = *(_QWORD *)(v183 + 56);
          if ( (unsigned __int64)v190 > 0x7FFFFFFFFFFFFF80LL )
LABEL_566:
            sub_4261EA(v113, v37, v105);
          v338 = sub_22077B0(v190);
          v187 = v405;
          v188 = v397;
          v339 = v338;
        }
        else
        {
          v339 = 0;
        }
        v340 = v188;
        v341 = v339;
        if ( v187 != v188 )
        {
          v406 = v339;
          v342 = v190;
          v343 = v183;
          v344 = v187;
          do
          {
            if ( v341 )
            {
              *(_QWORD *)v341 = *(_QWORD *)v340;
              *(_BYTE *)(v341 + 8) = *(_BYTE *)(v340 + 8);
              *(_BYTE *)(v341 + 9) = *(_BYTE *)(v340 + 9);
              *(_BYTE *)(v341 + 10) = *(_BYTE *)(v340 + 10);
              *(_BYTE *)(v341 + 16) = *(_BYTE *)(v340 + 16);
              *(_BYTE *)(v341 + 17) = *(_BYTE *)(v340 + 17);
              *(_QWORD *)(v341 + 24) = *(_QWORD *)(v340 + 24);
              sub_C8CD80(v341 + 32, v341 + 64, v340 + 32, v188, v181, v187);
              sub_C8CD80(v341 + 80, v341 + 112, v340 + 80, v345, v346, v347);
              *(_BYTE *)(v341 + 128) = *(_BYTE *)(v340 + 128);
            }
            v340 += 136;
            v341 += 136;
          }
          while ( v344 != v340 );
          v183 = v343;
          v190 = v342;
          v339 = v406;
        }
        v348 = *(_QWORD *)(v399 + 48);
        v349 = *(_QWORD *)(v399 + 40);
        if ( v348 == v349 )
        {
LABEL_509:
          if ( v349 )
            j_j___libc_free_0(v349);
          v198 = v339 + v190;
          *(_QWORD *)(v399 + 40) = v339;
          *(_QWORD *)(v399 + 56) = v198;
          goto LABEL_227;
        }
        while ( 1 )
        {
          if ( *(_BYTE *)(v349 + 108) )
          {
            if ( *(_BYTE *)(v349 + 60) )
              goto LABEL_504;
          }
          else
          {
            _libc_free(*(_QWORD *)(v349 + 88));
            if ( *(_BYTE *)(v349 + 60) )
            {
LABEL_504:
              v349 += 136LL;
              if ( v348 == v349 )
                goto LABEL_508;
              continue;
            }
          }
          v350 = *(_QWORD *)(v349 + 40);
          v349 += 136LL;
          _libc_free(v350);
          if ( v348 == v349 )
          {
LABEL_508:
            v349 = *(_QWORD *)(v399 + 40);
            goto LABEL_509;
          }
        }
      }
    }
  }
  v101 = &v461;
  while ( 1 )
  {
    v37 = (__int64)&v447;
    v447 = *((_QWORD *)v376 - 1);
    v102 = sub_270EC80(a3, &v447);
    v103 = (__int64 *)v102[19];
    v104 = (__int64)v102;
    v428 = v103;
    v384 = &v103[*((unsigned int *)v102 + 40)];
    if ( v103 == v384 )
      goto LABEL_172;
    v105 = *(unsigned int *)(a3 + 24);
    v37 = *v103;
    v106 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v105 )
    {
      v107 = (v105 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v108 = v106 + 192LL * v107;
      v109 = *(_QWORD *)v108;
      if ( v37 == *(_QWORD *)v108 )
        goto LABEL_129;
      v236 = 1;
      while ( v109 != -4096 )
      {
        v107 = (v105 - 1) & (v236 + v107);
        v108 = v106 + 192LL * v107;
        v109 = *(_QWORD *)v108;
        if ( v37 == *(_QWORD *)v108 )
          goto LABEL_129;
        ++v236;
      }
    }
    v108 = v106 + 192 * v105;
LABEL_129:
    v402 = v104 + 64;
    if ( v104 + 64 != v108 + 72 )
    {
      v37 = 16LL * *(unsigned int *)(v104 + 88);
      sub_C7D6A0(*(_QWORD *)(v104 + 72), v37, 8);
      v110 = *(unsigned int *)(v108 + 96);
      *(_DWORD *)(v104 + 88) = v110;
      if ( (_DWORD)v110 )
      {
        v111 = (void *)sub_C7D670(16 * v110, 8);
        v112 = *(unsigned int *)(v104 + 88);
        *(_QWORD *)(v104 + 72) = v111;
        *(_DWORD *)(v104 + 80) = *(_DWORD *)(v108 + 88);
        *(_DWORD *)(v104 + 84) = *(_DWORD *)(v108 + 92);
        v37 = *(_QWORD *)(v108 + 80);
        memcpy(v111, (const void *)v37, 16 * v112);
      }
      else
      {
        *(_QWORD *)(v104 + 72) = 0;
        *(_QWORD *)(v104 + 80) = 0;
      }
    }
    v371 = (__int64 *)(v104 + 96);
    v113 = v104 + 96;
    if ( v104 + 96 != v108 + 104 )
    {
      v114 = *(_QWORD *)(v108 + 112);
      v115 = *(_QWORD *)(v108 + 104);
      v116 = *(_QWORD *)(v104 + 96);
      v117 = v114 - v115;
      if ( *(_QWORD *)(v104 + 112) - v116 >= (unsigned __int64)(v114 - v115) )
      {
        v118 = *(_QWORD *)(v104 + 104);
        v119 = v118 - v116;
        v37 = v118 - v116;
        if ( v117 > (unsigned __int64)(v118 - v116) )
        {
          if ( v119 > 0 )
          {
            v254 = v115 + 32;
            v419 = v104;
            v255 = v116 + 32;
            v394 = v108;
            v256 = v254;
            v257 = 0xF0F0F0F0F0F0F0F1LL * (v119 >> 3);
            do
            {
              *(_QWORD *)(v255 - 32) = *(_QWORD *)(v256 - 32);
              *(_BYTE *)(v255 - 24) = *(_BYTE *)(v256 - 24);
              *(_BYTE *)(v255 - 23) = *(_BYTE *)(v256 - 23);
              *(_BYTE *)(v255 - 22) = *(_BYTE *)(v256 - 22);
              *(_BYTE *)(v255 - 16) = *(_BYTE *)(v256 - 16);
              *(_BYTE *)(v255 - 15) = *(_BYTE *)(v256 - 15);
              *(_QWORD *)(v255 - 8) = *(_QWORD *)(v256 - 8);
              if ( v255 != v256 )
                sub_C8CE00(v255, v255 + 32, v256, v254, v106, v114);
              if ( v256 + 48 != v255 + 48 )
                sub_C8CE00(v255 + 48, v255 + 80, v256 + 48, v254, v106, v114);
              v258 = *(_BYTE *)(v256 + 96);
              v255 += 136;
              v256 += 136;
              *(_BYTE *)(v255 - 40) = v258;
              --v257;
            }
            while ( v257 );
            v104 = v419;
            v108 = v394;
            v118 = *(_QWORD *)(v419 + 104);
            v116 = *(_QWORD *)(v419 + 96);
            v114 = *(_QWORD *)(v394 + 112);
            v115 = *(_QWORD *)(v394 + 104);
            v37 = v118 - v116;
          }
          v259 = v37 + v115;
          if ( v259 == v114 )
          {
            v253 = v116 + v117;
          }
          else
          {
            v420 = v108;
            v260 = v118;
            v261 = v259;
            v395 = v101;
            v262 = v117;
            v263 = v104;
            v264 = v114;
            do
            {
              if ( v260 )
              {
                *(_QWORD *)v260 = *(_QWORD *)v261;
                *(_BYTE *)(v260 + 8) = *(_BYTE *)(v261 + 8);
                *(_BYTE *)(v260 + 9) = *(_BYTE *)(v261 + 9);
                *(_BYTE *)(v260 + 10) = *(_BYTE *)(v261 + 10);
                *(_BYTE *)(v260 + 16) = *(_BYTE *)(v261 + 16);
                *(_BYTE *)(v260 + 17) = *(_BYTE *)(v261 + 17);
                *(_QWORD *)(v260 + 24) = *(_QWORD *)(v261 + 24);
                sub_C8CD80(v260 + 32, v260 + 64, v261 + 32, v259, v106, v114);
                v37 = v260 + 112;
                sub_C8CD80(v260 + 80, v260 + 112, v261 + 80, v265, v266, v267);
                *(_BYTE *)(v260 + 128) = *(_BYTE *)(v261 + 128);
              }
              v261 += 136;
              v260 += 136;
            }
            while ( v261 != v264 );
            v104 = v263;
            v108 = v420;
            v268 = v262;
            v101 = v395;
            v253 = *(_QWORD *)(v104 + 96) + v268;
          }
        }
        else
        {
          if ( v117 > 0 )
          {
            v380 = *(_QWORD *)(v104 + 104);
            v120 = v115 + 32;
            v121 = v116 + 32;
            v411 = v104;
            v122 = 0xF0F0F0F0F0F0F0F1LL * (v117 >> 3);
            v387 = v117;
            v123 = v116 + 32;
            v124 = v120;
            v377 = v108;
            v125 = v122;
            do
            {
              *(_QWORD *)(v123 - 32) = *(_QWORD *)(v124 - 32);
              *(_BYTE *)(v123 - 24) = *(_BYTE *)(v124 - 24);
              *(_BYTE *)(v123 - 23) = *(_BYTE *)(v124 - 23);
              *(_BYTE *)(v123 - 22) = *(_BYTE *)(v124 - 22);
              *(_BYTE *)(v123 - 16) = *(_BYTE *)(v124 - 16);
              *(_BYTE *)(v123 - 15) = *(_BYTE *)(v124 - 15);
              *(_QWORD *)(v123 - 8) = *(_QWORD *)(v124 - 8);
              if ( v123 != v124 )
              {
                v37 = v123 + 32;
                sub_C8CE00(v123, v123 + 32, v124, v121, v120, v122);
              }
              if ( v124 + 48 != v123 + 48 )
              {
                v37 = v123 + 80;
                sub_C8CE00(v123 + 48, v123 + 80, v124 + 48, v121, v120, v122);
              }
              v126 = *(_BYTE *)(v124 + 96);
              v123 += 136;
              v124 += 136;
              *(_BYTE *)(v123 - 40) = v126;
              --v125;
            }
            while ( v125 );
            v117 = v387;
            v104 = v411;
            v118 = v380;
            v108 = v377;
            v116 += v387;
          }
          if ( v118 != v116 )
          {
            v412 = v108;
            v127 = v118;
            while ( 1 )
            {
              if ( *(_BYTE *)(v116 + 108) )
              {
                if ( *(_BYTE *)(v116 + 60) )
                  goto LABEL_146;
LABEL_149:
                v128 = *(_QWORD *)(v116 + 40);
                v116 += 136;
                _libc_free(v128);
                if ( v127 == v116 )
                {
LABEL_150:
                  v108 = v412;
                  *(_QWORD *)(v104 + 104) = *(_QWORD *)(v104 + 96) + v117;
                  goto LABEL_151;
                }
              }
              else
              {
                _libc_free(*(_QWORD *)(v116 + 88));
                if ( !*(_BYTE *)(v116 + 60) )
                  goto LABEL_149;
LABEL_146:
                v116 += 136;
                if ( v127 == v116 )
                  goto LABEL_150;
              }
            }
          }
          v253 = *(_QWORD *)(v104 + 96) + v117;
        }
        goto LABEL_344;
      }
      if ( v117 )
      {
        if ( (unsigned __int64)v117 > 0x7FFFFFFFFFFFFF80LL )
          goto LABEL_566;
        v391 = *(_QWORD *)(v108 + 112);
        v416 = *(_QWORD *)(v108 + 104);
        v239 = sub_22077B0(v391 - v115);
        v115 = v416;
        v114 = v391;
        v240 = v239;
      }
      else
      {
        v240 = 0;
      }
      v241 = v240;
      if ( v114 != v115 )
      {
        v417 = v240;
        v242 = v104;
        v243 = v115;
        v392 = v108;
        v244 = v241;
        v383 = v101;
        v245 = v117;
        v246 = v114;
        do
        {
          if ( v244 )
          {
            *(_QWORD *)v244 = *(_QWORD *)v243;
            *(_BYTE *)(v244 + 8) = *(_BYTE *)(v243 + 8);
            *(_BYTE *)(v244 + 9) = *(_BYTE *)(v243 + 9);
            *(_BYTE *)(v244 + 10) = *(_BYTE *)(v243 + 10);
            *(_BYTE *)(v244 + 16) = *(_BYTE *)(v243 + 16);
            *(_BYTE *)(v244 + 17) = *(_BYTE *)(v243 + 17);
            *(_QWORD *)(v244 + 24) = *(_QWORD *)(v243 + 24);
            sub_C8CD80(v244 + 32, v244 + 64, v243 + 32, v115, v106, v114);
            v37 = v244 + 112;
            sub_C8CD80(v244 + 80, v244 + 112, v243 + 80, v247, v248, v249);
            *(_BYTE *)(v244 + 128) = *(_BYTE *)(v243 + 128);
          }
          v243 += 136;
          v244 += 136;
        }
        while ( v246 != v243 );
        v104 = v242;
        v117 = v245;
        v240 = v417;
        v108 = v392;
        v101 = v383;
      }
      v250 = *(_QWORD *)(v104 + 96);
      v418 = *(_QWORD *)(v104 + 104);
      if ( v418 == v250 )
      {
LABEL_341:
        if ( v250 )
        {
          v37 = *(_QWORD *)(v104 + 112) - v250;
          j_j___libc_free_0(v250);
        }
        v253 = v240 + v117;
        *(_QWORD *)(v104 + 96) = v240;
        *(_QWORD *)(v104 + 112) = v253;
LABEL_344:
        *(_QWORD *)(v104 + 104) = v253;
        goto LABEL_151;
      }
      v393 = v108;
      v251 = *(_QWORD *)(v104 + 96);
      while ( 1 )
      {
        if ( *(_BYTE *)(v251 + 108) )
        {
          if ( *(_BYTE *)(v251 + 60) )
            goto LABEL_336;
LABEL_339:
          v252 = *(_QWORD *)(v251 + 40);
          v251 += 136;
          _libc_free(v252);
          if ( v418 == v251 )
          {
LABEL_340:
            v108 = v393;
            v250 = *(_QWORD *)(v104 + 96);
            goto LABEL_341;
          }
        }
        else
        {
          _libc_free(*(_QWORD *)(v251 + 88));
          if ( !*(_BYTE *)(v251 + 60) )
            goto LABEL_339;
LABEL_336:
          v251 += 136;
          if ( v418 == v251 )
            goto LABEL_340;
        }
      }
    }
LABEL_151:
    v129 = *(_DWORD *)(v108 + 12);
    *(_DWORD *)(v104 + 4) = v129;
    v388 = v428 + 1;
    if ( v384 != v428 + 1 )
    {
      while ( 1 )
      {
        v37 = *v388;
        v130 = *(_QWORD *)(a3 + 8);
        v131 = *(unsigned int *)(a3 + 24);
        if ( (_DWORD)v131 )
        {
          LODWORD(v132) = (v131 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v133 = (__int64 *)(v130 + 192LL * (unsigned int)v132);
          v134 = *v133;
          if ( v37 == *v133 )
            goto LABEL_154;
          v215 = 1;
          while ( v134 != -4096 )
          {
            v132 = ((_DWORD)v131 - 1) & (unsigned int)(v132 + v215);
            v133 = (__int64 *)(v130 + 192 * v132);
            v134 = *v133;
            if ( v37 == *v133 )
              goto LABEL_154;
            ++v215;
          }
        }
        v133 = (__int64 *)(v130 + 192 * v131);
LABEL_154:
        if ( v129 == -1 )
          goto LABEL_246;
        v135 = v129 + *((_DWORD *)v133 + 3);
        *(_DWORD *)(v104 + 4) = v135;
        if ( v135 == -1 )
        {
          sub_270F800(v402);
          v216 = *(_QWORD *)(v104 + 96);
          v230 = *(_QWORD *)(v104 + 104);
          if ( v216 == v230 )
            goto LABEL_246;
          v231 = *(_QWORD *)(v104 + 96);
          while ( 2 )
          {
            if ( *(_BYTE *)(v231 + 108) )
            {
              if ( !*(_BYTE *)(v231 + 60) )
                goto LABEL_297;
            }
            else
            {
              _libc_free(*(_QWORD *)(v231 + 88));
              if ( !*(_BYTE *)(v231 + 60) )
LABEL_297:
                _libc_free(*(_QWORD *)(v231 + 40));
            }
            v231 += 136;
            if ( v230 == v231 )
            {
LABEL_263:
              *(_QWORD *)(v104 + 104) = v216;
              goto LABEL_246;
            }
            continue;
          }
        }
        if ( v135 < *((_DWORD *)v133 + 3) )
        {
          *(_DWORD *)(v104 + 4) = -1;
          sub_270F800(v402);
          v216 = *(_QWORD *)(v104 + 96);
          v217 = *(_QWORD *)(v104 + 104);
          if ( v216 == v217 )
            goto LABEL_246;
          v218 = *(_QWORD *)(v104 + 96);
          while ( 1 )
          {
            if ( *(_BYTE *)(v218 + 108) )
            {
              if ( *(_BYTE *)(v218 + 60) )
                goto LABEL_259;
            }
            else
            {
              _libc_free(*(_QWORD *)(v218 + 88));
              if ( *(_BYTE *)(v218 + 60) )
              {
LABEL_259:
                v218 += 136;
                if ( v217 == v218 )
                  goto LABEL_263;
                continue;
              }
            }
            v219 = *(_QWORD *)(v218 + 40);
            v218 += 136;
            _libc_free(v219);
            if ( v217 == v218 )
              goto LABEL_263;
          }
        }
        v413 = v133[14];
        for ( k = v133[13]; v413 != k; k += 136 )
        {
          while ( 1 )
          {
            v142 = *(_QWORD *)k;
            v459 = 0;
            v458 = (_DWORD *)v142;
            sub_2712ED0((__int64)v101, v402, (__int64 *)&v458, &v459);
            if ( (_BYTE)v465 )
            {
              v144 = *(_QWORD *)(v104 + 104) - *(_QWORD *)(v104 + 96);
              v145 = 0xF0F0F0F0F0F0F0F1LL * (v144 >> 3);
              *(_QWORD *)(v463 + 8) = v145;
              v146 = *(_QWORD *)(v104 + 104);
              if ( v146 == *(_QWORD *)(v104 + 112) )
              {
                v435 = v144;
                sub_270FD10(v371, *(_QWORD *)(v104 + 104), k, v145, v144, v146);
                v144 = v435;
              }
              else
              {
                if ( v146 )
                {
                  v381 = v144;
                  v430 = *(_QWORD *)(v104 + 104);
                  *(_QWORD *)v146 = *(_QWORD *)k;
                  *(_BYTE *)(v146 + 8) = *(_BYTE *)(k + 8);
                  *(_BYTE *)(v146 + 9) = *(_BYTE *)(k + 9);
                  *(_BYTE *)(v146 + 10) = *(_BYTE *)(k + 10);
                  *(_BYTE *)(v146 + 16) = *(_BYTE *)(k + 16);
                  *(_BYTE *)(v146 + 17) = *(_BYTE *)(k + 17);
                  *(_QWORD *)(v146 + 24) = *(_QWORD *)(k + 24);
                  sub_C8CD80(v146 + 32, v146 + 64, k + 32, v145, v144, v146);
                  sub_C8CD80(v430 + 80, v430 + 112, k + 80, v147, v148, v430);
                  v144 = v381;
                  *(_BYTE *)(v430 + 128) = *(_BYTE *)(k + 128);
                  v146 = *(_QWORD *)(v104 + 104);
                }
                *(_QWORD *)(v104 + 104) = v146 + 136;
              }
              v149 = *(_QWORD *)(v104 + 96) + v144;
              v465 = v469;
              v461 = 0;
              v141 = v149 + 8;
              v462 = 0;
              v463 = 0;
              v464 = 0;
              v466 = 2;
              v467 = 0;
              v468 = 1;
              v470 = 0;
              v471 = v475;
              v472 = 2;
              v473 = 0;
              v474 = 1;
              v476 = 0;
            }
            else
            {
              v137 = *(_QWORD *)(v104 + 96);
              v429 = v137 + 136LL * *(_QWORD *)(v463 + 8) + 8;
              LOBYTE(v461) = *(_BYTE *)(k + 8);
              *(_WORD *)((char *)&v461 + 1) = *(_WORD *)(k + 9);
              LOWORD(v462) = *(_WORD *)(k + 16);
              v463 = *(_QWORD *)(k + 24);
              sub_C8CD80((__int64)&v464, (__int64)v469, k + 32, v137, v429, v143);
              sub_C8CD80((__int64)&v470, (__int64)v475, k + 80, v138, v139, v140);
              v141 = v429;
              v476 = *(_BYTE *)(k + 128);
            }
            v37 = (__int64)v101;
            sub_271D550(v141, v101, 0);
            if ( !v474 )
              _libc_free((unsigned __int64)v471);
            if ( !v468 )
              break;
            k += 136;
            if ( v413 == k )
              goto LABEL_235;
          }
          _libc_free((unsigned __int64)v465);
        }
LABEL_235:
        v206 = *(__int64 **)(v104 + 104);
        for ( m = *(__int64 **)(v104 + 96); v206 != m; m += 17 )
        {
          while ( 1 )
          {
            v211 = *((unsigned int *)v133 + 24);
            v37 = *m;
            v212 = v133[10];
            if ( !(_DWORD)v211 )
              goto LABEL_242;
            v208 = (v211 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
            v209 = (__int64 *)(v212 + 16LL * v208);
            v210 = *v209;
            if ( *v209 == v37 )
              break;
            v213 = 1;
            while ( v210 != -4096 )
            {
              v214 = v213 + 1;
              v208 = (v211 - 1) & (v213 + v208);
              v209 = (__int64 *)(v212 + 16LL * v208);
              v210 = *v209;
              if ( v37 == *v209 )
                goto LABEL_238;
              v213 = v214;
            }
LABEL_242:
            v37 = (__int64)v101;
            v465 = v469;
            v461 = 0;
            v462 = 0;
            v463 = 0;
            v464 = 0;
            v466 = 2;
            v467 = 0;
            v468 = 1;
            v470 = 0;
            v471 = v475;
            v472 = 2;
            v473 = 0;
            v474 = 1;
            v476 = 0;
            sub_271D550(m + 1, v101, 0);
            if ( !v474 )
              _libc_free((unsigned __int64)v471);
            if ( v468 )
              goto LABEL_240;
            m += 17;
            _libc_free((unsigned __int64)v465);
            if ( v206 == m )
              goto LABEL_246;
          }
LABEL_238:
          if ( v209 == (__int64 *)(v212 + 16 * v211) || v133[14] == v133[13] + 136 * v209[1] )
            goto LABEL_242;
LABEL_240:
          ;
        }
LABEL_246:
        if ( v384 == ++v388 )
          break;
        v129 = *(_DWORD *)(v104 + 4);
      }
    }
LABEL_172:
    v150 = 0;
    if ( v447 + 48 != *(_QWORD *)(v447 + 56) )
      break;
LABEL_179:
    v153 = *(_QWORD *)(v104 + 120);
    v154 = v153 + 8LL * *(unsigned int *)(v104 + 128);
    if ( v153 == v154 )
    {
      v375 |= v150;
      v158 = *(_BYTE *)(a1 + 208);
    }
    else
    {
      v432 = v101;
      v155 = v104;
      v156 = *(_QWORD *)(v104 + 120);
      do
      {
        v157 = *(_QWORD *)(*(_QWORD *)v156 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v157 )
          goto LABEL_578;
        if ( *(_BYTE *)(v157 - 24) == 34 )
        {
          v37 = v157 - 24;
          v150 |= sub_27148F0(a1, v157 - 24, v447, a4, v155);
        }
        v156 += 8;
      }
      while ( v154 != v156 );
      v375 |= v150;
      v101 = v432;
      v158 = *(_BYTE *)(a1 + 208);
    }
    if ( v158 )
    {
      v232 = 0;
      goto LABEL_299;
    }
    v376 -= 8;
    if ( (_BYTE *)v370 == v376 )
      goto LABEL_188;
  }
  v431 = *(_QWORD **)(v447 + 56);
  v151 = (_QWORD *)(v447 + 48);
  while ( 1 )
  {
    while ( 1 )
    {
      v152 = *v151 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v152 )
        goto LABEL_578;
      v37 = v152 - 24;
      if ( *(_BYTE *)(v152 - 24) != 34 )
        break;
      v151 = (_QWORD *)(*v151 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v431 == (_QWORD *)v152 )
        goto LABEL_179;
    }
    v150 |= sub_27148F0(a1, v37, v447, a4, v104);
    if ( (unsigned int)qword_4FF9AC8 < -252645135
                                     * (unsigned int)((__int64)(*(_QWORD *)(v104 + 104) - *(_QWORD *)(v104 + 96)) >> 3) )
      break;
    v151 = (_QWORD *)(*v151 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v431 == v151 )
      goto LABEL_179;
  }
  v232 = 0;
  *(_BYTE *)(a1 + 208) = 1;
LABEL_299:
  if ( v444 != v446 )
    _libc_free((unsigned __int64)v444);
  if ( v441 != v443 )
    _libc_free((unsigned __int64)v441);
  return v232;
}
