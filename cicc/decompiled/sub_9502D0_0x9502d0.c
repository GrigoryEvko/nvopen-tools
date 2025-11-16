// Function: sub_9502D0
// Address: 0x9502d0
//
__int64 __fastcall sub_9502D0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 *a5)
{
  int v7; // r14d
  int v8; // ecx
  __int64 v9; // rdx
  int i; // eax
  __int64 v11; // rdx
  int j; // eax
  signed int v13; // r14d
  int v14; // eax
  bool v15; // cc
  __int64 result; // rax
  __int64 v17; // rax
  char m; // dl
  __int64 k; // rax
  _BYTE *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // r14d
  int v25; // eax
  __int64 v26; // r13
  bool v27; // bl
  _BYTE *v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // rbx
  __m128i *v31; // rax
  _BYTE *v32; // rsi
  __int64 v33; // rdx
  _BYTE *v34; // rax
  _BYTE *v35; // rsi
  __int64 v36; // rdx
  _BYTE *v37; // rax
  _BYTE *v38; // rsi
  _BYTE *v39; // r8
  _BYTE *v40; // rsi
  __int64 v41; // rax
  char *v42; // r15
  __int64 v43; // r13
  __m128i *v44; // rax
  _BYTE *v45; // rsi
  __int64 v46; // rax
  _BYTE *v47; // rsi
  char *v48; // rax
  _BYTE *v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r13
  __m128i *v52; // rax
  __int64 v53; // r8
  __m128i *v54; // rax
  _BYTE *v55; // rsi
  __int64 v56; // rax
  _BYTE *v57; // rsi
  __m128i *v58; // rax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // rax
  _BYTE *v62; // rsi
  __int64 v63; // rdx
  _BYTE *v64; // rsi
  _BYTE *v65; // rsi
  _BYTE *v66; // rsi
  __int64 v67; // rdi
  char *v68; // r15
  char *v69; // rbx
  char *v70; // rax
  _BYTE *v71; // rdx
  __int64 v72; // rcx
  int v73; // r13d
  __int64 v74; // rax
  int v75; // ecx
  unsigned __int64 v76; // rsi
  int v77; // r15d
  unsigned int **v78; // r13
  __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // r13
  __int64 v82; // rdx
  __int64 v83; // rsi
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdi
  _BYTE *v88; // rbx
  _BYTE *v89; // r13
  __int64 (__fastcall *v90)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v91; // rax
  __int64 v92; // r13
  int v93; // r14d
  int v94; // eax
  __int64 v95; // r15
  int v96; // eax
  __int64 v97; // rdi
  __int64 v98; // rax
  int v99; // r15d
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned __int8 v105; // al
  __int64 v106; // rax
  __int64 v107; // rbx
  unsigned int *v108; // r12
  unsigned int *v109; // rbx
  __int64 v110; // rdx
  __int64 v111; // rsi
  size_t v112; // rax
  size_t v113; // rbx
  _QWORD *v114; // rdx
  _BYTE *v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // rdx
  __int64 v118; // rbx
  __m128i *v119; // rax
  __int64 v120; // rbx
  __int64 v121; // rax
  _BYTE *v122; // rax
  __int64 v123; // rsi
  __m128i *v124; // r15
  __int64 v125; // rax
  _BYTE *v126; // rax
  __int64 v127; // rsi
  __int64 v128; // rcx
  __int64 v129; // rcx
  __int64 v130; // rcx
  __int64 v131; // r13
  __m128i *v132; // rax
  __int64 v133; // rax
  _BYTE *v134; // rax
  __m128i *v135; // rax
  __int64 v136; // rax
  _BYTE *v137; // rax
  _BYTE *v138; // rsi
  __int64 v139; // rdx
  _BYTE *v140; // rax
  _BYTE *v141; // rsi
  __int64 v142; // rdx
  _BYTE *v143; // rax
  _BYTE *v144; // rsi
  _BYTE *v145; // r8
  _BYTE *v146; // rsi
  _BYTE *v147; // rax
  char *v148; // r15
  char *v149; // rax
  _BYTE *v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // rax
  _BYTE *v153; // rax
  __int64 v154; // rax
  _QWORD *v155; // rdi
  __int64 v156; // rcx
  _BYTE *v157; // [rsp+38h] [rbp-498h]
  __m128i *v158; // [rsp+38h] [rbp-498h]
  __int64 v159; // [rsp+38h] [rbp-498h]
  __int64 v160; // [rsp+48h] [rbp-488h]
  __int64 v161; // [rsp+50h] [rbp-480h]
  int v162; // [rsp+50h] [rbp-480h]
  unsigned int v163; // [rsp+8Ch] [rbp-444h]
  int v164; // [rsp+98h] [rbp-438h]
  unsigned int v165; // [rsp+9Ch] [rbp-434h]
  _QWORD *v166; // [rsp+A0h] [rbp-430h]
  __int64 v167; // [rsp+A0h] [rbp-430h]
  __m128i *v168; // [rsp+A0h] [rbp-430h]
  int v169; // [rsp+A8h] [rbp-428h]
  char *v170; // [rsp+A8h] [rbp-428h]
  _QWORD *v171; // [rsp+A8h] [rbp-428h]
  char *v172; // [rsp+A8h] [rbp-428h]
  __int64 v173; // [rsp+A8h] [rbp-428h]
  bool v175; // [rsp+B8h] [rbp-418h]
  char *v176; // [rsp+B8h] [rbp-418h]
  char *v177; // [rsp+B8h] [rbp-418h]
  __int64 v178; // [rsp+B8h] [rbp-418h]
  __int64 v179; // [rsp+B8h] [rbp-418h]
  int v181; // [rsp+C0h] [rbp-410h]
  _QWORD *v183; // [rsp+C8h] [rbp-408h]
  __int64 v184; // [rsp+C8h] [rbp-408h]
  __int64 v185; // [rsp+C8h] [rbp-408h]
  bool v186; // [rsp+D3h] [rbp-3FDh] BYREF
  int v187; // [rsp+D4h] [rbp-3FCh] BYREF
  __int64 v188; // [rsp+D8h] [rbp-3F8h] BYREF
  _QWORD *v189; // [rsp+E0h] [rbp-3F0h] BYREF
  __int64 v190; // [rsp+E8h] [rbp-3E8h] BYREF
  __m128i *v191; // [rsp+F0h] [rbp-3E0h] BYREF
  __int64 v192; // [rsp+F8h] [rbp-3D8h] BYREF
  __int64 v193; // [rsp+100h] [rbp-3D0h] BYREF
  __m128i *v194; // [rsp+108h] [rbp-3C8h] BYREF
  __int64 v195; // [rsp+110h] [rbp-3C0h] BYREF
  __int64 v196; // [rsp+118h] [rbp-3B8h] BYREF
  _BYTE *v197; // [rsp+120h] [rbp-3B0h] BYREF
  __m128i *v198; // [rsp+128h] [rbp-3A8h] BYREF
  __int64 v199; // [rsp+130h] [rbp-3A0h] BYREF
  __int64 v200; // [rsp+138h] [rbp-398h] BYREF
  __int64 v201; // [rsp+140h] [rbp-390h] BYREF
  __m128i *v202; // [rsp+148h] [rbp-388h] BYREF
  __int64 v203; // [rsp+150h] [rbp-380h] BYREF
  __int64 v204; // [rsp+158h] [rbp-378h] BYREF
  __int64 v205; // [rsp+160h] [rbp-370h] BYREF
  __int64 v206; // [rsp+168h] [rbp-368h] BYREF
  __int64 v207; // [rsp+170h] [rbp-360h] BYREF
  _BYTE *v208; // [rsp+178h] [rbp-358h]
  _BYTE *v209; // [rsp+180h] [rbp-350h]
  __int64 v210; // [rsp+190h] [rbp-340h] BYREF
  _BYTE *v211; // [rsp+198h] [rbp-338h]
  _BYTE *v212; // [rsp+1A0h] [rbp-330h]
  __int64 v213; // [rsp+1B0h] [rbp-320h] BYREF
  signed int v214; // [rsp+1B8h] [rbp-318h]
  __int64 *v215; // [rsp+1C0h] [rbp-310h]
  _BYTE *v216; // [rsp+1D0h] [rbp-300h] BYREF
  __int64 v217; // [rsp+1D8h] [rbp-2F8h]
  _QWORD v218[2]; // [rsp+1E0h] [rbp-2F0h] BYREF
  _BYTE *v219; // [rsp+1F0h] [rbp-2E0h] BYREF
  __int64 v220; // [rsp+1F8h] [rbp-2D8h]
  _QWORD v221[2]; // [rsp+200h] [rbp-2D0h] BYREF
  _QWORD *v222; // [rsp+210h] [rbp-2C0h]
  __int64 v223; // [rsp+218h] [rbp-2B8h]
  _QWORD v224[2]; // [rsp+220h] [rbp-2B0h] BYREF
  _BYTE *v225; // [rsp+230h] [rbp-2A0h] BYREF
  __int64 v226; // [rsp+238h] [rbp-298h]
  _QWORD v227[2]; // [rsp+240h] [rbp-290h] BYREF
  __m128i dest; // [rsp+250h] [rbp-280h] BYREF
  _QWORD v229[2]; // [rsp+260h] [rbp-270h] BYREF
  _QWORD *v230; // [rsp+270h] [rbp-260h] BYREF
  __int64 v231; // [rsp+278h] [rbp-258h]
  _QWORD v232[2]; // [rsp+280h] [rbp-250h] BYREF
  _QWORD *v233; // [rsp+290h] [rbp-240h] BYREF
  __int64 v234; // [rsp+298h] [rbp-238h]
  _QWORD v235[2]; // [rsp+2A0h] [rbp-230h] BYREF
  _QWORD v236[4]; // [rsp+2B0h] [rbp-220h] BYREF
  __int64 v237[2]; // [rsp+2D0h] [rbp-200h] BYREF
  _QWORD v238[2]; // [rsp+2E0h] [rbp-1F0h] BYREF
  __int64 v239[2]; // [rsp+2F0h] [rbp-1E0h] BYREF
  char v240; // [rsp+300h] [rbp-1D0h] BYREF
  _QWORD *v241; // [rsp+310h] [rbp-1C0h] BYREF
  __int64 v242; // [rsp+318h] [rbp-1B8h]
  _QWORD v243[2]; // [rsp+320h] [rbp-1B0h] BYREF
  __m128i v244; // [rsp+330h] [rbp-1A0h] BYREF
  _QWORD v245[2]; // [rsp+340h] [rbp-190h] BYREF
  __m128i v246; // [rsp+350h] [rbp-180h] BYREF
  _QWORD v247[2]; // [rsp+360h] [rbp-170h] BYREF
  __m128i v248; // [rsp+370h] [rbp-160h] BYREF
  _QWORD v249[2]; // [rsp+380h] [rbp-150h] BYREF
  __m128i v250; // [rsp+390h] [rbp-140h] BYREF
  _QWORD v251[2]; // [rsp+3A0h] [rbp-130h] BYREF
  __m128i v252; // [rsp+3B0h] [rbp-120h] BYREF
  _QWORD v253[2]; // [rsp+3C0h] [rbp-110h] BYREF
  __m128i v254; // [rsp+3D0h] [rbp-100h] BYREF
  _QWORD v255[2]; // [rsp+3E0h] [rbp-F0h] BYREF
  __m128i v256; // [rsp+3F0h] [rbp-E0h] BYREF
  _QWORD v257[2]; // [rsp+400h] [rbp-D0h] BYREF
  __m128i v258; // [rsp+410h] [rbp-C0h] BYREF
  _QWORD v259[2]; // [rsp+420h] [rbp-B0h] BYREF
  __int16 v260; // [rsp+430h] [rbp-A0h]
  __m128i v261; // [rsp+440h] [rbp-90h] BYREF
  _QWORD v262[2]; // [rsp+450h] [rbp-80h] BYREF
  __int16 v263; // [rsp+460h] [rbp-70h]
  bool *v264; // [rsp+470h] [rbp-60h] BYREF
  __int64 *v265; // [rsp+478h] [rbp-58h]
  __int64 v266; // [rsp+480h] [rbp-50h]
  int *v267; // [rsp+488h] [rbp-48h]
  _QWORD **v268; // [rsp+490h] [rbp-40h]
  __int64 *v269; // [rsp+498h] [rbp-38h]

  v216 = v218;
  v219 = v221;
  v222 = v224;
  v225 = v227;
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v217 = 0;
  LOBYTE(v218[0]) = 0;
  v220 = 0;
  LOBYTE(v221[0]) = 0;
  v223 = 0;
  LOBYTE(v224[0]) = 0;
  v226 = 0;
  LOBYTE(v227[0]) = 0;
  dest.m128i_i64[0] = (__int64)v229;
  dest.m128i_i64[1] = 0;
  LOBYTE(v229[0]) = 0;
  v230 = v232;
  v231 = 0;
  LOBYTE(v232[0]) = 0;
  v233 = v235;
  v234 = 0;
  LOBYTE(v235[0]) = 0;
  v186 = unk_4D045E8 <= 0x45u;
  if ( a3 == 366 )
  {
    v187 = sub_620EE0((_WORD *)(*(_QWORD *)(a4 + 56) + 176LL), 1, &v264);
    v13 = sub_620EE0((_WORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 56LL) + 176LL), 1, &v264);
    v14 = v187;
    if ( !v186 )
      goto LABEL_22;
  }
  else
  {
    switch ( a3 )
    {
      case 417:
      case 423:
        v7 = 4;
        v8 = 3;
        goto LABEL_4;
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
        v7 = 2;
        v9 = *(_QWORD *)(a4 + 16);
        goto LABEL_6;
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
      case 429:
      case 430:
      case 431:
      case 432:
      case 433:
      case 434:
      case 435:
      case 436:
      case 437:
      case 438:
      case 439:
      case 440:
      case 441:
      case 442:
      case 443:
      case 444:
      case 445:
      case 446:
      case 447:
      case 448:
      case 449:
      case 450:
      case 451:
      case 452:
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
      case 463:
      case 464:
      case 465:
        v7 = 3;
        v8 = 2;
        goto LABEL_4;
      case 462:
        v7 = 5;
        v8 = 4;
LABEL_4:
        v9 = a4;
        for ( i = 0; i != v8; ++i )
          v9 = *(_QWORD *)(v9 + 16);
LABEL_6:
        v187 = sub_620EE0((_WORD *)(*(_QWORD *)(v9 + 56) + 176LL), 1, &v264);
        break;
      case 469:
        v91 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL) + 16LL) + 16LL);
        v92 = *(_QWORD *)(v91 + 16);
        v93 = sub_620EE0((_WORD *)(*(_QWORD *)(v91 + 56) + 176LL), 1, &v264);
        v94 = sub_620EE0((_WORD *)(*(_QWORD *)(v92 + 56) + 176LL), 1, &v264);
        if ( v94 <= v93 )
          v94 = v93;
        v7 = 7;
        v187 = v94;
        break;
      case 470:
      case 471:
      case 472:
      case 473:
        v22 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL) + 16LL);
        v23 = *(_QWORD *)(v22 + 16);
        v24 = sub_620EE0((_WORD *)(*(_QWORD *)(v22 + 56) + 176LL), 1, &v264);
        v25 = sub_620EE0((_WORD *)(*(_QWORD *)(v23 + 56) + 176LL), 1, &v264);
        if ( v25 <= v24 )
          v25 = v24;
        v7 = 6;
        v187 = v25;
        break;
      default:
        sub_91B980("unexpected atomic builtin function", 0);
    }
    v11 = a4;
    for ( j = 0; j != v7; ++j )
      v11 = *(_QWORD *)(v11 + 16);
    v13 = sub_620EE0((_WORD *)(*(_QWORD *)(v11 + 56) + 176LL), 1, &v264);
    if ( !v186 )
    {
      v14 = v187;
      if ( (unsigned int)(a3 - 423) <= 4 )
      {
        if ( v187 == 3 )
        {
LABEL_209:
          sub_948750((__int64)&v230, "release");
        }
        else
        {
          if ( v187 != 5 && v187 )
LABEL_14:
            sub_91B980("unexpected memory order.", 0);
LABEL_208:
          sub_948750((__int64)&v230, "relaxed");
        }
LABEL_15:
        v15 = v13 <= 2;
        if ( v13 != 2 )
          goto LABEL_16;
LABEL_27:
        if ( unk_4D045E8 > 0x59u )
          sub_948750((__int64)&v233, "cluster");
        else
          sub_948750((__int64)&v233, "gpu");
        goto LABEL_19;
      }
LABEL_22:
      switch ( v14 )
      {
        case 0:
          goto LABEL_208;
        case 1:
        case 2:
        case 5:
          sub_948750((__int64)&v230, "acquire");
          goto LABEL_15;
        case 3:
          goto LABEL_209;
        case 4:
          sub_948750((__int64)&v230, "acq_rel");
          goto LABEL_15;
        default:
          goto LABEL_14;
      }
    }
  }
  sub_2241130(&v230, 0, v231, "volatile", 8);
  v15 = v13 <= 2;
  if ( v13 == 2 )
    goto LABEL_27;
LABEL_16:
  if ( v15 )
  {
    if ( (unsigned int)v13 <= 1 )
    {
      sub_2241130(&v233, 0, v234, "cta", 3);
      goto LABEL_19;
    }
LABEL_255:
    sub_91B980("unexpected atomic operation scope.", 0);
  }
  if ( v13 == 3 )
  {
    sub_2241130(&v233, 0, v234, "gpu", 3);
  }
  else
  {
    if ( v13 != 4 )
      goto LABEL_255;
    sub_2241130(&v233, 0, v234, "sys", 3);
  }
LABEL_19:
  v266 = a2;
  v215 = &v188;
  v269 = &v188;
  v264 = &v186;
  v188 = a2;
  v213 = a2;
  v214 = v13;
  v265 = &v213;
  v267 = &v187;
  v268 = &v233;
  v175 = v186;
  if ( a3 != 366 )
  {
    v169 = v187;
    switch ( a3 )
    {
      case 417:
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
        sub_948750((__int64)&v225, "ld");
        break;
      case 423:
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
        sub_948750((__int64)&v225, "st");
        break;
      case 429:
      case 430:
      case 431:
      case 432:
      case 433:
      case 434:
      case 435:
      case 436:
      case 437:
      case 438:
      case 439:
      case 440:
        sub_948750((__int64)&v225, "atom.add");
        break;
      case 441:
      case 442:
        sub_948750((__int64)&v225, "atom.and");
        break;
      case 443:
      case 444:
        sub_948750((__int64)&v225, "atom.or");
        break;
      case 445:
      case 446:
        sub_948750((__int64)&v225, "atom.xor");
        break;
      case 447:
      case 448:
      case 449:
      case 450:
      case 451:
      case 452:
        sub_948750((__int64)&v225, "atom.max");
        break;
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
        sub_948750((__int64)&v225, "atom.min");
        break;
      case 462:
      case 463:
      case 464:
      case 465:
        sub_948750((__int64)&v225, "atom.exch");
        break;
      case 469:
      case 470:
      case 471:
      case 472:
      case 473:
        sub_948750((__int64)&v225, "atom.cas");
        break;
      default:
        sub_91B980("unexpected atomic builtin function.", 0);
    }
    v190 = a2;
    v236[0] = "b";
    v236[1] = "u";
    v236[2] = "s";
    v236[3] = "f";
    v189 = v236;
    switch ( a3 )
    {
      case 417:
      case 423:
      case 462:
      case 469:
        v164 = 0;
        v165 = sub_620EE0((_WORD *)(*(_QWORD *)(a4 + 56) + 176LL), 0, &v261);
        goto LABEL_62;
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
      case 463:
      case 464:
      case 465:
        for ( k = *a5; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        goto LABEL_61;
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
      case 441:
      case 442:
      case 443:
      case 444:
      case 445:
      case 446:
        for ( k = **(_QWORD **)(a4 + 16); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        goto LABEL_61;
      case 429:
      case 430:
      case 431:
      case 432:
      case 433:
      case 434:
      case 435:
      case 436:
      case 437:
      case 438:
      case 439:
      case 440:
      case 447:
      case 448:
      case 449:
      case 450:
      case 451:
      case 452:
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
        v17 = *a5;
        for ( m = *(_BYTE *)(*a5 + 140); m == 12; m = *(_BYTE *)(v17 + 140) )
          v17 = *(_QWORD *)(v17 + 160);
        v165 = *(_DWORD *)(v17 + 128);
        if ( m == 2 )
        {
          v164 = 1 - ((byte_4B6DF90[*(unsigned __int8 *)(v17 + 160)] == 0) - 1);
        }
        else
        {
          v164 = 3;
          if ( m != 3 )
          {
            v164 = 1;
            if ( m != 6 )
              sub_91B980("unexpected type.", 0);
          }
        }
        goto LABEL_62;
      case 470:
      case 471:
      case 472:
      case 473:
        for ( k = **(_QWORD **)(*(_QWORD *)(a4 + 16) + 16LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
LABEL_61:
        v164 = 0;
        v165 = *(_DWORD *)(k + 128);
LABEL_62:
        switch ( a3 )
        {
          case 417:
          case 418:
          case 419:
          case 420:
          case 421:
          case 422:
          case 429:
          case 430:
          case 431:
          case 432:
          case 433:
          case 434:
          case 435:
          case 436:
          case 437:
          case 438:
          case 439:
          case 440:
          case 441:
          case 442:
          case 443:
          case 444:
          case 445:
          case 446:
          case 447:
          case 448:
          case 449:
          case 450:
          case 451:
          case 452:
          case 453:
          case 454:
          case 455:
          case 456:
          case 457:
          case 458:
          case 463:
          case 464:
          case 465:
            v160 = 0;
            v163 = 0;
            goto LABEL_64;
          case 423:
          case 424:
          case 425:
          case 426:
          case 427:
          case 428:
            sub_949150(v261.m128i_i64, (__int64 *)&v189, v165, v164);
            sub_9487F0((__int64)&dest, (__int64)&v261);
            if ( (_QWORD *)v261.m128i_i64[0] != v262 )
              j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
            v160 = 0;
            v161 = sub_BCB120(*(_QWORD *)(a2 + 40));
            v163 = 0;
            goto LABEL_87;
          case 462:
          case 469:
          case 470:
          case 471:
          case 472:
          case 473:
            v163 = v165;
            if ( v165 > 0x10 || ((1LL << v165) & 0x10116) == 0 )
              sub_91B980("unexpected size1", 0);
            v160 = sub_948510((__int64)&v190, v165, v164);
LABEL_64:
            sub_949150(v261.m128i_i64, (__int64 *)&v189, v165, v164);
            v20 = (_BYTE *)dest.m128i_i64[0];
            if ( (_QWORD *)v261.m128i_i64[0] == v262 )
            {
              v117 = v261.m128i_i64[1];
              if ( v261.m128i_i64[1] )
              {
                if ( v261.m128i_i64[1] == 1 )
                  *(_BYTE *)dest.m128i_i64[0] = v262[0];
                else
                  memcpy((void *)dest.m128i_i64[0], v262, v261.m128i_u64[1]);
                v117 = v261.m128i_i64[1];
                v20 = (_BYTE *)dest.m128i_i64[0];
              }
              dest.m128i_i64[1] = v117;
              v20[v117] = 0;
              v20 = (_BYTE *)v261.m128i_i64[0];
            }
            else
            {
              if ( (_QWORD *)dest.m128i_i64[0] == v229 )
              {
                dest = v261;
                v229[0] = v262[0];
              }
              else
              {
                v21 = v229[0];
                dest = v261;
                v229[0] = v262[0];
                if ( v20 )
                {
                  v261.m128i_i64[0] = (__int64)v20;
                  v262[0] = v21;
                  goto LABEL_68;
                }
              }
              v261.m128i_i64[0] = (__int64)v262;
              v20 = v262;
            }
LABEL_68:
            v261.m128i_i64[1] = 0;
            *v20 = 0;
            if ( (_QWORD *)v261.m128i_i64[0] != v262 )
              j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
            if ( v164 == 3 )
            {
              if ( v165 == 4 )
              {
                v161 = sub_BCB160(*(_QWORD *)(v190 + 40));
              }
              else
              {
                if ( v165 != 8 )
                  sub_91B980("unexpected size3", 0);
                v161 = sub_BCB170(*(_QWORD *)(v190 + 40));
              }
            }
            else
            {
              switch ( v165 )
              {
                case 1u:
                  v161 = sub_BCB2B0(*(_QWORD *)(v190 + 40));
                  break;
                case 2u:
                  v161 = sub_BCB2C0(*(_QWORD *)(v190 + 40));
                  break;
                case 4u:
                  v161 = sub_BCB2D0(*(_QWORD *)(v190 + 40));
                  break;
                case 8u:
                  v161 = sub_BCB2E0(*(_QWORD *)(v190 + 40));
                  break;
                case 0x10u:
                  v161 = sub_BCB2F0(*(_QWORD *)(v190 + 40));
                  break;
                default:
                  sub_91B980("unexpected size4", 0);
              }
            }
LABEL_87:
            if ( v186 && (unsigned int)(v187 - 3) <= 2 )
              sub_94F9E0((__int64)&v261, &v213);
            break;
          default:
            sub_91B980("unexpected atomic builtin function.", 0);
        }
        return result;
      default:
        sub_91B980("unexpected atomic builtin function.", 0);
    }
    switch ( a3 )
    {
      case 417:
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
        v43 = a4;
        if ( a3 == 417 )
          v43 = *(_QWORD *)(a4 + 16);
        if ( !v175 && v169 == 5 )
        {
          if ( *v264 )
            sub_94F9E0((__int64)&v261, v265);
          else
            sub_94FDF0((__int64)&v261, &v264);
        }
        v44 = sub_92F410(a2, v43);
        v45 = v208;
        v191 = v44;
        v46 = v44->m128i_i64[1];
        v192 = v46;
        if ( v208 == v209 )
        {
          sub_918210((__int64)&v207, v208, &v192);
        }
        else
        {
          if ( v208 )
          {
            *(_QWORD *)v208 = v46;
            v45 = v208;
          }
          v208 = v45 + 8;
        }
        v47 = v211;
        if ( v211 == v212 )
        {
          sub_9281F0((__int64)&v210, v211, &v191);
        }
        else
        {
          if ( v211 )
          {
            *(_QWORD *)v211 = v191;
            v47 = v211;
          }
          v211 = v47 + 8;
        }
        v177 = sub_948310(v192);
        v48 = sub_948310(v161);
        v250.m128i_i64[1] = 1;
        LOWORD(v251[0]) = 61;
        v250.m128i_i64[0] = (__int64)v251;
        sub_94F930(&v252, (__int64)&v250, v48);
        sub_94F930(&v254, (__int64)&v252, ",");
        sub_94F930(&v256, (__int64)&v254, v177);
        sub_94F930(&v258, (__int64)&v256, ",");
        sub_94F930(&v261, (__int64)&v258, "~{memory}");
        sub_9487F0((__int64)&v219, (__int64)&v261);
        if ( (_QWORD *)v261.m128i_i64[0] != v262 )
          j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
        if ( (_QWORD *)v258.m128i_i64[0] != v259 )
          j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
        if ( (_QWORD *)v256.m128i_i64[0] != v257 )
          j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
        if ( (_QWORD *)v254.m128i_i64[0] != v255 )
          j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
        if ( (_QWORD *)v252.m128i_i64[0] != v253 )
          j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
        if ( (_QWORD *)v250.m128i_i64[0] != v251 )
          j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
        v49 = &v225[v226];
        if ( v186 )
        {
          v237[0] = (__int64)v238;
          sub_9486A0(v237, v225, (__int64)v49);
          if ( v237[1] != 0x3FFFFFFFFFFFFFFFLL )
          {
            sub_2241490(v237, ".", 1, v50);
            sub_948780(&v254, (__int64)v237, (__int64)v230, v231);
            sub_94F930(&v256, (__int64)&v254, ".");
            sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
            sub_94F930(&v261, (__int64)&v258, " $0, [$1];");
            sub_9487F0((__int64)&v216, (__int64)&v261);
            sub_2240A30(&v261);
            sub_2240A30(&v258);
            if ( (_QWORD *)v256.m128i_i64[0] != v257 )
              j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
            sub_2240A30(&v254);
            if ( (_QWORD *)v237[0] != v238 )
              j_j___libc_free_0(v237[0], v238[0] + 1LL);
            goto LABEL_149;
          }
        }
        else
        {
          v239[0] = (__int64)&v240;
          sub_9486A0(v239, v225, (__int64)v49);
          if ( v239[1] != 0x3FFFFFFFFFFFFFFFLL )
          {
            sub_2241490(v239, ".", 1, v129);
            sub_948780(&v250, (__int64)v239, (__int64)v230, v231);
            sub_94F930(&v252, (__int64)&v250, ".");
            sub_948780(&v254, (__int64)&v252, (__int64)v233, v234);
            sub_94F930(&v256, (__int64)&v254, ".");
            sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
            sub_94F930(&v261, (__int64)&v258, " $0, [$1];");
            sub_9487F0((__int64)&v216, (__int64)&v261);
            sub_2240A30(&v261);
            sub_2240A30(&v258);
            sub_2240A30(&v256);
            sub_2240A30(&v254);
            sub_2240A30(&v252);
            sub_2240A30(&v250);
            sub_2240A30(v239);
LABEL_149:
            v157 = 0;
            goto LABEL_218;
          }
        }
        goto LABEL_420;
      case 423:
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
        if ( a3 == 423 )
        {
          v131 = sub_948510((__int64)&v190, v165, v164);
          if ( v165 > 0x10 || ((1LL << v165) & 0x10116) == 0 )
            sub_91B980("unexpected size1", 0);
          v30 = *(_QWORD *)(a4 + 16);
          v132 = sub_92F410(a2, *(_QWORD *)(v30 + 16));
          v263 = 257;
          v167 = (__int64)v132;
          v133 = sub_BCE760(v131, 0);
          v134 = sub_94BCF0((unsigned int **)(a2 + 48), v167, v133, (__int64)&v261);
          v193 = sub_926480(a2, (unsigned __int64)v134, v131, v165, 0);
        }
        else
        {
          v30 = a4;
          v193 = (__int64)sub_92F410(a2, *(_QWORD *)(a4 + 16));
        }
        if ( !v175 && v169 == 5 )
        {
          if ( *v264 )
            sub_94F9E0((__int64)&v261, v265);
          else
            sub_94FDF0((__int64)&v261, &v264);
        }
        v31 = sub_92F410(a2, v30);
        v32 = v208;
        v33 = v31->m128i_i64[1];
        v194 = v31;
        v195 = v33;
        v196 = *(_QWORD *)(v193 + 8);
        v34 = v209;
        if ( v208 == v209 )
        {
          sub_918210((__int64)&v207, v208, &v195);
          v37 = v208;
          v35 = v209;
          if ( v208 != v209 )
          {
            if ( !v208 )
              goto LABEL_110;
            v36 = v196;
            v35 = v208;
LABEL_109:
            *(_QWORD *)v35 = v36;
            v37 = v208;
LABEL_110:
            v208 = v37 + 8;
            goto LABEL_111;
          }
        }
        else
        {
          if ( v208 )
          {
            *(_QWORD *)v208 = v33;
            v34 = v209;
            v32 = v208;
          }
          v35 = v32 + 8;
          v36 = v196;
          v208 = v35;
          if ( v34 != v35 )
            goto LABEL_109;
        }
        sub_918210((__int64)&v207, v35, &v196);
LABEL_111:
        v38 = v211;
        v39 = v212;
        if ( v211 == v212 )
        {
          sub_9281F0((__int64)&v210, v211, &v194);
          v40 = v211;
          v39 = v212;
          if ( v211 != v212 )
          {
            if ( !v211 )
              goto LABEL_116;
            v41 = v193;
LABEL_115:
            *(_QWORD *)v40 = v41;
            v40 = v211;
LABEL_116:
            v211 = v40 + 8;
            goto LABEL_117;
          }
        }
        else
        {
          if ( v211 )
          {
            *(_QWORD *)v211 = v194;
            v38 = v211;
            v39 = v212;
          }
          v40 = v38 + 8;
          v41 = v193;
          v211 = v40;
          if ( v40 != v39 )
            goto LABEL_115;
        }
        sub_9281F0((__int64)&v210, v39, &v193);
LABEL_117:
        v176 = sub_948310(v196);
        v42 = sub_948310(v195);
        v252.m128i_i64[0] = (__int64)v253;
        if ( !v42 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v112 = strlen(v42);
        v261.m128i_i64[0] = v112;
        v113 = v112;
        if ( v112 > 0xF )
        {
          v252.m128i_i64[0] = sub_22409D0(&v252, &v261, 0);
          v155 = (_QWORD *)v252.m128i_i64[0];
          v253[0] = v261.m128i_i64[0];
        }
        else
        {
          if ( v112 == 1 )
          {
            LOBYTE(v253[0]) = *v42;
            v114 = v253;
LABEL_260:
            v252.m128i_i64[1] = v112;
            *((_BYTE *)v114 + v112) = 0;
            sub_94F930(&v254, (__int64)&v252, ",");
            sub_94F930(&v256, (__int64)&v254, v176);
            sub_94F930(&v258, (__int64)&v256, ",");
            sub_94F930(&v261, (__int64)&v258, "~{memory}");
            sub_9487F0((__int64)&v219, (__int64)&v261);
            if ( (_QWORD *)v261.m128i_i64[0] != v262 )
              j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
            if ( (_QWORD *)v258.m128i_i64[0] != v259 )
              j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
            if ( (_QWORD *)v256.m128i_i64[0] != v257 )
              j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
            if ( (_QWORD *)v254.m128i_i64[0] != v255 )
              j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
            if ( (_QWORD *)v252.m128i_i64[0] != v253 )
              j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
            v115 = &v225[v226];
            if ( v186 )
            {
              v241 = v243;
              sub_9486A0((__int64 *)&v241, v225, (__int64)v115);
              if ( v242 != 0x3FFFFFFFFFFFFFFFLL )
              {
                sub_2241490(&v241, ".", 1, v116);
                sub_948780(&v254, (__int64)&v241, (__int64)v230, v231);
                sub_94F930(&v256, (__int64)&v254, ".");
                sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
                sub_94F930(&v261, (__int64)&v258, " [$0], $1;");
                sub_9487F0((__int64)&v216, (__int64)&v261);
                sub_2240A30(&v261);
                if ( (_QWORD *)v258.m128i_i64[0] != v259 )
                  j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
                if ( (_QWORD *)v256.m128i_i64[0] != v257 )
                  j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
                if ( (_QWORD *)v254.m128i_i64[0] != v255 )
                  j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
                sub_2240A30(&v241);
                goto LABEL_149;
              }
LABEL_420:
              sub_4262D8((__int64)"basic_string::append");
            }
            v244.m128i_i64[0] = (__int64)v245;
            v158 = &v244;
            sub_9486A0(v244.m128i_i64, v225, (__int64)v115);
            if ( v244.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
              goto LABEL_420;
            sub_2241490(&v244, ".", 1, v130);
            sub_948780(&v250, (__int64)&v244, (__int64)v230, v231);
            sub_94F930(&v252, (__int64)&v250, ".");
            sub_948780(&v254, (__int64)&v252, (__int64)v233, v234);
            sub_94F930(&v256, (__int64)&v254, ".");
            sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
            sub_94F930(&v261, (__int64)&v258, " [$0], $1;");
            sub_9487F0((__int64)&v216, (__int64)&v261);
            sub_2240A30(&v261);
            if ( (_QWORD *)v258.m128i_i64[0] != v259 )
              j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
            if ( (_QWORD *)v256.m128i_i64[0] != v257 )
              j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
            sub_2240A30(&v254);
            if ( (_QWORD *)v252.m128i_i64[0] != v253 )
              j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
            if ( (_QWORD *)v250.m128i_i64[0] != v251 )
              j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
            goto LABEL_309;
          }
          if ( !v112 )
          {
            v114 = v253;
            goto LABEL_260;
          }
          v155 = v253;
        }
        memcpy(v155, v42, v113);
        v112 = v261.m128i_i64[0];
        v114 = (_QWORD *)v252.m128i_i64[0];
        goto LABEL_260;
      case 429:
      case 430:
      case 431:
      case 432:
      case 433:
      case 434:
      case 435:
      case 436:
      case 437:
      case 438:
      case 439:
      case 440:
      case 441:
      case 442:
      case 443:
      case 444:
      case 445:
      case 446:
      case 447:
      case 448:
      case 449:
      case 450:
      case 451:
      case 452:
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
      case 462:
      case 463:
      case 464:
      case 465:
        v26 = *(_QWORD *)(a4 + 16);
        v27 = !v175 && v169 == 5;
        if ( a3 == 462 )
        {
          v135 = sub_92F410(a2, *(_QWORD *)(v26 + 16));
          v263 = 257;
          v178 = (__int64)v135;
          v136 = sub_BCE760(v160, 0);
          v137 = sub_94BCF0((unsigned int **)(a2 + 48), v178, v136, (__int64)&v261);
          v197 = (_BYTE *)sub_926480(a2, (unsigned __int64)v137, v160, v163, 0);
          if ( !v27 )
          {
            v198 = sub_92F410(a2, v26);
            v28 = v197;
            goto LABEL_316;
          }
        }
        else
        {
          v26 = a4;
          v197 = sub_92F410(a2, *(_QWORD *)(a4 + 16));
          if ( !v27 )
            goto LABEL_93;
        }
        if ( *v264 )
          sub_94F9E0((__int64)&v261, v265);
        else
          sub_94FDF0((__int64)&v261, &v264);
LABEL_93:
        v198 = sub_92F410(a2, v26);
        if ( a3 != 436 && a3 != 439 && a3 != 435 && a3 != 438 )
        {
          v28 = v197;
          if ( a3 == 437 || a3 == 440 )
          {
            v29 = (_BYTE *)sub_AD6530(*((_QWORD *)v197 + 1));
            v258.m128i_i32[1] = 0;
            v263 = 257;
            v28 = (_BYTE *)sub_94AB40((unsigned int **)(a2 + 48), v29, v197, v258.m128i_u32[0], (__int64)&v261, 0);
            v197 = v28;
          }
LABEL_316:
          v138 = v208;
          v200 = *((_QWORD *)v28 + 1);
          v139 = v198->m128i_i64[1];
          v140 = v209;
          v199 = v139;
          if ( v208 == v209 )
          {
            sub_918210((__int64)&v207, v208, &v199);
            v143 = v208;
            v141 = v209;
            if ( v208 != v209 )
            {
              if ( !v208 )
                goto LABEL_321;
              v142 = v200;
              v141 = v208;
LABEL_320:
              *(_QWORD *)v141 = v142;
              v143 = v208;
LABEL_321:
              v208 = v143 + 8;
              goto LABEL_322;
            }
          }
          else
          {
            if ( v208 )
            {
              *(_QWORD *)v208 = v139;
              v140 = v209;
              v138 = v208;
            }
            v141 = v138 + 8;
            v142 = v200;
            v208 = v141;
            if ( v140 != v141 )
              goto LABEL_320;
          }
          sub_918210((__int64)&v207, v141, &v200);
LABEL_322:
          v144 = v211;
          v145 = v212;
          if ( v211 == v212 )
          {
            sub_9281F0((__int64)&v210, v211, &v198);
            v146 = v211;
            v145 = v212;
            if ( v211 != v212 )
            {
              if ( !v211 )
                goto LABEL_327;
              v147 = v197;
LABEL_326:
              *(_QWORD *)v146 = v147;
              v146 = v211;
LABEL_327:
              v211 = v146 + 8;
LABEL_328:
              v172 = sub_948310(v200);
              v148 = sub_948310(v199);
              v149 = sub_948310(v161);
              v246.m128i_i64[1] = 1;
              v158 = &v246;
              v246.m128i_i64[0] = (__int64)v247;
              LOWORD(v247[0]) = 61;
              sub_94F930(&v248, (__int64)&v246, v149);
              sub_94F930(&v250, (__int64)&v248, ",");
              sub_94F930(&v252, (__int64)&v250, v148);
              sub_94F930(&v254, (__int64)&v252, ",");
              sub_94F930(&v256, (__int64)&v254, v172);
              sub_94F930(&v258, (__int64)&v256, ",");
              sub_94F930(&v261, (__int64)&v258, "~{memory}");
              sub_9487F0((__int64)&v219, (__int64)&v261);
              if ( (_QWORD *)v261.m128i_i64[0] != v262 )
                j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
              if ( (_QWORD *)v258.m128i_i64[0] != v259 )
                j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
              if ( (_QWORD *)v256.m128i_i64[0] != v257 )
                j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
              if ( (_QWORD *)v254.m128i_i64[0] != v255 )
                j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
              if ( (_QWORD *)v252.m128i_i64[0] != v253 )
                j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
              if ( (_QWORD *)v250.m128i_i64[0] != v251 )
                j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
              if ( (_QWORD *)v248.m128i_i64[0] != v249 )
                j_j___libc_free_0(v248.m128i_i64[0], v249[0] + 1LL);
              if ( (_QWORD *)v246.m128i_i64[0] != v247 )
                j_j___libc_free_0(v246.m128i_i64[0], v247[0] + 1LL);
              v150 = &v225[v226];
              if ( v186 )
              {
                v250.m128i_i64[0] = (__int64)v251;
                sub_9486A0(v250.m128i_i64, v225, (__int64)v150);
                if ( v250.m128i_i64[1] != 0x3FFFFFFFFFFFFFFFLL )
                {
                  sub_2241490(&v250, ".", 1, v151);
                  sub_948780(&v254, (__int64)&v250, (__int64)v233, v234);
                  sub_94F930(&v256, (__int64)&v254, ".");
                  sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
                  sub_94F930(&v261, (__int64)&v258, " $0,[$1],$2;");
                  sub_9487F0((__int64)&v216, (__int64)&v261);
                  sub_2240A30(&v261);
                  if ( (_QWORD *)v258.m128i_i64[0] != v259 )
                    j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
                  if ( (_QWORD *)v256.m128i_i64[0] != v257 )
                    j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
                  if ( (_QWORD *)v254.m128i_i64[0] != v255 )
                    j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
                  if ( (_QWORD *)v250.m128i_i64[0] != v251 )
                    j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
                  goto LABEL_149;
                }
                goto LABEL_420;
              }
              v246.m128i_i64[0] = (__int64)v247;
              sub_9486A0(v246.m128i_i64, v225, (__int64)v150);
              if ( v246.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
                goto LABEL_420;
              sub_2241490(&v246, ".", 1, v156);
              sub_948780(&v250, (__int64)&v246, (__int64)v230, v231);
              sub_94F930(&v252, (__int64)&v250, ".");
              sub_948780(&v254, (__int64)&v252, (__int64)v233, v234);
              sub_94F930(&v256, (__int64)&v254, ".");
              sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
              sub_94F930(&v261, (__int64)&v258, " $0,[$1],$2;");
              sub_9487F0((__int64)&v216, (__int64)&v261);
              sub_2240A30(&v261);
              if ( (_QWORD *)v258.m128i_i64[0] != v259 )
                j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
              if ( (_QWORD *)v256.m128i_i64[0] != v257 )
                j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
              sub_2240A30(&v254);
              if ( (_QWORD *)v252.m128i_i64[0] != v253 )
                j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
              sub_2240A30(&v250);
LABEL_309:
              sub_2240A30(v158);
              v157 = 0;
LABEL_218:
              v73 = sub_BCF480(v161, v207, (__int64)&v208[-v207] >> 3, 0);
              v258.m128i_i64[0] = (__int64)v259;
              sub_9486A0(v258.m128i_i64, v219, (__int64)&v219[v220]);
              v256.m128i_i64[0] = (__int64)v257;
              sub_9486A0(v256.m128i_i64, v216, (__int64)&v216[v217]);
              v74 = sub_B41A60(
                      v73,
                      v256.m128i_i32[0],
                      v256.m128i_i32[2],
                      v258.m128i_i32[0],
                      v258.m128i_i32[2],
                      1,
                      0,
                      0,
                      0);
              v75 = v210;
              v76 = 0;
              v77 = v74;
              v263 = 257;
              v78 = (unsigned int **)(v188 + 48);
              v79 = (__int64)&v211[-v210] >> 3;
              if ( v74 )
              {
                v162 = v210;
                v80 = sub_B3B7D0(v74, 0);
                v75 = v162;
                v76 = v80;
              }
              v81 = sub_921880(v78, v76, v77, v75, v79, (__int64)&v261, 0);
              v83 = sub_BD5C60(v81, v76, v82);
              *(_QWORD *)(v81 + 72) = sub_A7A090(v81 + 72, v83, 0xFFFFFFFFLL, 41);
              v85 = sub_BD5C60(v81, v83, v84);
              *(_QWORD *)(v81 + 72) = sub_A7A090(v81 + 72, v85, 0xFFFFFFFFLL, 6);
              if ( (_QWORD *)v256.m128i_i64[0] != v257 )
                j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
              if ( (_QWORD *)v258.m128i_i64[0] != v259 )
                j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
              if ( (unsigned int)(a3 - 469) <= 4 )
              {
                v95 = sub_926480(a2, (unsigned __int64)v157, v160, v163, 0);
                v166 = (_QWORD *)sub_945CA0(a2, (__int64)"cond.true", 0, 0);
                v183 = (_QWORD *)sub_945CA0(a2, (__int64)"cond.false", 0, 0);
                v171 = (_QWORD *)sub_945CA0(a2, (__int64)"cond.end", 0, 0);
                v261.m128i_i64[0] = (__int64)"cmp";
                v263 = 259;
                v96 = sub_92B530((unsigned int **)(a2 + 48), 0x20u, v95, (_BYTE *)v81, (__int64)&v261);
                sub_945D00(a2, v96, (int)v166, (int)v183, 0);
                v97 = *(_QWORD *)(a2 + 40);
                v263 = 259;
                v261.m128i_i64[0] = (__int64)"temp";
                v98 = sub_BCB2B0(v97);
                v99 = sub_921B80(a2, v98, (__int64)&v261, 0, 0);
                sub_92FEA0(a2, v166, 0);
                v100 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
                v101 = sub_ACD640(v100, 1, 0);
                sub_949050((unsigned int **)(a2 + 48), v101, v99, 0, 0);
                sub_92FD90(a2, (__int64)v171);
                sub_92FEA0(a2, v183, 0);
                v102 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
                v103 = sub_ACD640(v102, 0, 0);
                sub_949050((unsigned int **)(a2 + 48), v103, v99, 0, 0);
                sub_92FD90(a2, (__int64)v171);
                sub_92FEA0(a2, v171, 0);
                sub_923130(a2, v81, (unsigned __int64)v157, v163, 0);
                v260 = 257;
                v184 = sub_BCB2B0(*(_QWORD *)(a2 + 40));
                v104 = sub_AA4E30(*(_QWORD *)(a2 + 96));
                v105 = sub_AE5020(v104, v184);
                v263 = 257;
                v181 = v105;
                v106 = sub_BD2C40(80, unk_3F10A14);
                v81 = v106;
                if ( v106 )
                  sub_B4D190(v106, v184, v99, (unsigned int)&v261, 0, v181, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
                  *(_QWORD *)(a2 + 136),
                  v81,
                  &v258,
                  *(_QWORD *)(a2 + 104),
                  *(_QWORD *)(a2 + 112));
                v107 = 4LL * *(unsigned int *)(a2 + 56);
                v108 = *(unsigned int **)(a2 + 48);
                v109 = &v108[v107];
                while ( v109 != v108 )
                {
                  v110 = *((_QWORD *)v108 + 1);
                  v111 = *v108;
                  v108 += 4;
                  sub_B99FD0(v81, v111, v110);
                }
              }
              else if ( a3 == 417 )
              {
                v118 = *(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL);
                if ( v165 > 0x10 || ((1LL << v165) & 0x10116) == 0 )
                  sub_91B980("unexpected size1", 0);
                v185 = sub_948510((__int64)&v190, v165, v164);
                v119 = sub_92F410(a2, v118);
                v263 = 257;
                v120 = (__int64)v119;
                v121 = sub_BCE760(v185, 0);
                v122 = sub_94BCF0((unsigned int **)(a2 + 48), v120, v121, (__int64)&v261);
                v123 = v81;
                v81 = 0;
                sub_923130(a2, v123, (unsigned __int64)v122, v165, 0);
              }
              else if ( a3 == 462 )
              {
                v124 = sub_92F410(a2, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL));
                v263 = 257;
                v125 = sub_BCE760(v160, 0);
                v126 = sub_94BCF0((unsigned int **)(a2 + 48), (__int64)v124, v125, (__int64)&v261);
                v127 = v81;
                v81 = 0;
                sub_923130(a2, v127, (unsigned __int64)v126, v163, 0);
              }
              if ( !v186 )
                goto LABEL_231;
              if ( v187 > 2 )
              {
                if ( (unsigned int)(v187 - 4) > 1 )
                  goto LABEL_231;
              }
              else if ( v187 <= 0 )
              {
LABEL_231:
                *(_BYTE *)(a1 + 12) &= ~1u;
                *(_QWORD *)a1 = v81;
                *(_DWORD *)(a1 + 8) = 0;
                *(_DWORD *)(a1 + 16) = 0;
                goto LABEL_31;
              }
              sub_94F9E0((__int64)&v261, &v213);
              goto LABEL_231;
            }
          }
          else
          {
            if ( v211 )
            {
              *(_QWORD *)v211 = v198;
              v144 = v211;
              v145 = v212;
            }
            v146 = v144 + 8;
            v147 = v197;
            v211 = v146;
            if ( v146 != v145 )
              goto LABEL_326;
          }
          sub_9281F0((__int64)&v210, v145, &v197);
          goto LABEL_328;
        }
        v86 = sub_AD6530(*((_QWORD *)v197 + 1));
        v87 = *(_QWORD *)(a2 + 128);
        v88 = v197;
        v89 = (_BYTE *)v86;
        v260 = 257;
        v90 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v87 + 32LL);
        if ( v90 == sub_9201A0 )
        {
          if ( *v89 > 0x15u || *v197 > 0x15u )
            goto LABEL_363;
          if ( (unsigned __int8)sub_AC47B0(15) )
            v28 = (_BYTE *)sub_AD5570(15, v89, v88, 2, 0);
          else
            v28 = (_BYTE *)sub_AABE40(15, v89, v88);
        }
        else
        {
          v28 = (_BYTE *)v90(v87, 15u, v89, v197, 0, 1);
        }
        if ( v28 )
        {
LABEL_244:
          v197 = v28;
          goto LABEL_316;
        }
LABEL_363:
        v263 = 257;
        v173 = sub_B504D0(15, v89, v88, &v261, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v173,
          &v258,
          *(_QWORD *)(a2 + 104),
          *(_QWORD *)(a2 + 112));
        sub_94AAF0((unsigned int **)(a2 + 48), v173);
        sub_B44850(v173, 1);
        v28 = (_BYTE *)v173;
        goto LABEL_244;
      case 469:
      case 470:
      case 471:
      case 472:
      case 473:
        v51 = *(_QWORD *)(a4 + 16);
        if ( a3 == 469 )
        {
          v159 = *(_QWORD *)(v51 + 16);
          v168 = sub_92F410(a2, *(_QWORD *)(v159 + 16));
          v263 = 257;
          v152 = sub_BCE760(v160, 0);
          v153 = sub_94BCF0((unsigned int **)(a2 + 48), (__int64)v168, v152, (__int64)&v261);
          v154 = sub_926480(a2, (unsigned __int64)v153, v160, v163, 0);
          v53 = v51;
          v201 = v154;
          v51 = v159;
        }
        else
        {
          v52 = sub_92F410(a2, *(_QWORD *)(v51 + 16));
          v53 = a4;
          v201 = (__int64)v52;
        }
        if ( !v175 && v169 == 5 )
        {
          v179 = v53;
          if ( *v264 )
            sub_94F9E0((__int64)&v261, v265);
          else
            sub_94FDF0((__int64)&v261, &v264);
          v53 = v179;
        }
        v54 = sub_92F410(a2, v53);
        v55 = v208;
        v202 = v54;
        v56 = v54->m128i_i64[1];
        v203 = v56;
        if ( v208 == v209 )
        {
          sub_918210((__int64)&v207, v208, &v203);
        }
        else
        {
          if ( v208 )
          {
            *(_QWORD *)v208 = v56;
            v55 = v208;
          }
          v208 = v55 + 8;
        }
        v57 = v211;
        if ( v211 == v212 )
        {
          sub_9281F0((__int64)&v210, v211, &v202);
        }
        else
        {
          if ( v211 )
          {
            *(_QWORD *)v211 = v202;
            v57 = v211;
          }
          v211 = v57 + 8;
        }
        v58 = sub_92F410(a2, v51);
        v263 = 257;
        v59 = (__int64)v58;
        v60 = sub_BCE760(v160, 0);
        v157 = sub_94BCF0((unsigned int **)(a2 + 48), v59, v60, (__int64)&v261);
        v61 = sub_926480(a2, (unsigned __int64)v157, v160, v163, 0);
        v62 = v211;
        v63 = *(_QWORD *)(v61 + 8);
        v204 = v61;
        v205 = v63;
        if ( v211 == v212 )
        {
          sub_9281F0((__int64)&v210, v211, &v204);
        }
        else
        {
          if ( v211 )
          {
            *(_QWORD *)v211 = v61;
            v62 = v211;
          }
          v211 = v62 + 8;
        }
        v64 = v208;
        if ( v208 == v209 )
        {
          sub_918210((__int64)&v207, v208, &v205);
        }
        else
        {
          if ( v208 )
          {
            *(_QWORD *)v208 = v205;
            v64 = v208;
          }
          v208 = v64 + 8;
        }
        v65 = v211;
        v206 = *(_QWORD *)(v201 + 8);
        if ( v211 == v212 )
        {
          sub_9281F0((__int64)&v210, v211, &v201);
        }
        else
        {
          if ( v211 )
          {
            *(_QWORD *)v211 = v201;
            v65 = v211;
          }
          v211 = v65 + 8;
        }
        v66 = v208;
        if ( v208 == v209 )
        {
          sub_918210((__int64)&v207, v208, &v206);
          v67 = v206;
        }
        else
        {
          v67 = v206;
          if ( v208 )
          {
            *(_QWORD *)v208 = v206;
            v66 = v208;
          }
          v208 = v66 + 8;
        }
        v170 = sub_948310(v67);
        v68 = sub_948310(v205);
        v69 = sub_948310(v203);
        v70 = sub_948310(v161);
        LOWORD(v243[0]) = 61;
        v241 = v243;
        v242 = 1;
        sub_94F930(&v244, (__int64)&v241, v70);
        sub_94F930(&v246, (__int64)&v244, ",");
        sub_94F930(&v248, (__int64)&v246, v69);
        sub_94F930(&v250, (__int64)&v248, ",");
        sub_94F930(&v252, (__int64)&v250, v68);
        sub_94F930(&v254, (__int64)&v252, ",");
        sub_94F930(&v256, (__int64)&v254, v170);
        sub_94F930(&v258, (__int64)&v256, ",");
        sub_94F930(&v261, (__int64)&v258, "~{memory}");
        sub_9487F0((__int64)&v219, (__int64)&v261);
        if ( (_QWORD *)v261.m128i_i64[0] != v262 )
          j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
        if ( (_QWORD *)v258.m128i_i64[0] != v259 )
          j_j___libc_free_0(v258.m128i_i64[0], v259[0] + 1LL);
        if ( (_QWORD *)v256.m128i_i64[0] != v257 )
          j_j___libc_free_0(v256.m128i_i64[0], v257[0] + 1LL);
        if ( (_QWORD *)v254.m128i_i64[0] != v255 )
          j_j___libc_free_0(v254.m128i_i64[0], v255[0] + 1LL);
        if ( (_QWORD *)v252.m128i_i64[0] != v253 )
          j_j___libc_free_0(v252.m128i_i64[0], v253[0] + 1LL);
        if ( (_QWORD *)v250.m128i_i64[0] != v251 )
          j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
        if ( (_QWORD *)v248.m128i_i64[0] != v249 )
          j_j___libc_free_0(v248.m128i_i64[0], v249[0] + 1LL);
        if ( (_QWORD *)v246.m128i_i64[0] != v247 )
          j_j___libc_free_0(v246.m128i_i64[0], v247[0] + 1LL);
        if ( (_QWORD *)v244.m128i_i64[0] != v245 )
          j_j___libc_free_0(v244.m128i_i64[0], v245[0] + 1LL);
        if ( v241 != v243 )
          j_j___libc_free_0(v241, v243[0] + 1LL);
        v71 = &v225[v226];
        if ( v186 )
        {
          v252.m128i_i64[0] = (__int64)v253;
          sub_9486A0(v252.m128i_i64, v225, (__int64)v71);
          if ( v252.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
            goto LABEL_420;
          sub_2241490(&v252, ".", 1, v72);
          sub_948780(&v254, (__int64)&v252, (__int64)v233, v234);
          sub_94F930(&v256, (__int64)&v254, ".");
          sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
          sub_94F930(&v261, (__int64)&v258, " $0,[$1],$2,$3;");
          sub_9487F0((__int64)&v216, (__int64)&v261);
          if ( (_QWORD *)v261.m128i_i64[0] != v262 )
            j_j___libc_free_0(v261.m128i_i64[0], v262[0] + 1LL);
          sub_2240A30(&v258);
          sub_2240A30(&v256);
          sub_2240A30(&v254);
          sub_2240A30(&v252);
        }
        else
        {
          v248.m128i_i64[0] = (__int64)v249;
          sub_9486A0(v248.m128i_i64, v225, (__int64)v71);
          if ( v248.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
            goto LABEL_420;
          sub_2241490(&v248, ".", 1, v128);
          sub_948780(&v250, (__int64)&v248, (__int64)v230, v231);
          sub_94F930(&v252, (__int64)&v250, ".");
          sub_948780(&v254, (__int64)&v252, (__int64)v233, v234);
          sub_94F930(&v256, (__int64)&v254, ".");
          sub_948780(&v258, (__int64)&v256, dest.m128i_i64[0], dest.m128i_i64[1]);
          sub_94F930(&v261, (__int64)&v258, " $0,[$1],$2,$3;");
          sub_9487F0((__int64)&v216, (__int64)&v261);
          sub_2240A30(&v261);
          sub_2240A30(&v258);
          sub_2240A30(&v256);
          sub_2240A30(&v254);
          sub_2240A30(&v252);
          if ( (_QWORD *)v250.m128i_i64[0] != v251 )
            j_j___libc_free_0(v250.m128i_i64[0], v251[0] + 1LL);
          sub_2240A30(&v248);
        }
        goto LABEL_218;
      default:
        sub_91B980("unexpected atomic builtin function", 0);
    }
  }
  if ( v186 )
    sub_94F9E0(a1, &v213);
  else
    sub_94FDF0(a1, &v264);
LABEL_31:
  if ( v233 != v235 )
    j_j___libc_free_0(v233, v235[0] + 1LL);
  if ( v230 != v232 )
    j_j___libc_free_0(v230, v232[0] + 1LL);
  if ( (_QWORD *)dest.m128i_i64[0] != v229 )
    j_j___libc_free_0(dest.m128i_i64[0], v229[0] + 1LL);
  if ( v225 != (_BYTE *)v227 )
    j_j___libc_free_0(v225, v227[0] + 1LL);
  if ( v222 != v224 )
    j_j___libc_free_0(v222, v224[0] + 1LL);
  if ( v219 != (_BYTE *)v221 )
    j_j___libc_free_0(v219, v221[0] + 1LL);
  if ( v216 != (_BYTE *)v218 )
    j_j___libc_free_0(v216, v218[0] + 1LL);
  if ( v210 )
    j_j___libc_free_0(v210, &v212[-v210]);
  if ( v207 )
    j_j___libc_free_0(v207, &v209[-v207]);
  return a1;
}
