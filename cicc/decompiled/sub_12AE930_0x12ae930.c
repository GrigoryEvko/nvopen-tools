// Function: sub_12AE930
// Address: 0x12ae930
//
__int64 __fastcall sub_12AE930(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 *a5)
{
  int v7; // ecx
  int v8; // r14d
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
  __int64 v20; // r13
  bool v21; // r15
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // r13d
  __int64 v30; // r13
  char *v31; // rax
  _BYTE *v32; // rsi
  __int64 v33; // rdx
  _BYTE *v34; // rax
  _BYTE *v35; // rsi
  __int64 v36; // rdx
  _BYTE *v37; // rax
  _BYTE *v38; // rsi
  _BYTE *v39; // r8
  _BYTE *v40; // rsi
  __int64 *v41; // rax
  char *v42; // r13
  char *v43; // rax
  const char *v44; // rdx
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // r13
  char *v50; // rax
  _BYTE *v51; // rsi
  __int64 v52; // rax
  _BYTE *v53; // rsi
  char *v54; // r13
  char *v55; // r12
  __int64 v56; // rax
  __int64 v57; // r15
  int v58; // r14d
  int v59; // eax
  __int64 v60; // r13
  char *v61; // rax
  __int64 v62; // r9
  char *v63; // rax
  _BYTE *v64; // rsi
  __int64 v65; // rax
  _BYTE *v66; // rsi
  char *v67; // rax
  __int64 v68; // r12
  __int64 v69; // rax
  void **v70; // rax
  _BYTE *v71; // rsi
  _BYTE *v72; // rsi
  _BYTE *v73; // rsi
  _BYTE *v74; // rsi
  __int64 v75; // rdi
  char *v76; // r13
  char *v77; // r12
  char *v78; // r15
  __int64 v79; // rax
  _BYTE *v80; // rsi
  __int64 v81; // rdx
  _BYTE *v82; // rax
  _BYTE *v83; // rsi
  __int64 v84; // rdx
  _BYTE *v85; // rax
  _BYTE *v86; // rsi
  _BYTE *v87; // r8
  _BYTE *v88; // rsi
  __int64 *v89; // rax
  char *v90; // r12
  char *v91; // r15
  __int64 v92; // rax
  __int64 v93; // r15
  int v94; // r14d
  int v95; // eax
  _BYTE *v96; // r12
  __int64 v97; // rax
  __int64 v98; // rdi
  __int64 v99; // rax
  _QWORD *v100; // r12
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rdi
  __int64 v106; // rax
  __int64 v107; // r15
  char *v108; // rax
  __int64 v109; // rax
  unsigned __int64 v110; // rax
  __int64 v111; // rsi
  char *v112; // r12
  __int64 v113; // rax
  unsigned __int64 v114; // rax
  __int64 v115; // rsi
  const char *v116; // rdx
  __int64 v117; // rax
  char *v118; // rax
  __int64 v119; // rax
  unsigned __int64 v120; // rax
  char *v121; // rax
  __int64 v122; // rax
  unsigned __int64 v123; // rax
  __int64 **v124; // rax
  char *v125; // rax
  __int64 v126; // rax
  unsigned __int64 v127; // rax
  char *v128; // rax
  char *v129; // [rsp+20h] [rbp-400h]
  __int64 v130; // [rsp+28h] [rbp-3F8h]
  __int64 v131; // [rsp+28h] [rbp-3F8h]
  char *v132; // [rsp+30h] [rbp-3F0h]
  __int64 v133; // [rsp+30h] [rbp-3F0h]
  __int64 v134; // [rsp+30h] [rbp-3F0h]
  int v135; // [rsp+38h] [rbp-3E8h]
  unsigned __int64 v136; // [rsp+38h] [rbp-3E8h]
  __int64 v137; // [rsp+50h] [rbp-3D0h]
  bool v138; // [rsp+58h] [rbp-3C8h]
  __int64 *v139; // [rsp+58h] [rbp-3C8h]
  int v140; // [rsp+58h] [rbp-3C8h]
  __int64 v141; // [rsp+58h] [rbp-3C8h]
  __int64 v142; // [rsp+58h] [rbp-3C8h]
  __int64 v143; // [rsp+98h] [rbp-388h]
  unsigned int v144; // [rsp+A4h] [rbp-37Ch]
  int v145; // [rsp+A8h] [rbp-378h]
  _QWORD *v146; // [rsp+A8h] [rbp-378h]
  unsigned int v147; // [rsp+B0h] [rbp-370h]
  _QWORD *v148; // [rsp+B0h] [rbp-370h]
  _QWORD *v152; // [rsp+C8h] [rbp-358h]
  __int64 v153; // [rsp+C8h] [rbp-358h]
  __int64 v154; // [rsp+C8h] [rbp-358h]
  bool v155; // [rsp+D3h] [rbp-34Dh] BYREF
  int v156; // [rsp+D4h] [rbp-34Ch] BYREF
  __int64 v157; // [rsp+D8h] [rbp-348h] BYREF
  _QWORD *v158; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v159; // [rsp+E8h] [rbp-338h] BYREF
  __int64 **v160; // [rsp+F0h] [rbp-330h] BYREF
  char *v161; // [rsp+F8h] [rbp-328h] BYREF
  __int64 v162; // [rsp+100h] [rbp-320h] BYREF
  void **v163; // [rsp+108h] [rbp-318h] BYREF
  __int64 *v164; // [rsp+110h] [rbp-310h] BYREF
  __int64 *v165; // [rsp+118h] [rbp-308h] BYREF
  __int64 v166; // [rsp+120h] [rbp-300h] BYREF
  _BYTE *v167; // [rsp+128h] [rbp-2F8h]
  _BYTE *v168; // [rsp+130h] [rbp-2F0h]
  __int64 v169; // [rsp+140h] [rbp-2E0h] BYREF
  _BYTE *v170; // [rsp+148h] [rbp-2D8h]
  _BYTE *v171; // [rsp+150h] [rbp-2D0h]
  __int64 v172; // [rsp+160h] [rbp-2C0h] BYREF
  signed int v173; // [rsp+168h] [rbp-2B8h]
  __int64 *v174; // [rsp+170h] [rbp-2B0h]
  _BYTE *v175; // [rsp+180h] [rbp-2A0h] BYREF
  __int64 v176; // [rsp+188h] [rbp-298h]
  _QWORD v177[2]; // [rsp+190h] [rbp-290h] BYREF
  _BYTE *v178; // [rsp+1A0h] [rbp-280h] BYREF
  __int64 v179; // [rsp+1A8h] [rbp-278h]
  _QWORD v180[2]; // [rsp+1B0h] [rbp-270h] BYREF
  _QWORD *v181; // [rsp+1C0h] [rbp-260h]
  __int64 v182; // [rsp+1C8h] [rbp-258h]
  _QWORD v183[2]; // [rsp+1D0h] [rbp-250h] BYREF
  _BYTE *v184; // [rsp+1E0h] [rbp-240h] BYREF
  __int64 v185; // [rsp+1E8h] [rbp-238h]
  _QWORD v186[2]; // [rsp+1F0h] [rbp-230h] BYREF
  _QWORD *v187; // [rsp+200h] [rbp-220h] BYREF
  __int64 v188; // [rsp+208h] [rbp-218h]
  _QWORD v189[2]; // [rsp+210h] [rbp-210h] BYREF
  _QWORD *v190; // [rsp+220h] [rbp-200h] BYREF
  __int64 v191; // [rsp+228h] [rbp-1F8h]
  _QWORD v192[2]; // [rsp+230h] [rbp-1F0h] BYREF
  _QWORD *v193; // [rsp+240h] [rbp-1E0h] BYREF
  __int64 v194; // [rsp+248h] [rbp-1D8h]
  _QWORD v195[2]; // [rsp+250h] [rbp-1D0h] BYREF
  _QWORD v196[4]; // [rsp+260h] [rbp-1C0h] BYREF
  __int64 v197[4]; // [rsp+280h] [rbp-1A0h] BYREF
  __m128i v198[2]; // [rsp+2A0h] [rbp-180h] BYREF
  __m128i v199[2]; // [rsp+2C0h] [rbp-160h] BYREF
  __m128i v200; // [rsp+2E0h] [rbp-140h] BYREF
  char v201; // [rsp+2F0h] [rbp-130h] BYREF
  __m128i v202[2]; // [rsp+300h] [rbp-120h] BYREF
  __m128i v203[2]; // [rsp+320h] [rbp-100h] BYREF
  __m128i v204[2]; // [rsp+340h] [rbp-E0h] BYREF
  __m128i v205; // [rsp+360h] [rbp-C0h] BYREF
  __int16 v206; // [rsp+370h] [rbp-B0h]
  __m128i v207; // [rsp+380h] [rbp-A0h] BYREF
  _QWORD v208[2]; // [rsp+390h] [rbp-90h] BYREF
  __m128i v209; // [rsp+3A0h] [rbp-80h] BYREF
  _QWORD v210[2]; // [rsp+3B0h] [rbp-70h] BYREF
  bool *v211; // [rsp+3C0h] [rbp-60h] BYREF
  __int64 *v212; // [rsp+3C8h] [rbp-58h]
  __int64 v213; // [rsp+3D0h] [rbp-50h]
  int *v214; // [rsp+3D8h] [rbp-48h]
  _QWORD **v215; // [rsp+3E0h] [rbp-40h]
  __int64 *v216; // [rsp+3E8h] [rbp-38h]

  v175 = v177;
  v178 = v180;
  v181 = v183;
  v184 = v186;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v176 = 0;
  LOBYTE(v177[0]) = 0;
  v179 = 0;
  LOBYTE(v180[0]) = 0;
  v182 = 0;
  LOBYTE(v183[0]) = 0;
  v185 = 0;
  LOBYTE(v186[0]) = 0;
  v187 = v189;
  v188 = 0;
  LOBYTE(v189[0]) = 0;
  v190 = v192;
  v191 = 0;
  LOBYTE(v192[0]) = 0;
  v193 = v195;
  v194 = 0;
  LOBYTE(v195[0]) = 0;
  v155 = unk_4D045E8 <= 0x45u;
  if ( a3 == 366 )
  {
    v156 = sub_620EE0((_WORD *)(*(_QWORD *)(a4 + 56) + 176LL), 1, &v211);
    v13 = sub_620EE0((_WORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 56LL) + 176LL), 1, &v211);
    v14 = v156;
    if ( !v155 )
      goto LABEL_22;
LABEL_26:
    sub_12A7380((__int64)&v190, "volatile");
    v15 = v13 <= 2;
    if ( v13 != 2 )
      goto LABEL_16;
LABEL_27:
    if ( unk_4D045E8 > 0x59u )
    {
      sub_12A7380((__int64)&v193, "cluster");
      goto LABEL_19;
    }
    goto LABEL_218;
  }
  switch ( a3 )
  {
    case 417:
    case 423:
      v7 = 3;
      v8 = 4;
      goto LABEL_4;
    case 418:
    case 419:
    case 420:
    case 421:
    case 422:
      v8 = 2;
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
      v7 = 2;
      v8 = 3;
      goto LABEL_4;
    case 462:
      v7 = 4;
      v8 = 5;
LABEL_4:
      v9 = a4;
      for ( i = 0; i != v7; ++i )
        v9 = *(_QWORD *)(v9 + 16);
LABEL_6:
      v156 = sub_620EE0((_WORD *)(*(_QWORD *)(v9 + 56) + 176LL), 1, &v211);
      break;
    case 469:
      v92 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL) + 16LL) + 16LL);
      v93 = *(_QWORD *)(v92 + 16);
      v94 = sub_620EE0((_WORD *)(*(_QWORD *)(v92 + 56) + 176LL), 1, &v211);
      v95 = sub_620EE0((_WORD *)(*(_QWORD *)(v93 + 56) + 176LL), 1, &v211);
      if ( v95 <= v94 )
        v95 = v94;
      v8 = 7;
      v156 = v95;
      break;
    case 470:
    case 471:
    case 472:
    case 473:
      v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL) + 16LL);
      v57 = *(_QWORD *)(v56 + 16);
      v58 = sub_620EE0((_WORD *)(*(_QWORD *)(v56 + 56) + 176LL), 1, &v211);
      v59 = sub_620EE0((_WORD *)(*(_QWORD *)(v57 + 56) + 176LL), 1, &v211);
      if ( v59 <= v58 )
        v59 = v58;
      v8 = 6;
      v156 = v59;
      break;
    default:
      sub_127B630("unexpected atomic builtin function", 0);
  }
  v11 = a4;
  for ( j = 0; j != v8; ++j )
    v11 = *(_QWORD *)(v11 + 16);
  v13 = sub_620EE0((_WORD *)(*(_QWORD *)(v11 + 56) + 176LL), 1, &v211);
  if ( v155 )
    goto LABEL_26;
  v14 = v156;
  if ( (unsigned int)(a3 - 423) > 4 )
  {
LABEL_22:
    switch ( v14 )
    {
      case 0:
        goto LABEL_177;
      case 1:
      case 2:
      case 5:
        sub_12A7380((__int64)&v190, "acquire");
        goto LABEL_15;
      case 3:
        goto LABEL_178;
      case 4:
        sub_12A7380((__int64)&v190, "acq_rel");
        goto LABEL_15;
      default:
        goto LABEL_14;
    }
  }
  if ( v156 == 3 )
  {
LABEL_178:
    sub_12A7380((__int64)&v190, "release");
  }
  else
  {
    if ( v156 != 5 && v156 )
LABEL_14:
      sub_127B630("unexpected memory order.", 0);
LABEL_177:
    sub_12A7380((__int64)&v190, "relaxed");
  }
LABEL_15:
  v15 = v13 <= 2;
  if ( v13 == 2 )
    goto LABEL_27;
LABEL_16:
  if ( v15 )
  {
    if ( (unsigned int)v13 <= 1 )
    {
      sub_12A7380((__int64)&v193, "cta");
      goto LABEL_19;
    }
LABEL_217:
    sub_127B630("unexpected atomic operation scope.", 0);
  }
  if ( v13 == 3 )
  {
LABEL_218:
    sub_12A7380((__int64)&v193, "gpu");
    goto LABEL_19;
  }
  if ( v13 != 4 )
    goto LABEL_217;
  sub_12A7380((__int64)&v193, "sys");
LABEL_19:
  v213 = a2;
  v174 = &v157;
  v216 = &v157;
  v211 = &v155;
  v157 = a2;
  v172 = a2;
  v173 = v13;
  v212 = &v172;
  v214 = &v156;
  v215 = &v193;
  v138 = v155;
  if ( a3 == 366 )
  {
    if ( v155 )
      sub_12AE0E0(a1, &v172);
    else
      sub_12AE4B0(a1, &v211);
  }
  else
  {
    v135 = v156;
    switch ( a3 )
    {
      case 417:
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
        sub_12A7380((__int64)&v184, "ld");
        break;
      case 423:
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
        sub_12A7380((__int64)&v184, "st");
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
        sub_12A7380((__int64)&v184, "atom.add");
        break;
      case 441:
      case 442:
        sub_12A7380((__int64)&v184, "atom.and");
        break;
      case 443:
      case 444:
        sub_12A7380((__int64)&v184, "atom.or");
        break;
      case 445:
      case 446:
        sub_12A7380((__int64)&v184, "atom.xor");
        break;
      case 447:
      case 448:
      case 449:
      case 450:
      case 451:
      case 452:
        sub_12A7380((__int64)&v184, "atom.max");
        break;
      case 453:
      case 454:
      case 455:
      case 456:
      case 457:
      case 458:
        sub_12A7380((__int64)&v184, "atom.min");
        break;
      case 462:
      case 463:
      case 464:
      case 465:
        sub_12A7380((__int64)&v184, "atom.exch");
        break;
      case 469:
      case 470:
      case 471:
      case 472:
      case 473:
        sub_12A7380((__int64)&v184, "atom.cas");
        break;
      default:
        sub_127B630("unexpected atomic builtin function.", 0);
    }
    v159 = a2;
    v196[0] = "b";
    v196[1] = "u";
    v196[2] = "s";
    v196[3] = "f";
    v158 = v196;
    switch ( a3 )
    {
      case 417:
      case 423:
      case 462:
      case 469:
        v145 = 0;
        v147 = sub_620EE0((_WORD *)(*(_QWORD *)(a4 + 56) + 176LL), 0, &v209);
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
        v147 = *(_DWORD *)(v17 + 128);
        switch ( m )
        {
          case 2:
            if ( !byte_4B6DF90[*(unsigned __int8 *)(v17 + 160)] )
            {
LABEL_57:
              v145 = 1;
              break;
            }
            v145 = 2;
            break;
          case 3:
            v145 = 3;
            break;
          case 6:
            goto LABEL_57;
          default:
            sub_127B630("unexpected type.", 0);
        }
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
            v137 = 0;
            v144 = 0;
            goto LABEL_64;
          case 423:
          case 424:
          case 425:
          case 426:
          case 427:
          case 428:
            sub_12A7F20(v209.m128i_i64, (__int64 *)&v158, v147, v145);
            sub_12A7490((__int64)&v187, (__int64)&v209);
            sub_2240A30(&v209);
            v137 = 0;
            v143 = sub_1643270(*(_QWORD *)(a2 + 40));
            v144 = 0;
            goto LABEL_65;
          case 462:
          case 469:
          case 470:
          case 471:
          case 472:
          case 473:
            v144 = v147;
            if ( v147 > 0x10 || ((1LL << v147) & 0x10116) == 0 )
              sub_127B630("unexpected size1", 0);
            v137 = sub_12A7200((__int64)&v159, v147, v145);
LABEL_64:
            sub_12A7F20(v209.m128i_i64, (__int64 *)&v158, v147, v145);
            sub_12A7490((__int64)&v187, (__int64)&v209);
            sub_2240A30(&v209);
            v143 = sub_12A7200((__int64)&v159, v147, v145);
LABEL_65:
            if ( v155 && (unsigned int)(v156 - 3) <= 2 )
              sub_12AE0E0((__int64)&v209, &v172);
            break;
          default:
            sub_127B630("unexpected atomic builtin function.", 0);
        }
        return result;
      case 470:
      case 471:
      case 472:
      case 473:
        for ( k = **(_QWORD **)(*(_QWORD *)(a4 + 16) + 16LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
LABEL_61:
        v145 = 0;
        v147 = *(_DWORD *)(k + 128);
        goto LABEL_62;
      default:
        sub_127B630("unexpected atomic builtin function.", 0);
    }
    switch ( a3 )
    {
      case 417:
      case 418:
      case 419:
      case 420:
      case 421:
      case 422:
        v49 = a4;
        if ( a3 == 417 )
          v49 = *(_QWORD *)(a4 + 16);
        if ( !v138 && v135 == 5 )
        {
          if ( *v211 )
            sub_12AE0E0((__int64)&v209, v212);
          else
            sub_12AE4B0((__int64)&v209, &v211);
        }
        v50 = sub_128F980(a2, v49);
        v51 = v167;
        v198[0].m128i_i64[0] = (__int64)v50;
        v52 = *(_QWORD *)v50;
        v199[0].m128i_i64[0] = v52;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, v199);
        }
        else
        {
          if ( v167 )
          {
            *(_QWORD *)v167 = v52;
            v51 = v167;
          }
          v167 = v51 + 8;
        }
        v53 = v170;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, v198);
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v198[0].m128i_i64[0];
            v53 = v170;
          }
          v170 = v53 + 8;
        }
        v54 = sub_12A6FA0(v199[0].m128i_i64[0]);
        v55 = sub_12A6FA0(v143);
        sub_12A7780(v202[0].m128i_i64, "=");
        sub_94F930(v203, (__int64)v202, v55);
        sub_94F930(v204, (__int64)v203, ",");
        sub_94F930(&v205, (__int64)v204, v54);
        sub_94F930(&v207, (__int64)&v205, ",");
        sub_94F930(&v209, (__int64)&v207, "~{memory}");
        sub_12A7490((__int64)&v178, (__int64)&v209);
        sub_2240A30(&v209);
        sub_2240A30(&v207);
        sub_2240A30(&v205);
        sub_2240A30(v204);
        sub_2240A30(v203);
        sub_2240A30(v202);
        if ( v155 )
        {
          sub_2241BD0(v203, &v184);
          sub_2241520(v203, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v190, v191);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          v44 = " $0, [$1];";
          goto LABEL_111;
        }
        sub_2241BD0(&v200, &v184);
        sub_2241520(&v200, ".");
        sub_12A7420(v202, (__int64)&v200, (__int64)v190, v191);
        sub_94F930(v203, (__int64)v202, ".");
        sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
        sub_94F930(&v205, (__int64)v204, ".");
        sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
        v116 = " $0, [$1];";
        goto LABEL_225;
      case 423:
      case 424:
      case 425:
      case 426:
      case 427:
      case 428:
        if ( a3 == 423 )
        {
          v117 = sub_12A7200((__int64)&v159, v147, v145);
          if ( v147 > 0x10 || ((1LL << v147) & 0x10116) == 0 )
            sub_127B630("unexpected size1", 0);
          v130 = v117;
          v30 = *(_QWORD *)(a4 + 16);
          v118 = sub_128F980(a2, *(_QWORD *)(v30 + 16));
          LOWORD(v210[0]) = 257;
          v133 = (__int64)v118;
          v119 = sub_1646BA0(v130, 0);
          v120 = sub_12A95D0((__int64 *)(a2 + 48), v133, v119, (__int64)&v209);
          v165 = sub_12810A0((__int64 *)a2, v120, v147, 0);
        }
        else
        {
          v30 = a4;
          v165 = (__int64 *)sub_128F980(a2, *(_QWORD *)(a4 + 16));
        }
        if ( v135 == 5 && !v138 )
        {
          if ( *v211 )
            sub_12AE0E0((__int64)&v209, v212);
          else
            sub_12AE4B0((__int64)&v209, &v211);
        }
        v31 = sub_128F980(a2, v30);
        v32 = v167;
        v197[0] = (__int64)v31;
        v33 = *(_QWORD *)v31;
        v198[0].m128i_i64[0] = *(_QWORD *)v31;
        v199[0].m128i_i64[0] = *v165;
        v34 = v168;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, v198);
          v37 = v167;
          v35 = v168;
          if ( v167 != v168 )
          {
            if ( !v167 )
              goto LABEL_102;
            v36 = v199[0].m128i_i64[0];
            v35 = v167;
LABEL_101:
            *(_QWORD *)v35 = v36;
            v37 = v167;
LABEL_102:
            v167 = v37 + 8;
            goto LABEL_103;
          }
        }
        else
        {
          if ( v167 )
          {
            *(_QWORD *)v167 = v33;
            v34 = v168;
            v32 = v167;
          }
          v35 = v32 + 8;
          v36 = v199[0].m128i_i64[0];
          v167 = v35;
          if ( v34 != v35 )
            goto LABEL_101;
        }
        sub_1277EB0((__int64)&v166, v35, v199);
LABEL_103:
        v38 = v170;
        v39 = v171;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, v197);
          v40 = v170;
          v39 = v171;
          if ( v170 != v171 )
          {
            if ( !v170 )
              goto LABEL_108;
            v41 = v165;
LABEL_107:
            *(_QWORD *)v40 = v41;
            v40 = v170;
LABEL_108:
            v170 = v40 + 8;
            goto LABEL_109;
          }
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v197[0];
            v39 = v171;
            v38 = v170;
          }
          v40 = v38 + 8;
          v41 = v165;
          v170 = v40;
          if ( v40 != v39 )
            goto LABEL_107;
        }
        sub_1287830((__int64)&v169, v39, &v165);
LABEL_109:
        v42 = sub_12A6FA0(v199[0].m128i_i64[0]);
        v43 = sub_12A6FA0(v198[0].m128i_i64[0]);
        sub_12A7780(v203[0].m128i_i64, v43);
        sub_94F930(v204, (__int64)v203, ",");
        sub_94F930(&v205, (__int64)v204, v42);
        sub_94F930(&v207, (__int64)&v205, ",");
        sub_94F930(&v209, (__int64)&v207, "~{memory}");
        sub_12A7490((__int64)&v178, (__int64)&v209);
        sub_2240A30(&v209);
        sub_2240A30(&v207);
        sub_2240A30(&v205);
        sub_2240A30(v204);
        sub_2240A30(v203);
        if ( v155 )
        {
          sub_2241BD0(v203, &v184);
          sub_2241520(v203, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v190, v191);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          v44 = " [$0], $1;";
          goto LABEL_111;
        }
        sub_2241BD0(&v200, &v184);
        sub_2241520(&v200, ".");
        sub_12A7420(v202, (__int64)&v200, (__int64)v190, v191);
        sub_94F930(v203, (__int64)v202, ".");
        sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
        sub_94F930(&v205, (__int64)v204, ".");
        sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
        sub_94F930(&v209, (__int64)&v207, " [$0], $1;");
        sub_12A7490((__int64)&v175, (__int64)&v209);
        sub_2240A30(&v209);
        sub_2240A30(&v207);
        sub_2240A30(&v205);
        sub_2240A30(v204);
        sub_2240A30(v203);
        sub_2240A30(v202);
        sub_2240A30(&v200);
        goto LABEL_112;
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
        v20 = *(_QWORD *)(a4 + 16);
        v21 = v135 == 5 && !v138;
        if ( a3 == 462 )
        {
          v125 = sub_128F980(a2, *(_QWORD *)(v20 + 16));
          LOWORD(v210[0]) = 257;
          v141 = (__int64)v125;
          v126 = sub_1646BA0(v137, 0);
          v127 = sub_12A95D0((__int64 *)(a2 + 48), v141, v126, (__int64)&v209);
          v164 = sub_12810A0((__int64 *)a2, v127, v144, 0);
          if ( !v21 )
          {
            v128 = sub_128F980(a2, v20);
            v22 = v164;
            v165 = (__int64 *)v128;
            goto LABEL_197;
          }
        }
        else
        {
          v20 = a4;
          v164 = (__int64 *)sub_128F980(a2, *(_QWORD *)(a4 + 16));
          if ( !v21 )
            goto LABEL_71;
        }
        if ( *v211 )
          sub_12AE0E0((__int64)&v209, v212);
        else
          sub_12AE4B0((__int64)&v209, &v211);
LABEL_71:
        v165 = (__int64 *)sub_128F980(a2, v20);
        if ( a3 == 436 || a3 == 439 || a3 == 435 || a3 == 438 )
        {
          v79 = sub_15A06D0(*v164);
          LOWORD(v208[0]) = 257;
          if ( *(_BYTE *)(v79 + 16) > 0x10u || *((_BYTE *)v164 + 16) > 0x10u )
          {
            LOWORD(v210[0]) = 257;
            v22 = (__int64 *)sub_15FB440(13, v79, v164, &v209, 0);
            sub_12A73B0((__int64)v22, (__int64)&v207, *(_QWORD *)(a2 + 56), *(__int64 **)(a2 + 64));
            sub_12A86E0((__int64 *)(a2 + 48), (__int64)v22);
            sub_15F2330(v22, 1);
          }
          else
          {
            v22 = (__int64 *)sub_15A2B60(v79, v164, 0, 1);
          }
LABEL_196:
          v164 = v22;
          goto LABEL_197;
        }
        v22 = v164;
        if ( a3 == 437 || a3 == 440 )
        {
          v23 = sub_15A06D0(*v164);
          LOWORD(v208[0]) = 257;
          v24 = v164;
          v25 = v23;
          if ( *(_BYTE *)(v23 + 16) > 0x10u
            || *((_BYTE *)v164 + 16) > 0x10u
            || (v139 = v164, v26 = sub_15A2A30(14, v23, v164, 0, 0), v24 = v139, (v22 = (__int64 *)v26) == 0) )
          {
            LOWORD(v210[0]) = 257;
            v27 = sub_15FB440(14, v25, v24, &v209, 0);
            v28 = *(_QWORD *)(a2 + 80);
            v29 = *(_DWORD *)(a2 + 88);
            v22 = (__int64 *)v27;
            if ( v28 )
              sub_1625C10(v27, 3, v28);
            sub_15F2440(v22, v29);
            sub_12A73B0((__int64)v22, (__int64)&v207, *(_QWORD *)(a2 + 56), *(__int64 **)(a2 + 64));
            sub_12A86E0((__int64 *)(a2 + 48), (__int64)v22);
          }
          goto LABEL_196;
        }
LABEL_197:
        v80 = v167;
        v81 = *v165;
        v197[0] = *v165;
        v198[0].m128i_i64[0] = *v22;
        v82 = v168;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, v197);
          v85 = v167;
          v83 = v168;
          if ( v167 != v168 )
          {
            if ( !v167 )
              goto LABEL_202;
            v84 = v198[0].m128i_i64[0];
            v83 = v167;
LABEL_201:
            *(_QWORD *)v83 = v84;
            v85 = v167;
LABEL_202:
            v167 = v85 + 8;
            goto LABEL_203;
          }
        }
        else
        {
          if ( v167 )
          {
            *(_QWORD *)v167 = v81;
            v82 = v168;
            v80 = v167;
          }
          v83 = v80 + 8;
          v84 = v198[0].m128i_i64[0];
          v167 = v83;
          if ( v82 != v83 )
            goto LABEL_201;
        }
        sub_1277EB0((__int64)&v166, v83, v198);
LABEL_203:
        v86 = v170;
        v87 = v171;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, &v165);
          v88 = v170;
          v87 = v171;
          if ( v170 != v171 )
          {
            if ( !v170 )
              goto LABEL_208;
            v89 = v164;
LABEL_207:
            *(_QWORD *)v88 = v89;
            v88 = v170;
LABEL_208:
            v170 = v88 + 8;
            goto LABEL_209;
          }
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v165;
            v87 = v171;
            v86 = v170;
          }
          v88 = v86 + 8;
          v89 = v164;
          v170 = v88;
          if ( v88 != v87 )
            goto LABEL_207;
        }
        sub_1287830((__int64)&v169, v87, &v164);
LABEL_209:
        v129 = sub_12A6FA0(v198[0].m128i_i64[0]);
        v90 = sub_12A6FA0(v197[0]);
        v91 = sub_12A6FA0(v143);
        sub_12A7780(v199[0].m128i_i64, "=");
        sub_94F930(&v200, (__int64)v199, v91);
        sub_94F930(v202, (__int64)&v200, ",");
        sub_94F930(v203, (__int64)v202, v90);
        sub_94F930(v204, (__int64)v203, ",");
        sub_94F930(&v205, (__int64)v204, v129);
        sub_94F930(&v207, (__int64)&v205, ",");
        sub_94F930(&v209, (__int64)&v207, "~{memory}");
        sub_12A7490((__int64)&v178, (__int64)&v209);
        sub_2240A30(&v209);
        sub_2240A30(&v207);
        sub_2240A30(&v205);
        sub_2240A30(v204);
        sub_2240A30(v203);
        sub_2240A30(v202);
        sub_2240A30(&v200);
        sub_2240A30(v199);
        if ( v155 )
        {
          sub_2241BD0(v203, &v184);
          sub_2241520(v203, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          v44 = " $0,[$1],$2;";
LABEL_111:
          sub_94F930(&v209, (__int64)&v207, v44);
          sub_12A7490((__int64)&v175, (__int64)&v209);
          sub_2240A30(&v209);
          sub_2240A30(&v207);
          sub_2240A30(&v205);
          sub_2240A30(v204);
          sub_2240A30(v203);
        }
        else
        {
          sub_2241BD0(&v200, &v184);
          sub_2241520(&v200, ".");
          sub_12A7420(v202, (__int64)&v200, (__int64)v190, v191);
          sub_94F930(v203, (__int64)v202, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          v116 = " $0,[$1],$2;";
LABEL_225:
          sub_94F930(&v209, (__int64)&v207, v116);
          sub_12A7490((__int64)&v175, (__int64)&v209);
          sub_2240A30(&v209);
          sub_2240A30(&v207);
          sub_2240A30(&v205);
          sub_2240A30(v204);
          sub_2240A30(v203);
          sub_2240A30(v202);
          sub_2240A30(&v200);
        }
LABEL_112:
        v136 = 0;
LABEL_113:
        v45 = sub_1644EA0(v143, v166, (__int64)&v167[-v166] >> 3, 0);
        v209.m128i_i64[0] = (__int64)v210;
        v140 = v45;
        sub_12A72D0(v209.m128i_i64, v178, (__int64)&v178[v179]);
        v207.m128i_i64[0] = (__int64)v208;
        sub_12A72D0(v207.m128i_i64, v175, (__int64)&v175[v176]);
        v46 = sub_15EE570(v140, v207.m128i_i32[0], v207.m128i_i32[2], v209.m128i_i32[0], v209.m128i_i32[2], 1, 0, 0);
        v206 = 257;
        v47 = sub_1285290(
                (__int64 *)(v157 + 48),
                *(_QWORD *)(*(_QWORD *)v46 + 24LL),
                v46,
                v169,
                (__int64)&v170[-v169] >> 3,
                (__int64)&v205,
                0);
        v205.m128i_i64[0] = *(_QWORD *)(v47 + 56);
        v48 = sub_16498A0(v47);
        v205.m128i_i64[0] = sub_1563AB0(&v205, v48, 0xFFFFFFFFLL, 30);
        *(_QWORD *)(v47 + 56) = v205.m128i_i64[0];
        if ( (_QWORD *)v207.m128i_i64[0] != v208 )
          j_j___libc_free_0(v207.m128i_i64[0], v208[0] + 1LL);
        if ( (_QWORD *)v209.m128i_i64[0] != v210 )
          j_j___libc_free_0(v209.m128i_i64[0], v210[0] + 1LL);
        if ( (unsigned int)(a3 - 469) <= 4 )
        {
          v96 = sub_12810A0((__int64 *)a2, v136, v144, 0);
          v146 = (_QWORD *)sub_12A4D50(a2, (__int64)"cond.true", 0, 0);
          v152 = (_QWORD *)sub_12A4D50(a2, (__int64)"cond.false", 0, 0);
          v148 = (_QWORD *)sub_12A4D50(a2, (__int64)"cond.end", 0, 0);
          v209.m128i_i64[0] = (__int64)"cmp";
          LOWORD(v210[0]) = 259;
          v97 = sub_12AA0C0((__int64 *)(a2 + 48), 0x20u, v96, v47, (__int64)&v209);
          sub_12A4DB0((_QWORD *)a2, v97, (__int64)v146, (__int64)v152, 0);
          v98 = *(_QWORD *)(a2 + 40);
          v209.m128i_i64[0] = (__int64)"temp";
          LOWORD(v210[0]) = 259;
          v99 = sub_1643330(v98);
          v100 = sub_127FC40((_QWORD *)a2, v99, (__int64)&v209, 0, 0);
          sub_1290AF0((_QWORD *)a2, v146, 0);
          v101 = sub_1643330(*(_QWORD *)(a2 + 40));
          v102 = sub_159C470(v101, 1, 0);
          sub_12A8F50((__int64 *)(a2 + 48), v102, (__int64)v100, 0);
          sub_12909B0((_QWORD *)a2, (__int64)v148);
          sub_1290AF0((_QWORD *)a2, v152, 0);
          v103 = sub_1643330(*(_QWORD *)(a2 + 40));
          v104 = sub_159C470(v103, 0, 0);
          sub_12A8F50((__int64 *)(a2 + 48), v104, (__int64)v100, 0);
          sub_12909B0((_QWORD *)a2, (__int64)v148);
          sub_1290AF0((_QWORD *)a2, v148, 0);
          sub_1280F50((__int64 *)a2, v47, v136, v144, 0);
          v105 = *(_QWORD *)(a2 + 40);
          LOWORD(v210[0]) = 257;
          v106 = sub_1643330(v105);
          v47 = (__int64)sub_12A93B0((__int64 *)(a2 + 48), v106, (__int64)v100, (__int64)&v209);
        }
        else if ( a3 == 417 )
        {
          if ( v147 > 0x10 || ((1LL << v147) & 0x10116) == 0 )
            sub_127B630("unexpected size1", 0);
          v153 = *(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL);
          v107 = sub_12A7200((__int64)&v159, v147, v145);
          v108 = sub_128F980(a2, v153);
          LOWORD(v210[0]) = 257;
          v154 = (__int64)v108;
          v109 = sub_1646BA0(v107, 0);
          v110 = sub_12A95D0((__int64 *)(a2 + 48), v154, v109, (__int64)&v209);
          v111 = v47;
          v47 = 0;
          sub_1280F50((__int64 *)a2, v111, v110, v147, 0);
        }
        else if ( a3 == 462 )
        {
          v112 = sub_128F980(a2, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 16) + 16LL) + 16LL));
          LOWORD(v210[0]) = 257;
          v113 = sub_1646BA0(v137, 0);
          v114 = sub_12A95D0((__int64 *)(a2 + 48), (__int64)v112, v113, (__int64)&v209);
          v115 = v47;
          v47 = 0;
          sub_1280F50((__int64 *)a2, v115, v114, v144, 0);
        }
        if ( !v155 )
          goto LABEL_124;
        if ( v156 > 2 )
        {
          if ( (unsigned int)(v156 - 4) > 1 )
            goto LABEL_124;
        }
        else if ( v156 <= 0 )
        {
          goto LABEL_124;
        }
        sub_12AE0E0((__int64)&v209, &v172);
LABEL_124:
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v47;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
        break;
      case 469:
      case 470:
      case 471:
      case 472:
      case 473:
        v60 = *(_QWORD *)(a4 + 16);
        if ( a3 == 469 )
        {
          v131 = *(_QWORD *)(v60 + 16);
          v121 = sub_128F980(a2, *(_QWORD *)(v131 + 16));
          LOWORD(v210[0]) = 257;
          v134 = (__int64)v121;
          v122 = sub_1646BA0(v137, 0);
          v123 = sub_12A95D0((__int64 *)(a2 + 48), v134, v122, (__int64)&v209);
          v124 = (__int64 **)sub_12810A0((__int64 *)a2, v123, v144, 0);
          v62 = v60;
          v160 = v124;
          v60 = v131;
        }
        else
        {
          v61 = sub_128F980(a2, *(_QWORD *)(v60 + 16));
          v62 = a4;
          v160 = (__int64 **)v61;
        }
        if ( !v138 && v135 == 5 )
        {
          v142 = v62;
          if ( *v211 )
            sub_12AE0E0((__int64)&v209, v212);
          else
            sub_12AE4B0((__int64)&v209, &v211);
          v62 = v142;
        }
        v63 = sub_128F980(a2, v62);
        v64 = v167;
        v161 = v63;
        v65 = *(_QWORD *)v63;
        v162 = v65;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, &v162);
        }
        else
        {
          if ( v167 )
          {
            *(_QWORD *)v167 = v65;
            v64 = v167;
          }
          v167 = v64 + 8;
        }
        v66 = v170;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, &v161);
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v161;
            v66 = v170;
          }
          v170 = v66 + 8;
        }
        v67 = sub_128F980(a2, v60);
        LOWORD(v210[0]) = 257;
        v68 = (__int64)v67;
        v69 = sub_1646BA0(v137, 0);
        v136 = sub_12A95D0((__int64 *)(a2 + 48), v68, v69, (__int64)&v209);
        v70 = (void **)sub_12810A0((__int64 *)a2, v136, v144, 0);
        v71 = v170;
        v163 = v70;
        v164 = (__int64 *)*v70;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, &v163);
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v70;
            v71 = v170;
          }
          v170 = v71 + 8;
        }
        v72 = v167;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, &v164);
        }
        else
        {
          if ( v167 )
          {
            *(_QWORD *)v167 = v164;
            v72 = v167;
          }
          v167 = v72 + 8;
        }
        v73 = v170;
        v165 = *v160;
        if ( v170 == v171 )
        {
          sub_1287830((__int64)&v169, v170, &v160);
        }
        else
        {
          if ( v170 )
          {
            *(_QWORD *)v170 = v160;
            v73 = v170;
          }
          v170 = v73 + 8;
        }
        v74 = v167;
        if ( v167 == v168 )
        {
          sub_1277EB0((__int64)&v166, v167, &v165);
          v75 = (__int64)v165;
        }
        else
        {
          v75 = (__int64)v165;
          if ( v167 )
          {
            *(_QWORD *)v167 = v165;
            v74 = v167;
          }
          v167 = v74 + 8;
        }
        v76 = sub_12A6FA0(v75);
        v77 = sub_12A6FA0((__int64)v164);
        v132 = sub_12A6FA0(v162);
        v78 = sub_12A6FA0(v143);
        sub_12A7780(v197, "=");
        sub_94F930(v198, (__int64)v197, v78);
        sub_94F930(v199, (__int64)v198, ",");
        sub_94F930(&v200, (__int64)v199, v132);
        sub_94F930(v202, (__int64)&v200, ",");
        sub_94F930(v203, (__int64)v202, v77);
        sub_94F930(v204, (__int64)v203, ",");
        sub_94F930(&v205, (__int64)v204, v76);
        sub_94F930(&v207, (__int64)&v205, ",");
        sub_94F930(&v209, (__int64)&v207, "~{memory}");
        sub_12A7490((__int64)&v178, (__int64)&v209);
        sub_2240A30(&v209);
        sub_2240A30(&v207);
        sub_2240A30(&v205);
        sub_2240A30(v204);
        sub_2240A30(v203);
        sub_2240A30(v202);
        sub_2240A30(&v200);
        sub_2240A30(v199);
        sub_2240A30(v198);
        sub_2240A30(v197);
        if ( v155 )
        {
          sub_2241BD0(v203, &v184);
          sub_2241520(v203, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          sub_94F930(&v209, (__int64)&v207, " $0,[$1],$2,$3;");
          sub_12A7490((__int64)&v175, (__int64)&v209);
          sub_2240A30(&v209);
          sub_2240A30(&v207);
          sub_2240A30(&v205);
          sub_2240A30(v204);
          sub_2240A30(v203);
        }
        else
        {
          v200.m128i_i64[0] = (__int64)&v201;
          sub_12A72D0(v200.m128i_i64, v184, (__int64)&v184[v185]);
          sub_2241520(&v200, ".");
          sub_12A7420(v202, (__int64)&v200, (__int64)v190, v191);
          sub_94F930(v203, (__int64)v202, ".");
          sub_12A7420(v204, (__int64)v203, (__int64)v193, v194);
          sub_94F930(&v205, (__int64)v204, ".");
          sub_12A7420(&v207, (__int64)&v205, (__int64)v187, v188);
          sub_94F930(&v209, (__int64)&v207, " $0,[$1],$2,$3;");
          sub_12A7490((__int64)&v175, (__int64)&v209);
          sub_2240A30(&v209);
          sub_2240A30(&v207);
          sub_2240A30(&v205);
          sub_2240A30(v204);
          sub_2240A30(v203);
          sub_2240A30(v202);
          sub_2240A30(&v200);
        }
        goto LABEL_113;
      default:
        sub_127B630("unexpected atomic builtin function", 0);
    }
  }
  if ( v193 != v195 )
    j_j___libc_free_0(v193, v195[0] + 1LL);
  if ( v190 != v192 )
    j_j___libc_free_0(v190, v192[0] + 1LL);
  if ( v187 != v189 )
    j_j___libc_free_0(v187, v189[0] + 1LL);
  if ( v184 != (_BYTE *)v186 )
    j_j___libc_free_0(v184, v186[0] + 1LL);
  if ( v181 != v183 )
    j_j___libc_free_0(v181, v183[0] + 1LL);
  if ( v178 != (_BYTE *)v180 )
    j_j___libc_free_0(v178, v180[0] + 1LL);
  if ( v175 != (_BYTE *)v177 )
    j_j___libc_free_0(v175, v177[0] + 1LL);
  if ( v169 )
    j_j___libc_free_0(v169, &v171[-v169]);
  if ( v166 )
    j_j___libc_free_0(v166, &v168[-v166]);
  return a1;
}
