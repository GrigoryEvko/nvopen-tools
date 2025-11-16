// Function: sub_F7E3C0
// Address: 0xf7e3c0
//
__int64 __fastcall sub_F7E3C0(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 *v13; // rax
  unsigned int *v14; // r10
  char v16; // dl
  char v17; // al
  int v18; // r8d
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r8d
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r14
  int v25; // edx
  int v26; // r13d
  __int64 v27; // rax
  int v28; // edx
  int v29; // r15d
  bool v30; // of
  __int64 v31; // r14
  unsigned int *v32; // r12
  __int64 v33; // r8
  unsigned __int64 v34; // r14
  __int64 *v35; // r13
  __int64 *i; // r15
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r9
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // r11
  __int64 v41; // rbx
  int v42; // edx
  __int64 v43; // rax
  signed __int64 v44; // rax
  bool v45; // zf
  bool v46; // sf
  __int64 v47; // rax
  int v48; // edx
  int v49; // r13d
  __int64 v50; // r14
  __int64 v51; // rax
  unsigned int v52; // edx
  __int64 v53; // r8
  __int64 v54; // r14
  unsigned int *v55; // r12
  unsigned int *v56; // r9
  unsigned __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rdi
  __int64 v61; // r13
  int v62; // edx
  __int64 v63; // rax
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  int v66; // edx
  int v67; // ecx
  int v68; // r8d
  unsigned __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r14
  int v72; // edx
  int v73; // r13d
  __int64 v74; // rax
  int v75; // edx
  __int64 v76; // r14
  unsigned int *v77; // r12
  __int64 v78; // r8
  unsigned __int64 v79; // r14
  __int64 *v80; // r13
  __int64 *k; // r15
  unsigned __int64 v82; // rcx
  unsigned __int64 v83; // r9
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // r11
  __int64 v86; // rbx
  int v87; // edx
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 *v90; // r15
  __int64 v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rax
  int v94; // edx
  int v95; // r13d
  __int64 v96; // r14
  __int64 v97; // rax
  _BOOL4 v98; // edx
  _BOOL4 v99; // r15d
  __int64 v100; // r11
  unsigned int *v101; // r12
  unsigned int *v102; // r9
  __int64 v103; // r8
  unsigned __int64 j; // r13
  unsigned __int64 v105; // rbx
  unsigned __int64 v106; // r14
  unsigned __int64 v107; // rax
  unsigned __int64 v108; // rcx
  int v109; // edx
  __int64 v110; // r14
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // r13
  __int64 v114; // rax
  __int64 v115; // rdx
  _BYTE *v116; // r12
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // r8
  __int64 v120; // r9
  _BYTE *v121; // r12
  __int64 v122; // rdx
  __int64 v123; // r12
  unsigned __int64 v124; // rax
  int v125; // ecx
  _QWORD *v126; // rax
  __int64 v127; // rax
  __int64 *v128; // r12
  __int64 *v129; // r14
  _QWORD *v130; // rdx
  int v131; // ecx
  unsigned __int64 v132; // rdx
  __int64 v133; // r13
  _QWORD *v134; // rdx
  unsigned __int64 v135; // rax
  int v136; // edx
  unsigned int v137; // edx
  unsigned int v138; // r12d
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rdx
  __int64 v142; // rdx
  int v143; // r12d
  __int64 v144; // r8
  __int64 v145; // r9
  __int64 v146; // rdx
  __int64 v147; // r9
  unsigned __int64 v148; // rcx
  __int64 v149; // r12
  unsigned __int64 v150; // rax
  int v151; // edx
  _QWORD *v152; // rax
  __int64 v153; // rax
  unsigned int v154; // edx
  __int64 v155; // rax
  unsigned int v156; // r12d
  __int64 v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rax
  unsigned __int64 v160; // r13
  __int64 v161; // rax
  __int64 v162; // rdx
  _BYTE *v163; // r12
  __int64 v164; // rax
  __int64 v165; // rdx
  __int64 v166; // r8
  __int64 v167; // r9
  _BYTE *v168; // r12
  __int64 v169; // rdx
  __int64 v170; // r12
  unsigned __int64 v171; // rax
  int v172; // ecx
  _QWORD *v173; // rax
  __int64 v174; // rax
  __int64 *v175; // r12
  __int64 *v176; // r14
  _QWORD *v177; // rdx
  int v178; // ecx
  unsigned __int64 v179; // rdx
  __int64 v180; // r13
  _QWORD *v181; // rdx
  unsigned int v182; // r12d
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // rdx
  unsigned int v186; // edx
  unsigned __int64 v187; // rax
  int v188; // edx
  __int64 v189; // rax
  __int64 v190; // rdx
  unsigned int v191; // r15d
  _BYTE *v192; // r13
  int v193; // r12d
  __int64 v194; // rax
  __int64 v195; // rdx
  __int64 v196; // r8
  __int64 v197; // r9
  __int64 v198; // rdx
  unsigned __int64 v199; // rax
  unsigned __int64 v200; // rcx
  __int64 v201; // r12
  _QWORD *v202; // rax
  unsigned __int64 v203; // rax
  __int64 v204; // r12
  _QWORD *v205; // rax
  __int64 v206; // rax
  __int64 v207; // rax
  unsigned int v208; // r12d
  __int64 v209; // rax
  __int64 v210; // rdx
  __int64 v211; // rax
  _BOOL4 v212; // edx
  __int64 v213; // rax
  _BOOL4 v214; // edx
  __int64 v215; // rax
  _BOOL4 v216; // edx
  __int64 v217; // rax
  _BOOL4 v218; // edx
  __int64 v219; // rax
  _BOOL4 v220; // edx
  __int64 v221; // rax
  _BOOL4 v222; // edx
  __int64 v223; // rax
  _BOOL4 v224; // edx
  __int64 v225; // r9
  __int64 *v226; // rax
  __int64 v227; // r9
  __int64 *v228; // rax
  __int64 v229; // r11
  __int64 *v230; // rax
  __int64 v231; // rbx
  __int64 *v232; // rax
  __int64 v233; // rax
  int v234; // edx
  unsigned __int64 v235; // r13
  unsigned int v236; // edx
  unsigned __int64 v237; // rax
  signed __int64 v238; // rax
  int v239; // edx
  unsigned __int64 v240; // r13
  __int64 v241; // rax
  int v242; // edx
  int v243; // eax
  __int64 v244; // rax
  int v245; // edx
  unsigned __int64 v246; // r13
  unsigned int v247; // edx
  unsigned __int64 v248; // rax
  signed __int64 v249; // rax
  int v250; // edx
  unsigned __int64 v251; // r13
  __int64 v252; // rax
  int v253; // edx
  int v254; // eax
  __int64 v255; // rax
  int v256; // r8d
  int v257; // edx
  unsigned __int64 v258; // r13
  __int64 v259; // rax
  int v260; // r8d
  int v261; // edx
  unsigned __int64 v262; // r13
  __int64 v263; // rax
  int v264; // edx
  __int64 v265; // r13
  __int64 v266; // rax
  int v267; // edx
  unsigned __int64 v268; // r13
  __int64 v269; // rax
  int v270; // edx
  unsigned __int64 v271; // r13
  __int64 v272; // rax
  int v273; // edx
  _QWORD *v274; // rax
  _QWORD *v275; // rax
  _QWORD *v276; // rax
  _QWORD *v277; // rax
  __int64 v278; // r9
  _QWORD *v279; // rax
  bool v280; // cc
  unsigned __int64 v281; // rax
  unsigned __int64 v282; // rax
  unsigned __int64 v283; // rax
  unsigned __int64 v284; // rax
  unsigned __int64 v285; // rax
  unsigned __int64 v286; // rax
  unsigned __int64 v287; // rax
  unsigned __int64 v288; // rax
  unsigned __int64 v289; // rax
  unsigned __int64 v290; // rax
  unsigned int *v291; // [rsp+8h] [rbp-318h]
  unsigned int v292; // [rsp+8h] [rbp-318h]
  int v293; // [rsp+10h] [rbp-310h]
  unsigned int v294; // [rsp+10h] [rbp-310h]
  unsigned int *v295; // [rsp+10h] [rbp-310h]
  unsigned __int8 v296; // [rsp+10h] [rbp-310h]
  unsigned int v297; // [rsp+18h] [rbp-308h]
  unsigned int v298; // [rsp+18h] [rbp-308h]
  __int64 v299; // [rsp+18h] [rbp-308h]
  __int64 v300; // [rsp+18h] [rbp-308h]
  unsigned int *v301; // [rsp+18h] [rbp-308h]
  unsigned __int8 v302; // [rsp+18h] [rbp-308h]
  unsigned __int8 v303; // [rsp+20h] [rbp-300h]
  __int64 v304; // [rsp+20h] [rbp-300h]
  unsigned int *v305; // [rsp+20h] [rbp-300h]
  signed __int64 v306; // [rsp+28h] [rbp-2F8h]
  unsigned __int8 v307; // [rsp+28h] [rbp-2F8h]
  __int64 *v308; // [rsp+28h] [rbp-2F8h]
  unsigned __int8 v309; // [rsp+28h] [rbp-2F8h]
  unsigned __int8 v310; // [rsp+28h] [rbp-2F8h]
  __int64 v311; // [rsp+28h] [rbp-2F8h]
  unsigned int v312; // [rsp+28h] [rbp-2F8h]
  __int64 v313; // [rsp+28h] [rbp-2F8h]
  __int64 v314; // [rsp+28h] [rbp-2F8h]
  __int64 v315; // [rsp+28h] [rbp-2F8h]
  __int64 v317; // [rsp+30h] [rbp-2F0h]
  int v318; // [rsp+30h] [rbp-2F0h]
  int v319; // [rsp+30h] [rbp-2F0h]
  _BYTE *v320; // [rsp+30h] [rbp-2F0h]
  unsigned __int8 v321; // [rsp+30h] [rbp-2F0h]
  int v322; // [rsp+30h] [rbp-2F0h]
  __int64 v323; // [rsp+30h] [rbp-2F0h]
  __int64 v324; // [rsp+30h] [rbp-2F0h]
  int v325; // [rsp+30h] [rbp-2F0h]
  int v326; // [rsp+30h] [rbp-2F0h]
  unsigned int v327; // [rsp+30h] [rbp-2F0h]
  unsigned int v328; // [rsp+30h] [rbp-2F0h]
  unsigned int v330; // [rsp+40h] [rbp-2E0h] BYREF
  unsigned int v331; // [rsp+44h] [rbp-2DCh] BYREF
  unsigned int v332; // [rsp+48h] [rbp-2D8h] BYREF
  unsigned int v333; // [rsp+4Ch] [rbp-2D4h] BYREF
  __int64 v334; // [rsp+50h] [rbp-2D0h] BYREF
  __int64 v335; // [rsp+58h] [rbp-2C8h] BYREF
  __int64 v336; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v337; // [rsp+68h] [rbp-2B8h] BYREF
  unsigned __int64 *v338[4]; // [rsp+70h] [rbp-2B0h] BYREF
  unsigned __int64 *v339[4]; // [rsp+90h] [rbp-290h] BYREF
  unsigned __int64 *v340[4]; // [rsp+B0h] [rbp-270h] BYREF
  unsigned __int64 *v341[4]; // [rsp+D0h] [rbp-250h] BYREF
  unsigned __int64 *v342[4]; // [rsp+F0h] [rbp-230h] BYREF
  unsigned __int64 *v343[4]; // [rsp+110h] [rbp-210h] BYREF
  unsigned __int64 *v344[4]; // [rsp+130h] [rbp-1F0h] BYREF
  unsigned __int64 *v345[4]; // [rsp+150h] [rbp-1D0h] BYREF
  unsigned __int64 *v346[4]; // [rsp+170h] [rbp-1B0h] BYREF
  unsigned __int64 *v347[4]; // [rsp+190h] [rbp-190h] BYREF
  unsigned __int64 *v348[4]; // [rsp+1B0h] [rbp-170h] BYREF
  unsigned __int64 *v349[4]; // [rsp+1D0h] [rbp-150h] BYREF
  unsigned int *v350; // [rsp+1F0h] [rbp-130h] BYREF
  __int64 v351; // [rsp+1F8h] [rbp-128h]
  _BYTE v352[48]; // [rsp+200h] [rbp-120h] BYREF
  __int64 v353; // [rsp+230h] [rbp-F0h] BYREF
  __int64 v354; // [rsp+238h] [rbp-E8h]
  _BYTE v355[48]; // [rsp+240h] [rbp-E0h] BYREF
  __int64 v356; // [rsp+270h] [rbp-B0h] BYREF
  __int64 v357; // [rsp+278h] [rbp-A8h]
  _BYTE v358[48]; // [rsp+280h] [rbp-A0h] BYREF
  __int64 v359; // [rsp+2B0h] [rbp-70h] BYREF
  __int64 v360; // [rsp+2B8h] [rbp-68h]
  _QWORD v361[12]; // [rsp+2C0h] [rbp-60h] BYREF

  v9 = a4;
  v10 = (__int64)a3;
  v11 = *((unsigned int *)a5 + 2);
  v306 = (unsigned int)a6;
  if ( (_DWORD)v11 )
  {
    if ( (int)v11 <= 0 )
      goto LABEL_3;
LABEL_16:
    LODWORD(v14) = 1;
    return (unsigned int)v14;
  }
  if ( (unsigned int)a6 < (__int64)*a5 )
    goto LABEL_16;
LABEL_3:
  v12 = *(_QWORD *)(a2 + 8);
  if ( *(_WORD *)(v12 + 24) )
  {
    if ( !*(_BYTE *)(a8 + 28) )
      goto LABEL_11;
    v13 = *(__int64 **)(a8 + 8);
    a4 = *(unsigned int *)(a8 + 20);
    a3 = &v13[a4];
    if ( v13 != a3 )
    {
      while ( v12 != *v13 )
      {
        if ( a3 == ++v13 )
          goto LABEL_21;
      }
      goto LABEL_9;
    }
LABEL_21:
    if ( (unsigned int)a4 < *(_DWORD *)(a8 + 16) )
    {
      *(_DWORD *)(a8 + 20) = a4 + 1;
      *a3 = v12;
      ++*(_QWORD *)a8;
    }
    else
    {
LABEL_11:
      sub_C8CC70(a8, *(_QWORD *)(a2 + 8), (__int64)a3, a4, v11, a6);
      if ( !v16 )
      {
LABEL_9:
        LODWORD(v14) = 0;
        return (unsigned int)v14;
      }
    }
  }
  v303 = sub_F7E1C0(a1, (__int64 *)v12, v9, v10);
  if ( v303 )
    goto LABEL_9;
  v17 = sub_B2D610(*(_QWORD *)(**(_QWORD **)(v10 + 32) + 72LL), 18);
  LODWORD(v14) = 0;
  if ( v17 )
  {
    switch ( *(_WORD *)(v12 + 24) )
    {
      case 0:
        sub_D95540(v12);
        v65 = sub_DFB040(a7);
        v67 = v66;
        if ( v66 == 1 )
          *((_DWORD *)a5 + 2) = 1;
        else
          v67 = *((_DWORD *)a5 + 2);
        if ( __OFADD__(*a5, v65) )
        {
          v280 = v65 <= 0;
          v44 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v280 )
            v44 = 0x8000000000000000LL;
        }
        else
        {
          v44 = *a5 + v65;
        }
        *a5 = v44;
        v45 = v67 == 0;
        v46 = v67 < 0;
        if ( !v67 )
          goto LABEL_84;
        goto LABEL_54;
      case 1:
      case 0xF:
        return (unsigned int)v14;
      case 2:
      case 3:
      case 4:
      case 0xE:
        v21 = 2;
        goto LABEL_24;
      case 5:
      case 6:
      case 9:
      case 0xA:
      case 0xB:
      case 0xC:
      case 0xD:
        v18 = 2;
        goto LABEL_19;
      case 7:
        v293 = 2;
        goto LABEL_110;
      case 8:
        v68 = 2;
        goto LABEL_86;
      default:
        goto LABEL_408;
    }
  }
  switch ( *(_WORD *)(v12 + 24) )
  {
    case 0:
    case 1:
    case 0xF:
      goto LABEL_9;
    case 2:
    case 3:
    case 4:
    case 0xE:
      v21 = 0;
LABEL_24:
      v351 = 0x200000000LL;
      v22 = (__int64)&v350;
      v23 = *(_QWORD *)(a2 + 8);
      v338[3] = (unsigned __int64 *)&v330;
      v338[1] = (unsigned __int64 *)a7;
      v339[3] = (unsigned __int64 *)&v330;
      v340[3] = (unsigned __int64 *)&v330;
      v330 = v21;
      v334 = v23;
      v350 = (unsigned int *)v352;
      v338[0] = (unsigned __int64 *)&v350;
      v338[2] = (unsigned __int64 *)&v334;
      v339[0] = (unsigned __int64 *)&v350;
      v339[1] = (unsigned __int64 *)a7;
      v339[2] = (unsigned __int64 *)&v334;
      v340[0] = (unsigned __int64 *)&v350;
      v340[1] = (unsigned __int64 *)&v334;
      v340[2] = (unsigned __int64 *)a7;
      switch ( *(_WORD *)(v23 + 24) )
      {
        case 0:
        case 1:
        case 0xF:
          goto LABEL_117;
        case 2:
          v321 = 0;
          v22 = 38;
          goto LABEL_173;
        case 3:
          v321 = 0;
          v22 = 39;
          goto LABEL_173;
        case 4:
          v321 = 0;
          v22 = 40;
          goto LABEL_173;
        case 5:
          v22 = 13;
          v321 = 0;
          v153 = sub_F7B720(v339, 0xDu, 0);
          goto LABEL_174;
        case 6:
          v22 = 17;
          v321 = 0;
          v153 = sub_F7B720(v339, 0x11u, 0);
          goto LABEL_174;
        case 7:
          v155 = *(_QWORD *)(v23 + 32);
          v156 = 19;
          if ( !*(_WORD *)(v155 + 24) )
          {
            v157 = *(_QWORD *)(v155 + 32);
            if ( *(_DWORD *)(v157 + 32) > 0x40u )
            {
              v243 = sub_C44630(v157 + 24);
              LOBYTE(v14) = 0;
              if ( v243 == 1 )
                v156 = 26;
            }
            else
            {
              v158 = *(_QWORD *)(v157 + 24);
              if ( v158 )
                v156 = (v158 & (v158 - 1)) == 0 ? 26 : 19;
            }
          }
          v22 = v156;
          v321 = (unsigned __int8)v14;
          v153 = sub_F7B720(v339, v156, 1u);
          goto LABEL_174;
        case 8:
          v359 = 0;
          LODWORD(v360) = 0;
          v356 = sub_DFD270((__int64)a7, 55, v21);
          v357 = v142;
          sub_F79970((__int64)&v356, (__int64)&v359);
          v359 = 0;
          LODWORD(v360) = 0;
          v320 = (_BYTE *)v356;
          v143 = v357;
          v22 = (__int64)&v359;
          v356 = sub_DFD800((__int64)a7, 0xDu, *(_QWORD *)(v334 + 40), v330, v144, v145, 0, 0, 0, 0);
          v357 = v146;
          sub_F79970((__int64)&v356, (__int64)&v359);
          v53 = (unsigned int)v357;
          LODWORD(v14) = 0;
          if ( (_DWORD)v357 != 1 )
            v53 = v143 == 1;
          v54 = (__int64)&v320[v356];
          if ( __OFADD__(v356, v320) )
          {
            v54 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v356 <= 0 )
              v54 = 0x8000000000000000LL;
          }
          v148 = *(unsigned int *)(a9 + 12);
          v149 = *(_QWORD *)(v334 + 32);
          v150 = *(unsigned int *)(a9 + 8);
          v151 = *(_DWORD *)(a9 + 8);
          if ( v150 >= v148 )
          {
            if ( v148 < v150 + 1 )
            {
              v22 = a9 + 16;
              v328 = v53;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v150 + 1, 0x10u, v53, v147);
              LODWORD(v14) = v303;
              v53 = v328;
            }
            v279 = (_QWORD *)(*(_QWORD *)a9 + 16LL * *(unsigned int *)(a9 + 8));
            *v279 = 55;
            v279[1] = v149;
            ++*(_DWORD *)(a9 + 8);
LABEL_315:
            v55 = v350;
            v56 = &v350[6 * (unsigned int)v351];
          }
          else
          {
            v152 = (_QWORD *)(*(_QWORD *)a9 + 16 * v150);
            if ( v152 )
            {
              *v152 = 55;
              v152[1] = v149;
              v151 = *(_DWORD *)(a9 + 8);
            }
            *(_DWORD *)(a9 + 8) = v151 + 1;
LABEL_59:
            v55 = v350;
            v56 = &v350[6 * (unsigned int)v351];
          }
LABEL_60:
          if ( v55 != v56 )
          {
            do
            {
              v57 = *((_QWORD *)v55 + 1);
              v58 = 0;
              if ( v57 )
              {
                if ( *((_QWORD *)v55 + 2) <= v57 )
                  v57 = *((_QWORD *)v55 + 2);
                v58 = (unsigned int)v57;
              }
              v59 = *(unsigned int *)(a9 + 8);
              v60 = *(unsigned int *)(a9 + 12);
              v22 = *v55;
              v61 = *(_QWORD *)(v334 + 32);
              v62 = *(_DWORD *)(a9 + 8);
              if ( v59 >= v60 )
              {
                v229 = (v58 << 32) | (unsigned int)v22;
                if ( v60 < v59 + 1 )
                {
                  v22 = a9 + 16;
                  v302 = (unsigned __int8)v14;
                  v305 = v56;
                  v312 = v53;
                  v324 = v229;
                  sub_C8D5F0(a9, (const void *)(a9 + 16), v59 + 1, 0x10u, v53, (__int64)v56);
                  v59 = *(unsigned int *)(a9 + 8);
                  v229 = v324;
                  v53 = v312;
                  v56 = v305;
                  LODWORD(v14) = v302;
                }
                v230 = (__int64 *)(*(_QWORD *)a9 + 16 * v59);
                *v230 = v229;
                v230[1] = v61;
                ++*(_DWORD *)(a9 + 8);
              }
              else
              {
                v63 = *(_QWORD *)a9 + 16 * v59;
                if ( v63 )
                {
                  *(_DWORD *)v63 = v22;
                  *(_DWORD *)(v63 + 4) = v58;
                  *(_QWORD *)(v63 + 8) = v61;
                  v62 = *(_DWORD *)(a9 + 8);
                }
                *(_DWORD *)(a9 + 8) = v62 + 1;
              }
              v55 += 6;
            }
            while ( v56 != v55 );
            v56 = v350;
          }
          if ( v56 != (unsigned int *)v352 )
          {
            v307 = (unsigned __int8)v14;
            v318 = v53;
            _libc_free(v56, v22);
            LODWORD(v14) = v307;
            LODWORD(v53) = v318;
          }
          if ( (_DWORD)v53 == 1 )
            *((_DWORD *)a5 + 2) = 1;
          v64 = *a5 + v54;
          if ( __OFADD__(*a5, v54) )
          {
            v64 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v54 <= 0 )
              v64 = 0x8000000000000000LL;
          }
          break;
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
          v47 = sub_F7BD00(v340, 0x35u, 0, 1u);
          v22 = 57;
          v49 = v48;
          v50 = v47;
          v51 = sub_F7BD00(v340, 0x39u, 0, 2u);
          LODWORD(v14) = 0;
          v53 = v52;
          if ( v52 != 1 )
            v53 = v49 == 1;
          v30 = __OFADD__(v51, v50);
          v54 = v51 + v50;
          if ( v30 )
          {
            v54 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v51 <= 0 )
              v54 = 0x8000000000000000LL;
          }
          if ( *(_WORD *)(v334 + 24) != 13 )
            goto LABEL_59;
          v325 = v53;
          v255 = sub_F7BD00(v340, 0x35u, 0, 0);
          v256 = v325;
          if ( v257 == 1 )
            v256 = 1;
          v258 = v255 + v54;
          if ( __OFADD__(v255, v54) )
          {
            v258 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v255 <= 0 )
              v258 = 0x8000000000000000LL;
          }
          v326 = v256;
          v259 = sub_F7B720(v339, 0x1Du, 0);
          v260 = v326;
          if ( v261 == 1 )
            v260 = 1;
          v30 = __OFADD__(v259, v258);
          v262 = v259 + v258;
          if ( v30 )
          {
            v262 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v259 <= 0 )
              v262 = 0x8000000000000000LL;
          }
          v22 = 57;
          v327 = v260;
          v263 = sub_F7BD00(v340, 0x39u, 1u, 1u);
          v53 = v327;
          LODWORD(v14) = v303;
          if ( v264 == 1 )
            v53 = 1;
          v30 = __OFADD__(v263, v262);
          v265 = v263 + v262;
          if ( v30 )
          {
            v54 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v263 <= 0 )
              v54 = 0x8000000000000000LL;
          }
          else
          {
            v54 = v265;
          }
          goto LABEL_315;
        case 0xE:
          v321 = 0;
          v22 = 47;
LABEL_173:
          v153 = sub_F799E0(v338, v22);
LABEL_174:
          v54 = v153;
          v55 = v350;
          v53 = v154;
          LODWORD(v14) = v321;
          v56 = &v350[6 * (unsigned int)v351];
          goto LABEL_60;
        case 0x10:
          goto LABEL_408;
        default:
          v56 = (unsigned int *)v352;
          v55 = (unsigned int *)v352;
          v53 = 0;
          v54 = 0;
          goto LABEL_60;
      }
      goto LABEL_78;
    case 5:
    case 6:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
      v18 = 0;
LABEL_19:
      v356 = (__int64)v358;
      v19 = (unsigned __int64)&v356;
      v20 = *(_QWORD *)(a2 + 8);
      v357 = 0x200000000LL;
      v344[3] = (unsigned __int64 *)&v332;
      v344[1] = (unsigned __int64 *)a7;
      v345[3] = (unsigned __int64 *)&v332;
      v346[3] = (unsigned __int64 *)&v332;
      v332 = v18;
      v336 = v20;
      v344[0] = (unsigned __int64 *)&v356;
      v344[2] = (unsigned __int64 *)&v336;
      v345[0] = (unsigned __int64 *)&v356;
      v345[1] = (unsigned __int64 *)a7;
      v345[2] = (unsigned __int64 *)&v336;
      v346[0] = (unsigned __int64 *)&v356;
      v346[1] = (unsigned __int64 *)&v336;
      v346[2] = (unsigned __int64 *)a7;
      switch ( *(_WORD *)(v20 + 24) )
      {
        case 0:
        case 1:
        case 0xF:
          goto LABEL_116;
        case 2:
          v19 = 38;
          v135 = sub_F79C90(v344, 0x26u);
          goto LABEL_154;
        case 3:
          v19 = 39;
          v135 = sub_F79C90(v344, 0x27u);
          goto LABEL_154;
        case 4:
          v19 = 40;
          v135 = sub_F79C90(v344, 0x28u);
          goto LABEL_154;
        case 5:
          v19 = 13;
          v137 = *(_QWORD *)(v20 + 40) - 1;
          goto LABEL_156;
        case 6:
          v19 = 17;
          v137 = *(_QWORD *)(v20 + 40) - 1;
          goto LABEL_156;
        case 7:
          v138 = 19;
          v139 = *(_QWORD *)(*(_QWORD *)(v20 + 32) + 8LL);
          if ( !*(_WORD *)(v139 + 24) )
          {
            v140 = *(_QWORD *)(v139 + 32);
            if ( *(_DWORD *)(v140 + 32) > 0x40u )
            {
              if ( (unsigned int)sub_C44630(v140 + 24) == 1 )
                v138 = 26;
            }
            else
            {
              v141 = *(_QWORD *)(v140 + 24);
              if ( v141 )
                v138 = (v141 & (v141 - 1)) == 0 ? 26 : 19;
            }
          }
          v137 = 1;
          v19 = v138;
LABEL_156:
          v135 = sub_F7B870(v345, v19, v137);
          goto LABEL_154;
        case 8:
          v112 = *(_QWORD *)(v20 + 40);
          LODWORD(v360) = 0;
          v113 = (unsigned int)(v112 - 1);
          v359 = v113;
          v114 = sub_DFD270((__int64)a7, 55, v18);
          v354 = v115;
          v353 = v114;
          sub_F79970((__int64)&v353, (__int64)&v359);
          v359 = v113;
          LODWORD(v360) = 0;
          v297 = v332;
          v116 = (_BYTE *)v353;
          v319 = v354;
          v117 = sub_D95540(v336);
          v353 = sub_DFD800((__int64)a7, 0xDu, v117, v297, 0, 0, 0, 0, 0, 0);
          v354 = v118;
          sub_F79970((__int64)&v353, (__int64)&v359);
          v29 = v354;
          if ( (_DWORD)v354 != 1 )
            v29 = v319 == 1;
          v30 = __OFADD__(v353, v116);
          v121 = &v116[v353];
          if ( v30 )
          {
            v283 = 0x8000000000000000LL;
            if ( v353 > 0 )
              v283 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v283;
          }
          else
          {
            v317 = (__int64)v121;
          }
          v122 = v336;
          v19 = *(unsigned int *)(a9 + 12);
          v123 = **(_QWORD **)(v336 + 32);
          v124 = *(unsigned int *)(a9 + 8);
          v125 = *(_DWORD *)(a9 + 8);
          if ( v124 >= v19 )
          {
            if ( v19 < v124 + 1 )
            {
              v19 = a9 + 16;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v124 + 1, 0x10u, v119, v120);
            }
            v274 = (_QWORD *)(*(_QWORD *)a9 + 16LL * *(unsigned int *)(a9 + 8));
            *v274 = 55;
            v122 = v336;
            v274[1] = v123;
            ++*(_DWORD *)(a9 + 8);
          }
          else
          {
            v126 = (_QWORD *)(*(_QWORD *)a9 + 16 * v124);
            if ( v126 )
            {
              *v126 = 55;
              v122 = v336;
              v126[1] = v123;
              v125 = *(_DWORD *)(a9 + 8);
            }
            *(_DWORD *)(a9 + 8) = v125 + 1;
          }
          v127 = *(_QWORD *)(v122 + 32);
          v128 = (__int64 *)(v127 + 8);
          v129 = (__int64 *)(v127 + 8LL * *(_QWORD *)(v122 + 40));
          if ( (__int64 *)(v127 + 8) == v129 )
            goto LABEL_277;
          do
          {
            v132 = *(unsigned int *)(a9 + 8);
            v19 = *(unsigned int *)(a9 + 12);
            v133 = *v128;
            v131 = *(_DWORD *)(a9 + 8);
            if ( v132 < v19 )
            {
              v130 = (_QWORD *)(*(_QWORD *)a9 + 16 * v132);
              if ( v130 )
              {
                *v130 = 0x10000000DLL;
                v130[1] = v133;
                v131 = *(_DWORD *)(a9 + 8);
              }
              *(_DWORD *)(a9 + 8) = v131 + 1;
            }
            else
            {
              if ( v19 < v132 + 1 )
              {
                v19 = a9 + 16;
                sub_C8D5F0(a9, (const void *)(a9 + 16), v132 + 1, 0x10u, v132 + 1, v120);
                v132 = *(unsigned int *)(a9 + 8);
              }
              v134 = (_QWORD *)(*(_QWORD *)a9 + 16 * v132);
              *v134 = 0x10000000DLL;
              v134[1] = v133;
              ++*(_DWORD *)(a9 + 8);
            }
            ++v128;
          }
          while ( v129 != v128 );
          goto LABEL_31;
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
          v19 = 57;
          v24 = sub_F7BEA0(v346, 0x35u, (unsigned int)*(_QWORD *)(v20 + 40) - 1, 1u);
          v26 = v25;
          v27 = sub_F7BEA0(v346, 0x39u, (unsigned int)*(_QWORD *)(v336 + 40) - 1, 2u);
          v29 = v28;
          if ( v28 != 1 )
            v29 = v26 == 1;
          v30 = __OFADD__(v27, v24);
          v31 = v27 + v24;
          if ( v30 )
          {
            v280 = v27 <= 0;
            v290 = 0x8000000000000000LL;
            if ( !v280 )
              v290 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v290;
          }
          else
          {
            v317 = v31;
          }
          if ( *(_WORD *)(v336 + 24) != 13 )
            goto LABEL_31;
          v233 = sub_F7BEA0(v346, 0x35u, (unsigned int)*(_QWORD *)(v336 + 40) - 1, 0);
          if ( v234 == 1 )
            v29 = 1;
          if ( __OFADD__(v233, v317) )
          {
            v280 = v233 <= 0;
            v284 = 0x8000000000000000LL;
            if ( !v280 )
              v284 = 0x7FFFFFFFFFFFFFFFLL;
            v235 = v284;
          }
          else
          {
            v235 = v233 + v317;
          }
          v236 = 0;
          v237 = *(_QWORD *)(v336 + 40);
          if ( v237 > 2 )
            v236 = v237 - 2;
          v238 = sub_F7B870(v345, 0x1Du, v236);
          if ( v239 == 1 )
            v29 = 1;
          if ( __OFADD__(v238, v235) )
          {
            v280 = v238 <= 0;
            v289 = 0x8000000000000000LL;
            if ( !v280 )
              v289 = 0x7FFFFFFFFFFFFFFFLL;
            v240 = v289;
          }
          else
          {
            v240 = v238 + v235;
          }
          v19 = 57;
          v241 = sub_F7BEA0(v346, 0x39u, 1u, 1u);
          if ( v242 == 1 )
            v29 = 1;
          if ( __OFADD__(v241, v240) )
          {
            v280 = v241 <= 0;
            v288 = 0x8000000000000000LL;
            if ( !v280 )
              v288 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v288;
          }
          else
          {
            v317 = v241 + v240;
          }
LABEL_277:
          v32 = (unsigned int *)v356;
          v14 = (unsigned int *)(v356 + 24LL * (unsigned int)v357);
          goto LABEL_32;
        case 0xE:
          v19 = 47;
          v135 = sub_F79C90(v344, 0x2Fu);
LABEL_154:
          v317 = v135;
          v29 = v136;
LABEL_31:
          v32 = (unsigned int *)v356;
          v14 = (unsigned int *)(v356 + 24LL * (unsigned int)v357);
          goto LABEL_32;
        case 0x10:
          goto LABEL_408;
        default:
          v32 = (unsigned int *)v358;
          v29 = 0;
          v317 = 0;
          v14 = (unsigned int *)v358;
LABEL_32:
          if ( v32 != v14 )
          {
            v33 = (unsigned int)v29;
            do
            {
              v34 = 0;
              v35 = *(__int64 **)(v336 + 32);
              for ( i = &v35[*(_QWORD *)(v336 + 40)]; i != v35; ++v34 )
              {
                v37 = *((_QWORD *)v32 + 2);
                v38 = v34;
                if ( *((_QWORD *)v32 + 1) >= v34 )
                  v38 = *((_QWORD *)v32 + 1);
                v39 = *(unsigned int *)(a9 + 8);
                v40 = *(unsigned int *)(a9 + 12);
                v19 = *v32;
                v41 = *v35;
                v42 = *(_DWORD *)(a9 + 8);
                if ( v38 <= v37 )
                  v37 = v38;
                if ( v39 >= v40 )
                {
                  v225 = (v37 << 32) | (unsigned int)v19;
                  if ( v40 < v39 + 1 )
                  {
                    v19 = a9 + 16;
                    v291 = v14;
                    v294 = v33;
                    v299 = v225;
                    sub_C8D5F0(a9, (const void *)(a9 + 16), v39 + 1, 0x10u, v33, v225);
                    v14 = v291;
                    v33 = v294;
                    v225 = v299;
                    v39 = *(unsigned int *)(a9 + 8);
                  }
                  v226 = (__int64 *)(*(_QWORD *)a9 + 16 * v39);
                  *v226 = v225;
                  v226[1] = v41;
                  ++*(_DWORD *)(a9 + 8);
                }
                else
                {
                  v43 = *(_QWORD *)a9 + 16 * v39;
                  if ( v43 )
                  {
                    *(_DWORD *)v43 = v19;
                    *(_DWORD *)(v43 + 4) = v37;
                    *(_QWORD *)(v43 + 8) = v41;
                    v42 = *(_DWORD *)(a9 + 8);
                  }
                  *(_DWORD *)(a9 + 8) = v42 + 1;
                }
                ++v35;
              }
              v32 += 6;
            }
            while ( v14 != v32 );
            v14 = (unsigned int *)v356;
            v29 = v33;
          }
          if ( v14 != (unsigned int *)v358 )
            _libc_free(v14, v19);
          break;
      }
      goto LABEL_48;
    case 7:
      v293 = 0;
LABEL_110:
      v308 = (__int64 *)*a1;
      v89 = sub_D95540(v12);
      v361[1] = sub_DA2C50((__int64)v308, v89, 1, 0);
      v359 = (__int64)v361;
      v361[0] = v12;
      v360 = 0x200000002LL;
      v90 = sub_DC7EB0(v308, (__int64)&v359, 0, 0);
      if ( (_QWORD *)v359 != v361 )
        _libc_free(v359, &v359);
      LODWORD(v14) = sub_F7E1C0(a1, v90, v9, v10);
      if ( (_BYTE)v14 )
        goto LABEL_9;
      v353 = (__int64)v355;
      v91 = *(_QWORD *)(a2 + 8);
      v354 = 0x200000000LL;
      v331 = v293;
      v92 = (__int64)&v353;
      v341[1] = (unsigned __int64 *)a7;
      v341[3] = (unsigned __int64 *)&v331;
      v342[3] = (unsigned __int64 *)&v331;
      v343[3] = (unsigned __int64 *)&v331;
      v335 = v91;
      v341[0] = (unsigned __int64 *)&v353;
      v341[2] = (unsigned __int64 *)&v335;
      v342[0] = (unsigned __int64 *)&v353;
      v342[1] = (unsigned __int64 *)a7;
      v342[2] = (unsigned __int64 *)&v335;
      v343[0] = (unsigned __int64 *)&v353;
      v343[1] = (unsigned __int64 *)&v335;
      v343[2] = (unsigned __int64 *)a7;
      switch ( *(_WORD *)(v91 + 24) )
      {
        case 0:
        case 1:
        case 0xF:
LABEL_117:
          v64 = *a5;
          break;
        case 2:
          v92 = 38;
          v221 = sub_F79B30(v341, 0x26u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v221;
          v99 = v222;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 3:
          v92 = 39;
          v219 = sub_F79B30(v341, 0x27u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v219;
          v99 = v220;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 4:
          v92 = 40;
          v217 = sub_F79B30(v341, 0x28u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v217;
          v99 = v218;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 5:
          v92 = 13;
          v215 = sub_F7BA00(v342, 0xDu, 1u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v215;
          v99 = v216;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 6:
          v92 = 17;
          v213 = sub_F7BA00(v342, 0x11u, 1u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v213;
          v99 = v214;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 7:
          v207 = *(_QWORD *)(v91 + 40);
          v208 = 19;
          if ( !*(_WORD *)(v207 + 24) )
          {
            v209 = *(_QWORD *)(v207 + 32);
            if ( *(_DWORD *)(v209 + 32) > 0x40u )
            {
              v254 = sub_C44630(v209 + 24);
              LOBYTE(v14) = 0;
              if ( v254 == 1 )
                v208 = 26;
            }
            else
            {
              v210 = *(_QWORD *)(v209 + 24);
              if ( v210 )
                v208 = (v210 & (v210 - 1)) == 0 ? 26 : 19;
            }
          }
          v92 = v208;
          v309 = (unsigned __int8)v14;
          v211 = sub_F7BA00(v342, v208, 1u);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = v309;
          v100 = v211;
          v99 = v212;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 8:
          v359 = 1;
          LODWORD(v360) = 0;
          v189 = sub_DFD270((__int64)a7, 55, v293);
          v357 = v190;
          v356 = v189;
          sub_F79970((__int64)&v356, (__int64)&v359);
          v359 = 1;
          LODWORD(v360) = 0;
          v191 = v331;
          v192 = (_BYTE *)v356;
          v193 = v357;
          v194 = sub_D95540(*(_QWORD *)(v335 + 40));
          v92 = (__int64)&v359;
          v356 = sub_DFD800((__int64)a7, 0xDu, v194, v191, 0, 0, 0, 0, 0, 0);
          v357 = v195;
          sub_F79970((__int64)&v356, (__int64)&v359);
          v99 = v357;
          LODWORD(v14) = 0;
          if ( (_DWORD)v357 != 1 )
            v99 = v193 == 1;
          v100 = (__int64)&v192[v356];
          if ( __OFADD__(v356, v192) )
          {
            v100 = 0x8000000000000000LL;
            if ( v356 > 0 )
              v100 = 0x7FFFFFFFFFFFFFFFLL;
          }
          v198 = v335;
          v199 = *(unsigned int *)(a9 + 8);
          v200 = *(unsigned int *)(a9 + 12);
          v201 = *(_QWORD *)(v335 + 32);
          if ( v199 >= v200 )
          {
            if ( v200 < v199 + 1 )
            {
              v92 = a9 + 16;
              v314 = v100;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v199 + 1, 0x10u, v196, v197);
              v199 = *(unsigned int *)(a9 + 8);
              LODWORD(v14) = 0;
              v100 = v314;
            }
            v275 = (_QWORD *)(*(_QWORD *)a9 + 16 * v199);
            *v275 = 55;
            v198 = v335;
            v275[1] = v201;
            v200 = *(unsigned int *)(a9 + 12);
            v203 = (unsigned int)(*(_DWORD *)(a9 + 8) + 1);
            *(_DWORD *)(a9 + 8) = v203;
          }
          else
          {
            v202 = (_QWORD *)(*(_QWORD *)a9 + 16 * v199);
            if ( v202 )
            {
              *v202 = 55;
              v198 = v335;
              v202[1] = v201;
              v200 = *(unsigned int *)(a9 + 12);
            }
            v203 = (unsigned int)(*(_DWORD *)(a9 + 8) + 1);
            *(_DWORD *)(a9 + 8) = v203;
          }
          v204 = *(_QWORD *)(v198 + 40);
          if ( v203 >= v200 )
          {
            if ( v200 < v203 + 1 )
            {
              v315 = v100;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v203 + 1, 0x10u, v196, v197);
              v100 = v315;
              LODWORD(v14) = 0;
            }
            v92 = 0x10000000DLL;
            v277 = (_QWORD *)(*(_QWORD *)a9 + 16LL * *(unsigned int *)(a9 + 8));
            *v277 = 0x10000000DLL;
            v277[1] = v204;
            v278 = (unsigned int)v354;
            v101 = (unsigned int *)v353;
            ++*(_DWORD *)(a9 + 8);
            v102 = &v101[6 * v278];
          }
          else
          {
            v205 = (_QWORD *)(*(_QWORD *)a9 + 16 * v203);
            if ( v205 )
            {
              v92 = 0x10000000DLL;
              v205[1] = v204;
              *v205 = 0x10000000DLL;
            }
            v206 = (unsigned int)v354;
            v101 = (unsigned int *)v353;
            ++*(_DWORD *)(a9 + 8);
            v102 = &v101[6 * v206];
          }
          goto LABEL_123;
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
          v93 = sub_F7C080(v343, 0x35u, 1u);
          v92 = 57;
          v95 = v94;
          v96 = v93;
          v97 = sub_F7C080(v343, 0x39u, 2u);
          LODWORD(v14) = 0;
          v99 = v98;
          if ( !v98 )
            v99 = v95 == 1;
          v100 = v97 + v96;
          if ( __OFADD__(v97, v96) )
          {
            v100 = 0x8000000000000000LL;
            if ( v97 > 0 )
              v100 = 0x7FFFFFFFFFFFFFFFLL;
          }
          if ( *(_WORD *)(v335 + 24) == 13 )
          {
            v313 = v100;
            v266 = sub_F7C080(v343, 0x35u, 0);
            if ( v267 == 1 )
              v99 = 1;
            v268 = v266 + v313;
            if ( __OFADD__(v266, v313) )
            {
              v268 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v266 <= 0 )
                v268 = 0x8000000000000000LL;
            }
            v269 = sub_F7BA00(v342, 0x1Du, 0);
            if ( v270 == 1 )
              v99 = 1;
            v30 = __OFADD__(v269, v268);
            v271 = v269 + v268;
            if ( v30 )
            {
              v271 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v269 <= 0 )
                v271 = 0x8000000000000000LL;
            }
            v92 = 57;
            v272 = sub_F7C080(v343, 0x39u, 1u);
            LODWORD(v14) = 0;
            if ( v273 == 1 )
              v99 = 1;
            v100 = v272 + v271;
            if ( __OFADD__(v272, v271) )
            {
              v100 = 0x8000000000000000LL;
              if ( v272 > 0 )
                v100 = 0x7FFFFFFFFFFFFFFFLL;
            }
            v101 = (unsigned int *)v353;
            v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          }
          else
          {
            v101 = (unsigned int *)v353;
            v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          }
          goto LABEL_123;
        case 0xE:
          v92 = 47;
          v223 = sub_F79B30(v341, 0x2Fu);
          v101 = (unsigned int *)v353;
          LODWORD(v14) = 0;
          v100 = v223;
          v99 = v224;
          v102 = (unsigned int *)(v353 + 24LL * (unsigned int)v354);
          goto LABEL_123;
        case 0x10:
          goto LABEL_408;
        default:
          v101 = (unsigned int *)v355;
          v99 = 0;
          v100 = 0;
          v102 = (unsigned int *)v355;
LABEL_123:
          if ( v101 != v102 )
          {
            do
            {
              v103 = v335;
              for ( j = 0; ; j = 1 )
              {
                v105 = *((_QWORD *)v101 + 2);
                v106 = j;
                if ( *((_QWORD *)v101 + 1) >= j )
                  v106 = *((_QWORD *)v101 + 1);
                v107 = *(unsigned int *)(a9 + 8);
                v108 = *(unsigned int *)(a9 + 12);
                v92 = *v101;
                v109 = *(_DWORD *)(a9 + 8);
                if ( v106 <= v105 )
                  v105 = v106;
                v110 = *(_QWORD *)(v103 + 8 * j + 32);
                if ( v107 >= v108 )
                {
                  v231 = v92 | (v105 << 32);
                  if ( v108 < v107 + 1 )
                  {
                    v92 = a9 + 16;
                    v296 = (unsigned __int8)v14;
                    v301 = v102;
                    v304 = v100;
                    v311 = v103;
                    sub_C8D5F0(a9, (const void *)(a9 + 16), v107 + 1, 0x10u, v103, (__int64)v102);
                    LODWORD(v14) = v296;
                    v102 = v301;
                    v100 = v304;
                    v107 = *(unsigned int *)(a9 + 8);
                    v103 = v311;
                  }
                  v232 = (__int64 *)(*(_QWORD *)a9 + 16 * v107);
                  *v232 = v231;
                  v232[1] = v110;
                  ++*(_DWORD *)(a9 + 8);
                }
                else
                {
                  v111 = *(_QWORD *)a9 + 16 * v107;
                  if ( v111 )
                  {
                    *(_DWORD *)v111 = v92;
                    *(_DWORD *)(v111 + 4) = v105;
                    *(_QWORD *)(v111 + 8) = v110;
                    v109 = *(_DWORD *)(a9 + 8);
                  }
                  *(_DWORD *)(a9 + 8) = v109 + 1;
                }
                if ( j == 1 )
                  break;
              }
              v101 += 6;
            }
            while ( v102 != v101 );
            v102 = (unsigned int *)v353;
          }
          if ( v102 != (unsigned int *)v355 )
          {
            v310 = (unsigned __int8)v14;
            v323 = v100;
            _libc_free(v102, v92);
            LODWORD(v14) = v310;
            v100 = v323;
          }
          if ( v99 )
            *((_DWORD *)a5 + 2) = 1;
          v64 = *a5 + v100;
          if ( __OFADD__(*a5, v100) )
          {
            v64 = 0x7FFFFFFFFFFFFFFFLL;
            if ( v100 <= 0 )
              v64 = 0x8000000000000000LL;
          }
          break;
      }
LABEL_78:
      *a5 = v64;
      break;
    case 8:
      v68 = 0;
LABEL_86:
      v69 = (unsigned __int64)&v359;
      v359 = (__int64)v361;
      v70 = *(_QWORD *)(a2 + 8);
      v360 = 0x200000000LL;
      v347[3] = (unsigned __int64 *)&v333;
      v347[1] = (unsigned __int64 *)a7;
      v348[3] = (unsigned __int64 *)&v333;
      v349[3] = (unsigned __int64 *)&v333;
      v333 = v68;
      v337 = v70;
      v347[0] = (unsigned __int64 *)&v359;
      v347[2] = (unsigned __int64 *)&v337;
      v348[0] = (unsigned __int64 *)&v359;
      v348[1] = (unsigned __int64 *)a7;
      v348[2] = (unsigned __int64 *)&v337;
      v349[0] = (unsigned __int64 *)&v359;
      v349[1] = (unsigned __int64 *)&v337;
      v349[2] = (unsigned __int64 *)a7;
      switch ( *(_WORD *)(v70 + 24) )
      {
        case 0:
        case 1:
        case 0xF:
LABEL_116:
          v29 = *((_DWORD *)a5 + 2);
          v44 = *a5;
          goto LABEL_53;
        case 2:
          v69 = 38;
          v187 = sub_F79DF0(v347, 0x26u);
          goto LABEL_210;
        case 3:
          v69 = 39;
          v187 = sub_F79DF0(v347, 0x27u);
          goto LABEL_210;
        case 4:
          v69 = 40;
          v187 = sub_F79DF0(v347, 0x28u);
          goto LABEL_210;
        case 5:
          v69 = 13;
          v186 = *(_QWORD *)(v70 + 40) - 1;
          goto LABEL_209;
        case 6:
          v69 = 17;
          v186 = *(_QWORD *)(v70 + 40) - 1;
          goto LABEL_209;
        case 7:
          v182 = 19;
          v183 = *(_QWORD *)(*(_QWORD *)(v70 + 32) + 8LL);
          if ( !*(_WORD *)(v183 + 24) )
          {
            v184 = *(_QWORD *)(v183 + 32);
            if ( *(_DWORD *)(v184 + 32) > 0x40u )
            {
              if ( (unsigned int)sub_C44630(v184 + 24) == 1 )
                v182 = 26;
            }
            else
            {
              v185 = *(_QWORD *)(v184 + 24);
              if ( v185 )
                v182 = (v185 & (v185 - 1)) == 0 ? 26 : 19;
            }
          }
          v186 = 1;
          v69 = v182;
LABEL_209:
          v187 = sub_F7BB60(v348, v69, v186);
          goto LABEL_210;
        case 8:
          v159 = *(_QWORD *)(v70 + 40);
          LODWORD(v357) = 0;
          v160 = (unsigned int)(v159 - 1);
          v356 = v160;
          v161 = sub_DFD270((__int64)a7, 55, v68);
          v354 = v162;
          v353 = v161;
          sub_F79970((__int64)&v353, (__int64)&v356);
          v356 = v160;
          LODWORD(v357) = 0;
          v322 = v354;
          v298 = v333;
          v163 = (_BYTE *)v353;
          v164 = sub_D95540(**(_QWORD **)(v337 + 32));
          v353 = sub_DFD800((__int64)a7, 0xDu, v164, v298, 0, 0, 0, 0, 0, 0);
          v354 = v165;
          sub_F79970((__int64)&v353, (__int64)&v356);
          v29 = v354;
          if ( (_DWORD)v354 != 1 )
            v29 = v322 == 1;
          v30 = __OFADD__(v353, v163);
          v168 = &v163[v353];
          if ( v30 )
          {
            v286 = 0x8000000000000000LL;
            if ( v353 > 0 )
              v286 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v286;
          }
          else
          {
            v317 = (__int64)v168;
          }
          v169 = v337;
          v69 = *(unsigned int *)(a9 + 12);
          v170 = **(_QWORD **)(v337 + 32);
          v171 = *(unsigned int *)(a9 + 8);
          v172 = *(_DWORD *)(a9 + 8);
          if ( v171 >= v69 )
          {
            if ( v69 < v171 + 1 )
            {
              v69 = a9 + 16;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v171 + 1, 0x10u, v166, v167);
            }
            v276 = (_QWORD *)(*(_QWORD *)a9 + 16LL * *(unsigned int *)(a9 + 8));
            *v276 = 55;
            v169 = v337;
            v276[1] = v170;
            ++*(_DWORD *)(a9 + 8);
          }
          else
          {
            v173 = (_QWORD *)(*(_QWORD *)a9 + 16 * v171);
            if ( v173 )
            {
              *v173 = 55;
              v169 = v337;
              v173[1] = v170;
              v172 = *(_DWORD *)(a9 + 8);
            }
            *(_DWORD *)(a9 + 8) = v172 + 1;
          }
          v174 = *(_QWORD *)(v169 + 32);
          v175 = (__int64 *)(v174 + 8);
          v176 = (__int64 *)(v174 + 8LL * *(_QWORD *)(v169 + 40));
          if ( (__int64 *)(v174 + 8) == v176 )
            goto LABEL_301;
          do
          {
            v179 = *(unsigned int *)(a9 + 8);
            v69 = *(unsigned int *)(a9 + 12);
            v180 = *v175;
            v178 = *(_DWORD *)(a9 + 8);
            if ( v179 < v69 )
            {
              v177 = (_QWORD *)(*(_QWORD *)a9 + 16 * v179);
              if ( v177 )
              {
                *v177 = 0x10000000DLL;
                v177[1] = v180;
                v178 = *(_DWORD *)(a9 + 8);
              }
              *(_DWORD *)(a9 + 8) = v178 + 1;
            }
            else
            {
              if ( v69 < v179 + 1 )
              {
                v69 = a9 + 16;
                sub_C8D5F0(a9, (const void *)(a9 + 16), v179 + 1, 0x10u, v179 + 1, v167);
                v179 = *(unsigned int *)(a9 + 8);
              }
              v181 = (_QWORD *)(*(_QWORD *)a9 + 16 * v179);
              *v181 = 0x10000000DLL;
              v181[1] = v180;
              ++*(_DWORD *)(a9 + 8);
            }
            ++v175;
          }
          while ( v176 != v175 );
          goto LABEL_92;
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
          v69 = 57;
          v71 = sub_F7C220(v349, 0x35u, (unsigned int)*(_QWORD *)(v70 + 40) - 1, 1u);
          v73 = v72;
          v74 = sub_F7C220(v349, 0x39u, (unsigned int)*(_QWORD *)(v337 + 40) - 1, 2u);
          v29 = v75;
          if ( v75 != 1 )
            v29 = v73 == 1;
          v30 = __OFADD__(v74, v71);
          v76 = v74 + v71;
          if ( v30 )
          {
            v280 = v74 <= 0;
            v287 = 0x8000000000000000LL;
            if ( !v280 )
              v287 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v287;
          }
          else
          {
            v317 = v76;
          }
          if ( *(_WORD *)(v337 + 24) != 13 )
            goto LABEL_92;
          v244 = sub_F7C220(v349, 0x35u, (unsigned int)*(_QWORD *)(v337 + 40) - 1, 0);
          if ( v245 == 1 )
            v29 = 1;
          if ( __OFADD__(v244, v317) )
          {
            v280 = v244 <= 0;
            v285 = 0x8000000000000000LL;
            if ( !v280 )
              v285 = 0x7FFFFFFFFFFFFFFFLL;
            v246 = v285;
          }
          else
          {
            v246 = v244 + v317;
          }
          v247 = 0;
          v248 = *(_QWORD *)(v337 + 40);
          if ( v248 > 2 )
            v247 = v248 - 2;
          v249 = sub_F7BB60(v348, 0x1Du, v247);
          if ( v250 == 1 )
            v29 = 1;
          if ( __OFADD__(v249, v246) )
          {
            v280 = v249 <= 0;
            v282 = 0x8000000000000000LL;
            if ( !v280 )
              v282 = 0x7FFFFFFFFFFFFFFFLL;
            v251 = v282;
          }
          else
          {
            v251 = v249 + v246;
          }
          v69 = 57;
          v252 = sub_F7C220(v349, 0x39u, 1u, 1u);
          if ( v253 == 1 )
            v29 = 1;
          if ( __OFADD__(v252, v251) )
          {
            v280 = v252 <= 0;
            v281 = 0x8000000000000000LL;
            if ( !v280 )
              v281 = 0x7FFFFFFFFFFFFFFFLL;
            v317 = v281;
          }
          else
          {
            v317 = v252 + v251;
          }
LABEL_301:
          v77 = (unsigned int *)v359;
          v14 = (unsigned int *)(v359 + 24LL * (unsigned int)v360);
          goto LABEL_93;
        case 0xE:
          v69 = 47;
          v187 = sub_F79DF0(v347, 0x2Fu);
LABEL_210:
          v317 = v187;
          v29 = v188;
LABEL_92:
          v77 = (unsigned int *)v359;
          v14 = (unsigned int *)(v359 + 24LL * (unsigned int)v360);
          goto LABEL_93;
        case 0x10:
          goto LABEL_408;
        default:
          v77 = (unsigned int *)v361;
          v29 = 0;
          v317 = 0;
          v14 = (unsigned int *)v361;
LABEL_93:
          if ( v14 != v77 )
          {
            v78 = (unsigned int)v29;
            do
            {
              v79 = 0;
              v80 = *(__int64 **)(v337 + 32);
              for ( k = &v80[*(_QWORD *)(v337 + 40)]; k != v80; ++v79 )
              {
                v82 = *((_QWORD *)v77 + 2);
                v83 = v79;
                if ( *((_QWORD *)v77 + 1) >= v79 )
                  v83 = *((_QWORD *)v77 + 1);
                v84 = *(unsigned int *)(a9 + 8);
                v85 = *(unsigned int *)(a9 + 12);
                v69 = *v77;
                v86 = *v80;
                v87 = *(_DWORD *)(a9 + 8);
                if ( v83 <= v82 )
                  v82 = v83;
                if ( v84 >= v85 )
                {
                  v227 = (v82 << 32) | (unsigned int)v69;
                  if ( v85 < v84 + 1 )
                  {
                    v69 = a9 + 16;
                    v292 = v78;
                    v295 = v14;
                    v300 = v227;
                    sub_C8D5F0(a9, (const void *)(a9 + 16), v84 + 1, 0x10u, v78, v227);
                    v78 = v292;
                    v14 = v295;
                    v227 = v300;
                    v84 = *(unsigned int *)(a9 + 8);
                  }
                  v228 = (__int64 *)(*(_QWORD *)a9 + 16 * v84);
                  *v228 = v227;
                  v228[1] = v86;
                  ++*(_DWORD *)(a9 + 8);
                }
                else
                {
                  v88 = *(_QWORD *)a9 + 16 * v84;
                  if ( v88 )
                  {
                    *(_DWORD *)v88 = v69;
                    *(_DWORD *)(v88 + 4) = v82;
                    *(_QWORD *)(v88 + 8) = v86;
                    v87 = *(_DWORD *)(a9 + 8);
                  }
                  *(_DWORD *)(a9 + 8) = v87 + 1;
                }
                ++v80;
              }
              v77 += 6;
            }
            while ( v14 != v77 );
            v77 = (unsigned int *)v359;
            v29 = v78;
          }
          if ( v77 != (unsigned int *)v361 )
            _libc_free(v77, v69);
          break;
      }
LABEL_48:
      if ( v29 == 1 )
        *((_DWORD *)a5 + 2) = 1;
      else
        v29 = *((_DWORD *)a5 + 2);
      v44 = *a5 + v317;
      if ( __OFADD__(*a5, v317) )
      {
        v44 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v317 <= 0 )
          v44 = 0x8000000000000000LL;
      }
LABEL_53:
      *a5 = v44;
      v45 = v29 == 0;
      v46 = v29 < 0;
      if ( v29 )
LABEL_54:
        LOBYTE(v14) = !v46 && !v45;
      else
LABEL_84:
        LOBYTE(v14) = v306 < v44;
      break;
    default:
LABEL_408:
      BUG();
  }
  return (unsigned int)v14;
}
