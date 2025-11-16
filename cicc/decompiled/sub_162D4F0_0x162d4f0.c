// Function: sub_162D4F0
// Address: 0x162d4f0
//
unsigned int *__fastcall sub_162D4F0(
        __int64 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  int v11; // eax
  __int64 v12; // rax
  int v13; // r12d
  __int64 v14; // r12
  unsigned int v15; // esi
  unsigned int **v16; // rdx
  int v17; // eax
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  int v21; // eax
  __int64 v22; // r14
  __int64 v23; // r13
  unsigned int v24; // esi
  int v25; // eax
  _QWORD *v26; // rax
  _QWORD *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned int v30; // esi
  int v31; // eax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // rbx
  int v35; // eax
  __int64 v36; // r13
  unsigned int v37; // esi
  int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *v41; // rbx
  int v42; // eax
  __int64 v43; // r13
  unsigned int v44; // esi
  int v45; // eax
  _QWORD *v46; // rax
  _QWORD *v47; // rbx
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // r13
  unsigned int v51; // esi
  int v52; // eax
  _QWORD *v53; // rax
  unsigned int *result; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  _QWORD *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // r13
  unsigned int v60; // esi
  int v61; // eax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  _QWORD *v64; // rbx
  int v65; // r15d
  __int64 v66; // r14
  __int64 v67; // r13
  unsigned int v68; // esi
  int v69; // eax
  _QWORD *v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  _QWORD *v79; // rax
  _QWORD *v80; // rbx
  unsigned int **v81; // rax
  __int64 v82; // rax
  __int64 v83; // r13
  unsigned int v84; // esi
  int v85; // eax
  _QWORD *v86; // rax
  _QWORD *v87; // rax
  _QWORD *v88; // rax
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rax
  __int64 v94; // rdx
  _QWORD *v95; // rbx
  int v96; // r14d
  __int64 v97; // r15
  __int64 v98; // r13
  unsigned int v99; // esi
  int v100; // eax
  _QWORD *v101; // rax
  _QWORD *v102; // rax
  unsigned int v103; // edx
  unsigned int **v104; // rax
  unsigned int *v105; // rcx
  int v106; // edi
  int v107; // r14d
  unsigned int v108; // ecx
  unsigned int **v109; // rax
  unsigned int *v110; // rdx
  int v111; // r9d
  int v112; // eax
  __int64 v113; // r9
  unsigned int v114; // esi
  unsigned int **v115; // rdx
  unsigned int *v116; // rcx
  __int64 v117; // rdi
  int v118; // eax
  __int64 v119; // r11
  unsigned int v120; // ecx
  unsigned int **v121; // rax
  unsigned int *v122; // rdx
  __int64 v123; // rsi
  unsigned int v124; // edx
  unsigned int **v125; // rax
  unsigned int *v126; // rcx
  int v127; // r8d
  int v128; // r12d
  unsigned int v129; // eax
  __int64 v130; // r10
  unsigned int v131; // ecx
  unsigned int **v132; // rax
  unsigned int *v133; // rdx
  unsigned int v134; // ecx
  unsigned int **v135; // rdx
  unsigned int *v136; // rsi
  int v137; // r9d
  unsigned int v138; // ecx
  unsigned int **v139; // rdx
  unsigned int *v140; // rsi
  int v141; // r10d
  unsigned int v142; // ecx
  unsigned int **v143; // rdx
  unsigned int *v144; // rsi
  int v145; // r9d
  unsigned int v146; // esi
  unsigned int **v147; // rdx
  unsigned int *v148; // rcx
  int v149; // r10d
  __int64 v150; // rdi
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rcx
  unsigned int v154; // ecx
  unsigned int *v155; // rsi
  int v156; // r10d
  unsigned int **v157; // rdi
  __int64 v158; // r8
  __int64 v159; // r9
  __int64 v160; // rcx
  unsigned int v161; // ecx
  unsigned int *v162; // rsi
  int v163; // r10d
  unsigned int **v164; // rdi
  __int64 v165; // r9
  __int64 v166; // r14
  int v167; // r14d
  __int64 v168; // rcx
  unsigned int v169; // ecx
  unsigned int *v170; // rsi
  int v171; // r8d
  unsigned int **v172; // rdi
  __int64 v173; // r11
  __int64 v174; // r10
  __int64 v175; // rdx
  unsigned int v176; // ecx
  unsigned int *v177; // rsi
  int v178; // r8d
  unsigned int **v179; // rdi
  __int64 v180; // rsi
  unsigned int *v181; // rdi
  __int64 v182; // r10
  __int64 v183; // r12
  __int64 v184; // rdx
  unsigned int *v185; // rcx
  int v186; // r12d
  unsigned int v187; // ecx
  unsigned int *v188; // rsi
  int v189; // r8d
  unsigned int **v190; // rdi
  __int64 v191; // r11
  __int64 v192; // r10
  __int64 v193; // rdx
  unsigned int v194; // ecx
  unsigned int *v195; // rsi
  int v196; // r8d
  unsigned int **v197; // rdi
  __int64 v198; // r15
  __int64 v199; // r14
  int v200; // r14d
  __int64 v201; // rcx
  unsigned int v202; // ecx
  unsigned int *v203; // rsi
  int v204; // r8d
  unsigned int **v205; // rdi
  __int64 v206; // r10
  __int64 v207; // r9
  __int64 v208; // rdx
  unsigned int v209; // ecx
  unsigned int *v210; // rsi
  int v211; // r8d
  unsigned int **v212; // rdi
  __int64 v213; // r11
  __int64 v214; // r10
  __int64 v215; // rdx
  unsigned int v216; // ecx
  unsigned int *v217; // rsi
  int v218; // r8d
  unsigned int **v219; // rdi
  __int64 v220; // r15
  __int64 v221; // r14
  int v222; // r14d
  __int64 v223; // rcx
  unsigned int v224; // ecx
  unsigned int *v225; // rsi
  int v226; // r8d
  unsigned int **v227; // rdi
  int v228; // eax
  int v229; // eax
  int v230; // eax
  int v231; // eax
  int v232; // eax
  int v233; // eax
  int v234; // eax
  int v235; // eax
  int v236; // eax
  int v237; // eax
  __int64 v238; // [rsp+20h] [rbp-E0h]
  __int64 v239; // [rsp+30h] [rbp-D0h]
  int v240; // [rsp+48h] [rbp-B8h]
  int v241; // [rsp+48h] [rbp-B8h]
  __int64 v242; // [rsp+48h] [rbp-B8h]
  __int64 v243; // [rsp+48h] [rbp-B8h]
  __int64 v244; // [rsp+50h] [rbp-B0h]
  __int64 v245; // [rsp+50h] [rbp-B0h]
  int v246; // [rsp+50h] [rbp-B0h]
  int v247; // [rsp+50h] [rbp-B0h]
  __int64 v248; // [rsp+50h] [rbp-B0h]
  __int64 v249; // [rsp+50h] [rbp-B0h]
  int v250; // [rsp+50h] [rbp-B0h]
  __int64 v251; // [rsp+50h] [rbp-B0h]
  int v252; // [rsp+50h] [rbp-B0h]
  __int64 v253; // [rsp+58h] [rbp-A8h]
  int v254; // [rsp+58h] [rbp-A8h]
  __int64 v255; // [rsp+58h] [rbp-A8h]
  int v256; // [rsp+58h] [rbp-A8h]
  int v257; // [rsp+58h] [rbp-A8h]
  __int64 v258; // [rsp+58h] [rbp-A8h]
  int v259; // [rsp+58h] [rbp-A8h]
  __int64 v260; // [rsp+58h] [rbp-A8h]
  __int64 v261; // [rsp+60h] [rbp-A0h]
  int v262; // [rsp+60h] [rbp-A0h]
  int v263; // [rsp+60h] [rbp-A0h]
  int v264; // [rsp+60h] [rbp-A0h]
  __int64 v265; // [rsp+60h] [rbp-A0h]
  __int64 v266; // [rsp+60h] [rbp-A0h]
  int v267; // [rsp+60h] [rbp-A0h]
  __int64 v268; // [rsp+68h] [rbp-98h]
  __int64 v269; // [rsp+68h] [rbp-98h]
  int v270; // [rsp+68h] [rbp-98h]
  int v271; // [rsp+68h] [rbp-98h]
  __int64 v272; // [rsp+68h] [rbp-98h]
  unsigned int *v273; // [rsp+78h] [rbp-88h] BYREF
  unsigned int **v274; // [rsp+80h] [rbp-80h] BYREF
  __int64 v275; // [rsp+88h] [rbp-78h] BYREF
  unsigned int *v276; // [rsp+90h] [rbp-70h] BYREF
  __int64 v277; // [rsp+98h] [rbp-68h] BYREF
  __int64 v278; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v279; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v280; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v281; // [rsp+B8h] [rbp-48h]
  unsigned int v282; // [rsp+C0h] [rbp-40h]
  unsigned int v283; // [rsp+C4h] [rbp-3Ch]
  __int64 v284[7]; // [rsp+C8h] [rbp-38h] BYREF

  switch ( *(_BYTE *)a1 )
  {
    case 4:
      sub_161F280(a1);
      v78 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v78 = (_QWORD *)*v78;
      return (unsigned int *)sub_1629450(a1, *v78 + 496LL);
    case 5:
      v77 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v77 = (_QWORD *)*v77;
      return (unsigned int *)sub_16295D0(a1, *v77 + 528LL);
    case 6:
      v76 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v76 = (_QWORD *)*v76;
      return (unsigned int *)sub_16298C0(a1, *v76 + 560LL);
    case 7:
      v75 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v75 = (_QWORD *)*v75;
      return (unsigned int *)sub_1629AE0(a1, *v75 + 592LL);
    case 8:
      sub_15B0CB0(a1);
      v74 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v74 = (_QWORD *)*v74;
      return (unsigned int *)sub_1629D40(a1, *v74 + 624LL);
    case 9:
      v73 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v73 = (_QWORD *)*v73;
      return (unsigned int *)sub_162A180(a1, *v73 + 656LL);
    case 0xA:
      v72 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v72 = (_QWORD *)*v72;
      return (unsigned int *)sub_162A3D0(a1, *v72 + 688LL);
    case 0xB:
      v91 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v91 = (_QWORD *)*v91;
      return (unsigned int *)sub_162A650(a1, *v91 + 720LL);
    case 0xC:
      v90 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v90 = (_QWORD *)*v90;
      return (unsigned int *)sub_162A940(a1, *v90 + 752LL);
    case 0xD:
      v87 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v87 = (_QWORD *)*v87;
      return (unsigned int *)sub_162AE30(a1, *v87 + 784LL);
    case 0xE:
      v86 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v86 = (_QWORD *)*v86;
      return (unsigned int *)sub_162B2E0(a1, *v86 + 816LL);
    case 0xF:
      v89 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v89 = (_QWORD *)*v89;
      return (unsigned int *)sub_162B570(a1, *v89 + 848LL, a2, a3, a4, a5, a6, a7, a8, a9);
    case 0x10:
    case 0x22:
      v79 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v79 = (_QWORD *)*v79;
      v80 = (_QWORD *)*v79;
      v81 = *(unsigned int ***)(a1 + 24);
      v273 = (unsigned int *)a1;
      v274 = v81;
      v275 = *(_QWORD *)(a1 + 32);
      LOBYTE(v276) = *(_BYTE *)(a1 + 40);
      v82 = *(unsigned int *)(a1 + 8);
      v277 = *(_QWORD *)(a1 - 8 * v82);
      v278 = *(_QWORD *)(a1 + 8 * (1 - v82));
      v279 = *(_QWORD *)(a1 + 8 * (2 - v82));
      v280 = *(_QWORD *)(a1 + 8 * (3 - v82));
      if ( !*((_DWORD *)v80 + 362) )
        goto LABEL_104;
      v240 = *((_DWORD *)v80 + 362);
      v244 = v80[179];
      v103 = (v240 - 1) & sub_15B3480((__int64 *)&v274, &v275, (char *)&v276, &v279, &v280, &v277, &v278);
      v104 = (unsigned int **)(v244 + 8LL * v103);
      v105 = *v104;
      if ( *v104 == (unsigned int *)-8LL )
        goto LABEL_367;
      v106 = 1;
      while ( 2 )
      {
        if ( v105 != (unsigned int *)-16LL
          && v274 == *((unsigned int ***)v105 + 3)
          && v275 == *((_QWORD *)v105 + 4)
          && (_BYTE)v276 == *((_BYTE *)v105 + 40)
          && (v238 = v105[2], v277 == *(_QWORD *)&v105[-2 * v238])
          && v278 == *(_QWORD *)&v105[2 * (1 - v238)]
          && v279 == *(_QWORD *)&v105[2 * (2 - v238)]
          && v280 == *(_QWORD *)&v105[2 * (3 - v238)] )
        {
          v173 = v80[179];
          v174 = *((unsigned int *)v80 + 362);
          if ( v104 != (unsigned int **)(v173 + 8 * v174) )
          {
            result = *v104;
            if ( result )
              return result;
          }
        }
        else
        {
          v103 = (v240 - 1) & (v106 + v103);
          v104 = (unsigned int **)(v244 + 8LL * v103);
          v105 = *v104;
          if ( *v104 != (unsigned int *)-8LL )
          {
            ++v106;
            continue;
          }
LABEL_367:
          v173 = v80[179];
          LODWORD(v174) = *((_DWORD *)v80 + 362);
        }
        break;
      }
      v242 = v173;
      v250 = v174;
      if ( !(_DWORD)v174 )
      {
LABEL_104:
        ++v80[178];
        v83 = (__int64)(v80 + 178);
        v84 = 0;
        goto LABEL_105;
      }
      v274 = (unsigned int **)*((_QWORD *)v273 + 3);
      v275 = *((_QWORD *)v273 + 4);
      LOBYTE(v276) = *((_BYTE *)v273 + 40);
      v175 = v273[2];
      v277 = *(_QWORD *)&v273[-2 * v175];
      v278 = *(_QWORD *)&v273[2 * (1 - v175)];
      v279 = *(_QWORD *)&v273[2 * (2 - v175)];
      v280 = *(_QWORD *)&v273[2 * (3 - v175)];
      v176 = (v174 - 1) & sub_15B3480((__int64 *)&v274, &v275, (char *)&v276, &v279, &v280, &v277, &v278);
      v16 = (unsigned int **)(v242 + 8LL * v176);
      result = v273;
      v177 = *v16;
      if ( *v16 != v273 )
      {
        v178 = 1;
        v179 = 0;
        while ( v177 != (unsigned int *)-8LL )
        {
          if ( !v179 && v177 == (unsigned int *)-16LL )
            v179 = v16;
          v176 = (v250 - 1) & (v178 + v176);
          v16 = (unsigned int **)(v242 + 8LL * v176);
          v177 = *v16;
          if ( v273 == *v16 )
            return result;
          ++v178;
        }
        v235 = *((_DWORD *)v80 + 360);
        v84 = *((_DWORD *)v80 + 362);
        v83 = (__int64)(v80 + 178);
        if ( v179 )
          v16 = v179;
        ++v80[178];
        v85 = v235 + 1;
        if ( 4 * v85 < 3 * v84 )
        {
          if ( v84 - (v85 + *((_DWORD *)v80 + 361)) > v84 >> 3 )
          {
LABEL_107:
            *((_DWORD *)v80 + 360) = v85;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v80 + 361);
            goto LABEL_64;
          }
LABEL_106:
          sub_15BB7A0(v83, v84);
          sub_15B7680(v83, (__int64 *)&v273, &v274);
          v16 = v274;
          v85 = *((_DWORD *)v80 + 360) + 1;
          goto LABEL_107;
        }
LABEL_105:
        v84 *= 2;
        goto LABEL_106;
      }
      return result;
    case 0x11:
      v88 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v88 = (_QWORD *)*v88;
      return (unsigned int *)sub_162B9C0(a1, *v88 + 880LL);
    case 0x12:
      v55 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v55 = (_QWORD *)*v55;
      return (unsigned int *)sub_162BD00(a1, *v55 + 912LL);
    case 0x13:
      v53 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v53 = (_QWORD *)*v53;
      return (unsigned int *)sub_162C000(a1, *v53 + 944LL);
    case 0x14:
      v102 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v102 = (_QWORD *)*v102;
      return (unsigned int *)sub_162C2C0(a1, *v102 + 976LL);
    case 0x15:
      v101 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v101 = (_QWORD *)*v101;
      return (unsigned int *)sub_162C570(a1, *v101 + 1008LL);
    case 0x16:
      v93 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      v94 = *(unsigned int *)(a1 + 8);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v93 = (_QWORD *)*v93;
      v95 = (_QWORD *)*v93;
      v273 = (unsigned int *)a1;
      v274 = *(unsigned int ***)(a1 - 8 * v94);
      v275 = *(_QWORD *)(a1 + 8 * (1 - v94));
      v96 = *((_DWORD *)v95 + 266);
      v97 = v95[131];
      if ( !v96 )
        goto LABEL_133;
      v107 = v96 - 1;
      v108 = v107 & sub_15B2D00((__int64 *)&v274, &v275);
      v109 = (unsigned int **)(v97 + 8LL * v108);
      v110 = *v109;
      if ( *v109 == (unsigned int *)-8LL )
        goto LABEL_366;
      v111 = 1;
      while ( 2 )
      {
        if ( v110 != (unsigned int *)-16LL
          && v274 == *(unsigned int ***)&v110[-2 * v110[2]]
          && v275 == *(_QWORD *)&v110[2 * (1LL - v110[2])] )
        {
          v198 = v95[131];
          v199 = *((unsigned int *)v95 + 266);
          if ( v109 != (unsigned int **)(v198 + 8 * v199) )
          {
            result = *v109;
            if ( result )
              return result;
          }
        }
        else
        {
          v108 = v107 & (v111 + v108);
          v109 = (unsigned int **)(v97 + 8LL * v108);
          v110 = *v109;
          if ( *v109 != (unsigned int *)-8LL )
          {
            ++v111;
            continue;
          }
LABEL_366:
          v198 = v95[131];
          LODWORD(v199) = *((_DWORD *)v95 + 266);
        }
        break;
      }
      if ( (_DWORD)v199 )
      {
        v200 = v199 - 1;
        v201 = v273[2];
        v274 = *(unsigned int ***)&v273[-2 * v201];
        v275 = *(_QWORD *)&v273[2 * (1 - v201)];
        v202 = v200 & sub_15B2D00((__int64 *)&v274, &v275);
        v16 = (unsigned int **)(v198 + 8LL * v202);
        result = v273;
        v203 = *v16;
        if ( v273 == *v16 )
          return result;
        v204 = 1;
        v205 = 0;
        while ( v203 != (unsigned int *)-8LL )
        {
          if ( v203 == (unsigned int *)-16LL && !v205 )
            v205 = v16;
          v202 = v200 & (v204 + v202);
          v16 = (unsigned int **)(v198 + 8LL * v202);
          v203 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v204;
        }
        v234 = *((_DWORD *)v95 + 264);
        v99 = *((_DWORD *)v95 + 266);
        v98 = (__int64)(v95 + 130);
        if ( v205 )
          v16 = v205;
        ++v95[130];
        v100 = v234 + 1;
        if ( 4 * v100 < 3 * v99 )
        {
          if ( v99 - (v100 + *((_DWORD *)v95 + 265)) > v99 >> 3 )
          {
LABEL_136:
            *((_DWORD *)v95 + 264) = v100;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v95 + 265);
            goto LABEL_64;
          }
LABEL_135:
          sub_15C2120(v98, v99);
          sub_15B8C40(v98, (__int64 *)&v273, &v274);
          v16 = v274;
          v100 = *((_DWORD *)v95 + 264) + 1;
          goto LABEL_136;
        }
      }
      else
      {
LABEL_133:
        ++v95[130];
        v98 = (__int64)(v95 + 130);
        v99 = 0;
      }
      v99 *= 2;
      goto LABEL_135;
    case 0x17:
      v92 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v92 = (_QWORD *)*v92;
      return (unsigned int *)sub_162C8F0(a1, *v92 + 1072LL);
    case 0x18:
      v71 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v71 = (_QWORD *)*v71;
      return (unsigned int *)sub_162CBF0(a1, *v71 + 1104LL);
    case 0x19:
      v70 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v70 = (_QWORD *)*v70;
      return (unsigned int *)sub_162D0E0(a1, *v70 + 1136LL);
    case 0x1A:
      v62 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      v63 = *(unsigned int *)(a1 + 8);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v62 = (_QWORD *)*v62;
      v64 = (_QWORD *)*v62;
      v273 = (unsigned int *)a1;
      v274 = *(unsigned int ***)(a1 - 8 * v63);
      v275 = *(_QWORD *)(a1 + 8 * (1 - v63));
      v276 = *(unsigned int **)(a1 + 8 * (2 - v63));
      LODWORD(v277) = *(_DWORD *)(a1 + 24);
      v65 = *((_DWORD *)v64 + 298);
      v66 = v64[147];
      if ( !v65 )
        goto LABEL_69;
      v146 = (v65 - 1) & sub_15B3EF0((__int64 *)&v274, &v275, (int *)&v277);
      v147 = (unsigned int **)(v66 + 8LL * v146);
      v148 = *v147;
      if ( *v147 == (unsigned int *)-8LL )
        goto LABEL_360;
      v149 = 1;
      while ( 2 )
      {
        if ( v148 != (unsigned int *)-16LL
          && (v150 = v148[2], v274 == *(unsigned int ***)&v148[-2 * v150])
          && v275 == *(_QWORD *)&v148[2 * (1 - v150)]
          && v276 == *(unsigned int **)&v148[2 * (2 - v150)]
          && (_DWORD)v277 == v148[6] )
        {
          v220 = v64[147];
          v221 = *((unsigned int *)v64 + 298);
          if ( v147 != (unsigned int **)(v220 + 8 * v221) )
          {
            result = *v147;
            if ( *v147 )
              return result;
          }
        }
        else
        {
          v146 = (v65 - 1) & (v149 + v146);
          v147 = (unsigned int **)(v66 + 8LL * v146);
          v148 = *v147;
          if ( *v147 != (unsigned int *)-8LL )
          {
            ++v149;
            continue;
          }
LABEL_360:
          v220 = v64[147];
          LODWORD(v221) = *((_DWORD *)v64 + 298);
        }
        break;
      }
      if ( (_DWORD)v221 )
      {
        v222 = v221 - 1;
        v223 = v273[2];
        v274 = *(unsigned int ***)&v273[-2 * v223];
        v275 = *(_QWORD *)&v273[2 * (1 - v223)];
        v276 = *(unsigned int **)&v273[2 * (2 - v223)];
        LODWORD(v277) = v273[6];
        v224 = v222 & sub_15B3EF0((__int64 *)&v274, &v275, (int *)&v277);
        v16 = (unsigned int **)(v220 + 8LL * v224);
        result = v273;
        v225 = *v16;
        if ( v273 == *v16 )
          return result;
        v226 = 1;
        v227 = 0;
        while ( v225 != (unsigned int *)-8LL )
        {
          if ( v225 == (unsigned int *)-16LL && !v227 )
            v227 = v16;
          v224 = v222 & (v226 + v224);
          v16 = (unsigned int **)(v220 + 8LL * v224);
          v225 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v226;
        }
        v232 = *((_DWORD *)v64 + 296);
        v68 = *((_DWORD *)v64 + 298);
        v67 = (__int64)(v64 + 146);
        if ( v227 )
          v16 = v227;
        ++v64[146];
        v69 = v232 + 1;
        if ( 4 * v69 < 3 * v68 )
        {
          if ( v68 - (v69 + *((_DWORD *)v64 + 297)) > v68 >> 3 )
          {
LABEL_72:
            *((_DWORD *)v64 + 296) = v69;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v64 + 297);
            goto LABEL_64;
          }
LABEL_71:
          sub_15C3AB0(v67, v68);
          sub_15B9100(v67, (__int64 *)&v273, &v274);
          v16 = v274;
          v69 = *((_DWORD *)v64 + 296) + 1;
          goto LABEL_72;
        }
      }
      else
      {
LABEL_69:
        ++v64[146];
        v67 = (__int64)(v64 + 146);
        v68 = 0;
      }
      v68 *= 2;
      goto LABEL_71;
    case 0x1B:
      v56 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v56 = (_QWORD *)*v56;
      v57 = (_QWORD *)*v56;
      v58 = *(unsigned int *)(a1 + 8);
      v273 = (unsigned int *)a1;
      v274 = *(unsigned int ***)(a1 - 8 * v58);
      v275 = *(_QWORD *)(a1 + 8 * (1 - v58));
      LODWORD(v276) = *(_DWORD *)(a1 + 24);
      v277 = *(_QWORD *)(a1 + 8 * (2 - v58));
      v278 = *(_QWORD *)(a1 + 8 * (3 - v58));
      LODWORD(v279) = *(_DWORD *)(a1 + 28);
      v280 = *(_QWORD *)(a1 + 8 * (4 - v58));
      if ( !*((_DWORD *)v57 + 306) )
        goto LABEL_59;
      v241 = *((_DWORD *)v57 + 306);
      v245 = v57[151];
      v118 = sub_15B52D0((__int64 *)&v274, &v275, (int *)&v276, &v277, &v278, (int *)&v279, &v280);
      v119 = v245;
      v120 = (v241 - 1) & v118;
      v121 = (unsigned int **)(v245 + 8LL * v120);
      v122 = *v121;
      if ( *v121 == (unsigned int *)-8LL )
        goto LABEL_362;
      v246 = 1;
      while ( 2 )
      {
        if ( v122 != (unsigned int *)-16LL
          && (v123 = v122[2], v274 == *(unsigned int ***)&v122[-2 * v123])
          && v275 == *(_QWORD *)&v122[2 * (1 - v123)]
          && (_DWORD)v276 == v122[6]
          && v277 == *(_QWORD *)&v122[2 * (2 - v123)]
          && v278 == *(_QWORD *)&v122[2 * (3 - v123)]
          && (_DWORD)v279 == v122[7]
          && v280 == *(_QWORD *)&v122[2 * (4 - v123)] )
        {
          v213 = v57[151];
          v214 = *((unsigned int *)v57 + 306);
          if ( v121 != (unsigned int **)(v213 + 8 * v214) )
          {
            result = *v121;
            if ( result )
              return result;
          }
        }
        else
        {
          v120 = (v241 - 1) & (v246 + v120);
          v121 = (unsigned int **)(v119 + 8LL * v120);
          v122 = *v121;
          if ( *v121 != (unsigned int *)-8LL )
          {
            ++v246;
            continue;
          }
LABEL_362:
          v213 = v57[151];
          LODWORD(v214) = *((_DWORD *)v57 + 306);
        }
        break;
      }
      v243 = v213;
      v252 = v214;
      if ( (_DWORD)v214 )
      {
        v215 = v273[2];
        v274 = *(unsigned int ***)&v273[-2 * v215];
        v275 = *(_QWORD *)&v273[2 * (1 - v215)];
        LODWORD(v276) = v273[6];
        v277 = *(_QWORD *)&v273[2 * (2 - v215)];
        v278 = *(_QWORD *)&v273[2 * (3 - v215)];
        LODWORD(v279) = v273[7];
        v280 = *(_QWORD *)&v273[2 * (4 - v215)];
        v216 = (v214 - 1) & sub_15B52D0((__int64 *)&v274, &v275, (int *)&v276, &v277, &v278, (int *)&v279, &v280);
        v16 = (unsigned int **)(v243 + 8LL * v216);
        result = v273;
        v217 = *v16;
        if ( *v16 == v273 )
          return result;
        v218 = 1;
        v219 = 0;
        while ( v217 != (unsigned int *)-8LL )
        {
          if ( v217 == (unsigned int *)-16LL && !v219 )
            v219 = v16;
          v216 = (v252 - 1) & (v218 + v216);
          v16 = (unsigned int **)(v243 + 8LL * v216);
          v217 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v218;
        }
        v230 = *((_DWORD *)v57 + 304);
        v60 = *((_DWORD *)v57 + 306);
        v59 = (__int64)(v57 + 150);
        if ( v219 )
          v16 = v219;
        ++v57[150];
        v61 = v230 + 1;
        if ( 4 * v61 < 3 * v60 )
        {
          if ( v60 - (v61 + *((_DWORD *)v57 + 305)) > v60 >> 3 )
          {
LABEL_62:
            *((_DWORD *)v57 + 304) = v61;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v57 + 305);
            goto LABEL_64;
          }
LABEL_61:
          sub_15C5700(v59, v60);
          sub_15B93D0(v59, &v273, &v274);
          v16 = v274;
          v61 = *((_DWORD *)v57 + 304) + 1;
          goto LABEL_62;
        }
      }
      else
      {
LABEL_59:
        ++v57[150];
        v59 = (__int64)(v57 + 150);
        v60 = 0;
      }
      v60 *= 2;
      goto LABEL_61;
    case 0x1C:
      v46 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v46 = (_QWORD *)*v46;
      v47 = (_QWORD *)*v46;
      v48 = *(unsigned __int16 *)(a1 + 2);
      v273 = (unsigned int *)a1;
      LODWORD(v274) = v48;
      v49 = *(unsigned int *)(a1 + 8);
      v275 = *(_QWORD *)(a1 - 8 * v49);
      v276 = *(unsigned int **)(a1 + 8 * (1 - v49));
      v277 = *(_QWORD *)(a1 + 8 * (3 - v49));
      LODWORD(v278) = *(_DWORD *)(a1 + 24);
      v279 = *(_QWORD *)(a1 + 8 * (2 - v49));
      if ( !*((_DWORD *)v47 + 314) )
        goto LABEL_45;
      v247 = *((_DWORD *)v47 + 314);
      v255 = v47[155];
      v124 = (v247 - 1) & sub_15B6E40((int *)&v274, &v275, (__int64 *)&v276, &v277, (int *)&v278, &v279);
      v125 = (unsigned int **)(v255 + 8LL * v124);
      v126 = *v125;
      if ( *v125 == (unsigned int *)-8LL )
        goto LABEL_363;
      v127 = 1;
      while ( 2 )
      {
        if ( v126 != (unsigned int *)-16LL
          && (_DWORD)v274 == *((unsigned __int16 *)v126 + 1)
          && (v239 = v126[2], v275 == *(_QWORD *)&v126[-2 * v239])
          && v276 == *(unsigned int **)&v126[2 * (1 - v239)]
          && v277 == *(_QWORD *)&v126[2 * (3 - v239)]
          && (_DWORD)v278 == v126[6]
          && v279 == *(_QWORD *)&v126[2 * (2 - v239)] )
        {
          v191 = v47[155];
          v192 = *((unsigned int *)v47 + 314);
          if ( v125 != (unsigned int **)(v191 + 8 * v192) )
          {
            result = *v125;
            if ( result )
              return result;
          }
        }
        else
        {
          v124 = (v247 - 1) & (v127 + v124);
          v125 = (unsigned int **)(v255 + 8LL * v124);
          v126 = *v125;
          if ( *v125 != (unsigned int *)-8LL )
          {
            ++v127;
            continue;
          }
LABEL_363:
          v191 = v47[155];
          LODWORD(v192) = *((_DWORD *)v47 + 314);
        }
        break;
      }
      v251 = v191;
      v259 = v192;
      if ( (_DWORD)v192 )
      {
        LODWORD(v274) = *((unsigned __int16 *)v273 + 1);
        v193 = v273[2];
        v275 = *(_QWORD *)&v273[-2 * v193];
        v276 = *(unsigned int **)&v273[2 * (1 - v193)];
        v277 = *(_QWORD *)&v273[2 * (3 - v193)];
        LODWORD(v278) = v273[6];
        v279 = *(_QWORD *)&v273[2 * (2 - v193)];
        v194 = (v192 - 1) & sub_15B6E40((int *)&v274, &v275, (__int64 *)&v276, &v277, (int *)&v278, &v279);
        v16 = (unsigned int **)(v251 + 8LL * v194);
        result = v273;
        v195 = *v16;
        if ( v273 == *v16 )
          return result;
        v196 = 1;
        v197 = 0;
        while ( v195 != (unsigned int *)-8LL )
        {
          if ( v195 == (unsigned int *)-16LL && !v197 )
            v197 = v16;
          v194 = (v259 - 1) & (v196 + v194);
          v16 = (unsigned int **)(v251 + 8LL * v194);
          v195 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v196;
        }
        v228 = *((_DWORD *)v47 + 312);
        v51 = *((_DWORD *)v47 + 314);
        v50 = (__int64)(v47 + 154);
        if ( v197 )
          v16 = v197;
        ++v47[154];
        v52 = v228 + 1;
        if ( 4 * v52 < 3 * v51 )
        {
          if ( v51 - (v52 + *((_DWORD *)v47 + 313)) > v51 >> 3 )
          {
LABEL_48:
            *((_DWORD *)v47 + 312) = v52;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v47 + 313);
            goto LABEL_64;
          }
LABEL_47:
          sub_15C5E40(v50, v51);
          sub_15B9520(v50, (__int64 *)&v273, &v274);
          v16 = v274;
          v52 = *((_DWORD *)v47 + 312) + 1;
          goto LABEL_48;
        }
      }
      else
      {
LABEL_45:
        ++v47[154];
        v50 = (__int64)(v47 + 154);
        v51 = 0;
      }
      v51 *= 2;
      goto LABEL_47;
    case 0x1D:
      v39 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      v40 = *(unsigned int *)(a1 + 8);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v39 = (_QWORD *)*v39;
      v41 = (_QWORD *)*v39;
      v42 = *(unsigned __int16 *)(a1 + 2);
      v273 = (unsigned int *)a1;
      LODWORD(v274) = v42;
      HIDWORD(v274) = *(_DWORD *)(a1 + 24);
      v275 = *(_QWORD *)(a1 - 8 * v40);
      v276 = *(unsigned int **)(a1 + 8 * (1 - v40));
      if ( !*((_DWORD *)v41 + 322) )
        goto LABEL_37;
      v263 = *((_DWORD *)v41 + 322);
      v268 = v41[159];
      v134 = (v263 - 1) & sub_15B6820((int *)&v274, (int *)&v274 + 1, &v275, (__int64 *)&v276);
      v135 = (unsigned int **)(v268 + 8LL * v134);
      v136 = *v135;
      if ( *v135 == (unsigned int *)-8LL )
        goto LABEL_361;
      v137 = 1;
      while ( 2 )
      {
        if ( v136 != (unsigned int *)-16LL
          && v274 == (unsigned int **)__PAIR64__(v136[6], *((unsigned __int16 *)v136 + 1))
          && (v249 = v136[2], v275 == *(_QWORD *)&v136[-2 * v249])
          && v276 == *(unsigned int **)&v136[2 * (1 - v249)] )
        {
          v158 = v41[159];
          v159 = *((unsigned int *)v41 + 322);
          if ( v135 != (unsigned int **)(v158 + 8 * v159) )
          {
            result = *v135;
            if ( *v135 )
              return result;
          }
        }
        else
        {
          v134 = (v263 - 1) & (v137 + v134);
          v135 = (unsigned int **)(v268 + 8LL * v134);
          v136 = *v135;
          if ( *v135 != (unsigned int *)-8LL )
          {
            ++v137;
            continue;
          }
LABEL_361:
          v158 = v41[159];
          LODWORD(v159) = *((_DWORD *)v41 + 322);
        }
        break;
      }
      v266 = v158;
      v271 = v159;
      if ( (_DWORD)v159 )
      {
        LODWORD(v274) = *((unsigned __int16 *)v273 + 1);
        HIDWORD(v274) = v273[6];
        v160 = v273[2];
        v275 = *(_QWORD *)&v273[-2 * v160];
        v276 = *(unsigned int **)&v273[2 * (1 - v160)];
        v161 = (v159 - 1) & sub_15B6820((int *)&v274, (int *)&v274 + 1, &v275, (__int64 *)&v276);
        v16 = (unsigned int **)(v266 + 8LL * v161);
        result = v273;
        v162 = *v16;
        if ( v273 == *v16 )
          return result;
        v163 = 1;
        v164 = 0;
        while ( v162 != (unsigned int *)-8LL )
        {
          if ( v162 == (unsigned int *)-16LL && !v164 )
            v164 = v16;
          v161 = (v271 - 1) & (v163 + v161);
          v16 = (unsigned int **)(v266 + 8LL * v161);
          v162 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v163;
        }
        v237 = *((_DWORD *)v41 + 320);
        v44 = *((_DWORD *)v41 + 322);
        v43 = (__int64)(v41 + 158);
        if ( v164 )
          v16 = v164;
        ++v41[158];
        v45 = v237 + 1;
        if ( 4 * v45 < 3 * v44 )
        {
          if ( v44 - (v45 + *((_DWORD *)v41 + 321)) > v44 >> 3 )
          {
LABEL_40:
            *((_DWORD *)v41 + 320) = v45;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v41 + 321);
            goto LABEL_64;
          }
LABEL_39:
          sub_15C64E0(v43, v44);
          sub_15B9650(v43, (__int64 *)&v273, &v274);
          v16 = v274;
          v45 = *((_DWORD *)v41 + 320) + 1;
          goto LABEL_40;
        }
      }
      else
      {
LABEL_37:
        ++v41[158];
        v43 = (__int64)(v41 + 158);
        v44 = 0;
      }
      v44 *= 2;
      goto LABEL_39;
    case 0x1E:
      v32 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      v33 = *(unsigned int *)(a1 + 8);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v32 = (_QWORD *)*v32;
      v34 = (_QWORD *)*v32;
      v35 = *(unsigned __int16 *)(a1 + 2);
      v273 = (unsigned int *)a1;
      LODWORD(v274) = v35;
      HIDWORD(v274) = *(_DWORD *)(a1 + 24);
      v275 = *(_QWORD *)(a1 - 8 * v33);
      v276 = *(unsigned int **)(a1 + 8 * (1 - v33));
      if ( !*((_DWORD *)v34 + 330) )
        goto LABEL_29;
      v264 = *((_DWORD *)v34 + 330);
      v269 = v34[163];
      v142 = (v264 - 1) & sub_15B3100((int *)&v274, (int *)&v274 + 1, &v275, (__int64 *)&v276);
      v143 = (unsigned int **)(v269 + 8LL * v142);
      v144 = *v143;
      if ( *v143 == (unsigned int *)-8LL )
        goto LABEL_365;
      v145 = 1;
      while ( 2 )
      {
        if ( v144 != (unsigned int *)-16LL
          && v274 == (unsigned int **)__PAIR64__(v144[6], *((unsigned __int16 *)v144 + 1))
          && (v248 = v144[2], v275 == *(_QWORD *)&v144[-2 * v248])
          && v276 == *(unsigned int **)&v144[2 * (1 - v248)] )
        {
          v151 = v34[163];
          v152 = *((unsigned int *)v34 + 330);
          if ( v143 != (unsigned int **)(v151 + 8 * v152) )
          {
            result = *v143;
            if ( *v143 )
              return result;
          }
        }
        else
        {
          v142 = (v264 - 1) & (v145 + v142);
          v143 = (unsigned int **)(v269 + 8LL * v142);
          v144 = *v143;
          if ( *v143 != (unsigned int *)-8LL )
          {
            ++v145;
            continue;
          }
LABEL_365:
          v151 = v34[163];
          LODWORD(v152) = *((_DWORD *)v34 + 330);
        }
        break;
      }
      v265 = v151;
      v270 = v152;
      if ( (_DWORD)v152 )
      {
        LODWORD(v274) = *((unsigned __int16 *)v273 + 1);
        HIDWORD(v274) = v273[6];
        v153 = v273[2];
        v275 = *(_QWORD *)&v273[-2 * v153];
        v276 = *(unsigned int **)&v273[2 * (1 - v153)];
        v154 = (v152 - 1) & sub_15B3100((int *)&v274, (int *)&v274 + 1, &v275, (__int64 *)&v276);
        v16 = (unsigned int **)(v265 + 8LL * v154);
        result = v273;
        v155 = *v16;
        if ( v273 == *v16 )
          return result;
        v156 = 1;
        v157 = 0;
        while ( v155 != (unsigned int *)-8LL )
        {
          if ( v155 == (unsigned int *)-16LL && !v157 )
            v157 = v16;
          v154 = (v270 - 1) & (v156 + v154);
          v16 = (unsigned int **)(v265 + 8LL * v154);
          v155 = *v16;
          if ( v273 == *v16 )
            return result;
          ++v156;
        }
        v233 = *((_DWORD *)v34 + 328);
        v37 = *((_DWORD *)v34 + 330);
        v36 = (__int64)(v34 + 162);
        if ( v157 )
          v16 = v157;
        ++v34[162];
        v38 = v233 + 1;
        if ( 4 * v38 < 3 * v37 )
        {
          if ( v37 - (v38 + *((_DWORD *)v34 + 329)) > v37 >> 3 )
          {
LABEL_32:
            *((_DWORD *)v34 + 328) = v38;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v34 + 329);
            goto LABEL_64;
          }
LABEL_31:
          sub_15C6AB0(v36, v37);
          sub_15B9760(v36, (__int64 *)&v273, &v274);
          v16 = v274;
          v38 = *((_DWORD *)v34 + 328) + 1;
          goto LABEL_32;
        }
      }
      else
      {
LABEL_29:
        ++v34[162];
        v36 = (__int64)(v34 + 162);
        v37 = 0;
      }
      v37 *= 2;
      goto LABEL_31;
    case 0x1F:
      v26 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v26 = (_QWORD *)*v26;
      v27 = (_QWORD *)*v26;
      v28 = *(unsigned int *)(a1 + 8);
      v273 = (unsigned int *)a1;
      v274 = *(unsigned int ***)(a1 - 8 * v28);
      v275 = *(_QWORD *)(a1 + 8 * (1 - v28));
      v276 = *(unsigned int **)(a1 + 8 * (2 - v28));
      v277 = *(_QWORD *)(a1 + 8 * (3 - v28));
      LODWORD(v278) = *(_DWORD *)(a1 + 24);
      if ( !*((_DWORD *)v27 + 338) )
        goto LABEL_21;
      v254 = *((_DWORD *)v27 + 338);
      v261 = v27[167];
      v112 = sub_15B3B60((__int64 *)&v274, &v275, (__int64 *)&v276, &v277, (int *)&v278);
      v113 = v261;
      v114 = (v254 - 1) & v112;
      v115 = (unsigned int **)(v261 + 8LL * v114);
      v116 = *v115;
      if ( *v115 == (unsigned int *)-8LL )
        goto LABEL_368;
      v262 = 1;
      while ( 2 )
      {
        if ( v116 != (unsigned int *)-16LL
          && (v117 = v116[2], v274 == *(unsigned int ***)&v116[-2 * v117])
          && v275 == *(_QWORD *)&v116[2 * (1 - v117)]
          && v276 == *(unsigned int **)&v116[2 * (2 - v117)]
          && v277 == *(_QWORD *)&v116[2 * (3 - v117)]
          && (_DWORD)v278 == v116[6] )
        {
          v206 = v27[167];
          v207 = *((unsigned int *)v27 + 338);
          if ( v115 != (unsigned int **)(v206 + 8 * v207) )
          {
            result = *v115;
            if ( *v115 )
              return result;
          }
        }
        else
        {
          v114 = (v254 - 1) & (v262 + v114);
          v115 = (unsigned int **)(v113 + 8LL * v114);
          v116 = *v115;
          if ( *v115 != (unsigned int *)-8LL )
          {
            ++v262;
            continue;
          }
LABEL_368:
          v206 = v27[167];
          LODWORD(v207) = *((_DWORD *)v27 + 338);
        }
        break;
      }
      v260 = v206;
      v267 = v207;
      if ( (_DWORD)v207 )
      {
        v208 = v273[2];
        v274 = *(unsigned int ***)&v273[-2 * v208];
        v275 = *(_QWORD *)&v273[2 * (1 - v208)];
        v276 = *(unsigned int **)&v273[2 * (2 - v208)];
        v277 = *(_QWORD *)&v273[2 * (3 - v208)];
        LODWORD(v278) = v273[6];
        v209 = (v207 - 1) & sub_15B3B60((__int64 *)&v274, &v275, (__int64 *)&v276, &v277, (int *)&v278);
        v16 = (unsigned int **)(v260 + 8LL * v209);
        result = v273;
        v210 = *v16;
        if ( v273 == *v16 )
          return result;
        v211 = 1;
        v212 = 0;
        while ( v210 != (unsigned int *)-8LL )
        {
          if ( !v212 && v210 == (unsigned int *)-16LL )
            v212 = v16;
          v209 = (v267 - 1) & (v211 + v209);
          v16 = (unsigned int **)(v260 + 8LL * v209);
          v210 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v211;
        }
        v231 = *((_DWORD *)v27 + 336);
        v30 = *((_DWORD *)v27 + 338);
        v29 = (__int64)(v27 + 166);
        if ( v212 )
          v16 = v212;
        ++v27[166];
        v31 = v231 + 1;
        if ( 4 * v31 < 3 * v30 )
        {
          if ( v30 - (v31 + *((_DWORD *)v27 + 337)) > v30 >> 3 )
          {
LABEL_24:
            *((_DWORD *)v27 + 336) = v31;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v27 + 337);
            goto LABEL_64;
          }
LABEL_23:
          sub_15C1420(v29, v30);
          sub_15B89F0(v29, (__int64 *)&v273, &v274);
          v16 = v274;
          v31 = *((_DWORD *)v27 + 336) + 1;
          goto LABEL_24;
        }
      }
      else
      {
LABEL_21:
        ++v27[166];
        v29 = (__int64)(v27 + 166);
        v30 = 0;
      }
      v30 *= 2;
      goto LABEL_23;
    case 0x20:
      v18 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      v19 = *(unsigned int *)(a1 + 8);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v18 = (_QWORD *)*v18;
      v20 = (_QWORD *)*v18;
      v21 = *(unsigned __int16 *)(a1 + 2);
      v273 = (unsigned int *)a1;
      LODWORD(v274) = v21;
      v275 = *(_QWORD *)(a1 + 8 * (2 - v19));
      v276 = *(unsigned int **)(a1 + 8 * (3 - v19));
      v277 = *(_QWORD *)(a1 + 8 * (4 - v19));
      v278 = *(_QWORD *)(a1 + 32);
      v279 = *(_QWORD *)(a1 + 48);
      v22 = v20[171];
      if ( !*((_DWORD *)v20 + 346) )
        goto LABEL_13;
      v257 = *((_DWORD *)v20 + 346);
      v138 = (v257 - 1) & sub_15B4F20((int *)&v274, &v275, &v278, (int *)&v279, (int *)&v279 + 1);
      v139 = (unsigned int **)(v22 + 8LL * v138);
      v140 = *v139;
      if ( *v139 == (unsigned int *)-8LL )
        goto LABEL_364;
      v141 = 1;
      while ( 2 )
      {
        if ( v140 != (unsigned int *)-16LL
          && (_DWORD)v274 == *((unsigned __int16 *)v140 + 1)
          && v275 == *(_QWORD *)&v140[2 * (2LL - v140[2])]
          && v278 == *((_QWORD *)v140 + 4)
          && v279 == *((_QWORD *)v140 + 6) )
        {
          v165 = v20[171];
          v166 = *((unsigned int *)v20 + 346);
          if ( v139 != (unsigned int **)(v165 + 8 * v166) )
          {
            result = *v139;
            if ( *v139 )
              return result;
          }
        }
        else
        {
          v138 = (v257 - 1) & (v141 + v138);
          v139 = (unsigned int **)(v22 + 8LL * v138);
          v140 = *v139;
          if ( *v139 != (unsigned int *)-8LL )
          {
            ++v141;
            continue;
          }
LABEL_364:
          v165 = v20[171];
          LODWORD(v166) = *((_DWORD *)v20 + 346);
        }
        break;
      }
      v272 = v165;
      if ( (_DWORD)v166 )
      {
        v167 = v166 - 1;
        LODWORD(v274) = *((unsigned __int16 *)v273 + 1);
        v168 = v273[2];
        v275 = *(_QWORD *)&v273[2 * (2 - v168)];
        v276 = *(unsigned int **)&v273[2 * (3 - v168)];
        v277 = *(_QWORD *)&v273[2 * (4 - v168)];
        v278 = *((_QWORD *)v273 + 4);
        v279 = *((_QWORD *)v273 + 6);
        v169 = v167 & sub_15B4F20((int *)&v274, &v275, &v278, (int *)&v279, (int *)&v279 + 1);
        v16 = (unsigned int **)(v272 + 8LL * v169);
        result = v273;
        v170 = *v16;
        if ( v273 == *v16 )
          return result;
        v171 = 1;
        v172 = 0;
        while ( v170 != (unsigned int *)-8LL )
        {
          if ( !v172 && v170 == (unsigned int *)-16LL )
            v172 = v16;
          v169 = v167 & (v171 + v169);
          v16 = (unsigned int **)(v272 + 8LL * v169);
          v170 = *v16;
          if ( *v16 == v273 )
            return result;
          ++v171;
        }
        v236 = *((_DWORD *)v20 + 344);
        v24 = *((_DWORD *)v20 + 346);
        v23 = (__int64)(v20 + 170);
        if ( v172 )
          v16 = v172;
        ++v20[170];
        v25 = v236 + 1;
        if ( 4 * v25 < 3 * v24 )
        {
          if ( v24 - (v25 + *((_DWORD *)v20 + 345)) > v24 >> 3 )
          {
LABEL_16:
            *((_DWORD *)v20 + 344) = v25;
            if ( *v16 != (unsigned int *)-8LL )
              --*((_DWORD *)v20 + 345);
            goto LABEL_64;
          }
LABEL_15:
          sub_15BCA70(v23, v24);
          sub_15B79C0(v23, (__int64 *)&v273, &v274);
          v16 = v274;
          v25 = *((_DWORD *)v20 + 344) + 1;
          goto LABEL_16;
        }
      }
      else
      {
LABEL_13:
        ++v20[170];
        v23 = (__int64)(v20 + 170);
        v24 = 0;
      }
      v24 *= 2;
      goto LABEL_15;
    case 0x21:
      v9 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v9 = (_QWORD *)*v9;
      v10 = (_QWORD *)*v9;
      v11 = *(unsigned __int16 *)(a1 + 2);
      v273 = (unsigned int *)a1;
      LODWORD(v274) = v11;
      v12 = *(unsigned int *)(a1 + 8);
      v275 = *(_QWORD *)(a1 + 8 * (2 - v12));
      v276 = *(unsigned int **)(a1 - 8 * v12);
      LODWORD(v277) = *(_DWORD *)(a1 + 24);
      v278 = *(_QWORD *)(a1 + 8 * (1 - v12));
      v279 = *(_QWORD *)(a1 + 8 * (3 - v12));
      v280 = *(_QWORD *)(a1 + 32);
      v281 = *(_QWORD *)(a1 + 40);
      v282 = *(_DWORD *)(a1 + 48);
      v283 = *(_DWORD *)(a1 + 28);
      v284[0] = *(_QWORD *)(a1 + 8 * (4 - v12));
      v13 = *((_DWORD *)v10 + 354);
      v253 = v10[175];
      if ( !v13 )
        goto LABEL_5;
      v128 = v13 - 1;
      v129 = sub_15B5D10(&v275, (__int64 *)&v276, (int *)&v277, &v279, &v278, v284);
      v130 = v253;
      v131 = v128 & v129;
      v132 = (unsigned int **)(v253 + 8LL * (v128 & v129));
      v133 = *v132;
      if ( *v132 == (unsigned int *)-8LL )
        goto LABEL_369;
      v256 = 1;
      while ( 2 )
      {
        if ( v133 == (unsigned int *)-16LL )
          goto LABEL_178;
        if ( (_DWORD)v274 != *((unsigned __int16 *)v133 + 1) )
          goto LABEL_178;
        v180 = v133[2];
        if ( v275 != *(_QWORD *)&v133[2 * (2 - v180)] )
          goto LABEL_178;
        v181 = v133;
        if ( *(_BYTE *)v133 != 15 )
          v181 = *(unsigned int **)&v133[-2 * v180];
        if ( v276 == v181
          && (_DWORD)v277 == v133[6]
          && v278 == *(_QWORD *)&v133[2 * (1 - v180)]
          && v279 == *(_QWORD *)&v133[2 * (3 - v180)]
          && v280 == *((_QWORD *)v133 + 4)
          && v282 == v133[12]
          && v281 == *((_QWORD *)v133 + 5)
          && v283 == v133[7]
          && v284[0] == *(_QWORD *)&v133[2 * (4 - v180)] )
        {
          v182 = v10[175];
          v183 = *((unsigned int *)v10 + 354);
          if ( v132 != (unsigned int **)(v182 + 8 * v183) )
          {
            result = *v132;
            if ( result )
              return result;
          }
        }
        else
        {
LABEL_178:
          v131 = v128 & (v256 + v131);
          v132 = (unsigned int **)(v130 + 8LL * v131);
          v133 = *v132;
          if ( *v132 != (unsigned int *)-8LL )
          {
            ++v256;
            continue;
          }
LABEL_369:
          v182 = v10[175];
          LODWORD(v183) = *((_DWORD *)v10 + 354);
        }
        break;
      }
      if ( !(_DWORD)v183 )
      {
LABEL_5:
        ++v10[174];
        v14 = (__int64)(v10 + 174);
        v15 = 0;
LABEL_6:
        v15 *= 2;
LABEL_7:
        sub_15BE350(v14, v15);
        sub_15B7F60(v14, (__int64 *)&v273, &v274);
        v16 = v274;
        v17 = *((_DWORD *)v10 + 352) + 1;
        goto LABEL_8;
      }
      LODWORD(v274) = *((unsigned __int16 *)v273 + 1);
      v184 = v273[2];
      v275 = *(_QWORD *)&v273[2 * (2 - v184)];
      v185 = v273;
      if ( *(_BYTE *)v273 != 15 )
        v185 = *(unsigned int **)&v273[-2 * v273[2]];
      v276 = v185;
      v258 = v182;
      v186 = v183 - 1;
      LODWORD(v277) = v273[6];
      v278 = *(_QWORD *)&v273[2 * (1 - v184)];
      v279 = *(_QWORD *)&v273[2 * (3 - v184)];
      v280 = *((_QWORD *)v273 + 4);
      v281 = *((_QWORD *)v273 + 5);
      v282 = v273[12];
      v283 = v273[7];
      v284[0] = *(_QWORD *)&v273[2 * (4 - v184)];
      v187 = v186 & sub_15B5D10(&v275, (__int64 *)&v276, (int *)&v277, &v279, &v278, v284);
      v16 = (unsigned int **)(v258 + 8LL * v187);
      result = v273;
      v188 = *v16;
      if ( *v16 == v273 )
        return result;
      v189 = 1;
      v190 = 0;
      while ( v188 != (unsigned int *)-8LL )
      {
        if ( !v190 && v188 == (unsigned int *)-16LL )
          v190 = v16;
        v187 = v186 & (v189 + v187);
        v16 = (unsigned int **)(v258 + 8LL * v187);
        v188 = *v16;
        if ( *v16 == v273 )
          return result;
        ++v189;
      }
      v229 = *((_DWORD *)v10 + 352);
      v15 = *((_DWORD *)v10 + 354);
      v14 = (__int64)(v10 + 174);
      if ( v190 )
        v16 = v190;
      ++v10[174];
      v17 = v229 + 1;
      if ( 4 * v17 >= 3 * v15 )
        goto LABEL_6;
      if ( v15 - (v17 + *((_DWORD *)v10 + 353)) <= v15 >> 3 )
        goto LABEL_7;
LABEL_8:
      *((_DWORD *)v10 + 352) = v17;
      if ( *v16 != (unsigned int *)-8LL )
        --*((_DWORD *)v10 + 353);
LABEL_64:
      *v16 = v273;
      return v273;
  }
}
