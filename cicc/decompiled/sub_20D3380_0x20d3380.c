// Function: sub_20D3380
// Address: 0x20d3380
//
__int64 __fastcall sub_20D3380(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 (*v13)(); // rax
  __int64 v14; // rax
  __int64 (*v15)(); // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 i; // r12
  unsigned int v22; // r15d
  __int64 v24; // r15
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rdi
  unsigned __int8 *v30; // r15
  _DWORD *v31; // r12
  __int64 v32; // r14
  char v33; // al
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rax
  unsigned __int64 v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  double v59; // xmm4_8
  double v60; // xmm5_8
  __int64 v61; // rax
  unsigned __int64 v62; // rbx
  __int64 v63; // rdi
  __int64 (*v64)(); // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 *v67; // r12
  __int64 v68; // rax
  unsigned __int8 *v69; // rsi
  __int64 ***v70; // r13
  __int64 **v71; // rax
  __int64 **v72; // rdx
  _QWORD *v73; // r12
  unsigned __int64 *v74; // rbx
  __int64 v75; // rax
  unsigned __int64 v76; // rcx
  __int64 v77; // rsi
  unsigned __int8 *v78; // rsi
  double v79; // xmm4_8
  double v80; // xmm5_8
  __int16 v81; // cx
  __int16 v82; // ax
  __int64 **v83; // rdx
  _QWORD *v84; // r13
  __int64 v85; // rdi
  __int64 v86; // r13
  __int64 (*v87)(); // rax
  __int64 v88; // rax
  unsigned __int8 *v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 **v92; // r14
  __int64 v93; // r12
  __int64 *v94; // r9
  __int64 v95; // rax
  __int64 **v96; // rax
  __int64 v97; // r9
  _QWORD *v98; // rax
  __int64 v99; // r14
  __int64 *v100; // rbx
  __int64 v101; // rax
  __int64 v102; // rcx
  __int64 v103; // rsi
  unsigned __int8 *v104; // rsi
  __int16 v105; // dx
  int v106; // eax
  __int64 v107; // rdi
  __int64 (*v108)(); // rax
  __int64 v109; // rax
  unsigned __int8 *v110; // rsi
  unsigned int v111; // r8d
  __int64 *v112; // rbx
  __int64 v113; // r13
  __int64 *v114; // rax
  __int64 *v115; // r12
  unsigned __int64 *v116; // rbx
  __int64 v117; // rax
  unsigned __int64 v118; // rcx
  __int64 v119; // rsi
  unsigned __int8 *v120; // rsi
  double v121; // xmm4_8
  double v122; // xmm5_8
  char v123; // bl
  unsigned int v124; // eax
  unsigned int v125; // r12d
  __int64 v126; // rdx
  unsigned int v127; // eax
  unsigned int v128; // ebx
  __int64 v129; // rdi
  __int64 (*v130)(); // rax
  __int64 v131; // rsi
  __int16 v132; // ax
  __int64 v133; // rax
  unsigned __int8 *v134; // rsi
  __int64 v135; // rdi
  __int64 v136; // rax
  _QWORD *(__fastcall *v137)(__int64, __int64 *, __int64, int); // r8
  _QWORD *v138; // rax
  __int64 v139; // rsi
  __int64 v140; // rax
  __int64 v141; // rsi
  __int64 v142; // rdx
  unsigned __int8 *v143; // rsi
  _QWORD *(__fastcall *v144)(__int64, __int64 *, __int64, int); // rax
  _QWORD *v145; // rax
  __int64 v146; // rbx
  __int64 *v147; // r12
  __int64 v148; // rax
  __int64 v149; // rcx
  __int64 v150; // rsi
  unsigned __int8 *v151; // rsi
  unsigned __int64 *v152; // rbx
  __int64 **v153; // rax
  unsigned __int64 v154; // rcx
  __int64 v155; // rsi
  unsigned __int8 *v156; // rsi
  unsigned __int64 *v157; // rbx
  __int64 v158; // rax
  unsigned __int64 v159; // rcx
  __int64 v160; // rsi
  __int64 v161; // rdx
  unsigned __int8 *v162; // rsi
  __int64 v163; // rsi
  __int64 *v164; // rbx
  __int64 v165; // rax
  __int64 v166; // rcx
  __int64 v167; // rsi
  __int64 v168; // rdx
  unsigned __int8 *v169; // rsi
  __int64 v170; // r9
  __int64 *v171; // rbx
  __int64 v172; // rcx
  __int64 v173; // rax
  __int64 v174; // rsi
  __int64 v175; // r14
  unsigned __int8 *v176; // rsi
  unsigned int v177; // eax
  __int16 v178; // ax
  unsigned int v179; // ebx
  bool v180; // al
  _QWORD *v181; // r13
  __int64 v182; // rbx
  unsigned int v183; // r12d
  __int64 v184; // rax
  unsigned int v185; // r12d
  __int64 v186; // rdi
  double v187; // xmm4_8
  double v188; // xmm5_8
  __int64 v189; // rsi
  __int64 v190; // rax
  __int64 v191; // rdi
  __int64 (*v192)(); // rax
  double v193; // xmm4_8
  double v194; // xmm5_8
  __int64 v195; // rax
  unsigned __int64 v196; // r13
  __int64 v197; // rax
  __int64 v198; // rax
  __int64 v199; // rax
  unsigned __int8 *v200; // rsi
  __int64 ***v201; // r12
  __int64 **v202; // rax
  __int64 **v203; // rax
  __int64 v204; // rsi
  __int64 v205; // rsi
  unsigned int v206; // eax
  int v207; // ebx
  _QWORD *v208; // rax
  __int64 *v209; // r12
  __int64 v210; // rcx
  __int64 v211; // rax
  __int64 v212; // rsi
  unsigned __int8 *v213; // rsi
  __int16 v214; // ax
  __int64 v215; // rax
  __int64 v216; // r12
  __int64 v217; // rax
  __int64 ***v218; // rax
  __int64 **v219; // rdx
  __int64 v220; // rax
  __int64 v221; // rax
  __int64 v222; // rax
  double v223; // xmm4_8
  double v224; // xmm5_8
  __int64 v225; // rax
  __int64 *v226; // rbx
  __int64 v227; // rax
  __int64 v228; // rcx
  __int64 v229; // rsi
  __int64 v230; // rdx
  unsigned __int8 *v231; // rsi
  __int64 v232; // rax
  __int64 *v233; // rbx
  __int64 v234; // rax
  unsigned __int64 v235; // rcx
  __int64 v236; // rsi
  __int64 v237; // rdx
  unsigned __int8 *v238; // rsi
  __int64 v239; // rax
  unsigned __int64 *v240; // rbx
  unsigned __int64 v241; // rcx
  __int64 v242; // rax
  __int64 v243; // rsi
  __int64 v244; // rdx
  unsigned __int8 *v245; // rsi
  __int64 v246; // rax
  unsigned __int64 *v247; // rbx
  __int64 **v248; // rax
  unsigned __int64 v249; // rcx
  __int64 v250; // rsi
  __int64 v251; // rdx
  unsigned __int8 *v252; // rsi
  __int64 v253; // rax
  __int64 v254; // rax
  unsigned __int64 v255; // rbx
  __int64 v256; // rax
  unsigned __int64 v257; // rbx
  __int64 v258; // rax
  __int64 v259; // r12
  int *v260; // r13
  __int64 v261; // rax
  __int64 v262; // rdi
  __int64 v263; // rsi
  __int64 v264; // rbx
  __int64 v265; // rax
  __int64 v266; // rdi
  double v267; // xmm4_8
  double v268; // xmm5_8
  __int64 v269; // rsi
  __int64 v270; // rax
  __int64 v271; // rax
  __int64 v272; // rax
  unsigned __int64 v273; // rbx
  __int64 v274; // rdi
  __int64 (*v275)(); // rax
  unsigned int v276; // eax
  __int64 v277; // rax
  __int64 v278; // rax
  unsigned __int64 v279; // rbx
  __int64 v280; // rax
  unsigned __int64 v281; // rbx
  __int64 v282; // rax
  double v283; // xmm4_8
  double v284; // xmm5_8
  __int64 (*v285)(); // rax
  __int64 v286; // rdi
  __int64 (*v287)(); // rax
  __int64 v288; // rdi
  __int64 (*v289)(); // rax
  unsigned int v290; // eax
  unsigned int v291; // [rsp+8h] [rbp-148h]
  __int64 *v292; // [rsp+10h] [rbp-140h]
  __int64 v293; // [rsp+10h] [rbp-140h]
  __int64 v294; // [rsp+20h] [rbp-130h]
  __int64 v295; // [rsp+28h] [rbp-128h]
  __int64 v296; // [rsp+28h] [rbp-128h]
  _QWORD *v297; // [rsp+28h] [rbp-128h]
  __int64 v298; // [rsp+28h] [rbp-128h]
  __int64 *v299; // [rsp+28h] [rbp-128h]
  __int64 v300; // [rsp+28h] [rbp-128h]
  unsigned __int64 v301; // [rsp+28h] [rbp-128h]
  __int64 v302; // [rsp+28h] [rbp-128h]
  char v303; // [rsp+30h] [rbp-120h]
  unsigned __int8 v304; // [rsp+37h] [rbp-119h]
  int v305; // [rsp+38h] [rbp-118h]
  __int64 v306; // [rsp+38h] [rbp-118h]
  _QWORD *v307; // [rsp+38h] [rbp-118h]
  __int64 v308; // [rsp+38h] [rbp-118h]
  __int64 v310; // [rsp+40h] [rbp-110h]
  __int64 v311; // [rsp+40h] [rbp-110h]
  __int64 v312; // [rsp+40h] [rbp-110h]
  __int64 v313; // [rsp+40h] [rbp-110h]
  __int64 v314; // [rsp+40h] [rbp-110h]
  __int64 v315; // [rsp+40h] [rbp-110h]
  __int64 *v316; // [rsp+48h] [rbp-108h]
  unsigned __int8 v317; // [rsp+50h] [rbp-100h]
  __int64 v318; // [rsp+50h] [rbp-100h]
  __int64 v319; // [rsp+50h] [rbp-100h]
  __int64 v320; // [rsp+50h] [rbp-100h]
  __int64 v321; // [rsp+50h] [rbp-100h]
  __int64 v322; // [rsp+50h] [rbp-100h]
  __int64 v323; // [rsp+50h] [rbp-100h]
  __int64 v324; // [rsp+50h] [rbp-100h]
  __int64 v325; // [rsp+50h] [rbp-100h]
  __int64 *v326; // [rsp+58h] [rbp-F8h]
  unsigned __int8 *v327; // [rsp+68h] [rbp-E8h] BYREF
  __int64 *v328; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v329; // [rsp+78h] [rbp-D8h]
  _BYTE v330[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v331[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v332; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v333[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v334; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v335; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v336; // [rsp+D8h] [rbp-78h]
  __int64 *v337; // [rsp+E0h] [rbp-70h]
  __int64 v338; // [rsp+E8h] [rbp-68h]
  __int64 v339; // [rsp+F0h] [rbp-60h]
  int v340; // [rsp+F8h] [rbp-58h]
  __int64 v341; // [rsp+100h] [rbp-50h]
  __int64 v342; // [rsp+108h] [rbp-48h]

  v10 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCBA30, 1u);
  if ( !v10 )
    return 0;
  v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_4FCBA30);
  if ( !v11 )
    return 0;
  v12 = *(_QWORD *)(v11 + 208);
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 16LL);
  if ( v13 == sub_16FF750 )
    BUG();
  v14 = ((__int64 (__fastcall *)(__int64, __int64))v13)(v12, a2);
  v304 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 184LL))(v14);
  if ( !v304 )
    return 0;
  v15 = *(__int64 (**)())(*(_QWORD *)v12 + 16LL);
  if ( v15 == sub_16FF750 )
    BUG();
  v16 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v12, a2);
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 56LL);
  v18 = 0;
  if ( v17 != sub_1D12D20 )
    v18 = ((__int64 (__fastcall *)(__int64))v17)(v16);
  v19 = a2 + 72;
  *(_QWORD *)(a1 + 160) = v18;
  v20 = *(_QWORD *)(a2 + 80);
  v328 = (__int64 *)v330;
  v329 = 0x100000000LL;
  if ( a2 + 72 == v20 )
    return 0;
  if ( !v20 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v20 + 24);
    if ( i != v20 + 16 )
      break;
    v20 = *(_QWORD *)(v20 + 8);
    if ( v19 == v20 )
      return 0;
    if ( !v20 )
      BUG();
  }
LABEL_16:
  while ( v19 != v20 )
  {
    v24 = i - 24;
    if ( !i )
      v24 = 0;
    if ( sub_15F32D0(v24) && *(_BYTE *)(v24 + 16) != 57 )
    {
      v27 = (unsigned int)v329;
      if ( (unsigned int)v329 >= HIDWORD(v329) )
      {
        sub_16CD150((__int64)&v328, v330, 0, 8, v25, v26);
        v27 = (unsigned int)v329;
      }
      v328[v27] = v24;
      LODWORD(v329) = v329 + 1;
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v20 + 24) )
    {
      v28 = v20 - 24;
      if ( !v20 )
        v28 = 0;
      if ( i != v28 + 40 )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( v19 == v20 )
        goto LABEL_16;
      if ( !v20 )
        goto LABEL_398;
    }
  }
  v29 = v328;
  v316 = &v328[(unsigned int)v329];
  if ( v316 == v328 )
  {
    v22 = 0;
    goto LABEL_43;
  }
  v326 = v328;
  v317 = 0;
  v30 = (unsigned __int8 *)a1;
  do
  {
    v31 = (_DWORD *)*((_QWORD *)v30 + 20);
    v32 = *v326;
    v33 = *(_BYTE *)(*v326 + 16);
    switch ( v33 )
    {
      case '6':
        v34 = 1;
        v35 = sub_15F2050(*v326);
        v36 = sub_1632FA0(v35);
        v39 = *(_QWORD *)v32;
        v40 = v36;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v39 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v48 = *(_QWORD *)(v39 + 32);
              v39 = *(_QWORD *)(v39 + 24);
              v34 *= v48;
              continue;
            case 1:
              v41 = 16;
              goto LABEL_38;
            case 2:
              v41 = 32;
              goto LABEL_38;
            case 3:
            case 9:
              v41 = 64;
              goto LABEL_38;
            case 4:
              v41 = 80;
              goto LABEL_38;
            case 5:
            case 6:
              v41 = 128;
              goto LABEL_38;
            case 7:
              v41 = 8 * (unsigned int)sub_15A9520(v40, 0);
              goto LABEL_38;
            case 0xB:
              v41 = *(_DWORD *)(v39 + 8) >> 8;
              goto LABEL_38;
            case 0xD:
              v41 = 8LL * *(_QWORD *)sub_15A9930(v40, v39);
              goto LABEL_38;
            case 0xE:
              v295 = *(_QWORD *)(v39 + 24);
              v310 = *(_QWORD *)(v39 + 32);
              v49 = (unsigned int)sub_15A9FE0(v40, v295);
              v41 = 8 * v310 * v49 * ((v49 + ((unsigned __int64)(sub_127FA20(v40, v295) + 7) >> 3) - 1) / v49);
              goto LABEL_38;
            case 0xF:
              v41 = 8 * (unsigned int)sub_15A9520(v40, *(_DWORD *)(v39 + 8) >> 8);
LABEL_38:
              v42 = (unsigned __int64)(v41 * v34 + 7) >> 3;
              if ( 1 << (*(unsigned __int16 *)(v32 + 18) >> 1) >> 1 < (unsigned int)v42
                || v31[25] >> 3 < (unsigned int)v42 )
              {
                v43 = sub_15F2050(v32);
                v44 = sub_1632FA0(v43);
                v45 = sub_127FA20(v44, *(_QWORD *)v32);
                sub_20D1500(
                  (__int64)v30,
                  v32,
                  (unsigned __int64)(v45 + 7) >> 3,
                  1 << (*(unsigned __int16 *)(v32 + 18) >> 1) >> 1,
                  *(_QWORD *)(v32 - 24),
                  0,
                  a3,
                  a4,
                  a5,
                  a6,
                  v46,
                  v47,
                  a9,
                  a10,
                  0,
                  (*(unsigned __int16 *)(v32 + 18) >> 7) & 7,
                  0,
                  dword_430ABD0);
                goto LABEL_40;
              }
              v63 = *((_QWORD *)v30 + 20);
              v64 = *(__int64 (**)())(*(_QWORD *)v63 + 600LL);
              if ( v64 == sub_1F3CAD0 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v64)(v63, v32) )
                goto LABEL_73;
              v124 = *(unsigned __int16 *)(v32 + 18);
              v125 = (v124 >> 7) & 7;
              if ( !byte_428C1E0[8 * v125 + 4] )
                goto LABEL_149;
              v132 = v124 & 0x7C7F;
              v306 = v32;
              v86 = 0;
              v297 = 0;
              HIBYTE(v132) |= 1u;
              v294 = 0;
              *(_WORD *)(v32 + 18) = *(_WORD *)(v32 + 18) & 0x8000 | v132;
              break;
            default:
              goto LABEL_398;
          }
          break;
        }
LABEL_166:
        if ( v125 != 2 )
        {
          v133 = sub_16498A0(v32);
          v335 = 0;
          v338 = v133;
          v339 = 0;
          v340 = 0;
          v341 = 0;
          v342 = 0;
          v336 = *(_QWORD *)(v32 + 40);
          v337 = (__int64 *)(v32 + 24);
          v134 = *(unsigned __int8 **)(v32 + 48);
          v333[0] = v134;
          if ( v134 )
          {
            sub_1623A60((__int64)v333, (__int64)v134, 2);
            if ( v335 )
              sub_161E7C0((__int64)&v335, (__int64)v335);
            v335 = v333[0];
            if ( v333[0] )
              sub_1623210((__int64)v333, v333[0], (__int64)&v335);
          }
          v135 = *((_QWORD *)v30 + 20);
          v136 = *(_QWORD *)v135;
          v137 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(*(_QWORD *)v135 + 624LL);
          if ( v137 == sub_1F3D1E0 )
          {
            v312 = 0;
            if ( !byte_428C1E0[8 * v125 + 5] )
              goto LABEL_185;
            if ( (unsigned __int8)sub_15F3310(v32) )
            {
              v334 = 257;
              v138 = sub_1648A60(64, 0);
              v312 = (__int64)v138;
              if ( v138 )
                sub_15F9C80((__int64)v138, v338, v125, 1, 0);
              if ( v336 )
              {
                v292 = v337;
                sub_157E9D0(v336 + 40, v312);
                v139 = *v292;
                v140 = *(_QWORD *)(v312 + 24);
                *(_QWORD *)(v312 + 32) = v292;
                v139 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v312 + 24) = v139 | v140 & 7;
                *(_QWORD *)(v139 + 8) = v312 + 24;
                *v292 = *v292 & 7 | (v312 + 24);
              }
              sub_164B780(v312, (__int64 *)v333);
              if ( v335 )
              {
                v331[0] = (__int64)v335;
                sub_1623A60((__int64)v331, (__int64)v335, 2);
                v141 = *(_QWORD *)(v312 + 48);
                v142 = v312 + 48;
                if ( v141 )
                {
                  sub_161E7C0(v312 + 48, v141);
                  v142 = v312 + 48;
                }
                v143 = (unsigned __int8 *)v331[0];
                *(_QWORD *)(v312 + 48) = v331[0];
                if ( v143 )
                  sub_1623210((__int64)v331, v143, v142);
              }
            }
          }
          else
          {
            v312 = (__int64)v137(v135, (__int64 *)&v335, v32, v125);
          }
          v135 = *((_QWORD *)v30 + 20);
          v136 = *(_QWORD *)v135;
LABEL_185:
          v144 = *(_QWORD *(__fastcall **)(__int64, __int64 *, __int64, int))(v136 + 632);
          if ( v144 == sub_1F3D380 )
          {
            if ( !byte_428C1E0[8 * v125 + 4] )
            {
LABEL_198:
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v317 |= v312 != 0;
              goto LABEL_152;
            }
            v334 = 257;
            v145 = sub_1648A60(64, 0);
            v146 = (__int64)v145;
            if ( v145 )
              sub_15F9C80((__int64)v145, v338, v125, 1, 0);
            if ( v336 )
            {
              v147 = v337;
              sub_157E9D0(v336 + 40, v146);
              v148 = *(_QWORD *)(v146 + 24);
              v149 = *v147;
              *(_QWORD *)(v146 + 32) = v147;
              v149 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v146 + 24) = v149 | v148 & 7;
              *(_QWORD *)(v149 + 8) = v146 + 24;
              *v147 = *v147 & 7 | (v146 + 24);
            }
            sub_164B780(v146, (__int64 *)v333);
            if ( v335 )
            {
              v331[0] = (__int64)v335;
              sub_1623A60((__int64)v331, (__int64)v335, 2);
              v150 = *(_QWORD *)(v146 + 48);
              if ( v150 )
                sub_161E7C0(v146 + 48, v150);
              v151 = (unsigned __int8 *)v331[0];
              *(_QWORD *)(v146 + 48) = v331[0];
              if ( v151 )
                sub_1623210((__int64)v331, v151, v146 + 48);
            }
          }
          else
          {
            v146 = (__int64)v144(v135, (__int64 *)&v335, v32, v125);
          }
          v312 |= v146;
          if ( v146 )
            sub_15F2300((_QWORD *)v146, v32);
          goto LABEL_198;
        }
LABEL_152:
        if ( v306 )
        {
          v32 = v306;
LABEL_73:
          if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v32 + 8LL) - 1) > 5u )
          {
            v73 = (_QWORD *)v32;
          }
          else
          {
            v65 = sub_15F2050(v32);
            v66 = sub_1632FA0(v65);
            v67 = (__int64 *)sub_20CA660(*(_QWORD *)v32, v66);
            v68 = sub_16498A0(v32);
            v335 = 0;
            v338 = v68;
            v339 = 0;
            v340 = 0;
            v341 = 0;
            v342 = 0;
            v336 = *(_QWORD *)(v32 + 40);
            v337 = (__int64 *)(v32 + 24);
            v69 = *(unsigned __int8 **)(v32 + 48);
            v333[0] = v69;
            if ( v69 )
            {
              sub_1623A60((__int64)v333, (__int64)v69, 2);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v335 = v333[0];
              if ( v333[0] )
                sub_1623210((__int64)v333, v333[0], (__int64)&v335);
            }
            v70 = *(__int64 ****)(v32 - 24);
            v71 = *v70;
            if ( *((_BYTE *)*v70 + 8) == 16 )
              v71 = (__int64 **)*v71[2];
            v72 = (__int64 **)sub_1646BA0(v67, *((_DWORD *)v71 + 2) >> 8);
            v332 = 257;
            if ( v72 != *v70 )
            {
              if ( *((_BYTE *)v70 + 16) > 0x10u )
              {
                v334 = 257;
                v70 = (__int64 ***)sub_15FDBD0(47, (__int64)v70, (__int64)v72, (__int64)v333, 0);
                if ( v336 )
                {
                  v152 = (unsigned __int64 *)v337;
                  sub_157E9D0(v336 + 40, (__int64)v70);
                  v153 = v70[3];
                  v154 = *v152;
                  v70[4] = (__int64 **)v152;
                  v154 &= 0xFFFFFFFFFFFFFFF8LL;
                  v70[3] = (__int64 **)(v154 | (unsigned __int8)v153 & 7);
                  *(_QWORD *)(v154 + 8) = v70 + 3;
                  *v152 = *v152 & 7 | (unsigned __int64)(v70 + 3);
                }
                sub_164B780((__int64)v70, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v155 = (__int64)v70[6];
                  if ( v155 )
                    sub_161E7C0((__int64)(v70 + 6), v155);
                  v156 = v327;
                  v70[6] = (__int64 **)v327;
                  if ( v156 )
                    sub_1623210((__int64)&v327, v156, (__int64)(v70 + 6));
                }
              }
              else
              {
                v70 = (__int64 ***)sub_15A46C0(47, v70, v72, 0);
              }
            }
            v334 = 257;
            v73 = sub_1648A60(64, 1u);
            if ( v73 )
              sub_15F9210((__int64)v73, (__int64)(*v70)[3], (__int64)v70, 0, 0, 0);
            if ( v336 )
            {
              v74 = (unsigned __int64 *)v337;
              sub_157E9D0(v336 + 40, (__int64)v73);
              v75 = v73[3];
              v76 = *v74;
              v73[4] = v74;
              v76 &= 0xFFFFFFFFFFFFFFF8LL;
              v73[3] = v76 | v75 & 7;
              *(_QWORD *)(v76 + 8) = v73 + 3;
              *v74 = *v74 & 7 | (unsigned __int64)(v73 + 3);
            }
            sub_164B780((__int64)v73, (__int64 *)v333);
            if ( v335 )
            {
              v331[0] = (__int64)v335;
              sub_1623A60((__int64)v331, (__int64)v335, 2);
              v77 = v73[6];
              if ( v77 )
                sub_161E7C0((__int64)(v73 + 6), v77);
              v78 = (unsigned __int8 *)v331[0];
              v73[6] = v331[0];
              if ( v78 )
                sub_1623210((__int64)v331, v78, (__int64)(v73 + 6));
            }
            sub_15F8F50((__int64)v73, 1 << (*(unsigned __int16 *)(v32 + 18) >> 1) >> 1);
            v81 = *(_WORD *)(v32 + 18) & 1 | *((_WORD *)v73 + 9) & 0xFFFE;
            *((_WORD *)v73 + 9) = v81;
            v82 = (*(_WORD *)(v32 + 18) >> 7) & 7;
            *((_BYTE *)v73 + 56) = *(_BYTE *)(v32 + 56);
            *((_WORD *)v73 + 9) = v81 & 0x8000 | v81 & 0x7C7F | (v82 << 7);
            v332 = 257;
            v83 = *(__int64 ***)v32;
            if ( *(_QWORD *)v32 == *v73 )
            {
              v84 = v73;
            }
            else if ( *((_BYTE *)v73 + 16) > 0x10u )
            {
              v334 = 257;
              v84 = (_QWORD *)sub_15FDBD0(47, (__int64)v73, (__int64)v83, (__int64)v333, 0);
              if ( v336 )
              {
                v157 = (unsigned __int64 *)v337;
                sub_157E9D0(v336 + 40, (__int64)v84);
                v158 = v84[3];
                v159 = *v157;
                v84[4] = v157;
                v159 &= 0xFFFFFFFFFFFFFFF8LL;
                v84[3] = v159 | v158 & 7;
                *(_QWORD *)(v159 + 8) = v84 + 3;
                *v157 = *v157 & 7 | (unsigned __int64)(v84 + 3);
              }
              sub_164B780((__int64)v84, v331);
              if ( v335 )
              {
                v327 = v335;
                sub_1623A60((__int64)&v327, (__int64)v335, 2);
                v160 = v84[6];
                v161 = (__int64)(v84 + 6);
                if ( v160 )
                {
                  sub_161E7C0((__int64)(v84 + 6), v160);
                  v161 = (__int64)(v84 + 6);
                }
                v162 = v327;
                v84[6] = v327;
                if ( v162 )
                  sub_1623210((__int64)&v327, v162, v161);
              }
            }
            else
            {
              v84 = (_QWORD *)sub_15A46C0(47, (__int64 ***)v73, v83, 0);
            }
            sub_164D160(v32, (__int64)v84, a3, a4, a5, a6, v79, v80, a9, a10);
            sub_15F20C0((_QWORD *)v32);
            if ( v335 )
              sub_161E7C0((__int64)&v335, (__int64)v335);
            v317 = v304;
          }
          v317 |= sub_20CADE0((__int64)v30, (__int64)v73, a3, a4, a5, a6, v37, v38, a9, a10);
          goto LABEL_41;
        }
        if ( v86 )
        {
LABEL_102:
          if ( (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(v86 - 48) + 8LL) - 1) > 5u )
          {
            v99 = v86;
          }
          else
          {
            v88 = sub_16498A0(v86);
            v335 = 0;
            v338 = v88;
            v339 = 0;
            v340 = 0;
            v341 = 0;
            v342 = 0;
            v336 = *(_QWORD *)(v86 + 40);
            v337 = (__int64 *)(v86 + 24);
            v89 = *(unsigned __int8 **)(v86 + 48);
            v333[0] = v89;
            if ( v89 )
            {
              sub_1623A60((__int64)v333, (__int64)v89, 2);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v335 = v333[0];
              if ( v333[0] )
                sub_1623210((__int64)v333, v333[0], (__int64)&v335);
            }
            v90 = sub_15F2050(v86);
            v91 = sub_1632FA0(v90);
            v92 = (__int64 **)sub_20CA660(**(_QWORD **)(v86 - 48), v91);
            v332 = 257;
            v93 = *(_QWORD *)(v86 - 48);
            if ( v92 != *(__int64 ***)v93 )
            {
              if ( *(_BYTE *)(v93 + 16) > 0x10u )
              {
                v163 = *(_QWORD *)(v86 - 48);
                v334 = 257;
                v93 = sub_15FDBD0(47, v163, (__int64)v92, (__int64)v333, 0);
                if ( v336 )
                {
                  v164 = v337;
                  sub_157E9D0(v336 + 40, v93);
                  v165 = *(_QWORD *)(v93 + 24);
                  v166 = *v164;
                  *(_QWORD *)(v93 + 32) = v164;
                  v166 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v93 + 24) = v166 | v165 & 7;
                  *(_QWORD *)(v166 + 8) = v93 + 24;
                  *v164 = *v164 & 7 | (v93 + 24);
                }
                sub_164B780(v93, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v167 = *(_QWORD *)(v93 + 48);
                  v168 = v93 + 48;
                  if ( v167 )
                  {
                    sub_161E7C0(v93 + 48, v167);
                    v168 = v93 + 48;
                  }
                  v169 = v327;
                  *(_QWORD *)(v93 + 48) = v327;
                  if ( v169 )
                    sub_1623210((__int64)&v327, v169, v168);
                }
              }
              else
              {
                v93 = sub_15A46C0(47, *(__int64 ****)(v86 - 48), v92, 0);
              }
            }
            v94 = *(__int64 **)(v86 - 24);
            v95 = *v94;
            if ( *(_BYTE *)(*v94 + 8) == 16 )
              v95 = **(_QWORD **)(v95 + 16);
            v318 = *(_QWORD *)(v86 - 24);
            v96 = (__int64 **)sub_1646BA0((__int64 *)v92, *(_DWORD *)(v95 + 8) >> 8);
            v97 = v318;
            v332 = 257;
            if ( v96 != *(__int64 ***)v318 )
            {
              if ( *(_BYTE *)(v318 + 16) > 0x10u )
              {
                v334 = 257;
                v170 = sub_15FDBD0(47, v318, (__int64)v96, (__int64)v333, 0);
                if ( v336 )
                {
                  v171 = v337;
                  v320 = v170;
                  sub_157E9D0(v336 + 40, v170);
                  v170 = v320;
                  v172 = *v171;
                  v173 = *(_QWORD *)(v320 + 24);
                  *(_QWORD *)(v320 + 32) = v171;
                  v172 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v320 + 24) = v172 | v173 & 7;
                  *(_QWORD *)(v172 + 8) = v320 + 24;
                  *v171 = *v171 & 7 | (v320 + 24);
                }
                v321 = v170;
                sub_164B780(v170, v331);
                v97 = v321;
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v97 = v321;
                  v174 = *(_QWORD *)(v321 + 48);
                  v175 = v321 + 48;
                  if ( v174 )
                  {
                    sub_161E7C0(v321 + 48, v174);
                    v97 = v321;
                  }
                  v176 = v327;
                  *(_QWORD *)(v97 + 48) = v327;
                  if ( v176 )
                  {
                    v322 = v97;
                    sub_1623210((__int64)&v327, v176, v175);
                    v97 = v322;
                  }
                }
              }
              else
              {
                v97 = sub_15A46C0(47, (__int64 ***)v318, v96, 0);
              }
            }
            v319 = v97;
            v334 = 257;
            v98 = sub_1648A60(64, 2u);
            v99 = (__int64)v98;
            if ( v98 )
              sub_15F9650((__int64)v98, v93, v319, 0, 0);
            if ( v336 )
            {
              v100 = v337;
              sub_157E9D0(v336 + 40, v99);
              v101 = *(_QWORD *)(v99 + 24);
              v102 = *v100;
              *(_QWORD *)(v99 + 32) = v100;
              v102 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v99 + 24) = v102 | v101 & 7;
              *(_QWORD *)(v102 + 8) = v99 + 24;
              *v100 = *v100 & 7 | (v99 + 24);
            }
            sub_164B780(v99, (__int64 *)v333);
            if ( v335 )
            {
              v331[0] = (__int64)v335;
              sub_1623A60((__int64)v331, (__int64)v335, 2);
              v103 = *(_QWORD *)(v99 + 48);
              if ( v103 )
                sub_161E7C0(v99 + 48, v103);
              v104 = (unsigned __int8 *)v331[0];
              *(_QWORD *)(v99 + 48) = v331[0];
              if ( v104 )
                sub_1623210((__int64)v331, v104, v99 + 48);
            }
            sub_15F9450(v99, 1 << (*(unsigned __int16 *)(v86 + 18) >> 1) >> 1);
            v105 = *(_WORD *)(v86 + 18) & 1 | *(_WORD *)(v99 + 18) & 0xFFFE;
            *(_WORD *)(v99 + 18) = v105;
            v106 = (*(unsigned __int16 *)(v86 + 18) >> 7) & 7;
            *(_BYTE *)(v99 + 56) = *(_BYTE *)(v86 + 56);
            *(_WORD *)(v99 + 18) = v105 & 0x8000 | v105 & 0x7C7F | ((_WORD)v106 << 7);
            sub_15F20C0((_QWORD *)v86);
            if ( v335 )
              sub_161E7C0((__int64)&v335, (__int64)v335);
            v317 = v304;
          }
          v107 = *((_QWORD *)v30 + 20);
          v108 = *(__int64 (**)())(*(_QWORD *)v107 + 648LL);
          if ( v108 != sub_1F3CAF0 && ((unsigned __int8 (__fastcall *)(__int64, __int64))v108)(v107, v99) )
          {
            v109 = sub_16498A0(v99);
            v335 = 0;
            v338 = v109;
            v339 = 0;
            v340 = 0;
            v341 = 0;
            v342 = 0;
            v336 = *(_QWORD *)(v99 + 40);
            v337 = (__int64 *)(v99 + 24);
            v110 = *(unsigned __int8 **)(v99 + 48);
            v333[0] = v110;
            if ( v110 )
            {
              sub_1623A60((__int64)v333, (__int64)v110, 2);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v335 = v333[0];
              if ( v333[0] )
                sub_1623210((__int64)v333, v333[0], (__int64)&v335);
            }
            v111 = *(unsigned __int16 *)(v99 + 18);
            v112 = *(__int64 **)(v99 - 48);
            v113 = *(_QWORD *)(v99 - 24);
            v334 = 257;
            v305 = (v111 >> 7) & 7;
            v114 = sub_1648A60(64, 2u);
            v115 = v114;
            if ( v114 )
              sub_15F9C10((__int64)v114, 0, v113, v112, v305, 1, 0);
            if ( v336 )
            {
              v116 = (unsigned __int64 *)v337;
              sub_157E9D0(v336 + 40, (__int64)v115);
              v117 = v115[3];
              v118 = *v116;
              v115[4] = (__int64)v116;
              v118 &= 0xFFFFFFFFFFFFFFF8LL;
              v115[3] = v118 | v117 & 7;
              *(_QWORD *)(v118 + 8) = v115 + 3;
              *v116 = *v116 & 7 | (unsigned __int64)(v115 + 3);
            }
            sub_164B780((__int64)v115, (__int64 *)v333);
            if ( v335 )
            {
              v331[0] = (__int64)v335;
              sub_1623A60((__int64)v331, (__int64)v335, 2);
              v119 = v115[6];
              if ( v119 )
                sub_161E7C0((__int64)(v115 + 6), v119);
              v120 = (unsigned __int8 *)v331[0];
              v115[6] = v331[0];
              if ( v120 )
                sub_1623210((__int64)v331, v120, (__int64)(v115 + 6));
            }
            sub_15F20C0((_QWORD *)v99);
            v123 = sub_20CC3C0((__int64)v30, v115, a3, a4, a5, a6, v121, v122, a9, a10);
            if ( v335 )
              sub_161E7C0((__int64)&v335, (__int64)v335);
            v317 |= v123;
          }
          goto LABEL_41;
        }
        goto LABEL_154;
      case '7':
        v50 = 1;
        v51 = sub_15F2050(*v326);
        v52 = sub_1632FA0(v51);
        v53 = **(_QWORD **)(v32 - 48);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v53 + 8) )
          {
            case 1:
              v54 = 16;
              goto LABEL_59;
            case 2:
              v54 = 32;
              goto LABEL_59;
            case 3:
            case 9:
              v54 = 64;
              goto LABEL_59;
            case 4:
              v54 = 80;
              goto LABEL_59;
            case 5:
            case 6:
              v54 = 128;
              goto LABEL_59;
            case 7:
              v54 = 8 * (unsigned int)sub_15A9520(v52, 0);
              goto LABEL_59;
            case 0xB:
              v54 = *(_DWORD *)(v53 + 8) >> 8;
              goto LABEL_59;
            case 0xD:
              v54 = 8LL * *(_QWORD *)sub_15A9930(v52, v53);
              goto LABEL_59;
            case 0xE:
              v296 = *(_QWORD *)(v53 + 24);
              v311 = *(_QWORD *)(v53 + 32);
              v62 = (unsigned int)sub_15A9FE0(v52, v296);
              v54 = 8 * v311 * v62 * ((v62 + ((unsigned __int64)(sub_127FA20(v52, v296) + 7) >> 3) - 1) / v62);
              goto LABEL_59;
            case 0xF:
              v54 = 8 * (unsigned int)sub_15A9520(v52, *(_DWORD *)(v53 + 8) >> 8);
LABEL_59:
              v55 = (unsigned __int64)(v54 * v50 + 7) >> 3;
              if ( 1 << (*(unsigned __int16 *)(v32 + 18) >> 1) >> 1 < (unsigned int)v55
                || v31[25] >> 3 < (unsigned int)v55 )
              {
                v56 = sub_15F2050(v32);
                v57 = sub_1632FA0(v56);
                v58 = sub_127FA20(v57, **(_QWORD **)(v32 - 48));
                sub_20D1500(
                  (__int64)v30,
                  v32,
                  (unsigned __int64)(v58 + 7) >> 3,
                  1 << (*(unsigned __int16 *)(v32 + 18) >> 1) >> 1,
                  *(_QWORD *)(v32 - 24),
                  *(_QWORD **)(v32 - 48),
                  a3,
                  a4,
                  a5,
                  a6,
                  v59,
                  v60,
                  a9,
                  a10,
                  0,
                  (*(unsigned __int16 *)(v32 + 18) >> 7) & 7,
                  0,
                  dword_430ABB0);
LABEL_40:
                v317 = v304;
                goto LABEL_41;
              }
              v85 = *((_QWORD *)v30 + 20);
              v86 = v32;
              v87 = *(__int64 (**)())(*(_QWORD *)v85 + 600LL);
              if ( v87 == sub_1F3CAD0 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v87)(v85, v32) )
                goto LABEL_102;
              v177 = *(unsigned __int16 *)(v32 + 18);
              v125 = (v177 >> 7) & 7;
              if ( byte_428C1E0[8 * v125 + 5] )
              {
                v178 = v177 & 0x7C7F;
                v297 = 0;
                v294 = 0;
                HIBYTE(v178) |= 1u;
                v306 = 0;
                *(_WORD *)(v32 + 18) = *(_WORD *)(v32 + 18) & 0x8000 | v178;
                goto LABEL_166;
              }
              v306 = 0;
              break;
            case 0x10:
              v61 = *(_QWORD *)(v53 + 32);
              v53 = *(_QWORD *)(v53 + 24);
              v50 *= v61;
              continue;
            default:
              goto LABEL_398;
          }
          goto LABEL_150;
        }
      case ';':
        v253 = sub_15F2050(*v326);
        v254 = sub_1632FA0(v253);
        v255 = sub_127FA20(v254, **(_QWORD **)(v32 - 24)) + 7;
        v256 = sub_15F2050(v32);
        v257 = v255 >> 3;
        v258 = sub_1632FA0(v256);
        if ( (unsigned int)v257 > (unsigned int)((unsigned __int64)(sub_127FA20(v258, **(_QWORD **)(v32 - 24)) + 7) >> 3)
          || v31[25] >> 3 < (unsigned int)v257 )
        {
          switch ( (*(unsigned __int16 *)(v32 + 18) >> 5) & 0x7FFFBFF )
          {
            case 0:
              v259 = 6;
              v260 = (int *)&unk_430AB70;
              goto LABEL_334;
            case 1:
              v259 = 6;
              v260 = (int *)&unk_430AB50;
              goto LABEL_334;
            case 2:
              v259 = 6;
              v260 = (int *)&unk_430AB30;
              goto LABEL_334;
            case 3:
              v259 = 6;
              v260 = (int *)&unk_430AB10;
              goto LABEL_334;
            case 4:
              v259 = 6;
              v260 = (int *)&unk_430AAB0;
              goto LABEL_334;
            case 5:
              v259 = 6;
              v260 = (int *)&unk_430AAF0;
              goto LABEL_334;
            case 6:
              v259 = 6;
              v260 = (int *)&unk_430AAD0;
              goto LABEL_334;
            case 7:
            case 8:
            case 9:
            case 0xA:
              v259 = 0;
              v260 = 0;
LABEL_334:
              v261 = sub_15F2050(v32);
              v325 = 1;
              v262 = sub_1632FA0(v261);
              v263 = **(_QWORD **)(v32 - 24);
              while ( 2 )
              {
                switch ( *(_BYTE *)(v263 + 8) )
                {
                  case 1:
                    v264 = 16;
                    goto LABEL_337;
                  case 2:
                    v264 = 32;
                    goto LABEL_337;
                  case 3:
                  case 9:
                    v264 = 64;
                    goto LABEL_337;
                  case 4:
                    v264 = 80;
                    goto LABEL_337;
                  case 5:
                  case 6:
                    v264 = 128;
                    goto LABEL_337;
                  case 7:
                    v264 = 8 * (unsigned int)sub_15A9520(v262, 0);
                    goto LABEL_337;
                  case 0xB:
                    v264 = *(_DWORD *)(v263 + 8) >> 8;
                    goto LABEL_337;
                  case 0xD:
                    v264 = 8LL * *(_QWORD *)sub_15A9930(v262, v263);
                    goto LABEL_337;
                  case 0xE:
                    v302 = *(_QWORD *)(v263 + 24);
                    v315 = *(_QWORD *)(v263 + 32);
                    v273 = (unsigned int)sub_15A9FE0(v262, v302);
                    v264 = 8
                         * v315
                         * v273
                         * ((v273 + ((unsigned __int64)(sub_127FA20(v262, v302) + 7) >> 3) - 1)
                          / v273);
                    goto LABEL_337;
                  case 0xF:
                    v264 = 8 * (unsigned int)sub_15A9520(v262, *(_DWORD *)(v263 + 8) >> 8);
LABEL_337:
                    v265 = sub_15F2050(v32);
                    v314 = 1;
                    v266 = sub_1632FA0(v265);
                    v269 = **(_QWORD **)(v32 - 24);
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v269 + 8) )
                      {
                        case 1:
                          v270 = 16;
                          goto LABEL_340;
                        case 2:
                          v270 = 32;
                          goto LABEL_340;
                        case 3:
                        case 9:
                          v270 = 64;
                          goto LABEL_340;
                        case 4:
                          v270 = 80;
                          goto LABEL_340;
                        case 5:
                        case 6:
                          v270 = 128;
                          goto LABEL_340;
                        case 7:
                          v270 = 8 * (unsigned int)sub_15A9520(v266, 0);
                          goto LABEL_340;
                        case 0xB:
                          v270 = *(_DWORD *)(v269 + 8) >> 8;
                          goto LABEL_340;
                        case 0xD:
                          v270 = 8LL * *(_QWORD *)sub_15A9930(v266, v269);
                          goto LABEL_340;
                        case 0xE:
                          v293 = *(_QWORD *)(v269 + 24);
                          v308 = *(_QWORD *)(v269 + 32);
                          v301 = (unsigned int)sub_15A9FE0(v266, v293);
                          v270 = 8
                               * v308
                               * v301
                               * ((v301 + ((unsigned __int64)(sub_127FA20(v266, v293) + 7) >> 3) - 1)
                                / v301);
                          goto LABEL_340;
                        case 0xF:
                          v270 = 8 * (unsigned int)sub_15A9520(v266, *(_DWORD *)(v269 + 8) >> 8);
LABEL_340:
                          if ( !v259
                            || !sub_20D1500(
                                  (__int64)v30,
                                  v32,
                                  (unsigned __int64)(v264 * v325 + 7) >> 3,
                                  (unsigned __int64)(v314 * v270 + 7) >> 3,
                                  *(_QWORD *)(v32 - 48),
                                  *(_QWORD **)(v32 - 24),
                                  a3,
                                  a4,
                                  a5,
                                  a6,
                                  v267,
                                  v268,
                                  a9,
                                  a10,
                                  0,
                                  (*(unsigned __int16 *)(v32 + 18) >> 2) & 7,
                                  0,
                                  v260) )
                          {
                            v335 = v30;
                            sub_20CAAD0(
                              (__int64 *)v32,
                              (void (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64, _QWORD, unsigned __int8 **, __int64 *))sub_20D3190,
                              (__int64)&v335,
                              a3,
                              a4,
                              a5,
                              a6,
                              v267,
                              v268,
                              a9,
                              a10);
                          }
                          goto LABEL_40;
                        case 0x10:
                          v271 = v314 * *(_QWORD *)(v269 + 32);
                          v269 = *(_QWORD *)(v269 + 24);
                          v314 = v271;
                          continue;
                        default:
                          goto LABEL_398;
                      }
                    }
                  case 0x10:
                    v272 = v325 * *(_QWORD *)(v263 + 32);
                    v263 = *(_QWORD *)(v263 + 24);
                    v325 = v272;
                    continue;
                  default:
                    goto LABEL_398;
                }
              }
            default:
              break;
          }
LABEL_398:
          BUG();
        }
        v274 = *((_QWORD *)v30 + 20);
        v275 = *(__int64 (**)())(*(_QWORD *)v274 + 600LL);
        if ( v275 == sub_1F3CAD0 )
        {
LABEL_156:
          v126 = *(_QWORD *)(v32 - 24);
          if ( *(_BYTE *)(v126 + 16) != 13 )
            goto LABEL_217;
          v127 = (*(unsigned __int16 *)(v32 + 18) >> 5) & 0x7FFFBFF;
          if ( v127 == 3 )
          {
            v179 = *(_DWORD *)(v126 + 32);
            if ( v179 <= 0x40 )
              v180 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v179) == *(_QWORD *)(v126 + 24);
            else
              v180 = v179 == (unsigned int)sub_16A58F0(v126 + 24);
            if ( !v180 )
              goto LABEL_217;
          }
          else
          {
            if ( v127 <= 3 )
            {
              if ( v127 - 1 <= 1 )
                goto LABEL_160;
LABEL_217:
              v317 |= sub_20CC3C0((__int64)v30, (__int64 *)v32, a3, a4, a5, a6, v37, v38, a9, a10);
              goto LABEL_41;
            }
            if ( v127 - 5 > 1 )
              goto LABEL_217;
LABEL_160:
            v128 = *(_DWORD *)(v126 + 32);
            if ( v128 <= 0x40 )
            {
              if ( *(_QWORD *)(v126 + 24) )
                goto LABEL_217;
            }
            else if ( v128 != (unsigned int)sub_16A57B0(v126 + 24) )
            {
              goto LABEL_217;
            }
          }
          v129 = *((_QWORD *)v30 + 20);
          v130 = *(__int64 (**)())(*(_QWORD *)v129 + 688LL);
          if ( v130 != sub_1F3CB40 )
          {
            v131 = ((__int64 (__fastcall *)(__int64, __int64))v130)(v129, v32);
            if ( v131 )
            {
              sub_20CADE0((__int64)v30, v131, a3, a4, a5, a6, v37, v38, a9, a10);
              v317 = v304;
              goto LABEL_41;
            }
          }
          goto LABEL_217;
        }
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v275)(v274, v32) )
        {
          v297 = 0;
          goto LABEL_155;
        }
        v276 = *(unsigned __int16 *)(v32 + 18);
        v125 = (v276 >> 2) & 7;
        if ( byte_428C1E0[8 * v125 + 5] || byte_428C1E0[8 * v125 + 4] )
        {
          v294 = v32;
          v86 = 0;
          v297 = 0;
          v306 = 0;
          *(_WORD *)(v32 + 18) = *(_WORD *)(v32 + 18) & 0x8000 | v276 & 0x7FE3 | 8;
          goto LABEL_166;
        }
        v306 = 0;
        v86 = 0;
LABEL_151:
        v294 = v32;
        v297 = 0;
        goto LABEL_152;
      case ':':
        v181 = (_QWORD *)*v326;
        v277 = sub_15F2050(*v326);
        v278 = sub_1632FA0(v277);
        v279 = sub_127FA20(v278, **(_QWORD **)(v32 - 48)) + 7;
        v280 = sub_15F2050(v32);
        v281 = v279 >> 3;
        v282 = sub_1632FA0(v280);
        if ( (unsigned int)v281 > (unsigned int)((unsigned __int64)(sub_127FA20(v282, **(_QWORD **)(v32 - 48)) + 7) >> 3)
          || v31[25] >> 3 < (unsigned int)v281 )
        {
          sub_20D2E80((__int64)v30, v32, a3, a4, a5, a6, v283, v284, a9, a10);
          v317 = v304;
          goto LABEL_41;
        }
        v286 = *((_QWORD *)v30 + 20);
        v287 = *(__int64 (**)())(*(_QWORD *)v286 + 600LL);
        if ( v287 == sub_1F3CAD0 )
          goto LABEL_249;
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v287)(v286, v32) )
        {
          v297 = (_QWORD *)v32;
LABEL_248:
          v181 = v297;
          v32 = (__int64)v297;
LABEL_249:
          if ( *(_BYTE *)(**(_QWORD **)(v32 - 48) + 8LL) == 15 )
          {
            v197 = sub_15F2050((__int64)v181);
            v198 = sub_1632FA0(v197);
            v299 = (__int64 *)sub_20CA660(**(_QWORD **)(v32 - 48), v198);
            v199 = sub_16498A0((__int64)v181);
            v335 = 0;
            v338 = v199;
            v339 = 0;
            v340 = 0;
            v341 = 0;
            v342 = 0;
            v336 = *(_QWORD *)(v32 + 40);
            v337 = (__int64 *)(v32 + 24);
            v200 = *(unsigned __int8 **)(v32 + 48);
            v333[0] = v200;
            if ( v200 )
            {
              sub_1623A60((__int64)v333, (__int64)v200, 2);
              if ( v335 )
                sub_161E7C0((__int64)&v335, (__int64)v335);
              v335 = v333[0];
              if ( v333[0] )
                sub_1623210((__int64)v333, v333[0], (__int64)&v335);
            }
            v201 = *(__int64 ****)(v32 - 72);
            v202 = *v201;
            if ( *((_BYTE *)*v201 + 8) == 16 )
              v202 = (__int64 **)*v202[2];
            v203 = (__int64 **)sub_1646BA0(v299, *((_DWORD *)v202 + 2) >> 8);
            v332 = 257;
            if ( v203 != *v201 )
            {
              if ( *((_BYTE *)v201 + 16) > 0x10u )
              {
                v334 = 257;
                v246 = sub_15FDBD0(47, (__int64)v201, (__int64)v203, (__int64)v333, 0);
                v201 = (__int64 ***)v246;
                if ( v336 )
                {
                  v247 = (unsigned __int64 *)v337;
                  sub_157E9D0(v336 + 40, v246);
                  v248 = v201[3];
                  v249 = *v247;
                  v201[4] = (__int64 **)v247;
                  v249 &= 0xFFFFFFFFFFFFFFF8LL;
                  v201[3] = (__int64 **)(v249 | (unsigned __int8)v248 & 7);
                  *(_QWORD *)(v249 + 8) = v201 + 3;
                  *v247 = *v247 & 7 | (unsigned __int64)(v201 + 3);
                }
                sub_164B780((__int64)v201, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v250 = (__int64)v201[6];
                  v251 = (__int64)(v201 + 6);
                  if ( v250 )
                  {
                    sub_161E7C0((__int64)(v201 + 6), v250);
                    v251 = (__int64)(v201 + 6);
                  }
                  v252 = v327;
                  v201[6] = (__int64 **)v327;
                  if ( v252 )
                    sub_1623210((__int64)&v327, v252, v251);
                }
              }
              else
              {
                v201 = (__int64 ***)sub_15A46C0(47, v201, v203, 0);
              }
            }
            v332 = 257;
            v204 = *(_QWORD *)(v32 - 48);
            v323 = v204;
            if ( v299 != *(__int64 **)v204 )
            {
              if ( *(_BYTE *)(v204 + 16) > 0x10u )
              {
                v334 = 257;
                v232 = sub_15FDBD0(45, v204, (__int64)v299, (__int64)v333, 0);
                v323 = v232;
                if ( v336 )
                {
                  v233 = v337;
                  sub_157E9D0(v336 + 40, v232);
                  v234 = *(_QWORD *)(v323 + 24);
                  v235 = *v233 & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v323 + 32) = v233;
                  *(_QWORD *)(v323 + 24) = v235 | v234 & 7;
                  *(_QWORD *)(v235 + 8) = v323 + 24;
                  *v233 = *v233 & 7 | (v323 + 24);
                }
                sub_164B780(v323, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v236 = *(_QWORD *)(v323 + 48);
                  v237 = v323 + 48;
                  if ( v236 )
                  {
                    sub_161E7C0(v323 + 48, v236);
                    v237 = v323 + 48;
                  }
                  v238 = v327;
                  *(_QWORD *)(v323 + 48) = v327;
                  if ( v238 )
                    sub_1623210((__int64)&v327, v238, v237);
                }
              }
              else
              {
                v323 = sub_15A46C0(45, (__int64 ***)v204, (__int64 **)v299, 0);
              }
            }
            v332 = 257;
            v205 = *(_QWORD *)(v32 - 24);
            v307 = (_QWORD *)v205;
            if ( v299 != *(__int64 **)v205 )
            {
              if ( *(_BYTE *)(v205 + 16) > 0x10u )
              {
                v334 = 257;
                v239 = sub_15FDBD0(45, v205, (__int64)v299, (__int64)v333, 0);
                v307 = (_QWORD *)v239;
                if ( v336 )
                {
                  v240 = (unsigned __int64 *)v337;
                  sub_157E9D0(v336 + 40, v239);
                  v241 = *v240;
                  v242 = v307[3];
                  v307[4] = v240;
                  v241 &= 0xFFFFFFFFFFFFFFF8LL;
                  v307[3] = v241 | v242 & 7;
                  *(_QWORD *)(v241 + 8) = v307 + 3;
                  *v240 = *v240 & 7 | (unsigned __int64)(v307 + 3);
                }
                sub_164B780((__int64)v307, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v243 = v307[6];
                  v244 = (__int64)(v307 + 6);
                  if ( v243 )
                  {
                    sub_161E7C0((__int64)(v307 + 6), v243);
                    v244 = (__int64)(v307 + 6);
                  }
                  v245 = v327;
                  v307[6] = v327;
                  if ( v245 )
                    sub_1623210((__int64)&v327, v245, v244);
                }
              }
              else
              {
                v307 = (_QWORD *)sub_15A46C0(45, (__int64 ***)v205, (__int64 **)v299, 0);
              }
            }
            v206 = *(unsigned __int16 *)(v32 + 18);
            v303 = *(_BYTE *)(v32 + 56);
            v207 = (v206 >> 2) & 7;
            v291 = (unsigned __int8)v206 >> 5;
            v334 = 257;
            v208 = sub_1648A60(64, 3u);
            v300 = (__int64)v208;
            if ( v208 )
              sub_15F99E0((__int64)v208, (__int64)v201, (__int64 **)v323, (__int64)v307, v207, v291, v303, 0);
            if ( v336 )
            {
              v209 = v337;
              sub_157E9D0(v336 + 40, v300);
              v210 = *v209;
              v211 = *(_QWORD *)(v300 + 24);
              *(_QWORD *)(v300 + 32) = v209;
              v210 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v300 + 24) = v210 | v211 & 7;
              *(_QWORD *)(v210 + 8) = v300 + 24;
              *v209 = *v209 & 7 | (v300 + 24);
            }
            sub_164B780(v300, (__int64 *)v333);
            if ( v335 )
            {
              v331[0] = (__int64)v335;
              sub_1623A60((__int64)v331, (__int64)v335, 2);
              v212 = *(_QWORD *)(v300 + 48);
              if ( v212 )
                sub_161E7C0(v300 + 48, v212);
              v213 = (unsigned __int8 *)v331[0];
              *(_QWORD *)(v300 + 48) = v331[0];
              if ( v213 )
                sub_1623210((__int64)v331, v213, v300 + 48);
            }
            v214 = *(_WORD *)(v300 + 18) & 0x8000 | *(_WORD *)(v300 + 18) & 0x7FFE | *(_WORD *)(v32 + 18) & 1;
            *(_WORD *)(v300 + 18) = v214;
            *(_WORD *)(v300 + 18) = v214 & 0x7EFF | ((*(_BYTE *)(v32 + 19) & 1) << 8) | v214 & 0x8000;
            v334 = 257;
            LODWORD(v331[0]) = 0;
            v215 = sub_12A9E60((__int64 *)&v335, v300, (__int64)v331, 1, (__int64)v333);
            v334 = 257;
            v216 = v215;
            LODWORD(v331[0]) = 1;
            v217 = sub_12A9E60((__int64 *)&v335, v300, (__int64)v331, 1, (__int64)v333);
            v332 = 257;
            v324 = v217;
            v218 = *(__int64 ****)(v32 - 48);
            v219 = *v218;
            if ( *v218 != *(__int64 ***)v216 )
            {
              if ( *(_BYTE *)(v216 + 16) > 0x10u )
              {
                v334 = 257;
                v225 = sub_15FDBD0(46, v216, (__int64)v219, (__int64)v333, 0);
                v216 = v225;
                if ( v336 )
                {
                  v226 = v337;
                  sub_157E9D0(v336 + 40, v225);
                  v227 = *(_QWORD *)(v216 + 24);
                  v228 = *v226;
                  *(_QWORD *)(v216 + 32) = v226;
                  v228 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v216 + 24) = v228 | v227 & 7;
                  *(_QWORD *)(v228 + 8) = v216 + 24;
                  *v226 = *v226 & 7 | (v216 + 24);
                }
                sub_164B780(v216, v331);
                if ( v335 )
                {
                  v327 = v335;
                  sub_1623A60((__int64)&v327, (__int64)v335, 2);
                  v229 = *(_QWORD *)(v216 + 48);
                  v230 = v216 + 48;
                  if ( v229 )
                  {
                    sub_161E7C0(v216 + 48, v229);
                    v230 = v216 + 48;
                  }
                  v231 = v327;
                  *(_QWORD *)(v216 + 48) = v327;
                  if ( v231 )
                    sub_1623210((__int64)&v327, v231, v230);
                }
              }
              else
              {
                v216 = sub_15A46C0(46, (__int64 ***)v216, v219, 0);
              }
            }
            v220 = sub_1599EF0(*(__int64 ***)v32);
            v334 = 257;
            LODWORD(v331[0]) = 0;
            v221 = sub_17FE490((__int64 *)&v335, v220, v216, v331, 1, (__int64 *)v333);
            v334 = 257;
            LODWORD(v331[0]) = 1;
            v222 = sub_17FE490((__int64 *)&v335, v221, v324, v331, 1, (__int64 *)v333);
            sub_164D160((__int64)v181, v222, a3, a4, a5, a6, v223, v224, a9, a10);
            sub_15F20C0(v181);
            if ( v335 )
              sub_161E7C0((__int64)&v335, (__int64)v335);
            v181 = (_QWORD *)v300;
            v32 = v300;
            v317 = v304;
          }
          v182 = 1;
          v183 = *(_DWORD *)(*((_QWORD *)v30 + 20) + 104LL);
          v184 = sub_15F2050((__int64)v181);
          v185 = v183 >> 3;
          v186 = sub_1632FA0(v184);
          v189 = **(_QWORD **)(v32 - 48);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v189 + 8) )
            {
              case 1:
                v190 = 16;
                goto LABEL_253;
              case 2:
                v190 = 32;
                goto LABEL_253;
              case 3:
              case 9:
                v190 = 64;
                goto LABEL_253;
              case 4:
                v190 = 80;
                goto LABEL_253;
              case 5:
              case 6:
                v190 = 128;
                goto LABEL_253;
              case 7:
                v190 = 8 * (unsigned int)sub_15A9520(v186, 0);
                goto LABEL_253;
              case 0xB:
                v190 = *(_DWORD *)(v189 + 8) >> 8;
                goto LABEL_253;
              case 0xD:
                v190 = 8LL * *(_QWORD *)sub_15A9930(v186, v189);
                goto LABEL_253;
              case 0xE:
                v298 = *(_QWORD *)(v189 + 24);
                v313 = *(_QWORD *)(v189 + 32);
                v196 = (unsigned int)sub_15A9FE0(v186, v298);
                v190 = 8 * v313 * v196 * ((v196 + ((unsigned __int64)(sub_127FA20(v186, v298) + 7) >> 3) - 1) / v196);
                goto LABEL_253;
              case 0xF:
                v190 = 8 * (unsigned int)sub_15A9520(v186, *(_DWORD *)(v189 + 8) >> 8);
LABEL_253:
                if ( v185 > (unsigned int)((unsigned __int64)(v182 * v190 + 7) >> 3) )
                {
                  sub_20CD3E0((__int64)v30, v32, a3, a4, a5, a6, v187, v188, a9, a10);
                }
                else
                {
                  v191 = *((_QWORD *)v30 + 20);
                  v192 = *(__int64 (**)())(*(_QWORD *)v191 + 672LL);
                  if ( v192 != sub_1F3CB20 && ((unsigned __int8 (__fastcall *)(__int64, __int64))v192)(v191, v32) )
                    v317 |= sub_20CEB70((__int64)v30, v32, a3, a4, a5, a6, v193, v194, a9, a10);
                }
                goto LABEL_41;
              case 0x10:
                v195 = *(_QWORD *)(v189 + 32);
                v189 = *(_QWORD *)(v189 + 24);
                v182 *= v195;
                continue;
              default:
                goto LABEL_398;
            }
          }
        }
        v288 = *((_QWORD *)v30 + 20);
        v289 = *(__int64 (**)())(*(_QWORD *)v288 + 672LL);
        if ( v289 == sub_1F3CB20
          || (v297 = (_QWORD *)v32, !((unsigned __int8 (__fastcall *)(__int64, __int64))v289)(v288, v32)) )
        {
          v290 = *(unsigned __int16 *)(v32 + 18);
          v125 = (v290 >> 2) & 7;
          if ( byte_428C1E0[8 * v125 + 5] || byte_428C1E0[8 * v125 + 4] )
          {
            v297 = (_QWORD *)v32;
            v86 = 0;
            v294 = 0;
            v306 = 0;
            *(_WORD *)(v32 + 18) = *(_WORD *)(v32 + 18) & 0x8000 | v290 & 0x7F03 | 0x48;
            goto LABEL_166;
          }
          v297 = (_QWORD *)v32;
        }
        break;
      default:
        v285 = *(__int64 (**)())(*(_QWORD *)v31 + 600LL);
        if ( v285 == sub_1F3CAD0 )
          goto LABEL_41;
        if ( ((unsigned __int8 (__fastcall *)(_QWORD, __int64))v285)(*((_QWORD *)v30 + 20), *v326) )
        {
          v32 = 0;
LABEL_149:
          v306 = v32;
          v32 = 0;
LABEL_150:
          v86 = v32;
          v32 = 0;
          goto LABEL_151;
        }
        v297 = 0;
        break;
    }
    v294 = 0;
LABEL_154:
    v32 = v294;
LABEL_155:
    if ( v32 )
      goto LABEL_156;
    if ( v297 )
      goto LABEL_248;
LABEL_41:
    ++v326;
  }
  while ( v316 != v326 );
  v22 = v317;
  v29 = v328;
LABEL_43:
  if ( v29 != (__int64 *)v330 )
    _libc_free((unsigned __int64)v29);
  return v22;
}
