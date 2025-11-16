// Function: sub_1F50270
// Address: 0x1f50270
//
_BOOL8 __fastcall sub_1F50270(
        __int64 a1,
        unsigned __int64 *a2,
        _QWORD *a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        char a7)
{
  unsigned int v8; // ebx
  __int64 v9; // r15
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  int v12; // r13d
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // r13d
  __int64 v17; // rax
  unsigned int v19; // ebx
  bool v20; // al
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rbx
  int v25; // r12d
  char *v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // r9d
  unsigned int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // r12
  unsigned __int64 v33; // rax
  unsigned int v34; // ecx
  __int64 v35; // r8
  __int64 v36; // r13
  __int64 v37; // rax
  unsigned __int64 v38; // rbx
  int v39; // eax
  __int64 v40; // rbx
  __int64 *v41; // rdx
  __int64 v42; // r8
  _BYTE *v43; // r9
  __int64 v44; // rcx
  __int16 v45; // ax
  __int64 v46; // rsi
  __int64 v47; // r8
  int v48; // r9d
  __int64 v49; // r13
  __int64 v50; // r14
  __int64 v51; // r15
  unsigned __int8 v52; // al
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rsi
  int v58; // r13d
  __int64 v59; // rbx
  __int64 v60; // rdx
  __int64 v61; // rax
  __int16 v62; // ax
  __int64 v63; // rax
  __int64 v64; // rsi
  unsigned int v65; // edi
  __int64 *v66; // rcx
  __int64 v67; // r9
  int v68; // r12d
  unsigned int v69; // eax
  __int64 v70; // rax
  __int16 v71; // dx
  char v72; // al
  __int64 v73; // rdi
  __int64 (*v74)(); // rax
  unsigned int v75; // eax
  __int64 v76; // rdi
  __int64 v77; // rsi
  _QWORD *v78; // r12
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // r8
  int v82; // r9d
  unsigned int v83; // eax
  __int64 v84; // r8
  __int64 v85; // rdi
  unsigned int v86; // r12d
  __int64 (*v87)(); // rax
  __int64 v88; // r8
  __int64 v89; // rbx
  __int64 *v90; // r13
  __int64 v91; // rdi
  __int64 v92; // rdx
  __int64 v93; // rbx
  __int64 *v94; // r13
  __int64 v95; // rdi
  unsigned int v96; // eax
  __int64 v97; // r13
  __int64 v98; // r8
  unsigned __int64 v99; // r9
  _QWORD *v100; // rdi
  __int64 v101; // rax
  __int64 v102; // r13
  __int64 v103; // rbx
  int v104; // esi
  char v105; // al
  int v106; // r10d
  unsigned int v107; // esi
  __int64 v108; // rdi
  _DWORD *v109; // rdx
  int v110; // ecx
  int v111; // eax
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 v114; // rdi
  _WORD *v115; // r9
  __int16 *v116; // rsi
  unsigned __int16 v117; // cx
  _WORD *v118; // r8
  int v119; // eax
  unsigned __int16 *v120; // rdi
  unsigned int v121; // r8d
  unsigned int i; // edx
  bool v123; // cf
  __int16 *v124; // r8
  __int16 v125; // si
  __int64 v126; // rax
  __int64 v127; // rcx
  int v128; // r9d
  unsigned int v129; // edx
  __int64 v130; // rsi
  __int64 v131; // r12
  unsigned __int64 v132; // rdx
  unsigned int v133; // eax
  __int64 v134; // r8
  __int64 v135; // r13
  __int64 v136; // rax
  unsigned __int64 v137; // rbx
  int v138; // eax
  __int64 v139; // rbx
  __int64 *v140; // rdx
  __int64 v141; // r13
  __int16 v142; // ax
  __int64 v143; // rcx
  __int64 v144; // r8
  _BYTE *v145; // r9
  __int16 v146; // ax
  __int64 v147; // rax
  __int16 v148; // ax
  char v149; // al
  __int16 v150; // ax
  __int64 v151; // rax
  __int64 v152; // rsi
  __int64 v153; // rcx
  int v154; // r9d
  __int64 v155; // r8
  __int64 v156; // r12
  int v157; // ebx
  __int64 v158; // rdx
  __int16 v159; // ax
  __int64 v160; // rax
  __int16 v161; // ax
  char v162; // al
  __int16 v163; // ax
  __int64 v164; // rax
  __int64 v165; // rbx
  int v166; // r13d
  unsigned __int8 v167; // r14
  __int64 v168; // rdx
  __int64 v169; // rcx
  __int64 v170; // r8
  int v171; // r9d
  __int64 v172; // rdx
  __int64 v173; // rbx
  __int64 v174; // rdi
  __int64 v175; // rbx
  __int64 v176; // rdi
  __int64 v177; // rbx
  __int64 v178; // rdi
  __int64 v179; // rbx
  __int64 v180; // rdi
  __int64 v181; // rcx
  char v182; // al
  __int64 v183; // rdx
  unsigned int v184; // r13d
  __int64 v185; // rdx
  char *v186; // rax
  unsigned int v187; // r13d
  __int64 v188; // rcx
  char *v189; // rax
  __int64 v190; // r9
  int v191; // edi
  int v192; // edx
  int v193; // ecx
  int v194; // r10d
  __int64 v195; // rax
  int v196; // edx
  __int64 v197; // rdi
  unsigned int v198; // r8d
  int v199; // esi
  int v200; // r11d
  _DWORD *v201; // rcx
  int v202; // edx
  __int64 v203; // r8
  int v204; // edi
  int v205; // esi
  unsigned __int64 *v206; // rax
  int *v207; // r13
  __int64 v208; // r14
  unsigned __int64 v209; // rdx
  unsigned __int64 v210; // rcx
  __int16 v211; // dx
  __int64 v212; // rdx
  unsigned __int64 *v213; // rcx
  unsigned __int64 *n; // rdx
  unsigned __int64 v215; // r10
  unsigned __int64 v216; // rdi
  char *v217; // rcx
  __int64 v218; // r8
  __int16 v219; // ax
  __int64 v220; // rax
  __int16 v221; // ax
  char v222; // al
  __int16 v223; // ax
  __int64 v224; // rax
  __int64 v225; // r9
  __int64 v226; // r10
  __int64 v227; // rbx
  int *v228; // r9
  unsigned __int64 v229; // r13
  __int64 v230; // r14
  unsigned int v231; // r12d
  char *v232; // rax
  char *v233; // rax
  _BYTE *v234; // rdi
  _BYTE *v235; // r10
  __int64 v236; // rbx
  unsigned int v237; // eax
  __int64 *v238; // rdx
  __int64 *v239; // rcx
  int *v240; // rdx
  int *v241; // rcx
  char *v242; // rsi
  int v243; // r9d
  char *v244; // rcx
  int v245; // edx
  char *v246; // r8
  __int64 v247; // rax
  __int64 v248; // rdi
  __int64 v249; // rax
  int *v250; // rdi
  __int64 v251; // rax
  __int64 v252; // rdx
  int *v253; // r8
  __int64 v254; // rdx
  int *v255; // r8
  __int64 v256; // r14
  int *v257; // r12
  __int64 v258; // rsi
  __int64 v259; // rdx
  bool v260; // al
  __int64 v261; // rsi
  _QWORD *v262; // rcx
  _QWORD *v263; // rax
  __int64 v264; // r8
  __int64 v265; // rdx
  __int64 v266; // rdi
  __int64 v267; // rax
  __int64 v268; // rsi
  _QWORD *v269; // rdx
  _QWORD *v270; // rax
  _BYTE *v271; // rdi
  bool v272; // zf
  __int64 v273; // rbx
  __int64 j; // r12
  int v275; // r13d
  __int64 v276; // rax
  __int64 v277; // rdi
  int *v278; // r8
  __int64 v279; // r8
  unsigned __int64 v280; // r9
  __int16 v281; // r8
  unsigned __int64 v282; // rdi
  unsigned __int64 v283; // rsi
  unsigned __int64 v284; // rdi
  unsigned __int64 v285; // rax
  _QWORD *v286; // rdi
  __int16 v287; // r9
  unsigned __int64 v288; // r8
  __int64 v289; // r8
  unsigned __int64 v290; // r9
  unsigned __int64 v291; // rax
  unsigned __int64 v292; // rdx
  unsigned __int64 v293; // r12
  __int64 v294; // rax
  __int64 v295; // r15
  unsigned __int64 v296; // rbx
  __int64 v297; // r13
  __int64 v298; // r12
  int v299; // r8d
  int v300; // r9d
  int v301; // r12d
  __int16 v302; // cx
  __int64 v303; // rt0
  __int64 v304; // rdx
  __int64 v305; // rax
  unsigned __int64 k; // r12
  __int16 v307; // cx
  unsigned __int64 v308; // r12
  unsigned __int64 v309; // rax
  __int64 *v310; // rax
  unsigned __int64 v311; // rbx
  __int16 v312; // dx
  unsigned __int64 v313; // rdx
  __int16 v314; // si
  unsigned __int64 v315; // rcx
  unsigned __int64 *v316; // rdx
  unsigned __int64 v317; // rsi
  __int64 v318; // rcx
  unsigned __int64 v319; // rax
  unsigned __int64 v320; // rdx
  _QWORD *v321; // rdi
  __int64 v322; // r8
  unsigned __int64 v323; // r9
  __int16 v324; // cx
  unsigned __int64 v325; // [rsp-10h] [rbp-260h]
  __int64 v326; // [rsp+0h] [rbp-250h]
  int *v327; // [rsp+0h] [rbp-250h]
  int *v328; // [rsp+0h] [rbp-250h]
  __int64 v329; // [rsp+10h] [rbp-240h]
  unsigned int v330; // [rsp+18h] [rbp-238h]
  __int64 v331; // [rsp+18h] [rbp-238h]
  unsigned __int64 v332; // [rsp+20h] [rbp-230h]
  char v333; // [rsp+28h] [rbp-228h]
  __int64 m; // [rsp+28h] [rbp-228h]
  int v335; // [rsp+30h] [rbp-220h]
  int v336; // [rsp+38h] [rbp-218h]
  char v337; // [rsp+40h] [rbp-210h]
  unsigned int v338; // [rsp+40h] [rbp-210h]
  unsigned __int8 v339; // [rsp+40h] [rbp-210h]
  int v340; // [rsp+4Ch] [rbp-204h]
  char v341; // [rsp+4Ch] [rbp-204h]
  char v342; // [rsp+4Ch] [rbp-204h]
  unsigned __int8 v343; // [rsp+50h] [rbp-200h]
  int v344; // [rsp+50h] [rbp-200h]
  int v345; // [rsp+50h] [rbp-200h]
  int v346; // [rsp+50h] [rbp-200h]
  int v347; // [rsp+58h] [rbp-1F8h]
  _DWORD *v348; // [rsp+58h] [rbp-1F8h]
  unsigned __int8 v349; // [rsp+58h] [rbp-1F8h]
  __int64 v350; // [rsp+58h] [rbp-1F8h]
  int v351; // [rsp+60h] [rbp-1F0h]
  int v352; // [rsp+60h] [rbp-1F0h]
  int v353; // [rsp+60h] [rbp-1F0h]
  __int64 v354; // [rsp+60h] [rbp-1F0h]
  __int64 *v355; // [rsp+60h] [rbp-1F0h]
  __int64 v356; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 v357; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 v358; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 *v359; // [rsp+68h] [rbp-1E8h]
  __int64 v360; // [rsp+70h] [rbp-1E0h]
  __int64 v361; // [rsp+70h] [rbp-1E0h]
  __int64 v362; // [rsp+70h] [rbp-1E0h]
  __int64 v363; // [rsp+70h] [rbp-1E0h]
  unsigned __int64 v364; // [rsp+70h] [rbp-1E0h]
  unsigned __int64 v365; // [rsp+70h] [rbp-1E0h]
  int v366; // [rsp+78h] [rbp-1D8h]
  unsigned int v367; // [rsp+7Ch] [rbp-1D4h]
  bool v368; // [rsp+80h] [rbp-1D0h]
  __int64 *v369; // [rsp+80h] [rbp-1D0h]
  __int64 *v370; // [rsp+80h] [rbp-1D0h]
  bool v372; // [rsp+90h] [rbp-1C0h]
  __int64 v373; // [rsp+90h] [rbp-1C0h]
  unsigned int v374; // [rsp+90h] [rbp-1C0h]
  unsigned __int64 v375; // [rsp+90h] [rbp-1C0h]
  __int64 v376; // [rsp+90h] [rbp-1C0h]
  int v377; // [rsp+90h] [rbp-1C0h]
  int v378; // [rsp+90h] [rbp-1C0h]
  __int64 v379; // [rsp+98h] [rbp-1B8h]
  __int64 v380; // [rsp+98h] [rbp-1B8h]
  bool v383; // [rsp+AFh] [rbp-1A1h]
  char v384; // [rsp+BBh] [rbp-195h] BYREF
  int v385; // [rsp+BCh] [rbp-194h] BYREF
  _BYTE *v386; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v387; // [rsp+C8h] [rbp-188h]
  _BYTE v388[16]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 *v389; // [rsp+E0h] [rbp-170h] BYREF
  __int64 v390; // [rsp+E8h] [rbp-168h]
  _BYTE v391[16]; // [rsp+F0h] [rbp-160h] BYREF
  int v392; // [rsp+100h] [rbp-150h] BYREF
  __int64 v393; // [rsp+108h] [rbp-148h]
  int *v394; // [rsp+110h] [rbp-140h]
  int *v395; // [rsp+118h] [rbp-138h]
  __int64 v396; // [rsp+120h] [rbp-130h]
  char *v397; // [rsp+130h] [rbp-120h] BYREF
  __int64 v398; // [rsp+138h] [rbp-118h]
  _BYTE v399[16]; // [rsp+140h] [rbp-110h] BYREF
  int v400; // [rsp+150h] [rbp-100h] BYREF
  __int64 v401; // [rsp+158h] [rbp-F8h]
  int *v402; // [rsp+160h] [rbp-F0h]
  int *v403; // [rsp+168h] [rbp-E8h]
  __int64 v404; // [rsp+170h] [rbp-E0h]
  void *dest; // [rsp+180h] [rbp-D0h] BYREF
  __int64 v406; // [rsp+188h] [rbp-C8h]
  _BYTE v407[16]; // [rsp+190h] [rbp-C0h] BYREF
  int v408; // [rsp+1A0h] [rbp-B0h] BYREF
  __int64 v409; // [rsp+1A8h] [rbp-A8h]
  int *v410; // [rsp+1B0h] [rbp-A0h]
  int *v411; // [rsp+1B8h] [rbp-98h]
  __int64 v412; // [rsp+1C0h] [rbp-90h]
  int *v413; // [rsp+1D0h] [rbp-80h] BYREF
  __int64 v414; // [rsp+1D8h] [rbp-78h]
  _BYTE v415[16]; // [rsp+1E0h] [rbp-70h] BYREF
  int v416; // [rsp+1F0h] [rbp-60h] BYREF
  __int64 v417; // [rsp+1F8h] [rbp-58h]
  int *v418; // [rsp+200h] [rbp-50h]
  int *v419; // [rsp+208h] [rbp-48h]
  __int64 v420; // [rsp+210h] [rbp-40h]

  if ( !*(_DWORD *)(a1 + 296) )
    return 0;
  v8 = a4;
  v9 = a1;
  v10 = *a2;
  v360 = 40LL * a5;
  v11 = *(_QWORD *)(*a2 + 32);
  v356 = 40LL * a4;
  v12 = *(_DWORD *)(v11 + v360 + 8);
  v366 = v12;
  v367 = *(_DWORD *)(v11 + v356 + 8);
  v383 = sub_1F4D330(*a2, v367, *(_QWORD *)(a1 + 264), *(_QWORD *)(a1 + 280), 1, a6);
  if ( v12 < 0 )
    sub_1F4E620(a1, v12);
  LODWORD(dest) = v8;
  v14 = *(_QWORD *)(v10 + 16);
  if ( (*(_BYTE *)(v14 + 10) & 0x20) != 0 )
  {
    v15 = *(_QWORD *)(v10 + 32);
    v16 = *(unsigned __int16 *)(v14 + 2);
    v17 = *(unsigned __int8 *)(v14 + 4);
    v340 = *(_DWORD *)(v15 + v360 + 8);
    v351 = *(_DWORD *)(v15 + v356 + 8);
    LODWORD(v413) = v17;
    if ( v16 > (unsigned int)v17 )
    {
      v337 = 0;
      v372 = v383;
      while ( 1 )
      {
        if ( (_DWORD)v17 == v8
          || *(_BYTE *)(*(_QWORD *)(v10 + 32) + 40 * v17)
          || !(*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, void **, int **))(**(_QWORD **)(a1 + 240)
                                                                                           + 176LL))(
                *(_QWORD *)(a1 + 240),
                v10,
                &dest,
                &v413) )
        {
          goto LABEL_76;
        }
        v19 = *(_DWORD *)(*(_QWORD *)(v10 + 32) + 40LL * (unsigned int)v413 + 8);
        v20 = sub_1F4D330(v10, v19, *(_QWORD *)(a1 + 264), *(_QWORD *)(a1 + 280), 0, v13);
        v368 = v20;
        if ( v372 || (LOBYTE(v13) = 0, !v20) )
        {
          v13 = sub_1F4D480(a1, v340, v351, v19, v10, a6);
          if ( !(_BYTE)v13 )
            goto LABEL_76;
        }
        v343 = v13;
        v347 = *(_DWORD *)(*(_QWORD *)(v10 + 32) + 40LL * (unsigned int)v413 + 8);
        if ( !sub_1F3ADB0(*(_QWORD *)(a1 + 240), v10, 0, (int)dest, (int)v413) )
          goto LABEL_76;
        v21 = sub_1F4CA20(v347, a1 + 552);
        v22 = a1 + 552;
        v13 = v343;
        if ( v21 )
          break;
LABEL_16:
        if ( !(_BYTE)v13 )
        {
          v9 = a1;
          goto LABEL_18;
        }
        v351 = v19;
        v337 = v13;
        v372 = v368;
LABEL_76:
        v17 = (unsigned int)((_DWORD)v413 + 1);
        LODWORD(v413) = v17;
        if ( v16 <= (unsigned int)v17 )
        {
          v9 = a1;
          if ( !v337 )
            goto LABEL_79;
LABEL_18:
          if ( (*(_BYTE *)(*(_QWORD *)(v10 + 16) + 10LL) & 0x40) != 0 && !a7 )
          {
            v367 = *(_DWORD *)(*(_QWORD *)(v10 + 32) + v356 + 8);
            v383 = sub_1F4D330(v10, v367, *(_QWORD *)(v9 + 264), *(_QWORD *)(v9 + 280), 1, v13);
            v23 = *(_QWORD *)(v10 + 16);
            v24 = (*(_QWORD *)(v23 + 8) >> 22) & 1LL;
            if ( ((*(_QWORD *)(v23 + 8) >> 22) & 1) != 0 )
            {
LABEL_21:
              if ( !v383 )
                goto LABEL_111;
              v25 = sub_1F4CA20(v367, v9 + 552);
              if ( v25 )
              {
                v111 = sub_1F4CA20(v366, v9 + 584);
                if ( v111 )
                {
                  if ( v25 != v111 )
                  {
                    if ( v25 >= 0 && v111 >= 0 )
                    {
                      v112 = *(_QWORD *)(v9 + 248);
                      v113 = *(_QWORD *)(v112 + 8);
                      v114 = *(_QWORD *)(v112 + 56);
                      v115 = (_WORD *)(v114 + 2LL * (*(_DWORD *)(v113 + 24LL * (unsigned int)v25 + 16) >> 4));
                      v116 = v115 + 1;
                      v117 = *v115 + v25 * (*(_WORD *)(v113 + 24LL * (unsigned int)v25 + 16) & 0xF);
                      LODWORD(v113) = *(_DWORD *)(v113 + 24LL * (unsigned int)v111 + 16);
                      v119 = (v113 & 0xF) * v111;
                      v118 = (_WORD *)(v114 + 2LL * ((unsigned int)v113 >> 4));
                      LOWORD(v119) = *v118 + v119;
                      v120 = v118 + 1;
                      v121 = v117;
                      for ( i = (unsigned __int16)v119; ; i = (unsigned __int16)v119 )
                      {
                        v123 = v121 < i;
                        if ( v121 == i )
                          break;
                        while ( v123 )
                        {
                          v124 = v116 + 1;
                          v125 = *v116;
                          v117 += v125;
                          if ( !v125 )
                            goto LABEL_111;
                          v116 = v124;
                          v121 = v117;
                          v123 = v117 < i;
                          if ( v117 == i )
                            goto LABEL_23;
                        }
                        v192 = *v120;
                        if ( !(_WORD)v192 )
                          goto LABEL_111;
                        v119 += v192;
                        ++v120;
                      }
                      goto LABEL_23;
                    }
LABEL_111:
                    if ( (unsigned __int8)sub_1F4EF20(v9, (__int64 *)a2, a3, v366, v367, a6) )
                      return 1;
                  }
                }
              }
LABEL_23:
              if ( !(_BYTE)v24 )
                goto LABEL_24;
            }
          }
          return 0;
        }
        v8 = (unsigned int)dest;
      }
      v106 = *(_DWORD *)(*(_QWORD *)(v10 + 32) + v360 + 8);
      v107 = *(_DWORD *)(a1 + 576);
      if ( v107 )
      {
        v108 = *(_QWORD *)(a1 + 560);
        v352 = 37 * v106;
        v374 = (v107 - 1) & (37 * v106);
        v109 = (_DWORD *)(v108 + 8LL * v374);
        v110 = *v109;
        if ( v106 == *v109 )
        {
LABEL_108:
          v109[1] = v21;
          goto LABEL_16;
        }
        v344 = 1;
        v348 = 0;
        while ( v110 != -1 )
        {
          if ( v110 == -2 )
          {
            if ( v348 )
              v109 = v348;
            v348 = v109;
          }
          v374 = (v107 - 1) & (v344 + v374);
          v109 = (_DWORD *)(v108 + 8LL * v374);
          v110 = *v109;
          if ( v106 == *v109 )
            goto LABEL_108;
          ++v344;
        }
        if ( v348 )
          v109 = v348;
        v191 = *(_DWORD *)(a1 + 568);
        ++*(_QWORD *)(a1 + 552);
        v377 = v191 + 1;
        if ( 4 * (v191 + 1) < 3 * v107 )
        {
          if ( v107 - *(_DWORD *)(a1 + 572) - v377 > v107 >> 3 )
            goto LABEL_258;
          v335 = v106;
          v336 = v21;
          v339 = v13;
          sub_1392B70(v22, v107);
          v202 = *(_DWORD *)(a1 + 576);
          if ( !v202 )
          {
LABEL_606:
            ++*(_DWORD *)(a1 + 568);
            BUG();
          }
          v106 = v335;
          v201 = 0;
          v346 = v202 - 1;
          LODWORD(v203) = (v202 - 1) & v352;
          v13 = v339;
          v204 = 1;
          v350 = *(_QWORD *)(a1 + 560);
          v109 = (_DWORD *)(v350 + 8LL * (unsigned int)v203);
          v205 = *v109;
          v377 = *(_DWORD *)(a1 + 568) + 1;
          v21 = v336;
          if ( v335 == *v109 )
            goto LABEL_258;
          while ( v205 != -1 )
          {
            if ( v205 == -2 && !v201 )
              v201 = v109;
            v203 = v346 & (unsigned int)(v203 + v204);
            v109 = (_DWORD *)(v350 + 8 * v203);
            v205 = *v109;
            if ( v335 == *v109 )
              goto LABEL_258;
            ++v204;
          }
          goto LABEL_274;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 552);
      }
      v378 = v106;
      v345 = v21;
      v349 = v13;
      sub_1392B70(v22, 2 * v107);
      v196 = *(_DWORD *)(a1 + 576);
      if ( !v196 )
        goto LABEL_606;
      v106 = v378;
      v197 = *(_QWORD *)(a1 + 560);
      v353 = v196 - 1;
      v13 = v349;
      v198 = (v196 - 1) & (37 * v378);
      v109 = (_DWORD *)(v197 + 8LL * v198);
      v199 = *v109;
      v377 = *(_DWORD *)(a1 + 568) + 1;
      v21 = v345;
      if ( v106 == *v109 )
        goto LABEL_258;
      v200 = 1;
      v201 = 0;
      while ( v199 != -1 )
      {
        if ( !v201 && v199 == -2 )
          v201 = v109;
        v198 = v353 & (v200 + v198);
        v109 = (_DWORD *)(v197 + 8LL * v198);
        v199 = *v109;
        if ( v106 == *v109 )
          goto LABEL_258;
        ++v200;
      }
LABEL_274:
      if ( v201 )
        v109 = v201;
LABEL_258:
      *(_DWORD *)(a1 + 568) = v377;
      if ( *v109 != -1 )
        --*(_DWORD *)(a1 + 572);
      *v109 = v106;
      v109[1] = 0;
      goto LABEL_108;
    }
  }
LABEL_79:
  if ( a7 )
    return 0;
  LOBYTE(v24) = byte_4FCE3E0;
  if ( !byte_4FCE3E0 )
  {
    v70 = *(_QWORD *)(v10 + 16);
    if ( (*(_BYTE *)(v70 + 10) & 0x40) != 0 )
      goto LABEL_21;
LABEL_82:
    if ( *(_WORD *)v70 != 1 || (*(_BYTE *)(*(_QWORD *)(v10 + 32) + 64LL) & 8) == 0 )
    {
      v71 = *(_WORD *)(v10 + 46);
      if ( (v71 & 4) == 0 && (v71 & 8) != 0 )
        v72 = sub_1E15D00(v10, 0x10000u, 1);
      else
        v72 = WORD1(*(_QWORD *)(v70 + 8)) & 1;
      if ( !v72 )
        return 0;
    }
    if ( v383 )
      return 0;
    v73 = *(_QWORD *)(v9 + 240);
    v74 = *(__int64 (**)())(*(_QWORD *)v73 + 568LL);
    if ( v74 == sub_1E1C870 )
      return v383;
    v75 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64 **))v74)(
            v73,
            **(unsigned __int16 **)(v10 + 16),
            1,
            0,
            &v389);
    if ( !v75 )
      return v383;
    v76 = *(_QWORD *)(v9 + 240);
    v77 = *(_QWORD *)(v76 + 8) + ((unsigned __int64)v75 << 6);
    if ( *(_BYTE *)(v77 + 4) != 1 )
      return v383;
    v78 = *(_QWORD **)(v9 + 248);
    v79 = (_QWORD *)sub_1F3AD60(v76, v77, (unsigned int)v389, v78, *(_QWORD *)(v9 + 232));
    v80 = sub_1F4AAF0((__int64)v78, v79);
    v83 = sub_1E6B9A0(*(_QWORD *)(v9 + 264), (__int64)v80, (unsigned __int8 *)byte_3F871B3, 0, v81, v82);
    v85 = *(_QWORD *)(v9 + 240);
    v86 = v83;
    dest = v407;
    v406 = 0x200000000LL;
    v87 = *(__int64 (**)())(*(_QWORD *)v85 + 552LL);
    if ( v87 == sub_1E1C860 )
      return v383;
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, unsigned __int64, _QWORD, __int64, _QWORD, void **, __int64))v87)(
           v85,
           *(_QWORD *)(v9 + 232),
           v10,
           v86,
           1,
           0,
           &dest,
           v84) )
    {
      sub_1E1AFE0(*((_QWORD *)dest + 1), v86, *(_QWORD **)(v9 + 248), 0, v88, v325);
      v89 = *(_QWORD *)dest;
      v90 = (__int64 *)*a2;
      sub_1DD5BA0((__int64 *)(*(_QWORD *)(v9 + 304) + 16LL), *(_QWORD *)dest);
      v91 = *(_QWORD *)v89;
      v92 = *v90;
      *(_QWORD *)(v89 + 8) = v90;
      v92 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v89 = v92 | v91 & 7;
      *(_QWORD *)(v92 + 8) = v89;
      *v90 = v89 | *v90 & 7;
      v93 = *((_QWORD *)dest + 1);
      v94 = (__int64 *)*a2;
      sub_1DD5BA0((__int64 *)(*(_QWORD *)(v9 + 304) + 16LL), v93);
      v95 = *v94;
      *(_QWORD *)(v93 + 8) = v94;
      *(_QWORD *)v93 = v95 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v93 & 7LL;
      *(_QWORD *)((v95 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v93;
      *v94 = v93 | *v94 & 7;
      LODWORD(v93) = sub_1E16810(*((_QWORD *)dest + 1), v366, 0, 0, 0);
      v96 = sub_1E165A0(*((_QWORD *)dest + 1), v367, 0, 0);
      v97 = v96;
      v397 = (char *)*((_QWORD *)dest + 1);
      sub_1F50270(v9, (unsigned int)&v397, (_DWORD)a2, v96, v93, a6, 1);
      v100 = dest;
      if ( (((*(_BYTE *)(*(_QWORD *)(*((_QWORD *)dest + 1) + 32LL) + 40 * v97 + 3) & 0x40) != 0)
          & ((*(_BYTE *)(*(_QWORD *)(*((_QWORD *)dest + 1) + 32LL) + 40 * v97 + 3) >> 4)
           ^ 1)) != 0 )
      {
        if ( *(_QWORD *)(v9 + 272) )
        {
          v101 = *(unsigned int *)(v10 + 40);
          if ( (_DWORD)v101 )
          {
            v102 = 0;
            v380 = 40 * v101;
            do
            {
              v103 = v102 + *(_QWORD *)(v10 + 32);
              if ( !*(_BYTE *)v103 )
              {
                v104 = *(_DWORD *)(v103 + 8);
                if ( v104 < 0 )
                {
                  v105 = *(_BYTE *)(v103 + 3);
                  if ( (v105 & 0x10) != 0 )
                  {
                    if ( (unsigned __int8)sub_1F4E120(*(char **)(v9 + 272), v104, v10) )
                    {
                      if ( (unsigned int)sub_1E16810(*((_QWORD *)dest + 1), *(_DWORD *)(v103 + 8), 1, 0, 0) == -1 )
                        sub_1F4E310(*(_QWORD *)(v9 + 272), *(_DWORD *)(v103 + 8), *(_QWORD *)dest, 0, v279, v280);
                      else
                        sub_1F4E310(*(_QWORD *)(v9 + 272), *(_DWORD *)(v103 + 8), *((_QWORD *)dest + 1), 0, v279, v280);
                    }
                  }
                  else if ( (v105 & 0x40) != 0 )
                  {
                    if ( (unsigned int)sub_1E165A0(*(_QWORD *)dest, v104, 1, 0) == -1 )
                      sub_1DCCCA0(*(char **)(v9 + 272), *(_DWORD *)(v103 + 8), v10, *((_QWORD *)dest + 1));
                    else
                      sub_1DCCCA0(*(char **)(v9 + 272), *(_DWORD *)(v103 + 8), v10, *(_QWORD *)dest);
                  }
                }
              }
              v102 += 40;
            }
            while ( v102 != v380 );
            v100 = dest;
          }
          sub_1F4E280(*(_QWORD *)(v9 + 272), v86, v100[1], 0, v98, v99);
        }
        v272 = *(_QWORD *)(v9 + 280) == 0;
        v413 = (int *)v415;
        v414 = 0x400000000LL;
        if ( !v272 )
        {
          v273 = *(_QWORD *)(v10 + 32);
          for ( j = v273 + 40LL * *(unsigned int *)(v10 + 40); j != v273; v273 += 40 )
          {
            if ( !*(_BYTE *)v273 )
            {
              v275 = *(_DWORD *)(v273 + 8);
              v276 = (unsigned int)v414;
              if ( (unsigned int)v414 >= HIDWORD(v414) )
              {
                sub_16CD150((__int64)&v413, v415, 0, 4, v98, v99);
                v276 = (unsigned int)v414;
              }
              v413[v276] = v275;
              LODWORD(v414) = v414 + 1;
            }
          }
        }
        sub_1E16240(v10);
        v277 = *(_QWORD *)(v9 + 280);
        if ( v277 )
          sub_1DBF6C0(v277, *(_QWORD *)(v9 + 304), *(_QWORD *)dest, *((_QWORD *)dest + 1), v413, (unsigned int)v414);
        v271 = dest;
        v278 = v413;
        *a2 = *((_QWORD *)dest + 1);
        if ( v278 != (int *)v415 )
        {
          _libc_free((unsigned __int64)v278);
          v271 = dest;
        }
        goto LABEL_473;
      }
      sub_1E16240(*(_QWORD *)dest);
      sub_1E16240(*((_QWORD *)dest + 1));
    }
    v271 = dest;
LABEL_473:
    if ( v271 != v407 )
      _libc_free((unsigned __int64)v271);
    return v383;
  }
  v26 = *(char **)(v9 + 272);
  if ( !v26 && !*(_QWORD *)(v9 + 280) )
    goto LABEL_240;
  v126 = *(unsigned int *)(v9 + 336);
  if ( !(_DWORD)v126 )
    goto LABEL_240;
  v127 = *(_QWORD *)(v9 + 320);
  v128 = 1;
  v375 = *a2;
  v129 = (v126 - 1) & (((unsigned int)v375 >> 9) ^ ((unsigned int)v375 >> 4));
  v370 = (__int64 *)(v127 + 16LL * v129);
  v130 = *v370;
  if ( *a2 != *v370 )
  {
    while ( v130 != -8 )
    {
      v129 = (v126 - 1) & (v128 + v129);
      v370 = (__int64 *)(v127 + 16LL * v129);
      v130 = *v370;
      if ( v375 == *v370 )
        goto LABEL_127;
      ++v128;
    }
    goto LABEL_240;
  }
LABEL_127:
  if ( v370 == (__int64 *)(v127 + 16 * v126) )
  {
LABEL_240:
    LOBYTE(v24) = 0;
    if ( (*(_BYTE *)(*(_QWORD *)(v10 + 16) + 10LL) & 0x40) == 0 )
      goto LABEL_26;
    goto LABEL_21;
  }
  v131 = *(_QWORD *)(v9 + 280);
  if ( v131 )
  {
    v132 = *(unsigned int *)(v131 + 408);
    v133 = v367 & 0x7FFFFFFF;
    v134 = v367 & 0x7FFFFFFF;
    if ( (v367 & 0x7FFFFFFF) < (unsigned int)v132 )
    {
      v135 = *(_QWORD *)(*(_QWORD *)(v131 + 400) + 8LL * v133);
      if ( v135 )
      {
LABEL_131:
        v136 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v131 + 272) + 392LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(v9 + 304) + 48LL)
                         + 8);
        v137 = v136 & 0xFFFFFFFFFFFFFFF8LL;
        v138 = (v136 >> 1) & 3;
        if ( v138 )
          v139 = (2LL * (v138 - 1)) | v137;
        else
          v139 = *(_QWORD *)v137 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v140 = (__int64 *)sub_1DB3C70((__int64 *)v135, v139);
        if ( v140 != (__int64 *)(*(_QWORD *)v135 + 24LL * *(unsigned int *)(v135 + 8))
          && (*(_DWORD *)((*v140 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v140 >> 1) & 3)) < (*(_DWORD *)((v139 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v139 >> 1) & 3)
          || (*(v140 - 2) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          goto LABEL_211;
        }
        v141 = *(_QWORD *)((*(v140 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 16);
        goto LABEL_137;
      }
    }
    v187 = v133 + 1;
    if ( (unsigned int)v132 < v133 + 1 )
    {
      v195 = v187;
      if ( v187 < v132 )
      {
        *(_DWORD *)(v131 + 408) = v187;
        v188 = *(_QWORD *)(v131 + 400);
        goto LABEL_246;
      }
      if ( v187 > v132 )
      {
        if ( v187 > (unsigned __int64)*(unsigned int *)(v131 + 412) )
        {
          sub_16CD150(v131 + 400, (const void *)(v131 + 416), v187, 8, v134, v128);
          v134 = v367 & 0x7FFFFFFF;
          v195 = v187;
        }
        v188 = *(_QWORD *)(v131 + 400);
        v268 = *(_QWORD *)(v131 + 416);
        v269 = (_QWORD *)(v188 + 8 * v195);
        v270 = (_QWORD *)(v188 + 8LL * *(unsigned int *)(v131 + 408));
        if ( v269 != v270 )
        {
          do
            *v270++ = v268;
          while ( v269 != v270 );
          v188 = *(_QWORD *)(v131 + 400);
        }
        *(_DWORD *)(v131 + 408) = v187;
        goto LABEL_246;
      }
    }
    v188 = *(_QWORD *)(v131 + 400);
LABEL_246:
    v363 = v134;
    *(_QWORD *)(v188 + 8LL * (v367 & 0x7FFFFFFF)) = sub_1DBA290(v367);
    v135 = *(_QWORD *)(*(_QWORD *)(v131 + 400) + 8 * v363);
    sub_1DBB110((_QWORD *)v131, v135);
    v131 = *(_QWORD *)(v9 + 280);
    goto LABEL_131;
  }
  v189 = sub_1DCC790(v26, v367);
  v141 = sub_1DCB3F0((__int64)v189, *(_QWORD *)(v9 + 304));
LABEL_137:
  if ( !v141 )
    goto LABEL_211;
  if ( v375 == v141 )
    goto LABEL_211;
  v142 = **(_WORD **)(v141 + 16);
  if ( v142 == 15 || v142 == 10 || sub_1E17880(v141) )
    goto LABEL_211;
  v146 = *(_WORD *)(v141 + 46);
  if ( (v146 & 4) != 0 || (v146 & 8) == 0 )
    v147 = (*(_QWORD *)(*(_QWORD *)(v141 + 16) + 8LL) >> 4) & 1LL;
  else
    LOBYTE(v147) = sub_1E15D00(v141, 0x10u, 1);
  if ( (_BYTE)v147 )
    goto LABEL_211;
  v148 = *(_WORD *)(v141 + 46);
  if ( (v148 & 4) != 0 || (v148 & 8) == 0 )
    v149 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v141 + 16) + 8LL) >> 7;
  else
    v149 = sub_1E15D00(v141, 0x80u, 1);
  if ( v149 )
    goto LABEL_211;
  v150 = *(_WORD *)(v141 + 46);
  if ( (v150 & 4) != 0 || (v150 & 8) == 0 )
    v151 = (*(_QWORD *)(*(_QWORD *)(v141 + 16) + 8LL) >> 6) & 1LL;
  else
    LOBYTE(v151) = sub_1E15D00(v141, 0x40u, 1);
  if ( (_BYTE)v151 )
    goto LABEL_211;
  if ( (unsigned __int8)sub_1F4C460(v141, v367, &v386, v143, v144, v145) )
    goto LABEL_211;
  v152 = *(_QWORD *)(v9 + 288);
  LOBYTE(v385) = 1;
  v342 = sub_1E17B50(v375, v152, &v385);
  if ( !v342
    || (*(unsigned int (__fastcall **)(_QWORD, _QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(v9 + 240) + 848LL))(
         *(_QWORD *)(v9 + 240),
         *(_QWORD *)(v9 + 256),
         v375,
         0) > 1 )
  {
    goto LABEL_211;
  }
  v397 = v399;
  dest = v407;
  v398 = 0x200000000LL;
  v406 = 0x200000000LL;
  v413 = (int *)v415;
  v414 = 0x200000000LL;
  v155 = *(_QWORD *)(v375 + 32);
  v156 = v155;
  v362 = v155 + 40LL * *(unsigned int *)(v375 + 40);
  while ( v362 != v156 )
  {
    if ( !*(_BYTE *)v156 )
    {
      v157 = *(_DWORD *)(v156 + 8);
      if ( v157 )
      {
        if ( (*(_BYTE *)(v156 + 3) & 0x10) != 0 )
        {
          if ( (unsigned int)v414 >= HIDWORD(v414) )
            sub_16CD150((__int64)&v413, v415, 0, 4, v155, v154);
          v413[(unsigned int)v414] = v157;
          LODWORD(v414) = v414 + 1;
        }
        else
        {
          if ( (unsigned int)v398 >= HIDWORD(v398) )
            sub_16CD150((__int64)&v397, v399, 0, 4, v155, v154);
          *(_DWORD *)&v397[4 * (unsigned int)v398] = v157;
          LODWORD(v398) = v398 + 1;
          if ( v367 != v157
            && ((((*(_BYTE *)(v156 + 3) & 0x40) != 0) & ((*(_BYTE *)(v156 + 3) >> 4) ^ 1)) != 0
             || (v158 = *(_QWORD *)(v9 + 280)) != 0 && sub_1F4D060(v375, v157, v158, v153, v155, v154)) )
          {
            if ( (unsigned int)v406 >= HIDWORD(v406) )
              sub_16CD150((__int64)&dest, v407, 0, 4, v155, v154);
            *((_DWORD *)dest + (unsigned int)v406) = v157;
            LODWORD(v406) = v406 + 1;
          }
        }
      }
    }
    v156 += 40;
  }
  v291 = v375;
  v389 = (__int64 *)v375;
  v292 = v375;
  if ( (*(_BYTE *)v375 & 4) == 0 )
  {
    do
    {
      v302 = *(_WORD *)(v291 + 46);
      v292 = v291;
      v291 = *(_QWORD *)(v291 + 8);
    }
    while ( (v302 & 8) != 0 );
  }
  v293 = *(_QWORD *)(v292 + 8);
  if ( **(_WORD **)(v293 + 16) == 15 )
  {
    v294 = v9;
    v295 = v141;
    v364 = *(_QWORD *)(v292 + 8);
    v296 = v364;
    v297 = v294;
    do
    {
      v298 = *(_QWORD *)(v296 + 32);
      if ( !(unsigned __int8)sub_1F4C920((__int64)&v413, *(_DWORD *)(v298 + 48), *(_QWORD *)(v297 + 248)) )
        break;
      v301 = *(_DWORD *)(v298 + 8);
      if ( (unsigned int)v414 >= HIDWORD(v414) )
        sub_16CD150((__int64)&v413, v415, 0, 4, v299, v300);
      v413[(unsigned int)v414] = v301;
      LODWORD(v414) = v414 + 1;
      if ( (*(_BYTE *)v296 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v296 + 46) & 8) != 0 )
          v296 = *(_QWORD *)(v296 + 8);
      }
      v296 = *(_QWORD *)(v296 + 8);
    }
    while ( **(_WORD **)(v296 + 16) == 15 );
    v359 = (unsigned __int64 *)v296;
    v303 = v295;
    v9 = v297;
    v141 = v303;
    v293 = v364;
  }
  else
  {
    v359 = *(unsigned __int64 **)(v292 + 8);
  }
  v304 = v141;
  v305 = v141;
  if ( (*(_BYTE *)v141 & 4) == 0 )
  {
    do
    {
      v307 = *(_WORD *)(v305 + 46);
      v304 = v305;
      v305 = *(_QWORD *)(v305 + 8);
    }
    while ( (v307 & 8) != 0 );
  }
  v338 = 0;
  v365 = v10;
  v355 = *(__int64 **)(v304 + 8);
  v331 = v141;
  v332 = v293;
  for ( k = (unsigned __int64)v359; (__int64 *)k != v355; k = *(_QWORD *)(k + 8) )
  {
    if ( (unsigned __int16)(**(_WORD **)(k + 16) - 12) > 1u )
    {
      if ( v338 > 0xA
        || (++v338, sub_1E17880(k))
        || ((v159 = *(_WORD *)(k + 46), (v159 & 4) != 0) || (v159 & 8) == 0
          ? (v160 = (*(_QWORD *)(*(_QWORD *)(k + 16) + 8LL) >> 4) & 1LL)
          : (LOBYTE(v160) = sub_1E15D00(k, 0x10u, 1)),
            (_BYTE)v160
         || ((v161 = *(_WORD *)(k + 46), (v161 & 4) != 0) || (v161 & 8) == 0
           ? (v162 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(k + 16) + 8LL) >> 7)
           : (v162 = sub_1E15D00(k, 0x80u, 1)),
             v162
          || ((v163 = *(_WORD *)(k + 46), (v163 & 4) != 0) || (v163 & 8) == 0
            ? (v164 = (*(_QWORD *)(*(_QWORD *)(k + 16) + 8LL) >> 6) & 1LL)
            : (LOBYTE(v164) = sub_1E15D00(k, 0x40u, 1)),
              (_BYTE)v164))) )
      {
LABEL_203:
        v342 = 0;
        v10 = v365;
        goto LABEL_204;
      }
      v165 = *(_QWORD *)(k + 32);
      for ( m = v165 + 40LL * *(unsigned int *)(k + 40); m != v165; v165 += 40 )
      {
        if ( *(_BYTE *)v165 )
          continue;
        v166 = *(_DWORD *)(v165 + 8);
        if ( !v166 )
          continue;
        v167 = *(_BYTE *)(v165 + 3);
        v168 = *(_QWORD *)(v9 + 248);
        if ( (v167 & 0x10) != 0 )
        {
          v329 = *(_QWORD *)(v9 + 248);
          if ( (unsigned __int8)sub_1F4C920((__int64)&v397, v166, v168)
            || (((v167 & 0x10) != 0) & (v167 >> 6)) == 0 && (unsigned __int8)sub_1F4C920((__int64)&v413, v166, v329) )
          {
            goto LABEL_203;
          }
        }
        else
        {
          if ( (unsigned __int8)sub_1F4C920((__int64)&v413, v166, v168) )
            goto LABEL_203;
          if ( (v167 & 0x40) != 0 || (v172 = *(_QWORD *)(v9 + 280)) != 0 && sub_1F4D060(k, v166, v172, v169, v170, v171) )
          {
            if ( v367 == v166 )
              continue;
            if ( (unsigned __int8)sub_1F4C920((__int64)&v397, v166, *(_QWORD *)(v9 + 248)) )
              goto LABEL_203;
          }
          else if ( v367 == v166 )
          {
            goto LABEL_203;
          }
          if ( (unsigned __int8)sub_1F4C920((__int64)&dest, v166, *(_QWORD *)(v9 + 248)) )
            goto LABEL_203;
        }
      }
    }
    if ( (*(_BYTE *)k & 4) == 0 )
    {
      while ( (*(_BYTE *)(k + 46) & 8) != 0 )
        k = *(_QWORD *)(k + 8);
    }
  }
  v308 = v332;
  v10 = v365;
  while ( *(__int64 **)(*(_QWORD *)(v9 + 304) + 32LL) != v389 )
  {
    if ( (*v389 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v309 = *v389 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)v309 & 4) != 0 )
        goto LABEL_566;
    }
    else
    {
      v309 = 0;
    }
    while ( (*(_BYTE *)(v309 + 46) & 4) != 0 )
      v309 = *(_QWORD *)v309 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_566:
    if ( (unsigned __int16)(**(_WORD **)(v309 + 16) - 12) > 1u )
      break;
    sub_1F4E0E0((unsigned __int64 *)&v389);
  }
  *a3 = v359;
  if ( !*(_QWORD *)(v9 + 280) )
    goto LABEL_591;
  v310 = v355;
  while ( 2 )
  {
    if ( (unsigned __int64 *)v308 != v359 )
    {
      v311 = 0;
      if ( !v308 )
        goto LABEL_579;
      if ( (*(_BYTE *)v308 & 4) != 0 )
      {
        v311 = *(_QWORD *)(v308 + 8);
        v315 = v308;
        if ( (__int64 *)v308 != v310 )
          goto LABEL_584;
      }
      else
      {
        v311 = v308;
        do
        {
LABEL_579:
          v312 = *(_WORD *)(v311 + 46);
          v311 = *(_QWORD *)(v311 + 8);
        }
        while ( (v312 & 8) != 0 );
        if ( (__int64 *)v308 != v310 )
        {
          if ( !v308 )
          {
            v313 = 0;
            goto LABEL_583;
          }
          v313 = v308;
          if ( (*(_BYTE *)v308 & 4) != 0 )
          {
            v315 = v308;
          }
          else
          {
            do
            {
LABEL_583:
              v314 = *(_WORD *)(v313 + 46);
              v315 = v313;
              v313 = *(_QWORD *)(v313 + 8);
            }
            while ( (v314 & 8) != 0 );
          }
LABEL_584:
          v316 = *(unsigned __int64 **)(v315 + 8);
          if ( (unsigned __int64 *)v308 != v316 && v310 != (__int64 *)v316 && v316 != (unsigned __int64 *)v308 )
          {
            v317 = *v316 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*(_QWORD *)v308 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v316;
            *v316 = *v316 & 7 | *(_QWORD *)v308 & 0xFFFFFFFFFFFFFFF8LL;
            v318 = *v310;
            *(_QWORD *)(v317 + 8) = v310;
            v318 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v308 = v318 | *(_QWORD *)v308 & 7LL;
            *(_QWORD *)(v318 + 8) = v308;
            *v310 = v317 | *v310 & 7;
          }
        }
      }
      sub_1DC1A70(*(_QWORD **)(v9 + 280), v308, 0);
      v310 = (__int64 *)v308;
      v308 = v311;
      continue;
    }
    break;
  }
  v355 = v310;
  v319 = v375;
  v320 = v375;
  if ( (*(_BYTE *)v375 & 4) == 0 )
  {
    do
    {
      v324 = *(_WORD *)(v319 + 46);
      v320 = v319;
      v319 = *(_QWORD *)(v319 + 8);
    }
    while ( (v324 & 8) != 0 );
  }
  v359 = *(unsigned __int64 **)(v320 + 8);
LABEL_591:
  sub_1F4D850(*(_QWORD *)(v9 + 304), v355, *(_QWORD *)(v9 + 304), v389, v359);
  *v370 = -16;
  v321 = *(_QWORD **)(v9 + 280);
  --*(_DWORD *)(v9 + 328);
  ++*(_DWORD *)(v9 + 332);
  if ( v321 )
  {
    sub_1DC1A70(v321, v375, 0);
  }
  else
  {
    sub_1F4E1D0(*(char **)(v9 + 272), v367, v331);
    sub_1F4E280(*(_QWORD *)(v9 + 272), v367, v375, 0, v322, v323);
  }
LABEL_204:
  if ( v413 != (int *)v415 )
    _libc_free((unsigned __int64)v413);
  if ( dest != v407 )
    _libc_free((unsigned __int64)dest);
  if ( v397 != v399 )
    _libc_free((unsigned __int64)v397);
  if ( v342 )
    return 1;
LABEL_211:
  if ( (*(_BYTE *)(*(_QWORD *)(v10 + 16) + 10LL) & 0x40) != 0 )
  {
    LOBYTE(v24) = 0;
    goto LABEL_21;
  }
LABEL_24:
  if ( !byte_4FCE3E0 )
    goto LABEL_110;
  v26 = *(char **)(v9 + 272);
LABEL_26:
  if ( !v26 && !*(_QWORD *)(v9 + 280) )
    goto LABEL_110;
  v27 = *(unsigned int *)(v9 + 336);
  if ( !(_DWORD)v27 )
    goto LABEL_110;
  v28 = *(_QWORD *)(v9 + 320);
  v29 = 1;
  v379 = *a2;
  v30 = (v27 - 1) & (((unsigned int)v379 >> 9) ^ ((unsigned int)v379 >> 4));
  v369 = (__int64 *)(v28 + 16LL * v30);
  v31 = *v369;
  if ( *a2 != *v369 )
  {
    while ( 1 )
    {
      if ( v31 == -8 )
        goto LABEL_110;
      v30 = (v27 - 1) & (v29 + v30);
      v369 = (__int64 *)(v28 + 16LL * v30);
      v31 = *v369;
      if ( v379 == *v369 )
        break;
      ++v29;
    }
  }
  if ( v369 == (__int64 *)(v28 + 16 * v27) )
    goto LABEL_110;
  v32 = *(_QWORD *)(v9 + 280);
  if ( v32 )
  {
    v33 = *(unsigned int *)(v32 + 408);
    v34 = v367 & 0x7FFFFFFF;
    v35 = v367 & 0x7FFFFFFF;
    if ( (v367 & 0x7FFFFFFF) < (unsigned int)v33 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)(v32 + 400) + 8LL * v34);
      if ( v36 )
        goto LABEL_33;
    }
    v184 = v34 + 1;
    if ( (unsigned int)v33 < v34 + 1 )
    {
      v190 = v184;
      if ( v184 >= v33 )
      {
        if ( v184 <= v33 )
          goto LABEL_237;
        if ( v184 > (unsigned __int64)*(unsigned int *)(v32 + 412) )
        {
          sub_16CD150(v32 + 400, (const void *)(v32 + 416), v184, 8, v35, v184);
          v33 = *(unsigned int *)(v32 + 408);
          v35 = v367 & 0x7FFFFFFF;
          v190 = v184;
        }
        v185 = *(_QWORD *)(v32 + 400);
        v261 = *(_QWORD *)(v32 + 416);
        v262 = (_QWORD *)(v185 + 8 * v190);
        v263 = (_QWORD *)(v185 + 8 * v33);
        if ( v262 != v263 )
        {
          do
            *v263++ = v261;
          while ( v262 != v263 );
          v185 = *(_QWORD *)(v32 + 400);
        }
        *(_DWORD *)(v32 + 408) = v184;
      }
      else
      {
        *(_DWORD *)(v32 + 408) = v184;
        v185 = *(_QWORD *)(v32 + 400);
      }
    }
    else
    {
LABEL_237:
      v185 = *(_QWORD *)(v32 + 400);
    }
    v376 = v35;
    *(_QWORD *)(v185 + 8LL * (v367 & 0x7FFFFFFF)) = sub_1DBA290(v367);
    v36 = *(_QWORD *)(*(_QWORD *)(v32 + 400) + 8 * v376);
    sub_1DBB110((_QWORD *)v32, v36);
    v32 = *(_QWORD *)(v9 + 280);
LABEL_33:
    v37 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v32 + 272) + 392LL)
                    + 16LL * *(unsigned int *)(*(_QWORD *)(v9 + 304) + 48LL)
                    + 8);
    v38 = v37 & 0xFFFFFFFFFFFFFFF8LL;
    v39 = (v37 >> 1) & 3;
    if ( v39 )
      v40 = (2LL * (v39 - 1)) | v38;
    else
      v40 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v41 = (__int64 *)sub_1DB3C70((__int64 *)v36, v40);
    v44 = 3LL * *(unsigned int *)(v36 + 8);
    if ( v41 != (__int64 *)(*(_QWORD *)v36 + 24LL * *(unsigned int *)(v36 + 8)) )
    {
      v44 = *(_DWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v41 >> 1) & 3);
      if ( (unsigned int)v44 < (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v40 >> 1) & 3) )
        goto LABEL_110;
    }
    if ( (*(v41 - 2) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_110;
    v373 = *(_QWORD *)((*(v41 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 16);
  }
  else
  {
    v186 = sub_1DCC790(v26, v367);
    v373 = sub_1DCB3F0((__int64)v186, *(_QWORD *)(v9 + 304));
  }
  if ( !v373 )
    goto LABEL_110;
  if ( v379 == v373 )
    goto LABEL_110;
  v45 = **(_WORD **)(v373 + 16);
  if ( v45 == 15 )
    goto LABEL_110;
  if ( v45 == 10 )
    goto LABEL_110;
  v341 = sub_1F4C460(v373, v367, &v385, v44, v42, v43);
  if ( v341 )
    goto LABEL_110;
  v46 = *(_QWORD *)(v9 + 288);
  v384 = 1;
  v333 = sub_1E17B50(v373, v46, &v384);
  if ( !v333 )
    goto LABEL_110;
  v392 = 0;
  v394 = &v392;
  v395 = &v392;
  v397 = v399;
  v402 = &v400;
  v403 = &v400;
  v389 = (__int64 *)v391;
  dest = v407;
  v390 = 0x200000000LL;
  v398 = 0x200000000LL;
  v406 = 0x200000000LL;
  v410 = &v408;
  v411 = &v408;
  v414 = 0x200000000LL;
  v413 = (int *)v415;
  v393 = 0;
  v396 = 0;
  v400 = 0;
  v401 = 0;
  v404 = 0;
  v408 = 0;
  v409 = 0;
  v412 = 0;
  v416 = 0;
  v417 = 0;
  v418 = &v416;
  v419 = &v416;
  v420 = 0;
  v49 = *(_QWORD *)(v373 + 32);
  v361 = v49 + 40LL * *(unsigned int *)(v373 + 40);
  if ( v49 != v361 )
  {
    v357 = v10;
    v50 = v9;
    v51 = *(_QWORD *)(v373 + 32);
    while ( 1 )
    {
      if ( !*(_BYTE *)v51 )
      {
        v54 = *(unsigned int *)(v51 + 8);
        LODWORD(v386) = *(_DWORD *)(v51 + 8);
        v52 = *(_BYTE *)(v51 + 3);
        v53 = v52 & 0x10;
        if ( (v52 & 0x10) != 0 )
        {
          if ( (int)v54 > 0 )
          {
            sub_1B94BE0((__int64)&dest, (unsigned int *)&v386, v53, v54, v47);
            v183 = *(unsigned __int8 *)(v51 + 3);
            v182 = (unsigned __int8)v183 >> 4;
            LOBYTE(v183) = (unsigned __int8)v183 >> 6;
            if ( (v182 & 1 & (unsigned __int8)v183) == 0 )
              sub_1B94BE0((__int64)&v413, (unsigned int *)&v386, v183, v181, v47);
          }
          goto LABEL_53;
        }
        if ( (_DWORD)v54 )
        {
          v57 = *(_QWORD *)(v50 + 264);
          v58 = *((_DWORD *)v369 + 2);
          if ( (int)v54 < 0 )
          {
            v54 = *(_QWORD *)(v57 + 24) + 16 * (v54 & 0x7FFFFFFF);
            v59 = *(_QWORD *)(v54 + 8);
          }
          else
          {
            v59 = *(_QWORD *)(*(_QWORD *)(v57 + 272) + 8 * v54);
          }
          if ( v59 )
          {
            v53 = *(_BYTE *)(v59 + 3) & 0x10;
            if ( (*(_BYTE *)(v59 + 3) & 0x10) != 0
              || (v59 = *(_QWORD *)(v59 + 32)) != 0
              && (v53 = *(_BYTE *)(v59 + 3) & 0x10, (*(_BYTE *)(v59 + 3) & 0x10) != 0) )
            {
              v60 = *(_QWORD *)(v59 + 16);
              if ( *(_QWORD *)(v60 + 24) != *(_QWORD *)(v50 + 304) )
                goto LABEL_62;
LABEL_67:
              v62 = **(_WORD **)(v60 + 16);
              if ( v62 == 15 || v62 == 10 )
                goto LABEL_62;
              if ( v60 == v379 )
                goto LABEL_214;
              v63 = *(unsigned int *)(v50 + 336);
              if ( !(_DWORD)v63 )
                goto LABEL_214;
              v64 = *(_QWORD *)(v50 + 320);
              v65 = (v63 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v66 = (__int64 *)(v64 + 16LL * v65);
              v67 = *v66;
              if ( v60 != *v66 )
              {
                v193 = 1;
                while ( v67 != -8 )
                {
                  v194 = v193 + 1;
                  v65 = (v63 - 1) & (v65 + v193);
                  v66 = (__int64 *)(v64 + 16LL * v65);
                  v67 = *v66;
                  if ( v60 == *v66 )
                    goto LABEL_72;
                  v193 = v194;
                }
LABEL_214:
                v9 = v50;
                v10 = v357;
                goto LABEL_215;
              }
LABEL_72:
              if ( v66 == (__int64 *)(v64 + 16 * v63) )
                goto LABEL_214;
              v68 = *((_DWORD *)v66 + 2);
              v69 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD))(**(_QWORD **)(v50 + 240) + 848LL))(
                      *(_QWORD *)(v50 + 240),
                      *(_QWORD *)(v50 + 256),
                      v60,
                      0);
              v47 = v69;
              if ( v69 > v58 - v68 )
                goto LABEL_214;
              v61 = *(_QWORD *)(v59 + 16);
              while ( 1 )
              {
                v59 = *(_QWORD *)(v59 + 32);
                if ( !v59 || (*(_BYTE *)(v59 + 3) & 0x10) == 0 )
                  break;
                v60 = *(_QWORD *)(v59 + 16);
                if ( v60 != v61 )
                {
                  if ( *(_QWORD *)(v60 + 24) == *(_QWORD *)(v50 + 304) )
                    goto LABEL_67;
LABEL_62:
                  v61 = v60;
                  continue;
                }
              }
              v52 = *(_BYTE *)(v51 + 3);
              v53 = v52 & 0x10;
            }
          }
          LOBYTE(v53) = (_BYTE)v53 == 0;
          if ( ((unsigned __int8)v53 & (v52 >> 6)) != 0 )
          {
LABEL_51:
            sub_1B94BE0((__int64)&v389, (unsigned int *)&v386, v53, v54, v47);
            if ( v367 != (_DWORD)v386 )
              sub_1B94BE0((__int64)&v397, (unsigned int *)&v386, v55, v56, v47);
            goto LABEL_53;
          }
          v53 = *(_QWORD *)(v50 + 280);
          if ( v53 )
          {
            if ( sub_1F4D060(v373, (int)v386, v53, v54, v47, v48) )
              goto LABEL_51;
            if ( v367 == (_DWORD)v386 )
              goto LABEL_214;
          }
          else if ( v367 == (_DWORD)v386 )
          {
            goto LABEL_214;
          }
          sub_1B94BE0((__int64)&v389, (unsigned int *)&v386, v53, v54, v47);
        }
      }
LABEL_53:
      v51 += 40;
      if ( v361 == v51 )
      {
        v9 = v50;
        v10 = v357;
        break;
      }
    }
  }
  v206 = (unsigned __int64 *)*a2;
  if ( v373 == *a2 )
    goto LABEL_295;
  v207 = &v408;
  v330 = 0;
  v358 = v10;
  v208 = *a2;
  while ( (unsigned __int16)(**(_WORD **)(v208 + 16) - 12) <= 1u )
  {
LABEL_292:
    if ( (*(_BYTE *)v208 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v208 + 46) & 8) != 0 )
        v208 = *(_QWORD *)(v208 + 8);
    }
    v208 = *(_QWORD *)(v208 + 8);
    if ( v373 == v208 )
    {
      v10 = v358;
      v206 = (unsigned __int64 *)*a2;
LABEL_295:
      while ( *(unsigned __int64 **)(*(_QWORD *)(v9 + 304) + 32LL) != v206 )
      {
        v209 = *v206 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v209 )
          BUG();
        v210 = *v206 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)v209 & 4) != 0 )
        {
          v211 = **(_WORD **)(v209 + 16);
          if ( v211 != 12 && v211 != 13 )
            break;
        }
        else
        {
          v281 = *(_WORD *)(v209 + 46);
          v282 = *v206 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v281 & 4) != 0 )
          {
            do
              v282 = *(_QWORD *)v282 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v282 + 46) & 4) != 0 );
          }
          if ( (unsigned __int16)(**(_WORD **)(v282 + 16) - 12) > 1u )
            break;
          if ( (v281 & 4) != 0 )
          {
            do
              v210 = *(_QWORD *)v210 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v210 + 46) & 4) != 0 );
          }
        }
        v206 = (unsigned __int64 *)v210;
      }
      v212 = v373;
      if ( (*(_BYTE *)v373 & 4) == 0 && (*(_BYTE *)(v373 + 46) & 8) != 0 )
      {
        do
          v212 = *(_QWORD *)(v212 + 8);
        while ( (*(_BYTE *)(v212 + 46) & 8) != 0 );
      }
      v213 = *(unsigned __int64 **)(v212 + 8);
      for ( n = (unsigned __int64 *)v373; ; n = (unsigned __int64 *)v216 )
      {
        v215 = *n & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v215 )
          BUG();
        v216 = *n & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)v215 & 4) != 0 )
        {
          if ( (unsigned __int16)(**(_WORD **)(v215 + 16) - 12) > 1u )
            goto LABEL_506;
        }
        else
        {
          v287 = *(_WORD *)(v215 + 46);
          v288 = *n & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v287 & 4) != 0 )
          {
            do
              v288 = *(_QWORD *)v288 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v288 + 46) & 4) != 0 );
          }
          if ( (unsigned __int16)(**(_WORD **)(v288 + 16) - 12) > 1u )
          {
LABEL_506:
            if ( v213 != n && v213 != v206 && v213 != n )
            {
              v283 = *v213;
              *(_QWORD *)(v215 + 8) = v213;
              v283 &= 0xFFFFFFFFFFFFFFF8LL;
              *v213 = *v213 & 7 | *n & 0xFFFFFFFFFFFFFFF8LL;
              v284 = *v206;
              *(_QWORD *)(v283 + 8) = v206;
              v284 &= 0xFFFFFFFFFFFFFFF8LL;
              *n = v284 | *n & 7;
              *(_QWORD *)(v284 + 8) = n;
              *v206 = v283 | *v206 & 7;
            }
            v285 = *v206 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v285 )
              BUG();
            if ( (*(_BYTE *)v285 & 4) == 0 && (*(_BYTE *)(v285 + 46) & 4) != 0 )
            {
              do
                v285 = *(_QWORD *)v285 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v285 + 46) & 4) != 0 );
            }
            *a3 = v285;
            *v369 = -16;
            v286 = *(_QWORD **)(v9 + 280);
            --*(_DWORD *)(v9 + 328);
            ++*(_DWORD *)(v9 + 332);
            if ( v286 )
            {
              sub_1DC1A70(v286, v373, 0);
            }
            else
            {
              sub_1F4E1D0(*(char **)(v9 + 272), v367, v373);
              sub_1F4E280(*(_QWORD *)(v9 + 272), v367, v379, 0, v289, v290);
            }
            v341 = v333;
            goto LABEL_215;
          }
          if ( (v287 & 4) != 0 )
          {
            do
              v216 = *(_QWORD *)v216 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v216 + 46) & 4) != 0 );
          }
        }
      }
    }
  }
  if ( v330 > 0xA
    || sub_1E17880(v208)
    || ((v219 = *(_WORD *)(v208 + 46), (v219 & 4) != 0) || (v219 & 8) == 0
      ? (v220 = (*(_QWORD *)(*(_QWORD *)(v208 + 16) + 8LL) >> 4) & 1LL)
      : (LOBYTE(v220) = sub_1E15D00(v208, 0x10u, 1)),
        (_BYTE)v220
     || ((v221 = *(_WORD *)(v208 + 46), (v221 & 4) != 0) || (v221 & 8) == 0
       ? (v222 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v208 + 16) + 8LL) >> 7)
       : (v222 = sub_1E15D00(v208, 0x80u, 1)),
         v222
      || ((v223 = *(_WORD *)(v208 + 46), (v223 & 4) != 0) || (v223 & 8) == 0
        ? (v224 = (*(_QWORD *)(*(_QWORD *)(v208 + 16) + 8LL) >> 6) & 1LL)
        : (LOBYTE(v224) = sub_1E15D00(v208, 0x40u, 1)),
          (_BYTE)v224))) )
  {
    v10 = v358;
    goto LABEL_215;
  }
  v386 = v388;
  v387 = 0x200000000LL;
  v225 = *(_QWORD *)(v208 + 32);
  v226 = v225 + 40LL * *(unsigned int *)(v208 + 40);
  if ( v225 == v226 )
  {
LABEL_368:
    ++v330;
    goto LABEL_292;
  }
  v227 = *(_QWORD *)(v208 + 32);
  v228 = v207;
  v229 = v208;
  v230 = v226;
  do
  {
    if ( !*(_BYTE *)v227 )
    {
      v231 = *(_DWORD *)(v227 + 8);
      if ( v231 )
      {
        if ( (*(_BYTE *)(v227 + 3) & 0x10) != 0 )
        {
          v251 = (unsigned int)v387;
          if ( (unsigned int)v387 >= HIDWORD(v387) )
          {
            v328 = v228;
            sub_16CD150((__int64)&v386, v388, 0, 4, v218, (int)v228);
            v251 = (unsigned int)v387;
            v228 = v328;
          }
          *(_DWORD *)&v386[4 * v251] = v231;
          LODWORD(v387) = v387 + 1;
        }
        else
        {
          if ( v412 )
          {
            v247 = v409;
            if ( v409 )
            {
              v248 = (__int64)v228;
              do
              {
                v217 = *(char **)(v247 + 24);
                if ( v231 > *(_DWORD *)(v247 + 32) )
                {
                  v247 = *(_QWORD *)(v247 + 24);
                }
                else
                {
                  v248 = v247;
                  v247 = *(_QWORD *)(v247 + 16);
                }
              }
              while ( v247 );
              if ( (int *)v248 != v228 && v231 >= *(_DWORD *)(v248 + 32) )
                goto LABEL_386;
            }
          }
          else
          {
            v232 = (char *)dest;
            v217 = (char *)dest + 4 * (unsigned int)v406;
            if ( dest != v217 )
            {
              while ( v231 != *(_DWORD *)v232 )
              {
                v232 += 4;
                if ( v217 == v232 )
                  goto LABEL_336;
              }
              if ( v232 != v217 )
              {
LABEL_386:
                v10 = v358;
                v235 = v386;
                goto LABEL_387;
              }
            }
          }
LABEL_336:
          if ( v404 )
          {
            v249 = v401;
            if ( v401 )
            {
              v250 = &v400;
              do
              {
                v217 = *(char **)(v249 + 24);
                if ( v231 > *(_DWORD *)(v249 + 32) )
                {
                  v249 = *(_QWORD *)(v249 + 24);
                }
                else
                {
                  v250 = (int *)v249;
                  v249 = *(_QWORD *)(v249 + 16);
                }
              }
              while ( v249 );
              if ( v250 != &v400 && v231 >= v250[8] )
                goto LABEL_386;
            }
          }
          else
          {
            v233 = v397;
            v217 = &v397[4 * (unsigned int)v398];
            if ( v397 != v217 )
            {
              while ( v231 != *(_DWORD *)v233 )
              {
                v233 += 4;
                if ( v217 == v233 )
                  goto LABEL_342;
              }
              if ( v233 != v217 )
                goto LABEL_386;
            }
          }
LABEL_342:
          if ( v367 == v231 && v379 != v229 && (*(_BYTE *)(v227 + 3) & 0x40) == 0 )
          {
            v327 = v228;
            v259 = *(_QWORD *)(v9 + 280);
            if ( !v259 )
              goto LABEL_386;
            v260 = sub_1F4D060(v229, v231, v259, (__int64)v217, v218, (int)v228);
            v228 = v327;
            if ( !v260 )
              goto LABEL_386;
          }
        }
      }
    }
    v227 += 40;
  }
  while ( v230 != v227 );
  v234 = v386;
  v208 = v229;
  v207 = v228;
  v235 = v386;
  if ( !(_DWORD)v387 )
  {
LABEL_366:
    if ( v234 != v388 )
      _libc_free((unsigned __int64)v234);
    goto LABEL_368;
  }
  v326 = v208;
  v236 = 0;
  v354 = 4LL * (unsigned int)v387;
  while ( 1 )
  {
    v237 = *(_DWORD *)&v234[v236];
    if ( !v396 )
      break;
    v252 = v393;
    if ( v393 )
    {
      v253 = &v392;
      do
      {
        if ( v237 > *(_DWORD *)(v252 + 32) )
        {
          v252 = *(_QWORD *)(v252 + 24);
        }
        else
        {
          v253 = (int *)v252;
          v252 = *(_QWORD *)(v252 + 16);
        }
      }
      while ( v252 );
      if ( v253 != &v392 && v237 >= v253[8] )
        goto LABEL_400;
    }
LABEL_353:
    if ( (int)v237 > 0 )
    {
      if ( v420 )
      {
        v254 = v417;
        if ( v417 )
        {
          v255 = &v416;
          do
          {
            if ( v237 > *(_DWORD *)(v254 + 32) )
            {
              v254 = *(_QWORD *)(v254 + 24);
            }
            else
            {
              v255 = (int *)v254;
              v254 = *(_QWORD *)(v254 + 16);
            }
          }
          while ( v254 );
          if ( v255 != &v416 && v237 >= v255[8] )
            goto LABEL_400;
        }
      }
      else
      {
        v240 = v413;
        v241 = &v413[(unsigned int)v414];
        if ( v413 != v241 )
        {
          while ( v237 != *v240 )
          {
            if ( v241 == ++v240 )
              goto LABEL_360;
          }
          if ( v240 != v241 )
            goto LABEL_400;
        }
      }
    }
LABEL_360:
    if ( v412 )
    {
      if ( v409 )
      {
        v256 = v409;
        v257 = v207;
        while ( 1 )
        {
          if ( v237 > *(_DWORD *)(v256 + 32) )
          {
            v256 = *(_QWORD *)(v256 + 24);
          }
          else
          {
            v258 = *(_QWORD *)(v256 + 16);
            if ( v237 >= *(_DWORD *)(v256 + 32) )
            {
              v264 = *(_QWORD *)(v256 + 24);
              while ( v264 )
              {
                if ( v237 >= *(_DWORD *)(v264 + 32) )
                {
                  v264 = *(_QWORD *)(v264 + 24);
                }
                else
                {
                  v257 = (int *)v264;
                  v264 = *(_QWORD *)(v264 + 16);
                }
              }
              while ( v258 )
              {
                while ( 1 )
                {
                  v265 = *(_QWORD *)(v258 + 24);
                  if ( v237 <= *(_DWORD *)(v258 + 32) )
                    break;
                  v258 = *(_QWORD *)(v258 + 24);
                  if ( !v265 )
                    goto LABEL_455;
                }
                v256 = v258;
                v258 = *(_QWORD *)(v258 + 16);
              }
LABEL_455:
              if ( v410 != (int *)v256 || v257 != v207 )
              {
                if ( v257 != (int *)v256 )
                {
                  do
                  {
                    v266 = v256;
                    v256 = sub_220EF30(v256);
                    v267 = sub_220F330(v266, v207);
                    j_j___libc_free_0(v267, 40);
                    --v412;
                  }
                  while ( v257 != (int *)v256 );
                  v234 = v386;
                }
                goto LABEL_428;
              }
LABEL_427:
              sub_1F4CC10(v409);
              v410 = v207;
              v234 = v386;
              v409 = 0;
              v411 = v207;
              v412 = 0;
              goto LABEL_428;
            }
            v257 = (int *)v256;
            v256 = *(_QWORD *)(v256 + 16);
          }
          if ( !v256 )
            goto LABEL_425;
        }
      }
      v257 = v207;
LABEL_425:
      if ( v410 == v257 && v257 == v207 )
        goto LABEL_427;
LABEL_428:
      v235 = v234;
    }
    else
    {
      v242 = (char *)dest;
      v243 = v406;
      v244 = (char *)dest + 4 * (unsigned int)v406;
      while ( v242 != v244 )
      {
        v245 = *(_DWORD *)v242;
        v246 = v242;
        v242 += 4;
        if ( v237 == v245 )
        {
          if ( v242 != v244 )
          {
            memmove(v246, v242, v244 - v242);
            v234 = v386;
            v243 = v406;
            v235 = v386;
          }
          LODWORD(v406) = v243 - 1;
          break;
        }
      }
    }
    v236 += 4;
    if ( v236 == v354 )
    {
      v208 = v326;
      goto LABEL_366;
    }
  }
  v238 = v389;
  v239 = (__int64 *)((char *)v389 + 4 * (unsigned int)v390);
  if ( v389 == v239 )
    goto LABEL_353;
  while ( v237 != *(_DWORD *)v238 )
  {
    v238 = (__int64 *)((char *)v238 + 4);
    if ( v239 == v238 )
      goto LABEL_353;
  }
  if ( v239 == v238 )
    goto LABEL_353;
LABEL_400:
  v10 = v358;
LABEL_387:
  if ( v235 != v388 )
    _libc_free((unsigned __int64)v235);
LABEL_215:
  v173 = v417;
  while ( v173 )
  {
    sub_1F4CC10(*(_QWORD *)(v173 + 24));
    v174 = v173;
    v173 = *(_QWORD *)(v173 + 16);
    j_j___libc_free_0(v174, 40);
  }
  if ( v413 != (int *)v415 )
    _libc_free((unsigned __int64)v413);
  v175 = v409;
  while ( v175 )
  {
    sub_1F4CC10(*(_QWORD *)(v175 + 24));
    v176 = v175;
    v175 = *(_QWORD *)(v175 + 16);
    j_j___libc_free_0(v176, 40);
  }
  if ( dest != v407 )
    _libc_free((unsigned __int64)dest);
  v177 = v401;
  while ( v177 )
  {
    sub_1F4CC10(*(_QWORD *)(v177 + 24));
    v178 = v177;
    v177 = *(_QWORD *)(v177 + 16);
    j_j___libc_free_0(v178, 40);
  }
  if ( v397 != v399 )
    _libc_free((unsigned __int64)v397);
  v179 = v393;
  while ( v179 )
  {
    sub_1F4CC10(*(_QWORD *)(v179 + 24));
    v180 = v179;
    v179 = *(_QWORD *)(v179 + 16);
    j_j___libc_free_0(v180, 40);
  }
  if ( v389 != (__int64 *)v391 )
    _libc_free((unsigned __int64)v389);
  if ( !v341 )
  {
LABEL_110:
    v70 = *(_QWORD *)(v10 + 16);
    goto LABEL_82;
  }
  return 1;
}
