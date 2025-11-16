// Function: sub_69ED20
// Address: 0x69ed20
//
__int64 __fastcall sub_69ED20(__int64 a1, __m128i *a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r15d
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r8
  unsigned __int16 v9; // ax
  const char *v10; // r9
  __int64 v11; // rdx
  int v12; // esi
  char v13; // al
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int16 v19; // ax
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __m128i v23; // xmm4
  __m128i v24; // xmm5
  __m128i v25; // xmm6
  __m128i v26; // xmm7
  __int64 v27; // rdx
  __m128i v28; // xmm0
  unsigned __int16 v29; // ax
  __int64 v30; // rdx
  char v31; // al
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm7
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __m128i v39; // xmm7
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __m128i v42; // xmm4
  __m128i v43; // xmm5
  __m128i v44; // xmm6
  __m128i v45; // xmm7
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 result; // rax
  char v53; // al
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __m128i v59; // xmm2
  __m128i v60; // xmm3
  __m128i v61; // xmm4
  __m128i v62; // xmm5
  __m128i v63; // xmm6
  __m128i v64; // xmm7
  __m128i v65; // xmm1
  __m128i v66; // xmm2
  __m128i v67; // xmm3
  __m128i v68; // xmm4
  __m128i v69; // xmm5
  __m128i v70; // xmm6
  __int64 v71; // rsi
  __int64 v72; // rdi
  char v73; // al
  __int64 v74; // rdi
  char v75; // dl
  __int64 v76; // rax
  __m128i v77; // xmm4
  __m128i v78; // xmm5
  __m128i v79; // xmm6
  __m128i v80; // xmm7
  __m128i v81; // xmm1
  __m128i v82; // xmm2
  __m128i v83; // xmm3
  __m128i v84; // xmm4
  __m128i v85; // xmm5
  __m128i v86; // xmm6
  __m128i v87; // xmm7
  __m128i v88; // xmm1
  __m128i v89; // xmm5
  __m128i v90; // xmm6
  __m128i v91; // xmm7
  __m128i v92; // xmm2
  __m128i v93; // xmm3
  __m128i v94; // xmm4
  __m128i v95; // xmm1
  __m128i v96; // xmm5
  __m128i v97; // xmm6
  __m128i v98; // xmm7
  __m128i v99; // xmm2
  __m128i v100; // xmm3
  char v101; // al
  __int64 v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 v106; // r9
  unsigned __int64 v107; // rdi
  __int64 v108; // rcx
  __int64 v109; // rdx
  unsigned int *v110; // rdx
  int v111; // esi
  int v112; // r8d
  int v113; // eax
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 m; // rdi
  __int64 v119; // rdx
  __int64 v120; // rcx
  __int64 v121; // r8
  __int64 v122; // r9
  __int64 v123; // rdi
  __int64 v124; // r13
  unsigned __int16 v125; // ax
  int v126; // r8d
  __int64 v127; // r8
  __int64 v128; // rdx
  char v129; // al
  __m128i *v130; // rsi
  __int64 v131; // rdi
  __int64 v132; // rax
  __int64 v133; // rcx
  __int64 v134; // r9
  _QWORD *v135; // rdx
  __int64 v136; // rax
  __m128i *v137; // r8
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rcx
  __int64 v141; // r9
  __m128i *v142; // rdi
  __int64 **v143; // r12
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // r8
  __int64 v147; // r9
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 v150; // rdx
  __int64 v151; // rcx
  unsigned __int16 v152; // ax
  __int64 v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rcx
  __int64 v156; // r12
  char v157; // al
  __m128i *v158; // rsi
  __int64 v159; // rax
  int v160; // eax
  __int64 v161; // rax
  __int64 v162; // rdi
  __int64 v163; // r12
  __int64 v164; // r10
  _BYTE *v165; // rax
  __int64 v166; // rax
  int v167; // eax
  int v168; // eax
  int v169; // eax
  int v170; // eax
  int v171; // eax
  int v172; // eax
  int v173; // eax
  __int64 v174; // r12
  __int64 v175; // rdx
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // r9
  __int64 v179; // rax
  __int64 v180; // rcx
  int v181; // eax
  int v182; // eax
  int v183; // eax
  int v184; // eax
  __int64 v185; // rax
  __int64 v186; // rax
  unsigned __int8 v187; // al
  __int64 v188; // rdi
  __int64 v189; // r12
  __int64 v190; // rdx
  __int64 v191; // rcx
  __int64 v192; // r8
  __int64 v193; // r9
  __int64 v194; // rdx
  __int64 v195; // rcx
  __int64 v196; // r8
  __int64 v197; // r9
  __int64 v198; // rdx
  char v199; // al
  __int64 v200; // r12
  __int64 v201; // r15
  __int64 i; // rbx
  __int64 v203; // rdi
  __int64 v204; // rdi
  __int64 v205; // rax
  unsigned int v206; // edi
  __m128i v207; // xmm3
  __m128i v208; // xmm4
  __m128i v209; // xmm5
  __m128i v210; // xmm6
  __m128i v211; // xmm2
  __m128i v212; // xmm3
  __m128i v213; // xmm4
  __m128i v214; // xmm5
  int v215; // eax
  int v216; // eax
  __int16 v217; // ax
  unsigned __int16 v218; // ax
  __int64 v219; // rdx
  __int64 v220; // rcx
  __int64 v221; // r8
  __int64 v222; // r9
  unsigned __int64 v223; // rax
  _BOOL4 v224; // r8d
  __int64 v225; // rdx
  __int64 v226; // rcx
  __int64 v227; // rdx
  __int64 v228; // rcx
  __int64 v229; // r12
  __int64 v230; // rax
  __int64 v231; // rax
  __int64 v232; // rdx
  char n; // al
  __m128i v234; // xmm2
  __m128i v235; // xmm3
  __m128i v236; // xmm4
  __m128i v237; // xmm5
  __m128i v238; // xmm6
  __m128i v239; // xmm2
  __m128i v240; // xmm3
  __m128i v241; // xmm4
  __m128i v242; // xmm5
  __m128i v243; // xmm6
  __m128i v244; // xmm2
  __m128i v245; // xmm3
  char v246; // al
  __int32 v247; // edx
  __int64 v248; // rsi
  unsigned __int8 v249; // al
  unsigned __int8 v250; // dl
  unsigned __int8 v251; // al
  __int16 v252; // ax
  __int64 v253; // rax
  __int64 v254; // rdx
  unsigned int *v255; // rdx
  size_t v256; // rdx
  unsigned int v257; // eax
  __int64 v258; // r12
  __int64 v259; // rax
  __int64 j; // r12
  _DWORD *v261; // r12
  __int64 v262; // rdx
  __int64 v263; // rcx
  char v264; // cl
  __int64 v265; // r10
  __int64 *v266; // r12
  int v267; // r10d
  __int64 v268; // rax
  __int64 v269; // rax
  unsigned int v270; // eax
  __int64 v271; // rcx
  __m128i *v272; // rdi
  __int64 v273; // rax
  __int32 *v274; // rsi
  __int64 v275; // rdx
  __int16 v276; // ax
  __int64 ii; // rcx
  __int64 v278; // rax
  int v279; // eax
  unsigned __int8 v280; // al
  __int64 v281; // rax
  __int64 v282; // rdx
  __int64 v283; // rcx
  __int64 v284; // rdx
  __int64 v285; // rcx
  __int64 v286; // r8
  __int64 v287; // r9
  __int64 v288; // rdx
  __int64 v289; // rcx
  __int64 v290; // r8
  __int64 v291; // r9
  __int64 v292; // rax
  char v293; // dl
  __int64 v294; // rdx
  int v295; // eax
  __int64 v296; // rdx
  __int64 v297; // rax
  __int64 v298; // rdx
  __int64 v299; // r12
  char v300; // al
  __int64 v301; // rax
  __int64 v302; // rax
  __int64 v303; // rdx
  __int64 v304; // rcx
  __int64 v305; // r8
  __int64 v306; // rdx
  unsigned __int8 v307; // cl
  __int16 v308; // ax
  const char *v309; // r12
  __int64 v310; // rax
  __int64 v311; // [rsp-8h] [rbp-778h]
  int v312; // [rsp+8h] [rbp-768h]
  unsigned __int64 v313; // [rsp+10h] [rbp-760h]
  int v314; // [rsp+18h] [rbp-758h]
  unsigned __int64 v315; // [rsp+18h] [rbp-758h]
  bool v316; // [rsp+20h] [rbp-750h]
  __int64 v317; // [rsp+28h] [rbp-748h]
  __m128i *v318; // [rsp+28h] [rbp-748h]
  _QWORD *v319; // [rsp+28h] [rbp-748h]
  unsigned int v320; // [rsp+28h] [rbp-748h]
  __int64 k; // [rsp+28h] [rbp-748h]
  __int64 v322; // [rsp+28h] [rbp-748h]
  int v323; // [rsp+30h] [rbp-740h]
  __int64 v324; // [rsp+30h] [rbp-740h]
  __int32 v325; // [rsp+30h] [rbp-740h]
  unsigned int v326; // [rsp+30h] [rbp-740h]
  __int16 v327; // [rsp+30h] [rbp-740h]
  int v328; // [rsp+30h] [rbp-740h]
  __int64 v329; // [rsp+30h] [rbp-740h]
  __int64 v330; // [rsp+30h] [rbp-740h]
  __int16 v332; // [rsp+42h] [rbp-72Eh]
  int v333; // [rsp+44h] [rbp-72Ch]
  __int64 v334; // [rsp+48h] [rbp-728h]
  int v335; // [rsp+50h] [rbp-720h]
  int v336; // [rsp+50h] [rbp-720h]
  __int16 v337; // [rsp+50h] [rbp-720h]
  __int16 v338; // [rsp+50h] [rbp-720h]
  __int64 v339; // [rsp+50h] [rbp-720h]
  unsigned int v340; // [rsp+50h] [rbp-720h]
  unsigned int v341; // [rsp+50h] [rbp-720h]
  int v342; // [rsp+50h] [rbp-720h]
  int v343; // [rsp+50h] [rbp-720h]
  int v344; // [rsp+50h] [rbp-720h]
  int v345; // [rsp+50h] [rbp-720h]
  __int64 v346; // [rsp+50h] [rbp-720h]
  __int64 v347; // [rsp+50h] [rbp-720h]
  int v348; // [rsp+50h] [rbp-720h]
  int v349; // [rsp+50h] [rbp-720h]
  int v350; // [rsp+50h] [rbp-720h]
  int v351; // [rsp+50h] [rbp-720h]
  char v352; // [rsp+58h] [rbp-718h]
  char v353; // [rsp+5Ch] [rbp-714h]
  int v354; // [rsp+5Ch] [rbp-714h]
  int v355; // [rsp+60h] [rbp-710h]
  _QWORD *v356; // [rsp+68h] [rbp-708h]
  unsigned int v357; // [rsp+78h] [rbp-6F8h] BYREF
  int v358; // [rsp+7Ch] [rbp-6F4h] BYREF
  __int64 v359; // [rsp+80h] [rbp-6F0h] BYREF
  __int64 v360; // [rsp+88h] [rbp-6E8h] BYREF
  __int64 v361; // [rsp+90h] [rbp-6E0h] BYREF
  __int64 v362; // [rsp+98h] [rbp-6D8h] BYREF
  char v363[160]; // [rsp+A0h] [rbp-6D0h] BYREF
  __m128i v364; // [rsp+140h] [rbp-630h] BYREF
  __m128i v365; // [rsp+150h] [rbp-620h]
  __m128i v366; // [rsp+160h] [rbp-610h]
  __m128i v367; // [rsp+170h] [rbp-600h]
  __m128i v368; // [rsp+180h] [rbp-5F0h]
  __m128i v369; // [rsp+190h] [rbp-5E0h]
  __m128i v370; // [rsp+1A0h] [rbp-5D0h]
  __m128i v371; // [rsp+1B0h] [rbp-5C0h]
  __m128i v372; // [rsp+1C0h] [rbp-5B0h]
  __m128i v373; // [rsp+1D0h] [rbp-5A0h]
  __m128i v374; // [rsp+1E0h] [rbp-590h]
  __m128i v375; // [rsp+1F0h] [rbp-580h]
  __m128i v376; // [rsp+200h] [rbp-570h]
  __m128i v377; // [rsp+210h] [rbp-560h]
  __m128i v378; // [rsp+220h] [rbp-550h]
  __m128i v379; // [rsp+230h] [rbp-540h]
  __m128i v380; // [rsp+240h] [rbp-530h]
  __m128i v381; // [rsp+250h] [rbp-520h]
  __m128i v382; // [rsp+260h] [rbp-510h]
  __m128i v383; // [rsp+270h] [rbp-500h]
  __m128i v384; // [rsp+280h] [rbp-4F0h]
  __m128i v385; // [rsp+290h] [rbp-4E0h]
  __m128i v386; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i v387; // [rsp+2B0h] [rbp-4C0h] BYREF
  __m128i v388; // [rsp+2C0h] [rbp-4B0h] BYREF
  __m128i v389; // [rsp+2D0h] [rbp-4A0h] BYREF
  __m256i v390; // [rsp+2E0h] [rbp-490h] BYREF
  __m128i v391; // [rsp+300h] [rbp-470h] BYREF
  __m128i v392; // [rsp+310h] [rbp-460h] BYREF
  __m128i v393; // [rsp+320h] [rbp-450h] BYREF
  __m128i v394; // [rsp+330h] [rbp-440h] BYREF
  __m128i v395; // [rsp+340h] [rbp-430h] BYREF
  __m128i v396; // [rsp+350h] [rbp-420h] BYREF
  __m128i v397; // [rsp+360h] [rbp-410h] BYREF
  __m128i v398; // [rsp+370h] [rbp-400h] BYREF
  __m128i v399; // [rsp+380h] [rbp-3F0h] BYREF
  __m128i v400; // [rsp+390h] [rbp-3E0h] BYREF
  __m128i v401; // [rsp+3A0h] [rbp-3D0h] BYREF
  __m128i v402; // [rsp+3B0h] [rbp-3C0h] BYREF
  __m128i v403; // [rsp+3C0h] [rbp-3B0h] BYREF
  __m128i v404; // [rsp+3D0h] [rbp-3A0h] BYREF
  __m128i v405; // [rsp+3E0h] [rbp-390h] BYREF
  __m128i v406; // [rsp+3F0h] [rbp-380h] BYREF
  __m128i v407; // [rsp+400h] [rbp-370h] BYREF
  __m128i v408; // [rsp+410h] [rbp-360h] BYREF
  __m128i v409; // [rsp+420h] [rbp-350h] BYREF
  __m128i v410; // [rsp+430h] [rbp-340h] BYREF
  __m128i v411; // [rsp+440h] [rbp-330h] BYREF
  __m128i v412; // [rsp+450h] [rbp-320h] BYREF
  __m128i v413; // [rsp+460h] [rbp-310h] BYREF
  __m128i v414; // [rsp+470h] [rbp-300h] BYREF
  __m128i v415; // [rsp+480h] [rbp-2F0h] BYREF
  __m128i v416; // [rsp+490h] [rbp-2E0h] BYREF
  __m128i v417; // [rsp+4A0h] [rbp-2D0h] BYREF
  __m128i v418; // [rsp+4B0h] [rbp-2C0h] BYREF
  __m128i v419; // [rsp+4C0h] [rbp-2B0h] BYREF
  __m128i v420; // [rsp+4D0h] [rbp-2A0h] BYREF
  __m128i v421; // [rsp+4E0h] [rbp-290h] BYREF
  __m128i v422; // [rsp+4F0h] [rbp-280h] BYREF
  __m128i v423; // [rsp+500h] [rbp-270h] BYREF
  __m128i v424; // [rsp+510h] [rbp-260h] BYREF
  __m128i v425; // [rsp+520h] [rbp-250h] BYREF
  __m128i v426; // [rsp+530h] [rbp-240h] BYREF
  __m128i v427; // [rsp+540h] [rbp-230h] BYREF
  __m128i v428; // [rsp+550h] [rbp-220h] BYREF
  __m128i v429[33]; // [rsp+560h] [rbp-210h] BYREF

  v4 = a3;
  v5 = a4;
  v356 = (_QWORD *)a1;
  v6 = qword_4D03C50;
  v332 = *(_WORD *)(qword_4D03C50 + 20LL);
  v352 = 0;
  if ( (a4 & 0x4000) != 0 )
  {
    v352 = 1;
    BYTE1(v5) = BYTE1(a4) & 0xBF;
  }
  if ( (v5 & 0x10000) != 0 )
    *(_BYTE *)(qword_4D03C50 + 21LL) |= 1u;
  v7 = *(_QWORD *)(v6 + 136);
  if ( v7 && *(_QWORD *)v7 && (unsigned int)sub_6E6800(a1) )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 68);
    goto LABEL_72;
  }
  v355 = 0;
  v353 = 0;
  if ( word_4F06418[0] == 187 )
  {
    sub_7B8B50(a1, v7, a3, a4);
    a4 = *(unsigned __int8 *)(qword_4D03C50 + 20LL);
    v101 = a4 | 0x80;
    LOBYTE(a4) = (unsigned __int8)a4 >> 7;
    v7 = (unsigned __int8)a4;
    v353 = a4;
    *(_BYTE *)(qword_4D03C50 + 20LL) = v101;
    v355 = 1;
  }
  v361 = *(_QWORD *)&dword_4F063F8;
  v334 = unk_4D03C48;
  unk_4D03C48 = 0;
  v333 = HIDWORD(qword_4F077B4);
  if ( HIDWORD(qword_4F077B4) )
  {
    v333 = 0;
    if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x40000100) == 0x40000000 )
    {
      a1 = 4;
      sub_6E1E00(4, v363, 0, 0);
      v11 = qword_4D03C50;
      a4 = *(_QWORD *)qword_4D03C50;
      v12 = *(unsigned __int8 *)(qword_4D03C50 + 18LL);
      v13 = v12 | 9;
      v7 = v12 | 8u;
      *(_BYTE *)(qword_4D03C50 + 18LL) = v7;
      if ( (*(_BYTE *)(a4 + 18) & 1) == 0 )
        v13 = v7;
      *(_BYTE *)(v11 + 18) = v13;
      v333 = 1;
      *(_QWORD *)(v11 + 40) = *(_QWORD *)(a4 + 40);
    }
  }
  v8 = v5 & 8;
  if ( (v5 & 8) != 0 )
  {
LABEL_19:
    sub_6D1B40(&v386, &v407, v5);
    goto LABEL_20;
  }
  v9 = word_4F06418[0];
  if ( word_4F06418[0] == 185 )
  {
    if ( dword_4F077C4 != 2 )
    {
LABEL_119:
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      sub_6E5F20(29);
LABEL_120:
      sub_6E6260(&v386);
      goto LABEL_20;
    }
    v7 = 0;
    a1 = 0x4000;
    sub_7C0F00(0x4000, 0);
    v9 = word_4F06418[0];
    v8 = v5 & 8;
  }
  v10 = "__is_signed";
  while ( 2 )
  {
    switch ( v9 )
    {
      case 1u:
        v164 = qword_4D04A00;
        if ( !qword_4D04A00 || (*(_BYTE *)(qword_4D04A00 + 73) & 2) == 0 )
        {
          if ( dword_4F077C4 != 2 )
            goto LABEL_266;
          goto LABEL_431;
        }
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( strcmp(*(const char **)(qword_4D04A00 + 8), "__is_signed") )
            goto LABEL_265;
          v322 = qword_4D04A00;
          v328 = v8;
          if ( sub_688AA0() )
          {
            word_4F06418[0] = 324;
            sub_6A98C0(0, &v386);
            goto LABEL_20;
          }
          LODWORD(v8) = v328;
          v164 = v322;
          if ( !HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4 )
          {
LABEL_491:
            v218 = word_4F06418[0];
            goto LABEL_333;
          }
        }
        else if ( !HIDWORD(qword_4F077B4) )
        {
LABEL_265:
          if ( dword_4F077C4 == 2 )
            goto LABEL_431;
          goto LABEL_266;
        }
        if ( qword_4F077A8 <= 0x249EFu )
          goto LABEL_491;
        v330 = v164;
        v342 = v8;
        if ( sub_688AA0() )
        {
          v309 = *(const char **)(v330 + 8);
          if ( !strcmp(v309, "__is_pointer") )
          {
            word_4F06418[0] = 320;
            sub_6A98C0(0, &v386);
          }
          else if ( !strcmp(v309, "__is_invocable") )
          {
            word_4F06418[0] = 225;
            sub_6A9CC0(112, 0, &v386);
          }
          else if ( !strcmp(v309, "__is_nothrow_invocable") )
          {
            word_4F06418[0] = 226;
            sub_6A9CC0(113, 0, &v386);
          }
          goto LABEL_20;
        }
        goto LABEL_332;
      case 2u:
      case 3u:
      case 0x7Bu:
      case 0x7Cu:
      case 0x7Du:
        v110 = &word_4D04898;
        if ( word_4D04898 )
        {
          v110 = (unsigned int *)qword_4D03C50;
          if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
            goto LABEL_148;
        }
        v7 = (__int64)&qword_4F077B4 + 4;
        if ( !HIDWORD(qword_4F077B4)
          || (a4 = qword_4D03C50, v110 = (unsigned int *)*(unsigned __int8 *)(qword_4D03C50 + 16LL), !(_BYTE)v110) )
        {
          if ( (v5 & 2) != 0 && v9 == 2 )
          {
            v276 = sub_7BE840(0, 0);
            v7 = v4;
            v8 = sub_688800(v276, v4, v5);
          }
          if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
          {
LABEL_194:
            v125 = word_4F06418[0];
            if ( word_4F06418[0] != 124 )
            {
              if ( word_4F06418[0] != 125 )
                goto LABEL_196;
              v349 = v8;
              v107 = (unsigned __int64)&v386;
              sub_6E8C70(&v386, v7, v110, a4, v8, v10);
              v126 = v349;
LABEL_198:
              if ( !v126 )
                goto LABEL_185;
              goto LABEL_145;
            }
LABEL_504:
            v351 = v8;
            v107 = (unsigned __int64)&v386;
            sub_6E8BF0(&v386, v7, v110, a4, v8, v10);
            v126 = v351;
            goto LABEL_198;
          }
          v110 = (unsigned int *)*(unsigned __int8 *)(qword_4D03C50 + 16LL);
          if ( !(_BYTE)v110 )
          {
            v350 = v8;
            v7 = (__int64)&v386;
            v107 = (unsigned int)sub_6E92D0();
            sub_6E6890(v107, &v386);
            v126 = v350;
            goto LABEL_198;
          }
          if ( (_BYTE)v110 == 1 )
            goto LABEL_399;
          goto LABEL_193;
        }
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
        {
          if ( (_BYTE)v110 == 1 )
          {
LABEL_472:
            v125 = word_4F06418[0];
            v8 = 1;
            if ( word_4F06418[0] == 124 )
              goto LABEL_504;
            if ( word_4F06418[0] != 125 )
            {
LABEL_196:
              if ( v125 != 123 )
              {
                v107 = (unsigned __int64)xmmword_4F06300;
                v336 = v8;
                v7 = (__int64)&v386;
                sub_6E6A50(xmmword_4F06300, &v386);
                v126 = v336;
                goto LABEL_198;
              }
              goto LABEL_468;
            }
LABEL_474:
            v107 = (unsigned __int64)&v386;
            sub_6E8C70(&v386, v7, v110, a4, v8, v10);
            goto LABEL_145;
          }
          v8 = 1;
LABEL_193:
          if ( (_BYTE)v110 != 2 || dword_4D04800 )
            goto LABEL_194;
LABEL_399:
          if ( !(_DWORD)v8 )
          {
LABEL_184:
            v257 = sub_6E92D0();
            sub_6E6890(v257, &v386);
LABEL_185:
            v7 = (__int64)&v386;
            v107 = 1;
            sub_6E26D0(1, &v386);
            goto LABEL_145;
          }
          goto LABEL_472;
        }
LABEL_148:
        if ( v9 != 124 )
        {
          if ( v9 != 125 )
          {
            if ( v9 != 123 )
            {
              v107 = (unsigned __int64)xmmword_4F06300;
              v7 = (__int64)&v386;
              sub_6E6A50(xmmword_4F06300, &v386);
              goto LABEL_145;
            }
            v8 = 1;
LABEL_468:
            v348 = v8;
            v107 = (unsigned __int64)&v386;
            sub_6E8CF0(&v386, v7, v110, a4, v8, v10);
            v126 = v348;
            goto LABEL_198;
          }
          goto LABEL_474;
        }
        v107 = (unsigned __int64)&v386;
        sub_6E8BF0(&v386, v7, v110, a4, v8, "__is_signed");
        goto LABEL_145;
      case 4u:
      case 5u:
      case 6u:
      case 0xB5u:
      case 0xB6u:
        if ( byte_4F063A9[0] < 0 )
        {
          if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
          {
            v256 = unk_4F06400;
            if ( (unsigned __int64)(unk_4F06400 + 1LL) > unk_4F06C48 )
            {
              sub_729510(unk_4F06400 + 1LL, v7, unk_4F06400);
              v256 = unk_4F06400;
            }
            strncpy((char *)qword_4F06C50, unk_4F06410, v256);
            *((_BYTE *)qword_4F06C50 + unk_4F06400) = 0;
            sub_6E5DE0(4, 193, &dword_4F063F8, qword_4F06C50);
          }
          byte_4F063A9[0] &= ~0x80u;
        }
        v107 = (unsigned __int64)xmmword_4F06300;
        v7 = (__int64)&v386;
        sub_6E6A50(xmmword_4F06300, &v386);
        unk_4F061D8 = *(__int64 *)((char *)&v390.m256i_i64[1] + 4);
        v109 = (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
        if ( qword_4D0495C && (byte_4F063A9[0] & 2) != 0 )
          v387.m128i_i8[3] |= 0x40u;
        goto LABEL_145;
      case 7u:
        v335 = sub_693ED0();
        sub_6E7020(xmmword_4F06300, &v386);
        v387.m128i_i8[3] = (16 * (v335 == 0)) | v387.m128i_i8[3] & 0xEF;
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 && *(_BYTE *)(qword_4D03C50 + 16LL) <= 1u )
          goto LABEL_184;
        goto LABEL_185;
      case 8u:
        v386.m128i_i64[0] = 0;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( !qword_4F06218 )
        {
          v7 = (__int64)&dword_4F063F8;
          sub_6851C0(0x9B6u, &dword_4F063F8);
          v204 = v386.m128i_i64[0];
          v364.m128i_i32[0] = 0;
          goto LABEL_385;
        }
        v7 = (__int64)&v386;
        if ( !(unsigned int)sub_64A6B0(&v364, &v386) )
        {
          if ( (unsigned int)sub_6E5430(&v364, &v386, v194, v195, v196, v197) && v386.m128i_i64[0] )
          {
            v347 = *qword_4F06218;
            v261 = sub_67D9D0(0x9B7u, &dword_4F063F8);
            v7 = *(_QWORD *)(v347 + 16) - 11LL;
            sub_881010((void *)(*(_QWORD *)(v347 + 8) + 11LL));
            sub_685910((__int64)v261, (FILE *)v7);
          }
          sub_6E6260(&v407);
          v204 = v386.m128i_i64[0];
          goto LABEL_385;
        }
        if ( !v364.m128i_i32[0] )
        {
          v204 = v386.m128i_i64[0];
          if ( !unk_4F063AD )
          {
            if ( (unsigned int)sub_6E5430(v386.m128i_i64[0], &v386, v194, v195, v196, v197) )
            {
              v7 = (__int64)&dword_4F063F8;
              sub_6851C0(0xB4Du, &dword_4F063F8);
            }
            goto LABEL_402;
          }
          goto LABEL_385;
        }
        v429[0].m128i_i64[0] = 0;
        if ( *((_BYTE *)qword_4F06218 + 80) != 17 )
        {
          v198 = **(_QWORD **)(qword_4F06218[11] + 328LL);
          v199 = *(_BYTE *)(*(_QWORD *)(v198 + 8) + 80LL);
          if ( v199 == 3 )
            goto LABEL_441;
          if ( v199 == 2 )
          {
            v346 = **(_QWORD **)(qword_4F06218[11] + 328LL);
            if ( (unsigned int)sub_8D3A70(*(_QWORD *)(*(_QWORD *)(v198 + 64) + 128LL))
              || (unsigned int)sub_8D3F60(*(_QWORD *)(*(_QWORD *)(v346 + 64) + 128LL)) )
            {
              goto LABEL_441;
            }
          }
LABEL_306:
          sub_7296C0(&v407);
          v429[0].m128i_i64[0] = sub_725090(3);
          v200 = v429[0].m128i_i64[0];
          goto LABEL_307;
        }
        if ( *(_BYTE *)(xmmword_4F06380[0].m128i_i64[0] + 140) != 8 )
          goto LABEL_306;
LABEL_441:
        if ( (unsigned int)sub_689CB0(v386.m128i_i64, v386.m128i_i64) )
        {
          v204 = v386.m128i_i64[0];
          if ( v386.m128i_i64[0] )
          {
            v246 = *(_BYTE *)(v386.m128i_i64[0] + 80);
            goto LABEL_387;
          }
LABEL_402:
          v107 = (unsigned __int64)&v386;
          sub_6E6260(&v386);
          goto LABEL_145;
        }
        sub_7296C0(&v407);
        v258 = sub_8D4050(unk_4F062A0);
        v259 = sub_725090(0);
        for ( v429[0].m128i_i64[0] = v259; *(_BYTE *)(v258 + 140) == 12; v258 = *(_QWORD *)(v258 + 160) )
          ;
        *(_QWORD *)(v259 + 32) = v258;
        v200 = sub_725090(3);
        *(_QWORD *)v429[0].m128i_i64[0] = v200;
LABEL_307:
        sub_740640(&unk_4F06220);
        if ( *(_QWORD *)(unk_4F062D0 + 120LL) )
        {
          v340 = v4;
          v201 = unk_4F062D0;
          v327 = v5;
          for ( i = *(_QWORD *)(unk_4F062D0 + 120LL); ; i = *(_QWORD *)(i + 120) )
          {
            v319 = (_QWORD *)v200;
            v200 = sub_725090(1);
            *v319 = v200;
            *(_QWORD *)(v200 + 32) = v201;
            *(_BYTE *)(v200 + 24) |= 8u;
            *(_QWORD *)(v201 + 120) = 0;
            v201 = i;
            if ( !*(_QWORD *)(i + 120) )
              break;
          }
          v4 = v340;
          LOWORD(v5) = v327;
        }
        sub_729730(v407.m128i_u32[0]);
        v203 = (__int64)qword_4F06218;
        if ( *((_BYTE *)qword_4F06218 + 80) == 17 )
        {
          for ( j = qword_4F06218[11]; j; j = *(_QWORD *)(j + 8) )
          {
            if ( *(_BYTE *)(j + 80) == 20 )
            {
              v7 = v429[0].m128i_i64[0];
              if ( sub_8B1C20(j, v429[0].m128i_i64[0], 0, 0, 0x20000) )
              {
                if ( v386.m128i_i64[0] )
                {
                  v7 = (__int64)&dword_4F063F8;
                  sub_6851C0(0x9B7u, &dword_4F063F8);
                }
                else
                {
                  v386.m128i_i64[0] = j;
                }
              }
            }
          }
          v203 = v386.m128i_i64[0];
          if ( !v386.m128i_i64[0] )
            goto LABEL_402;
        }
        else
        {
          v386.m128i_i64[0] = (__int64)qword_4F06218;
        }
        v7 = (__int64)v429;
        v386.m128i_i64[0] = sub_8B74F0(v203, v429, 1, &dword_4F063F8);
        v204 = v386.m128i_i64[0];
LABEL_385:
        if ( !v204 )
          goto LABEL_402;
        v246 = *(_BYTE *)(v204 + 80);
        if ( v246 == 17 )
          goto LABEL_402;
LABEL_387:
        v429[0].m128i_i64[0] = qword_4F063F0;
        if ( v246 == 11 )
        {
          sub_875C60(v204, 0, &dword_4F063F8);
          LODWORD(v204) = v386.m128i_i32[0];
        }
        sub_6EAB60(v204, 0, 1, (unsigned int)&dword_4F063F8, (unsigned int)v429, 0, (__int64)&v407);
        v14 = (__int64)&v407;
        sub_6F69D0(&v407, 28);
        qword_4F06218 = (_QWORD *)v386.m128i_i64[0];
        v7 = v311;
        if ( !v386.m128i_i64[0] )
          goto LABEL_402;
        v247 = 0;
        if ( !v364.m128i_i32[0] )
        {
          v264 = *(_BYTE *)(v386.m128i_i64[0] + 80);
          if ( v264 == 11 )
          {
            v265 = *(_QWORD *)(*(_QWORD *)(v386.m128i_i64[0] + 88) + 152LL);
            for ( k = *(_QWORD *)(v386.m128i_i64[0] + 88); *(_BYTE *)(v265 + 140) == 12; v265 = *(_QWORD *)(v265 + 160) )
              ;
            v314 = v265;
            v316 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0;
            v266 = (__int64 *)sub_6E2F40(0);
            sub_6E6A50(xmmword_4F06300, v266[3] + 8);
            v267 = v314;
            if ( unk_4F063AD == 2 && (*(_BYTE *)(qword_4F06218[11] + 207LL) & 1) == 0 )
            {
              v312 = v314;
              v315 = *(_QWORD *)word_4F063B0;
              v313 = qword_4F06B40[unk_4F063A8 & 7];
              v310 = sub_6E2F40(0);
              *v266 = v310;
              sub_6E7080(*(_QWORD *)(v310 + 24) + 8LL, v315 / v313 - 1);
              v267 = v312;
            }
            *(_BYTE *)(qword_4D03C50 + 18LL) |= 8u;
            sub_6C0910(v267, k, 0, (unsigned int)v429, 0, 0, 0, 0, 0, 1, (__int64)v266, 0, 0, 0, 0);
            *(_BYTE *)(qword_4D03C50 + 18LL) = *(_BYTE *)(qword_4D03C50 + 18LL) & 0xF7 | (8 * v316);
            v268 = v429[0].m128i_i64[0];
            if ( *(_BYTE *)(v429[0].m128i_i64[0] + 24) == 2 )
              *(_QWORD *)(*(_QWORD *)(v429[0].m128i_i64[0] + 56) + 144LL) = 0;
            v269 = *(_QWORD *)(v268 + 16);
            if ( v269 && *(_BYTE *)(v269 + 24) == 2 )
              *(_QWORD *)(*(_QWORD *)(v269 + 56) + 144LL) = 0;
            sub_6E1990(v266);
            v247 = v429[0].m128i_i32[0];
          }
          else if ( v264 != 20 )
          {
LABEL_80:
            sub_721090(v14);
          }
        }
        v7 = (__int64)v429;
        v107 = (unsigned __int64)&v407;
        sub_7022F0(
          (unsigned int)&v407,
          (unsigned int)v429,
          v247,
          1,
          0,
          0,
          0,
          1,
          (__int64)&dword_4F077C8,
          (__int64)&dword_4F063F8,
          (__int64)&dword_4F077C8,
          (__int64)&v386,
          (__int64)&v362,
          (__int64)&v364);
        if ( !(_DWORD)v362 )
        {
          v7 = (__int64)&v386;
          v107 = 2;
          sub_6E26D0(2, &v386);
        }
LABEL_145:
        sub_7B8B50(v107, v7, v109, v108);
LABEL_20:
        if ( v355 )
        {
          sub_68B170((__int64)&v386);
          *(_BYTE *)(qword_4D03C50 + 20LL) = (v353 << 7) | *(_BYTE *)(qword_4D03C50 + 20LL) & 0x7F;
        }
LABEL_22:
        if ( dword_4F077C4 == 1 )
          goto LABEL_39;
        while ( 1 )
        {
          v14 = word_4F06418[0];
          v15 = v4;
          if ( sub_688800(word_4F06418[0], v4, v5) )
            break;
          v19 = word_4F06418[0];
          if ( (v5 & 0x1000) != 0 && (unsigned __int16)(word_4F06418[0] - 52) > 1u )
          {
            if ( (v5 & 0x2000) != 0 )
            {
              if ( word_4F06418[0] == 25 )
              {
                if ( dword_4D043F8 )
                {
                  v15 = 0;
                  if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
                    break;
                  v19 = word_4F06418[0];
                  goto LABEL_94;
                }
              }
              else
              {
LABEL_94:
                if ( (v19 == 27 || v19 == 73) && sub_6879B0() )
                  break;
              }
              v15 = (__int64)&dword_4F063F8;
              v14 = 3039;
              sub_6851C0(0xBDFu, &dword_4F063F8);
              v19 = word_4F06418[0];
            }
            BYTE1(v5) &= ~0x10u;
          }
          if ( (*(_BYTE *)(qword_4D03C50 + 21LL) & 1) != 0 && v19 == 42 )
          {
            v15 = 0;
            v14 = 0;
            if ( (unsigned __int16)sub_7BE840(0, 0) == 44 )
              break;
            v19 = word_4F06418[0];
          }
          if ( v387.m128i_i8[0] != 6 || v19 == 27 || v19 == 72 )
            goto LABEL_29;
          if ( !unk_4D04878 || v19 != 43 )
          {
            v54 = v393.m128i_i64[1];
            sub_886080(v393.m128i_i64[1]);
            if ( (unsigned int)sub_6E5430(v54, v15, v55, v56, v57, v58) )
            {
              v15 = *(_QWORD *)(*(_QWORD *)v393.m128i_i64[1] + 8LL);
              sub_6851F0(0x14u, v15);
            }
            v14 = (__int64)&v386;
            sub_6E6260(&v386);
            v19 = word_4F06418[0];
LABEL_29:
            if ( (v387.m128i_i8[2] & 1) != 0 && v19 != 27 )
            {
LABEL_89:
              v15 = (__int64)&v407;
              v14 = (__int64)&v386;
              if ( !(unsigned int)sub_6F68A0(&v386, &v407) )
              {
                v15 = (__int64)&v386;
                v14 = 300;
                sub_6E68E0(300, &v386);
                v387.m128i_i8[2] &= ~1u;
              }
              v19 = word_4F06418[0];
            }
            v20 = _mm_loadu_si128(&v387);
            v21 = _mm_loadu_si128(&v388);
            v22 = _mm_loadu_si128(&v389);
            v23 = _mm_loadu_si128((const __m128i *)&v390);
            v24 = _mm_loadu_si128((const __m128i *)&v390.m256i_u64[2]);
            v364 = _mm_loadu_si128(&v386);
            v25 = _mm_loadu_si128(&v391);
            v26 = _mm_loadu_si128(&v392);
            v365 = v20;
            v27 = v387.m128i_u8[0];
            v366 = v21;
            v28 = _mm_loadu_si128(&v393);
            v367 = v22;
            v368 = v23;
            v369 = v24;
            v370 = v25;
            v371 = v26;
            v372 = v28;
            if ( v387.m128i_i8[0] != 2 )
            {
              if ( v387.m128i_i8[0] == 5 || v387.m128i_i8[0] == 1 )
              {
                v27 = v394.m128i_i64[0];
                v29 = v19 - 25;
                v373.m128i_i64[0] = v394.m128i_i64[0];
                if ( v29 > 0x7Bu )
                  goto LABEL_80;
              }
              else
              {
LABEL_34:
                v29 = v19 - 25;
              }
              switch ( v29 )
              {
                case 0u:
                  sub_6D30E0(&v364, 0, 0, &v386);
                  goto LABEL_22;
                case 2u:
                case 0x2Fu:
                  sub_6C4D80(&v364, &v407, &v386);
                  goto LABEL_22;
                case 4u:
                case 5u:
                  sub_6D7FC0(&v364, 0, 0, 0, &v386, &v407);
                  goto LABEL_22;
                case 6u:
                case 7u:
                  sub_69D0B0((__int64)&v364, 0, (__int64)&v386, v16, v17, v18);
                  goto LABEL_22;
                case 8u:
                case 0x19u:
                case 0x1Au:
                  sub_6B38B0(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 9u:
                case 0xEu:
                case 0xFu:
                  sub_6B1D00(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0xAu:
                case 0xBu:
                  sub_6B21E0(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x10u:
                case 0x11u:
                  sub_6B2B40(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x12u:
                case 0x13u:
                case 0x14u:
                case 0x15u:
                  goto LABEL_53;
                case 0x16u:
                case 0x17u:
                  sub_6B3030(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x18u:
                  v362 = *(_QWORD *)&dword_4F063F8;
                  v354 = dword_4F06650[0];
                  sub_7B8B50(v14, v15, v27, v16);
                  sub_69ED20(v429, 0, 10, 0);
                  sub_68FEF0(&v364, v429, &v362, v354, 0, (__int64)&v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x1Bu:
                case 0x1Cu:
                  sub_6B3BD0(&v364, 0, v5 & 0x3000, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x1Du:
                  v53 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  if ( (v53 & 8) != 0 )
                  {
                    *(_BYTE *)(qword_4D03C50 + 20LL) = v53 & 0xF7;
                    sub_6B4800(&v364, 0, &v386);
                    v30 = qword_4D03C50;
                    v31 = *(_BYTE *)(qword_4D03C50 + 20LL) | 8;
                    *(_BYTE *)(qword_4D03C50 + 20LL) = v31;
                  }
                  else
                  {
                    sub_6B4800(&v364, 0, &v386);
                    v30 = qword_4D03C50;
                    v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  }
                  goto LABEL_38;
                case 0x1Fu:
                  sub_6CEC90(&v364, 0, v429, &v386);
                  if ( v429[0].m128i_i32[0] )
                    goto LABEL_57;
                  goto LABEL_37;
                case 0x20u:
                case 0x21u:
                case 0x22u:
                case 0x23u:
                case 0x24u:
                case 0x25u:
                case 0x26u:
                case 0x27u:
                case 0x28u:
                case 0x29u:
                  sub_6CF140(&v364, 0, v429, &v386);
                  if ( !v429[0].m128i_i32[0] )
                    goto LABEL_37;
LABEL_57:
                  v15 = 2;
                  if ( !sub_688800(word_4F06418[0], 2, v5) )
                    goto LABEL_58;
LABEL_37:
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  break;
                case 0x2Au:
                  if ( dword_4F077C4 == 2 && v352 )
                    sub_6E5C80((unsigned int)(unk_4F07778 > 202001) + 4, 3008, &dword_4F063F8);
                  sub_6B70D0(&v364, 0, &v386);
                  goto LABEL_22;
                case 0x2Du:
                case 0x2Eu:
                  sub_6B3200(&v364, 0, &v386);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                case 0x7Au:
                case 0x7Bu:
                  sub_6B0A80(&v364, 0, 0, &v386, &v407);
                  v30 = qword_4D03C50;
                  v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
                  goto LABEL_38;
                default:
                  goto LABEL_80;
              }
              goto LABEL_38;
            }
            v59 = _mm_loadu_si128(&v395);
            v60 = _mm_loadu_si128(&v396);
            v61 = _mm_loadu_si128(&v397);
            v62 = _mm_loadu_si128(&v398);
            v63 = _mm_loadu_si128(&v399);
            v373 = _mm_loadu_si128(&v394);
            v64 = _mm_loadu_si128(&v400);
            v65 = _mm_loadu_si128(&v401);
            v374 = v59;
            v375 = v60;
            v66 = _mm_loadu_si128(&v402);
            v67 = _mm_loadu_si128(&v403);
            v376 = v61;
            v68 = _mm_loadu_si128(&v404);
            v377 = v62;
            v69 = _mm_loadu_si128(&v405);
            v378 = v63;
            v70 = _mm_loadu_si128(&v406);
            v379 = v64;
            v380 = v65;
            v381 = v66;
            v382 = v67;
            v383 = v68;
            v384 = v69;
            v385 = v70;
            goto LABEL_34;
          }
          if ( (v387.m128i_i8[2] & 1) != 0 )
            goto LABEL_89;
          v32 = _mm_loadu_si128(&v388);
          v33 = _mm_loadu_si128(&v389);
          v34 = _mm_loadu_si128((const __m128i *)&v390);
          v364 = _mm_loadu_si128(&v386);
          v35 = _mm_loadu_si128(&v387);
          v36 = _mm_loadu_si128((const __m128i *)&v390.m256i_u64[2]);
          v37 = _mm_loadu_si128(&v391);
          v38 = _mm_loadu_si128(&v392);
          v366 = v32;
          v365 = v35;
          v39 = _mm_loadu_si128(&v393);
          v367 = v33;
          v368 = v34;
          v369 = v36;
          v370 = v37;
          v371 = v38;
          v372 = v39;
LABEL_53:
          sub_6B2F50(&v364, 0, &v386);
          v30 = qword_4D03C50;
          v31 = *(_BYTE *)(qword_4D03C50 + 20LL);
LABEL_38:
          *(_BYTE *)(v30 + 20) = v31 & 0xF7;
          if ( dword_4F077C4 == 1 )
LABEL_39:
            sub_6886E0();
        }
LABEL_58:
        *(_QWORD *)dword_4F07508 = v361;
        if ( v387.m128i_i8[0] == 6 )
        {
          v102 = v393.m128i_i64[1];
          sub_886080(v393.m128i_i64[1]);
          if ( (unsigned int)sub_6E5430(v102, v15, v103, v104, v105, v106) )
            sub_6851F0(0x14u, *(_QWORD *)(*(_QWORD *)v393.m128i_i64[1] + 8LL));
          sub_6E6260(&v386);
        }
        if ( (v387.m128i_i8[2] & 1) == 0 )
        {
LABEL_107:
          v48 = 0;
          goto LABEL_65;
        }
        if ( (v5 & 0x10) == 0 )
        {
          if ( !(unsigned int)sub_6F68A0(&v386, &v407) )
          {
            sub_6E68E0(300, &v386);
            v387.m128i_i8[2] &= ~1u;
          }
          goto LABEL_107;
        }
        v40 = _mm_loadu_si128(&v408);
        v41 = _mm_loadu_si128(&v409);
        *a2 = _mm_loadu_si128(&v407);
        v42 = _mm_loadu_si128(&v410);
        v43 = _mm_loadu_si128(&v411);
        a2[1] = v40;
        v44 = _mm_loadu_si128(&v412);
        v45 = _mm_loadu_si128(&v413);
        v46 = _mm_loadu_si128(&v414);
        v47 = _mm_loadu_si128(&v415);
        a2[2] = v41;
        a2[3] = v42;
        a2[4] = v43;
        a2[5] = v44;
        a2[6] = v45;
        a2[7] = v46;
        a2[8] = v47;
        if ( v408.m128i_i8[0] == 2 )
        {
          v89 = _mm_loadu_si128(&v417);
          v90 = _mm_loadu_si128(&v418);
          v91 = _mm_loadu_si128(&v419);
          v92 = _mm_loadu_si128(&v420);
          a2[9] = _mm_loadu_si128(&v416);
          v93 = _mm_loadu_si128(&v421);
          v94 = _mm_loadu_si128(&v422);
          a2[10] = v89;
          v95 = _mm_loadu_si128(&v426);
          v96 = _mm_loadu_si128(&v423);
          a2[11] = v90;
          a2[12] = v91;
          v97 = _mm_loadu_si128(&v424);
          v98 = _mm_loadu_si128(&v425);
          a2[13] = v92;
          v99 = _mm_loadu_si128(&v427);
          a2[14] = v93;
          v100 = _mm_loadu_si128(&v428);
          a2[15] = v94;
          a2[16] = v96;
          a2[17] = v97;
          a2[18] = v98;
          a2[19] = v95;
          a2[20] = v99;
          a2[21] = v100;
        }
        else if ( v408.m128i_i8[0] == 5 || v408.m128i_i8[0] == 1 )
        {
          a2[9].m128i_i64[0] = v416.m128i_i64[0];
        }
        v48 = a2[5].m128i_i64[1];
LABEL_65:
        *(__m128i *)v356 = _mm_loadu_si128(&v386);
        *((__m128i *)v356 + 1) = _mm_loadu_si128(&v387);
        *((__m128i *)v356 + 2) = _mm_loadu_si128(&v388);
        *((__m128i *)v356 + 3) = _mm_loadu_si128(&v389);
        *((__m128i *)v356 + 4) = _mm_loadu_si128((const __m128i *)&v390);
        *((__m128i *)v356 + 5) = _mm_loadu_si128((const __m128i *)&v390.m256i_u64[2]);
        *((__m128i *)v356 + 6) = _mm_loadu_si128(&v391);
        *((__m128i *)v356 + 7) = _mm_loadu_si128(&v392);
        *((__m128i *)v356 + 8) = _mm_loadu_si128(&v393);
        if ( v387.m128i_i8[0] == 2 )
        {
          v77 = _mm_loadu_si128(&v395);
          v78 = _mm_loadu_si128(&v396);
          v79 = _mm_loadu_si128(&v397);
          v80 = _mm_loadu_si128(&v398);
          *((__m128i *)v356 + 9) = _mm_loadu_si128(&v394);
          v81 = _mm_loadu_si128(&v399);
          v82 = _mm_loadu_si128(&v400);
          *((__m128i *)v356 + 10) = v77;
          v83 = _mm_loadu_si128(&v401);
          v84 = _mm_loadu_si128(&v402);
          *((__m128i *)v356 + 11) = v78;
          *((__m128i *)v356 + 12) = v79;
          v85 = _mm_loadu_si128(&v403);
          v86 = _mm_loadu_si128(&v404);
          *((__m128i *)v356 + 13) = v80;
          v87 = _mm_loadu_si128(&v405);
          *((__m128i *)v356 + 14) = v81;
          v88 = _mm_loadu_si128(&v406);
          *((__m128i *)v356 + 15) = v82;
          *((__m128i *)v356 + 16) = v83;
          *((__m128i *)v356 + 17) = v84;
          *((__m128i *)v356 + 18) = v85;
          *((__m128i *)v356 + 19) = v86;
          *((__m128i *)v356 + 20) = v87;
          *((__m128i *)v356 + 21) = v88;
          goto LABEL_68;
        }
        if ( v387.m128i_i8[0] == 5 || v387.m128i_i8[0] == 1 )
        {
          v356[18] = v394.m128i_i64[0];
          if ( !v333 )
            goto LABEL_69;
LABEL_109:
          v71 = 4;
          v72 = (__int64)v356;
          sub_6F69D0(v356, 4);
          v73 = *((_BYTE *)v356 + 16);
          if ( v73 == 2 )
          {
            if ( *((_BYTE *)v356 + 17) == 2 || (v72 = (__int64)v356, (unsigned int)sub_6ED0A0(v356)) )
            {
LABEL_116:
              sub_6E2B30(v72, v71);
              goto LABEL_69;
            }
            v73 = *((_BYTE *)v356 + 16);
          }
          if ( !v73 )
            goto LABEL_115;
          v74 = *v356;
          v75 = *(_BYTE *)(*v356 + 140LL);
          if ( v75 == 12 )
          {
            v76 = *v356;
            do
            {
              v76 = *(_QWORD *)(v76 + 160);
              v75 = *(_BYTE *)(v76 + 140);
            }
            while ( v75 == 12 );
          }
          if ( v75 )
          {
            if ( (unsigned int)sub_8D3350(v74) || (v74 = *v356, (unsigned int)sub_8D3D40(*v356)) )
            {
              v124 = *v356;
              v429[0].m128i_i64[0] = sub_724DC0(v74, 4, v114, v115, v116, v117);
              sub_72BB40(v124, v429[0].m128i_i64[0]);
              v71 = (__int64)v356;
              sub_6E6A50(v429[0].m128i_i64[0], v356);
              v72 = (__int64)v429;
              sub_724E30(v429);
              *(_OWORD *)((char *)v356 + 68) = *(_OWORD *)((char *)v390.m256i_i64 + 4);
            }
            else
            {
              for ( m = *v356; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                ;
              if ( (unsigned int)sub_8D2600(m) )
              {
                v429[0].m128i_i64[0] = sub_724DC0(m, 4, v119, v120, v121, v122);
                sub_724C70(v429[0].m128i_i64[0], 14);
                v123 = v429[0].m128i_i64[0];
                v71 = (__int64)v356;
                *(_QWORD *)(v429[0].m128i_i64[0] + 128) = *v356;
                sub_6E6A50(v123, v356);
                v72 = (__int64)v429;
                *(_OWORD *)((char *)v356 + 68) = *(_OWORD *)((char *)v390.m256i_i64 + 4);
                sub_724E30(v429);
              }
              else
              {
                v71 = (__int64)v356;
                v72 = 28;
                sub_6E68E0(28, v356);
              }
            }
          }
          else
          {
LABEL_115:
            v72 = (__int64)v356;
            sub_6E6870(v356);
          }
          goto LABEL_116;
        }
LABEL_68:
        if ( v333 )
          goto LABEL_109;
LABEL_69:
        sub_6E16F0(v356[11], v48);
        v49 = v334;
        if ( v334 )
        {
          do
          {
            v50 = v49;
            v49 = *(_QWORD *)(v49 + 40);
          }
          while ( v49 );
          *(_QWORD *)(v50 + 40) = unk_4D03C48;
          unk_4D03C48 = v334;
        }
LABEL_72:
        v51 = *(unsigned __int16 *)(qword_4D03C50 + 20LL);
        LOWORD(v51) = v51 & 0xFEF7;
        result = v332 & 0x108 | v51;
        *(_WORD *)(qword_4D03C50 + 20LL) = result;
        return result;
      case 0x12u:
      case 0x50u:
      case 0x55u:
      case 0x59u:
      case 0x5Du:
      case 0x5Eu:
      case 0x61u:
      case 0x62u:
      case 0x69u:
      case 0x6Au:
      case 0x7Eu:
      case 0x7Fu:
      case 0x80u:
      case 0x85u:
      case 0x86u:
      case 0x87u:
      case 0x88u:
      case 0xA5u:
      case 0xB4u:
      case 0xB7u:
      case 0xB9u:
      case 0xBDu:
      case 0xECu:
      case 0xEFu:
      case 0x14Bu:
      case 0x14Cu:
      case 0x14Du:
      case 0x14Eu:
      case 0x14Fu:
      case 0x153u:
      case 0x154u:
      case 0x155u:
      case 0x156u:
      case 0x157u:
      case 0x158u:
      case 0x159u:
      case 0x15Au:
      case 0x15Bu:
      case 0x15Cu:
      case 0x15Du:
      case 0x15Eu:
      case 0x15Fu:
      case 0x160u:
      case 0x161u:
      case 0x162u:
        if ( dword_4F077C4 != 2 )
          goto LABEL_119;
        if ( v9 == 183 )
        {
          if ( dword_4D041A8 && (unsigned __int16)sub_7BE840(0, 0) == 25 )
            v407.m128i_i64[0] = sub_6B8420(0, 0);
          else
            sub_671BC0((__int64)&v407, (__int64)v429, 0, 0, 0, 0);
        }
        else if ( v9 == 185 )
        {
          v407.m128i_i64[0] = sub_6B7D60(0, 0);
        }
        else
        {
LABEL_156:
          if ( v9 == 18 )
          {
            v407.m128i_i64[0] = unk_4D04A38;
            sub_7B8B50(a1, v7, &dword_4F077C4, a4);
          }
          else if ( v9 == 77 )
          {
LABEL_233:
            v407.m128i_i64[0] = sub_72B6D0(&dword_4F063F8, 0);
            sub_7B8B50(&dword_4F063F8, 0, v148, v149);
          }
          else if ( v9 == 236 || (unsigned __int16)(v9 - 339) <= 0xFu )
          {
            v407.m128i_i64[0] = sub_6911B0(a1, (unsigned int *)v7);
          }
          else if ( v9 == 189 )
          {
            v407.m128i_i64[0] = sub_6B8C50(0, 0);
          }
          else
          {
            v255 = &dword_4F077BC;
            if ( dword_4F077BC
              && (v255 = (unsigned int *)&qword_4F077A8, qword_4F077A8 <= 0x76BFu)
              && ((LOBYTE(v255) = v9 == 104, LOBYTE(a4) = v9 == 104 || v9 == 101, (_BYTE)a4) || ((v9 - 87) & 0xFFBF) == 0) )
            {
              v407.m128i_i64[0] = sub_64ED10(a1, v7, v255, a4, v8, "__is_signed");
              if ( (*(_BYTE *)(v407.m128i_i64[0] + 140) & 0xFB) == 8
                && (unsigned int)sub_8D4C10(v407.m128i_i64[0], dword_4F077C4 != 2) )
              {
                sub_69A8C0(1596, dword_4F07508, v284, v285, v286, v287);
              }
            }
            else
            {
              v407.m128i_i64[0] = sub_64EDF0(a1, v7, (__int64)v255, a4, v8, (__int64)"__is_signed");
              sub_7B8B50(a1, v7, v282, v283);
            }
          }
        }
        *(_QWORD *)dword_4F07508 = v361;
        if ( word_4F06418[0] == 27 || dword_4D04428 && word_4F06418[0] == 73 )
          sub_6CA0E0(0, 0, 0, 0, v407.m128i_i32[0], (unsigned int)&v361, (__int64)&v386, v5);
        else
          sub_6E6890(254, &v386);
        goto LABEL_20;
      case 0x19u:
        if ( dword_4D041A8 )
        {
          v7 = 0;
          a1 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) == 55 )
          {
            sub_6D7930(&v386);
            goto LABEL_20;
          }
        }
        if ( !dword_4D0448C )
          goto LABEL_119;
        v127 = qword_4D03C50;
        v407.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        v128 = *(unsigned __int8 *)(qword_4D03C50 + 19LL);
        if ( (v128 & 0x40) != 0 )
        {
          if ( (unsigned int)sub_6E5430(a1, v7, v128, a4, qword_4D03C50, v10) )
            sub_6851C0(0x6E3u, &v407);
          goto LABEL_515;
        }
        if ( unk_4D041C4 )
          goto LABEL_522;
        v129 = *(_BYTE *)(qword_4D03C50 + 17LL);
        if ( (v129 & 2) != 0 )
        {
          v278 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( (*(_BYTE *)(v278 + 6) & 0x20) != 0 )
          {
            if ( (unsigned int)sub_6E5430(a1, v7, v128, qword_4F04C68, qword_4D03C50, v10) )
              sub_6851C0(0xB32u, &v407);
            goto LABEL_515;
          }
          if ( (*(_BYTE *)(v278 + 12) & 1) != 0
            || (v128 & 2) != 0
            && ((v128 = *(unsigned __int8 *)(v278 + 4), (_BYTE)v128 == 1)
             || (_BYTE)v128 == 8
             && (*(_BYTE *)(**(_QWORD **)(v278 + 616) + 122LL) & 4) == 0
             && (*(_BYTE *)(qword_4D03C50 + 18LL) & 1) == 0) )
          {
            if ( (unsigned int)sub_6E5430(a1, v7, v128, qword_4F04C68, qword_4D03C50, v10) )
              sub_6851C0(0xB50u, &v407);
            goto LABEL_515;
          }
LABEL_522:
          if ( dword_4D041E0
            || dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x1387Fu
            || (v329 = qword_4D03C50, v279 = sub_6E91E0(1763, &v407), v127 = v329, !v279) )
          {
LABEL_210:
            v130 = v429;
            v317 = v127;
            v131 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
            sub_6E1E00(v131, v429, 0, 1);
            v429[7].m128i_i64[0] = 0;
            v132 = sub_609F00();
            v324 = v132;
            if ( !v132 )
            {
              sub_6E6260(&v386);
              sub_6E2B30(&v386, v429);
              v390.m256i_i32[1] = v407.m128i_i32[0];
              v390.m256i_i16[4] = v407.m128i_i16[2];
              *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
              *(__int64 *)((char *)&v390.m256i_i64[1] + 4) = unk_4F061D8;
              sub_6E3280(&v386, &v407);
              goto LABEL_20;
            }
            if ( unk_4D041C4 && (*(_BYTE *)(*(_QWORD *)(v132 + 16) + 193LL) & 2) == 0 )
            {
              v135 = qword_4F04C68;
              v136 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              if ( (*(_BYTE *)(v136 + 6) & 0x20) != 0 )
              {
                if ( (unsigned int)sub_6E5430(v131, v429, qword_4F04C68, v133, v317, v134) )
                {
                  v130 = &v407;
                  sub_6851C0(0xB32u, &v407);
                }
                goto LABEL_547;
              }
              if ( (*(_BYTE *)(v136 + 12) & 1) != 0
                || (*(_BYTE *)(v317 + 19) & 2) != 0
                && ((v135 = (_QWORD *)*(unsigned __int8 *)(v136 + 4), (_BYTE)v135 == 1)
                 || (_BYTE)v135 == 8
                 && (v281 = *(_QWORD *)(v136 + 616)) != 0
                 && (*(_BYTE *)(*(_QWORD *)v281 + 122LL) & 4) == 0
                 && (*(_BYTE *)(v317 + 18) & 1) == 0) )
              {
                if ( (unsigned int)sub_6E5430(v131, v429, v135, v133, v317, v134) )
                {
                  v130 = &v407;
                  sub_6851C0(0xB50u, &v407);
                }
LABEL_547:
                v142 = &v386;
                sub_6E6260(&v386);
                goto LABEL_219;
              }
            }
            v137 = (__m128i *)sub_726700(6);
            v138 = *(_QWORD *)(v324 + 8);
            v137[4].m128i_i64[0] = v324;
            v318 = v137;
            v137->m128i_i64[0] = v138;
            v130 = &v386;
            v137[3].m128i_i64[1] = sub_6892A0((_QWORD *)v324, (__int64)v429, v139, v140, (__int64)v137, v141);
            v142 = v318;
            sub_6E70E0(v318, &v386);
LABEL_219:
            sub_6E2B30(v142, v130);
            v390.m256i_i32[1] = v407.m128i_i32[0];
            v390.m256i_i16[4] = v407.m128i_i16[2];
            *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
            *(__int64 *)((char *)&v390.m256i_i64[1] + 4) = unk_4F061D8;
            sub_6E3280(&v386, &v407);
            goto LABEL_220;
          }
        }
        else
        {
          if ( (*(_BYTE *)(qword_4D03C50 + 22LL) & 1) != 0 )
          {
            *(_BYTE *)(qword_4D03C50 + 17LL) = v129 | 0x10;
            *(_QWORD *)(v127 + 96) = v407.m128i_i64[0];
            goto LABEL_210;
          }
          if ( (unsigned int)sub_6E5430(a1, v7, v128, a4, qword_4D03C50, v10) )
            sub_6851C0(0x6F1u, &v407);
        }
LABEL_515:
        sub_6E1E00(*(unsigned __int8 *)(qword_4D03C50 + 16LL), v429, 0, 1);
        v429[7].m128i_i64[0] = 0;
        v324 = sub_609F00();
        sub_6E6260(&v386);
        sub_6E2B30(&v386, v429);
        v390.m256i_i32[1] = v407.m128i_i32[0];
        v390.m256i_i16[4] = v407.m128i_i16[2];
        *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
        *(__int64 *)((char *)&v390.m256i_i64[1] + 4) = unk_4F061D8;
        sub_6E3280(&v386, &v407);
        if ( !v324 )
          goto LABEL_20;
LABEL_220:
        v143 = *(__int64 ***)v324;
        if ( *(_QWORD *)v324 )
        {
          v337 = v5;
          v5 = 0;
          do
          {
            if ( ((_BYTE)v143[4] & 1) != 0 )
            {
              sub_643C70((__int64)v143[2]);
              v143[2] = 0;
            }
            v143 = (__int64 **)*v143;
            ++v5;
          }
          while ( v143 );
          v248 = v5;
          LOWORD(v5) = v337;
          v249 = *(_BYTE *)(v324 + 25);
          v250 = v249 >> 3;
          v251 = v249 >> 4;
          if ( ((v251 | v250) & 1) == 0 )
            goto LABEL_20;
          if ( (unsigned int)v248 > 0x3FE )
          {
            sub_684AA0(7u, 0xE0Bu, (_DWORD *)(v324 + 44));
            goto LABEL_20;
          }
          v280 = v251 & 1;
        }
        else
        {
          v248 = 0;
          v280 = (*(_BYTE *)(v324 + 25) & 0x10) != 0;
          if ( !(v280 | ((*(_BYTE *)(v324 + 25) & 8) != 0)) )
            goto LABEL_20;
        }
        sub_826910(v280, v248);
        goto LABEL_20;
      case 0x1Bu:
        goto LABEL_19;
      case 0x1Fu:
      case 0x20u:
        sub_6A4340(0, &v386);
        goto LABEL_20;
      case 0x21u:
        sub_6A49A0(0, &v386, v5 & 0x100);
        goto LABEL_20;
      case 0x22u:
        sub_6A59A0(0, &v386);
        goto LABEL_20;
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
        sub_6A5FF0(0, &v386);
        goto LABEL_20;
      case 0x32u:
        if ( !dword_4D041A8 )
          goto LABEL_119;
        v362 = *(_QWORD *)&dword_4F063F8;
        v360 = sub_724DC0(a1, v7, v9, a4, v8, "__is_signed");
        sub_724C70(v360, 15);
        sub_68B050((unsigned int)dword_4F04C64, (__int64)&v357, &v359);
        sub_6E2140(5, &v407, 0, 0, 0);
        v358 = 0;
        *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x28u;
        sub_7B8B50(5, &v407, v150, v151);
        v364.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        v152 = word_4F06418[0];
        if ( word_4F06418[0] != 146 )
          goto LABEL_236;
        v153 = 0;
        v252 = sub_7BE840(0, 0);
        if ( v252 != 1 && v252 != 155 && v252 != 152 )
        {
          v253 = v360;
          *(_BYTE *)(v360 + 176) = 23;
          v254 = unk_4F07288;
          goto LABEL_414;
        }
        v152 = word_4F06418[0];
LABEL_236:
        if ( dword_4F077C4 == 2 )
        {
          if ( (v152 != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(16385, 0) )
            goto LABEL_242;
        }
        else if ( v152 != 1 )
        {
          goto LABEL_242;
        }
        v153 = 16385;
        v154 = sub_7BF130(16385, 0, &v358);
        v156 = v154;
        if ( v154 )
        {
          v157 = *(_BYTE *)(v154 + 80);
          if ( v157 == 23 )
          {
            v254 = *(_QWORD *)(v156 + 88);
            v253 = v360;
            if ( (*(_BYTE *)(v254 + 124) & 1) != 0 )
            {
              *(_BYTE *)(v360 + 176) = 28;
              *(_QWORD *)(v253 + 184) = v254;
              goto LABEL_415;
            }
            *(_BYTE *)(v360 + 176) = 23;
            v254 = *(_QWORD *)(v254 + 128);
LABEL_414:
            *(_QWORD *)(v253 + 184) = v254;
LABEL_415:
            v158 = (__m128i *)(unsigned int)qword_4F063F0;
            v325 = qword_4F063F0;
            v338 = WORD2(qword_4F063F0);
            sub_7B8B50(v153, (unsigned int)qword_4F063F0, v254, v155);
            goto LABEL_416;
          }
          if ( (unsigned __int8)(v157 - 21) <= 1u || v157 == 19 )
          {
            v153 = 0;
            v308 = sub_7BE840(0, 0);
            if ( v308 != 73 && v308 != 27 )
            {
              v254 = *(_QWORD *)(*(_QWORD *)(v156 + 88) + 104LL);
              v253 = v360;
              *(_BYTE *)(v360 + 176) = 59;
              goto LABEL_414;
            }
          }
        }
LABEL_242:
        if ( (unsigned int)sub_679C10(1u) )
        {
          v158 = v429;
          memset(v429, 0, 0x1D8u);
          v429[9].m128i_i64[1] = (__int64)v429;
          v429[1].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
            v429[11].m128i_i8[2] |= 1u;
          sub_65C7C0((__int64)v429);
          if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
            sub_64EC60((__int64)v429);
          v159 = v360;
          *(_BYTE *)(v360 + 176) = 6;
          v153 = v429[18].m128i_i64[0];
          *(_QWORD *)(v159 + 184) = v429[18].m128i_i64[0];
          v160 = sub_8DC060(v153);
          v325 = unk_4F061D8;
          v338 = unk_4F061DC;
          if ( v160 )
            goto LABEL_250;
          goto LABEL_416;
        }
        v158 = 0;
        sub_69ED20(v429, 0, 18, 0x8000);
        v292 = v429[8].m128i_i64[1];
        if ( v429[8].m128i_i64[1] )
        {
          v293 = *(_BYTE *)(v429[8].m128i_i64[1] + 80);
          if ( v293 == 8 )
          {
            v298 = v360;
            *(_BYTE *)(v360 + 176) = 8;
            *(_QWORD *)(v298 + 184) = *(_QWORD *)(v292 + 88);
          }
          else if ( (unsigned __int8)(v293 - 10) <= 1u )
          {
            v296 = v360;
            *(_BYTE *)(v360 + 176) = 11;
            *(_QWORD *)(v296 + 184) = *(_QWORD *)(v292 + 88);
          }
          else if ( (unsigned __int8)(v293 - 19) > 3u )
          {
            if ( v293 == 17 )
            {
              v158 = &v364;
              sub_6851C0(0xD2Cu, &v364);
              v297 = v360;
              *(_BYTE *)(v360 + 176) = 0;
              *(_QWORD *)(v297 + 184) = 0;
            }
          }
          else
          {
            v294 = v360;
            *(_BYTE *)(v360 + 176) = 59;
            *(_QWORD *)(v294 + 184) = *(_QWORD *)(*(_QWORD *)(v292 + 88) + 104LL);
          }
          goto LABEL_560;
        }
        v386.m128i_i64[0] = sub_724DC0(v429, 0, v288, v289, v290, v291);
        v158 = (__m128i *)v386.m128i_i64[0];
        if ( v429[1].m128i_i8[0] == 2 )
        {
          sub_72A510(&v429[9], v386.m128i_i64[0]);
        }
        else
        {
          v158 = 0;
          v299 = sub_6F6F40(v429, 0);
          v300 = *(_BYTE *)(v299 + 24);
          if ( v300 == 3 )
          {
            if ( (*(_BYTE *)(v299 + 25) & 1) != 0 )
              goto LABEL_574;
          }
          else if ( v300 == 20 )
          {
LABEL_574:
            v301 = v360;
            *(_BYTE *)(v360 + 176) = 13;
            *(_QWORD *)(v301 + 184) = v299;
LABEL_575:
            v302 = v360;
            v303 = *(_QWORD *)(v360 + 184);
            if ( (*(_BYTE *)(v303 - 8) & 1) == 0 )
            {
              v158 = (__m128i *)qword_4F04C68;
              *(_DWORD *)(v360 + 192) = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
            }
            if ( *(_BYTE *)(v302 + 176) == 13 )
            {
              v307 = *(_BYTE *)(v303 + 24);
              if ( v307 == 4 )
              {
                *(_BYTE *)(v302 + 176) = 8;
                *(_QWORD *)(v302 + 184) = *(_QWORD *)(v303 + 56);
              }
              else if ( v307 > 4u )
              {
                if ( v307 == 20 )
                {
                  *(_BYTE *)(v302 + 176) = 11;
                  *(_QWORD *)(v302 + 184) = *(_QWORD *)(v303 + 56);
                }
              }
              else if ( v307 == 2 )
              {
                *(_BYTE *)(v302 + 176) = 2;
                *(_QWORD *)(v302 + 184) = *(_QWORD *)(v303 + 56);
              }
              else if ( v307 == 3 )
              {
                *(_BYTE *)(v302 + 176) = 7;
                *(_QWORD *)(v302 + 184) = *(_QWORD *)(v303 + 56);
              }
              if ( (*(_BYTE *)(*(_QWORD *)(v302 + 184) - 8LL) & 1) != 0 )
                *(_DWORD *)(v302 + 192) = 0;
            }
            if ( v386.m128i_i64[0] )
              sub_724E30(&v386);
LABEL_560:
            v153 = (__int64)v429;
            v295 = sub_696840((__int64)v429);
            v325 = unk_4F061D8;
            v338 = unk_4F061DC;
            if ( v295 )
            {
LABEL_250:
              v161 = *(_QWORD *)&dword_4D03B80;
              goto LABEL_251;
            }
LABEL_416:
            v161 = sub_72CD60(v153, v158);
LABEL_251:
            v162 = v360;
            *(_QWORD *)(v360 + 128) = v161;
            sub_6E6A50(v162, &v386);
            v390.m256i_i32[1] = v362;
            v390.m256i_i16[4] = WORD2(v362);
            *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
            v390.m256i_i32[3] = v325;
            v390.m256i_i16[8] = v338;
            unk_4F061D8 = *(__int64 *)((char *)&v390.m256i_i64[1] + 4);
            sub_6E3280(&v386, &v362);
            sub_6E3BA0(&v386, &v362, 0, &v364);
            sub_6E2B30(&v386, &v362);
            v163 = v359;
            sub_729730(v357);
            qword_4F06BC0 = v163;
            sub_724E30(&v360);
            goto LABEL_20;
          }
          v158 = (__m128i *)v386.m128i_i64[0];
          if ( !(unsigned int)sub_716120(v299, v386.m128i_i64[0]) )
            goto LABEL_574;
        }
        v306 = v360;
        *(_BYTE *)(v360 + 176) = 2;
        *(_QWORD *)(v306 + 184) = sub_724E50(&v386, v158, v306, v304, v305);
        goto LABEL_575;
      case 0x34u:
        if ( !HIDWORD(qword_4F077B4) || (unsigned __int16)sub_7BE840(0, 0) != 1 )
          goto LABEL_119;
        sub_68CED0((__int64)&v386, 0, v144, v145, v146, v147);
        goto LABEL_20;
      case 0x4Du:
        if ( dword_4F077C4 == 2 && unk_4F07778 > 202301
          || dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1D4BFu && dword_4F077C4 == 2 )
        {
          goto LABEL_233;
        }
        goto LABEL_119;
      case 0x57u:
      case 0x65u:
      case 0x68u:
      case 0x97u:
        a1 = dword_4F077BC;
        if ( dword_4F077BC && qword_4F077A8 <= 0x76BFu && dword_4F077C4 == 2 )
          goto LABEL_156;
        goto LABEL_119;
      case 0x63u:
      case 0x11Cu:
        sub_6A6540(0, &v386);
        goto LABEL_20;
      case 0x6Du:
        sub_6DE240(&v386);
        goto LABEL_20;
      case 0x6Fu:
      case 0xF7u:
        sub_6A7EC0(0, &v386);
        goto LABEL_20;
      case 0x70u:
        sub_6AF090(0, &v386);
        goto LABEL_20;
      case 0x71u:
        sub_6ACB40(&v386, 0, 0);
        goto LABEL_20;
      case 0x72u:
        sub_6AC240(&v386, 0);
        goto LABEL_20;
      case 0x73u:
        sub_6AC740(&v386, 0);
        goto LABEL_20;
      case 0x74u:
        sub_6AC910(&v386, 0);
        goto LABEL_20;
      case 0x75u:
        sub_6DBAB0(0, &v386);
        goto LABEL_20;
      case 0x8Au:
      case 0x8Bu:
      case 0x8Cu:
      case 0x8Du:
        sub_6944D0((__int64)&v386, v7);
        goto LABEL_20;
      case 0x8Fu:
        v343 = sub_72BA30(5);
        if ( dword_4F077C4 == 2 && (unsigned int)sub_6E5430(5, v7, v219, v220, v221, v222) )
          sub_6851A0(0x59Cu, &dword_4F063F8, (__int64)off_4B6D868[0]);
        sub_6A9320(0, 21, v343, 1, 1, 0, (__int64)&v386);
        if ( dword_4F077C4 == 2 )
          goto LABEL_170;
        goto LABEL_20;
      case 0x90u:
      case 0x91u:
        sub_6AABA0(0, &v386);
        goto LABEL_20;
      case 0x92u:
        if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
        {
          sub_69A8C0(58, &dword_4F063F8, v9, a4, v8, (__int64)"__is_signed");
          sub_7B8B50(58, &dword_4F063F8, v262, v263);
          sub_6E6260(&v386);
          goto LABEL_20;
        }
        v342 = v8;
        v217 = sub_7BE840(0, 0);
        if ( v217 == 155 )
        {
LABEL_283:
          sub_6C6830(0, &v386);
          if ( !sub_688800(word_4F06418[0], 18, v5) )
            sub_6E5C80(7, 3097, &dword_4F063F8);
          goto LABEL_20;
        }
        if ( v217 == 152 )
        {
LABEL_282:
          sub_6D53F0(0, &v386);
          goto LABEL_20;
        }
LABEL_332:
        v218 = word_4F06418[0];
        LODWORD(v8) = v342;
LABEL_333:
        if ( dword_4F077C4 == 2 )
        {
          if ( v218 == 1 )
          {
LABEL_431:
            if ( (unk_4D04A11 & 2) != 0 )
              goto LABEL_266;
          }
LABEL_382:
          v345 = v8;
          if ( !(unsigned int)sub_7C0F00(0x4000, 0) )
            goto LABEL_119;
          LODWORD(v8) = v345;
        }
        else if ( v218 != 1 )
        {
          goto LABEL_119;
        }
LABEL_266:
        if ( !(_DWORD)qword_4F077B4
          || (v326 = v8, (unk_4D04A10 & 1) != 0)
          || !qword_4D04A00
          || (v165 = *(_BYTE **)(qword_4D04A00 + 8), v339 = qword_4D04A00, *v165 != 95)
          || v165[1] != 95
          || v165[2] != 105
          || v165[3] != 115
          || v165[4] != 95
          || (v7 = 0, a1 = 0, (unsigned __int16)sub_7BE840(0, 0) != 27)
          || (v166 = *(_QWORD *)(v339 + 24)) == 0 )
        {
LABEL_280:
          sub_6CC940((unsigned int)&v386, v5, v4, 0, 0, 0, 0, 0, (__int64)v429);
          goto LABEL_20;
        }
        v8 = v326;
        v10 = "__is_signed";
        while ( *(_BYTE *)(v166 + 80) )
        {
          v166 = *(_QWORD *)(v166 + 8);
          if ( !v166 )
            goto LABEL_280;
        }
        v9 = *(_WORD *)(v166 + 88);
        word_4F06418[0] = v9;
        continue;
      case 0x98u:
        goto LABEL_282;
      case 0x9Bu:
        goto LABEL_283;
      case 0x9Cu:
        if ( dword_4F077C4 == 2 )
          goto LABEL_382;
        goto LABEL_119;
      case 0xA1u:
        sub_693B00((__int64)&v386, v7, v9, a4);
        goto LABEL_20;
      case 0xA2u:
        sub_6AF3D0(0, &v386);
        goto LABEL_20;
      case 0xA6u:
        sub_6AE340(0, &v386);
        goto LABEL_20;
      case 0xA7u:
        sub_6AD6A0(0, &v386);
        goto LABEL_20;
      case 0xB0u:
        sub_6AEB80(0, &v386);
        goto LABEL_20;
      case 0xB1u:
        sub_6C0800(0, &v386);
        goto LABEL_20;
      case 0xB2u:
        sub_6A2D50(0, &v386);
        goto LABEL_20;
      case 0xBCu:
        v185 = sub_72CBE0(a1, v7, v9, a4, v8, "__is_signed");
        v186 = sub_72D2E0(v185, 0);
        v187 = sub_622A90(*(_DWORD *)(v186 + 128) * dword_4F06BA0, 1);
        if ( v187 == 13 )
          v187 = 5;
        v188 = v187;
        v189 = sub_72BA30(v187);
        v429[0].m128i_i64[0] = sub_724DC0(v188, 1, v190, v191, v192, v193);
        sub_72BB40(v189, v429[0].m128i_i64[0]);
        v7 = (__int64)&v386;
        sub_6E6A50(v429[0].m128i_i64[0], &v386);
        v107 = (unsigned __int64)v429;
        sub_724E30(v429);
        v404.m128i_i8[9] |= 8u;
        goto LABEL_145;
      case 0xC3u:
      case 0xC4u:
      case 0xC5u:
      case 0xC6u:
      case 0xC7u:
      case 0xC8u:
      case 0xC9u:
      case 0xCAu:
      case 0xCBu:
      case 0xCCu:
      case 0xCDu:
      case 0xCEu:
      case 0xD0u:
      case 0xD4u:
      case 0xD5u:
      case 0xD6u:
      case 0xD7u:
      case 0xD8u:
      case 0xD9u:
      case 0xDAu:
      case 0xDBu:
      case 0xDCu:
      case 0xDDu:
      case 0xDEu:
      case 0xDFu:
      case 0xE0u:
      case 0xF2u:
      case 0x11Du:
      case 0x11Eu:
      case 0x125u:
      case 0x131u:
      case 0x134u:
      case 0x135u:
      case 0x136u:
      case 0x137u:
      case 0x138u:
      case 0x139u:
      case 0x13Au:
      case 0x13Bu:
      case 0x13Cu:
      case 0x13Du:
      case 0x13Eu:
      case 0x13Fu:
      case 0x140u:
      case 0x141u:
      case 0x142u:
      case 0x143u:
      case 0x144u:
      case 0x145u:
      case 0x146u:
      case 0x147u:
      case 0x148u:
      case 0x149u:
      case 0x14Au:
      case 0x150u:
      case 0x151u:
      case 0x152u:
      case 0x163u:
      case 0x164u:
        sub_6A98C0(0, &v386);
        goto LABEL_20;
      case 0xCFu:
      case 0xD1u:
      case 0xD2u:
      case 0xD3u:
      case 0x120u:
      case 0x121u:
      case 0x122u:
      case 0x123u:
      case 0x124u:
      case 0x12Au:
      case 0x12Bu:
        sub_6A9D40(0, &v386);
        goto LABEL_20;
      case 0xE1u:
        v167 = sub_68AFD0(0x70u);
        sub_6A9320(0, 112, v167, 1, 1, 1, (__int64)&v386);
        goto LABEL_169;
      case 0xE2u:
        v181 = sub_68AFD0(0x71u);
        sub_6A9320(0, 113, v181, 1, 1, 1, (__int64)&v386);
        goto LABEL_169;
      case 0xE3u:
        v182 = sub_68AFD0(0x1Eu);
        sub_6A9320(0, 30, v182, 1, 1, 1, (__int64)&v386);
        goto LABEL_169;
      case 0xE4u:
        v183 = sub_68AFD0(0x1Fu);
        sub_6A9320(0, 31, v183, 1, 1, 1, (__int64)&v386);
        goto LABEL_169;
      case 0xE5u:
        v184 = sub_68AFD0(0x29u);
        sub_6A9320(0, 41, v184, 1, 1, 1, (__int64)&v386);
        goto LABEL_169;
      case 0xE6u:
        v168 = sub_68AFD0(0x2Au);
        sub_6A9320(0, 42, v168, 1, 0, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xE7u:
        v169 = sub_68AFD0(0x2Bu);
        sub_6A9320(0, 43, v169, 1, 0, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xE8u:
        v170 = sub_68AFD0(0x2Cu);
        sub_6A9320(0, 44, v170, 1, 0, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xE9u:
        v171 = sub_68AFD0(0x2Du);
        sub_6A9320(0, 45, v171, 1, 1, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xEAu:
        v172 = sub_68AFD0(0x2Eu);
        sub_6A9320(0, 46, v172, 1, 1, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xEBu:
        v173 = sub_68AFD0(0x37u);
        sub_6A9320(0, 55, v173, 1, 0, 0, (__int64)&v386);
        goto LABEL_169;
      case 0xEDu:
        v174 = sub_72C570();
        v429[0].m128i_i64[0] = sub_724DC0(a1, v7, v175, v176, v177, v178);
        sub_72BB40(v174, v429[0].m128i_i64[0]);
        sub_6E6A50(v429[0].m128i_i64[0], &v386);
        v179 = sub_724E30(v429);
        LOBYTE(v179) = word_4F06418[0] == 237;
        v404.m128i_i8[9] = (16 * (word_4F06418[0] == 237)) | v404.m128i_i8[9] & 0xEF;
        sub_7B8B50(v429, &v386, (unsigned int)(16 * v179), v180);
        goto LABEL_20;
      case 0xF3u:
        sub_6ABC20(0, &v386);
        goto LABEL_20;
      case 0xFBu:
        sub_6DDBA0(&v386, 48);
        goto LABEL_20;
      case 0xFCu:
        sub_6DDBA0(&v386, 49);
        goto LABEL_20;
      case 0xFDu:
        sub_6DDE70(50, &v386);
        goto LABEL_20;
      case 0xFEu:
        sub_6DDE70(51, &v386);
        goto LABEL_20;
      case 0xFFu:
        sub_6DDE70(52, &v386);
        goto LABEL_20;
      case 0x100u:
        sub_6DDBA0(&v386, 47);
        goto LABEL_20;
      case 0x101u:
      case 0x102u:
        sub_6D3EC0(0, &v386);
        goto LABEL_20;
      case 0x103u:
        sub_6AFBA0(0, &v386);
        goto LABEL_20;
      case 0x105u:
        sub_6B0100(0, &v386);
        goto LABEL_20;
      case 0x106u:
        sub_6AB110(&v386);
        goto LABEL_20;
      case 0x10Bu:
        v407.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        if ( unk_4F04C50 )
        {
          v205 = *(_QWORD *)(unk_4F04C50 + 32LL);
          if ( !v205 || (*(_BYTE *)(v205 + 198) & 0x10) == 0 )
          {
            if ( (unsigned int)sub_86FC80(a1) )
            {
              a1 = 2669;
              sub_6851C0(0xA6Du, &v407);
            }
            v320 = dword_4F06650[0];
            sub_7B8B50(a1, &v407, v227, v228);
            if ( word_4F06418[0] == 73 && dword_4D04428 )
              v229 = sub_6BA760(0, 0);
            else
              v229 = sub_6A2B80(0);
            if ( (dword_4F04C44 != -1
               || (v230 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v230 + 6) & 6) != 0)
               || *(_BYTE *)(v230 + 4) == 12)
              && (unsigned int)sub_82ED80(v229) )
            {
              sub_6F41B0(v229);
              if ( *(_BYTE *)(v229 + 8) == 1 )
                sub_6E9FE0(v229, v429);
              else
                sub_6E6610(v229, v429, 1);
              sub_7032B0(118, v429, &v386, &v407, v320);
            }
            else
            {
              v231 = sub_695660(v229, 1, 0);
              sub_6E70E0(v231, &v386);
              if ( v387.m128i_i8[0] )
              {
                v232 = v386.m128i_i64[0];
                for ( n = *(_BYTE *)(v386.m128i_i64[0] + 140); n == 12; n = *(_BYTE *)(v232 + 140) )
                  v232 = *(_QWORD *)(v232 + 160);
                if ( n )
                  sub_6980A0(&v386, &v407, v320, 1, 0, 0);
              }
            }
LABEL_320:
            v390.m256i_i32[1] = v407.m128i_i32[0];
            v390.m256i_i16[4] = v407.m128i_i16[2];
            *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
            *(__int64 *)((char *)&v390.m256i_i64[1] + 4) = unk_4F061D8;
LABEL_321:
            sub_6E3280(&v386, &v407);
            sub_6E26D0(1, &v386);
            goto LABEL_20;
          }
          v206 = 3708;
        }
        else
        {
          v206 = 2747;
        }
        sub_6851C0(v206, &dword_4F063F8);
        sub_6E6260(&v386);
        sub_7BE180();
        goto LABEL_320;
      case 0x10Du:
        v407.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
        v341 = dword_4F06650[0];
        sub_7B8B50(a1, v7, v9, a4);
        sub_69ED20(v429, 0, 18, 0);
        v207 = _mm_loadu_si128(&v429[1]);
        v208 = _mm_loadu_si128(&v429[2]);
        v209 = _mm_loadu_si128(&v429[3]);
        v210 = _mm_loadu_si128(&v429[4]);
        v386 = _mm_loadu_si128(v429);
        v387 = v207;
        v211 = _mm_loadu_si128(&v429[5]);
        v212 = _mm_loadu_si128(&v429[6]);
        v388 = v208;
        v213 = _mm_loadu_si128(&v429[7]);
        v389 = v209;
        v214 = _mm_loadu_si128(&v429[8]);
        *(__m128i *)v390.m256i_i8 = v210;
        *(__m128i *)&v390.m256i_u64[2] = v211;
        v391 = v212;
        v392 = v213;
        v393 = v214;
        if ( v429[1].m128i_i8[0] == 2 )
        {
          v234 = _mm_loadu_si128(&v429[10]);
          v235 = _mm_loadu_si128(&v429[11]);
          v236 = _mm_loadu_si128(&v429[12]);
          v237 = _mm_loadu_si128(&v429[13]);
          v394 = _mm_loadu_si128(&v429[9]);
          v238 = _mm_loadu_si128(&v429[14]);
          v395 = v234;
          v239 = _mm_loadu_si128(&v429[15]);
          v396 = v235;
          v240 = _mm_loadu_si128(&v429[16]);
          v397 = v236;
          v241 = _mm_loadu_si128(&v429[17]);
          v398 = v237;
          v242 = _mm_loadu_si128(&v429[18]);
          v399 = v238;
          v243 = _mm_loadu_si128(&v429[19]);
          v400 = v239;
          v244 = _mm_loadu_si128(&v429[20]);
          v401 = v240;
          v245 = _mm_loadu_si128(&v429[21]);
          v402 = v241;
          v403 = v242;
          v404 = v243;
          v405 = v244;
          v406 = v245;
        }
        else if ( v429[1].m128i_i8[0] == 5 || v429[1].m128i_i8[0] == 1 )
        {
          v394.m128i_i64[0] = v429[9].m128i_i64[0];
        }
        sub_6980A0(&v386, &v407, v341, 0, 0, 0);
        v390.m256i_i32[1] = v407.m128i_i32[0];
        v390.m256i_i16[4] = v407.m128i_i16[2];
        *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v390.m256i_i64 + 4);
        *(__int64 *)((char *)&v390.m256i_i64[1] + 4) = *(__int64 *)((char *)&v429[4].m128i_i64[1] + 4);
        unk_4F061D8 = *(__int64 *)((char *)&v429[4].m128i_i64[1] + 4);
        goto LABEL_321;
      case 0x10Eu:
        v215 = sub_68AFD0(0x3Cu);
        sub_6A9320(0, 60, v215, 1, 1, 0, (__int64)&v386);
        goto LABEL_169;
      case 0x10Fu:
        sub_6A8930(0, &v386);
        goto LABEL_20;
      case 0x11Au:
        v344 = v8;
        sub_7B8B50(a1, v7, v9, a4);
        sub_7BE280(27, 125, 0, 0);
        ++*(_BYTE *)(qword_4F061C8 + 36LL);
        if ( word_4F06418[0] == 4 )
        {
          if ( (int)sub_6210B0((__int64)xmmword_4F06300, 0) >= 0 )
          {
            v223 = sub_620FD0((__int64)xmmword_4F06300, v429);
            v224 = v344;
            if ( !v429[0].m128i_i32[0] )
              v224 = unk_4D03C70 > (int)v223;
            LODWORD(v360) = v224;
            if ( v223 <= 0xFFFFFFFF && v224 )
            {
              v270 = sub_826AE0(v223, (unsigned int)v360);
              v271 = 36;
              v272 = &v386;
              v273 = *(_QWORD *)(unk_4D03C78 + 8LL * v270);
              v274 = (__int32 *)v273;
              while ( v271 )
              {
                v272->m128i_i32[0] = *v274++;
                v272 = (__m128i *)((char *)v272 + 4);
                --v271;
              }
              v275 = *(unsigned __int8 *)(v273 + 16);
              if ( (_BYTE)v275 == 2 )
              {
                v272 = &v394;
                v274 = (__int32 *)(v273 + 144);
                for ( ii = 52; ii; --ii )
                {
                  v272->m128i_i32[0] = *v274++;
                  v272 = (__m128i *)((char *)v272 + 4);
                }
              }
              else if ( (_BYTE)v275 == 5 || (LOBYTE(v275) = v275 - 1, !(_BYTE)v275) )
              {
                v394.m128i_i64[0] = *(_QWORD *)(v273 + 144);
              }
              sub_7B8B50(v272, v274, v275, 0);
              sub_7BE280(28, 18, 0, 0);
              --*(_BYTE *)(qword_4F061C8 + 36LL);
              goto LABEL_20;
            }
          }
          sub_6851C0(0x3Du, &dword_4F063F8);
          sub_7B8B50(61, &dword_4F063F8, v225, v226);
        }
        else
        {
          sub_6E5F20(661);
        }
        sub_7BE280(28, 18, 0, 0);
        --*(_BYTE *)(qword_4F061C8 + 36LL);
        goto LABEL_120;
      case 0x126u:
        sub_69E1D0(v386.m128i_i64, v7);
        goto LABEL_20;
      case 0x128u:
        sub_6AA060(&v386);
        goto LABEL_20;
      case 0x129u:
        sub_6AA570(0, &v386);
        goto LABEL_20;
      case 0x12Cu:
      case 0x12Du:
        sub_6A9E80(0, &v386);
        goto LABEL_20;
      case 0x12Eu:
      case 0x12Fu:
        sub_6A9F70(0, &v386);
        goto LABEL_20;
      case 0x130u:
        v216 = sub_68AFD0(0x4Eu);
        sub_6A9320(0, 78, v216, 2, 1, 0, (__int64)&v386);
        goto LABEL_169;
      case 0x132u:
      case 0x133u:
        if ( v9 == 306 )
        {
          sub_68AFD0(0x50u);
          v111 = 80;
          v112 = 0;
        }
        else
        {
          sub_68AFD0(0x51u);
          v111 = 81;
          v112 = 3;
        }
        v323 = v112;
        v113 = sub_72BA30(unk_4F06A51);
        sub_6A9320(0, v111, v113, 1, v323, 0, (__int64)&v386);
LABEL_169:
        if ( !dword_4D044B0 )
LABEL_170:
          sub_6E6840(&v386);
        goto LABEL_20;
      default:
        goto LABEL_119;
    }
  }
}
