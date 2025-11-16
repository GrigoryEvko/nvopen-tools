// Function: sub_1A3B290
// Address: 0x1a3b290
//
__int64 __fastcall sub_1A3B290(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r13
  __int64 v12; // rax
  unsigned int v13; // r15d
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // r12d
  __int64 v28; // r14
  __int64 v29; // r13
  _BYTE *v30; // rdx
  __int64 v31; // rdi
  _QWORD *v32; // rbx
  __int64 v33; // r14
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  unsigned int v36; // r13d
  __int64 v37; // rcx
  __int64 v38; // rdi
  _BYTE *v39; // r12
  double v40; // xmm4_8
  double v41; // xmm5_8
  int v42; // r14d
  _BYTE *v43; // rbx
  unsigned __int64 v44; // r12
  __int64 v45; // rdi
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // edx
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rax
  int v58; // eax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // rdi
  int v62; // eax
  unsigned __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r13
  __m128i v66; // kr00_16
  _QWORD *v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __m128i v72; // rax
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r15
  _QWORD *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __m128i v82; // rax
  __int64 **v83; // rsi
  unsigned __int64 v84; // rax
  __int64 v85; // rdx
  double v86; // xmm4_8
  double v87; // xmm5_8
  __int64 v88; // rcx
  __int64 *v89; // r12
  __int64 v90; // r13
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 v93; // rbx
  __int64 v94; // r13
  __int64 v95; // rsi
  __int64 v96; // rax
  double v97; // xmm4_8
  double v98; // xmm5_8
  __int64 *v99; // r12
  __int64 *v100; // rbx
  __int64 v101; // rsi
  __int64 *v102; // rsi
  _QWORD *v103; // r12
  double v104; // xmm4_8
  double v105; // xmm5_8
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 *v108; // rsi
  _QWORD *v109; // r12
  double v110; // xmm4_8
  double v111; // xmm5_8
  __int64 *v112; // rax
  __int64 v113; // r8
  __int64 v114; // rdx
  unsigned int v115; // r15d
  unsigned int v116; // eax
  __int64 v117; // rdx
  __int64 *v118; // rax
  unsigned __int64 v119; // rax
  __int64 v120; // r13
  char v121; // bl
  char v122; // dl
  __int64 v123; // rax
  _QWORD *v124; // rax
  __int64 *v125; // r12
  __int64 v126; // r15
  __int64 v127; // rbx
  char v128; // bl
  __int64 v129; // r12
  __int64 v130; // r12
  __int64 v131; // rdi
  _QWORD *v132; // rax
  _QWORD *v133; // rdx
  __int64 v134; // rbx
  __int64 v135; // rax
  char v136; // al
  char v137; // al
  __int64 *v138; // r13
  __int64 v139; // r12
  __int64 v140; // rbx
  __int64 v141; // rdi
  char v142; // al
  _QWORD *v143; // rax
  __int64 v144; // rdx
  char v145; // bl
  _QWORD *v146; // rax
  __int64 v147; // rdx
  __int64 v148; // rsi
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rax
  __int64 v152; // r14
  __int64 v153; // rdi
  _QWORD *v154; // rax
  __int64 v155; // rdx
  __int64 v156; // r13
  __int64 v157; // rax
  char v158; // al
  __int64 v159; // rax
  _QWORD *v160; // rdx
  unsigned __int64 v161; // rax
  __int64 v162; // rax
  char v163; // dl
  unsigned int v164; // r14d
  bool v165; // al
  __int64 v166; // r15
  _QWORD *v167; // rax
  __int64 v168; // r15
  __int64 v169; // r13
  _QWORD *v170; // rax
  __int64 v171; // r11
  __int64 v172; // rax
  __int64 *v173; // r8
  __int64 v174; // rsi
  __int64 v175; // rax
  __int64 v176; // r11
  __int64 v177; // rsi
  __int64 v178; // rdx
  unsigned __int8 *v179; // rsi
  unsigned __int64 v180; // rax
  __m128i v181; // kr10_16
  _QWORD *v182; // rax
  __int64 v183; // r8
  __int64 v184; // rax
  __int64 v185; // rax
  unsigned __int64 v186; // rax
  _QWORD *v187; // rax
  __int64 v188; // [rsp+10h] [rbp-5B0h]
  __int64 *v189; // [rsp+10h] [rbp-5B0h]
  unsigned __int64 v190; // [rsp+18h] [rbp-5A8h]
  __int64 *v191; // [rsp+18h] [rbp-5A8h]
  char v192; // [rsp+20h] [rbp-5A0h]
  __int64 *v193; // [rsp+20h] [rbp-5A0h]
  unsigned __int64 v194; // [rsp+28h] [rbp-598h]
  __int64 v195; // [rsp+30h] [rbp-590h]
  __int64 v196; // [rsp+38h] [rbp-588h]
  __int64 v197; // [rsp+40h] [rbp-580h]
  unsigned __int64 v198; // [rsp+48h] [rbp-578h]
  unsigned __int64 v199; // [rsp+48h] [rbp-578h]
  char v200; // [rsp+50h] [rbp-570h]
  __int64 v201; // [rsp+50h] [rbp-570h]
  unsigned __int64 v202; // [rsp+50h] [rbp-570h]
  __int64 v203; // [rsp+50h] [rbp-570h]
  __int64 v204; // [rsp+58h] [rbp-568h]
  __int64 v205; // [rsp+58h] [rbp-568h]
  __int64 v206; // [rsp+58h] [rbp-568h]
  bool v207; // [rsp+60h] [rbp-560h]
  __int64 v208; // [rsp+60h] [rbp-560h]
  __int64 *v209; // [rsp+60h] [rbp-560h]
  __int64 v210; // [rsp+60h] [rbp-560h]
  __int64 v211; // [rsp+68h] [rbp-558h]
  unsigned __int64 v212; // [rsp+68h] [rbp-558h]
  char v213; // [rsp+68h] [rbp-558h]
  __int64 v214; // [rsp+70h] [rbp-550h]
  __int64 v215; // [rsp+70h] [rbp-550h]
  __int64 v216; // [rsp+70h] [rbp-550h]
  unsigned __int64 v217; // [rsp+70h] [rbp-550h]
  __int64 v218; // [rsp+70h] [rbp-550h]
  unsigned int v219; // [rsp+70h] [rbp-550h]
  __int64 *v220; // [rsp+70h] [rbp-550h]
  __int64 v221; // [rsp+70h] [rbp-550h]
  unsigned __int64 v222; // [rsp+78h] [rbp-548h]
  __m128i v223; // [rsp+78h] [rbp-548h]
  unsigned __int64 v224; // [rsp+78h] [rbp-548h]
  __int64 v225; // [rsp+78h] [rbp-548h]
  __int64 *v226; // [rsp+78h] [rbp-548h]
  unsigned __int8 v227; // [rsp+78h] [rbp-548h]
  __int64 v228; // [rsp+78h] [rbp-548h]
  __m128i v229; // [rsp+78h] [rbp-548h]
  __int64 v230; // [rsp+78h] [rbp-548h]
  __int64 *v231; // [rsp+78h] [rbp-548h]
  unsigned int v232; // [rsp+80h] [rbp-540h]
  unsigned int v233; // [rsp+80h] [rbp-540h]
  unsigned int v234; // [rsp+80h] [rbp-540h]
  __int64 v235; // [rsp+80h] [rbp-540h]
  __int64 v236; // [rsp+80h] [rbp-540h]
  unsigned int v237; // [rsp+80h] [rbp-540h]
  __int64 v238; // [rsp+88h] [rbp-538h]
  char v239; // [rsp+88h] [rbp-538h]
  unsigned int v240; // [rsp+88h] [rbp-538h]
  __int64 *v241; // [rsp+88h] [rbp-538h]
  __int64 v242; // [rsp+88h] [rbp-538h]
  __int64 v243; // [rsp+88h] [rbp-538h]
  __int64 v244; // [rsp+88h] [rbp-538h]
  __int64 v245; // [rsp+88h] [rbp-538h]
  __int64 *v246; // [rsp+88h] [rbp-538h]
  __int64 v247; // [rsp+88h] [rbp-538h]
  int v248; // [rsp+90h] [rbp-530h]
  __int64 v249; // [rsp+90h] [rbp-530h]
  __int64 v250; // [rsp+90h] [rbp-530h]
  unsigned int v251; // [rsp+90h] [rbp-530h]
  __int64 *v252; // [rsp+90h] [rbp-530h]
  _QWORD *v253; // [rsp+90h] [rbp-530h]
  __int64 *v254; // [rsp+98h] [rbp-528h]
  __int64 *v255; // [rsp+A0h] [rbp-520h]
  __int64 v256; // [rsp+A0h] [rbp-520h]
  __int64 v257; // [rsp+A8h] [rbp-518h]
  __int64 *v258; // [rsp+A8h] [rbp-518h]
  __int64 v259; // [rsp+C8h] [rbp-4F8h]
  __int64 *v260; // [rsp+C8h] [rbp-4F8h]
  __int64 v261; // [rsp+D8h] [rbp-4E8h] BYREF
  __int64 *v262; // [rsp+E0h] [rbp-4E0h] BYREF
  __int64 v263; // [rsp+E8h] [rbp-4D8h]
  __int64 v264; // [rsp+F0h] [rbp-4D0h]
  __m128i v265; // [rsp+100h] [rbp-4C0h] BYREF
  __int64 v266; // [rsp+110h] [rbp-4B0h]
  __int64 v267; // [rsp+118h] [rbp-4A8h]
  __int64 v268; // [rsp+120h] [rbp-4A0h]
  __m128i v269; // [rsp+130h] [rbp-490h] BYREF
  __int64 v270; // [rsp+140h] [rbp-480h]
  __int64 v271; // [rsp+148h] [rbp-478h]
  __int64 v272; // [rsp+150h] [rbp-470h]
  __m128i v273; // [rsp+160h] [rbp-460h] BYREF
  __int64 v274; // [rsp+170h] [rbp-450h] BYREF
  __int64 v275; // [rsp+178h] [rbp-448h]
  __int64 v276; // [rsp+180h] [rbp-440h]
  __m128i v277; // [rsp+190h] [rbp-430h] BYREF
  _QWORD v278[6]; // [rsp+1A0h] [rbp-420h] BYREF
  _BYTE v279[48]; // [rsp+1D0h] [rbp-3F0h] BYREF
  __int64 v280; // [rsp+200h] [rbp-3C0h] BYREF
  __int64 *v281; // [rsp+208h] [rbp-3B8h]
  _BYTE *v282; // [rsp+210h] [rbp-3B0h]
  __int64 v283; // [rsp+218h] [rbp-3A8h]
  _BYTE v284[64]; // [rsp+220h] [rbp-3A0h] BYREF
  __int64 v285; // [rsp+260h] [rbp-360h]
  _BYTE *v286; // [rsp+268h] [rbp-358h]
  _BYTE *v287; // [rsp+270h] [rbp-350h]
  __int64 v288; // [rsp+278h] [rbp-348h]
  int v289; // [rsp+280h] [rbp-340h]
  _BYTE v290[64]; // [rsp+288h] [rbp-338h] BYREF
  __int64 *v291; // [rsp+2C8h] [rbp-2F8h]
  _QWORD *v292; // [rsp+2D0h] [rbp-2F0h] BYREF
  __int64 v293; // [rsp+2D8h] [rbp-2E8h]
  _BYTE *v294; // [rsp+2E0h] [rbp-2E0h] BYREF
  _QWORD *v295; // [rsp+2E8h] [rbp-2D8h]
  __int64 v296; // [rsp+2F0h] [rbp-2D0h]
  int v297; // [rsp+2F8h] [rbp-2C8h]
  __int64 v298; // [rsp+300h] [rbp-2C0h]
  __int64 v299; // [rsp+308h] [rbp-2B8h]
  _QWORD *v300; // [rsp+310h] [rbp-2B0h]
  __int64 v301; // [rsp+318h] [rbp-2A8h]
  _QWORD v302[3]; // [rsp+320h] [rbp-2A0h] BYREF
  _BYTE *v303; // [rsp+338h] [rbp-288h]
  __int64 v304; // [rsp+340h] [rbp-280h]
  _BYTE v305[16]; // [rsp+348h] [rbp-278h] BYREF
  _QWORD *v306; // [rsp+358h] [rbp-268h]
  __int64 v307; // [rsp+360h] [rbp-260h]
  _QWORD v308[4]; // [rsp+368h] [rbp-258h] BYREF
  __int64 v309; // [rsp+388h] [rbp-238h]
  __m128i v310; // [rsp+390h] [rbp-230h]
  __int64 v311; // [rsp+3A0h] [rbp-220h]
  __int64 v312; // [rsp+3B0h] [rbp-210h] BYREF
  unsigned __int64 v313; // [rsp+3B8h] [rbp-208h]
  unsigned __int64 v314; // [rsp+3C0h] [rbp-200h]
  _BYTE *v315; // [rsp+3C8h] [rbp-1F8h] BYREF
  __int64 v316; // [rsp+3D0h] [rbp-1F0h]
  _BYTE v317[176]; // [rsp+3D8h] [rbp-1E8h] BYREF
  __int64 *v318; // [rsp+488h] [rbp-138h]
  int v319; // [rsp+490h] [rbp-130h]
  __int64 v320; // [rsp+498h] [rbp-128h] BYREF
  _BYTE *v321; // [rsp+4A0h] [rbp-120h]
  _BYTE *v322; // [rsp+4A8h] [rbp-118h]
  __int64 v323; // [rsp+4B0h] [rbp-110h]
  int v324; // [rsp+4B8h] [rbp-108h]
  _BYTE v325[24]; // [rsp+4C0h] [rbp-100h] BYREF
  char *v326; // [rsp+4D8h] [rbp-E8h]
  char v327; // [rsp+4E8h] [rbp-D8h] BYREF
  unsigned __int64 v328; // [rsp+500h] [rbp-C0h]
  char v329; // [rsp+508h] [rbp-B8h]
  _QWORD *v330; // [rsp+510h] [rbp-B0h] BYREF
  unsigned int v331; // [rsp+518h] [rbp-A8h]
  unsigned __int64 v332; // [rsp+520h] [rbp-A0h]
  __int64 *v333; // [rsp+528h] [rbp-98h]
  __int64 v334; // [rsp+530h] [rbp-90h]
  __int64 v335; // [rsp+538h] [rbp-88h] BYREF
  __int64 v336; // [rsp+540h] [rbp-80h]
  __int64 v337; // [rsp+548h] [rbp-78h] BYREF
  char v338; // [rsp+588h] [rbp-38h] BYREF

  v10 = a2;
  if ( !a2[1] )
  {
    v13 = 1;
    sub_15F20C0(a2);
    return v13;
  }
  v12 = sub_15F2050((__int64)a2);
  v259 = sub_1632FA0(v12);
  v13 = sub_15F8BF0((__int64)a2);
  if ( (_BYTE)v13 )
    return 0;
  v15 = a2[7];
  v16 = *(unsigned __int8 *)(v15 + 8);
  if ( (unsigned __int8)v16 > 0xFu || (v34 = 35454, !_bittest64(&v34, v16)) )
  {
    if ( (unsigned int)(v16 - 13) > 1 && (_DWORD)v16 != 16 || !sub_16435F0(v15, 0) )
      return 0;
    v15 = a2[7];
  }
  if ( !sub_12BE0A0(v259, v15) || sub_12BE0A0(v259, a2[7]) > (unsigned int)dword_4FB3E60 )
    return 0;
  v17 = *(__int64 **)(a1 + 24);
  v18 = *(_QWORD *)(a1 + 8);
  v283 = 0x800000000LL;
  v281 = v17;
  v282 = v284;
  v280 = v18;
  v285 = 0;
  v286 = v290;
  v287 = v290;
  v288 = 8;
  v289 = 0;
  v19 = sub_15F2050((__int64)a2);
  v20 = sub_1632FA0(v19);
  v21 = a2[7];
  v313 = 0;
  v312 = v20;
  v315 = v317;
  v314 = 0;
  v316 = 0x800000000LL;
  v320 = 0;
  v321 = v325;
  v322 = v325;
  v323 = 8;
  v324 = 0;
  v331 = 1;
  v330 = 0;
  v22 = sub_12BE0A0(v20, v21);
  v334 = v280;
  v335 = 0;
  v336 = 1;
  v332 = v22;
  v23 = &v337;
  v333 = v281;
  do
  {
    *v23 = -8;
    v23 += 2;
  }
  while ( v23 != (__int64 *)&v338 );
  v24 = sub_15A9650(v312, *v10);
  v329 = 1;
  LODWORD(v293) = *(_DWORD *)(v24 + 8) >> 8;
  if ( (unsigned int)v293 > 0x40 )
    sub_16A4EF0((__int64)&v292, 0, 0);
  else
    v292 = 0;
  if ( v331 > 0x40 && v330 )
    j_j___libc_free_0_0(v330);
  v313 &= 3u;
  v314 &= 3u;
  v330 = v292;
  v331 = v293;
  sub_386EA80(&v312, v10);
  v25 = (unsigned int)v316;
  if ( !(_DWORD)v316 )
  {
LABEL_158:
    LOBYTE(v35) = v313;
    goto LABEL_37;
  }
  v257 = a1;
  v255 = v10;
  while ( 2 )
  {
    v26 = (__int64)&v315[24 * v25 - 24];
    v27 = *(_DWORD *)(v26 + 16);
    v28 = *(_QWORD *)v26;
    *(_DWORD *)(v26 + 16) = 0;
    v29 = *(_QWORD *)(v26 + 8);
    LODWORD(v316) = v316 - 1;
    v30 = &v315[24 * (unsigned int)v316];
    if ( *((_DWORD *)v30 + 4) > 0x40u )
    {
      v31 = *((_QWORD *)v30 + 1);
      if ( v31 )
        j_j___libc_free_0_0(v31);
    }
    v32 = (_QWORD *)(v28 & 0xFFFFFFFFFFFFFFF8LL);
    v328 = v28 & 0xFFFFFFFFFFFFFFF8LL;
    v329 = (v28 >> 2) & 1;
    if ( v329 )
    {
      if ( v331 > 0x40 && v330 )
      {
        j_j___libc_free_0_0(v330);
        v32 = (_QWORD *)v328;
      }
      v331 = v27;
      v27 = 0;
      v330 = (_QWORD *)v29;
    }
    v33 = (__int64)sub_1648700((__int64)v32);
    switch ( *(_BYTE *)(v33 + 16) )
    {
      case 0x18:
      case 0x19:
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1E:
      case 0x1F:
      case 0x20:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26:
      case 0x27:
      case 0x28:
      case 0x29:
      case 0x2A:
      case 0x2B:
      case 0x2C:
      case 0x2D:
      case 0x2E:
      case 0x2F:
      case 0x30:
      case 0x31:
      case 0x32:
      case 0x33:
      case 0x34:
      case 0x35:
      case 0x39:
      case 0x3A:
      case 0x3B:
      case 0x3C:
      case 0x3D:
      case 0x3E:
      case 0x3F:
      case 0x40:
      case 0x41:
      case 0x42:
      case 0x43:
      case 0x44:
      case 0x46:
      case 0x49:
      case 0x4A:
      case 0x4B:
      case 0x4C:
      case 0x50:
      case 0x51:
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
      case 0x56:
      case 0x57:
      case 0x58:
        goto LABEL_29;
      case 0x1D:
        goto LABEL_75;
      case 0x36:
        goto LABEL_80;
      case 0x37:
        LOBYTE(v35) = v313;
        if ( *(_QWORD *)(v33 - 48) != *v32 )
          goto LABEL_81;
        LOBYTE(v35) = v33 | v313 & 3 | 4;
        v313 = v33 | v313 & 3 | 4;
        v314 = v33 | v314 & 3 | 4;
        goto LABEL_31;
      case 0x38:
        if ( !*(_QWORD *)(v33 + 8) )
          goto LABEL_98;
        if ( !(unsigned __int8)sub_386E8D0(&v312, v33) )
        {
          v329 = 0;
          LODWORD(v293) = 1;
          v292 = 0;
          sub_1A1A780((__int64 *)&v330, (__int64 *)&v292);
          sub_135E100((__int64 *)&v292);
        }
        goto LABEL_94;
      case 0x45:
        v314 = v33 | v314 & 3 | 4;
        LOBYTE(v35) = v313;
        goto LABEL_31;
      case 0x47:
        if ( !*(_QWORD *)(v33 + 8) )
          goto LABEL_98;
        goto LABEL_94;
      case 0x48:
        if ( !*(_QWORD *)(v33 + 8) )
          goto LABEL_98;
        v55 = *(_QWORD *)v33;
        if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) == 16 )
          v55 = **(_QWORD **)(v55 + 16);
        v56 = sub_15A9520(v312, *(_DWORD *)(v55 + 8) >> 8);
        sub_16A5D10((__int64)&v292, (__int64)&v330, 8 * v56);
        sub_1A1A780((__int64 *)&v330, (__int64 *)&v292);
        sub_135E100((__int64 *)&v292);
        sub_386EA80(&v312, v33);
        LOBYTE(v35) = v313;
        goto LABEL_31;
      case 0x4D:
        if ( !*(_QWORD *)(v33 + 8) )
          goto LABEL_98;
        v200 = v329;
        if ( !v329 )
          goto LABEL_29;
        v292 = (_QWORD *)v33;
        v53 = sub_1A28710((__int64)&v335, (__int64 *)&v292);
        if ( !v53[1] )
        {
          v54 = sub_1A1E740(&v312, v33, (unsigned __int64 *)v53 + 1);
          if ( v54 )
            goto LABEL_103;
        }
        v122 = v313;
        if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) != 15 )
          goto LABEL_238;
        v123 = *(_QWORD *)(v33 + 40);
        v273.m128i_i64[1] = 0x400000000LL;
        v218 = v123;
        v273.m128i_i64[0] = (__int64)&v274;
        sub_1A24E50(v33, (__int64)&v273);
        v277.m128i_i64[0] = (__int64)v278;
        v277.m128i_i64[1] = 0x400000000LL;
        sub_1A251A0(v33, (__int64)&v277);
        if ( v277.m128i_i32[2] )
        {
          v213 = 0;
          v237 = v27;
          v241 = (__int64 *)v277.m128i_i64[0];
          v188 = v277.m128i_i64[0] + 8LL * v277.m128i_u32[2];
          v228 = v29;
          while ( 1 )
          {
            v129 = *v241;
            if ( sub_15F32D0(*v241) || (*(_BYTE *)(v129 + 18) & 1) != 0 || v218 != *(_QWORD *)(v129 + 40) )
              break;
            v210 = *(_QWORD *)(v129 - 24);
            v196 = v33;
            v152 = v33 + 24;
            v193 = 0;
            v199 = (unsigned __int64)(sub_127FA20(v312, *(_QWORD *)v129) + 7) >> 3;
            while ( 1 )
            {
              v153 = v152 - 24;
              if ( v129 == v152 - 24 )
                break;
              if ( *(_BYTE *)(v152 - 8) == 55 )
              {
                v154 = (_QWORD *)v273.m128i_i64[0];
                v155 = v273.m128i_i64[0] + 8LL * v273.m128i_u32[2];
                if ( v273.m128i_i64[0] == v155 )
                {
LABEL_302:
                  v156 = *(_QWORD *)(v152 - 48);
                  v191 = *(__int64 **)(v152 - 72);
                  v157 = sub_127FA20(v312, *v191);
                  v292 = (_QWORD *)v156;
                  v294 = 0;
                  v194 = (unsigned __int64)(v157 + 7) >> 3;
                  v293 = 1;
                  v295 = 0;
                  v296 = 0;
                  v269.m128i_i64[0] = v210;
                  v269.m128i_i64[1] = 1;
                  v270 = 0;
                  v271 = 0;
                  v272 = 0;
                  if ( (unsigned __int8)sub_134CB50((__int64)v333, (__int64)&v269, (__int64)&v292) == 3 && v199 <= v194 )
                  {
                    v193 = v191;
                  }
                  else
                  {
                    v292 = (_QWORD *)v156;
                    v293 = v194;
                    v269.m128i_i64[0] = v210;
                    v294 = 0;
                    v295 = 0;
                    v296 = 0;
                    v269.m128i_i64[1] = v199;
                    v270 = 0;
                    v271 = 0;
                    v272 = 0;
                    if ( (unsigned __int8)sub_134CB50((__int64)v333, (__int64)&v269, (__int64)&v292) )
                    {
LABEL_301:
                      v33 = v196;
                      v27 = v237;
                      v128 = 0;
                      v29 = v228;
                      v13 = (unsigned __int8)v13;
                      goto LABEL_233;
                    }
                  }
                }
                else
                {
                  while ( v153 != *v154 )
                  {
                    if ( (_QWORD *)v155 == ++v154 )
                      goto LABEL_302;
                  }
                }
              }
              else if ( (unsigned __int8)sub_15F3040(v153) )
              {
                goto LABEL_301;
              }
              v152 = *(_QWORD *)(v152 + 8);
              if ( !v152 )
LABEL_358:
                BUG();
            }
            v158 = v213;
            v33 = v196;
            if ( !v193 )
              v158 = v200;
            v13 = (unsigned __int8)v13;
            ++v241;
            v213 = v158;
            if ( (__int64 *)v188 == v241 )
            {
              v27 = v237;
              v29 = v228;
              goto LABEL_223;
            }
          }
          v27 = v237;
          v29 = v228;
LABEL_244:
          v128 = 0;
          goto LABEL_233;
        }
        v213 = 0;
        if ( !v273.m128i_i32[2] )
          goto LABEL_244;
LABEL_223:
        v124 = &v294;
        v292 = 0;
        v293 = 1;
        do
        {
          *v124 = -8;
          v124 += 2;
        }
        while ( v124 != v302 );
        v208 = v273.m128i_i64[0] + 8LL * v273.m128i_u32[2];
        if ( v273.m128i_i64[0] != v208 )
        {
          v236 = v29;
          v227 = v13;
          v240 = v27;
          v125 = (__int64 *)v273.m128i_i64[0];
          while ( 1 )
          {
            v126 = *v125;
            v127 = *(_QWORD *)(*v125 - 48);
            if ( sub_15F32D0(*v125) || (*(_BYTE *)(v126 + 18) & 1) != 0 || v218 != *(_QWORD *)(v126 + 40) )
            {
LABEL_230:
              v27 = v240;
              v29 = v236;
              v128 = 0;
              v13 = v227;
              goto LABEL_231;
            }
            v205 = *(_QWORD *)(v126 - 24);
            v202 = (unsigned __int64)(sub_127FA20(v312, **(_QWORD **)(v126 - 48)) + 7) >> 3;
            v138 = v125;
            v139 = v127;
            v140 = v33 + 24;
            while ( 1 )
            {
              v141 = v140 - 24;
              if ( v126 == v140 - 24 )
                break;
              v142 = *(_BYTE *)(v140 - 8);
              if ( v142 == 54 )
              {
                v143 = (_QWORD *)v277.m128i_i64[0];
                v144 = v277.m128i_i64[0] + 8LL * v277.m128i_u32[2];
                if ( v277.m128i_i64[0] != v144 )
                {
                  while ( v141 != *v143 )
                  {
                    if ( (_QWORD *)v144 == ++v143 )
                      goto LABEL_283;
                  }
                  goto LABEL_271;
                }
LABEL_283:
                v148 = *(_QWORD *)(v140 - 24);
LABEL_284:
                v149 = sub_127FA20(v312, v148);
                v150 = *(_QWORD *)(v140 - 48);
                v270 = 0;
                v269.m128i_i64[0] = v150;
                v269.m128i_i64[1] = (unsigned __int64)(v149 + 7) >> 3;
                v271 = 0;
                v265.m128i_i64[0] = v205;
                v272 = 0;
                v265.m128i_i64[1] = v202;
                v266 = 0;
                v267 = 0;
                v268 = 0;
                if ( (unsigned __int8)sub_134CB50((__int64)v333, (__int64)&v265, (__int64)&v269) )
                  goto LABEL_230;
                goto LABEL_271;
              }
              if ( v142 == 55 )
              {
                v146 = (_QWORD *)v273.m128i_i64[0];
                v147 = v273.m128i_i64[0] + 8LL * v273.m128i_u32[2];
                if ( v273.m128i_i64[0] != v147 )
                {
                  while ( v141 != *v146 )
                  {
                    if ( (_QWORD *)v147 == ++v146 )
                      goto LABEL_290;
                  }
                  goto LABEL_271;
                }
LABEL_290:
                v148 = **(_QWORD **)(v140 - 72);
                goto LABEL_284;
              }
              if ( (unsigned __int8)sub_15F3040(v141) || (unsigned __int8)sub_15F2ED0(v140 - 24) )
                goto LABEL_230;
LABEL_271:
              v140 = *(_QWORD *)(v140 + 8);
              if ( !v140 )
                BUG();
            }
            v145 = sub_1A29B80(v334, v139, v218, (__int64)&v292);
            if ( !v145 )
              goto LABEL_230;
            v125 = v138 + 1;
            if ( (__int64 *)v208 == v138 + 1 )
            {
              v27 = v240;
              v29 = v236;
              v13 = v227;
              goto LABEL_276;
            }
          }
        }
        v145 = 0;
LABEL_276:
        v128 = v213 | v145;
LABEL_231:
        if ( (v293 & 1) == 0 )
          j___libc_free_0(v294);
LABEL_233:
        if ( (_QWORD *)v277.m128i_i64[0] != v278 )
          _libc_free(v277.m128i_u64[0]);
        if ( (__int64 *)v273.m128i_i64[0] != &v274 )
          _libc_free(v273.m128i_u64[0]);
        v122 = v313;
        LOBYTE(v35) = v313;
        if ( !v128 )
        {
LABEL_238:
          LOBYTE(v35) = v33 | v122 & 3 | 4;
          v313 = v33 | v122 & 3 | 4;
        }
        goto LABEL_31;
      case 0x4E:
        v50 = *(_QWORD *)(v33 - 24);
        if ( *(_BYTE *)(v50 + 16) )
          goto LABEL_75;
        v51 = *(_DWORD *)(v50 + 36);
        if ( v51 == 135 )
          goto LABEL_186;
        if ( v51 > 0x87 )
        {
          if ( v51 == 213 )
            goto LABEL_75;
          if ( v51 > 0xD5 )
          {
            if ( v51 == 214 )
            {
LABEL_75:
              LOBYTE(v35) = v33 & 0xF8 | v313 & 3 | 4;
              v314 = v33 & 0xFFFFFFFFFFFFFFF8LL | v314 & 3 | 4;
              v313 = v33 & 0xFFFFFFFFFFFFFFF8LL | v313 & 3 | 4;
              goto LABEL_31;
            }
            goto LABEL_89;
          }
          if ( v51 != 137 )
          {
            if ( v51 == 212 )
              goto LABEL_75;
LABEL_89:
            LOBYTE(v35) = v313;
LABEL_90:
            v35 = v33 & 0xFFFFFFFFFFFFFFF8LL | v35 & 3 | 4;
            v313 = v35;
            v314 = v33 & 0xFFFFFFFFFFFFFFF8LL | v314 & 3 | 4;
            goto LABEL_31;
          }
          v48 = *(_QWORD *)(v33 + 24 * (2LL - (*(_DWORD *)(v33 + 20) & 0xFFFFFFF)));
          if ( *(_BYTE *)(v48 + 16) == 13 )
          {
            if ( *(_DWORD *)(v48 + 32) <= 0x40u )
            {
              v49 = *(_QWORD *)(v48 + 24);
              goto LABEL_79;
            }
            v248 = *(_DWORD *)(v48 + 32);
            if ( v248 - (unsigned int)sub_16A57B0(v48 + 24) <= 0x40 )
            {
              v49 = **(_QWORD **)(v48 + 24);
LABEL_79:
              if ( !v49 )
                goto LABEL_98;
            }
          }
LABEL_80:
          LOBYTE(v35) = v313;
LABEL_81:
          if ( !v329 )
          {
LABEL_30:
            v35 = v33 | v35 & 3 | 4;
            v313 = v35;
          }
          goto LABEL_31;
        }
        if ( v51 <= 0x26 )
        {
          LOBYTE(v35) = v313;
          if ( v51 > 0x23 )
            goto LABEL_31;
          if ( !v51 )
            goto LABEL_75;
          goto LABEL_89;
        }
        if ( v51 == 133 )
        {
LABEL_186:
          sub_1A1E260((__int64)&v312, v33);
LABEL_98:
          LOBYTE(v35) = v313;
          goto LABEL_31;
        }
        LOBYTE(v35) = v313;
        if ( v51 - 116 > 1 )
          goto LABEL_90;
LABEL_31:
        if ( (v35 & 4) == 0 )
        {
          if ( v27 > 0x40 && v29 )
            j_j___libc_free_0_0(v29);
          v25 = (unsigned int)v316;
          if ( !(_DWORD)v316 )
          {
            a1 = v257;
            v10 = v255;
            LOBYTE(v35) = v313;
            goto LABEL_37;
          }
          continue;
        }
        v88 = v29;
        a1 = v257;
        v10 = v255;
        if ( v27 > 0x40 )
        {
          if ( v88 )
            j_j___libc_free_0_0(v88);
          goto LABEL_158;
        }
LABEL_37:
        if ( (v314 & 4) == 0 && (v35 & 4) == 0 )
        {
          sub_1A1B710((__int64)&v280, (__int64)v10);
          if ( (_DWORD)v283 )
          {
            v256 = a1;
            v13 = 0;
            v254 = v10;
            v36 = v283;
            while ( 2 )
            {
              v37 = v36--;
              v38 = *(_QWORD *)&v282[8 * v37 - 8];
              LODWORD(v283) = v36;
              v291 = (__int64 *)v38;
              v39 = sub_1648700(v38);
              switch ( v39[16] )
              {
                case 0x19:
                case 0x1A:
                case 0x1B:
                case 0x1C:
                case 0x1D:
                case 0x1E:
                case 0x1F:
                case 0x20:
                case 0x21:
                case 0x22:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x49:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                case 0x56:
                case 0x57:
                case 0x58:
                  goto LABEL_43;
                case 0x36:
                  if ( sub_15F32D0((__int64)v39) )
                    goto LABEL_121;
                  if ( (v39[18] & 1) != 0 )
                    goto LABEL_121;
                  v74 = *(unsigned __int8 *)(*(_QWORD *)v39 + 8LL);
                  if ( (unsigned __int8)v74 <= 0x10u )
                  {
                    v75 = 100990;
                    if ( _bittest64(&v75, v74) )
                      goto LABEL_121;
                  }
                  v273 = 0u;
                  v274 = 0;
                  sub_14A8180((__int64)v39, v273.m128i_i64, 0);
                  v76 = v274;
                  v223 = v273;
                  v215 = *v291;
                  v77 = (_QWORD *)sub_16498A0((__int64)v39);
                  v292 = 0;
                  v295 = v77;
                  v300 = v302;
                  v294 = 0;
                  v296 = 0;
                  v297 = 0;
                  v298 = 0;
                  v299 = 0;
                  v293 = 0;
                  v301 = 0;
                  LOBYTE(v302[0]) = 0;
                  sub_17050D0((__int64 *)&v292, (__int64)v39);
                  v303 = v305;
                  v304 = 0x400000000LL;
                  v78 = sub_1643350(v295);
                  v79 = sub_159C470(v78, 0, 0);
                  v307 = 0x400000001LL;
                  v309 = v215;
                  v310 = v223;
                  v311 = v76;
                  v306 = v308;
                  v308[0] = v79;
                  v262 = (__int64 *)sub_1599EF0(*(__int64 ***)v39);
                  v80 = sub_15F2050((__int64)v39);
                  v81 = sub_1632FA0(v80);
                  v265.m128i_i64[0] = 0;
                  v224 = sub_127FA20(v81, *(_QWORD *)v39);
                  v233 = 1 << (*((unsigned __int16 *)v39 + 9) >> 1) >> 1;
                  v82.m128i_i64[0] = (__int64)sub_1649960((__int64)v39);
                  v269 = v82;
                  LOWORD(v278[0]) = 773;
                  v277.m128i_i64[0] = (__int64)&v269;
                  v277.m128i_i64[1] = (__int64)".fca";
                  v83 = *(__int64 ***)v39;
                  v84 = *(unsigned __int8 *)(*(_QWORD *)v39 + 8LL);
                  if ( (unsigned __int8)v84 <= 0x10u && (v85 = 100990, _bittest64(&v85, v84)) )
                    sub_1A1EF10((__int64)&v292, (__int64)v83, &v262, &v277, v233, &v265);
                  else
                    sub_1A1F2B0(
                      (__int64)&v292,
                      (__int64)v83,
                      &v262,
                      (__int64)&v277,
                      v233,
                      (unsigned __int64 *)&v265,
                      v224);
                  sub_164D160((__int64)v39, (__int64)v262, a3, a4, a5, a6, v86, v87, a9, a10);
                  sub_15F20C0(v39);
                  v73 = (unsigned __int64)v306;
                  if ( v306 == v308 )
                    goto LABEL_134;
                  goto LABEL_133;
                case 0x37:
                  if ( sub_15F32D0((__int64)v39) )
                    goto LABEL_121;
                  if ( (v39[18] & 1) != 0 )
                    goto LABEL_121;
                  if ( *((_QWORD *)v39 - 3) != *v291 )
                    goto LABEL_121;
                  v262 = (__int64 *)*((_QWORD *)v39 - 6);
                  v63 = *(unsigned __int8 *)(*v262 + 8);
                  if ( (unsigned __int8)v63 <= 0x10u )
                  {
                    v64 = 100990;
                    if ( _bittest64(&v64, v63) )
                      goto LABEL_121;
                  }
                  v273 = 0u;
                  v274 = 0;
                  sub_14A8180((__int64)v39, v273.m128i_i64, 0);
                  v65 = *v291;
                  v66 = v273;
                  v214 = v274;
                  v67 = (_QWORD *)sub_16498A0((__int64)v39);
                  LOBYTE(v302[0]) = 0;
                  v295 = v67;
                  v292 = 0;
                  v296 = 0;
                  v297 = 0;
                  v298 = 0;
                  v299 = 0;
                  v300 = v302;
                  v301 = 0;
                  v293 = *((_QWORD *)v39 + 5);
                  v294 = v39 + 24;
                  v68 = *((_QWORD *)v39 + 6);
                  v277.m128i_i64[0] = v68;
                  if ( v68 )
                  {
                    sub_1623A60((__int64)&v277, v68, 2);
                    if ( v292 )
                      sub_161E7C0((__int64)&v292, (__int64)v292);
                    v292 = (_QWORD *)v277.m128i_i64[0];
                    if ( v277.m128i_i64[0] )
                      sub_1623210((__int64)&v277, (unsigned __int8 *)v277.m128i_i64[0], (__int64)&v292);
                  }
                  v303 = v305;
                  v304 = 0x400000000LL;
                  v69 = sub_1643350(v295);
                  v308[0] = sub_159C470(v69, 0, 0);
                  v306 = v308;
                  v310 = v66;
                  v307 = 0x400000001LL;
                  v309 = v65;
                  v311 = v214;
                  v265.m128i_i64[0] = 0;
                  v70 = sub_15F2050((__int64)v39);
                  v71 = sub_1632FA0(v70);
                  v222 = sub_127FA20(v71, *v262);
                  v232 = 1 << (*((unsigned __int16 *)v39 + 9) >> 1) >> 1;
                  v72.m128i_i64[0] = (__int64)sub_1649960((__int64)v262);
                  v269 = v72;
                  LOWORD(v278[0]) = 773;
                  v277.m128i_i64[1] = (__int64)".fca";
                  v277.m128i_i64[0] = (__int64)&v269;
                  sub_1A202B0((__int64)&v292, *v262, (__int64 *)&v262, &v277, v232, &v265, v222);
                  sub_15F20C0(v39);
                  v73 = (unsigned __int64)v306;
                  if ( v306 != v308 )
LABEL_133:
                    _libc_free(v73);
LABEL_134:
                  if ( v303 != v305 )
                    _libc_free((unsigned __int64)v303);
                  if ( v300 != v302 )
                    j_j___libc_free_0(v300, v302[0] + 1LL);
                  if ( v292 )
                    sub_161E7C0((__int64)&v292, (__int64)v292);
                  v13 = 1;
                  goto LABEL_121;
                case 0x38:
                case 0x47:
                case 0x48:
                  sub_1A1B710((__int64)&v280, (__int64)v39);
                  v36 = v283;
                  goto LABEL_43;
                case 0x4D:
                case 0x4F:
                  v42 = sub_1A2D070((__int64)v39, a3, a4, a5, a6, v40, v41, a9, a10);
                  sub_1A1B710((__int64)&v280, (__int64)v39);
                  v36 = v283;
                  v13 |= v42;
                  goto LABEL_43;
                case 0x4E:
                  v57 = *((_QWORD *)v39 - 3);
                  if ( *(_BYTE *)(v57 + 16) )
                    goto LABEL_43;
                  v58 = *(_DWORD *)(v57 + 36);
                  if ( v58 == 133 )
                  {
                    v216 = sub_1649C60(*(_QWORD *)&v39[24 * (1LL - (*((_DWORD *)v39 + 5) & 0xFFFFFFF))]);
                    v112 = (__int64 *)sub_1649C60(*(_QWORD *)&v39[-24 * (*((_DWORD *)v39 + 5) & 0xFFFFFFF)]);
                    v113 = *(_QWORD *)v216;
                    v225 = (__int64)v112;
                    v114 = *v112;
                    if ( *(_BYTE *)(*(_QWORD *)v216 + 8LL) != 15 )
                      v113 = 0;
                    if ( *(_BYTE *)(v114 + 8) != 15 )
                      goto LABEL_121;
                    v211 = *(_QWORD *)&v39[24 * (2LL - (*((_DWORD *)v39 + 5) & 0xFFFFFFF))];
                    if ( *(_BYTE *)(v211 + 16) != 13 )
                      goto LABEL_121;
                    v207 = v114 == v113 && v113 != 0;
                    if ( !v207 || (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v113 + 24) + 8LL) - 13 > 1 )
                      goto LABEL_121;
                    v250 = v113;
                    v115 = sub_15603A0((_QWORD *)v39 + 7, 0);
                    v116 = sub_15603A0((_QWORD *)v39 + 7, 1);
                    if ( v115 <= v116 )
                      v116 = v115;
                    v117 = *((_DWORD *)v39 + 5) & 0xFFFFFFF;
                    v234 = v116;
                    if ( *(_QWORD *)&v39[24 * (1 - v117)] == *v291 )
                    {
                      v273 = 0u;
                      v274 = 0;
                      sub_14A8180((__int64)v39, v273.m128i_i64, 0);
                      v181 = v273;
                      v203 = v274;
                      v182 = (_QWORD *)sub_16498A0((__int64)v39);
                      LOBYTE(v302[0]) = 0;
                      v183 = v250;
                      v295 = v182;
                      v292 = 0;
                      v296 = 0;
                      v297 = 0;
                      v298 = 0;
                      v299 = 0;
                      v300 = v302;
                      v301 = 0;
                      v293 = *((_QWORD *)v39 + 5);
                      v294 = v39 + 24;
                      v277.m128i_i64[0] = *((_QWORD *)v39 + 6);
                      if ( v277.m128i_i64[0] )
                      {
                        sub_1623A60((__int64)&v277, v277.m128i_i64[0], 2);
                        v183 = v250;
                        if ( v292 )
                        {
                          sub_161E7C0((__int64)&v292, (__int64)v292);
                          v183 = v250;
                        }
                        v292 = (_QWORD *)v277.m128i_i64[0];
                        if ( v277.m128i_i64[0] )
                        {
                          v247 = v183;
                          sub_1623210((__int64)&v277, (unsigned __int8 *)v277.m128i_i64[0], (__int64)&v292);
                          v183 = v247;
                        }
                      }
                      v197 = v183;
                      v303 = v305;
                      v304 = 0x400000000LL;
                      v184 = sub_1643350(v295);
                      v308[0] = sub_159C470(v184, 0, 0);
                      v306 = v308;
                      v309 = v216;
                      v307 = 0x400000001LL;
                      v310 = v181;
                      v311 = v203;
                      v185 = sub_1599EF0(*(__int64 ***)(v197 + 24));
                      v269.m128i_i64[0] = 0;
                      v265.m128i_i64[0] = v185;
                      v186 = sub_179D670((_DWORD *)(v211 + 24), 0xFFFFFFFFFFFFFFFFLL);
                      v277.m128i_i64[0] = (__int64)"memcpy.load.fca";
                      LOWORD(v278[0]) = 259;
                      sub_1A1F890(
                        (__int64)&v292,
                        *(_QWORD *)(v197 + 24),
                        (__int64 **)&v265,
                        &v277,
                        v234,
                        &v269,
                        8 * v186);
                      sub_1A1B630((__int64)&v277, (__int64)v39);
                      v187 = sub_1A1CF60(v277.m128i_i64, v265.m128i_i64[0], v225, 0);
                      sub_15F9450((__int64)v187, v234);
                      sub_2240A30(v279);
                      sub_17CD270(v277.m128i_i64);
                      if ( v306 != v308 )
                        _libc_free((unsigned __int64)v306);
                      if ( v303 != v305 )
                        _libc_free((unsigned __int64)v303);
                      if ( v300 != v302 )
                        j_j___libc_free_0(v300, v302[0] + 1LL);
                      if ( v292 )
                        sub_161E7C0((__int64)&v292, (__int64)v292);
                    }
                    else if ( *v291 == *(_QWORD *)&v39[-24 * v117] )
                    {
                      v262 = 0;
                      v263 = 0;
                      v264 = 0;
                      sub_14A8180((__int64)v39, (__int64 *)&v262, 0);
                      v168 = v264;
                      v252 = v262;
                      v169 = v263;
                      sub_1A1B810((__int64)&v292, (__int64)v39, v225);
                      v311 = v168;
                      v310.m128i_i64[0] = (__int64)v252;
                      v310.m128i_i64[1] = v169;
                      v261 = 0;
                      sub_1A1B630((__int64)&v277, (__int64)v39);
                      LOWORD(v266) = 257;
                      v170 = sub_1648A60(64, 1u);
                      v171 = (__int64)v170;
                      if ( v170 )
                      {
                        v253 = v170;
                        sub_15F9210((__int64)v170, *(_QWORD *)(*(_QWORD *)v216 + 24LL), v216, 0, 0, 0);
                        v171 = (__int64)v253;
                      }
                      v172 = v277.m128i_i64[1];
                      v173 = (__int64 *)v278[0];
                      if ( (unsigned __int8)v266 > 1u )
                      {
                        v206 = v277.m128i_i64[1];
                        v221 = v171;
                        v231 = (__int64 *)v278[0];
                        v273.m128i_i64[0] = (__int64)v279;
                        LOWORD(v274) = 260;
                        sub_14EC200(&v269, &v273, &v265);
                        v173 = v231;
                        v171 = v221;
                        v172 = v206;
                      }
                      else
                      {
                        a3 = (__m128)_mm_loadu_si128(&v265);
                        v270 = v266;
                        v269 = (__m128i)a3;
                      }
                      v220 = v173;
                      if ( v172 )
                      {
                        v242 = v171;
                        sub_157E9D0(v172 + 40, v171);
                        v171 = v242;
                        v174 = *v220;
                        v175 = *(_QWORD *)(v242 + 24);
                        *(_QWORD *)(v242 + 32) = v220;
                        v174 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v242 + 24) = v174 | v175 & 7;
                        *(_QWORD *)(v174 + 8) = v242 + 24;
                        *v220 = *v220 & 7 | (v242 + 24);
                      }
                      v243 = v171;
                      sub_164B780(v171, v269.m128i_i64);
                      v176 = v243;
                      if ( v277.m128i_i64[0] )
                      {
                        v273.m128i_i64[0] = v277.m128i_i64[0];
                        sub_1623A60((__int64)&v273, v277.m128i_i64[0], 2);
                        v176 = v243;
                        v177 = *(_QWORD *)(v243 + 48);
                        v178 = v243 + 48;
                        if ( v177 )
                        {
                          v230 = v243;
                          v244 = v243 + 48;
                          sub_161E7C0(v244, v177);
                          v176 = v230;
                          v178 = v244;
                        }
                        v179 = (unsigned __int8 *)v273.m128i_i64[0];
                        *(_QWORD *)(v176 + 48) = v273.m128i_i64[0];
                        if ( v179 )
                        {
                          v245 = v176;
                          sub_1623210((__int64)&v273, v179, v178);
                          v176 = v245;
                        }
                      }
                      v246 = (__int64 *)v176;
                      sub_15F8F50(v176, v234);
                      v269.m128i_i64[0] = (__int64)v246;
                      v180 = sub_179D670((_DWORD *)(v211 + 24), 0xFFFFFFFFFFFFFFFFLL);
                      v273.m128i_i64[0] = (__int64)"memcpy.store.fca";
                      LOWORD(v274) = 259;
                      sub_1A202B0((__int64)&v292, *v246, v269.m128i_i64, &v273, v234, &v261, 8 * v180);
                      sub_2240A30(v279);
                      sub_17CD270(v277.m128i_i64);
                      sub_1A1B230((__int64 *)&v292);
                    }
                    sub_15F20C0(v39);
                    v36 = v283;
                    v13 = v207;
                  }
                  else
                  {
                    if ( v58 != 137 )
                      goto LABEL_43;
                    v238 = sub_1649C60(*(_QWORD *)&v39[-24 * (*((_DWORD *)v39 + 5) & 0xFFFFFFF)]);
                    if ( *(_BYTE *)(*(_QWORD *)v238 + 8LL) != 15 )
                      goto LABEL_121;
                    v249 = *(_QWORD *)(*(_QWORD *)v238 + 24LL);
                    v59 = *(_QWORD *)&v39[24 * (2LL - (*((_DWORD *)v39 + 5) & 0xFFFFFFF))];
                    if ( *(_BYTE *)(v59 + 16) != 13 )
                    {
                      v151 = sub_15F2050((__int64)v39);
                      sub_1632FA0(v151);
LABEL_121:
                      v36 = v283;
                      goto LABEL_43;
                    }
                    v60 = sub_15F2050((__int64)v39);
                    v61 = sub_1632FA0(v60);
                    v62 = *(unsigned __int8 *)(v249 + 8);
                    if ( v62 != 13 && v62 != 14 )
                      goto LABEL_121;
                    v159 = sub_127FA20(v61, v249);
                    v160 = *(_QWORD **)(v59 + 24);
                    v161 = (unsigned __int64)(v159 + 7) >> 3;
                    if ( *(_DWORD *)(v59 + 32) > 0x40u )
                      v160 = (_QWORD *)*v160;
                    if ( (_QWORD *)v161 != v160 )
                      goto LABEL_121;
                    v262 = 0;
                    v162 = *(_QWORD *)&v39[24 * (1LL - (*((_DWORD *)v39 + 5) & 0xFFFFFFF))];
                    v163 = *(_BYTE *)(v162 + 16);
                    if ( v163 == 13 )
                    {
                      v164 = *(_DWORD *)(v162 + 32);
                      if ( v164 <= 0x40 )
                        v165 = *(_QWORD *)(v162 + 24) == 0;
                      else
                        v165 = v164 == (unsigned int)sub_16A57B0(v162 + 24);
                      if ( !v165 )
                        goto LABEL_121;
                      v262 = (__int64 *)sub_1598F00((__int64 **)v249);
                    }
                    else
                    {
                      if ( v163 != 9 )
                        goto LABEL_121;
                      v262 = (__int64 *)sub_1599EF0((__int64 **)v249);
                    }
                    if ( !v262 )
                      goto LABEL_121;
                    v219 = sub_15603A0((_QWORD *)v39 + 7, 0);
                    v269 = 0u;
                    v270 = 0;
                    sub_14A8180((__int64)v39, v269.m128i_i64, 0);
                    v166 = v270;
                    v229 = v269;
                    sub_1A1B810((__int64)&v292, (__int64)v39, v238);
                    v311 = v166;
                    v310 = v229;
                    v265.m128i_i64[0] = 0;
                    sub_1A1B630((__int64)&v277, (__int64)v39);
                    v167 = *(_QWORD **)(v59 + 24);
                    if ( *(_DWORD *)(v59 + 32) > 0x40u )
                      v167 = (_QWORD *)*v167;
                    v273.m128i_i64[0] = (__int64)"memset.store.fca";
                    LOWORD(v274) = 259;
                    sub_1A202B0((__int64)&v292, v249, (__int64 *)&v262, &v273, v219, &v265, 8LL * (_QWORD)v167);
                    sub_15F20C0(v39);
                    sub_2240A30(v279);
                    v13 = 1;
                    sub_17CD270(v277.m128i_i64);
                    sub_1A1B230((__int64 *)&v292);
                    v36 = v283;
                  }
LABEL_43:
                  if ( v36 )
                    continue;
                  a1 = v256;
                  v10 = v254;
                  break;
                default:
                  goto LABEL_358;
              }
              break;
            }
          }
        }
        if ( (v336 & 1) == 0 )
          j___libc_free_0(v337);
        if ( v331 > 0x40 && v330 )
          j_j___libc_free_0_0(v330);
        if ( v322 != v321 )
          _libc_free((unsigned __int64)v322);
        v43 = v315;
        v44 = (unsigned __int64)&v315[24 * (unsigned int)v316];
        if ( v315 != (_BYTE *)v44 )
        {
          do
          {
            v44 -= 24LL;
            if ( *(_DWORD *)(v44 + 16) > 0x40u )
            {
              v45 = *(_QWORD *)(v44 + 8);
              if ( v45 )
                j_j___libc_free_0_0(v45);
            }
          }
          while ( v43 != (_BYTE *)v44 );
          v44 = (unsigned __int64)v315;
        }
        if ( (_BYTE *)v44 != v317 )
          _libc_free(v44);
        sub_1A28EA0((unsigned __int64 *)&v312, v259, v10);
        if ( !v312 )
        {
          v89 = v318;
          v260 = &v318[v319];
          if ( v318 != v260 )
          {
            v258 = v10;
            do
            {
              v90 = *v89;
              v292 = (_QWORD *)v90;
              v91 = 24LL * (*(_DWORD *)(v90 + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(v90 + 23) & 0x40) != 0 )
              {
                v92 = *(_QWORD *)(v90 - 8);
                v93 = v92 + v91;
              }
              else
              {
                v92 = v90 - v91;
                v93 = v90;
              }
              if ( v92 != v93 )
              {
                v94 = v92;
                do
                {
                  v95 = v94;
                  v94 += 24;
                  sub_1A2F250(a1, v95);
                }
                while ( v93 != v94 );
                v90 = (__int64)v292;
              }
              ++v89;
              v96 = sub_1599EF0(*(__int64 ***)v90);
              sub_164D160(v90, v96, a3, a4, a5, a6, v97, v98, a9, a10);
              sub_1A2EDE0(a1 + 208, (__int64 *)&v292);
            }
            while ( v260 != v89 );
            v10 = v258;
            v13 = 1;
          }
          v99 = v333;
          v100 = &v333[(unsigned int)v334];
          if ( v100 != v333 )
          {
            do
            {
              v101 = *v99++;
              sub_1A2F250(a1, v101);
            }
            while ( v100 != v99 );
            v13 = 1;
          }
          if ( 3LL * (unsigned int)v314 )
          {
            v13 |= sub_1A396A0(a1, (__int64)v10, (__int64)&v312, a3, a4, a5, a6, v46, v47, a9, a10);
            while ( 1 )
            {
              v106 = *(unsigned int *)(a1 + 560);
              if ( !(_DWORD)v106 )
                break;
              v102 = (__int64 *)(*(_QWORD *)(a1 + 552) + 8 * v106 - 8);
              v103 = (_QWORD *)*v102;
              if ( (unsigned __int8)sub_1A27740(a1 + 520, v102, &v292) )
              {
                *v292 = -16;
                --*(_DWORD *)(a1 + 536);
                ++*(_DWORD *)(a1 + 540);
              }
              --*(_DWORD *)(a1 + 560);
              sub_1A2AC20((__int64)v103, *(_QWORD *)(a1 + 8));
              sub_1A3A670((__int64)v103, *(_QWORD *)(a1 + 8), a3, a4, a5, a6, v104, v105, a9, a10);
              if ( !v103[1] )
                sub_15F20C0(v103);
            }
            while ( 1 )
            {
              v107 = *(unsigned int *)(a1 + 624);
              if ( !(_DWORD)v107 )
                break;
              v108 = (__int64 *)(*(_QWORD *)(a1 + 616) + 8 * v107 - 8);
              v109 = (_QWORD *)*v108;
              if ( (unsigned __int8)sub_1A277F0(a1 + 584, v108, &v292) )
              {
                *v292 = -16;
                --*(_DWORD *)(a1 + 600);
                ++*(_DWORD *)(a1 + 604);
              }
              --*(_DWORD *)(a1 + 624);
              sub_1A2C2F0((__int64)v109, a3, a4, a5, a6, v110, v111, a9, a10);
              if ( !v109[1] )
                sub_15F20C0(v109);
            }
          }
        }
        if ( v333 != &v335 )
          _libc_free((unsigned __int64)v333);
        if ( v326 != &v327 )
          _libc_free((unsigned __int64)v326);
        if ( v318 != &v320 )
          _libc_free((unsigned __int64)v318);
        if ( (_BYTE **)v313 != &v315 )
          _libc_free(v313);
        if ( v287 != v286 )
          _libc_free((unsigned __int64)v287);
        if ( v282 != v284 )
          _libc_free((unsigned __int64)v282);
        return v13;
      case 0x4F:
        if ( !*(_QWORD *)(v33 + 8) )
          goto LABEL_98;
        v52 = sub_1A1ABE0(v33);
        if ( v52 )
        {
          if ( v52 == *v32 )
          {
LABEL_94:
            sub_386EA80(&v312, v33);
            LOBYTE(v35) = v313;
            goto LABEL_31;
          }
          goto LABEL_98;
        }
        v239 = v329;
        if ( !v329 )
        {
LABEL_29:
          LOBYTE(v35) = v313;
          goto LABEL_30;
        }
        v292 = (_QWORD *)v33;
        v118 = sub_1A28710((__int64)&v335, (__int64 *)&v292);
        if ( !v118[1] )
        {
          v54 = sub_1A1E740(&v312, v33, (unsigned __int64 *)v118 + 1);
          if ( v54 )
          {
LABEL_103:
            v35 = v313 & 3 | v54 | 4;
            v313 = v35;
            goto LABEL_31;
          }
        }
        v217 = *(_QWORD *)(v33 - 48);
        v119 = *(_QWORD *)(v33 - 24);
        v277.m128i_i64[1] = 0x400000000LL;
        v212 = v119;
        v277.m128i_i64[0] = (__int64)v278;
        sub_1A251A0(v33, (__int64)&v277);
        v292 = &v294;
        v293 = 0x400000000LL;
        sub_1A24E50(v33, (__int64)&v292);
        if ( !v277.m128i_i32[2] )
          goto LABEL_214;
        v239 = 0;
        v251 = v27;
        v204 = *(_QWORD *)(v33 + 40);
        v235 = v29;
        v226 = (__int64 *)v277.m128i_i64[0];
        v195 = v277.m128i_i64[0] + 8LL * v277.m128i_u32[2];
        do
        {
          v120 = *v226;
          if ( !(unsigned __int8)sub_13F86A0(v217, 1 << (*(unsigned __int16 *)(*v226 + 18) >> 1) >> 1, v312, *v226, 0)
            || (v121 = sub_13F86A0(v212, 1 << (*(unsigned __int16 *)(v120 + 18) >> 1) >> 1, v312, v120, 0)) == 0
            || sub_15F32D0(v120)
            || (*(_BYTE *)(v120 + 18) & 1) != 0
            || v204 != *(_QWORD *)(v120 + 40) )
          {
LABEL_213:
            v239 = 0;
            v27 = v251;
            v29 = v235;
            goto LABEL_214;
          }
          v130 = v33 + 24;
          v201 = *(_QWORD *)(v120 - 24);
          v192 = v121;
          v209 = 0;
          v198 = (unsigned __int64)(sub_127FA20(v312, *(_QWORD *)v120) + 7) >> 3;
          while ( 1 )
          {
            v131 = v130 - 24;
            if ( v120 == v130 - 24 )
              break;
            if ( *(_BYTE *)(v130 - 8) == 55 )
            {
              v132 = v292;
              v133 = &v292[(unsigned int)v293];
              if ( v292 == v133 )
              {
LABEL_256:
                v134 = *(_QWORD *)(v130 - 48);
                v189 = *(__int64 **)(v130 - 72);
                v135 = sub_127FA20(v312, *v189);
                v273.m128i_i64[0] = v134;
                v273.m128i_i64[1] = 1;
                v274 = 0;
                v190 = (unsigned __int64)(v135 + 7) >> 3;
                v275 = 0;
                v276 = 0;
                v269.m128i_i64[0] = v201;
                v269.m128i_i64[1] = 1;
                v270 = 0;
                v271 = 0;
                v272 = 0;
                v136 = sub_134CB50((__int64)v333, (__int64)&v269, (__int64)&v273);
                if ( v198 <= v190 && v136 == 3 )
                {
                  v209 = v189;
                }
                else
                {
                  v273.m128i_i64[0] = v134;
                  v273.m128i_i64[1] = v190;
                  v269.m128i_i64[0] = v201;
                  v274 = 0;
                  v275 = 0;
                  v276 = 0;
                  v269.m128i_i64[1] = v198;
                  v270 = 0;
                  v271 = 0;
                  v272 = 0;
                  if ( (unsigned __int8)sub_134CB50((__int64)v333, (__int64)&v269, (__int64)&v273) )
                    goto LABEL_213;
                }
              }
              else
              {
                while ( v131 != *v132 )
                {
                  if ( v133 == ++v132 )
                    goto LABEL_256;
                }
              }
            }
            else if ( (unsigned __int8)sub_15F3040(v131) )
            {
              goto LABEL_213;
            }
            v130 = *(_QWORD *)(v130 + 8);
            if ( !v130 )
              BUG();
          }
          v137 = v239;
          if ( !v209 )
            v137 = v192;
          ++v226;
          v239 = v137;
        }
        while ( (__int64 *)v195 != v226 );
        v27 = v251;
        v29 = v235;
LABEL_214:
        if ( v292 != &v294 )
          _libc_free((unsigned __int64)v292);
        if ( (_QWORD *)v277.m128i_i64[0] != v278 )
          _libc_free(v277.m128i_u64[0]);
        LOBYTE(v35) = v313;
        if ( v239 )
          goto LABEL_31;
        goto LABEL_30;
    }
  }
}
