// Function: sub_24B1E50
// Address: 0x24b1e50
//
__int64 __fastcall sub_24B1E50(__int64 a1, int *a2, _QWORD **a3, __int64 a4)
{
  _BYTE *v4; // rsi
  _QWORD *v5; // rdx
  bool v6; // zf
  __int64 v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rdi
  size_t v12; // rdx
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r14
  char v20; // al
  unsigned __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r12
  __int16 v28; // dx
  __int64 v29; // r9
  char v30; // al
  char v31; // dl
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  size_t *v37; // rax
  __int64 v38; // r14
  __int64 v39; // rbx
  char *v40; // r15
  _BYTE *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // r12
  __int16 v44; // dx
  __int64 v45; // r9
  char v46; // al
  char v47; // dl
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 *v54; // r12
  __int64 v55; // r15
  __int16 v56; // dx
  __int64 v57; // rbx
  _QWORD *v58; // rdi
  __int16 v59; // ax
  __int64 v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // r15
  unsigned __int64 *v65; // rax
  int v66; // ecx
  unsigned __int64 *v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned int v70; // ebx
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // r13
  __int64 v77; // rsi
  int v78; // eax
  __int64 v79; // rsi
  __int64 *v80; // r14
  __int64 *v81; // r15
  signed __int64 v82; // r12
  __int64 v83; // rax
  char *v84; // rbx
  char *v85; // r13
  char *v86; // r8
  __int64 v87; // rdx
  signed __int64 v88; // r9
  __int64 v89; // rax
  __int64 v90; // rcx
  bool v91; // cf
  unsigned __int64 v92; // rax
  unsigned __int64 v93; // r12
  __int64 v94; // rax
  char *v95; // rcx
  unsigned __int64 v96; // r12
  char *v97; // r14
  __int64 v98; // rax
  char *v99; // rbx
  _BYTE *v100; // rsi
  __int64 v101; // r15
  __int64 v102; // r13
  unsigned __int64 v103; // rax
  int v104; // edx
  _BYTE *v105; // r12
  _BYTE *v106; // rax
  unsigned int v107; // esi
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // rax
  __int64 v111; // rax
  char *v112; // rax
  __int64 v113; // rax
  __int64 **v114; // rax
  __int64 *v115; // rdx
  __int64 *v116; // r14
  __int64 v117; // r15
  __int64 v118; // r12
  char v119; // al
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rbx
  __int64 v124; // rbx
  __int64 v125; // r13
  __int64 v126; // rbx
  __int64 v127; // rax
  _DWORD *v128; // rax
  __int64 *v129; // rdx
  unsigned int v130; // ecx
  int v131; // eax
  __int64 v132; // rax
  _BYTE *v133; // rdi
  __int64 v134; // r8
  __int64 v135; // r13
  _QWORD *v136; // r12
  char *v137; // r13
  __int64 v138; // r9
  unsigned __int64 v139; // rax
  __m128i *v140; // rax
  __int64 v141; // r12
  __int64 v142; // rbx
  __m128i *v143; // r13
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // r11
  __int64 v151; // r10
  __m128i *v152; // rdx
  int v153; // eax
  __int64 v154; // rcx
  unsigned int v155; // eax
  int v156; // r15d
  _QWORD *v157; // rax
  __int64 v158; // r12
  __int64 v159; // r13
  char *v160; // r13
  char *v161; // rbx
  __int64 v162; // rdx
  unsigned int v163; // esi
  unsigned __int64 v164; // rbx
  unsigned __int64 *v165; // r12
  unsigned __int64 v166; // rdi
  int v167; // r13d
  __int64 *v168; // rax
  __int64 **v169; // r13
  unsigned int v170; // ebx
  unsigned int v171; // eax
  __int64 (__fastcall *v172)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v173; // rax
  char *v174; // r13
  char *v175; // rbx
  __int64 v176; // rdx
  unsigned int v177; // esi
  unsigned int v178; // r12d
  __int64 **v179; // r13
  __int64 (__fastcall *v180)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v181; // rbx
  __int64 v182; // rax
  __int64 v183; // rcx
  unsigned int v184; // edx
  __int64 *v185; // rax
  __int64 v186; // rdi
  __int64 v187; // rax
  _QWORD **v188; // rdi
  __int64 v189; // rax
  __int64 v190; // r9
  int v191; // ecx
  unsigned __int64 v192; // rdx
  __int64 v193; // rbx
  __int64 v194; // r12
  __m128i *v195; // rsi
  _QWORD *v196; // rax
  __int64 (__fastcall *v197)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 *v198; // rbx
  __int64 v199; // rax
  unsigned __int64 *v200; // rax
  unsigned __int64 v201; // r12
  unsigned __int64 v202; // rbx
  int v203; // r13d
  char *v204; // r13
  __int64 v205; // rax
  char *v206; // r15
  __int64 v207; // rdx
  unsigned int v208; // esi
  int v209; // eax
  __int64 v210; // rax
  __int64 v211; // r15
  __int64 v212; // rbx
  __int64 v213; // r12
  __int64 v214; // r14
  __int64 v215; // rax
  __int64 v216; // rbx
  __int64 v217; // r12
  __int64 v218; // r13
  __int64 v219; // rax
  __int64 v220; // rcx
  void *v221; // rdi
  size_t v222; // rdx
  char *v223; // r8
  size_t v224; // rdx
  unsigned __int64 v225; // rcx
  unsigned __int64 v226; // rcx
  unsigned __int64 v227; // rsi
  __int64 v228; // rax
  char *v229; // r10
  char *v230; // rcx
  char *v231; // rax
  char *v232; // rdx
  _QWORD *v233; // rsi
  char *v234; // rax
  char *v235; // r12
  char *v236; // r12
  __int64 v237; // rax
  int v238; // eax
  __int64 v239; // rbx
  unsigned int v240; // r14d
  _QWORD *v241; // rax
  char *v242; // r15
  char *v243; // r13
  __int64 v244; // rdx
  unsigned int v245; // esi
  __m128i *v246; // r12
  __m128i *v247; // rax
  int v248; // ebx
  __m128i *v249; // rcx
  _QWORD *v250; // rax
  int v251; // ebx
  int v252; // r8d
  __int64 v253; // [rsp+8h] [rbp-528h]
  char *v254; // [rsp+10h] [rbp-520h]
  char *v255; // [rsp+18h] [rbp-518h]
  size_t v256; // [rsp+20h] [rbp-510h]
  char *v257; // [rsp+20h] [rbp-510h]
  __int64 v258; // [rsp+28h] [rbp-508h]
  __int64 v259; // [rsp+38h] [rbp-4F8h]
  __int64 v260; // [rsp+48h] [rbp-4E8h]
  __int64 *v261; // [rsp+50h] [rbp-4E0h]
  int v263; // [rsp+64h] [rbp-4CCh]
  __int64 v264; // [rsp+68h] [rbp-4C8h]
  __int64 v265; // [rsp+78h] [rbp-4B8h]
  _QWORD **v266; // [rsp+80h] [rbp-4B0h]
  unsigned int v267; // [rsp+90h] [rbp-4A0h]
  __int64 v268; // [rsp+90h] [rbp-4A0h]
  int v269; // [rsp+98h] [rbp-498h]
  __int64 *v270; // [rsp+A8h] [rbp-488h]
  __int64 *v271; // [rsp+A8h] [rbp-488h]
  _QWORD *v272; // [rsp+B0h] [rbp-480h]
  __int64 v273; // [rsp+B8h] [rbp-478h]
  __int64 v274; // [rsp+B8h] [rbp-478h]
  void *v275; // [rsp+C0h] [rbp-470h]
  unsigned int v276; // [rsp+C0h] [rbp-470h]
  unsigned int v277; // [rsp+C0h] [rbp-470h]
  __int64 v278; // [rsp+C0h] [rbp-470h]
  char *v279; // [rsp+C0h] [rbp-470h]
  int v280; // [rsp+D0h] [rbp-460h]
  unsigned int v281; // [rsp+D4h] [rbp-45Ch]
  unsigned int v282; // [rsp+D8h] [rbp-458h]
  __m128i *v283; // [rsp+D8h] [rbp-458h]
  unsigned __int64 v284; // [rsp+D8h] [rbp-458h]
  char v286; // [rsp+E8h] [rbp-448h]
  unsigned __int64 v287; // [rsp+E8h] [rbp-448h]
  unsigned int i; // [rsp+E8h] [rbp-448h]
  __int64 v289; // [rsp+E8h] [rbp-448h]
  char v290; // [rsp+F0h] [rbp-440h]
  __int64 v291; // [rsp+F0h] [rbp-440h]
  __int64 v292; // [rsp+F0h] [rbp-440h]
  __int64 v293; // [rsp+F0h] [rbp-440h]
  char *src; // [rsp+F8h] [rbp-438h]
  char *srca; // [rsp+F8h] [rbp-438h]
  __int64 n; // [rsp+100h] [rbp-430h]
  unsigned int na; // [rsp+100h] [rbp-430h]
  __int64 v298; // [rsp+108h] [rbp-428h]
  char *v299; // [rsp+108h] [rbp-428h]
  char *v300; // [rsp+108h] [rbp-428h]
  char *v301; // [rsp+108h] [rbp-428h]
  __int64 v302; // [rsp+108h] [rbp-428h]
  __int64 v303; // [rsp+110h] [rbp-420h]
  unsigned int v304; // [rsp+13Ch] [rbp-3F4h] BYREF
  unsigned __int64 v305; // [rsp+140h] [rbp-3F0h] BYREF
  size_t v306; // [rsp+148h] [rbp-3E8h] BYREF
  unsigned __int64 v307; // [rsp+150h] [rbp-3E0h] BYREF
  _BYTE *v308; // [rsp+158h] [rbp-3D8h]
  _BYTE *v309; // [rsp+160h] [rbp-3D0h]
  __int64 v310[6]; // [rsp+170h] [rbp-3C0h] BYREF
  void *v311; // [rsp+1A0h] [rbp-390h] BYREF
  size_t v312; // [rsp+1A8h] [rbp-388h]
  _QWORD v313[2]; // [rsp+1B0h] [rbp-380h] BYREF
  __int16 v314; // [rsp+1C0h] [rbp-370h]
  void **v315; // [rsp+1D0h] [rbp-360h] BYREF
  void *v316[6]; // [rsp+1E0h] [rbp-350h] BYREF
  void *s; // [rsp+210h] [rbp-320h] BYREF
  __int64 v318; // [rsp+218h] [rbp-318h]
  _QWORD *v319; // [rsp+220h] [rbp-310h]
  __int64 v320; // [rsp+228h] [rbp-308h]
  int v321; // [rsp+230h] [rbp-300h]
  __int64 v322; // [rsp+238h] [rbp-2F8h]
  _QWORD v323[2]; // [rsp+240h] [rbp-2F0h] BYREF
  void *dest; // [rsp+250h] [rbp-2E0h] BYREF
  size_t v325; // [rsp+258h] [rbp-2D8h]
  __m128i v326; // [rsp+260h] [rbp-2D0h] BYREF
  void *v327; // [rsp+270h] [rbp-2C0h]
  void *v328; // [rsp+278h] [rbp-2B8h]
  __int64 v329; // [rsp+280h] [rbp-2B0h]
  __m128i *v330; // [rsp+290h] [rbp-2A0h] BYREF
  __int64 v331; // [rsp+298h] [rbp-298h]
  _QWORD v332[2]; // [rsp+2A0h] [rbp-290h] BYREF
  int v333; // [rsp+2B0h] [rbp-280h]
  char v334; // [rsp+2B4h] [rbp-27Ch]
  char *v335; // [rsp+2E0h] [rbp-250h] BYREF
  __int64 v336; // [rsp+2E8h] [rbp-248h]
  size_t *v337; // [rsp+2F0h] [rbp-240h] BYREF
  void **v338; // [rsp+2F8h] [rbp-238h]
  void **p_dest; // [rsp+300h] [rbp-230h]
  __int64 v340; // [rsp+310h] [rbp-220h]
  __int64 v341; // [rsp+318h] [rbp-218h]
  __int64 v342; // [rsp+320h] [rbp-210h]
  _QWORD *v343; // [rsp+328h] [rbp-208h]
  void **v344; // [rsp+330h] [rbp-200h]
  void **v345; // [rsp+338h] [rbp-1F8h]
  __int64 v346; // [rsp+340h] [rbp-1F0h]
  int v347; // [rsp+348h] [rbp-1E8h]
  __int16 v348; // [rsp+34Ch] [rbp-1E4h]
  char v349; // [rsp+34Eh] [rbp-1E2h]
  __int64 v350; // [rsp+350h] [rbp-1E0h]
  __int64 v351; // [rsp+358h] [rbp-1D8h]
  void *v352; // [rsp+360h] [rbp-1D0h] BYREF
  void *v353; // [rsp+368h] [rbp-1C8h] BYREF
  __int64 v354[2]; // [rsp+370h] [rbp-1C0h] BYREF
  char v355; // [rsp+380h] [rbp-1B0h]
  char v356; // [rsp+390h] [rbp-1A0h]
  char v357; // [rsp+391h] [rbp-19Fh]
  __int64 v358; // [rsp+398h] [rbp-198h]
  __int64 v359; // [rsp+3B0h] [rbp-180h] BYREF
  int v360; // [rsp+3B8h] [rbp-178h]
  int v361; // [rsp+3BCh] [rbp-174h]
  unsigned int *v362; // [rsp+3C0h] [rbp-170h]
  unsigned int v363; // [rsp+3C8h] [rbp-168h]
  unsigned __int64 v364; // [rsp+3D0h] [rbp-160h]
  __int64 v365; // [rsp+3D8h] [rbp-158h]
  char v366; // [rsp+3E8h] [rbp-148h]
  unsigned __int64 v367; // [rsp+430h] [rbp-100h]
  __int64 v368; // [rsp+438h] [rbp-F8h]
  char v369[8]; // [rsp+440h] [rbp-F0h] BYREF
  __int64 *v370; // [rsp+448h] [rbp-E8h]
  __int64 *v371; // [rsp+450h] [rbp-E0h]
  _BYTE v372[136]; // [rsp+4A8h] [rbp-88h] BYREF

  v265 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  v280 = *a2;
  if ( *a2 == 1 )
    sub_24A5670((__int64)a3, 1);
  v4 = a3[29];
  v5 = a3[30];
  v315 = v316;
  sub_24A3020((__int64 *)&v315, v4, (__int64)v5 + (_QWORD)v4);
  v6 = *((_DWORD *)a3 + 71) == 3;
  v316[2] = a3[33];
  v316[3] = a3[34];
  v316[4] = a3[35];
  if ( !v6 && LOBYTE(qword_4F8A568[8]) )
  {
    v354[0] = (__int64)"VTable value profiling is presently not supported for non-ELF object formats";
    v36 = (__int64)*a3;
    v37 = a3[21];
    v357 = 1;
    v356 = 3;
    v337 = v37;
    v336 = 0x100000017LL;
    v335 = (char *)&unk_49D9CA8;
    v338 = (void **)v354;
    sub_B6EB20(v36, (__int64)&v335);
  }
  s = v323;
  v318 = 1;
  v319 = 0;
  v320 = 0;
  v321 = 1065353216;
  v322 = 0;
  v323[0] = 0;
  v266 = a3 + 3;
  if ( byte_4FEC008 )
  {
    sub_24B1CF0(a3, (unsigned __int64 *)&s);
    v272 = a3[4];
    if ( v272 == a3 + 3 )
      goto LABEL_16;
  }
  else
  {
    v11 = v323;
    v12 = 8;
    v272 = a3[4];
    if ( v266 == v272 )
      goto LABEL_19;
  }
  do
  {
    v7 = 0;
    if ( v272 )
      v7 = (__int64)(v272 - 7);
    v273 = v7;
    v8 = v7;
    if ( sub_B2FC80(v7)
      || sub_24A3530(v8)
      || (unsigned __int8)sub_B2D610(v8, 20)
      || (unsigned __int8)sub_B2D610(v8, 33)
      || (unsigned __int8)sub_B2D610(v8, 66)
      || (unsigned int)sub_B2BED0(v8) < (unsigned int)qword_4FEACE8 )
    {
      goto LABEL_15;
    }
    if ( LOBYTE(qword_4FEA8E0[17]) )
    {
      sub_B2EE70((__int64)v354, v273, 0);
      if ( v355 )
      {
        if ( v354[0] > (unsigned __int64)qword_4FEAB28 )
          goto LABEL_15;
      }
      else if ( !byte_4FEAA48 )
      {
        goto LABEL_15;
      }
    }
    v14 = sub_BC1CD0(v265, &unk_4F6D3F8, v273) + 8;
    v15 = sub_BC1CD0(v265, &unk_4F8E5A8, v273) + 8;
    v16 = sub_BC1CD0(v265, &unk_4F8D9A8, v273) + 8;
    v19 = sub_BC1CD0(v265, &unk_4F875F0, v273) + 8;
    v20 = qword_4FEB4E8;
    if ( !(_BYTE)qword_4FEB4E8 )
    {
      sub_F429C0(v273, 0, v15, v16, v17, v18);
      v20 = qword_4FEB4E8;
    }
    if ( byte_4FEB788 )
    {
      sub_24AF800(v354, v273, v14, (__int64)&s, v280 != 3, v15, v16, v19, v280 == 2, 1u, qword_4FEB6A8, v20);
      v21 = v273;
      if ( v280 != 3 )
        goto LABEL_31;
    }
    else
    {
      if ( v280 != 3 )
      {
        sub_24AF800(v354, v273, v14, (__int64)&s, 1, v15, v16, v19, v280 == 2, 0, qword_4FEB6A8, v20);
LABEL_31:
        v21 = v367;
        goto LABEL_32;
      }
      sub_24AF800(v354, v273, v14, (__int64)&s, 0, v15, v16, v19, 0, 1u, qword_4FEB6A8, v20);
      v21 = v273;
    }
LABEL_32:
    v22 = v368;
    v305 = v21;
    v23 = sub_BCB2E0(*a3);
    v24 = sub_ACD640(v23, v22, 0);
    v25 = *a3;
    v306 = v24;
    v26 = sub_BCE3C0(v25, 0);
    v275 = (void *)sub_ADB060(v305, v26);
    if ( (_BYTE)qword_4FEB5C8 )
    {
      v27 = *(_QWORD *)(v273 + 80);
      if ( v27 )
        v27 -= 24;
      v29 = sub_AA5190(v27);
      if ( v29 )
      {
        v30 = v28;
        v31 = HIBYTE(v28);
      }
      else
      {
        v31 = 0;
        v30 = 0;
      }
      v32 = v264;
      LOBYTE(v32) = v30;
      v33 = v32;
      BYTE1(v33) = v31;
      v264 = v33;
      sub_2412230((__int64)&v335, v27, v29, v33, 0, v29, 0, 0);
      LOWORD(v333) = 257;
      dest = v275;
      HIDWORD(v311) = 0;
      v325 = v306;
      v34 = sub_BCB2D0(v343);
      v326.m128i_i64[0] = sub_ACD640(v34, 1, 0);
      v35 = sub_BCB2D0(v343);
      v326.m128i_i64[1] = sub_ACD640(v35, 0, 0);
      sub_B33D10((__int64)&v335, 0xC5u, 0, 0, (int)&dest, 4, (__int64)v311, (__int64)&v330);
      nullsub_61();
      v352 = &unk_49DA100;
      nullsub_63();
      if ( v335 != (char *)&v337 )
        _libc_free((unsigned __int64)v335);
      goto LABEL_39;
    }
    v307 = 0;
    v308 = 0;
    v309 = 0;
    if ( v372[80] )
    {
      v38 = *(_QWORD *)(v354[0] + 80);
      v39 = v354[0] + 72;
      if ( v38 != v354[0] + 72 )
      {
        while ( 1 )
        {
          v40 = (char *)(v38 - 24);
          if ( !v38 )
            v40 = 0;
          if ( !(unsigned __int8)sub_3158140(v372, v40) )
            goto LABEL_45;
          v335 = v40;
          v41 = v308;
          if ( v308 == v309 )
          {
            sub_F38A10((__int64)&v307, v308, &v335);
LABEL_45:
            v38 = *(_QWORD *)(v38 + 8);
            if ( v39 == v38 )
              goto LABEL_53;
          }
          else
          {
            if ( v308 )
            {
              *(_QWORD *)v308 = v40;
              v41 = v308;
            }
            v308 = v41 + 8;
            v38 = *(_QWORD *)(v38 + 8);
            if ( v39 == v38 )
              goto LABEL_53;
          }
        }
      }
      LODWORD(v42) = 0;
      goto LABEL_54;
    }
    v80 = v371;
    v81 = v370;
    v82 = (char *)v371 - (char *)v370;
    if ( (unsigned __int64)((char *)v371 - (char *)v370) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"vector::reserve");
    if ( v82 )
    {
      v83 = sub_22077B0((char *)v371 - (char *)v370);
      v81 = v370;
      v80 = v371;
      v84 = (char *)v83;
      src = (char *)(v83 + v82);
      if ( v370 == v371 )
      {
        v287 = v83;
      }
      else
      {
LABEL_114:
        v85 = src;
        v86 = v84;
        do
        {
          while ( 1 )
          {
            v87 = *v81;
            if ( v85 != v84 )
              break;
            v88 = v85 - v86;
            v89 = (v85 - v86) >> 3;
            if ( v89 == 0xFFFFFFFFFFFFFFFLL )
              sub_4262D8((__int64)"vector::_M_realloc_insert");
            v90 = 1;
            if ( v89 )
              v90 = (v85 - v86) >> 3;
            v91 = __CFADD__(v90, v89);
            v92 = v90 + v89;
            if ( v91 )
            {
              v93 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_127:
              srca = v86;
              n = v85 - v86;
              v298 = *v81;
              v94 = sub_22077B0(v93);
              v87 = v298;
              v88 = n;
              v86 = srca;
              v95 = (char *)v94;
              v96 = v94 + v93;
              goto LABEL_128;
            }
            if ( v92 )
            {
              if ( v92 > 0xFFFFFFFFFFFFFFFLL )
                v92 = 0xFFFFFFFFFFFFFFFLL;
              v93 = 8 * v92;
              goto LABEL_127;
            }
            v96 = 0;
            v95 = 0;
LABEL_128:
            if ( &v95[v88] )
              *(_QWORD *)&v95[v88] = v87;
            v84 = &v95[v88 + 8];
            if ( v88 > 0 )
            {
              v300 = v86;
              v112 = (char *)memmove(v95, v86, v88);
              v86 = v300;
              v95 = v112;
LABEL_167:
              v301 = v95;
              j_j___libc_free_0((unsigned __int64)v86);
              v95 = v301;
              goto LABEL_132;
            }
            if ( v86 )
              goto LABEL_167;
LABEL_132:
            ++v81;
            v85 = (char *)v96;
            v86 = v95;
            if ( v80 == v81 )
              goto LABEL_133;
          }
          if ( v84 )
            *(_QWORD *)v84 = v87;
          ++v81;
          v84 += 8;
        }
        while ( v80 != v81 );
LABEL_133:
        v287 = (unsigned __int64)v86;
        if ( v84 != v86 )
        {
          v299 = v84;
          v97 = v86;
          while ( 1 )
          {
            v101 = *(_QWORD *)v97;
            if ( *(_BYTE *)(*(_QWORD *)v97 + 24LL) || *(_BYTE *)(v101 + 25) )
              goto LABEL_142;
            v99 = *(char **)v101;
            v102 = *(_QWORD *)(v101 + 8);
            if ( *(_QWORD *)v101 )
            {
              if ( !v102 )
                goto LABEL_138;
              n = (__int64)(v99 + 48);
              v103 = *((_QWORD *)v99 + 6) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v99 + 48 == (char *)v103 )
              {
                v105 = 0;
              }
              else
              {
                if ( !v103 )
                  goto LABEL_448;
                v104 = *(unsigned __int8 *)(v103 - 24);
                v105 = 0;
                v106 = (_BYTE *)(v103 - 24);
                if ( (unsigned int)(v104 - 30) < 0xB )
                  v105 = v106;
              }
              if ( (unsigned int)sub_B46E30((__int64)v105) <= 1 )
              {
                v111 = sub_AA5190((__int64)v99);
                if ( !v111 || n != v111 )
                  goto LABEL_138;
              }
              else if ( *(_BYTE *)(v101 + 26) )
              {
                v107 = sub_D0E820((__int64)v99, v102);
                if ( *v105 != 33 )
                {
                  LOWORD(p_dest) = 257;
                  v330 = 0;
                  v331 = 0;
                  v332[0] = 0;
                  v332[1] = 0;
                  v333 = 0;
                  v334 = 1;
                  v108 = sub_F451F0((__int64)v105, v107, (__int64)&v330, (void **)&v335);
                  v109 = v108;
                  if ( v108 )
                  {
                    n = (__int64)v369;
                    sub_24A7690((__int64)v369, (__int64)v99, v108, 0);
                    *(_BYTE *)(sub_24A7690((__int64)v369, v109, v102, 0) + 24) = 1;
                    *(_BYTE *)(v101 + 25) = 1;
                    v110 = sub_AA5190(v109);
                    if ( !v110 || v110 != v109 + 48 )
                    {
                      v99 = (char *)v109;
LABEL_138:
                      v335 = v99;
                      v100 = v308;
                      if ( v308 != v309 )
                        goto LABEL_139;
                      goto LABEL_160;
                    }
                  }
                }
              }
              else
              {
                v98 = sub_AA5190(v102);
                if ( !v98 || v98 != v102 + 48 )
                {
                  v99 = (char *)v102;
                  goto LABEL_138;
                }
              }
LABEL_142:
              v97 += 8;
              if ( v299 == v97 )
                break;
            }
            else
            {
              v335 = *(char **)(v101 + 8);
              if ( !v102 )
                goto LABEL_142;
              v100 = v308;
              v99 = (char *)v102;
              if ( v308 != v309 )
              {
LABEL_139:
                if ( v100 )
                {
                  *(_QWORD *)v100 = v99;
                  v100 = v308;
                }
                v308 = v100 + 8;
                goto LABEL_142;
              }
LABEL_160:
              v97 += 8;
              sub_9319A0((__int64)&v307, v100, &v335);
              if ( v299 == v97 )
                break;
            }
          }
        }
      }
      if ( v287 )
        j_j___libc_free_0(v287);
      goto LABEL_53;
    }
    src = 0;
    v84 = 0;
    if ( v371 != v370 )
      goto LABEL_114;
LABEL_53:
    v42 = (__int64)&v308[-v307] >> 3;
LABEL_54:
    v267 = v360 + v42;
    if ( v280 != 3 )
      goto LABEL_55;
    v210 = sub_B6E160((__int64 *)a3, 0xC4u, 0, 0);
    LODWORD(v311) = 0;
    v330 = (__m128i *)v210;
    v211 = *(_QWORD *)(v273 + 80);
    v302 = v273 + 72;
    if ( v211 == v273 + 72 )
      goto LABEL_55;
    v212 = 0x8000000000041LL;
    do
    {
      if ( !v211 )
LABEL_447:
        BUG();
      v213 = *(_QWORD *)(v211 + 32);
      v214 = v211 + 24;
      if ( v213 != v211 + 24 )
      {
        while ( 1 )
        {
          if ( !v213 )
            goto LABEL_448;
          if ( (unsigned __int8)(*(_BYTE *)(v213 - 24) - 34) > 0x33u
            || !_bittest64(&v212, (unsigned int)*(unsigned __int8 *)(v213 - 24) - 34)
            || **(_BYTE **)(v213 - 56) == 25 )
          {
            goto LABEL_329;
          }
          if ( sub_B491E0(v213 - 24) )
          {
            LODWORD(v311) = (_DWORD)v311 + 1;
LABEL_336:
            v213 = *(_QWORD *)(v213 + 8);
            if ( v214 == v213 )
              break;
          }
          else
          {
            v215 = *(_QWORD *)(v213 - 56);
            if ( v215
              && !*(_BYTE *)v215
              && *(_QWORD *)(v215 + 24) == *(_QWORD *)(v213 + 56)
              && (*(_BYTE *)(v215 + 33) & 0x20) == 0 )
            {
              LODWORD(v311) = (_DWORD)v311 + 1;
              goto LABEL_336;
            }
LABEL_329:
            v213 = *(_QWORD *)(v213 + 8);
            if ( v214 == v213 )
              break;
          }
        }
      }
      v211 = *(_QWORD *)(v211 + 8);
    }
    while ( v273 + 72 != v211 );
    v216 = *(_QWORD *)(v273 + 80);
    LODWORD(dest) = 0;
    v335 = (char *)&v330;
    v336 = (__int64)&v305;
    v337 = &v306;
    v338 = &v311;
    p_dest = &dest;
    if ( v302 != v216 )
    {
      while ( 1 )
      {
        if ( !v216 )
          goto LABEL_447;
        v217 = *(_QWORD *)(v216 + 32);
        v218 = v216 + 24;
        if ( v217 != v216 + 24 )
          break;
LABEL_352:
        v216 = *(_QWORD *)(v216 + 8);
        if ( v216 == v302 )
          goto LABEL_55;
      }
      while ( v217 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v217 - 24) - 34) <= 0x33u
          && (v220 = 0x8000000000041LL, _bittest64(&v220, (unsigned int)*(unsigned __int8 *)(v217 - 24) - 34))
          && **(_BYTE **)(v217 - 56) != 25
          && (sub_B491E0(v217 - 24)
           || (v219 = *(_QWORD *)(v217 - 56)) != 0
           && !*(_BYTE *)v219
           && *(_QWORD *)(v219 + 24) == *(_QWORD *)(v217 + 56)
           && (*(_BYTE *)(v219 + 33) & 0x20) == 0) )
        {
          sub_24AB080((__int64)&v335, v217 - 24);
          v217 = *(_QWORD *)(v217 + 8);
          if ( v218 == v217 )
            goto LABEL_352;
        }
        else
        {
          v217 = *(_QWORD *)(v217 + 8);
          if ( v218 == v217 )
            goto LABEL_352;
        }
      }
LABEL_448:
      BUG();
    }
LABEL_55:
    v304 = 0;
    if ( (_BYTE)qword_4FEB328 )
    {
      v267 += (_BYTE)qword_4FEB4E8 == 0 ? 1 : 8;
      v43 = *(_QWORD *)(v273 + 80);
      if ( v43 )
        v43 -= 24;
      v45 = sub_AA5190(v43);
      if ( v45 )
      {
        v46 = v44;
        v47 = HIBYTE(v44);
      }
      else
      {
        v47 = 0;
        v46 = 0;
      }
      v48 = v260;
      LOBYTE(v48) = v46;
      v49 = v48;
      BYTE1(v49) = v47;
      v260 = v49;
      sub_2412230((__int64)&v335, v43, v45, v49, 0, v45, 0, 0);
      LOWORD(v333) = 257;
      dest = v275;
      HIDWORD(v311) = 0;
      v325 = v306;
      v50 = sub_BCB2D0(v343);
      v51 = sub_ACD640(v50, v267, 0);
      v52 = v304;
      v326.m128i_i64[0] = v51;
      v53 = sub_BCB2D0(v343);
      v326.m128i_i64[1] = sub_ACD640(v53, v52, 0);
      sub_B33D10((__int64)&v335, 0xCAu, 0, 0, (int)&dest, 4, (__int64)v311, (__int64)&v330);
      v304 += (_BYTE)qword_4FEB4E8 == 0 ? 1 : 8;
      nullsub_61();
      v352 = &unk_49DA100;
      nullsub_63();
      if ( v335 != (char *)&v337 )
        _libc_free((unsigned __int64)v335);
    }
    v54 = (__int64 *)v307;
    v270 = (__int64 *)v308;
    if ( (_BYTE *)v307 != v308 )
    {
      n = (__int64)&unk_49DA100;
      while ( 1 )
      {
        v55 = *v54;
        v57 = sub_AA5190(*v54);
        if ( v57 )
        {
          v290 = v56;
          v286 = HIBYTE(v56);
        }
        else
        {
          v286 = 0;
          v290 = 0;
        }
        v58 = (_QWORD *)sub_AA48A0(v55);
        v345 = &v353;
        v343 = v58;
        v335 = (char *)&v337;
        v336 = 0x200000000LL;
        v346 = 0;
        v344 = &v352;
        v347 = 0;
        v352 = &unk_49DA100;
        v348 = 512;
        v349 = 7;
        v353 = &unk_49DA0B0;
        LOBYTE(v59) = v290;
        v350 = 0;
        HIBYTE(v59) = v286;
        v340 = v55;
        v351 = 0;
        v341 = v57;
        LOWORD(v342) = v59;
        if ( v57 == v55 + 48 )
          goto LABEL_78;
        v60 = v57 - 24;
        if ( !v57 )
          v60 = 0;
        v61 = *(_QWORD *)sub_B46C60(v60);
        v330 = (__m128i *)v61;
        if ( !v61 )
          break;
        sub_B96E90((__int64)&v330, v61, 1);
        v64 = (__int64)v330;
        if ( !v330 )
          break;
        v65 = (unsigned __int64 *)v335;
        v66 = v336;
        v67 = (unsigned __int64 *)&v335[16 * (unsigned int)v336];
        if ( v335 == (char *)v67 )
        {
LABEL_105:
          if ( (unsigned int)v336 >= (unsigned __int64)HIDWORD(v336) )
          {
            v202 = v258 & 0xFFFFFFFF00000000LL;
            v258 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v336) < (unsigned __int64)(unsigned int)v336 + 1 )
            {
              sub_C8D5F0((__int64)&v335, &v337, (unsigned int)v336 + 1LL, 0x10u, v62, v63);
              v67 = (unsigned __int64 *)&v335[16 * (unsigned int)v336];
            }
            *v67 = v202;
            v67[1] = v64;
            v64 = (__int64)v330;
            LODWORD(v336) = v336 + 1;
          }
          else
          {
            if ( v67 )
            {
              *(_DWORD *)v67 = 0;
              v67[1] = v64;
              v66 = v336;
              v64 = (__int64)v330;
            }
            LODWORD(v336) = v66 + 1;
          }
LABEL_103:
          if ( !v64 )
            goto LABEL_77;
          goto LABEL_76;
        }
        while ( *(_DWORD *)v65 )
        {
          v65 += 2;
          if ( v67 == v65 )
            goto LABEL_105;
        }
        v65[1] = (unsigned __int64)v330;
LABEL_76:
        sub_B91220((__int64)&v330, v64);
LABEL_77:
        v58 = v343;
LABEL_78:
        HIDWORD(v311) = 0;
        LOWORD(v333) = 257;
        dest = v275;
        v325 = v306;
        v68 = sub_BCB2D0(v58);
        v69 = sub_ACD640(v68, v267, 0);
        v70 = v304;
        v326.m128i_i64[0] = v69;
        ++v304;
        v71 = sub_BCB2D0(v343);
        v326.m128i_i64[1] = sub_ACD640(v71, v70, 0);
        sub_B33D10(
          (__int64)&v335,
          ((_BYTE)qword_4FEB4E8 == 0) + 197,
          0,
          0,
          (int)&dest,
          4,
          (__int64)v311,
          (__int64)&v330);
        nullsub_61();
        v352 = &unk_49DA100;
        nullsub_63();
        if ( v335 != (char *)&v337 )
          _libc_free((unsigned __int64)v335);
        if ( v270 == ++v54 )
          goto LABEL_81;
      }
      sub_93FB40((__int64)&v335, 0);
      v64 = (__int64)v330;
      goto LABEL_103;
    }
LABEL_81:
    v361 = 1;
    v362 = &v304;
    v363 = v267;
    v365 = v368;
    v364 = v305;
    v72 = *(_QWORD *)(v359 + 80);
    v73 = v359 + 72;
    while ( v73 != v72 )
    {
      v74 = v72;
      v72 = *(_QWORD *)(v72 + 8);
      v75 = *(_QWORD *)(v74 + 32);
      v76 = v74 + 24;
LABEL_84:
      while ( v76 != v75 )
      {
        while ( 1 )
        {
          v77 = v75;
          v75 = *(_QWORD *)(v75 + 8);
          v78 = *(unsigned __int8 *)(v77 - 24);
          if ( v78 == 86 )
            break;
          if ( (unsigned int)(v78 - 29) <= 0x39 )
          {
            if ( (unsigned int)(v78 - 30) > 0x37 )
              goto LABEL_448;
            goto LABEL_84;
          }
          if ( (unsigned int)(v78 - 87) > 9 )
            goto LABEL_448;
          if ( v76 == v75 )
            goto LABEL_89;
        }
        if ( (_BYTE)qword_4FEBC88
          && !(_BYTE)qword_4FEB5C8
          && !v366
          && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v77 - 120) + 8LL) + 8LL) - 17 > 1 )
        {
          v79 = v77 - 24;
          if ( v361 == 1 )
          {
            sub_24AAB30((__int64)&v359, v79);
          }
          else if ( v361 == 2 )
          {
            sub_24ADFC0(&v359, v79);
          }
          else
          {
            if ( v361 )
              goto LABEL_448;
            ++v360;
          }
        }
      }
LABEL_89:
      ;
    }
    if ( (unsigned __int8)qword_4FEC2A8 | (v280 == 3) )
      goto LABEL_91;
    if ( (*(_BYTE *)(v273 + 2) & 8) != 0 )
    {
      v237 = sub_B2E500(v273);
      v238 = sub_B2A630(v237);
      if ( v238 > 10 )
      {
        if ( v238 == 12 )
        {
LABEL_395:
          sub_B2AF20((__int64)&v335, v273);
          sub_C7D6A0(0, 0, 8);
          v239 = v336;
          ++v335;
          v269 = (int)v337;
          v268 = v336;
          v240 = (unsigned int)v338;
          v263 = (int)v338;
          v336 = 0;
          v337 = 0;
          LODWORD(v338) = 0;
          sub_C7D6A0(0, 0, 8);
          v259 = 16LL * v240;
          v261 = (__int64 *)(v259 + v239);
          goto LABEL_171;
        }
      }
      else if ( v238 > 6 )
      {
        goto LABEL_395;
      }
    }
    v261 = 0;
    v259 = 0;
    v263 = 0;
    v269 = 0;
    v268 = 0;
LABEL_171:
    v274 = 0;
    v113 = v358;
    while ( v274 == 1 )
    {
      if ( (_BYTE)qword_4FEB948 )
      {
        v115 = *(__int64 **)(v113 + 24);
        v271 = *(__int64 **)(v113 + 32);
        if ( v115 != v271 )
        {
LABEL_174:
          v116 = v115;
          for ( i = 0; ; i = v277 )
          {
            v117 = *v116;
            v118 = v116[2];
            sub_23D0AB0((__int64)&v335, v116[1], 0, 0, 0);
            v119 = *(_BYTE *)(*(_QWORD *)(v117 + 8) + 8LL);
            if ( v119 != 12 )
            {
              if ( v119 != 14 )
              {
                v117 = 0;
                goto LABEL_178;
              }
              LOWORD(v327) = 257;
              v179 = (__int64 **)sub_BCB2E0(v343);
              if ( v179 == *(__int64 ***)(v117 + 8) )
              {
                v181 = v117;
              }
              else
              {
                v180 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v344 + 15);
                if ( v180 != sub_920130 )
                {
                  v181 = v180((__int64)v344, 47u, (_BYTE *)v117, (__int64)v179);
                  goto LABEL_253;
                }
                if ( *(_BYTE *)v117 > 0x15u )
                  goto LABEL_309;
                v181 = (unsigned __int8)sub_AC4810(0x2Fu)
                     ? sub_ADAB70(47, v117, v179, 0)
                     : sub_AA93C0(0x2Fu, v117, (__int64)v179);
LABEL_253:
                if ( !v181 )
                {
LABEL_309:
                  LOWORD(v333) = 257;
                  v181 = sub_B51D30(47, v117, (__int64)v179, (__int64)&v330, 0, 0);
                  if ( (unsigned __int8)sub_920620(v181) )
                  {
                    v203 = v347;
                    if ( v346 )
                      sub_B99FD0(v181, 3u, v346);
                    sub_B45150(v181, v203);
                  }
                  (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v345 + 2))(
                    v345,
                    v181,
                    &dest,
                    v341,
                    v342);
                  v204 = v335;
                  v205 = 16LL * (unsigned int)v336;
                  v206 = &v335[v205];
                  if ( v335 != &v335[v205] )
                  {
                    do
                    {
                      v207 = *((_QWORD *)v204 + 1);
                      v208 = *(_DWORD *)v204;
                      v204 += 16;
                      sub_B99FD0(v181, v208, v207);
                    }
                    while ( v206 != v204 );
                  }
                }
              }
LABEL_254:
              v117 = v181;
              goto LABEL_178;
            }
            LOWORD(v327) = 257;
            v169 = (__int64 **)sub_BCB2E0(v343);
            n = *(_QWORD *)(v117 + 8);
            v170 = sub_BCB060(n);
            v171 = sub_BCB060((__int64)v169);
            if ( v170 < v171 )
            {
              if ( v169 == (__int64 **)n )
                goto LABEL_178;
              v197 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v344 + 15);
              if ( v197 == sub_920130 )
              {
                if ( *(_BYTE *)v117 > 0x15u )
                  goto LABEL_398;
                if ( (unsigned __int8)sub_AC4810(0x27u) )
                  v173 = sub_ADAB70(39, v117, v169, 0);
                else
                  v173 = sub_AA93C0(0x27u, v117, (__int64)v169);
              }
              else
              {
                v173 = v197((__int64)v344, 39u, (_BYTE *)v117, (__int64)v169);
              }
              if ( !v173 )
              {
LABEL_398:
                LOWORD(v333) = 257;
                v241 = sub_BD2C40(72, unk_3F10A14);
                v181 = (__int64)v241;
                if ( v241 )
                  sub_B515B0((__int64)v241, v117, (__int64)v169, (__int64)&v330, 0, 0);
                (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v345 + 2))(
                  v345,
                  v181,
                  &dest,
                  v341,
                  v342);
                v242 = v335;
                v243 = &v335[16 * (unsigned int)v336];
                if ( v335 != v243 )
                {
                  do
                  {
                    v244 = *((_QWORD *)v242 + 1);
                    v245 = *(_DWORD *)v242;
                    v242 += 16;
                    sub_B99FD0(v181, v245, v244);
                  }
                  while ( v243 != v242 );
                }
                goto LABEL_254;
              }
LABEL_291:
              v117 = v173;
              goto LABEL_178;
            }
            if ( v169 != (__int64 **)n && v170 != v171 )
              break;
LABEL_178:
            v120 = sub_BCE3C0(*a3, 0);
            v291 = sub_ADB060(v305, v120);
            v330 = (__m128i *)v332;
            v331 = 0x100000000LL;
            if ( *(_BYTE *)v118 != 85 )
            {
              LOBYTE(n) = *(_BYTE *)v118 == 34 || *(_BYTE *)v118 == 40;
              if ( !(_BYTE)n )
              {
                v141 = 0;
                v142 = 0;
                v143 = (__m128i *)v332;
                v283 = (__m128i *)v332;
                goto LABEL_206;
              }
LABEL_180:
              if ( *(char *)(v118 + 7) < 0 )
              {
                v121 = sub_BD2BC0(v118);
                v123 = v121 + v122;
                if ( *(char *)(v118 + 7) < 0 )
                  v123 -= sub_BD2BC0(v118);
                v124 = v123 >> 4;
                if ( (_DWORD)v124 )
                {
                  v125 = 0;
                  v126 = 16LL * (unsigned int)v124;
                  while ( 1 )
                  {
                    v127 = 0;
                    if ( *(char *)(v118 + 7) < 0 )
                      v127 = sub_BD2BC0(v118);
                    v128 = (_DWORD *)(v125 + v127);
                    v129 = *(__int64 **)v128;
                    if ( *(_DWORD *)(*(_QWORD *)v128 + 8LL) == 1 )
                      break;
                    v125 += 16;
                    if ( v126 == v125 )
                      goto LABEL_246;
                  }
                  v130 = v128[2];
                  v282 = v128[3];
                  v131 = *(_DWORD *)(v118 + 4);
                  dest = &v326;
                  v325 = 0;
                  v276 = v131 & 0x7FFFFFF;
                  v326.m128i_i8[0] = 0;
                  v327 = 0;
                  v328 = 0;
                  v329 = 0;
                  v132 = *v129;
                  na = v130;
                  v311 = v313;
                  sub_24A2F70((__int64 *)&v311, (_BYTE *)v129 + 16, (__int64)v129 + v132 + 16);
                  v133 = dest;
                  if ( v311 == v313 )
                  {
                    v222 = v312;
                    if ( v312 )
                    {
                      if ( v312 == 1 )
                        *(_BYTE *)dest = v313[0];
                      else
                        memcpy(dest, v311, v312);
                      v222 = v312;
                      v133 = dest;
                    }
                    v325 = v222;
                    v133[v222] = 0;
                    v133 = v311;
                  }
                  else
                  {
                    if ( dest == &v326 )
                    {
                      dest = v311;
                      v325 = v312;
                      v326.m128i_i64[0] = v313[0];
                    }
                    else
                    {
                      v134 = v326.m128i_i64[0];
                      dest = v311;
                      v325 = v312;
                      v326.m128i_i64[0] = v313[0];
                      if ( v133 )
                      {
                        v311 = v133;
                        v313[0] = v134;
                        goto LABEL_193;
                      }
                    }
                    v311 = v313;
                    v133 = v313;
                  }
LABEL_193:
                  v312 = 0;
                  *v133 = 0;
                  v135 = 32LL * na;
                  v136 = (_QWORD *)(v135 - 32LL * v276 + v118);
                  n = 32LL * v282 - v135;
                  sub_2240A30((unsigned __int64 *)&v311);
                  v137 = (char *)v328;
                  v138 = (__int64)v136 + n;
                  if ( v136 != (_QWORD *)((char *)v136 + n) )
                  {
                    v139 = n >> 5;
                    if ( n >> 5 <= (unsigned __int64)((v329 - (__int64)v328) >> 3) )
                    {
                      do
                      {
                        if ( v137 )
                          *(_QWORD *)v137 = *v136;
                        v136 += 4;
                        v137 += 8;
                      }
                      while ( (_QWORD *)v138 != v136 );
                      v328 = (char *)v328 + 8 * v139;
                      goto LABEL_199;
                    }
                    v223 = (char *)v327;
                    v224 = (_BYTE *)v328 - (_BYTE *)v327;
                    v225 = ((_BYTE *)v328 - (_BYTE *)v327) >> 3;
                    if ( v139 > 0xFFFFFFFFFFFFFFFLL - v225 )
                      sub_4262D8((__int64)"vector::_M_range_insert");
                    if ( v139 < v225 )
                      v139 = ((_BYTE *)v328 - (_BYTE *)v327) >> 3;
                    v91 = __CFADD__(v139, v225);
                    v226 = v139 + v225;
                    n = v226;
                    if ( v91 )
                    {
                      v227 = 0x7FFFFFFFFFFFFFF8LL;
                      goto LABEL_381;
                    }
                    if ( v226 )
                    {
                      if ( v226 > 0xFFFFFFFFFFFFFFFLL )
                        v226 = 0xFFFFFFFFFFFFFFFLL;
                      v227 = 8 * v226;
LABEL_381:
                      v278 = v138;
                      v228 = sub_22077B0(v227);
                      v229 = (char *)v328;
                      v230 = (char *)v228;
                      v223 = (char *)v327;
                      v138 = v278;
                      v224 = v137 - (_BYTE *)v327;
                      n = (_BYTE *)v328 - v137;
                      v284 = v228 + v227;
                    }
                    else
                    {
                      v284 = 0;
                      v229 = (char *)v328;
                      v230 = 0;
                    }
                    if ( v137 != v223 )
                    {
                      v253 = v138;
                      v254 = v229;
                      v256 = v224;
                      v279 = v223;
                      v231 = (char *)memmove(v230, v223, v224);
                      v138 = v253;
                      v229 = v254;
                      v224 = v256;
                      v230 = v231;
                      v223 = v279;
                    }
                    v232 = &v230[v224];
                    v233 = v136;
                    v234 = v232;
                    do
                    {
                      if ( v234 )
                        *(_QWORD *)v234 = *v136;
                      v136 += 4;
                      v234 += 8;
                    }
                    while ( (_QWORD *)v138 != v136 );
                    v235 = &v232[8 * ((unsigned __int64)(v138 - (_QWORD)v233 - 32) >> 5) + 8];
                    if ( v229 != v137 )
                    {
                      v255 = v230;
                      v257 = v223;
                      memcpy(v235, v137, n);
                      v230 = v255;
                      v223 = v257;
                    }
                    v236 = &v235[n];
                    if ( v223 )
                    {
                      n = (__int64)v230;
                      j_j___libc_free_0((unsigned __int64)v223);
                      v230 = (char *)n;
                    }
                    v327 = v230;
                    v328 = v236;
                    v329 = v284;
                  }
LABEL_199:
                  if ( HIDWORD(v331) <= (unsigned int)v331 )
                  {
                    n = (__int64)&v330;
                    v246 = (__m128i *)sub_C8D7D0(
                                        (__int64)&v330,
                                        (__int64)v332,
                                        0,
                                        0x38u,
                                        (unsigned __int64 *)&v311,
                                        v138);
                    v247 = (__m128i *)((char *)v246 + 56 * (unsigned int)v331);
                    if ( v247 )
                    {
                      v247->m128i_i64[0] = (__int64)v247[1].m128i_i64;
                      if ( dest == &v326 )
                      {
                        v247[1] = _mm_load_si128(&v326);
                      }
                      else
                      {
                        v247->m128i_i64[0] = (__int64)dest;
                        v247[1].m128i_i64[0] = v326.m128i_i64[0];
                      }
                      v247->m128i_i64[1] = v325;
                      dest = &v326;
                      v325 = 0;
                      v326.m128i_i8[0] = 0;
                      v247[2].m128i_i64[0] = (__int64)v327;
                      v247[2].m128i_i64[1] = (__int64)v328;
                      v247[3].m128i_i64[0] = v329;
                      v329 = 0;
                      v328 = 0;
                      v327 = 0;
                    }
                    sub_B56820((__int64)&v330, v246);
                    v248 = (int)v311;
                    if ( v330 != (__m128i *)v332 )
                      _libc_free((unsigned __int64)v330);
                    v221 = v327;
                    v330 = v246;
                    LODWORD(v331) = v331 + 1;
                    HIDWORD(v331) = v248;
                  }
                  else
                  {
                    v140 = (__m128i *)((char *)v330 + 56 * (unsigned int)v331);
                    if ( v140 )
                    {
                      v140->m128i_i64[0] = (__int64)v140[1].m128i_i64;
                      if ( dest == &v326 )
                      {
                        v140[1] = _mm_load_si128(&v326);
                      }
                      else
                      {
                        v140->m128i_i64[0] = (__int64)dest;
                        v140[1].m128i_i64[0] = v326.m128i_i64[0];
                      }
                      v140->m128i_i64[1] = v325;
                      dest = &v326;
                      v325 = 0;
                      v326.m128i_i8[0] = 0;
                      v140[2].m128i_i64[0] = (__int64)v327;
                      v140[2].m128i_i64[1] = (__int64)v328;
                      v140[3].m128i_i64[0] = v329;
                      LODWORD(v331) = v331 + 1;
                      goto LABEL_204;
                    }
                    v221 = v327;
                    LODWORD(v331) = v331 + 1;
                  }
                  if ( v221 )
                    j_j___libc_free_0((unsigned __int64)v221);
LABEL_204:
                  sub_2240A30((unsigned __int64 *)&dest);
                }
              }
LABEL_246:
              v142 = (unsigned int)v331;
              v143 = v330;
              v178 = v331;
              goto LABEL_247;
            }
            v182 = *(_QWORD *)(v118 - 32);
            if ( !v182
              || *(_BYTE *)v182
              || *(_QWORD *)(v182 + 24) != *(_QWORD *)(v118 + 80)
              || (*(_BYTE *)(v182 + 33) & 0x20) == 0 )
            {
              goto LABEL_180;
            }
            if ( v269 )
            {
              v183 = *(_QWORD *)(v118 + 40);
              if ( v263 )
              {
                v184 = (v263 - 1) & (((unsigned int)v183 >> 9) ^ ((unsigned int)v183 >> 4));
                v185 = (__int64 *)(v268 + 16LL * v184);
                v186 = *v185;
                if ( v183 == *v185 )
                  goto LABEL_263;
                v209 = 1;
                while ( v186 != -4096 )
                {
                  v252 = v209 + 1;
                  v184 = (v263 - 1) & (v209 + v184);
                  v185 = (__int64 *)(v268 + 16LL * v184);
                  v186 = *v185;
                  if ( v183 == *v185 )
                    goto LABEL_263;
                  v209 = v252;
                }
              }
              v185 = v261;
LABEL_263:
              v187 = v185[1];
              v188 = (_QWORD **)(v187 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v187 & 4) != 0 )
                v188 = (_QWORD **)**v188;
              v189 = sub_AA4FF0((__int64)v188);
              if ( !v189 )
                goto LABEL_448;
              v191 = v331;
              v192 = (unsigned int)*(unsigned __int8 *)(v189 - 24) - 39;
              v178 = v331;
              if ( (unsigned int)v192 <= 0x38 && (v193 = 0x100060000000001LL, _bittest64(&v193, v192)) )
              {
                v194 = v189 - 24;
                if ( (unsigned int)v331 >= HIDWORD(v331) )
                {
                  n = (__int64)&v330;
                  v143 = (__m128i *)sub_C8D7D0((__int64)&v330, (__int64)v332, 0, 0x38u, (unsigned __int64 *)&v311, v190);
                  sub_24A46A0((__int64 *)&dest, "funclet");
                  v249 = (__m128i *)((char *)v143 + 56 * (unsigned int)v331);
                  if ( v249 )
                  {
                    v249->m128i_i64[0] = (__int64)v249[1].m128i_i64;
                    if ( dest == &v326 )
                    {
                      v249[1] = _mm_load_si128(&v326);
                    }
                    else
                    {
                      v249->m128i_i64[0] = (__int64)dest;
                      v249[1].m128i_i64[0] = v326.m128i_i64[0];
                    }
                    n = (__int64)v249;
                    v249->m128i_i64[1] = v325;
                    dest = &v326;
                    v325 = 0;
                    v326.m128i_i8[0] = 0;
                    v249[2].m128i_i64[0] = 0;
                    v249[2].m128i_i64[1] = 0;
                    v249[3].m128i_i64[0] = 0;
                    v250 = (_QWORD *)sub_22077B0(8u);
                    *(_QWORD *)(n + 32) = v250;
                    *(_QWORD *)(n + 48) = v250 + 1;
                    *v250 = v194;
                    *(_QWORD *)(n + 40) = v250 + 1;
                  }
                  if ( dest != &v326 )
                  {
                    n = (__int64)&v330;
                    j_j___libc_free_0((unsigned __int64)dest);
                  }
                  sub_B56820((__int64)&v330, v143);
                  v251 = (int)v311;
                  if ( v330 != (__m128i *)v332 )
                    _libc_free((unsigned __int64)v330);
                  v330 = v143;
                  HIDWORD(v331) = v251;
                  v178 = v331 + 1;
                  LODWORD(v331) = v331 + 1;
                }
                else
                {
                  v143 = v330;
                  dest = &v326;
                  v195 = (__m128i *)((char *)v330 + 56 * (unsigned int)v331);
                  v326.m128i_i64[0] = 0x74656C636E7566LL;
                  v325 = 7;
                  if ( v195 )
                  {
                    v195->m128i_i64[0] = (__int64)v195[1].m128i_i64;
                    if ( dest == &v326 )
                    {
                      v195[1] = _mm_load_si128(&v326);
                    }
                    else
                    {
                      v195->m128i_i64[0] = (__int64)dest;
                      v195[1].m128i_i64[0] = v326.m128i_i64[0];
                    }
                    LODWORD(n) = (_DWORD)v195;
                    v195->m128i_i64[1] = v325;
                    dest = &v326;
                    v325 = 0;
                    v326.m128i_i8[0] = 0;
                    v195[2].m128i_i64[0] = 0;
                    v195[2].m128i_i64[1] = 0;
                    v195[3].m128i_i64[0] = 0;
                    v196 = (_QWORD *)sub_22077B0(8u);
                    v195[2].m128i_i64[0] = (__int64)v196;
                    v195[3].m128i_i64[0] = (__int64)(v196 + 1);
                    *v196 = v194;
                    v195[2].m128i_i64[1] = (__int64)(v196 + 1);
                    if ( dest != &v326 )
                      j_j___libc_free_0((unsigned __int64)dest);
                    v191 = v331;
                    v143 = v330;
                  }
                  v178 = v191 + 1;
                  LODWORD(v331) = v191 + 1;
                }
              }
              else
              {
                v143 = v330;
              }
              v142 = v178;
LABEL_247:
              v141 = 16 * v178;
              LOBYTE(n) = (_DWORD)v141 != 0;
              v283 = (__m128i *)((char *)v143 + 56 * v142);
              goto LABEL_206;
            }
            v143 = (__m128i *)v332;
            v141 = 0;
            v142 = 0;
            LOBYTE(n) = 0;
            v283 = (__m128i *)v332;
LABEL_206:
            v144 = v291;
            v314 = 257;
            v292 = v368;
            v310[0] = v144;
            v145 = sub_BCB2E0(v343);
            v146 = sub_ACD640(v145, v292, 0);
            v310[2] = v117;
            v310[1] = v146;
            v147 = sub_BCB2D0(v343);
            v310[3] = sub_ACD640(v147, v274, 0);
            v277 = i + 1;
            v148 = sub_BCB2D0(v343);
            v310[4] = sub_ACD640(v148, i, 0);
            v149 = sub_B6E160((__int64 *)a3, 0xCBu, 0, 0);
            v150 = 0;
            v151 = v149;
            if ( v149 )
              v150 = *(_QWORD *)(v149 + 24);
            LOWORD(v327) = 257;
            if ( v143 == v283 )
            {
              v156 = 6;
              v155 = 6;
            }
            else
            {
              v152 = v143;
              v153 = 0;
              do
              {
                v154 = v152[2].m128i_i64[1] - v152[2].m128i_i64[0];
                v152 = (__m128i *)((char *)v152 + 56);
                v153 += v154 >> 3;
              }
              while ( v152 != v283 );
              v155 = v153 + 6;
              v156 = v155 & 0x7FFFFFF;
            }
            v289 = v150;
            v293 = v151;
            v157 = sub_BD2CC0(88, (v141 << 32) | v155);
            v158 = (__int64)v157;
            if ( v157 )
            {
              v303 = (__int64)v143;
              v159 = (__int64)v157;
              v281 = v281 & 0xE0000000 | ((_DWORD)n << 28) | v156;
              sub_B44260((__int64)v157, **(_QWORD **)(v289 + 16), 56, v281, 0, 0);
              *(_QWORD *)(v158 + 72) = 0;
              sub_B4A290(v158, v289, v293, v310, 5, (__int64)&dest, v303, v142);
            }
            else
            {
              v159 = 0;
            }
            if ( (_BYTE)v348 )
            {
              v168 = (__int64 *)sub_BD5C60(v159);
              *(_QWORD *)(v158 + 72) = sub_A7A090((__int64 *)(v158 + 72), v168, -1, 72);
            }
            if ( (unsigned __int8)sub_920620(v159) )
            {
              v167 = v347;
              if ( v346 )
                sub_B99FD0(v158, 3u, v346);
              sub_B45150(v158, v167);
            }
            (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v345 + 2))(
              v345,
              v158,
              &v311,
              v341,
              v342);
            v160 = v335;
            v161 = &v335[16 * (unsigned int)v336];
            if ( v335 != v161 )
            {
              do
              {
                v162 = *((_QWORD *)v160 + 1);
                v163 = *(_DWORD *)v160;
                v160 += 16;
                sub_B99FD0(v158, v163, v162);
              }
              while ( v161 != v160 );
            }
            v164 = (unsigned __int64)v330;
            v165 = (unsigned __int64 *)v330 + 7 * (unsigned int)v331;
            if ( v330 != (__m128i *)v165 )
            {
              do
              {
                v166 = *(v165 - 3);
                v165 -= 7;
                if ( v166 )
                  j_j___libc_free_0(v166);
                if ( (unsigned __int64 *)*v165 != v165 + 2 )
                  j_j___libc_free_0(*v165);
              }
              while ( (unsigned __int64 *)v164 != v165 );
              v165 = (unsigned __int64 *)v330;
            }
            if ( v165 != v332 )
              _libc_free((unsigned __int64)v165);
            nullsub_61();
            v352 = &unk_49DA100;
            nullsub_63();
            if ( v335 != (char *)&v337 )
              _libc_free((unsigned __int64)v335);
            v116 += 3;
            if ( v271 == v116 )
              goto LABEL_270;
          }
          v172 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v344 + 15);
          if ( v172 == sub_920130 )
          {
            if ( *(_BYTE *)v117 <= 0x15u )
            {
              if ( (unsigned __int8)sub_AC4810(0x26u) )
                v173 = sub_ADAB70(38, v117, v169, 0);
              else
                v173 = sub_AA93C0(0x26u, v117, (__int64)v169);
              goto LABEL_242;
            }
          }
          else
          {
            v173 = v172((__int64)v344, 38u, (_BYTE *)v117, (__int64)v169);
LABEL_242:
            if ( v173 )
              goto LABEL_291;
          }
          LOWORD(v333) = 257;
          v117 = sub_B51D30(38, v117, (__int64)v169, (__int64)&v330, 0, 0);
          (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v345 + 2))(
            v345,
            v117,
            &dest,
            v341,
            v342);
          v174 = v335;
          v175 = &v335[16 * (unsigned int)v336];
          if ( v335 != v175 )
          {
            do
            {
              v176 = *((_QWORD *)v174 + 1);
              v177 = *(_DWORD *)v174;
              v174 += 16;
              sub_B99FD0(v117, v177, v176);
            }
            while ( v175 != v174 );
          }
          goto LABEL_178;
        }
      }
LABEL_272:
      ++v274;
    }
    v114 = (__int64 **)(v113 + 24 * v274);
    v115 = *v114;
    v271 = v114[1];
    if ( *v114 != v271 )
      goto LABEL_174;
LABEL_270:
    if ( (_DWORD)v274 != 2 )
    {
      v113 = v358;
      goto LABEL_272;
    }
    v198 = (__int64 *)v268;
    if ( v263 )
    {
      do
      {
        if ( *v198 != -8192 && *v198 != -4096 )
        {
          v199 = v198[1];
          if ( v199 )
          {
            if ( (v199 & 4) != 0 )
            {
              v200 = (unsigned __int64 *)(v199 & 0xFFFFFFFFFFFFFFF8LL);
              v201 = (unsigned __int64)v200;
              if ( v200 )
              {
                if ( (unsigned __int64 *)*v200 != v200 + 2 )
                  _libc_free(*v200);
                j_j___libc_free_0(v201);
              }
            }
          }
        }
        v198 += 2;
      }
      while ( v198 != v261 );
    }
    sub_C7D6A0(v268, v259, 8);
LABEL_91:
    if ( v307 )
      j_j___libc_free_0(v307);
LABEL_39:
    sub_24A58D0((__int64)v354);
LABEL_15:
    v272 = (_QWORD *)v272[1];
  }
  while ( v272 != v266 );
LABEL_16:
  v9 = v319;
  while ( v9 )
  {
    v10 = (unsigned __int64)v9;
    v9 = (_QWORD *)*v9;
    j_j___libc_free_0(v10);
  }
  v11 = s;
  v12 = 8 * v318;
LABEL_19:
  memset(v11, 0, v12);
  v320 = 0;
  v319 = 0;
  if ( s != v323 )
    j_j___libc_free_0((unsigned __int64)s);
  if ( v315 != v316 )
    j_j___libc_free_0((unsigned __int64)v315);
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_BYTE *)(a1 + 28) = 1;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
