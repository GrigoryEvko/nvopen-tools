// Function: sub_34D6FB0
// Address: 0x34d6fb0
//
__int64 __fastcall sub_34D6FB0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r13
  unsigned int v6; // r15d
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // r10
  unsigned __int8 v12; // al
  char v13; // al
  int v14; // ecx
  unsigned __int64 v15; // rdx
  int v16; // edi
  int v17; // esi
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // edi
  unsigned int v21; // edx
  unsigned __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  bool v25; // r9
  __int64 v26; // r15
  char *v27; // rax
  unsigned __int8 v28; // bl
  unsigned __int16 v29; // ax
  __int64 v30; // r8
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int8 **v34; // rax
  unsigned __int8 *v35; // r15
  unsigned __int8 *v36; // r13
  int v37; // ebx
  int v38; // r15d
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rbx
  __int64 v42; // rax
  __int64 *v43; // r10
  bool v44; // of
  unsigned __int64 v45; // rbx
  int v46; // eax
  unsigned int v47; // esi
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 *v50; // r10
  __int64 *v51; // r15
  int v52; // edx
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rbx
  __int64 v57; // rcx
  __int64 v58; // rax
  _BYTE *v59; // rdx
  bool v60; // cf
  bool v61; // zf
  __int64 v62; // rdx
  unsigned __int64 v63; // rax
  unsigned __int8 v64; // cl
  char v65; // r8
  int v66; // esi
  _QWORD **v67; // rdx
  __int64 v68; // rax
  _BYTE *v69; // rdx
  bool v70; // cf
  bool v71; // zf
  __int64 v72; // rdx
  unsigned __int64 v73; // rax
  _BYTE *v74; // rbx
  unsigned __int16 v75; // ax
  __int64 *v76; // rax
  __int64 v77; // rbx
  _BYTE *v78; // r13
  unsigned __int16 v79; // ax
  unsigned __int8 v80; // cl
  unsigned __int8 v81; // cl
  unsigned int v82; // r11d
  __int64 *v83; // rdx
  __int64 v84; // rsi
  int v85; // eax
  __int64 v86; // r10
  unsigned int v87; // r11d
  int v88; // eax
  __int64 v89; // rdx
  __int64 (__fastcall *v90)(__int16, __int64); // rbx
  __int16 v91; // dx
  __int64 v92; // rsi
  int v93; // eax
  __int64 *v94; // r9
  __int64 v95; // rbx
  int v96; // edx
  int v97; // eax
  char v98; // al
  __int64 *v99; // r9
  __int64 *v100; // r15
  __int64 v101; // r13
  unsigned int v102; // r12d
  unsigned __int64 v103; // r14
  int v104; // ebx
  signed __int64 v105; // rax
  int v106; // edx
  unsigned __int64 v107; // rbx
  signed __int64 v108; // rax
  _QWORD *v109; // rcx
  __int64 *v110; // rax
  char v111; // r15
  __int64 v112; // r13
  __int16 v113; // ax
  unsigned __int8 v114; // bl
  unsigned __int8 v115; // al
  __int64 v116; // rax
  char v117; // r13
  __int64 v118; // r15
  __int16 v119; // ax
  unsigned __int8 v120; // bl
  unsigned __int8 v121; // al
  unsigned int v122; // eax
  __int64 v123; // rdi
  __int64 *v124; // r10
  __int64 v125; // rsi
  __int64 v126; // rdx
  __int64 (*v127)(); // rax
  __int64 v128; // rax
  unsigned int v129; // ebx
  int v130; // eax
  bool v131; // bl
  char v132; // bl
  __int64 v133; // r10
  __int64 v134; // r9
  unsigned __int64 v135; // rdx
  unsigned __int64 v136; // rax
  unsigned int v137; // eax
  __int64 v138; // rax
  __int64 v139; // rbx
  __int64 v140; // rax
  unsigned __int64 v141; // rbx
  __int64 v142; // rax
  unsigned __int64 v143; // rbx
  __int64 v144; // rax
  unsigned __int64 v145; // rbx
  __int64 v146; // rax
  unsigned __int64 v147; // rbx
  __int64 v148; // rcx
  unsigned __int64 v149; // rdi
  _QWORD *v150; // rdx
  __int64 v151; // rcx
  _QWORD *v152; // rax
  _QWORD *v153; // rax
  __int64 v154; // rcx
  __int64 v155; // rsi
  _QWORD *v156; // rdx
  _QWORD *v157; // rdx
  __int64 v158; // rcx
  _QWORD *v159; // rax
  unsigned int v160; // eax
  __int64 v161; // rbx
  char v162; // al
  const void **v163; // r8
  unsigned int v164; // edx
  __int64 *v165; // rsi
  unsigned __int64 v166; // rcx
  int v167; // eax
  unsigned __int64 v168; // rcx
  int v169; // ecx
  unsigned int v170; // ecx
  __int64 v171; // rax
  __int64 v172; // rcx
  unsigned __int64 v173; // rdx
  unsigned int v174; // r13d
  int v175; // r13d
  int v176; // eax
  __int64 v177; // r10
  int v178; // r15d
  __int64 v179; // rax
  unsigned __int64 v180; // rdx
  __int64 v181; // rcx
  unsigned __int64 v182; // rax
  unsigned __int64 v183; // kr30_8
  unsigned __int64 v184; // r13
  unsigned int v185; // edi
  __int64 v186; // rdx
  __int64 v187; // rdx
  __int64 v188; // rbx
  __int64 v189; // r15
  bool v190; // al
  char *v191; // r8
  __int64 v192; // r15
  char *v193; // rax
  unsigned __int8 v194; // bl
  unsigned __int16 v195; // ax
  __int64 v196; // r8
  __int64 v197; // rax
  __int64 v198; // rcx
  int v199; // eax
  bool v200; // al
  __int64 v201; // rcx
  __int64 v202; // r8
  __int16 v203; // ax
  __int64 v204; // rdx
  char v205; // al
  int v206; // eax
  unsigned __int64 v207; // rax
  int v208; // eax
  char v209; // al
  __int64 v210; // r14
  __int64 v211; // rcx
  __int16 v212; // bx
  __int64 v213; // rdx
  __int64 v214; // r13
  __int64 v215; // rax
  char v216; // al
  char *v217; // rsi
  _QWORD *v218; // rax
  __int64 *v219; // r9
  __int64 *v220; // rax
  __int64 v221; // rax
  int v222; // edx
  __int64 v223; // r13
  __int64 *v224; // rbx
  unsigned int v225; // r14d
  size_t v226; // rax
  __int64 v227; // rax
  __int64 v228; // rcx
  __int64 v229; // rax
  __int64 v230; // rax
  __int64 v231; // r10
  unsigned int v232; // eax
  unsigned int v233; // eax
  __int64 v234; // rbx
  unsigned __int64 v235; // r13
  unsigned __int64 *v236; // r10
  unsigned int *v237; // r11
  unsigned int v238; // r14d
  __int64 v239; // r12
  __int64 *v240; // rbx
  __int64 v241; // r10
  unsigned int v242; // eax
  __int64 v243; // rax
  __int64 *v244; // rdi
  __int64 v245; // rax
  signed __int64 v246; // rax
  bool v247; // cc
  unsigned __int64 v248; // rax
  __int128 v249; // [rsp-18h] [rbp-258h]
  __int128 v250; // [rsp-18h] [rbp-258h]
  __int128 v251; // [rsp-18h] [rbp-258h]
  int v252; // [rsp-10h] [rbp-250h]
  unsigned __int16 v253; // [rsp+2h] [rbp-23Eh]
  char v254; // [rsp+4h] [rbp-23Ch]
  __int64 v255; // [rsp+8h] [rbp-238h]
  unsigned int v256; // [rsp+20h] [rbp-220h]
  _QWORD *v257; // [rsp+20h] [rbp-220h]
  unsigned int v258; // [rsp+28h] [rbp-218h]
  unsigned int v259; // [rsp+28h] [rbp-218h]
  __int64 v260; // [rsp+30h] [rbp-210h]
  const void **v261; // [rsp+30h] [rbp-210h]
  __int64 *v262; // [rsp+30h] [rbp-210h]
  __int64 v263; // [rsp+30h] [rbp-210h]
  int v264; // [rsp+38h] [rbp-208h]
  unsigned int v265; // [rsp+38h] [rbp-208h]
  unsigned int v266; // [rsp+38h] [rbp-208h]
  unsigned int v267; // [rsp+38h] [rbp-208h]
  __int64 v268; // [rsp+38h] [rbp-208h]
  __int64 v269; // [rsp+40h] [rbp-200h]
  unsigned int v270; // [rsp+40h] [rbp-200h]
  unsigned int v271; // [rsp+40h] [rbp-200h]
  __int64 v272; // [rsp+40h] [rbp-200h]
  __int64 v273; // [rsp+40h] [rbp-200h]
  __int64 v274; // [rsp+40h] [rbp-200h]
  __int64 v275; // [rsp+40h] [rbp-200h]
  __int64 v276; // [rsp+40h] [rbp-200h]
  __int64 v277; // [rsp+40h] [rbp-200h]
  __int64 v278; // [rsp+40h] [rbp-200h]
  int v279; // [rsp+40h] [rbp-200h]
  __int64 v280; // [rsp+40h] [rbp-200h]
  __int64 v281; // [rsp+40h] [rbp-200h]
  __int64 v282; // [rsp+40h] [rbp-200h]
  __int64 v283; // [rsp+40h] [rbp-200h]
  __int64 v284; // [rsp+48h] [rbp-1F8h]
  int v285; // [rsp+48h] [rbp-1F8h]
  bool v286; // [rsp+48h] [rbp-1F8h]
  __int64 *v287; // [rsp+48h] [rbp-1F8h]
  unsigned int v288; // [rsp+48h] [rbp-1F8h]
  _QWORD **v289; // [rsp+48h] [rbp-1F8h]
  __int64 *v290; // [rsp+48h] [rbp-1F8h]
  unsigned int *v291; // [rsp+48h] [rbp-1F8h]
  unsigned __int8 *v292; // [rsp+50h] [rbp-1F0h]
  __int64 v293; // [rsp+50h] [rbp-1F0h]
  __int64 v294; // [rsp+50h] [rbp-1F0h]
  __int64 *v295; // [rsp+50h] [rbp-1F0h]
  __int64 *v296; // [rsp+50h] [rbp-1F0h]
  __int64 v297; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v298; // [rsp+50h] [rbp-1F0h]
  __int64 v299; // [rsp+50h] [rbp-1F0h]
  __int64 v300; // [rsp+58h] [rbp-1E8h]
  int v301; // [rsp+60h] [rbp-1E0h]
  unsigned __int8 *v302; // [rsp+60h] [rbp-1E0h]
  __int64 v303; // [rsp+60h] [rbp-1E0h]
  __int64 v304; // [rsp+60h] [rbp-1E0h]
  _QWORD **v305; // [rsp+60h] [rbp-1E0h]
  __int64 v306; // [rsp+68h] [rbp-1D8h]
  __int64 v307; // [rsp+68h] [rbp-1D8h]
  __int64 v308; // [rsp+68h] [rbp-1D8h]
  __int64 v309; // [rsp+68h] [rbp-1D8h]
  __int64 v310; // [rsp+68h] [rbp-1D8h]
  __int64 v311; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 v312; // [rsp+68h] [rbp-1D8h]
  __int64 *v313; // [rsp+68h] [rbp-1D8h]
  __int64 *v314; // [rsp+68h] [rbp-1D8h]
  char v315; // [rsp+68h] [rbp-1D8h]
  char v316; // [rsp+68h] [rbp-1D8h]
  __int64 v317; // [rsp+68h] [rbp-1D8h]
  __int64 v318; // [rsp+68h] [rbp-1D8h]
  __int64 v319; // [rsp+68h] [rbp-1D8h]
  __int64 v320; // [rsp+68h] [rbp-1D8h]
  __int64 v321; // [rsp+68h] [rbp-1D8h]
  __int64 v322; // [rsp+68h] [rbp-1D8h]
  __int64 v323; // [rsp+68h] [rbp-1D8h]
  __int64 v324; // [rsp+68h] [rbp-1D8h]
  __int64 v325; // [rsp+68h] [rbp-1D8h]
  char *v326; // [rsp+68h] [rbp-1D8h]
  __int64 v327; // [rsp+68h] [rbp-1D8h]
  __int64 v328; // [rsp+68h] [rbp-1D8h]
  signed __int64 v329; // [rsp+68h] [rbp-1D8h]
  __int64 v330; // [rsp+68h] [rbp-1D8h]
  __int64 v331; // [rsp+68h] [rbp-1D8h]
  __int64 v332; // [rsp+78h] [rbp-1C8h] BYREF
  __int64 v333[2]; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v334; // [rsp+90h] [rbp-1B0h]
  unsigned __int64 v335; // [rsp+98h] [rbp-1A8h]
  _DWORD v336[2]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v337; // [rsp+A8h] [rbp-198h]
  unsigned __int64 v338; // [rsp+B0h] [rbp-190h] BYREF
  unsigned int v339; // [rsp+B8h] [rbp-188h] BYREF
  unsigned __int64 v340; // [rsp+C0h] [rbp-180h]
  unsigned int v341; // [rsp+C8h] [rbp-178h]
  __int64 v342; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v343; // [rsp+D8h] [rbp-168h]
  char *v344; // [rsp+E8h] [rbp-158h]
  char v345; // [rsp+F8h] [rbp-148h] BYREF
  char *v346; // [rsp+118h] [rbp-128h]
  char v347; // [rsp+128h] [rbp-118h] BYREF
  __int64 v348; // [rsp+170h] [rbp-D0h] BYREF
  __int64 v349; // [rsp+178h] [rbp-C8h]
  unsigned __int64 v350; // [rsp+180h] [rbp-C0h] BYREF
  _BYTE *v351; // [rsp+188h] [rbp-B8h]
  _BYTE v352[32]; // [rsp+198h] [rbp-A8h] BYREF
  _BYTE *v353; // [rsp+1B8h] [rbp-88h]
  _BYTE v354[120]; // [rsp+1C8h] [rbp-78h] BYREF

  v4 = a2;
  v6 = *(_DWORD *)(a2 + 16);
  if ( v6 <= 0xD3 )
  {
    if ( v6 > 0x94 )
    {
      switch ( v6 )
      {
        case 0x95u:
        case 0x96u:
        case 0x9Bu:
        case 0xA9u:
        case 0xCCu:
        case 0xCDu:
        case 0xCEu:
        case 0xD0u:
        case 0xD2u:
        case 0xD3u:
          return 0;
        default:
          goto LABEL_5;
      }
    }
    switch ( v6 )
    {
      case 5u:
      case 6u:
      case 7u:
      case 8u:
      case 0xBu:
      case 0x1Bu:
      case 0x1Cu:
      case 0x27u:
      case 0x28u:
      case 0x2Bu:
      case 0x2Eu:
      case 0x2Fu:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x44u:
      case 0x45u:
      case 0x46u:
      case 0x47u:
        return 0;
      default:
        goto LABEL_5;
    }
  }
  if ( v6 <= 0x178 )
  {
    if ( v6 > 0x143 )
    {
      v7 = 0x10000020401001LL;
      if ( _bittest64(&v7, v6 - 324) )
        return 0;
    }
    else if ( v6 == 282 || v6 - 291 <= 1 )
    {
      return 0;
    }
  }
LABEL_5:
  if ( sub_B60C40(v6) )
    return 1;
  if ( sub_B5A760(*(_DWORD *)(a2 + 16)) )
  {
    v338 = sub_B5A790(*(_DWORD *)(a2 + 16), a2, v9, v10);
    v15 = HIDWORD(v338);
    if ( BYTE4(v338) )
    {
      v16 = *(_DWORD *)(a2 + 16);
      v17 = v338;
      if ( v16 == 438 )
      {
        v192 = *(_QWORD *)v4;
        if ( !*(_QWORD *)v4 )
          goto LABEL_255;
        v193 = *(char **)(v192 - 32);
        if ( !v193 )
          goto LABEL_394;
        v194 = *v193;
        if ( *v193 || *((_QWORD *)v193 + 3) != *(_QWORD *)(v192 + 80) )
          goto LABEL_394;
        v17 = v338;
        if ( sub_B5A760(*((_DWORD *)v193 + 9)) )
        {
          v195 = sub_B5A5E0(v192);
          v17 = v338;
          if ( HIBYTE(v195) )
            v194 = v195;
        }
        else
        {
LABEL_255:
          v194 = 0;
        }
        v196 = 0;
        if ( *(_DWORD *)(v4 + 32) > 1u )
        {
          v197 = **(_QWORD **)(v4 + 24);
          if ( *(_BYTE *)(v197 + 8) == 14 )
            v196 = *(_DWORD *)(v197 + 8) >> 8;
        }
        v198 = v194;
        BYTE1(v198) = 1;
        return sub_34D2F80(a1, v17, *(_QWORD *)(v4 + 8), v198, v196, a3);
      }
      if ( v16 == 481 )
      {
        v26 = *(_QWORD *)v4;
        if ( !*(_QWORD *)v4 )
          goto LABEL_252;
        v27 = *(char **)(v26 - 32);
        if ( !v27 )
          goto LABEL_394;
        v28 = *v27;
        if ( *v27 || *((_QWORD *)v27 + 3) != *(_QWORD *)(v26 + 80) )
          goto LABEL_394;
        v17 = v338;
        if ( sub_B5A760(*((_DWORD *)v27 + 9)) )
        {
          v29 = sub_B5A5E0(v26);
          v17 = v338;
          if ( HIBYTE(v29) )
            v28 = v29;
        }
        else
        {
LABEL_252:
          v28 = 0;
        }
        v30 = 0;
        v31 = *(__int64 **)(v4 + 24);
        if ( *(_DWORD *)(v4 + 32) > 1u )
        {
          v32 = v31[1];
          if ( *(_BYTE *)(v32 + 8) == 14 )
            v30 = *(_DWORD *)(v32 + 8) >> 8;
        }
        v33 = v28;
        BYTE1(v33) = 1;
        return sub_34D2F80(a1, v17, *v31, v33, v30, a3);
      }
      if ( (unsigned __int8)sub_B5B050(v16) )
        return sub_34D2250(a1, v338, *(_QWORD *)(v4 + 8), a3, 0, 0, 0, 0, 0);
      v20 = *(_DWORD *)(v4 + 16);
      if ( v20 == 424 )
        return sub_34D2250(a1, v338, *(_QWORD *)(v4 + 8), a3, 0, 0, 0, 0, 0);
      a2 = (unsigned int)v338;
      if ( sub_B5B010(v20, (unsigned int)v338, v18, v19) )
        return sub_34D3270(a1, v338, *(_QWORD *)(v4 + 8), **(_QWORD **)(v4 + 24), 0, a3, 0);
      if ( sub_B5B040(*(_DWORD *)(v4 + 16)) && *(_QWORD *)v4 )
      {
        sub_B5B080(*(_QWORD *)v4);
        return sub_34D1290(a1, v338, **(__int64 ***)(v4 + 24), *(_QWORD *)(v4 + 8), v199, a3, 0, 0, 0);
      }
    }
    v342 = sub_B5A9F0(*(_DWORD *)(v4 + 16), a2, v15, v14);
    if ( BYTE4(v342) )
    {
      v188 = *(unsigned int *)(v4 + 32);
      v326 = *(char **)(v4 + 24);
      v189 = v188 - 2;
      v190 = sub_B5B000(*(_DWORD *)(v4 + 16));
      v191 = v326;
      if ( (_DWORD)v342 != 394 && (_DWORD)v342 != 389 && v190 )
      {
        v189 = v188 - 3;
        v191 = v326 + 8;
      }
      *((_QWORD *)&v251 + 1) = 1;
      *(_QWORD *)&v251 = 0;
      sub_DF8CB0((__int64)&v348, v342, *(_QWORD *)(v4 + 8), v191, v189, *(_DWORD *)(v4 + 120), 0, v251);
      result = sub_34D6FB0(a1, &v348, a3);
      v24 = (unsigned __int64)v353;
      if ( v353 == v354 )
        goto LABEL_36;
LABEL_35:
      v308 = result;
      _libc_free(v24);
      result = v308;
LABEL_36:
      if ( v351 != v352 )
      {
        v309 = result;
        _libc_free((unsigned __int64)v351);
        return v309;
      }
      return result;
    }
  }
  if ( !*(_DWORD *)(v4 + 80) )
    return sub_34D9E20(a1, v4, a3);
  v11 = *(_QWORD *)(v4 + 8);
  v12 = *(_BYTE *)(v11 + 8);
  v332 = v11;
  if ( v12 == 15 )
  {
    v306 = v11;
    v13 = sub_E456C0(v11);
    v11 = v306;
    if ( !v13 )
      goto LABEL_18;
    v81 = *(_BYTE *)(v306 + 8);
    if ( v81 == 15 )
    {
      v187 = **(_QWORD **)(v306 + 16);
      v12 = *(_BYTE *)(v187 + 8);
    }
    else
    {
      v12 = *(_BYTE *)(v306 + 8);
      v187 = v306;
    }
  }
  else
  {
    if ( (unsigned int)v12 - 17 > 1 )
    {
LABEL_18:
      v307 = *(_QWORD *)v4;
      v301 = *(_DWORD *)(v4 + 120);
      if ( v6 > 0xF9 )
      {
        switch ( v6 )
        {
          case 0x11Du:
            v286 = 0;
            v82 = 1;
            goto LABEL_186;
          case 0x146u:
          case 0x147u:
LABEL_94:
            v286 = 0;
            v81 = *(_BYTE *)(v11 + 8);
            v82 = 1;
            goto LABEL_95;
          case 0x159u:
            v81 = *(_BYTE *)(v11 + 8);
            goto LABEL_181;
          case 0x17Du:
            v81 = *(_BYTE *)(v11 + 8);
            goto LABEL_176;
          case 0x17Eu:
            goto LABEL_171;
          case 0x183u:
          case 0x184u:
          case 0x186u:
          case 0x187u:
          case 0x188u:
          case 0x189u:
          case 0x18Bu:
          case 0x18Cu:
          case 0x18Du:
          case 0x18Eu:
          case 0x18Fu:
          case 0x190u:
          case 0x191u:
            goto LABEL_59;
          case 0x185u:
          case 0x18Au:
            goto LABEL_124;
          case 0x192u:
            return sub_34D5BE0(a1, 1, *(_QWORD *)(**(_QWORD **)(v4 + 72) + 8LL), 0, 0, a3, 0, v11);
          case 0x193u:
            goto LABEL_167;
          default:
            goto LABEL_32;
        }
      }
      if ( v6 > 0xE0 )
      {
        switch ( v6 )
        {
          case 0xE1u:
            goto LABEL_91;
          case 0xE2u:
            goto LABEL_88;
          case 0xE3u:
            goto LABEL_82;
          case 0xE5u:
            goto LABEL_76;
          case 0xEEu:
            return 4;
          case 0xF9u:
            goto LABEL_94;
          default:
            goto LABEL_32;
        }
      }
      if ( v6 > 0xB9 )
        goto LABEL_32;
      if ( v6 > 0x90 )
      {
        switch ( v6 )
        {
          case 0x91u:
            goto LABEL_135;
          case 0xA2u:
          case 0xB9u:
            return sub_34D9E20(a1, v4, a3);
          case 0xA7u:
            goto LABEL_130;
          case 0xA8u:
            goto LABEL_125;
          case 0xB4u:
          case 0xB5u:
            goto LABEL_61;
          default:
            goto LABEL_32;
        }
      }
      if ( v6 != 65 )
      {
        if ( v6 != 67 )
        {
LABEL_32:
          v21 = 1;
          v22 = 0;
LABEL_33:
          *((_QWORD *)&v249 + 1) = v21 | v300 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v249 = v22;
          sub_DF8CB0((__int64)&v348, v6, v11, *(char **)(v4 + 24), *(unsigned int *)(v4 + 32), v301, v307, v249);
          v23 = a3;
LABEL_34:
          result = sub_34D9E20(a1, &v348, v23);
          v24 = (unsigned __int64)v353;
          if ( v353 == v354 )
            goto LABEL_36;
          goto LABEL_35;
        }
LABEL_41:
        v284 = v11;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 288LL))(
                *(_QWORD *)(a1 + 24),
                v11) )
          goto LABEL_47;
        return 1;
      }
LABEL_46:
      v284 = v11;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 296LL))(
              *(_QWORD *)(a1 + 24),
              v11) )
      {
LABEL_47:
        v11 = v284;
        goto LABEL_32;
      }
      return 1;
    }
    v81 = v12;
    v187 = v11;
  }
  v82 = *(_DWORD *)(v187 + 32);
  v286 = v12 == 18;
  v307 = *(_QWORD *)v4;
  v301 = *(_DWORD *)(v4 + 120);
  if ( v6 <= 0xF9 )
  {
    if ( v6 > 0xE0 )
    {
      switch ( v6 )
      {
        case 0xE1u:
LABEL_91:
          v76 = *(__int64 **)(v4 + 72);
          v77 = *v76;
          v78 = (_BYTE *)v76[2];
          v79 = sub_A74840((_QWORD *)(v307 + 72), 1);
          v80 = 0;
          if ( HIBYTE(v79) )
            v80 = v79;
          return sub_34D46A0(a1, 33, *(_QWORD ***)(v77 + 8), v80, *v78 > 0x15u, 1, a3, 0);
        case 0xE2u:
LABEL_88:
          v303 = v11;
          v74 = *(_BYTE **)(*(_QWORD *)(v4 + 72) + 8LL);
          v75 = sub_A74840((_QWORD *)(v307 + 72), 0);
          v11 = v303;
          v64 = 0;
          if ( HIBYTE(v75) )
            v64 = v75;
          v252 = a3;
          v65 = *v74 > 0x15u;
          goto LABEL_87;
        case 0xE3u:
LABEL_82:
          v68 = *(_QWORD *)(v4 + 72);
          v69 = *(_BYTE **)(v68 + 16);
          v70 = *v69 < 0x15u;
          v71 = *v69 == 21;
          v72 = *(_QWORD *)(v68 + 8);
          v73 = *(_QWORD *)(v72 + 24);
          if ( *(_DWORD *)(v72 + 32) > 0x40u )
            v73 = *(_QWORD *)v73;
          v64 = 0;
          if ( v73 )
          {
            _BitScanReverse64(&v73, v73);
            v64 = 63 - (v73 ^ 0x3F);
          }
          v65 = !v70 && !v71;
          v252 = a3;
LABEL_87:
          v67 = (_QWORD **)v11;
          v66 = 32;
          return sub_34D46A0(a1, v66, v67, v64, v65, 1, v252, 0);
        case 0xE5u:
LABEL_76:
          v58 = *(_QWORD *)(v4 + 72);
          v59 = *(_BYTE **)(v58 + 24);
          v60 = *v59 < 0x15u;
          v61 = *v59 == 21;
          v62 = *(_QWORD *)(v58 + 16);
          v63 = *(_QWORD *)(v62 + 24);
          if ( *(_DWORD *)(v62 + 32) > 0x40u )
            v63 = *(_QWORD *)v63;
          v64 = 0;
          if ( v63 )
          {
            _BitScanReverse64(&v63, v63);
            v64 = 63 - (v63 ^ 0x3F);
          }
          v65 = !v60 && !v61;
          v66 = 33;
          v67 = **(_QWORD ****)(v4 + 24);
          v252 = a3;
          return sub_34D46A0(a1, v66, v67, v64, v65, 1, v252, 0);
        case 0xEEu:
          return 4;
        case 0xF9u:
          goto LABEL_95;
        default:
          goto LABEL_221;
      }
    }
    if ( v6 <= 0xB9 )
    {
      if ( v6 > 0x90 )
      {
        switch ( v6 )
        {
          case 0x91u:
LABEL_135:
            v295 = (__int64 *)v11;
            v122 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), **(__int64 ***)(v4 + 24), 1);
            v123 = *(_QWORD *)(a1 + 24);
            v124 = v295;
            v336[0] = v122;
            v125 = v122;
            v337 = v126;
            v127 = *(__int64 (**)())(*(_QWORD *)v123 + 136LL);
            if ( v127 != sub_2FE2E80 )
            {
              v209 = ((__int64 (__fastcall *)(__int64, __int64))v127)(v123, v125);
              v124 = v295;
              if ( !v209 )
              {
                v210 = *v295;
                v329 = 1;
                v211 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v295, 0);
                v212 = v211;
                v214 = v213;
                while ( 1 )
                {
                  LOWORD(v211) = v212;
                  sub_2FE6CC0((__int64)&v348, *(_QWORD *)(a1 + 24), v210, v211, v214);
                  if ( (_BYTE)v348 == 10 )
                    break;
                  if ( !(_BYTE)v348 )
                    return v329;
                  if ( (v348 & 0xFB) == 2 )
                  {
                    v215 = 2 * v329;
                    if ( !is_mul_ok(2u, v329) )
                    {
                      v215 = 0x7FFFFFFFFFFFFFFFLL;
                      if ( v329 <= 0 )
                        v215 = 0x8000000000000000LL;
                    }
                    v329 = v215;
                  }
                  if ( v212 == (_WORD)v349 && (v214 == v350 || (_WORD)v349) )
                    return v329;
                  v211 = v349;
                  v214 = v350;
                  v212 = v349;
                }
                return 0;
              }
            }
            v128 = *(_QWORD *)(*(_QWORD *)(v4 + 72) + 8LL);
            v129 = *(_DWORD *)(v128 + 32);
            if ( v129 <= 0x40 )
            {
              v131 = *(_QWORD *)(v128 + 24) == 0;
            }
            else
            {
              v296 = v124;
              v130 = sub_C444A0(v128 + 24);
              v124 = v296;
              v131 = v129 == v130;
            }
            v132 = !v131;
            v274 = (__int64)v124;
            LODWORD(v349) = 64;
            v348 = 0;
            LODWORD(v343) = 64;
            v342 = 1;
            sub_AADC30((__int64)&v338, (__int64)&v342, &v348);
            v133 = v274;
            if ( (unsigned int)v343 > 0x40 && v342 )
            {
              j_j___libc_free_0_0(v342);
              v133 = v274;
            }
            if ( (unsigned int)v349 > 0x40 && v348 )
            {
              v275 = v133;
              j_j___libc_free_0_0(v348);
              v133 = v275;
            }
            if ( *(_BYTE *)(**(_QWORD **)(v4 + 24) + 8LL) == 18 )
            {
              if ( v307 )
              {
                v281 = v133;
                v229 = sub_B491C0(v307);
                v133 = v281;
                if ( v229 )
                {
                  v230 = sub_B491C0(v307);
                  sub_988CD0((__int64)&v348, v230, 0x40u);
                  v231 = v281;
                  if ( v339 > 0x40 && v338 )
                  {
                    j_j___libc_free_0_0(v338);
                    v231 = v281;
                  }
                  v338 = v348;
                  v232 = v349;
                  LODWORD(v349) = 0;
                  v339 = v232;
                  if ( v341 > 0x40 && v340 )
                  {
                    v282 = v231;
                    j_j___libc_free_0_0(v340);
                    v231 = v282;
                  }
                  v283 = v231;
                  v340 = v350;
                  v233 = (unsigned int)v351;
                  LODWORD(v351) = 0;
                  v341 = v233;
                  sub_969240((__int64 *)&v350);
                  sub_969240(&v348);
                  v133 = v283;
                }
              }
            }
            v134 = *(_QWORD *)(a1 + 24);
            if ( LOWORD(v336[0]) )
            {
              LOBYTE(v135) = (unsigned __int16)(LOWORD(v336[0]) - 176) <= 0x34u;
              LODWORD(v136) = word_4456340[LOWORD(v336[0]) - 1];
            }
            else
            {
              v268 = *(_QWORD *)(a1 + 24);
              v278 = v133;
              v136 = sub_3007240((__int64)v336);
              v134 = v268;
              v133 = v278;
              v335 = v136;
              v135 = HIDWORD(v136);
            }
            LODWORD(v335) = v136;
            BYTE4(v335) = v135;
            v289 = (_QWORD **)v133;
            v333[1] = v335;
            v137 = sub_2FE69A0(v134, v133, v335, v132, (__int64)&v338);
            v290 = (__int64 *)sub_BCD140(*v289, v137);
            v138 = *(_QWORD *)(**(_QWORD **)(v4 + 72) + 8LL);
            v61 = *(_BYTE *)(v138 + 8) == 18;
            LODWORD(v138) = *(_DWORD *)(v138 + 32);
            BYTE4(v334) = v61;
            LODWORD(v334) = v138;
            v333[0] = sub_BCE1B0(v290, v334);
            *((_QWORD *)&v250 + 1) = 1;
            *(_QWORD *)&v250 = 0;
            sub_DF8CB0((__int64)&v342, 345, v333[0], 0, 0, v301, 0, v250);
            v139 = sub_34D6FB0(a1, &v342, a3);
            v140 = sub_34D2250(a1, 0xFu, v333[0], a3, 0, 0, 0, 0, 0);
            v44 = __OFADD__(v140, v139);
            v141 = v140 + v139;
            if ( v44 )
            {
              v141 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v140 <= 0 )
                v141 = 0x8000000000000000LL;
            }
            v142 = sub_34D3270(a1, 0x28u, v333[0], *(_QWORD *)(**(_QWORD **)(v4 + 72) + 8LL), 0, a3, 0);
            v44 = __OFADD__(v142, v141);
            v143 = v142 + v141;
            if ( v44 )
            {
              v143 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v142 <= 0 )
                v143 = 0x8000000000000000LL;
            }
            v144 = sub_34D2250(a1, 0x1Cu, v333[0], a3, 0, 0, 0, 0, 0);
            v44 = __OFADD__(v144, v143);
            v145 = v144 + v143;
            if ( v44 )
            {
              v145 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v144 <= 0 )
                v145 = 0x8000000000000000LL;
            }
            sub_DF8CB0((__int64)&v348, 399, (__int64)v290, (char *)v333, 1, v301, v307, 1u);
            v146 = sub_34D9E20(a1, &v348, a3);
            v44 = __OFADD__(v146, v145);
            v147 = v146 + v145;
            if ( v44 )
            {
              v147 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v146 <= 0 )
                v147 = 0x8000000000000000LL;
            }
            v148 = sub_34D2250(a1, 0xFu, (__int64)v290, a3, 0, 0, 0, 0, 0);
            result = v148 + v147;
            if ( __OFADD__(v148, v147) )
            {
              result = 0x7FFFFFFFFFFFFFFFLL;
              if ( v148 <= 0 )
                result = 0x8000000000000000LL;
            }
            if ( v353 != v354 )
            {
              v317 = result;
              _libc_free((unsigned __int64)v353);
              result = v317;
            }
            if ( v351 != v352 )
            {
              v318 = result;
              _libc_free((unsigned __int64)v351);
              result = v318;
            }
            if ( v346 != &v347 )
            {
              v319 = result;
              _libc_free((unsigned __int64)v346);
              result = v319;
            }
            if ( v344 != &v345 )
            {
              v320 = result;
              _libc_free((unsigned __int64)v344);
              result = v320;
            }
            if ( v341 > 0x40 && v340 )
            {
              v321 = result;
              j_j___libc_free_0_0(v340);
              result = v321;
            }
            if ( v339 > 0x40 )
            {
              v149 = v338;
              if ( v338 )
                goto LABEL_166;
            }
            return result;
          case 0xA2u:
          case 0xB9u:
            return sub_34D9E20(a1, v4, a3);
          case 0xA7u:
LABEL_130:
            v116 = *(_QWORD *)(v4 + 72);
            v117 = 1;
            if ( **(_BYTE **)(v116 + 16) <= 0x15u )
              v117 = **(_BYTE **)(v116 + 24) > 0x15u;
            v118 = *(_QWORD *)(v11 + 24);
            v305 = (_QWORD **)v11;
            v119 = sub_A74840((_QWORD *)(v307 + 72), 0);
            v120 = v119;
            v316 = HIBYTE(v119);
            v121 = sub_AE5020(*(_QWORD *)(a1 + 8), v118);
            if ( v316 )
              v121 = v120;
            return sub_34D46A0(a1, 32, v305, v121, v117, 1, a3, 0);
          case 0xA8u:
LABEL_125:
            v110 = *(__int64 **)(v4 + 72);
            v111 = 1;
            v112 = *v110;
            if ( *(_BYTE *)v110[3] <= 0x15u )
              v111 = *(_BYTE *)v110[4] > 0x15u;
            v304 = *(_QWORD *)(*(_QWORD *)(v112 + 8) + 24LL);
            v113 = sub_A74840((_QWORD *)(v307 + 72), 1);
            v114 = v113;
            v315 = HIBYTE(v113);
            v115 = sub_AE5020(*(_QWORD *)(a1 + 8), v304);
            if ( v315 )
              v115 = v114;
            return sub_34D46A0(a1, 33, *(_QWORD ***)(v112 + 8), v115, v111, 1, a3, 0);
          case 0xB4u:
          case 0xB5u:
LABEL_61:
            v34 = *(unsigned __int8 ***)(v4 + 72);
            v310 = v11;
            v35 = v34[1];
            v36 = v34[2];
            v302 = *v34;
            v292 = v35;
            v37 = sub_DFB770(*v34);
            v285 = sub_DFB770(v35);
            v38 = sub_DFB770(v36);
            v269 = v310;
            v311 = sub_34D2250(a1, 0x1Du, v310, a3, 0, 0, 0, 0, 0);
            v39 = sub_34D2250(a1, 0xFu, v269, a3, 0, 0, 0, 0, 0);
            if ( __OFADD__(v39, v311) )
            {
              v247 = v39 <= 0;
              v248 = 0x8000000000000000LL;
              if ( !v247 )
                v248 = 0x7FFFFFFFFFFFFFFFLL;
              v312 = v248;
            }
            else
            {
              v312 = v39 + v311;
            }
            v40 = sub_34D2250(a1, 0x19u, v269, a3, v37, v38, 0, 0, 0);
            v41 = v40 + v312;
            if ( __OFADD__(v40, v312) )
            {
              v41 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v40 <= 0 )
                v41 = 0x8000000000000000LL;
            }
            v42 = sub_34D2250(a1, 0x1Au, v269, a3, v285, v38, 0, 0, 0);
            v43 = (__int64 *)v269;
            v44 = __OFADD__(v42, v41);
            v45 = v42 + v41;
            if ( v44 )
            {
              v45 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v42 <= 0 )
                v45 = 0x8000000000000000LL;
            }
            if ( (unsigned int)(v38 - 2) > 1 )
            {
              v46 = sub_BCB060(v269);
              v47 = 22;
              if ( v46 )
                v47 = ((v46 - 1) & v46) == 0 ? 28 : 22;
              v48 = sub_34D2250(a1, v47, v269, a3, v38, 2, 0, 0, 0);
              v43 = (__int64 *)v269;
              v44 = __OFADD__(v48, v45);
              v45 += v48;
              if ( v44 )
              {
                v45 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v48 <= 0 )
                  v45 = 0x8000000000000000LL;
              }
            }
            if ( v302 != v292 )
            {
              v313 = v43;
              v49 = sub_BCD140((_QWORD *)*v43, 1u);
              v50 = v313;
              v51 = (__int64 *)v49;
              v52 = *((unsigned __int8 *)v313 + 8);
              if ( (unsigned int)(v52 - 17) <= 1 )
              {
                v53 = *((_DWORD *)v313 + 8);
                BYTE4(v348) = (_BYTE)v52 == 18;
                LODWORD(v348) = v53;
                v54 = sub_BCE1B0(v51, v348);
                v50 = v313;
                v51 = (__int64 *)v54;
              }
              v314 = v50;
              v55 = sub_34D1290(a1, 53, v50, (__int64)v51, 32, a3, 0, 0, 0);
              v44 = __OFADD__(v55, v45);
              v56 = v55 + v45;
              if ( v44 )
              {
                v56 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v55 <= 0 )
                  v56 = 0x8000000000000000LL;
              }
              v57 = sub_34D1290(a1, 57, v314, (__int64)v51, 32, a3, 0, 0, 0);
              v44 = __OFADD__(v57, v56);
              v45 = v57 + v56;
              if ( v44 )
              {
                v45 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v57 <= 0 )
                  return 0x8000000000000000LL;
              }
            }
            return v45;
          default:
            goto LABEL_221;
        }
      }
      if ( v6 == 65 )
      {
        v25 = v12 != 18;
        if ( v82 == 1 && v12 != 18 )
          goto LABEL_46;
LABEL_106:
        v300 = 0;
        if ( v82 <= 1 || !v25 )
          goto LABEL_32;
        v98 = *(_BYTE *)(v11 + 8);
        if ( v98 != 7 )
        {
          if ( v98 != 15 )
          {
            v99 = &v332;
            v287 = v333;
LABEL_111:
            v272 = v11;
            v265 = v6;
            v100 = v99;
            v260 = v4;
            v101 = a1;
            v102 = a3;
            v103 = 0;
            v104 = 0;
            do
            {
              v105 = sub_34D2080(v101, *v100, 1, 0);
              if ( v106 == 1 )
                v104 = 1;
              if ( __OFADD__(v105, v103) )
              {
                v103 = 0x8000000000000000LL;
                if ( v105 > 0 )
                  v103 = 0x7FFFFFFFFFFFFFFFLL;
              }
              else
              {
                v103 += v105;
              }
              ++v100;
            }
            while ( v287 != v100 );
            v288 = v104;
            v11 = v272;
            v107 = v103;
            a3 = v102;
            v6 = v265;
            a1 = v101;
            v4 = v260;
            goto LABEL_118;
          }
          v99 = *(__int64 **)(v11 + 16);
          v287 = &v99[*(unsigned int *)(v11 + 12)];
          if ( v287 != v99 )
            goto LABEL_111;
        }
        v288 = 0;
        v107 = 0;
LABEL_118:
        v273 = v11;
        v108 = sub_34D0EB0(
                 a1,
                 *(_BYTE ***)(v4 + 72),
                 *(_DWORD *)(v4 + 80),
                 *(__int64 **)(v4 + 24),
                 *(unsigned int *)(v4 + 32),
                 a3);
        v11 = v273;
        if ( v21 != 1 )
          v21 = v288;
        v44 = __OFADD__(v108, v107);
        v22 = v108 + v107;
        if ( v44 )
        {
          v22 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v108 <= 0 )
            v22 = 0x8000000000000000LL;
        }
        goto LABEL_33;
      }
      if ( v6 == 67 )
      {
        v25 = v12 != 18;
        if ( v82 == 1 && v12 != 18 )
          goto LABEL_41;
        goto LABEL_106;
      }
    }
LABEL_221:
    v25 = v12 != 18;
    goto LABEL_106;
  }
  switch ( v6 )
  {
    case 0x11Du:
LABEL_186:
      v161 = *(_QWORD *)(*(_QWORD *)(v4 + 72) + 8LL);
      if ( *(_BYTE *)v161 != 17 )
        goto LABEL_105;
      v266 = v82;
      v276 = v11;
      v297 = *(_QWORD *)(*(_QWORD *)(v307 + 40) + 72LL);
      v162 = sub_B2D610(v297, 47);
      v11 = v276;
      v163 = (const void **)(v161 + 24);
      v82 = v266;
      if ( v162 )
      {
        v164 = *(_DWORD *)(v161 + 32);
        v165 = *(__int64 **)(v161 + 24);
        if ( v164 <= 0x40 )
        {
          if ( !v164 )
            goto LABEL_105;
          goto LABEL_190;
        }
      }
      else
      {
        v205 = sub_B2D610(v297, 18);
        v164 = *(_DWORD *)(v161 + 32);
        v165 = *(__int64 **)(v161 + 24);
        v11 = v276;
        v82 = v266;
        v163 = (const void **)(v161 + 24);
        if ( v164 <= 0x40 )
        {
          if ( v164 )
          {
            if ( v205 )
            {
LABEL_190:
              v166 = abs64((__int64)((_QWORD)v165 << (64 - (unsigned __int8)v164)) >> (64 - (unsigned __int8)v164));
              goto LABEL_191;
            }
          }
          else if ( v205 )
          {
            goto LABEL_105;
          }
          v172 = *(_QWORD *)(v161 + 24);
          if ( _bittest64(&v172, v164 - 1) )
            goto LABEL_196;
          goto LABEL_273;
        }
        if ( !v205 )
        {
          v170 = v164 - 1;
          v171 = 1LL << ((unsigned __int8)v164 - 1);
          goto LABEL_286;
        }
      }
      v166 = abs64(*v165);
LABEL_191:
      v256 = v164;
      v261 = v163;
      v267 = v82;
      v277 = v11;
      v298 = v166;
      v167 = sub_39FAC40(v166);
      v11 = v277;
      v82 = v267;
      v163 = v261;
      v164 = v256;
      if ( v298 )
      {
        _BitScanReverse64(&v168, v298);
        v169 = v168 ^ 0x3F;
      }
      else
      {
        v169 = 64;
      }
      if ( (unsigned int)(v167 - v169 + 63) > 6 )
        goto LABEL_105;
      v170 = v256 - 1;
      v171 = 1LL << ((unsigned __int8)v256 - 1);
      if ( v256 <= 0x40 )
      {
        v172 = *(_QWORD *)(v161 + 24);
        if ( (v171 & v172) != 0 )
        {
LABEL_196:
          LODWORD(v349) = v164;
          v348 = v172;
          goto LABEL_197;
        }
LABEL_273:
        LODWORD(v343) = v164;
        v342 = *(_QWORD *)(v161 + 24);
        goto LABEL_274;
      }
LABEL_286:
      if ( (v165[v170 >> 6] & v171) == 0 )
      {
        v328 = v11;
        LODWORD(v343) = v164;
        sub_C43780((__int64)&v342, v163);
        v174 = v343;
        v11 = v328;
        goto LABEL_202;
      }
      v331 = v11;
      LODWORD(v349) = v164;
      sub_C43780((__int64)&v348, v163);
      v11 = v331;
LABEL_197:
      if ( (unsigned int)v349 > 0x40 )
      {
        v330 = v11;
        sub_C43D10((__int64)&v348);
        v11 = v330;
      }
      else
      {
        v173 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v349;
        if ( !(_DWORD)v349 )
          v173 = 0;
        v348 = v173 & ~v348;
      }
      v323 = v11;
      sub_C46250((__int64)&v348);
      v174 = v349;
      v11 = v323;
      LODWORD(v343) = v349;
      v342 = v348;
LABEL_202:
      if ( v174 > 0x40 )
      {
        v324 = v11;
        v175 = v174 - sub_C444A0((__int64)&v342);
        v176 = sub_C44630((__int64)&v342);
        v177 = v324;
        v178 = v176;
        goto LABEL_204;
      }
LABEL_274:
      v206 = 64;
      if ( v342 )
      {
        _BitScanReverse64(&v207, v342);
        v206 = v207 ^ 0x3F;
      }
      v327 = v11;
      v175 = 64 - v206;
      v208 = sub_39FAC40(v342);
      v177 = v327;
      v178 = v208;
LABEL_204:
      v325 = v177;
      v179 = sub_34D2250(a1, 0x12u, v177, a3, 0, 0, 0, 0, 0);
      v180 = (unsigned int)(v178 + v175 - 2);
      v181 = v179;
      v183 = v179;
      v182 = v179 * v180;
      if ( is_mul_ok(v183, v180) )
      {
        v184 = v182;
      }
      else if ( v181 <= 0 || (v184 = 0x7FFFFFFFFFFFFFFFLL, !v180) )
      {
        v184 = 0x8000000000000000LL;
      }
      v185 = *(_DWORD *)(v161 + 32);
      v186 = *(_QWORD *)(v161 + 24);
      if ( v185 > 0x40 )
        v186 = *(_QWORD *)(v186 + 8LL * ((v185 - 1) >> 6));
      if ( (v186 & (1LL << ((unsigned __int8)v185 - 1))) != 0 )
      {
        v228 = sub_34D2250(a1, 0x15u, v325, a3, 0, 0, 0, 0, 0);
        if ( __OFADD__(v228, v184) )
        {
          v184 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v228 <= 0 )
            v184 = 0x8000000000000000LL;
        }
        else
        {
          v184 += v228;
        }
      }
      result = v184;
      if ( (unsigned int)v343 > 0x40 )
      {
        v149 = v342;
        if ( v342 )
        {
LABEL_166:
          v322 = result;
          j_j___libc_free_0_0(v149);
          return v322;
        }
      }
      return result;
    case 0x146u:
    case 0x147u:
LABEL_95:
      v83 = (__int64 *)v11;
      if ( v81 == 15 )
        v83 = **(__int64 ***)(v11 + 16);
      v84 = *(_QWORD *)(a1 + 8);
      v270 = v82;
      v293 = v11;
      v85 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), v84, v83, 0);
      v86 = v293;
      v87 = v270;
      LODWORD(v348) = v85;
      v88 = *(_DWORD *)(v4 + 16);
      v349 = v89;
      switch ( v88 )
      {
        case 326:
          v90 = sub_2FE5F00;
LABEL_100:
          v91 = v348;
          if ( (_WORD)v348 )
          {
            if ( (unsigned __int16)(v348 - 17) <= 0xD3u )
            {
              v92 = 0;
              v91 = word_4456580[(unsigned __int16)v348 - 1];
              goto LABEL_103;
            }
          }
          else
          {
            v200 = sub_30070B0((__int64)&v348);
            v86 = v293;
            v87 = v270;
            v91 = 0;
            if ( v200 )
            {
              v203 = sub_3009970((__int64)&v348, v84, 0, v201, v202);
              v86 = v293;
              v87 = v270;
              v92 = v204;
              v91 = v203;
              goto LABEL_103;
            }
          }
          v92 = v349;
LABEL_103:
          v271 = v87;
          v294 = v86;
          v93 = v90(v91, v92);
          v94 = *(__int64 **)(v4 + 144);
          v95 = *(_QWORD *)(v4 + 8);
          v96 = v93;
          v97 = *(_DWORD *)(v4 + 16);
          v11 = v294;
          v338 = v95;
          v82 = v271;
          v264 = v97;
          if ( !v94
            || *(_BYTE *)(v95 + 8) != 15
            || (v258 = v271, v262 = v94, v279 = v96, v216 = sub_E456C0(v95), v11 = v294, v82 = v258, !v216)
            || (v217 = *(char **)(*(_QWORD *)(a1 + 24) + 8LL * v279 + 525288)) == 0 )
          {
LABEL_105:
            v25 = !v286;
            goto LABEL_106;
          }
          v218 = *(_QWORD **)v95;
          v342 = v95;
          v219 = v262;
          v257 = v218;
          if ( *(_BYTE *)(v95 + 8) == 15 )
            v220 = *(__int64 **)(v95 + 16);
          else
            v220 = &v342;
          v221 = *v220;
          v280 = 0;
          v222 = *(_DWORD *)(v221 + 32);
          v61 = *(_BYTE *)(v221 + 8) == 18;
          LOWORD(v336[0]) = 256;
          LODWORD(v342) = v222;
          v263 = a1;
          v255 = v4;
          v223 = v95;
          v224 = v219;
          v259 = a3;
          v225 = v82;
          BYTE4(v342) = v61;
          while ( 1 )
          {
            v254 = *((_BYTE *)v336 + v280);
            v226 = strlen(v217);
            v227 = sub_97F930(*v224, v217, v226, (__int64)&v342, v254);
            if ( v227 )
              break;
            if ( v280 == 1 )
            {
              v82 = v225;
              v11 = v294;
              v4 = v255;
              a3 = v259;
              goto LABEL_105;
            }
            v280 = 1;
          }
          v234 = v223;
          if ( *(_BYTE *)(v227 + 40) )
          {
            v244 = (__int64 *)sub_BCB2A0(v257);
            v245 = sub_BCE1B0(v244, v342);
            v246 = sub_34D5BE0(a1, 0, v245, 0, 0, v259, 0, 0);
            v235 = v246 + 10;
            if ( __OFADD__(10, v246) )
            {
              v235 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v246 <= 0 )
                v235 = 0x8000000000000000LL;
            }
          }
          else
          {
            v235 = 10;
          }
          if ( *(_BYTE *)(v234 + 8) == 15 )
          {
            v236 = *(unsigned __int64 **)(v234 + 16);
            v237 = (unsigned int *)&v236[*(unsigned int *)(v234 + 12)];
            if ( v237 == (unsigned int *)v236 )
              return v235;
          }
          else
          {
            v237 = &v339;
            v236 = &v338;
          }
          v238 = v253;
          v239 = 0;
          v240 = (__int64 *)v236;
          v241 = v263;
          do
          {
            if ( v239 || v264 != 249 )
            {
              v291 = v237;
              v299 = v241;
              LOBYTE(v238) = sub_AE5020(*(_QWORD *)(v241 + 8), *v240);
              v242 = v238;
              BYTE1(v242) = 1;
              v238 = v242;
              v243 = sub_34D2F80(v299, 32, *v240, v242, 0, v259);
              v241 = v299;
              v237 = v291;
              v44 = __OFADD__(v243, v235);
              v235 += v243;
              if ( v44 )
              {
                v235 = 0x8000000000000000LL;
                if ( v243 > 0 )
                  v235 = 0x7FFFFFFFFFFFFFFFLL;
              }
            }
            ++v240;
            ++v239;
          }
          while ( v240 != (__int64 *)v237 );
          return v235;
        case 327:
          v90 = sub_2FE5F30;
          goto LABEL_100;
        case 249:
          v90 = sub_2FE5F60;
          goto LABEL_100;
      }
      break;
    case 0x159u:
LABEL_181:
      if ( v81 != 18 )
        return 1;
      v160 = *(_DWORD *)(v4 + 16);
      if ( v160 <= 0xD3 )
      {
        if ( v160 <= 0x94 )
        {
          switch ( v160 )
          {
            case 5u:
            case 6u:
            case 7u:
            case 8u:
            case 0xBu:
            case 0x1Bu:
            case 0x1Cu:
            case 0x27u:
            case 0x28u:
            case 0x2Bu:
            case 0x2Eu:
            case 0x2Fu:
            case 0x3Au:
            case 0x3Bu:
            case 0x3Cu:
            case 0x44u:
            case 0x45u:
            case 0x46u:
            case 0x47u:
              return 0;
            default:
              return 1;
          }
        }
        switch ( v160 )
        {
          case 0x95u:
          case 0x96u:
          case 0x9Bu:
          case 0xA9u:
          case 0xCCu:
          case 0xCDu:
          case 0xCEu:
          case 0xD0u:
          case 0xD2u:
          case 0xD3u:
            return 0;
          case 0xA1u:
            goto LABEL_328;
          default:
            return 1;
        }
      }
      goto LABEL_256;
    case 0x17Du:
LABEL_176:
      if ( v81 != 18 )
      {
        v157 = *(_QWORD **)(v4 + 72);
        v158 = v157[1];
        v159 = *(_QWORD **)(v158 + 24);
        if ( *(_DWORD *)(v158 + 32) > 0x40u )
          v159 = (_QWORD *)*v159;
        return sub_34D5BE0(a1, 5, *(_QWORD *)(*v157 + 8LL), 0, 0, a3, (signed int)v159, v11);
      }
      v160 = *(_DWORD *)(v4 + 16);
      if ( v160 <= 0xD3 )
      {
        if ( v160 <= 0x94 )
        {
          switch ( v160 )
          {
            case 5u:
            case 6u:
            case 7u:
            case 8u:
            case 0xBu:
            case 0x1Bu:
            case 0x1Cu:
            case 0x27u:
            case 0x28u:
            case 0x2Bu:
            case 0x2Eu:
            case 0x2Fu:
            case 0x3Au:
            case 0x3Bu:
            case 0x3Cu:
            case 0x44u:
            case 0x45u:
            case 0x46u:
            case 0x47u:
              return 0;
            default:
              return 1;
          }
        }
        switch ( v160 )
        {
          case 0x95u:
          case 0x96u:
          case 0x9Bu:
          case 0xA9u:
          case 0xCCu:
          case 0xCDu:
          case 0xCEu:
          case 0xD0u:
          case 0xD2u:
          case 0xD3u:
            return 0;
          case 0xA1u:
            goto LABEL_328;
          default:
            return 1;
        }
      }
      goto LABEL_256;
    case 0x17Eu:
LABEL_171:
      v153 = *(_QWORD **)(v4 + 72);
      v154 = *(_QWORD *)(v153[1] + 8LL);
      if ( *(_BYTE *)(v154 + 8) == 18 )
      {
        v160 = *(_DWORD *)(v4 + 16);
        if ( v160 > 0xD3 )
        {
LABEL_256:
          if ( v160 > 0x178 )
          {
            return 1;
          }
          else if ( v160 > 0x143 )
          {
            return ((1LL << ((unsigned __int8)v160 - 68)) & 0x10000020401001LL) == 0;
          }
          else
          {
            return v160 != 282 && v160 - 291 >= 2;
          }
        }
        else
        {
          if ( v160 <= 0x94 )
          {
            switch ( v160 )
            {
              case 5u:
              case 6u:
              case 7u:
              case 8u:
              case 0xBu:
              case 0x1Bu:
              case 0x1Cu:
              case 0x27u:
              case 0x28u:
              case 0x2Bu:
              case 0x2Eu:
              case 0x2Fu:
              case 0x3Au:
              case 0x3Bu:
              case 0x3Cu:
              case 0x44u:
              case 0x45u:
              case 0x46u:
              case 0x47u:
                return 0;
              default:
                return 1;
            }
          }
          switch ( v160 )
          {
            case 0x95u:
            case 0x96u:
            case 0x9Bu:
            case 0xA9u:
            case 0xCCu:
            case 0xCDu:
            case 0xCEu:
            case 0xD0u:
            case 0xD2u:
            case 0xD3u:
              return 0;
            case 0xA1u:
LABEL_328:
              result = 0;
              break;
            default:
              return 1;
          }
        }
      }
      else
      {
        v155 = v153[2];
        v156 = *(_QWORD **)(v155 + 24);
        if ( *(_DWORD *)(v155 + 32) > 0x40u )
          v156 = (_QWORD *)*v156;
        return sub_34D5BE0(a1, 4, *(_QWORD *)(*v153 + 8LL), 0, 0, a3, (signed int)v156, v154);
      }
      return result;
    case 0x183u:
    case 0x184u:
    case 0x186u:
    case 0x187u:
    case 0x188u:
    case 0x189u:
    case 0x18Bu:
    case 0x18Cu:
    case 0x18Du:
    case 0x18Eu:
    case 0x18Fu:
    case 0x190u:
    case 0x191u:
LABEL_59:
      v342 = *(_QWORD *)(**(_QWORD **)(v4 + 72) + 8LL);
      sub_DF8CB0((__int64)&v348, v6, v11, (char *)&v342, 1, v301, v307, 1u);
      goto LABEL_60;
    case 0x185u:
    case 0x18Au:
LABEL_124:
      v109 = *(_QWORD **)(v4 + 72);
      v342 = *(_QWORD *)(*v109 + 8LL);
      v343 = *(_QWORD *)(v109[1] + 8LL);
      sub_DF8CB0((__int64)&v348, v6, v11, (char *)&v342, 2, v301, v307, 1u);
LABEL_60:
      v23 = a3;
      goto LABEL_34;
    case 0x192u:
      return sub_34D5BE0(a1, 1, *(_QWORD *)(**(_QWORD **)(v4 + 72) + 8LL), 0, 0, a3, 0, v11);
    case 0x193u:
LABEL_167:
      v150 = *(_QWORD **)(v4 + 72);
      v151 = v150[2];
      v152 = *(_QWORD **)(v151 + 24);
      if ( *(_DWORD *)(v151 + 32) > 0x40u )
        v152 = (_QWORD *)*v152;
      return sub_34D5BE0(a1, 8, *(_QWORD *)(*v150 + 8LL), 0, 0, a3, (signed int)v152, v11);
    default:
      goto LABEL_221;
  }
LABEL_394:
  BUG();
}
