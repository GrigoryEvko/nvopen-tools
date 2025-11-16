// Function: sub_306A930
// Address: 0x306a930
//
signed __int64 __fastcall sub_306A930(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r15
  unsigned int v6; // r14d
  __int64 v7; // rdx
  signed __int64 result; // rax
  __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // r10
  unsigned __int8 v12; // al
  char v13; // al
  int v14; // ecx
  unsigned __int64 v15; // rdx
  int v16; // edi
  __int64 v17; // rdx
  int v18; // ecx
  int v19; // edi
  bool v20; // r11
  unsigned int v21; // eax
  unsigned __int64 v22; // r11
  __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  __int64 v25; // r14
  char *v26; // rax
  unsigned __int16 v27; // ax
  unsigned __int8 v28; // dl
  __int64 v29; // r8
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rcx
  unsigned __int8 **v33; // rax
  unsigned __int8 *v34; // r15
  unsigned __int8 *v35; // r14
  int v36; // r15d
  __int64 v37; // rbx
  __int64 v38; // rax
  bool v39; // of
  unsigned __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  __int64 *v44; // r10
  unsigned __int64 v45; // rbx
  int v46; // eax
  int v47; // esi
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
  _QWORD ***v65; // rax
  char v66; // r8
  int v67; // esi
  _QWORD **v68; // rdx
  __int64 v69; // rax
  _BYTE *v70; // rdx
  bool v71; // cf
  bool v72; // zf
  __int64 v73; // rdx
  unsigned __int64 v74; // rax
  _BYTE *v75; // rbx
  unsigned __int16 v76; // ax
  __int64 *v77; // rax
  __int64 v78; // rbx
  _BYTE *v79; // r14
  unsigned __int16 v80; // ax
  unsigned __int8 v81; // cl
  unsigned __int8 v82; // cl
  unsigned int v83; // r9d
  __int64 *v84; // rdx
  __int64 v85; // rsi
  int v86; // eax
  __int64 v87; // r10
  unsigned int v88; // r9d
  int v89; // eax
  __int64 v90; // rdx
  __int64 (__fastcall *v91)(__int16, __int64); // rbx
  __int16 v92; // dx
  __int64 v93; // rsi
  int v94; // eax
  __int64 *v95; // r11
  __int64 v96; // rbx
  int v97; // edx
  int v98; // eax
  char v99; // al
  __int64 *v100; // rbx
  unsigned __int64 v101; // r14
  __int64 *v102; // r15
  int v103; // ebx
  __int64 v104; // r10
  unsigned int v105; // edx
  unsigned __int64 v106; // rsi
  __int64 v107; // rdx
  signed __int64 v108; // rax
  int v109; // edx
  _QWORD *v110; // rcx
  __int64 *v111; // rax
  char v112; // r15
  __int64 v113; // r14
  __int16 v114; // ax
  unsigned __int8 v115; // bl
  unsigned __int8 v116; // al
  __int64 v117; // rax
  char v118; // r14
  __int64 v119; // r15
  __int16 v120; // ax
  unsigned __int8 v121; // bl
  unsigned __int8 v122; // al
  unsigned int v123; // eax
  __int64 v124; // rdi
  __int64 *v125; // r10
  __int64 v126; // rsi
  __int64 v127; // rdx
  __int64 (*v128)(); // rax
  __int64 v129; // rax
  unsigned int v130; // ebx
  int v131; // eax
  bool v132; // bl
  char v133; // bl
  __int64 v134; // r10
  __int64 v135; // r9
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rax
  unsigned int v138; // eax
  __int64 v139; // rax
  __int64 v140; // rbx
  __int64 v141; // rax
  unsigned __int64 v142; // rbx
  signed __int64 v143; // rax
  unsigned __int64 v144; // rbx
  __int64 v145; // rax
  unsigned __int64 v146; // rbx
  __int64 v147; // rax
  unsigned __int64 v148; // rbx
  __int64 v149; // rcx
  unsigned __int64 v150; // rdi
  _QWORD *v151; // rax
  __int64 v152; // rcx
  __int64 v153; // rsi
  _QWORD *v154; // rdx
  _QWORD *v155; // rdx
  __int64 v156; // rcx
  _QWORD *v157; // rax
  _QWORD *v158; // rdx
  __int64 v159; // rcx
  _QWORD *v160; // rax
  unsigned int v161; // eax
  __int64 v162; // rbx
  char v163; // al
  const void **v164; // r8
  unsigned int v165; // edx
  __int64 *v166; // rsi
  unsigned __int64 v167; // rcx
  int v168; // eax
  unsigned __int64 v169; // rcx
  int v170; // ecx
  unsigned int v171; // ecx
  __int64 v172; // rax
  __int64 v173; // rcx
  unsigned __int64 v174; // rdx
  unsigned int v175; // r14d
  int v176; // r14d
  int v177; // eax
  int v178; // r10d
  int v179; // r15d
  __int64 v180; // rax
  unsigned __int64 v181; // rdx
  __int64 v182; // rcx
  unsigned __int64 v183; // rax
  unsigned __int64 v184; // kr30_8
  unsigned __int64 v185; // r14
  unsigned int v186; // edi
  __int64 v187; // rdx
  __int64 v188; // rdx
  __int64 v189; // rbx
  bool v190; // al
  __int64 v191; // r8
  char *v192; // r10
  __int64 v193; // r14
  char *v194; // rax
  unsigned __int16 v195; // ax
  unsigned __int8 v196; // dl
  __int64 v197; // r8
  __int64 v198; // rax
  __int64 v199; // rcx
  int v200; // eax
  unsigned __int64 v201; // r11
  unsigned __int64 v202; // rax
  __int64 v203; // rcx
  unsigned int v204; // edx
  bool v205; // al
  __int64 v206; // rcx
  __int64 v207; // r8
  __int16 v208; // ax
  __int64 v209; // rdx
  char v210; // al
  int v211; // eax
  unsigned __int64 v212; // rax
  int v213; // eax
  char v214; // al
  __int64 v215; // r13
  signed __int64 v216; // rbx
  __int64 v217; // rcx
  __int16 v218; // r15
  __int64 v219; // rdx
  __int64 v220; // r14
  __int64 v221; // rax
  __int64 v222; // r12
  __int64 v223; // r13
  __int64 v224; // rcx
  char v225; // al
  char *v226; // rsi
  _QWORD *v227; // rax
  __int64 *v228; // r11
  __int64 *v229; // rax
  __int64 v230; // rax
  int v231; // edx
  unsigned int v232; // r15d
  __int64 v233; // r13
  __int64 *v234; // rbx
  size_t v235; // rax
  __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rax
  __int64 v239; // r10
  unsigned int v240; // eax
  unsigned int v241; // eax
  unsigned __int64 v242; // r14
  unsigned __int64 *v243; // r10
  unsigned int *v244; // r11
  unsigned int v245; // r13d
  __int64 v246; // r12
  __int64 *v247; // rbx
  __int64 v248; // r10
  unsigned int v249; // eax
  __int64 v250; // rax
  __int64 *v251; // rax
  __int64 v252; // rax
  signed __int64 v253; // rax
  bool v254; // cc
  __int128 v255; // [rsp-18h] [rbp-278h]
  __int128 v256; // [rsp-18h] [rbp-278h]
  __int128 v257; // [rsp-18h] [rbp-278h]
  int v258; // [rsp-10h] [rbp-270h]
  unsigned __int16 v259; // [rsp+Ah] [rbp-256h]
  char v260; // [rsp+Ch] [rbp-254h]
  __int64 v261; // [rsp+10h] [rbp-250h]
  __int64 v262; // [rsp+30h] [rbp-230h]
  unsigned int v263; // [rsp+30h] [rbp-230h]
  _QWORD *v264; // [rsp+30h] [rbp-230h]
  unsigned int v265; // [rsp+38h] [rbp-228h]
  unsigned int v266; // [rsp+38h] [rbp-228h]
  unsigned int v267; // [rsp+38h] [rbp-228h]
  int v268; // [rsp+40h] [rbp-220h]
  __int64 v269; // [rsp+50h] [rbp-210h]
  unsigned int v270; // [rsp+50h] [rbp-210h]
  __int64 *v271; // [rsp+50h] [rbp-210h]
  __int64 v272; // [rsp+50h] [rbp-210h]
  int v273; // [rsp+58h] [rbp-208h]
  signed __int64 v274; // [rsp+58h] [rbp-208h]
  unsigned int v275; // [rsp+58h] [rbp-208h]
  const void **v276; // [rsp+58h] [rbp-208h]
  __int64 v277; // [rsp+58h] [rbp-208h]
  unsigned __int64 v278; // [rsp+58h] [rbp-208h]
  __int64 v279; // [rsp+58h] [rbp-208h]
  __int64 v280; // [rsp+60h] [rbp-200h]
  bool v281; // [rsp+60h] [rbp-200h]
  __int64 *v282; // [rsp+60h] [rbp-200h]
  __int64 v283; // [rsp+60h] [rbp-200h]
  __int64 v284; // [rsp+60h] [rbp-200h]
  _QWORD **v285; // [rsp+60h] [rbp-200h]
  __int64 *v286; // [rsp+60h] [rbp-200h]
  __int64 v287; // [rsp+60h] [rbp-200h]
  __int64 v288; // [rsp+60h] [rbp-200h]
  __int64 v289; // [rsp+60h] [rbp-200h]
  __int64 v290; // [rsp+60h] [rbp-200h]
  __int64 v291; // [rsp+60h] [rbp-200h]
  __int64 v292; // [rsp+68h] [rbp-1F8h]
  int v293; // [rsp+68h] [rbp-1F8h]
  unsigned int v294; // [rsp+68h] [rbp-1F8h]
  unsigned int v295; // [rsp+68h] [rbp-1F8h]
  __int64 v296; // [rsp+68h] [rbp-1F8h]
  __int64 v297; // [rsp+68h] [rbp-1F8h]
  int v298; // [rsp+68h] [rbp-1F8h]
  __int64 v299; // [rsp+68h] [rbp-1F8h]
  unsigned int *v300; // [rsp+68h] [rbp-1F8h]
  int v301; // [rsp+70h] [rbp-1F0h]
  __int64 v302; // [rsp+70h] [rbp-1F0h]
  __int64 v303; // [rsp+70h] [rbp-1F0h]
  __int64 *v304; // [rsp+70h] [rbp-1F0h]
  __int64 *v305; // [rsp+70h] [rbp-1F0h]
  __int64 v306; // [rsp+70h] [rbp-1F0h]
  unsigned __int64 v307; // [rsp+70h] [rbp-1F0h]
  __int64 v308; // [rsp+70h] [rbp-1F0h]
  __int64 v309; // [rsp+78h] [rbp-1E8h]
  int v310; // [rsp+80h] [rbp-1E0h]
  unsigned __int8 *v311; // [rsp+80h] [rbp-1E0h]
  __int64 v312; // [rsp+80h] [rbp-1E0h]
  __int64 v313; // [rsp+80h] [rbp-1E0h]
  _QWORD **v314; // [rsp+80h] [rbp-1E0h]
  char *v315; // [rsp+80h] [rbp-1E0h]
  __int64 v316; // [rsp+88h] [rbp-1D8h]
  __int64 v317; // [rsp+88h] [rbp-1D8h]
  signed __int64 v318; // [rsp+88h] [rbp-1D8h]
  signed __int64 v319; // [rsp+88h] [rbp-1D8h]
  unsigned __int8 v320; // [rsp+88h] [rbp-1D8h]
  unsigned __int8 *v321; // [rsp+88h] [rbp-1D8h]
  __int64 *v322; // [rsp+88h] [rbp-1D8h]
  __int64 *v323; // [rsp+88h] [rbp-1D8h]
  char v324; // [rsp+88h] [rbp-1D8h]
  char v325; // [rsp+88h] [rbp-1D8h]
  signed __int64 v326; // [rsp+88h] [rbp-1D8h]
  signed __int64 v327; // [rsp+88h] [rbp-1D8h]
  signed __int64 v328; // [rsp+88h] [rbp-1D8h]
  signed __int64 v329; // [rsp+88h] [rbp-1D8h]
  signed __int64 v330; // [rsp+88h] [rbp-1D8h]
  signed __int64 v331; // [rsp+88h] [rbp-1D8h]
  int v332; // [rsp+88h] [rbp-1D8h]
  int v333; // [rsp+88h] [rbp-1D8h]
  int v334; // [rsp+88h] [rbp-1D8h]
  unsigned __int8 v335; // [rsp+88h] [rbp-1D8h]
  int v336; // [rsp+88h] [rbp-1D8h]
  int v337; // [rsp+88h] [rbp-1D8h]
  int v338; // [rsp+88h] [rbp-1D8h]
  int v339; // [rsp+88h] [rbp-1D8h]
  __int64 v340; // [rsp+98h] [rbp-1C8h] BYREF
  __int64 v341[2]; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v342; // [rsp+B0h] [rbp-1B0h]
  unsigned __int64 v343; // [rsp+B8h] [rbp-1A8h]
  _DWORD v344[2]; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v345; // [rsp+C8h] [rbp-198h]
  unsigned __int64 v346; // [rsp+D0h] [rbp-190h] BYREF
  unsigned int v347; // [rsp+D8h] [rbp-188h] BYREF
  unsigned __int64 v348; // [rsp+E0h] [rbp-180h]
  unsigned int v349; // [rsp+E8h] [rbp-178h]
  __int64 v350; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v351; // [rsp+F8h] [rbp-168h]
  char *v352; // [rsp+108h] [rbp-158h]
  char v353; // [rsp+118h] [rbp-148h] BYREF
  char *v354; // [rsp+138h] [rbp-128h]
  char v355; // [rsp+148h] [rbp-118h] BYREF
  __int64 v356; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v357; // [rsp+198h] [rbp-C8h]
  unsigned __int64 v358; // [rsp+1A0h] [rbp-C0h] BYREF
  _BYTE *v359; // [rsp+1A8h] [rbp-B8h]
  _BYTE v360[32]; // [rsp+1B8h] [rbp-A8h] BYREF
  _BYTE *v361; // [rsp+1D8h] [rbp-88h]
  _BYTE v362[120]; // [rsp+1E8h] [rbp-78h] BYREF

  v3 = a2;
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
    v346 = sub_B5A790(*(_DWORD *)(a2 + 16), a2, v9, v10);
    v15 = HIDWORD(v346);
    if ( BYTE4(v346) )
    {
      v16 = *(_DWORD *)(a2 + 16);
      if ( v16 == 438 )
      {
        v193 = *(_QWORD *)a2;
        if ( !*(_QWORD *)a2 )
          goto LABEL_258;
        v194 = *(char **)(v193 - 32);
        if ( !v194 || *v194 || *((_QWORD *)v194 + 3) != *(_QWORD *)(v193 + 80) )
          goto LABEL_401;
        v335 = *v194;
        if ( sub_B5A760(*((_DWORD *)v194 + 9)) )
        {
          v195 = sub_B5A5E0(v193);
          v196 = v335;
          if ( HIBYTE(v195) )
            v196 = v195;
        }
        else
        {
LABEL_258:
          v196 = 0;
        }
        v197 = 0;
        if ( *(_DWORD *)(a2 + 32) > 1u )
        {
          v198 = **(_QWORD **)(a2 + 24);
          if ( *(_BYTE *)(v198 + 8) == 14 )
            v197 = *(_DWORD *)(v198 + 8) >> 8;
        }
        v199 = v196;
        BYTE1(v199) = 1;
        return sub_30670E0(a1, v346, *(_QWORD *)(a2 + 8), v199, v197, a3);
      }
      if ( v16 == 481 )
      {
        v25 = *(_QWORD *)a2;
        if ( !*(_QWORD *)a2 )
          goto LABEL_255;
        v26 = *(char **)(v25 - 32);
        if ( !v26 || *v26 || *((_QWORD *)v26 + 3) != *(_QWORD *)(v25 + 80) )
          goto LABEL_401;
        v320 = *v26;
        if ( sub_B5A760(*((_DWORD *)v26 + 9)) )
        {
          v27 = sub_B5A5E0(v25);
          v28 = v320;
          if ( HIBYTE(v27) )
            v28 = v27;
        }
        else
        {
LABEL_255:
          v28 = 0;
        }
        v29 = 0;
        v30 = *(__int64 **)(a2 + 24);
        if ( *(_DWORD *)(a2 + 32) > 1u )
        {
          v31 = v30[1];
          if ( *(_BYTE *)(v31 + 8) == 14 )
            v29 = *(_DWORD *)(v31 + 8) >> 8;
        }
        v32 = v28;
        BYTE1(v32) = 1;
        return sub_30670E0(a1, v346, *v30, v32, v29, a3);
      }
      if ( (unsigned __int8)sub_B5B050(v16) )
        return sub_3075ED0(a1, v346, *(_QWORD *)(a2 + 8), a3, 0, 0, 0, 0, 0);
      v19 = *(_DWORD *)(a2 + 16);
      if ( v19 == 424 )
        return sub_3075ED0(a1, v346, *(_QWORD *)(a2 + 8), a3, 0, 0, 0, 0, 0);
      if ( sub_B5B010(v19, a2, v17, v18) )
        return sub_3065900(a1, v346, *(_QWORD *)(a2 + 8), **(_QWORD **)(a2 + 24), 0, a3, 0);
      if ( sub_B5B040(*(_DWORD *)(a2 + 16)) && *(_QWORD *)a2 )
      {
        sub_B5B080(*(_QWORD *)a2);
        return sub_3066CD0(a1, v346, **(__int64 ***)(a2 + 24), *(_QWORD *)(a2 + 8), v200, a3, 0, 0, 0);
      }
    }
    v350 = sub_B5A9F0(*(_DWORD *)(a2 + 16), a2, v15, v14);
    if ( BYTE4(v350) )
    {
      v189 = *(unsigned int *)(a2 + 32);
      v315 = *(char **)(a2 + 24);
      v190 = sub_B5B000(*(_DWORD *)(a2 + 16));
      v191 = v189 - 2;
      v192 = v315;
      if ( (_DWORD)v350 != 389 && (_DWORD)v350 != 394 && v190 )
      {
        v191 = v189 - 3;
        v192 = v315 + 8;
      }
      *((_QWORD *)&v257 + 1) = 1;
      *(_QWORD *)&v257 = 0;
      sub_DF8CB0((__int64)&v356, v350, *(_QWORD *)(a2 + 8), v192, v191, *(_DWORD *)(a2 + 120), 0, v257);
      result = sub_3078340(a1, &v356, a3);
      v24 = (unsigned __int64)v361;
      if ( v361 == v362 )
        goto LABEL_45;
LABEL_44:
      v318 = result;
      _libc_free(v24);
      result = v318;
LABEL_45:
      if ( v359 != v360 )
      {
        v319 = result;
        _libc_free((unsigned __int64)v359);
        return v319;
      }
      return result;
    }
  }
  if ( !*(_DWORD *)(a2 + 80) )
    return sub_306D850(a1, a2, a3);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_BYTE *)(v11 + 8);
  v340 = v11;
  if ( v12 == 15 )
  {
    v316 = v11;
    v13 = sub_E456C0(v11);
    v11 = v316;
    if ( !v13 )
      goto LABEL_18;
    v82 = *(_BYTE *)(v316 + 8);
    if ( v82 == 15 )
    {
      v188 = **(_QWORD **)(v316 + 16);
      v12 = *(_BYTE *)(v188 + 8);
    }
    else
    {
      v12 = *(_BYTE *)(v316 + 8);
      v188 = v316;
    }
  }
  else
  {
    if ( (unsigned int)v12 - 17 > 1 )
    {
LABEL_18:
      v317 = *(_QWORD *)a2;
      v310 = *(_DWORD *)(a2 + 120);
      if ( v6 > 0xF9 )
      {
        switch ( v6 )
        {
          case 0x11Du:
            v281 = 0;
            v83 = 1;
            goto LABEL_188;
          case 0x146u:
          case 0x147u:
LABEL_92:
            v281 = 0;
            v82 = *(_BYTE *)(v11 + 8);
            v83 = 1;
            goto LABEL_93;
          case 0x159u:
            v82 = *(_BYTE *)(v11 + 8);
            goto LABEL_183;
          case 0x17Du:
            v82 = *(_BYTE *)(v11 + 8);
            goto LABEL_178;
          case 0x17Eu:
            goto LABEL_169;
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
            goto LABEL_58;
          case 0x185u:
          case 0x18Au:
            goto LABEL_126;
          case 0x192u:
            return sub_30690B0(a1, 1, *(_QWORD *)(**(_QWORD **)(a2 + 72) + 8LL), 0, 0, a3, 0, v11);
          case 0x193u:
            goto LABEL_173;
          default:
            goto LABEL_41;
        }
      }
      if ( v6 > 0xE0 )
      {
        switch ( v6 )
        {
          case 0xE1u:
            goto LABEL_89;
          case 0xE2u:
            goto LABEL_86;
          case 0xE3u:
            goto LABEL_80;
          case 0xE5u:
            goto LABEL_74;
          case 0xEEu:
            return 4;
          case 0xF9u:
            goto LABEL_92;
          default:
            goto LABEL_41;
        }
      }
      if ( v6 > 0xB9 )
        goto LABEL_41;
      if ( v6 > 0x90 )
      {
        switch ( v6 )
        {
          case 0x91u:
            goto LABEL_137;
          case 0xA2u:
          case 0xB9u:
            return sub_306D850(a1, a2, a3);
          case 0xA7u:
            goto LABEL_132;
          case 0xA8u:
            goto LABEL_127;
          case 0xB4u:
          case 0xB5u:
            goto LABEL_60;
          default:
            goto LABEL_41;
        }
      }
      if ( v6 != 65 )
      {
        if ( v6 != 67 )
        {
LABEL_41:
          v21 = 1;
          v22 = 0;
LABEL_42:
          *((_QWORD *)&v255 + 1) = v21 | v309 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v255 = v22;
          sub_DF8CB0((__int64)&v356, v6, v11, *(char **)(v3 + 24), *(unsigned int *)(v3 + 32), v310, v317, v255);
          v23 = a3;
LABEL_43:
          result = sub_306D850(a1, &v356, v23);
          v24 = (unsigned __int64)v361;
          if ( v361 == v362 )
            goto LABEL_45;
          goto LABEL_44;
        }
LABEL_34:
        v292 = v11;
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 288LL))(
                *(_QWORD *)(a1 + 24),
                v11) )
          goto LABEL_40;
        return 1;
      }
LABEL_39:
      v292 = v11;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 296LL))(
              *(_QWORD *)(a1 + 24),
              v11) )
      {
LABEL_40:
        v11 = v292;
        goto LABEL_41;
      }
      return 1;
    }
    v82 = v12;
    v188 = v11;
  }
  v83 = *(_DWORD *)(v188 + 32);
  v281 = v12 == 18;
  v317 = *(_QWORD *)a2;
  v310 = *(_DWORD *)(a2 + 120);
  if ( v6 <= 0xF9 )
  {
    if ( v6 > 0xE0 )
    {
      switch ( v6 )
      {
        case 0xE1u:
LABEL_89:
          v77 = *(__int64 **)(a2 + 72);
          v78 = *v77;
          v79 = (_BYTE *)v77[2];
          v80 = sub_A74840((_QWORD *)(v317 + 72), 1);
          v81 = 0;
          if ( HIBYTE(v80) )
            v81 = v80;
          return sub_3067E40(a1, 33, *(_QWORD ***)(v78 + 8), v81, *v79 > 0x15u, 1, a3, 0);
        case 0xE2u:
LABEL_86:
          v312 = v11;
          v75 = *(_BYTE **)(*(_QWORD *)(a2 + 72) + 8LL);
          v76 = sub_A74840((_QWORD *)(v317 + 72), 0);
          v11 = v312;
          v64 = 0;
          if ( HIBYTE(v76) )
            v64 = v76;
          v258 = a3;
          v66 = *v75 > 0x15u;
          goto LABEL_85;
        case 0xE3u:
LABEL_80:
          v69 = *(_QWORD *)(a2 + 72);
          v70 = *(_BYTE **)(v69 + 16);
          v71 = *v70 < 0x15u;
          v72 = *v70 == 21;
          v73 = *(_QWORD *)(v69 + 8);
          v74 = *(_QWORD *)(v73 + 24);
          if ( *(_DWORD *)(v73 + 32) > 0x40u )
            v74 = *(_QWORD *)v74;
          v64 = 0;
          if ( v74 )
          {
            _BitScanReverse64(&v74, v74);
            v64 = 63 - (v74 ^ 0x3F);
          }
          v66 = !v71 && !v72;
          v258 = a3;
LABEL_85:
          v68 = (_QWORD **)v11;
          v67 = 32;
          return sub_3067E40(a1, v67, v68, v64, v66, 1, v258, 0);
        case 0xE5u:
LABEL_74:
          v58 = *(_QWORD *)(a2 + 72);
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
          v65 = *(_QWORD ****)(a2 + 24);
          v66 = !v60 && !v61;
          v67 = 33;
          v68 = *v65;
          v258 = a3;
          return sub_3067E40(a1, v67, v68, v64, v66, 1, v258, 0);
        case 0xEEu:
          return 4;
        case 0xF9u:
          goto LABEL_93;
        default:
          goto LABEL_223;
      }
    }
    if ( v6 <= 0xB9 )
    {
      if ( v6 > 0x90 )
      {
        switch ( v6 )
        {
          case 0x91u:
LABEL_137:
            v304 = (__int64 *)v11;
            v123 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), **(__int64 ***)(a2 + 24), 1);
            v124 = *(_QWORD *)(a1 + 24);
            v125 = v304;
            v344[0] = v123;
            v126 = v123;
            v345 = v127;
            v128 = *(__int64 (**)())(*(_QWORD *)v124 + 136LL);
            if ( v128 != sub_2FE2E80 )
            {
              v214 = ((__int64 (__fastcall *)(__int64, __int64))v128)(v124, v126);
              v125 = v304;
              if ( !v214 )
              {
                v215 = *v304;
                v216 = 1;
                v217 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v304, 0);
                v218 = v217;
                v220 = v219;
                v221 = a1;
                v222 = v215;
                v223 = v221;
                while ( 1 )
                {
                  LOWORD(v217) = v218;
                  sub_2FE6CC0((__int64)&v356, *(_QWORD *)(v223 + 24), v222, v217, v220);
                  if ( (_BYTE)v356 == 10 )
                    break;
                  if ( !(_BYTE)v356 )
                    return v216;
                  if ( (v356 & 0xFB) == 2 )
                  {
                    if ( is_mul_ok(2u, v216) )
                    {
                      v216 *= 2LL;
                    }
                    else
                    {
                      v254 = v216 <= 0;
                      v216 = 0x7FFFFFFFFFFFFFFFLL;
                      if ( v254 )
                        v216 = 0x8000000000000000LL;
                    }
                  }
                  if ( (_WORD)v357 == v218 && (v358 == v220 || (_WORD)v357) )
                    return v216;
                  v217 = v357;
                  v220 = v358;
                  v218 = v357;
                }
                return 0;
              }
            }
            v129 = *(_QWORD *)(*(_QWORD *)(v3 + 72) + 8LL);
            v130 = *(_DWORD *)(v129 + 32);
            if ( v130 <= 0x40 )
            {
              v132 = *(_QWORD *)(v129 + 24) == 0;
            }
            else
            {
              v305 = v125;
              v131 = sub_C444A0(v129 + 24);
              v125 = v305;
              v132 = v130 == v131;
            }
            v133 = !v132;
            v283 = (__int64)v125;
            LODWORD(v357) = 64;
            v356 = 0;
            LODWORD(v351) = 64;
            v350 = 1;
            sub_AADC30((__int64)&v346, (__int64)&v350, &v356);
            v134 = v283;
            if ( (unsigned int)v351 > 0x40 && v350 )
            {
              j_j___libc_free_0_0(v350);
              v134 = v283;
            }
            if ( (unsigned int)v357 > 0x40 && v356 )
            {
              v284 = v134;
              j_j___libc_free_0_0(v356);
              v134 = v284;
            }
            if ( *(_BYTE *)(**(_QWORD **)(v3 + 24) + 8LL) == 18 )
            {
              if ( v317 )
              {
                v289 = v134;
                v237 = sub_B491C0(v317);
                v134 = v289;
                if ( v237 )
                {
                  v238 = sub_B491C0(v317);
                  sub_988CD0((__int64)&v356, v238, 0x40u);
                  v239 = v289;
                  if ( v347 > 0x40 && v346 )
                  {
                    j_j___libc_free_0_0(v346);
                    v239 = v289;
                  }
                  v346 = v356;
                  v240 = v357;
                  LODWORD(v357) = 0;
                  v347 = v240;
                  if ( v349 > 0x40 && v348 )
                  {
                    v290 = v239;
                    j_j___libc_free_0_0(v348);
                    v239 = v290;
                  }
                  v291 = v239;
                  v348 = v358;
                  v241 = (unsigned int)v359;
                  LODWORD(v359) = 0;
                  v349 = v241;
                  sub_969240((__int64 *)&v358);
                  sub_969240(&v356);
                  v134 = v291;
                }
              }
            }
            v135 = *(_QWORD *)(a1 + 24);
            if ( LOWORD(v344[0]) )
            {
              LOBYTE(v136) = (unsigned __int16)(LOWORD(v344[0]) - 176) <= 0x34u;
              LODWORD(v137) = word_4456340[LOWORD(v344[0]) - 1];
            }
            else
            {
              v279 = *(_QWORD *)(a1 + 24);
              v288 = v134;
              v137 = sub_3007240((__int64)v344);
              v135 = v279;
              v134 = v288;
              v343 = v137;
              v136 = HIDWORD(v137);
            }
            LODWORD(v343) = v137;
            BYTE4(v343) = v136;
            v285 = (_QWORD **)v134;
            v341[1] = v343;
            v138 = sub_2FE69A0(v135, v134, v343, v133, (__int64)&v346);
            v286 = (__int64 *)sub_BCD140(*v285, v138);
            v139 = *(_QWORD *)(**(_QWORD **)(v3 + 72) + 8LL);
            v61 = *(_BYTE *)(v139 + 8) == 18;
            LODWORD(v139) = *(_DWORD *)(v139 + 32);
            BYTE4(v342) = v61;
            LODWORD(v342) = v139;
            v341[0] = sub_BCE1B0(v286, v342);
            *((_QWORD *)&v256 + 1) = 1;
            *(_QWORD *)&v256 = 0;
            sub_DF8CB0((__int64)&v350, 345, v341[0], 0, 0, v310, 0, v256);
            v140 = sub_3078340(a1, &v350, a3);
            v141 = sub_3075ED0(a1, 15, v341[0], a3, 0, 0, 0, 0, 0);
            v39 = __OFADD__(v141, v140);
            v142 = v141 + v140;
            if ( v39 )
            {
              v142 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v141 <= 0 )
                v142 = 0x8000000000000000LL;
            }
            v143 = sub_3065900(a1, 0x28u, v341[0], *(_QWORD *)(**(_QWORD **)(v3 + 72) + 8LL), 0, a3, 0);
            v39 = __OFADD__(v143, v142);
            v144 = v143 + v142;
            if ( v39 )
            {
              v144 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v143 <= 0 )
                v144 = 0x8000000000000000LL;
            }
            v145 = sub_3075ED0(a1, 28, v341[0], a3, 0, 0, 0, 0, 0);
            v39 = __OFADD__(v145, v144);
            v146 = v145 + v144;
            if ( v39 )
            {
              v146 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v145 <= 0 )
                v146 = 0x8000000000000000LL;
            }
            sub_DF8CB0((__int64)&v356, 399, (__int64)v286, (char *)v341, 1, v310, v317, 1u);
            v147 = sub_306D850(a1, &v356, a3);
            v39 = __OFADD__(v147, v146);
            v148 = v147 + v146;
            if ( v39 )
            {
              v148 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v147 <= 0 )
                v148 = 0x8000000000000000LL;
            }
            v149 = sub_3075ED0(a1, 15, (_DWORD)v286, a3, 0, 0, 0, 0, 0);
            result = v149 + v148;
            if ( __OFADD__(v149, v148) )
            {
              result = 0x7FFFFFFFFFFFFFFFLL;
              if ( v149 <= 0 )
                result = 0x8000000000000000LL;
            }
            if ( v361 != v362 )
            {
              v326 = result;
              _libc_free((unsigned __int64)v361);
              result = v326;
            }
            if ( v359 != v360 )
            {
              v327 = result;
              _libc_free((unsigned __int64)v359);
              result = v327;
            }
            if ( v354 != &v355 )
            {
              v328 = result;
              _libc_free((unsigned __int64)v354);
              result = v328;
            }
            if ( v352 != &v353 )
            {
              v329 = result;
              _libc_free((unsigned __int64)v352);
              result = v329;
            }
            if ( v349 > 0x40 && v348 )
            {
              v330 = result;
              j_j___libc_free_0_0(v348);
              result = v330;
            }
            if ( v347 > 0x40 )
            {
              v150 = v346;
              if ( v346 )
                goto LABEL_168;
            }
            return result;
          case 0xA2u:
          case 0xB9u:
            return sub_306D850(a1, a2, a3);
          case 0xA7u:
LABEL_132:
            v117 = *(_QWORD *)(a2 + 72);
            v118 = 1;
            if ( **(_BYTE **)(v117 + 16) <= 0x15u )
              v118 = **(_BYTE **)(v117 + 24) > 0x15u;
            v119 = *(_QWORD *)(v11 + 24);
            v314 = (_QWORD **)v11;
            v120 = sub_A74840((_QWORD *)(v317 + 72), 0);
            v121 = v120;
            v325 = HIBYTE(v120);
            v122 = sub_AE5020(*(_QWORD *)(a1 + 8), v119);
            if ( v325 )
              v122 = v121;
            return sub_3067E40(a1, 32, v314, v122, v118, 1, a3, 0);
          case 0xA8u:
LABEL_127:
            v111 = *(__int64 **)(a2 + 72);
            v112 = 1;
            v113 = *v111;
            if ( *(_BYTE *)v111[3] <= 0x15u )
              v112 = *(_BYTE *)v111[4] > 0x15u;
            v313 = *(_QWORD *)(*(_QWORD *)(v113 + 8) + 24LL);
            v114 = sub_A74840((_QWORD *)(v317 + 72), 1);
            v115 = v114;
            v324 = HIBYTE(v114);
            v116 = sub_AE5020(*(_QWORD *)(a1 + 8), v313);
            if ( v324 )
              v116 = v115;
            return sub_3067E40(a1, 33, *(_QWORD ***)(v113 + 8), v116, v112, 1, a3, 0);
          case 0xB4u:
          case 0xB5u:
LABEL_60:
            v33 = *(unsigned __int8 ***)(a2 + 72);
            v280 = v11;
            v34 = v33[1];
            v35 = v33[2];
            v321 = *v33;
            v311 = v34;
            v301 = sub_DFB770(*v33);
            v293 = sub_DFB770(v34);
            v36 = sub_DFB770(v35);
            v37 = sub_3075ED0(a1, 29, v280, a3, 0, 0, 0, 0, 0);
            v38 = sub_3075ED0(a1, 15, v280, a3, 0, 0, 0, 0, 0);
            v39 = __OFADD__(v38, v37);
            v40 = v38 + v37;
            if ( v39 )
            {
              v40 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v38 <= 0 )
                v40 = 0x8000000000000000LL;
            }
            v41 = sub_3075ED0(a1, 25, v280, a3, v301, v36, 0, 0, 0);
            v39 = __OFADD__(v41, v40);
            v42 = v41 + v40;
            if ( v39 )
            {
              v42 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v41 <= 0 )
                v42 = 0x8000000000000000LL;
            }
            v43 = sub_3075ED0(a1, 26, v280, a3, v293, v36, 0, 0, 0);
            v44 = (__int64 *)v280;
            v39 = __OFADD__(v43, v42);
            v45 = v43 + v42;
            if ( v39 )
            {
              v45 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v43 <= 0 )
                v45 = 0x8000000000000000LL;
            }
            if ( (unsigned int)(v36 - 2) > 1 )
            {
              v46 = sub_BCB060(v280);
              v47 = 22;
              if ( v46 )
                v47 = ((v46 - 1) & v46) == 0 ? 28 : 22;
              v48 = sub_3075ED0(a1, v47, v280, a3, v36, 2, 0, 0, 0);
              v44 = (__int64 *)v280;
              v39 = __OFADD__(v48, v45);
              v45 += v48;
              if ( v39 )
              {
                v45 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v48 <= 0 )
                  v45 = 0x8000000000000000LL;
              }
            }
            if ( v321 != v311 )
            {
              v322 = v44;
              v49 = sub_BCD140((_QWORD *)*v44, 1u);
              v50 = v322;
              v51 = (__int64 *)v49;
              v52 = *((unsigned __int8 *)v322 + 8);
              if ( (unsigned int)(v52 - 17) <= 1 )
              {
                v53 = *((_DWORD *)v322 + 8);
                BYTE4(v356) = (_BYTE)v52 == 18;
                LODWORD(v356) = v53;
                v54 = sub_BCE1B0(v51, v356);
                v50 = v322;
                v51 = (__int64 *)v54;
              }
              v323 = v50;
              v55 = sub_3066CD0(a1, 53, v50, (__int64)v51, 32, a3, 0, 0, 0);
              v39 = __OFADD__(v55, v45);
              v56 = v55 + v45;
              if ( v39 )
              {
                v56 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v55 <= 0 )
                  v56 = 0x8000000000000000LL;
              }
              v57 = sub_3066CD0(a1, 57, v323, (__int64)v51, 32, a3, 0, 0, 0);
              v39 = __OFADD__(v57, v56);
              v45 = v57 + v56;
              if ( v39 )
              {
                v45 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v57 <= 0 )
                  return 0x8000000000000000LL;
              }
            }
            return v45;
          default:
            goto LABEL_223;
        }
      }
      if ( v6 == 65 )
      {
        v20 = v12 != 18;
        if ( v83 == 1 && v12 != 18 )
          goto LABEL_39;
LABEL_104:
        v309 = 0;
        if ( v83 <= 1 || !v20 )
          goto LABEL_41;
        v99 = *(_BYTE *)(v11 + 8);
        if ( v99 != 7 )
        {
          if ( v99 != 15 )
          {
            v100 = &v340;
            v282 = v341;
LABEL_109:
            v265 = v6;
            v262 = v3;
            v101 = 0;
            v102 = v100;
            v103 = 0;
            v269 = v11;
            do
            {
              v104 = *v102;
              if ( *(_BYTE *)(*v102 + 8) == 18 )
              {
                v103 = 1;
              }
              else
              {
                v105 = *(_DWORD *)(v104 + 32);
                LODWORD(v357) = v105;
                if ( v105 > 0x40 )
                {
                  v277 = v104;
                  sub_C43690((__int64)&v356, -1, 1);
                  v104 = v277;
                }
                else
                {
                  v106 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v105;
                  v61 = v105 == 0;
                  v107 = 0;
                  if ( !v61 )
                    v107 = v106;
                  v356 = v107;
                }
                v108 = sub_3064F80(a1, v104, &v356, 1, 0);
                if ( (unsigned int)v357 > 0x40 && v356 )
                {
                  v268 = v109;
                  v274 = v108;
                  j_j___libc_free_0_0(v356);
                  v109 = v268;
                  v108 = v274;
                }
                if ( v109 == 1 )
                  v103 = 1;
                v39 = __OFADD__(v108, v101);
                v101 += v108;
                if ( v39 )
                {
                  v101 = 0x8000000000000000LL;
                  if ( v108 > 0 )
                    v101 = 0x7FFFFFFFFFFFFFFFLL;
                }
              }
              ++v102;
            }
            while ( v282 != v102 );
            v201 = v101;
            v11 = v269;
            v6 = v265;
            v3 = v262;
            goto LABEL_260;
          }
          v100 = *(__int64 **)(v11 + 16);
          v282 = &v100[*(unsigned int *)(v11 + 12)];
          if ( v282 != v100 )
            goto LABEL_109;
        }
        v103 = 0;
        v201 = 0;
LABEL_260:
        v278 = v201;
        v287 = v11;
        v202 = sub_3065600(
                 a1,
                 *(_BYTE ***)(v3 + 72),
                 *(_DWORD *)(v3 + 80),
                 *(__int64 **)(v3 + 24),
                 *(unsigned int *)(v3 + 32),
                 a3);
        v11 = v287;
        v203 = v202;
        v21 = v204;
        if ( v204 != 1 )
          v21 = v103;
        v22 = v203 + v278;
        if ( __OFADD__(v203, v278) )
        {
          v22 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v203 <= 0 )
            v22 = 0x8000000000000000LL;
        }
        goto LABEL_42;
      }
      if ( v6 == 67 )
      {
        v20 = v12 != 18;
        if ( v83 == 1 && v12 != 18 )
          goto LABEL_34;
        goto LABEL_104;
      }
    }
LABEL_223:
    v20 = v12 != 18;
    goto LABEL_104;
  }
  switch ( v6 )
  {
    case 0x11Du:
LABEL_188:
      v162 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 8LL);
      if ( *(_BYTE *)v162 != 17 )
        goto LABEL_103;
      v275 = v83;
      v296 = v11;
      v306 = *(_QWORD *)(*(_QWORD *)(v317 + 40) + 72LL);
      v163 = sub_B2D610(v306, 47);
      v11 = v296;
      v164 = (const void **)(v162 + 24);
      v83 = v275;
      if ( v163 )
      {
        v165 = *(_DWORD *)(v162 + 32);
        v166 = *(__int64 **)(v162 + 24);
        if ( v165 <= 0x40 )
        {
          if ( !v165 )
            goto LABEL_103;
          goto LABEL_192;
        }
      }
      else
      {
        v210 = sub_B2D610(v306, 18);
        v165 = *(_DWORD *)(v162 + 32);
        v166 = *(__int64 **)(v162 + 24);
        v11 = v296;
        v164 = (const void **)(v162 + 24);
        v83 = v275;
        if ( v165 <= 0x40 )
        {
          if ( v165 )
          {
            if ( v210 )
            {
LABEL_192:
              v167 = abs64((__int64)((_QWORD)v166 << (64 - (unsigned __int8)v165)) >> (64 - (unsigned __int8)v165));
              goto LABEL_193;
            }
          }
          else if ( v210 )
          {
            goto LABEL_103;
          }
          v173 = *(_QWORD *)(v162 + 24);
          if ( _bittest64(&v173, v165 - 1) )
            goto LABEL_198;
          goto LABEL_288;
        }
        if ( !v210 )
        {
          v171 = v165 - 1;
          v172 = 1LL << ((unsigned __int8)v165 - 1);
          goto LABEL_281;
        }
      }
      v167 = abs64(*v166);
LABEL_193:
      v263 = v165;
      v270 = v83;
      v276 = v164;
      v297 = v11;
      v307 = v167;
      v168 = sub_39FAC40(v167);
      v11 = v297;
      v164 = v276;
      v83 = v270;
      v165 = v263;
      if ( v307 )
      {
        _BitScanReverse64(&v169, v307);
        v170 = v169 ^ 0x3F;
      }
      else
      {
        v170 = 64;
      }
      if ( (unsigned int)(v168 - v170 + 63) > 6 )
        goto LABEL_103;
      v171 = v263 - 1;
      v172 = 1LL << ((unsigned __int8)v263 - 1);
      if ( v263 <= 0x40 )
      {
        v173 = *(_QWORD *)(v162 + 24);
        if ( (v173 & v172) != 0 )
        {
LABEL_198:
          LODWORD(v357) = v165;
          v356 = v173;
          goto LABEL_199;
        }
LABEL_288:
        LODWORD(v351) = v165;
        v350 = *(_QWORD *)(v162 + 24);
        goto LABEL_289;
      }
LABEL_281:
      if ( (v166[v171 >> 6] & v172) == 0 )
      {
        v336 = v11;
        LODWORD(v351) = v165;
        sub_C43780((__int64)&v350, v164);
        v175 = v351;
        LODWORD(v11) = v336;
        goto LABEL_204;
      }
      v339 = v11;
      LODWORD(v357) = v165;
      sub_C43780((__int64)&v356, v164);
      LODWORD(v11) = v339;
LABEL_199:
      if ( (unsigned int)v357 > 0x40 )
      {
        v338 = v11;
        sub_C43D10((__int64)&v356);
        LODWORD(v11) = v338;
      }
      else
      {
        v174 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v357;
        if ( !(_DWORD)v357 )
          v174 = 0;
        v356 = v174 & ~v356;
      }
      v332 = v11;
      sub_C46250((__int64)&v356);
      v175 = v357;
      LODWORD(v11) = v332;
      LODWORD(v351) = v357;
      v350 = v356;
LABEL_204:
      if ( v175 > 0x40 )
      {
        v333 = v11;
        v176 = v175 - sub_C444A0((__int64)&v350);
        v177 = sub_C44630((__int64)&v350);
        v178 = v333;
        v179 = v177;
        goto LABEL_206;
      }
LABEL_289:
      v211 = 64;
      if ( v350 )
      {
        _BitScanReverse64(&v212, v350);
        v211 = v212 ^ 0x3F;
      }
      v337 = v11;
      v176 = 64 - v211;
      v213 = sub_39FAC40(v350);
      v178 = v337;
      v179 = v213;
LABEL_206:
      v334 = v178;
      v180 = sub_3075ED0(a1, 18, v178, a3, 0, 0, 0, 0, 0);
      v181 = (unsigned int)(v176 + v179 - 2);
      v182 = v180;
      v184 = v180;
      v183 = v180 * v181;
      if ( is_mul_ok(v184, v181) )
      {
        v185 = v183;
      }
      else if ( v182 <= 0 || (v185 = 0x7FFFFFFFFFFFFFFFLL, !v181) )
      {
        v185 = 0x8000000000000000LL;
      }
      v186 = *(_DWORD *)(v162 + 32);
      v187 = *(_QWORD *)(v162 + 24);
      if ( v186 > 0x40 )
        v187 = *(_QWORD *)(v187 + 8LL * ((v186 - 1) >> 6));
      if ( (v187 & (1LL << ((unsigned __int8)v186 - 1))) != 0 )
      {
        v224 = sub_3075ED0(a1, 21, v334, a3, 0, 0, 0, 0, 0);
        if ( __OFADD__(v224, v185) )
        {
          v185 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v224 <= 0 )
            v185 = 0x8000000000000000LL;
        }
        else
        {
          v185 += v224;
        }
      }
      result = v185;
      if ( (unsigned int)v351 > 0x40 )
      {
        v150 = v350;
        if ( v350 )
        {
LABEL_168:
          v331 = result;
          j_j___libc_free_0_0(v150);
          return v331;
        }
      }
      return result;
    case 0x146u:
    case 0x147u:
LABEL_93:
      v84 = (__int64 *)v11;
      if ( v82 == 15 )
        v84 = **(__int64 ***)(v11 + 16);
      v85 = *(_QWORD *)(a1 + 8);
      v294 = v83;
      v302 = v11;
      v86 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), v85, v84, 0);
      v87 = v302;
      v88 = v294;
      LODWORD(v356) = v86;
      v89 = *(_DWORD *)(v3 + 16);
      v357 = v90;
      switch ( v89 )
      {
        case 326:
          v91 = sub_2FE5F00;
LABEL_98:
          v92 = v356;
          if ( (_WORD)v356 )
          {
            if ( (unsigned __int16)(v356 - 17) <= 0xD3u )
            {
              v93 = 0;
              v92 = word_4456580[(unsigned __int16)v356 - 1];
              goto LABEL_101;
            }
          }
          else
          {
            v205 = sub_30070B0((__int64)&v356);
            v87 = v302;
            v88 = v294;
            v92 = 0;
            if ( v205 )
            {
              v208 = sub_3009970((__int64)&v356, v85, 0, v206, v207);
              v87 = v302;
              v88 = v294;
              v93 = v209;
              v92 = v208;
              goto LABEL_101;
            }
          }
          v93 = v357;
LABEL_101:
          v295 = v88;
          v303 = v87;
          v94 = v91(v92, v93);
          v95 = *(__int64 **)(v3 + 144);
          v96 = *(_QWORD *)(v3 + 8);
          v97 = v94;
          v98 = *(_DWORD *)(v3 + 16);
          v11 = v303;
          v346 = v96;
          v83 = v295;
          v273 = v98;
          if ( !v95
            || *(_BYTE *)(v96 + 8) != 15
            || (v266 = v295, v271 = v95, v298 = v97, v225 = sub_E456C0(v96), v11 = v303, v83 = v266, !v225)
            || (v226 = *(char **)(*(_QWORD *)(a1 + 24) + 8LL * v298 + 525288)) == 0 )
          {
LABEL_103:
            v20 = !v281;
            goto LABEL_104;
          }
          v227 = *(_QWORD **)v96;
          v350 = v96;
          v228 = v271;
          v264 = v227;
          if ( *(_BYTE *)(v96 + 8) == 15 )
            v229 = *(__int64 **)(v96 + 16);
          else
            v229 = &v350;
          v230 = *v229;
          v299 = 0;
          v231 = *(_DWORD *)(v230 + 32);
          v61 = *(_BYTE *)(v230 + 8) == 18;
          LOWORD(v344[0]) = 256;
          LODWORD(v350) = v231;
          v272 = a1;
          v261 = v3;
          v232 = v266;
          v267 = a3;
          v233 = v96;
          v234 = v228;
          BYTE4(v350) = v61;
          while ( 1 )
          {
            v260 = *((_BYTE *)v344 + v299);
            v235 = strlen(v226);
            v236 = sub_97F930(*v234, v226, v235, (__int64)&v350, v260);
            if ( v236 )
              break;
            if ( v299 == 1 )
            {
              v83 = v232;
              v11 = v303;
              v3 = v261;
              a3 = v267;
              goto LABEL_103;
            }
            v299 = 1;
          }
          if ( *(_BYTE *)(v236 + 40) )
          {
            v251 = (__int64 *)sub_BCB2A0(v264);
            v252 = sub_BCE1B0(v251, v350);
            v253 = sub_30690B0(a1, 0, v252, 0, 0, v267, 0, 0);
            v242 = v253 + 10;
            if ( __OFADD__(10, v253) )
            {
              v242 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v253 <= 0 )
                v242 = 0x8000000000000000LL;
            }
          }
          else
          {
            v242 = 10;
          }
          if ( *(_BYTE *)(v233 + 8) == 15 )
          {
            v243 = *(unsigned __int64 **)(v233 + 16);
            v244 = (unsigned int *)&v243[*(unsigned int *)(v233 + 12)];
            if ( v244 == (unsigned int *)v243 )
              return v242;
          }
          else
          {
            v244 = &v347;
            v243 = &v346;
          }
          v245 = v259;
          v246 = 0;
          v247 = (__int64 *)v243;
          v248 = v272;
          do
          {
            if ( v246 || v273 != 249 )
            {
              v300 = v244;
              v308 = v248;
              LOBYTE(v245) = sub_AE5020(*(_QWORD *)(v248 + 8), *v247);
              v249 = v245;
              BYTE1(v249) = 1;
              v245 = v249;
              v250 = sub_30670E0(v308, 32, *v247, v249, 0, v267);
              v248 = v308;
              v244 = v300;
              v39 = __OFADD__(v250, v242);
              v242 += v250;
              if ( v39 )
              {
                v242 = 0x8000000000000000LL;
                if ( v250 > 0 )
                  v242 = 0x7FFFFFFFFFFFFFFFLL;
              }
            }
            ++v247;
            ++v246;
          }
          while ( v247 != (__int64 *)v244 );
          return v242;
        case 327:
          v91 = sub_2FE5F30;
          goto LABEL_98;
        case 249:
          v91 = sub_2FE5F60;
          goto LABEL_98;
      }
      break;
    case 0x159u:
LABEL_183:
      if ( v82 != 18 )
        return 1;
      v161 = *(_DWORD *)(a2 + 16);
      if ( v161 <= 0xD3 )
      {
        if ( v161 <= 0x94 )
        {
          switch ( v161 )
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
        switch ( v161 )
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
            goto LABEL_350;
          default:
            return 1;
        }
      }
      goto LABEL_266;
    case 0x17Du:
LABEL_178:
      if ( v82 != 18 )
      {
        v158 = *(_QWORD **)(a2 + 72);
        v159 = v158[1];
        v160 = *(_QWORD **)(v159 + 24);
        if ( *(_DWORD *)(v159 + 32) > 0x40u )
          v160 = (_QWORD *)*v160;
        return sub_30690B0(a1, 5, *(_QWORD *)(*v158 + 8LL), 0, 0, a3, (signed int)v160, v11);
      }
      v161 = *(_DWORD *)(a2 + 16);
      if ( v161 <= 0xD3 )
      {
        if ( v161 <= 0x94 )
        {
          switch ( v161 )
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
        switch ( v161 )
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
            goto LABEL_350;
          default:
            return 1;
        }
      }
      goto LABEL_266;
    case 0x17Eu:
LABEL_169:
      v151 = *(_QWORD **)(a2 + 72);
      v152 = *(_QWORD *)(v151[1] + 8LL);
      if ( *(_BYTE *)(v152 + 8) == 18 )
      {
        v161 = *(_DWORD *)(a2 + 16);
        if ( v161 > 0xD3 )
        {
LABEL_266:
          if ( v161 > 0x178 )
          {
            return 1;
          }
          else if ( v161 > 0x143 )
          {
            return ((1LL << ((unsigned __int8)v161 - 68)) & 0x10000020401001LL) == 0;
          }
          else
          {
            return v161 != 282 && v161 - 291 >= 2;
          }
        }
        else
        {
          if ( v161 <= 0x94 )
          {
            switch ( v161 )
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
          switch ( v161 )
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
LABEL_350:
              result = 0;
              break;
            default:
              return 1;
          }
        }
      }
      else
      {
        v153 = v151[2];
        v154 = *(_QWORD **)(v153 + 24);
        if ( *(_DWORD *)(v153 + 32) > 0x40u )
          v154 = (_QWORD *)*v154;
        return sub_30690B0(a1, 4, *(_QWORD *)(*v151 + 8LL), 0, 0, a3, (signed int)v154, v152);
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
LABEL_58:
      v350 = *(_QWORD *)(**(_QWORD **)(a2 + 72) + 8LL);
      sub_DF8CB0((__int64)&v356, v6, v11, (char *)&v350, 1, v310, v317, 1u);
      goto LABEL_59;
    case 0x185u:
    case 0x18Au:
LABEL_126:
      v110 = *(_QWORD **)(a2 + 72);
      v350 = *(_QWORD *)(*v110 + 8LL);
      v351 = *(_QWORD *)(v110[1] + 8LL);
      sub_DF8CB0((__int64)&v356, v6, v11, (char *)&v350, 2, v310, v317, 1u);
LABEL_59:
      v23 = a3;
      goto LABEL_43;
    case 0x192u:
      return sub_30690B0(a1, 1, *(_QWORD *)(**(_QWORD **)(a2 + 72) + 8LL), 0, 0, a3, 0, v11);
    case 0x193u:
LABEL_173:
      v155 = *(_QWORD **)(a2 + 72);
      v156 = v155[2];
      v157 = *(_QWORD **)(v156 + 24);
      if ( *(_DWORD *)(v156 + 32) > 0x40u )
        v157 = (_QWORD *)*v157;
      return sub_30690B0(a1, 8, *(_QWORD *)(*v155 + 8LL), 0, 0, a3, (signed int)v157, v11);
    default:
      goto LABEL_223;
  }
LABEL_401:
  BUG();
}
