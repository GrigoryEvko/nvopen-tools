// Function: sub_32F4AC0
// Address: 0x32f4ac0
//
__int64 __fastcall sub_32F4AC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v8; // rax
  __int64 v9; // rcx
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  __int64 v18; // rsi
  unsigned int v19; // r12d
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r12
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r11
  __int64 v27; // r10
  __int64 v28; // r12
  __int64 v29; // rdx
  __int16 v30; // ax
  __int64 v31; // rdx
  int v32; // esi
  bool v33; // al
  __int64 v34; // rax
  __int64 v35; // rsi
  __m128i v36; // xmm2
  __int64 v37; // rcx
  __int128 v38; // xmm3
  __m128i v39; // xmm4
  __int64 v40; // rax
  __int16 *v41; // rax
  __int16 v42; // cx
  unsigned __int16 *v43; // rax
  __int64 v44; // r8
  __int64 v45; // rax
  int v46; // eax
  int v47; // edx
  bool v48; // al
  __int64 v49; // rax
  bool v50; // al
  __int64 v51; // rax
  bool v52; // al
  bool v53; // al
  bool v54; // al
  __int64 v55; // rsi
  unsigned int v56; // edx
  __int64 v57; // rdi
  int v58; // eax
  unsigned int v60; // eax
  int v61; // r9d
  int v62; // edx
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rax
  bool v69; // al
  __int64 v70; // rax
  bool v71; // al
  __int128 v72; // rax
  int v73; // r9d
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  const __m128i *v77; // rax
  __int64 v78; // rsi
  __int64 v79; // r12
  __int64 v80; // rcx
  __int64 v81; // rdi
  int v82; // r13d
  __int64 (__fastcall *v83)(__int64, unsigned int); // rax
  char (__fastcall *v84)(__int64, unsigned int); // rax
  __int64 v85; // r12
  _DWORD *v86; // rbx
  __int64 *v87; // rax
  char v88; // al
  __int64 v89; // r11
  __int64 v90; // r10
  int v91; // r9d
  bool v92; // al
  int v93; // r10d
  int v94; // r9d
  char v95; // al
  __int64 v96; // rcx
  __int64 v97; // rax
  bool v98; // al
  __int64 v99; // r13
  unsigned __int16 v100; // r14
  __int64 v101; // r13
  unsigned __int16 v102; // ax
  int v103; // eax
  unsigned __int16 *v104; // rax
  __int64 v105; // r9
  int v106; // r15d
  __int128 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  int v112; // r12d
  __int64 *v113; // rax
  __int64 v114; // r10
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r11
  __int64 v118; // rsi
  __int64 v119; // rdi
  __int64 v120; // rcx
  __int64 v121; // r8
  __int64 v122; // rdx
  __int64 v123; // r9
  unsigned __int16 v124; // ax
  __int64 v125; // rcx
  __int128 v126; // rax
  int v127; // r9d
  __int64 v128; // r15
  __int64 v129; // rax
  const __m128i *v130; // rax
  __int128 v131; // rax
  __int64 v132; // rcx
  __int64 v133; // r8
  __int64 v134; // r9
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  __int64 v139; // rsi
  __int64 v140; // rax
  __int64 v141; // rax
  const __m128i *v142; // rdx
  __int64 v143; // rsi
  __int64 v144; // rdi
  __int64 v145; // rax
  __int64 v146; // rcx
  unsigned __int16 v147; // dx
  __int64 v148; // r8
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // r11
  unsigned __int16 *v152; // rdx
  __int64 v153; // rax
  __int64 v154; // rdx
  __int64 v155; // rsi
  __int64 v156; // rdi
  __int64 v157; // rax
  __int64 v158; // rcx
  unsigned __int16 v159; // dx
  __int64 v160; // r8
  __int64 v161; // rax
  __int64 v162; // rdx
  __int64 v163; // r11
  unsigned __int16 *v164; // rdx
  int v165; // r13d
  __int64 v166; // rax
  char v167; // r15
  __int64 v168; // rax
  __int64 v169; // rdx
  __int64 v170; // rax
  __int64 v171; // rdx
  unsigned int v172; // eax
  __int64 v173; // r10
  __int64 v174; // rax
  unsigned int v175; // edx
  int v176; // eax
  bool v177; // al
  __int64 v178; // rax
  unsigned int v179; // edx
  int v180; // eax
  __int64 v181; // rax
  __int64 v182; // rdx
  unsigned int v183; // edx
  int v184; // r9d
  int v185; // eax
  int v186; // edx
  int v187; // edx
  int v188; // r9d
  int v189; // edx
  int v190; // eax
  __int128 v191; // rax
  int v192; // r9d
  __int64 v193; // rdx
  __int64 v194; // rsi
  unsigned __int16 *v195; // rax
  __int64 v196; // rax
  char v197; // al
  __int64 v198; // rax
  __int64 v199; // rcx
  int v200; // edx
  int v201; // eax
  __int64 v202; // rsi
  __int64 v203; // rdx
  __int64 v204; // rcx
  __int64 v205; // r8
  unsigned int v206; // eax
  int v207; // ecx
  int v208; // edx
  int v209; // r9d
  const __m128i *v210; // rax
  __int128 v211; // rax
  __int64 v212; // rcx
  __int64 v213; // r8
  __int64 v214; // r9
  __int64 v215; // rdx
  __int64 v216; // rcx
  __int64 v217; // r8
  __int64 v218; // r9
  __int64 v219; // rsi
  __int64 v220; // rdx
  __int64 v221; // rax
  int v222; // r9d
  int v223; // r10d
  bool v224; // al
  __int64 v225; // r13
  __int64 v226; // rax
  __int64 v227; // rdx
  __int64 v228; // rax
  __int64 v229; // rdx
  int v230; // r9d
  int v231; // r10d
  bool v232; // al
  __int64 v233; // rax
  unsigned int v234; // edx
  __int64 v235; // r12
  __int64 v236; // rdx
  __int64 v237; // rsi
  unsigned __int16 *v238; // rax
  __int64 v239; // r8
  __int64 v240; // r14
  __int64 v241; // rsi
  __int64 v242; // r13
  int v243; // r15d
  __int64 v244; // rax
  __int64 v245; // rdx
  __int64 v246; // r13
  __int64 v247; // r8
  __int64 v248; // r9
  __int64 v249; // rsi
  const __m128i *v250; // r14
  __int64 v251; // rax
  unsigned __int16 v252; // r15
  __int64 v253; // rbx
  int v254; // ecx
  __m128i v255; // xmm5
  __int64 v256; // r15
  __int64 v257; // r12
  __int64 v258; // r9
  __int64 v259; // r14
  __int64 v260; // rdx
  __int64 v261; // rdx
  __int16 v262; // ax
  __int64 v263; // rdx
  int v264; // esi
  __int64 v265; // rbx
  bool v266; // al
  __int64 v267; // rax
  __int128 v268; // rax
  __int128 v269; // [rsp-30h] [rbp-260h]
  __int128 v270; // [rsp-20h] [rbp-250h]
  __int128 v271; // [rsp-20h] [rbp-250h]
  __int128 v272; // [rsp-20h] [rbp-250h]
  __int128 v273; // [rsp-20h] [rbp-250h]
  __int128 v274; // [rsp-20h] [rbp-250h]
  __int128 v275; // [rsp-20h] [rbp-250h]
  __int64 v276; // [rsp-20h] [rbp-250h]
  __int128 v277; // [rsp-20h] [rbp-250h]
  __int128 v278; // [rsp-10h] [rbp-240h]
  __int128 v279; // [rsp-10h] [rbp-240h]
  __int128 v280; // [rsp-10h] [rbp-240h]
  __int128 v281; // [rsp-10h] [rbp-240h]
  __int128 v282; // [rsp-10h] [rbp-240h]
  __int128 v283; // [rsp-10h] [rbp-240h]
  __int128 v284; // [rsp-10h] [rbp-240h]
  __int128 v285; // [rsp-10h] [rbp-240h]
  __int128 v286; // [rsp-10h] [rbp-240h]
  __int128 v287; // [rsp-10h] [rbp-240h]
  unsigned __int64 v288; // [rsp+8h] [rbp-228h]
  bool v289; // [rsp+10h] [rbp-220h]
  unsigned __int64 v290; // [rsp+10h] [rbp-220h]
  __int64 v291; // [rsp+18h] [rbp-218h]
  int v292; // [rsp+20h] [rbp-210h]
  const void **v293; // [rsp+20h] [rbp-210h]
  int v294; // [rsp+20h] [rbp-210h]
  __int64 v295; // [rsp+28h] [rbp-208h]
  unsigned int v296; // [rsp+28h] [rbp-208h]
  __int64 v297; // [rsp+28h] [rbp-208h]
  __int64 v298; // [rsp+28h] [rbp-208h]
  unsigned int v299; // [rsp+30h] [rbp-200h]
  __int64 v300; // [rsp+30h] [rbp-200h]
  const void **v301; // [rsp+30h] [rbp-200h]
  int v302; // [rsp+30h] [rbp-200h]
  __int64 v303; // [rsp+30h] [rbp-200h]
  __int16 v304; // [rsp+38h] [rbp-1F8h]
  unsigned int v305; // [rsp+38h] [rbp-1F8h]
  unsigned int v306; // [rsp+38h] [rbp-1F8h]
  bool v307; // [rsp+38h] [rbp-1F8h]
  unsigned int v308; // [rsp+38h] [rbp-1F8h]
  __int64 v309; // [rsp+38h] [rbp-1F8h]
  __int64 v310; // [rsp+38h] [rbp-1F8h]
  int v311; // [rsp+40h] [rbp-1F0h]
  __int128 v312; // [rsp+50h] [rbp-1E0h]
  __int64 v313; // [rsp+50h] [rbp-1E0h]
  __m128i v314; // [rsp+50h] [rbp-1E0h]
  __m128i v315; // [rsp+50h] [rbp-1E0h]
  __int64 v316; // [rsp+60h] [rbp-1D0h]
  int v317; // [rsp+60h] [rbp-1D0h]
  __int64 v318; // [rsp+60h] [rbp-1D0h]
  __int64 v319; // [rsp+60h] [rbp-1D0h]
  __m128i v320; // [rsp+60h] [rbp-1D0h]
  __int128 v321; // [rsp+60h] [rbp-1D0h]
  __int128 v322; // [rsp+60h] [rbp-1D0h]
  __int128 v323; // [rsp+60h] [rbp-1D0h]
  __int64 v324; // [rsp+60h] [rbp-1D0h]
  __int128 v325; // [rsp+60h] [rbp-1D0h]
  __int64 v326; // [rsp+68h] [rbp-1C8h]
  __int64 v327; // [rsp+70h] [rbp-1C0h]
  int v328; // [rsp+70h] [rbp-1C0h]
  __int64 v329; // [rsp+70h] [rbp-1C0h]
  __int64 v330; // [rsp+70h] [rbp-1C0h]
  int v331; // [rsp+70h] [rbp-1C0h]
  __int64 v332; // [rsp+70h] [rbp-1C0h]
  __int16 v333; // [rsp+80h] [rbp-1B0h]
  char v334; // [rsp+80h] [rbp-1B0h]
  __m128i v335; // [rsp+80h] [rbp-1B0h]
  __int64 v336; // [rsp+90h] [rbp-1A0h]
  int v337; // [rsp+90h] [rbp-1A0h]
  __int64 v338; // [rsp+90h] [rbp-1A0h]
  unsigned int v339; // [rsp+A0h] [rbp-190h]
  __m128i v340; // [rsp+A0h] [rbp-190h]
  __int64 v341; // [rsp+A0h] [rbp-190h]
  __int64 v342; // [rsp+A0h] [rbp-190h]
  int v343; // [rsp+A0h] [rbp-190h]
  __int64 v344; // [rsp+B0h] [rbp-180h]
  unsigned int v345; // [rsp+B8h] [rbp-178h]
  unsigned int v346; // [rsp+BCh] [rbp-174h]
  __int64 v347; // [rsp+C0h] [rbp-170h]
  __int64 v348; // [rsp+C0h] [rbp-170h]
  __int64 v349; // [rsp+C8h] [rbp-168h]
  __int64 v350; // [rsp+D0h] [rbp-160h]
  __int64 v351; // [rsp+D0h] [rbp-160h]
  __int64 v352; // [rsp+D0h] [rbp-160h]
  __int128 v353; // [rsp+D0h] [rbp-160h]
  __int64 v354; // [rsp+D8h] [rbp-158h]
  __int32 v355; // [rsp+E0h] [rbp-150h]
  int v356; // [rsp+E0h] [rbp-150h]
  unsigned int v357; // [rsp+E0h] [rbp-150h]
  int v358; // [rsp+E0h] [rbp-150h]
  int v359; // [rsp+E8h] [rbp-148h]
  __int64 v360; // [rsp+E8h] [rbp-148h]
  __int64 v361; // [rsp+E8h] [rbp-148h]
  __int64 v362; // [rsp+E8h] [rbp-148h]
  __int64 v363; // [rsp+E8h] [rbp-148h]
  __int64 v364; // [rsp+E8h] [rbp-148h]
  __int64 v365; // [rsp+E8h] [rbp-148h]
  __int64 v366; // [rsp+F0h] [rbp-140h]
  __int64 v367; // [rsp+F0h] [rbp-140h]
  int v368; // [rsp+F0h] [rbp-140h]
  int v369; // [rsp+F0h] [rbp-140h]
  __int64 v370; // [rsp+100h] [rbp-130h]
  __m128i v371; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v372; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v373; // [rsp+188h] [rbp-A8h]
  __int64 v374; // [rsp+190h] [rbp-A0h] BYREF
  int v375; // [rsp+198h] [rbp-98h]
  __int128 v376; // [rsp+1A0h] [rbp-90h] BYREF
  unsigned int v377; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v378; // [rsp+1B8h] [rbp-78h]
  unsigned int v379; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v380; // [rsp+1C8h] [rbp-68h]
  __int64 v381; // [rsp+1D0h] [rbp-60h] BYREF
  int v382; // [rsp+1D8h] [rbp-58h]
  __int64 v383; // [rsp+1E0h] [rbp-50h] BYREF
  unsigned int v384; // [rsp+1E8h] [rbp-48h]
  __int64 v385; // [rsp+1F0h] [rbp-40h] BYREF
  __int64 v386; // [rsp+1F8h] [rbp-38h]

  v8 = *(__int64 **)(a2 + 40);
  v9 = *v8;
  v10 = _mm_loadu_si128((const __m128i *)v8);
  v11 = _mm_loadu_si128((const __m128i *)(v8 + 5));
  v12 = v8[10];
  v347 = v8[5];
  v13 = v8[11];
  v14 = *((_DWORD *)v8 + 12);
  LODWORD(v8) = *((_DWORD *)v8 + 22);
  v371 = v10;
  v346 = v14;
  v339 = (unsigned int)v8;
  v15 = *(_QWORD *)(a2 + 48);
  v350 = v9;
  v16 = *(_QWORD *)(v15 + 8);
  LOWORD(v372) = *(_WORD *)v15;
  v373 = v16;
  v344 = 16LL * v10.m128i_u32[2];
  v17 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + v344);
  v18 = *(_QWORD *)(a2 + 80);
  v333 = *v17;
  v19 = *v17;
  v336 = *((_QWORD *)v17 + 1);
  v374 = v18;
  if ( v18 )
    sub_B96E90((__int64)&v374, v18, 1);
  v375 = *(_DWORD *)(a2 + 72);
  v359 = *(_DWORD *)(a2 + 28);
  v20 = sub_33E28A0(*a1, v371.m128i_i32[0], v371.m128i_i32[2], v11.m128i_i32[0], v11.m128i_i32[2], a6, v12, v13);
  if ( v20 )
    goto LABEL_4;
  v20 = sub_326AD10(*(const __m128i **)(a2 + 40), *(unsigned __int16 **)(a2 + 48), (__int64)&v374, *a1, v21);
  if ( v20 )
    goto LABEL_4;
  v24 = sub_326C2C0(v371.m128i_i64[0], v371.m128i_i64[1], *a1, 0);
  if ( v24 )
  {
    v26 = v25;
    v27 = v24;
    v28 = *a1;
    v29 = *(_QWORD *)(v24 + 48) + 16LL * (unsigned int)v25;
    v30 = *(_WORD *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    LOWORD(v385) = v30;
    v386 = v31;
    if ( v30 )
    {
      v32 = ((unsigned __int16)(v30 - 17) < 0xD4u) + 205;
    }
    else
    {
      v351 = v27;
      v354 = v26;
      v33 = sub_30070B0((__int64)&v385);
      v26 = v354;
      v27 = v351;
      v32 = 205 - (!v33 - 1);
    }
    *((_QWORD *)&v270 + 1) = v13;
    *(_QWORD *)&v270 = v12;
    v22 = sub_340EC60(v28, v32, (unsigned int)&v374, v372, v373, 0, v27, v26, v270, *(_OWORD *)&v11);
    *(_DWORD *)(v22 + 28) = v359;
    goto LABEL_5;
  }
  v34 = *(_QWORD *)(a2 + 40);
  v35 = *(_QWORD *)(a2 + 80);
  v36 = _mm_loadu_si128((const __m128i *)v34);
  v37 = *(_QWORD *)(v34 + 40);
  v38 = (__int128)_mm_loadu_si128((const __m128i *)(v34 + 40));
  v39 = _mm_loadu_si128((const __m128i *)(v34 + 80));
  v40 = *(_QWORD *)(v34 + 80);
  v376 = (__int128)v36;
  v327 = v37;
  v316 = v40;
  v41 = *(__int16 **)(a2 + 48);
  v42 = *v41;
  v312 = (__int128)v39;
  v378 = *((_QWORD *)v41 + 1);
  LOWORD(v377) = v42;
  v43 = (unsigned __int16 *)(*(_QWORD *)(v36.m128i_i64[0] + 48) + 16LL * v36.m128i_u32[2]);
  v44 = *v43;
  v45 = *((_QWORD *)v43 + 1);
  v381 = v35;
  LOWORD(v379) = v44;
  v380 = v45;
  if ( v35 )
  {
    v299 = v44;
    v304 = v42;
    sub_B96E90((__int64)&v381, v35, 1);
    v44 = v299;
    v42 = v304;
  }
  v382 = *(_DWORD *)(a2 + 72);
  if ( v42 )
  {
    if ( (unsigned __int16)(v42 - 2) > 7u
      && (unsigned __int16)(v42 - 17) > 0x6Cu
      && (unsigned __int16)(v42 - 176) > 0x1Fu )
    {
      goto LABEL_22;
    }
  }
  else
  {
    v305 = v44;
    v48 = sub_3007070((__int64)&v377);
    v44 = v305;
    if ( !v48 )
      goto LABEL_22;
  }
  v46 = *(_DWORD *)(v327 + 24);
  v47 = *(_DWORD *)(v316 + 24);
  if ( v46 != 11 && v46 != 35 || v47 != 11 && v47 != 35 )
  {
LABEL_22:
    if ( v381 )
      sub_B91220((__int64)&v381, v381);
    goto LABEL_24;
  }
  if ( (_WORD)v44 != 2 )
  {
    if ( (_WORD)v44 )
    {
      if ( (unsigned __int16)(v44 - 2) > 7u && (unsigned __int16)(v44 - 17) > 0x6Cu )
      {
        LOWORD(v44) = v44 - 176;
        if ( (unsigned __int16)v44 > 0x1Fu )
          goto LABEL_22;
      }
    }
    else if ( !sub_3007070((__int64)&v379) )
    {
      goto LABEL_22;
    }
    goto LABEL_69;
  }
  if ( *((_BYTE *)a1 + 33) )
  {
LABEL_69:
    v67 = a1[1];
    if ( *(_DWORD *)(v67 + 64) != 1 || *(_DWORD *)(v67 + 60) != 1 )
      goto LABEL_22;
    v68 = *(_QWORD *)(v327 + 96);
    if ( *(_DWORD *)(v68 + 32) <= 0x40u )
    {
      v69 = *(_QWORD *)(v68 + 24) == 0;
    }
    else
    {
      v328 = *(_DWORD *)(v68 + 32);
      v69 = v328 == (unsigned int)sub_C444A0(v68 + 24);
    }
    if ( !v69 )
      goto LABEL_22;
    v70 = *(_QWORD *)(v316 + 96);
    if ( *(_DWORD *)(v70 + 32) <= 0x40u )
    {
      v71 = *(_QWORD *)(v70 + 24) == 1;
    }
    else
    {
      v317 = *(_DWORD *)(v70 + 32);
      v71 = v317 - 1 == (unsigned int)sub_C444A0(v70 + 24);
    }
    if ( !v71 )
      goto LABEL_22;
    v318 = *a1;
    *(_QWORD *)&v72 = sub_3400BD0(*a1, 1, (unsigned int)&v381, v379, v380, 0, 0, v44);
    v74 = sub_3406EB0(v318, 188, (unsigned int)&v381, v379, v380, v73, v376, v72);
    v329 = v75;
    v319 = v74;
    if ( sub_3280240((__int64)&v377, v379, v380) )
      v65 = v319;
    else
      v65 = sub_33FB310(*a1, v319, v329, &v381, v377, v378);
    goto LABEL_64;
  }
  v49 = *(_QWORD *)(v327 + 96);
  v300 = v49;
  v295 = v49 + 24;
  v306 = *(_DWORD *)(v49 + 32);
  if ( v306 <= 0x40 )
    v50 = *(_QWORD *)(v49 + 24) == 1;
  else
    v50 = v306 - 1 == (unsigned int)sub_C444A0(v295);
  if ( v50 )
  {
    v51 = *(_QWORD *)(v316 + 96);
    if ( *(_DWORD *)(v51 + 32) <= 0x40u )
    {
      v52 = *(_QWORD *)(v51 + 24) == 0;
    }
    else
    {
      v292 = *(_DWORD *)(v51 + 32);
      v52 = v292 == (unsigned int)sub_C444A0(v51 + 24);
    }
    if ( v52 )
    {
      v65 = sub_33FB310(*a1, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
      goto LABEL_64;
    }
  }
  if ( !v306
    || (v306 <= 0x40
      ? (v53 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v306) == *(_QWORD *)(v300 + 24))
      : (v53 = v306 == (unsigned int)sub_C445E0(v295)),
        v53) )
  {
    v97 = *(_QWORD *)(v316 + 96);
    if ( *(_DWORD *)(v97 + 32) <= 0x40u )
    {
      v98 = *(_QWORD *)(v97 + 24) == 0;
    }
    else
    {
      v294 = *(_DWORD *)(v97 + 32);
      v98 = v294 == (unsigned int)sub_C444A0(v97 + 24);
    }
    if ( v98 )
    {
      v65 = sub_33FB160(*a1, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
      goto LABEL_64;
    }
  }
  if ( v306 <= 0x40 )
    v54 = *(_QWORD *)(v300 + 24) == 0;
  else
    v54 = v306 == (unsigned int)sub_C444A0(v295);
  if ( v54 )
  {
    v55 = *(_QWORD *)(v316 + 96);
    v56 = *(_DWORD *)(v55 + 32);
    v57 = v55 + 24;
    if ( v56 <= 0x40 )
    {
      if ( *(_QWORD *)(v55 + 24) != 1 )
      {
LABEL_47:
        if ( v56 )
        {
          if ( !(v56 <= 0x40
               ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v56) == *(_QWORD *)(v55 + 24)
               : v56 == (unsigned int)sub_C445E0(v57)) )
            goto LABEL_51;
        }
        v168 = sub_34074A0(*a1, &v381, v376, *((_QWORD *)&v376 + 1), 2, 0);
        v65 = sub_33FB160(*a1, v168, v169, &v381, v377, v378);
LABEL_64:
        v66 = v381;
        if ( !v381 )
          goto LABEL_65;
        goto LABEL_207;
      }
    }
    else
    {
      v296 = *(_DWORD *)(v55 + 32);
      v58 = sub_C444A0(v57);
      v56 = v296;
      v57 = v55 + 24;
      if ( v58 != v296 - 1 )
        goto LABEL_47;
    }
    v170 = sub_34074A0(*a1, &v381, v376, *((_QWORD *)&v376 + 1), 2, 0);
    v65 = sub_33FB310(*a1, v170, v171, &v381, v377, v378);
    goto LABEL_64;
  }
LABEL_51:
  if ( !(unsigned __int8)sub_326C7E0((__int64 *)&v376, v377, v378, a1[1]) )
    goto LABEL_22;
  v291 = *(_QWORD *)(v327 + 96);
  v301 = (const void **)(v291 + 24);
  v297 = *(_QWORD *)(v316 + 96);
  v293 = (const void **)(v297 + 24);
  v384 = *(_DWORD *)(v291 + 32);
  if ( v384 > 0x40 )
    sub_C43780((__int64)&v383, v301);
  else
    v383 = *(_QWORD *)(v291 + 24);
  sub_C46F20((__int64)&v383, 1u);
  v60 = v384;
  v384 = 0;
  LODWORD(v386) = v60;
  v385 = v383;
  if ( v60 <= 0x40 )
  {
    v289 = *(_QWORD *)(v297 + 24) == v383;
  }
  else
  {
    v288 = v383;
    v289 = sub_C43C50((__int64)&v385, v293);
    if ( v288 )
      j_j___libc_free_0_0(v288);
  }
  if ( v384 > 0x40 && v383 )
    j_j___libc_free_0_0(v383);
  if ( v289 )
  {
    *(_QWORD *)&v376 = sub_33FB310(*a1, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
    v63 = v62;
LABEL_62:
    DWORD2(v376) = v63;
    v64 = sub_3406EB0(*a1, 56, (unsigned int)&v381, v377, v378, v61, v376, v312);
LABEL_63:
    v65 = v64;
    goto LABEL_64;
  }
  v384 = *(_DWORD *)(v291 + 32);
  if ( v384 > 0x40 )
    sub_C43780((__int64)&v383, v301);
  else
    v383 = *(_QWORD *)(v291 + 24);
  sub_C46A40((__int64)&v383, 1);
  v172 = v384;
  v384 = 0;
  LODWORD(v386) = v172;
  v385 = v383;
  if ( v172 <= 0x40 )
  {
    v307 = *(_QWORD *)(v297 + 24) == v383;
  }
  else
  {
    v290 = v383;
    v307 = sub_C43C50((__int64)&v385, v293);
    if ( v290 )
      j_j___libc_free_0_0(v290);
  }
  if ( v384 > 0x40 && v383 )
    j_j___libc_free_0_0(v383);
  v173 = *a1;
  if ( v307 )
  {
    *(_QWORD *)&v376 = sub_33FB160(*a1, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
    v63 = v186;
    goto LABEL_62;
  }
  if ( *(_DWORD *)(v291 + 32) > 0x40u )
  {
    v310 = *a1;
    v185 = sub_C44630((__int64)v301);
    v173 = v310;
    if ( v185 != 1 )
      goto LABEL_223;
  }
  else
  {
    v174 = *(_QWORD *)(v291 + 24);
    if ( !v174 || (v174 & (v174 - 1)) != 0 )
      goto LABEL_223;
  }
  v175 = *(_DWORD *)(v297 + 32);
  if ( v175 <= 0x40 )
  {
    v177 = *(_QWORD *)(v297 + 24) == 0;
  }
  else
  {
    v298 = v173;
    v308 = v175;
    v176 = sub_C444A0((__int64)v293);
    v173 = v298;
    v177 = v308 == v176;
  }
  if ( v177 )
  {
    *(_QWORD *)&v376 = sub_33FB310(v173, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
    DWORD2(v376) = v189;
    v190 = sub_10BBC70((__int64)v301);
    *(_QWORD *)&v191 = sub_3400E40(*a1, v190, v377, v378, &v381);
    v64 = sub_3406EB0(*a1, 190, (unsigned int)&v381, v377, v378, v192, v376, v191);
    goto LABEL_63;
  }
LABEL_223:
  v178 = *(_QWORD *)(v327 + 96);
  v179 = *(_DWORD *)(v178 + 32);
  if ( !v179 )
    goto LABEL_232;
  if ( v179 > 0x40 )
  {
    v309 = v173;
    v331 = *(_DWORD *)(v178 + 32);
    v180 = sub_C445E0(v178 + 24);
    v173 = v309;
    if ( v331 != v180 )
      goto LABEL_226;
LABEL_232:
    *(_QWORD *)&v376 = sub_33FB160(v173, v376, *((_QWORD *)&v376 + 1), &v381, v377, v378);
    DWORD2(v376) = v187;
    v64 = sub_3406EB0(*a1, 187, (unsigned int)&v381, v377, v378, v188, v376, v312);
    goto LABEL_63;
  }
  if ( *(_QWORD *)(v178 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v179) )
    goto LABEL_232;
LABEL_226:
  v332 = v173;
  if ( sub_986760(*(_QWORD *)(v316 + 96) + 24LL) )
  {
    v181 = sub_34074A0(v332, &v381, v376, *((_QWORD *)&v376 + 1), 2, 0);
    v326 = v182;
    v370 = sub_33FB160(*a1, v181, v182, &v381, v377, v378);
    *((_QWORD *)&v275 + 1) = v183 | v326 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v275 = v370;
    v64 = sub_3406EB0(*a1, 187, (unsigned int)&v381, v377, v378, v184, v275, v38);
    goto LABEL_63;
  }
  v65 = sub_326B270(a2, (int)&v381, v332);
  if ( !v65 )
    goto LABEL_22;
  v66 = v381;
  if ( !v381 )
    goto LABEL_66;
LABEL_207:
  v324 = v65;
  sub_B91220((__int64)&v381, v66);
  v65 = v324;
LABEL_65:
  if ( v65 )
  {
LABEL_66:
    v22 = v65;
    goto LABEL_5;
  }
LABEL_24:
  if ( (unsigned __int8)sub_32EFE10(a1, a2, v11.m128i_i64[0], v11.m128i_i64[1], v12, v13) )
  {
    v22 = a2;
    goto LABEL_5;
  }
  if ( v333 != 2 )
    goto LABEL_87;
  v334 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)a1[1] + 1224LL))(
           a1[1],
           *(_QWORD *)(*a1 + 64),
           (unsigned int)v372,
           v373);
  if ( *(_DWORD *)(v350 + 24) != 186 )
    goto LABEL_81;
  v129 = *(_QWORD *)(v350 + 56);
  if ( v129 && !*(_QWORD *)(v129 + 32) )
  {
    v130 = *(const __m128i **)(v350 + 40);
    *((_QWORD *)&v279 + 1) = v13;
    *(_QWORD *)&v279 = v12;
    v314 = _mm_loadu_si128(v130);
    *(_QWORD *)&v131 = sub_340EC60(
                         *a1,
                         205,
                         (unsigned int)&v374,
                         *(unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346),
                         *(_QWORD *)(*(_QWORD *)(v347 + 48) + 16LL * v346 + 8),
                         v359,
                         v130[2].m128i_i64[1],
                         v130[3].m128i_i64[0],
                         *(_OWORD *)&v11,
                         v279);
    v321 = v131;
    if ( v334
      || (v139 = DWORD2(v131), (unsigned __int8)sub_33CF8A0(
                                                  v131,
                                                  DWORD2(v131),
                                                  *((_QWORD *)&v131 + 1),
                                                  v132,
                                                  v133,
                                                  v134)) )
    {
      *((_QWORD *)&v280 + 1) = v13;
      *(_QWORD *)&v280 = v12;
      v140 = sub_340EC60(
               *a1,
               205,
               (unsigned int)&v374,
               *(unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346),
               *(_QWORD *)(*(_QWORD *)(v347 + 48) + 16LL * v346 + 8),
               v359,
               v314.m128i_i64[0],
               v314.m128i_i64[1],
               v321,
               v280);
LABEL_166:
      v22 = v140;
      goto LABEL_5;
    }
    if ( !(unsigned __int8)sub_33CF8A0(v321, v139, v135, v136, v137, v138) )
      sub_32CF870((__int64)a1, v321);
LABEL_81:
    if ( *(_DWORD *)(v350 + 24) == 187 )
    {
      v76 = *(_QWORD *)(v350 + 56);
      if ( v76 )
      {
        if ( !*(_QWORD *)(v76 + 32) )
        {
          v210 = *(const __m128i **)(v350 + 40);
          *((_QWORD *)&v284 + 1) = v13;
          *(_QWORD *)&v284 = v12;
          v315 = _mm_loadu_si128(v210);
          *(_QWORD *)&v211 = sub_340EC60(
                               *a1,
                               205,
                               (unsigned int)&v374,
                               *(unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346),
                               *(_QWORD *)(*(_QWORD *)(v347 + 48) + 16LL * v346 + 8),
                               v359,
                               v210[2].m128i_i64[1],
                               v210[3].m128i_i64[0],
                               *(_OWORD *)&v11,
                               v284);
          v325 = v211;
          if ( v334
            || (v219 = DWORD2(v211),
                (unsigned __int8)sub_33CF8A0(v211, DWORD2(v211), *((_QWORD *)&v211 + 1), v212, v213, v214)) )
          {
            v140 = sub_340EC60(
                     *a1,
                     205,
                     (unsigned int)&v374,
                     *(unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346),
                     *(_QWORD *)(*(_QWORD *)(v347 + 48) + 16LL * v346 + 8),
                     v359,
                     v315.m128i_i64[0],
                     v315.m128i_i64[1],
                     *(_OWORD *)&v11,
                     v325);
            goto LABEL_166;
          }
          if ( !(unsigned __int8)sub_33CF8A0(v325, v219, v215, v216, v217, v218) )
            sub_32CF870((__int64)a1, v325);
        }
      }
    }
  }
  if ( *(_DWORD *)(v347 + 24) == 205 )
  {
    v153 = *(_QWORD *)(v347 + 56);
    if ( v153 )
    {
      if ( !*(_QWORD *)(v153 + 32) )
      {
        v154 = *(_QWORD *)(v347 + 40);
        v155 = *(_QWORD *)v154;
        v156 = *(_QWORD *)(v154 + 8);
        v323 = (__int128)_mm_loadu_si128((const __m128i *)(v154 + 40));
        if ( *(_DWORD *)(v154 + 88) == v339 && *(_QWORD *)(v154 + 80) == v12 )
        {
          v157 = *(_QWORD *)(*(_QWORD *)v154 + 48LL) + 16LL * *(unsigned int *)(v154 + 8);
          v158 = *(_QWORD *)(v350 + 48) + v344;
          v159 = *(_WORD *)v158;
          if ( *(_WORD *)v158 == *(_WORD *)v157 )
          {
            v160 = *(_QWORD *)(v158 + 8);
            if ( *(_QWORD *)(v157 + 8) == v160 || v159 )
            {
              if ( !v334 )
              {
                *((_QWORD *)&v282 + 1) = v156;
                *(_QWORD *)&v282 = v155;
                v161 = sub_3406EB0(*a1, 186, (unsigned int)&v374, v159, v160, *a1, *(_OWORD *)&v371, v282);
                v163 = v162;
                v164 = (unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346);
                *((_QWORD *)&v283 + 1) = v13;
                *(_QWORD *)&v283 = v12;
                v140 = sub_340EC60(
                         *a1,
                         205,
                         (unsigned int)&v374,
                         *v164,
                         *((_QWORD *)v164 + 1),
                         v359,
                         v161,
                         v163,
                         v323,
                         v283);
                goto LABEL_166;
              }
              v237 = sub_32F45B0(a1, v371.m128i_i64[0], v371.m128i_i64[1], v155, v156, a2);
              if ( v237 )
              {
                v238 = (unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346);
                *((_QWORD *)&v286 + 1) = v13;
                *(_QWORD *)&v286 = v12;
                v196 = sub_340EC60(
                         *a1,
                         205,
                         (unsigned int)&v374,
                         *v238,
                         *((_QWORD *)v238 + 1),
                         v359,
                         v237,
                         v236,
                         v323,
                         v286);
                goto LABEL_239;
              }
            }
          }
        }
      }
    }
  }
  if ( *(_DWORD *)(v12 + 24) == 205 )
  {
    v141 = *(_QWORD *)(v12 + 56);
    if ( v141 )
    {
      if ( !*(_QWORD *)(v141 + 32) )
      {
        v142 = *(const __m128i **)(v12 + 40);
        v143 = v142->m128i_i64[0];
        v144 = v142->m128i_i64[1];
        v322 = (__int128)_mm_loadu_si128(v142 + 5);
        if ( v142[3].m128i_i32[0] == v346 && v142[2].m128i_i64[1] == v347 )
        {
          v145 = *(_QWORD *)(v142->m128i_i64[0] + 48) + 16LL * v142->m128i_u32[2];
          v146 = *(_QWORD *)(v350 + 48) + v344;
          v147 = *(_WORD *)v146;
          if ( *(_WORD *)v146 == *(_WORD *)v145 )
          {
            v148 = *(_QWORD *)(v146 + 8);
            if ( *(_QWORD *)(v145 + 8) == v148 || v147 )
            {
              if ( !v334 )
              {
                *((_QWORD *)&v281 + 1) = v144;
                *(_QWORD *)&v281 = v143;
                v149 = sub_3406EB0(*a1, 187, (unsigned int)&v374, v147, v148, *a1, *(_OWORD *)&v371, v281);
                v151 = v150;
                v152 = (unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346);
                v140 = sub_340EC60(
                         *a1,
                         205,
                         (unsigned int)&v374,
                         *v152,
                         *((_QWORD *)v152 + 1),
                         v359,
                         v149,
                         v151,
                         *(_OWORD *)&v11,
                         v322);
                goto LABEL_166;
              }
              v194 = sub_32B5EB0(a1, v371.m128i_i64[0], v371.m128i_i64[1], v143, v144, (__int64)&v374);
              if ( v194 )
              {
                v195 = (unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346);
                v196 = sub_340EC60(
                         *a1,
                         205,
                         (unsigned int)&v374,
                         *v195,
                         *((_QWORD *)v195 + 1),
                         v359,
                         v194,
                         v193,
                         *(_OWORD *)&v11,
                         v322);
LABEL_239:
                v22 = v196;
                goto LABEL_5;
              }
            }
          }
        }
      }
    }
  }
  if ( *(_DWORD *)(v350 + 24) == 79 && v10.m128i_i32[2] == 1 )
  {
    if ( v12 == v350 && !v339 && *(_DWORD *)(v347 + 24) == 57 )
    {
      v228 = *(_QWORD *)(v347 + 40);
      v229 = *(_QWORD *)(v12 + 40);
      if ( *(_QWORD *)v229 == *(_QWORD *)(v228 + 40)
        && *(_DWORD *)(v229 + 8) == *(_DWORD *)(v228 + 48)
        && *(_QWORD *)(v229 + 40) == *(_QWORD *)v228
        && *(_DWORD *)(v229 + 48) == *(_DWORD *)(v228 + 8) )
      {
        v230 = v372;
        v231 = v373;
        if ( !*((_BYTE *)a1 + 33) || (v232 = sub_328D6E0(a1[1], 0xB3u, v372), v231 = v373, v230 = v372, v232) )
        {
          v22 = sub_3406EB0(
                  *a1,
                  179,
                  (unsigned int)&v374,
                  v230,
                  v231,
                  v230,
                  *(_OWORD *)*(_QWORD *)(v350 + 40),
                  *(_OWORD *)(*(_QWORD *)(v350 + 40) + 40LL));
          goto LABEL_5;
        }
      }
    }
    if ( v346 )
      goto LABEL_88;
    if ( v347 != v350 )
      goto LABEL_88;
    if ( *(_DWORD *)(v12 + 24) != 57 )
      goto LABEL_88;
    v220 = *(_QWORD *)(v12 + 40);
    v221 = *(_QWORD *)(v347 + 40);
    if ( *(_QWORD *)v220 != *(_QWORD *)(v221 + 40) )
      goto LABEL_88;
    if ( *(_DWORD *)(v220 + 8) != *(_DWORD *)(v221 + 48) )
      goto LABEL_88;
    if ( *(_QWORD *)(v220 + 40) != *(_QWORD *)v221 )
      goto LABEL_88;
    if ( *(_DWORD *)(v220 + 48) != *(_DWORD *)(v221 + 8) )
      goto LABEL_88;
    v222 = v372;
    v223 = v373;
    if ( *((_BYTE *)a1 + 33) )
    {
      v224 = sub_328D6E0(a1[1], 0xB3u, v372);
      v223 = v373;
      v222 = v372;
      if ( !v224 )
        goto LABEL_88;
    }
    v225 = *a1;
    v226 = sub_3406EB0(
             *a1,
             179,
             (unsigned int)&v374,
             v222,
             v223,
             v222,
             *(_OWORD *)*(_QWORD *)(v350 + 40),
             *(_OWORD *)(*(_QWORD *)(v350 + 40) + 40LL));
    v20 = sub_3407430(v225, v226, v227, &v374, (unsigned int)v372, v373);
LABEL_4:
    v22 = v20;
    goto LABEL_5;
  }
LABEL_87:
  if ( *(_DWORD *)(v350 + 24) != 208 )
    goto LABEL_88;
  v87 = *(__int64 **)(v350 + 40);
  v313 = *v87;
  v335 = _mm_loadu_si128((const __m128i *)v87);
  v311 = *((_DWORD *)v87 + 2);
  v320 = _mm_loadu_si128((const __m128i *)(v87 + 5));
  v330 = v87[5];
  v345 = *(_DWORD *)(v87[10] + 96);
  if ( (unsigned __int8)sub_3286E00(&v371) )
  {
    if ( (unsigned __int8)sub_3260270(*a1, v11.m128i_i64[0], v11.m128i_i64[1], v12, v13, v359, a1[1]) )
    {
      v96 = sub_3270620(
              (__int64)a1,
              (__int64)&v374,
              v372,
              v373,
              v335.m128i_i64[0],
              v335.m128i_i64[1],
              *(_OWORD *)&v320,
              *(_OWORD *)&v11,
              v12,
              v339,
              v345);
      if ( v96 )
      {
LABEL_114:
        v22 = v96;
        goto LABEL_5;
      }
    }
  }
  if ( *((_BYTE *)a1 + 33) )
    goto LABEL_190;
  v302 = v372;
  v341 = v373;
  v361 = a1[1];
  v88 = sub_328A020(v361, 0x4Du, v372, v373, 0);
  v89 = v361;
  v90 = v341;
  v91 = v302;
  if ( v345 == 10 )
  {
    if ( v88 )
    {
      v303 = v341;
      v343 = v91;
      v197 = sub_3286E00(&v371);
      v89 = v361;
      v91 = v343;
      v90 = v303;
      if ( v197 )
      {
        if ( (unsigned __int8)sub_33CF460(v11.m128i_i64[0], v11.m128i_i64[1]) )
        {
          if ( *(_DWORD *)(v12 + 24) == 56 )
          {
            v198 = *(_QWORD *)(v12 + 40);
            if ( v313 == *(_QWORD *)v198 && v311 == *(_DWORD *)(v198 + 8) )
            {
              v199 = *(_QWORD *)(v198 + 40);
              v200 = *(_DWORD *)(v199 + 24);
              v201 = *(_DWORD *)(v330 + 24);
              if ( (v200 == 35 || v200 == 11) && (v201 == 11 || v201 == 35) )
              {
                v363 = v199;
                v202 = *(_QWORD *)(v330 + 96) + 24LL;
                sub_9865C0((__int64)&v383, v202);
                sub_987160((__int64)&v383, v202, v203, v204, v205);
                v206 = v384;
                v384 = 0;
                LODWORD(v386) = v206;
                v385 = v383;
                LOBYTE(v363) = sub_AAD8B0(*(_QWORD *)(v363 + 96) + 24LL, &v385);
                sub_969240(&v385);
                sub_969240(&v383);
                if ( (_BYTE)v363 )
                {
                  v207 = sub_33E5110(*a1, (unsigned int)v372, v373, v19, v336);
                  v276 = sub_3411F20(
                           *a1,
                           77,
                           (unsigned int)&v374,
                           v207,
                           v208,
                           v209,
                           *(_OWORD *)&v335,
                           *(_OWORD *)(*(_QWORD *)(v12 + 40) + 40LL));
                  v22 = sub_3288B20(*a1, (int)&v374, v372, v373, v276, 1, *(_OWORD *)&v11, (unsigned __int64)v276, 0);
                  goto LABEL_5;
                }
              }
            }
          }
        }
LABEL_190:
        v89 = a1[1];
        v91 = v372;
        v90 = v373;
      }
    }
  }
  v356 = v91;
  v362 = v90;
  v342 = v89;
  v92 = sub_328D6E0(v89, 0xCFu, v91);
  v93 = v362;
  v94 = v356;
  if ( v92 || !*((_BYTE *)a1 + 33) && (v95 = sub_328A020(v342, 0xCFu, v356, v362, 0), v93 = v362, v94 = v356, v95) )
  {
    v165 = *(_DWORD *)(v350 + 28);
    *((_QWORD *)&v274 + 1) = v13;
    *(_QWORD *)&v274 = v12;
    v166 = sub_33FC1D0(
             *a1,
             207,
             (unsigned int)&v374,
             v94,
             v93,
             v94,
             *(_OWORD *)&v335,
             *(_OWORD *)&v320,
             *(_OWORD *)&v11,
             v274,
             *(_OWORD *)(*(_QWORD *)(v350 + 40) + 80LL));
    *(_DWORD *)(v166 + 28) = v165;
    v22 = v166;
    goto LABEL_5;
  }
  *((_QWORD *)&v271 + 1) = v13;
  *(_QWORD *)&v271 = v12;
  v96 = sub_32AD6C0(
          a1,
          v335.m128i_i64[0],
          v335.m128i_i64[1],
          v320.m128i_i64[0],
          v320.m128i_i64[1],
          v345,
          *(_OWORD *)&v11,
          v271,
          (__int64)&v374);
  if ( v96 )
    goto LABEL_114;
  *((_QWORD *)&v277 + 1) = v13;
  *(_QWORD *)&v277 = v12;
  v233 = sub_32C7250(
           a1,
           (__int64)&v374,
           **(_QWORD **)(v350 + 40),
           *(_QWORD *)(*(_QWORD *)(v350 + 40) + 8LL),
           *(_QWORD *)(*(_QWORD *)(v350 + 40) + 40LL),
           *(_QWORD *)(*(_QWORD *)(v350 + 40) + 48LL),
           *(_OWORD *)&v11,
           v277,
           *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v350 + 40) + 80LL) + 96LL),
           0);
  v357 = v234;
  v235 = v233;
  if ( !v233 )
  {
LABEL_88:
    if ( sub_32801E0((__int64)&v372) )
      goto LABEL_100;
    v77 = *(const __m128i **)(a2 + 40);
    v78 = *(_QWORD *)(a2 + 80);
    v79 = v77->m128i_i64[0];
    v366 = v77[2].m128i_i64[1];
    v340 = _mm_loadu_si128(v77);
    v355 = v77[3].m128i_i32[0];
    v80 = v77[5].m128i_i64[0];
    LODWORD(v77) = v77[5].m128i_i32[2];
    v385 = v78;
    v360 = v80;
    v337 = (int)v77;
    if ( v78 )
      sub_B96E90((__int64)&v385, v78, 1);
    v81 = a1[1];
    LODWORD(v386) = *(_DWORD *)(a2 + 72);
    v82 = *(_DWORD *)(v366 + 24);
    v83 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v81 + 1368LL);
    if ( v83 == sub_2FE4300 )
    {
      v84 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v81 + 1360LL);
      if ( v84 == sub_2FE3400 )
      {
        if ( v82 <= 98 )
        {
          if ( v82 > 55 )
          {
            switch ( v82 )
            {
              case '8':
              case ':':
              case '?':
              case '@':
              case 'D':
              case 'F':
              case 'L':
              case 'M':
              case 'R':
              case 'S':
              case '`':
              case 'b':
                goto LABEL_141;
              default:
                break;
            }
          }
LABEL_95:
          if ( v82 > 56 )
          {
LABEL_96:
            switch ( v82 )
            {
              case '9':
              case ';':
              case '<':
              case '=':
              case '>':
              case 'T':
              case 'U':
              case 'a':
              case 'c':
              case 'd':
                goto LABEL_141;
              default:
                goto LABEL_98;
            }
          }
          goto LABEL_98;
        }
        if ( v82 > 188 )
        {
          if ( (unsigned int)(v82 - 279) > 7 )
            goto LABEL_140;
        }
        else if ( v82 <= 185 && (unsigned int)(v82 - 172) > 0xB )
        {
          if ( v82 <= 100 )
            goto LABEL_96;
          goto LABEL_140;
        }
      }
      else if ( !v84(v81, v82) )
      {
        if ( v82 <= 100 )
          goto LABEL_95;
LABEL_140:
        if ( (unsigned int)(v82 - 190) > 4 )
          goto LABEL_98;
      }
    }
    else if ( !(unsigned __int8)v83(v81, v82) )
    {
      goto LABEL_98;
    }
LABEL_141:
    if ( *(_DWORD *)(v360 + 24) == v82 && v355 == v337 )
    {
      v109 = *(_QWORD *)(v79 + 56);
      if ( v109 )
      {
        if ( !*(_QWORD *)(v109 + 32) )
        {
          v110 = *(_QWORD *)(v366 + 56);
          if ( v110 )
          {
            if ( !*(_QWORD *)(v110 + 32) )
            {
              v111 = *(_QWORD *)(v360 + 56);
              if ( v111 )
              {
                if ( !*(_QWORD *)(v111 + 32) )
                {
                  v112 = *(_DWORD *)(v366 + 68);
                  v113 = *(__int64 **)(v366 + 40);
                  v338 = *(_QWORD *)(v366 + 48);
                  v114 = v113[5];
                  v115 = *(_QWORD *)(v360 + 40);
                  v116 = *v113;
                  v117 = *(_QWORD *)(v115 + 40);
                  if ( v114 == v117 && *((_DWORD *)v113 + 12) == *(_DWORD *)(v115 + 48) )
                  {
                    *(_QWORD *)&v268 = sub_3288B20(
                                         *a1,
                                         (int)&v385,
                                         *(unsigned __int16 *)(*(_QWORD *)(v116 + 48)
                                                             + 16LL * *((unsigned int *)v113 + 2)),
                                         *(_QWORD *)(*(_QWORD *)(v116 + 48) + 16LL * *((unsigned int *)v113 + 2) + 8),
                                         v340.m128i_i64[0],
                                         v340.m128i_i64[1],
                                         *(_OWORD *)v113,
                                         *(_OWORD *)v115,
                                         0);
                    v128 = v366;
                    v278 = *(_OWORD *)(*(_QWORD *)(v366 + 40) + 40LL);
                    v273 = v268;
LABEL_156:
                    v22 = sub_3411F20(*a1, v82, (unsigned int)&v385, v338, v112, v127, v273, v278);
                    *(_DWORD *)(v22 + 28) = *(_DWORD *)(v128 + 28);
                    sub_33D00A0(v22, *(unsigned int *)(v360 + 28));
                    if ( v385 )
                      sub_B91220((__int64)&v385, v385);
                    goto LABEL_5;
                  }
                  if ( v116 == *(_QWORD *)v115 && *((_DWORD *)v113 + 2) == *(_DWORD *)(v115 + 8) )
                  {
                    v118 = *(_QWORD *)(v115 + 40);
                    v119 = *(_QWORD *)(v115 + 48);
                    v120 = *(_QWORD *)(v114 + 48) + 16LL * *((unsigned int *)v113 + 12);
                    v121 = v113[5];
                    v122 = *(_QWORD *)(v117 + 48) + 16LL * *(unsigned int *)(v115 + 48);
                    v123 = v113[6];
                    v124 = *(_WORD *)v120;
                    v125 = *(_QWORD *)(v120 + 8);
                    if ( *(_WORD *)v122 == v124 && (*(_QWORD *)(v122 + 8) == v125 || v124) )
                    {
                      *((_QWORD *)&v272 + 1) = v119;
                      *(_QWORD *)&v272 = v118;
                      *((_QWORD *)&v269 + 1) = v123;
                      *(_QWORD *)&v269 = v121;
                      *(_QWORD *)&v126 = sub_3288B20(
                                           *a1,
                                           (int)&v385,
                                           v124,
                                           v125,
                                           v340.m128i_i64[0],
                                           v340.m128i_i64[1],
                                           v269,
                                           v272,
                                           0);
                      v128 = v366;
                      v278 = v126;
                      v273 = *(_OWORD *)*(_QWORD *)(v366 + 40);
                      goto LABEL_156;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_98:
    if ( v385 )
      sub_B91220((__int64)&v385, v385);
LABEL_100:
    v85 = *a1;
    v367 = v371.m128i_i64[1];
    v86 = *(_DWORD **)(*a1 + 16);
    if ( !(unsigned __int8)sub_33CF170(v12, v13) )
      goto LABEL_101;
    v99 = *(_QWORD *)(v350 + 48) + v344;
    v100 = *(_WORD *)v99;
    v101 = *(_QWORD *)(v99 + 8);
    LOWORD(v385) = v100;
    v386 = v101;
    if ( v100 )
    {
      v102 = v100 - 17;
      if ( (unsigned __int16)(v100 - 10) > 6u && (unsigned __int16)(v100 - 126) > 0x31u )
      {
        if ( v102 > 0xD3u )
        {
          if ( (unsigned __int16)(v100 - 208) > 0x14u )
          {
LABEL_126:
            v103 = v86[15];
            goto LABEL_127;
          }
          goto LABEL_168;
        }
LABEL_169:
        v103 = v86[17];
LABEL_127:
        if ( v103 == 1
          && *(_DWORD *)(v347 + 24) == 186
          && (unsigned __int8)sub_33CF4D0(
                                *(_QWORD *)(*(_QWORD *)(v347 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v347 + 40) + 48LL)) )
        {
          v104 = (unsigned __int16 *)(*(_QWORD *)(v347 + 48) + 16LL * v346);
          v105 = *((_QWORD *)v104 + 1);
          v106 = *v104;
          if ( v100 == *v104 && (v100 || v101 == v105) )
          {
            *(_QWORD *)&v107 = v350;
            *((_QWORD *)&v107 + 1) = v367 & 0xFFFFFFFF00000000LL | v10.m128i_u32[2];
          }
          else
          {
            *((_QWORD *)&v285 + 1) = v101;
            *(_QWORD *)&v285 = v100;
            v368 = *((_QWORD *)v104 + 1);
            *(_QWORD *)&v107 = sub_33FB620(v85, v350, v10.m128i_i32[2], (unsigned int)&v374, *v104, v105, v285);
            LODWORD(v105) = v368;
          }
          v108 = sub_3406EB0(v85, 186, (unsigned int)&v374, v106, v105, v105, v107, *(_OWORD *)*(_QWORD *)(v347 + 40));
          if ( v108 )
          {
            v22 = v108;
            goto LABEL_5;
          }
        }
LABEL_101:
        v22 = 0;
        goto LABEL_5;
      }
      if ( v102 <= 0xD3u )
        goto LABEL_169;
    }
    else
    {
      v167 = sub_3007030((__int64)&v385);
      if ( sub_30070B0((__int64)&v385) )
        goto LABEL_169;
      if ( !v167 )
        goto LABEL_126;
    }
LABEL_168:
    v103 = v86[16];
    goto LABEL_127;
  }
  if ( *(_DWORD *)(v233 + 24) == 207 )
  {
    v239 = *(_QWORD *)(v233 + 40);
    v240 = *a1;
    v241 = *(_QWORD *)(v350 + 80);
    v369 = *(_DWORD *)(v350 + 28);
    v242 = *(_QWORD *)(*(_QWORD *)(v350 + 48) + v344 + 8);
    v243 = *(unsigned __int16 *)(*(_QWORD *)(v350 + 48) + 16LL * v10.m128i_u32[2]);
    v385 = v241;
    if ( v241 )
    {
      v348 = v239;
      sub_B96E90((__int64)&v385, v241, 1);
      v239 = v348;
    }
    LODWORD(v386) = *(_DWORD *)(v350 + 72);
    v244 = sub_340EC60(
             v240,
             208,
             (unsigned int)&v385,
             v243,
             v242,
             v369,
             *(_QWORD *)v239,
             *(_QWORD *)(v239 + 8),
             *(_OWORD *)(v239 + 40),
             *(_OWORD *)(v239 + 160));
    v352 = v245;
    v246 = v244;
    if ( v385 )
      sub_B91220((__int64)&v385, v385);
    if ( *(_DWORD *)(v246 + 24) != 328 )
    {
      v385 = v246;
      sub_32B3B20((__int64)(a1 + 71), &v385);
      if ( *(int *)(v246 + 88) < 0 )
      {
        *(_DWORD *)(v246 + 88) = *((_DWORD *)a1 + 12);
        v267 = *((unsigned int *)a1 + 12);
        if ( v267 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
        {
          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v267 + 1, 8u, v247, v248);
          v267 = *((unsigned int *)a1 + 12);
        }
        *(_QWORD *)(a1[5] + 8 * v267) = v246;
        ++*((_DWORD *)a1 + 12);
      }
    }
    v249 = *(_QWORD *)(v235 + 80);
    v250 = *(const __m128i **)(v235 + 40);
    v364 = *a1;
    v251 = *(_QWORD *)(v235 + 48) + 16LL * v357;
    v252 = *(_WORD *)v251;
    v253 = *(_QWORD *)(v251 + 8);
    v383 = v249;
    if ( v249 )
      sub_B96E90((__int64)&v383, v249, 1);
    v254 = v252;
    v384 = *(_DWORD *)(v235 + 72);
    v255 = _mm_loadu_si128(v250 + 5);
    v256 = v250[8].m128i_i64[0];
    v257 = (unsigned int)v352;
    v258 = v352;
    v259 = v250[7].m128i_i64[1];
    v260 = (unsigned int)v352;
    v353 = (__int128)v255;
    v261 = *(_QWORD *)(v246 + 48) + 16 * v260;
    v262 = *(_WORD *)v261;
    v263 = *(_QWORD *)(v261 + 8);
    LOWORD(v385) = v262;
    v386 = v263;
    if ( v262 )
    {
      v264 = 206 - ((unsigned __int16)(v262 - 17) >= 0xD4u);
    }
    else
    {
      v358 = v254;
      v349 = v258;
      v266 = sub_30070B0((__int64)&v385);
      v254 = v358;
      v258 = v349;
      v264 = 206 - !v266;
    }
    *((_QWORD *)&v287 + 1) = v256;
    *(_QWORD *)&v287 = v259;
    v233 = sub_340EC60(
             v364,
             v264,
             (unsigned int)&v383,
             v254,
             v253,
             0,
             v246,
             v257 | v258 & 0xFFFFFFFF00000000LL,
             v353,
             v287);
    v265 = v233;
    if ( v383 )
    {
      v365 = v233;
      sub_B91220((__int64)&v383, v383);
      v233 = v365;
    }
    *(_DWORD *)(v265 + 28) = v369;
  }
  v22 = v233;
LABEL_5:
  if ( v374 )
    sub_B91220((__int64)&v374, v374);
  return v22;
}
