// Function: sub_2832000
// Address: 0x2832000
//
__int64 __fastcall sub_2832000(__int64 *a1)
{
  __int64 *v1; // r12
  __int64 v2; // rdi
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // r10
  unsigned __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r9
  unsigned __int64 v19; // rsi
  __int64 v20; // r10
  __int64 v21; // r8
  __int64 v22; // r9
  char *v23; // rdi
  char *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  char *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // r15
  _BYTE *v33; // rax
  __int64 v34; // r11
  __int64 **v35; // rdi
  __int64 **v36; // rdx
  __int64 **v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // r8
  unsigned int v41; // esi
  int v42; // r11d
  _QWORD *v43; // rax
  unsigned int v44; // r9d
  _QWORD *v45; // rdx
  __int64 v46; // rcx
  __int64 *v47; // rax
  __int64 v48; // rcx
  _BYTE *v49; // rsi
  _BYTE *v50; // rsi
  _BYTE *v51; // rsi
  __int64 *v52; // r12
  unsigned int v53; // r12d
  _BYTE *v55; // rsi
  _BYTE *v56; // r10
  _BYTE *v57; // r9
  __int64 *v58; // rdi
  int v59; // eax
  size_t v60; // r14
  __int64 v61; // r8
  __int64 v62; // r8
  __int64 v63; // r13
  __int64 *v64; // r14
  __int64 *v65; // rbx
  __int64 v66; // rax
  __int64 v67; // rsi
  int v68; // edi
  __int64 v69; // r9
  int v70; // edi
  __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // r11
  __int64 *v74; // r9
  __int64 v75; // r9
  _QWORD *v76; // rax
  __int64 v77; // rdi
  __int64 v78; // rax
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 *v83; // r13
  __int64 *v84; // r11
  __int64 *v85; // r10
  __int64 *v86; // rax
  __int64 v87; // r14
  __int64 v88; // r12
  unsigned int v89; // esi
  unsigned int v90; // r15d
  unsigned int v91; // edi
  __int64 *v92; // rax
  _BYTE *v93; // rax
  __int64 *v94; // rdi
  __int64 *v95; // rax
  _BYTE *v96; // rsi
  __int64 v97; // rsi
  _QWORD *v98; // rax
  __int64 v99; // rbx
  unsigned int v100; // esi
  __int64 v101; // r8
  unsigned int v102; // r14d
  __int64 v103; // rcx
  _QWORD *v104; // rdx
  __int64 v105; // rdi
  __int64 *v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // r14
  __int64 v110; // rax
  __int64 v111; // rbx
  __int64 v112; // rsi
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rsi
  __int64 v116; // rax
  __int64 v117; // rax
  _QWORD *v118; // rdi
  __int64 v119; // rbx
  __int64 v120; // r8
  __int64 v121; // r9
  __int64 v122; // rdx
  __int64 v123; // r15
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rbx
  __int64 v127; // r8
  __int64 v128; // r9
  __int64 v129; // rdx
  __int64 v130; // r15
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rbx
  _QWORD **v134; // r15
  _QWORD *v135; // r14
  char v136; // dh
  __int64 v137; // rsi
  char v138; // al
  __int64 v139; // r14
  __int64 *v140; // r15
  __int64 v141; // rbx
  _QWORD *v142; // r12
  char v143; // dh
  __int64 v144; // rsi
  char v145; // al
  __int64 v146; // rdx
  __int64 v147; // r15
  __int64 v148; // rbx
  __int64 v149; // rax
  __int64 v150; // rax
  __int64 v151; // rsi
  int v152; // ecx
  __int64 v153; // rdi
  int v154; // ecx
  unsigned int v155; // edx
  __int64 *v156; // rax
  __int64 v157; // r11
  __int64 v158; // rax
  __int64 v159; // r14
  __int64 v160; // rdx
  __int64 *v161; // rax
  __int64 v162; // rsi
  __int64 v163; // rsi
  __int64 v164; // rsi
  __int64 v165; // r12
  __int64 v166; // rdx
  __int64 v167; // rcx
  __int64 v168; // rdx
  int v169; // eax
  int v170; // eax
  unsigned int v171; // esi
  __int64 v172; // rax
  __int64 v173; // rsi
  __int64 v174; // rsi
  __int64 v175; // rsi
  char v176; // dh
  char v177; // al
  __int64 v178; // rax
  __int64 v179; // rdx
  __int64 v180; // rdx
  __int64 v181; // r14
  __int64 v182; // rdx
  __int64 v183; // r8
  __int64 v184; // r9
  __int64 v185; // r15
  __int64 v186; // r12
  _QWORD *v187; // rax
  _QWORD *v188; // rdx
  __int64 v189; // rax
  unsigned __int64 v190; // rdx
  __int64 v191; // rax
  __int64 v192; // rdx
  __int64 v193; // r8
  __int64 v194; // r9
  __int64 v195; // r15
  __int64 v196; // r12
  _QWORD *v197; // rax
  _QWORD *v198; // rdx
  __int64 v199; // rax
  unsigned __int64 v200; // rdx
  __int64 v201; // rax
  _QWORD **v202; // r14
  __int64 v203; // rbx
  _QWORD *v204; // r12
  char v205; // dh
  __int64 v206; // rsi
  char v207; // al
  _QWORD **v208; // r14
  __int64 v209; // rbx
  _QWORD *v210; // r12
  char v211; // dh
  __int64 v212; // rsi
  char v213; // al
  __int64 v214; // r9
  unsigned __int64 v215; // rdx
  __int64 v216; // rax
  __int64 v217; // rbx
  unsigned __int64 v218; // r15
  __int64 v219; // r14
  unsigned __int64 v220; // r8
  int v221; // eax
  __int64 v222; // r10
  unsigned int v223; // r10d
  __int64 v224; // r8
  __int64 *v225; // rax
  __int64 *v226; // rax
  unsigned int v227; // esi
  unsigned int v228; // edx
  __int64 *v229; // rax
  __int64 v230; // rdi
  __int64 v231; // rax
  _QWORD *v232; // rax
  __int64 v233; // rdx
  _QWORD *v234; // rdx
  int v235; // eax
  int v236; // r8d
  __int64 v237; // rdx
  int v238; // eax
  int v239; // eax
  unsigned int v240; // r15d
  __int64 v241; // rsi
  int v242; // edi
  int v243; // eax
  unsigned int v244; // r15d
  int v245; // edi
  __int64 v246; // rsi
  __int64 v247; // rax
  int v248; // r11d
  _QWORD *v249; // rax
  int v250; // ecx
  int v251; // edx
  int v252; // edx
  int v253; // r11d
  int v254; // r11d
  __int64 v255; // r10
  __int64 v256; // r8
  __int64 v257; // rsi
  int v258; // r9d
  _QWORD *v259; // rcx
  int v260; // r10d
  int v261; // r10d
  __int64 v262; // r9
  int v263; // r8d
  __int64 v264; // r14
  __int64 v265; // rsi
  int v266; // r11d
  int v267; // r11d
  __int64 v268; // r10
  __int64 v269; // rcx
  __int64 v270; // r8
  int v271; // edi
  _QWORD *v272; // rsi
  int v273; // r10d
  int v274; // r10d
  __int64 v275; // r9
  __int64 v276; // r11
  int v277; // esi
  _QWORD *v278; // rcx
  __int64 v279; // rdi
  int v280; // eax
  int v281; // ecx
  unsigned int v282; // [rsp+4h] [rbp-44Ch]
  __int64 v283; // [rsp+8h] [rbp-448h]
  __int64 v284; // [rsp+10h] [rbp-440h]
  __int64 v285; // [rsp+18h] [rbp-438h]
  __int64 v286; // [rsp+20h] [rbp-430h]
  __int64 v287; // [rsp+28h] [rbp-428h]
  __int64 v288; // [rsp+30h] [rbp-420h]
  __int64 v289; // [rsp+38h] [rbp-418h]
  __int64 v290; // [rsp+48h] [rbp-408h]
  __int64 *v291; // [rsp+48h] [rbp-408h]
  __int64 v292; // [rsp+50h] [rbp-400h]
  __int64 *v293; // [rsp+58h] [rbp-3F8h]
  __int64 v294; // [rsp+58h] [rbp-3F8h]
  __int64 v295; // [rsp+58h] [rbp-3F8h]
  int v296; // [rsp+58h] [rbp-3F8h]
  __int64 *v297; // [rsp+58h] [rbp-3F8h]
  __int64 *v298; // [rsp+58h] [rbp-3F8h]
  __int64 v299; // [rsp+60h] [rbp-3F0h]
  __int64 v300; // [rsp+60h] [rbp-3F0h]
  __int64 v301; // [rsp+68h] [rbp-3E8h]
  __int64 v302; // [rsp+68h] [rbp-3E8h]
  __int64 *v303; // [rsp+68h] [rbp-3E8h]
  __int64 *v304; // [rsp+68h] [rbp-3E8h]
  _BYTE *v305; // [rsp+68h] [rbp-3E8h]
  __int64 v306; // [rsp+70h] [rbp-3E0h]
  unsigned __int64 v307; // [rsp+78h] [rbp-3D8h]
  __int64 v308; // [rsp+78h] [rbp-3D8h]
  __int64 v309; // [rsp+78h] [rbp-3D8h]
  __int64 *v310; // [rsp+78h] [rbp-3D8h]
  __int64 v311; // [rsp+78h] [rbp-3D8h]
  __int64 v312; // [rsp+80h] [rbp-3D0h]
  __int64 v313; // [rsp+80h] [rbp-3D0h]
  __int64 *v314; // [rsp+80h] [rbp-3D0h]
  __int64 v315; // [rsp+88h] [rbp-3C8h]
  unsigned __int64 v316; // [rsp+88h] [rbp-3C8h]
  __int64 v317; // [rsp+88h] [rbp-3C8h]
  __int64 v318; // [rsp+88h] [rbp-3C8h]
  __int64 v319; // [rsp+88h] [rbp-3C8h]
  __int64 v320; // [rsp+88h] [rbp-3C8h]
  _BYTE *v321; // [rsp+88h] [rbp-3C8h]
  __int64 v322; // [rsp+88h] [rbp-3C8h]
  __int64 v323; // [rsp+90h] [rbp-3C0h]
  __int64 v324; // [rsp+90h] [rbp-3C0h]
  __int64 v325; // [rsp+90h] [rbp-3C0h]
  int v326; // [rsp+90h] [rbp-3C0h]
  __int64 v327; // [rsp+90h] [rbp-3C0h]
  __int64 *v328; // [rsp+90h] [rbp-3C0h]
  __int64 v329; // [rsp+90h] [rbp-3C0h]
  _QWORD **v330; // [rsp+90h] [rbp-3C0h]
  __int64 *v331; // [rsp+90h] [rbp-3C0h]
  __int64 v332; // [rsp+90h] [rbp-3C0h]
  __int64 *v333; // [rsp+90h] [rbp-3C0h]
  __int64 *v334; // [rsp+90h] [rbp-3C0h]
  _QWORD **v335; // [rsp+90h] [rbp-3C0h]
  _QWORD **v336; // [rsp+90h] [rbp-3C0h]
  unsigned __int64 v337; // [rsp+90h] [rbp-3C0h]
  __int64 v338; // [rsp+98h] [rbp-3B8h]
  __int64 v339; // [rsp+A0h] [rbp-3B0h]
  __int64 v340; // [rsp+A8h] [rbp-3A8h]
  __int64 v341; // [rsp+B0h] [rbp-3A0h]
  __int64 v342; // [rsp+B8h] [rbp-398h]
  unsigned __int64 *v343; // [rsp+C0h] [rbp-390h] BYREF
  __int64 v344; // [rsp+C8h] [rbp-388h]
  __int64 v345; // [rsp+D0h] [rbp-380h]
  _BYTE *v346; // [rsp+E0h] [rbp-370h] BYREF
  __int64 v347; // [rsp+E8h] [rbp-368h]
  _BYTE v348[32]; // [rsp+F0h] [rbp-360h] BYREF
  _BYTE *v349; // [rsp+110h] [rbp-340h] BYREF
  __int64 v350; // [rsp+118h] [rbp-338h]
  _BYTE v351[64]; // [rsp+120h] [rbp-330h] BYREF
  __int64 *v352; // [rsp+160h] [rbp-2F0h] BYREF
  __int64 v353; // [rsp+168h] [rbp-2E8h]
  _BYTE dest[736]; // [rsp+170h] [rbp-2E0h] BYREF

  v1 = a1;
  v2 = *a1;
  v343 = 0;
  v344 = 0;
  v345 = 0;
  v342 = sub_D4B130(v2);
  v341 = sub_D4B130(v1[1]);
  v4 = *(_QWORD *)(v342 + 56);
  if ( !v4 )
    goto LABEL_497;
  if ( *(_BYTE *)(v4 - 24) == 84 || !sub_AA5510(v342) )
    v342 = sub_F67CB0(*v1, v1[4], v1[3], 0, 1, v3);
  v340 = **(_QWORD **)(*v1 + 32);
  if ( v341 == v340 )
  {
    v341 = sub_F67CB0(v1[1], v1[4], v1[3], 0, 1, v3);
    v340 = **(_QWORD **)(*v1 + 32);
  }
  v5 = v1[1];
  v6 = **(_QWORD **)(v5 + 32);
  v339 = v6;
  v7 = sub_D47930(v5);
  v338 = sub_D47930(*v1);
  v315 = sub_AA5510(v342);
  v8 = sub_AA5510(v7);
  v9 = *(_QWORD *)(v338 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v338 + 48 )
    goto LABEL_497;
  if ( !v9 )
    goto LABEL_497;
  v10 = v9 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
    goto LABEL_497;
  if ( *(_BYTE *)(v9 - 24) != 31 )
    v10 = 0;
  v11 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == v7 + 48 )
    goto LABEL_497;
  if ( !v11 )
    goto LABEL_497;
  v12 = v11 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
    goto LABEL_497;
  if ( *(_BYTE *)(v11 - 24) != 31 )
    v12 = 0;
  v13 = *(_QWORD *)(v340 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 == v340 + 48 )
    goto LABEL_497;
  if ( !v13 )
    goto LABEL_497;
  v14 = v13 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
    goto LABEL_497;
  if ( *(_BYTE *)(v13 - 24) != 31 )
    v14 = 0;
  v15 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 == v6 + 48 )
    goto LABEL_497;
  if ( !v15 )
    goto LABEL_497;
  v16 = v15 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
    goto LABEL_497;
  if ( *(_BYTE *)(v15 - 24) != 31 )
    v16 = 0;
  if ( !v315 )
    goto LABEL_84;
  if ( !v8 )
    goto LABEL_84;
  if ( !v10 )
    goto LABEL_84;
  if ( !v12 )
    goto LABEL_84;
  v323 = v14;
  if ( !v14 || !v16 )
    goto LABEL_84;
  v17 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v17 == v8 + 48 )
    goto LABEL_497;
  if ( !v17 )
    goto LABEL_497;
  v18 = v17 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 > 0xA )
    goto LABEL_497;
  if ( *(_BYTE *)(v17 - 24) != 31 )
    v18 = 0;
  v19 = *(_QWORD *)(v315 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 == v315 + 48 )
LABEL_88:
    BUG();
  if ( !v19 )
LABEL_497:
    BUG();
  v316 = *(_QWORD *)(v315 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
    goto LABEL_88;
  if ( *(_BYTE *)(v19 - 24) != 31 || (v307 = v18) == 0 || (v312 = sub_AA5780(v339)) == 0 )
  {
LABEL_84:
    v53 = 0;
    goto LABEL_85;
  }
  sub_2831360(v316 - 24, v342, v341, (__int64)&v343);
  v20 = v323;
  v21 = v312;
  v22 = v307;
  if ( (*(_BYTE *)(v323 + 7) & 0x40) != 0 )
  {
    v24 = *(char **)(v323 - 8);
    v23 = &v24[32 * (*(_DWORD *)(v323 + 4) & 0x7FFFFFF)];
    if ( (*(_DWORD *)(v323 + 4) & 0x7FFFFFF) == 3 )
      v24 += 32;
  }
  else
  {
    v23 = (char *)v323;
    v24 = (char *)(v323 - 32LL * (*(_DWORD *)(v323 + 4) & 0x7FFFFFF));
    if ( (*(_DWORD *)(v323 + 4) & 0x7FFFFFF) == 3 )
      v24 += 32;
  }
  v25 = (v23 - v24) >> 7;
  v26 = (v23 - v24) >> 5;
  if ( v25 > 0 )
  {
    v27 = &v24[128 * v25];
    while ( v338 != *(_QWORD *)v24 )
    {
      if ( v338 == *((_QWORD *)v24 + 4) )
      {
        v24 += 32;
        goto LABEL_53;
      }
      if ( v338 == *((_QWORD *)v24 + 8) )
      {
        v24 += 64;
        goto LABEL_53;
      }
      if ( v338 == *((_QWORD *)v24 + 12) )
      {
        v24 += 96;
        goto LABEL_53;
      }
      v24 += 128;
      if ( v24 == v27 )
      {
        v26 = (v23 - v24) >> 5;
        goto LABEL_333;
      }
    }
    goto LABEL_53;
  }
LABEL_333:
  if ( v26 == 2 )
  {
LABEL_378:
    if ( v338 != *(_QWORD *)v24 )
    {
      v24 += 32;
LABEL_336:
      if ( v338 != *(_QWORD *)v24 )
        goto LABEL_55;
      goto LABEL_53;
    }
    goto LABEL_53;
  }
  if ( v26 != 3 )
  {
    if ( v26 != 1 )
      goto LABEL_55;
    goto LABEL_336;
  }
  if ( v338 != *(_QWORD *)v24 )
  {
    v24 += 32;
    goto LABEL_378;
  }
LABEL_53:
  if ( v24 != v23 )
  {
    v308 = v312;
    v313 = v22;
    sub_2831360(v323, v338, v7, (__int64)&v343);
    v21 = v308;
    v22 = v313;
    v20 = v323;
  }
LABEL_55:
  v306 = v22;
  v324 = v21;
  sub_2831360(v20, v341, v21, (__int64)&v343);
  sub_AA5D60(v324, v339, v340);
  sub_2831360(v16, v324, v342, (__int64)&v343);
  v28 = *(_QWORD *)(v12 - 32);
  v309 = v28;
  if ( v28 && v339 == v28 )
    v309 = *(_QWORD *)(v12 - 64);
  sub_2831360(v306, v7, v309, (__int64)&v343);
  v29 = *(_QWORD *)(v10 - 32);
  v292 = v29;
  if ( v29 && v29 == v340 )
    v292 = *(_QWORD *)(v10 - 64);
  sub_2831360(v12, v309, v292, (__int64)&v343);
  sub_2831360(v10, v292, v7, (__int64)&v343);
  v30 = v1[4];
  sub_B26290((__int64)&v352, v343, (v344 - (__int64)v343) >> 4, 1u);
  sub_B24D40(v30, (__int64)&v352, 0);
  sub_B1A8B0((__int64)&v352, (__int64)&v352);
  v31 = *v1;
  v32 = v1[1];
  v290 = *v1 + 32;
  v325 = *(_QWORD *)*v1;
  v352 = (__int64 *)v341;
  v33 = sub_282FA80(*(_QWORD **)(v31 + 32), *(_QWORD *)(v31 + 40), (__int64 *)&v352);
  sub_F681A0(v34, v33);
  v289 = v31 + 56;
  if ( *(_BYTE *)(v31 + 84) )
  {
    v35 = *(__int64 ***)(v31 + 64);
    v36 = &v35[*(unsigned int *)(v31 + 76)];
    v37 = v35;
    if ( v35 != v36 )
    {
      while ( v352 != *v37 )
      {
        if ( v36 == ++v37 )
          goto LABEL_67;
      }
      v38 = (unsigned int)(*(_DWORD *)(v31 + 76) - 1);
      *(_DWORD *)(v31 + 76) = v38;
      *v37 = v35[v38];
      ++*(_QWORD *)(v31 + 56);
    }
  }
  else
  {
    v226 = sub_C8CA60(v289, (__int64)v352);
    if ( v226 )
    {
      *v226 = -2;
      ++*(_DWORD *)(v31 + 80);
      ++*(_QWORD *)(v31 + 56);
    }
  }
LABEL_67:
  v39 = v1[3];
  v40 = *(_QWORD *)(v39 + 8);
  v41 = *(_DWORD *)(v39 + 24);
  v317 = v39;
  if ( v325 )
  {
    if ( v41 )
    {
      v42 = 1;
      v43 = 0;
      v44 = (v41 - 1) & (((unsigned int)v341 >> 4) ^ ((unsigned int)v341 >> 9));
      v45 = (_QWORD *)(v40 + 16LL * v44);
      v46 = *v45;
      if ( v341 == *v45 )
      {
LABEL_70:
        v47 = v45 + 1;
LABEL_71:
        *v47 = v325;
        sub_2831230(v325, v31);
        sub_2831230(v31, v32);
        v352 = (__int64 *)v32;
        *(_QWORD *)v32 = v325;
        v49 = *(_BYTE **)(v325 + 16);
        if ( v49 == *(_BYTE **)(v325 + 24) )
        {
          sub_D4C7F0(v325 + 8, v49, &v352);
        }
        else
        {
          if ( v49 )
          {
            *(_QWORD *)v49 = v352;
            v49 = *(_BYTE **)(v325 + 16);
          }
          *(_QWORD *)(v325 + 16) = v49 + 8;
        }
        goto LABEL_75;
      }
      while ( v46 != -4096 )
      {
        if ( !v43 && v46 == -8192 )
          v43 = v45;
        v44 = (v41 - 1) & (v42 + v44);
        v45 = (_QWORD *)(v40 + 16LL * v44);
        v46 = *v45;
        if ( v341 == *v45 )
          goto LABEL_70;
        ++v42;
      }
      if ( !v43 )
        v43 = v45;
      ++*(_QWORD *)v317;
      v252 = *(_DWORD *)(v317 + 16) + 1;
      if ( 4 * v252 < 3 * v41 )
      {
        if ( v41 - *(_DWORD *)(v317 + 20) - v252 > v41 >> 3 )
        {
LABEL_422:
          *(_DWORD *)(v317 + 16) = v252;
          if ( *v43 != -4096 )
            --*(_DWORD *)(v317 + 20);
          v43[1] = 0;
          v47 = v43 + 1;
          *(v47 - 1) = v341;
          goto LABEL_71;
        }
        sub_D4F150(v317, v41);
        v260 = *(_DWORD *)(v317 + 24);
        if ( v260 )
        {
          v261 = v260 - 1;
          v262 = *(_QWORD *)(v317 + 8);
          v263 = 1;
          LODWORD(v264) = v261 & (((unsigned int)v341 >> 4) ^ ((unsigned int)v341 >> 9));
          v252 = *(_DWORD *)(v317 + 16) + 1;
          v259 = 0;
          v43 = (_QWORD *)(v262 + 16LL * (unsigned int)v264);
          v265 = *v43;
          if ( v341 == *v43 )
            goto LABEL_422;
          while ( v265 != -4096 )
          {
            if ( !v259 && v265 == -8192 )
              v259 = v43;
            v264 = v261 & (unsigned int)(v264 + v263);
            v43 = (_QWORD *)(v262 + 16 * v264);
            v265 = *v43;
            if ( v341 == *v43 )
              goto LABEL_422;
            ++v263;
          }
          goto LABEL_437;
        }
        goto LABEL_494;
      }
    }
    else
    {
      ++*(_QWORD *)v39;
    }
    sub_D4F150(v317, 2 * v41);
    v253 = *(_DWORD *)(v317 + 24);
    if ( v253 )
    {
      v254 = v253 - 1;
      v255 = *(_QWORD *)(v317 + 8);
      v252 = *(_DWORD *)(v317 + 16) + 1;
      LODWORD(v256) = v254 & (((unsigned int)v341 >> 9) ^ ((unsigned int)v341 >> 4));
      v43 = (_QWORD *)(v255 + 16LL * (unsigned int)v256);
      v257 = *v43;
      if ( v341 == *v43 )
        goto LABEL_422;
      v258 = 1;
      v259 = 0;
      while ( v257 != -4096 )
      {
        if ( !v259 && v257 == -8192 )
          v259 = v43;
        v256 = v254 & (unsigned int)(v256 + v258);
        v43 = (_QWORD *)(v255 + 16 * v256);
        v257 = *v43;
        if ( v341 == *v43 )
          goto LABEL_422;
        ++v258;
      }
LABEL_437:
      if ( v259 )
        v43 = v259;
      goto LABEL_422;
    }
LABEL_494:
    ++*(_DWORD *)(v317 + 16);
    BUG();
  }
  if ( v41 )
  {
    v227 = v41 - 1;
    v228 = v227 & (((unsigned int)v341 >> 9) ^ ((unsigned int)v341 >> 4));
    v229 = (__int64 *)(v40 + 16LL * v228);
    v230 = *v229;
    if ( v341 == *v229 )
    {
LABEL_343:
      *v229 = -8192;
      --*(_DWORD *)(v317 + 16);
      ++*(_DWORD *)(v317 + 20);
    }
    else
    {
      v280 = 1;
      while ( v230 != -4096 )
      {
        v281 = v280 + 1;
        v228 = v227 & (v280 + v228);
        v229 = (__int64 *)(v40 + 16LL * v228);
        v230 = *v229;
        if ( v341 == *v229 )
          goto LABEL_343;
        v280 = v281;
      }
    }
  }
  sub_2831230(v31, v32);
  v231 = v1[3];
  v48 = *(_QWORD *)(v231 + 40);
  v232 = *(_QWORD **)(v231 + 32);
  v233 = (v48 - (__int64)v232) >> 5;
  if ( v233 > 0 )
  {
    v234 = &v232[4 * v233];
    while ( v31 != *v232 )
    {
      if ( v31 == v232[1] )
      {
        ++v232;
        goto LABEL_351;
      }
      if ( v31 == v232[2] )
      {
        v232 += 2;
        goto LABEL_351;
      }
      if ( v31 == v232[3] )
      {
        v232 += 3;
        goto LABEL_351;
      }
      v232 += 4;
      if ( v232 == v234 )
        goto LABEL_396;
    }
    goto LABEL_351;
  }
  v234 = v232;
LABEL_396:
  v247 = v48 - (_QWORD)v234;
  if ( v48 - (_QWORD)v234 == 16 )
  {
LABEL_427:
    v232 = v234;
    if ( v31 == *v234 )
      goto LABEL_351;
    ++v234;
LABEL_429:
    v232 = (_QWORD *)v48;
    if ( v31 == *v234 )
      v232 = v234;
    goto LABEL_351;
  }
  if ( v247 == 24 )
  {
    v232 = v234;
    if ( v31 == *v234 )
      goto LABEL_351;
    ++v234;
    goto LABEL_427;
  }
  if ( v247 == 8 )
    goto LABEL_429;
  v232 = (_QWORD *)v48;
LABEL_351:
  *v232 = v32;
LABEL_75:
  v50 = *(_BYTE **)(v32 + 8);
  v318 = v32 + 8;
  if ( v50 != *(_BYTE **)(v32 + 16) )
  {
    v314 = v1;
    do
    {
      v52 = *(__int64 **)v50;
      sub_D4C9B0(v318, v50);
      v352 = v52;
      *v52 = v31;
      v51 = *(_BYTE **)(v31 + 16);
      if ( v51 == *(_BYTE **)(v31 + 24) )
      {
        sub_D4C7F0(v31 + 8, v51, &v352);
      }
      else
      {
        if ( v51 )
        {
          *(_QWORD *)v51 = v352;
          v51 = *(_BYTE **)(v31 + 16);
        }
        *(_QWORD *)(v31 + 16) = v51 + 8;
      }
      v50 = *(_BYTE **)(v32 + 8);
    }
    while ( *(_BYTE **)(v32 + 16) != v50 );
    v1 = v314;
  }
  v352 = (__int64 *)v31;
  *(_QWORD *)v31 = v32;
  v55 = *(_BYTE **)(v32 + 16);
  if ( v55 == *(_BYTE **)(v32 + 24) )
  {
    sub_D4C7F0(v318, v55, &v352);
  }
  else
  {
    if ( v55 )
    {
      *(_QWORD *)v55 = v352;
      v55 = *(_BYTE **)(v32 + 16);
    }
    *(_QWORD *)(v32 + 16) = v55 + 8;
  }
  v56 = *(_BYTE **)(v32 + 40);
  v57 = *(_BYTE **)(v32 + 32);
  v58 = (__int64 *)dest;
  v353 = 0x800000000LL;
  v59 = 0;
  v60 = v56 - v57;
  v352 = (__int64 *)dest;
  v61 = (v56 - v57) >> 3;
  if ( (unsigned __int64)(v56 - v57) > 0x40 )
  {
    v305 = v56;
    v321 = v57;
    v337 = (v56 - v57) >> 3;
    sub_C8D5F0((__int64)&v352, dest, v337, 8u, v61, (__int64)v57);
    v59 = v353;
    v56 = v305;
    v57 = v321;
    LODWORD(v61) = v337;
    v58 = &v352[(unsigned int)v353];
  }
  if ( v57 != v56 )
  {
    v326 = v61;
    memcpy(v58, v57, v60);
    v59 = v353;
    LODWORD(v61) = v326;
  }
  LODWORD(v353) = v61 + v59;
  if ( *(_QWORD *)(v31 + 32) != *(_QWORD *)(v31 + 40) )
  {
    v327 = v7;
    v62 = v32 + 56;
    v63 = v31;
    v64 = *(__int64 **)(v31 + 32);
    v65 = *(__int64 **)(v31 + 40);
    while ( 1 )
    {
      v66 = v1[3];
      v67 = *v64;
      v68 = *(_DWORD *)(v66 + 24);
      v69 = *(_QWORD *)(v66 + 8);
      if ( v68 )
      {
        v70 = v68 - 1;
        v71 = v70 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v72 = (__int64 *)(v69 + 16 * v71);
        v73 = *v72;
        if ( v67 != *v72 )
        {
          v221 = 1;
          while ( v73 != -4096 )
          {
            v48 = (unsigned int)(v221 + 1);
            v71 = v70 & (unsigned int)(v221 + v71);
            v72 = (__int64 *)(v69 + 16LL * (unsigned int)v71);
            v73 = *v72;
            if ( v67 == *v72 )
              goto LABEL_106;
            v221 = v48;
          }
          goto LABEL_103;
        }
LABEL_106:
        if ( v63 == v72[1] )
        {
          v349 = (_BYTE *)*v64;
          v74 = *(__int64 **)(v32 + 40);
          if ( v74 == *(__int64 **)(v32 + 48) )
          {
            v322 = v62;
            sub_9319A0(v32 + 32, *(_BYTE **)(v32 + 40), &v349);
            v67 = (__int64)v349;
            v62 = v322;
          }
          else
          {
            if ( v74 )
            {
              *v74 = v67;
              v74 = *(__int64 **)(v32 + 40);
            }
            v75 = (__int64)(v74 + 1);
            *(_QWORD *)(v32 + 40) = v75;
          }
          if ( !*(_BYTE *)(v32 + 84) )
            goto LABEL_314;
          v76 = *(_QWORD **)(v32 + 64);
          v77 = *(unsigned int *)(v32 + 76);
          v71 = (__int64)&v76[v77];
          if ( v76 != (_QWORD *)v71 )
          {
            while ( *v76 != v67 )
            {
              if ( (_QWORD *)v71 == ++v76 )
                goto LABEL_115;
            }
            goto LABEL_103;
          }
LABEL_115:
          if ( (unsigned int)v77 < *(_DWORD *)(v32 + 72) )
          {
            *(_DWORD *)(v32 + 76) = v77 + 1;
            *(_QWORD *)v71 = v67;
            ++*(_QWORD *)(v32 + 56);
          }
          else
          {
LABEL_314:
            v320 = v62;
            sub_C8CC70(v62, v67, v71, v48, v62, v75);
            v62 = v320;
          }
        }
      }
LABEL_103:
      if ( v65 == ++v64 )
      {
        v31 = v63;
        v7 = v327;
        break;
      }
    }
  }
  v301 = **(_QWORD **)(v32 + 32);
  v78 = sub_D47930(v32);
  v81 = (unsigned int)v353;
  v299 = v78;
  v82 = (__int64)&v352[(unsigned int)v353];
  v328 = (__int64 *)v82;
  if ( v352 == (__int64 *)v82 )
    goto LABEL_136;
  v288 = v7;
  v83 = v352;
  v84 = v1;
  v319 = v32;
  do
  {
    v87 = v84[3];
    v88 = *v83;
    v89 = *(_DWORD *)(v87 + 24);
    v82 = *(_QWORD *)(v87 + 8);
    if ( v89 )
    {
      v80 = v89 - 1;
      v90 = ((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4);
      v91 = v80 & v90;
      v92 = (__int64 *)(v82 + 16LL * ((unsigned int)v80 & v90));
      v81 = *v92;
      v79 = (__int64)v92;
      if ( v88 == *v92 )
      {
LABEL_126:
        if ( v319 == *(_QWORD *)(v79 + 8) )
        {
          if ( v301 != v88 && v299 != v88 )
          {
            v79 = 1;
            v85 = 0;
            if ( v88 == v81 )
            {
LABEL_121:
              v86 = v92 + 1;
LABEL_122:
              *v86 = v31;
              goto LABEL_123;
            }
            while ( v81 != -4096 )
            {
              if ( !v85 && v81 == -8192 )
                v85 = v92;
              v91 = v80 & (v79 + v91);
              v92 = (__int64 *)(v82 + 16LL * v91);
              v81 = *v92;
              if ( v88 == *v92 )
                goto LABEL_121;
              v79 = (unsigned int)(v79 + 1);
            }
            v79 = 2 * v89;
            if ( !v85 )
              v85 = v92;
            v238 = *(_DWORD *)(v87 + 16);
            ++*(_QWORD *)v87;
            v82 = (unsigned int)(v238 + 1);
            if ( 4 * (int)v82 >= 3 * v89 )
            {
              v297 = v84;
              sub_D4F150(v87, v79);
              v239 = *(_DWORD *)(v87 + 24);
              if ( !v239 )
                goto LABEL_493;
              v80 = (unsigned int)(v239 - 1);
              v79 = *(_QWORD *)(v87 + 8);
              v240 = v80 & v90;
              v84 = v297;
              v82 = (unsigned int)(*(_DWORD *)(v87 + 16) + 1);
              v85 = (__int64 *)(v79 + 16LL * v240);
              v241 = *v85;
              if ( v88 != *v85 )
              {
                v242 = 1;
                v81 = 0;
                while ( v241 != -4096 )
                {
                  if ( !v81 && v241 == -8192 )
                    v81 = (__int64)v85;
                  v240 = v80 & (v242 + v240);
                  v85 = (__int64 *)(v79 + 16LL * v240);
                  v241 = *v85;
                  if ( v88 == *v85 )
                    goto LABEL_373;
                  ++v242;
                }
LABEL_384:
                if ( v81 )
                  v85 = (__int64 *)v81;
              }
            }
            else
            {
              v81 = v89 >> 3;
              if ( v89 - *(_DWORD *)(v87 + 20) - (unsigned int)v82 <= (unsigned int)v81 )
              {
                v298 = v84;
                sub_D4F150(v87, v89);
                v243 = *(_DWORD *)(v87 + 24);
                if ( !v243 )
                {
LABEL_493:
                  ++*(_DWORD *)(v87 + 16);
                  BUG();
                }
                v80 = (unsigned int)(v243 - 1);
                v79 = *(_QWORD *)(v87 + 8);
                v81 = 0;
                v244 = v80 & v90;
                v84 = v298;
                v245 = 1;
                v82 = (unsigned int)(*(_DWORD *)(v87 + 16) + 1);
                v85 = (__int64 *)(v79 + 16LL * v244);
                v246 = *v85;
                if ( v88 != *v85 )
                {
                  while ( v246 != -4096 )
                  {
                    if ( !v81 && v246 == -8192 )
                      v81 = (__int64)v85;
                    v244 = v80 & (v245 + v244);
                    v85 = (__int64 *)(v79 + 16LL * v244);
                    v246 = *v85;
                    if ( v88 == *v85 )
                      goto LABEL_373;
                    ++v245;
                  }
                  goto LABEL_384;
                }
              }
            }
LABEL_373:
            *(_DWORD *)(v87 + 16) = v82;
            if ( *v85 != -4096 )
              --*(_DWORD *)(v87 + 20);
            *v85 = v88;
            v86 = v85 + 1;
            v85[1] = 0;
            goto LABEL_122;
          }
          v349 = (_BYTE *)*v83;
          v293 = v84;
          v93 = sub_282FA80(*(_QWORD **)(v31 + 32), *(_QWORD *)(v31 + 40), (__int64 *)&v349);
          sub_F681A0(v290, v93);
          v84 = v293;
          if ( *(_BYTE *)(v31 + 84) )
          {
            v94 = *(__int64 **)(v31 + 64);
            v81 = (__int64)&v94[*(unsigned int *)(v31 + 76)];
            v82 = *(unsigned int *)(v31 + 76);
            v95 = v94;
            if ( v94 != (__int64 *)v81 )
            {
              while ( v349 != (_BYTE *)*v95 )
              {
                if ( (__int64 *)v81 == ++v95 )
                  goto LABEL_123;
              }
              v82 = (unsigned int)(v82 - 1);
              *(_DWORD *)(v31 + 76) = v82;
              v81 = v94[v82];
              *v95 = v81;
              ++*(_QWORD *)(v31 + 56);
            }
          }
          else
          {
            v225 = sub_C8CA60(v289, (__int64)v349);
            v84 = v293;
            if ( v225 )
            {
              *v225 = -2;
              ++*(_DWORD *)(v31 + 80);
              ++*(_QWORD *)(v31 + 56);
            }
          }
        }
      }
      else
      {
        v296 = v80 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
        v222 = *v92;
        v79 = 1;
        while ( v222 != -4096 )
        {
          v223 = v79 + 1;
          v224 = (unsigned int)v80 & (v296 + (_DWORD)v79);
          v282 = v223;
          v296 = v224;
          v79 = v82 + 16 * v224;
          v222 = *(_QWORD *)v79;
          if ( v88 == *(_QWORD *)v79 )
            goto LABEL_126;
          v79 = v282;
        }
      }
    }
LABEL_123:
    ++v83;
  }
  while ( v328 != v83 );
  v32 = v319;
  v7 = v288;
  v1 = v84;
LABEL_136:
  v349 = (_BYTE *)v342;
  v96 = *(_BYTE **)(v32 + 40);
  if ( v96 == *(_BYTE **)(v32 + 48) )
  {
    sub_9319A0(v32 + 32, v96, &v349);
    v97 = (__int64)v349;
  }
  else
  {
    if ( v96 )
    {
      *(_QWORD *)v96 = v342;
      v96 = *(_BYTE **)(v32 + 40);
    }
    *(_QWORD *)(v32 + 40) = v96 + 8;
    v97 = v342;
  }
  if ( !*(_BYTE *)(v32 + 84) )
    goto LABEL_325;
  v98 = *(_QWORD **)(v32 + 64);
  v82 = *(unsigned int *)(v32 + 76);
  v81 = (__int64)&v98[v82];
  if ( v98 == (_QWORD *)v81 )
  {
LABEL_327:
    if ( (unsigned int)v82 < *(_DWORD *)(v32 + 72) )
    {
      *(_DWORD *)(v32 + 76) = v82 + 1;
      *(_QWORD *)v81 = v97;
      ++*(_QWORD *)(v32 + 56);
      goto LABEL_145;
    }
LABEL_325:
    sub_C8CC70(v32 + 56, v97, v81, v82, v79, v80);
    goto LABEL_145;
  }
  while ( *v98 != v97 )
  {
    if ( (_QWORD *)v81 == ++v98 )
      goto LABEL_327;
  }
LABEL_145:
  v99 = v1[3];
  v100 = *(_DWORD *)(v99 + 24);
  if ( !v100 )
  {
    ++*(_QWORD *)v99;
    goto LABEL_449;
  }
  v101 = *(_QWORD *)(v99 + 8);
  v102 = ((unsigned int)v342 >> 9) ^ ((unsigned int)v342 >> 4);
  v103 = (v100 - 1) & v102;
  v104 = (_QWORD *)(v101 + 16 * v103);
  v105 = *v104;
  if ( v342 != *v104 )
  {
    v248 = 1;
    v249 = 0;
    while ( v105 != -4096 )
    {
      if ( v105 == -8192 && !v249 )
        v249 = v104;
      LODWORD(v103) = (v100 - 1) & (v248 + v103);
      v104 = (_QWORD *)(v101 + 16LL * (unsigned int)v103);
      v105 = *v104;
      if ( v342 == *v104 )
        goto LABEL_147;
      ++v248;
    }
    v250 = *(_DWORD *)(v99 + 16);
    if ( !v249 )
      v249 = v104;
    ++*(_QWORD *)v99;
    v251 = v250 + 1;
    if ( 4 * (v250 + 1) < 3 * v100 )
    {
      if ( v100 - *(_DWORD *)(v99 + 20) - v251 > v100 >> 3 )
      {
LABEL_409:
        *(_DWORD *)(v99 + 16) = v251;
        if ( *v249 != -4096 )
          --*(_DWORD *)(v99 + 20);
        v249[1] = 0;
        v106 = v249 + 1;
        *(v106 - 1) = v342;
        goto LABEL_148;
      }
      sub_D4F150(v99, v100);
      v273 = *(_DWORD *)(v99 + 24);
      if ( v273 )
      {
        v274 = v273 - 1;
        v275 = *(_QWORD *)(v99 + 8);
        LODWORD(v276) = v274 & v102;
        v277 = 1;
        v251 = *(_DWORD *)(v99 + 16) + 1;
        v278 = 0;
        v249 = (_QWORD *)(v275 + 16LL * (v274 & v102));
        v279 = *v249;
        if ( v342 != *v249 )
        {
          while ( v279 != -4096 )
          {
            if ( v279 == -8192 && !v278 )
              v278 = v249;
            v276 = v274 & (unsigned int)(v276 + v277);
            v249 = (_QWORD *)(v275 + 16 * v276);
            v279 = *v249;
            if ( v342 == *v249 )
              goto LABEL_409;
            ++v277;
          }
          if ( v278 )
            v249 = v278;
        }
        goto LABEL_409;
      }
LABEL_496:
      ++*(_DWORD *)(v99 + 16);
      BUG();
    }
LABEL_449:
    sub_D4F150(v99, 2 * v100);
    v266 = *(_DWORD *)(v99 + 24);
    if ( v266 )
    {
      v267 = v266 - 1;
      v268 = *(_QWORD *)(v99 + 8);
      v251 = *(_DWORD *)(v99 + 16) + 1;
      LODWORD(v269) = v267 & (((unsigned int)v342 >> 9) ^ ((unsigned int)v342 >> 4));
      v249 = (_QWORD *)(v268 + 16LL * (unsigned int)v269);
      v270 = *v249;
      if ( v342 != *v249 )
      {
        v271 = 1;
        v272 = 0;
        while ( v270 != -4096 )
        {
          if ( !v272 && v270 == -8192 )
            v272 = v249;
          v269 = v267 & (unsigned int)(v269 + v271);
          v249 = (_QWORD *)(v268 + 16 * v269);
          v270 = *v249;
          if ( v342 == *v249 )
            goto LABEL_409;
          ++v271;
        }
        if ( v272 )
          v249 = v272;
      }
      goto LABEL_409;
    }
    goto LABEL_496;
  }
LABEL_147:
  v106 = v104 + 1;
LABEL_148:
  *v106 = v32;
  sub_DAC210(v1[2], v32);
  if ( v352 != (__int64 *)dest )
    _libc_free((unsigned __int64)v352);
  v302 = v1[3];
  v300 = v1[1];
  v294 = sub_D47470(v300);
  v107 = sub_AA5930(v309);
  v329 = v108;
  v109 = v107;
  while ( v109 != v329 )
  {
    if ( !v109 )
      goto LABEL_495;
    v110 = *(_QWORD *)(v109 + 32);
    if ( !v110 )
      goto LABEL_497;
    v111 = 0;
    if ( *(_BYTE *)(v110 - 24) == 84 )
      v111 = v110 - 24;
    v112 = *(_QWORD *)(v109 - 8);
    v113 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v109 + 4) & 0x7FFFFFF) != 0 )
    {
      v114 = 0;
      do
      {
        if ( v7 == *(_QWORD *)(v112 + 32LL * *(unsigned int *)(v109 + 72) + 8 * v114) )
        {
          v113 = 32 * v114;
          goto LABEL_161;
        }
        ++v114;
      }
      while ( (*(_DWORD *)(v109 + 4) & 0x7FFFFFF) != (_DWORD)v114 );
      v113 = 0x1FFFFFFFE0LL;
    }
LABEL_161:
    v115 = *(_QWORD *)(v112 + v113);
    v116 = v115;
    while ( *(_BYTE *)v116 == 84 && (*(_DWORD *)(v116 + 4) & 0x7FFFFFF) == 1 )
    {
      v116 = **(_QWORD **)(v116 - 8);
      if ( !v116 )
        goto LABEL_497;
    }
    v117 = *(_QWORD *)(v116 + 40);
    if ( v7 == v117 || v339 == v117 )
    {
      sub_BD84D0(v109, v115);
      v118 = (_QWORD *)v109;
      v109 = v111;
      sub_B43D60(v118);
    }
    else
    {
      v109 = v111;
    }
  }
  v349 = v351;
  v350 = 0x800000000LL;
  v119 = sub_AA5930(v309);
  v123 = v122;
  v124 = (unsigned int)v350;
  if ( v119 != v122 )
  {
    while ( 1 )
    {
      if ( v124 + 1 > (unsigned __int64)HIDWORD(v350) )
      {
        sub_C8D5F0((__int64)&v349, v351, v124 + 1, 8u, v120, v121);
        v124 = (unsigned int)v350;
      }
      *(_QWORD *)&v349[8 * v124] = v119;
      v124 = (unsigned int)(v350 + 1);
      LODWORD(v350) = v350 + 1;
      if ( !v119 )
        break;
      v125 = *(_QWORD *)(v119 + 32);
      if ( !v125 )
        goto LABEL_497;
      v119 = 0;
      if ( *(_BYTE *)(v125 - 24) == 84 )
        v119 = v125 - 24;
      if ( v123 == v119 )
        goto LABEL_179;
    }
LABEL_495:
    BUG();
  }
LABEL_179:
  v352 = (__int64 *)dest;
  v353 = 0x800000000LL;
  v126 = sub_AA5930(v7);
  v130 = v129;
  v131 = (unsigned int)v353;
  while ( v130 != v126 )
  {
    if ( v131 + 1 > (unsigned __int64)HIDWORD(v353) )
    {
      sub_C8D5F0((__int64)&v352, dest, v131 + 1, 8u, v127, v128);
      v131 = (unsigned int)v353;
    }
    v352[v131] = v126;
    v131 = (unsigned int)(v353 + 1);
    LODWORD(v353) = v353 + 1;
    if ( !v126 )
      goto LABEL_495;
    v132 = *(_QWORD *)(v126 + 32);
    if ( !v132 )
      goto LABEL_497;
    v126 = 0;
    if ( *(_BYTE *)(v132 - 24) == 84 )
      v126 = v132 - 24;
  }
  if ( v349 != &v349[8 * (unsigned int)v350] )
  {
    v330 = (_QWORD **)&v349[8 * (unsigned int)v350];
    v133 = v286;
    v134 = (_QWORD **)v349;
    do
    {
      v135 = *v134;
      LOBYTE(v133) = 1;
      v137 = sub_AA4FF0(v7);
      v138 = 0;
      if ( v137 )
        v138 = v136;
      ++v134;
      BYTE1(v133) = v138;
      sub_B444E0(v135, v137, v133);
    }
    while ( v330 != v134 );
  }
  if ( v352 != &v352[(unsigned int)v353] )
  {
    v139 = v309;
    v310 = v1;
    v140 = v352;
    v331 = &v352[(unsigned int)v353];
    v141 = v287;
    do
    {
      v142 = (_QWORD *)*v140;
      LOBYTE(v141) = 1;
      v144 = sub_AA4FF0(v139);
      v145 = 0;
      if ( v144 )
        v145 = v143;
      ++v140;
      BYTE1(v141) = v145;
      sub_B444E0(v142, v144, v141);
    }
    while ( v331 != v140 );
    v1 = v310;
  }
  if ( v294 )
  {
    v147 = sub_AA5930(v294);
    if ( v147 != v146 )
    {
      v291 = v1;
      v148 = v283;
      v332 = v146;
      while ( 1 )
      {
        if ( (*(_DWORD *)(v147 + 4) & 0x7FFFFFF) == 1 )
        {
          v150 = **(_QWORD **)(v147 - 8);
          if ( *(_BYTE *)v150 > 0x1Cu )
            break;
        }
LABEL_201:
        v149 = *(_QWORD *)(v147 + 32);
        if ( !v149 )
          goto LABEL_497;
        v147 = 0;
        if ( *(_BYTE *)(v149 - 24) == 84 )
          v147 = v149 - 24;
        if ( v332 == v147 )
        {
          v1 = v291;
          goto LABEL_247;
        }
      }
      v151 = *(_QWORD *)(v150 + 40);
      v152 = *(_DWORD *)(v302 + 24);
      v153 = *(_QWORD *)(v302 + 8);
      if ( v152 )
      {
        v154 = v152 - 1;
        v155 = v154 & (((unsigned int)v151 >> 9) ^ ((unsigned int)v151 >> 4));
        v156 = (__int64 *)(v153 + 16LL * v155);
        v157 = *v156;
        if ( *v156 == v151 )
        {
LABEL_209:
          v158 = v156[1];
          goto LABEL_210;
        }
        v235 = 1;
        while ( v157 != -4096 )
        {
          v236 = v235 + 1;
          v155 = v154 & (v235 + v155);
          v156 = (__int64 *)(v153 + 16LL * v155);
          v157 = *v156;
          if ( v151 == *v156 )
            goto LABEL_209;
          v235 = v236;
        }
      }
      v158 = 0;
LABEL_210:
      if ( v300 != v158 )
      {
        v159 = sub_B47F80((_BYTE *)v147);
        if ( *(_BYTE *)v159 != 84 )
          v159 = 0;
        v160 = **(_QWORD **)(v147 - 8);
        v161 = *(__int64 **)(v159 - 8);
        v162 = *v161;
        if ( v160 )
        {
          if ( v162 )
          {
            v163 = v161[1];
            *(_QWORD *)v161[2] = v163;
            if ( v163 )
              *(_QWORD *)(v163 + 16) = v161[2];
          }
          *v161 = v160;
          v164 = *(_QWORD *)(v160 + 16);
          v161[1] = v164;
          if ( v164 )
            *(_QWORD *)(v164 + 16) = v161 + 1;
          v161[2] = v160 + 16;
          *(_QWORD *)(v160 + 16) = v161;
          v161 = *(__int64 **)(v159 - 8);
        }
        else if ( v162 )
        {
          v237 = v161[1];
          *(_QWORD *)v161[2] = v237;
          if ( v237 )
            *(_QWORD *)(v237 + 16) = v161[2];
          *v161 = 0;
          v161 = *(__int64 **)(v159 - 8);
        }
        v161[4 * *(unsigned int *)(v159 + 72)] = v338;
        v165 = *(_QWORD *)(v7 + 16);
        if ( v165 )
        {
          while ( 1 )
          {
            v166 = *(_QWORD *)(v165 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v166 - 30) <= 0xAu )
              break;
            v165 = *(_QWORD *)(v165 + 8);
            if ( !v165 )
              goto LABEL_236;
          }
LABEL_224:
          v167 = *(_QWORD *)(v166 + 40);
          if ( v338 != v167 )
          {
            v168 = **(_QWORD **)(v147 - 8);
            v169 = *(_DWORD *)(v159 + 4) & 0x7FFFFFF;
            if ( v169 == *(_DWORD *)(v159 + 72) )
            {
              v295 = **(_QWORD **)(v147 - 8);
              v311 = v167;
              sub_B48D90(v159);
              v168 = v295;
              v167 = v311;
              v169 = *(_DWORD *)(v159 + 4) & 0x7FFFFFF;
            }
            v170 = (v169 + 1) & 0x7FFFFFF;
            v171 = v170 | *(_DWORD *)(v159 + 4) & 0xF8000000;
            v172 = *(_QWORD *)(v159 - 8) + 32LL * (unsigned int)(v170 - 1);
            *(_DWORD *)(v159 + 4) = v171;
            if ( *(_QWORD *)v172 )
            {
              v173 = *(_QWORD *)(v172 + 8);
              **(_QWORD **)(v172 + 16) = v173;
              if ( v173 )
                *(_QWORD *)(v173 + 16) = *(_QWORD *)(v172 + 16);
            }
            *(_QWORD *)v172 = v168;
            if ( v168 )
            {
              v174 = *(_QWORD *)(v168 + 16);
              *(_QWORD *)(v172 + 8) = v174;
              if ( v174 )
                *(_QWORD *)(v174 + 16) = v172 + 8;
              *(_QWORD *)(v172 + 16) = v168 + 16;
              *(_QWORD *)(v168 + 16) = v172;
            }
            *(_QWORD *)(*(_QWORD *)(v159 - 8)
                      + 32LL * *(unsigned int *)(v159 + 72)
                      + 8LL * ((*(_DWORD *)(v159 + 4) & 0x7FFFFFFu) - 1)) = v167;
          }
          while ( 1 )
          {
            v165 = *(_QWORD *)(v165 + 8);
            if ( !v165 )
              break;
            v166 = *(_QWORD *)(v165 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v166 - 30) <= 0xAu )
              goto LABEL_224;
          }
        }
LABEL_236:
        LOBYTE(v148) = 1;
        v175 = sub_AA4FF0(v7);
        v177 = v176;
        if ( !v175 )
          v177 = 0;
        BYTE1(v148) = v177;
        sub_B44220((_QWORD *)v159, v175, v148);
        v178 = *(_QWORD *)(v147 - 8);
        if ( *(_QWORD *)v178 )
        {
          v179 = *(_QWORD *)(v178 + 8);
          **(_QWORD **)(v178 + 16) = v179;
          if ( v179 )
            *(_QWORD *)(v179 + 16) = *(_QWORD *)(v178 + 16);
        }
        *(_QWORD *)v178 = v159;
        v180 = *(_QWORD *)(v159 + 16);
        *(_QWORD *)(v178 + 8) = v180;
        if ( v180 )
          *(_QWORD *)(v180 + 16) = v178 + 8;
        *(_QWORD *)(v178 + 16) = v159 + 16;
        *(_QWORD *)(v159 + 16) = v178;
      }
      goto LABEL_201;
    }
  }
LABEL_247:
  sub_AA5D60(v7, v7, v338);
  if ( v352 != (__int64 *)dest )
    _libc_free((unsigned __int64)v352);
  if ( v349 != v351 )
    _libc_free((unsigned __int64)v349);
  sub_AA5D60(v292, v338, v7);
  v181 = v1[5];
  v346 = v348;
  v349 = v351;
  v347 = 0x400000000LL;
  v350 = 0x400000000LL;
  v185 = sub_AA5930(v339);
  if ( v185 != v182 )
  {
    v333 = v1;
    v186 = v182;
    while ( 1 )
    {
      if ( *(_BYTE *)(v181 + 60) )
      {
        v187 = *(_QWORD **)(v181 + 40);
        v188 = &v187[*(unsigned int *)(v181 + 52)];
        if ( v187 != v188 )
        {
          while ( v185 != *v187 )
          {
            if ( v188 == ++v187 )
              goto LABEL_261;
          }
LABEL_258:
          v189 = (unsigned int)v347;
          v190 = (unsigned int)v347 + 1LL;
          if ( v190 > HIDWORD(v347) )
          {
            sub_C8D5F0((__int64)&v346, v348, v190, 8u, v183, v184);
            v189 = (unsigned int)v347;
          }
          *(_QWORD *)&v346[8 * v189] = v185;
          LODWORD(v347) = v347 + 1;
        }
      }
      else if ( sub_C8CA60(v181 + 32, v185) )
      {
        goto LABEL_258;
      }
LABEL_261:
      if ( !v185 )
        goto LABEL_495;
      v191 = *(_QWORD *)(v185 + 32);
      if ( !v191 )
        goto LABEL_497;
      v185 = 0;
      if ( *(_BYTE *)(v191 - 24) == 84 )
        v185 = v191 - 24;
      if ( v186 == v185 )
      {
        v1 = v333;
        break;
      }
    }
  }
  v195 = sub_AA5930(v340);
  if ( v192 == v195 )
    goto LABEL_283;
  v334 = v1;
  v196 = v192;
  while ( 2 )
  {
    if ( *(_BYTE *)(v181 + 60) )
    {
      v197 = *(_QWORD **)(v181 + 40);
      v198 = &v197[*(unsigned int *)(v181 + 52)];
      if ( v197 == v198 )
        goto LABEL_277;
      while ( v195 != *v197 )
      {
        if ( v198 == ++v197 )
          goto LABEL_277;
      }
    }
    else if ( !sub_C8CA60(v181 + 32, v195) )
    {
      goto LABEL_277;
    }
    v199 = (unsigned int)v350;
    v200 = (unsigned int)v350 + 1LL;
    if ( v200 > HIDWORD(v350) )
    {
      sub_C8D5F0((__int64)&v349, v351, v200, 8u, v193, v194);
      v199 = (unsigned int)v350;
    }
    *(_QWORD *)&v349[8 * v199] = v195;
    LODWORD(v350) = v350 + 1;
LABEL_277:
    if ( !v195 )
      goto LABEL_495;
    v201 = *(_QWORD *)(v195 + 32);
    if ( !v201 )
      goto LABEL_497;
    v195 = 0;
    if ( *(_BYTE *)(v201 - 24) == 84 )
      v195 = v201 - 24;
    if ( v196 != v195 )
      continue;
    break;
  }
  v1 = v334;
LABEL_283:
  v202 = (_QWORD **)v349;
  if ( &v349[8 * (unsigned int)v350] != v349 )
  {
    v303 = v1;
    v335 = (_QWORD **)&v349[8 * (unsigned int)v350];
    v203 = v284;
    do
    {
      v204 = *v202;
      LOBYTE(v203) = 1;
      v206 = sub_AA4FF0(v339);
      v207 = 0;
      if ( v206 )
        v207 = v205;
      ++v202;
      BYTE1(v203) = v207;
      sub_B444E0(v204, v206, v203);
    }
    while ( v335 != v202 );
    v1 = v303;
  }
  v208 = (_QWORD **)v346;
  if ( &v346[8 * (unsigned int)v347] != v346 )
  {
    v304 = v1;
    v336 = (_QWORD **)&v346[8 * (unsigned int)v347];
    v209 = v285;
    do
    {
      v210 = *v208;
      LOBYTE(v209) = 1;
      v212 = sub_AA4FF0(v340);
      v213 = 0;
      if ( v212 )
        v213 = v211;
      ++v208;
      BYTE1(v209) = v213;
      sub_B444E0(v210, v212, v209);
    }
    while ( v336 != v208 );
    v1 = v304;
  }
  sub_AA5D60(v340, v341, v342);
  sub_AA5D60(v340, v7, v338);
  sub_AA5D60(v339, v342, v341);
  sub_AA5D60(v339, v338, v7);
  v215 = 4;
  v352 = (__int64 *)dest;
  v353 = 0x400000000LL;
  v216 = 0;
  v217 = *(_QWORD *)(v340 + 56);
  v218 = *(_QWORD *)(v340 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v217 != v218 )
  {
    while ( 1 )
    {
      v219 = v217 - 24;
      v220 = v216 + 1;
      if ( !v217 )
        v219 = 0;
      if ( v220 > v215 )
      {
        sub_C8D5F0((__int64)&v352, dest, v216 + 1, 8u, v220, v214);
        v216 = (unsigned int)v353;
      }
      v352[v216] = v219;
      v216 = (unsigned int)(v353 + 1);
      LODWORD(v353) = v353 + 1;
      v217 = *(_QWORD *)(v217 + 8);
      if ( v217 == v218 )
        break;
      v215 = HIDWORD(v353);
    }
  }
  sub_11D0BA0((__int64)&v352, v1[4], v1[3], v1[2], 0, 0);
  if ( v352 != (__int64 *)dest )
    _libc_free((unsigned __int64)v352);
  if ( v349 != v351 )
    _libc_free((unsigned __int64)v349);
  if ( v346 != v348 )
    _libc_free((unsigned __int64)v346);
  v53 = 1;
LABEL_85:
  if ( v343 )
    j_j___libc_free_0((unsigned __int64)v343);
  return v53;
}
