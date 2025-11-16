// Function: sub_2D4C710
// Address: 0x2d4c710
//
__int64 __fastcall sub_2D4C710(unsigned int *a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned __int8 v4; // al
  unsigned int v5; // eax
  unsigned int v6; // r14d
  unsigned int v8; // eax
  _QWORD **v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // edx
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned __int16 v19; // cx
  unsigned __int64 v20; // rax
  _QWORD **v21; // rbx
  unsigned int v22; // eax
  unsigned int v23; // r8d
  unsigned __int16 v24; // si
  unsigned __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned int v30; // edx
  unsigned __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v35; // edx
  unsigned __int16 v36; // cx
  unsigned __int64 v37; // rax
  _QWORD **v38; // rdi
  _QWORD *v39; // rdx
  __int64 (__fastcall *v40)(__int64, __int64); // rax
  unsigned __int8 v41; // al
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 (*v45)(); // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 (*v48)(); // rax
  __int64 (*v49)(); // rax
  int v50; // eax
  __int64 v51; // rdx
  __int16 v52; // cx
  __int16 v53; // ax
  unsigned int v54; // r12d
  bool v55; // al
  __int64 (*v56)(); // rax
  int v57; // r8d
  __int64 v58; // rax
  __int64 (*v59)(); // rax
  unsigned __int16 v60; // ax
  __int64 v61; // rdx
  unsigned __int16 v62; // si
  __int16 v63; // cx
  __int64 (*v64)(); // r9
  __int16 v65; // ax
  __int16 v66; // ax
  __int64 (__fastcall *v67)(__int64, __int64); // rax
  unsigned __int8 v68; // al
  __int64 v69; // rax
  int *v70; // rbx
  unsigned __int64 v71; // rax
  unsigned __int16 v72; // cx
  unsigned int v73; // edx
  __int64 (__fastcall *v74)(__int64, __int64); // rax
  unsigned __int8 v75; // cl
  __int64 v76; // rax
  __int64 v77; // r13
  __int64 v78; // r12
  __int16 v79; // r9
  unsigned __int64 v80; // rax
  __int16 v81; // bx
  _QWORD *v82; // rax
  _QWORD *v83; // r14
  unsigned int *v84; // rbx
  unsigned int *v85; // r12
  __int64 v86; // rdx
  unsigned int v87; // esi
  unsigned int v88; // r13d
  unsigned int v89; // r12d
  __int64 (*v90)(); // rax
  int v91; // ecx
  _QWORD *v92; // rax
  unsigned int v93; // r12d
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // rsi
  __int64 (*v97)(); // rax
  _QWORD *v98; // r14
  __int64 v99; // r13
  __int64 (*v100)(); // rax
  __int16 v101; // dx
  __int16 v102; // cx
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  unsigned __int64 v110; // rax
  _QWORD *v111; // r13
  unsigned int *v112; // r14
  unsigned int *v113; // rbx
  __int64 v114; // rdx
  unsigned int v115; // esi
  __int64 v116; // rdi
  __int64 v117; // r13
  __int64 v118; // rbx
  _QWORD *v119; // r14
  _QWORD *v120; // rax
  __int64 v121; // r9
  __int64 v122; // r13
  unsigned int *v123; // r14
  unsigned int *v124; // rbx
  __int64 v125; // rdx
  unsigned int v126; // esi
  _QWORD *v127; // rax
  __int64 v128; // r13
  unsigned int *v129; // r14
  unsigned int *v130; // rbx
  __int64 v131; // rdx
  unsigned int v132; // esi
  __int64 v133; // rdx
  __int64 v134; // rax
  __int64 v135; // rbx
  __int64 v136; // rax
  __int64 v137; // r13
  _QWORD *v138; // r14
  __int64 v139; // rbx
  bool v140; // zf
  _QWORD *v141; // rax
  __int64 v142; // r13
  unsigned int *v143; // r14
  unsigned int *v144; // rbx
  __int64 v145; // rdx
  unsigned int v146; // esi
  __int64 v147; // rbx
  __int64 v148; // r13
  _QWORD *v149; // rax
  __int64 v150; // r9
  __int64 v151; // r13
  unsigned int *v152; // r14
  unsigned int *v153; // rbx
  __int64 v154; // rdx
  unsigned int v155; // esi
  __int64 (*v156)(); // rax
  _QWORD *v157; // rax
  __int64 v158; // r13
  unsigned int *v159; // r14
  unsigned int *v160; // rbx
  __int64 v161; // rdx
  unsigned int v162; // esi
  void (*v163)(); // rax
  _QWORD *v164; // rax
  __int64 v165; // r13
  unsigned int *v166; // r14
  unsigned int *v167; // rbx
  __int64 v168; // rdx
  unsigned int v169; // esi
  _QWORD *v170; // rax
  __int64 v171; // r13
  unsigned int *v172; // r15
  unsigned int *v173; // rbx
  __int64 v174; // rdx
  unsigned int v175; // esi
  _BYTE *v176; // r13
  __int64 v177; // rax
  __int64 v178; // r14
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // r15
  __int64 v182; // rbx
  __int64 v183; // r8
  __int64 v184; // r9
  __int64 v185; // rax
  unsigned __int64 v186; // rdx
  char *v187; // rbx
  char *v188; // r15
  _QWORD *v189; // rdi
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // rax
  int v193; // eax
  _QWORD *v194; // rax
  __int64 v195; // r13
  unsigned int *v196; // r14
  unsigned int *v197; // rbx
  __int64 v198; // rdx
  unsigned int v199; // esi
  __int64 v200; // rax
  int v201; // ecx
  int v202; // edx
  _QWORD *v203; // rdi
  __int64 *v204; // rax
  __int64 v205; // rsi
  unsigned int *v206; // r13
  unsigned int *v207; // rbx
  __int64 v208; // rdx
  unsigned int v209; // esi
  __int64 v210; // rax
  int v211; // ecx
  int v212; // edx
  _QWORD *v213; // rdi
  __int64 *v214; // rax
  __int64 v215; // rsi
  unsigned int *v216; // r13
  unsigned int *v217; // rbx
  __int64 v218; // rdx
  unsigned int v219; // esi
  __int64 v220; // rax
  int v221; // ecx
  int v222; // edx
  _QWORD *v223; // rdi
  __int64 *v224; // rax
  __int64 v225; // rsi
  unsigned int *v226; // r13
  unsigned int *v227; // rbx
  __int64 v228; // rdx
  unsigned int v229; // esi
  int v230; // eax
  _QWORD **v231; // rbx
  unsigned __int64 v232; // rsi
  unsigned __int64 v233; // rcx
  __int64 v234; // rax
  __int64 v235; // rcx
  __int64 v236; // r14
  __int64 v237; // rbx
  __int64 v238; // rbx
  __int64 v239; // r10
  unsigned __int16 v240; // cx
  unsigned __int64 v241; // rax
  __int16 v242; // bx
  __int16 v243; // r14
  unsigned int *v244; // r14
  unsigned int *v245; // r12
  __int64 v246; // rdx
  unsigned int v247; // esi
  __int16 v248; // ax
  __int64 v249; // r14
  __int64 v250; // rax
  __int64 v251; // rbx
  __int64 v252; // r12
  __int64 v253; // rax
  __int64 v254; // rbx
  __int64 *v255; // r14
  __int64 v256; // r12
  _QWORD *v257; // rax
  unsigned int *v258; // r14
  unsigned int *v259; // rbx
  __int64 v260; // rdx
  unsigned int v261; // esi
  __int64 *v262; // rax
  unsigned int *v263; // r12
  unsigned int *v264; // rbx
  __int64 v265; // rdx
  unsigned int v266; // esi
  int v267; // r14d
  unsigned int *v268; // r14
  unsigned int *v269; // rbx
  __int64 v270; // rdx
  unsigned int v271; // esi
  char v272; // al
  __int64 v273; // r10
  int v274; // r14d
  unsigned int *v275; // r13
  unsigned int *v276; // r12
  __int64 v277; // rbx
  __int64 v278; // rdx
  unsigned int v279; // esi
  int v280; // ebx
  __int64 v281; // rbx
  unsigned int *v282; // r13
  unsigned int *v283; // r12
  __int64 v284; // rdx
  unsigned int v285; // esi
  __int64 v286; // [rsp-10h] [rbp-2C0h]
  __int64 v287; // [rsp+8h] [rbp-2A8h]
  char v288; // [rsp+20h] [rbp-290h]
  __int64 v289; // [rsp+20h] [rbp-290h]
  unsigned int v290; // [rsp+28h] [rbp-288h]
  unsigned __int16 v291; // [rsp+2Ch] [rbp-284h]
  char v292; // [rsp+2Fh] [rbp-281h]
  __int64 v293; // [rsp+30h] [rbp-280h]
  __int64 v294; // [rsp+38h] [rbp-278h]
  __int64 v295; // [rsp+40h] [rbp-270h]
  __int64 v296; // [rsp+50h] [rbp-260h]
  __int64 v297; // [rsp+58h] [rbp-258h]
  __int64 v298; // [rsp+60h] [rbp-250h]
  char v299; // [rsp+68h] [rbp-248h]
  __int64 v300; // [rsp+68h] [rbp-248h]
  __int64 v301; // [rsp+68h] [rbp-248h]
  unsigned __int8 v302; // [rsp+70h] [rbp-240h]
  __int64 v303; // [rsp+70h] [rbp-240h]
  char v304; // [rsp+70h] [rbp-240h]
  unsigned int v305; // [rsp+78h] [rbp-238h]
  unsigned int v306; // [rsp+78h] [rbp-238h]
  _QWORD *v307; // [rsp+78h] [rbp-238h]
  __int64 v308; // [rsp+78h] [rbp-238h]
  int v309; // [rsp+78h] [rbp-238h]
  __int64 v310; // [rsp+80h] [rbp-230h]
  __int64 v311; // [rsp+80h] [rbp-230h]
  unsigned int v312; // [rsp+88h] [rbp-228h]
  __int64 v313; // [rsp+88h] [rbp-228h]
  __int64 v314; // [rsp+88h] [rbp-228h]
  __int64 v315; // [rsp+88h] [rbp-228h]
  __int64 v316; // [rsp+90h] [rbp-220h]
  __int64 *v317; // [rsp+90h] [rbp-220h]
  __int64 v318; // [rsp+90h] [rbp-220h]
  unsigned int v319; // [rsp+98h] [rbp-218h]
  __int64 v320; // [rsp+98h] [rbp-218h]
  unsigned __int8 v321; // [rsp+98h] [rbp-218h]
  int v322; // [rsp+98h] [rbp-218h]
  _QWORD *v323; // [rsp+A0h] [rbp-210h]
  __int64 v324; // [rsp+A0h] [rbp-210h]
  __int64 v325; // [rsp+A0h] [rbp-210h]
  __int64 v326; // [rsp+A0h] [rbp-210h]
  __int64 v327; // [rsp+A0h] [rbp-210h]
  __int64 v328; // [rsp+A0h] [rbp-210h]
  __int64 v329; // [rsp+A8h] [rbp-208h]
  char v330; // [rsp+A8h] [rbp-208h]
  unsigned __int16 v331; // [rsp+A8h] [rbp-208h]
  int v332; // [rsp+B4h] [rbp-1FCh] BYREF
  __int64 v333; // [rsp+B8h] [rbp-1F8h]
  __int64 v334; // [rsp+C0h] [rbp-1F0h]
  __int64 v335; // [rsp+C8h] [rbp-1E8h]
  char *v336; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v337; // [rsp+D8h] [rbp-1D8h]
  _BYTE v338[16]; // [rsp+E0h] [rbp-1D0h] BYREF
  char v339; // [rsp+F0h] [rbp-1C0h]
  char v340; // [rsp+F1h] [rbp-1BFh]
  _QWORD v341[4]; // [rsp+100h] [rbp-1B0h] BYREF
  __int16 v342; // [rsp+120h] [rbp-190h]
  __int64 v343; // [rsp+130h] [rbp-180h] BYREF
  __int64 v344; // [rsp+138h] [rbp-178h]
  __int64 v345; // [rsp+148h] [rbp-168h]
  __int16 v346; // [rsp+150h] [rbp-160h]
  char *v347; // [rsp+170h] [rbp-140h] BYREF
  __int64 v348; // [rsp+178h] [rbp-138h]
  _BYTE v349[16]; // [rsp+180h] [rbp-130h] BYREF
  char v350; // [rsp+190h] [rbp-120h]
  char v351; // [rsp+191h] [rbp-11Fh]
  _QWORD *v352; // [rsp+1A0h] [rbp-110h]
  _QWORD *v353; // [rsp+1A8h] [rbp-108h]
  __int64 v354; // [rsp+1B0h] [rbp-100h]
  __int64 v355; // [rsp+1B8h] [rbp-F8h]
  void **v356; // [rsp+1C0h] [rbp-F0h]
  void **v357; // [rsp+1C8h] [rbp-E8h]
  __int64 v358; // [rsp+1D0h] [rbp-E0h]
  int v359; // [rsp+1D8h] [rbp-D8h]
  __int16 v360; // [rsp+1DCh] [rbp-D4h]
  char v361; // [rsp+1DEh] [rbp-D2h]
  __int64 v362; // [rsp+1E0h] [rbp-D0h]
  __int64 v363; // [rsp+1E8h] [rbp-C8h]
  void *v364; // [rsp+1F0h] [rbp-C0h] BYREF
  void *v365; // [rsp+1F8h] [rbp-B8h] BYREF
  _QWORD v366[12]; // [rsp+250h] [rbp-60h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 61 )
  {
    if ( v4 == 62 )
    {
      LOBYTE(v8) = sub_B46500((unsigned __int8 *)a2);
      v6 = v8;
      if ( !(_BYTE)v8 )
        return v6;
      v9 = *(_QWORD ***)a1;
      v10 = sub_B43CC0(a2);
      v11 = sub_9208B0(v10, *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL));
      v348 = v12;
      v347 = (char *)((unsigned __int64)(v11 + 7) >> 3);
      v13 = sub_CA1930(&v347);
      _BitScanReverse64(&v14, 1LL << (*(_WORD *)(a2 + 2) >> 1));
      if ( 0x8000000000000000LL >> ((unsigned __int8)v14 ^ 0x3Fu) < v13 || *((_DWORD *)v9 + 21) >> 3 < v13 )
      {
        v15 = sub_B43CC0(a2);
        v16 = sub_9208B0(v15, *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL));
        v348 = v17;
        v347 = (char *)((unsigned __int64)(v16 + 7) >> 3);
        v18 = sub_CA1930(&v347);
        v19 = *(_WORD *)(a2 + 2);
        _BitScanReverse64(&v20, 1LL << (v19 >> 1));
        if ( !sub_2D4A460(
                a1,
                a2,
                v18,
                63 - ((unsigned __int8)v20 ^ 0x3Fu),
                *(_QWORD *)(a2 - 32),
                *(_QWORD *)(a2 - 64),
                0,
                (v19 >> 7) & 7,
                0,
                dword_444C430) )
          sub_C64ED0("expandAtomicOpToLibcall shouldn't fail for Store", 1u);
        return 1;
      }
      v38 = *(_QWORD ***)a1;
      v39 = **(_QWORD ***)a1;
      v67 = (__int64 (__fastcall *)(__int64, __int64))v39[146];
      if ( v67 == sub_2D42F00 )
      {
        v68 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL) + 8LL);
        if ( v68 <= 3u || v68 == 5 || (v68 & 0xFD) == 4 )
          goto LABEL_75;
      }
      else
      {
        if ( (unsigned int)v67((__int64)v38, a2) == 1 )
        {
LABEL_75:
          v42 = 0;
          v44 = 0;
          v69 = sub_2D47210((__int64)a1, a2);
          v38 = *(_QWORD ***)a1;
          v323 = 0;
          v329 = v69;
          v3 = v69;
          v39 = **(_QWORD ***)a1;
          goto LABEL_20;
        }
        v38 = *(_QWORD ***)a1;
        v39 = **(_QWORD ***)a1;
      }
      v329 = a2;
      v6 = 0;
      v42 = 0;
      v44 = 0;
      v323 = 0;
      goto LABEL_20;
    }
    if ( v4 != 66 )
    {
      if ( v4 != 65 )
        return 0;
      v231 = *(_QWORD ***)a1;
      v232 = (unsigned int)sub_2D44290(a2);
      _BitScanReverse64(&v233, 1LL << *(_BYTE *)(v3 + 3));
      if ( v232 > 0x8000000000000000LL >> ((unsigned __int8)v233 ^ 0x3Fu)
        || (unsigned int)v232 > *((_DWORD *)v231 + 21) >> 3 )
      {
        v6 = 1;
        sub_2D4B8E0(a1, v3);
        return v6;
      }
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 - 64) + 8LL) + 8LL) == 14 )
      {
        v234 = sub_B43CA0(v3);
        v236 = sub_2D43EB0(*(_QWORD *)a1, *(__int64 **)(*(_QWORD *)(v3 - 64) + 8LL), v234 + 312, v235);
        sub_2D46B10((__int64)&v347, v3, *((_QWORD *)a1 + 1));
        v237 = *(_QWORD *)(v3 - 64);
        v314 = *(_QWORD *)(v3 - 96);
        v342 = 257;
        if ( v236 == *(_QWORD *)(v237 + 8) )
        {
          v318 = v237;
        }
        else
        {
          v318 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v356 + 15))(v356, 47, v237, v236);
          if ( !v318 )
          {
            v346 = 257;
            v318 = sub_B51D30(47, v237, v236, (__int64)&v343, 0, 0);
            if ( (unsigned __int8)sub_920620(v318) )
            {
              v280 = v359;
              if ( v358 )
                sub_B99FD0(v318, 3u, v358);
              sub_B45150(v318, v280);
            }
            (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
              v357,
              v318,
              v341,
              v353,
              v354);
            v281 = 16LL * (unsigned int)v348;
            if ( v347 != &v347[v281] )
            {
              v328 = v3;
              v282 = (unsigned int *)v347;
              v283 = (unsigned int *)&v347[v281];
              do
              {
                v284 = *((_QWORD *)v282 + 1);
                v285 = *v282;
                v282 += 4;
                sub_B99FD0(v318, v285, v284);
              }
              while ( v283 != v282 );
              v3 = v328;
            }
          }
        }
        v238 = *(_QWORD *)(v3 - 32);
        v342 = 257;
        if ( v236 == *(_QWORD *)(v238 + 8) )
        {
          v239 = v238;
        }
        else
        {
          v239 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v356 + 15))(v356, 47, v238, v236);
          if ( !v239 )
          {
            v346 = 257;
            v324 = sub_B51D30(47, v238, v236, (__int64)&v343, 0, 0);
            v272 = sub_920620(v324);
            v273 = v324;
            if ( v272 )
            {
              v274 = v359;
              if ( v358 )
              {
                sub_B99FD0(v324, 3u, v358);
                v273 = v324;
              }
              v325 = v273;
              sub_B45150(v273, v274);
              v273 = v325;
            }
            v326 = v273;
            (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
              v357,
              v273,
              v341,
              v353,
              v354);
            v239 = v326;
            if ( v347 != &v347[16 * (unsigned int)v348] )
            {
              v327 = v3;
              v275 = (unsigned int *)v347;
              v276 = (unsigned int *)&v347[16 * (unsigned int)v348];
              v277 = v239;
              do
              {
                v278 = *((_QWORD *)v275 + 1);
                v279 = *v275;
                v275 += 4;
                sub_B99FD0(v277, v279, v278);
              }
              while ( v276 != v275 );
              v3 = v327;
              v239 = v277;
            }
          }
        }
        v240 = *(_WORD *)(v3 + 2);
        v301 = v239;
        v304 = *(_BYTE *)(v3 + 72);
        _BitScanReverse64(&v241, 1LL << SHIBYTE(v240));
        v242 = (unsigned __int8)v240 >> 5;
        v243 = (v240 >> 2) & 7;
        v346 = 257;
        v309 = (unsigned __int8)(63 - (v241 ^ 0x3F));
        v323 = sub_BD2C40(80, unk_3F148C4);
        if ( v323 )
          sub_B4D5A0((__int64)v323, v314, v318, v301, v309, v243, v242, v304, 0, 0);
        (*((void (__fastcall **)(void **, _QWORD *, __int64 *, _QWORD *, __int64))*v357 + 2))(
          v357,
          v323,
          &v343,
          v353,
          v354);
        if ( v347 != &v347[16 * (unsigned int)v348] )
        {
          v244 = (unsigned int *)&v347[16 * (unsigned int)v348];
          v245 = (unsigned int *)v347;
          do
          {
            v246 = *((_QWORD *)v245 + 1);
            v247 = *v245;
            v245 += 4;
            sub_B99FD0((__int64)v323, v247, v246);
          }
          while ( v244 != v245 );
        }
        v248 = *((_WORD *)v323 + 1) & 0xFFFE | *(_WORD *)(v3 + 2) & 1;
        *((_WORD *)v323 + 1) = v248;
        *((_WORD *)v323 + 1) = *(_WORD *)(v3 + 2) & 2 | v248 & 0xFFFD;
        v346 = 257;
        LODWORD(v341[0]) = 0;
        v249 = sub_94D3D0((unsigned int **)&v347, (__int64)v323, (__int64)v341, 1, (__int64)&v343);
        v346 = 257;
        LODWORD(v341[0]) = 1;
        v315 = sub_94D3D0((unsigned int **)&v347, (__int64)v323, (__int64)v341, 1, (__int64)&v343);
        v250 = *(_QWORD *)(v3 - 64);
        v342 = 257;
        v251 = *(_QWORD *)(v250 + 8);
        if ( v251 == *(_QWORD *)(v249 + 8) )
        {
          v252 = v249;
        }
        else
        {
          v252 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD))*v356 + 15))(
                   v356,
                   48,
                   v249,
                   *(_QWORD *)(v250 + 8));
          if ( !v252 )
          {
            v346 = 257;
            v252 = sub_B51D30(48, v249, v251, (__int64)&v343, 0, 0);
            if ( (unsigned __int8)sub_920620(v252) )
            {
              v267 = v359;
              if ( v358 )
                sub_B99FD0(v252, 3u, v358);
              sub_B45150(v252, v267);
            }
            (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
              v357,
              v252,
              v341,
              v353,
              v354);
            v268 = (unsigned int *)v347;
            v269 = (unsigned int *)&v347[16 * (unsigned int)v348];
            if ( v347 != (char *)v269 )
            {
              do
              {
                v270 = *((_QWORD *)v268 + 1);
                v271 = *v268;
                v268 += 4;
                sub_B99FD0(v252, v271, v270);
              }
              while ( v269 != v268 );
            }
          }
        }
        v253 = sub_ACADE0(*(__int64 ***)(v3 + 8));
        v342 = 257;
        v254 = v253;
        LODWORD(v336) = 0;
        v255 = (__int64 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, char **, __int64))*v356 + 11))(
                            v356,
                            v253,
                            v252,
                            &v336,
                            1);
        if ( !v255 )
        {
          v346 = 257;
          v262 = sub_BD2C40(104, unk_3F148BC);
          v255 = v262;
          if ( v262 )
          {
            sub_B44260((__int64)v262, *(_QWORD *)(v254 + 8), 65, 2u, 0, 0);
            v255[9] = (__int64)(v255 + 11);
            v255[10] = 0x400000000LL;
            sub_B4FD20((__int64)v255, v254, v252, &v336, 1, (__int64)&v343);
          }
          (*((void (__fastcall **)(void **, __int64 *, _QWORD *, _QWORD *, __int64))*v357 + 2))(
            v357,
            v255,
            v341,
            v353,
            v354);
          v263 = (unsigned int *)v347;
          v264 = (unsigned int *)&v347[16 * (unsigned int)v348];
          if ( v347 != (char *)v264 )
          {
            do
            {
              v265 = *((_QWORD *)v263 + 1);
              v266 = *v263;
              v263 += 4;
              sub_B99FD0((__int64)v255, v266, v265);
            }
            while ( v264 != v263 );
          }
        }
        v342 = 257;
        LODWORD(v336) = 1;
        v256 = (*((__int64 (__fastcall **)(void **, __int64 *, __int64, char **, __int64))*v356 + 11))(
                 v356,
                 v255,
                 v315,
                 &v336,
                 1);
        if ( !v256 )
        {
          v346 = 257;
          v257 = sub_BD2C40(104, unk_3F148BC);
          v256 = (__int64)v257;
          if ( v257 )
          {
            sub_B44260((__int64)v257, v255[1], 65, 2u, 0, 0);
            *(_QWORD *)(v256 + 72) = v256 + 88;
            *(_QWORD *)(v256 + 80) = 0x400000000LL;
            sub_B4FD20(v256, (__int64)v255, v315, &v336, 1, (__int64)&v343);
          }
          (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
            v357,
            v256,
            v341,
            v353,
            v354);
          v258 = (unsigned int *)v347;
          v259 = (unsigned int *)&v347[16 * (unsigned int)v348];
          if ( v347 != (char *)v259 )
          {
            do
            {
              v260 = *((_QWORD *)v258 + 1);
              v261 = *v258;
              v258 += 4;
              sub_B99FD0(v256, v261, v260);
            }
            while ( v259 != v258 );
          }
        }
        sub_BD84D0(v3, v256);
        sub_B43D60((_QWORD *)v3);
        sub_B32BF0(v366);
        v364 = &unk_49E5698;
        v365 = &unk_49D94D0;
        nullsub_63();
        nullsub_63();
        if ( v347 != v349 )
          _libc_free((unsigned __int64)v347);
        v38 = *(_QWORD ***)a1;
        v3 = (__int64)v323;
        v42 = 0;
        v44 = 0;
        v329 = 0;
        v6 = 1;
        v39 = **(_QWORD ***)a1;
      }
      else
      {
        v38 = *(_QWORD ***)a1;
        v323 = (_QWORD *)v3;
        v6 = 0;
        v42 = 0;
        v329 = 0;
        v44 = 0;
        v39 = **(_QWORD ***)a1;
      }
      goto LABEL_20;
    }
    v21 = *(_QWORD ***)a1;
    v22 = sub_2D44250(a2);
    v24 = *(_WORD *)(a2 + 2);
    _BitScanReverse64(&v25, 1LL << (*(_WORD *)(v3 + 2) >> 9));
    if ( v22 > 0x8000000000000000LL >> ((unsigned __int8)v25 ^ 0x3Fu) || v22 > *((_DWORD *)v21 + 21) >> 3 )
    {
      switch ( (v24 >> 4) & 0x1F )
      {
        case 0:
          v70 = (int *)&unk_444C3F0;
          goto LABEL_81;
        case 1:
          v70 = (int *)&unk_444C3D0;
          goto LABEL_81;
        case 2:
          v70 = (int *)&unk_444C3B0;
          goto LABEL_81;
        case 3:
          v70 = (int *)&unk_444C390;
          goto LABEL_81;
        case 4:
          v70 = (int *)&unk_444C330;
          goto LABEL_81;
        case 5:
          v70 = (int *)&unk_444C370;
          goto LABEL_81;
        case 6:
          v70 = (int *)&unk_444C350;
LABEL_81:
          LODWORD(v71) = sub_2D44250(v3);
          v72 = *(_WORD *)(v3 + 2);
          v73 = v71;
          _BitScanReverse64(&v71, 1LL << (v72 >> 9));
          if ( !sub_2D4A460(
                  a1,
                  v3,
                  v73,
                  63 - ((unsigned __int8)v71 ^ 0x3Fu),
                  *(_QWORD *)(v3 - 64),
                  *(_QWORD *)(v3 - 32),
                  0,
                  (v72 >> 1) & 7,
                  0,
                  v70) )
            goto LABEL_79;
          return 1;
        case 7:
        case 8:
        case 9:
        case 0xA:
        case 0xB:
        case 0xC:
        case 0xD:
        case 0xE:
        case 0xF:
        case 0x10:
        case 0x11:
        case 0x12:
          sub_2D44250(v3);
LABEL_79:
          v347 = (char *)a1;
          sub_2D48D60(
            v3,
            (void (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD *, __int64 *, __int64))sub_2D4B980,
            (__int64)&v347);
          return 1;
        default:
          goto LABEL_356;
      }
    }
    v38 = *(_QWORD ***)a1;
    v39 = **(_QWORD ***)a1;
    v74 = (__int64 (__fastcall *)(__int64, __int64))v39[149];
    if ( v74 == sub_2D43010 )
    {
      if ( (v24 & 0x1F0) == 0 )
      {
        v75 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 - 32) + 8LL) + 8LL);
        if ( v75 > 3u && v75 != 5 )
        {
          LOBYTE(v23) = v75 == 14 || (v75 & 0xFD) == 4;
          v6 = v23;
          if ( !(_BYTE)v23 )
          {
            v323 = 0;
            v42 = v3;
            v44 = 0;
            v329 = 0;
            goto LABEL_20;
          }
        }
        goto LABEL_92;
      }
    }
    else
    {
      if ( (unsigned int)v74((__int64)v38, v3) == 1 )
      {
LABEL_92:
        v6 = 1;
        v44 = 0;
        v76 = sub_2D47520((__int64)a1, v3);
        v38 = *(_QWORD ***)a1;
        v323 = 0;
        v329 = 0;
        v3 = v76;
        v42 = v76;
        v39 = **(_QWORD ***)a1;
        goto LABEL_20;
      }
      v38 = *(_QWORD ***)a1;
      v39 = **(_QWORD ***)a1;
    }
    v42 = v3;
    v6 = 0;
    v44 = 0;
    v323 = 0;
    v329 = 0;
    goto LABEL_20;
  }
  LOBYTE(v5) = sub_B46500((unsigned __int8 *)a2);
  v6 = v5;
  if ( !(_BYTE)v5 )
    return v6;
  v26 = *(_QWORD *)a1;
  v27 = sub_B43CC0(a2);
  v28 = sub_9208B0(v27, *(_QWORD *)(a2 + 8));
  v348 = v29;
  v347 = (char *)((unsigned __int64)(v28 + 7) >> 3);
  v30 = sub_CA1930(&v347);
  _BitScanReverse64(&v31, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  if ( 0x8000000000000000LL >> ((unsigned __int8)v31 ^ 0x3Fu) < v30 || *(_DWORD *)(v26 + 84) >> 3 < v30 )
  {
    v32 = sub_B43CC0(a2);
    v33 = sub_9208B0(v32, *(_QWORD *)(a2 + 8));
    v348 = v34;
    v347 = (char *)((unsigned __int64)(v33 + 7) >> 3);
    v35 = sub_CA1930(&v347);
    v36 = *(_WORD *)(a2 + 2);
    _BitScanReverse64(&v37, 1LL << (v36 >> 1));
    if ( !sub_2D4A460(
            a1,
            a2,
            v35,
            63 - ((unsigned __int8)v37 ^ 0x3Fu),
            *(_QWORD *)(a2 - 32),
            0,
            0,
            (v36 >> 7) & 7,
            0,
            dword_444C450) )
      sub_C64ED0("expandAtomicOpToLibcall shouldn't fail for Load", 1u);
    return 1;
  }
  v38 = *(_QWORD ***)a1;
  v39 = **(_QWORD ***)a1;
  v40 = (__int64 (__fastcall *)(__int64, __int64))v39[144];
  if ( v40 == sub_2D42ED0 )
  {
    v41 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
    if ( v41 > 3u && v41 != 5 && (v41 & 0xFD) != 4 )
      goto LABEL_27;
  }
  else if ( (unsigned int)v40((__int64)v38, a2) != 1 )
  {
    v38 = *(_QWORD ***)a1;
    v39 = **(_QWORD ***)a1;
LABEL_27:
    v323 = 0;
    v44 = a2;
    v6 = 0;
    v42 = 0;
    v329 = 0;
    goto LABEL_20;
  }
  v42 = 0;
  v43 = sub_2D46EE0((__int64)a1, a2);
  v38 = *(_QWORD ***)a1;
  v323 = 0;
  v329 = 0;
  v3 = v43;
  v44 = v43;
  v39 = **(_QWORD ***)a1;
LABEL_20:
  v45 = (__int64 (*)())v39[126];
  if ( v45 == sub_2D42A40 || !((unsigned __int8 (__fastcall *)(_QWORD **, __int64))v45)(v38, v3) )
  {
    if ( (unsigned __int8)sub_B46540((_BYTE *)v3) )
    {
      v48 = *(__int64 (**)())(**(_QWORD **)a1 + 1024LL);
      if ( v48 != sub_2D42A60 )
      {
        v302 = ((__int64 (__fastcall *)(_QWORD, __int64))v48)(*(_QWORD *)a1, v3);
        if ( v302 )
        {
          if ( v329 )
          {
            v91 = (*(_WORD *)(v329 + 2) >> 7) & 7;
          }
          else if ( v42 )
          {
            v91 = (*(_WORD *)(v42 + 2) >> 1) & 7;
          }
          else if ( v323
                 && ((v97 = *(__int64 (**)())(**(_QWORD **)a1 + 1176LL), v97 == sub_2D42AA0)
                  || ((unsigned int (__fastcall *)(_QWORD, _QWORD *))v97)(*(_QWORD *)a1, v323) != 2) )
          {
            v91 = (*((_WORD *)v323 + 1) >> 2) & 7;
          }
          else
          {
            v91 = 2;
          }
          v305 = v91;
          v355 = sub_BD5C60(v3);
          v356 = &v364;
          v360 = 512;
          LOWORD(v354) = 0;
          v347 = v349;
          v364 = &unk_49DA100;
          v357 = &v365;
          v348 = 0x200000000LL;
          v365 = &unk_49DA0B0;
          v358 = 0;
          v359 = 0;
          v361 = 7;
          v362 = 0;
          v363 = 0;
          v352 = 0;
          v353 = 0;
          sub_D5F1F0((__int64)&v347, v3);
          v92 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, char **, __int64, _QWORD))(**(_QWORD **)a1 + 1112LL))(
                            *(_QWORD *)a1,
                            &v347,
                            v3,
                            v305);
          if ( v92 )
          {
            sub_B44530(v92, v3);
            v6 = v302;
          }
          nullsub_61();
          v364 = &unk_49DA100;
          nullsub_63();
          if ( v347 != v349 )
            _libc_free((unsigned __int64)v347);
        }
      }
    }
  }
  else
  {
    if ( v44
      && (v46 = *(unsigned __int16 *)(v44 + 2),
          v47 = (__int64)byte_3F8E4E0,
          byte_3F8E4E0[8 * ((*(_WORD *)(v44 + 2) >> 7) & 7) + 4]) )
    {
      LOWORD(v46) = v46 & 0xFC7F;
      v57 = (*(_WORD *)(v44 + 2) >> 7) & 7;
      BYTE1(v46) |= 1u;
      *(_WORD *)(v44 + 2) = v46;
    }
    else if ( v329
           && (v46 = *(unsigned __int16 *)(v329 + 2),
               v47 = (__int64)byte_3F8E4E0,
               byte_3F8E4E0[8 * ((*(_WORD *)(v329 + 2) >> 7) & 7) + 5]) )
    {
      LOWORD(v46) = v46 & 0xFC7F;
      v57 = (*(_WORD *)(v329 + 2) >> 7) & 7;
      BYTE1(v46) |= 1u;
      *(_WORD *)(v329 + 2) = v46;
    }
    else if ( v42
           && ((v47 = (__int64)byte_3F8E4E0, v46 = (*(_WORD *)(v42 + 2) >> 1) & 7, byte_3F8E4E0[8 * v46 + 5])
            || byte_3F8E4E0[8 * v46 + 4]) )
    {
      v57 = (*(_WORD *)(v42 + 2) >> 1) & 7;
      *(_WORD *)(v42 + 2) = *(_WORD *)(v42 + 2) & 0xFFF1 | 4;
    }
    else
    {
      if ( !v323 )
        goto LABEL_23;
      v59 = *(__int64 (**)())(**(_QWORD **)a1 + 1176LL);
      if ( v59 != sub_2D42AA0 )
      {
        if ( ((unsigned int (__fastcall *)(_QWORD, _QWORD *))v59)(*(_QWORD *)a1, v323) )
          goto LABEL_23;
      }
      v60 = *((_WORD *)v323 + 1);
      v61 = (v60 >> 2) & 7;
      v47 = (v60 >> 5) & 7;
      v62 = (v60 >> 2) & 7;
      if ( !byte_3F8E4E0[8 * v61 + 5] && !byte_3F8E4E0[8 * v61 + 4] )
      {
        v46 = (unsigned __int8)v60 >> 5;
        if ( !byte_3F8E4E0[8 * v46 + 4] )
          goto LABEL_23;
      }
      if ( (unsigned __int8)v60 >> 5 == 7 )
      {
        v57 = 7;
      }
      else
      {
        v57 = v62;
        if ( (unsigned __int8)v60 >> 5 == 4 )
        {
          if ( v62 == 2 )
          {
            v57 = 4;
          }
          else if ( v62 == 5 )
          {
            v57 = 6;
          }
        }
      }
      v63 = 64;
      v64 = *(__int64 (**)())(**(_QWORD **)a1 + 1016LL);
      v46 = 8;
      if ( v64 != sub_2D42A50 )
      {
        v322 = v57;
        v193 = ((__int64 (__fastcall *)(_QWORD, _QWORD *, __int64, __int64))v64)(*(_QWORD *)a1, v323, 8, 64);
        v57 = v322;
        v63 = 32 * v193;
        v46 = (unsigned int)(4 * v193);
        v60 = *((_WORD *)v323 + 1);
      }
      v65 = v46 | v60 & 0xFFE3;
      LOBYTE(v65) = v65 & 0x1F;
      v66 = v63 | v65;
      v47 = (__int64)v323;
      *((_WORD *)v323 + 1) = v66;
    }
    v319 = v57;
    if ( v57 != 2 )
    {
      sub_2D46B10((__int64)&v347, v3, *((_QWORD *)a1 + 1));
      v312 = v319;
      v320 = (*(__int64 (__fastcall **)(_QWORD, char **, __int64, _QWORD))(**(_QWORD **)a1 + 1104LL))(
               *(_QWORD *)a1,
               &v347,
               v3,
               v319);
      v58 = (*(__int64 (__fastcall **)(_QWORD, char **, __int64, _QWORD))(**(_QWORD **)a1 + 1112LL))(
              *(_QWORD *)a1,
              &v347,
              v3,
              v312);
      if ( v58 )
      {
        v316 = v58;
        sub_B44530((_QWORD *)v58, v3);
        v58 = v316;
      }
      LOBYTE(v3) = (v58 | v320) != 0;
      sub_B32BF0(v366);
      v364 = &unk_49E5698;
      v365 = &unk_49D94D0;
      nullsub_63();
      nullsub_63();
      if ( v347 != v349 )
        _libc_free((unsigned __int64)v347);
      v6 |= v3;
    }
  }
LABEL_23:
  if ( v44 )
  {
    v6 |= sub_2D49200((unsigned __int64 *)a1, v44, v46, v47);
    return v6;
  }
  if ( !v329 )
  {
    if ( !v42 )
    {
      if ( v323 )
      {
        v88 = *(_DWORD *)(*(_QWORD *)a1 + 96LL);
        v89 = sub_2D44290((__int64)v323);
        v90 = *(__int64 (**)())(**(_QWORD **)a1 + 1176LL);
        if ( v90 != sub_2D42AA0 )
        {
          switch ( ((unsigned int (__fastcall *)(_QWORD, _QWORD *))v90)(*(_QWORD *)a1, v323) )
          {
            case 0u:
              break;
            case 2u:
              v98 = (_QWORD *)v323[5];
              v291 = *((_WORD *)v323 + 1);
              v99 = v98[9];
              v290 = (v291 >> 2) & 7;
              v310 = *(v323 - 12);
              v317 = (__int64 *)sub_B2BE50(v99);
              v100 = *(__int64 (**)())(**(_QWORD **)a1 + 1008LL);
              if ( v100 == sub_2D42A40
                || (v292 = ((__int64 (__fastcall *)(_QWORD, _QWORD *))v100)(*(_QWORD *)a1, v323)) == 0 )
              {
                if ( ((*((_WORD *)v323 + 1) >> 5) & 7) == 7 )
                {
                  v306 = 7;
                }
                else
                {
                  v101 = *((_WORD *)v323 + 1) >> 2;
                  v102 = v101 & 7;
                  v306 = v101 & 7;
                  if ( ((*((_WORD *)v323 + 1) >> 5) & 7) == 4 )
                  {
                    if ( v102 == 2 )
                    {
                      v306 = 4;
                    }
                    else
                    {
                      v230 = 6;
                      if ( v102 != 5 )
                        v230 = v101 & 7;
                      v306 = v230;
                    }
                  }
                }
                v292 = 0;
                v299 = 0;
              }
              else
              {
                v331 = *((_WORD *)v323 + 1);
                v299 = (v331 & 2) != 0;
                if ( ((v331 >> 1) & 1) != 0 )
                {
                  v306 = 2;
                  v299 = 0;
                  v292 = (v331 & 2) != 0;
                }
                else
                {
                  v306 = 2;
                  if ( ((((v291 >> 2) & 7) - 2) & 0xFFFD) != 0 )
                    v299 = sub_B2D610(v99, 18) ^ 1;
                }
              }
              v330 = sub_B2D610(v99, 18);
              v288 = v292;
              if ( v330 )
              {
                v288 = v292 & ((*((_WORD *)v323 + 1) & 2) != 0);
                v330 = v292 & ((*((_WORD *)v323 + 1) & 2) == 0);
              }
              v347 = "cmpxchg.end";
              v351 = 1;
              v350 = 3;
              v103 = sub_AA8550(v98, v323 + 3, 0, (__int64)&v347, 0);
              v351 = 1;
              v297 = v103;
              v347 = "cmpxchg.failure";
              v350 = 3;
              v313 = sub_22077B0(0x50u);
              if ( v313 )
                sub_AA4D50(v313, (__int64)v317, (__int64)&v347, v99, v297);
              v351 = 1;
              v347 = "cmpxchg.nostore";
              v350 = 3;
              v104 = sub_22077B0(0x50u);
              v298 = v104;
              if ( v104 )
                sub_AA4D50(v104, (__int64)v317, (__int64)&v347, v99, v313);
              v351 = 1;
              v347 = "cmpxchg.success";
              v350 = 3;
              v105 = sub_22077B0(0x50u);
              v303 = v105;
              if ( v105 )
                sub_AA4D50(v105, (__int64)v317, (__int64)&v347, v99, v298);
              v351 = 1;
              v347 = "cmpxchg.releasedload";
              v350 = 3;
              v106 = sub_22077B0(0x50u);
              v293 = v106;
              if ( v106 )
                sub_AA4D50(v106, (__int64)v317, (__int64)&v347, v99, v303);
              v351 = 1;
              v347 = "cmpxchg.trystore";
              v350 = 3;
              v107 = sub_22077B0(0x50u);
              v294 = v107;
              if ( v107 )
                sub_AA4D50(v107, (__int64)v317, (__int64)&v347, v99, v293);
              v351 = 1;
              v347 = "cmpxchg.fencedstore";
              v350 = 3;
              v108 = sub_22077B0(0x50u);
              v295 = v108;
              if ( v108 )
                sub_AA4D50(v108, (__int64)v317, (__int64)&v347, v99, v294);
              v351 = 1;
              v347 = "cmpxchg.start";
              v350 = 3;
              v109 = sub_22077B0(0x50u);
              v296 = v109;
              if ( v109 )
                sub_AA4D50(v109, (__int64)v317, (__int64)&v347, v99, v295);
              sub_2D46B10((__int64)&v347, (__int64)v323, *((_QWORD *)a1 + 1));
              if ( (v98[6] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                v42 = (v98[6] & 0xFFFFFFFFFFFFFFF8LL) - 24;
              sub_B43D60((_QWORD *)v42);
              v352 = v98;
              v353 = v98 + 6;
              LOWORD(v354) = 0;
              if ( v330 )
                (*(void (__fastcall **)(_QWORD, char **, _QWORD *, _QWORD))(**(_QWORD **)a1 + 1104LL))(
                  *(_QWORD *)a1,
                  &v347,
                  v323,
                  v290);
              _BitScanReverse64(&v110, 1LL << *((_BYTE *)v323 + 3));
              sub_2D44EF0(
                (__int64)&v343,
                (__int64)&v347,
                (__int64)v323,
                *(_QWORD *)(*(v323 - 8) + 8LL),
                v310,
                63 - (v110 ^ 0x3F),
                *(_DWORD *)(*(_QWORD *)a1 + 96LL) >> 3);
              v342 = 257;
              v111 = sub_BD2C40(72, 1u);
              if ( v111 )
                sub_B4C8F0((__int64)v111, v296, 1u, 0, 0);
              (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v111,
                v341,
                v353,
                v354);
              v112 = (unsigned int *)v347;
              v113 = (unsigned int *)&v347[16 * (unsigned int)v348];
              if ( v347 != (char *)v113 )
              {
                do
                {
                  v114 = *((_QWORD *)v112 + 1);
                  v115 = *v112;
                  v112 += 4;
                  sub_B99FD0((__int64)v111, v115, v114);
                }
                while ( v113 != v112 );
              }
              v116 = *(_QWORD *)a1;
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v296;
              v353 = (_QWORD *)(v296 + 48);
              v311 = (*(__int64 (__fastcall **)(__int64, char **, __int64, __int64, _QWORD))(*(_QWORD *)v116 + 1032LL))(
                       v116,
                       &v347,
                       v343,
                       v345,
                       v306);
              v117 = v311;
              if ( v343 != v344 )
                v117 = sub_2D44750((__int64 *)&v347, v311, &v343);
              v336 = "should_store";
              v340 = 1;
              v339 = 3;
              v118 = *(v323 - 8);
              v119 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v356 + 7))(
                                 v356,
                                 32,
                                 v117,
                                 v118);
              if ( !v119 )
              {
                v342 = 257;
                v119 = sub_BD2C40(72, unk_3F10FD0);
                if ( v119 )
                {
                  v210 = *(_QWORD *)(v117 + 8);
                  v211 = *(unsigned __int8 *)(v210 + 8);
                  if ( (unsigned int)(v211 - 17) > 1 )
                  {
                    v215 = sub_BCB2A0(*(_QWORD **)v210);
                  }
                  else
                  {
                    v212 = *(_DWORD *)(v210 + 32);
                    v213 = *(_QWORD **)v210;
                    BYTE4(v333) = (_BYTE)v211 == 18;
                    LODWORD(v333) = v212;
                    v214 = (__int64 *)sub_BCB2A0(v213);
                    v215 = sub_BCE1B0(v214, v333);
                  }
                  sub_B523C0((__int64)v119, v215, 53, 32, v117, v118, (__int64)v341, 0, 0, 0);
                }
                (*((void (__fastcall **)(void **, _QWORD *, char **, _QWORD *, __int64))*v357 + 2))(
                  v357,
                  v119,
                  &v336,
                  v353,
                  v354);
                v216 = (unsigned int *)v347;
                v217 = (unsigned int *)&v347[16 * (unsigned int)v348];
                if ( v347 != (char *)v217 )
                {
                  do
                  {
                    v218 = *((_QWORD *)v216 + 1);
                    v219 = *v216;
                    v216 += 4;
                    sub_B99FD0((__int64)v119, v219, v218);
                  }
                  while ( v217 != v216 );
                }
              }
              v342 = 257;
              v120 = sub_BD2C40(72, 3u);
              v122 = (__int64)v120;
              if ( v120 )
              {
                sub_B4C9A0((__int64)v120, v295, v298, (__int64)v119, 3u, v121, 0, 0);
                v121 = v286;
              }
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64, __int64))*v357 + 2))(
                v357,
                v122,
                v341,
                v353,
                v354,
                v121);
              v123 = (unsigned int *)v347;
              v124 = (unsigned int *)&v347[16 * (unsigned int)v348];
              if ( v347 != (char *)v124 )
              {
                do
                {
                  v125 = *((_QWORD *)v123 + 1);
                  v126 = *v123;
                  v123 += 4;
                  sub_B99FD0(v122, v126, v125);
                }
                while ( v124 != v123 );
              }
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v295;
              v353 = (_QWORD *)(v295 + 48);
              if ( v288 )
                (*(void (__fastcall **)(_QWORD, char **, _QWORD *, _QWORD))(**(_QWORD **)a1 + 1104LL))(
                  *(_QWORD *)a1,
                  &v347,
                  v323,
                  v290);
              v342 = 257;
              v127 = sub_BD2C40(72, 1u);
              v128 = (__int64)v127;
              if ( v127 )
                sub_B4C8F0((__int64)v127, v294, 1u, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v128,
                v341,
                v353,
                v354);
              v129 = (unsigned int *)v347;
              v130 = (unsigned int *)&v347[16 * (unsigned int)v348];
              if ( v347 != (char *)v130 )
              {
                do
                {
                  v131 = *((_QWORD *)v129 + 1);
                  v132 = *v129;
                  v129 += 4;
                  sub_B99FD0(v128, v132, v131);
                }
                while ( v130 != v129 );
              }
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v294;
              v353 = (_QWORD *)(v294 + 48);
              v341[0] = "loaded.trystore";
              v342 = 259;
              v289 = sub_D5C860((__int64 *)&v347, v343, 2, (__int64)v341);
              sub_F0A850(v289, v311, v295);
              v133 = *(v323 - 4);
              if ( v343 != v344 )
                v133 = sub_2D442D0((__int64 *)&v347, v289, v133, &v343);
              v134 = (*(__int64 (__fastcall **)(_QWORD, char **, __int64, __int64, _QWORD))(**(_QWORD **)a1 + 1040LL))(
                       *(_QWORD *)a1,
                       &v347,
                       v133,
                       v345,
                       v306);
              v340 = 1;
              v135 = v134;
              v339 = 3;
              v336 = "success";
              v136 = sub_BCB2D0(v317);
              v137 = sub_ACD640(v136, 0, 0);
              v138 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v356 + 7))(
                                 v356,
                                 32,
                                 v135,
                                 v137);
              if ( !v138 )
              {
                v342 = 257;
                v138 = sub_BD2C40(72, unk_3F10FD0);
                if ( v138 )
                {
                  v200 = *(_QWORD *)(v135 + 8);
                  v201 = *(unsigned __int8 *)(v200 + 8);
                  if ( (unsigned int)(v201 - 17) > 1 )
                  {
                    v205 = sub_BCB2A0(*(_QWORD **)v200);
                  }
                  else
                  {
                    v202 = *(_DWORD *)(v200 + 32);
                    v203 = *(_QWORD **)v200;
                    BYTE4(v334) = (_BYTE)v201 == 18;
                    LODWORD(v334) = v202;
                    v204 = (__int64 *)sub_BCB2A0(v203);
                    v205 = sub_BCE1B0(v204, v334);
                  }
                  sub_B523C0((__int64)v138, v205, 53, 32, v135, v137, (__int64)v341, 0, 0, 0);
                }
                (*((void (__fastcall **)(void **, _QWORD *, char **, _QWORD *, __int64))*v357 + 2))(
                  v357,
                  v138,
                  &v336,
                  v353,
                  v354);
                v206 = (unsigned int *)v347;
                v207 = (unsigned int *)&v347[16 * (unsigned int)v348];
                if ( v347 != (char *)v207 )
                {
                  do
                  {
                    v208 = *((_QWORD *)v206 + 1);
                    v209 = *v206;
                    v206 += 4;
                    sub_B99FD0((__int64)v138, v209, v208);
                  }
                  while ( v207 != v206 );
                }
              }
              v139 = v296;
              if ( v299 )
                v139 = v293;
              v140 = (*((_BYTE *)v323 + 2) & 2) == 0;
              v342 = 257;
              if ( !v140 )
                v139 = v313;
              v141 = sub_BD2C40(72, 3u);
              v142 = (__int64)v141;
              if ( v141 )
                sub_B4C9A0((__int64)v141, v303, v139, (__int64)v138, 3u, 0, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v142,
                v341,
                v353,
                v354);
              v143 = (unsigned int *)&v347[16 * (unsigned int)v348];
              v144 = (unsigned int *)v347;
              if ( v347 != (char *)v143 )
              {
                do
                {
                  v145 = *((_QWORD *)v144 + 1);
                  v146 = *v144;
                  v144 += 4;
                  sub_B99FD0(v142, v146, v145);
                }
                while ( v143 != v144 );
              }
              v352 = (_QWORD *)v293;
              v353 = (_QWORD *)(v293 + 48);
              LOWORD(v354) = 0;
              if ( v299 )
              {
                v287 = (*(__int64 (__fastcall **)(_QWORD, char **, __int64, __int64, _QWORD))(**(_QWORD **)a1 + 1032LL))(
                         *(_QWORD *)a1,
                         &v347,
                         v343,
                         v345,
                         v306);
                v147 = v287;
                if ( v343 != v344 )
                  v147 = sub_2D44750((__int64 *)&v347, v287, &v343);
                v336 = "should_store";
                v340 = 1;
                v339 = 3;
                v148 = *(v323 - 8);
                v307 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v356 + 7))(
                                   v356,
                                   32,
                                   v147,
                                   v148);
                if ( !v307 )
                {
                  v342 = 257;
                  v307 = sub_BD2C40(72, unk_3F10FD0);
                  if ( v307 )
                  {
                    v220 = *(_QWORD *)(v147 + 8);
                    v221 = *(unsigned __int8 *)(v220 + 8);
                    if ( (unsigned int)(v221 - 17) > 1 )
                    {
                      v225 = sub_BCB2A0(*(_QWORD **)v220);
                    }
                    else
                    {
                      v222 = *(_DWORD *)(v220 + 32);
                      v223 = *(_QWORD **)v220;
                      BYTE4(v335) = (_BYTE)v221 == 18;
                      LODWORD(v335) = v222;
                      v224 = (__int64 *)sub_BCB2A0(v223);
                      v225 = sub_BCE1B0(v224, v335);
                    }
                    sub_B523C0((__int64)v307, v225, 53, 32, v147, v148, (__int64)v341, 0, 0, 0);
                  }
                  (*((void (__fastcall **)(void **, _QWORD *, char **, _QWORD *, __int64))*v357 + 2))(
                    v357,
                    v307,
                    &v336,
                    v353,
                    v354);
                  v226 = (unsigned int *)v347;
                  v227 = (unsigned int *)&v347[16 * (unsigned int)v348];
                  if ( v347 != (char *)v227 )
                  {
                    do
                    {
                      v228 = *((_QWORD *)v226 + 1);
                      v229 = *v226;
                      v226 += 4;
                      sub_B99FD0((__int64)v307, v229, v228);
                    }
                    while ( v227 != v226 );
                  }
                }
                v342 = 257;
                v149 = sub_BD2C40(72, 3u);
                v151 = (__int64)v149;
                if ( v149 )
                  sub_B4C9A0((__int64)v149, v294, v298, (__int64)v307, 3u, v150, 0, 0);
                (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                  v357,
                  v151,
                  v341,
                  v353,
                  v354);
                v152 = (unsigned int *)&v347[16 * (unsigned int)v348];
                v153 = (unsigned int *)v347;
                if ( v347 != (char *)v152 )
                {
                  do
                  {
                    v154 = *((_QWORD *)v153 + 1);
                    v155 = *v153;
                    v153 += 4;
                    sub_B99FD0(v151, v155, v154);
                  }
                  while ( v152 != v153 );
                }
                sub_F0A850(v289, v287, v293);
              }
              else
              {
                v342 = 257;
                v194 = sub_BD2C40(72, unk_3F148B8);
                v195 = (__int64)v194;
                if ( v194 )
                  sub_B4C8A0((__int64)v194, v355, 0, 0);
                (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                  v357,
                  v195,
                  v341,
                  v353,
                  v354);
                v196 = (unsigned int *)&v347[16 * (unsigned int)v348];
                v197 = (unsigned int *)v347;
                if ( v347 != (char *)v196 )
                {
                  do
                  {
                    v198 = *((_QWORD *)v197 + 1);
                    v199 = *v197;
                    v197 += 4;
                    sub_B99FD0(v195, v199, v198);
                  }
                  while ( v196 != v197 );
                }
              }
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v303;
              v353 = (_QWORD *)(v303 + 48);
              if ( v292
                || (v156 = *(__int64 (**)())(**(_QWORD **)a1 + 1024LL), v156 != sub_2D42A60)
                && ((unsigned __int8 (__fastcall *)(_QWORD, _QWORD *))v156)(*(_QWORD *)a1, v323) )
              {
                (*(void (__fastcall **)(_QWORD, char **, _QWORD *, _QWORD))(**(_QWORD **)a1 + 1112LL))(
                  *(_QWORD *)a1,
                  &v347,
                  v323,
                  v290);
              }
              v342 = 257;
              v157 = sub_BD2C40(72, 1u);
              v158 = (__int64)v157;
              if ( v157 )
                sub_B4C8F0((__int64)v157, v297, 1u, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v158,
                v341,
                v353,
                v354);
              v159 = (unsigned int *)&v347[16 * (unsigned int)v348];
              v160 = (unsigned int *)v347;
              if ( v347 != (char *)v159 )
              {
                do
                {
                  v161 = *((_QWORD *)v160 + 1);
                  v162 = *v160;
                  v160 += 4;
                  sub_B99FD0(v158, v162, v161);
                }
                while ( v159 != v160 );
              }
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v298;
              v353 = (_QWORD *)(v298 + 48);
              v341[0] = "loaded.nostore";
              v342 = 259;
              v308 = sub_D5C860((__int64 *)&v347, *(_QWORD *)(v311 + 8), 2, (__int64)v341);
              sub_F0A850(v308, v311, v296);
              if ( v299 )
                sub_F0A850(v308, v287, v293);
              v163 = *(void (**)())(**(_QWORD **)a1 + 1120LL);
              if ( v163 != nullsub_1577 )
                ((void (__fastcall *)(_QWORD, char **))v163)(*(_QWORD *)a1, &v347);
              v342 = 257;
              v164 = sub_BD2C40(72, 1u);
              v165 = (__int64)v164;
              if ( v164 )
                sub_B4C8F0((__int64)v164, v313, 1u, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v165,
                v341,
                v353,
                v354);
              v166 = (unsigned int *)&v347[16 * (unsigned int)v348];
              v167 = (unsigned int *)v347;
              if ( v347 != (char *)v166 )
              {
                do
                {
                  v168 = *((_QWORD *)v167 + 1);
                  v169 = *v167;
                  v167 += 4;
                  sub_B99FD0(v165, v169, v168);
                }
                while ( v166 != v167 );
              }
              LOWORD(v354) = 0;
              v352 = (_QWORD *)v313;
              v353 = (_QWORD *)(v313 + 48);
              v341[0] = "loaded.failure";
              v342 = 259;
              v300 = sub_D5C860((__int64 *)&v347, *(_QWORD *)(v311 + 8), 2, (__int64)v341);
              sub_F0A850(v300, v308, v298);
              if ( (*((_BYTE *)v323 + 2) & 2) != 0 )
                sub_F0A850(v300, v289, v294);
              if ( v292 )
                (*(void (__fastcall **)(_QWORD, char **, _QWORD *, _QWORD))(**(_QWORD **)a1 + 1112LL))(
                  *(_QWORD *)a1,
                  &v347,
                  v323,
                  (v291 >> 5) & 7);
              v342 = 257;
              v170 = sub_BD2C40(72, 1u);
              v171 = (__int64)v170;
              if ( v170 )
                sub_B4C8F0((__int64)v170, v297, 1u, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _QWORD *, __int64))*v357 + 2))(
                v357,
                v171,
                v341,
                v353,
                v354);
              v172 = (unsigned int *)v347;
              v173 = (unsigned int *)&v347[16 * (unsigned int)v348];
              if ( v347 != (char *)v173 )
              {
                do
                {
                  v174 = *((_QWORD *)v172 + 1);
                  v175 = *v172;
                  v172 += 4;
                  sub_B99FD0(v171, v175, v174);
                }
                while ( v173 != v172 );
              }
              sub_A88F30((__int64)&v347, v297, *(_QWORD *)(v297 + 56), 1);
              v341[0] = "loaded.exit";
              v342 = 259;
              v176 = (_BYTE *)sub_D5C860((__int64 *)&v347, *(_QWORD *)(v311 + 8), 2, (__int64)v341);
              sub_F0A850((__int64)v176, v289, v303);
              sub_F0A850((__int64)v176, v300, v313);
              v341[0] = "success";
              v342 = 259;
              v177 = sub_BCB2A0(v317);
              v178 = sub_D5C860((__int64 *)&v347, v177, 2, (__int64)v341);
              v179 = sub_ACD6D0(v317);
              sub_F0A850(v178, v179, v303);
              v180 = sub_ACD720(v317);
              sub_F0A850(v178, v180, v313);
              sub_A88F30((__int64)&v347, v297, *(_QWORD *)(v178 + 32), 0);
              if ( v343 != v344 )
                v176 = (_BYTE *)sub_2D44750((__int64 *)&v347, (__int64)v176, &v343);
              v336 = v338;
              v337 = 0x200000000LL;
              v181 = v323[2];
              if ( v181 )
              {
                do
                {
                  v182 = *(_QWORD *)(v181 + 24);
                  if ( *(_BYTE *)v182 == 93 )
                  {
                    if ( **(_DWORD **)(v182 + 72) )
                      sub_BD84D0(*(_QWORD *)(v181 + 24), v178);
                    else
                      sub_BD84D0(*(_QWORD *)(v181 + 24), (__int64)v176);
                    v185 = (unsigned int)v337;
                    v186 = (unsigned int)v337 + 1LL;
                    if ( v186 > HIDWORD(v337) )
                    {
                      sub_C8D5F0((__int64)&v336, v338, v186, 8u, v183, v184);
                      v185 = (unsigned int)v337;
                    }
                    *(_QWORD *)&v336[8 * v185] = v182;
                    LODWORD(v337) = v337 + 1;
                  }
                  v181 = *(_QWORD *)(v181 + 8);
                }
                while ( v181 );
                v187 = &v336[8 * (unsigned int)v337];
                if ( v336 != v187 )
                {
                  v188 = v336;
                  do
                  {
                    v189 = *(_QWORD **)v188;
                    v188 += 8;
                    sub_B43D60(v189);
                  }
                  while ( v187 != v188 );
                }
                if ( v323[2] )
                {
                  v332 = 0;
                  v342 = 257;
                  v190 = sub_ACADE0((__int64 **)v323[1]);
                  v191 = sub_2466140((__int64 *)&v347, v190, v176, &v332, 1, (__int64)v341);
                  v342 = 257;
                  v332 = 1;
                  v192 = sub_2466140((__int64 *)&v347, v191, (_BYTE *)v178, &v332, 1, (__int64)v341);
                  sub_BD84D0((__int64)v323, v192);
                }
              }
              sub_B43D60(v323);
              if ( v336 != v338 )
                _libc_free((unsigned __int64)v336);
              goto LABEL_102;
            case 5u:
              v6 = 1;
              sub_2D48230((unsigned __int64 *)a1, (__int64)v323);
              return v6;
            case 8u:
              (*(void (__fastcall **)(_QWORD, _QWORD *))(**(_QWORD **)a1 + 1064LL))(*(_QWORD *)a1, v323);
              return 1;
            case 9u:
              v6 |= sub_2A2D840((__int64)v323);
              return v6;
            default:
              goto LABEL_356;
          }
        }
        if ( v88 >> 3 > v89 )
          v6 |= sub_2D492F0((__int64 *)a1, (__int64)v323);
      }
      return v6;
    }
    v51 = *(_QWORD *)(v42 - 32);
    if ( *(_BYTE *)v51 == 17 )
    {
      v52 = *(_WORD *)(v42 + 2) >> 4;
      v53 = v52 & 0x1F;
      if ( v53 == 3 )
      {
        v93 = *(_DWORD *)(v51 + 32);
        if ( !v93 )
          goto LABEL_41;
        if ( v93 <= 0x40 )
          v55 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v93) == *(_QWORD *)(v51 + 24);
        else
          v55 = v93 == (unsigned int)sub_C445E0(v51 + 24);
      }
      else
      {
        if ( (v52 & 0x1C) != 0 )
        {
          if ( (unsigned __int16)(v53 - 5) > 1u )
            goto LABEL_42;
        }
        else if ( (unsigned __int16)(v53 - 1) > 1u )
        {
          goto LABEL_42;
        }
        v54 = *(_DWORD *)(v51 + 32);
        if ( v54 <= 0x40 )
          v55 = *(_QWORD *)(v51 + 24) == 0;
        else
          v55 = v54 == (unsigned int)sub_C444A0(v51 + 24);
      }
      if ( v55 )
      {
LABEL_41:
        v56 = *(__int64 (**)())(**(_QWORD **)a1 + 1200LL);
        if ( v56 != sub_2D42AD0 )
        {
          v96 = ((__int64 (__fastcall *)(_QWORD, unsigned __int64))v56)(*(_QWORD *)a1, v42);
          if ( v96 )
          {
            v6 = 1;
            sub_2D49200((unsigned __int64 *)a1, v96, v94, v95);
            return v6;
          }
        }
      }
    }
LABEL_42:
    v6 |= sub_2D4BB90((unsigned __int64 *)a1, (_QWORD *)v42);
    return v6;
  }
  v49 = *(__int64 (**)())(**(_QWORD **)a1 + 1160LL);
  if ( v49 == sub_2D42A90 )
    return v6;
  v50 = ((__int64 (__fastcall *)(_QWORD, __int64))v49)(*(_QWORD *)a1, v329);
  if ( v50 == 8 )
  {
    sub_2D46B10((__int64)&v347, v329, *((_QWORD *)a1 + 1));
    v77 = *(_QWORD *)(v329 - 64);
    v78 = *(_QWORD *)(v329 - 32);
    v79 = (*(_WORD *)(v329 + 2) >> 7) & 7;
    if ( v79 == 1 )
      v79 = 2;
    _BitScanReverse64(&v80, 1LL << (*(_WORD *)(v329 + 2) >> 1));
    v81 = v79;
    v346 = 257;
    v321 = 63 - (v80 ^ 0x3F);
    v82 = sub_BD2C40(80, unk_3F148C0);
    v83 = v82;
    if ( v82 )
      sub_B4D750((__int64)v82, 0, v78, v77, v321, v81, 1, 0, 0);
    (*((void (__fastcall **)(void **, _QWORD *, __int64 *, _QWORD *, __int64))*v357 + 2))(v357, v83, &v343, v353, v354);
    v84 = (unsigned int *)v347;
    v85 = (unsigned int *)&v347[16 * (unsigned int)v348];
    if ( v347 != (char *)v85 )
    {
      do
      {
        v86 = *((_QWORD *)v84 + 1);
        v87 = *v84;
        v84 += 4;
        sub_B99FD0((__int64)v83, v87, v86);
      }
      while ( v85 != v84 );
    }
    sub_B43D60((_QWORD *)v329);
    sub_2D4BB90((unsigned __int64 *)a1, v83);
LABEL_102:
    sub_B32BF0(v366);
    v364 = &unk_49E5698;
    v365 = &unk_49D94D0;
    nullsub_63();
    nullsub_63();
    if ( v347 != v349 )
      _libc_free((unsigned __int64)v347);
    return 1;
  }
  if ( v50 == 9 )
  {
    v6 = 1;
    *(_WORD *)(v329 + 2) &= 0xFC7Fu;
    *(_BYTE *)(v329 + 72) = 1;
  }
  else if ( v50 )
  {
LABEL_356:
    BUG();
  }
  return v6;
}
