// Function: sub_1F87CB0
// Address: 0x1f87cb0
//
__int64 *__fastcall sub_1F87CB0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10,
        __int64 a11,
        __int128 a12,
        unsigned int a13,
        char a14)
{
  __int64 v15; // rax
  char v16; // si
  int v17; // eax
  int v18; // eax
  __int64 v19; // r14
  unsigned __int8 *v20; // rdx
  unsigned __int8 (__fastcall *v21)(__int64, __int64, __int64, __int64, __int64); // r15
  __int64 v22; // rax
  int v23; // esi
  __int64 v24; // rax
  int v25; // ecx
  __int64 v26; // rdi
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r14
  int v30; // eax
  int v31; // eax
  unsigned __int8 *v32; // rcx
  bool v33; // dl
  unsigned __int8 *v34; // rax
  __m128 si128; // xmm0
  __int64 v36; // rdx
  char v37; // cl
  const void **v38; // r15
  unsigned __int8 v39; // r14
  __int64 v40; // r14
  unsigned int v41; // r15d
  __int64 v42; // r8
  __int64 v43; // rax
  _DWORD *v44; // r9
  unsigned __int8 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rsi
  bool v48; // cl
  int v49; // eax
  int v50; // eax
  bool v51; // al
  __int64 v52; // rax
  __int64 *v53; // r14
  unsigned __int8 (__fastcall *v54)(_DWORD *, __int64, __int64, __int64, __int64); // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r14
  unsigned int v63; // edx
  __int64 v64; // rax
  unsigned __int64 v65; // r15
  __int64 v66; // rdx
  const void **v67; // rsi
  __int64 v68; // rax
  char v69; // r8
  const void **v70; // rax
  __int64 v71; // r14
  unsigned int v72; // edx
  unsigned int v73; // r12d
  __int64 v74; // rax
  unsigned int v75; // r15d
  bool v76; // cl
  __int64 *result; // rax
  __int64 v78; // r9
  __int64 *v79; // r15
  __int64 v80; // r12
  char v81; // r13
  __int64 v82; // rdi
  __int64 v83; // rsi
  __int64 v84; // rax
  unsigned int v85; // eax
  __int64 v86; // rsi
  unsigned int v87; // r12d
  const void **v88; // rdx
  const void **v89; // r13
  __int64 v90; // rax
  unsigned int v91; // r14d
  unsigned __int64 v92; // rax
  int v93; // eax
  __int128 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rdi
  unsigned int v97; // r13d
  bool v98; // al
  int v99; // eax
  unsigned int v100; // eax
  __int64 v101; // rax
  unsigned int v102; // eax
  char v103; // di
  const void **v104; // rdx
  __int64 v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rcx
  __int64 v108; // rdx
  unsigned int v109; // eax
  __int64 v110; // rdx
  unsigned int v111; // ecx
  unsigned __int64 v112; // rdx
  __int128 v113; // rax
  __int128 v114; // rax
  __int64 v115; // rcx
  __int64 v116; // rdx
  __int64 v117; // rdi
  __int64 v118; // r8
  __int64 (*v119)(); // rax
  __int64 v120; // rax
  __int64 v121; // rax
  int v122; // eax
  __int64 v123; // rax
  unsigned int v124; // r12d
  __int64 v125; // r15
  bool v126; // r14
  bool v127; // al
  bool v128; // al
  int v129; // eax
  unsigned __int8 v130; // r12
  unsigned int v131; // eax
  __int64 v132; // rbx
  __int64 v133; // r14
  unsigned int v134; // r15d
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 *v137; // rdi
  __int64 v138; // rax
  __int64 *v139; // rdx
  __int64 v140; // rax
  __int64 v141; // r14
  unsigned __int64 v142; // r15
  int v143; // ecx
  __int64 v144; // rax
  int v145; // eax
  __int64 v146; // r12
  unsigned __int8 *v147; // rax
  __int64 v148; // rax
  const void **v149; // rdx
  const void **v150; // rbx
  unsigned int v151; // eax
  __int64 v152; // rdx
  __int64 *v153; // r12
  __int64 *v154; // rax
  unsigned __int64 v155; // rdx
  unsigned __int64 v156; // r15
  __int64 v157; // r14
  unsigned __int8 *v158; // rax
  __int64 v159; // r12
  __int64 v160; // rax
  __int64 v161; // rdx
  __int64 v162; // rdx
  __int64 *v163; // r12
  __int64 *v164; // r14
  unsigned __int64 v165; // rdx
  unsigned __int64 v166; // r15
  __int64 v167; // rdi
  __int64 (__fastcall *v168)(__int64); // rcx
  __int64 (*v169)(); // rax
  char v170; // al
  bool v171; // al
  __int64 *v172; // rax
  __int64 v173; // rdx
  int v174; // eax
  unsigned __int8 *v175; // rcx
  char v176; // r14
  const void **v177; // rax
  __int64 v178; // rdi
  int v179; // eax
  bool v180; // al
  bool v181; // al
  __int64 v182; // rbx
  __int64 v183; // rsi
  __int64 *v184; // r12
  char v185; // r15
  int v186; // eax
  __int64 v187; // r14
  unsigned int v188; // ebx
  __int64 v189; // rax
  unsigned int v190; // eax
  const void **v191; // rdx
  const void **v192; // r14
  unsigned int v193; // ebx
  int v194; // eax
  __int128 v195; // rax
  __int128 v196; // rax
  __int128 v197; // kr00_16
  __int64 v198; // rdx
  int v199; // eax
  __int64 v200; // r10
  __int128 v201; // rax
  __int128 v202; // rax
  __int64 v203; // rax
  __int64 v204; // rdx
  bool v205; // al
  unsigned int v206; // ecx
  bool v207; // di
  unsigned int v208; // eax
  __int64 v209; // rdx
  unsigned int v210; // eax
  __int64 v211; // rdx
  __int64 v212; // rax
  __int64 *v213; // rdi
  __int64 v214; // rax
  __int64 **v215; // rax
  __int64 *v216; // r15
  __int64 v217; // r14
  __int128 v218; // rdi
  __int64 v219; // rcx
  __int64 *v220; // r12
  int v221; // r15d
  __int64 v222; // rax
  unsigned int v223; // edx
  unsigned __int8 v224; // al
  unsigned __int64 v225; // rdx
  __int64 v226; // rdx
  unsigned int v227; // eax
  __int64 v228; // r15
  unsigned int v229; // r14d
  __int64 v230; // rdx
  __int64 v231; // r15
  __int64 *v232; // r14
  __int64 v233; // rdi
  unsigned __int8 (__fastcall *v234)(__int64, __int64, _QWORD, __int64, __int64); // r9
  __int64 v235; // rax
  __int64 v236; // rdx
  __int64 v237; // rcx
  __int64 v238; // r8
  __int64 v239; // r9
  __int64 v240; // rax
  __int64 v241; // rdx
  __int64 *v242; // rax
  unsigned int v243; // edx
  unsigned int v244; // r14d
  const void **v245; // r11
  unsigned int v246; // r10d
  __int64 v247; // rax
  char v248; // dl
  __int64 v249; // rax
  unsigned __int128 v250; // kr10_16
  unsigned int v251; // esi
  __int64 *v252; // r14
  __int64 v253; // rdx
  __int64 v254; // r15
  const void ***v255; // rax
  __int64 *v256; // r15
  unsigned int v257; // edx
  unsigned int v258; // r12d
  _QWORD *v259; // r14
  __int64 v260; // rsi
  int v261; // eax
  __int64 v262; // rax
  __int64 v263; // rdx
  __int64 v264; // r14
  unsigned int v265; // edx
  __int64 v266; // r15
  __int64 v267; // rsi
  __int64 *v268; // r12
  const void **v269; // r8
  __int64 v270; // rcx
  unsigned int v271; // edx
  char v272; // r8
  unsigned int v273; // eax
  unsigned int v274; // edx
  bool v275; // al
  unsigned int v276; // eax
  __int128 v277; // [rsp-20h] [rbp-220h]
  __int128 v278; // [rsp-10h] [rbp-210h]
  __int128 v279; // [rsp-10h] [rbp-210h]
  __int128 v280; // [rsp-10h] [rbp-210h]
  __int128 v281; // [rsp-10h] [rbp-210h]
  __int64 v282; // [rsp+10h] [rbp-1F0h]
  __int64 v283; // [rsp+18h] [rbp-1E8h]
  __int64 *v284; // [rsp+18h] [rbp-1E8h]
  __int64 v285; // [rsp+20h] [rbp-1E0h]
  unsigned int v286; // [rsp+20h] [rbp-1E0h]
  __int64 v287; // [rsp+28h] [rbp-1D8h]
  const void **v288; // [rsp+28h] [rbp-1D8h]
  unsigned int v289; // [rsp+28h] [rbp-1D8h]
  unsigned int v290; // [rsp+28h] [rbp-1D8h]
  unsigned int v291; // [rsp+30h] [rbp-1D0h]
  char v292; // [rsp+30h] [rbp-1D0h]
  bool v293; // [rsp+30h] [rbp-1D0h]
  bool v294; // [rsp+30h] [rbp-1D0h]
  __int64 v295; // [rsp+30h] [rbp-1D0h]
  __int64 v296; // [rsp+30h] [rbp-1D0h]
  const void **v297; // [rsp+38h] [rbp-1C8h]
  char v298; // [rsp+40h] [rbp-1C0h]
  unsigned int *v299; // [rsp+40h] [rbp-1C0h]
  __int64 v300; // [rsp+40h] [rbp-1C0h]
  __int64 v301; // [rsp+48h] [rbp-1B8h]
  __int64 v302; // [rsp+60h] [rbp-1A0h]
  __int64 v303; // [rsp+68h] [rbp-198h]
  __int64 v304; // [rsp+70h] [rbp-190h]
  __int64 v305; // [rsp+70h] [rbp-190h]
  __int64 v306; // [rsp+78h] [rbp-188h]
  __int64 v307; // [rsp+80h] [rbp-180h]
  __int64 v308; // [rsp+80h] [rbp-180h]
  __int64 v309; // [rsp+88h] [rbp-178h]
  unsigned __int8 (__fastcall *v310)(__int64, __int64, _QWORD, __int64, __int64); // [rsp+88h] [rbp-178h]
  __int64 v311; // [rsp+90h] [rbp-170h]
  __int128 v312; // [rsp+90h] [rbp-170h]
  __int64 v313; // [rsp+A0h] [rbp-160h]
  unsigned __int8 v314; // [rsp+A0h] [rbp-160h]
  __int64 v315; // [rsp+A0h] [rbp-160h]
  __int128 v316; // [rsp+A0h] [rbp-160h]
  __int64 v317; // [rsp+A0h] [rbp-160h]
  _DWORD *v318; // [rsp+A0h] [rbp-160h]
  unsigned int v319; // [rsp+A0h] [rbp-160h]
  __int64 v320; // [rsp+B0h] [rbp-150h]
  _DWORD *v321; // [rsp+B0h] [rbp-150h]
  _DWORD *v322; // [rsp+B0h] [rbp-150h]
  __int64 v323; // [rsp+B0h] [rbp-150h]
  bool v324; // [rsp+B0h] [rbp-150h]
  int v325; // [rsp+B0h] [rbp-150h]
  __int64 v326; // [rsp+B8h] [rbp-148h]
  __int64 v327; // [rsp+C0h] [rbp-140h]
  __int64 v328; // [rsp+C8h] [rbp-138h]
  int v329; // [rsp+C8h] [rbp-138h]
  __int64 v330; // [rsp+C8h] [rbp-138h]
  unsigned int v331; // [rsp+D0h] [rbp-130h]
  const void **v332; // [rsp+D0h] [rbp-130h]
  __int64 v333; // [rsp+D0h] [rbp-130h]
  __int64 v334; // [rsp+D0h] [rbp-130h]
  char v335; // [rsp+D0h] [rbp-130h]
  unsigned __int64 v336; // [rsp+D8h] [rbp-128h]
  __int64 *v338; // [rsp+E0h] [rbp-120h]
  __int64 v339; // [rsp+E0h] [rbp-120h]
  unsigned __int8 *v340; // [rsp+E0h] [rbp-120h]
  unsigned __int8 *v341; // [rsp+E0h] [rbp-120h]
  unsigned int v342; // [rsp+E0h] [rbp-120h]
  __int64 *v343; // [rsp+E0h] [rbp-120h]
  unsigned int v344; // [rsp+E0h] [rbp-120h]
  unsigned __int128 v346; // [rsp+F0h] [rbp-110h] BYREF
  __int128 v347; // [rsp+100h] [rbp-100h]
  __int128 v348; // [rsp+110h] [rbp-F0h]
  __int64 v349; // [rsp+120h] [rbp-E0h]
  __int64 v350; // [rsp+128h] [rbp-D8h]
  __int64 v351; // [rsp+130h] [rbp-D0h]
  __int64 v352; // [rsp+138h] [rbp-C8h]
  __int64 v353; // [rsp+140h] [rbp-C0h]
  __int64 v354; // [rsp+148h] [rbp-B8h]
  __int64 v355; // [rsp+150h] [rbp-B0h]
  __int64 v356; // [rsp+158h] [rbp-A8h]
  unsigned int v357; // [rsp+160h] [rbp-A0h] BYREF
  const void **v358; // [rsp+168h] [rbp-98h]
  unsigned int v359; // [rsp+170h] [rbp-90h] BYREF
  const void **v360; // [rsp+178h] [rbp-88h]
  __int64 *v361; // [rsp+180h] [rbp-80h] BYREF
  __int64 v362; // [rsp+188h] [rbp-78h]
  __int64 *v363; // [rsp+190h] [rbp-70h] BYREF
  const void **v364; // [rsp+198h] [rbp-68h]
  __int64 v365; // [rsp+1A0h] [rbp-60h]
  __int128 v366; // [rsp+1B0h] [rbp-50h] BYREF
  __int64 v367; // [rsp+1C0h] [rbp-40h]

  v346 = __PAIR128__(a4, a3);
  *(_QWORD *)&v348 = a5;
  *(_QWORD *)&v347 = a3;
  v331 = a4;
  v303 = a10;
  v307 = a12;
  if ( (_QWORD)a12 == a10 && DWORD2(a12) == (_DWORD)a11 )
    return (__int64 *)a10;
  v311 = 16LL * (unsigned int)a11;
  v15 = *(_QWORD *)(a10 + 40) + v311;
  v16 = *(_BYTE *)v15;
  v297 = *(const void ***)(v15 + 8);
  v358 = v297;
  v298 = v16;
  v17 = *(unsigned __int16 *)(v348 + 24);
  LOBYTE(v357) = v16;
  if ( v17 == 32 || v17 == 10 )
  {
    v302 = v348;
    v18 = *(unsigned __int16 *)(a10 + 24);
    if ( v18 == 32 )
      goto LABEL_5;
  }
  else
  {
    v302 = 0;
    v18 = *(unsigned __int16 *)(a10 + 24);
    if ( v18 == 32 )
    {
LABEL_5:
      v304 = a10;
      goto LABEL_6;
    }
  }
  if ( v18 == 10 )
    goto LABEL_5;
  v304 = 0;
LABEL_6:
  v19 = a1[1];
  v301 = (unsigned int)a4;
  v309 = 16LL * (unsigned int)a4;
  v20 = (unsigned __int8 *)(v309 + *(_QWORD *)(v347 + 40));
  v313 = *((_QWORD *)v20 + 1);
  v21 = *(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v19 + 264LL);
  v327 = *v20;
  v320 = *(_QWORD *)(*a1 + 48LL);
  v22 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
  v23 = v21(v19, v22, v320, v327, v313);
  v24 = *a1;
  v25 = *((_DWORD *)a1 + 4);
  *(_QWORD *)&v366 = a1;
  v26 = a1[1];
  BYTE12(v366) = 0;
  v367 = v24;
  DWORD2(v366) = v25;
  v28 = sub_20ACAE0(v26, v23, v27, v346, DWORD2(v346), a13, v348, a6, 0, (__int64)&v366, a2);
  v29 = v28;
  if ( v28 )
  {
    sub_1F81BC0((__int64)a1, v28);
    v30 = *(unsigned __int16 *)(v29 + 24);
    if ( v30 == 10 || v30 == 32 )
    {
      v96 = *(_QWORD *)(v29 + 88);
      v97 = *(_DWORD *)(v96 + 32);
      if ( v97 <= 0x40 )
        v98 = *(_QWORD *)(v96 + 24) == 0;
      else
        v98 = v97 == (unsigned int)sub_16A57B0(v96 + 24);
      if ( !v98 )
        return (__int64 *)a10;
      return (__int64 *)a12;
    }
  }
  v31 = *(unsigned __int16 *)(a10 + 24);
  v32 = *(unsigned __int8 **)(a10 + 40);
  if ( v31 == 33 || (v33 = v31 == 11, v34 = &v32[v311], v33) )
  {
    v99 = *(unsigned __int16 *)(a12 + 24);
    if ( v99 != 33 && v99 != 11 )
    {
      v34 = &v32[v311];
      goto LABEL_11;
    }
    v34 = &v32[v311];
    v116 = v32[16 * (unsigned int)a11];
    if ( (_BYTE)v116 )
    {
      v117 = a1[1];
      if ( *(_QWORD *)(v117 + 8 * v116 + 120) )
      {
        if ( *(_BYTE *)(v117 + 259LL * (unsigned __int8)v116 + 2433) )
        {
          v118 = a1[1];
          v119 = *(__int64 (**)())(*(_QWORD *)v117 + 328LL);
          if ( v119 != sub_1F3CA70 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, __int64))v119)(
                   v117,
                   *(_QWORD *)(a10 + 88) + 24LL,
                   *v32,
                   *((_QWORD *)v32 + 1),
                   v118) )
            {
LABEL_102:
              v34 = (unsigned __int8 *)(*(_QWORD *)(a10 + 40) + v311);
              goto LABEL_11;
            }
            v118 = a1[1];
            v119 = *(__int64 (**)())(*(_QWORD *)v118 + 328LL);
          }
          if ( (v119 == sub_1F3CA70
             || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v119)(
                   v118,
                   *(_QWORD *)(a12 + 88) + 24LL,
                   **(unsigned __int8 **)(a12 + 40),
                   *(_QWORD *)(*(_QWORD *)(a12 + 40) + 8LL)))
            && ((v120 = *(_QWORD *)(a10 + 48)) != 0 && !*(_QWORD *)(v120 + 32)
             || (v121 = *(_QWORD *)(a12 + 48)) != 0 && !*(_QWORD *)(v121 + 32)) )
          {
            v215 = *(__int64 ***)(a12 + 88);
            v362 = *(_QWORD *)(a10 + 88);
            v361 = (__int64 *)v215;
            v216 = *v215;
            v217 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
            *((_QWORD *)&v218 + 1) = &v361;
            *(_QWORD *)&v218 = sub_1645D80(v216, 2);
            v220 = (__int64 *)sub_159DFD0(v218, 2, v219);
            v333 = *a1;
            v221 = sub_15AAE50(v217, (__int64)v216);
            v222 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
            v223 = 8 * sub_15A9520(v222, 0);
            if ( v223 == 32 )
            {
              v224 = 5;
            }
            else if ( v223 > 0x20 )
            {
              v224 = 6;
              if ( v223 != 64 )
              {
                v224 = 0;
                if ( v223 == 128 )
                  v224 = 7;
              }
            }
            else
            {
              v224 = 3;
              if ( v223 != 8 )
                v224 = 4 * (v223 == 16);
            }
            v334 = sub_1D2A150(v333, v220, v224, 0, v221, 0, 0, 0);
            v336 = v225;
            v319 = v225;
            v325 = *(_DWORD *)(v334 + 100);
            v306 = sub_1D38E70(*a1, 0, a2, 0, a7, a8, a9);
            v330 = v226;
            v227 = sub_12BE0A0(v217, *v361);
            v228 = *a1;
            v229 = v227;
            *(_QWORD *)&v366 = *(_QWORD *)(a12 + 72);
            if ( (_QWORD)v366 )
              sub_1F6CA20((__int64 *)&v366);
            DWORD2(v366) = *(_DWORD *)(a12 + 64);
            *(_QWORD *)&v312 = sub_1D38E70(v228, v229, (__int64)&v366, 0, a7, a8, a9);
            *((_QWORD *)&v312 + 1) = v230;
            if ( (_QWORD)v366 )
              sub_161E7C0((__int64)&v366, v366);
            v231 = a1[1];
            v232 = (__int64 *)*a1;
            v233 = *(_QWORD *)(*a1 + 32LL);
            v305 = *(_QWORD *)(*(_QWORD *)(v347 + 40) + v309 + 8);
            v234 = *(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)v231 + 264LL);
            v308 = *(unsigned __int8 *)(*(_QWORD *)(v347 + 40) + v309);
            *(_QWORD *)&v347 = *(_QWORD *)(*a1 + 48LL);
            v310 = v234;
            v235 = sub_1E0A0C0(v233);
            LODWORD(v231) = v310(v231, v235, v347, v308, v305);
            *(_QWORD *)&v347 = v236;
            *((_QWORD *)&v348 + 1) = a6;
            v240 = sub_1D28D50(v232, a13, v236, v237, v238, v239);
            *((_QWORD *)&v277 + 1) = a6;
            *(_QWORD *)&v277 = v348;
            v242 = sub_1D3A900(
                     v232,
                     0x89u,
                     a2,
                     (unsigned int)v231,
                     (const void **)v347,
                     0,
                     (__m128)a7,
                     a8,
                     a9,
                     v346,
                     *((__int16 **)&v346 + 1),
                     v277,
                     v240,
                     v241);
            v244 = v243;
            *(_QWORD *)&v347 = v242;
            sub_1F81BC0((__int64)a1, (__int64)v242);
            *(_QWORD *)&v348 = *a1;
            v245 = *(const void ***)(*(_QWORD *)(v306 + 40) + 16LL * (unsigned int)v330 + 8);
            v246 = *(unsigned __int8 *)(*(_QWORD *)(v306 + 40) + 16LL * (unsigned int)v330);
            v247 = *(_QWORD *)(v347 + 40) + 16LL * v244;
            v248 = *(_BYTE *)v247;
            v249 = *(_QWORD *)(v247 + 8);
            LOBYTE(v366) = v248;
            *((_QWORD *)&v366 + 1) = v249;
            v250 = __PAIR128__(v244, v347);
            if ( v248 )
            {
              v251 = 135 - ((unsigned __int8)(v248 - 14) >= 0x60u);
            }
            else
            {
              v344 = v246;
              *(_QWORD *)&v346 = v347;
              *(_QWORD *)&v347 = v245;
              *((_QWORD *)&v346 + 1) = v244;
              v275 = sub_1F58D20((__int64)&v366);
              v246 = v344;
              v250 = v346;
              v245 = (const void **)v347;
              v251 = 135 - !v275;
            }
            v252 = sub_1D3A900(
                     (__int64 *)v348,
                     v251,
                     a2,
                     v246,
                     v245,
                     0,
                     (__m128)a7,
                     a8,
                     a9,
                     v250,
                     *((__int16 **)&v250 + 1),
                     v312,
                     v306,
                     v330);
            v254 = v253;
            sub_1F81BC0((__int64)a1, (__int64)v252);
            v255 = (const void ***)(*(_QWORD *)(v334 + 40) + 16LL * v319);
            *((_QWORD *)&v280 + 1) = v254;
            *(_QWORD *)&v280 = v252;
            v256 = sub_1D332F0(
                     (__int64 *)*a1,
                     52,
                     a2,
                     *(unsigned __int8 *)v255,
                     v255[1],
                     0,
                     *(double *)a7.m128i_i64,
                     a8,
                     a9,
                     v334,
                     v336,
                     v280);
            v258 = v257;
            sub_1F81BC0((__int64)a1, (__int64)v256);
            v259 = (_QWORD *)*a1;
            v363 = 0;
            v364 = 0;
            v365 = 0;
            sub_1E34190((__int64)&v366, v259[4]);
            return (__int64 *)sub_1D2B730(
                                v259,
                                **(unsigned __int8 **)(a10 + 40),
                                *(_QWORD *)(*(_QWORD *)(a10 + 40) + 8LL),
                                a2,
                                *a1 + 88LL,
                                0,
                                (__int64)v256,
                                v258 | v336 & 0xFFFFFFFF00000000LL,
                                v366,
                                v367,
                                v325,
                                0,
                                (__int64)&v363,
                                0);
          }
          goto LABEL_102;
        }
      }
    }
  }
LABEL_11:
  v326 = a11;
  si128 = (__m128)_mm_load_si128((const __m128i *)&v346);
  v36 = *(_QWORD *)(v347 + 40) + v309;
  v37 = *(_BYTE *)v36;
  v360 = *(const void ***)(v36 + 8);
  v38 = (const void **)*((_QWORD *)v34 + 1);
  LOBYTE(v359) = v37;
  v39 = *v34;
  if ( !sub_1D185B0(a12) )
    goto LABEL_12;
  LOBYTE(v366) = v39;
  *((_QWORD *)&v366 + 1) = v38;
  if ( v39 != (_BYTE)v359 )
  {
    if ( (_BYTE)v359 )
    {
      v291 = sub_1F6C8D0(v359);
      goto LABEL_64;
    }
LABEL_138:
    v291 = sub_1F58D40((__int64)&v359);
LABEL_64:
    if ( v39 )
      v100 = sub_1F6C8D0(v39);
    else
      v100 = sub_1F58D40((__int64)&v366);
    if ( v100 > v291 )
      goto LABEL_12;
    goto LABEL_69;
  }
  if ( !v39 && v38 != v360 )
    goto LABEL_138;
LABEL_69:
  if ( a13 == 18 )
  {
    v167 = a1[1];
    v168 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v167 + 224LL);
    if ( v168 == sub_1F3D090 )
    {
      v169 = *(__int64 (**)())(*(_QWORD *)v167 + 216LL);
      v326 = a11;
      if ( v169 == sub_1F3CA20 )
        goto LABEL_13;
      v170 = ((__int64 (__fastcall *)(__int64, __int64))v169)(v167, a10);
    }
    else
    {
      v326 = a11;
      v170 = ((__int64 (__fastcall *)(__int64, __int64))v168)(v167, a10);
    }
    if ( !v170 )
      goto LABEL_13;
    if ( sub_1D188A0(v348) )
    {
LABEL_72:
      v287 = a1[1];
      v292 = *((_BYTE *)a1 + 25);
      v283 = *(_QWORD *)(*(_QWORD *)(v347 + 40) + v309 + 8);
      v285 = *(unsigned __int8 *)(*(_QWORD *)(v347 + 40) + v309);
      v101 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
      v102 = sub_1F40B60(v287, v285, v283, v101, v292);
      v103 = v359;
      v288 = v104;
      v286 = v102;
      v293 = *(_WORD *)(a10 + 24) == 10 || *(_WORD *)(a10 + 24) == 32;
      if ( !v293 )
        goto LABEL_192;
      v105 = *(_QWORD *)(a10 + 88);
      LODWORD(v362) = *(_DWORD *)(v105 + 32);
      if ( (unsigned int)v362 > 0x40 )
        sub_16A4FD0((__int64)&v361, (const void **)(v105 + 24));
      else
        v361 = *(__int64 **)(v105 + 24);
      sub_16A7800((__int64)&v361, 1u);
      v106 = v362;
      LODWORD(v362) = 0;
      v107 = *(_QWORD *)(a10 + 88);
      LODWORD(v364) = v106;
      v363 = v361;
      if ( v106 > 0x40 )
      {
        sub_16A8890((__int64 *)&v363, (__int64 *)(v107 + 24));
        v206 = (unsigned int)v364;
        v108 = (__int64)v363;
        LODWORD(v364) = 0;
        DWORD2(v366) = v206;
        *(_QWORD *)&v366 = v363;
        if ( v206 > 0x40 )
        {
          v284 = v363;
          if ( v206 - (unsigned int)sub_16A57B0((__int64)&v366) <= 0x40 )
          {
            v207 = v293;
            if ( *v284 )
              v207 = 0;
            v294 = v207;
          }
          else
          {
            v294 = 0;
          }
          if ( (_QWORD)v366 )
            j_j___libc_free_0_0(v366);
LABEL_78:
          if ( (unsigned int)v364 > 0x40 && v363 )
            j_j___libc_free_0_0(v363);
          if ( (unsigned int)v362 > 0x40 && v361 )
            j_j___libc_free_0_0(v361);
          v103 = v359;
          if ( v294 )
          {
            if ( (_BYTE)v359 )
              v109 = sub_1F6C8D0(v359);
            else
              v109 = sub_1F58D40((__int64)&v359);
            v110 = *(_QWORD *)(a10 + 88);
            v111 = *(_DWORD *)(v110 + 32);
            if ( v111 > 0x40 )
            {
              v109 = v109 - v111 + sub_16A57B0(v110 + 24);
            }
            else
            {
              v112 = *(_QWORD *)(v110 + 24);
              if ( v112 )
              {
                _BitScanReverse64(&v112, v112);
                v109 = v109 + (v112 ^ 0x3F) - 64;
              }
            }
            *(_QWORD *)&v113 = sub_1D38BB0(*a1, v109, a2, v286, v288, 0, (__m128i)si128, a8, a9, 0);
            *(_QWORD *)&v114 = sub_1D332F0(
                                 (__int64 *)*a1,
                                 124,
                                 a2,
                                 v359,
                                 v360,
                                 0,
                                 *(double *)si128.m128_u64,
                                 a8,
                                 a9,
                                 v347,
                                 si128.m128_u64[1] & 0xFFFFFFFF00000000LL | v331,
                                 v113);
            v316 = v114;
            v295 = v114;
            sub_1F81BC0((__int64)a1, v114);
            LOBYTE(v366) = v39;
            *((_QWORD *)&v366 + 1) = v38;
            if ( v39 == (_BYTE)v359 )
            {
              if ( v39 || v38 == v360 )
              {
LABEL_92:
                if ( a13 != 18 )
                  goto LABEL_93;
                v214 = v282;
                LOBYTE(v214) = v39;
                v282 = v214;
                v295 = (__int64)sub_1D3C080(
                                  (__int64 *)*a1,
                                  a2,
                                  v295,
                                  *((unsigned __int64 *)&v316 + 1),
                                  (unsigned int)v214,
                                  v38,
                                  (__m128i)si128,
                                  a8,
                                  a9);
                v353 = v295;
                v354 = v204;
                goto LABEL_198;
              }
            }
            else if ( (_BYTE)v359 )
            {
              v290 = sub_1F6C8D0(v359);
              goto LABEL_231;
            }
            v290 = sub_1F58D40((__int64)&v359);
LABEL_231:
            if ( v39 )
              v210 = sub_1F6C8D0(v39);
            else
              v210 = sub_1F58D40((__int64)&v366);
            if ( v210 < v290 )
            {
              v282 = v39;
              v295 = sub_1D309E0(
                       (__int64 *)*a1,
                       145,
                       a2,
                       v39,
                       v38,
                       0,
                       *(double *)si128.m128_u64,
                       a8,
                       *(double *)a9.m128i_i64,
                       v316);
              v355 = v295;
              v356 = v211;
              *((_QWORD *)&v316 + 1) = (unsigned int)v211 | *((_QWORD *)&v316 + 1) & 0xFFFFFFFF00000000LL;
              sub_1F81BC0((__int64)a1, v295);
            }
            goto LABEL_92;
          }
LABEL_192:
          if ( v103 )
          {
            v199 = sub_1F6C8D0(v103);
          }
          else
          {
            v296 = *a1;
            v199 = sub_1F58D40((__int64)&v359);
            v200 = v296;
          }
          *(_QWORD *)&v201 = sub_1D38BB0(v200, (unsigned int)(v199 - 1), a2, v286, v288, 0, (__m128i)si128, a8, a9, 0);
          *(_QWORD *)&v202 = sub_1D332F0(
                               (__int64 *)*a1,
                               123,
                               a2,
                               v359,
                               v360,
                               0,
                               *(double *)si128.m128_u64,
                               a8,
                               a9,
                               v347,
                               si128.m128_u64[1] & 0xFFFFFFFF00000000LL | v331,
                               v201);
          v316 = v202;
          v295 = v202;
          sub_1F81BC0((__int64)a1, v202);
          LOBYTE(v366) = v39;
          *((_QWORD *)&v366 + 1) = v38;
          if ( v39 == (_BYTE)v359 )
          {
            if ( v39 || v38 == v360 )
            {
LABEL_196:
              if ( a13 != 18 )
                goto LABEL_93;
              v203 = v282;
              LOBYTE(v203) = v39;
              v282 = v203;
              v295 = (__int64)sub_1D3C080(
                                (__int64 *)*a1,
                                a2,
                                v295,
                                *((unsigned __int64 *)&v316 + 1),
                                (unsigned int)v203,
                                v38,
                                (__m128i)si128,
                                a8,
                                a9);
              v349 = v295;
              v350 = v204;
LABEL_198:
              *((_QWORD *)&v316 + 1) = (unsigned int)v204 | *((_QWORD *)&v316 + 1) & 0xFFFFFFFF00000000LL;
LABEL_93:
              v115 = v282;
              LOBYTE(v115) = v39;
              *((_QWORD *)&v279 + 1) = (unsigned int)a11 | v326 & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v279 = a10;
              result = sub_1D332F0(
                         (__int64 *)*a1,
                         118,
                         a2,
                         v115,
                         v38,
                         0,
                         *(double *)si128.m128_u64,
                         a8,
                         a9,
                         v295,
                         *((unsigned __int64 *)&v316 + 1),
                         v279);
              if ( result )
                return result;
              goto LABEL_12;
            }
          }
          else if ( (_BYTE)v359 )
          {
            v289 = sub_1F6C8D0(v359);
            goto LABEL_224;
          }
          v289 = sub_1F58D40((__int64)&v359);
LABEL_224:
          if ( v39 )
            v208 = sub_1F6C8D0(v39);
          else
            v208 = sub_1F58D40((__int64)&v366);
          if ( v208 < v289 )
          {
            v282 = v39;
            v295 = sub_1D309E0(
                     (__int64 *)*a1,
                     145,
                     a2,
                     v39,
                     v38,
                     0,
                     *(double *)si128.m128_u64,
                     a8,
                     *(double *)a9.m128i_i64,
                     v316);
            v351 = v295;
            v352 = v209;
            *((_QWORD *)&v316 + 1) = (unsigned int)v209 | *((_QWORD *)&v316 + 1) & 0xFFFFFFFF00000000LL;
            sub_1F81BC0((__int64)a1, v295);
          }
          goto LABEL_196;
        }
      }
      else
      {
        v108 = *(_QWORD *)(v107 + 24) & (unsigned __int64)v361;
        DWORD2(v366) = v106;
        v363 = (__int64 *)v108;
        *(_QWORD *)&v366 = v108;
        LODWORD(v364) = 0;
      }
      v294 = v108 == 0;
      goto LABEL_78;
    }
    v171 = sub_1D185B0(v348);
    if ( a10 != (_QWORD)v347 || (_DWORD)a11 != v331 )
      goto LABEL_13;
LABEL_165:
    if ( !v171 )
      goto LABEL_13;
    goto LABEL_72;
  }
  if ( a13 == 20 )
  {
    if ( sub_1D185B0(v348) )
      goto LABEL_72;
    v171 = sub_1D18910(v348);
    if ( (_DWORD)a11 != v331 || a10 != (_QWORD)v347 )
      goto LABEL_13;
    goto LABEL_165;
  }
LABEL_12:
  if ( a13 == 17 && *(_WORD *)(v347 + 24) == 118 )
  {
    v138 = *(_QWORD *)(v347 + 40);
    if ( *(_BYTE *)v138 == v298
      && (*(const void ***)(v138 + 8) == v297 || v298)
      && sub_1D185B0(v348)
      && sub_1D185B0(a10) )
    {
      v139 = *(__int64 **)(v347 + 32);
      v140 = v139[5];
      v141 = *v139;
      v142 = v139[1];
      v143 = *(unsigned __int16 *)(v140 + 24);
      if ( v143 == 32 || v143 == 10 )
      {
        v144 = *(_QWORD *)(v140 + 88);
        v299 = *(unsigned int **)(v347 + 32);
        v317 = v144;
        v323 = v144 + 24;
        v145 = *(_DWORD *)(v144 + 32) > 0x40u ? sub_16A5940(v323) : sub_39FAC40(*(_QWORD *)(v144 + 24));
        if ( v145 == 1 )
        {
          v146 = *a1;
          v147 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v299 + 40LL) + 16LL * v299[2]);
          v148 = sub_1F6BF40((__int64)a1, *v147, *((_QWORD *)v147 + 1));
          v150 = v149;
          *(_QWORD *)&v348 = v148;
          sub_1F80610((__int64)&v366, v141);
          v151 = sub_1455840(v323);
          *(_QWORD *)&v348 = sub_1D38BB0(v146, v151, (__int64)&v366, v348, v150, 0, (__m128i)si128, a8, a9, 0);
          *((_QWORD *)&v348 + 1) = v152;
          sub_17CD270((__int64 *)&v366);
          v153 = (__int64 *)*a1;
          *(_QWORD *)&v346 = v347;
          sub_1F80610((__int64)&v366, v347);
          v154 = sub_1D332F0(
                   v153,
                   122,
                   (__int64)&v366,
                   v357,
                   v358,
                   0,
                   *(double *)si128.m128_u64,
                   a8,
                   a9,
                   v141,
                   v142,
                   v348);
          v156 = v155;
          v157 = (__int64)v154;
          sub_17CD270((__int64 *)&v366);
          v158 = (unsigned __int8 *)(*(_QWORD *)(v157 + 40) + 16LL * (unsigned int)v156);
          v159 = *a1;
          v160 = sub_1F6BF40((__int64)a1, *v158, *((_QWORD *)v158 + 1));
          *(_QWORD *)&v347 = v161;
          *(_QWORD *)&v348 = v160;
          sub_1F80610((__int64)&v366, v157);
          *(_QWORD *)&v348 = sub_1D38BB0(
                               v159,
                               (unsigned int)(*(_DWORD *)(v317 + 32) - 1),
                               (__int64)&v366,
                               v348,
                               (const void **)v347,
                               0,
                               (__m128i)si128,
                               a8,
                               a9,
                               0);
          *((_QWORD *)&v348 + 1) = v162;
          sub_17CD270((__int64 *)&v366);
          v163 = (__int64 *)*a1;
          sub_1F80610((__int64)&v366, v346);
          v164 = sub_1D332F0(
                   v163,
                   123,
                   (__int64)&v366,
                   v357,
                   v358,
                   0,
                   *(double *)si128.m128_u64,
                   a8,
                   a9,
                   v157,
                   v156,
                   v348);
          v166 = v165;
          sub_17CD270((__int64 *)&v366);
          return sub_1D332F0(
                   (__int64 *)*a1,
                   118,
                   a2,
                   v357,
                   v358,
                   0,
                   *(double *)si128.m128_u64,
                   a8,
                   a9,
                   (__int64)v164,
                   v166,
                   a12);
        }
      }
    }
  }
LABEL_13:
  if ( !v304 || !sub_1D185B0(a12) )
    goto LABEL_104;
  v40 = *(_QWORD *)(v304 + 88);
  v41 = *(_DWORD *)(v40 + 32);
  v42 = v40 + 24;
  if ( v41 > 0x40 )
  {
    v122 = sub_16A5940(v40 + 24);
    v42 = v40 + 24;
    if ( v122 != 1 )
      goto LABEL_104;
  }
  else
  {
    v43 = *(_QWORD *)(v40 + 24);
    if ( !v43 || (v43 & (v43 - 1)) != 0 )
      goto LABEL_104;
  }
  v44 = (_DWORD *)a1[1];
  v45 = (unsigned __int8 *)(*(_QWORD *)(v347 + 40) + v309);
  v46 = *v45;
  v47 = *((_QWORD *)v45 + 1);
  LOBYTE(v366) = v46;
  *((_QWORD *)&v366 + 1) = v47;
  if ( (_BYTE)v46 )
  {
    if ( (unsigned __int8)(v46 - 14) > 0x5Fu )
    {
      v48 = (unsigned __int8)(v46 - 86) <= 0x17u || (unsigned __int8)(v46 - 8) <= 5u;
      goto LABEL_21;
    }
  }
  else
  {
    v300 = v42;
    v318 = v44;
    v324 = sub_1F58CD0((__int64)&v366);
    v205 = sub_1F58D20((__int64)&v366);
    v48 = v324;
    v44 = v318;
    v42 = v300;
    v46 = 0;
    if ( !v205 )
    {
LABEL_21:
      if ( v48 )
        v49 = v44[16];
      else
        v49 = v44[15];
      goto LABEL_23;
    }
  }
  v49 = v44[17];
LABEL_23:
  if ( v49 != 1 )
    goto LABEL_104;
  if ( a14 )
  {
    if ( v41 <= 0x40 )
    {
      v51 = *(_QWORD *)(v40 + 24) == 1;
    }
    else
    {
      v314 = v46;
      v321 = v44;
      v50 = sub_16A57B0(v42);
      v44 = v321;
      v46 = v314;
      v51 = v41 - 1 == v50;
    }
    if ( v51 )
      return 0;
  }
  if ( *((_BYTE *)a1 + 24) )
  {
    if ( (v52 = 1, (_BYTE)v46 != 1)
      && (!(_BYTE)v46 || (v52 = (unsigned __int8)v46, !*(_QWORD *)&v44[2 * (unsigned __int8)v46 + 30]))
      || *((_BYTE *)v44 + 259 * v52 + 2559) )
    {
LABEL_104:
      if ( !v302 )
        return 0;
      v123 = *(_QWORD *)(v302 + 88);
      v124 = *(_DWORD *)(v123 + 32);
      *(_QWORD *)&v348 = v123;
      v125 = v123 + 24;
      if ( v124 <= 0x40 )
        v126 = *(_QWORD *)(v123 + 24) == 0;
      else
        v126 = v124 == (unsigned int)sub_16A57B0(v123 + 24);
      if ( v126 && a13 - 18 <= 1 )
      {
        if ( (_QWORD)v347 != a10 || (_DWORD)a11 != v331 )
          goto LABEL_114;
        if ( *(_WORD *)(a12 + 24) != 53 )
        {
LABEL_155:
          if ( a13 - 20 <= 1 )
          {
LABEL_156:
            if ( (_QWORD)a12 != (_QWORD)v347 )
              goto LABEL_157;
            if ( DWORD2(a12) != v331 )
              goto LABEL_157;
            if ( *(_WORD *)(a10 + 24) != 53 )
              goto LABEL_157;
            v172 = *(__int64 **)(a10 + 32);
            if ( v172[5] != (_QWORD)a12 || *((_DWORD *)v172 + 12) != v331 )
              goto LABEL_157;
            goto LABEL_176;
          }
LABEL_114:
          if ( v124 <= 0x40 )
          {
            if ( *(_QWORD *)(v348 + 24) != 1 )
              goto LABEL_158;
          }
          else if ( (unsigned int)sub_16A57B0(v125) != v124 - 1 )
          {
LABEL_116:
            v128 = v124 == (unsigned int)sub_16A57B0(v125);
            goto LABEL_117;
          }
          if ( a13 != 20 )
          {
LABEL_157:
            if ( v124 > 0x40 )
              goto LABEL_116;
LABEL_158:
            v128 = *(_QWORD *)(v348 + 24) == 0;
LABEL_117:
            if ( v128 )
            {
              if ( a13 != 17 )
              {
                if ( a13 != 22 )
                  return 0;
                v307 = a10;
                v303 = a12;
              }
              v129 = *(unsigned __int16 *)(v303 + 24);
              if ( v129 != 10 && v129 != 32 )
                return 0;
              v130 = v357;
              if ( (_BYTE)v357 )
                v131 = sub_1F6C8D0(v357);
              else
                v131 = sub_1F58D40((__int64)&v357);
              v132 = v131;
              v133 = *(_QWORD *)(v303 + 88);
              v134 = *(_DWORD *)(v133 + 32);
              if ( v134 > 0x40 )
              {
                if ( v134 - (unsigned int)sub_16A57B0(v133 + 24) > 0x40 )
                  return 0;
                v135 = **(_QWORD **)(v133 + 24);
              }
              else
              {
                v135 = *(_QWORD *)(v133 + 24);
              }
              if ( v132 == v135 )
              {
                LOWORD(v348) = *(_WORD *)(v307 + 24);
                if ( (v348 & 0xFFFB) == 0x80 )
                {
                  v136 = *(_QWORD *)(v307 + 32);
                  if ( *(_QWORD *)v136 == (_QWORD)v347
                    && v331 == *(_DWORD *)(v136 + 8)
                    && (!*((_BYTE *)a1 + 24) || sub_1F6C830(a1[1], 0x80u, v130)) )
                  {
                    v137 = (__int64 *)*a1;
                    *(_QWORD *)&v346 = v347;
                    return (__int64 *)sub_1D309E0(
                                        v137,
                                        128,
                                        a2,
                                        v357,
                                        v358,
                                        0,
                                        *(double *)si128.m128_u64,
                                        a8,
                                        *(double *)a9.m128i_i64,
                                        __PAIR128__(v331 | *((_QWORD *)&v346 + 1) & 0xFFFFFFFF00000000LL, v347));
                  }
                }
                else if ( (v348 & 0xFFFB) == 0x81 )
                {
                  v212 = *(_QWORD *)(v307 + 32);
                  if ( *(_QWORD *)v212 == (_QWORD)v347
                    && v331 == *(_DWORD *)(v212 + 8)
                    && (!*((_BYTE *)a1 + 24) || sub_1F6C830(a1[1], 0x81u, v130)) )
                  {
                    v213 = (__int64 *)*a1;
                    *(_QWORD *)&v346 = v347;
                    return (__int64 *)sub_1D309E0(
                                        v213,
                                        129,
                                        a2,
                                        v357,
                                        v358,
                                        0,
                                        *(double *)si128.m128_u64,
                                        a8,
                                        *(double *)a9.m128i_i64,
                                        __PAIR128__(v331 | *((_QWORD *)&v346 + 1) & 0xFFFFFFFF00000000LL, v347));
                  }
                }
              }
            }
            return 0;
          }
          goto LABEL_156;
        }
      }
      else
      {
        if ( v124 <= 0x40 )
          v127 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v124) == *(_QWORD *)(v348 + 24);
        else
          v127 = v124 == (unsigned int)sub_16A58F0(v125);
        if ( a13 != 18 || !v127 )
          goto LABEL_113;
        if ( (_QWORD)v347 != a10 || (_DWORD)a11 != v331 )
          goto LABEL_114;
        if ( *(_WORD *)(a12 + 24) != 53 )
          goto LABEL_113;
      }
      v172 = *(__int64 **)(a12 + 32);
      if ( v172[5] == (_QWORD)v347 && *((_DWORD *)v172 + 12) == v331 )
      {
LABEL_176:
        v173 = *v172;
        v174 = *(unsigned __int16 *)(*v172 + 24);
        if ( v174 == 10 || v174 == 32 )
        {
          v175 = (unsigned __int8 *)(*(_QWORD *)(v347 + 40) + v309);
          v176 = *v175;
          v177 = (const void **)*((_QWORD *)v175 + 1);
          LOBYTE(v363) = *v175;
          v364 = v177;
          v178 = *(_QWORD *)(v173 + 88);
          if ( *(_DWORD *)(v178 + 32) <= 0x40u )
          {
            v180 = *(_QWORD *)(v178 + 24) == 0;
          }
          else
          {
            v329 = *(_DWORD *)(v178 + 32);
            v340 = v175;
            v179 = sub_16A57B0(v178 + 24);
            v175 = v340;
            v180 = v329 == v179;
          }
          if ( v180 )
          {
            if ( v176 )
            {
              v181 = (unsigned __int8)(v176 - 14) <= 0x47u || (unsigned __int8)(v176 - 2) <= 5u;
            }
            else
            {
              v341 = v175;
              v181 = sub_1F58CF0((__int64)&v363);
              v175 = v341;
            }
            if ( v181 )
            {
              v182 = v347;
              v183 = *(_QWORD *)(v347 + 72);
              *(_QWORD *)&v366 = v183;
              if ( v183 )
              {
                sub_1623A60((__int64)&v366, v183, 2);
                v175 = (unsigned __int8 *)(*(_QWORD *)(v182 + 40) + v309);
              }
              v184 = (__int64 *)*a1;
              v185 = *((_BYTE *)a1 + 25);
              v186 = *(_DWORD *)(v347 + 64);
              *(_QWORD *)&v348 = a1[1];
              DWORD2(v366) = v186;
              v187 = *((_QWORD *)v175 + 1);
              v188 = *v175;
              v189 = sub_1E0A0C0(v184[4]);
              v190 = sub_1F40B60(v348, v188, v187, v189, v185);
              v192 = v191;
              v193 = v190;
              if ( (_BYTE)v363 )
                v194 = sub_1F6C8D0((char)v363);
              else
                v194 = sub_1F58D40((__int64)&v363);
              *(_QWORD *)&v195 = sub_1D38BB0(
                                   (__int64)v184,
                                   (unsigned int)(v194 - 1),
                                   (__int64)&v366,
                                   v193,
                                   v192,
                                   0,
                                   (__m128i)si128,
                                   a8,
                                   a9,
                                   0);
              *(_QWORD *)&v346 = v347;
              *((_QWORD *)&v346 + 1) = v331 | *((_QWORD *)&v346 + 1) & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v196 = sub_1D332F0(
                                   v184,
                                   123,
                                   (__int64)&v366,
                                   (unsigned int)v363,
                                   v364,
                                   0,
                                   *(double *)si128.m128_u64,
                                   a8,
                                   a9,
                                   v347,
                                   *((unsigned __int64 *)&v346 + 1),
                                   v195);
              v197 = v196;
              *(_QWORD *)&v348 = sub_1D332F0(
                                   (__int64 *)*a1,
                                   52,
                                   (__int64)&v366,
                                   (unsigned int)v363,
                                   v364,
                                   0,
                                   *(double *)si128.m128_u64,
                                   a8,
                                   a9,
                                   v346,
                                   *((unsigned __int64 *)&v346 + 1),
                                   v196);
              *((_QWORD *)&v348 + 1) = v198;
              sub_1F81BC0((__int64)a1, v197);
              sub_1F81BC0((__int64)a1, v348);
              result = sub_1D332F0(
                         (__int64 *)*a1,
                         120,
                         (__int64)&v366,
                         (unsigned int)v363,
                         v364,
                         0,
                         *(double *)si128.m128_u64,
                         a8,
                         a9,
                         v348,
                         *((unsigned __int64 *)&v348 + 1),
                         v197);
              if ( (_QWORD)v366 )
                goto LABEL_47;
              return result;
            }
          }
        }
        goto LABEL_157;
      }
LABEL_113:
      if ( !v126 )
        goto LABEL_114;
      goto LABEL_155;
    }
  }
  v53 = (__int64 *)*a1;
  if ( *((_BYTE *)a1 + 25) )
  {
    v322 = v44;
    v54 = *(unsigned __int8 (__fastcall **)(_DWORD *, __int64, __int64, __int64, __int64))(*(_QWORD *)v44 + 264LL);
    v315 = (unsigned __int8)v46;
    v328 = v53[6];
    v55 = sub_1E0A0C0(v53[4]);
    LODWORD(v54) = v54(v322, v55, v328, v315, v47);
    v332 = (const void **)v56;
    *(_QWORD *)&v346 = v347;
    *((_QWORD *)&v346 + 1) = v301 | *((_QWORD *)&v346 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v347 = v348;
    *((_QWORD *)&v347 + 1) = a6;
    v60 = sub_1D28D50(v53, a13, v56, v57, v58, v59);
    *(_QWORD *)&v347 = sub_1D3A900(
                         v53,
                         0x89u,
                         a2,
                         (unsigned int)v54,
                         v332,
                         0,
                         si128,
                         a8,
                         a9,
                         v346,
                         *((__int16 **)&v346 + 1),
                         v347,
                         v60,
                         v61);
    v62 = v347;
    v64 = v63;
    v65 = v63;
    v66 = *(_QWORD *)(a10 + 40) + v311;
    v67 = *(const void ***)(v66 + 8);
    LOBYTE(v363) = *(_BYTE *)v66;
    v364 = v67;
    v68 = *(_QWORD *)(v347 + 40) + 16 * v64;
    v69 = *(_BYTE *)v68;
    v70 = *(const void ***)(v68 + 8);
    LOBYTE(v348) = (_BYTE)v363;
    *(_QWORD *)&v346 = v67;
    LOBYTE(v366) = v69;
    *((_QWORD *)&v366 + 1) = v70;
    if ( v69 == (_BYTE)v363 )
    {
      if ( v69 || v70 == v67 )
      {
LABEL_36:
        v338 = (__int64 *)*a1;
        *(_QWORD *)&v348 = (unsigned __int8)v348;
        sub_1F80610((__int64)&v366, a10);
        *((_QWORD *)&v278 + 1) = v65;
        *(_QWORD *)&v278 = v62;
        *(_QWORD *)&v348 = sub_1D309E0(
                             v338,
                             143,
                             (__int64)&v366,
                             v348,
                             (const void **)v346,
                             0,
                             *(double *)si128.m128_u64,
                             a8,
                             *(double *)a9.m128i_i64,
                             v278);
        v71 = v348;
        v73 = v72;
        *((_QWORD *)&v348 + 1) = v72;
        sub_17CD270((__int64 *)&v366);
        goto LABEL_37;
      }
    }
    else if ( (_BYTE)v348 )
    {
      v342 = sub_1F6C8D0(v348);
      goto LABEL_286;
    }
    v335 = v69;
    v276 = sub_1F58D40((__int64)&v363);
    v272 = v335;
    v342 = v276;
LABEL_286:
    if ( v272 )
      v273 = sub_1F6C8D0(v272);
    else
      v273 = sub_1F58D40((__int64)&v366);
    if ( v273 > v342 )
    {
      v343 = (__int64 *)*a1;
      *(_QWORD *)&v348 = (unsigned __int8)v348;
      sub_1F80610((__int64)&v366, a10);
      *(_QWORD *)&v348 = sub_1D3BC50(v343, v62, v65, (__int64)&v366, v348, v346, (__m128i)si128, a8, a9);
      v71 = v348;
      v73 = v274;
      *((_QWORD *)&v348 + 1) = v274;
      sub_17CD270((__int64 *)&v366);
      goto LABEL_37;
    }
    goto LABEL_36;
  }
  v260 = *(_QWORD *)(v347 + 72);
  *(_QWORD *)&v366 = v260;
  if ( v260 )
    sub_1623A60((__int64)&v366, v260, 2);
  v261 = *(_DWORD *)(v347 + 64);
  *(_QWORD *)&v346 = v347;
  DWORD2(v366) = v261;
  *((_QWORD *)&v348 + 1) = a6;
  *((_QWORD *)&v346 + 1) = v301 | *((_QWORD *)&v346 + 1) & 0xFFFFFFFF00000000LL;
  v262 = sub_1D28D50(v53, a13, v46, v347, v348, a6);
  *(_QWORD *)&v347 = sub_1D3A900(
                       v53,
                       0x89u,
                       (__int64)&v366,
                       2u,
                       0,
                       0,
                       si128,
                       a8,
                       a9,
                       v346,
                       *((__int16 **)&v346 + 1),
                       v348,
                       v262,
                       v263);
  v264 = v347;
  v266 = v265;
  if ( (_QWORD)v366 )
    sub_161E7C0((__int64)&v366, v366);
  v267 = *(_QWORD *)(a10 + 72);
  v268 = (__int64 *)*a1;
  v269 = *(const void ***)(*(_QWORD *)(a10 + 40) + v311 + 8);
  v270 = *(unsigned __int8 *)(*(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11);
  *(_QWORD *)&v366 = v267;
  if ( v267 )
  {
    *(_QWORD *)&v346 = v270;
    *(_QWORD *)&v348 = v269;
    sub_1623A60((__int64)&v366, v267, 2);
    v270 = v346;
    v269 = (const void **)v348;
  }
  *((_QWORD *)&v281 + 1) = v266;
  *(_QWORD *)&v281 = v264;
  DWORD2(v366) = *(_DWORD *)(a10 + 64);
  *(_QWORD *)&v348 = sub_1D309E0(
                       v268,
                       143,
                       (__int64)&v366,
                       v270,
                       v269,
                       0,
                       *(double *)si128.m128_u64,
                       a8,
                       *(double *)a9.m128i_i64,
                       v281);
  v71 = v348;
  v73 = v271;
  *((_QWORD *)&v348 + 1) = v271;
  if ( (_QWORD)v366 )
    sub_161E7C0((__int64)&v366, v366);
LABEL_37:
  sub_1F81BC0((__int64)a1, v347);
  sub_1F81BC0((__int64)a1, v71);
  v74 = *(_QWORD *)(v304 + 88);
  v75 = *(_DWORD *)(v74 + 32);
  if ( v75 <= 0x40 )
    v76 = *(_QWORD *)(v74 + 24) == 1;
  else
    v76 = v75 - 1 == (unsigned int)sub_16A57B0(v74 + 24);
  result = (__int64 *)v71;
  if ( !v76 )
  {
    v78 = v73;
    v79 = (__int64 *)*a1;
    v80 = a1[1];
    v81 = *((_BYTE *)a1 + 25);
    v339 = v78;
    v82 = v79[4];
    v83 = *(unsigned __int8 *)(*(_QWORD *)(v71 + 40) + 16 * v78);
    *(_QWORD *)&v346 = *(_QWORD *)(*(_QWORD *)(v71 + 40) + 16 * v78 + 8);
    *(_QWORD *)&v347 = v83;
    v84 = sub_1E0A0C0(v82);
    v85 = sub_1F40B60(v80, v347, v346, v84, v81);
    v86 = *(_QWORD *)(v71 + 72);
    *(_QWORD *)&v348 = v71;
    v87 = v85;
    v89 = v88;
    *(_QWORD *)&v366 = v86;
    *((_QWORD *)&v348 + 1) = v339 | *((_QWORD *)&v348 + 1) & 0xFFFFFFFF00000000LL;
    if ( v86 )
      sub_1623A60((__int64)&v366, v86, 2);
    DWORD2(v366) = *(_DWORD *)(v71 + 64);
    v90 = *(_QWORD *)(v304 + 88);
    v91 = *(_DWORD *)(v90 + 32);
    if ( v91 > 0x40 )
    {
      v93 = sub_16A57B0(v90 + 24);
    }
    else
    {
      v92 = *(_QWORD *)(v90 + 24);
      if ( v92 )
      {
        _BitScanReverse64(&v92, v92);
        LODWORD(v92) = v92 ^ 0x3F;
      }
      else
      {
        LODWORD(v92) = 64;
      }
      v93 = v91 + v92 - 64;
    }
    *(_QWORD *)&v94 = sub_1D38BB0((__int64)v79, v91 - 1 - v93, (__int64)&v366, v87, v89, 0, (__m128i)si128, a8, a9, 0);
    result = sub_1D332F0(
               v79,
               122,
               a2,
               *(unsigned __int8 *)(*(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11),
               *(const void ***)(*(_QWORD *)(a10 + 40) + 16LL * (unsigned int)a11 + 8),
               0,
               *(double *)si128.m128_u64,
               a8,
               a9,
               v348,
               *((unsigned __int64 *)&v348 + 1),
               v94);
    if ( (_QWORD)v366 )
    {
LABEL_47:
      *(_QWORD *)&v347 = v95;
      *(_QWORD *)&v348 = result;
      sub_161E7C0((__int64)&v366, v366);
      return (__int64 *)v348;
    }
  }
  return result;
}
