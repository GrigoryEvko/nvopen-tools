// Function: sub_128D0F0
// Address: 0x128d0f0
//
char *__fastcall sub_128D0F0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  __int64 *v8; // rdi
  _DWORD *v9; // rcx
  char *result; // rax
  __int64 *v11; // rsi
  __int64 v12; // r15
  __int64 *v13; // rsi
  unsigned __int64 v14; // rax
  char n; // dl
  __int64 *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  _QWORD *v19; // r14
  unsigned __int8 v20; // al
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  const char **v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // rbx
  __int64 v29; // rax
  const char **v30; // rbx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rbx
  __int64 *v34; // r13
  __int64 *v35; // r14
  __int64 *v36; // rsi
  __int64 *v37; // rt0
  __int64 *v38; // rt1
  __int64 i; // rax
  char v40; // dl
  __int64 v41; // r14
  __int64 *v42; // r13
  __int64 v43; // r15
  __int64 *v44; // rax
  __int64 *v45; // r12
  __int64 v46; // rax
  char v47; // dl
  __int64 v48; // rsi
  __int64 v49; // rcx
  unsigned __int8 v50; // dl
  __int64 *v51; // r14
  __int64 v52; // r15
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 *v56; // rdi
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v59; // r14
  __int64 v60; // r13
  __int64 *v61; // rsi
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 *v64; // rdi
  __int64 *v65; // rbx
  __int64 v66; // rax
  __int64 *v67; // rdi
  __int64 v68; // r14
  __int64 *v69; // rax
  const char **v70; // rbx
  __int64 *v71; // rsi
  __int64 v72; // rax
  __int64 v73; // r13
  const char *v74; // rax
  __int64 *v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // rax
  const char *v78; // rsi
  __int64 v79; // rdx
  const char *v80; // rsi
  int v81; // eax
  __int64 v82; // rax
  int v83; // edx
  __int64 v84; // rdx
  __int64 *v85; // rax
  __int64 v86; // rcx
  unsigned __int64 v87; // rsi
  __int64 *v88; // rbx
  __int64 v89; // rcx
  unsigned __int64 v90; // rsi
  __int64 *v91; // rbx
  __int64 v92; // rax
  __int64 v93; // rcx
  int v94; // eax
  __int64 v95; // rax
  int v96; // edx
  __int64 v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 v103; // r13
  const char **v104; // rax
  const char *v105; // rsi
  const char *v106; // rsi
  __int64 *v107; // rdx
  __int64 v108; // rax
  __int64 v109; // r14
  __int64 *v110; // rdi
  __int64 *v111; // rax
  __int64 *v112; // rdi
  __int64 *v113; // r15
  __int64 v114; // rax
  __int64 *v115; // r13
  __int64 v116; // rbx
  __int64 v117; // rax
  __int64 *v118; // rdi
  __int64 v119; // r14
  __int64 v120; // rax
  __int64 v121; // r13
  __int64 v122; // r14
  __int64 v123; // rax
  __int64 v124; // r14
  __int64 v125; // rbx
  int v126; // eax
  __int64 v127; // rax
  int v128; // edx
  __int64 v129; // rdx
  __int64 *v130; // rax
  __int64 v131; // rcx
  unsigned __int64 v132; // rsi
  __int64 v133; // rcx
  __int64 v134; // rdx
  __int64 v135; // rcx
  const char *v136; // rsi
  const char *v137; // rsi
  const char *v138; // rax
  __int64 v139; // rsi
  __int64 **v140; // rax
  __int64 *v141; // rdi
  unsigned __int64 v142; // rsi
  __int64 v143; // rsi
  __int64 v144; // rcx
  __int64 v145; // rsi
  __int64 v146; // rbx
  __int64 *v147; // rcx
  int v148; // eax
  __int64 v149; // rax
  int v150; // esi
  __int64 v151; // r14
  __int64 *v152; // rdi
  __int64 *v153; // rax
  __int64 *v154; // rdi
  __int64 *v155; // r15
  __int64 v156; // rax
  __int64 *v157; // r13
  __int64 v158; // rbx
  __int64 v159; // rax
  __int64 *v160; // rdi
  __int64 v161; // r14
  __int64 v162; // rax
  __int64 v163; // r13
  __int64 v164; // r14
  __int64 v165; // rax
  __int64 v166; // r14
  __int64 v167; // rbx
  int v168; // eax
  __int64 v169; // rax
  int v170; // edx
  __int64 v171; // rdx
  __int64 *v172; // rax
  __int64 v173; // rcx
  unsigned __int64 v174; // rsi
  __int64 v175; // rcx
  __int64 v176; // rdx
  __int64 v177; // rcx
  const char *v178; // rsi
  const char *v179; // rsi
  __int64 *v180; // rdi
  __int64 v181; // rsi
  __int64 **v182; // rax
  __int64 *v183; // rdi
  unsigned __int64 v184; // rsi
  __int64 v185; // rsi
  __int64 v186; // rcx
  __int64 v187; // rsi
  __int64 v188; // rbx
  __int64 *v189; // rcx
  int v190; // eax
  __int64 v191; // rax
  int v192; // esi
  __int64 v193; // r15
  __int64 **v194; // rdx
  __int64 *v195; // r8
  __int64 *k; // rax
  __int64 *v197; // rax
  __int64 *v198; // rsi
  __int64 v199; // rdx
  __int64 v200; // rcx
  __int64 *v201; // r8
  __int64 v202; // r9
  unsigned int v203; // ebx
  __int64 *v204; // r10
  __int64 *v205; // rsi
  unsigned __int64 v206; // rax
  char m; // dl
  void *v208; // rdx
  __int64 v209; // r14
  _BYTE *v210; // rax
  __int64 *v211; // rsi
  __int64 v212; // rdx
  _DWORD *v213; // r14
  __int64 v214; // rcx
  __int64 v215; // r8
  __int64 v216; // r9
  int v217; // ebx
  __int64 *v218; // r8
  _BYTE *v219; // rax
  char **v220; // rsi
  __int64 *v221; // rdi
  __int64 v222; // rcx
  __int64 v223; // r9
  __int64 *v224; // rsi
  __int64 v225; // r8
  __int64 v226; // rcx
  __int64 v227; // rdx
  _QWORD *v228; // rbx
  __int64 v229; // r15
  __int64 v230; // r14
  __int64 v231; // r15
  __int64 v232; // rax
  __int64 v233; // rdx
  __int64 v234; // rcx
  char v235; // dl
  __int64 v236; // rax
  const char **v237; // rbx
  __int64 v238; // r13
  __int64 v239; // rax
  __int64 v240; // rax
  __int64 v241; // rax
  __int64 *v242; // r13
  __int64 *v243; // rbx
  _BYTE *v244; // r14
  __int64 *v245; // r9
  __int64 j; // rax
  char v247; // dl
  const char *v248; // rax
  __int64 *v249; // r12
  __int64 v250; // rax
  __int64 v251; // rcx
  const char *v252; // rsi
  const char *v253; // rsi
  _BYTE *v254; // rax
  __int64 *v255; // rdi
  __int64 v256; // rdx
  __int64 v257; // rcx
  __int64 v258; // r8
  __int64 v259; // r9
  __int64 *v260; // r13
  __int64 v261; // rax
  __int64 v262; // rax
  const char *v263; // rax
  __int64 *v264; // r14
  __int64 v265; // rax
  __int64 v266; // rcx
  const char *v267; // rsi
  const char *v268; // rsi
  int v269; // eax
  __int64 *v270; // r9
  __int64 *v271; // rsi
  __int64 v272; // rax
  __int64 v273; // rax
  __int64 *v274; // rax
  __int64 *v275; // rdi
  const char *v276; // rax
  __int64 *v277; // r14
  __int64 v278; // rax
  __int64 v279; // rcx
  const char *v280; // rsi
  const char *v281; // rsi
  const char *v282; // rax
  __int64 *v283; // rdx
  __int64 v284; // rsi
  __int64 v285; // rax
  const char *v286; // rsi
  __int64 v287; // rdx
  const char *v288; // rsi
  const char *v289; // rax
  __int64 *v290; // rdx
  __int64 v291; // rsi
  __int64 v292; // rax
  const char *v293; // rsi
  __int64 v294; // rdx
  const char *v295; // rsi
  char **v296; // rsi
  __int64 v297; // rcx
  __int64 *v298; // rdi
  __int64 v299; // rax
  __int64 v300; // r14
  __int64 v301; // rax
  _QWORD *v302; // rbx
  __int64 v303; // rax
  __int64 v304; // rax
  __int64 v305; // rdi
  __int64 v306; // rax
  __int64 v307; // rax
  unsigned __int64 *v308; // r12
  __int64 v309; // rax
  unsigned __int64 v310; // rcx
  _BYTE *v311; // rsi
  _BYTE *v312; // rsi
  __int64 v313; // rax
  __int64 *v314; // r12
  __int64 v315; // rax
  __int64 v316; // rcx
  _BYTE *v317; // rsi
  _BYTE *v318; // rsi
  __int64 v319; // [rsp-38h] [rbp-168h] BYREF
  _BYTE *v320; // [rsp-30h] [rbp-160h]
  _BYTE *v321; // [rsp-28h] [rbp-158h]
  __int64 v322; // [rsp-20h] [rbp-150h]
  __int64 v323; // [rsp-18h] [rbp-148h]
  __int64 *v324; // [rsp-10h] [rbp-140h]
  __int64 *v325; // [rsp-8h] [rbp-138h]
  __int64 *v326; // [rsp+0h] [rbp-130h]
  __int64 *v327; // [rsp+8h] [rbp-128h]
  __int64 *v328; // [rsp+10h] [rbp-120h]
  __int64 v329; // [rsp+18h] [rbp-118h]
  _BYTE *v330; // [rsp+28h] [rbp-108h] BYREF
  _BYTE *v331; // [rsp+30h] [rbp-100h] BYREF
  int v332; // [rsp+38h] [rbp-F8h]
  char v333; // [rsp+3Ch] [rbp-F4h]
  __int64 v334; // [rsp+40h] [rbp-F0h]
  _BYTE *v335; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v336; // [rsp+58h] [rbp-D8h]
  __int64 v337; // [rsp+60h] [rbp-D0h]
  const char *v338; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE *v339; // [rsp+78h] [rbp-B8h]
  __int64 v340; // [rsp+80h] [rbp-B0h]
  __int64 v341; // [rsp+88h] [rbp-A8h]
  __int64 *v342; // [rsp+90h] [rbp-A0h]
  __int64 *v343; // [rsp+98h] [rbp-98h]
  char *v344; // [rsp+A0h] [rbp-90h] BYREF
  _BYTE *v345; // [rsp+A8h] [rbp-88h]
  __int64 v346; // [rsp+B0h] [rbp-80h]
  __int64 v347; // [rsp+B8h] [rbp-78h]
  __int64 *v348; // [rsp+C0h] [rbp-70h]
  __int64 v349; // [rsp+C8h] [rbp-68h]
  const char *v350; // [rsp+D0h] [rbp-60h] BYREF
  _BYTE *v351; // [rsp+D8h] [rbp-58h]
  __int64 v352; // [rsp+E0h] [rbp-50h]
  __int64 v353; // [rsp+E8h] [rbp-48h]
  __int64 v354; // [rsp+F0h] [rbp-40h]
  __int64 v355; // [rsp+F8h] [rbp-38h]

  v8 = *a1;
  while ( 2 )
  {
    v9 = &dword_4D04720;
    if ( !(dword_4D04720 | dword_4D04658) && (*(_WORD *)(a2 + 24) & 0x10FF) != 0x1002 )
    {
      v350 = *(const char **)(a2 + 36);
      if ( (_DWORD)v350 )
      {
        sub_1290930(v8, &v350);
        sub_127C770(&v350);
      }
    }
    switch ( *(_BYTE *)(a2 + 24) )
    {
      case 1:
        switch ( *(_BYTE *)(a2 + 56) )
        {
          case 0:
            v21 = *(_QWORD *)(a2 + 72);
            v22 = sub_72B0F0(v21, 0);
            sub_127C5E0(v22, (_DWORD *)(a2 + 36));
            sub_1286D80((__int64)&v350, *a1, v21, v23, v24);
            return v351;
          case 3:
          case 6:
          case 8:
          case 0x5C:
          case 0x5E:
          case 0x5F:
            return (char *)sub_1287ED0(a1, a2, a3, (__int64)v9, a5);
          case 5:
            v16 = *(__int64 **)(a2 + 72);
            if ( !sub_127B420(*v16) )
            {
              v17 = sub_128D0F0(a1, v16);
              v18 = *(_QWORD *)a2;
              v19 = (_QWORD *)v17;
              v20 = sub_127B3A0(*v16);
              return (char *)sub_128B370((__int64 *)a1, v19, v20, v18, (_DWORD *)(a2 + 36));
            }
            v12 = 0;
            sub_12A6C40(*a1, v16, 0, 0, 0);
            return (char *)v12;
          case 0x15:
            sub_1286D80((__int64)&v350, *a1, *(_QWORD *)(a2 + 72), (__int64)v9, a5);
            v12 = (__int64)v351;
            if ( sub_8D23B0(**(_QWORD **)(a2 + 72)) )
            {
              v27 = sub_127A030((*a1)[4] + 8, *(_QWORD *)a2, 0);
              LOWORD(v340) = 257;
              v28 = a1[1];
              if ( v27 != *(_QWORD *)v12 )
              {
                if ( *(_BYTE *)(v12 + 16) > 0x10u )
                {
                  LOWORD(v346) = 257;
                  v12 = sub_15FDBD0(47, v12, v27, &v344, 0);
                  v313 = v28[1];
                  if ( v313 )
                  {
                    v314 = (__int64 *)v28[2];
                    sub_157E9D0(v313 + 40, v12);
                    v315 = *(_QWORD *)(v12 + 24);
                    v316 = *v314;
                    *(_QWORD *)(v12 + 32) = v314;
                    v316 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v12 + 24) = v316 | v315 & 7;
                    *(_QWORD *)(v316 + 8) = v12 + 24;
                    *v314 = *v314 & 7 | (v12 + 24);
                  }
                  sub_164B780(v12, &v338);
                  v317 = (_BYTE *)*v28;
                  if ( *v28 )
                  {
                    v335 = (_BYTE *)*v28;
                    sub_1623A60(&v335, v317, 2);
                    if ( *(_QWORD *)(v12 + 48) )
                      sub_161E7C0(v12 + 48);
                    v318 = v335;
                    *(_QWORD *)(v12 + 48) = v335;
                    if ( v318 )
                      sub_1623210(&v335, v318, v12 + 48);
                  }
                }
                else
                {
                  return (char *)sub_15A46C0(47, v12, v27, 0);
                }
              }
            }
            else
            {
              v260 = a1[1];
              v338 = "arraydecay";
              LOWORD(v340) = 259;
              v261 = sub_1643350(v260[3]);
              v335 = (_BYTE *)sub_159C470(v261, 0, 0);
              v262 = sub_1643350(v260[3]);
              v336 = sub_159C470(v262, 0, 0);
              if ( *(_BYTE *)(v12 + 16) > 0x10u )
              {
                LOWORD(v346) = 257;
                v299 = *(_QWORD *)v12;
                if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
                  v299 = **(_QWORD **)(v299 + 16);
                v300 = *(_QWORD *)(v299 + 24);
                v301 = sub_1648A60(72, 3);
                v302 = (_QWORD *)v301;
                if ( v301 )
                {
                  v329 = v301 - 72;
                  v303 = *(_QWORD *)v12;
                  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
                    v303 = **(_QWORD **)(v303 + 16);
                  LODWORD(v328) = *(_DWORD *)(v303 + 8) >> 8;
                  v304 = sub_15F9F50(v300, &v335, 2);
                  v305 = sub_1646BA0(v304, (unsigned int)v328);
                  v306 = *(_QWORD *)v12;
                  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16
                    || (v306 = *(_QWORD *)v335, *(_BYTE *)(*(_QWORD *)v335 + 8LL) == 16)
                    || (v306 = *(_QWORD *)v336, *(_BYTE *)(*(_QWORD *)v336 + 8LL) == 16) )
                  {
                    v305 = sub_16463B0(v305, *(_QWORD *)(v306 + 32));
                  }
                  sub_15F1EA0(v302, v305, 32, v329, 3, 0);
                  v302[7] = v300;
                  v302[8] = sub_15F9F50(v300, &v335, 2);
                  sub_15F9CE0(v302, v12, &v335, 2, &v344);
                }
                sub_15FA2E0(v302, 1);
                v307 = v260[1];
                if ( v307 )
                {
                  v308 = (unsigned __int64 *)v260[2];
                  sub_157E9D0(v307 + 40, v302);
                  v309 = v302[3];
                  v310 = *v308;
                  v302[4] = v308;
                  v310 &= 0xFFFFFFFFFFFFFFF8LL;
                  v302[3] = v310 | v309 & 7;
                  *(_QWORD *)(v310 + 8) = v302 + 3;
                  *v308 = *v308 & 7 | (unsigned __int64)(v302 + 3);
                }
                sub_164B780(v302, &v338);
                v311 = (_BYTE *)*v260;
                if ( *v260 )
                {
                  v331 = (_BYTE *)*v260;
                  sub_1623A60(&v331, v311, 2);
                  if ( v302[6] )
                    sub_161E7C0(v302 + 6);
                  v312 = v331;
                  v302[6] = v331;
                  if ( v312 )
                    sub_1623210(&v331, v312, v302 + 6);
                }
                return (char *)v302;
              }
              else
              {
                v325 = v5;
                BYTE4(v344) = 0;
                return (char *)sub_15A2E80(0, v12, (unsigned int)&v335, 2, 1, (unsigned int)&v344, 0);
              }
            }
            return (char *)v12;
          case 0x19:
            a2 = *(_QWORD *)(a2 + 72);
            v8 = *a1;
            continue;
          case 0x1A:
            return (char *)sub_128FDE0(a1, a2);
          case 0x1C:
            v25 = sub_128D0F0(a1, *(_QWORD *)(a2 + 72));
            v26 = (const char **)a1[1];
            LOWORD(v346) = 259;
            v344 = "not";
            if ( *(_BYTE *)(v25 + 16) <= 0x10u )
              return (char *)sub_15A2B00();
            LOWORD(v352) = 257;
            v12 = sub_15FB630(v25, &v350, 0);
            v248 = v26[1];
            if ( !v248 )
              goto LABEL_250;
            goto LABEL_249;
          case 0x1D:
            v29 = sub_127FEC0((__int64)*a1, *(_QWORD *)(a2 + 72));
            v30 = (const char **)a1[1];
            LOWORD(v346) = 259;
            v344 = "lnot";
            if ( *(_BYTE *)(v29 + 16) > 0x10u )
            {
              LOWORD(v352) = 257;
              v12 = sub_15FB630(v29, &v350, 0);
              v276 = v30[1];
              if ( v276 )
              {
                v277 = (__int64 *)v30[2];
                sub_157E9D0(v276 + 40, v12);
                v278 = *(_QWORD *)(v12 + 24);
                v279 = *v277;
                *(_QWORD *)(v12 + 32) = v277;
                v279 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v12 + 24) = v279 | v278 & 7;
                *(_QWORD *)(v279 + 8) = v12 + 24;
                *v277 = *v277 & 7 | (v12 + 24);
              }
              sub_164B780(v12, &v344);
              v280 = *v30;
              if ( *v30 )
              {
                v338 = *v30;
                sub_1623A60(&v338, v280, 2);
                if ( *(_QWORD *)(v12 + 48) )
                  sub_161E7C0(v12 + 48);
                v281 = v338;
                *(_QWORD *)(v12 + 48) = v338;
                if ( v281 )
                  sub_1623210(&v338, v281, v12 + 48);
              }
            }
            else
            {
              v12 = sub_15A2B00();
            }
            v26 = (const char **)a1[1];
            v344 = "lnot.ext";
            v31 = *a1;
            LOWORD(v346) = 259;
            v32 = sub_127A030(v31[4] + 8, *(_QWORD *)a2, 0);
            if ( v32 != *(_QWORD *)v12 )
              goto LABEL_34;
            return (char *)v12;
          case 0x1E:
          case 0x1F:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x59:
          case 0x5A:
          case 0x5D:
          case 0x68:
            return (char *)sub_127D2C0((*a1)[4], *(_QWORD *)a2);
          case 0x23:
            v49 = 0;
            v50 = 1;
            return (char *)sub_128C390((__int64)a1, a2, v50, v49, a5);
          case 0x24:
            v49 = 0;
            goto LABEL_46;
          case 0x25:
            v49 = 1;
            v50 = 1;
            return (char *)sub_128C390((__int64)a1, a2, v50, v49, a5);
          case 0x26:
            v49 = 1;
LABEL_46:
            v50 = 0;
            return (char *)sub_128C390((__int64)a1, a2, v50, v49, a5);
          case 0x27:
          case 0x28:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x35:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
            return (char *)sub_128F9F0(a1, a2);
          case 0x32:
            v33 = *(__int64 **)(a2 + 72);
            v34 = (__int64 *)v33[2];
            v35 = (__int64 *)sub_128D0F0(a1, v33);
            v36 = (__int64 *)sub_128D0F0(a1, v34);
            if ( *(_BYTE *)(*v35 + 8) == 15 )
            {
              v37 = v35;
              v35 = v36;
              v36 = v37;
              v38 = v34;
              v34 = v33;
              v33 = v38;
            }
            for ( i = *v34; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            do
            {
              i = *(_QWORD *)(i + 160);
              v40 = *(_BYTE *)(i + 140);
            }
            while ( v40 == 12 );
            return (char *)sub_128BE50((__int64)a1, v36, v35, *v33, v40 == 1);
          case 0x33:
            v242 = *(__int64 **)(a2 + 72);
            v243 = (__int64 *)v242[2];
            v244 = (_BYTE *)sub_128D0F0(a1, v242);
            v245 = (__int64 *)sub_128D0F0(a1, v243);
            for ( j = *v242; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            do
            {
              j = *(_QWORD *)(j + 160);
              v247 = *(_BYTE *)(j + 140);
            }
            while ( v247 == 12 );
            return (char *)sub_128B750((__int64)a1, v244, v245, *v243, v247 == 1);
          case 0x34:
            v228 = *(_QWORD **)(a2 + 72);
            v229 = v228[2];
            v230 = sub_128D0F0(a1, v228);
            v231 = sub_128D0F0(a1, v229);
            v232 = *v228;
            if ( *(_BYTE *)(*v228 + 140LL) == 12 )
            {
              v233 = *v228;
              do
                v233 = *(_QWORD *)(v233 + 160);
              while ( *(_BYTE *)(v233 + 140) == 12 );
              v234 = *(_QWORD *)(v233 + 160);
              do
                v232 = *(_QWORD *)(v232 + 160);
              while ( *(_BYTE *)(v232 + 140) == 12 );
            }
            else
            {
              v234 = *(_QWORD *)(v232 + 160);
            }
            do
            {
              v232 = *(_QWORD *)(v232 + 160);
              v235 = *(_BYTE *)(v232 + 140);
            }
            while ( v235 == 12 );
            v329 = 1;
            if ( v235 != 1 && *(_BYTE *)(*(_QWORD *)v230 + 8LL) != 12 )
            {
              while ( *(_BYTE *)(v234 + 140) == 12 )
                v234 = *(_QWORD *)(v234 + 160);
              v329 = *(_QWORD *)(v234 + 128);
            }
            v236 = sub_127A030((*a1)[4] + 8, *(_QWORD *)a2, 0);
            v237 = (const char **)a1[1];
            v238 = v236;
            LOWORD(v346) = 259;
            v344 = "sub.ptr.lhs.cast";
            if ( v236 != *(_QWORD *)v230 )
            {
              if ( *(_BYTE *)(v230 + 16) > 0x10u )
              {
                LOWORD(v352) = 257;
                v230 = sub_15FDBD0(45, v230, v236, &v350, 0);
                v282 = v237[1];
                if ( v282 )
                {
                  v328 = (__int64 *)v237[2];
                  sub_157E9D0(v282 + 40, v230);
                  v283 = v328;
                  v284 = *v328;
                  v285 = *(_QWORD *)(v230 + 24) & 7LL;
                  *(_QWORD *)(v230 + 32) = v328;
                  v284 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v230 + 24) = v284 | v285;
                  *(_QWORD *)(v284 + 8) = v230 + 24;
                  *v283 = *v283 & 7 | (v230 + 24);
                }
                sub_164B780(v230, &v344);
                v286 = *v237;
                if ( *v237 )
                {
                  v338 = *v237;
                  sub_1623A60(&v338, v286, 2);
                  v287 = v230 + 48;
                  if ( *(_QWORD *)(v230 + 48) )
                  {
                    v328 = (__int64 *)(v230 + 48);
                    sub_161E7C0(v230 + 48);
                    v287 = v230 + 48;
                  }
                  v288 = v338;
                  *(_QWORD *)(v230 + 48) = v338;
                  if ( v288 )
                    sub_1623210(&v338, v288, v287);
                }
                v237 = (const char **)a1[1];
              }
              else
              {
                v239 = sub_15A46C0(45, v230, v236, 0);
                v237 = (const char **)a1[1];
                v230 = v239;
              }
            }
            v344 = "sub.ptr.rhs.cast";
            LOWORD(v346) = 259;
            if ( v238 != *(_QWORD *)v231 )
            {
              if ( *(_BYTE *)(v231 + 16) > 0x10u )
              {
                LOWORD(v352) = 257;
                v231 = sub_15FDBD0(45, v231, v238, &v350, 0);
                v289 = v237[1];
                if ( v289 )
                {
                  v328 = (__int64 *)v237[2];
                  sub_157E9D0(v289 + 40, v231);
                  v290 = v328;
                  v291 = *v328;
                  v292 = *(_QWORD *)(v231 + 24) & 7LL;
                  *(_QWORD *)(v231 + 32) = v328;
                  v291 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v231 + 24) = v291 | v292;
                  *(_QWORD *)(v291 + 8) = v231 + 24;
                  *v290 = *v290 & 7 | (v231 + 24);
                }
                sub_164B780(v231, &v344);
                v293 = *v237;
                if ( *v237 )
                {
                  v338 = *v237;
                  sub_1623A60(&v338, v293, 2);
                  v294 = v231 + 48;
                  if ( *(_QWORD *)(v231 + 48) )
                  {
                    v328 = (__int64 *)(v231 + 48);
                    sub_161E7C0(v231 + 48);
                    v294 = v231 + 48;
                  }
                  v295 = v338;
                  *(_QWORD *)(v231 + 48) = v338;
                  if ( v295 )
                    sub_1623210(&v338, v295, v294);
                }
                v237 = (const char **)a1[1];
              }
              else
              {
                v240 = sub_15A46C0(45, v231, v238, 0);
                v237 = (const char **)a1[1];
                v231 = v240;
              }
            }
            v344 = "sub.ptr.sub";
            LOWORD(v346) = 259;
            if ( *(_BYTE *)(v230 + 16) > 0x10u || *(_BYTE *)(v231 + 16) > 0x10u )
            {
              LOWORD(v352) = 257;
              v12 = sub_15FB440(13, v230, v231, &v350, 0);
              v263 = v237[1];
              if ( v263 )
              {
                v264 = (__int64 *)v237[2];
                sub_157E9D0(v263 + 40, v12);
                v265 = *(_QWORD *)(v12 + 24);
                v266 = *v264;
                *(_QWORD *)(v12 + 32) = v264;
                v266 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v12 + 24) = v266 | v265 & 7;
                *(_QWORD *)(v266 + 8) = v12 + 24;
                *v264 = *v264 & 7 | (v12 + 24);
              }
              sub_164B780(v12, &v344);
              v267 = *v237;
              if ( *v237 )
              {
                v338 = *v237;
                sub_1623A60(&v338, v267, 2);
                if ( *(_QWORD *)(v12 + 48) )
                  sub_161E7C0(v12 + 48);
                v268 = v338;
                *(_QWORD *)(v12 + 48) = v338;
                if ( v268 )
                  sub_1623210(&v338, v268, v12 + 48);
              }
            }
            else
            {
              v12 = sub_15A2B60(v230, v231, 0, 0);
            }
            if ( v329 == 1 )
              return (char *)v12;
            v241 = sub_15A0680(v238, v329, 0);
            v26 = (const char **)a1[1];
            LOWORD(v346) = 259;
            v344 = "sub.ptr.div";
            if ( *(_BYTE *)(v12 + 16) <= 0x10u && *(_BYTE *)(v241 + 16) <= 0x10u )
              return (char *)sub_15A2C90(v12, v241, 1);
            LOWORD(v352) = 257;
            v12 = sub_15FB440(18, v12, v241, &v350, 0);
            sub_15F2350(v12, 1);
            v248 = v26[1];
            if ( !v248 )
              goto LABEL_250;
            goto LABEL_249;
          case 0x3A:
            v225 = 1;
            v226 = 32;
            v227 = 32;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x3B:
            v225 = 14;
            v226 = 33;
            v227 = 33;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x3C:
            v225 = 2;
            v226 = 38;
            v227 = 34;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x3D:
            v225 = 4;
            v226 = 40;
            v227 = 36;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x3E:
            v225 = 3;
            v226 = 39;
            v227 = 35;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x3F:
            v225 = 5;
            v226 = 41;
            v227 = 37;
            return (char *)sub_128F580(a1, a2, v227, v226, v225);
          case 0x49:
            v209 = *(_QWORD *)(a2 + 72);
            v210 = (_BYTE *)sub_128D0F0(a1, *(_QWORD *)(v209 + 16));
            v211 = *a1;
            v212 = v209;
            v330 = v210;
            v213 = (_DWORD *)(v209 + 36);
            sub_1286D80((__int64)&v344, v211, v212, v214, v215);
            v217 = (int)v344;
            if ( (_DWORD)v344 == 1 )
            {
              v218 = *a1;
              v219 = v330;
              if ( (v349 & 1) == 0 )
              {
                BYTE4(v351) &= ~1u;
                v296 = &v344;
                v297 = 12;
                v298 = &v319;
                v350 = v330;
                LODWORD(v351) = 0;
                LODWORD(v352) = 0;
                while ( v297 )
                {
                  *(_DWORD *)v298 = *(_DWORD *)v296;
                  v296 = (char **)((char *)v296 + 4);
                  v298 = (__int64 *)((char *)v298 + 4);
                  --v297;
                }
                sub_1282050(
                  v218,
                  v213,
                  (__int64 *)&v330,
                  0,
                  (__int64)v218,
                  v216,
                  (__int64)v219,
                  (int)v351,
                  v352,
                  v319,
                  v320,
                  (__int64)v321,
                  v322,
                  v323,
                  (__int64)v324);
                return v330;
              }
              BYTE4(v336) &= ~1u;
              v220 = &v344;
              v221 = &v319;
              v335 = v330;
              v222 = 12;
              LODWORD(v336) = 0;
              LODWORD(v337) = 0;
              while ( v222 )
              {
                *(_DWORD *)v221 = *(_DWORD *)v220;
                v220 = (char **)((char *)v220 + 4);
                v221 = (__int64 *)((char *)v221 + 4);
                --v222;
              }
              sub_1282050(
                v218,
                v213,
                0,
                0,
                (__int64)v218,
                v216,
                (__int64)v219,
                v336,
                v337,
                v319,
                v320,
                (__int64)v321,
                v322,
                v323,
                (__int64)v324);
            }
            else
            {
              v254 = (_BYTE *)sub_1289860(*a1, *(_QWORD *)(*(_QWORD *)v345 + 24LL), v330);
              v255 = *a1;
              v333 &= ~1u;
              v332 = 0;
              LODWORD(v334) = 0;
              v330 = v254;
              v331 = v254;
              sub_12843D0(
                v255,
                v213,
                v256,
                v257,
                v258,
                v259,
                (__int64)v254,
                0,
                0,
                (__int64)v344,
                v345,
                v346,
                v347,
                (__int64)v348,
                v349);
            }
            v12 = 0;
            if ( (*(_BYTE *)(a2 + 25) & 4) == 0 )
            {
              LODWORD(v344) = v217;
              v325 = (__int64 *)v349;
              v324 = v348;
              v323 = v347;
              v322 = v346;
              v321 = v345;
              v320 = v344;
              v354 = (__int64)v348;
              v224 = *a1;
              v351 = v345;
              v355 = v349;
              v350 = v344;
              v352 = v346;
              v353 = v347;
              sub_1287CD0(
                (__int64)&v338,
                v224,
                v213,
                v346,
                (__int64)&v338,
                v223,
                (__int64)v344,
                v345,
                v346,
                v347,
                (__int64)v348,
                v349);
              return (char *)v338;
            }
            return (char *)v12;
          case 0x4A:
            v208 = sub_1288F60;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x4B:
            v208 = sub_1288370;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x4C:
            v208 = sub_1288770;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x4D:
            v208 = sub_1289D20;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x4E:
            v208 = sub_1288DC0;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x4F:
            v208 = sub_1288B70;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x50:
            v208 = sub_1289360;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x51:
            v208 = sub_1288090;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x52:
            v208 = sub_1287F30;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x53:
            v208 = sub_1288230;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x54:
            v208 = sub_128BE50;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x55:
            v208 = sub_128B750;
            return (char *)sub_12901D0(a1, a2, v208, 0);
          case 0x56:
            v193 = *(_QWORD *)(a2 + 72);
            v194 = *(__int64 ***)(v193 + 16);
            v195 = *v194;
            for ( k = *v194; *((_BYTE *)k + 140) == 12; k = (__int64 *)k[20] )
              ;
            v197 = (__int64 *)k[16];
            v198 = *a1;
            v328 = *v194;
            v329 = (__int64)v197;
            sub_1286D80((__int64)&v344, v198, (__int64)v194, (__int64)v9, (__int64)v195);
            v203 = v346;
            v204 = (__int64 *)v345;
            if ( dword_4D04810 )
            {
              v327 = (__int64 *)v345;
              v269 = sub_731770(v193, 0, v199, v200, (__int64)v201, v202);
              v201 = v328;
              v204 = v327;
              if ( v269 )
              {
                v270 = *a1;
                v271 = v328;
                v350 = "bassign.tmp";
                LOWORD(v352) = 259;
                v272 = v270[4];
                v328 = v270;
                v273 = sub_127A030(v272 + 8, (unsigned __int64)v271, 0);
                v274 = sub_127FC40(v328, v273, (__int64)&v350, v203, 0);
                v325 = v328;
                v324 = 0;
                v275 = *a1;
                v328 = v274;
                sub_12897E0((__int64)v275, (int)v274, (int)v327, v329, v203, v203, 0);
                v201 = v324;
                v204 = v328;
              }
            }
            v205 = *a1;
            v328 = v204;
            sub_1286D80((__int64)&v350, v205, v193, v200, (__int64)v201);
            sub_12897E0((__int64)*a1, (int)v351, (int)v328, v329, v352, v203, 0);
            v206 = *(_QWORD *)a2;
            for ( m = *(_BYTE *)(*(_QWORD *)a2 + 140LL); m == 12; m = *(_BYTE *)(v206 + 140) )
              v206 = *(_QWORD *)(v206 + 160);
            if ( m != 1 )
              sub_127B550("expected result type of bassign to be void!", (_DWORD *)(a2 + 36), 1);
            return 0;
          case 0x57:
            v151 = *(_QWORD *)(a2 + 72);
            v152 = *a1;
            v327 = *(__int64 **)(v151 + 16);
            v153 = (__int64 *)sub_12A4D50(v152, "land.end", 0, 0);
            v154 = *a1;
            v155 = v153;
            v329 = (__int64)v153;
            v156 = sub_12A4D50(v154, "land.rhs", 0, 0);
            v157 = *a1;
            v158 = v156;
            v326 = (__int64 *)v156;
            v159 = sub_127FEC0((__int64)v157, v151);
            sub_12A4DB0(v157, v159, v158, v155, 0);
            v160 = a1[2];
            LOWORD(v352) = 257;
            v161 = sub_1643320(v160);
            v162 = sub_1648B60(64);
            v12 = v162;
            if ( v162 )
            {
              v163 = v162;
              sub_15F1F50(v162, v161, 53, 0, 0, v329, v326, v327);
              *(_DWORD *)(v12 + 56) = 2;
              sub_164B780(v12, &v350);
              sub_1648880(v12, *(unsigned int *)(v12 + 56), 1);
            }
            else
            {
              v163 = 0;
            }
            v164 = *(_QWORD *)(v329 + 8);
            if ( !v164 )
              goto LABEL_142;
            while ( 1 )
            {
              v165 = sub_1648700(v164);
              if ( (unsigned __int8)(*(_BYTE *)(v165 + 16) - 25) <= 9u )
                break;
              v164 = *(_QWORD *)(v164 + 8);
              if ( !v164 )
                goto LABEL_142;
            }
LABEL_177:
            v188 = *(_QWORD *)(v165 + 40);
            v189 = (__int64 *)sub_159C540(a1[2]);
            v190 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            if ( v190 == *(_DWORD *)(v12 + 56) )
            {
              v328 = v189;
              sub_15F55D0(v12);
              v189 = v328;
              v190 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            }
            v191 = (v190 + 1) & 0xFFFFFFF;
            v192 = v191 | *(_DWORD *)(v12 + 20) & 0xF0000000;
            *(_DWORD *)(v12 + 20) = v192;
            if ( (v192 & 0x40000000) != 0 )
              v181 = *(_QWORD *)(v12 - 8);
            else
              v181 = v163 - 24 * v191;
            v182 = (__int64 **)(v181 + 24LL * (unsigned int)(v191 - 1));
            if ( *v182 )
            {
              v183 = v182[1];
              v184 = (unsigned __int64)v182[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v184 = v183;
              if ( v183 )
                v183[2] = v183[2] & 3 | v184;
            }
            *v182 = v189;
            if ( v189 )
            {
              v185 = v189[1];
              v182[1] = (__int64 *)v185;
              if ( v185 )
                *(_QWORD *)(v185 + 16) = (unsigned __int64)(v182 + 1) | *(_QWORD *)(v185 + 16) & 3LL;
              v182[2] = (__int64 *)((unsigned __int64)(v189 + 1) | (unsigned __int64)v182[2] & 3);
              v189[1] = (__int64)v182;
            }
            v186 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
              v187 = *(_QWORD *)(v12 - 8);
            else
              v187 = v163 - 24 * v186;
            *(_QWORD *)(v187 + 8LL * (unsigned int)(v186 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v188;
            while ( 1 )
            {
              v164 = *(_QWORD *)(v164 + 8);
              if ( !v164 )
                break;
              v165 = sub_1648700(v164);
              if ( (unsigned __int8)(*(_BYTE *)(v165 + 16) - 25) <= 9u )
                goto LABEL_177;
            }
LABEL_142:
            sub_1290AF0(*a1, v326, 0);
            v166 = sub_127FEC0((__int64)*a1, (__int64)v327);
            v167 = a1[1][1];
            sub_1290AF0(*a1, v329, 0);
            v168 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            if ( v168 == *(_DWORD *)(v12 + 56) )
            {
              sub_15F55D0(v12);
              v168 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            }
            v169 = (v168 + 1) & 0xFFFFFFF;
            v170 = v169 | *(_DWORD *)(v12 + 20) & 0xF0000000;
            *(_DWORD *)(v12 + 20) = v170;
            if ( (v170 & 0x40000000) != 0 )
              v171 = *(_QWORD *)(v12 - 8);
            else
              v171 = v163 - 24 * v169;
            v172 = (__int64 *)(v171 + 24LL * (unsigned int)(v169 - 1));
            if ( *v172 )
            {
              v173 = v172[1];
              v174 = v172[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v174 = v173;
              if ( v173 )
                *(_QWORD *)(v173 + 16) = v174 | *(_QWORD *)(v173 + 16) & 3LL;
            }
            *v172 = v166;
            if ( v166 )
            {
              v175 = *(_QWORD *)(v166 + 8);
              v172[1] = v175;
              if ( v175 )
                *(_QWORD *)(v175 + 16) = (unsigned __int64)(v172 + 1) | *(_QWORD *)(v175 + 16) & 3LL;
              v172[2] = (v166 + 8) | v172[2] & 3;
              *(_QWORD *)(v166 + 8) = v172;
            }
            v176 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
              v177 = *(_QWORD *)(v12 - 8);
            else
              v177 = v163 - 24 * v176;
            *(_QWORD *)(v177 + 8LL * (unsigned int)(v176 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v167;
            v26 = (const char **)a1[1];
            v178 = *v26;
            if ( *v26 )
            {
              v350 = *v26;
              sub_1623A60(&v350, v178, 2);
              if ( *(_QWORD *)(v12 + 48) )
                sub_161E7C0(v12 + 48);
              v179 = v350;
              *(_QWORD *)(v12 + 48) = v350;
              if ( v179 )
                sub_1623210(&v350, v179, v12 + 48);
              v26 = (const char **)a1[1];
            }
            BYTE1(v346) = 1;
            v138 = "land.ext";
            goto LABEL_162;
          case 0x58:
            v109 = *(_QWORD *)(a2 + 72);
            v110 = *a1;
            v327 = *(__int64 **)(v109 + 16);
            v111 = (__int64 *)sub_12A4D50(v110, "lor.end", 0, 0);
            v112 = *a1;
            v113 = v111;
            v329 = (__int64)v111;
            v114 = sub_12A4D50(v112, "lor.rhs", 0, 0);
            v115 = *a1;
            v116 = v114;
            v326 = (__int64 *)v114;
            v117 = sub_127FEC0((__int64)v115, v109);
            sub_12A4DB0(v115, v117, v113, v116, 0);
            v118 = a1[2];
            LOWORD(v352) = 257;
            v119 = sub_1643320(v118);
            v120 = sub_1648B60(64);
            v12 = v120;
            if ( v120 )
            {
              v121 = v120;
              sub_15F1F50(v120, v119, 53, 0, 0, v329, v326, v327);
              *(_DWORD *)(v12 + 56) = 2;
              sub_164B780(v12, &v350);
              sub_1648880(v12, *(unsigned int *)(v12 + 56), 1);
            }
            else
            {
              v121 = 0;
            }
            v122 = *(_QWORD *)(v329 + 8);
            if ( !v122 )
              goto LABEL_100;
            break;
          case 0x5B:
            v107 = *(__int64 **)(a2 + 72);
            a2 = v107[2];
            sub_127FF60((__int64)&v350, (__int64)*a1, v107, 0, 0, 0);
            v8 = *a1;
            if ( !(*a1)[7] )
            {
              v329 = (__int64)*a1;
              v108 = sub_12A4D50(v8, byte_3F871B3, 0, 0);
              sub_1290AF0(v329, v108, 0);
              v8 = *a1;
            }
            continue;
          case 0x67:
            v56 = *a1;
            v57 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 16LL);
            v329 = *(_QWORD *)(a2 + 72);
            v328 = *(__int64 **)(v57 + 16);
            v58 = sub_12A4D50(v56, "cond.true", 0, 0);
            v59 = sub_12A4D50(*a1, "cond.false", 0, 0);
            v60 = sub_12A4D50(*a1, "cond.end", 0, 0);
            v61 = (__int64 *)v329;
            v329 = (__int64)*a1;
            v62 = sub_127FEC0(v329, (__int64)v61);
            sub_12A4DB0(v329, v62, v58, v59, 0);
            sub_1290AF0(*a1, v58, 0);
            v63 = (__int64 *)sub_128D0F0(a1, v57);
            v64 = *a1;
            v329 = (__int64)v63;
            v65 = v63;
            v327 = (__int64 *)a1[1][1];
            sub_12909B0(v64, v60);
            sub_1290AF0(*a1, v59, 0);
            v66 = sub_128D0F0(a1, v328);
            v67 = *a1;
            v68 = v66;
            v328 = (__int64 *)a1[1][1];
            sub_12909B0(v67, v60);
            sub_1290AF0(*a1, v60, 0);
            v69 = v65;
            if ( !v65 )
              return (char *)v68;
            v12 = (__int64)v65;
            if ( v68 )
            {
              v70 = (const char **)a1[1];
              v344 = "cond";
              LOWORD(v346) = 259;
              v71 = (__int64 *)*v69;
              LOWORD(v352) = 257;
              v326 = v71;
              v72 = sub_1648B60(64);
              v12 = v72;
              if ( v72 )
              {
                v73 = v72;
                sub_15F1EA0(v72, v71, 53, 0, 0, 0);
                *(_DWORD *)(v12 + 56) = 2;
                sub_164B780(v12, &v350);
                sub_1648880(v12, *(unsigned int *)(v12 + 56), 1);
              }
              else
              {
                v73 = 0;
              }
              v74 = v70[1];
              if ( v74 )
              {
                v326 = (__int64 *)v70[2];
                sub_157E9D0(v74 + 40, v12);
                v75 = v326;
                v76 = *v326;
                v77 = *(_QWORD *)(v12 + 24) & 7LL;
                *(_QWORD *)(v12 + 32) = v326;
                v76 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v12 + 24) = v76 | v77;
                *(_QWORD *)(v76 + 8) = v12 + 24;
                *v75 = *v75 & 7 | (v12 + 24);
              }
              sub_164B780(v73, &v344);
              v78 = *v70;
              if ( *v70 )
              {
                v338 = *v70;
                sub_1623A60(&v338, v78, 2);
                v79 = v12 + 48;
                if ( *(_QWORD *)(v12 + 48) )
                {
                  v326 = (__int64 *)(v12 + 48);
                  sub_161E7C0(v12 + 48);
                  v79 = v12 + 48;
                }
                v80 = v338;
                *(_QWORD *)(v12 + 48) = v338;
                if ( v80 )
                  sub_1623210(&v338, v80, v79);
              }
              v81 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              if ( v81 == *(_DWORD *)(v12 + 56) )
              {
                sub_15F55D0(v12);
                v81 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              }
              v82 = (v81 + 1) & 0xFFFFFFF;
              v83 = v82 | *(_DWORD *)(v12 + 20) & 0xF0000000;
              *(_DWORD *)(v12 + 20) = v83;
              if ( (v83 & 0x40000000) != 0 )
                v84 = *(_QWORD *)(v12 - 8);
              else
                v84 = v73 - 24 * v82;
              v85 = (__int64 *)(v84 + 24LL * (unsigned int)(v82 - 1));
              if ( *v85 )
              {
                v86 = v85[1];
                v87 = v85[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v87 = v86;
                if ( v86 )
                  *(_QWORD *)(v86 + 16) = v87 | *(_QWORD *)(v86 + 16) & 3LL;
              }
              v88 = (__int64 *)v329;
              *v85 = v329;
              v89 = v88[1];
              v90 = (unsigned __int64)(v88 + 1);
              v85[1] = v89;
              if ( v89 )
                *(_QWORD *)(v89 + 16) = (unsigned __int64)(v85 + 1) | *(_QWORD *)(v89 + 16) & 3LL;
              v91 = (__int64 *)v329;
              v85[2] = v90 | v85[2] & 3;
              v91[1] = (__int64)v85;
              v92 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
                v93 = *(_QWORD *)(v12 - 8);
              else
                v93 = v73 - 24 * v92;
              *(_QWORD *)(v93 + 8LL * (unsigned int)(v92 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v327;
              v94 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              if ( v94 == *(_DWORD *)(v12 + 56) )
              {
                sub_15F55D0(v12);
                v94 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              }
              v95 = (v94 + 1) & 0xFFFFFFF;
              v96 = v95 | *(_DWORD *)(v12 + 20) & 0xF0000000;
              *(_DWORD *)(v12 + 20) = v96;
              if ( (v96 & 0x40000000) != 0 )
                v97 = *(_QWORD *)(v12 - 8);
              else
                v97 = v73 - 24 * v95;
              v98 = (__int64 *)(v97 + 24LL * (unsigned int)(v95 - 1));
              if ( *v98 )
              {
                v99 = v98[1];
                v100 = v98[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v100 = v99;
                if ( v99 )
                  *(_QWORD *)(v99 + 16) = v100 | *(_QWORD *)(v99 + 16) & 3LL;
              }
              *v98 = v68;
              v101 = *(_QWORD *)(v68 + 8);
              v98[1] = v101;
              if ( v101 )
                *(_QWORD *)(v101 + 16) = (unsigned __int64)(v98 + 1) | *(_QWORD *)(v101 + 16) & 3LL;
              v98[2] = (v68 + 8) | v98[2] & 3;
              *(_QWORD *)(v68 + 8) = v98;
              v102 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
                v103 = *(_QWORD *)(v12 - 8);
              else
                v103 = v73 - 24 * v102;
              *(_QWORD *)(v103 + 8LL * (unsigned int)(v102 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v328;
              v104 = (const char **)a1[1];
              v105 = *v104;
              if ( *v104 )
              {
                v350 = *v104;
                sub_1623A60(&v350, v105, 2);
                if ( *(_QWORD *)(v12 + 48) )
                  sub_161E7C0(v12 + 48);
                v106 = v350;
                *(_QWORD *)(v12 + 48) = v350;
                if ( v106 )
                  sub_1623210(&v350, v106, v12 + 48);
              }
            }
            return (char *)v12;
          case 0x69:
            sub_1281200((__int64)&v350);
            return (char *)v350;
          case 0x6F:
            v45 = *a1;
            v55 = sub_12A4D00(v45, *(_QWORD *)(a2 + 72));
            v47 = 1;
            v48 = v55;
            return (char *)sub_1285E30(v45, v48, v47);
          case 0x70:
            v51 = *a1;
            v52 = *(_QWORD *)(a2 + 72);
            v53 = sub_127A030((*a1)[4] + 8, *(_QWORD *)a2, 0);
            v54 = sub_12A4D00(*a1, v52);
            return (char *)sub_12812E0(v51, v54, v53);
          case 0x71:
            v45 = *a1;
            v46 = sub_12A4D00(v45, *(_QWORD *)(a2 + 72));
            v47 = 0;
            v48 = v46;
            return (char *)sub_1285E30(v45, v48, v47);
          case 0x72:
            v41 = *(_QWORD *)(a2 + 72);
            v42 = *a1;
            v43 = sub_12A4D00(*a1, *(_QWORD *)(v41 + 16));
            v44 = (__int64 *)sub_12A4D00(*a1, v41);
            return (char *)sub_1286000(v42, v44, v43);
          default:
            sub_127B550("unsupported operation expression!", (_DWORD *)(a2 + 36), 1);
        }
        return result;
      case 2:
        return (char *)sub_127F650((__int64)*a1, *(const __m128i **)(a2 + 56), 0);
      case 3:
        sub_1286D80((__int64)&v338, *a1, a2, (__int64)v9, a5);
        v351 = v339;
        v350 = v338;
        v325 = v343;
        v324 = v342;
        v323 = v341;
        v322 = v340;
        v321 = v339;
        v320 = v338;
        v11 = *a1;
        v352 = v340;
        v353 = v341;
        v354 = (__int64)v342;
        v355 = (__int64)v343;
        sub_1287CD0(
          (__int64)&v344,
          v11,
          (_DWORD *)(a2 + 36),
          v340,
          (__int64)v342,
          (__int64)v343,
          (__int64)v338,
          v339,
          v340,
          v341,
          (__int64)v342,
          (__int64)v343);
        return v344;
      case 0x11:
        v14 = *(_QWORD *)a2;
        for ( n = *(_BYTE *)(*(_QWORD *)a2 + 140LL); n == 12; n = *(_BYTE *)(v14 + 140) )
          v14 = *(_QWORD *)(v14 + 160);
        sub_1296570((unsigned int)&v350, (unsigned int)*a1, *(_QWORD *)(a2 + 56), n != 1, 0, 0, 0);
        return (char *)v350;
      case 0x13:
        sub_1281220((__int64)&v350, (__int64)*a1, (__int64 *)a2);
        return (char *)v350;
      case 0x14:
        sub_1286D80((__int64)&v344, *a1, a2, (__int64)v9, a5);
        v351 = v345;
        v350 = v344;
        v325 = (__int64 *)v349;
        v324 = v348;
        v323 = v347;
        v322 = v346;
        v321 = v345;
        v320 = v344;
        v13 = *a1;
        v352 = v346;
        v353 = v347;
        v354 = (__int64)v348;
        v355 = v349;
        sub_1287CD0(
          (__int64)&v338,
          v13,
          (_DWORD *)(a2 + 36),
          v346,
          (__int64)v348,
          v349,
          (__int64)v344,
          v345,
          v346,
          v347,
          (__int64)v348,
          v349);
        return (char *)v338;
      default:
        sub_127B550("unsupported expression!", (_DWORD *)(a2 + 36), 1);
    }
  }
  while ( 1 )
  {
    v123 = sub_1648700(v122);
    if ( (unsigned __int8)(*(_BYTE *)(v123 + 16) - 25) <= 9u )
      break;
    v122 = *(_QWORD *)(v122 + 8);
    if ( !v122 )
      goto LABEL_100;
  }
LABEL_133:
  v146 = *(_QWORD *)(v123 + 40);
  v147 = (__int64 *)sub_159C4F0(a1[2]);
  v148 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( v148 == *(_DWORD *)(v12 + 56) )
  {
    v328 = v147;
    sub_15F55D0(v12);
    v147 = v328;
    v148 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  }
  v149 = (v148 + 1) & 0xFFFFFFF;
  v150 = v149 | *(_DWORD *)(v12 + 20) & 0xF0000000;
  *(_DWORD *)(v12 + 20) = v150;
  if ( (v150 & 0x40000000) != 0 )
    v139 = *(_QWORD *)(v12 - 8);
  else
    v139 = v121 - 24 * v149;
  v140 = (__int64 **)(v139 + 24LL * (unsigned int)(v149 - 1));
  if ( *v140 )
  {
    v141 = v140[1];
    v142 = (unsigned __int64)v140[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v142 = v141;
    if ( v141 )
      v141[2] = v141[2] & 3 | v142;
  }
  *v140 = v147;
  if ( v147 )
  {
    v143 = v147[1];
    v140[1] = (__int64 *)v143;
    if ( v143 )
      *(_QWORD *)(v143 + 16) = (unsigned __int64)(v140 + 1) | *(_QWORD *)(v143 + 16) & 3LL;
    v140[2] = (__int64 *)((unsigned __int64)(v147 + 1) | (unsigned __int64)v140[2] & 3);
    v147[1] = (__int64)v140;
  }
  v144 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
    v145 = *(_QWORD *)(v12 - 8);
  else
    v145 = v121 - 24 * v144;
  *(_QWORD *)(v145 + 8LL * (unsigned int)(v144 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v146;
  while ( 1 )
  {
    v122 = *(_QWORD *)(v122 + 8);
    if ( !v122 )
      break;
    v123 = sub_1648700(v122);
    if ( (unsigned __int8)(*(_BYTE *)(v123 + 16) - 25) <= 9u )
      goto LABEL_133;
  }
LABEL_100:
  sub_1290AF0(*a1, v326, 0);
  v124 = sub_127FEC0((__int64)*a1, (__int64)v327);
  v125 = a1[1][1];
  sub_1290AF0(*a1, v329, 0);
  v126 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( v126 == *(_DWORD *)(v12 + 56) )
  {
    sub_15F55D0(v12);
    v126 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  }
  v127 = (v126 + 1) & 0xFFFFFFF;
  v128 = v127 | *(_DWORD *)(v12 + 20) & 0xF0000000;
  *(_DWORD *)(v12 + 20) = v128;
  if ( (v128 & 0x40000000) != 0 )
    v129 = *(_QWORD *)(v12 - 8);
  else
    v129 = v121 - 24 * v127;
  v130 = (__int64 *)(v129 + 24LL * (unsigned int)(v127 - 1));
  if ( *v130 )
  {
    v131 = v130[1];
    v132 = v130[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v132 = v131;
    if ( v131 )
      *(_QWORD *)(v131 + 16) = v132 | *(_QWORD *)(v131 + 16) & 3LL;
  }
  *v130 = v124;
  if ( v124 )
  {
    v133 = *(_QWORD *)(v124 + 8);
    v130[1] = v133;
    if ( v133 )
      *(_QWORD *)(v133 + 16) = (unsigned __int64)(v130 + 1) | *(_QWORD *)(v133 + 16) & 3LL;
    v130[2] = (v124 + 8) | v130[2] & 3;
    *(_QWORD *)(v124 + 8) = v130;
  }
  v134 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
    v135 = *(_QWORD *)(v12 - 8);
  else
    v135 = v121 - 24 * v134;
  *(_QWORD *)(v135 + 8LL * (unsigned int)(v134 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v125;
  v26 = (const char **)a1[1];
  v136 = *v26;
  if ( *v26 )
  {
    v350 = *v26;
    sub_1623A60(&v350, v136, 2);
    if ( *(_QWORD *)(v12 + 48) )
      sub_161E7C0(v12 + 48);
    v137 = v350;
    *(_QWORD *)(v12 + 48) = v350;
    if ( v137 )
      sub_1623210(&v350, v137, v12 + 48);
    v26 = (const char **)a1[1];
  }
  BYTE1(v346) = 1;
  v138 = "lor.ext";
LABEL_162:
  v180 = a1[2];
  v344 = (char *)v138;
  LOBYTE(v346) = 3;
  v32 = sub_1643350(v180);
  if ( v32 != *(_QWORD *)v12 )
  {
LABEL_34:
    if ( *(_BYTE *)(v12 + 16) > 0x10u )
    {
      LOWORD(v352) = 257;
      v12 = sub_15FDBD0(37, v12, v32, &v350, 0);
      v248 = v26[1];
      if ( v248 )
      {
LABEL_249:
        v249 = (__int64 *)v26[2];
        sub_157E9D0(v248 + 40, v12);
        v250 = *(_QWORD *)(v12 + 24);
        v251 = *v249;
        *(_QWORD *)(v12 + 32) = v249;
        v251 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v251 | v250 & 7;
        *(_QWORD *)(v251 + 8) = v12 + 24;
        *v249 = *v249 & 7 | (v12 + 24);
      }
LABEL_250:
      sub_164B780(v12, &v344);
      v252 = *v26;
      if ( *v26 )
      {
        v338 = *v26;
        sub_1623A60(&v338, v252, 2);
        if ( *(_QWORD *)(v12 + 48) )
          sub_161E7C0(v12 + 48);
        v253 = v338;
        *(_QWORD *)(v12 + 48) = v338;
        if ( v253 )
          sub_1623210(&v338, v253, v12 + 48);
      }
    }
    else
    {
      return (char *)sub_15A46C0(37, v12, v32, 0);
    }
  }
  return (char *)v12;
}
