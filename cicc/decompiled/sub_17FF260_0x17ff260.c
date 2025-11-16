// Function: sub_17FF260
// Address: 0x17ff260
//
__int64 __fastcall sub_17FF260(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 v19; // r12
  __int64 v20; // rax
  char v21; // al
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned int v24; // r13d
  __int64 v25; // r14
  int v26; // eax
  __int64 v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // r13
  int v31; // eax
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // r12
  _QWORD *v40; // rax
  unsigned __int8 *v41; // rsi
  char v42; // al
  __int64 v43; // r14
  int v44; // eax
  __int64 v45; // rbx
  __int64 *v46; // r13
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // r13
  _QWORD *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rbx
  bool v56; // bl
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // r12d
  __int64 v63; // rbx
  int v64; // eax
  __int64 v65; // r14
  __int64 *v66; // r13
  __int64 v67; // rax
  __int64 v68; // rbx
  int v69; // eax
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // rbx
  _QWORD *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r13
  __int64 v76; // rdi
  __int64 v77; // rbx
  _QWORD *v78; // rax
  unsigned __int8 *v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // rsi
  char v82; // r12
  unsigned __int64 *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 *v87; // rdi
  __int64 v88; // rdi
  __int64 v89; // rbx
  _QWORD *v90; // rax
  unsigned __int8 *v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rsi
  char v96; // r15
  unsigned __int64 *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rdx
  __int64 *v101; // rdi
  __int64 v102; // rbx
  __int64 v103; // r13
  __int64 v104; // r13
  int v105; // eax
  __int64 v106; // rsi
  __int64 v107; // rax
  int v108; // eax
  unsigned int v109; // edx
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rax
  __int64 ***v114; // r13
  __int64 **v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rax
  double v119; // xmm4_8
  double v120; // xmm5_8
  unsigned __int64 v121; // rsi
  __int64 v122; // rax
  __int64 v123; // rsi
  __int64 v124; // rdx
  unsigned __int8 *v125; // rsi
  __int64 *v126; // r12
  __int64 *v127; // rbx
  __int64 v128; // rsi
  __int64 *v129; // rbx
  __int64 v130; // r15
  _QWORD *v131; // rax
  unsigned __int8 *v132; // rsi
  __int64 v133; // rax
  __int64 v134; // rsi
  __int64 v135; // r8
  __int64 v136; // rsi
  __int64 v137; // rax
  __int64 v138; // r13
  __int64 **v139; // rsi
  __int64 v140; // rax
  __int64 v141; // r13
  __int64 v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rsi
  __int64 v145; // r8
  __int64 **v146; // rsi
  __int64 v147; // rax
  __int64 v148; // r9
  __int64 **v149; // rsi
  __int64 v150; // rax
  __int64 v151; // r9
  int v152; // eax
  __int64 v153; // r13
  __int64 v154; // rax
  bool v155; // zf
  __int64 v156; // r13
  __int64 v157; // r14
  _QWORD *v158; // rax
  __int64 v159; // r8
  __int64 *v160; // r13
  __int64 v161; // rcx
  __int64 v162; // rax
  __int64 v163; // rsi
  __int64 v164; // rdx
  unsigned __int8 *v165; // rsi
  __int64 v166; // r9
  __int64 v167; // rax
  __int64 v168; // rsi
  __int64 v169; // rsi
  __int64 v170; // rdx
  unsigned __int8 *v171; // rsi
  _QWORD *v172; // rdi
  __int64 v173; // r9
  __int64 v174; // rax
  __int64 v175; // rsi
  __int64 v176; // rsi
  __int64 v177; // rdx
  unsigned __int8 *v178; // rsi
  __int64 v179; // rsi
  __int64 v180; // rax
  __int64 v181; // rsi
  __int64 v182; // rdx
  unsigned __int8 *v183; // rsi
  __int64 v184; // rdi
  __int64 v185; // r8
  __int64 *v186; // r13
  __int64 v187; // rcx
  __int64 v188; // rax
  __int64 v189; // rsi
  __int64 v190; // rdx
  unsigned __int8 *v191; // rsi
  __int64 v192; // rsi
  __int64 v193; // rax
  __int64 v194; // rsi
  __int64 v195; // rdx
  unsigned __int8 *v196; // rsi
  __int64 v197; // rdi
  __int64 v198; // rsi
  __int64 v199; // rax
  __int64 v200; // rsi
  __int64 v201; // rdx
  unsigned __int8 *v202; // rsi
  __int64 **v203; // rax
  __int64 *v204; // rax
  __int64 v205; // rsi
  unsigned __int64 *v206; // rbx
  __int64 v207; // rax
  unsigned __int64 v208; // rcx
  __int64 v209; // rsi
  unsigned __int8 *v210; // rsi
  _QWORD *v211; // rdi
  __int64 *v212; // r13
  __int64 v213; // rax
  __int64 v214; // rcx
  __int64 v215; // rsi
  unsigned __int8 *v216; // rsi
  __int64 v217; // rsi
  __int64 v218; // rax
  __int64 v219; // rsi
  __int64 v220; // rdx
  unsigned __int8 *v221; // rsi
  __int64 v222; // rsi
  __int64 v223; // rax
  __int64 v224; // rsi
  __int64 v225; // rdx
  unsigned __int8 *v226; // rsi
  unsigned __int64 *v227; // rbx
  __int64 **v228; // rax
  unsigned __int64 v229; // rcx
  __int64 v230; // rsi
  unsigned __int8 *v231; // rsi
  __int64 v232; // [rsp+0h] [rbp-3C0h]
  __int64 v233; // [rsp+8h] [rbp-3B8h]
  __int64 *v234; // [rsp+10h] [rbp-3B0h]
  __int64 *v235; // [rsp+18h] [rbp-3A8h]
  __int64 v236; // [rsp+20h] [rbp-3A0h]
  __int64 v237; // [rsp+28h] [rbp-398h]
  int v238; // [rsp+40h] [rbp-380h]
  _QWORD *v239; // [rsp+40h] [rbp-380h]
  _QWORD *v240; // [rsp+68h] [rbp-358h]
  unsigned __int8 v242; // [rsp+90h] [rbp-330h]
  __int64 v243; // [rsp+A8h] [rbp-318h]
  __int64 *v244; // [rsp+A8h] [rbp-318h]
  __int64 *v245; // [rsp+A8h] [rbp-318h]
  __int64 v246; // [rsp+A8h] [rbp-318h]
  __int64 v248; // [rsp+B8h] [rbp-308h]
  __int64 *v249; // [rsp+B8h] [rbp-308h]
  __int64 v250; // [rsp+B8h] [rbp-308h]
  __int64 v251; // [rsp+B8h] [rbp-308h]
  __int64 v252; // [rsp+B8h] [rbp-308h]
  __int64 v253; // [rsp+B8h] [rbp-308h]
  __int64 v254; // [rsp+B8h] [rbp-308h]
  __int64 v255; // [rsp+B8h] [rbp-308h]
  __int64 v256; // [rsp+B8h] [rbp-308h]
  __int64 v257; // [rsp+B8h] [rbp-308h]
  __int64 v258; // [rsp+B8h] [rbp-308h]
  __int64 v259; // [rsp+B8h] [rbp-308h]
  __int64 v260; // [rsp+B8h] [rbp-308h]
  __int64 v261; // [rsp+B8h] [rbp-308h]
  __int64 v262; // [rsp+B8h] [rbp-308h]
  __int64 v263; // [rsp+B8h] [rbp-308h]
  __int64 *v264; // [rsp+B8h] [rbp-308h]
  __int64 *v265; // [rsp+B8h] [rbp-308h]
  unsigned __int8 v266; // [rsp+CEh] [rbp-2F2h]
  bool v267; // [rsp+CFh] [rbp-2F1h]
  __int64 v268; // [rsp+D0h] [rbp-2F0h]
  unsigned __int64 *v269; // [rsp+D0h] [rbp-2F0h]
  unsigned __int8 v270; // [rsp+D0h] [rbp-2F0h]
  __int64 *v271; // [rsp+D0h] [rbp-2F0h]
  __int64 *v272; // [rsp+D0h] [rbp-2F0h]
  __int64 *v273; // [rsp+D0h] [rbp-2F0h]
  unsigned __int64 v274; // [rsp+D8h] [rbp-2E8h]
  __int64 *i; // [rsp+D8h] [rbp-2E8h]
  __int64 *v276; // [rsp+D8h] [rbp-2E8h]
  unsigned __int8 *v277; // [rsp+E8h] [rbp-2D8h] BYREF
  __int64 v278[2]; // [rsp+F0h] [rbp-2D0h] BYREF
  __int16 v279; // [rsp+100h] [rbp-2C0h]
  __int64 v280[2]; // [rsp+110h] [rbp-2B0h] BYREF
  __int16 v281; // [rsp+120h] [rbp-2A0h]
  __int64 v282[2]; // [rsp+130h] [rbp-290h] BYREF
  __int16 v283; // [rsp+140h] [rbp-280h]
  unsigned __int8 *v284; // [rsp+150h] [rbp-270h] BYREF
  __int64 v285; // [rsp+158h] [rbp-268h]
  __int64 v286; // [rsp+160h] [rbp-260h]
  __int64 v287[2]; // [rsp+170h] [rbp-250h] BYREF
  __int16 v288; // [rsp+180h] [rbp-240h]
  unsigned __int8 *v289; // [rsp+190h] [rbp-230h] BYREF
  __int64 v290; // [rsp+198h] [rbp-228h]
  __int64 v291; // [rsp+1A0h] [rbp-220h]
  _QWORD *v292; // [rsp+1A8h] [rbp-218h]
  __int64 v293; // [rsp+1B0h] [rbp-210h]
  int v294; // [rsp+1B8h] [rbp-208h]
  __int64 v295; // [rsp+1C0h] [rbp-200h]
  __int64 v296; // [rsp+1C8h] [rbp-1F8h]
  __int64 *v297; // [rsp+1E0h] [rbp-1E0h] BYREF
  __int64 v298; // [rsp+1E8h] [rbp-1D8h]
  _BYTE v299[64]; // [rsp+1F0h] [rbp-1D0h] BYREF
  _BYTE *v300; // [rsp+230h] [rbp-190h] BYREF
  __int64 v301; // [rsp+238h] [rbp-188h]
  _BYTE v302[64]; // [rsp+240h] [rbp-180h] BYREF
  __int64 *v303; // [rsp+280h] [rbp-140h] BYREF
  __int64 v304; // [rsp+288h] [rbp-138h]
  _BYTE v305[64]; // [rsp+290h] [rbp-130h] BYREF
  __int64 *v306; // [rsp+2D0h] [rbp-F0h] BYREF
  __int64 v307; // [rsp+2D8h] [rbp-E8h]
  _BYTE v308[64]; // [rsp+2E0h] [rbp-E0h] BYREF
  unsigned __int8 *v309; // [rsp+320h] [rbp-A0h] BYREF
  const char *v310; // [rsp+328h] [rbp-98h]
  unsigned __int64 *v311; // [rsp+330h] [rbp-90h]
  _QWORD *v312; // [rsp+338h] [rbp-88h]
  __int64 v313; // [rsp+340h] [rbp-80h] BYREF
  __int64 v314; // [rsp+348h] [rbp-78h]
  __int64 v315; // [rsp+350h] [rbp-70h]
  __int64 v316; // [rsp+358h] [rbp-68h]
  __int64 v317; // [rsp+360h] [rbp-60h]
  int v318; // [rsp+368h] [rbp-58h]
  __int64 v319; // [rsp+370h] [rbp-50h]
  __int64 v320; // [rsp+378h] [rbp-48h]
  char v321; // [rsp+388h] [rbp-38h]
  char v322; // [rsp+389h] [rbp-37h]

  sub_17FC580(a1, *(_QWORD **)(a2 + 40));
  v297 = (__int64 *)v299;
  v298 = 0x800000000LL;
  v300 = v302;
  v301 = 0x800000000LL;
  v304 = 0x800000000LL;
  v307 = 0x800000000LL;
  v303 = (__int64 *)v305;
  v306 = (__int64 *)v308;
  v240 = (_QWORD *)(a2 + 112);
  v266 = sub_1560180(a2 + 112, 45);
  v10 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v11 = (__int64 *)a1[1];
  v12 = v10;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_300:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F9B6E8 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_300;
  }
  v267 = 0;
  v248 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
           *(_QWORD *)(v13 + 8),
           &unk_4F9B6E8)
       + 360;
  if ( *(_QWORD *)(a2 + 80) != a2 + 72 )
  {
    v274 = v12;
    v17 = *(_QWORD *)(a2 + 80);
    while ( 1 )
    {
      if ( !v17 )
        BUG();
      v18 = *(_QWORD *)(v17 + 24);
      v19 = v17 + 16;
      if ( v18 != v17 + 16 )
        break;
LABEL_22:
      sub_17FBC80((__int64 *)&v300, (__int64)&v297, v274);
      v17 = *(_QWORD *)(v17 + 8);
      if ( a2 + 72 == v17 )
      {
        v12 = v274;
        goto LABEL_24;
      }
    }
    while ( 1 )
    {
      if ( !v18 )
        BUG();
      v21 = *(_BYTE *)(v18 - 8);
      v22 = v18 - 24;
      if ( v21 != 54 && v21 != 55 )
        break;
      if ( sub_15F32D0(v18 - 24) && *(_BYTE *)(v18 + 32) )
      {
LABEL_11:
        v20 = (unsigned int)v304;
        if ( (unsigned int)v304 >= HIDWORD(v304) )
        {
          sub_16CD150((__int64)&v303, v305, 0, 8, v15, v16);
          v20 = (unsigned int)v304;
        }
        v303[v20] = v22;
        LODWORD(v304) = v304 + 1;
        goto LABEL_14;
      }
      v21 = *(_BYTE *)(v18 - 8);
      if ( (unsigned __int8)(v21 - 54) > 1u )
      {
LABEL_56:
        v56 = v21 == 78 || v21 == 29;
        if ( v56 )
        {
          if ( v21 == 78 )
          {
            sub_1AED190(v18 - 24, v248);
            if ( *(_BYTE *)(v18 - 8) == 78 )
            {
              v59 = *(_QWORD *)(v18 - 48);
              if ( !*(_BYTE *)(v59 + 16)
                && (*(_BYTE *)(v59 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v59 + 36) - 133) <= 4
                && ((1LL << (*(_BYTE *)(v59 + 36) + 123)) & 0x15) != 0 )
              {
                v60 = (unsigned int)v307;
                if ( (unsigned int)v307 >= HIDWORD(v307) )
                {
                  sub_16CD150((__int64)&v306, v308, 0, 8, v57, v58);
                  v60 = (unsigned int)v307;
                }
                v306[v60] = v22;
                LODWORD(v307) = v307 + 1;
              }
            }
          }
          sub_17FBC80((__int64 *)&v300, (__int64)&v297, v274);
          v267 = v56;
        }
LABEL_14:
        v18 = *(_QWORD *)(v18 + 8);
        if ( v19 == v18 )
          goto LABEL_22;
      }
      else
      {
        v23 = (unsigned int)v301;
        if ( (unsigned int)v301 >= HIDWORD(v301) )
        {
          sub_16CD150((__int64)&v300, v302, 0, 8, v15, v16);
          v23 = (unsigned int)v301;
        }
        *(_QWORD *)&v300[8 * v23] = v22;
        LODWORD(v301) = v301 + 1;
        v18 = *(_QWORD *)(v18 + 8);
        if ( v19 == v18 )
          goto LABEL_22;
      }
    }
    if ( (unsigned __int8)(v21 - 57) <= 2u )
      goto LABEL_11;
    goto LABEL_56;
  }
LABEL_24:
  v24 = v266;
  LOBYTE(v24) = byte_4FA6BA0 & v266;
  if ( ((unsigned __int8)byte_4FA6BA0 & v266) != 0 )
  {
    v126 = v297;
    v24 = 0;
    v127 = &v297[(unsigned int)v298];
    if ( v297 != v127 )
    {
      do
      {
        v128 = *v126++;
        v24 |= sub_17FE640((__int64)a1, v128, v12);
      }
      while ( v127 != v126 );
    }
  }
  v242 = byte_4FA6900;
  if ( byte_4FA6900 )
  {
    v249 = &v303[(unsigned int)v304];
    if ( v303 != v249 )
    {
      for ( i = v303; v249 != i; ++i )
      {
        v39 = *i;
        v40 = (_QWORD *)sub_16498A0(*i);
        v313 = 0;
        v309 = 0;
        v312 = v40;
        LODWORD(v314) = 0;
        v315 = 0;
        v316 = 0;
        v310 = *(const char **)(v39 + 40);
        v311 = (unsigned __int64 *)(v39 + 24);
        v41 = *(unsigned __int8 **)(v39 + 48);
        v289 = v41;
        if ( v41 )
        {
          sub_1623A60((__int64)&v289, (__int64)v41, 2);
          if ( v309 )
            sub_161E7C0((__int64)&v309, (__int64)v309);
          v309 = v289;
          if ( v289 )
            sub_1623210((__int64)&v289, v289, (__int64)&v309);
        }
        v42 = *(_BYTE *)(v39 + 16);
        switch ( v42 )
        {
          case '6':
            v25 = *(_QWORD *)(v39 - 24);
            v26 = sub_17FC080(*(_QWORD *)v25, v12);
            v27 = v26;
            if ( v26 >= 0 )
            {
              v28 = (__int64 *)sub_1644C60(v312, 8 << v26);
              v29 = sub_1647190(v28, 0);
              v288 = 257;
              if ( v29 == *(_QWORD *)v25 )
              {
                v30 = (unsigned __int8 *)v25;
              }
              else if ( *(_BYTE *)(v25 + 16) > 0x10u )
              {
                LOWORD(v291) = 257;
                v30 = (unsigned __int8 *)sub_15FDFF0(v25, v29, (__int64)&v289, 0);
                if ( v310 )
                {
                  v269 = v311;
                  sub_157E9D0((__int64)(v310 + 40), (__int64)v30);
                  v121 = *v269;
                  v122 = *((_QWORD *)v30 + 3) & 7LL;
                  *((_QWORD *)v30 + 4) = v269;
                  v121 &= 0xFFFFFFFFFFFFFFF8LL;
                  *((_QWORD *)v30 + 3) = v121 | v122;
                  *(_QWORD *)(v121 + 8) = v30 + 24;
                  *v269 = *v269 & 7 | (unsigned __int64)(v30 + 24);
                }
                sub_164B780((__int64)v30, v287);
                if ( v309 )
                {
                  v284 = v309;
                  sub_1623A60((__int64)&v284, (__int64)v309, 2);
                  v123 = *((_QWORD *)v30 + 6);
                  v124 = (__int64)(v30 + 48);
                  if ( v123 )
                  {
                    sub_161E7C0((__int64)(v30 + 48), v123);
                    v124 = (__int64)(v30 + 48);
                  }
                  v125 = v284;
                  *((_QWORD *)v30 + 6) = v284;
                  if ( v125 )
                    sub_1623210((__int64)&v284, v125, v124);
                }
              }
              else
              {
                v30 = (unsigned __int8 *)sub_15A4A70((__int64 ***)v25, v29);
              }
              v284 = v30;
              v31 = (*(unsigned __int16 *)(v39 + 18) >> 7) & 7;
              v32 = (unsigned int)(v31 - 2);
              if ( (unsigned int)(v31 - 4) >= 4 )
                v32 = 0;
              v33 = sub_1643350(v312);
              v285 = sub_159C470(v33, v32, 0);
              v34 = *(_QWORD *)(*(_QWORD *)v25 + 24LL);
              LOWORD(v291) = 257;
              v35 = sub_1285290(
                      (__int64 *)&v309,
                      *(_QWORD *)(a1[v27 + 46] + 24LL),
                      a1[v27 + 46],
                      (int)&v284,
                      2,
                      (__int64)&v289,
                      0);
              LOWORD(v291) = 257;
              v36 = sub_17FE280((__int64 *)&v309, v35, v34, (__int64 *)&v289);
              sub_164D160(v39, v36, a3, a4, a5, a6, v37, v38, a9, a10);
              v24 = v242;
            }
            break;
          case '7':
            v43 = *(_QWORD *)(v39 - 24);
            v44 = sub_17FC080(*(_QWORD *)v43, v12);
            v45 = v44;
            if ( v44 >= 0 )
            {
              v46 = (__int64 *)sub_1644C60(v312, 8 << v44);
              v47 = sub_1647190(v46, 0);
              v288 = 257;
              if ( v47 != *(_QWORD *)v43 )
              {
                if ( *(_BYTE *)(v43 + 16) > 0x10u )
                {
                  LOWORD(v291) = 257;
                  v43 = sub_15FDFF0(v43, v47, (__int64)&v289, 0);
                  if ( v310 )
                  {
                    v271 = (__int64 *)v311;
                    sub_157E9D0((__int64)(v310 + 40), v43);
                    v179 = *v271;
                    v180 = *(_QWORD *)(v43 + 24) & 7LL;
                    *(_QWORD *)(v43 + 32) = v271;
                    v179 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v43 + 24) = v179 | v180;
                    *(_QWORD *)(v179 + 8) = v43 + 24;
                    *v271 = *v271 & 7 | (v43 + 24);
                  }
                  sub_164B780(v43, v287);
                  if ( v309 )
                  {
                    v284 = v309;
                    sub_1623A60((__int64)&v284, (__int64)v309, 2);
                    v181 = *(_QWORD *)(v43 + 48);
                    v182 = v43 + 48;
                    if ( v181 )
                    {
                      sub_161E7C0(v43 + 48, v181);
                      v182 = v43 + 48;
                    }
                    v183 = v284;
                    *(_QWORD *)(v43 + 48) = v284;
                    if ( v183 )
                      sub_1623210((__int64)&v284, v183, v182);
                  }
                }
                else
                {
                  v43 = sub_15A4A70((__int64 ***)v43, v47);
                }
              }
              v284 = (unsigned __int8 *)v43;
              LOWORD(v291) = 257;
              v285 = sub_17FE280((__int64 *)&v309, *(_QWORD *)(v39 - 48), (__int64)v46, (__int64 *)&v289);
              v48 = (*(unsigned __int16 *)(v39 + 18) >> 7) & 7;
              v49 = (unsigned int)(v48 - 2);
              if ( (unsigned int)(v48 - 4) >= 4 )
                v49 = 0;
              v50 = sub_1643350(v312);
              v286 = sub_159C470(v50, v49, 0);
              LOWORD(v291) = 257;
              v51 = a1[v45 + 51];
              v52 = *(_QWORD *)(*(_QWORD *)v51 + 24LL);
              v53 = sub_1648AB0(72, 4u, 0);
              v55 = (__int64)v53;
              if ( v53 )
              {
                sub_15F1EA0((__int64)v53, **(_QWORD **)(v52 + 16), 54, (__int64)(v53 - 12), 4, 0);
                *(_QWORD *)(v55 + 56) = 0;
                sub_15F5B40(v55, v52, v51, (__int64 *)&v284, 3, (__int64)&v289, 0, 0);
              }
LABEL_53:
              sub_1AA6530(v39, v55, v54);
              v24 = v242;
            }
            break;
          case ';':
            v63 = *(_QWORD *)(v39 - 48);
            v64 = sub_17FC080(*(_QWORD *)v63, v12);
            if ( v64 >= 0 )
            {
              v65 = a1[5 * ((*(unsigned __int16 *)(v39 + 18) >> 5) & 0x3FF) + 56 + v64];
              if ( v65 )
              {
                v66 = (__int64 *)sub_1644C60(v312, 8 << v64);
                v67 = sub_1647190(v66, 0);
                v288 = 257;
                if ( v67 != *(_QWORD *)v63 )
                {
                  if ( *(_BYTE *)(v63 + 16) > 0x10u )
                  {
                    LOWORD(v291) = 257;
                    v63 = sub_15FDFF0(v63, v67, (__int64)&v289, 0);
                    if ( v310 )
                    {
                      v272 = (__int64 *)v311;
                      sub_157E9D0((__int64)(v310 + 40), v63);
                      v217 = *v272;
                      v218 = *(_QWORD *)(v63 + 24) & 7LL;
                      *(_QWORD *)(v63 + 32) = v272;
                      v217 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v63 + 24) = v217 | v218;
                      *(_QWORD *)(v217 + 8) = v63 + 24;
                      *v272 = *v272 & 7 | (v63 + 24);
                    }
                    sub_164B780(v63, v287);
                    if ( v309 )
                    {
                      v284 = v309;
                      sub_1623A60((__int64)&v284, (__int64)v309, 2);
                      v219 = *(_QWORD *)(v63 + 48);
                      v220 = v63 + 48;
                      if ( v219 )
                      {
                        sub_161E7C0(v63 + 48, v219);
                        v220 = v63 + 48;
                      }
                      v221 = v284;
                      *(_QWORD *)(v63 + 48) = v284;
                      if ( v221 )
                        sub_1623210((__int64)&v284, v221, v220);
                    }
                  }
                  else
                  {
                    v63 = sub_15A4A70((__int64 ***)v63, v67);
                  }
                }
                v284 = (unsigned __int8 *)v63;
                v288 = 257;
                v68 = *(_QWORD *)(v39 - 24);
                if ( v66 != *(__int64 **)v68 )
                {
                  if ( *(_BYTE *)(v68 + 16) > 0x10u )
                  {
                    v211 = *(_QWORD **)(v39 - 24);
                    LOWORD(v291) = 257;
                    v68 = sub_15FE0A0(v211, (__int64)v66, 0, (__int64)&v289, 0);
                    if ( v310 )
                    {
                      v212 = (__int64 *)v311;
                      sub_157E9D0((__int64)(v310 + 40), v68);
                      v213 = *(_QWORD *)(v68 + 24);
                      v214 = *v212;
                      *(_QWORD *)(v68 + 32) = v212;
                      v214 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v68 + 24) = v214 | v213 & 7;
                      *(_QWORD *)(v214 + 8) = v68 + 24;
                      *v212 = *v212 & 7 | (v68 + 24);
                    }
                    sub_164B780(v68, v287);
                    if ( v309 )
                    {
                      v282[0] = (__int64)v309;
                      sub_1623A60((__int64)v282, (__int64)v309, 2);
                      v215 = *(_QWORD *)(v68 + 48);
                      if ( v215 )
                        sub_161E7C0(v68 + 48, v215);
                      v216 = (unsigned __int8 *)v282[0];
                      *(_QWORD *)(v68 + 48) = v282[0];
                      if ( v216 )
                        sub_1623210((__int64)v282, v216, v68 + 48);
                    }
                  }
                  else
                  {
                    v68 = sub_15A4750(*(__int64 ****)(v39 - 24), (__int64 **)v66, 0);
                  }
                }
                v285 = v68;
                v69 = (*(unsigned __int16 *)(v39 + 18) >> 2) & 7;
                v70 = (unsigned int)(v69 - 2);
                if ( (unsigned int)(v69 - 4) >= 4 )
                  v70 = 0;
                v71 = sub_1643350(v312);
                v286 = sub_159C470(v71, v70, 0);
                LOWORD(v291) = 257;
                v72 = *(_QWORD *)(*(_QWORD *)v65 + 24LL);
                v73 = sub_1648AB0(72, 4u, 0);
                v75 = (__int64)v73;
                if ( v73 )
                {
                  sub_15F1EA0((__int64)v73, **(_QWORD **)(v72 + 16), 54, (__int64)(v73 - 12), 4, 0);
                  *(_QWORD *)(v75 + 56) = 0;
                  sub_15F5B40(v75, v72, v65, (__int64 *)&v284, 3, (__int64)&v289, 0, 0);
                  v74 = v233;
                }
                sub_1AA6530(v39, v75, v74);
                v24 = v242;
              }
            }
            break;
          case ':':
            v102 = *(_QWORD *)(v39 - 72);
            v238 = sub_17FC080(*(_QWORD *)v102, v12);
            if ( v238 >= 0 )
            {
              v235 = (__int64 *)sub_1644C60(v312, 8 << v238);
              v268 = sub_1647190(v235, 0);
              LOWORD(v291) = 257;
              v237 = sub_17FE280((__int64 *)&v309, *(_QWORD *)(v39 - 48), (__int64)v235, (__int64 *)&v289);
              LOWORD(v291) = 257;
              v103 = sub_17FE280((__int64 *)&v309, *(_QWORD *)(v39 - 24), (__int64)v235, (__int64 *)&v289);
              v288 = 257;
              if ( v268 != *(_QWORD *)v102 )
              {
                if ( *(_BYTE *)(v102 + 16) > 0x10u )
                {
                  LOWORD(v291) = 257;
                  v102 = sub_15FDFF0(v102, v268, (__int64)&v289, 0);
                  if ( v310 )
                  {
                    v273 = (__int64 *)v311;
                    sub_157E9D0((__int64)(v310 + 40), v102);
                    v222 = *v273;
                    v223 = *(_QWORD *)(v102 + 24) & 7LL;
                    *(_QWORD *)(v102 + 32) = v273;
                    v222 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v102 + 24) = v222 | v223;
                    *(_QWORD *)(v222 + 8) = v102 + 24;
                    *v273 = *v273 & 7 | (v102 + 24);
                  }
                  sub_164B780(v102, v287);
                  if ( v309 )
                  {
                    v284 = v309;
                    sub_1623A60((__int64)&v284, (__int64)v309, 2);
                    v224 = *(_QWORD *)(v102 + 48);
                    v225 = v102 + 48;
                    if ( v224 )
                    {
                      sub_161E7C0(v102 + 48, v224);
                      v225 = v102 + 48;
                    }
                    v226 = v284;
                    *(_QWORD *)(v102 + 48) = v284;
                    if ( v226 )
                      sub_1623210((__int64)&v284, v226, v225);
                  }
                }
                else
                {
                  v102 = sub_15A4A70((__int64 ***)v102, v268);
                }
              }
              v289 = (unsigned __int8 *)v102;
              v291 = v103;
              v104 = 0;
              v290 = v237;
              v105 = (*(unsigned __int16 *)(v39 + 18) >> 2) & 7;
              v106 = (unsigned int)(v105 - 2);
              if ( (unsigned int)(v105 - 4) >= 4 )
                v106 = 0;
              v107 = sub_1643350(v312);
              v292 = (_QWORD *)sub_159C470(v107, v106, 0);
              v108 = (*(unsigned __int16 *)(v39 + 18) >> 5) & 7;
              v109 = v108 - 4;
              v110 = (unsigned int)(v108 - 2);
              if ( v109 < 4 )
                v104 = v110;
              v111 = sub_1643350(v312);
              v293 = sub_159C470(v111, v104, 0);
              v288 = 257;
              v112 = a1[v238 + 111];
              v113 = sub_1285290((__int64 *)&v309, *(_QWORD *)(v112 + 24), v112, (int)&v289, 5, (__int64)v287, 0);
              LOWORD(v286) = 257;
              v114 = (__int64 ***)v113;
              if ( *(_BYTE *)(v113 + 16) > 0x10u || *(_BYTE *)(v237 + 16) > 0x10u )
              {
                v288 = 257;
                v239 = sub_1648A60(56, 2u);
                if ( v239 )
                {
                  v236 = (__int64)v239;
                  v203 = *v114;
                  if ( *((_BYTE *)*v114 + 8) == 16 )
                  {
                    v234 = v203[4];
                    v204 = (__int64 *)sub_1643320(*v203);
                    v205 = (__int64)sub_16463B0(v204, (unsigned int)v234);
                  }
                  else
                  {
                    v205 = sub_1643320(*v203);
                  }
                  sub_15FEC10((__int64)v239, v205, 51, 32, (__int64)v114, v237, (__int64)v287, 0);
                }
                else
                {
                  v236 = 0;
                }
                if ( v310 )
                {
                  v206 = v311;
                  sub_157E9D0((__int64)(v310 + 40), (__int64)v239);
                  v207 = v239[3];
                  v208 = *v206 & 0xFFFFFFFFFFFFFFF8LL;
                  v239[4] = v206;
                  v239[3] = v208 | v207 & 7;
                  *(_QWORD *)(v208 + 8) = v239 + 3;
                  *v206 = *v206 & 7 | (unsigned __int64)(v239 + 3);
                }
                sub_164B780(v236, (__int64 *)&v284);
                if ( v309 )
                {
                  v282[0] = (__int64)v309;
                  sub_1623A60((__int64)v282, (__int64)v309, 2);
                  v209 = v239[6];
                  if ( v209 )
                    sub_161E7C0((__int64)(v239 + 6), v209);
                  v210 = (unsigned __int8 *)v282[0];
                  v239[6] = v282[0];
                  if ( v210 )
                    sub_1623210((__int64)v282, v210, (__int64)(v239 + 6));
                }
              }
              else
              {
                v239 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v113, (_QWORD *)v237, 0);
              }
              v115 = **(__int64 ****)(v39 - 24);
              if ( v235 != (__int64 *)v115 )
              {
                LOWORD(v286) = 257;
                if ( v115 != *v114 )
                {
                  if ( *((_BYTE *)v114 + 16) > 0x10u )
                  {
                    v288 = 257;
                    v114 = (__int64 ***)sub_15FDBD0(46, (__int64)v114, (__int64)v115, (__int64)v287, 0);
                    if ( v310 )
                    {
                      v227 = v311;
                      sub_157E9D0((__int64)(v310 + 40), (__int64)v114);
                      v228 = v114[3];
                      v229 = *v227;
                      v114[4] = (__int64 **)v227;
                      v229 &= 0xFFFFFFFFFFFFFFF8LL;
                      v114[3] = (__int64 **)(v229 | (unsigned __int8)v228 & 7);
                      *(_QWORD *)(v229 + 8) = v114 + 3;
                      *v227 = *v227 & 7 | (unsigned __int64)(v114 + 3);
                    }
                    sub_164B780((__int64)v114, (__int64 *)&v284);
                    if ( v309 )
                    {
                      v282[0] = (__int64)v309;
                      sub_1623A60((__int64)v282, (__int64)v309, 2);
                      v230 = (__int64)v114[6];
                      if ( v230 )
                        sub_161E7C0((__int64)(v114 + 6), v230);
                      v231 = (unsigned __int8 *)v282[0];
                      v114[6] = (__int64 **)v282[0];
                      if ( v231 )
                        sub_1623210((__int64)v282, v231, (__int64)(v114 + 6));
                    }
                  }
                  else
                  {
                    v114 = (__int64 ***)sub_15A46C0(46, v114, v115, 0);
                  }
                }
              }
              LODWORD(v284) = 0;
              v288 = 257;
              v116 = sub_1599EF0(*(__int64 ***)v39);
              v117 = sub_17FE490((__int64 *)&v309, v116, (__int64)v114, &v284, 1, v287);
              v288 = 257;
              LODWORD(v284) = 1;
              v118 = sub_17FE490((__int64 *)&v309, v117, (__int64)v239, &v284, 1, v287);
              sub_164D160(v39, v118, a3, a4, a5, a6, v119, v120, a9, a10);
              sub_15F20C0((_QWORD *)v39);
              v24 = v242;
            }
            break;
          default:
            v24 = v242;
            if ( v42 == 57 )
            {
              v152 = (*(unsigned __int16 *)(v39 + 18) >> 1) & 0x7FFFBFFF;
              v153 = (unsigned int)(v152 - 2);
              if ( (unsigned int)(v152 - 4) >= 4 )
                v153 = 0;
              v154 = sub_1643350(v312);
              v287[0] = sub_159C470(v154, v153, 0);
              v155 = *(_BYTE *)(v39 + 56) == 0;
              LOWORD(v291) = 257;
              v156 = a1[117];
              if ( !v155 )
                v156 = a1[116];
              v157 = *(_QWORD *)(*(_QWORD *)v156 + 24LL);
              v158 = sub_1648AB0(72, 2u, 0);
              v55 = (__int64)v158;
              if ( v158 )
              {
                sub_15F1EA0((__int64)v158, **(_QWORD **)(v157 + 16), 54, (__int64)(v158 - 6), 2, 0);
                *(_QWORD *)(v55 + 56) = 0;
                sub_15F5B40(v55, v157, v156, v287, 1, (__int64)&v289, 0, 0);
              }
              goto LABEL_53;
            }
            break;
        }
        if ( v309 )
          sub_161E7C0((__int64)&v309, (__int64)v309);
      }
    }
  }
  if ( byte_4FA6820 )
  {
    if ( v266 )
    {
      v129 = v306;
      v276 = &v306[(unsigned int)v307];
      if ( v306 != v276 )
      {
        v270 = v24;
        while ( 1 )
        {
          v130 = *v129;
          v131 = (_QWORD *)sub_16498A0(*v129);
          v313 = 0;
          v309 = 0;
          v312 = v131;
          LODWORD(v314) = 0;
          v315 = 0;
          v316 = 0;
          v310 = *(const char **)(v130 + 40);
          v311 = (unsigned __int64 *)(v130 + 24);
          v132 = *(unsigned __int8 **)(v130 + 48);
          v289 = v132;
          if ( v132 )
          {
            sub_1623A60((__int64)&v289, (__int64)v132, 2);
            if ( v309 )
              sub_161E7C0((__int64)&v309, (__int64)v309);
            v309 = v289;
            if ( v289 )
              sub_1623210((__int64)&v289, v289, (__int64)&v309);
          }
          if ( *(_BYTE *)(v130 + 16) != 78 )
            goto LABEL_154;
          v133 = *(_QWORD *)(v130 - 24);
          if ( *(_BYTE *)(v133 + 16) || (*(_BYTE *)(v133 + 33) & 0x20) == 0 )
            goto LABEL_154;
          if ( *(_DWORD *)(v133 + 36) == 137 )
          {
            v288 = 257;
            v283 = 257;
            v144 = sub_16471D0(v312, 0);
            v145 = *(_QWORD *)(v130 - 24LL * (*(_DWORD *)(v130 + 20) & 0xFFFFFFF));
            if ( v144 != *(_QWORD *)v145 )
            {
              if ( *(_BYTE *)(v145 + 16) > 0x10u )
              {
                LOWORD(v291) = 257;
                v159 = sub_15FDFF0(v145, v144, (__int64)&v289, 0);
                if ( v310 )
                {
                  v160 = (__int64 *)v311;
                  v250 = v159;
                  sub_157E9D0((__int64)(v310 + 40), v159);
                  v159 = v250;
                  v161 = *v160;
                  v162 = *(_QWORD *)(v250 + 24);
                  *(_QWORD *)(v250 + 32) = v160;
                  v161 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v250 + 24) = v161 | v162 & 7;
                  *(_QWORD *)(v161 + 8) = v250 + 24;
                  *v160 = *v160 & 7 | (v250 + 24);
                }
                v251 = v159;
                sub_164B780(v159, v282);
                v145 = v251;
                if ( v309 )
                {
                  v284 = v309;
                  sub_1623A60((__int64)&v284, (__int64)v309, 2);
                  v145 = v251;
                  v163 = *(_QWORD *)(v251 + 48);
                  v164 = v251 + 48;
                  if ( v163 )
                  {
                    v243 = v251;
                    v252 = v251 + 48;
                    sub_161E7C0(v252, v163);
                    v145 = v243;
                    v164 = v252;
                  }
                  v165 = v284;
                  *(_QWORD *)(v145 + 48) = v284;
                  if ( v165 )
                  {
                    v253 = v145;
                    sub_1623210((__int64)&v284, v165, v164);
                    v145 = v253;
                  }
                }
              }
              else
              {
                v145 = sub_15A4A70(*(__int64 ****)(v130 - 24LL * (*(_DWORD *)(v130 + 20) & 0xFFFFFFF)), v144);
              }
            }
            v284 = (unsigned __int8 *)v145;
            v281 = 257;
            v146 = (__int64 **)sub_1643350(v312);
            v147 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
            v148 = *(_QWORD *)(v130 + 24 * (1 - v147));
            if ( v146 != *(__int64 ***)v148 )
            {
              if ( *(_BYTE *)(v148 + 16) > 0x10u )
              {
                v172 = *(_QWORD **)(v130 + 24 * (1 - v147));
                LOWORD(v291) = 257;
                v173 = sub_15FE0A0(v172, (__int64)v146, 0, (__int64)&v289, 0);
                if ( v310 )
                {
                  v257 = v173;
                  v245 = (__int64 *)v311;
                  sub_157E9D0((__int64)(v310 + 40), v173);
                  v173 = v257;
                  v174 = *(_QWORD *)(v257 + 24);
                  v175 = *v245;
                  *(_QWORD *)(v257 + 32) = v245;
                  v175 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v257 + 24) = v175 | v174 & 7;
                  *(_QWORD *)(v175 + 8) = v257 + 24;
                  *v245 = *v245 & 7 | (v257 + 24);
                }
                v258 = v173;
                sub_164B780(v173, v280);
                v148 = v258;
                if ( v309 )
                {
                  v278[0] = (__int64)v309;
                  sub_1623A60((__int64)v278, (__int64)v309, 2);
                  v148 = v258;
                  v176 = *(_QWORD *)(v258 + 48);
                  v177 = v258 + 48;
                  if ( v176 )
                  {
                    sub_161E7C0(v258 + 48, v176);
                    v148 = v258;
                    v177 = v258 + 48;
                  }
                  v178 = (unsigned __int8 *)v278[0];
                  *(_QWORD *)(v148 + 48) = v278[0];
                  if ( v178 )
                  {
                    v259 = v148;
                    sub_1623210((__int64)v278, v178, v177);
                    v148 = v259;
                  }
                }
                v147 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
              }
              else
              {
                v148 = sub_15A4750(*(__int64 ****)(v130 + 24 * (1 - v147)), v146, 0);
                v147 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
              }
            }
            v285 = v148;
            v149 = (__int64 **)a1[20];
            v279 = 257;
            v150 = 3 * (2 - v147);
            v151 = *(_QWORD *)(v130 + 8 * v150);
            if ( v149 != *(__int64 ***)v151 )
            {
              if ( *(_BYTE *)(v151 + 16) > 0x10u )
              {
                LOWORD(v291) = 257;
                v166 = sub_15FE0A0((_QWORD *)v151, (__int64)v149, 0, (__int64)&v289, 0);
                if ( v310 )
                {
                  v254 = v166;
                  v244 = (__int64 *)v311;
                  sub_157E9D0((__int64)(v310 + 40), v166);
                  v166 = v254;
                  v167 = *(_QWORD *)(v254 + 24);
                  v168 = *v244;
                  *(_QWORD *)(v254 + 32) = v244;
                  v168 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v254 + 24) = v168 | v167 & 7;
                  *(_QWORD *)(v168 + 8) = v254 + 24;
                  *v244 = *v244 & 7 | (v254 + 24);
                }
                v255 = v166;
                sub_164B780(v166, v278);
                v151 = v255;
                if ( v309 )
                {
                  v277 = v309;
                  sub_1623A60((__int64)&v277, (__int64)v309, 2);
                  v151 = v255;
                  v169 = *(_QWORD *)(v255 + 48);
                  v170 = v255 + 48;
                  if ( v169 )
                  {
                    sub_161E7C0(v255 + 48, v169);
                    v151 = v255;
                    v170 = v255 + 48;
                  }
                  v171 = v277;
                  *(_QWORD *)(v151 + 48) = v277;
                  if ( v171 )
                  {
                    v256 = v151;
                    sub_1623210((__int64)&v277, v171, v170);
                    v151 = v256;
                  }
                }
              }
              else
              {
                v151 = sub_15A4750(*(__int64 ****)(v130 + 8 * v150), v149, 0);
              }
            }
            v286 = v151;
            v143 = a1[122];
          }
          else
          {
            if ( (*(_BYTE *)(v133 + 33) & 0x20) == 0 || (*(_DWORD *)(v133 + 36) & 0xFFFFFFFD) != 0x85 )
              goto LABEL_154;
            v288 = 257;
            v283 = 257;
            v134 = sub_16471D0(v312, 0);
            v135 = *(_QWORD *)(v130 - 24LL * (*(_DWORD *)(v130 + 20) & 0xFFFFFFF));
            if ( v134 != *(_QWORD *)v135 )
            {
              if ( *(_BYTE *)(v135 + 16) > 0x10u )
              {
                v184 = *(_QWORD *)(v130 - 24LL * (*(_DWORD *)(v130 + 20) & 0xFFFFFFF));
                LOWORD(v291) = 257;
                v185 = sub_15FDFF0(v184, v134, (__int64)&v289, 0);
                if ( v310 )
                {
                  v186 = (__int64 *)v311;
                  v260 = v185;
                  sub_157E9D0((__int64)(v310 + 40), v185);
                  v185 = v260;
                  v187 = *v186;
                  v188 = *(_QWORD *)(v260 + 24);
                  *(_QWORD *)(v260 + 32) = v186;
                  v187 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v260 + 24) = v187 | v188 & 7;
                  *(_QWORD *)(v187 + 8) = v260 + 24;
                  *v186 = *v186 & 7 | (v260 + 24);
                }
                v261 = v185;
                sub_164B780(v185, v282);
                v135 = v261;
                if ( v309 )
                {
                  v284 = v309;
                  sub_1623A60((__int64)&v284, (__int64)v309, 2);
                  v135 = v261;
                  v189 = *(_QWORD *)(v261 + 48);
                  v190 = v261 + 48;
                  if ( v189 )
                  {
                    v246 = v261;
                    v262 = v261 + 48;
                    sub_161E7C0(v262, v189);
                    v135 = v246;
                    v190 = v262;
                  }
                  v191 = v284;
                  *(_QWORD *)(v135 + 48) = v284;
                  if ( v191 )
                  {
                    v263 = v135;
                    sub_1623210((__int64)&v284, v191, v190);
                    v135 = v263;
                  }
                }
              }
              else
              {
                v135 = sub_15A4A70(*(__int64 ****)(v130 - 24LL * (*(_DWORD *)(v130 + 20) & 0xFFFFFFF)), v134);
              }
            }
            v284 = (unsigned __int8 *)v135;
            v281 = 257;
            v136 = sub_16471D0(v312, 0);
            v137 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
            v138 = *(_QWORD *)(v130 + 24 * (1 - v137));
            if ( v136 != *(_QWORD *)v138 )
            {
              if ( *(_BYTE *)(v138 + 16) > 0x10u )
              {
                v197 = *(_QWORD *)(v130 + 24 * (1 - v137));
                LOWORD(v291) = 257;
                v138 = sub_15FDFF0(v197, v136, (__int64)&v289, 0);
                if ( v310 )
                {
                  v265 = (__int64 *)v311;
                  sub_157E9D0((__int64)(v310 + 40), v138);
                  v198 = *v265;
                  v199 = *(_QWORD *)(v138 + 24) & 7LL;
                  *(_QWORD *)(v138 + 32) = v265;
                  v198 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v138 + 24) = v198 | v199;
                  *(_QWORD *)(v198 + 8) = v138 + 24;
                  *v265 = *v265 & 7 | (v138 + 24);
                }
                sub_164B780(v138, v280);
                if ( v309 )
                {
                  v278[0] = (__int64)v309;
                  sub_1623A60((__int64)v278, (__int64)v309, 2);
                  v200 = *(_QWORD *)(v138 + 48);
                  v201 = v138 + 48;
                  if ( v200 )
                  {
                    sub_161E7C0(v138 + 48, v200);
                    v201 = v138 + 48;
                  }
                  v202 = (unsigned __int8 *)v278[0];
                  *(_QWORD *)(v138 + 48) = v278[0];
                  if ( v202 )
                    sub_1623210((__int64)v278, v202, v201);
                }
                v137 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
              }
              else
              {
                v138 = sub_15A4A70(*(__int64 ****)(v130 + 24 * (1 - v137)), v136);
                v137 = *(_DWORD *)(v130 + 20) & 0xFFFFFFF;
              }
            }
            v285 = v138;
            v139 = (__int64 **)a1[20];
            v279 = 257;
            v140 = 3 * (2 - v137);
            v141 = *(_QWORD *)(v130 + 8 * v140);
            if ( v139 != *(__int64 ***)v141 )
            {
              if ( *(_BYTE *)(v141 + 16) > 0x10u )
              {
                LOWORD(v291) = 257;
                v141 = sub_15FE0A0((_QWORD *)v141, (__int64)v139, 0, (__int64)&v289, 0);
                if ( v310 )
                {
                  v264 = (__int64 *)v311;
                  sub_157E9D0((__int64)(v310 + 40), v141);
                  v192 = *v264;
                  v193 = *(_QWORD *)(v141 + 24) & 7LL;
                  *(_QWORD *)(v141 + 32) = v264;
                  v192 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v141 + 24) = v192 | v193;
                  *(_QWORD *)(v192 + 8) = v141 + 24;
                  *v264 = *v264 & 7 | (v141 + 24);
                }
                sub_164B780(v141, v278);
                if ( v309 )
                {
                  v277 = v309;
                  sub_1623A60((__int64)&v277, (__int64)v309, 2);
                  v194 = *(_QWORD *)(v141 + 48);
                  v195 = v141 + 48;
                  if ( v194 )
                  {
                    sub_161E7C0(v141 + 48, v194);
                    v195 = v141 + 48;
                  }
                  v196 = v277;
                  *(_QWORD *)(v141 + 48) = v277;
                  if ( v196 )
                    sub_1623210((__int64)&v277, v196, v195);
                }
              }
              else
              {
                v141 = sub_15A4750(*(__int64 ****)(v130 + 8 * v140), v139, 0);
              }
            }
            v286 = v141;
            v142 = *(_QWORD *)(v130 - 24);
            if ( *(_BYTE *)(v142 + 16) )
              BUG();
            v143 = a1[121];
            if ( *(_DWORD *)(v142 + 36) != 133 )
              v143 = a1[120];
          }
          sub_1285290((__int64 *)&v309, *(_QWORD *)(v143 + 24), v143, (int)&v284, 3, (__int64)v287, 0);
          sub_15F20C0((_QWORD *)v130);
LABEL_154:
          if ( v309 )
            sub_161E7C0((__int64)&v309, (__int64)v309);
          if ( v276 == ++v129 )
          {
            v24 = v270;
            break;
          }
        }
      }
    }
  }
  if ( sub_15602E0(v240, "sanitize_thread_no_checking_at_run_time", 0x27u) && v267 )
  {
    v76 = *(_QWORD *)(a2 + 80);
    if ( v76 )
      v76 -= 24;
    v77 = sub_157ED20(v76);
    v78 = (_QWORD *)sub_16498A0(v77);
    v289 = 0;
    v292 = v78;
    v293 = 0;
    v294 = 0;
    v295 = 0;
    v296 = 0;
    v290 = *(_QWORD *)(v77 + 40);
    v291 = v77 + 24;
    v79 = *(unsigned __int8 **)(v77 + 48);
    v309 = v79;
    if ( v79 )
    {
      sub_1623A60((__int64)&v309, (__int64)v79, 2);
      if ( v289 )
        sub_161E7C0((__int64)&v289, (__int64)v289);
      v289 = v309;
      if ( v309 )
        sub_1623210((__int64)&v309, v309, (__int64)&v289);
    }
    v80 = a1[24];
    LOWORD(v311) = 257;
    v81 = *(_QWORD *)(*(_QWORD *)v80 + 24LL);
    sub_1285290((__int64 *)&v289, v81, v80, 0, 0, (__int64)&v309, 0);
    v82 = byte_4FA69E0;
    v310 = "tsan_ignore_cleanup";
    v83 = *(unsigned __int64 **)(a2 + 80);
    v309 = (unsigned __int8 *)a2;
    v311 = v83;
    v312 = (_QWORD *)(a2 + 72);
    v84 = sub_15E0530(a2);
    v322 = v82;
    v85 = v232;
    v313 = 0;
    v315 = 0;
    v316 = v84;
    v317 = 0;
    v318 = 0;
    v319 = 0;
    v320 = 0;
    v314 = 0;
    v321 = 0;
    while ( 1 )
    {
      v87 = (__int64 *)sub_1AC5690(&v309, v81, v85);
      if ( !v87 )
        break;
      v86 = a1[25];
      v288 = 257;
      v81 = *(_QWORD *)(*(_QWORD *)v86 + 24LL);
      sub_1285290(v87, v81, v86, 0, 0, (__int64)v287, 0);
      v85 = v233;
    }
    if ( v313 )
      sub_161E7C0((__int64)&v313, v313);
    if ( v289 )
      sub_161E7C0((__int64)&v289, (__int64)v289);
  }
  else
  {
    v61 = v267;
    LOBYTE(v61) = v24 | v267;
    if ( !((unsigned __int8)v24 | v267) )
      goto LABEL_72;
  }
  v61 = (unsigned __int8)byte_4FA6AC0;
  if ( byte_4FA6AC0 )
  {
    v88 = *(_QWORD *)(a2 + 80);
    if ( v88 )
      v88 -= 24;
    v89 = sub_157ED20(v88);
    v90 = (_QWORD *)sub_16498A0(v89);
    v289 = 0;
    v292 = v90;
    v293 = 0;
    v294 = 0;
    v295 = 0;
    v296 = 0;
    v290 = *(_QWORD *)(v89 + 40);
    v291 = v89 + 24;
    v91 = *(unsigned __int8 **)(v89 + 48);
    v309 = v91;
    if ( v91 )
    {
      sub_1623A60((__int64)&v309, (__int64)v91, 2);
      if ( v289 )
        sub_161E7C0((__int64)&v289, (__int64)v289);
      v289 = v309;
      if ( v309 )
        sub_1623210((__int64)&v309, v309, (__int64)&v289);
    }
    LOWORD(v311) = 257;
    v92 = sub_1643350(v292);
    v287[0] = sub_159C470(v92, 0, 0);
    v93 = sub_15E26F0(*(__int64 **)(a2 + 40), 186, 0, 0);
    v284 = (unsigned __int8 *)sub_1285290((__int64 *)&v289, *(_QWORD *)(v93 + 24), v93, (int)v287, 1, (__int64)&v309, 0);
    LOWORD(v311) = 257;
    v94 = a1[22];
    v95 = *(_QWORD *)(v94 + 24);
    sub_1285290((__int64 *)&v289, v95, v94, (int)&v284, 1, (__int64)&v309, 0);
    v96 = byte_4FA69E0;
    v310 = "tsan_cleanup";
    v97 = *(unsigned __int64 **)(a2 + 80);
    v309 = (unsigned __int8 *)a2;
    v312 = (_QWORD *)(a2 + 72);
    v311 = v97;
    v98 = sub_15E0530(a2);
    v322 = v96;
    v313 = 0;
    v315 = 0;
    v316 = v98;
    v317 = 0;
    v318 = 0;
    v319 = 0;
    v320 = 0;
    v314 = 0;
    v321 = 0;
    while ( 1 )
    {
      v101 = (__int64 *)sub_1AC5690(&v309, v95, v99);
      if ( !v101 )
        break;
      v100 = a1[23];
      v288 = 257;
      v95 = *(_QWORD *)(v100 + 24);
      sub_1285290(v101, v95, v100, 0, 0, (__int64)v287, 0);
    }
    if ( v313 )
      sub_161E7C0((__int64)&v313, v313);
    if ( v289 )
      sub_161E7C0((__int64)&v289, (__int64)v289);
  }
  else
  {
    v61 = v24;
  }
LABEL_72:
  if ( v306 != (__int64 *)v308 )
    _libc_free((unsigned __int64)v306);
  if ( v303 != (__int64 *)v305 )
    _libc_free((unsigned __int64)v303);
  if ( v300 != v302 )
    _libc_free((unsigned __int64)v300);
  if ( v297 != (__int64 *)v299 )
    _libc_free((unsigned __int64)v297);
  return v61;
}
