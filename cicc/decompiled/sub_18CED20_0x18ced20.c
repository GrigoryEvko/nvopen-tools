// Function: sub_18CED20
// Address: 0x18ced20
//
__int64 __fastcall sub_18CED20(
        __int64 a1,
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
  __int64 v10; // r14
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  _QWORD *v13; // rbx
  __int64 v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 *v16; // r13
  __int64 i; // r15
  __int64 v19; // rax
  __int64 v20; // r12
  unsigned __int8 v21; // al
  bool v22; // al
  bool v23; // al
  __int64 v24; // rdi
  int v25; // edi
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned __int8 v30; // al
  int v31; // eax
  __int64 v32; // r13
  __int64 *v33; // rax
  __int64 v34; // rdi
  double v35; // xmm4_8
  double v36; // xmm5_8
  char v37; // al
  __int64 ***v38; // rax
  __int64 **v39; // rbx
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rbx
  _QWORD *v44; // rdi
  __int64 v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  char v48; // dl
  char v49; // al
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rdi
  int v53; // edi
  __int64 v54; // rax
  __int64 v55; // rbx
  unsigned __int8 v56; // al
  int v57; // eax
  __int64 *v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // r12
  _QWORD *v63; // rbx
  int v64; // r8d
  int v65; // r9d
  unsigned __int8 v66; // al
  __int64 v67; // rax
  __int64 v68; // rdi
  int v69; // edi
  __int64 v70; // rax
  __int64 v71; // rbx
  unsigned __int8 v72; // al
  int v73; // eax
  unsigned __int8 v74; // al
  __int64 v75; // rdx
  _QWORD *v76; // rcx
  _QWORD *v77; // rdx
  char v78; // al
  _QWORD *v79; // rax
  bool v80; // zf
  __int64 v81; // rcx
  unsigned __int64 v82; // rdx
  __int64 v83; // rdx
  char v84; // al
  __int64 v85; // rdi
  int v86; // esi
  __int64 *v87; // rdi
  unsigned int v88; // eax
  __int64 v89; // rdx
  __int64 *v90; // rdx
  __int64 v91; // rsi
  __int64 v92; // rbx
  __int64 v93; // rdx
  __int64 ii; // rdi
  int v95; // edi
  unsigned int v96; // eax
  __int64 v97; // rax
  unsigned __int8 v98; // al
  int v99; // eax
  _BYTE *v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rdi
  unsigned __int64 v103; // rax
  unsigned __int64 v104; // rax
  __int64 k; // rcx
  char v106; // al
  __int64 v107; // rdx
  double v108; // xmm4_8
  double v109; // xmm5_8
  __int64 v110; // rdx
  _QWORD *v111; // rax
  _QWORD *v112; // rcx
  __int64 v113; // rdx
  _QWORD *v114; // rsi
  _BYTE *v115; // rdi
  __int64 v116; // rbx
  int v117; // eax
  __int64 v118; // rdx
  unsigned __int8 v119; // al
  int v120; // edi
  __int64 v121; // rax
  __int64 v122; // rbx
  __int64 v123; // r12
  _QWORD *v124; // rdi
  int v125; // edi
  __int64 v126; // rax
  __int64 v127; // r13
  unsigned __int8 v128; // al
  int v129; // eax
  __int64 v130; // rax
  __int64 v131; // rbx
  _QWORD *v132; // rax
  __int64 v133; // r11
  __int64 *v134; // rsi
  __int64 v135; // rbx
  int v136; // esi
  double v137; // xmm4_8
  double v138; // xmm5_8
  __int64 v139; // rbx
  __int64 v140; // rax
  int v141; // eax
  __int64 v142; // rax
  __int64 *v143; // rbx
  __int64 *v144; // rax
  char v145; // al
  __int64 j; // rcx
  char v147; // al
  char v148; // al
  int v149; // eax
  __int64 v150; // rdi
  int v151; // eax
  __int64 m; // rdi
  int v153; // edi
  double v154; // xmm4_8
  double v155; // xmm5_8
  __int64 v156; // rax
  __int64 v157; // rbx
  unsigned __int8 v158; // al
  int v159; // eax
  __int64 v160; // rbx
  __int64 v161; // rbx
  __int64 *v162; // r12
  __int64 *v163; // rax
  __int64 *v164; // rax
  __int64 v165; // rax
  __int64 v166; // rax
  _QWORD *v167; // rax
  __int64 v168; // rcx
  unsigned __int64 v169; // rsi
  __int64 v170; // rcx
  __int64 *v171; // rax
  __int64 v172; // rbx
  int v173; // eax
  __int64 v174; // rdx
  __int64 jj; // rdi
  int v176; // edi
  __int64 v177; // rax
  __int64 v178; // rbx
  unsigned __int8 v179; // al
  char v180; // al
  __int64 v181; // rax
  unsigned __int64 v182; // rax
  __int64 v183; // rax
  __int64 v184; // rdx
  __int64 v185; // r12
  __int64 v186; // rax
  __int64 v187; // rax
  unsigned int *v188; // rax
  __int64 *v189; // r15
  __int64 v190; // r14
  __int64 v191; // r13
  __int64 v192; // r12
  unsigned int v193; // eax
  __int64 *v194; // rbx
  __int64 v195; // rax
  __m128i *v196; // rdi
  __int64 v197; // rdx
  __int64 v198; // rdx
  char *v199; // r14
  char *v200; // r8
  __int64 v201; // rax
  char *v202; // r13
  unsigned __int64 v203; // rax
  unsigned int v204; // eax
  __int64 v205; // rax
  __int64 *v206; // r13
  __int64 v207; // r12
  __int64 *v208; // r15
  __int64 v209; // rdi
  _QWORD *v210; // rax
  _QWORD *v211; // r13
  _QWORD *v212; // rax
  __int64 v213; // rcx
  unsigned __int64 v214; // rsi
  __int64 v215; // rcx
  int v216; // r8d
  int v217; // r9d
  __int64 v218; // rax
  __int64 *v219; // rax
  __int64 v220; // rax
  __int64 v221; // rbx
  __int64 *v222; // rax
  unsigned int v223; // edx
  __int64 v224; // rsi
  __int64 v225; // rdx
  unsigned __int64 **v226; // rax
  unsigned __int64 v227; // rdi
  __int64 v228; // r12
  unsigned __int64 v229; // rax
  __int64 v230; // rsi
  __m128 *v231; // r13
  __int64 *v232; // rax
  char *v233; // r9
  size_t v234; // rdx
  unsigned __int64 v235; // rsi
  bool v236; // cf
  unsigned __int64 v237; // rsi
  __int64 v238; // r12
  __int64 v239; // rax
  char *v240; // r15
  __int64 v241; // r12
  char *v242; // rdx
  char *v243; // rax
  char *v244; // rsi
  char *v245; // rdi
  __int64 v246; // r13
  char *v247; // rax
  char *v248; // r13
  size_t v249; // rdx
  int v250; // eax
  int v251; // edi
  char *v252; // [rsp+0h] [rbp-260h]
  size_t v253; // [rsp+8h] [rbp-258h]
  __int64 v254; // [rsp+10h] [rbp-250h]
  char *v255; // [rsp+20h] [rbp-240h]
  char *v256; // [rsp+20h] [rbp-240h]
  char *v257; // [rsp+20h] [rbp-240h]
  __int64 v258; // [rsp+28h] [rbp-238h]
  __int64 v259; // [rsp+30h] [rbp-230h]
  __int64 v260; // [rsp+38h] [rbp-228h]
  __int64 v261; // [rsp+40h] [rbp-220h]
  __int64 v262; // [rsp+48h] [rbp-218h]
  __int64 v263; // [rsp+50h] [rbp-210h]
  _QWORD *v264; // [rsp+58h] [rbp-208h]
  __int64 v265; // [rsp+60h] [rbp-200h]
  __int64 v266; // [rsp+68h] [rbp-1F8h]
  __int64 v267; // [rsp+88h] [rbp-1D8h]
  __int64 v268; // [rsp+88h] [rbp-1D8h]
  _QWORD *v269; // [rsp+90h] [rbp-1D0h]
  _QWORD *v270; // [rsp+90h] [rbp-1D0h]
  __int64 v271; // [rsp+90h] [rbp-1D0h]
  _QWORD *v272; // [rsp+90h] [rbp-1D0h]
  __int64 v273; // [rsp+90h] [rbp-1D0h]
  __int64 v274; // [rsp+90h] [rbp-1D0h]
  __int64 v275; // [rsp+90h] [rbp-1D0h]
  _QWORD *v276; // [rsp+90h] [rbp-1D0h]
  _QWORD *v277; // [rsp+98h] [rbp-1C8h]
  __int64 v278; // [rsp+98h] [rbp-1C8h]
  __int64 *v279; // [rsp+98h] [rbp-1C8h]
  __int64 v280; // [rsp+98h] [rbp-1C8h]
  __int64 v281; // [rsp+98h] [rbp-1C8h]
  __int64 v282; // [rsp+98h] [rbp-1C8h]
  _QWORD *v283; // [rsp+98h] [rbp-1C8h]
  _QWORD *v284; // [rsp+98h] [rbp-1C8h]
  _QWORD *v285; // [rsp+A0h] [rbp-1C0h]
  int v286; // [rsp+A8h] [rbp-1B8h]
  unsigned int v287; // [rsp+ACh] [rbp-1B4h]
  __int64 v288; // [rsp+B0h] [rbp-1B0h]
  __int64 v290; // [rsp+C0h] [rbp-1A0h]
  int v291; // [rsp+C8h] [rbp-198h]
  __int64 v292; // [rsp+C8h] [rbp-198h]
  __int64 v293; // [rsp+C8h] [rbp-198h]
  __int64 v294; // [rsp+C8h] [rbp-198h]
  __int64 *v295; // [rsp+C8h] [rbp-198h]
  __m128i *p_src; // [rsp+D0h] [rbp-190h] BYREF
  size_t n; // [rsp+D8h] [rbp-188h]
  __m128i src; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v299; // [rsp+F0h] [rbp-170h] BYREF
  _BYTE *v300; // [rsp+F8h] [rbp-168h]
  _QWORD *v301; // [rsp+100h] [rbp-160h]
  __int64 v302; // [rsp+108h] [rbp-158h]
  int v303; // [rsp+110h] [rbp-150h]
  _BYTE v304[40]; // [rsp+118h] [rbp-148h] BYREF
  __int64 *v305; // [rsp+140h] [rbp-120h] BYREF
  _BYTE *v306; // [rsp+148h] [rbp-118h]
  _BYTE *v307; // [rsp+150h] [rbp-110h]
  __int64 v308; // [rsp+158h] [rbp-108h]
  int v309; // [rsp+160h] [rbp-100h]
  _BYTE v310[40]; // [rsp+168h] [rbp-F8h] BYREF
  __int64 *v311; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v312; // [rsp+198h] [rbp-C8h]
  _WORD v313[32]; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 *v314; // [rsp+1E0h] [rbp-80h] BYREF
  __int64 v315; // [rsp+1E8h] [rbp-78h]
  __int64 v316; // [rsp+1F0h] [rbp-70h] BYREF
  __int64 v317; // [rsp+1F8h] [rbp-68h]

  *(_DWORD *)(a1 + 348) = 0;
  if ( (*(_BYTE *)(a2 + 18) & 8) == 0 )
  {
LABEL_2:
    v10 = *(_QWORD *)(a2 + 80);
    v11 = a2 + 72;
    if ( v10 != a2 + 72 )
    {
LABEL_3:
      v286 = 0;
      v287 = 0;
      v285 = 0;
      goto LABEL_4;
    }
    return j___libc_free_0(0);
  }
  v140 = sub_15E38F0(a2);
  v141 = sub_14DD7D0(v140);
  if ( v141 > 10 )
  {
    if ( v141 != 12 )
      goto LABEL_2;
  }
  else if ( v141 <= 6 )
  {
    v10 = *(_QWORD *)(a2 + 80);
    v11 = a2 + 72;
    if ( a2 + 72 != v10 )
      goto LABEL_3;
    return j___libc_free_0(0);
  }
  v11 = a2 + 72;
  sub_14DDFC0((__int64)&v314, a2);
  j___libc_free_0(0);
  v142 = v315;
  v314 = (__int64 *)((char *)v314 + 1);
  v315 = 0;
  v285 = (_QWORD *)v142;
  LODWORD(v142) = v316;
  v316 = 0;
  v286 = v142;
  LODWORD(v142) = v317;
  LODWORD(v317) = 0;
  v287 = v142;
  j___libc_free_0(0);
  v10 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v10 )
  {
    while ( 1 )
    {
LABEL_4:
      if ( !v10 )
LABEL_427:
        BUG();
      v12 = *(_QWORD *)(v10 + 24);
      if ( v12 != v10 + 16 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v10 == v11 )
        goto LABEL_7;
    }
    if ( v11 == v10 )
      goto LABEL_7;
    v288 = v11;
    while ( 2 )
    {
      for ( i = *(_QWORD *)(v12 + 8); ; i = *(_QWORD *)(v10 + 24) )
      {
        v19 = v10 - 24;
        if ( !v10 )
          v19 = 0;
        if ( i != v19 + 40 )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( v10 == v288 )
          break;
        if ( !v10 )
          goto LABEL_427;
      }
      v290 = v12 - 24;
      v20 = v12 - 24;
      v21 = *(_BYTE *)(v12 - 8);
      if ( v21 <= 0x17u )
      {
        v291 = 23;
        v22 = sub_1439C80(23);
        goto LABEL_31;
      }
      if ( v21 != 78 )
      {
        v291 = 2 * (v21 != 29) + 21;
        goto LABEL_30;
      }
      v34 = *(_QWORD *)(v12 - 48);
      if ( *(_BYTE *)(v34 + 16) )
      {
        v291 = 21;
        v22 = sub_1439C80(21);
        goto LABEL_31;
      }
      v291 = sub_1438F00(v34);
      switch ( v291 )
      {
        case 1:
          v68 = *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          while ( 2 )
          {
            v70 = sub_1649C60(v68);
            v69 = 23;
            v71 = v70;
            v72 = *(_BYTE *)(v70 + 16);
            if ( v72 <= 0x17u )
              goto LABEL_95;
            if ( v72 != 78 )
            {
              v69 = 2 * (v72 != 29) + 21;
LABEL_95:
              if ( (unsigned __int8)sub_1439C90(v69) )
                goto LABEL_96;
              break;
            }
            v69 = 21;
            if ( *(_BYTE *)(*(_QWORD *)(v71 - 24) + 16LL) )
              goto LABEL_95;
            v73 = sub_1438F00(*(_QWORD *)(v71 - 24));
            if ( (unsigned __int8)sub_1439C90(v73) )
            {
LABEL_96:
              v68 = *(_QWORD *)(v71 - 24LL * (*(_DWORD *)(v71 + 20) & 0xFFFFFFF));
              continue;
            }
            break;
          }
          v74 = *(_BYTE *)(v71 + 16);
          v75 = *(_QWORD *)(v12 + 16);
          if ( v74 > 0x17u )
          {
            if ( v74 == 78 )
            {
              v103 = v71 | 4;
              goto LABEL_163;
            }
            if ( v74 == 29 )
            {
              v103 = v71 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_163:
              v104 = v103 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v104 )
              {
                if ( v75 == *(_QWORD *)(v104 + 40) )
                {
                  for ( j = *(_QWORD *)(v104 + 32); ; j = *(_QWORD *)(j + 8) )
                  {
                    if ( !j )
                      goto LABEL_426;
                    v147 = *(_BYTE *)(j - 8);
                    v107 = j - 24;
                    if ( v147 != 71 )
                    {
                      if ( v147 != 56 )
                        break;
                      v275 = j;
                      v282 = j - 24;
                      v148 = sub_15FA1F0(j - 24);
                      v107 = v282;
                      j = v275;
                      if ( !v148 )
                        break;
                    }
                  }
LABEL_171:
                  if ( v290 == v107 )
                    goto LABEL_30;
                  v75 = *(_QWORD *)(v12 + 16);
                }
                else if ( *(_BYTE *)(v104 + 16) == 29 && v75 == *(_QWORD *)(v104 - 48) )
                {
                  for ( k = *(_QWORD *)(v75 + 48); k; k = *(_QWORD *)(k + 8) )
                  {
                    v106 = *(_BYTE *)(k - 8);
                    v107 = k - 24;
                    if ( v106 != 71 )
                    {
                      if ( v106 != 56 )
                        goto LABEL_171;
                      v274 = k;
                      v281 = k - 24;
                      v145 = sub_15FA1F0(k - 24);
                      v107 = v281;
                      k = v274;
                      if ( !v145 )
                        goto LABEL_171;
                    }
                  }
LABEL_426:
                  BUG();
                }
              }
            }
          }
          v76 = *(_QWORD **)(v75 + 48);
          if ( v76 == (_QWORD *)v12 )
            goto LABEL_110;
          v77 = (_QWORD *)(*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v76 != v77 )
          {
            while ( v77 )
            {
              v78 = *((_BYTE *)v77 - 8);
              if ( v78 != 71 )
              {
                if ( v78 != 56 )
                  goto LABEL_109;
                v270 = v76;
                v277 = v77;
                v84 = sub_15FA1F0((__int64)(v77 - 3));
                v77 = v277;
                v76 = v270;
                if ( !v84 )
                  goto LABEL_109;
              }
              v77 = (_QWORD *)(*v77 & 0xFFFFFFFFFFFFFFF8LL);
              if ( v76 == v77 )
                goto LABEL_120;
            }
            goto LABEL_426;
          }
LABEL_120:
          if ( !v77 )
            goto LABEL_426;
LABEL_109:
          if ( *((_BYTE *)v77 - 8) != 78 )
            goto LABEL_110;
          v150 = *(v77 - 6);
          v283 = v77;
          if ( *(_BYTE *)(v150 + 16) || (unsigned int)sub_1438F00(v150) != 6 )
            goto LABEL_110;
          v268 = v71;
          v151 = *((_DWORD *)v283 - 1);
          v276 = v283;
          v284 = v283 - 3;
          for ( m = v284[-3 * (v151 & 0xFFFFFFF)]; ; m = *(_QWORD *)(v157 - 24LL * (*(_DWORD *)(v157 + 20) & 0xFFFFFFF)) )
          {
            v156 = sub_1649C60(m);
            v153 = 23;
            v157 = v156;
            v158 = *(_BYTE *)(v156 + 16);
            if ( v158 > 0x17u )
            {
              if ( v158 != 78 )
              {
                v153 = 2 * (v158 != 29) + 21;
                goto LABEL_258;
              }
              v153 = 21;
              if ( !*(_BYTE *)(*(_QWORD *)(v157 - 24) + 16LL) )
                break;
            }
LABEL_258:
            if ( !(unsigned __int8)sub_1439C90(v153) )
              goto LABEL_264;
LABEL_259:
            ;
          }
          v159 = sub_1438F00(*(_QWORD *)(v157 - 24));
          if ( (unsigned __int8)sub_1439C90(v159) )
            goto LABEL_259;
LABEL_264:
          if ( v268 == v157 )
          {
            *(_BYTE *)(a1 + 153) = 1;
            v160 = v284[-3 * (*((_DWORD *)v276 - 1) & 0xFFFFFFF)];
            if ( *(v276 - 2) )
            {
              sub_164D160(
                (__int64)v284,
                v284[-3 * (*((_DWORD *)v276 - 1) & 0xFFFFFFF)],
                a3,
                a4,
                a5,
                a6,
                v154,
                v155,
                a9,
                a10);
              sub_15F20C0(v284);
            }
            else
            {
              sub_15F20C0(v284);
              sub_1AEB370(v160, 0);
            }
            v50 = -3LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF);
            v161 = *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
            if ( *(_QWORD *)(v12 - 16) )
            {
LABEL_69:
              sub_164D160(v290, *(_QWORD *)(v290 + 8 * v50), a3, a4, a5, a6, v35, v36, a9, a10);
              sub_15F20C0((_QWORD *)v290);
            }
            else
            {
              sub_15F20C0((_QWORD *)v290);
              sub_1AEB370(v161, 0);
            }
            goto LABEL_42;
          }
LABEL_110:
          *(_BYTE *)(a1 + 153) = 1;
          v79 = *(_QWORD **)(a1 + 256);
          if ( !v79 )
          {
            v143 = **(__int64 ***)(a1 + 232);
            v144 = (__int64 *)sub_1643330(v143);
            v311 = (__int64 *)sub_1646BA0(v144, 0);
            v280 = sub_1644EA0(v311, &v311, 1, 0);
            v314 = 0;
            v314 = (__int64 *)sub_1563AB0((__int64 *)&v314, v143, -1, 30);
            v79 = (_QWORD *)sub_1632080(*(_QWORD *)(a1 + 232), (__int64)"objc_retain", 11, v280, (__int64)v314);
            *(_QWORD *)(a1 + 256) = v79;
          }
          v80 = *(_QWORD *)(v12 - 48) == 0;
          *(_QWORD *)(v12 + 40) = *(_QWORD *)(*v79 + 24LL);
          if ( !v80 )
          {
            v81 = *(_QWORD *)(v12 - 40);
            v82 = *(_QWORD *)(v12 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v82 = v81;
            if ( v81 )
              *(_QWORD *)(v81 + 16) = *(_QWORD *)(v81 + 16) & 3LL | v82;
          }
          *(_QWORD *)(v12 - 48) = v79;
          v83 = v79[1];
          *(_QWORD *)(v12 - 40) = v83;
          if ( v83 )
            *(_QWORD *)(v83 + 16) = (v12 - 40) | *(_QWORD *)(v83 + 16) & 3LL;
          *(_QWORD *)(v12 - 32) = *(_QWORD *)(v12 - 32) & 3LL | (unsigned __int64)(v79 + 1);
          v79[1] = v12 - 48;
LABEL_30:
          v22 = sub_1439C80(v291);
LABEL_31:
          if ( !v22
            || *(_QWORD *)(v12 - 16)
            || (v116 = *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF)),
                v117 = *(unsigned __int8 *)(v116 + 16),
                (unsigned int)(v117 - 9) <= 7) )
          {
LABEL_33:
            v23 = sub_1439CD0(v291);
            goto LABEL_34;
          }
          v118 = *(_QWORD *)(v116 + 8);
          if ( !v118 )
          {
LABEL_199:
            if ( !(unsigned __int8)sub_18CE260(v116) )
              goto LABEL_33;
            v121 = v116;
            v122 = *(_QWORD *)(v116 + 8);
            if ( !v122 )
              goto LABEL_214;
            v272 = (_QWORD *)v12;
            v278 = v20;
            v123 = v121;
            while ( 2 )
            {
              v124 = sub_1648700(v122);
              if ( v124[1] )
              {
LABEL_273:
                v20 = v278;
                goto LABEL_33;
              }
LABEL_207:
              v126 = sub_1649C60((__int64)v124);
              v125 = 23;
              v127 = v126;
              v128 = *(_BYTE *)(v126 + 16);
              if ( v128 > 0x17u )
              {
                if ( v128 == 78 )
                {
                  v125 = 21;
                  if ( !*(_BYTE *)(*(_QWORD *)(v127 - 24) + 16LL) )
                  {
                    v129 = sub_1438F00(*(_QWORD *)(v127 - 24));
                    if ( !(unsigned __int8)sub_1439C90(v129) )
                    {
LABEL_211:
                      if ( v127 != v123 )
                        goto LABEL_273;
                      v122 = *(_QWORD *)(v122 + 8);
                      if ( !v122 )
                      {
                        v12 = (unsigned __int64)v272;
                        goto LABEL_214;
                      }
                      continue;
                    }
                    goto LABEL_206;
                  }
                }
                else
                {
                  v125 = 2 * (v128 != 29) + 21;
                }
              }
              break;
            }
            if ( !(unsigned __int8)sub_1439C90(v125) )
              goto LABEL_211;
LABEL_206:
            v124 = *(_QWORD **)(v127 - 24LL * (*(_DWORD *)(v127 + 20) & 0xFFFFFFF));
            goto LABEL_207;
          }
          while ( 2 )
          {
            if ( *(_QWORD *)(v118 + 8) )
              goto LABEL_199;
            if ( (unsigned __int8)v117 <= 0x17u )
              goto LABEL_192;
            if ( (_BYTE)v117 == 71 )
            {
              v116 = *(_QWORD *)(v116 - 24);
LABEL_197:
              v117 = *(unsigned __int8 *)(v116 + 16);
              if ( (unsigned int)(v117 - 9) <= 7 )
                goto LABEL_33;
              v118 = *(_QWORD *)(v116 + 8);
              if ( !v118 )
                goto LABEL_199;
              continue;
            }
            break;
          }
          if ( (_BYTE)v117 == 56 && (unsigned __int8)sub_15FA1F0(v116) )
            goto LABEL_196;
LABEL_192:
          v119 = *(_BYTE *)(v116 + 16);
          v120 = 23;
          if ( v119 > 0x17u )
          {
            if ( v119 == 78 )
            {
              v120 = 21;
              if ( !*(_BYTE *)(*(_QWORD *)(v116 - 24) + 16LL) )
                v120 = sub_1438F00(*(_QWORD *)(v116 - 24));
            }
            else
            {
              v120 = 2 * (v119 != 29) + 21;
            }
          }
          if ( (unsigned __int8)sub_1439C90(v120) )
          {
LABEL_196:
            v116 = *(_QWORD *)(v116 - 24LL * (*(_DWORD *)(v116 + 20) & 0xFFFFFFF));
            goto LABEL_197;
          }
          if ( !(unsigned __int8)sub_18CE260(v116) )
            goto LABEL_33;
LABEL_214:
          *(_BYTE *)(a1 + 153) = 1;
          v130 = sub_16498A0(v290);
          v131 = *(_QWORD *)(a1 + 248);
          v279 = (__int64 *)v130;
          if ( !v131 )
          {
            v162 = **(__int64 ***)(a1 + 232);
            v163 = (__int64 *)sub_1643330(v162);
            v311 = (__int64 *)sub_1646BA0(v163, 0);
            v314 = 0;
            v294 = sub_1563AB0((__int64 *)&v314, v162, -1, 30);
            v164 = (__int64 *)sub_1643270(v162);
            v165 = sub_1644EA0(v164, &v311, 1, 0);
            v166 = sub_1632080(*(_QWORD *)(a1 + 232), (__int64)"objc_release", 12, v165, v294);
            *(_QWORD *)(a1 + 248) = v166;
            v131 = v166;
          }
          LOWORD(v316) = 257;
          v311 = *(__int64 **)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          v292 = *(_QWORD *)(*(_QWORD *)v131 + 24LL);
          v132 = sub_1648AB0(72, 2u, 0);
          v133 = v292;
          v20 = (__int64)v132;
          if ( v132 )
          {
            v134 = *(__int64 **)(v292 + 16);
            v293 = (__int64)v132;
            v273 = v133;
            sub_15F1EA0((__int64)v132, *v134, 54, (__int64)(v132 - 6), 2, v290);
            *(_QWORD *)(v20 + 56) = 0;
            sub_15F5B40(v20, v273, v131, (__int64 *)&v311, 1, (__int64)&v314, 0, 0);
          }
          else
          {
            v293 = 0;
          }
          v135 = sub_1627350(v279, 0, 0, 0, 1);
          if ( *(_BYTE *)(a1 + 324) )
          {
            v136 = *(_DWORD *)(a1 + 320);
          }
          else
          {
            v136 = sub_1602B80(**(__int64 ***)(a1 + 312), "clang.imprecise_release", 0x17u);
            if ( *(_BYTE *)(a1 + 324) )
            {
              *(_DWORD *)(a1 + 320) = v136;
            }
            else
            {
              *(_DWORD *)(a1 + 320) = v136;
              *(_BYTE *)(a1 + 324) = 1;
            }
          }
          sub_1625C10(v293, v136, v135);
          v139 = *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          if ( *(_QWORD *)(v12 - 16) )
          {
            sub_164D160(
              v290,
              *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF)),
              a3,
              a4,
              a5,
              a6,
              v137,
              v138,
              a9,
              a10);
            sub_15F20C0((_QWORD *)v290);
          }
          else
          {
            sub_15F20C0((_QWORD *)v290);
            sub_1AEB370(v139, 0);
          }
          v291 = 4;
          v23 = sub_1439CD0(4);
LABEL_34:
          if ( v23 )
          {
            *(_BYTE *)(a1 + 153) = 1;
            *(_WORD *)(v20 + 18) = *(_WORD *)(v20 + 18) & 0xFFFC | 1;
          }
          if ( sub_1439CF0(v291) )
          {
            *(_BYTE *)(a1 + 153) = 1;
            *(_WORD *)(v20 + 18) &= 0xFFFCu;
          }
          if ( sub_1439D00(v291) )
          {
            *(_BYTE *)(a1 + 153) = 1;
            v314 = *(__int64 **)(v20 + 56);
            v33 = (__int64 *)sub_16498A0(v20);
            v314 = (__int64 *)sub_1563AB0((__int64 *)&v314, v33, -1, 30);
            *(_QWORD *)(v20 + 56) = v314;
          }
          if ( !sub_1439CC0(v291) )
          {
            *(_DWORD *)(a1 + 348) |= 1 << v291;
            goto LABEL_42;
          }
          v24 = *(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
          while ( 2 )
          {
            v28 = sub_1649C60(v24);
            v25 = 23;
            v29 = v28;
            v30 = *(_BYTE *)(v28 + 16);
            if ( v30 <= 0x17u )
              goto LABEL_47;
            if ( v30 != 78 )
            {
              v25 = 2 * (v30 != 29) + 21;
LABEL_47:
              if ( !(unsigned __int8)sub_1439C90(v25) )
                goto LABEL_53;
LABEL_48:
              v24 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
              continue;
            }
            break;
          }
          v25 = 21;
          if ( *(_BYTE *)(*(_QWORD *)(v29 - 24) + 16LL) )
            goto LABEL_47;
          v31 = sub_1438F00(*(_QWORD *)(v29 - 24));
          if ( (unsigned __int8)sub_1439C90(v31) )
            goto LABEL_48;
LABEL_53:
          if ( *(_BYTE *)(v29 + 16) == 15 || *(_BYTE *)(v29 + 16) == 9 )
          {
            *(_BYTE *)(a1 + 153) = 1;
            v32 = *(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
            if ( *(_QWORD *)(v20 + 8) )
            {
              sub_164D160(
                v20,
                *(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)),
                a3,
                a4,
                a5,
                a6,
                v26,
                v27,
                a9,
                a10);
              sub_15F20C0((_QWORD *)v20);
            }
            else
            {
              sub_15F20C0((_QWORD *)v20);
              sub_1AEB370(v32, 0);
            }
            goto LABEL_42;
          }
          *(_DWORD *)(a1 + 348) |= 1 << v291;
          if ( v291 != 4
            || (!*(_BYTE *)(a1 + 324)
              ? ((v149 = sub_1602B80(**(__int64 ***)(a1 + 312), "clang.imprecise_release", 0x17u),
                  v86 = v149,
                  !*(_BYTE *)(a1 + 324))
               ? (*(_DWORD *)(a1 + 320) = v149, *(_BYTE *)(a1 + 324) = 1)
               : (*(_DWORD *)(a1 + 320) = v149))
              : (v86 = *(_DWORD *)(a1 + 320)),
                (*(_QWORD *)(v20 + 48) || *(__int16 *)(v20 + 18) < 0) && sub_1625790(v20, v86)) )
          {
            v87 = &v316;
            v316 = v20;
            v315 = 0x400000001LL;
            v88 = 1;
            v314 = &v316;
            v317 = v29;
            v271 = i;
            v267 = v10;
            while ( 1 )
            {
              while ( 1 )
              {
                v89 = v88--;
                v90 = &v87[2 * v89 - 2];
                v10 = v90[1];
                v91 = *v90;
                LODWORD(v315) = v88;
                v290 = v91;
                if ( *(_BYTE *)(v10 + 16) == 77 && (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 0 )
                  break;
                if ( !v88 )
                  goto LABEL_158;
              }
              v12 = 0;
              v92 = 0;
              v20 = 8LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
              do
              {
                if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
                  v93 = *(_QWORD *)(v10 - 8);
                else
                  v93 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
                for ( ii = *(_QWORD *)(v93 + 3 * v92); ; ii = *(_QWORD *)(i - 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF)) )
                {
                  v97 = sub_1649C60(ii);
                  v95 = 23;
                  i = v97;
                  v98 = *(_BYTE *)(v97 + 16);
                  if ( v98 > 0x17u )
                  {
                    if ( v98 != 78 )
                    {
                      v95 = 2 * (v98 != 29) + 21;
                      goto LABEL_140;
                    }
                    v95 = 21;
                    if ( !*(_BYTE *)(*(_QWORD *)(i - 24) + 16LL) )
                      break;
                  }
LABEL_140:
                  v96 = sub_1439C90(v95);
                  if ( !(_BYTE)v96 )
                    goto LABEL_146;
LABEL_141:
                  ;
                }
                v99 = sub_1438F00(*(_QWORD *)(i - 24));
                v96 = sub_1439C90(v99);
                if ( (_BYTE)v96 )
                  goto LABEL_141;
LABEL_146:
                LOBYTE(v96) = *(_BYTE *)(i + 16) == 9 || *(_BYTE *)(i + 16) == 15;
                if ( (_BYTE)v96 )
                {
                  v12 = v96;
                  goto LABEL_148;
                }
                if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
                  v101 = *(_QWORD *)(v10 - 8);
                else
                  v101 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
                v102 = (*(_QWORD *)(*(_QWORD *)(v92 + v101 + 24LL * *(unsigned int *)(v10 + 56) + 8) + 40LL)
                      & 0xFFFFFFFFFFFFFFF8LL)
                     - 24;
                if ( (*(_QWORD *)(*(_QWORD *)(v92 + v101 + 24LL * *(unsigned int *)(v10 + 56) + 8) + 40LL)
                    & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  v102 = 0;
                if ( (unsigned int)sub_15F4D60(v102) != 1 )
                {
LABEL_156:
                  v88 = v315;
                  v87 = v314;
                  goto LABEL_157;
                }
LABEL_148:
                v92 += 8;
              }
              while ( v20 != v92 );
              if ( !(_BYTE)v12 )
                goto LABEL_156;
              v299 = 0;
              v300 = v304;
              v301 = v304;
              v100 = v310;
              v302 = 4;
              v306 = v310;
              v303 = 0;
              v305 = 0;
              v307 = v310;
              v308 = 4;
              v309 = 0;
              switch ( (unsigned int)v310 )
              {
                case 0u:
                case 3u:
                  goto LABEL_182;
                case 1u:
                case 2u:
                case 6u:
                  v115 = v310;
                  goto LABEL_235;
                case 4u:
                  sub_18DCE30(0, v10, *(_QWORD *)(v91 + 40), v91, (unsigned int)&v299, (unsigned int)&v305, a1 + 160);
                  goto LABEL_174;
                case 5u:
                  sub_18DCE30(1, v10, *(_QWORD *)(v91 + 40), v91, (unsigned int)&v299, (unsigned int)&v305, a1 + 160);
LABEL_174:
                  v110 = HIDWORD(v302);
                  if ( HIDWORD(v302) - v303 != 1 )
                    goto LABEL_182;
                  v111 = v301;
                  if ( v301 != (_QWORD *)v300 )
                    v110 = (unsigned int)v302;
                  v112 = &v301[v110];
                  v113 = *v301;
                  if ( v301 == v112 )
                    goto LABEL_181;
                  break;
                default:
                  goto LABEL_30;
              }
              while ( 1 )
              {
                v113 = *v111;
                v114 = v111;
                if ( *v111 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v112 == ++v111 )
                {
                  v113 = v114[1];
                  break;
                }
              }
LABEL_181:
              if ( v113 != v10 )
              {
LABEL_182:
                v115 = v307;
                if ( v307 != v306 )
                  goto LABEL_183;
                goto LABEL_184;
              }
              v266 = 0;
              *(_BYTE *)(a1 + 153) = 1;
              v173 = *(_DWORD *)(v290 + 20);
              v260 = **(_QWORD **)(v290 - 24LL * (v173 & 0xFFFFFFF));
              v259 = 8LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
              if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 0 )
              {
                v265 = v10;
                while ( 1 )
                {
                  if ( (*(_BYTE *)(v265 + 23) & 0x40) != 0 )
                    v174 = *(_QWORD *)(v265 - 8);
                  else
                    v174 = v265 - 24LL * (*(_DWORD *)(v265 + 20) & 0xFFFFFFF);
                  for ( jj = *(_QWORD *)(v174 + 3 * v266);
                        ;
                        jj = *(_QWORD *)(v178 - 24LL * (*(_DWORD *)(v178 + 20) & 0xFFFFFFF)) )
                  {
                    v177 = sub_1649C60(jj);
                    v176 = 23;
                    v178 = v177;
                    v179 = *(_BYTE *)(v177 + 16);
                    if ( v179 > 0x17u )
                    {
                      if ( v179 == 78 )
                      {
                        v176 = 21;
                        if ( !*(_BYTE *)(*(_QWORD *)(v178 - 24) + 16LL) )
                          v176 = sub_1438F00(*(_QWORD *)(v178 - 24));
                      }
                      else
                      {
                        v176 = 2 * (v179 != 29) + 21;
                      }
                    }
                    if ( !(unsigned __int8)sub_1439C90(v176) )
                      break;
                  }
                  v180 = *(_BYTE *)(v178 + 16);
                  if ( v180 == 15 || v180 == 9 )
                    goto LABEL_362;
                  if ( (*(_BYTE *)(v265 + 23) & 0x40) != 0 )
                    v181 = *(_QWORD *)(v265 - 8);
                  else
                    v181 = v265 - 24LL * (*(_DWORD *)(v265 + 20) & 0xFFFFFFF);
                  v264 = *(_QWORD **)(v181 + 3 * v266);
                  v182 = *(_QWORD *)(*(_QWORD *)(v266 + v181 + 24LL * *(unsigned int *)(v265 + 56) + 8) + 40LL)
                       & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v182 )
                    BUG();
                  v261 = v182 - 24;
                  v262 = *(_QWORD *)(v182 + 16);
                  v311 = (__int64 *)v313;
                  v312 = 0x100000000LL;
                  if ( *(char *)(v290 + 23) >= 0 )
                    goto LABEL_337;
                  v183 = sub_1648A40(v290);
                  v185 = v183 + v184;
                  if ( *(char *)(v290 + 23) >= 0 )
                    v186 = v185 >> 4;
                  else
                    LODWORD(v186) = (v185 - sub_1648A40(v290)) >> 4;
                  v263 = 0;
                  v258 = 16LL * (unsigned int)v186;
                  if ( !(_DWORD)v186 )
                    goto LABEL_337;
                  v254 = v178;
                  do
                  {
                    v187 = 0;
                    if ( *(char *)(v290 + 23) < 0 )
                      v187 = sub_1648A40(v290);
                    v188 = (unsigned int *)(v263 + v187);
                    v189 = *(__int64 **)v188;
                    v190 = v188[2];
                    v191 = v188[3];
                    v192 = *(_DWORD *)(v290 + 20) & 0xFFFFFFF;
                    if ( *(_DWORD *)(*(_QWORD *)v188 + 8LL) != 1 )
                    {
                      v193 = v312;
                      if ( (unsigned int)v312 >= HIDWORD(v312) )
                      {
                        sub_1740340((__int64)&v311, 0);
                        v193 = v312;
                      }
                      v194 = &v311[7 * v193];
                      if ( v194 )
                      {
                        *((_BYTE *)v194 + 16) = 0;
                        *v194 = (__int64)(v194 + 2);
                        v194[1] = 0;
                        v194[4] = 0;
                        v194[5] = 0;
                        v194[6] = 0;
                        v195 = *v189;
                        p_src = &src;
                        sub_18CD230((__int64 *)&p_src, (_BYTE *)v189 + 16, (__int64)v189 + v195 + 16);
                        v196 = (__m128i *)*v194;
                        if ( p_src == &src )
                        {
                          v249 = n;
                          if ( n )
                          {
                            if ( n == 1 )
                              v196->m128i_i8[0] = src.m128i_i8[0];
                            else
                              memcpy(v196, &src, n);
                            v249 = n;
                            v196 = (__m128i *)*v194;
                          }
                          v194[1] = v249;
                          v196->m128i_i8[v249] = 0;
                          v196 = p_src;
                        }
                        else
                        {
                          if ( v194 + 2 == (__int64 *)v196 )
                          {
                            *v194 = (__int64)p_src;
                            v194[1] = n;
                            v194[2] = src.m128i_i64[0];
                          }
                          else
                          {
                            *v194 = (__int64)p_src;
                            v197 = v194[2];
                            v194[1] = n;
                            v194[2] = src.m128i_i64[0];
                            if ( v196 )
                            {
                              p_src = v196;
                              src.m128i_i64[0] = v197;
                              goto LABEL_325;
                            }
                          }
                          p_src = &src;
                          v196 = &src;
                        }
LABEL_325:
                        n = 0;
                        v196->m128i_i8[0] = 0;
                        if ( p_src != &src )
                          j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
                        v198 = 3 * v190;
                        v199 = (char *)v194[5];
                        v198 *= 8;
                        v200 = (char *)(v290 + v198 - 24 * v192);
                        v201 = 24 * v191 - v198;
                        v202 = &v200[v201];
                        if ( v200 != &v200[v201] )
                        {
                          v203 = 0xAAAAAAAAAAAAAAABLL * (v201 >> 3);
                          if ( v203 <= (v194[6] - (__int64)v199) >> 3 )
                          {
                            do
                            {
                              if ( v199 )
                                *(_QWORD *)v199 = *(_QWORD *)v200;
                              v200 += 24;
                              v199 += 8;
                            }
                            while ( v202 != v200 );
                            v194[5] += 8 * v203;
                            goto LABEL_333;
                          }
                          v233 = (char *)v194[4];
                          v234 = v199 - v233;
                          v235 = (v199 - v233) >> 3;
                          if ( v203 > 0xFFFFFFFFFFFFFFFLL - v235 )
                            sub_4262D8((__int64)"vector::_M_range_insert");
                          if ( v203 < v235 )
                            v203 = (v199 - v233) >> 3;
                          v236 = __CFADD__(v203, v235);
                          v237 = v203 + v235;
                          if ( v236 )
                          {
                            v238 = 0x7FFFFFFFFFFFFFF8LL;
                            goto LABEL_391;
                          }
                          if ( v237 )
                          {
                            if ( v237 > 0xFFFFFFFFFFFFFFFLL )
                              v237 = 0xFFFFFFFFFFFFFFFLL;
                            v238 = 8 * v237;
LABEL_391:
                            v255 = v200;
                            v239 = sub_22077B0(v238);
                            v233 = (char *)v194[4];
                            v200 = v255;
                            v240 = (char *)v239;
                            v241 = v239 + v238;
                            v234 = v199 - v233;
                          }
                          else
                          {
                            v241 = 0;
                            v240 = 0;
                          }
                          if ( v233 != v199 )
                          {
                            v252 = v200;
                            v253 = v234;
                            v256 = v233;
                            memmove(v240, v233, v234);
                            v200 = v252;
                            v234 = v253;
                            v233 = v256;
                          }
                          v242 = &v240[v234];
                          v243 = v200;
                          v244 = v242;
                          do
                          {
                            if ( v244 )
                              *(_QWORD *)v244 = *(_QWORD *)v243;
                            v243 += 24;
                            v244 += 8;
                          }
                          while ( v202 != v243 );
                          v245 = &v242[0x5555555555555558LL * ((unsigned __int64)(v202 - v200 - 24) >> 3) + 8];
                          v246 = v194[5] - (_QWORD)v199;
                          if ( v199 != (char *)v194[5] )
                          {
                            v257 = v233;
                            v247 = (char *)memcpy(v245, v199, v194[5] - (_QWORD)v199);
                            v233 = v257;
                            v245 = v247;
                          }
                          v248 = &v245[v246];
                          if ( v233 )
                            j_j___libc_free_0(v233, v194[6] - (_QWORD)v233);
                          v194[4] = (__int64)v240;
                          v194[5] = (__int64)v248;
                          v194[6] = v241;
                        }
LABEL_333:
                        v193 = v312;
                      }
                      LODWORD(v312) = v193 + 1;
                    }
                    v263 += 16;
                  }
                  while ( v263 != v258 );
                  v178 = v254;
LABEL_337:
                  if ( !v286 )
                    goto LABEL_338;
                  v222 = &v285[2 * v287];
                  if ( v287 )
                  {
                    v223 = (v287 - 1) & (((unsigned int)v262 >> 9) ^ ((unsigned int)v262 >> 4));
                    v222 = &v285[2 * v223];
                    v224 = *v222;
                    if ( v262 != *v222 )
                    {
                      v250 = 1;
                      while ( v224 != -8 )
                      {
                        v251 = v250 + 1;
                        v223 = (v287 - 1) & (v250 + v223);
                        v222 = &v285[2 * v223];
                        v224 = *v222;
                        if ( v262 == *v222 )
                          goto LABEL_370;
                        v250 = v251;
                      }
                      v222 = &v285[2 * v287];
                    }
                  }
LABEL_370:
                  v225 = v222[1];
                  v226 = (unsigned __int64 **)(v225 & 0xFFFFFFFFFFFFFFF8LL);
                  v227 = v225 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (v225 & 4) != 0 || !v226 )
                    v227 = **v226;
                  v228 = sub_157ED20(v227);
                  v229 = (unsigned int)*(unsigned __int8 *)(v228 + 16) - 34;
                  if ( (unsigned int)v229 <= 0x36 && (v230 = 0x40018000000001LL, _bittest64(&v230, v229)) )
                  {
                    if ( (unsigned int)v312 >= HIDWORD(v312) )
                      sub_1740340((__int64)&v311, 0);
                    p_src = &src;
                    sub_18CD230((__int64 *)&p_src, "funclet", (__int64)"");
                    v231 = (__m128 *)&v311[7 * (unsigned int)v312];
                    if ( v231 )
                    {
                      v231->m128_u64[0] = (unsigned __int64)&v231[1];
                      if ( p_src == &src )
                      {
                        a3 = (__m128)_mm_load_si128(&src);
                        v231[1] = a3;
                      }
                      else
                      {
                        v231->m128_u64[0] = (unsigned __int64)p_src;
                        v231[1].m128_u64[0] = src.m128i_i64[0];
                      }
                      v231->m128_u64[1] = n;
                      p_src = &src;
                      n = 0;
                      src.m128i_i8[0] = 0;
                      v231[2].m128_u64[0] = 0;
                      v231[2].m128_u64[1] = 0;
                      v231[3].m128_u64[0] = 0;
                      v232 = (__int64 *)sub_22077B0(8);
                      v231[2].m128_u64[0] = (unsigned __int64)v232;
                      v231[3].m128_u64[0] = (unsigned __int64)(v232 + 1);
                      *v232 = v228;
                      v231[2].m128_u64[1] = (unsigned __int64)(v232 + 1);
                    }
                    if ( p_src != &src )
                      j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
                    v204 = v312 + 1;
                    LODWORD(v312) = v312 + 1;
                  }
                  else
                  {
LABEL_338:
                    v204 = v312;
                  }
                  v205 = sub_15F60C0(v290, v311, v204, 0);
                  v206 = v311;
                  v207 = v205;
                  v208 = &v311[7 * (unsigned int)v312];
                  if ( v311 != v208 )
                  {
                    do
                    {
                      v209 = *(v208 - 3);
                      v208 -= 7;
                      if ( v209 )
                        j_j___libc_free_0(v209, v208[6] - v209);
                      if ( (__int64 *)*v208 != v208 + 2 )
                        j_j___libc_free_0(*v208, v208[2] + 1);
                    }
                    while ( v206 != v208 );
                    v208 = v311;
                  }
                  if ( v208 != (__int64 *)v313 )
                    _libc_free((unsigned __int64)v208);
                  if ( v260 != *v264 )
                  {
                    v313[0] = 257;
                    v210 = sub_1648A60(56, 1u);
                    v211 = v210;
                    if ( v210 )
                      sub_15FD590((__int64)v210, (__int64)v264, v260, (__int64)&v311, v261);
                    v264 = v211;
                  }
                  v212 = (_QWORD *)(v207 - 24LL * (*(_DWORD *)(v207 + 20) & 0xFFFFFFF));
                  if ( *v212 )
                  {
                    v213 = v212[1];
                    v214 = v212[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v214 = v213;
                    if ( v213 )
                      *(_QWORD *)(v213 + 16) = v214 | *(_QWORD *)(v213 + 16) & 3LL;
                  }
                  *v212 = v264;
                  if ( v264 )
                  {
                    v215 = v264[1];
                    v212[1] = v215;
                    if ( v215 )
                      *(_QWORD *)(v215 + 16) = (unsigned __int64)(v212 + 1) | *(_QWORD *)(v215 + 16) & 3LL;
                    v212[2] = (unsigned __int64)(v264 + 1) | v212[2] & 3LL;
                    v264[1] = v212;
                  }
                  sub_15F2120(v207, v261);
                  v218 = (unsigned int)v315;
                  if ( (unsigned int)v315 >= HIDWORD(v315) )
                  {
                    sub_16CD150((__int64)&v314, &v316, 0, 16, v216, v217);
                    v218 = (unsigned int)v315;
                  }
                  v219 = &v314[2 * v218];
                  *v219 = v207;
                  v219[1] = v178;
                  LODWORD(v315) = v315 + 1;
LABEL_362:
                  v266 += 8;
                  if ( v259 == v266 )
                  {
                    v173 = *(_DWORD *)(v290 + 20);
                    break;
                  }
                }
              }
              v220 = -3LL * (v173 & 0xFFFFFFF);
              v221 = *(_QWORD *)(v290 + 8 * v220);
              if ( *(_QWORD *)(v290 + 8) )
              {
                sub_164D160(v290, *(_QWORD *)(v290 + 8 * v220), a3, a4, a5, a6, v108, v109, a9, a10);
                sub_15F20C0((_QWORD *)v290);
              }
              else
              {
                sub_15F20C0((_QWORD *)v290);
                sub_1AEB370(v221, 0);
              }
              v115 = v307;
              v100 = v306;
LABEL_235:
              if ( v115 != v100 )
LABEL_183:
                _libc_free((unsigned __int64)v115);
LABEL_184:
              if ( v301 == (_QWORD *)v300 )
                goto LABEL_156;
              _libc_free((unsigned __int64)v301);
              v88 = v315;
              v87 = v314;
LABEL_157:
              if ( !v88 )
              {
LABEL_158:
                i = v271;
                v10 = v267;
                if ( v87 != &v316 )
                  _libc_free((unsigned __int64)v87);
                break;
              }
            }
          }
LABEL_42:
          if ( v10 == v288 )
            break;
          v12 = i;
          continue;
        case 6:
          v52 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF) - 24);
          while ( 2 )
          {
            v54 = sub_1649C60(v52);
            v53 = 23;
            v55 = v54;
            v56 = *(_BYTE *)(v54 + 16);
            if ( v56 <= 0x17u )
              break;
            if ( v56 == 78 )
            {
              v53 = 21;
              if ( !*(_BYTE *)(*(_QWORD *)(v55 - 24) + 16LL) )
              {
                v57 = sub_1438F00(*(_QWORD *)(v55 - 24));
                if ( !(unsigned __int8)sub_1439C90(v57) )
                {
LABEL_78:
                  if ( (unsigned int)*(unsigned __int8 *)(v55 + 16) - 9 <= 7 )
                    goto LABEL_30;
                  v58 = &v316;
                  v316 = v55;
                  v315 = 0x200000001LL;
                  v59 = 1;
                  v314 = &v316;
                  if ( *(_BYTE *)(v55 + 16) == 77 )
                  {
                    sub_18CE920(v55, (__int64)&v314);
                    v269 = (_QWORD *)(v12 - 24);
                    v58 = v314;
                    v59 = v315;
                  }
                  else
                  {
                    v269 = (_QWORD *)(v12 - 24);
                  }
                  while ( 1 )
                  {
                    v60 = v59--;
                    v61 = v58[v60 - 1];
                    LODWORD(v315) = v59;
                    v62 = *(_QWORD *)(v61 + 8);
                    if ( !v62 )
                      goto LABEL_91;
                    do
                    {
                      v63 = sub_1648700(v62);
                      v66 = *((_BYTE *)v63 + 16);
                      if ( v66 <= 0x17u )
                        goto LABEL_89;
                      if ( v66 == 25 )
                        goto LABEL_124;
                      if ( v66 == 78 )
                      {
                        v85 = *(v63 - 3);
                        if ( *(_BYTE *)(v85 + 16) )
                          goto LABEL_89;
                        if ( (unsigned int)sub_1438F00(v85) == 1 )
                        {
LABEL_124:
                          v20 = (__int64)v269;
                          if ( v314 != &v316 )
                            _libc_free((unsigned __int64)v314);
                          goto LABEL_30;
                        }
                        v66 = *((_BYTE *)v63 + 16);
                      }
                      if ( v66 == 71 )
                      {
                        v67 = (unsigned int)v315;
                        if ( (unsigned int)v315 >= HIDWORD(v315) )
                        {
                          sub_16CD150((__int64)&v314, &v316, 0, 8, v64, v65);
                          v67 = (unsigned int)v315;
                        }
                        v314[v67] = (__int64)v63;
                        LODWORD(v315) = v315 + 1;
                      }
LABEL_89:
                      v62 = *(_QWORD *)(v62 + 8);
                    }
                    while ( v62 );
                    v59 = v315;
LABEL_91:
                    if ( !v59 )
                    {
                      v20 = (__int64)v269;
                      *(_BYTE *)(a1 + 153) = 1;
                      v167 = *(_QWORD **)(a1 + 272);
                      if ( !v167 )
                      {
                        v295 = **(__int64 ***)(a1 + 232);
                        v171 = (__int64 *)sub_1643330(v295);
                        v305 = (__int64 *)sub_1646BA0(v171, 0);
                        v172 = sub_1644EA0(v305, &v305, 1, 0);
                        v311 = 0;
                        v311 = (__int64 *)sub_1563AB0((__int64 *)&v311, v295, -1, 30);
                        v167 = (_QWORD *)sub_1632080(
                                           *(_QWORD *)(a1 + 232),
                                           (__int64)"objc_autorelease",
                                           16,
                                           v172,
                                           (__int64)v311);
                        *(_QWORD *)(a1 + 272) = v167;
                      }
                      v80 = *(_QWORD *)(v12 - 48) == 0;
                      *(_QWORD *)(v12 + 40) = *(_QWORD *)(*v167 + 24LL);
                      if ( !v80 )
                      {
                        v168 = *(_QWORD *)(v12 - 40);
                        v169 = *(_QWORD *)(v12 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v169 = v168;
                        if ( v168 )
                          *(_QWORD *)(v168 + 16) = v169 | *(_QWORD *)(v168 + 16) & 3LL;
                      }
                      *(_QWORD *)(v12 - 48) = v167;
                      v170 = v167[1];
                      *(_QWORD *)(v12 - 40) = v170;
                      if ( v170 )
                        *(_QWORD *)(v170 + 16) = (v12 - 40) | *(_QWORD *)(v170 + 16) & 3LL;
                      *(_QWORD *)(v12 - 32) = *(_QWORD *)(v12 - 32) & 3LL | (unsigned __int64)(v167 + 1);
                      v167[1] = v12 - 48;
                      *(_WORD *)(v12 - 6) &= 0xFFFCu;
                      if ( v314 != &v316 )
                        _libc_free((unsigned __int64)v314);
                      v291 = 5;
                      v22 = sub_1439C80(5);
                      goto LABEL_31;
                    }
                    v58 = v314;
                  }
                }
LABEL_73:
                v52 = *(_QWORD *)(v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF));
                continue;
              }
            }
            else
            {
              v53 = 2 * (v56 != 29) + 21;
            }
            break;
          }
          if ( !(unsigned __int8)sub_1439C90(v53) )
            goto LABEL_78;
          goto LABEL_73;
        case 9:
          *(_BYTE *)(a1 + 153) = 1;
          v50 = -3LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF);
          v51 = *(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          if ( *(_QWORD *)(v12 - 16) )
            goto LABEL_69;
          sub_15F20C0((_QWORD *)v290);
          sub_1AEB370(v51, 0);
          goto LABEL_42;
        case 12:
        case 13:
        case 14:
        case 15:
        case 18:
          v37 = *(_BYTE *)(*(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF)) + 16LL);
          if ( v37 == 9 || v37 == 15 )
            goto LABEL_60;
          goto LABEL_30;
        case 16:
        case 17:
          v48 = *(_BYTE *)(*(_QWORD *)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF)) + 16LL);
          if ( v48 != 15 && v48 != 9 )
          {
            v49 = *(_BYTE *)(*(_QWORD *)(v290 + 24 * (1LL - (*(_DWORD *)(v12 - 4) & 0xFFFFFFF))) + 16LL);
            if ( v49 != 15 && v49 != 9 )
              goto LABEL_30;
          }
LABEL_60:
          *(_BYTE *)(a1 + 153) = 1;
          v38 = *(__int64 ****)(v290 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF));
          v39 = *v38;
          v40 = sub_1599EF0((__int64 **)(*v38)[3]);
          v43 = sub_15A06D0(v39, v290, v41, v42);
          v44 = sub_1648A60(64, 2u);
          if ( v44 )
            sub_15F9660((__int64)v44, v40, v43, v290);
          v45 = sub_1599EF0(*(__int64 ***)(v12 - 24));
          sub_164D160(v290, v45, a3, a4, a5, a6, v46, v47, a9, a10);
          sub_15F20C0((_QWORD *)v290);
          goto LABEL_42;
        default:
          goto LABEL_30;
      }
      break;
    }
  }
LABEL_7:
  if ( v287 )
  {
    v13 = v285;
    do
    {
      if ( *v13 != -8 && *v13 != -16 )
      {
        v14 = v13[1];
        if ( (v14 & 4) != 0 )
        {
          v15 = (unsigned __int64 *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
          v16 = v15;
          if ( v15 )
          {
            if ( (unsigned __int64 *)*v15 != v15 + 2 )
              _libc_free(*v15);
            j_j___libc_free_0(v16, 48);
          }
        }
      }
      v13 += 2;
    }
    while ( &v285[2 * v287] != v13 );
  }
  return j___libc_free_0(v285);
}
