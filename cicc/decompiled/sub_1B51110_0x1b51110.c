// Function: sub_1B51110
// Address: 0x1b51110
//
_BOOL8 __fastcall sub_1B51110(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  unsigned __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  char v16; // dl
  __int64 v17; // rax
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 *v20; // rsi
  _QWORD *v21; // rdx
  unsigned __int8 v22; // al
  unsigned int v23; // ecx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 *v26; // rbx
  __int64 *v27; // r15
  __int64 *v28; // rbx
  unsigned __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // r14
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 *v38; // r15
  __int64 v39; // r14
  _QWORD *v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 *v43; // r13
  __int64 v44; // rbx
  __int64 v45; // rdx
  char v46; // cl
  __int64 *v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 *v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // r12
  __int64 v58; // r9
  const char *v59; // rax
  int v60; // r15d
  __int64 v61; // rdx
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r12
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 *v68; // r14
  __int64 v69; // r13
  __int64 v70; // rax
  __int64 v71; // r15
  int v72; // eax
  __int64 v73; // rax
  int v74; // ecx
  __int64 v75; // rcx
  _QWORD *v76; // rax
  __int64 v77; // rdi
  unsigned __int64 v78; // rcx
  __int64 v79; // rcx
  __int64 v80; // rcx
  __int64 v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rbx
  unsigned int v86; // ebx
  double v87; // xmm4_8
  double v88; // xmm5_8
  __int64 *v89; // rsi
  __int64 *v90; // rdi
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rsi
  __int64 *v100; // r12
  __int64 *v101; // r15
  __int64 v102; // r13
  __int64 v103; // rbx
  __int64 v104; // rax
  _QWORD *v105; // r12
  double v106; // xmm4_8
  double v107; // xmm5_8
  __int64 *v108; // rbx
  __int64 *v109; // r12
  _QWORD *v110; // rbx
  _QWORD *v111; // r12
  unsigned __int64 v112; // rdi
  unsigned __int64 v113; // rdx
  __int64 v114; // rdi
  __int64 v115; // rcx
  __int64 *v116; // rax
  __int64 v117; // rcx
  __int64 v118; // r9
  unsigned __int64 v119; // r8
  __int64 v120; // r8
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 *v127; // r12
  _QWORD *v128; // r12
  unsigned __int64 v129; // rax
  __int64 v130; // rax
  __int64 *v131; // rbx
  __int64 *v132; // r15
  _QWORD *v133; // rax
  __int64 v134; // r8
  __int64 v135; // rdi
  __int64 v136; // r14
  char v137; // al
  __int64 v138; // rsi
  __int64 *v139; // r15
  __int64 v140; // rdx
  __int64 v141; // rcx
  __int64 *v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rdx
  __int64 v145; // rdx
  __int64 v146; // rdx
  __int64 *v147; // rbx
  __int64 *v148; // r12
  unsigned __int64 v149; // r9
  __int64 *v150; // r8
  __int64 v151; // rsi
  __int64 *v152; // rdi
  __int64 *v153; // rax
  __int64 *v154; // rcx
  __int64 *v155; // rcx
  unsigned __int64 v156; // rax
  unsigned __int64 v157; // rax
  __int64 v158; // rdx
  __int64 v159; // r14
  __int64 v160; // r14
  _QWORD *v161; // rax
  __int64 v162; // rdi
  _QWORD *v163; // rax
  __int64 v164; // rsi
  char v165; // cl
  unsigned int v166; // edx
  __int64 v167; // rdi
  __int64 v168; // r10
  __int64 v169; // r8
  __int64 j; // rdi
  _QWORD *v171; // r11
  _QWORD *v172; // r9
  __int64 v173; // rdi
  __int64 v174; // rsi
  char v175; // cl
  __int64 v176; // rdx
  __int64 v177; // r8
  __int64 v178; // r9
  __int64 v179; // r8
  _QWORD *v180; // r11
  _QWORD *v181; // rcx
  __int64 v182; // rdx
  __int64 v183; // r14
  __int64 v184; // r14
  __int64 *v185; // rax
  char v186; // r14
  __int64 v187; // rsi
  char v188; // cl
  unsigned int v189; // edx
  __int64 v190; // r9
  __int64 v191; // rsi
  _QWORD *v192; // r11
  _QWORD *v193; // r10
  __int64 v194; // rsi
  _QWORD *v195; // rax
  __int64 v196; // rsi
  char v197; // cl
  unsigned int v198; // edx
  __int64 v199; // rdi
  __int64 v200; // r10
  __int64 v201; // r8
  __int64 i; // rdi
  _QWORD *v203; // r11
  _QWORD *v204; // r9
  __int64 v205; // rdi
  char v206; // al
  __int64 *v207; // r14
  __int64 v208; // rax
  __int64 v209; // rax
  int v210; // esi
  int v211; // r9d
  unsigned int v212; // edx
  __int64 *v213; // r12
  __int64 v214; // r8
  __int64 v215; // rdx
  unsigned int v216; // ecx
  __int64 v217; // rdx
  __int64 v218; // rdx
  __int64 v219; // rdx
  int v220; // edi
  __int64 *v221; // rcx
  int v222; // edx
  __int64 v223; // rdx
  __int64 v224; // rdx
  __int64 v225; // rdx
  __int64 v226; // rdx
  __int64 *v227; // rbx
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 v230; // rax
  __int64 v231; // rax
  __int64 v232; // rax
  __int64 v233; // rax
  __int64 v234; // rax
  __int64 v235; // rax
  signed __int64 v236; // rax
  __int64 v237; // rax
  __int64 v238; // rax
  __int64 v239; // rax
  __int64 v240; // rax
  __int64 v241; // rax
  __int64 v242; // rax
  __int64 v243; // rax
  __int64 v244; // rax
  signed __int64 v245; // rax
  __int64 v246; // rax
  __int64 v247; // rax
  __int64 v248; // rax
  __int64 v249; // rax
  __int64 *v250; // rax
  unsigned int v251; // [rsp+4h] [rbp-26Ch]
  __int64 v253; // [rsp+18h] [rbp-258h]
  __int64 v254; // [rsp+18h] [rbp-258h]
  __int64 v255; // [rsp+18h] [rbp-258h]
  __int64 v256; // [rsp+18h] [rbp-258h]
  __int64 v257; // [rsp+20h] [rbp-250h]
  unsigned int v258; // [rsp+20h] [rbp-250h]
  __int64 v259; // [rsp+28h] [rbp-248h]
  __int64 v260; // [rsp+28h] [rbp-248h]
  __int64 *v261; // [rsp+40h] [rbp-230h]
  __int64 *v262; // [rsp+48h] [rbp-228h]
  __int64 v263; // [rsp+48h] [rbp-228h]
  char v264; // [rsp+48h] [rbp-228h]
  __int64 v265; // [rsp+50h] [rbp-220h]
  __int64 v266; // [rsp+50h] [rbp-220h]
  int v267; // [rsp+58h] [rbp-218h]
  bool v268; // [rsp+68h] [rbp-208h]
  __int64 v269; // [rsp+68h] [rbp-208h]
  __int64 *v270; // [rsp+68h] [rbp-208h]
  int v271; // [rsp+68h] [rbp-208h]
  int v272; // [rsp+68h] [rbp-208h]
  __int64 v273; // [rsp+68h] [rbp-208h]
  __int64 v274; // [rsp+70h] [rbp-200h]
  __int64 v275; // [rsp+70h] [rbp-200h]
  char v276; // [rsp+7Bh] [rbp-1F5h]
  char v277; // [rsp+7Bh] [rbp-1F5h]
  unsigned int v278; // [rsp+7Ch] [rbp-1F4h]
  __int64 v279; // [rsp+80h] [rbp-1F0h]
  __int64 v280; // [rsp+80h] [rbp-1F0h]
  __int64 v281; // [rsp+88h] [rbp-1E8h]
  __int64 *v282; // [rsp+88h] [rbp-1E8h]
  unsigned __int64 v283; // [rsp+88h] [rbp-1E8h]
  char *v284; // [rsp+90h] [rbp-1E0h]
  __int64 v285; // [rsp+90h] [rbp-1E0h]
  unsigned __int64 v286; // [rsp+90h] [rbp-1E0h]
  __int64 v287; // [rsp+90h] [rbp-1E0h]
  __int64 *v288; // [rsp+98h] [rbp-1D8h]
  bool v289; // [rsp+98h] [rbp-1D8h]
  __int64 v290; // [rsp+98h] [rbp-1D8h]
  _QWORD v291[2]; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 *v292; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 *v293; // [rsp+B8h] [rbp-1B8h]
  __int64 **v294; // [rsp+C0h] [rbp-1B0h]
  __int64 v295[2]; // [rsp+D0h] [rbp-1A0h] BYREF
  __int16 v296; // [rsp+E0h] [rbp-190h]
  __int64 v297; // [rsp+F0h] [rbp-180h] BYREF
  _QWORD *v298; // [rsp+F8h] [rbp-178h]
  __int64 v299; // [rsp+100h] [rbp-170h]
  unsigned int v300; // [rsp+108h] [rbp-168h]
  __int64 *v301; // [rsp+110h] [rbp-160h] BYREF
  __int64 v302; // [rsp+118h] [rbp-158h]
  _BYTE v303[32]; // [rsp+120h] [rbp-150h] BYREF
  __int64 *v304; // [rsp+140h] [rbp-130h] BYREF
  __int64 v305; // [rsp+148h] [rbp-128h]
  _BYTE v306[32]; // [rsp+150h] [rbp-120h] BYREF
  __int64 *v307; // [rsp+170h] [rbp-100h] BYREF
  __int64 v308; // [rsp+178h] [rbp-F8h]
  _BYTE v309[32]; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v310; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 *v311; // [rsp+1A8h] [rbp-C8h]
  __int64 *v312; // [rsp+1B0h] [rbp-C0h]
  __int64 v313; // [rsp+1B8h] [rbp-B8h]
  int v314; // [rsp+1C0h] [rbp-B0h]
  _BYTE v315[40]; // [rsp+1C8h] [rbp-A8h] BYREF
  __int64 v316[2]; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 *v317; // [rsp+200h] [rbp-70h]
  __int64 v318; // [rsp+208h] [rbp-68h]
  _BYTE v319[32]; // [rsp+210h] [rbp-60h] BYREF
  char v320; // [rsp+230h] [rbp-40h]

  v301 = (__int64 *)v303;
  v302 = 0x400000000LL;
  v316[0] = *(_QWORD *)(a1 + 8);
  sub_15CDD40(v316);
  v9 = v316[0];
  if ( !v316[0] )
  {
    v10 = 0;
    goto LABEL_13;
  }
  v10 = 0;
  v11 = sub_1648700(v316[0]);
LABEL_8:
  v12 = v11[5];
  v13 = sub_157EBA0(v12);
  v16 = *(_BYTE *)(v13 + 16);
  if ( v16 == 26 )
  {
    if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) == 1 )
    {
      v17 = (unsigned int)v302;
      if ( (unsigned int)v302 >= HIDWORD(v302) )
      {
        sub_16CD150((__int64)&v301, v303, 0, 8, v14, v15);
        v17 = (unsigned int)v302;
      }
      v301[v17] = v12;
      LODWORD(v302) = v302 + 1;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 )
        goto LABEL_7;
      goto LABEL_13;
    }
  }
  else if ( v16 != 27 )
  {
LABEL_101:
    v289 = 0;
    goto LABEL_102;
  }
  if ( v10 )
    goto LABEL_101;
  v10 = v13;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      break;
LABEL_7:
    v11 = sub_1648700(v9);
    if ( (unsigned __int8)(*((_BYTE *)v11 + 16) - 25) <= 9u )
      goto LABEL_8;
  }
LABEL_13:
  if ( (unsigned int)v302 <= 1 )
    goto LABEL_101;
  v316[1] = (unsigned int)v302;
  v311 = (__int64 *)v315;
  v312 = (__int64 *)v315;
  v317 = (__int64 *)v319;
  v310 = 0;
  v313 = 4;
  v314 = 0;
  v297 = 0;
  v298 = 0;
  v299 = 0;
  v300 = 0;
  v316[0] = (__int64)v301;
  v318 = 0x400000000LL;
  sub_1B43580((__int64)v316);
  if ( v320 )
  {
    v292 = &v297;
    v293 = &v310;
    v294 = &v301;
    goto LABEL_232;
  }
  v18 = v317;
  v19 = (unsigned int)v318;
  v278 = 0;
LABEL_16:
  while ( 2 )
  {
    while ( 2 )
    {
      v274 = 8 * v19;
      v288 = &v18[v19];
      v279 = *v18;
      if ( v288 != v18 )
      {
        v20 = v18;
        while ( 1 )
        {
          v21 = (_QWORD *)*v20;
          v22 = *(_BYTE *)(*v20 + 16);
          if ( v22 == 77 )
            goto LABEL_32;
          v23 = v22 - 34;
          if ( v23 <= 0x36 && ((1LL << v23) & 0x40018000000001LL) != 0 )
            goto LABEL_32;
          if ( v22 == 53 || *(_BYTE *)(*v21 + 8LL) == 10 )
            goto LABEL_32;
          if ( v22 == 78 )
            break;
          if ( v22 != 55 )
            goto LABEL_25;
LABEL_27:
          if ( v288 == ++v20 )
          {
            v25 = *v18;
            v26 = v18 + 1;
            while ( sub_15F4220(v25, v279, 0) )
            {
              if ( v288 == v26 )
                goto LABEL_237;
              v25 = *v26++;
            }
            goto LABEL_32;
          }
        }
        if ( *(_BYTE *)(*(v21 - 3) + 16LL) == 20 )
          goto LABEL_32;
LABEL_25:
        v24 = v21[1];
        if ( !v24 || *(_QWORD *)(v24 + 8) )
          goto LABEL_32;
        goto LABEL_27;
      }
LABEL_237:
      if ( *(_BYTE *)(v279 + 16) == 55 )
      {
LABEL_473:
        if ( v274 >> 5 )
        {
          v227 = v18;
          while ( 1 )
          {
            v244 = *v227;
            v237 = (*(_BYTE *)(*v227 + 23) & 0x40) != 0
                 ? *(_QWORD *)(v244 - 8)
                 : v244 - 24LL * (*(_DWORD *)(v244 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)(v237 + 24) + 16LL) == 53 )
              goto LABEL_449;
            v238 = v227[1];
            if ( (*(_BYTE *)(v238 + 23) & 0x40) != 0 )
              v239 = *(_QWORD *)(v238 - 8);
            else
              v239 = v238 - 24LL * (*(_DWORD *)(v238 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)(v239 + 24) + 16LL) == 53 )
            {
LABEL_455:
              ++v227;
              goto LABEL_449;
            }
            v240 = v227[2];
            if ( (*(_BYTE *)(v240 + 23) & 0x40) != 0 )
              v241 = *(_QWORD *)(v240 - 8);
            else
              v241 = v240 - 24LL * (*(_DWORD *)(v240 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)(v241 + 24) + 16LL) == 53 )
            {
LABEL_452:
              v227 += 2;
              goto LABEL_449;
            }
            v242 = v227[3];
            if ( (*(_BYTE *)(v242 + 23) & 0x40) != 0 )
              v243 = *(_QWORD *)(v242 - 8);
            else
              v243 = v242 - 24LL * (*(_DWORD *)(v242 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)(v243 + 24) + 16LL) == 53 )
            {
LABEL_448:
              v227 += 3;
              goto LABEL_449;
            }
            v227 += 4;
            if ( &v18[4 * (v274 >> 5)] == v227 )
              goto LABEL_493;
          }
        }
        v227 = v18;
LABEL_493:
        v245 = (char *)v288 - (char *)v227;
        if ( (char *)v288 - (char *)v227 != 16 )
        {
          if ( v245 != 24 )
          {
            if ( v245 != 8 )
              goto LABEL_250;
LABEL_496:
            if ( *(_BYTE *)(*(_QWORD *)(sub_13CF970(*v227) + 24) + 16LL) != 53 )
              goto LABEL_250;
            goto LABEL_449;
          }
          v248 = *v227;
          if ( (*(_BYTE *)(*v227 + 23) & 0x40) != 0 )
            v249 = *(_QWORD *)(v248 - 8);
          else
            v249 = v248 - 24LL * (*(_DWORD *)(v248 + 20) & 0xFFFFFFF);
          if ( *(_BYTE *)(*(_QWORD *)(v249 + 24) + 16LL) == 53 )
            goto LABEL_449;
          ++v227;
        }
        if ( *(_BYTE *)(*(_QWORD *)(sub_13CF970(*v227) + 24) + 16LL) == 53 )
          goto LABEL_449;
        ++v227;
        goto LABEL_496;
      }
      v128 = sub_1648700(*(_QWORD *)(v279 + 8));
      if ( *((_BYTE *)v128 + 16) != 77 )
        v128 = 0;
      v295[0] = (__int64)v128;
      v129 = sub_157EBA0(*(_QWORD *)(v279 + 40));
      v265 = sub_15F4DF0(v129, 0);
      v304 = (__int64 *)v265;
      v307 = v295;
      v308 = (__int64)&v304;
      v130 = v274 >> 3;
      v259 = v274 >> 5;
      if ( v274 >> 5 )
      {
        v286 = v10;
        v131 = v18 + 3;
        v282 = v18;
        v261 = &v18[4 * (v274 >> 5)];
        while ( 1 )
        {
          v132 = v131;
          v269 = *(v131 - 3);
          v133 = sub_1648700(*(_QWORD *)(v269 + 8));
          v134 = v269;
          v135 = *(_QWORD *)(v269 + 40);
          if ( !v128 )
            break;
          v257 = v128[5];
          if ( v265 != v257 )
          {
            if ( v135 != v133[5] )
              goto LABEL_296;
            v136 = *(v131 - 2);
            v270 = v18 + 1;
            if ( sub_1648700(*(_QWORD *)(v136 + 8))[5] != *(_QWORD *)(v136 + 40)
              || (v159 = *(v131 - 1), v270 = v18 + 2, sub_1648700(*(_QWORD *)(v159 + 8))[5] != *(_QWORD *)(v159 + 40)) )
            {
              v10 = v286;
              v18 = v282;
              v132 = v270;
              goto LABEL_247;
            }
LABEL_298:
            v160 = *v131;
            v161 = sub_1648700(*(_QWORD *)(*v131 + 8));
            v162 = *(_QWORD *)(v160 + 40);
LABEL_299:
            if ( v161[5] != v162 )
            {
              v10 = v286;
              v18 = v282;
              goto LABEL_247;
            }
            goto LABEL_327;
          }
          v186 = *((_BYTE *)v128 + 23);
          v187 = 0x2FFFFFFFDLL;
          v272 = *((_DWORD *)v128 + 5);
          v188 = v186 & 0x40;
          v189 = v272 & 0xFFFFFFF;
          if ( (v272 & 0xFFFFFFF) != 0 )
          {
            v190 = 24LL * *((unsigned int *)v128 + 14) + 8;
            v191 = 0;
            do
            {
              v192 = &v128[-3 * v189];
              if ( v188 )
                v192 = (_QWORD *)*(v128 - 1);
              if ( *(_QWORD *)((char *)v192 + v190) == v135 )
              {
                v187 = 3 * v191;
                goto LABEL_339;
              }
              ++v191;
              v190 += 8;
            }
            while ( v189 != (_DWORD)v191 );
            v187 = 0x2FFFFFFFDLL;
          }
LABEL_339:
          if ( v188 )
            v193 = (_QWORD *)*(v128 - 1);
          else
            v193 = &v128[-3 * v189];
          v194 = v193[v187];
          if ( v134 == v194 && v194 )
          {
            v276 = *((_BYTE *)v128 + 23) & 0x40;
            v255 = *(v131 - 2);
            v262 = v18 + 1;
            v195 = sub_1648700(*(_QWORD *)(v255 + 8));
            v196 = v255;
            v197 = v276;
            v198 = v272 & 0xFFFFFFF;
          }
          else
          {
            if ( v135 != v133[5] )
            {
LABEL_296:
              v132 = v18;
              v10 = v286;
              v18 = v282;
              goto LABEL_247;
            }
            v262 = v18 + 1;
            v254 = *(v131 - 2);
            v195 = sub_1648700(*(_QWORD *)(v254 + 8));
            v196 = v254;
            v197 = v186 & 0x40;
            v198 = v272 & 0xFFFFFFF;
          }
          v199 = 0x2FFFFFFFDLL;
          v200 = *(_QWORD *)(v196 + 40);
          if ( v198 )
          {
            v201 = 24LL * *((unsigned int *)v128 + 14) + 8;
            for ( i = 0; i != v198; ++i )
            {
              v203 = &v128[-3 * v198];
              if ( v197 )
                v203 = (_QWORD *)*(v128 - 1);
              if ( v200 == *(_QWORD *)((char *)v203 + v201) )
              {
                v199 = 3 * i;
                goto LABEL_352;
              }
              v201 += 8;
            }
            v199 = 0x2FFFFFFFDLL;
          }
LABEL_352:
          if ( v197 )
            v204 = (_QWORD *)*(v128 - 1);
          else
            v204 = &v128[-3 * v198];
          v205 = v204[v199];
          if ( v205 && v205 == v196 )
          {
            v251 = v198;
            v277 = v197;
            v256 = *(v131 - 1);
            v262 = v18 + 2;
            v163 = sub_1648700(*(_QWORD *)(v256 + 8));
            v164 = v256;
            v165 = v277;
            v166 = v251;
          }
          else
          {
            if ( v195[5] != v200 )
            {
LABEL_357:
              v10 = v286;
              v18 = v282;
              v132 = v262;
              goto LABEL_247;
            }
            v262 = v18 + 2;
            v253 = *(v131 - 1);
            v163 = sub_1648700(*(_QWORD *)(v253 + 8));
            v164 = v253;
            v165 = v186 & 0x40;
            v166 = v272 & 0xFFFFFFF;
          }
          v167 = 0x2FFFFFFFDLL;
          v168 = *(_QWORD *)(v164 + 40);
          if ( v166 )
          {
            v169 = 24LL * *((unsigned int *)v128 + 14) + 8;
            for ( j = 0; j != v166; ++j )
            {
              v171 = &v128[-3 * v166];
              if ( v165 )
                v171 = (_QWORD *)*(v128 - 1);
              if ( v168 == *(_QWORD *)((char *)v171 + v169) )
              {
                v167 = 3 * j;
                goto LABEL_309;
              }
              v169 += 8;
            }
            v167 = 0x2FFFFFFFDLL;
          }
LABEL_309:
          if ( v165 )
            v172 = (_QWORD *)*(v128 - 1);
          else
            v172 = &v128[-3 * v166];
          v173 = v172[v167];
          if ( v173 == v164 && v173 )
          {
            v258 = v166;
            v264 = v165;
            v273 = *v131;
            v161 = sub_1648700(*(_QWORD *)(*v131 + 8));
            v174 = v273;
            v175 = v264;
            v176 = v258;
          }
          else
          {
            if ( v168 != v163[5] )
              goto LABEL_357;
            v263 = *v131;
            v161 = sub_1648700(*(_QWORD *)(*v131 + 8));
            v174 = v263;
            v162 = *(_QWORD *)(v263 + 40);
            if ( v265 != v257 )
              goto LABEL_299;
            v175 = v186 & 0x40;
            v176 = v272 & 0xFFFFFFF;
          }
          v177 = 0x2FFFFFFFDLL;
          v162 = *(_QWORD *)(v174 + 40);
          if ( (_DWORD)v176 )
          {
            v178 = 24LL * *((unsigned int *)v128 + 14) + 8;
            v179 = 0;
            do
            {
              v180 = &v128[-3 * (unsigned int)v176];
              if ( v175 )
                v180 = (_QWORD *)*(v128 - 1);
              if ( v162 == *(_QWORD *)((char *)v180 + v178) )
              {
                v177 = 3 * v179;
                goto LABEL_323;
              }
              ++v179;
              v178 += 8;
            }
            while ( (unsigned int)v176 != v179 );
            v177 = 0x2FFFFFFFDLL;
          }
LABEL_323:
          if ( v175 )
            v181 = (_QWORD *)*(v128 - 1);
          else
            v181 = &v128[-3 * v176];
          v182 = v181[v177];
          if ( !v182 || v182 != v174 )
            goto LABEL_299;
LABEL_327:
          v18 += 4;
          v131 += 4;
          if ( v261 == v18 )
          {
            v132 = v261;
            v10 = v286;
            v18 = v282;
            v130 = v288 - v261;
            goto LABEL_463;
          }
        }
        if ( v135 != v133[5] )
          goto LABEL_296;
        v183 = *(v131 - 2);
        if ( sub_1648700(*(_QWORD *)(v183 + 8))[5] != *(_QWORD *)(v183 + 40) )
        {
          v250 = v18;
          v10 = v286;
          v18 = v282;
          v132 = v250 + 1;
          goto LABEL_247;
        }
        v184 = *(v131 - 1);
        if ( sub_1648700(*(_QWORD *)(v184 + 8))[5] != *(_QWORD *)(v184 + 40) )
        {
          v185 = v18;
          v10 = v286;
          v18 = v282;
          v132 = v185 + 2;
          goto LABEL_247;
        }
        goto LABEL_298;
      }
      v132 = v18;
LABEL_463:
      if ( v130 == 2 )
      {
LABEL_518:
        if ( sub_1B448C0(&v307, *v132) )
        {
          ++v132;
          goto LABEL_466;
        }
        goto LABEL_247;
      }
      if ( v130 != 3 )
      {
        if ( v130 != 1 )
          goto LABEL_248;
LABEL_466:
        if ( sub_1B448C0(&v307, *v132) )
          goto LABEL_248;
        goto LABEL_247;
      }
      if ( sub_1B448C0(&v307, *v132) )
      {
        ++v132;
        goto LABEL_518;
      }
LABEL_247:
      if ( v288 != v132 )
        goto LABEL_32;
LABEL_248:
      v137 = *(_BYTE *)(v279 + 16);
      if ( v137 == 55 )
        goto LABEL_473;
      if ( v137 == 54 )
      {
        if ( v259 )
        {
          v227 = v18;
          while ( 1 )
          {
            v235 = *v227;
            v228 = (*(_BYTE *)(*v227 + 23) & 0x40) != 0
                 ? *(_QWORD *)(v235 - 8)
                 : v235 - 24LL * (*(_DWORD *)(v235 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)v228 + 16LL) == 53 )
              goto LABEL_449;
            v229 = v227[1];
            if ( (*(_BYTE *)(v229 + 23) & 0x40) != 0 )
              v230 = *(_QWORD *)(v229 - 8);
            else
              v230 = v229 - 24LL * (*(_DWORD *)(v229 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)v230 + 16LL) == 53 )
              goto LABEL_455;
            v231 = v227[2];
            if ( (*(_BYTE *)(v231 + 23) & 0x40) != 0 )
              v232 = *(_QWORD *)(v231 - 8);
            else
              v232 = v231 - 24LL * (*(_DWORD *)(v231 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)v232 + 16LL) == 53 )
              goto LABEL_452;
            v233 = v227[3];
            if ( (*(_BYTE *)(v233 + 23) & 0x40) != 0 )
              v234 = *(_QWORD *)(v233 - 8);
            else
              v234 = v233 - 24LL * (*(_DWORD *)(v233 + 20) & 0xFFFFFFF);
            if ( *(_BYTE *)(*(_QWORD *)v234 + 16LL) == 53 )
              goto LABEL_448;
            v227 += 4;
            if ( &v18[4 * v259] == v227 )
              goto LABEL_457;
          }
        }
        v227 = v18;
LABEL_457:
        v236 = (char *)v288 - (char *)v227;
        if ( (char *)v288 - (char *)v227 != 16 )
        {
          if ( v236 != 24 )
          {
            if ( v236 != 8 )
              goto LABEL_250;
LABEL_460:
            if ( *(_BYTE *)(*(_QWORD *)sub_13CF970(*v227) + 16LL) != 53 )
              goto LABEL_250;
LABEL_449:
            if ( v288 != v227 )
              goto LABEL_32;
            goto LABEL_250;
          }
          v246 = *v227;
          if ( (*(_BYTE *)(*v227 + 23) & 0x40) != 0 )
            v247 = *(_QWORD *)(v246 - 8);
          else
            v247 = v246 - 24LL * (*(_DWORD *)(v246 + 20) & 0xFFFFFFF);
          if ( *(_BYTE *)(*(_QWORD *)v247 + 16LL) == 53 )
            goto LABEL_449;
          ++v227;
        }
        if ( *(_BYTE *)(*(_QWORD *)sub_13CF970(*v227) + 16LL) == 53 )
          goto LABEL_449;
        ++v227;
        goto LABEL_460;
      }
LABEL_250:
      v271 = *(_DWORD *)(v279 + 20) & 0xFFFFFFF;
      if ( !v271 )
        goto LABEL_270;
      v283 = v10;
      v287 = 0;
      v138 = v274;
      v275 = v274 >> 5;
      v266 = v138 >> 3;
      v139 = &v18[4 * v275];
      do
      {
        if ( (*(_BYTE *)(v279 + 23) & 0x40) != 0 )
          v140 = *(_QWORD *)(v279 - 8);
        else
          v140 = v279 - 24LL * (*(_DWORD *)(v279 + 20) & 0xFFFFFFF);
        v141 = *(_QWORD *)(v140 + 24 * v287);
        if ( *(_BYTE *)(*(_QWORD *)v141 + 8LL) == 10 )
          goto LABEL_414;
        if ( !v275 )
        {
          v217 = v266;
          v142 = v18;
LABEL_393:
          if ( v217 == 2 )
            goto LABEL_419;
          if ( v217 != 3 )
          {
            if ( v217 != 1 )
              goto LABEL_268;
            goto LABEL_396;
          }
          v223 = *v142;
          if ( (*(_BYTE *)(*v142 + 23) & 0x40) != 0 )
            v224 = *(_QWORD *)(v223 - 8);
          else
            v224 = v223 - 24LL * (*(_DWORD *)(v223 + 20) & 0xFFFFFFF);
          if ( v141 == *(_QWORD *)(v224 + 24 * v287) )
          {
            ++v142;
LABEL_419:
            v225 = *v142;
            if ( (*(_BYTE *)(*v142 + 23) & 0x40) != 0 )
              v226 = *(_QWORD *)(v225 - 8);
            else
              v226 = v225 - 24LL * (*(_DWORD *)(v225 + 20) & 0xFFFFFFF);
            if ( v141 == *(_QWORD *)(v226 + 24 * v287) )
            {
              ++v142;
LABEL_396:
              v218 = *v142;
              if ( (*(_BYTE *)(*v142 + 23) & 0x40) != 0 )
                v219 = *(_QWORD *)(v218 - 8);
              else
                v219 = v218 - 24LL * (*(_DWORD *)(v218 + 20) & 0xFFFFFFF);
              if ( v141 == *(_QWORD *)(v219 + 24 * v287) )
                goto LABEL_268;
            }
          }
LABEL_267:
          if ( v288 == v142 )
            goto LABEL_268;
LABEL_368:
          if ( !sub_1AED280(v279, v287)
            || ((v206 = *(_BYTE *)(v279 + 16), v206 == 29) || v206 == 78) && v271 - 1 == (_DWORD)v287 )
          {
LABEL_414:
            v10 = v283;
            goto LABEL_32;
          }
          v207 = v18;
          if ( v288 == v18 )
            goto LABEL_268;
          while ( 2 )
          {
            v209 = *v207;
            v210 = v300;
            v304 = (__int64 *)*v207;
            if ( v300 )
            {
              v211 = (int)v298;
              v212 = (v300 - 1) & (((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4));
              v213 = &v298[7 * v212];
              v214 = *v213;
              if ( v209 == *v213 )
              {
LABEL_377:
                v215 = *((unsigned int *)v213 + 4);
                v216 = *((_DWORD *)v213 + 5);
                goto LABEL_378;
              }
              v220 = 1;
              v221 = 0;
              while ( v214 != -8 )
              {
                if ( v214 == -16 && !v221 )
                  v221 = v213;
                v212 = (v300 - 1) & (v212 + v220);
                v213 = &v298[7 * v212];
                v214 = *v213;
                if ( v209 == *v213 )
                  goto LABEL_377;
                ++v220;
              }
              if ( v221 )
                v213 = v221;
              ++v297;
              v222 = v299 + 1;
              if ( 4 * ((int)v299 + 1) < 3 * v300 )
              {
                if ( v300 - HIDWORD(v299) - v222 > v300 >> 3 )
                {
LABEL_407:
                  LODWORD(v299) = v222;
                  if ( *v213 != -8 )
                    --HIDWORD(v299);
                  *v213 = v209;
                  v216 = 4;
                  v215 = 0;
                  v213[1] = (__int64)(v213 + 3);
                  v213[2] = 0x400000000LL;
                  v209 = (__int64)v304;
LABEL_378:
                  if ( (*(_BYTE *)(v209 + 23) & 0x40) != 0 )
                  {
                    v208 = *(_QWORD *)(*(_QWORD *)(v209 - 8) + 24 * v287);
                    if ( (unsigned int)v215 >= v216 )
                      goto LABEL_380;
                  }
                  else
                  {
                    v208 = *(_QWORD *)(v209 - 24LL * (*(_DWORD *)(v209 + 20) & 0xFFFFFFF) + 24 * v287);
                    if ( (unsigned int)v215 >= v216 )
                    {
LABEL_380:
                      v260 = v208;
                      sub_16CD150((__int64)(v213 + 1), v213 + 3, 0, 8, v214, v211);
                      v215 = *((unsigned int *)v213 + 4);
                      v208 = v260;
                    }
                  }
                  ++v207;
                  *(_QWORD *)(v213[1] + 8 * v215) = v208;
                  ++*((_DWORD *)v213 + 4);
                  if ( v288 == v207 )
                    goto LABEL_268;
                  continue;
                }
LABEL_412:
                sub_1B509C0((__int64)&v297, v210);
                sub_1B50540((__int64)&v297, (__int64 *)&v304, &v307);
                v213 = v307;
                v209 = (__int64)v304;
                v222 = v299 + 1;
                goto LABEL_407;
              }
            }
            else
            {
              ++v297;
            }
            break;
          }
          v210 = 2 * v300;
          goto LABEL_412;
        }
        v142 = v18;
        while ( 1 )
        {
          v146 = *v142;
          if ( (*(_BYTE *)(*v142 + 23) & 0x40) != 0 )
          {
            if ( v141 != *(_QWORD *)(*(_QWORD *)(v146 - 8) + 24 * v287) )
              goto LABEL_267;
          }
          else if ( v141 != *(_QWORD *)(v146 - 24LL * (*(_DWORD *)(v146 + 20) & 0xFFFFFFF) + 24 * v287) )
          {
            goto LABEL_267;
          }
          v143 = v142[1];
          if ( (*(_BYTE *)(v143 + 23) & 0x40) != 0 )
          {
            if ( v141 != *(_QWORD *)(*(_QWORD *)(v143 - 8) + 24 * v287) )
              break;
            goto LABEL_260;
          }
          if ( v141 != *(_QWORD *)(v143 - 24LL * (*(_DWORD *)(v143 + 20) & 0xFFFFFFF) + 24 * v287) )
            break;
LABEL_260:
          v144 = v142[2];
          if ( (*(_BYTE *)(v144 + 23) & 0x40) != 0 )
          {
            if ( v141 != *(_QWORD *)(*(_QWORD *)(v144 - 8) + 24 * v287) )
              goto LABEL_382;
          }
          else if ( v141 != *(_QWORD *)(v144 - 24LL * (*(_DWORD *)(v144 + 20) & 0xFFFFFFF) + 24 * v287) )
          {
LABEL_382:
            v142 += 2;
            goto LABEL_267;
          }
          v145 = v142[3];
          if ( (*(_BYTE *)(v145 + 23) & 0x40) != 0 )
          {
            if ( v141 != *(_QWORD *)(*(_QWORD *)(v145 - 8) + 24 * v287) )
              goto LABEL_384;
          }
          else if ( v141 != *(_QWORD *)(v145 - 24LL * (*(_DWORD *)(v145 + 20) & 0xFFFFFFF) + 24 * v287) )
          {
LABEL_384:
            v142 += 3;
            goto LABEL_267;
          }
          v142 += 4;
          if ( v139 == v142 )
          {
            v217 = v288 - v139;
            goto LABEL_393;
          }
        }
        if ( v288 != v142 + 1 )
          goto LABEL_368;
LABEL_268:
        ++v287;
      }
      while ( v271 != (_DWORD)v287 );
      v10 = v283;
LABEL_270:
      v147 = v317;
      v148 = &v317[(unsigned int)v318];
      if ( v148 == v317 )
      {
        ++v278;
        if ( v320 )
          goto LABEL_32;
        v19 = (unsigned int)v318;
        v18 = &v148[(unsigned int)v318];
        continue;
      }
      break;
    }
    v149 = (unsigned __int64)v312;
    v150 = v311;
LABEL_274:
    while ( 2 )
    {
      v151 = *v147;
      if ( v150 != (__int64 *)v149 )
      {
LABEL_272:
        sub_16CCBA0((__int64)&v310, v151);
        v149 = (unsigned __int64)v312;
        v150 = v311;
        goto LABEL_273;
      }
      v152 = &v150[HIDWORD(v313)];
      if ( v152 == v150 )
      {
LABEL_387:
        if ( HIDWORD(v313) < (unsigned int)v313 )
        {
          ++HIDWORD(v313);
          *v152 = v151;
          v150 = v311;
          ++v310;
          v149 = (unsigned __int64)v312;
          goto LABEL_273;
        }
        goto LABEL_272;
      }
      v153 = v150;
      v154 = 0;
      while ( v151 != *v153 )
      {
        if ( *v153 == -2 )
          v154 = v153;
        if ( v152 == ++v153 )
        {
          if ( !v154 )
            goto LABEL_387;
          ++v147;
          *v154 = v151;
          v149 = (unsigned __int64)v312;
          --v314;
          v150 = v311;
          ++v310;
          if ( v148 != v147 )
            goto LABEL_274;
          goto LABEL_283;
        }
      }
LABEL_273:
      if ( v148 != ++v147 )
        continue;
      break;
    }
LABEL_283:
    ++v278;
    if ( v320 )
      goto LABEL_32;
    v155 = v317;
    v19 = (unsigned int)v318;
    v18 = &v317[(unsigned int)v318];
    if ( v317 == v18 )
      continue;
    break;
  }
LABEL_287:
  v156 = *v155;
  if ( *(_QWORD *)(*(_QWORD *)(*v155 + 40) + 48LL) != *v155 + 24 )
  {
    do
    {
      v157 = *(_QWORD *)(v156 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v157 )
        break;
      v156 = v157 - 24;
      *v155 = v156;
      if ( *(_BYTE *)(v156 + 16) == 78 )
      {
        v158 = *(_QWORD *)(v156 - 24);
        if ( !*(_BYTE *)(v158 + 16)
          && (*(_BYTE *)(v158 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v158 + 36) - 35) <= 3 )
        {
          continue;
        }
      }
      if ( v18 != ++v155 )
        goto LABEL_287;
      if ( !v320 )
      {
        v18 = v317;
        v19 = (unsigned int)v318;
        goto LABEL_16;
      }
      goto LABEL_32;
    }
    while ( *(_QWORD *)(*(_QWORD *)(v156 + 40) + 48LL) != v156 + 24 );
  }
  *v155 = 0;
  v320 = 1;
LABEL_32:
  v292 = &v297;
  v293 = &v310;
  v294 = &v301;
  v268 = v10 != 0 && v278 != 0;
  if ( !v268 )
  {
    if ( v278 )
      goto LABEL_34;
LABEL_232:
    v289 = 0;
    goto LABEL_169;
  }
  v86 = 0;
  sub_1B43580((__int64)v316);
  while ( 1 )
  {
    v268 = sub_1B50CE0((__int64 *)&v292, (__int64)v316) && v86 < v278;
    if ( !v268 )
      goto LABEL_232;
    if ( !(unsigned __int8)sub_14AF470(*v317, 0, 0, 0) )
      break;
    if ( !v320 )
    {
      v89 = v317;
      v90 = &v317[(unsigned int)v318];
      if ( v317 != v90 )
      {
LABEL_114:
        v91 = *v89;
        if ( *(_QWORD *)(*(_QWORD *)(*v89 + 40) + 48LL) != *v89 + 24 )
        {
          do
          {
            v92 = *(_QWORD *)(v91 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v92 )
              break;
            v91 = v92 - 24;
            *v89 = v91;
            if ( *(_BYTE *)(v91 + 16) == 78 )
            {
              v93 = *(_QWORD *)(v91 - 24);
              if ( !*(_BYTE *)(v93 + 16)
                && (*(_BYTE *)(v93 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v93 + 36) - 35) <= 3 )
              {
                continue;
              }
            }
            if ( v90 == ++v89 )
              goto LABEL_122;
            goto LABEL_114;
          }
          while ( *(_QWORD *)(*(_QWORD *)(v91 + 40) + 48LL) != v91 + 24 );
        }
        *v89 = 0;
        v320 = 1;
      }
    }
LABEL_122:
    ++v86;
  }
  if ( !sub_1AAB350(a1, v301, (unsigned int)v302, ".sink.split", 0, 0, a2, a3, a4, a5, v87, v88, a8, a9, 0) )
    goto LABEL_232;
LABEL_34:
  v267 = 0;
LABEL_35:
  sub_1B43580((__int64)v316);
  v289 = v268;
  v268 = sub_1B50CE0((__int64 *)&v292, (__int64)v316);
  if ( !v268 )
    goto LABEL_169;
  v27 = v301;
  v28 = &v301[(unsigned int)v302];
  v29 = sub_157EBA0(*v301);
  v280 = sub_15F4DF0(v29, 0);
  v304 = (__int64 *)v306;
  v305 = 0x400000000LL;
  if ( v27 == v28 )
  {
    v38 = (__int64 *)v306;
  }
  else
  {
LABEL_41:
    while ( 2 )
    {
      v33 = *v27;
      v34 = sub_157EBA0(*v27);
      do
      {
        if ( *(_QWORD *)(*(_QWORD *)(v34 + 40) + 48LL) == v34 + 24
          || (v35 = *(_QWORD *)(v34 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          BUG();
        }
        v34 = v35 - 24;
        if ( *(_BYTE *)(v35 - 8) != 78
          || (v36 = *(_QWORD *)(v35 - 48), *(_BYTE *)(v36 + 16))
          || (*(_BYTE *)(v36 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v36 + 36) - 35) > 3 )
        {
          v32 = (unsigned int)v305;
          if ( (unsigned int)v305 >= HIDWORD(v305) )
          {
            sub_16CD150((__int64)&v304, v306, 0, 8, v30, v31);
            v32 = (unsigned int)v305;
          }
          ++v27;
          v304[v32] = v34;
          LODWORD(v305) = v305 + 1;
          if ( v28 == v27 )
            goto LABEL_51;
          goto LABEL_41;
        }
        v37 = *(_QWORD *)(v33 + 48);
      }
      while ( !v37 || v34 != v37 - 24 );
      if ( v28 != ++v27 )
        continue;
      break;
    }
LABEL_51:
    v38 = v304;
  }
  v39 = *v38;
  if ( *(_BYTE *)(*v38 + 16) == 55 )
  {
LABEL_59:
    v307 = (__int64 *)v309;
    v308 = 0x400000000LL;
    if ( (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) == 0 )
      goto LABEL_150;
    v44 = 0;
    v285 = 24LL * ((*(_DWORD *)(v39 + 20) & 0xFFFFFFFu) - 1);
LABEL_61:
    v45 = 8LL * (unsigned int)v305;
    v46 = *(_BYTE *)(v39 + 23) & 0x40;
    v47 = &v38[(unsigned __int64)v45 / 8];
    v48 = v45 >> 3;
    v49 = v45 >> 5;
    if ( !v49 )
    {
LABEL_141:
      if ( v48 != 2 )
      {
        if ( v48 != 3 )
        {
          if ( v48 != 1 )
            goto LABEL_125;
          v96 = *v38;
          if ( (*(_BYTE *)(*v38 + 23) & 0x40) != 0 )
          {
LABEL_145:
            v97 = *(_QWORD *)(*(_QWORD *)(v96 - 8) + v44);
            if ( v46 )
              goto LABEL_146;
LABEL_206:
            v98 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
            goto LABEL_147;
          }
LABEL_205:
          v97 = *(_QWORD *)(v96 - 24LL * (*(_DWORD *)(v96 + 20) & 0xFFFFFFF) + v44);
          if ( !v46 )
            goto LABEL_206;
LABEL_146:
          v98 = *(_QWORD *)(v39 - 8);
LABEL_147:
          if ( v97 == *(_QWORD *)(v98 + v44) )
          {
            if ( v46 )
              goto LABEL_126;
            goto LABEL_149;
          }
          goto LABEL_75;
        }
        v121 = *v38;
        if ( (*(_BYTE *)(*v38 + 23) & 0x40) != 0 )
          v122 = *(_QWORD *)(v121 - 8);
        else
          v122 = v121 - 24LL * (*(_DWORD *)(v121 + 20) & 0xFFFFFFF);
        if ( v46 )
          v123 = *(_QWORD *)(v39 - 8);
        else
          v123 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
        if ( *(_QWORD *)(v122 + v44) != *(_QWORD *)(v123 + v44) )
          goto LABEL_75;
        ++v38;
      }
      v124 = *v38;
      if ( (*(_BYTE *)(*v38 + 23) & 0x40) != 0 )
        v125 = *(_QWORD *)(v124 - 8);
      else
        v125 = v124 - 24LL * (*(_DWORD *)(v124 + 20) & 0xFFFFFFF);
      if ( v46 )
        v126 = *(_QWORD *)(v39 - 8);
      else
        v126 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
      if ( *(_QWORD *)(v125 + v44) == *(_QWORD *)(v126 + v44) )
      {
        v96 = v38[1];
        ++v38;
        if ( (*(_BYTE *)(v96 + 23) & 0x40) != 0 )
          goto LABEL_145;
        goto LABEL_205;
      }
LABEL_75:
      if ( v47 == v38 )
        goto LABEL_125;
LABEL_76:
      if ( v46 )
        v56 = *(_QWORD *)(v39 - 8);
      else
        v56 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
      v57 = *(__int64 **)(v56 + v44);
      v58 = *(_QWORD *)(v280 + 48);
      if ( v58 )
        v58 -= 24;
      v290 = v58;
      v59 = sub_1649960(*(_QWORD *)(v56 + v44));
      v60 = v305;
      v291[0] = v59;
      v295[0] = (__int64)v291;
      v291[1] = v61;
      v296 = 773;
      v295[1] = (__int64)".sink";
      v62 = *v57;
      v63 = sub_1648B60(64);
      v65 = v63;
      if ( v63 )
      {
        sub_15F1EA0(v63, v62, 53, 0, 0, v290);
        *(_DWORD *)(v65 + 56) = v60;
        sub_164B780(v65, v295);
        sub_1648880(v65, *(_DWORD *)(v65 + 56), 1);
      }
      v66 = (__int64)v304;
      if ( v304 != &v304[(unsigned int)v305] )
      {
        v281 = v39;
        v67 = v44;
        v68 = &v304[(unsigned int)v305];
        v69 = (__int64)v304;
        do
        {
          v83 = *(_QWORD *)v69;
          v84 = *(_QWORD *)(*(_QWORD *)v69 + 40LL);
          if ( (*(_BYTE *)(*(_QWORD *)v69 + 23LL) & 0x40) != 0 )
          {
            v70 = *(_QWORD *)(v83 - 8);
          }
          else
          {
            v64 = 24LL * (*(_DWORD *)(v83 + 20) & 0xFFFFFFF);
            v70 = v83 - v64;
          }
          v71 = *(_QWORD *)(v70 + v67);
          v72 = *(_DWORD *)(v65 + 20) & 0xFFFFFFF;
          if ( v72 == *(_DWORD *)(v65 + 56) )
          {
            sub_15F55D0(v65, v67, v66, v64, v30, v31);
            v72 = *(_DWORD *)(v65 + 20) & 0xFFFFFFF;
          }
          v73 = (v72 + 1) & 0xFFFFFFF;
          v74 = v73 | *(_DWORD *)(v65 + 20) & 0xF0000000;
          *(_DWORD *)(v65 + 20) = v74;
          if ( (v74 & 0x40000000) != 0 )
            v75 = *(_QWORD *)(v65 - 8);
          else
            v75 = v65 - 24 * v73;
          v76 = (_QWORD *)(v75 + 24LL * (unsigned int)(v73 - 1));
          if ( *v76 )
          {
            v77 = v76[1];
            v78 = v76[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v78 = v77;
            if ( v77 )
            {
              v30 = *(_QWORD *)(v77 + 16) & 3LL;
              *(_QWORD *)(v77 + 16) = v30 | v78;
            }
          }
          *v76 = v71;
          if ( v71 )
          {
            v79 = *(_QWORD *)(v71 + 8);
            v30 = v71 + 8;
            v76[1] = v79;
            if ( v79 )
            {
              v31 = (__int64)(v76 + 1);
              *(_QWORD *)(v79 + 16) = (unsigned __int64)(v76 + 1) | *(_QWORD *)(v79 + 16) & 3LL;
            }
            v76[2] = v30 | v76[2] & 3LL;
            *(_QWORD *)(v71 + 8) = v76;
          }
          v80 = *(_DWORD *)(v65 + 20) & 0xFFFFFFF;
          v81 = (unsigned int)(v80 - 1);
          if ( (*(_BYTE *)(v65 + 23) & 0x40) != 0 )
            v82 = *(_QWORD *)(v65 - 8);
          else
            v82 = v65 - 24 * v80;
          v69 += 8;
          v64 = 3LL * *(unsigned int *)(v65 + 56);
          *(_QWORD *)(v82 + 8 * v81 + 24LL * *(unsigned int *)(v65 + 56) + 8) = v84;
        }
        while ( v68 != (__int64 *)v69 );
        v39 = v281;
        v44 = v67;
      }
      v95 = (unsigned int)v308;
      if ( (unsigned int)v308 >= HIDWORD(v308) )
        goto LABEL_128;
      goto LABEL_137;
    }
    v50 = &v38[4 * v49];
    while ( 1 )
    {
      v55 = *v38;
      if ( (*(_BYTE *)(*v38 + 23) & 0x40) != 0 )
      {
        v30 = *(_QWORD *)(*(_QWORD *)(v55 - 8) + v44);
        if ( v46 )
          goto LABEL_64;
      }
      else
      {
        v30 = *(_QWORD *)(v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF) + v44);
        if ( v46 )
        {
LABEL_64:
          v51 = *(_QWORD *)(*(_QWORD *)(v39 - 8) + v44);
          if ( v30 != v51 )
            goto LABEL_75;
          goto LABEL_65;
        }
      }
      v51 = *(_QWORD *)(v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) + v44);
      if ( v30 != v51 )
        goto LABEL_75;
LABEL_65:
      v52 = v38[1];
      if ( (*(_BYTE *)(v52 + 23) & 0x40) != 0 )
      {
        if ( v51 != *(_QWORD *)(*(_QWORD *)(v52 - 8) + v44) )
          goto LABEL_124;
      }
      else
      {
        v30 = 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
        if ( v51 != *(_QWORD *)(v52 - v30 + v44) )
        {
LABEL_124:
          if ( v47 != v38 + 1 )
            goto LABEL_76;
LABEL_125:
          if ( v46 )
          {
LABEL_126:
            v94 = *(_QWORD *)(v39 - 8);
            goto LABEL_127;
          }
LABEL_149:
          v94 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
LABEL_127:
          v65 = *(_QWORD *)(v94 + v44);
          v95 = (unsigned int)v308;
          if ( (unsigned int)v308 >= HIDWORD(v308) )
          {
LABEL_128:
            sub_16CD150((__int64)&v307, v309, 0, 8, v30, v31);
            v95 = (unsigned int)v308;
          }
LABEL_137:
          v307[v95] = v65;
          LODWORD(v308) = v308 + 1;
          if ( v285 != v44 )
          {
            v38 = v304;
            v44 += 24;
            goto LABEL_61;
          }
          if ( (*(_DWORD *)(v39 + 20) & 0xFFFFFFF) != 0 )
          {
            v113 = 0;
            v114 = 8LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
            do
            {
              if ( (*(_BYTE *)(v39 + 23) & 0x40) != 0 )
                v115 = *(_QWORD *)(v39 - 8);
              else
                v115 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF);
              v116 = (__int64 *)(v115 + 3 * v113);
              v117 = v307[v113 / 8];
              if ( *v116 )
              {
                v118 = v116[1];
                v119 = v116[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v119 = v118;
                if ( v118 )
                  *(_QWORD *)(v118 + 16) = v119 | *(_QWORD *)(v118 + 16) & 3LL;
              }
              *v116 = v117;
              if ( v117 )
              {
                v120 = *(_QWORD *)(v117 + 8);
                v116[1] = v120;
                if ( v120 )
                  *(_QWORD *)(v120 + 16) = (unsigned __int64)(v116 + 1) | *(_QWORD *)(v120 + 16) & 3LL;
                v116[2] = (v117 + 8) | v116[2] & 3;
                *(_QWORD *)(v117 + 8) = v116;
              }
              v113 += 8LL;
            }
            while ( v114 != v113 );
          }
LABEL_150:
          v99 = sub_157EE30(v280);
          if ( v99 )
            v99 -= 24;
          sub_15F22F0((_QWORD *)v39, v99);
          v100 = v304;
          if ( v304 == &v304[(unsigned int)v305] )
          {
            if ( *(_BYTE *)(v39 + 16) != 55 )
              goto LABEL_158;
          }
          else
          {
            v101 = &v304[(unsigned int)v305];
            do
            {
              v102 = *v100;
              if ( v39 != *v100 )
              {
                v103 = sub_15C70A0(v102 + 48);
                v104 = sub_15C70A0(v39 + 48);
                sub_15AC0B0(v39, v104, v103);
                sub_1AEC340(v39, v102);
                sub_15F2780((unsigned __int8 *)v39, v102);
              }
              ++v100;
            }
            while ( v101 != v100 );
            if ( *(_BYTE *)(v39 + 16) != 55 )
            {
LABEL_158:
              v105 = sub_1648700(*(_QWORD *)(v39 + 8));
              sub_164D160((__int64)v105, v39, a2, a3, a4, a5, v106, v107, a8, a9);
              sub_15F20C0(v105);
            }
            v108 = v304;
            v109 = &v304[(unsigned int)v305];
            if ( v304 != v109 )
            {
              do
              {
                if ( v39 != *v108 )
                  sub_15F20C0((_QWORD *)*v108);
                ++v108;
              }
              while ( v109 != v108 );
            }
          }
          if ( v307 != (__int64 *)v309 )
            _libc_free((unsigned __int64)v307);
          if ( v304 != (__int64 *)v306 )
            _libc_free((unsigned __int64)v304);
          if ( ++v267 == v278 )
          {
            v289 = v268;
            goto LABEL_169;
          }
          goto LABEL_35;
        }
      }
      v53 = v38[2];
      if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
      {
        if ( v51 != *(_QWORD *)(*(_QWORD *)(v53 - 8) + v44) )
          goto LABEL_130;
      }
      else
      {
        v30 = 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
        if ( v51 != *(_QWORD *)(v53 - v30 + v44) )
        {
LABEL_130:
          v38 += 2;
          goto LABEL_75;
        }
      }
      v54 = v38[3];
      if ( (*(_BYTE *)(v54 + 23) & 0x40) != 0 )
      {
        if ( v51 != *(_QWORD *)(*(_QWORD *)(v54 - 8) + v44) )
          goto LABEL_132;
      }
      else
      {
        v30 = 24LL * (*(_DWORD *)(v54 + 20) & 0xFFFFFFF);
        if ( v51 != *(_QWORD *)(v54 - v30 + v44) )
        {
LABEL_132:
          v38 += 3;
          goto LABEL_75;
        }
      }
      v38 += 4;
      if ( v50 == v38 )
      {
        v48 = v47 - v38;
        goto LABEL_141;
      }
    }
  }
  v40 = sub_1648700(*(_QWORD *)(v39 + 8));
  v41 = 8LL * (unsigned int)v305;
  v284 = (char *)&v38[(unsigned __int64)v41 / 8];
  if ( *((_BYTE *)v40 + 16) != 77 )
  {
    v42 = v41 >> 3;
    if ( !((unsigned __int64)v41 >> 5) && v41 != 16 )
    {
      v43 = v38;
      v40 = 0;
      goto LABEL_57;
    }
    v43 = v38;
    goto LABEL_212;
  }
  v42 = v41 >> 3;
  if ( v41 >> 5 )
  {
    v43 = v38;
    v127 = &v38[4 * (v41 >> 5)];
    while ( 1 )
    {
      if ( v40 != sub_1648700(*(_QWORD *)(v43[1] + 8)) )
      {
        ++v43;
        goto LABEL_212;
      }
      if ( v40 != sub_1648700(*(_QWORD *)(v43[2] + 8)) )
      {
        v43 += 2;
        goto LABEL_212;
      }
      if ( v40 != sub_1648700(*(_QWORD *)(v43[3] + 8)) )
      {
        v43 += 3;
        goto LABEL_212;
      }
      v43 += 4;
      if ( v127 == v43 )
        break;
      if ( v40 != sub_1648700(*(_QWORD *)(*v43 + 8)) )
        goto LABEL_212;
    }
    v42 = (v284 - (char *)v43) >> 3;
    if ( v284 - (char *)v43 == 16 )
      goto LABEL_225;
LABEL_57:
    if ( v42 != 3 )
    {
      if ( v42 != 1 )
        goto LABEL_59;
      goto LABEL_227;
    }
    if ( v40 != sub_1648700(*(_QWORD *)(*v43 + 8)) )
      goto LABEL_212;
    ++v43;
LABEL_225:
    if ( v40 != sub_1648700(*(_QWORD *)(*v43 + 8)) )
      goto LABEL_212;
  }
  else
  {
    v43 = v38;
    if ( v41 != 16 )
      goto LABEL_57;
  }
  ++v43;
LABEL_227:
  if ( v40 == sub_1648700(*(_QWORD *)(*v43 + 8)) )
    goto LABEL_59;
LABEL_212:
  if ( v43 == (__int64 *)v284 )
    goto LABEL_59;
  if ( v38 != (__int64 *)v306 )
    _libc_free((unsigned __int64)v38);
LABEL_169:
  if ( v317 != (__int64 *)v319 )
    _libc_free((unsigned __int64)v317);
  if ( v300 )
  {
    v110 = v298;
    v111 = &v298[7 * v300];
    do
    {
      if ( *v110 != -16 && *v110 != -8 )
      {
        v112 = v110[1];
        if ( (_QWORD *)v112 != v110 + 3 )
          _libc_free(v112);
      }
      v110 += 7;
    }
    while ( v111 != v110 );
  }
  j___libc_free_0(v298);
  if ( v312 != v311 )
    _libc_free((unsigned __int64)v312);
LABEL_102:
  if ( v301 != (__int64 *)v303 )
    _libc_free((unsigned __int64)v301);
  return v289;
}
