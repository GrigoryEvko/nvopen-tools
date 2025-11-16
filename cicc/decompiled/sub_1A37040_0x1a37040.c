// Function: sub_1A37040
// Address: 0x1a37040
//
__int64 __fastcall sub_1A37040(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  __int64 v17; // r12
  _QWORD *v18; // r14
  _QWORD *v19; // rax
  unsigned __int8 v20; // cl
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rsi
  unsigned __int64 v27; // r15
  char v28; // al
  __int64 v29; // r15
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // r15
  unsigned __int8 *v33; // rax
  __int64 v34; // r13
  unsigned __int8 *v35; // rcx
  __int64 v36; // rsi
  unsigned __int64 v37; // r13
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  __int64 v40; // r14
  _QWORD *v41; // rax
  int v42; // r8d
  int v43; // r9d
  unsigned __int8 v44; // dl
  __int64 v45; // rax
  bool v46; // zf
  char v47; // al
  __int64 *v48; // r9
  __int64 v49; // rsi
  unsigned __int64 *v50; // rdx
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // ecx
  __int64 v54; // rdx
  __int64 v55; // r14
  char v56; // dl
  unsigned __int64 *v57; // rax
  unsigned __int64 *v58; // rbx
  unsigned __int64 *v59; // r13
  unsigned __int64 **v60; // r13
  unsigned __int64 **v61; // rbx
  char *v62; // r9
  __int64 v63; // rax
  _QWORD *v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rax
  _QWORD *v67; // rdi
  __int64 v68; // rax
  _QWORD *v69; // rax
  size_t v70; // rdx
  char *v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rbx
  char *v74; // r13
  unsigned __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // rbx
  char *v78; // rsi
  unsigned __int64 v79; // rdx
  char *v80; // rcx
  __int64 v81; // rax
  __int64 *v82; // rax
  __int64 v83; // r13
  unsigned __int64 v84; // rbx
  __int64 *v85; // r14
  unsigned __int64 v86; // rbx
  __int8 v87; // bl
  unsigned int v88; // eax
  __int64 v89; // r14
  unsigned int v90; // eax
  unsigned int v91; // r8d
  unsigned int v92; // eax
  __int64 v93; // rdx
  _QWORD *v94; // rax
  __int64 v95; // r11
  unsigned int v96; // edx
  __m128i *v97; // r8
  __int64 v98; // rsi
  __int64 v99; // rcx
  unsigned __int64 v100; // rsi
  unsigned __int64 *v101; // rax
  unsigned __int64 *v102; // rax
  _QWORD *v103; // rdx
  __int64 v104; // rdi
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rsi
  unsigned __int64 v108; // rax
  __int64 v109; // rax
  double v110; // xmm4_8
  double v111; // xmm5_8
  unsigned __int64 **v112; // r12
  unsigned __int64 **v113; // rbx
  char v114; // r13
  unsigned __int64 *v115; // rsi
  unsigned __int64 *v116; // r12
  unsigned __int64 *v117; // rbx
  char v118; // r13
  unsigned __int64 *v119; // rsi
  __int64 v120; // r15
  __int64 v121; // rax
  __int64 v122; // rax
  char v123; // r12
  __m128i *v124; // rax
  __int64 v125; // rbx
  __int64 v126; // rbx
  unsigned __int64 v127; // r15
  __int64 v128; // rax
  __int64 v129; // r12
  __int64 v130; // rdx
  __int64 *v131; // rbx
  __int64 v132; // r14
  __int64 i; // rax
  int v134; // ecx
  int v135; // ecx
  _QWORD *v136; // r9
  __int64 v137; // rsi
  unsigned int v138; // eax
  __int64 *v139; // rdi
  __int64 v140; // r8
  __int64 v141; // r12
  __int64 v143; // rax
  unsigned __int8 *v144; // rsi
  __int64 *v145; // rax
  double v146; // xmm4_8
  double v147; // xmm5_8
  __int64 **v148; // r13
  __int64 **v149; // rbx
  __int64 *v150; // rdi
  __int64 v151; // rax
  __int64 v152; // rax
  unsigned __int64 *v153; // rbx
  unsigned __int64 *v154; // r13
  __int64 v155; // r12
  int v156; // r9d
  _QWORD *v157; // r8
  int v158; // r11d
  _QWORD *v159; // r10
  unsigned int v160; // edx
  _QWORD *v161; // rdi
  __int64 v162; // rcx
  unsigned __int64 v163; // rax
  unsigned int v164; // esi
  int v165; // ecx
  __int64 v166; // rax
  __int64 *v167; // rbx
  __int64 v168; // rax
  __int64 *v169; // rsi
  __int64 *v170; // rax
  double v171; // xmm4_8
  double v172; // xmm5_8
  __int64 v173; // r14
  char v174; // dl
  unsigned int v175; // eax
  void *v176; // rbx
  __int64 v177; // r13
  unsigned __int64 v178; // r12
  unsigned __int64 v179; // r13
  __int64 **v180; // r9
  __int64 **v181; // r14
  __int64 *v182; // r12
  __int64 v183; // r15
  __int64 *v184; // r13
  char v185; // al
  __int64 v186; // rsi
  __int64 v187; // rax
  __int64 *v188; // rdx
  __int64 v189; // r13
  __int64 v190; // rbx
  __int64 v191; // r12
  __int64 v192; // rdx
  __int64 v193; // r14
  unsigned __int64 v194; // rax
  unsigned __int64 v195; // rax
  unsigned int v196; // eax
  int v197; // edi
  int v198; // r11d
  __int64 v199; // rsi
  unsigned __int8 *v200; // rsi
  unsigned int v201; // ebx
  _QWORD *v202; // rax
  unsigned __int64 *v203; // rbx
  __int64 v204; // r12
  unsigned __int64 *v205; // r13
  int v206; // r9d
  _QWORD *v207; // r8
  int v208; // r11d
  _QWORD *m128i_i64; // r10
  unsigned int v210; // edx
  _QWORD *v211; // rdi
  __int64 v212; // rcx
  unsigned __int64 v213; // rax
  unsigned int v214; // esi
  int v215; // ecx
  __int64 v216; // rax
  int v217; // edx
  int v218; // edx
  char *v219; // rbx
  __int64 v220; // rcx
  __int64 v221; // rdx
  char *v222; // rax
  char *v223; // rsi
  unsigned __int8 *v224; // rax
  unsigned __int8 *v225; // rcx
  _QWORD *v226; // rax
  __int64 v227; // rdx
  char *v228; // rax
  size_t v229; // rdx
  _BYTE *v230; // rsi
  __int64 v231; // rax
  _QWORD *v232; // r13
  unsigned int v233; // eax
  __int64 v234; // rax
  unsigned __int8 *v235; // [rsp+10h] [rbp-4B0h]
  unsigned int v237; // [rsp+50h] [rbp-470h]
  char v238; // [rsp+57h] [rbp-469h]
  __int64 *v240; // [rsp+68h] [rbp-458h]
  __int64 v241; // [rsp+80h] [rbp-440h]
  __int64 *v242; // [rsp+80h] [rbp-440h]
  __int64 v243; // [rsp+88h] [rbp-438h]
  __int64 v244; // [rsp+90h] [rbp-430h]
  __int64 v246; // [rsp+A0h] [rbp-420h]
  __int64 v247; // [rsp+A0h] [rbp-420h]
  __int64 v248; // [rsp+A8h] [rbp-418h]
  __int64 v249; // [rsp+A8h] [rbp-418h]
  unsigned int src; // [rsp+B0h] [rbp-410h]
  __int64 v251; // [rsp+B8h] [rbp-408h]
  unsigned __int64 v252; // [rsp+B8h] [rbp-408h]
  char *v253; // [rsp+B8h] [rbp-408h]
  __int64 v254; // [rsp+B8h] [rbp-408h]
  __int64 *v255; // [rsp+B8h] [rbp-408h]
  char v256; // [rsp+B8h] [rbp-408h]
  __int64 v257; // [rsp+B8h] [rbp-408h]
  __int64 v258; // [rsp+C0h] [rbp-400h]
  __int64 v259; // [rsp+C0h] [rbp-400h]
  __int64 v260; // [rsp+C0h] [rbp-400h]
  char v261; // [rsp+C0h] [rbp-400h]
  char v262; // [rsp+C8h] [rbp-3F8h]
  unsigned __int64 v263; // [rsp+C8h] [rbp-3F8h]
  __int64 v264; // [rsp+C8h] [rbp-3F8h]
  __int64 *v265; // [rsp+C8h] [rbp-3F8h]
  unsigned int v266; // [rsp+C8h] [rbp-3F8h]
  __int64 *v267; // [rsp+C8h] [rbp-3F8h]
  __int64 v268; // [rsp+D0h] [rbp-3F0h]
  _QWORD *v269; // [rsp+D0h] [rbp-3F0h]
  __int64 v270; // [rsp+D0h] [rbp-3F0h]
  __int64 v271; // [rsp+D0h] [rbp-3F0h]
  unsigned int v272; // [rsp+D0h] [rbp-3F0h]
  __int64 *v273; // [rsp+D0h] [rbp-3F0h]
  __int64 v274; // [rsp+D0h] [rbp-3F0h]
  char *v275; // [rsp+D8h] [rbp-3E8h]
  unsigned __int64 v276; // [rsp+D8h] [rbp-3E8h]
  char v277; // [rsp+D8h] [rbp-3E8h]
  unsigned __int64 *v278; // [rsp+D8h] [rbp-3E8h]
  char *v279; // [rsp+D8h] [rbp-3E8h]
  __int64 *v280; // [rsp+D8h] [rbp-3E8h]
  unsigned int v281; // [rsp+D8h] [rbp-3E8h]
  void *v282; // [rsp+D8h] [rbp-3E8h]
  __m128i *v283; // [rsp+D8h] [rbp-3E8h]
  __int64 *v284; // [rsp+D8h] [rbp-3E8h]
  void *v285; // [rsp+D8h] [rbp-3E8h]
  __int64 *v286; // [rsp+D8h] [rbp-3E8h]
  void *v287; // [rsp+D8h] [rbp-3E8h]
  __int64 **v288; // [rsp+D8h] [rbp-3E8h]
  __m128i *v289; // [rsp+D8h] [rbp-3E8h]
  char *v290; // [rsp+D8h] [rbp-3E8h]
  __int64 v291; // [rsp+E8h] [rbp-3D8h] BYREF
  __int64 *v292[6]; // [rsp+F0h] [rbp-3D0h] BYREF
  __int64 *v293[6]; // [rsp+120h] [rbp-3A0h] BYREF
  __m128i *v294; // [rsp+150h] [rbp-370h] BYREF
  __int64 v295; // [rsp+158h] [rbp-368h]
  _BYTE v296[32]; // [rsp+160h] [rbp-360h] BYREF
  unsigned __int64 v297; // [rsp+180h] [rbp-340h] BYREF
  __int64 v298; // [rsp+188h] [rbp-338h]
  _BYTE v299[32]; // [rsp+190h] [rbp-330h] BYREF
  __m128i *v300; // [rsp+1B0h] [rbp-310h] BYREF
  __int64 v301; // [rsp+1B8h] [rbp-308h]
  __m128i v302; // [rsp+1C0h] [rbp-300h] BYREF
  __int64 v303; // [rsp+1D0h] [rbp-2F0h]
  int v304; // [rsp+1D8h] [rbp-2E8h]
  __int64 v305; // [rsp+1E0h] [rbp-2E0h]
  __int64 v306; // [rsp+1E8h] [rbp-2D8h]
  _QWORD *v307; // [rsp+1F0h] [rbp-2D0h]
  __int64 v308; // [rsp+1F8h] [rbp-2C8h]
  _QWORD v309[4]; // [rsp+200h] [rbp-2C0h] BYREF
  __m128i v310; // [rsp+220h] [rbp-2A0h] BYREF
  __int64 v311; // [rsp+230h] [rbp-290h] BYREF
  __int64 *v312; // [rsp+270h] [rbp-250h] BYREF
  __int64 v313; // [rsp+278h] [rbp-248h]
  _BYTE v314[64]; // [rsp+280h] [rbp-240h] BYREF
  __m128i v315; // [rsp+2C0h] [rbp-200h] BYREF
  __int64 v316; // [rsp+2D0h] [rbp-1F0h] BYREF
  unsigned __int64 *v317; // [rsp+310h] [rbp-1B0h] BYREF
  __int64 v318; // [rsp+318h] [rbp-1A8h]
  _BYTE v319[64]; // [rsp+320h] [rbp-1A0h] BYREF
  __m128i v320; // [rsp+360h] [rbp-160h] BYREF
  _QWORD v321[10]; // [rsp+370h] [rbp-150h] BYREF
  __int64 v322; // [rsp+3C0h] [rbp-100h]
  unsigned __int64 v323; // [rsp+3C8h] [rbp-F8h]
  __int64 v324; // [rsp+3D0h] [rbp-F0h]
  __int64 v325; // [rsp+3D8h] [rbp-E8h]
  __int16 v326; // [rsp+3F8h] [rbp-C8h]
  __int64 v327; // [rsp+400h] [rbp-C0h]
  __int64 v328; // [rsp+408h] [rbp-B8h]
  __m128i *v329; // [rsp+410h] [rbp-B0h]
  __m128i *v330; // [rsp+418h] [rbp-A8h]
  __int64 v331[5]; // [rsp+420h] [rbp-A0h] BYREF
  int v332; // [rsp+448h] [rbp-78h]
  __int64 v333; // [rsp+450h] [rbp-70h]
  __int64 v334; // [rsp+458h] [rbp-68h]
  __m128i *v335; // [rsp+460h] [rbp-60h]
  __int64 v336; // [rsp+468h] [rbp-58h]
  _OWORD v337[5]; // [rsp+470h] [rbp-50h] BYREF

  v258 = a2;
  v13 = sub_15F2050(a2);
  v14 = sub_1632FA0(v13);
  v15 = *(_QWORD **)(a4 + 24);
  v16 = *(_QWORD **)(a4 + 16);
  v17 = v14;
  v275 = *(char **)(a4 + 8);
  if ( v16 == v15 )
  {
    v23 = *(_QWORD *)a4;
    v263 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
    goto LABEL_15;
  }
  v262 = 1;
  v18 = *(_QWORD **)(a4 + 16);
  v251 = 0;
  v268 = 0;
  do
  {
    v19 = sub_1648700(v18[2] & 0xFFFFFFFFFFFFFFF8LL);
    v20 = *((_BYTE *)v19 + 16);
    if ( v20 != 78 || (v49 = *(v19 - 3), *(_BYTE *)(v49 + 16)) || (*(_BYTE *)(v49 + 33) & 0x20) == 0 )
    {
      if ( *v18 == *v16 && v275 == (char *)v18[1] )
      {
        if ( v20 <= 0x17u )
          goto LABEL_9;
        if ( v20 != 54 )
        {
          if ( v20 != 55 )
            goto LABEL_9;
          v19 = (_QWORD *)*(v19 - 6);
        }
        v52 = *v19;
        if ( !v52 )
        {
LABEL_9:
          v262 = 0;
          goto LABEL_10;
        }
        if ( *(_BYTE *)(v52 + 8) == 11 )
        {
          v53 = *(_DWORD *)(v52 + 8);
          if ( (v53 & 0x700) != 0 || v53 >> 11 > (unsigned __int64)&v275[-*v16] )
            goto LABEL_10;
          v54 = v268;
          if ( v268 )
          {
            if ( *(_DWORD *)(v52 + 8) >> 8 > *(_DWORD *)(v268 + 8) >> 8 )
              v54 = v52;
            v268 = v54;
          }
          else
          {
            v268 = v52;
          }
        }
        if ( v251 && v52 != v251 )
          goto LABEL_9;
        v251 = v52;
      }
    }
LABEL_10:
    v18 += 3;
  }
  while ( v15 != v18 );
  v21 = v251;
  if ( !v262 )
    v21 = v268;
  v269 = (_QWORD *)v21;
  if ( v21 )
  {
    v22 = sub_12BE0A0(v17, v21);
    v23 = *(_QWORD *)a4;
    v263 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
    if ( v22 >= v263 )
      goto LABEL_54;
  }
  else
  {
    v23 = *(_QWORD *)a4;
    v263 = (unsigned __int64)&v275[-*(_QWORD *)a4];
  }
LABEL_15:
  v270 = a4;
  v24 = v17;
  v25 = *(_QWORD *)(a2 + 56);
  v26 = v25;
  if ( v23 )
    goto LABEL_25;
  while ( 2 )
  {
    if ( sub_12BE0A0(v24, v26) == v263 )
    {
      a4 = v270;
      v269 = (_QWORD *)sub_1A210B0(v17, v25);
      goto LABEL_53;
    }
    sub_15A9FE0(v17, v25);
    sub_127FA20(v17, v25);
LABEL_18:
    v27 = (unsigned int)sub_15A9FE0(v17, v25);
    if ( v27 * ((v27 + ((unsigned __int64)(sub_127FA20(v17, v25) + 7) >> 3) - 1) / v27) - v23 < v263 )
    {
LABEL_26:
      a4 = v270;
      goto LABEL_27;
    }
    v28 = *(_BYTE *)(v25 + 8);
    if ( ((v28 - 14) & 0xFD) != 0 )
    {
      if ( v28 != 13 )
        goto LABEL_26;
      v50 = (unsigned __int64 *)sub_15A9930(v17, v25);
      if ( v23 >= *v50 )
        goto LABEL_26;
      v252 = v23 + v263;
      if ( *v50 < v23 + v263 )
        goto LABEL_26;
      v278 = v50;
      src = sub_15A8020((__int64)v50, v23);
      v29 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 8LL * src);
      v23 -= v278[src + 2];
      v51 = sub_12BE0A0(v17, v29);
      if ( v23 >= v51 )
        goto LABEL_26;
      if ( !v23 && v51 <= v263 )
      {
        v48 = (__int64 *)v29;
        a4 = v270;
        if ( v51 != v263 )
        {
          v231 = *(_QWORD *)(v25 + 16);
          v232 = (_QWORD *)(v231 + 8LL * src);
          if ( v252 >= *v278 )
          {
            v234 = v231 + 8LL * *(unsigned int *)(v25 + 12);
            goto LABEL_417;
          }
          v233 = sub_15A8020((__int64)v278, v252);
          if ( src != v233 && v252 == v278[v233 + 2] )
          {
            v234 = *(_QWORD *)(v25 + 16) + 8LL * v233;
LABEL_417:
            v269 = (_QWORD *)sub_1645600(
                               *(_QWORD **)v25,
                               v232,
                               (v234 - (__int64)v232) >> 3,
                               (*(_DWORD *)(v25 + 8) & 0x200) != 0);
            if ( v263 == *(_QWORD *)sub_15A9930(v17, (__int64)v269) )
              goto LABEL_53;
          }
LABEL_27:
          v33 = *(unsigned __int8 **)(v17 + 24);
          v34 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
          v35 = &v33[*(unsigned int *)(v17 + 32)];
          v36 = 8 * v34;
          if ( v35 != v33 )
          {
            v269 = 0;
            goto LABEL_30;
          }
LABEL_118:
          v82 = (__int64 *)sub_1643330(*a1);
          v269 = sub_1645D80(v82, v34);
          goto LABEL_33;
        }
LABEL_427:
        v269 = (_QWORD *)sub_1A210B0(v17, (__int64)v48);
        goto LABEL_53;
      }
      if ( v51 < v23 + v263 )
        goto LABEL_26;
      goto LABEL_24;
    }
    v29 = *(_QWORD *)(v25 + 24);
    v276 = (unsigned int)sub_15A9FE0(v17, v29);
    v30 = (v276 + ((unsigned __int64)(sub_127FA20(v17, v29) + 7) >> 3) - 1) / v276 * v276;
    v31 = v23 % v30;
    if ( v23 / v30 >= *(_QWORD *)(v25 + 32) )
      goto LABEL_26;
    v23 %= v30;
    if ( v31 || v30 > v263 )
    {
      if ( v263 + v31 > v30 )
        goto LABEL_26;
LABEL_24:
      v25 = v29;
      v24 = v17;
      v26 = v29;
      if ( !v23 )
        continue;
LABEL_25:
      v32 = (unsigned int)sub_15A9FE0(v24, v26);
      if ( v23 > v32 * ((v32 + ((unsigned __int64)(sub_127FA20(v17, v25) + 7) >> 3) - 1) / v32) )
        goto LABEL_26;
      goto LABEL_18;
    }
    break;
  }
  v48 = (__int64 *)v29;
  a4 = v270;
  if ( v30 == v263 )
    goto LABEL_427;
  if ( v263 / v30 * v30 != v263 )
    goto LABEL_27;
  v269 = sub_1645D80(v48, v263 / v30);
LABEL_53:
  if ( !v269 )
    goto LABEL_27;
LABEL_54:
  if ( *((_BYTE *)v269 + 8) == 14 && *(_BYTE *)(*(_QWORD *)v269[2] + 8LL) == 11 )
  {
    v33 = *(unsigned __int8 **)(v17 + 24);
    v34 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
    v35 = &v33[*(unsigned int *)(v17 + 32)];
    v36 = 8 * v34;
    if ( v33 != v35 )
    {
LABEL_30:
      while ( *v33 != v36 )
      {
        if ( v35 == ++v33 )
          goto LABEL_32;
      }
      v269 = (_QWORD *)sub_1644C60(*a1, 8 * (int)v34);
LABEL_32:
      if ( !v269 )
      {
        v34 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
        goto LABEL_118;
      }
    }
  }
LABEL_33:
  v37 = sub_127FA20(v17, (__int64)v269);
  if ( v37 <= 0xFFFFFF && v37 == ((sub_127FA20(v17, (__int64)v269) + 7) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v55 = sub_1644C60((_QWORD *)*v269, v37);
    if ( sub_1A1E350(v17, (__int64)v269, v55) )
    {
      v56 = sub_1A1E350(v17, v55, (__int64)v269);
      if ( v56 )
      {
        v57 = *(unsigned __int64 **)(a4 + 16);
        v58 = *(unsigned __int64 **)(a4 + 24);
        if ( v57 == v58 )
        {
          v224 = *(unsigned __int8 **)(v17 + 24);
          v225 = &v224[*(unsigned int *)(v17 + 32)];
          if ( v224 == v225 )
          {
LABEL_425:
            v56 = 0;
          }
          else
          {
            while ( v37 != *v224 )
            {
              if ( v225 == ++v224 )
                goto LABEL_425;
            }
          }
          v320.m128i_i8[0] = v56;
LABEL_88:
          v60 = *(unsigned __int64 ***)(a4 + 32);
          v61 = &v60[*(unsigned int *)(a4 + 40)];
          if ( v60 == v61 )
          {
LABEL_394:
            v87 = v320.m128i_i8[0];
            if ( v320.m128i_i8[0] )
            {
              v83 = 0;
              goto LABEL_132;
            }
          }
          else
          {
            while ( sub_1A1EC30(*v60, *(_QWORD *)a4, (__int64)v269, v17, &v320) )
            {
              if ( v61 == ++v60 )
                goto LABEL_394;
            }
          }
        }
        else
        {
          v320.m128i_i8[0] = 0;
          v59 = v57;
          while ( sub_1A1EC30(v59, *(_QWORD *)a4, (__int64)v269, v17, &v320) )
          {
            v59 += 3;
            if ( v58 == v59 )
              goto LABEL_88;
          }
        }
      }
    }
  }
  v38 = *(_QWORD **)(a4 + 24);
  v320.m128i_i64[0] = (__int64)v321;
  v320.m128i_i64[1] = 0x400000000LL;
  if ( *(_QWORD **)(a4 + 16) == v38 )
  {
LABEL_213:
    v87 = 0;
    v83 = 0;
    goto LABEL_132;
  }
  v277 = 1;
  v39 = *(_QWORD **)(a4 + 16);
  v40 = 0;
  v264 = 0;
  while ( 2 )
  {
    if ( *v39 == *(_QWORD *)a4 && v39[1] == *(_QWORD *)(a4 + 8) )
    {
      v41 = sub_1648700(v39[2] & 0xFFFFFFFFFFFFFFF8LL);
      v44 = *((_BYTE *)v41 + 16);
      if ( v44 > 0x17u )
      {
        if ( v44 == 54 )
        {
LABEL_43:
          v45 = *v41;
          if ( *(_BYTE *)(v45 + 8) == 16 )
          {
            if ( (unsigned int)v40 >= v320.m128i_i32[3] )
            {
              v257 = v45;
              sub_16CD150((__int64)&v320, v321, 0, 8, v42, v43);
              v40 = v320.m128i_u32[2];
              v45 = v257;
            }
            *(_QWORD *)(v320.m128i_i64[0] + 8 * v40) = v45;
            v40 = (unsigned int)++v320.m128i_i32[2];
            if ( v264 )
            {
              v46 = *(_QWORD *)(v45 + 24) == v264;
              v47 = v277;
              if ( !v46 )
                v47 = 0;
              v277 = v47;
            }
            else
            {
              v264 = *(_QWORD *)(v45 + 24);
            }
          }
        }
        else if ( v44 == 55 )
        {
          v41 = (_QWORD *)*(v41 - 6);
          goto LABEL_43;
        }
      }
    }
    v39 += 3;
    if ( v38 != v39 )
      continue;
    break;
  }
  v62 = (char *)v320.m128i_i64[0];
  if ( !(_DWORD)v40 )
  {
LABEL_130:
    if ( v62 != (char *)v321 )
    {
      v87 = 0;
      v83 = 0;
      _libc_free((unsigned __int64)v62);
      goto LABEL_132;
    }
    goto LABEL_213;
  }
  if ( v277 )
  {
    v81 = 8;
    if ( (_DWORD)v40 != 1 )
      v320.m128i_i32[2] = 1;
    goto LABEL_121;
  }
  v63 = 8LL * (unsigned int)v40;
  v64 = (_QWORD *)(v320.m128i_i64[0] + v63);
  v65 = v63 >> 3;
  v66 = v63 >> 5;
  if ( !v66 )
  {
    v67 = (_QWORD *)v320.m128i_i64[0];
LABEL_401:
    if ( v65 != 2 )
    {
      if ( v65 != 3 )
      {
        if ( v65 != 1 )
        {
LABEL_404:
          v67 = v64;
LABEL_405:
          v71 = (char *)v67;
          goto LABEL_109;
        }
LABEL_423:
        if ( *(_BYTE *)(*(_QWORD *)(*v67 + 24LL) + 8LL) != 11 )
          goto LABEL_102;
        goto LABEL_404;
      }
      if ( *(_BYTE *)(*(_QWORD *)(*v67 + 24LL) + 8LL) != 11 )
        goto LABEL_102;
      ++v67;
    }
    if ( *(_BYTE *)(*(_QWORD *)(*v67 + 24LL) + 8LL) != 11 )
      goto LABEL_102;
    ++v67;
    goto LABEL_423;
  }
  v67 = (_QWORD *)v320.m128i_i64[0];
  v68 = v320.m128i_i64[0] + 32 * v66;
  while ( *(_BYTE *)(*(_QWORD *)(*v67 + 24LL) + 8LL) == 11 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v67[1] + 24LL) + 8LL) != 11 )
    {
      ++v67;
      break;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v67[2] + 24LL) + 8LL) != 11 )
    {
      v67 += 2;
      break;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v67[3] + 24LL) + 8LL) != 11 )
    {
      v67 += 3;
      break;
    }
    v67 += 4;
    if ( (_QWORD *)v68 == v67 )
    {
      v65 = v64 - v67;
      goto LABEL_401;
    }
  }
LABEL_102:
  if ( v64 == v67 )
    goto LABEL_405;
  v69 = v67 + 1;
  if ( v64 == v67 + 1 )
    goto LABEL_405;
  do
  {
    if ( *(_BYTE *)(*(_QWORD *)(*v69 + 24LL) + 8LL) == 11 )
      *v67++ = *v69;
    ++v69;
  }
  while ( v64 != v69 );
  v62 = (char *)v320.m128i_i64[0];
  v70 = v320.m128i_i64[0] + 8LL * v320.m128i_u32[2] - (_QWORD)v64;
  v71 = (char *)v67 + v70;
  if ( v64 != (_QWORD *)(v320.m128i_i64[0] + 8LL * v320.m128i_u32[2]) )
  {
    memmove(v67, v64, v70);
    v62 = (char *)v320.m128i_i64[0];
  }
LABEL_109:
  v72 = (v71 - v62) >> 3;
  v320.m128i_i32[2] = v72;
  if ( !(_DWORD)v72 )
    goto LABEL_130;
  v73 = 8LL * (unsigned int)v72;
  v279 = v62;
  v74 = &v62[v73];
  _BitScanReverse64(&v75, v73 >> 3);
  sub_1A1A7C0(v62, (__int64 *)&v62[v73], 2LL * (int)(63 - (v75 ^ 0x3F)), v17);
  if ( (unsigned __int64)v73 > 0x80 )
  {
    v219 = v279 + 128;
    sub_1A1A430(v279, v279 + 128);
    if ( v74 != v279 + 128 )
    {
      do
      {
        v220 = *(_QWORD *)v219;
        v221 = *((_QWORD *)v219 - 1);
        v222 = v219 - 8;
        if ( *(_QWORD *)(v221 + 32) <= *(_QWORD *)(*(_QWORD *)v219 + 32LL) )
        {
          v223 = v219;
        }
        else
        {
          do
          {
            *((_QWORD *)v222 + 1) = v221;
            v223 = v222;
            v221 = *((_QWORD *)v222 - 1);
            v222 -= 8;
          }
          while ( *(_QWORD *)(v220 + 32) < *(_QWORD *)(v221 + 32) );
        }
        v219 += 8;
        *(_QWORD *)v223 = v220;
      }
      while ( v74 != v219 );
    }
  }
  else
  {
    sub_1A1A430(v279, v74);
  }
  v62 = (char *)v320.m128i_i64[0];
  v76 = 8LL * v320.m128i_u32[2];
  v77 = v320.m128i_i64[0];
  v78 = (char *)(v320.m128i_i64[0] + v76);
  if ( v320.m128i_i64[0] != v320.m128i_i64[0] + v76 )
  {
    while ( 1 )
    {
      v80 = (char *)v77;
      v77 += 8;
      if ( v78 == (char *)v77 )
        break;
      v79 = *(_QWORD *)(*(_QWORD *)(v77 - 8) + 32LL);
      if ( v79 < *(_QWORD *)(*(_QWORD *)v77 + 32LL) )
      {
        if ( v78 == v80 )
        {
          v77 = v320.m128i_i64[0] + v76;
        }
        else
        {
          v226 = v80 + 16;
          if ( v78 != v80 + 16 )
          {
            while ( 1 )
            {
              if ( *(_QWORD *)(*v226 + 32LL) <= v79 )
              {
                *((_QWORD *)v80 + 1) = *v226;
                v80 += 8;
              }
              if ( v78 == (char *)++v226 )
                break;
              v79 = *(_QWORD *)(*(_QWORD *)v80 + 32LL);
            }
            v227 = 8LL * v320.m128i_u32[2];
            v228 = &v62[v227];
            v229 = v227 - v76;
            v77 = (__int64)&v80[v229 + 8];
            if ( v78 != v228 )
            {
              v290 = v62;
              memmove(v80 + 8, v78, v229);
              v62 = v290;
            }
          }
        }
        break;
      }
    }
  }
  v320.m128i_i32[2] = (v77 - (__int64)v62) >> 3;
  v81 = 8LL * v320.m128i_u32[2];
LABEL_121:
  v253 = &v62[v81];
  if ( &v62[v81] == v62 )
    goto LABEL_130;
  v265 = (__int64 *)v62;
  while ( 1 )
  {
    v83 = *v265;
    v84 = sub_127FA20(v17, *(_QWORD *)(*v265 + 24));
    if ( (v84 & 7) == 0 )
      break;
LABEL_128:
    if ( v253 == (char *)++v265 )
    {
      v62 = (char *)v320.m128i_i64[0];
      goto LABEL_130;
    }
  }
  v85 = *(__int64 **)(a4 + 16);
  v86 = v84 >> 3;
  v280 = *(__int64 **)(a4 + 24);
  if ( v85 != v280 )
  {
    while ( sub_1A21230((__int64 *)a4, v85, v83, v86, v17) )
    {
      v85 += 3;
      if ( v280 == v85 )
        goto LABEL_287;
    }
    goto LABEL_128;
  }
LABEL_287:
  v180 = *(__int64 ***)(a4 + 32);
  v288 = &v180[*(unsigned int *)(a4 + 40)];
  if ( v180 != v288 )
  {
    v181 = *(__int64 ***)(a4 + 32);
    while ( sub_1A21230((__int64 *)a4, *v181, v83, v86, v17) )
    {
      if ( v288 == ++v181 )
        goto LABEL_325;
    }
    goto LABEL_128;
  }
LABEL_325:
  if ( (_QWORD *)v320.m128i_i64[0] == v321 )
  {
    v269 = (_QWORD *)v83;
    v87 = 0;
  }
  else
  {
    _libc_free(v320.m128i_u64[0]);
    v87 = 0;
    v269 = (_QWORD *)v83;
  }
LABEL_132:
  if ( v269 == *(_QWORD **)(a2 + 56) )
  {
    v100 = *(_QWORD *)a4;
    if ( !*(_QWORD *)a4 )
    {
      v291 = a2;
      v99 = a2;
      goto LABEL_144;
    }
  }
  v88 = (unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1;
  if ( !v88 )
    v88 = sub_15A9FE0(v17, *(_QWORD *)(a2 + 56));
  v89 = (*(_QWORD *)a4 | v88) & -(*(_QWORD *)a4 | v88);
  v90 = sub_15A9FE0(v17, (__int64)v269);
  v91 = 0;
  if ( v90 < (unsigned int)v89 )
    v91 = v89;
  v92 = *(_DWORD *)(*(_QWORD *)a2 + 8LL);
  LOWORD(v316) = 268;
  v266 = v91;
  v281 = v92 >> 8;
  v297 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a4 + 16) - *(_QWORD *)(a3 + 8)) >> 3);
  v315.m128i_i64[0] = (__int64)&v297;
  v300 = (__m128i *)sub_1649960(a2);
  v310.m128i_i64[0] = (__int64)&v300;
  v310.m128i_i64[1] = (__int64)".sroa.";
  v301 = v93;
  LOWORD(v311) = 773;
  sub_14EC200(&v320, &v310, &v315);
  v94 = sub_1648A60(64, 1u);
  v95 = (__int64)v94;
  if ( v94 )
  {
    v96 = v281;
    v282 = v94;
    sub_15F8A50((__int64)v94, v269, v96, 0, v266, (__int64)&v320, a2);
    v95 = (__int64)v282;
  }
  v291 = v95;
  v97 = (__m128i *)(v95 + 48);
  v98 = *(_QWORD *)(a2 + 48);
  v320.m128i_i64[0] = v98;
  if ( v98 )
  {
    v283 = (__m128i *)(v95 + 48);
    v271 = v95;
    sub_1623A60((__int64)&v320, v98, 2);
    v97 = v283;
    if ( v283 == &v320 )
    {
      if ( v320.m128i_i64[0] )
        sub_161E7C0((__int64)&v320, v320.m128i_i64[0]);
      goto LABEL_143;
    }
    v95 = v271;
    v199 = *(_QWORD *)(v271 + 48);
    if ( !v199 )
    {
LABEL_323:
      v200 = (unsigned __int8 *)v320.m128i_i64[0];
      *(_QWORD *)(v95 + 48) = v320.m128i_i64[0];
      if ( v200 )
        sub_1623210((__int64)&v320, v200, (__int64)v97);
      goto LABEL_143;
    }
LABEL_322:
    v274 = v95;
    v289 = v97;
    sub_161E7C0((__int64)v97, v199);
    v95 = v274;
    v97 = v289;
    goto LABEL_323;
  }
  if ( v97 != &v320 )
  {
    v199 = *(_QWORD *)(v95 + 48);
    if ( v199 )
      goto LABEL_322;
  }
LABEL_143:
  v99 = v291;
  v100 = *(_QWORD *)a4;
  v258 = v291;
LABEL_144:
  v310.m128i_i64[0] = 0;
  v310.m128i_i64[1] = 1;
  v237 = *((_DWORD *)a1 + 90);
  v101 = (unsigned __int64 *)&v311;
  do
    *v101++ = -8;
  while ( v101 != (unsigned __int64 *)&v312 );
  v315.m128i_i64[0] = 0;
  v315.m128i_i64[1] = 1;
  v312 = (__int64 *)v314;
  v313 = 0x800000000LL;
  v102 = (unsigned __int64 *)&v316;
  do
    *v102++ = -8;
  while ( v102 != (unsigned __int64 *)&v317 );
  v320.m128i_i64[0] = v17;
  v317 = (unsigned __int64 *)v319;
  v103 = a1[1];
  v318 = 0x800000000LL;
  v104 = (__int64)a1[3];
  v321[0] = v103;
  v105 = *(_QWORD *)(a4 + 8);
  v320.m128i_i64[1] = v104;
  v321[1] = a3;
  v321[6] = v105;
  v106 = 0;
  v321[2] = a1;
  v321[3] = a2;
  v321[4] = v99;
  v321[5] = v100;
  v107 = *(_QWORD *)(v99 + 56);
  v321[7] = v107;
  if ( v87 )
  {
    v201 = sub_127FA20(v17, v107);
    v202 = (_QWORD *)sub_16498A0(v258);
    v106 = sub_1644C60(v202, v201);
  }
  v321[8] = v106;
  v321[9] = v83;
  if ( v83 )
  {
    v322 = *(_QWORD *)(v83 + 24);
    v108 = (unsigned __int64)sub_127FA20(v17, v322) >> 3;
  }
  else
  {
    v322 = 0;
    v108 = 0;
  }
  v323 = v108;
  v326 = 0;
  v329 = &v310;
  v324 = 0;
  v325 = 0;
  v327 = 0;
  v328 = 0;
  v330 = &v315;
  v300 = &v302;
  v301 = 0;
  v302.m128i_i8[0] = 0;
  v109 = sub_16498A0(v258);
  v331[4] = 0;
  v331[3] = v109;
  v335 = (__m128i *)v337;
  memset(v331, 0, 24);
  v332 = 0;
  v333 = 0;
  v334 = 0;
  if ( v300 == &v302 )
  {
    a5 = (__m128)_mm_load_si128(&v302);
    v337[0] = a5;
  }
  else
  {
    v335 = v300;
    *(_QWORD *)&v337[0] = v302.m128i_i64[0];
  }
  v112 = *(unsigned __int64 ***)(a4 + 32);
  v336 = v301;
  v113 = &v112[*(unsigned int *)(a4 + 40)];
  if ( v113 == v112 )
  {
    v238 = 1;
  }
  else
  {
    v114 = 1;
    do
    {
      v115 = *v112++;
      v114 &= sub_1A333C0((__int64)&v320, v115, a5, a6, a7, a8, v110, v111, a11, a12);
    }
    while ( v113 != v112 );
    v238 = v114;
  }
  v116 = *(unsigned __int64 **)(a4 + 16);
  v117 = *(unsigned __int64 **)(a4 + 24);
  if ( v117 != v116 )
  {
    v118 = v238;
    do
    {
      v119 = v116;
      v116 += 3;
      v118 &= sub_1A333C0((__int64)&v320, v119, a5, a6, a7, a8, v110, v111, a11, a12);
    }
    while ( v117 != v116 );
    v238 = v118;
  }
  v267 = v312;
  v240 = &v312[(unsigned int)v313];
  if ( v240 != v312 )
  {
LABEL_163:
    v120 = *v267;
    v241 = (__int64)a1[1];
    v248 = (__int64)a1[3];
    v121 = sub_15F2050(*v267);
    v243 = sub_1632FA0(v121);
    if ( *(_BYTE *)(*(_QWORD *)v120 + 8LL) != 15 )
      goto LABEL_192;
    v122 = *(_QWORD *)(v120 + 40);
    v295 = 0x400000000LL;
    v259 = v122;
    v294 = (__m128i *)v296;
    sub_1A247B0(v120, (__int64)&v294);
    v297 = (unsigned __int64)v299;
    v298 = 0x400000000LL;
    sub_1A24B00(v120, (__int64)&v297);
    if ( (_DWORD)v298 )
    {
      v167 = (__int64 *)v297;
      v256 = 0;
      v272 = 0;
      v286 = (__int64 *)(v297 + 8LL * (unsigned int)v298);
      do
      {
        while ( 1 )
        {
          v173 = *v167;
          v292[0] = 0;
          if ( sub_15F32D0(v173) )
            goto LABEL_176;
          if ( (*(_BYTE *)(v173 + 18) & 1) != 0 )
            goto LABEL_176;
          if ( v259 != *(_QWORD *)(v173 + 40) )
            goto LABEL_176;
          v174 = sub_1A1E4F0(v243, v248, (__int64 *)v173, v120, v292);
          if ( !v174 )
            goto LABEL_176;
          if ( v292[0] )
            break;
          v256 = v174;
          v175 = 1 << (*(unsigned __int16 *)(v173 + 18) >> 1) >> 1;
          if ( v272 >= v175 )
            v175 = v272;
          ++v167;
          v272 = v175;
          if ( v286 == v167 )
            goto LABEL_274;
        }
        v168 = sub_16498A0(v173);
        v307 = v309;
        v300 = 0;
        v302.m128i_i64[1] = v168;
        v303 = 0;
        v304 = 0;
        v305 = 0;
        v306 = 0;
        v308 = 0;
        LOBYTE(v309[0]) = 0;
        v301 = *(_QWORD *)(v173 + 40);
        v302.m128i_i64[0] = v173 + 24;
        v169 = *(__int64 **)(v173 + 48);
        v293[0] = v169;
        if ( v169 )
        {
          sub_1623A60((__int64)v293, (__int64)v169, 2);
          if ( v300 )
            sub_161E7C0((__int64)&v300, (__int64)v300);
          v300 = (__m128i *)v293[0];
          if ( v293[0] )
            sub_1623210((__int64)v293, (unsigned __int8 *)v293[0], (__int64)&v300);
        }
        v170 = sub_1A1C950(v243, (__int64 *)&v300, v292[0], *(_QWORD *)v173);
        sub_164D160(v173, (__int64)v170, a5, a6, a7, a8, v171, v172, a11, a12);
        sub_15F20C0((_QWORD *)v173);
        if ( v307 != v309 )
          j_j___libc_free_0(v307, v309[0] + 1LL);
        if ( v300 )
          sub_161E7C0((__int64)&v300, (__int64)v300);
        ++v167;
      }
      while ( v286 != v167 );
LABEL_274:
      v123 = v256;
      if ( !v241 )
        goto LABEL_275;
    }
    else
    {
      if ( !(_DWORD)v295 || !v241 )
        goto LABEL_176;
      v272 = 0;
      v123 = 0;
    }
    v124 = &v302;
    v300 = 0;
    v301 = 1;
    do
    {
      v124->m128i_i64[0] = -8;
      ++v124;
    }
    while ( v124 != (__m128i *)v309 );
    v284 = (__int64 *)v294;
    v235 = (unsigned __int8 *)v294 + 8 * (unsigned int)v295;
    if ( v294 == (__m128i *)v235 )
      goto LABEL_312;
    while ( 1 )
    {
      v254 = *v284;
      v125 = *v284;
      v246 = *(_QWORD *)(*v284 - 48);
      if ( sub_15F32D0(*v284) || (*(_BYTE *)(v125 + 18) & 1) != 0 || v259 != *(_QWORD *)(v125 + 40) )
        goto LABEL_174;
      v182 = *(__int64 **)(v125 - 24);
      v244 = v120;
      v183 = v120 + 24;
      v184 = (__int64 *)((unsigned __int64)(sub_127FA20(v243, **(_QWORD **)(v125 - 48)) + 7) >> 3);
      while ( v254 != v183 - 24 )
      {
        v185 = *(_BYTE *)(v183 - 8);
        if ( v185 == 54 )
        {
          v186 = *(_QWORD *)(v183 - 24);
          goto LABEL_297;
        }
        if ( v185 == 55 )
        {
          v186 = **(_QWORD **)(v183 - 72);
LABEL_297:
          v187 = sub_127FA20(v243, v186);
          v188 = *(__int64 **)(v183 - 48);
          memset(&v293[2], 0, 24);
          v293[0] = v188;
          v293[1] = (__int64 *)((unsigned __int64)(v187 + 7) >> 3);
          v292[0] = v182;
          v292[1] = v184;
          memset(&v292[2], 0, 24);
          if ( (unsigned __int8)sub_134CB50(v248, (__int64)v292, (__int64)v293) )
            goto LABEL_174;
          goto LABEL_298;
        }
        if ( (unsigned __int8)sub_15F3040(v183 - 24) || (unsigned __int8)sub_15F2ED0(v183 - 24) )
          goto LABEL_174;
LABEL_298:
        v183 = *(_QWORD *)(v183 + 8);
        if ( !v183 )
          BUG();
      }
      v120 = v244;
      v189 = *(_QWORD *)(v244 + 40);
      if ( (*(_DWORD *)(v244 + 20) & 0xFFFFFFF) != 0 )
      {
        v190 = 0;
        v191 = 8LL * (*(_DWORD *)(v244 + 20) & 0xFFFFFFF);
        while ( 1 )
        {
          v192 = (*(_BYTE *)(v244 + 23) & 0x40) != 0
               ? *(_QWORD *)(v244 - 8)
               : v244 - 24LL * (*(_DWORD *)(v244 + 20) & 0xFFFFFFF);
          v193 = *(_QWORD *)(v190 + v192 + 24LL * *(unsigned int *)(v244 + 56) + 8);
          v194 = sub_157EBA0(v193);
          if ( (unsigned int)sub_15F4D60(v194) != 1 )
            break;
          v195 = sub_157EBA0(v193);
          if ( v189 != sub_15F4DF0(v195, 0) )
            break;
          v190 += 8;
          if ( v191 == v190 )
            goto LABEL_308;
        }
LABEL_174:
        if ( (v301 & 1) == 0 )
          j___libc_free_0(v302.m128i_i64[0]);
LABEL_176:
        v123 = 0;
        goto LABEL_177;
      }
LABEL_308:
      v123 = sub_1A29B80(v241, v246, v259, (__int64)&v300);
      if ( !v123 )
        goto LABEL_174;
      v196 = 1 << (*(unsigned __int16 *)(v254 + 18) >> 1) >> 1;
      if ( v272 >= v196 )
        v196 = v272;
      ++v284;
      v272 = v196;
      if ( v235 == (unsigned __int8 *)v284 )
      {
LABEL_312:
        if ( (v301 & 1) == 0 )
          j___libc_free_0(v302.m128i_i64[0]);
LABEL_275:
        if ( !v123 )
          goto LABEL_176;
        v176 = 0;
        v287 = (void *)(8LL * (*(_DWORD *)(v120 + 20) & 0xFFFFFFF));
        if ( (*(_DWORD *)(v120 + 20) & 0xFFFFFFF) != 0 )
        {
          v261 = v123;
          while ( 1 )
          {
            v177 = (*(_BYTE *)(v120 + 23) & 0x40) != 0
                 ? *(_QWORD *)(v120 - 8)
                 : v120 - 24LL * (*(_DWORD *)(v120 + 20) & 0xFFFFFFF);
            v178 = sub_157EBA0(*(_QWORD *)((char *)v176 + 24 * *(unsigned int *)(v120 + 56) + v177 + 8));
            v179 = *(_QWORD *)(v177 + 3LL * (_QWORD)v176);
            if ( v179 == v178
              || (unsigned __int8)sub_15F3040(v178)
              || sub_15F3330(v178)
              || (unsigned int)sub_15F4D60(v178) != 1 && !(unsigned __int8)sub_13F86A0(v179, v272, v243, v178, 0) )
            {
              goto LABEL_176;
            }
            v176 = (char *)v176 + 8;
            if ( v287 == v176 )
            {
              v123 = v261;
              break;
            }
          }
        }
LABEL_177:
        if ( (_BYTE *)v297 != v299 )
          _libc_free(v297);
        if ( v294 != (__m128i *)v296 )
          _libc_free((unsigned __int64)v294);
        if ( !v123 )
          goto LABEL_192;
        if ( v240 == ++v267 )
          break;
        goto LABEL_163;
      }
    }
  }
  v242 = (__int64 *)&v317[(unsigned int)v318];
  if ( v242 == (__int64 *)v317 )
  {
LABEL_231:
    if ( !v238 )
      goto LABEL_193;
    v148 = *(__int64 ***)(a3 + 296);
    v149 = &v148[*(unsigned int *)(a3 + 304)];
    while ( v149 != v148 )
    {
      while ( 1 )
      {
        v150 = *v148;
        v151 = **v148;
        if ( *(_BYTE *)(v151 + 16) <= 0x17u )
          v151 = 0;
        v300 = (__m128i *)v151;
        sub_1649990((__int64)v150);
        if ( v300 )
        {
          if ( (unsigned __int8)sub_1AE9990(v300, 0) )
            break;
        }
        if ( v149 == ++v148 )
          goto LABEL_240;
      }
      ++v148;
      sub_1A2EDE0((__int64)(a1 + 26), (__int64 *)&v300);
    }
LABEL_240:
    if ( !(_DWORD)v313 )
    {
      v152 = (unsigned int)v318;
      if ( (_DWORD)v318 )
        goto LABEL_242;
      v230 = a1[63];
      if ( v230 == (_BYTE *)a1[64] )
      {
        sub_186B0F0((__int64)(a1 + 62), v230, &v291);
        v141 = v291;
      }
      else
      {
        v141 = v291;
        if ( v230 )
        {
          *(_QWORD *)v230 = v291;
          v230 = a1[63];
        }
        a1[63] = v230 + 8;
      }
      goto LABEL_200;
    }
    v203 = (unsigned __int64 *)v312;
    v204 = (__int64)(a1 + 65);
    v205 = (unsigned __int64 *)&v312[(unsigned int)v313];
    while ( 1 )
    {
      v213 = *v203;
      v214 = *((_DWORD *)a1 + 136);
      v297 = *v203;
      if ( !v214 )
        break;
      v206 = v214 - 1;
      v207 = a1[66];
      v208 = 1;
      m128i_i64 = 0;
      v210 = (v214 - 1) & (((unsigned int)v213 >> 9) ^ ((unsigned int)v213 >> 4));
      v211 = &v207[v210];
      v212 = *v211;
      if ( v213 == *v211 )
      {
LABEL_336:
        if ( v205 == ++v203 )
          goto LABEL_346;
      }
      else
      {
        while ( v212 != -8 )
        {
          if ( v212 != -16 || m128i_i64 )
            v211 = m128i_i64;
          v210 = v206 & (v208 + v210);
          v212 = v207[v210];
          if ( v213 == v212 )
            goto LABEL_336;
          ++v208;
          m128i_i64 = v211;
          v211 = &v207[v210];
        }
        v218 = *((_DWORD *)a1 + 134);
        if ( !m128i_i64 )
          m128i_i64 = v211;
        a1[65] = (_QWORD *)((char *)a1[65] + 1);
        v215 = v218 + 1;
        if ( 4 * (v218 + 1) < 3 * v214 )
        {
          if ( v214 - *((_DWORD *)a1 + 135) - v215 > v214 >> 3 )
            goto LABEL_341;
          goto LABEL_340;
        }
LABEL_339:
        v214 *= 2;
LABEL_340:
        sub_176FD40(v204, v214);
        sub_1A27740(v204, (__int64 *)&v297, &v300);
        m128i_i64 = v300->m128i_i64;
        v213 = v297;
        v215 = *((_DWORD *)a1 + 134) + 1;
LABEL_341:
        *((_DWORD *)a1 + 134) = v215;
        if ( *m128i_i64 != -8 )
          --*((_DWORD *)a1 + 135);
        *m128i_i64 = v213;
        v216 = *((unsigned int *)a1 + 140);
        if ( (unsigned int)v216 >= *((_DWORD *)a1 + 141) )
        {
          sub_16CD150((__int64)(a1 + 69), a1 + 71, 0, 8, (int)v207, v206);
          v216 = *((unsigned int *)a1 + 140);
        }
        ++v203;
        a1[69][v216] = v297;
        ++*((_DWORD *)a1 + 140);
        if ( v205 == v203 )
        {
LABEL_346:
          v152 = (unsigned int)v318;
LABEL_242:
          v153 = v317;
          v154 = &v317[v152];
          v155 = (__int64)(a1 + 73);
          if ( v154 == v317 )
          {
LABEL_199:
            sub_1A30BB0((__int64)(a1 + 4), &v291);
            v141 = v291;
            goto LABEL_200;
          }
          while ( 2 )
          {
            v163 = *v153;
            v164 = *((_DWORD *)a1 + 152);
            v297 = *v153;
            if ( !v164 )
            {
              a1[73] = (_QWORD *)((char *)a1[73] + 1);
              goto LABEL_248;
            }
            v156 = v164 - 1;
            v157 = a1[74];
            v158 = 1;
            v159 = 0;
            v160 = (v164 - 1) & (((unsigned int)v163 >> 9) ^ ((unsigned int)v163 >> 4));
            v161 = &v157[v160];
            v162 = *v161;
            if ( v163 == *v161 )
            {
LABEL_245:
              if ( v154 == ++v153 )
                goto LABEL_199;
              continue;
            }
            break;
          }
          while ( v162 != -8 )
          {
            if ( v162 != -16 || v159 )
              v161 = v159;
            v160 = v156 & (v158 + v160);
            v162 = v157[v160];
            if ( v163 == v162 )
              goto LABEL_245;
            ++v158;
            v159 = v161;
            v161 = &v157[v160];
          }
          v217 = *((_DWORD *)a1 + 150);
          if ( !v159 )
            v159 = v161;
          a1[73] = (_QWORD *)((char *)a1[73] + 1);
          v165 = v217 + 1;
          if ( 4 * (v217 + 1) < 3 * v164 )
          {
            if ( v164 - *((_DWORD *)a1 + 151) - v165 <= v164 >> 3 )
            {
LABEL_249:
              sub_1A36E90(v155, v164);
              sub_1A277F0(v155, (__int64 *)&v297, &v300);
              v159 = v300->m128i_i64;
              v163 = v297;
              v165 = *((_DWORD *)a1 + 150) + 1;
            }
            *((_DWORD *)a1 + 150) = v165;
            if ( *v159 != -8 )
              --*((_DWORD *)a1 + 151);
            *v159 = v163;
            v166 = *((unsigned int *)a1 + 156);
            if ( (unsigned int)v166 >= *((_DWORD *)a1 + 157) )
            {
              sub_16CD150((__int64)(a1 + 77), a1 + 79, 0, 8, (int)v157, v156);
              v166 = *((unsigned int *)a1 + 156);
            }
            a1[77][v166] = v297;
            ++*((_DWORD *)a1 + 156);
            goto LABEL_245;
          }
LABEL_248:
          v164 *= 2;
          goto LABEL_249;
        }
      }
    }
    a1[65] = (_QWORD *)((char *)a1[65] + 1);
    goto LABEL_339;
  }
  v255 = (__int64 *)v317;
  while ( 1 )
  {
    v126 = *v255;
    v247 = *v255;
    v127 = *(_QWORD *)(*v255 - 48);
    v249 = (__int64)a1[3];
    v285 = *(void **)(*v255 - 24);
    v128 = sub_15F2050(*v255);
    v129 = sub_1632FA0(v128);
    v297 = (unsigned __int64)v299;
    v298 = 0x400000000LL;
    sub_1A24B00(v126, (__int64)&v297);
    if ( !(_DWORD)v298 )
      break;
    v130 = *(_QWORD *)(v126 + 40);
    v131 = (__int64 *)v297;
    v260 = v130;
    v273 = (__int64 *)(v297 + 8LL * (unsigned int)v298);
    do
    {
      v132 = *v131;
      if ( !(unsigned __int8)sub_13F86A0(v127, 1 << (*(unsigned __int16 *)(*v131 + 18) >> 1) >> 1, v129, *v131, 0) )
        goto LABEL_190;
      if ( !(unsigned __int8)sub_13F86A0(
                               (unsigned __int64)v285,
                               1 << (*(unsigned __int16 *)(v132 + 18) >> 1) >> 1,
                               v129,
                               v132,
                               0) )
        goto LABEL_190;
      v293[0] = 0;
      if ( sub_15F32D0(v132)
        || (*(_BYTE *)(v132 + 18) & 1) != 0
        || v260 != *(_QWORD *)(v132 + 40)
        || !(unsigned __int8)sub_1A1E4F0(v129, v249, (__int64 *)v132, v247, v293) )
      {
        goto LABEL_190;
      }
      if ( v293[0] )
      {
        v143 = sub_16498A0(v132);
        LOBYTE(v309[0]) = 0;
        v302.m128i_i64[1] = v143;
        v300 = 0;
        v303 = 0;
        v304 = 0;
        v305 = 0;
        v306 = 0;
        v307 = v309;
        v308 = 0;
        v301 = *(_QWORD *)(v132 + 40);
        v302.m128i_i64[0] = v132 + 24;
        v144 = *(unsigned __int8 **)(v132 + 48);
        v294 = (__m128i *)v144;
        if ( v144 )
        {
          sub_1623A60((__int64)&v294, (__int64)v144, 2);
          if ( v300 )
            sub_161E7C0((__int64)&v300, (__int64)v300);
          v300 = v294;
          if ( v294 )
            sub_1623210((__int64)&v294, (unsigned __int8 *)v294, (__int64)&v300);
        }
        v145 = sub_1A1C950(v129, (__int64 *)&v300, v293[0], *(_QWORD *)v132);
        sub_164D160(v132, (__int64)v145, a5, a6, a7, a8, v146, v147, a11, a12);
        sub_15F20C0((_QWORD *)v132);
        if ( v307 != v309 )
          j_j___libc_free_0(v307, v309[0] + 1LL);
        if ( v300 )
          sub_161E7C0((__int64)&v300, (__int64)v300);
      }
      ++v131;
    }
    while ( v273 != v131 );
    if ( (_BYTE *)v297 != v299 )
      _libc_free(v297);
    if ( v242 == ++v255 )
      goto LABEL_231;
  }
LABEL_190:
  if ( (_BYTE *)v297 != v299 )
    _libc_free(v297);
LABEL_192:
  sub_1A26A20((__int64)&v310);
  LODWORD(v313) = 0;
  sub_1A26BD0((__int64)&v315);
  LODWORD(v318) = 0;
LABEL_193:
  for ( i = *((unsigned int *)a1 + 90); v237 < (unsigned int)i; *((_DWORD *)a1 + 90) = i )
  {
    v134 = *((_DWORD *)a1 + 86);
    if ( v134 )
    {
      v135 = v134 - 1;
      v136 = a1[41];
      v137 = a1[44][i - 1];
      v138 = v135 & (((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4));
      v139 = &v136[v138];
      v140 = *v139;
      if ( *v139 == v137 )
      {
LABEL_196:
        *v139 = -16;
        --*((_DWORD *)a1 + 84);
        ++*((_DWORD *)a1 + 85);
      }
      else
      {
        v197 = 1;
        while ( v140 != -8 )
        {
          v198 = v197 + 1;
          v138 = v135 & (v197 + v138);
          v139 = &v136[v138];
          v140 = *v139;
          if ( v137 == *v139 )
            goto LABEL_196;
          v197 = v198;
        }
      }
    }
    i = (unsigned int)(*((_DWORD *)a1 + 90) - 1);
  }
  if ( v291 != a2 )
    goto LABEL_199;
  v141 = 0;
LABEL_200:
  if ( v335 != (__m128i *)v337 )
    j_j___libc_free_0(v335, *(_QWORD *)&v337[0] + 1LL);
  if ( v331[0] )
    sub_161E7C0((__int64)v331, v331[0]);
  if ( v317 != (unsigned __int64 *)v319 )
    _libc_free((unsigned __int64)v317);
  if ( (v315.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v316);
  if ( v312 != (__int64 *)v314 )
    _libc_free((unsigned __int64)v312);
  if ( (v310.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v311);
  return v141;
}
