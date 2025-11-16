// Function: sub_672A20
// Address: 0x672a20
//
__int64 __fastcall sub_672A20(unsigned __int64 a1, __int64 a2, __int64 i, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  _BOOL8 v9; // r12
  __int64 v10; // rax
  int v11; // r11d
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  char v14; // bl
  char v15; // r10
  int v16; // eax
  unsigned __int64 v17; // rbx
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  bool v23; // zf
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  char *v28; // rsi
  unsigned int v29; // eax
  int v30; // ebx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  char k; // dl
  __int64 v34; // rdi
  unsigned int v35; // r13d
  char v36; // al
  __int64 v37; // rbx
  _BYTE *v38; // r13
  __int64 *v39; // r14
  __int64 v40; // r12
  __int64 *v41; // r15
  __int64 v42; // rbx
  char v43; // al
  char v44; // al
  __int64 *v45; // r12
  __int64 *v46; // rcx
  char v47; // dl
  char v48; // al
  unsigned __int8 v49; // dl
  _QWORD *v50; // rsi
  __int64 result; // rax
  __int64 v52; // rax
  char m; // dl
  __int64 v54; // rax
  int v55; // eax
  __int64 **v56; // r13
  int v57; // r14d
  unsigned __int64 v58; // rbx
  int v59; // eax
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rax
  int v63; // eax
  unsigned __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdx
  unsigned __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rax
  __m128i v71; // xmm0
  __m128i v72; // xmm2
  __m128i v73; // xmm3
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // rdx
  char v77; // ah
  __int64 v78; // rdx
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rax
  unsigned __int16 v81; // ax
  int v82; // ebx
  _BOOL4 v83; // r9d
  _BOOL4 v84; // ecx
  __int64 v85; // rax
  char v86; // al
  char v87; // dl
  int v88; // eax
  char v89; // al
  __int64 v90; // rbx
  bool v91; // al
  __int64 v92; // rax
  __int64 v93; // rax
  __int16 v94; // ax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  int v100; // eax
  __int64 v101; // rax
  char j; // dl
  int v103; // eax
  int v104; // edi
  int v105; // eax
  __int64 v106; // rdi
  __int64 v107; // rax
  char v108; // al
  char v109; // bl
  __int64 v110; // r14
  unsigned __int8 v111; // al
  unsigned __int8 v112; // cl
  char v113; // dl
  char v114; // al
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 *v119; // r8
  int v120; // r11d
  int v121; // eax
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rdx
  __int64 v125; // rcx
  int v126; // r11d
  unsigned __int16 v127; // ax
  bool v128; // bl
  int v129; // ebx
  unsigned __int64 v130; // rax
  __int64 **v131; // rax
  int v132; // eax
  __int16 v133; // ax
  __int64 v134; // rax
  _QWORD **v135; // rdi
  _QWORD **v136; // rax
  int v137; // eax
  __int64 v138; // rax
  unsigned int v139; // r13d
  __int64 v140; // rax
  __int64 v141; // rax
  int v142; // ebx
  __int64 v143; // rdi
  __int64 v144; // rdx
  __int64 v145; // rcx
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // rbx
  unsigned int v153; // ebx
  int v154; // eax
  __int64 v155; // rdi
  int v156; // eax
  int v157; // eax
  int v158; // eax
  int v159; // eax
  __int64 v160; // rax
  unsigned __int64 v161; // rdi
  char v162; // al
  unsigned __int64 v163; // rbx
  char v164; // al
  unsigned __int8 v165; // bl
  __int64 v166; // rax
  __int64 v167; // rax
  char v168; // dl
  __int64 v169; // rdx
  char v170; // cl
  int v171; // eax
  __int64 v172; // rax
  __int16 v173; // ax
  __int64 v174; // rcx
  __int64 n; // rax
  __int64 *v176; // rdx
  __int64 ii; // rax
  __int64 v178; // rax
  __int64 v179; // rax
  __int64 v180; // rcx
  int v181; // eax
  unsigned __int64 v182; // rax
  __int64 v183; // rcx
  unsigned __int64 v184; // rbx
  __int64 v185; // rdx
  __int64 v186; // rcx
  __int16 v187; // ax
  __int64 v188; // rdx
  __int64 v189; // rcx
  char v190; // dl
  __int64 v191; // rax
  __int64 v192; // rax
  _BOOL4 v193; // edx
  __int64 v194; // rdx
  __int64 v195; // rcx
  __int64 v196; // rdi
  bool v197; // al
  __int64 *v198; // r9
  __int64 v199; // r12
  char v200; // al
  __int64 v201; // rax
  __int64 v202; // rax
  unsigned int v203; // eax
  __int64 v204; // rdx
  __int64 v205; // rcx
  int v206; // eax
  __int64 v207; // r12
  __int16 v208; // ax
  unsigned __int8 v209; // cl
  __int64 v210; // rax
  char v211; // al
  __int64 v212; // rax
  int v213; // eax
  __int64 v214; // rsi
  unsigned int *v215; // rsi
  __int64 v216; // rdx
  __int64 v217; // rcx
  __int64 v218; // rdx
  __int64 v219; // rcx
  __int64 v220; // rax
  __int64 v221; // rax
  char v222; // si
  __int64 v223; // rax
  char v224; // dl
  __int64 v225; // rsi
  __int64 v226; // rax
  int v227; // eax
  __int64 v228; // r8
  __int64 v229; // rax
  __int16 v230; // ax
  __int64 v231; // rdi
  __int64 v232; // rax
  unsigned __int8 v233; // cl
  __int64 v234; // rdx
  char v235; // cl
  __int64 v236; // r12
  __int64 v237; // rdx
  __int64 v238; // rcx
  __int64 v239; // rsi
  __int64 v240; // rax
  char v241; // al
  char v242; // al
  int v243; // eax
  int v244; // eax
  __int64 v245; // [rsp-10h] [rbp-140h]
  unsigned __int16 *v246; // [rsp-8h] [rbp-138h]
  __int64 v247; // [rsp+8h] [rbp-128h]
  unsigned __int64 v248; // [rsp+10h] [rbp-120h]
  char v249; // [rsp+10h] [rbp-120h]
  __int64 v250; // [rsp+10h] [rbp-120h]
  int v251; // [rsp+18h] [rbp-118h]
  unsigned int v252; // [rsp+18h] [rbp-118h]
  __int64 v253; // [rsp+18h] [rbp-118h]
  unsigned int v254; // [rsp+20h] [rbp-110h]
  __int64 *v255; // [rsp+20h] [rbp-110h]
  int v256; // [rsp+20h] [rbp-110h]
  int v257; // [rsp+20h] [rbp-110h]
  __int16 v258; // [rsp+20h] [rbp-110h]
  int v259; // [rsp+20h] [rbp-110h]
  __int64 v260; // [rsp+20h] [rbp-110h]
  unsigned __int64 v261; // [rsp+20h] [rbp-110h]
  int v262; // [rsp+20h] [rbp-110h]
  int v263; // [rsp+20h] [rbp-110h]
  int v264; // [rsp+20h] [rbp-110h]
  int v265; // [rsp+28h] [rbp-108h]
  unsigned __int64 v266; // [rsp+28h] [rbp-108h]
  int v267; // [rsp+28h] [rbp-108h]
  int v268; // [rsp+28h] [rbp-108h]
  unsigned __int64 v269; // [rsp+28h] [rbp-108h]
  int v270; // [rsp+28h] [rbp-108h]
  unsigned __int64 v271; // [rsp+28h] [rbp-108h]
  int v272; // [rsp+28h] [rbp-108h]
  int v273; // [rsp+28h] [rbp-108h]
  unsigned __int64 v274; // [rsp+28h] [rbp-108h]
  int v275; // [rsp+28h] [rbp-108h]
  int v276; // [rsp+28h] [rbp-108h]
  __int64 *v277; // [rsp+28h] [rbp-108h]
  char v278; // [rsp+28h] [rbp-108h]
  __int64 *v279; // [rsp+28h] [rbp-108h]
  int v280; // [rsp+28h] [rbp-108h]
  __int64 v281; // [rsp+28h] [rbp-108h]
  __int64 *v282; // [rsp+28h] [rbp-108h]
  __int64 v283; // [rsp+28h] [rbp-108h]
  __int64 *v284; // [rsp+28h] [rbp-108h]
  __int64 *v285; // [rsp+28h] [rbp-108h]
  __int64 *v286; // [rsp+28h] [rbp-108h]
  unsigned __int16 v287; // [rsp+30h] [rbp-100h]
  bool v288; // [rsp+37h] [rbp-F9h]
  char v289; // [rsp+38h] [rbp-F8h]
  unsigned int v290; // [rsp+3Ch] [rbp-F4h]
  unsigned __int64 v291; // [rsp+40h] [rbp-F0h]
  _BOOL4 v292; // [rsp+48h] [rbp-E8h]
  __int64 v293; // [rsp+48h] [rbp-E8h]
  unsigned __int64 v294; // [rsp+50h] [rbp-E0h]
  int v295; // [rsp+50h] [rbp-E0h]
  _BOOL4 v296; // [rsp+58h] [rbp-D8h]
  int v297; // [rsp+5Ch] [rbp-D4h]
  __int64 *v298; // [rsp+60h] [rbp-D0h]
  int v299; // [rsp+68h] [rbp-C8h]
  unsigned int v300; // [rsp+6Ch] [rbp-C4h]
  unsigned __int64 v301; // [rsp+70h] [rbp-C0h]
  unsigned int v302; // [rsp+78h] [rbp-B8h]
  char v303; // [rsp+78h] [rbp-B8h]
  int v304; // [rsp+80h] [rbp-B0h]
  unsigned int v305; // [rsp+84h] [rbp-ACh]
  unsigned __int64 v306; // [rsp+88h] [rbp-A8h]
  int v307; // [rsp+90h] [rbp-A0h]
  char v308; // [rsp+90h] [rbp-A0h]
  char v309; // [rsp+90h] [rbp-A0h]
  char v310; // [rsp+90h] [rbp-A0h]
  unsigned __int8 v311; // [rsp+90h] [rbp-A0h]
  int v312; // [rsp+98h] [rbp-98h]
  int v313; // [rsp+98h] [rbp-98h]
  int v314; // [rsp+98h] [rbp-98h]
  __int64 *v315; // [rsp+98h] [rbp-98h]
  int v316; // [rsp+98h] [rbp-98h]
  unsigned int v317; // [rsp+98h] [rbp-98h]
  unsigned __int8 v318; // [rsp+98h] [rbp-98h]
  int v319; // [rsp+98h] [rbp-98h]
  char v320; // [rsp+98h] [rbp-98h]
  int v321; // [rsp+98h] [rbp-98h]
  int v322; // [rsp+98h] [rbp-98h]
  __int64 v323; // [rsp+ACh] [rbp-84h] BYREF
  unsigned int v324; // [rsp+B4h] [rbp-7Ch] BYREF
  unsigned int v325; // [rsp+B8h] [rbp-78h] BYREF
  _BOOL4 v326; // [rsp+BCh] [rbp-74h] BYREF
  unsigned __int64 v327; // [rsp+C0h] [rbp-70h] BYREF
  unsigned __int64 v328; // [rsp+C8h] [rbp-68h] BYREF
  _BYTE *v329; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v330; // [rsp+D8h] [rbp-58h] BYREF
  __int64 v331; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v332; // [rsp+E8h] [rbp-48h]
  _QWORD *v333; // [rsp+F0h] [rbp-40h]

  v5 = a2;
  v6 = i;
  v7 = a1;
  v323 = 0;
  v301 = a1 & 8;
  v288 = (a1 & 8) != 0;
  v291 = a1 & 4;
  v324 = 0;
  v312 = (a1 & 0x40000) != 0;
  v325 = 0;
  v298 = (__int64 *)(a2 + 272);
  if ( (a1 & 0x40000) != 0 )
    *(_BYTE *)(a2 + 228) |= 2u;
  v8 = (__int64)dword_4F07508;
  v327 = 0;
  v306 = a1 & 2;
  v9 = (a1 >> 7) & 1;
  v292 = (*(_QWORD *)(v5 + 120) & 0x4008000000LL) != 0;
  v10 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v5 + 32) = *(_QWORD *)&dword_4F063F8;
  if ( i && (*(_BYTE *)(v5 + 126) & 2) == 0 )
  {
    if ( *(_QWORD *)(v5 + 184) )
      v10 = *(_QWORD *)(v5 + 24);
    *(_QWORD *)(i + 32) = v10;
  }
  v11 = 0;
  v289 = a1;
  v296 = 0;
  v294 = a1 & 0x40;
  v307 = 0;
  v290 = (4 * (_WORD)a1) & 0x100;
  v299 = 0;
  v305 = 0;
  v302 = 0;
  v300 = 0;
  v12 = word_4F06418[0];
  v297 = 0;
  v304 = 0;
  while ( 2 )
  {
    v13 = (unsigned __int16)v12;
    switch ( (__int16)v12 )
    {
      case 1:
        if ( !qword_4D04A00 || (*(_BYTE *)(qword_4D04A00 + 73) & 2) == 0 )
        {
          v252 = dword_4F063F8;
          v287 = word_4F063FC[0];
          if ( dword_4F077C4 != 2 )
          {
LABEL_249:
            a1 = v325;
            if ( v325 )
              goto LABEL_250;
            goto LABEL_384;
          }
          v99 = v327;
          goto LABEL_575;
        }
        v265 = v11;
        if ( !(unsigned __int16)sub_8876F0() )
          goto LABEL_380;
LABEL_172:
        v331 = *(_QWORD *)&dword_4F063F8;
        v54 = sub_6911B0();
        *(_QWORD *)(v5 + 272) = v54;
        for ( i = *(unsigned __int8 *)(v54 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v54 + 140) )
          v54 = *(_QWORD *)(v54 + 160);
LABEL_174:
        if ( (_BYTE)i )
        {
          v55 = v305 | v302;
          v305 = 0;
          v302 = v325 | v55;
          if ( v325 | v55 )
          {
            v8 = (__int64)&v331;
            a1 = 84;
            sub_6851C0(84, &v331);
            v302 = 0;
            *(_QWORD *)(v5 + 272) = sub_72C930(84);
          }
        }
        v327 |= 4u;
        v325 = 20;
        v17 = v7 & 0x400;
        v12 = word_4F06418[0];
        goto LABEL_178;
      case 18:
        goto LABEL_469;
      case 25:
        v8 = 0;
        a1 = 0;
        v270 = v11;
        v94 = sub_7BE840(0, 0);
        v11 = v270;
        if ( v94 != 25 )
        {
          if ( v94 != 55 )
            goto LABEL_8;
          a5 = dword_4D041A8;
          if ( !dword_4D041A8 || *(char *)(v5 + 132) >= 0 )
            goto LABEL_8;
          sub_7ADF70(&v331, 0);
          dword_4F06648 += 2;
          v95 = sub_7AE2C0(183, dword_4F06648, &dword_4F063F8);
          if ( v332 )
            *v333 = v95;
          else
            v332 = v95;
          a1 = (unsigned __int64)&v331;
          v333 = (_QWORD *)v95;
          sub_7BC000(&v331);
          v11 = v270;
          v81 = word_4F06418[0];
LABEL_381:
          v252 = dword_4F063F8;
          v8 = word_4F063FC[0];
          v287 = word_4F063FC[0];
          if ( dword_4F077C4 != 2 )
            goto LABEL_382;
          v99 = v327;
          if ( v81 == 1 )
          {
LABEL_575:
            if ( (unk_4D04A11 & 2) != 0 )
              goto LABEL_576;
          }
LABEL_467:
          a1 = v290;
          v8 = 0;
          v256 = v11;
          v271 = v99;
          v100 = sub_7C0F00(v290, 0);
          v99 = v271;
          v11 = v256;
          if ( !v100 )
          {
            if ( word_4F06418[0] == 18 )
            {
LABEL_469:
              v331 = *(_QWORD *)&dword_4F063F8;
              v101 = unk_4D04A38;
              *(_QWORD *)(v5 + 272) = unk_4D04A38;
              for ( j = *(_BYTE *)(v101 + 140); j == 12; j = *(_BYTE *)(v101 + 140) )
                v101 = *(_QWORD *)(v101 + 160);
              if ( j )
              {
                v103 = v305 | v302;
                v305 = 0;
                v302 = v325 | v103;
                if ( v325 | v103 )
                {
                  v8 = (__int64)&v331;
                  a1 = 84;
                  sub_6851C0(84, &v331);
                  v302 = 0;
                  *(_QWORD *)(v5 + 272) = sub_72C930(84);
                }
              }
              v327 |= 4u;
              v325 = 20;
              v17 = v7 & 0x400;
              goto LABEL_186;
            }
            if ( (*(_BYTE *)(v5 + 125) & 4) == 0 )
              goto LABEL_8;
            v129 = 1;
            goto LABEL_577;
          }
LABEL_576:
          v129 = 0;
          if ( !v291 && (*(_BYTE *)(v5 + 125) & 4) == 0 )
          {
LABEL_597:
            v81 = word_4F06418[0];
LABEL_382:
            if ( v81 != 1 )
            {
              if ( v81 == 18 )
                goto LABEL_469;
              goto LABEL_384;
            }
            goto LABEL_249;
          }
LABEL_577:
          a4 = (__int64)&qword_4F077B4;
          v130 = -4978;
          if ( (_DWORD)qword_4F077B4 )
            v130 = (-(__int64)(qword_4F077A0 < 0x7724u) & 0xFFFFFFFFFFFFFFFELL) - 4978;
          if ( (v99 & v130) != 0 || (*(_BYTE *)(v5 + 268) & 0xFD) != 0 )
          {
            if ( (v99 & 8) != 0
              && (unk_4D04A12 & 2) != 0
              && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
            {
              a1 = *(unsigned __int8 *)(xmmword_4D04A20.m128i_i64[0] + 140);
              if ( (unsigned __int8)(a1 - 9) <= 2u
                && (*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 177) & 0x20) != 0
                && qword_4D04A00 == **(_QWORD **)xmmword_4D04A20.m128i_i64[0] )
              {
                v8 = 0;
                a1 = 0;
                v276 = v11;
                v133 = sub_7BE840(0, 0);
                v11 = v276;
                if ( v133 == 27 )
                {
                  if ( !unk_4D047C8 || (a1 = v294 != 0, v8 = 0, v134 = sub_6512E0(a1, 0, 0, 0, 0, 0), v11 = v276, !v134) )
                  {
                    *(_QWORD *)(v5 + 8) |= 0x400uLL;
                    goto LABEL_593;
                  }
                }
              }
            }
          }
          else
          {
            v275 = v11;
            v131 = (__int64 **)sub_6724F0();
            v11 = v275;
            a1 = (unsigned __int64)v131;
            if ( v131 )
            {
              v8 = v5;
              v132 = sub_672080(v131);
              v11 = v275;
              if ( v132 )
              {
                *(_QWORD *)(v5 + 8) |= 0x500uLL;
                v325 = 25;
                goto LABEL_593;
              }
              if ( qword_4D0495C )
              {
                if ( (unk_4D04A11 & 0x20) == 0 )
                {
                  v210 = *(_QWORD *)(qword_4D04A00 + 24);
                  if ( v210 )
                  {
                    v211 = *(_BYTE *)(v210 + 80);
                    if ( v211 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v211 - 4) <= 2u )
                    {
                      v8 = (__int64)&v331;
                      a1 = 27;
                      sub_7BEB10(27, &v331);
                      v11 = v275;
                      if ( (_WORD)v331 == 28 )
                      {
                        *(_QWORD *)(v5 + 8) |= 0x100uLL;
LABEL_593:
                        if ( v129 )
                          goto LABEL_8;
LABEL_250:
                        v14 = (v11 ^ 1) & (v6 != 0);
                        if ( (*(_BYTE *)(v5 + 9) & 4) != 0 )
                        {
                          if ( v307 )
                          {
                            a1 = 902;
                            sub_684B30(902, v5 + 72);
                            v327 &= ~2uLL;
                          }
                          v16 = dword_4F077C4;
                          v15 = 0;
                          v11 = 0;
                        }
                        else
                        {
                          v16 = dword_4F077C4;
                          v11 = 0;
                          v15 = v307 & 0x7F;
                        }
                        goto LABEL_43;
                      }
                    }
                  }
                }
              }
            }
          }
          if ( v129 )
            goto LABEL_8;
          goto LABEL_597;
        }
        v12 = (unsigned int)dword_4D043F8;
        if ( !dword_4D043F8 )
          goto LABEL_8;
LABEL_198:
        if ( (v7 & 0x100000) != 0 )
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 1570;
          v17 = v7 & 0x400;
          sub_684B30(1570, &dword_4F063F8);
          sub_5CCA00();
          v12 = word_4F06418[0];
LABEL_178:
          v11 = 1;
          LODWORD(v9) = 0;
LABEL_13:
          if ( v17 )
          {
            if ( (unsigned __int16)(v12 - 81) <= 0x26u )
            {
              i = 0x6004000001LL;
              if ( !_bittest64(&i, (unsigned int)(v12 - 81)) )
                goto LABEL_16;
            }
            else if ( (unsigned __int16)(v12 - 263) > 3u )
            {
              goto LABEL_16;
            }
          }
          else if ( v324 && (!dword_4F077C0 || *(_BYTE *)(v5 + 268) != 4) && (v7 & 0x800) != 0 )
          {
            if ( (unsigned __int16)(v12 - 80) <= 0x30u )
            {
              v19 = 0x1C70006066221LL;
              if ( _bittest64(&v19, (unsigned int)(v12 - 80)) )
                goto LABEL_42;
            }
            else if ( (_WORD)v12 == 180 || (_WORD)v12 == 165 || (unsigned __int16)(v12 - 331) <= 4u || (_WORD)v12 == 18 )
            {
LABEL_42:
              v15 = v307 & 0x7F;
              v16 = dword_4F077C4;
              v14 = (v11 ^ 1) & (v6 != 0);
              v11 = 1;
              goto LABEL_43;
            }
            i = (unsigned int)(unk_4D04548 | unk_4D04558);
            if ( unk_4D04548 | unk_4D04558 && (unsigned __int16)(v12 - 133) <= 3u || (_WORD)v12 == 239 )
              goto LABEL_33;
            if ( (unsigned __int16)(v12 - 272) <= 8u )
              goto LABEL_42;
            if ( (_DWORD)qword_4F077B4 && ((_WORD)v12 == 236 || (unsigned __int16)(v12 - 339) <= 0xFu) )
            {
LABEL_33:
              v15 = v307 & 0x7F;
              v14 = (v11 ^ 1) & (v6 != 0);
              v11 = 1;
              if ( dword_4F077C4 != 2 )
                goto LABEL_44;
              goto LABEL_34;
            }
            a4 = (unsigned int)(v12 - 151);
            if ( (unsigned __int16)(v12 - 151) > 0x27u )
            {
              if ( (unsigned __int16)(v12 - 87) > 0x11u )
              {
                LOBYTE(v31) = 0;
                goto LABEL_95;
              }
              i = 147457;
              if ( _bittest64(&i, (unsigned int)(v12 - 87)) )
                goto LABEL_33;
            }
            else
            {
              v31 = (0xC500000001uLL >> ((unsigned __int8)v12 + 105)) & 1;
LABEL_95:
              if ( (_WORD)v12 == 236 || (_BYTE)v31 )
                goto LABEL_33;
              if ( dword_4F0775C && (_WORD)v12 == 77 )
                goto LABEL_42;
            }
          }
          continue;
        }
        a1 = 5;
        v251 = v11;
        v331 = sub_5CC190(5);
        i = v331;
        if ( !v331 )
        {
          v11 = 1;
          if ( v6 )
            *(_QWORD *)(v6 + 40) = unk_4F061D8;
          goto LABEL_913;
        }
        a5 = v7 & 0x400000;
        v266 = v7 & 0x8000000;
        a4 = v289 & 0x40;
        v248 = v7;
        v56 = (__int64 **)&v331;
        v247 = v6;
        v57 = 0;
        v58 = a5;
        v254 = 0;
        while ( 2 )
        {
          if ( (v289 & 0x40) != 0 )
          {
            v59 = *(unsigned __int8 *)(i + 8);
            if ( (unsigned __int8)(v59 - 87) <= 0xFu )
            {
              v60 = 32793;
              if ( _bittest64(&v60, (unsigned int)(v59 - 87)) )
              {
                v8 = i + 56;
                sub_6851A0(3537, i + 56, *(_QWORD *)(i + 16));
                i = **v56;
                *v56 = (__int64 *)i;
                goto LABEL_204;
              }
            }
          }
          v8 = *(unsigned __int8 *)(i + 9);
          if ( !v58 && ((_BYTE)v8 == 2 || (*(_BYTE *)(i + 11) & 0x10) != 0) )
          {
LABEL_209:
            if ( !v57 )
            {
              v8 = 1098;
              sub_684AA0(dword_4F077BC == 0 ? 8 : 5, 1098, i + 56);
              i = (__int64)*v56;
            }
            i = *(_QWORD *)i;
            v57 = 1;
            *v56 = (__int64 *)i;
          }
          else
          {
            if ( ((_BYTE)v8 == 1 || (_BYTE)v8 == 4) && (dword_4F077C4 == 2 || unk_4F07778 <= 201111 || (_BYTE)v8 != 4) )
            {
              if ( !v266 )
                goto LABEL_209;
              v254 = 1;
            }
            v56 = (__int64 **)i;
            i = *(_QWORD *)i;
          }
LABEL_204:
          if ( i )
            continue;
          break;
        }
        v135 = (_QWORD **)(v5 + 208);
        v11 = v251;
        v7 = v248;
        v6 = v247;
        if ( *(_QWORD *)(v5 + 208) )
        {
          v136 = sub_5CB9F0(v135);
          v11 = v251;
          v135 = v136;
        }
        *v135 = (_QWORD *)v331;
        if ( v247 )
          *(_QWORD *)(v247 + 40) = unk_4F061D8;
        a1 = v254;
        if ( !v254 )
        {
          v11 = 1;
LABEL_913:
          v17 = v7 & 0x400;
          v12 = word_4F06418[0];
          goto LABEL_13;
        }
        if ( (v327 & 0x804) == 0 )
        {
          a1 = 1866;
          v17 = v248 & 0x400;
          v8 = *(_QWORD *)(*(_QWORD *)(v5 + 208) + 40LL);
          sub_6851C0(1866, v8);
          *(_QWORD *)(v5 + 208) = 0;
          v12 = word_4F06418[0];
          goto LABEL_178;
        }
LABEL_8:
        v14 = v6 != 0;
        v15 = v307 & 0x7F;
        if ( v11 )
        {
          v16 = dword_4F077C4;
          v14 = 0;
          v11 = 0;
        }
        else if ( (v7 & 0x10) != 0 )
        {
          *(_QWORD *)(v5 + 8) |= 0x100uLL;
          v16 = dword_4F077C4;
        }
        else
        {
          a1 = 79;
          sub_6851D0(79);
          v325 = 26;
          v16 = dword_4F077C4;
          v11 = 0;
          v15 = v307 & 0x7F;
          LODWORD(v323) = 1;
        }
LABEL_43:
        if ( v16 == 2 )
          goto LABEL_34;
LABEL_44:
        if ( (*(_BYTE *)(v5 + 125) & 1) != 0 && !*(_QWORD *)(v5 + 304) )
        {
          a5 = dword_4F07758;
          if ( dword_4F07758 )
          {
            a1 = dword_4F0775C;
            if ( dword_4F0775C || (_DWORD)qword_4F077B4 && dword_4F077C4 == 2 && qword_4F077A0 )
            {
              v310 = v15;
              LOBYTE(a1) = v301 == 0;
              a1 = v292 & (unsigned int)a1;
              v316 = v11;
              sub_668A70(a1, v296, v7, v5, v6, &v327, &v325, v298, &v323);
              v11 = v316;
              v15 = v310;
            }
          }
        }
        v20 = v300;
        if ( v300 )
        {
          if ( *(_BYTE *)(v5 + 268) == 4 )
          {
            v104 = 0;
            v105 = 1;
            if ( word_4F06418[0] == 75 )
              v105 = v304;
            LOBYTE(v104) = word_4F06418[0] != 75;
            v304 = v105;
            a1 = (unsigned int)(3 * v104 + 5);
          }
          else
          {
            v304 = 1;
            a1 = 8;
          }
          v309 = v15;
          v314 = v11;
          sub_684AA0(a1, v300, &v330);
          v15 = v309;
          v11 = v314;
        }
        *(_BYTE *)(v5 + 120) = v15 | *(_BYTE *)(v5 + 120) & 0x80;
        if ( v327 == 2048 )
        {
          *(_QWORD *)(v5 + 8) |= 0x40uLL;
        }
        else if ( v297 )
        {
          if ( (v327 & 8) != 0 && (a1 = (unsigned __int64)word_4F06418, word_4F06418[0] == 75)
            || !(v324 | (unsigned int)v323) && (v327 & 0x53) == 0 )
          {
            v21 = *(_QWORD *)(v5 + 8);
            v22 = v21;
            BYTE1(v22) = BYTE1(v21) | 2;
            v23 = v325 == 23;
            *(_QWORD *)(v5 + 8) = v22;
            if ( v23 )
              *(_QWORD *)(v5 + 8) = v21 | 0x10200;
          }
        }
        if ( v14 && (*(_BYTE *)(v5 + 126) & 2) == 0 )
        {
          v24 = *(_QWORD **)(v5 + 184);
          if ( v24 )
          {
            do
            {
              v25 = v24;
              v24 = (_QWORD *)*v24;
            }
            while ( v24 );
            v26 = v25[5];
            if ( v26 )
              v27 = *(_QWORD *)(v26 + 8);
            else
              v27 = v25[8];
            *(_QWORD *)(v6 + 40) = v27;
          }
          else
          {
            *(_QWORD *)(v6 + 32) = unk_4F077C8;
          }
        }
        v28 = (char *)dword_4F07508;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)(v5 + 32);
        if ( !v306 )
          goto LABEL_119;
        v29 = v325;
        if ( v325 != 25 && v325 || (v28 = (char *)(v302 | v305), v302 | v305) )
          *(_QWORD *)(v5 + 8) |= 1uLL;
        LOBYTE(v20) = v29 - 12 > 7;
        if ( ((v29 != 9) & (unsigned __int8)v20) != 0 && v299 )
        {
          if ( v29 )
          {
            if ( v29 != 20 )
              goto LABEL_1147;
            v232 = *(_QWORD *)(v5 + 272);
            if ( !v232 || *(_BYTE *)(v232 + 140) != 12 )
              goto LABEL_1147;
            v233 = *(_BYTE *)(v232 + 184);
            if ( (v233 > 0xCu
               || ((0x18C2uLL >> v233) & 1) == 0
               || !dword_4F077BC
               || (_DWORD)qword_4F077B4
               || !qword_4F077A8)
              && *(char *)(v232 + 185) >= 0 )
            {
              goto LABEL_1147;
            }
            v234 = *(_QWORD *)(v5 + 272);
            do
            {
              v234 = *(_QWORD *)(v234 + 160);
              v235 = *(_BYTE *)(v234 + 140);
            }
            while ( v235 == 12 );
            if ( v235 == 3 )
            {
              do
                v232 = *(_QWORD *)(v232 + 160);
              while ( *(_BYTE *)(v232 + 140) == 12 );
              a1 = *(unsigned __int8 *)(v232 + 160);
              v325 = sub_667F40(a1);
            }
            else
            {
LABEL_1147:
              v28 = "_Complex";
              a1 = 1043;
              if ( v299 != 1 )
                v28 = "_Imaginary";
              v321 = v11;
              sub_6851F0(1043, v28);
              v11 = v321;
              v304 = 1;
            }
          }
          else if ( dword_4F077C0 )
          {
            v325 = 15;
          }
          else
          {
            v28 = (char *)dword_4F07508;
            a1 = 1054;
            v322 = v11;
            sub_6851C0(1054, dword_4F07508);
            v11 = v322;
            v304 = 1;
          }
          *(_QWORD *)(v5 + 8) |= 1uLL;
        }
        if ( v11 )
          *(_QWORD *)(v5 + 8) |= 0x80uLL;
        if ( v304 )
        {
LABEL_486:
          *(_QWORD *)(v5 + 272) = sub_72C930(a1);
          LODWORD(v323) = 1;
          goto LABEL_487;
        }
        if ( unk_4D047EC )
        {
          if ( unk_4F04C38 )
          {
            a1 = *(_QWORD *)(v5 + 272);
            if ( a1 )
            {
              if ( (unsigned int)sub_8DD2A0() )
              {
                a1 = 1401;
                sub_6851C0(1401, dword_4F07508);
                goto LABEL_486;
              }
            }
          }
        }
        v30 = v325;
        if ( dword_4F077C4 != 1 && (v12 = dword_4F077BC) == 0 && ((a5 = dword_4F077C0) == 0 || qword_4F077A8 > 0x76BFu)
          || v325 != 20 )
        {
          switch ( v325 )
          {
            case 0u:
            case 8u:
LABEL_732:
              switch ( v305 )
              {
                case 0u:
                  if ( v302 == 2 )
                    goto LABEL_1203;
                  v157 = 0;
                  v106 = 5;
                  v153 = 5;
                  v305 = 0;
                  break;
                case 1u:
                  goto LABEL_788;
                case 2u:
                  goto LABEL_799;
                case 3u:
                  goto LABEL_779;
                case 5u:
                  if ( v302 == 2 )
                    goto LABEL_1206;
                  v305 = 5;
                  v106 = unk_4F06ACF;
                  v157 = 5;
                  v153 = unk_4F06ACF;
                  break;
                case 6u:
                  if ( v302 == 2 )
                    goto LABEL_1199;
                  v305 = 6;
                  v106 = unk_4F06ACD;
                  v157 = 6;
                  v153 = unk_4F06ACD;
                  break;
                case 7u:
                  if ( v302 == 2 )
                    goto LABEL_1213;
                  v305 = 7;
                  v106 = unk_4F06ACB;
                  v157 = 7;
                  v153 = unk_4F06ACB;
                  break;
                case 8u:
                  if ( v302 == 2 )
                    goto LABEL_1204;
                  v106 = 11;
                  v157 = 8;
                  v153 = 11;
                  v305 = 8;
                  break;
                default:
                  goto LABEL_703;
              }
LABEL_764:
              if ( v302 == 1 )
              {
                if ( unk_4D04548 | unk_4D04558 && v305 > 3 && v157 != 8 )
                  *(_QWORD *)(v5 + 272) = sub_72BDB0(v106);
                else
                  *(_QWORD *)(v5 + 272) = sub_72BCF0(v106);
                goto LABEL_507;
              }
              if ( unk_4D04558 )
                goto LABEL_838;
LABEL_837:
              if ( !unk_4D04548 )
                goto LABEL_506;
LABEL_838:
              if ( v157 == 8 || v305 <= 3 )
              {
LABEL_506:
                *(_QWORD *)(v5 + 272) = sub_72BA30(v106);
                goto LABEL_507;
              }
LABEL_840:
              *(_QWORD *)(v5 + 272) = sub_72BC30(v153);
              goto LABEL_507;
            case 1u:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72CBE0(a1, v28, v325, v20, a5, v12);
              goto LABEL_507;
            case 2u:
LABEL_500:
              if ( !v305 )
                goto LABEL_503;
              if ( v305 != 4 )
                goto LABEL_691;
LABEL_502:
              if ( !(unk_4D04548 | unk_4D04558) )
                goto LABEL_691;
LABEL_503:
              v106 = 1;
              if ( v302 != 1 )
              {
                v106 = 2;
                if ( v302 != 2 )
                  v106 = byte_4F068B0[0];
              }
              goto LABEL_506;
            case 3u:
LABEL_749:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72BF70();
              goto LABEL_507;
            case 4u:
LABEL_751:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72C030();
              goto LABEL_507;
            case 5u:
LABEL_753:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72C0F0();
              goto LABEL_507;
            case 6u:
LABEL_755:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72C1B0();
              goto LABEL_507;
            case 7u:
LABEL_757:
              if ( v302 | v305 )
                goto LABEL_691;
              *(_QWORD *)(v5 + 272) = sub_72C390();
              goto LABEL_507;
            case 9u:
            case 0xAu:
            case 0xBu:
            case 0xCu:
            case 0xDu:
            case 0xEu:
            case 0xFu:
            case 0x10u:
            case 0x11u:
            case 0x12u:
            case 0x13u:
LABEL_658:
              if ( v302 )
                goto LABEL_691;
              if ( v305 )
              {
                if ( v305 != 2 )
                  goto LABEL_691;
                if ( v30 != 12 )
                {
                  v139 = 6;
                  if ( unk_4F04C50 )
                  {
                    v212 = *(_QWORD *)(unk_4F04C50 + 32LL);
                    if ( v212 )
                    {
                      if ( (*(_BYTE *)(v212 + 198) & 0x10) != 0 )
                        sub_684B10(3665, dword_4F07508, "long double");
                    }
                  }
                  goto LABEL_671;
                }
                v139 = 4;
                if ( dword_4F077C4 != 1 )
                {
                  if ( dword_4D04964 )
                  {
                    if ( HIDWORD(qword_4F077B4) )
                      v231 = 7;
                    else
                      v231 = byte_4F07472[0];
                  }
                  else
                  {
                    v231 = 7;
                    if ( !HIDWORD(qword_4F077B4) )
                    {
                      sub_684B30(2252, dword_4F07508);
                      goto LABEL_671;
                    }
                  }
                  v139 = 4;
                  sub_684AC0(v231, 84);
                }
              }
              else
              {
                switch ( v30 )
                {
                  case 9:
                    v139 = 9;
                    break;
                  case 10:
                    v139 = 0;
                    break;
                  case 11:
                    v139 = 1;
                    break;
                  case 12:
                    v139 = 2;
                    break;
                  case 14:
                    v139 = 3;
                    break;
                  case 15:
                    v139 = 4;
                    break;
                  case 17:
                    v139 = 5;
                    break;
                  case 13:
                    v139 = 11;
                    break;
                  case 16:
                    v139 = 12;
                    break;
                  case 18:
                    v139 = 7;
                    break;
                  default:
                    v139 = 13;
                    if ( v30 == 19 )
                      break;
LABEL_703:
                    sub_721090(a1);
                }
              }
LABEL_671:
              if ( v299 == 1 )
              {
                *(_QWORD *)(v5 + 272) = sub_72C6F0(v139);
              }
              else if ( v299 == 2 )
              {
                *(_QWORD *)(v5 + 272) = sub_72C7D0(v139);
              }
              else
              {
                *(_QWORD *)(v5 + 272) = sub_72C610(v139);
              }
              goto LABEL_507;
            case 0x14u:
            case 0x15u:
            case 0x16u:
            case 0x17u:
            case 0x18u:
LABEL_690:
              if ( v305 | v302 )
                goto LABEL_691;
              goto LABEL_507;
            case 0x19u:
LABEL_748:
              *(_QWORD *)(v5 + 272) = sub_72CBA0();
              goto LABEL_507;
            case 0x1Au:
LABEL_759:
              *(_QWORD *)(v5 + 272) = sub_72C930(a1);
              goto LABEL_507;
            default:
              goto LABEL_703;
          }
        }
        if ( v302 | v305 )
        {
          v32 = *(_QWORD *)(v5 + 272);
          for ( k = *(_BYTE *)(v32 + 140); k == 12; k = *(_BYTE *)(v32 + 140) )
            v32 = *(_QWORD *)(v32 + 160);
          if ( k == 2 )
          {
            if ( (*(_DWORD *)(v32 + 160) & 0x40800) == 0 )
            {
              v153 = *(unsigned __int8 *)(v32 + 160);
              switch ( (char)v153 )
              {
                case 0:
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    sub_684B30(1616, dword_4F07508);
                  goto LABEL_500;
                case 1:
                  goto LABEL_821;
                case 2:
                  if ( byte_4F068B0[0] != 2 && v302 )
                    break;
LABEL_821:
                  if ( v305 )
                  {
                    if ( v305 != 4 || !(unk_4D04548 | unk_4D04558) )
                      break;
                    *(_QWORD *)(v5 + 272) = 0;
                    if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    {
                      sub_684B30(1616, dword_4F07508);
                      goto LABEL_502;
                    }
                  }
                  else
                  {
                    *(_QWORD *)(v5 + 272) = 0;
                    if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                      sub_684B30(1616, dword_4F07508);
                  }
                  goto LABEL_503;
                case 3:
                  if ( v305 && (!dword_4F077BC || v305 != 1 || (*(_BYTE *)(v5 + 124) & 1) == 0) )
                    break;
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    sub_684B30(1616, dword_4F07508);
LABEL_788:
                  if ( v302 != 2 )
                  {
                    v106 = 3;
                    v157 = 1;
                    v153 = 3;
                    v305 = 1;
                    goto LABEL_764;
                  }
                  v106 = 4;
                  v157 = 1;
                  v153 = 4;
                  v305 = 1;
                  goto LABEL_836;
                case 4:
                  if ( !dword_4F077BC || (*(_BYTE *)(v5 + 124) & 1) == 0 || v305 > 1 || v302 != 2 && (v305 != 1 || v302) )
                    break;
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    sub_684B30(1616, dword_4F07508);
                  v305 = 1;
                  v106 = 4;
                  v157 = 1;
                  goto LABEL_836;
                case 5:
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) )
                    goto LABEL_730;
                  goto LABEL_732;
                case 6:
                  if ( v302 )
                  {
                    if ( !dword_4F077BC || (*(_BYTE *)(v5 + 124) & 1) == 0 || v302 != 2 )
                      break;
                    *(_QWORD *)(v5 + 272) = 0;
                    v157 = HIDWORD(qword_4F077B4);
                    if ( !HIDWORD(qword_4F077B4) )
                    {
                      switch ( v305 )
                      {
                        case 0u:
                          v106 = 6;
                          goto LABEL_836;
                        case 1u:
LABEL_1239:
                          v106 = 4;
                          v157 = 1;
                          v153 = 4;
                          goto LABEL_836;
                        case 2u:
LABEL_1238:
                          v106 = 8;
                          v157 = 2;
                          v153 = 8;
                          goto LABEL_836;
                        case 3u:
LABEL_1237:
                          v106 = 10;
                          v157 = 3;
                          v153 = 10;
                          goto LABEL_836;
                        case 5u:
LABEL_1206:
                          v153 = unk_4F06ACE;
                          goto LABEL_1200;
                        case 6u:
LABEL_1199:
                          v153 = unk_4F06ACC;
                          goto LABEL_1200;
                        case 7u:
LABEL_1213:
                          v153 = unk_4F06ACA;
LABEL_1200:
                          if ( unk_4D04558 )
                            goto LABEL_840;
                          v106 = v153;
                          if ( unk_4D04548 )
                            goto LABEL_840;
                          goto LABEL_506;
                        case 8u:
LABEL_1204:
                          v106 = 12;
                          v157 = 8;
                          v153 = 12;
                          v305 = 8;
                          goto LABEL_836;
                        default:
                          goto LABEL_703;
                      }
                    }
                  }
                  else
                  {
                    *(_QWORD *)(v5 + 272) = 0;
                    if ( !HIDWORD(qword_4F077B4) )
                    {
                      switch ( v305 )
                      {
                        case 0u:
LABEL_1203:
                          v157 = 0;
                          v106 = 6;
                          v153 = 6;
                          v305 = 0;
                          goto LABEL_836;
                        case 1u:
                          goto LABEL_1239;
                        case 2u:
                          goto LABEL_1238;
                        case 3u:
                          goto LABEL_1237;
                        case 5u:
                          goto LABEL_1206;
                        case 6u:
                          goto LABEL_1199;
                        case 7u:
                          goto LABEL_1213;
                        case 8u:
                          goto LABEL_1204;
                        default:
                          goto LABEL_703;
                      }
                    }
                  }
                  v302 = 2;
LABEL_730:
                  if ( unk_4D04320 )
                  {
                    a1 = 1616;
                    sub_684B30(1616, dword_4F07508);
                  }
                  goto LABEL_732;
                case 7:
                  if ( v305
                    && (!dword_4F077C0 || v305 != 2)
                    && (!dword_4F077BC || (*(_BYTE *)(v5 + 124) & 1) == 0 || v305 != 2) )
                  {
                    break;
                  }
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    sub_684B30(1616, dword_4F07508);
LABEL_799:
                  if ( v302 != 2 )
                  {
                    v106 = 7;
                    v157 = 2;
                    v153 = 7;
                    v305 = 2;
                    goto LABEL_764;
                  }
                  v106 = 8;
                  v157 = 2;
                  v153 = 8;
                  v305 = 2;
                  goto LABEL_836;
                case 8:
                  if ( (!dword_4F077C0 || v305 != 2)
                    && (!dword_4F077BC
                     || (*(_BYTE *)(v5 + 124) & 1) == 0
                     || (v305 & 0xFFFFFFFD) != 0
                     || v302 != 2 && (v305 != 2 || v302)) )
                  {
                    break;
                  }
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                  {
                    sub_684B30(1616, dword_4F07508);
                    v106 = 8;
                    v157 = 2;
                    v305 = 2;
                  }
                  else
                  {
                    v305 = 2;
                    v106 = 8;
                    v157 = 2;
                  }
                  goto LABEL_836;
                case 9:
                  if ( v305 )
                    break;
                  *(_QWORD *)(v5 + 272) = 0;
                  if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
                    sub_684B30(1616, dword_4F07508);
LABEL_779:
                  if ( v302 != 2 )
                  {
                    v106 = 9;
                    v157 = 3;
                    v153 = 9;
                    v305 = 3;
                    goto LABEL_764;
                  }
                  v106 = 10;
                  v157 = 3;
                  v153 = 10;
                  v305 = 3;
LABEL_836:
                  if ( !unk_4D04558 )
                    goto LABEL_837;
                  goto LABEL_506;
                case 10:
                  goto LABEL_690;
                case 11:
                case 12:
                  if ( !dword_4F077BC || (*(_BYTE *)(v5 + 122) & 0x20) == 0 )
                    break;
                  if ( (_BYTE)v153 == 11 )
                  {
                    if ( v302 != 1 )
                      break;
                  }
                  else if ( v302 != 2 )
                  {
                    break;
                  }
                  v302 = 0;
                  goto LABEL_690;
                default:
                  goto LABEL_703;
              }
            }
          }
          else if ( k == 3 )
          {
            a1 = *(unsigned __int8 *)(v32 + 160);
            v30 = sub_667F40(a1);
            if ( v30 != 20 )
            {
              *(_QWORD *)(v5 + 272) = 0;
              if ( HIDWORD(qword_4F077B4) && unk_4D04320 )
              {
                a1 = 1616;
                sub_684B30(1616, dword_4F07508);
              }
              switch ( v30 )
              {
                case 0:
                case 8:
                  goto LABEL_732;
                case 1:
                  break;
                case 2:
                  goto LABEL_500;
                case 3:
                  goto LABEL_749;
                case 4:
                  goto LABEL_751;
                case 5:
                  goto LABEL_753;
                case 6:
                  goto LABEL_755;
                case 7:
                  goto LABEL_757;
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 16:
                case 17:
                case 18:
                case 19:
                  goto LABEL_658;
                case 21:
                case 22:
                case 23:
                case 24:
                  goto LABEL_690;
                case 25:
                  goto LABEL_748;
                case 26:
                  goto LABEL_759;
                default:
                  goto LABEL_703;
              }
            }
          }
LABEL_691:
          if ( dword_4F04C58 != -1 )
          {
            v141 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
            if ( v141 )
            {
              if ( (*(_BYTE *)(v141 + 197) & 2) == 0 && (*(_BYTE *)(v141 + 198) & 0x10) != 0 )
              {
                v142 = 1;
                goto LABEL_696;
              }
            }
          }
          goto LABEL_699;
        }
LABEL_507:
        if ( dword_4F04C58 == -1 )
          goto LABEL_510;
        v107 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
        if ( !v107 || (*(_BYTE *)(v107 + 197) & 2) != 0 || (*(_BYTE *)(v107 + 198) & 0x10) == 0 )
          goto LABEL_510;
        v142 = 0;
LABEL_696:
        v143 = *(_QWORD *)(v5 + 272);
        if ( v143 )
          sub_8E3700(v143);
        if ( !v142 )
        {
LABEL_510:
          v108 = *(_BYTE *)(v5 + 120);
          v109 = v108 & 0x7F;
          v35 = v108 & 0x7F;
          if ( (v108 & 0x7F) != 0 )
          {
            v110 = *(_QWORD *)(v5 + 272);
            v111 = v108 & 4;
            v112 = v111;
            if ( *(_BYTE *)(v110 + 140) == 12 )
            {
              v318 = v111;
              if ( dword_4F077C4 == 2 )
              {
                v213 = sub_8D2FB0(v110);
                v112 = v318;
                if ( v213 )
                {
                  *(_BYTE *)(v5 + 124) = (32 * ((v35 & 0xFFFFFFFB) != 0)) | *(_BYTE *)(v5 + 124) & 0xDF;
                  if ( !v318 )
                    goto LABEL_118;
                  v35 = v318;
                  if ( (unsigned int)sub_624110(v110, v5 + 80) )
                    goto LABEL_113;
                  goto LABEL_1015;
                }
              }
              else
              {
                v154 = sub_8D4C10(v110, 0);
                v112 = v318;
                if ( (v154 & v35) != 0 )
                {
                  v155 = 4;
                  v156 = dword_4D04964;
                  if ( dword_4D04964 )
                  {
                    if ( dword_4F077C4 == 2 || (v156 = 0, unk_4F07778 <= 199900) )
                    {
                      v155 = byte_4F07472[0];
                      v156 = byte_4F07472[0] == 8;
                    }
                  }
                  v311 = v318;
                  v319 = v156;
                  sub_684AC0(v155, 83);
                  v112 = v311;
                  v304 = v319;
                }
              }
            }
            if ( !v112 )
            {
              if ( (v109 & 0x70) == 0 )
                goto LABEL_113;
              goto LABEL_514;
            }
            if ( (unsigned int)sub_624110(v110, v5 + 80) )
            {
              if ( (v109 & 0x70) == 0 )
                goto LABEL_113;
              goto LABEL_514;
            }
LABEL_1015:
            v203 = v35 & 0xFFFFFFFB;
            if ( (v35 & 0x70) == 0 )
            {
              if ( !v203 )
                goto LABEL_1017;
              v304 = 1;
              v35 &= ~4u;
LABEL_113:
              v34 = v110;
              if ( (unsigned int)sub_8D2310(v110) )
              {
                if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 6) & 2) != 0 )
                {
                  v214 = v5 + 72;
                  if ( v35 == 4 )
                    v214 = v5 + 80;
                  LOBYTE(v35) = 0;
                  sub_684B30(925, v214);
                }
                else
                {
                  LOBYTE(v35) = 0;
                }
                goto LABEL_117;
              }
              if ( (*(_BYTE *)(v5 + 125) & 2) != 0
                && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x1ADAFu) )
              {
                LOBYTE(v35) = 0;
                sub_6851C0(2643, v5 + 104);
LABEL_117:
                if ( !v304 )
                {
LABEL_118:
                  *(_QWORD *)(v5 + 272) = v110;
                  goto LABEL_119;
                }
                LOBYTE(v203) = v35;
LABEL_1017:
                v113 = v203 & 0x7F;
                goto LABEL_517;
              }
              v222 = *(_BYTE *)(v110 + 140);
              if ( v222 == 12 )
              {
                v223 = v110;
                do
                {
                  v223 = *(_QWORD *)(v223 + 160);
                  v224 = *(_BYTE *)(v223 + 140);
                }
                while ( v224 == 12 );
              }
              else
              {
                v224 = *(_BYTE *)(v110 + 140);
              }
              if ( v224 != 21 || *(_QWORD *)(v5 + 304) )
              {
                if ( (v35 & 8) == 0 )
                {
LABEL_1119:
                  if ( (v222 & 0xFB) == 8 )
                  {
                    v227 = sub_8D4C10(v110, dword_4F077C4 != 2) & 3;
                    if ( v227 )
                    {
                      v225 = v35 & ~v227;
                      *(_WORD *)(v5 + 120) = ((v35 & 3 & (unsigned __int16)~(_WORD)v227) << 7)
                                           | *(_WORD *)(v5 + 120) & 0xC07F;
                    }
                    else
                    {
                      v225 = v35;
                      *(_WORD *)(v5 + 120) = *(_WORD *)(v5 + 120) & 0xC07F | ((v35 & 3) << 7);
                    }
                  }
                  else
                  {
                    v225 = v35;
                    *(_WORD *)(v5 + 120) = ((v35 & 3) << 7) | *(_WORD *)(v5 + 120) & 0xC07F;
                  }
                  v110 = sub_73C570(v110, v225, *(_QWORD *)(v5 + 392));
                  goto LABEL_117;
                }
                v228 = v5 + 72;
              }
              else
              {
                v34 = 5;
                v226 = sub_72BA30(5);
                v110 = v226;
                if ( (v35 & 8) == 0 )
                {
LABEL_1127:
                  v222 = *(_BYTE *)(v110 + 140);
                  goto LABEL_1119;
                }
                if ( !v226 )
                  goto LABEL_1125;
                v222 = *(_BYTE *)(v226 + 140);
                v228 = v5 + 72;
              }
              if ( v222 == 12 )
              {
                v229 = v110;
                do
                {
                  v229 = *(_QWORD *)(v229 + 160);
                  v222 = *(_BYTE *)(v229 + 140);
                }
                while ( v222 == 12 );
              }
              if ( v222 )
              {
                v110 = sub_6680B0(v110, v228, 1);
                goto LABEL_1126;
              }
LABEL_1125:
              v110 = sub_72C930(v34);
LABEL_1126:
              v35 &= ~8u;
              goto LABEL_1127;
            }
            v304 = 1;
            v35 &= ~4u;
LABEL_514:
            if ( !(unsigned int)sub_624240(v35, v110, v5 + 72) )
            {
              v35 &= 0xFFFFFF8F;
              if ( !v35 )
              {
                v113 = 0;
LABEL_517:
                v114 = *(_BYTE *)(v5 + 120);
                *(_QWORD *)(v5 + 272) = v110;
                LODWORD(v323) = 1;
                *(_BYTE *)(v5 + 120) = v113 | v114 & 0x80;
                goto LABEL_487;
              }
              v304 = 1;
            }
            goto LABEL_113;
          }
LABEL_119:
          v36 = v323 & 1;
          if ( v323 )
            goto LABEL_120;
          goto LABEL_121;
        }
LABEL_699:
        sub_6851C0(84, dword_4F07508);
        *(_QWORD *)(v5 + 272) = sub_72C930(84);
        LODWORD(v323) = 1;
LABEL_487:
        v36 = 1;
LABEL_120:
        *(_QWORD *)(v5 + 8) |= 0x10uLL;
LABEL_121:
        if ( v324 )
          *(_QWORD *)(v5 + 8) |= 0x20uLL;
        v37 = *(_QWORD *)(v5 + 184);
        v38 = *(_BYTE **)(v5 + 208);
        *(_BYTE *)(v5 + 269) = *(_BYTE *)(v5 + 268);
        *(_BYTE *)(v5 + 126) = (16 * v36) | *(_BYTE *)(v5 + 126) & 0xEF;
        if ( v37 )
        {
          v329 = 0;
          v39 = (__int64 *)(v5 + 208);
          v331 = 0;
          if ( v38 )
            goto LABEL_125;
          v45 = (__int64 *)(v5 + 184);
LABEL_136:
          v46 = &v331;
          while ( 2 )
          {
            while ( 1 )
            {
              v48 = *(_BYTE *)(v37 + 11);
              v49 = *(_BYTE *)(v37 + 8);
              if ( (v48 & 2) == 0 )
                break;
              if ( v49 <= 1u )
                goto LABEL_144;
              v47 = *(_BYTE *)(v37 + 9);
              if ( (v47 == 1 || v47 == 4) && (v48 & 0x10) == 0 )
              {
                v315 = v46;
                sub_5CCAE0(8u, v37);
                v46 = v315;
              }
              *v45 = *(_QWORD *)v37;
              *(_QWORD *)v37 = 0;
              *(_BYTE *)(v37 + 10) = 5;
              *v46 = v37;
              v46 = (__int64 *)v37;
              v37 = *v45;
              if ( !*v45 )
              {
LABEL_145:
                v37 = v331;
                goto LABEL_146;
              }
            }
            if ( v49 == 19 )
              *(_BYTE *)(v5 + 131) |= 0x80u;
LABEL_144:
            v45 = (__int64 *)*v45;
            v37 = *v45;
            if ( !*v45 )
              goto LABEL_145;
            continue;
          }
        }
        if ( !v38 )
          goto LABEL_150;
        v329 = 0;
        v39 = (__int64 *)(v5 + 208);
        v331 = 0;
LABEL_125:
        v40 = v5;
        v41 = (__int64 *)&v329;
        while ( 2 )
        {
          while ( 1 )
          {
            v44 = v38[11];
            if ( (v44 & 2) == 0 )
            {
              if ( v38[8] > 1u )
                goto LABEL_127;
              if ( v38[9] == 2 || (v44 & 0x10) != 0 )
                break;
            }
            v39 = (__int64 *)*v39;
LABEL_134:
            v38 = (_BYTE *)*v39;
            if ( !*v39 )
              goto LABEL_135;
          }
          if ( !(unsigned int)sub_8D3EA0(*(_QWORD *)(v40 + 272)) )
          {
            v52 = *(_QWORD *)(v40 + 272);
            for ( m = *(_BYTE *)(v52 + 140); m == 12; m = *(_BYTE *)(v52 + 140) )
              v52 = *(_QWORD *)(v52 + 160);
            if ( m != 21 )
            {
              v39 = (__int64 *)*v39;
              goto LABEL_134;
            }
          }
LABEL_127:
          v42 = *v39;
          v43 = *(_BYTE *)(*v39 + 9);
          if ( (v43 == 1 || v43 == 4)
            && (*(_BYTE *)(v42 + 11) & 0x10) == 0
            && (dword_4F077C4 == 2 || unk_4F07778 <= 201111 || v43 != 4) )
          {
            sub_5CCAE0(dword_4F077BC == 0 ? 8 : 5, *v39);
          }
          *v39 = *(_QWORD *)v42;
          v23 = *(_BYTE *)(v42 + 8) == 19;
          *(_QWORD *)v42 = 0;
          *(_BYTE *)(v42 + 10) = 1;
          if ( v23 )
            *(_BYTE *)(v40 + 131) |= 0x80u;
          *v41 = v42;
          v38 = (_BYTE *)*v39;
          v41 = (__int64 *)v42;
          if ( *v39 )
            continue;
          break;
        }
LABEL_135:
        v37 = *(_QWORD *)(v40 + 184);
        v5 = v40;
        v38 = v329;
        v45 = (__int64 *)(v40 + 184);
        if ( v37 )
          goto LABEL_136;
LABEL_146:
        *v45 = (__int64)v38;
        *v39 = v37;
        v50 = *(_QWORD **)(v5 + 208);
        if ( v50 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(v5 + 272) + 140LL) == 7 )
          {
            v115 = sub_7259C0(12);
            v116 = *(_QWORD *)(v5 + 272);
            *(_BYTE *)(v115 + 184) = 8;
            *(_QWORD *)(v115 + 160) = v116;
            v50 = *(_QWORD **)(v5 + 208);
            *(_QWORD *)(v5 + 272) = v115;
          }
          sub_5CF030(v298, v50, v5);
        }
LABEL_150:
        result = *(_QWORD *)(v5 + 272);
        *(_QWORD *)(v5 + 280) = result;
        *(_QWORD *)(v5 + 288) = result;
        if ( (*(_BYTE *)(v5 + 9) & 1) != 0 )
        {
          result = *(unsigned __int8 *)(v5 + 126);
          if ( (result & 2) == 0 && !*(_QWORD *)(v5 + 184) && !*(_QWORD *)(v5 + 208) )
          {
            result = (unsigned int)result | 8;
            *(_BYTE *)(v5 + 126) = result;
          }
        }
        return result;
      case 37:
        goto LABEL_534;
      case 76:
        if ( !dword_4D04408 || (v327 & 0x804) != 0 )
          goto LABEL_8;
        v8 = (__int64)&dword_4F063F8;
        a1 = 1913;
        v17 = v7 & 0x400;
        sub_6851C0(1913, &dword_4F063F8);
        LODWORD(v323) = 1;
        goto LABEL_186;
      case 77:
        v17 = v7 & 0x400;
        if ( (*(_BYTE *)(v5 + 125) & 3) != 0 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = !v292 ? 81 : 84;
          sub_6851C0(a1, dword_4F07508);
          goto LABEL_186;
        }
        if ( v292 && v288 )
        {
          v8 = 0;
          a1 = v5;
          if ( (unsigned int)sub_6726E0(v5, 0) )
          {
            *(_WORD *)(v5 + 124) |= 0x180u;
            v327 |= 4u;
            v325 = 20;
            v292 = 1;
            *(_QWORD *)(v5 + 104) = *(_QWORD *)&dword_4F063F8;
            goto LABEL_186;
          }
        }
        *(_WORD *)(v5 + 124) |= 0x180u;
        v8 = dword_4F07758;
        *(_QWORD *)(v5 + 104) = *(_QWORD *)&dword_4F063F8;
        a1 = (v327 & 0xFFFFFFFFFFFFFFB7LL) == 0;
        v296 = (v327 & 0xFFFFFFFFFFFFFFB7LL) == 0;
        if ( !(_DWORD)v8 )
          goto LABEL_644;
        a4 = dword_4F0775C;
        if ( !dword_4F0775C && (!(_DWORD)qword_4F077B4 || dword_4F077C4 != 2 || !qword_4F077A0) )
          goto LABEL_644;
        if ( (v327 & 4) != 0 )
          goto LABEL_186;
        v8 = 0;
        sub_7ADF70(&v331, 0);
        while ( 1 )
        {
          sub_7AE360(&v331);
          sub_7B8B50(&v331, 0, v194, v195);
          if ( (unsigned __int16)(word_4F06418[0] - 27) > 0x2Eu )
          {
            if ( word_4F06418[0] == 1 )
            {
              v198 = &v331;
              v199 = *(_QWORD *)(qword_4D04A00 + 24);
              if ( !v199 )
                goto LABEL_1059;
              while ( 1 )
              {
                v200 = *(_BYTE *)(v199 + 80);
                if ( v200 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v200 - 4) <= 2u )
                  break;
                if ( v200 == 19 )
                {
                  v8 = 0;
                  v282 = v198;
                  v230 = sub_7BE840(0, 0);
                  v198 = v282;
                  if ( v230 == 43 )
                    break;
                  v200 = *(_BYTE *)(v199 + 80);
                }
                if ( v200 == 23 )
                  break;
                v199 = *(_QWORD *)(v199 + 8);
                if ( !v199 )
                  goto LABEL_1059;
              }
LABEL_1004:
              a1 = (unsigned __int64)v198;
              sub_7BC000(v198);
LABEL_186:
              if ( v6 )
LABEL_187:
                *(_QWORD *)(v6 + 40) = qword_4F063F0;
LABEL_188:
              unk_4F061D8 = qword_4F063F0;
              sub_7B8B50(a1, v8, qword_4F063F0, a4);
LABEL_189:
              v12 = word_4F06418[0];
              goto LABEL_178;
            }
          }
          else
          {
            v196 = 0x4000220000C1LL;
            if ( _bittest64(&v196, (unsigned int)word_4F06418[0] - 27) )
            {
              v198 = &v331;
LABEL_1059:
              sub_7BC000(v198);
LABEL_644:
              v8 = v296;
              a1 = v292;
              sub_668A70(v292, v296, v7, v5, v6, &v327, &v325, v298, &v323);
              goto LABEL_186;
            }
          }
          v197 = 1;
          if ( (unsigned __int16)(word_4F06418[0] - 81) <= 0x1Au )
            v197 = (((unsigned __int64)&unk_4080001 >> (LOBYTE(word_4F06418[0]) - 81)) & 1) == 0;
          if ( word_4F06418[0] != 244 && v197 )
          {
            v198 = &v331;
            goto LABEL_1004;
          }
        }
      case 80:
      case 85:
      case 89:
      case 93:
      case 120:
      case 126:
      case 127:
      case 128:
      case 165:
      case 180:
      case 331:
      case 332:
      case 333:
      case 334:
      case 335:
        goto LABEL_85;
      case 81:
        if ( (v307 & 1) != 0 )
        {
          if ( dword_4F077C4 == 2 || (a1 = 5, unk_4F07778 <= 199900) )
          {
            a1 = 5;
            if ( dword_4D04964 )
              a1 = unk_4F07471;
          }
          goto LABEL_461;
        }
        a4 = *(unsigned int *)(v5 + 72);
        if ( !(_DWORD)a4 )
          *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
        v307 |= 1u;
        v327 |= 2u;
        v17 = v7 & 0x400;
        goto LABEL_186;
      case 87:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( v325 )
        {
          sub_6851C0(84, dword_4F07508);
          v8 = v7;
          a1 = v5;
          sub_66F9E0(v5, v7, 0, 0, &v331, 0, (unsigned int *)&v329, 0, v6);
          v304 = 1;
        }
        else
        {
          v8 = v7;
          a1 = v5;
          sub_66F9E0(v5, v7, v9, 0, v298, v5 + 232, (unsigned int *)&v323 + 1, &v324, v6);
          v201 = *(_QWORD *)(v5 + 272);
          for ( i = *(unsigned __int8 *)(v201 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v201 + 140) )
            v201 = *(_QWORD *)(v201 + 160);
          if ( (_BYTE)i )
          {
            v325 = 22;
            v297 = 1;
          }
          else
          {
            LODWORD(v323) = 1;
            v325 = 26;
          }
        }
        goto LABEL_628;
      case 88:
        if ( dword_4F077C4 != 2 )
          goto LABEL_465;
        if ( (unsigned __int16)sub_7BE840(0, 0) == 7 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 335;
          v17 = v7 & 0x400;
          sub_6851C0(335, dword_4F07508);
          LODWORD(v323) = 1;
          sub_7B8B50(335, dword_4F07508, v204, v205);
          goto LABEL_186;
        }
        v13 = (unsigned __int64)word_4F06418;
        LOWORD(v12) = word_4F06418[0];
LABEL_465:
        v246 = (unsigned __int16 *)v13;
        a1 = (unsigned __int16)v12;
        v8 = v7;
        v17 = v7 & 0x400;
        sub_668230((unsigned __int16)v12, v7, v5, v6, (v327 & 0xFFFFFFFFFFFFCFB7LL) == 0, &v327, &v323);
        i = v245;
        a4 = (__int64)v246;
        goto LABEL_189;
      case 94:
      case 97:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( !v305 )
        {
          v327 |= 4u;
          v305 = ((_WORD)v12 != 97) + 1;
          goto LABEL_186;
        }
        if ( (_WORD)v12 == 94 && v305 == 2 )
        {
          v305 = 3;
          if ( dword_4D04964 && !unk_4D04298 )
          {
            v8 = 450;
            a1 = unk_4F07471;
            sub_684AC0(unk_4F07471, 450);
          }
          goto LABEL_186;
        }
        if ( dword_4F077C0 && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9C3Fu )
        {
          v300 = 84;
          v330 = *(_QWORD *)&dword_4F063F8;
          goto LABEL_186;
        }
        if ( v305 == 1 && (_WORD)v12 == 97 )
        {
          v8 = 240;
          a1 = qword_4D0495C == 0 ? 8 : 5;
          sub_684AC0(a1, 240);
          v305 = 1;
          goto LABEL_186;
        }
        goto LABEL_185;
      case 95:
      case 100:
      case 174:
      case 193:
      case 194:
        goto LABEL_465;
      case 98:
      case 105:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( v302 )
        {
          if ( (v302 == 1) != ((_WORD)v12 == 98) )
            goto LABEL_185;
          a1 = 5;
          if ( !qword_4D0495C )
          {
            a1 = 8;
            if ( dword_4F077C0 )
            {
              if ( !(_DWORD)qword_4F077B4 )
              {
                a1 = 5;
                if ( qword_4F077A8 > 0x765Bu && (*(_BYTE *)(v5 + 268) != 4 || qword_4F077A8 > 0x9C3Fu) )
                  a1 = 8;
              }
            }
          }
          v8 = 240;
          sub_684AA0(a1, 240, &dword_4F063F8);
        }
        else
        {
          v327 |= 4u;
          v302 = ((_WORD)v12 != 98) + 1;
        }
        goto LABEL_186;
      case 101:
      case 104:
      case 151:
        goto LABEL_194;
      case 103:
        v312 = dword_4D04964;
        sub_6729D0(v5);
        v17 = v7 & 0x400;
        a1 = word_4F06418[0];
        v8 = v7;
        sub_668230(word_4F06418[0], v7, v5, v6, (v327 & 0xFFFFFFFFFFFFCFB7LL) == 0, &v327, &v323);
        if ( !v312 )
        {
          v11 = 1;
          v12 = word_4F06418[0];
          goto LABEL_13;
        }
        v312 = 0;
        goto LABEL_189;
      case 106:
        if ( dword_4F077C4 == 1 )
        {
          a4 = v325;
          if ( v325 )
          {
            if ( *(_BYTE *)(v5 + 268) == 4 )
            {
              v15 = v307 & 0x7F;
              v14 = (v11 ^ 1) & (v6 != 0);
              v11 = 0;
              goto LABEL_44;
            }
            if ( v306 )
              goto LABEL_567;
          }
          else if ( v306 )
          {
LABEL_813:
            v325 = 1;
            goto LABEL_604;
          }
        }
        else
        {
LABEL_85:
          if ( v306 )
          {
            if ( !v325 )
            {
              if ( (unsigned __int16)v12 <= 0x80u )
              {
                if ( (unsigned __int16)v12 > 0x4Fu )
                {
                  switch ( (__int16)v12 )
                  {
                    case 80:
                      v325 = 2;
                      goto LABEL_702;
                    case 85:
                      v325 = 15;
                      goto LABEL_702;
                    case 89:
                      v325 = 12;
                      goto LABEL_702;
                    case 93:
                      v325 = 8;
                      goto LABEL_702;
                    case 106:
                      goto LABEL_813;
                    case 120:
                      if ( dword_4F077C4 != 2 && unk_4F07778 > 202310 )
                      {
                        if ( qword_4D04A00 )
                        {
                          if ( *(_QWORD *)(qword_4D04A00 + 16) == 5 )
                          {
                            a1 = *(_QWORD *)(qword_4D04A00 + 8);
                            v8 = (__int64)"_Bool";
                            v258 = v12;
                            v278 = v11;
                            v158 = strncmp((const char *)a1, "_Bool", 5u);
                            LOBYTE(v11) = v278;
                            LOWORD(v12) = v258;
                            if ( !v158 )
                            {
                              a1 = dword_4F063F8;
                              v159 = sub_729F80(dword_4F063F8);
                              LOBYTE(v11) = v278;
                              if ( !v159 )
                              {
                                sub_684AA0(5 - (unsigned int)(dword_4D04964 == 0), 3292, &dword_4F063F8);
                                v8 = 1;
                                a1 = 3292;
                                sub_67D850(3292, 1, 0);
                                LOBYTE(v11) = v278;
                              }
                              LOWORD(v12) = word_4F06418[0];
                            }
                          }
                        }
                      }
                      goto LABEL_603;
                    case 126:
                      v325 = 5;
                      goto LABEL_702;
                    case 127:
                      v325 = 6;
                      goto LABEL_702;
                    case 128:
                      v325 = 4;
                      goto LABEL_702;
                    default:
                      goto LABEL_703;
                  }
                }
                goto LABEL_703;
              }
              if ( (unsigned __int16)v12 > 0x14Fu )
                goto LABEL_703;
              if ( (unsigned __int16)v12 > 0x14Au )
              {
                switch ( (__int16)v12 )
                {
                  case 332:
                    v325 = 14;
                    break;
                  case 333:
                    v325 = 16;
                    break;
                  case 334:
                    v325 = 17;
                    break;
                  case 335:
                    v325 = 19;
                    break;
                  default:
                    v325 = 13;
                    break;
                }
                goto LABEL_702;
              }
              if ( (_WORD)v12 == 165 )
              {
                v325 = 3;
LABEL_702:
                v327 |= 4u;
                v17 = v7 & 0x400;
                goto LABEL_186;
              }
              if ( (_WORD)v12 != 180 )
                goto LABEL_703;
LABEL_603:
              v325 = 7;
LABEL_604:
              v17 = v7 & 0x400;
              if ( (_WORD)v12 == 106 && (v11 & 1) == 0 )
              {
                v327 = 2048;
                goto LABEL_186;
              }
              goto LABEL_702;
            }
LABEL_567:
            if ( dword_4F077C0 )
            {
              if ( (_DWORD)qword_4F077B4 )
                goto LABEL_684;
              if ( qword_4F077A8 <= 0x9C3Fu )
              {
LABEL_574:
                v300 = 84;
                v17 = v7 & 0x400;
                v330 = *(_QWORD *)&dword_4F063F8;
                goto LABEL_186;
              }
              v8 = dword_4F077BC;
              if ( !dword_4F077BC )
              {
LABEL_684:
                v8 = (__int64)dword_4F07508;
                a1 = 84;
                v17 = v7 & 0x400;
                sub_6851C0(84, dword_4F07508);
                v304 = 1;
                goto LABEL_186;
              }
            }
            else if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || !qword_4F077A8 )
            {
              goto LABEL_684;
            }
            if ( (_WORD)v12 == 180 || (_WORD)v12 == 165 || (unsigned __int16)(v12 - 331) <= 4u )
            {
              a1 = dword_4F063F8;
              if ( (unsigned int)sub_729F80(dword_4F063F8) )
                goto LABEL_574;
            }
            goto LABEL_684;
          }
        }
LABEL_197:
        v8 = (__int64)dword_4F07508;
        a1 = 87;
        v17 = v7 & 0x400;
        sub_6851C0(87, dword_4F07508);
        LODWORD(v323) = 1;
        goto LABEL_186;
      case 107:
        if ( (v307 & 2) != 0 )
        {
          if ( dword_4F077C4 == 2 || (a1 = 5, unk_4F07778 <= 199900) )
          {
            a1 = 5;
            if ( dword_4D04964 )
              a1 = unk_4F07471;
          }
          goto LABEL_461;
        }
        if ( !*(_DWORD *)(v5 + 72) )
          *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
        v307 |= 2u;
        v327 |= 2u;
        v17 = v7 & 0x400;
        goto LABEL_186;
      case 118:
      case 119:
        if ( (v307 & 4) != 0 )
        {
          if ( dword_4F077C4 == 2 || (a1 = 5, unk_4F07778 <= 199900) )
          {
            a1 = 5;
            if ( dword_4D04964 )
              a1 = unk_4F07471;
          }
LABEL_461:
          v8 = 83;
          sub_684AC0(a1, 83);
          v17 = v7 & 0x400;
          goto LABEL_186;
        }
        v307 |= 4u;
        v327 |= 2u;
        *(_QWORD *)(v5 + 80) = *(_QWORD *)&dword_4F063F8;
        if ( !HIDWORD(qword_4F077B4) )
        {
          v17 = v7 & 0x400;
          goto LABEL_186;
        }
        if ( !unk_4D04320 )
        {
          v17 = v7 & 0x400;
          goto LABEL_186;
        }
        if ( (_WORD)v12 == 119 )
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 1614;
          v17 = v7 & 0x400;
          sub_684B30(1614, &dword_4F063F8);
          goto LABEL_186;
        }
        if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
        {
          v17 = v7 & 0x400;
          goto LABEL_186;
        }
        goto LABEL_703;
      case 121:
        v17 = v7 & 0x400;
        if ( v299 == 1 )
          goto LABEL_955;
        if ( v299 == 2 )
          goto LABEL_185;
        v299 = 1;
        goto LABEL_186;
      case 122:
        v17 = v7 & 0x400;
        if ( v299 == 2 )
        {
LABEL_955:
          v8 = (__int64)dword_4F07508;
          a1 = 240;
          sub_6851C0(240, dword_4F07508);
        }
        else if ( v299 == 1 )
        {
LABEL_185:
          v8 = (__int64)dword_4F07508;
          a1 = 84;
          sub_6851C0(84, dword_4F07508);
          v304 = 1;
        }
        else
        {
          v299 = 2;
        }
        goto LABEL_186;
      case 133:
      case 134:
      case 135:
      case 136:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( v325 | v305 )
          goto LABEL_185;
        if ( (_WORD)v12 == 133 )
        {
          v305 = 4;
          v181 = 2;
        }
        else
        {
          v305 = v12 - 129;
          v181 = 8;
        }
        v327 |= 4u;
        v325 = v181;
        goto LABEL_186;
      case 142:
      case 248:
        goto LABEL_198;
      case 146:
        goto LABEL_475;
      case 153:
        v17 = v7 & 0x400;
        if ( v301 )
          goto LABEL_875;
        if ( !v291 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 239;
          sub_6851C0(239, dword_4F07508);
          LODWORD(v323) = 1;
          goto LABEL_186;
        }
        v80 = v327;
        if ( (v327 & 8) != 0 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 240;
          sub_6851C0(240, dword_4F07508);
          LODWORD(v323) = 1;
          goto LABEL_186;
        }
        *(_QWORD *)(v5 + 8) |= 8uLL;
        v327 = v80 | 8;
        if ( (v80 | 8) != 8 )
        {
          if ( (v80 & 1) != 0 )
          {
            v8 = (__int64)dword_4F07508;
            a1 = 784;
            sub_6851C0(784, dword_4F07508);
            v327 &= ~1uLL;
            LODWORD(v323) = 1;
            *(_BYTE *)(v5 + 268) = 0;
          }
          else if ( (v80 & 0x80u) != 0LL )
          {
            v8 = v5 + 260;
            a1 = 719;
            sub_6851C0(719, v5 + 260);
            v327 &= ~0x80uLL;
            *(_QWORD *)(v5 + 8) &= ~0x1000uLL;
            LODWORD(v323) = 1;
          }
          goto LABEL_186;
        }
        if ( !qword_4D0495C )
          goto LABEL_186;
        sub_7B8B50(a1, v8, (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C), a4);
        if ( dword_4F077C4 == 2 )
        {
          if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
          {
            v8 = 0;
            a1 = 0;
            if ( !(unsigned int)sub_7C0F00(0, 0) )
            {
LABEL_1142:
              sub_7BEC40();
              word_4F06418[0] = 153;
              goto LABEL_186;
            }
          }
        }
        else if ( word_4F06418[0] != 1 )
        {
          goto LABEL_1142;
        }
        v331 = *(_QWORD *)&dword_4F063F8;
        v8 = 0;
        v207 = sub_7BF130(0, 2, &v329);
        v208 = sub_7BE840(0, 0);
        if ( v207 || v208 != 75 )
        {
          if ( (unk_4D04A11 & 0x40) == 0 )
          {
            unk_4D04A10 &= ~0x80u;
            unk_4D04A18 = 0;
          }
          sub_7BEC40();
          a1 = 153;
          word_4F06418[0] = 153;
          goto LABEL_186;
        }
        LODWORD(v9) = 0;
        sub_684AE0(451, &v331, "class");
LABEL_194:
        v17 = v7 & 0x400;
        if ( !v306 )
        {
LABEL_195:
          v8 = (__int64)dword_4F07508;
          a1 = 87;
          sub_6851C0(87, dword_4F07508);
          LODWORD(v323) = 1;
          goto LABEL_186;
        }
        if ( v325 )
        {
          sub_6851C0(84, dword_4F07508);
          v8 = v7;
          a1 = v5;
          sub_66AC40(v5, v7, 0, 0, v312, &v331, &v329, &v329, v6);
          v304 = 1;
        }
        else
        {
          v8 = v7;
          a1 = v5;
          if ( !(unsigned int)sub_66AC40(v5, v7, v9, (v327 >> 3) & 1, v312, v298, (_DWORD *)&v323 + 1, &v324, v6) )
            LODWORD(v323) = 1;
          v325 = 21;
          v297 = 1;
        }
LABEL_628:
        v327 |= 4u;
        v12 = word_4F06418[0];
        goto LABEL_178;
      case 154:
        v17 = v7 & 0x400;
        if ( v301 )
          goto LABEL_875;
        if ( (v7 & 0x20) == 0 )
          goto LABEL_318;
        if ( dword_4F077C4 == 2 )
        {
          if ( !dword_4D04824 && *(_BYTE *)(v5 + 268) == 1 )
          {
LABEL_318:
            v8 = 326;
            a1 = HIDWORD(qword_4F077B4) == 0 ? 8 : 5;
            sub_684AC0(a1, 326);
            LODWORD(v323) = HIDWORD(qword_4F077B4) == 0;
            goto LABEL_186;
          }
          v70 = v327;
          a1 = 8;
          if ( (v327 & 0x40) == 0 )
            goto LABEL_316;
        }
        else
        {
          v70 = v327;
          if ( (v327 & 0x40) == 0 )
          {
LABEL_316:
            if ( (v7 & 0x400) != 0 )
            {
              v8 = (__int64)dword_4F07508;
              a1 = 973;
              sub_684B30(973, dword_4F07508);
            }
            else
            {
              *(_QWORD *)(v5 + 8) |= 2uLL;
              v327 = v70 | 0x40;
              *(_QWORD *)(v5 + 88) = *(_QWORD *)&dword_4F063F8;
            }
            goto LABEL_186;
          }
          a1 = 3 * (unsigned int)(unk_4F07778 < 199901) + 5;
        }
        v8 = 240;
        sub_684AC0(a1, 240);
        if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 )
          LODWORD(v323) = 1;
        goto LABEL_186;
      case 156:
        if ( dword_4F077C4 == 2 )
        {
          a1 = 0;
          v320 = v11;
          sub_7C0F00(0, 0);
          LOBYTE(v11) = v320;
        }
        goto LABEL_321;
      case 160:
        v71 = _mm_loadu_si128(xmmword_4F06660);
        v72 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v73 = _mm_loadu_si128(&xmmword_4F06660[3]);
        unk_4D04A10 = _mm_loadu_si128(&xmmword_4F06660[1]);
        unk_4D04A11 |= 0x20u;
        *(__m128i *)&qword_4D04A00 = v71;
        xmmword_4D04A20 = v72;
        qword_4D04A08 = *(_QWORD *)&dword_4F063F8;
        unk_4D04A30 = v73;
        sub_6851C0(463, &dword_4F063F8);
        v8 = 0;
        a1 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) == 43 )
          sub_7BE180();
        else
          sub_7B8B50(0, 0, v74, v75);
        LODWORD(v323) = 1;
        v17 = v7 & 0x400;
        if ( v325 == 20 )
          *(_QWORD *)(v5 + 272) = sub_72C930(0);
        else
          v325 = 26;
        v12 = word_4F06418[0];
        goto LABEL_178;
      case 161:
        if ( !unk_4D048A4 )
          goto LABEL_8;
        v17 = v7 & 0x400;
        if ( v301 )
        {
          v76 = v327;
          if ( (v327 & 0x8000) != 0 )
          {
            v8 = (__int64)dword_4F07508;
            a1 = 240;
            sub_684B30(240, dword_4F07508);
          }
          else
          {
            a4 = (__int64)qword_4F04C68;
            if ( **(_QWORD **)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 208) + 168LL) )
            {
              v8 = (__int64)dword_4F07508;
              a1 = 3211;
              sub_6851C0(3211, dword_4F07508);
              LODWORD(v323) = 1;
            }
            else
            {
              BYTE1(v76) = BYTE1(v327) | 0x80;
              v327 = v76;
              *(_BYTE *)(v5 + 133) |= 0x40u;
            }
          }
        }
        else
        {
          v8 = (__int64)dword_4F07508;
          a1 = 3212;
          sub_6851C0(3212, dword_4F07508);
          LODWORD(v323) = 1;
        }
        goto LABEL_186;
      case 164:
        v17 = v7 & 0x400;
        if ( v301 )
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 238;
          sub_6851C0(238, &dword_4F063F8);
          LODWORD(v323) = 1;
        }
        else
        {
          v77 = BYTE1(v327);
          if ( (v327 & 8) != 0 )
          {
            v8 = (__int64)&dword_4F063F8;
            a1 = 377;
            sub_6851C0(377, &dword_4F063F8);
            LODWORD(v323) = 1;
          }
          else if ( v291 )
          {
            if ( (v7 & 0x200) != 0 && (*(_BYTE *)(v5 + 129) & 0x10) == 0 )
            {
              v8 = (__int64)&dword_4F063F8;
              a1 = 774;
              sub_6851C0(774, &dword_4F063F8);
              LODWORD(v323) = 1;
            }
            else if ( (v327 & 0x10) != 0 )
            {
              v8 = 240;
              a1 = 8;
              sub_684AA0(8, 240, &dword_4F063F8);
              LODWORD(v323) = 1;
            }
            else
            {
              a1 = (unsigned __int64)&dword_4F063F8;
              v327 |= 0x10u;
              v78 = *(_QWORD *)(v5 + 8);
              *(_QWORD *)(v5 + 8) = v78 | 4;
              a4 = *(_QWORD *)&dword_4F063F8;
              *(_QWORD *)(v5 + 96) = *(_QWORD *)&dword_4F063F8;
              if ( (v77 & 0x10) != 0 )
              {
                v8 = (unsigned int)dword_4D04888;
                if ( !dword_4D04888 )
                {
                  v8 = (__int64)&dword_4F063F8;
                  a1 = (v78 & 0x80000) == 0 ? 2924 : 2390;
                  sub_6851C0(a1, &dword_4F063F8);
                  *(_QWORD *)(v5 + 8) &= ~0x80000uLL;
                }
              }
            }
          }
          else
          {
            v8 = (__int64)&dword_4F063F8;
            a1 = 239;
            sub_6851C0(239, &dword_4F063F8);
            LODWORD(v323) = 1;
          }
        }
        goto LABEL_186;
      case 168:
        v17 = v7 & 0x400;
        if ( v301 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 238;
          sub_6851C0(238, dword_4F07508);
          LODWORD(v323) = 1;
          v292 = 0;
          goto LABEL_186;
        }
        v79 = v327;
        if ( (v327 & 0x20) != 0 || (*(_BYTE *)(v5 + 132) & 1) != 0 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 240;
          sub_6851C0(240, dword_4F07508);
          LODWORD(v323) = 1;
          v292 = 0;
          goto LABEL_186;
        }
        if ( dword_4D0483C )
          goto LABEL_354;
        if ( dword_4F077BC )
        {
          if ( !(_DWORD)qword_4F077B4 )
          {
            if ( qword_4F077A8 <= 0x15F8Fu )
              goto LABEL_356;
            goto LABEL_354;
          }
        }
        else if ( !(_DWORD)qword_4F077B4 )
        {
          goto LABEL_356;
        }
        if ( qword_4F077A0 <= 0x1869Fu )
          goto LABEL_356;
LABEL_354:
        if ( (unsigned __int16)sub_7BE840(0, 0) == 27 )
        {
          v215 = (unsigned int *)dword_4D0483C;
          if ( !dword_4D0483C )
          {
            v215 = &dword_4F063F8;
            sub_684B30(3401, &dword_4F063F8);
          }
          v253 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          v260 = sub_727670();
          v281 = sub_7276D0();
          v293 = sub_724D80(0);
          sub_7B8B50(0, v215, v216, v217);
          *(_BYTE *)(v260 + 8) = 84;
          *(_QWORD *)(v260 + 16) = sub_724840(unk_4F073B8, "explicit");
          *(_QWORD *)(v260 + 56) = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v260 + 32) = v281;
          sub_7B8B50(&dword_4F063F8, "explicit", v218, v219);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
            || *(char *)(v253 + 6) < 0 )
          {
            sub_6B9B50(v293);
          }
          else
          {
            *(_BYTE *)(v253 + 6) |= 0x80u;
            sub_6B9B50(v293);
            *(_BYTE *)(v253 + 6) &= ~0x80u;
          }
          v220 = *(_QWORD *)&dword_4F063F8;
          *(_BYTE *)(v281 + 10) = 3;
          *(_QWORD *)(v281 + 24) = v220;
          *(_QWORD *)(v281 + 40) = v293;
          *(_QWORD *)(v281 + 32) = unk_4F061D8;
          sub_7BE5B0(28, 18, 0, 0);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          v221 = *(_QWORD *)(v5 + 184);
          *(_BYTE *)(v5 + 132) |= 1u;
          *(_QWORD *)v260 = v221;
          *(_QWORD *)(v5 + 184) = v260;
          goto LABEL_357;
        }
        v79 = v327;
LABEL_356:
        *(_QWORD *)(v5 + 8) |= 0x2000uLL;
        v327 = v79 | 0x20;
LABEL_357:
        v8 = v5;
        a1 = (unsigned __int64)sub_667110;
        sub_643E40((__int64)sub_667110, v5, 0);
        v292 = 0;
        goto LABEL_186;
      case 170:
        v8 = (__int64)&dword_4F063F8;
        a1 = 3103;
        v17 = v7 & 0x400;
        sub_6851C0(3103, &dword_4F063F8);
        goto LABEL_186;
      case 183:
        if ( !v306 )
          goto LABEL_197;
        if ( dword_4D041A8 )
        {
          a1 = 0;
          v280 = v11;
          v187 = sub_7BE840(0, 0);
          v11 = v280;
          if ( v187 == 25 )
          {
            v81 = word_4F06418[0];
            goto LABEL_381;
          }
        }
        v17 = v7 & 0x400;
        if ( v325 )
        {
          v8 = 1554;
          a1 = 7;
          sub_684AC0(7, 1554);
          goto LABEL_186;
        }
        a1 = (unsigned __int64)v298;
        v8 = (__int64)&v331;
        v267 = v11;
        sub_671BC0((__int64)v298, (__int64)&v331, 0, 1, (_QWORD *)v5, v6);
        v11 = v267;
        if ( *(_QWORD *)(v5 + 272) )
        {
          v327 |= 4u;
          v325 = 23;
          v297 = 1;
          goto LABEL_189;
        }
        if ( unk_4F04C48 == -1
          || (i = (__int64)qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0) )
        {
          if ( dword_4F04C44 == -1 )
            goto LABEL_189;
        }
        goto LABEL_913;
      case 185:
        if ( !dword_4F07750 )
        {
LABEL_475:
          v252 = dword_4F063F8;
          v287 = word_4F063FC[0];
          if ( dword_4F077C4 != 2 )
          {
LABEL_384:
            v82 = v305 | v302;
            if ( v305 | v302 )
            {
              v16 = dword_4F077C4;
              if ( dword_4F077C4 != 1 )
              {
                v12 = dword_4F077BC;
                if ( !dword_4F077BC )
                  goto LABEL_17;
                a5 = (unsigned int)qword_4F077B4;
                if ( (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x76BFu )
                  goto LABEL_936;
                if ( (v7 & 0x80000) == 0 )
                  goto LABEL_17;
              }
            }
            if ( v301 || (v83 = 0, (*(_QWORD *)(v5 + 120) & 0x4000100000LL) != 0) )
              v83 = unk_4D04494 != 0;
            v84 = 0;
            if ( *(char *)(v5 + 132) < 0 )
              v84 = unk_4D04874 != 0;
            a1 = v294 != 0;
            v8 = 0;
            v268 = v11;
            v85 = sub_6512E0(a1, 0, 0, v84, 0, v83);
            v11 = v268;
            a5 = v85;
            if ( !v85 )
              goto LABEL_528;
            if ( *(_BYTE *)(v85 + 80) != 22 )
            {
              if ( dword_4F077C4 != 2 || !v291 || (v327 & 4) != 0 || (unk_4D04A10 & 1) != 0 )
                goto LABEL_398;
              v87 = *(_BYTE *)(v85 + 80);
              if ( v87 != 3 )
              {
                v86 = unk_4D04A12;
                if ( (unk_4D04A12 & 2) == 0 )
                  goto LABEL_402;
LABEL_400:
                if ( !dword_4F077BC || (v86 & 1) == 0 || (unsigned __int8)(v87 - 4) > 1u )
                  goto LABEL_402;
                a1 = *(_QWORD *)(a5 + 88);
                if ( (*(_BYTE *)(a1 + 177) & 0x10) == 0 || *(_QWORD *)a5 != **(_QWORD **)xmmword_4D04A20.m128i_i64[0] )
                {
                  if ( !v82 )
                    goto LABEL_405;
LABEL_404:
                  v249 = v11;
                  v255 = (__int64 *)a5;
                  v88 = sub_8D2930(a1);
                  a5 = (unsigned __int64)v255;
                  if ( v88 )
                  {
LABEL_405:
                    if ( dword_4F077C4 == 2 && unk_4D04A18 && (*(_DWORD *)(unk_4D04A18 + 80LL) & 0x41000) != 0 )
                    {
                      v8 = 0;
                      a1 = (unsigned __int64)&qword_4D04A00;
                      v277 = (__int64 *)a5;
                      sub_8841F0(&qword_4D04A00, 0, 0, 0);
                      a5 = (unsigned __int64)v277;
                    }
                    if ( (unk_4D04A11 & 0x20) != 0 )
                    {
                      v327 |= 4u;
                      LODWORD(v323) = 1;
                      v325 = 20;
                      *(_QWORD *)(v5 + 272) = sub_72C930(a1);
                    }
                    else
                    {
                      v89 = *(_BYTE *)(a5 + 80);
                      v90 = *(_QWORD *)(a5 + 88);
                      if ( v89 != 3 && v89 != 6 )
                      {
                        if ( unk_4D04A10 >= 0 )
                          goto LABEL_410;
                        goto LABEL_562;
                      }
                      if ( unk_4D04A10 < 0 )
                      {
LABEL_562:
                        v274 = a5;
                        sub_685440(unk_4F07470, 406, unk_4D04A18);
                        a5 = v274;
                        v89 = *(_BYTE *)(v274 + 80);
LABEL_410:
                        if ( v89 == 16 )
                        {
                          a5 = **(_QWORD **)(a5 + 88);
                          v89 = *(_BYTE *)(a5 + 80);
                        }
                        if ( v89 == 24 )
                          a5 = *(_QWORD *)(a5 + 88);
                      }
                      a1 = 4;
                      v269 = a5;
                      sub_8767A0(4, a5, &qword_4D04A08, 1);
                      if ( v306 )
                      {
                        v327 |= 4u;
                        v91 = 0;
                        *(_QWORD *)(v5 + 272) = v90;
                        v23 = *(_BYTE *)(v269 + 80) == 3;
                        v325 = 20;
                        if ( v23 )
                          v91 = *(_BYTE *)(v269 + 104) != 0;
                        *(_BYTE *)(v5 + 132) = (32 * v91) | *(_BYTE *)(v5 + 132) & 0xDF;
                        v8 = dword_4D04808;
                        if ( dword_4D04808 && *(_BYTE *)(v90 + 140) == 14 && (unk_4D04A12 & 1) == 0 )
                        {
                          if ( *(_DWORD *)(*(_QWORD *)(v90 + 168) + 28LL) == -2 )
                          {
                            *(_WORD *)(v5 + 124) |= 0x480u;
                            *(_QWORD *)(v5 + 304) = v90;
                            *(_QWORD *)(v5 + 104) = *(_QWORD *)&dword_4F063F8;
                          }
                        }
                        else
                        {
                          a4 = unk_4D04A18;
                          if ( *(_BYTE *)(unk_4D04A18 + 80LL) == 16
                            && (*(_BYTE *)(unk_4D04A18 + 96LL) & 4) == 0
                            && (unk_4D04A10 & 1) != 0
                            && dword_4F04C64 != -1
                            && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0 )
                          {
                            if ( *(_QWORD *)(*(_QWORD *)(unk_4D04A18 + 88LL) + 16LL) )
                            {
                              v90 = *(_QWORD *)(*(_QWORD *)(unk_4D04A18 + 88LL) + 16LL);
                            }
                            else
                            {
                              v250 = unk_4D04A18;
                              v283 = *(_QWORD *)(unk_4D04A18 + 64LL);
                              v236 = sub_7259C0(12);
                              sub_72EE40(v236, 6, *(_QWORD *)(*(_QWORD *)(v283 + 168) + 152LL));
                              sub_877D80(v236, v250);
                              sub_877E20(v250, v236, v283);
                              *(_QWORD *)(v236 + 160) = v90;
                              v8 = 0;
                              a1 = v236;
                              *(_BYTE *)(v236 + 187) |= 2u;
                              v90 = v236;
                              sub_7365B0(v236, 0);
                              a4 = v250;
                              *(_QWORD *)(*(_QWORD *)(v250 + 88) + 16LL) = v236;
                            }
                          }
                          *(_QWORD *)(v5 + 272) = v90;
                        }
                      }
                      else
                      {
                        v8 = (__int64)dword_4F07508;
                        a1 = 87;
                        sub_6851C0(87, dword_4F07508);
                        LODWORD(v323) = 1;
                      }
                    }
                    v17 = v7 & 0x400;
                    if ( v6 )
                    {
                      *(_DWORD *)(v6 + 16) = v252;
                      *(_WORD *)(v6 + 20) = v287;
                      *(_QWORD *)(v6 + 24) = qword_4F063F0;
                      goto LABEL_187;
                    }
                    goto LABEL_188;
                  }
                  v171 = sub_8D2A90(a1);
                  a5 = (unsigned __int64)v255;
                  LOBYTE(v11) = v249;
                  if ( v171 )
                  {
                    a1 = v302;
                    if ( v302 || v305 != 2 )
                    {
LABEL_16:
                      v16 = dword_4F077C4;
LABEL_17:
                      v15 = v307 & 0x7F;
                      v14 = (v11 ^ 1) & (v6 != 0);
                      v11 = 0;
                      goto LABEL_43;
                    }
                    goto LABEL_405;
                  }
LABEL_935:
                  v16 = dword_4F077C4;
LABEL_936:
                  v15 = v307 & 0x7F;
                  v14 = (v11 ^ 1) & (v6 != 0);
                  v11 = 0;
                  goto LABEL_43;
                }
                if ( (v327 & 0xFFFFFFFFFFFFECB7LL) != 0 )
                {
LABEL_402:
                  if ( v82 )
                  {
                    a1 = *(_QWORD *)(a5 + 88);
                    goto LABEL_404;
                  }
                  goto LABEL_405;
                }
LABEL_920:
                if ( v294 )
                  goto LABEL_402;
                v8 = 0;
                a1 = 0;
                v259 = v11;
                v279 = (__int64 *)a5;
                v173 = sub_7BE840(0, 0);
                a5 = (unsigned __int64)v279;
                v11 = v259;
                if ( v173 != 27 )
                  goto LABEL_402;
LABEL_922:
                v174 = *(_QWORD *)(a5 + 88);
                for ( n = v174; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                  ;
                v176 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)n + 96LL) + 8LL);
                if ( v176 )
                {
                  if ( (unk_4D04A12 & 2) == 0 )
                    BUG();
                  for ( ii = xmmword_4D04A20.m128i_i64[0]; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
                    ;
                  if ( ii == v174
                    || (v8 = (__int64)&dword_4F07588, (v12 = dword_4F07588) != 0)
                    && (v178 = *(_QWORD *)(ii + 32), *(_QWORD *)(v174 + 32) == v178)
                    && v178 )
                  {
                    v179 = *(_QWORD *)(v5 + 8);
                    a5 = (unsigned __int64)&qword_4D04A00;
                    v325 = 25;
                    v180 = v179;
                    BYTE1(v179) |= 4u;
                    BYTE1(v180) |= 5u;
                    if ( !v11 )
                      v179 = v180;
                    *(_QWORD *)(v5 + 8) = v179;
                    unk_4D04A18 = v176;
                    qword_4D04A00 = *v176;
                    goto LABEL_935;
                  }
                }
                goto LABEL_402;
              }
              v167 = *(_QWORD *)(v85 + 88);
              v168 = *(_BYTE *)(v167 + 140);
              if ( v168 == 12 )
              {
                v169 = *(_QWORD *)(a5 + 88);
                do
                {
                  v169 = *(_QWORD *)(v169 + 160);
                  v170 = *(_BYTE *)(v169 + 140);
                }
                while ( v170 == 12 );
                if ( v170 != 14 )
                  goto LABEL_892;
                do
                  v167 = *(_QWORD *)(v167 + 160);
                while ( *(_BYTE *)(v167 + 140) == 12 );
LABEL_1242:
                if ( *(_BYTE *)(v167 + 160) == 1 )
                {
                  v8 = 0;
                  v261 = a5;
                  sub_7ADF70(&v331, 0);
                  sub_7AE360(&v331);
                  sub_7B8B50(&v331, 0, v237, v238);
                  v119 = (__int64 *)v261;
                  v120 = v268;
                  if ( word_4F06418[0] != 27 )
                    goto LABEL_1244;
                  sub_7AE360(&v331);
                  sub_7B8B50(&v331, 0, v117, v118);
                  v119 = (__int64 *)v261;
                  v120 = v268;
                  if ( word_4F06418[0] == 1 )
                  {
                    if ( dword_4F077C4 != 2 )
                      goto LABEL_1244;
                    if ( (unk_4D04A11 & 2) != 0 )
                    {
                      if ( (unk_4D04A12 & 1) == 0 )
                        goto LABEL_1244;
                    }
                    else
                    {
                      v8 = 0;
                      v264 = v268;
                      v286 = v119;
                      v244 = sub_7C0F00(0, 0);
                      v119 = v286;
                      v120 = v264;
                      if ( !v244 || (unk_4D04A12 & 1) == 0 )
                        goto LABEL_1244;
                      if ( dword_4F077C4 != 2 )
                      {
                        if ( word_4F06418[0] != 1 )
                          goto LABEL_526;
                        goto LABEL_527;
                      }
                      if ( word_4F06418[0] != 1 )
                        goto LABEL_1270;
                    }
                    if ( (unk_4D04A11 & 2) != 0 )
                      goto LABEL_527;
                    goto LABEL_1270;
                  }
                  if ( word_4F06418[0] != 34 && word_4F06418[0] != 27 )
                  {
                    if ( dword_4F077C4 != 2 )
                    {
LABEL_526:
                      if ( word_4F06418[0] == 15 )
                        goto LABEL_1244;
                      goto LABEL_527;
                    }
                    if ( word_4F06418[0] != 33
                      && (!dword_4D04474 || word_4F06418[0] != 52)
                      && (!dword_4D0485C || word_4F06418[0] != 25)
                      && word_4F06418[0] != 156 )
                    {
LABEL_1270:
                      v8 = 0;
                      v263 = v120;
                      v285 = v119;
                      v243 = sub_7C0F00(0, 0);
                      v120 = v263;
                      if ( !v243 )
                      {
                        v119 = v285;
                        goto LABEL_526;
                      }
LABEL_527:
                      a1 = (unsigned __int64)&v331;
                      v272 = v120;
                      sub_7BC000(&v331);
                      v11 = v272;
LABEL_528:
                      if ( v294 )
                      {
                        v8 = (__int64)dword_4F07508;
                        a1 = 79;
                        v17 = v7 & 0x400;
                        sub_6851C0(79, dword_4F07508);
                        v325 = 20;
                        LODWORD(v323) = 1;
                        v172 = sub_72C930(79);
                        v327 |= 4u;
                        *(_QWORD *)(v5 + 272) = v172;
                        goto LABEL_186;
                      }
                      if ( (unk_4D04A10 & 0x18) != 0 )
                      {
LABEL_321:
                        v15 = v307 & 0x7F;
                        v14 = (v11 ^ 1) & (v6 != 0);
                        if ( (v7 & 0x80000) != 0 )
                        {
                          a1 = 502;
                          sub_6851D0(502);
                          v325 = 26;
                          v16 = dword_4F077C4;
                          v11 = 0;
                          LODWORD(v323) = 1;
                          v15 = v307 & 0x7F;
                          goto LABEL_43;
                        }
                        if ( (unk_4D04A10 & 0x10) != 0 && !(v325 | v305 | v302) )
                          v325 = 25;
                        v11 = v323;
                        v16 = dword_4F077C4;
                        if ( (_DWORD)v323 )
                        {
                          v11 = 0;
                          goto LABEL_43;
                        }
                        if ( v327 )
                          goto LABEL_43;
                        *(_QWORD *)(v5 + 8) |= 0x100uLL;
                        goto LABEL_44;
                      }
                      if ( (unk_4D04A10 & 0x20) != 0 && (v327 & 8) == 0 )
                      {
                        v273 = v11;
                        v121 = sub_64E7D0();
                        v11 = v273;
                        if ( v121 || (unk_4D04A10 & 1) == 0 )
                        {
LABEL_534:
                          if ( dword_4F077C4 != 2 || !v291 )
                            goto LABEL_8;
                          v122 = *(_QWORD *)(v5 + 8);
                          v123 = v122;
                          if ( !v11 )
                          {
                            BYTE1(v123) = BYTE1(v122) | 1;
                            v122 = v123;
                          }
                          BYTE1(v122) |= 8u;
                          *(_QWORD *)(v5 + 8) = v122;
                          if ( !(v325 | v305 | v302) )
                          {
                            v325 = 25;
                            v305 = 0;
                            v302 = 0;
                          }
                          v292 = 0;
                          v15 = v307 & 0x7F;
                          v14 = (v11 ^ 1) & (v6 != 0);
                          v11 = 0;
LABEL_34:
                          if ( (v327 & 0x41) == 0x41 )
                          {
                            v18 = *(_BYTE *)(v5 + 268);
                            if ( v18 != 2 )
                            {
                              v12 = dword_4D04824;
                              if ( !dword_4D04824 || v18 != 1 )
                              {
                                a1 = 327;
                                v308 = v15;
                                v313 = v11;
                                sub_6851C0(327, v5 + 260);
                                v15 = v308;
                                LODWORD(v323) = 1;
                                v11 = v313;
                              }
                            }
                          }
                          goto LABEL_44;
                        }
                      }
                      if ( (unk_4D04A10 & 0x12000) == 0x12000
                        || !v11 && ((v7 & 0x10) == 0 || (unk_4D04A10 & 0x2002) == 0x2002) )
                      {
LABEL_634:
                        if ( (unk_4D04A11 & 0x20) == 0 )
                        {
                          v8 = 0;
                          v152 = sub_7BF130(v290, 0, &v331);
                          if ( !(_DWORD)v331 )
                          {
                            if ( v152 )
                            {
                              if ( (unk_4D04A12 & 2) != 0
                                && (unsigned __int8)(*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 140) - 9) <= 2u
                                && (*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 177) & 0xB0) == 0x30
                                && !(unsigned int)sub_8D97B0(xmmword_4D04A20.m128i_i64[0]) )
                              {
                                v8 = v152;
                                sub_6854E0(2675, v152);
                              }
                              else
                              {
                                v8 = v152;
                                sub_6854E0(757, v152);
                              }
                            }
                            else
                            {
                              v8 = *(_QWORD *)(qword_4D04A00 + 8);
                              sub_6851F0(20, v8);
                            }
                          }
                          a1 = (unsigned __int64)&qword_4D04A00;
                          sub_8767B0(&qword_4D04A00);
                          if ( (unk_4D04A11 & 0x40) == 0 )
                          {
                            unk_4D04A10 &= ~0x80u;
                            unk_4D04A18 = 0;
                          }
                        }
                        LODWORD(v323) = 1;
                        v325 = 20;
                        v17 = v7 & 0x400;
                        v138 = sub_72C930(a1);
                        v327 |= 4u;
                        *(_QWORD *)(v5 + 272) = v138;
                        goto LABEL_186;
                      }
                      if ( v306 && !(v325 | v82) )
                      {
                        v8 = 0;
                        v257 = v11;
                        sub_7ADF70(&v331, 0);
                        sub_7AE360(&v331);
                        sub_7B8B50(&v331, 0, v124, v125);
                        v126 = v257;
                        v127 = word_4F06418[0];
                        if ( word_4F06418[0] == 1 )
                        {
                          if ( dword_4F077C4 != 2 )
                            goto LABEL_633;
                          if ( (unk_4D04A11 & 2) != 0 )
                          {
                            if ( (unk_4D04A12 & 1) != 0 )
                              goto LABEL_555;
                            goto LABEL_633;
                          }
                          v8 = 0;
                          v137 = sub_7C0F00(0, 0);
                          v126 = v257;
                          if ( v137 && (unk_4D04A12 & 1) != 0 )
                            goto LABEL_555;
                          v127 = word_4F06418[0];
                        }
                        else if ( word_4F06418[0] != 34 )
                        {
                          if ( word_4F06418[0] == 27
                            || dword_4F077C4 != 2
                            || word_4F06418[0] != 33
                            && (!dword_4D04474 || word_4F06418[0] != 52)
                            && (!dword_4D0485C || word_4F06418[0] != 25)
                            && word_4F06418[0] != 156 )
                          {
LABEL_555:
                            a1 = (unsigned __int64)&v331;
                            v295 = v126;
                            v317 = 0;
                            sub_7BC000(&v331);
                            a5 = (unsigned __int64)&qword_4D04A00;
                            v305 = 0;
                            v11 = v295;
                            goto LABEL_556;
                          }
LABEL_633:
                          a1 = (unsigned __int64)&v331;
                          sub_7BC000(&v331);
                          goto LABEL_634;
                        }
                        if ( v127 == 27 )
                          goto LABEL_555;
                        goto LABEL_633;
                      }
                      a5 = (unsigned __int64)&qword_4D04A00;
                      v317 = v302;
LABEL_556:
                      v128 = v6 != 0;
                      v15 = v307 & 0x7F;
                      if ( (v327 & 0xFFFFFFFFFFFFFCB7LL) != 0 )
                      {
                        if ( (v327 & 8) == 0 )
                        {
                          v302 = v317;
                          goto LABEL_8;
                        }
                        v16 = dword_4F077C4;
                        v14 = (v11 ^ 1) & v128;
                        if ( (unk_4D04A11 & 0x40) == 0 )
                        {
                          a1 = v317;
                          unk_4D04A10 &= ~0x80u;
                          v11 = 0;
                          unk_4D04A18 = 0;
                          v302 = v317;
                          goto LABEL_43;
                        }
                        goto LABEL_1219;
                      }
                      if ( !v11 )
                        *(_QWORD *)(v5 + 8) |= 0x100uLL;
                      v14 = (v11 ^ 1) & v128;
                      if ( (unk_4D04A12 & 2) == 0 )
                      {
LABEL_1224:
                        v16 = dword_4F077C4;
                        goto LABEL_1225;
                      }
                      v239 = xmmword_4D04A20.m128i_i64[0];
                      if ( (unsigned int)sub_8D2870(xmmword_4D04A20.m128i_i64[0]) )
                      {
                        v240 = sub_7D36A0(&qword_4D04A00, v239);
                        a5 = (unsigned __int64)&qword_4D04A00;
                        v15 = v307 & 0x7F;
                        a1 = v240;
                      }
                      else
                      {
                        sub_7D2AC0(&qword_4D04A00, v239, 4096);
                        a5 = (unsigned __int64)&qword_4D04A00;
                        v15 = v307 & 0x7F;
                        a1 = unk_4D04A18;
                      }
                      if ( a1 )
                      {
                        v303 = v15;
                        v241 = sub_877F80(a1);
                        v15 = v303;
                        a5 = (unsigned __int64)&qword_4D04A00;
                        if ( v241 == 1 )
                        {
                          *(_QWORD *)(v5 + 8) |= 0x400uLL;
                          v325 = 25;
                          goto LABEL_1224;
                        }
                        v242 = sub_877F80(a1);
                        v15 = v303;
                        a5 = (unsigned __int64)&qword_4D04A00;
                        if ( v242 != 2 )
                        {
                          v16 = dword_4F077C4;
                          if ( (unk_4D04A11 & 0x40) == 0 )
                          {
                            unk_4D04A10 &= ~0x80u;
                            v11 = 0;
                            unk_4D04A18 = 0;
                            v302 = v317;
                            goto LABEL_43;
                          }
LABEL_1225:
                          v11 = 0;
                          v302 = v317;
                          goto LABEL_43;
                        }
                        *(_QWORD *)(v5 + 8) |= 0x800uLL;
                        v325 = 25;
                      }
                      v16 = dword_4F077C4;
LABEL_1219:
                      a1 = v317;
                      v11 = 0;
                      v302 = v317;
                      goto LABEL_43;
                    }
                  }
LABEL_1244:
                  a1 = (unsigned __int64)&v331;
                  v262 = v120;
                  v284 = v119;
                  sub_7BC000(&v331);
                  a5 = (unsigned __int64)v284;
                  v11 = v262;
LABEL_398:
                  v86 = unk_4D04A12;
                  if ( (unk_4D04A12 & 2) == 0 )
                    goto LABEL_402;
                  v87 = *(_BYTE *)(a5 + 80);
                  if ( v87 != 3 )
                    goto LABEL_400;
LABEL_893:
                  if ( !*(_BYTE *)(a5 + 104) )
                  {
                    v87 = 3;
                    goto LABEL_400;
                  }
                  if ( (v327 & 0xFFFFFFFFFFFFECB7LL) != 0 )
                    goto LABEL_402;
                  if ( dword_4F077BC )
                    goto LABEL_920;
                  goto LABEL_922;
                }
              }
              else if ( v168 == 14 )
              {
                goto LABEL_1242;
              }
LABEL_892:
              v86 = unk_4D04A12;
              if ( (unk_4D04A12 & 2) == 0 )
                goto LABEL_402;
              goto LABEL_893;
            }
            v17 = v7 & 0x400;
            if ( v301 )
            {
              v8 = v85;
              a1 = v5;
              if ( (unsigned int)sub_6726E0(v5, v85) )
              {
                *(_WORD *)(v5 + 124) |= 0x180u;
                *(_QWORD *)(v5 + 104) = *(_QWORD *)&dword_4F063F8;
              }
              v327 |= 4u;
              v325 = 20;
              goto LABEL_186;
            }
            v8 = 0;
            a1 = v85;
            *(_QWORD *)(v5 + 408) = sub_8988D0(v85, 0);
            v12 = word_4F06418[0];
            if ( word_4F06418[0] != 77 )
            {
              if ( word_4F06418[0] == 185 && sub_6510D0() )
                goto LABEL_189;
              v8 = (__int64)&dword_4F063F8;
              a1 = 3096;
              sub_6851C0(3096, &dword_4F063F8);
              *(_QWORD *)(v5 + 408) = 0;
              v12 = word_4F06418[0];
            }
            goto LABEL_178;
          }
          v99 = v327;
          goto LABEL_467;
        }
        v265 = v11;
        if ( !sub_6510D0() )
        {
LABEL_380:
          v11 = v265;
          v81 = word_4F06418[0];
          goto LABEL_381;
        }
        if ( (*(_BYTE *)(v5 + 125) & 3) != 0 )
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 84;
          sub_6851C0(84, &dword_4F063F8);
        }
        else
        {
          v202 = *(_QWORD *)&dword_4F063F8;
          v8 = (v327 & 0xFFFFFFFFFFFFFFB7LL) == 0;
          *(_WORD *)(v5 + 124) |= 0x280u;
          v296 = v8;
          *(_QWORD *)(v5 + 104) = v202;
          a1 = dword_4F07750;
          sub_668A70(dword_4F07750, v8, v7, v5, v6, &v327, &v325, v298, &v323);
        }
        sub_7B8B50(a1, v8, v144, v145);
        sub_7B8B50(a1, v8, v146, v147);
        v17 = v7 & 0x400;
        sub_7B8B50(a1, v8, v148, v149);
        sub_7B8B50(a1, v8, v150, v151);
        v12 = word_4F06418[0];
        goto LABEL_178;
      case 186:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( v325 )
          goto LABEL_185;
        *(_WORD *)(v5 + 124) |= 0x180u;
        v327 |= 4u;
        v325 = 24;
        *(_QWORD *)(v5 + 104) = *(_QWORD *)&dword_4F063F8;
        v140 = sub_7259C0(21);
        v8 = v5;
        a1 = (unsigned __int64)sub_667AD0;
        *(_QWORD *)(v5 + 304) = v140;
        *(_QWORD *)(v5 + 272) = v140;
        sub_643E40((__int64)sub_667AD0, v5, 1);
        goto LABEL_186;
      case 189:
      case 190:
        v8 = v6;
        a1 = 0;
        v331 = *(_QWORD *)&dword_4F063F8;
        v62 = sub_6B8C50(0, v6);
        *(_QWORD *)(v5 + 272) = v62;
        for ( i = *(unsigned __int8 *)(v62 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v62 + 140) )
          v62 = *(_QWORD *)(v62 + 160);
        goto LABEL_174;
      case 191:
        v8 = 362;
        v17 = v7 & 0x400;
        a1 = unk_4F07470;
        sub_684AC0(unk_4F07470, 362);
        v327 |= 0x400u;
        v292 = 0;
        goto LABEL_186;
      case 192:
        v17 = v7 & 0x400;
        if ( v301 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 1378;
          sub_6851C0(1378, dword_4F07508);
        }
        else
        {
          v63 = *(_DWORD *)(v5 + 224);
          if ( (v63 & 1) != 0 )
          {
            v8 = (__int64)dword_4F07508;
            a1 = 1379;
            sub_6851C0(1379, dword_4F07508);
          }
          else
          {
            *(_DWORD *)(v5 + 224) = v63 | 1;
          }
        }
        goto LABEL_186;
      case 236:
      case 339:
      case 340:
      case 341:
      case 342:
      case 343:
      case 344:
      case 345:
      case 346:
      case 347:
      case 348:
      case 349:
      case 350:
      case 351:
      case 352:
      case 353:
      case 354:
        goto LABEL_172;
      case 239:
        v17 = v7 & 0x400;
        if ( !v306 )
          goto LABEL_195;
        if ( v325 | v305 )
          goto LABEL_185;
        v327 |= 4u;
        v325 = 8;
        v305 = 8;
        goto LABEL_186;
      case 244:
        v17 = v7 & 0x400;
        if ( v301 )
          goto LABEL_875;
        v64 = v327;
        if ( (v327 & 0x10) == 0 || dword_4D04888 )
        {
          if ( (v7 & 0x10000) != 0 )
          {
            v8 = (__int64)&dword_4F063F8;
            a1 = 2437;
            sub_6851C0(2437, &dword_4F063F8);
          }
          else
          {
            v65 = *(_QWORD *)(v5 + 8);
            if ( (v327 & 0x1000) != 0 )
            {
              v8 = (__int64)dword_4F07508;
              a1 = (*(_QWORD *)(v5 + 8) & 0x80000LL) == 0 ? 2923 : 240;
              sub_6851C0(a1, dword_4F07508);
            }
            else
            {
              BYTE1(v64) = BYTE1(v327) | 0x10;
              v8 = v5;
              v327 = v64;
              a1 = (unsigned __int64)sub_667870;
              *(_QWORD *)(v5 + 8) = v65 | 0x80000;
              *(_QWORD *)(v5 + 112) = *(_QWORD *)&dword_4F063F8;
              sub_643E40((__int64)sub_667870, v5, 1);
            }
          }
        }
        else
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 2390;
          sub_6851C0(2390, &dword_4F063F8);
        }
        goto LABEL_186;
      case 245:
        v17 = v7 & 0x400;
        if ( v301 )
          goto LABEL_875;
        v67 = v327;
        if ( (v327 & 0x10) == 0 || dword_4D04888 )
        {
          if ( (v7 & 0x10000) != 0 )
          {
            v8 = (__int64)&dword_4F063F8;
            a1 = 2925;
            sub_6851C0(2925, &dword_4F063F8);
          }
          else
          {
            v68 = *(_QWORD *)(v5 + 8);
            if ( (v327 & 0x1000) != 0 )
            {
              v8 = (__int64)dword_4F07508;
              a1 = (*(_QWORD *)(v5 + 8) & 0x100000LL) == 0 ? 2923 : 240;
              sub_6851C0(a1, dword_4F07508);
            }
            else
            {
              BYTE1(v67) = BYTE1(v327) | 0x10;
              v8 = v5;
              v327 = v67;
              a1 = (unsigned __int64)sub_666F40;
              *(_QWORD *)(v5 + 8) = v68 | 0x100000;
              *(_QWORD *)(v5 + 112) = *(_QWORD *)&dword_4F063F8;
              sub_643E40((__int64)sub_666F40, v5, 1);
            }
          }
        }
        else
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 2924;
          sub_6851C0(2924, &dword_4F063F8);
        }
        goto LABEL_186;
      case 246:
        v17 = v7 & 0x400;
        if ( v301 )
        {
LABEL_875:
          v8 = (__int64)dword_4F07508;
          a1 = 238;
          sub_6851C0(238, dword_4F07508);
          LODWORD(v323) = 1;
        }
        else
        {
          v69 = v327;
          if ( (v327 & 0x1000) != 0 )
          {
            v8 = (__int64)dword_4F07508;
            a1 = (*(_QWORD *)(v5 + 8) & 0x200000LL) == 0 ? 2923 : 240;
            sub_6851C0(a1, dword_4F07508);
          }
          else
          {
            BYTE1(v69) = BYTE1(v327) | 0x10;
            v8 = v5;
            *(_QWORD *)(v5 + 8) |= 0x200000uLL;
            a1 = (unsigned __int64)sub_667430;
            v327 = v69;
            *(_QWORD *)(v5 + 112) = *(_QWORD *)&dword_4F063F8;
            sub_643E40((__int64)sub_667430, v5, 1);
          }
        }
        goto LABEL_186;
      case 249:
      case 250:
        v331 = *(_QWORD *)&dword_4F063F8;
        v61 = sub_6913B0();
        *(_QWORD *)(v5 + 272) = v61;
        for ( i = *(unsigned __int8 *)(v61 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v61 + 140) )
          v61 = *(_QWORD *)(v61 + 160);
        goto LABEL_174;
      case 260:
        if ( dword_4F077C4 != 2
          && unk_4F07778 > 202310
          && qword_4D04A00
          && *(_QWORD *)(qword_4D04A00 + 16) == 9
          && !memcmp(*(const void **)(qword_4D04A00 + 8), "_Noreturn", 9u)
          && !(unsigned int)sub_729F80(dword_4F063F8) )
        {
          sub_684AA0(4 - ((unsigned int)(dword_4D04964 == 0) - 1), 3289, &dword_4F063F8);
          sub_67D850(3289, 1, 0);
        }
        if ( (v327 & 0x4000) != 0 )
        {
          v8 = (__int64)&dword_4F063F8;
          a1 = 240;
          sub_684B30(240, &dword_4F063F8);
        }
        else if ( v301 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 238;
          sub_6851C0(238, dword_4F07508);
          LODWORD(v323) = 1;
        }
        else
        {
          sub_7293C0(2, &dword_4F063F8, v5 + 448);
          v8 = v5;
          a1 = (unsigned __int64)sub_667320;
          sub_643E40((__int64)sub_667320, v5, 1);
        }
        v327 |= 0x4000u;
        v17 = v7 & 0x400;
        goto LABEL_186;
      case 263:
        v8 = 0;
        a1 = 0;
        v17 = v7 & 0x400;
        if ( (unsigned __int16)sub_7BE840(0, 0) == 27 )
        {
          sub_7B8B50(0, 0, v66, a4);
          sub_7B8B50(0, 0, v188, v189);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          sub_65CD60(&v331);
          if ( word_4F06418[0] != 28 )
            sub_6851D0(18);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          a1 = v331;
          if ( !v331 )
            goto LABEL_964;
          v190 = *(_BYTE *)(v331 + 140);
          if ( v190 == 12 )
          {
            v191 = v331;
            do
            {
              v191 = *(_QWORD *)(v191 + 160);
              v190 = *(_BYTE *)(v191 + 140);
            }
            while ( v190 == 12 );
          }
          if ( v190 )
          {
            v8 = (__int64)dword_4F07508;
            v192 = sub_6680B0(v331, (__int64)dword_4F07508, 0);
          }
          else
          {
LABEL_964:
            v192 = sub_72C930(v331);
          }
          *(_QWORD *)(v5 + 272) = v192;
          if ( v325 )
          {
            v8 = (__int64)dword_4F07508;
            a1 = 84;
            sub_6851C0(84, dword_4F07508);
            v304 = 1;
          }
          v327 |= 4u;
          v325 = 20;
        }
        else if ( (v307 & 8) != 0 )
        {
          v8 = (__int64)dword_4F07508;
          a1 = 83;
          sub_684B30(83, dword_4F07508);
        }
        else
        {
          if ( !*(_DWORD *)(v5 + 72) )
            *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
          v307 |= 8u;
          v327 |= 2u;
        }
        goto LABEL_186;
      case 264:
        v17 = v7 & 0x400;
        if ( (v307 & 0x10) != 0 )
          goto LABEL_882;
        if ( (v307 & 0x70) != 0 )
          goto LABEL_950;
        if ( !*(_DWORD *)(v5 + 72) )
          *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
        v307 |= 0x10u;
        v327 |= 2u;
        goto LABEL_186;
      case 265:
        v17 = v7 & 0x400;
        if ( (v307 & 0x20) != 0 )
          goto LABEL_882;
        if ( (v307 & 0x70) != 0 )
          goto LABEL_950;
        if ( !*(_DWORD *)(v5 + 72) )
          *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
        v307 |= 0x20u;
        v327 |= 2u;
        goto LABEL_186;
      case 266:
        v17 = v7 & 0x400;
        if ( (v307 & 0x40) != 0 )
        {
LABEL_882:
          v8 = (__int64)&dword_4F063F8;
          a1 = 83;
          sub_684B30(83, &dword_4F063F8);
        }
        else if ( (v307 & 0x70) != 0 )
        {
LABEL_950:
          v8 = (__int64)&dword_4F063F8;
          a1 = 2785;
          sub_6851C0(2785, &dword_4F063F8);
        }
        else
        {
          if ( !*(_DWORD *)(v5 + 72) )
            *(_QWORD *)(v5 + 72) = *(_QWORD *)&dword_4F063F8;
          v307 |= 0x40u;
          v327 |= 2u;
        }
        goto LABEL_186;
      case 272:
        sub_7B8B50(a1, v8, i, a4);
        v8 = 125;
        a1 = 27;
        if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
          goto LABEL_430;
        ++*(_BYTE *)(qword_4F061C8 + 36LL);
        if ( word_4F06418[0] != 4 )
        {
          sub_6851D0(661);
LABEL_429:
          v8 = 18;
          a1 = 28;
          sub_7BE280(28, 18, 0, 0);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          goto LABEL_430;
        }
        if ( (int)sub_6210B0((__int64)xmmword_4F06300, 0) < 0
          || (v182 = sub_620FD0((__int64)xmmword_4F06300, &v331), v184 = v182, (_DWORD)v331)
          || v182 >= unk_4F06CE8 )
        {
          sub_6851C0(61, &dword_4F063F8);
          sub_7B8B50(61, &dword_4F063F8, v185, v186);
          goto LABEL_429;
        }
        sub_7B8B50(xmmword_4F06300, &v331, (unsigned int)v331, v183);
        v8 = 18;
        a1 = 28;
        sub_7BE280(28, 18, 0, 0);
        --*(_BYTE *)(qword_4F061C8 + 36LL);
        v92 = *(_QWORD *)(unk_4F06CF0 + 8 * v184);
        goto LABEL_431;
      case 273:
        a1 = 0;
        v92 = sub_667B60(0, v8, i, a4);
        goto LABEL_431;
      case 274:
        a1 = 2;
        v92 = sub_667B60(2, v8, i, a4);
        goto LABEL_431;
      case 275:
        a1 = 3;
        v92 = sub_667B60(3, v8, i, a4);
        goto LABEL_431;
      case 276:
        v326 = 0;
        sub_7B8B50(a1, v8, i, a4);
        v8 = 125;
        a1 = 27;
        if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
        {
          v326 = 1;
          goto LABEL_430;
        }
        v331 = *(_QWORD *)&dword_4F063F8;
        v160 = qword_4F061C8;
        ++*(_BYTE *)(qword_4F061C8 + 36LL);
        ++*(_BYTE *)(v160 + 75);
        sub_65CD60(&v328);
        v161 = v328;
        v162 = *(_BYTE *)(v328 + 140);
        if ( v162 == 12 )
        {
          v163 = v328;
          do
            v163 = *(_QWORD *)(v163 + 160);
          while ( *(_BYTE *)(v163 + 140) == 12 );
        }
        else
        {
          if ( (v162 & 0xFB) != 8 )
            goto LABEL_981;
          v163 = v328;
        }
        if ( (unsigned int)sub_8D4C10(v328, dword_4F077C4 != 2) )
        {
          v326 = 1;
          v164 = *(_BYTE *)(v163 + 140);
          v161 = v163;
          goto LABEL_862;
        }
        v161 = v163;
LABEL_981:
        if ( (unsigned int)sub_8D2820() )
        {
          v193 = v326;
          if ( (unsigned __int8)(*(_BYTE *)(v161 + 160) - 9) > 1u )
            goto LABEL_983;
          v326 = 1;
          v164 = *(_BYTE *)(v161 + 140);
LABEL_862:
          while ( v164 == 12 )
          {
            v161 = *(_QWORD *)(v161 + 160);
LABEL_861:
            v164 = *(_BYTE *)(v161 + 140);
          }
          if ( !v164 )
          {
LABEL_864:
            sub_7BE280(67, 253, 0, 0);
            v165 = 0;
            if ( (unsigned int)sub_7BE280(4, 661, 0, 0) )
            {
              sub_620E00(word_4F063B0, 0, (__int64 *)&v329, (int *)&v326);
              if ( !v326 )
              {
                if ( (unsigned __int64)(v329 - 1) > 3
                  || (v206 = sub_8D29A0(v328), v165 = (unsigned __int8)v329, v206) && v329 == (_BYTE *)3 )
                {
                  v165 = 0;
                  sub_6851C0(3412, &v331);
                  v326 = 1;
                }
              }
            }
            v8 = 18;
            a1 = 28;
            sub_7BE280(28, 18, 0, 0);
            v166 = qword_4F061C8;
            --*(_BYTE *)(qword_4F061C8 + 75LL);
            --*(_BYTE *)(v166 + 36);
            if ( !v326 )
            {
              a1 = v328;
              v8 = v165;
              v92 = sub_72B620(v328, v165);
LABEL_431:
              *(_QWORD *)(v5 + 272) = v92;
              v327 |= 4u;
              v17 = v7 & 0x400;
              v325 = 20;
              v12 = word_4F06418[0];
              goto LABEL_178;
            }
LABEL_430:
            v92 = sub_72C930(a1);
            goto LABEL_431;
          }
LABEL_1082:
          sub_685360(3411, &v331);
          goto LABEL_864;
        }
        if ( (unsigned int)sub_8D2AC0(v161) )
        {
          v209 = *(_BYTE *)(v161 + 160);
          if ( v209 > 9u )
          {
            v326 = 1;
            goto LABEL_861;
          }
          v326 = ((0x216uLL >> v209) & 1) == 0;
          v193 = v326;
        }
        else
        {
          if ( !(unsigned int)sub_8D29A0(v161) )
          {
            v164 = *(_BYTE *)(v161 + 140);
            if ( v164 == 18 )
            {
              if ( !v326 )
                goto LABEL_864;
              goto LABEL_1082;
            }
            v326 = 1;
            goto LABEL_862;
          }
          v193 = v326;
        }
LABEL_983:
        if ( v193 )
          goto LABEL_861;
        goto LABEL_864;
      case 277:
        v17 = v7 & 0x400;
        a1 = unk_4F06A51;
        v93 = sub_72BA30(unk_4F06A51);
        v327 |= 4u;
        *(_QWORD *)(v5 + 272) = v93;
        v325 = 20;
        goto LABEL_186;
      case 278:
        v17 = v7 & 0x400;
        a1 = unk_4F06A60;
        v96 = sub_72BA30(unk_4F06A60);
        v327 |= 4u;
        *(_QWORD *)(v5 + 272) = v96;
        v325 = 20;
        goto LABEL_186;
      case 279:
        v97 = sub_72C390();
        v327 |= 4u;
        *(_QWORD *)(v5 + 272) = v97;
        v17 = v7 & 0x400;
        v325 = 20;
        goto LABEL_186;
      case 280:
        v98 = sub_72C270();
        v327 |= 4u;
        *(_QWORD *)(v5 + 272) = v98;
        v17 = v7 & 0x400;
        v325 = 20;
        goto LABEL_186;
      default:
        goto LABEL_8;
    }
  }
}
