// Function: sub_8708D0
// Address: 0x8708d0
//
__int64 __fastcall sub_8708D0(int a1, int a2)
{
  unsigned int *v3; // rsi
  __int64 v4; // rax
  unsigned __int16 v5; // ax
  __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // edx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // ebx
  char v22; // al
  unsigned __int8 v23; // di
  unsigned int *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  char v30; // al
  int v31; // r12d
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rbx
  unsigned int *v35; // rsi
  int v36; // esi
  unsigned __int64 v37; // rbx
  __int64 v38; // rax
  __int64 *v39; // r12
  FILE *v40; // rsi
  unsigned __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  _QWORD *v45; // r9
  unsigned __int8 v46; // di
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rbx
  _BYTE *v51; // r12
  int v52; // r12d
  unsigned int *v53; // rsi
  _BYTE *v54; // rbx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rsi
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rdx
  unsigned int *v69; // rsi
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rdi
  __int64 v79; // rax
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 *v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdi
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 *v89; // r9
  __int64 v90; // rbx
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __m128i *v95; // r9
  int v96; // r12d
  _BYTE *v97; // rax
  __int64 v98; // r8
  __int64 v99; // r9
  _BYTE *v100; // rcx
  __int64 v101; // rax
  _BYTE *v102; // rcx
  __int64 v103; // kr08_8
  __int64 v104; // rdx
  __int16 v105; // ax
  unsigned __int8 v106; // di
  unsigned int v107; // r14d
  unsigned int *v108; // rsi
  __int64 v109; // rsi
  unsigned __int64 v110; // rdi
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  __int64 v121; // r8
  __int64 v122; // rax
  bool v123; // bl
  __int64 v124; // r8
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // rsi
  unsigned __int8 v128; // r12
  char v129; // bl
  unsigned int *v130; // rsi
  unsigned int *v131; // r12
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  __int64 v135; // r9
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // r8
  __int64 v139; // r9
  __int64 v140; // r8
  int v141; // ebx
  unsigned int *v142; // rsi
  unsigned int *v143; // r12
  __int64 v144; // r8
  __int64 v145; // r9
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r9
  __int64 v150; // r8
  __int64 v151; // r9
  __int64 v152; // rax
  __int64 v153; // rdi
  unsigned __int16 v154; // bx
  __int64 v155; // rax
  __int64 v156; // r8
  __int64 v157; // r9
  __int64 v158; // rax
  __int64 v159; // r8
  __int64 v160; // r9
  __int64 v161; // r8
  __int64 v162; // rbx
  _BYTE *v163; // r12
  _QWORD *v164; // rax
  __int64 v165; // rdx
  __int64 v166; // r8
  __int64 v167; // r9
  __int64 v168; // rdx
  __int64 v169; // rcx
  __int64 v170; // rbx
  __m128i *v171; // r14
  unsigned int *v172; // rsi
  _BYTE *v173; // rax
  __int64 v174; // r12
  __int64 v175; // rdx
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // r9
  __int64 v179; // r8
  _DWORD *v180; // r9
  __int64 v181; // rax
  int v182; // ebx
  __int64 v183; // rcx
  __int64 v184; // rax
  __int64 v185; // r12
  unsigned __int64 v186; // rdi
  __int64 v187; // rdx
  __int64 v188; // rcx
  __int64 v189; // r8
  __int64 v190; // r9
  __int64 v191; // rax
  __int64 v192; // r8
  __int64 v193; // rcx
  __int64 v194; // rdx
  __int64 v195; // r9
  __int64 v196; // r8
  __int64 v197; // r8
  char v198; // bl
  unsigned __int64 v199; // rdi
  __int64 v200; // rdx
  __int64 v201; // rcx
  __int64 v202; // r8
  __int64 v203; // rdx
  __int64 v204; // rcx
  __int64 v205; // r8
  __int64 v206; // r9
  __int64 v207; // rax
  int v208; // eax
  __int64 v209; // rbx
  __int64 v210; // rdi
  __int64 v211; // rax
  _BYTE *v212; // rax
  __int64 v213; // rdx
  _BYTE *v214; // rax
  __int64 v215; // rbx
  __int64 v216; // rax
  __int64 v217; // rax
  char v218; // dl
  unsigned __int8 v219; // al
  __int64 v220; // rcx
  __int64 v221; // r9
  __int64 v222; // rax
  unsigned int *v223; // rsi
  _BYTE *v224; // rax
  _BYTE *v225; // rbx
  _BYTE *v226; // rax
  char v227; // cl
  __int64 v228; // rcx
  __int64 v229; // rcx
  __int64 v230; // rdx
  int v231; // eax
  __int64 v232; // rcx
  char v233; // al
  __int64 v234; // rcx
  __int64 v235; // rcx
  char v236; // al
  char v237; // bl
  __int64 v238; // rax
  const __m128i *v239; // rax
  __m128i *v240; // rax
  __int64 v241; // r14
  _QWORD *v242; // rax
  int v243; // eax
  __int64 v244; // rdi
  __int64 v245; // rdx
  __int64 v246; // rcx
  __int64 v247; // r8
  __int64 *v248; // r9
  _BYTE *v249; // r12
  _QWORD *v250; // rax
  __int64 v251; // r11
  __int64 v252; // r9
  __int64 v253; // rcx
  __int64 v254; // rdx
  __int64 v255; // rdx
  __int64 v256; // rdx
  __int64 v257; // rdi
  __int64 v258; // rsi
  __int64 v259; // rax
  unsigned int v260; // edi
  __int64 v261; // rdx
  __int64 v262; // rcx
  __int64 v263; // r8
  __int64 v264; // r9
  __int64 v265; // rdx
  __int64 v266; // rcx
  __int64 v267; // r8
  __int64 v268; // r9
  __int64 v269; // rdx
  __int64 v270; // rcx
  __int64 v271; // r8
  __int64 v272; // r9
  __int64 v273; // rdx
  __int64 v274; // rcx
  __int64 v275; // r8
  __int64 v276; // r9
  __int64 v277; // rax
  __int64 v278; // rax
  __int64 v279; // rdi
  __int64 v280; // rsi
  unsigned int v281; // eax
  __int64 v282; // rdx
  unsigned int v283; // ecx
  int v284; // ebx
  __int64 v285; // rcx
  __int64 v286; // rdx
  __int64 v287; // rax
  __int64 v288; // rax
  unsigned __int8 v289; // bl
  __int64 v290; // rcx
  char v291; // dl
  char v292; // al
  bool v293; // zf
  __int8 v294; // al
  unsigned int v295; // eax
  unsigned __int64 v296; // rdi
  __int64 v297; // rdx
  __int64 v298; // rcx
  __int64 v299; // r8
  char v300; // al
  _BOOL4 v301; // edi
  _QWORD **v302; // rbx
  __int64 v303; // rax
  int v304; // eax
  _DWORD *v305; // rbx
  __int64 v306; // rdi
  _BOOL4 v307; // eax
  int v308; // eax
  __int64 v309; // rax
  __int64 v310; // [rsp+10h] [rbp-120h]
  __int64 v311; // [rsp+18h] [rbp-118h]
  unsigned int v312; // [rsp+20h] [rbp-110h]
  int v313; // [rsp+24h] [rbp-10Ch]
  __int64 *v314; // [rsp+30h] [rbp-100h]
  bool v315; // [rsp+38h] [rbp-F8h]
  _BYTE *v316; // [rsp+40h] [rbp-F0h]
  char v317; // [rsp+40h] [rbp-F0h]
  char v318; // [rsp+48h] [rbp-E8h]
  __int64 v319; // [rsp+48h] [rbp-E8h]
  unsigned int *v320; // [rsp+50h] [rbp-E0h]
  _BYTE *v321; // [rsp+50h] [rbp-E0h]
  unsigned int v322; // [rsp+50h] [rbp-E0h]
  __int64 v323; // [rsp+50h] [rbp-E0h]
  unsigned int v324; // [rsp+50h] [rbp-E0h]
  unsigned int v325; // [rsp+58h] [rbp-D8h]
  unsigned int v326; // [rsp+58h] [rbp-D8h]
  char v327; // [rsp+58h] [rbp-D8h]
  __int64 v328; // [rsp+58h] [rbp-D8h]
  __int64 *v329; // [rsp+60h] [rbp-D0h]
  _BYTE *v330; // [rsp+60h] [rbp-D0h]
  char v331; // [rsp+60h] [rbp-D0h]
  __int64 v332; // [rsp+60h] [rbp-D0h]
  __m128i *v333; // [rsp+60h] [rbp-D0h]
  __int64 v334; // [rsp+60h] [rbp-D0h]
  __int64 v335; // [rsp+60h] [rbp-D0h]
  _BYTE *v336; // [rsp+68h] [rbp-C8h]
  int v337; // [rsp+68h] [rbp-C8h]
  __int64 v338; // [rsp+68h] [rbp-C8h]
  unsigned __int16 v339; // [rsp+70h] [rbp-C0h]
  _BYTE *v340; // [rsp+70h] [rbp-C0h]
  unsigned int v341; // [rsp+70h] [rbp-C0h]
  unsigned int v342; // [rsp+70h] [rbp-C0h]
  unsigned __int16 v343; // [rsp+70h] [rbp-C0h]
  __int64 v344; // [rsp+70h] [rbp-C0h]
  _BYTE *v345; // [rsp+70h] [rbp-C0h]
  char v347; // [rsp+7Fh] [rbp-B1h]
  unsigned __int16 v348; // [rsp+80h] [rbp-B0h]
  _BYTE *i; // [rsp+88h] [rbp-A8h]
  __int64 v350; // [rsp+88h] [rbp-A8h]
  _BYTE *v351; // [rsp+88h] [rbp-A8h]
  __int64 v352; // [rsp+88h] [rbp-A8h]
  __int64 v353; // [rsp+88h] [rbp-A8h]
  unsigned int v354; // [rsp+88h] [rbp-A8h]
  _BYTE *v355; // [rsp+88h] [rbp-A8h]
  __m128i *v356; // [rsp+88h] [rbp-A8h]
  unsigned int v357; // [rsp+90h] [rbp-A0h]
  __int64 v358; // [rsp+90h] [rbp-A0h]
  __int64 v359; // [rsp+90h] [rbp-A0h]
  FILE *v360; // [rsp+90h] [rbp-A0h]
  __int64 v361; // [rsp+90h] [rbp-A0h]
  _QWORD *v362; // [rsp+90h] [rbp-A0h]
  _BYTE *v363; // [rsp+98h] [rbp-98h]
  int v364; // [rsp+A8h] [rbp-88h] BYREF
  int v365; // [rsp+ACh] [rbp-84h] BYREF
  __int64 v366; // [rsp+B0h] [rbp-80h] BYREF
  __m128i *v367; // [rsp+B8h] [rbp-78h] BYREF
  __int64 v368; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v369; // [rsp+C8h] [rbp-68h] BYREF
  __m128i v370; // [rsp+D0h] [rbp-60h] BYREF
  unsigned int v371[20]; // [rsp+E0h] [rbp-50h] BYREF

  v364 = dword_4D0488C;
  v363 = (_BYTE *)qword_4F04C50;
  if ( qword_4F04C50 )
    v363 = *(_BYTE **)(qword_4F04C50 + 32LL);
  v347 = 8;
  v357 = 0;
  while ( 1 )
  {
    v3 = &word_4D04898;
    v4 = qword_4D03B98 + 176LL * unk_4D03B90;
    if ( word_4D04898 && (*(_BYTE *)(v4 + 5) & 4) != 0 )
      v364 = 1;
    if ( !*(_QWORD *)(v4 + 160) )
    {
      v368 = *(_QWORD *)&dword_4F063F8;
      *(_QWORD *)(v4 + 160) = &v368;
    }
    v5 = word_4F06418[0];
    if ( word_4F06418[0] == 25 )
    {
      if ( !dword_4D043F8 )
        goto LABEL_13;
      v3 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
      {
LABEL_12:
        v6 = qword_4D03B98 + 176LL * unk_4D03B90;
        *(_QWORD *)(v6 + 16) = sub_5CC190(1);
        goto LABEL_13;
      }
      v5 = word_4F06418[0];
    }
    if ( v5 == 248 || v5 == 142 )
      goto LABEL_12;
LABEL_13:
    v7 = v357;
    if ( sub_854590(v357) )
    {
      v3 = 0;
      v7 = (unsigned __int64)sub_854840(5u, 0, 0, 0);
      if ( v7 )
      {
        *(__int64 *)((char *)&qword_4F5FD78 + 4) = 0x100000000LL;
        sub_854000((__m128i *)v7);
      }
    }
    v10 = 176LL * unk_4D03B90 + qword_4D03B98;
    v11 = 176LL * unk_4D03B90;
    *(_QWORD *)(v10 + 168) = 0;
    if ( word_4F06418[0] > 0xA3u )
      break;
    if ( word_4F06418[0] > 0x48u )
    {
      switch ( word_4F06418[0] )
      {
        case 0x49u:
          sub_86FD00(0, 0, 0, 0, 0, 0);
          result = (__int64)&dword_4D04964;
          if ( dword_4D04964 )
            goto LABEL_28;
          goto LABEL_163;
        case 0x4Au:
          if ( !v357 )
            goto LABEL_27;
          if ( dword_4F077C4 == 2 )
          {
            if ( unk_4F07778 > 202301 )
              goto LABEL_161;
          }
          else if ( unk_4F07778 > 202310 )
          {
            goto LABEL_161;
          }
          if ( dword_4D04964 )
          {
            sub_684AC0(byte_4F07472[0], 0x7Fu);
LABEL_161:
            result = (__int64)sub_854AB0();
            goto LABEL_28;
          }
          sub_684B30(0x7Fu, dword_4F07508);
          result = (__int64)sub_854AB0();
          goto LABEL_28;
        case 0x4Bu:
          result = sub_86E8F0(v7, v3, v11);
          v364 = 1;
          goto LABEL_28;
        case 0x4Eu:
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          sub_86F7D0(0x6Fu, dword_4F07508);
          v90 = qword_4D03B98 + 176LL * unk_4D03B90;
          if ( qword_4D03B98 == v90 )
            goto LABEL_431;
          while ( (unsigned int)(*(_DWORD *)v90 - 3) > 4 )
          {
            v90 -= 176;
            if ( qword_4D03B98 == v90 )
            {
LABEL_431:
              sub_854AB0();
              sub_6851C0(0x74u, dword_4F07508);
              sub_7B8B50(0x74u, dword_4F07508, v273, v274, v275, v276);
              v96 = qword_4F063F0;
              v348 = WORD2(qword_4F063F0);
              goto LABEL_154;
            }
          }
          sub_854AB0();
          if ( *(_DWORD *)v90 == 3 && **(_QWORD **)(*(_QWORD *)(v90 + 8) + 80LL) )
          {
            v91 = qword_4F5FD78;
            *(_DWORD *)(v90 + 124) |= dword_4F5FD80;
            *(_QWORD *)(v90 + 116) |= v91;
          }
          sub_7B8B50(0x6Fu, dword_4F07508, v91, v92, v93, v94);
          v95 = *(__m128i **)(v90 + 64);
          v96 = qword_4F063F0;
          v348 = WORD2(qword_4F063F0);
          if ( !v95 )
          {
            v356 = sub_726410();
            sub_730430((__int64)v356);
            v95 = v356;
            *(_QWORD *)(v90 + 64) = v356;
            v294 = v356[7].m128i_i8[8];
            v356[7].m128i_i8[8] = v294 | 2;
            if ( *(_DWORD *)v90 == 3 )
              v356[7].m128i_i8[8] = v294 | 6;
          }
          v350 = (__int64)v95;
          v97 = sub_86E480(6u, v371);
          v99 = v350;
          *((_DWORD *)v97 + 2) = v96;
          v100 = v97;
          *((_WORD *)v97 + 6) = v348;
          if ( !dword_4F04C3C )
          {
            v344 = v350;
            v355 = v97;
            sub_8699D0((__int64)v97, 21, 0);
            v99 = v344;
            v100 = v355;
          }
          *((_QWORD *)v100 + 9) = v99;
          if ( dword_4F077C4 == 2
            && (v345 = v100,
                v278 = sub_7340A0(qword_4F06BC0),
                v100 = v345,
                *((_QWORD *)v345 + 10) = v278,
                dword_4F077C4 == 2)
            || dword_4D047EC )
          {
            v351 = v100;
            v101 = sub_86B2C0(2);
            v102 = v351;
            v352 = v101;
            *(_QWORD *)(v101 + 24) = *(_QWORD *)&dword_4F063F8;
            *(_QWORD *)(v101 + 40) = v102;
            sub_86CBE0(v101);
            *(_QWORD *)(v352 + 48) = *(_QWORD *)(v90 + 72);
            *(_QWORD *)(v90 + 72) = v352;
          }
LABEL_154:
          sub_7BE280(0x4Bu, 65, 0, 0, v98, v99);
          result = v348;
          dword_4F061D8 = v96;
          unk_4F061DC = v348;
          goto LABEL_28;
        case 0x4Fu:
          v182 = qword_4F5FD78;
          *(_BYTE *)(v10 + 5) |= 0x40u;
          v183 = qword_4D03B98;
          ++*(_BYTE *)(qword_4F061C8 + 63LL);
          v184 = v183 + v11;
          if ( v183 == v183 + v11 )
            goto LABEL_426;
          while ( *(_DWORD *)v184 != 3 )
          {
            v184 -= 176;
            if ( v183 == v184 )
            {
LABEL_426:
              v3 = dword_4F07508;
              sub_6851C0(0x79u, dword_4F07508);
              dword_4F5FD80 = 0;
              qword_4F5FD78 = 0x100000001LL;
              v185 = sub_72C930();
              *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
              sub_7B8B50(0x79u, dword_4F07508, v265, v266, v267, v268);
              v186 = v185;
              v354 = dword_4F063F8;
              v343 = word_4F063FC[0];
              v193 = sub_6D74C0(v185, (__int64)dword_4F07508, v269, v270, v271, v272);
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_267;
              v337 = 0;
              LOBYTE(v3) = v193 != 0;
              v194 = 0;
              goto LABEL_281;
            }
          }
          v185 = *(_QWORD *)(v184 + 96);
          LODWORD(qword_4F5FD78) = v182 | *(_DWORD *)(v184 + 104);
          *(__int64 *)((char *)&qword_4F5FD78 + 4) |= *(_QWORD *)(v184 + 108);
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          v337 = -1171354717 * ((v184 - v183) >> 4);
          v358 = 0x2E8BA2E8BA2E8BA3LL * ((v184 - v183) >> 4);
          sub_7B8B50(v7, v3, v358, v183, v8, v9);
          v186 = v185;
          v354 = dword_4F063F8;
          v343 = word_4F063FC[0];
          v191 = sub_6D74C0(v185, (__int64)v3, v187, v188, v189, v190);
          LODWORD(v194) = v358;
          v193 = v191;
          LOBYTE(v194) = (_DWORD)v358 != 0;
          if ( !HIDWORD(qword_4F077B4) )
          {
            v195 = 0;
            LOBYTE(v194) = (v191 != 0) & v194;
            goto LABEL_266;
          }
          LOBYTE(v3) = v191 != 0;
          v194 = (unsigned int)v3 & (unsigned int)v194;
LABEL_281:
          v195 = 0;
          if ( word_4F06418[0] == 76 )
          {
            v331 = v194;
            v359 = v193;
            sub_7B8B50(v186, v3, v194, v193, v192, 0);
            v207 = sub_6D74C0(v185, (__int64)v3, v203, v204, v205, v206);
            v193 = v359;
            LOBYTE(v194) = v331;
            v195 = v207;
            if ( v207 )
            {
              if ( (_BYTE)v3 )
              {
                if ( *(_BYTE *)(v359 + 173) == 1 && *(_BYTE *)(v207 + 173) == 1 )
                {
                  v327 = v331;
                  v332 = v207;
                  v208 = sub_621060(v359, v207);
                  v193 = v359;
                  v195 = v332;
                  LOBYTE(v194) = v327;
                  if ( v208 > 0 )
                  {
                    sub_6851C0(0x45Bu, dword_4F07508);
                    LOBYTE(v194) = v327;
                    v193 = v359;
                    v195 = 0;
                  }
                }
              }
            }
          }
LABEL_266:
          if ( (_BYTE)v194 )
          {
            v328 = v193;
            v334 = v195;
            v249 = sub_7267B0();
            v361 = qword_4D03B98 + 176LL * v337;
            v250 = sub_86E480(0xFu, v371);
            v251 = v361;
            v252 = v334;
            v253 = v328;
            if ( !dword_4F04C3C )
            {
              v335 = v361;
              v338 = v252;
              v362 = v250;
              sub_8699D0((__int64)v250, 21, 0);
              v253 = v328;
              v251 = v335;
              v252 = v338;
              v250 = v362;
            }
            v254 = *(_QWORD *)(v251 + 8);
            v250[10] = v249;
            v250[9] = v254;
            *((_QWORD *)v249 + 2) = v252;
            *(_QWORD *)v249 = v250;
            v255 = *(_QWORD *)&dword_4F061D8;
            *((_DWORD *)v249 + 10) = v354;
            *((_QWORD *)v249 + 1) = v253;
            *((_QWORD *)v249 + 6) = v255;
            v256 = *(_QWORD *)&dword_4F063F8;
            *((_WORD *)v249 + 22) = v343;
            *((_QWORD *)v249 + 7) = v256;
            v250[1] = qword_4F063F0;
            v249[64] = v249[64] & 0xFE | v182 & 1;
            sub_86D690((__int64)v249, v251);
          }
LABEL_267:
          *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          sub_7BE280(0x37u, 53, 0, 0, v192, (__int64)&dword_4F061D8);
          --*(_BYTE *)(qword_4F061C8 + 63LL);
          goto LABEL_240;
        case 0x52u:
          sub_86F7D0(0x6Fu, dword_4F07508);
          v169 = qword_4D03B98 + 176LL * unk_4D03B90;
          if ( qword_4D03B98 == v169 )
            goto LABEL_424;
          v170 = qword_4D03B98 + 176LL * unk_4D03B90;
          do
          {
            if ( (unsigned int)(*(_DWORD *)v170 - 4) <= 3 )
            {
              v171 = *(__m128i **)(v170 + 80);
              if ( !v171 )
              {
                v171 = sub_726410();
                sub_730430((__int64)v171);
                v277 = unk_4D03B90;
                *(_QWORD *)(v170 + 80) = v171;
                v171[7].m128i_i8[8] |= 8u;
                v169 = qword_4D03B98 + 176 * v277;
              }
              v172 = *(unsigned int **)(v169 + 160);
              if ( !v172 )
                v172 = &dword_4F063F8;
              v173 = sub_86E480(6u, v172);
              v174 = (__int64)v173;
              *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
              if ( !dword_4F04C3C )
                sub_8699D0((__int64)v173, 21, 0);
              *(_QWORD *)(v174 + 72) = v171;
              if ( dword_4F077C4 == 2 && (*(_QWORD *)(v174 + 80) = sub_7340A0(qword_4F06BC0), dword_4F077C4 == 2)
                || dword_4D047EC )
              {
                v181 = sub_86B2C0(2);
                *(_QWORD *)(v181 + 40) = v174;
                v353 = v181;
                *(_QWORD *)(v181 + 24) = *(_QWORD *)&dword_4F063F8;
                sub_86CBE0(v181);
                *(_QWORD *)(v353 + 48) = *(_QWORD *)(v170 + 88);
                *(_QWORD *)(v170 + 88) = v353;
              }
              sub_854980(0, v174);
              sub_7B8B50(0, (unsigned int *)v174, v175, v176, v177, v178);
              v180 = &dword_4F061D8;
              if ( word_4F06418[0] == 75 )
                *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
              *(_QWORD *)(v174 + 8) = *(_QWORD *)&dword_4F061D8;
              goto LABEL_257;
            }
            v170 -= 176;
          }
          while ( qword_4D03B98 != v170 );
LABEL_424:
          sub_6851C0(0x73u, dword_4F07508);
          ((void (*)(void))sub_86E8F0)();
          sub_7B8B50(0x73u, dword_4F07508, v261, v262, v263, v264);
          if ( word_4F06418[0] == 75 )
          {
            v180 = &dword_4F061D8;
            *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          }
LABEL_257:
          result = sub_7BE280(0x4Bu, 65, 0, 0, v179, (__int64)v180);
          goto LABEL_28;
        case 0x53u:
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          sub_7B8B50(v7, v3, v11, v10, v8, v9);
          *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          v162 = qword_4D03B98 + 176LL * unk_4D03B90;
          if ( qword_4D03B98 == v162 )
            goto LABEL_416;
          do
          {
            if ( *(_DWORD *)v162 == 3 )
            {
              v163 = sub_7267B0();
              v164 = sub_86E480(0xFu, v371);
              v165 = *(_QWORD *)(v162 + 8);
              v164[10] = v163;
              v164[9] = v165;
              *(_QWORD *)v163 = v164;
              *((_QWORD *)v163 + 5) = *(_QWORD *)v371;
              *((_QWORD *)v163 + 7) = *(_QWORD *)&dword_4F063F8;
              v164[1] = qword_4F063F0;
              v163[64] = qword_4F5FD78 & 1 | v163[64] & 0xFE;
              sub_86D690((__int64)v163, v162);
              v168 = *(_QWORD *)(v162 + 104) | qword_4F5FD78;
              dword_4F5FD80 |= *(_DWORD *)(v162 + 112);
              qword_4F5FD78 = v168;
              goto LABEL_239;
            }
            v162 -= 176;
          }
          while ( qword_4D03B98 != v162 );
LABEL_416:
          sub_6851C0(0x7Au, v371);
          dword_4F5FD80 = 0;
          qword_4F5FD78 = 0x100000001LL;
LABEL_239:
          sub_7BE280(0x37u, 53, 0, 0, v166, v167);
          goto LABEL_240;
        case 0x54u:
          v141 = dword_4F5FD80 | qword_4F5FD78;
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
          {
            sub_86EE70(v7, v3, v11);
            v10 = qword_4D03B98 + 176LL * unk_4D03B90;
          }
          v142 = *(unsigned int **)(v10 + 160);
          if ( !v142 )
            v142 = &dword_4F063F8;
          v143 = (unsigned int *)sub_86E480(0xCu, v142);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v143, 21, 0);
          sub_854980(0, (__int64)v143);
          sub_86D170(5, (__int64)v143, 0, 0, v144, v145);
          sub_7B8B50(5u, v143, v146, v147, v148, v149);
          ++*(_BYTE *)(qword_4F061C8 + 116LL);
          sub_8745D0();
          v152 = qword_4D03B98 + 176LL * unk_4D03B90;
          if ( !v141 && (*(_BYTE *)(v152 + 5) & 0x60) == 0 )
          {
            sub_684B30(0x80u, v371);
            dword_4F5FD80 = 1;
            v152 = qword_4D03B98 + 176LL * unk_4D03B90;
          }
          v153 = *(_QWORD *)(v152 + 80);
          if ( v153 )
            sub_86EEF0(v153, *(_QWORD *)(v152 + 88));
          v154 = word_4F063FC[0];
          v342 = dword_4F063F8;
          sub_7BE280(0x6Cu, 112, 0, 0, v150, v151);
          v155 = qword_4F061C8;
          --*(_BYTE *)(qword_4F061C8 + 116LL);
          ++*(_BYTE *)(v155 + 83);
          sub_869D70((__int64)v143, 21);
          if ( !dword_4F04C3C )
          {
            v158 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336) + 24LL);
            *(_DWORD *)v158 = v342;
            *(_WORD *)(v158 + 4) = v154;
          }
          sub_7BE280(0x1Bu, 125, 0, 0, v156, v157);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          *((_QWORD *)v143 + 6) = sub_6D7680(0);
          sub_7BE280(0x1Cu, 18, 0, 0, v159, v160);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          if ( word_4F06418[0] == 75 )
            *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          *((_QWORD *)v143 + 1) = *(_QWORD *)&dword_4F061D8;
          sub_7BE280(0x4Bu, 65, 0, 0, v161, (__int64)&dword_4F061D8);
          --*(_BYTE *)(qword_4F061C8 + 83LL);
          sub_86F030();
          sub_86C020((__int64)v143);
          result = (__int64)&dword_4F077C4;
          if ( dword_4F077C4 == 2 )
            goto LABEL_28;
          goto LABEL_125;
        case 0x56u:
        case 0x96u:
LABEL_27:
          v18 = qword_4F061C8;
          ++*(_BYTE *)(qword_4F061C8 + 83LL);
          ++*(_BYTE *)(v18 + 82);
          sub_6851D0(0x7Fu);
          v19 = qword_4F061C8;
          --*(_BYTE *)(qword_4F061C8 + 83LL);
          --*(_BYTE *)(v19 + 82);
          result = sub_86E8F0(127, v3, v20);
          goto LABEL_28;
        case 0x5Au:
          result = sub_874650(v7, v3, v11);
          goto LABEL_28;
        case 0x5Bu:
          sub_86F7D0(0x6Fu, dword_4F07508);
          if ( HIDWORD(qword_4F077B4) && (unsigned __int16)sub_7BE840(0, 0) == 34 )
          {
            v128 = 23;
            v129 = 23;
            if ( dword_4D04964 )
              sub_684AC0(byte_4F07472[0], 0x44Eu);
          }
          else
          {
            v128 = 6;
            v129 = 6;
          }
          v130 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
          if ( !v130 )
            v130 = &dword_4F063F8;
          v131 = (unsigned int *)sub_86E480(v128, v130);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v131, 21, 0);
          sub_854980(0, (__int64)v131);
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          sub_7B8B50(0, v131, v132, v133, v134, v135);
          ++*(_BYTE *)(qword_4F061C8 + 83LL);
          if ( v129 == 23 )
          {
            sub_7B8B50(0, v131, v136, v137, v138, v139);
            v239 = (const __m128i *)sub_72CBE0();
            v240 = sub_73C570(v239, 1);
            v241 = sub_72D2E0(v240);
            v242 = (_QWORD *)sub_72CBE0();
            v243 = sub_72D2E0(v242);
            *((_QWORD *)v131 + 6) = sub_6B9A60(v243, v241, 1101);
          }
          else
          {
            *((_QWORD *)v131 + 9) = sub_64E550(0, 0);
            if ( dword_4F077C4 == 2 )
              *((_QWORD *)v131 + 10) = sub_7340A0(qword_4F06BC0);
            sub_86D5C0((__int64)v131, v371);
          }
          if ( word_4F06418[0] == 75 )
            *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          *((_QWORD *)v131 + 1) = *(_QWORD *)&dword_4F061D8;
          sub_7BE280(0x4Bu, 65, 0, 0, v140, (__int64)&dword_4F061D8);
          --*(_BYTE *)(qword_4F061C8 + 83LL);
          result = (__int64)&dword_4F077C4;
          if ( dword_4F077C4 == 2 )
          {
            result = (__int64)&unk_4F07778;
            if ( unk_4F07778 > 202301 )
              goto LABEL_28;
          }
          goto LABEL_107;
        case 0x5Cu:
          v365 = 0;
          v103 = qword_4F5FD78;
          v313 = dword_4F5FD80;
          sub_86F7D0(0x6Fu, dword_4F07508);
          if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
            sub_86EE70(111, dword_4F07508, v104);
          if ( !qword_4D043C0 )
            goto LABEL_174;
          v105 = sub_7BE840(0, 0);
          if ( HIDWORD(qword_4D043C0) && v105 == 244 )
          {
            v326 = 0;
            v106 = 2;
            v107 = 2;
            v341 = 1;
            v318 = 2;
            goto LABEL_175;
          }
          if ( (_DWORD)qword_4D043C0 )
          {
            if ( v105 == 245 )
            {
              v326 = 1;
              v106 = 3;
              v107 = 1;
              v341 = 0;
              v318 = 3;
              goto LABEL_175;
            }
          }
          else
          {
            if ( !dword_4F077BC || (v341 = qword_4F077B4) != 0 )
            {
LABEL_174:
              v326 = 0;
              v106 = 1;
              v107 = 1;
              v341 = 0;
              v318 = 1;
              goto LABEL_175;
            }
            if ( qword_4F077A8 > 0x1D4BFu && dword_4F077C4 == 2 && unk_4F07778 > 202001 && v105 == 245 )
            {
              v107 = 1;
              sub_684B30(0xC8Fu, &dword_4F063F8);
              v318 = 3;
              v106 = 3;
              v326 = 1;
              goto LABEL_175;
            }
            if ( qword_4F077A8 <= 0x1D4BFu || dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
              goto LABEL_174;
          }
          if ( v105 != 38 || (unsigned __int16)sub_7BEB10(0x26u, v371) != 38 || LOWORD(v371[0]) != 245 )
            goto LABEL_174;
          v341 = qword_4D043C0;
          if ( (_DWORD)qword_4D043C0 )
          {
            v326 = 1;
            v106 = 4;
            v107 = 1;
            v341 = 0;
            v318 = 4;
          }
          else
          {
            v107 = 1;
            sub_684B30(0xC8Fu, &dword_4F063F8);
            v318 = 4;
            v106 = 4;
            v326 = 1;
          }
LABEL_175:
          v108 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
          if ( !v108 )
            v108 = &dword_4F063F8;
          v330 = sub_86E480(v106, v108);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v330, 21, 0);
          sub_854980(0, (__int64)v330);
          v109 = (__int64)v330;
          v110 = v107;
          sub_86D170(v107, (__int64)v330, 0, 0, v111, v112);
          v312 = dword_4F06650[0];
          sub_7B8B50(v107, (unsigned int *)v330, v113, v114, v115, v116);
          if ( v341 )
          {
            if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201702) )
            {
              v109 = 2912;
              v110 = (unsigned __int64)&dword_4F063F8;
              sub_684B40(&dword_4F063F8, 0xB60u);
            }
            sub_7B8B50(v110, (unsigned int *)v109, v117, v118, v119, v120);
            *(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 4) = (4
                                                                 * *(_BYTE *)(qword_4F04C68[0]
                                                                            + 776LL * dword_4F04C64
                                                                            + 14))
                                                                & 8
                                                                | *(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 4)
                                                                & 0xF7;
            v311 = *((_QWORD *)v330 + 9);
            if ( v326 )
              goto LABEL_182;
            sub_7BE280(0x1Bu, 125, 0, 0, v121, (__int64)qword_4F04C68);
            ++*(_BYTE *)(qword_4F061C8 + 36LL);
            v369 = *(_QWORD *)&dword_4F063F8;
            sub_86E100((__int64)v330, &v365);
            v367 = (__m128i *)sub_724DC0();
            if ( v365 )
              v279 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v330 + 6) + 56LL) + 16LL);
            else
              v279 = *((_QWORD *)v330 + 6);
            v280 = 1;
            v370 = 0u;
            v281 = sub_7A30C0(v279, 1, 1, v367, &v370);
            v283 = v281;
            if ( v281 )
            {
              if ( v367[10].m128i_i8[13] )
              {
                v284 = 1;
                v308 = sub_711520((__int64)v367, 1, v282, v281, (__int64)&v370);
                v283 = v341;
                v315 = v308 == 0;
              }
              else
              {
                v315 = 0;
                v283 = v341;
                v284 = 1;
              }
            }
            else if ( dword_4F04C44 != -1
                   || (v303 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v303 + 6) & 6) != 0)
                   || *(_BYTE *)(v303 + 4) == 12
                   || (v324 = v283, v304 = sub_6E5430(), v283 = v324, !v304) )
            {
              v315 = 0;
              v284 = 0;
            }
            else
            {
              v305 = sub_67D9D0(0x1Cu, &v369);
              v280 = (__int64)&v370;
              sub_67E370((__int64)v305, &v370);
              v306 = (__int64)v305;
              v284 = 0;
              sub_685910(v306, (FILE *)&v370);
              v315 = 0;
              v283 = v324;
            }
            v322 = v283;
            sub_67E3D0(&v370);
            sub_724E30((__int64)&v367);
            v285 = v322;
            v286 = qword_4D03B98 + 176LL * unk_4D03B90;
            if ( dword_4F04C44 != -1
              || (v287 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v287 + 6) & 6) != 0)
              || *(_BYTE *)(v287 + 4) == 12 )
            {
              v323 = qword_4D03B98 + 176LL * unk_4D03B90;
              *(_BYTE *)(v286 + 4) = *(_BYTE *)(v286 + 4) & 0xFD | (2 * ((v285 ^ 1) & 1));
              v295 = sub_866B30();
              v286 = v323;
              v285 = v295;
              LOBYTE(v280) = v295 != 0;
              v280 = v284 & (unsigned int)v280;
              if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 0x42) == 2 && !v295 )
              {
                sub_878D60(v371, v280);
                v309 = sub_7AE940();
                v317 = 0;
                v286 = v323;
                *(_QWORD *)v371 = v309;
                v320 = v371;
                v314 = 0;
              }
              else
              {
                if ( !(_BYTE)v280 )
                  goto LABEL_482;
LABEL_455:
                v310 = v286;
                v288 = sub_883780(v312, v280, v286, v285);
                v317 = 1;
                v286 = v310;
                v314 = (__int64 *)v288;
                v320 = 0;
              }
            }
            else
            {
              if ( unk_4F04C48 != -1 )
                goto LABEL_455;
LABEL_482:
              v317 = 0;
              v320 = 0;
              v314 = 0;
            }
            v289 = (2 * v315) | *(_BYTE *)(v311 + 24) & 0xFC | v284;
            *(_BYTE *)(v311 + 24) = v289;
            if ( (*(_BYTE *)(v286 + 4) & 2) == 0 )
            {
              v290 = qword_4D03B98 + 176LL * unk_4D03B90;
              v291 = *(_BYTE *)(v290 + 4);
              if ( (v291 & 8) == 0 )
              {
                v292 = ((v289 >> 1) ^ 1) & 1;
                *(_BYTE *)(v290 + 4) = (4 * v292) | v291 & 0xFB;
                *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) = (2 * v292)
                                                                          | *(_BYTE *)(qword_4F04C68[0]
                                                                                     + 776LL * dword_4F04C64
                                                                                     + 14)
                                                                          & 0xFD;
              }
            }
            goto LABEL_271;
          }
          v196 = v326;
          if ( v326 )
          {
            v369 = *(_QWORD *)&dword_4F063F8;
            if ( word_4F06418[0] == 38 )
              sub_7B8B50(v107, (unsigned int *)v330, v117, v118, v326, v120);
            sub_7B8B50(v107, (unsigned int *)v330, v117, v118, v196, v120);
            v311 = 0;
LABEL_182:
            if ( word_4F06418[0] != 75 )
            {
              v317 = 0;
              v320 = 0;
              v314 = 0;
              v122 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_184:
              v123 = (*(_BYTE *)(v122 + 13) & 0x20) != 0;
              ++*(_BYTE *)(qword_4F061C8 + 94LL);
              if ( v326 )
              {
                if ( (*(_BYTE *)(v122 + 13) & 0x20) != 0 )
                {
                  sub_684B30(0xC84u, &v369);
                }
                else if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 2) == 0 )
                {
                  sub_684B30(0xC86u, &v369);
                }
                *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = (32 * (v318 == 3))
                                                                          | *(_BYTE *)(qword_4F04C68[0]
                                                                                     + 776LL * dword_4F04C64
                                                                                     + 13)
                                                                          & 0xDF;
                if ( word_4F06418[0] != 73 )
                  sub_6851C0(0xC85u, &dword_4F063F8);
              }
              sub_8745D0();
              v125 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              v126 = *(_BYTE *)(v125 + 13) & 0xDF;
              *(_BYTE *)(v125 + 13) = *(_BYTE *)(v125 + 13) & 0xDF | (32 * v123);
              v127 = v341;
              if ( v341 && (*(_BYTE *)(v125 + 14) & 2) == 0 )
              {
                v313 = dword_4F5FD80;
                v103 = qword_4F5FD78;
              }
              --*(_BYTE *)(qword_4F061C8 + 94LL);
              goto LABEL_192;
            }
            v317 = 0;
            v320 = 0;
            v314 = 0;
            goto LABEL_422;
          }
          sub_7BE280(0x1Bu, 125, 0, 0, v326, v120);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          v369 = *(_QWORD *)&dword_4F063F8;
          sub_86E100((__int64)v330, &v365);
          v317 = 0;
          v311 = 0;
          v314 = 0;
          v320 = 0;
LABEL_271:
          v127 = 18;
          sub_7BE280(0x1Cu, 18, 0, 0, v197, (__int64)qword_4F04C68);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          if ( word_4F06418[0] != 75 )
            goto LABEL_272;
LABEL_422:
          v127 = 0;
          if ( (unsigned __int16)sub_7BE840(0, 0) != 86 )
          {
            v127 = (__int64)dword_4F07508;
            sub_684B00(0x715u, dword_4F07508);
          }
LABEL_272:
          v122 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          v198 = v317 & (*(_BYTE *)(v122 + 14) >> 1);
          if ( !v198 )
            goto LABEL_184;
          v199 = 0;
          sub_854590(0);
          sub_854B40();
          if ( !v314 )
            goto LABEL_465;
          v127 = v314[1];
          if ( !v127 )
            v127 = v314[2];
          v199 = *v314;
          if ( !(unsigned int)sub_7BC350(*v314, (unsigned int *)v127) )
LABEL_465:
            sub_7BE9B0(v199, v127, v200, v201, v202, (__int64)qword_4F04C68);
          ((void (*)(void))sub_86E8F0)();
          v317 = v198;
LABEL_192:
          if ( word_4F06418[0] == 86 )
          {
            if ( v341 )
            {
              v228 = v311;
              *(_QWORD *)(v311 + 16) = *(_QWORD *)&dword_4F063F8;
            }
            else
            {
              v228 = (__int64)v330;
              *((_QWORD *)v330 + 11) = *(_QWORD *)&dword_4F063F8;
            }
            if ( v320 )
              *((_QWORD *)v320 + 1) = unk_4F06640;
            sub_7B8B50((unsigned __int64)v320, (unsigned int *)v127, v126, v228, v124, (__int64)qword_4F04C68);
            if ( v326 )
            {
              if ( word_4F06418[0] != 73 )
                sub_6851C0(0xC85u, &dword_4F063F8);
            }
            else if ( word_4F06418[0] == 75 )
            {
              sub_684B00(0x716u, dword_4F07508);
            }
            v229 = qword_4F5FD78;
            v127 = 176LL * unk_4D03B90;
            v230 = v127 + qword_4D03B98;
            v231 = dword_4F5FD80 | *(_DWORD *)(v127 + qword_4D03B98 + 124);
            *(_QWORD *)(v230 + 48) = 0;
            *(_DWORD *)(v230 + 124) = v231;
            LOBYTE(v231) = *(_BYTE *)(v230 + 4);
            *(_QWORD *)(v230 + 116) |= v229;
            v232 = dword_4F04C64;
            v233 = v231 | 1;
            *(_QWORD *)(v230 + 56) = 0;
            *(_BYTE *)(v230 + 4) = v233;
            v234 = 776 * v232;
            if ( !v341 || (v233 & 2) != 0 || (v127 += qword_4D03B98, v300 = *(_BYTE *)(v127 + 4), (v300 & 8) != 0) )
            {
              v235 = qword_4F04C68[0] + v234;
            }
            else
            {
              v301 = (*(_BYTE *)(v311 + 24) & 2) != 0;
              *(_BYTE *)(v127 + 4) = (4 * v301) | v300 & 0xFB;
              v235 = qword_4F04C68[0] + v234;
              v127 = (unsigned int)(2 * v301);
              *(_BYTE *)(v235 + 14) = (2 * v301) | *(_BYTE *)(v235 + 14) & 0xFD;
            }
            if ( (*(_BYTE *)(v235 + 14) & 2) != 0 && v317 )
            {
              v296 = 0;
              sub_854590(0);
              sub_854B40();
              if ( !v314 || (v127 = v314[2], v296 = *v314, !(unsigned int)sub_7BC350(*v314, (unsigned int *)v127)) )
                sub_7BE9B0(v296, v127, v297, v298, v299, (__int64)qword_4F04C68);
              ((void (*)(void))sub_86E8F0)();
            }
            else
            {
              v236 = *(_BYTE *)(v235 + 13);
              LOBYTE(v127) = v318 == 4;
              v237 = 32 * ((v236 & 0x20) != 0);
              v127 = (unsigned int)(32 * v127);
              *(_BYTE *)(v235 + 13) = v127 | v236 & 0xDF;
              qword_4F5FD78 = *(_QWORD *)(v230 + 104);
              dword_4F5FD80 = *(_DWORD *)(v230 + 112);
              sub_8745D0();
              v238 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              *(_BYTE *)(v238 + 13) = *(_BYTE *)(v238 + 13) & 0xDF | v237;
              if ( v341 && (*(_BYTE *)(v238 + 14) & 2) == 0 )
              {
                v313 = dword_4F5FD80;
                v103 = qword_4F5FD78;
              }
            }
          }
          if ( v320 && word_4F06418[0] != 9 && !qword_4D03E88 )
          {
            v127 = v312;
            *((_QWORD *)v320 + 2) = unk_4F06640;
            sub_883690(v320, v312);
          }
          if ( v365 )
          {
            v244 = sub_86B2C0(5);
            sub_86CBE0(v244);
            sub_863FC0(v244, v127, v245, v246, v247, v248);
          }
          if ( v341 )
          {
            *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) = (*(_BYTE *)(qword_4D03B98
                                                                                  + 176LL * unk_4D03B90
                                                                                  + 4) >> 2)
                                                                      & 2
                                                                      | *(_BYTE *)(qword_4F04C68[0]
                                                                                 + 776LL * dword_4F04C64
                                                                                 + 14)
                                                                      & 0xFD;
            sub_86F030();
            if ( dword_4F04C44 == -1 )
            {
              v211 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              if ( (*(_BYTE *)(v211 + 6) & 6) == 0 && *(_BYTE *)(v211 + 4) != 12 )
              {
                qword_4F5FD78 = v103;
                dword_4F5FD80 = v313;
              }
            }
            v54 = v330;
          }
          else
          {
            sub_86F030();
            v54 = v330;
          }
LABEL_124:
          sub_86C020((__int64)v54);
          *((_QWORD *)v54 + 1) = *(_QWORD *)&dword_4F061D8;
          result = (__int64)&dword_4F077C4;
          if ( dword_4F077C4 != 2 )
          {
LABEL_125:
            result = (__int64)&unk_4F07778;
            if ( unk_4F07778 > 199900 )
              result = sub_86F430(*(_BYTE **)(qword_4D03B98 + 176LL * unk_4D03B90 + 8));
          }
          break;
        case 0x60u:
          goto LABEL_41;
        case 0x66u:
          v371[0] = 0;
          sub_86F7D0(0x6Fu, dword_4F07508);
          if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
            sub_86EE70(111, dword_4F07508, v68);
          v69 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
          if ( !v69 )
            v69 = &dword_4F063F8;
          v54 = sub_86E480(0x10u, v69);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v54, 21, 0);
          sub_854980(0, (__int64)v54);
          sub_86D170(3, (__int64)v54, 0, 0, v70, v71);
          sub_7B8B50(3u, (unsigned int *)v54, v72, v73, v74, v75);
          sub_7BE280(0x1Bu, 125, 0, 0, v76, v77);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          sub_86E100((__int64)v54, v371);
          if ( *(_BYTE *)(*((_QWORD *)v54 + 6) + 24LL) == 2 )
            sub_684B00(0xEDu, dword_4F07508);
          v78 = sub_86B2C0(0);
          v79 = *(_QWORD *)v54;
          *(_BYTE *)(v78 + 72) |= 4u;
          *(_QWORD *)(v78 + 24) = v79;
          sub_86CBE0(v78);
          *(_QWORD *)(qword_4D03B98 + 176LL * unk_4D03B90 + 96) = **((_QWORD **)v54 + 6);
          sub_7BE280(0x1Cu, 18, 0, 0, v80, v81);
          qword_4F5FD78 = 0;
          dword_4F5FD80 = 0;
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          sub_8745D0();
          if ( (*(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 5) & 0x10) != 0 )
          {
            v82 = (__int64 *)*((_QWORD *)v54 + 10);
            v83 = *v82;
            if ( *v82 )
            {
              do
              {
                *(_QWORD *)(v83 + 32) = 0;
                v83 = *(_QWORD *)(v83 + 24);
              }
              while ( v83 );
              v82 = (__int64 *)*((_QWORD *)v54 + 10);
            }
            v82[2] = 0;
          }
          v84 = sub_86B2C0(5);
          sub_86CBE0(v84);
          v63 = v371[0];
          if ( v371[0] )
            goto LABEL_142;
          goto LABEL_123;
        case 0x6Cu:
          v370.m128i_i32[0] = 0;
          v52 = dword_4F5FD80 | qword_4F5FD78;
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
          {
            sub_86EE70(v7, v3, v11);
            v10 = qword_4D03B98 + 176LL * unk_4D03B90;
          }
          v53 = *(unsigned int **)(v10 + 160);
          if ( !v53 )
            v53 = &dword_4F063F8;
          v54 = sub_86E480(5u, v53);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v54, 21, 0);
          sub_854980(0, (__int64)v54);
          sub_86D170(4, (__int64)v54, 0, 0, v55, v56);
          sub_7B8B50(4u, (unsigned int *)v54, v57, v58, v59, v60);
          sub_7BE280(0x1Bu, 125, 0, 0, v61, v62);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          sub_86E100((__int64)v54, &v370);
          v63 = 18;
          sub_7BE280(0x1Cu, 18, 0, 0, v64, v65);
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          sub_8745D0();
          v66 = qword_4D03B98 + 176LL * unk_4D03B90;
          if ( !v52 && (*(_BYTE *)(v66 + 5) & 0x60) == 0 )
          {
            v63 = (__int64)v371;
            sub_684B30(0x80u, v371);
            dword_4F5FD80 = 1;
            v66 = qword_4D03B98 + 176LL * unk_4D03B90;
          }
          v67 = *(_QWORD *)(v66 + 80);
          if ( v67 )
          {
            v63 = *(_QWORD *)(v66 + 88);
            sub_86EEF0(v67, v63);
          }
          if ( !v370.m128i_i32[0] )
            goto LABEL_123;
LABEL_142:
          v85 = sub_86B2C0(5);
          sub_86CBE0(v85);
          sub_863FC0(v85, v63, v86, v87, v88, v89);
LABEL_123:
          sub_86F030();
          goto LABEL_124;
        case 0x89u:
        case 0x95u:
          sub_86F7D0(0x6Fu, dword_4F07508);
          *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
          v50 = sub_64F620(1, 1, (__int64 *)(qword_4D03B98 + 176LL * unk_4D03B90 + 16), v47, v48, v49);
          v51 = sub_86E480(0x12u, v371);
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v51, 21, 0);
          *((_QWORD *)v51 + 9) = v50;
          *((_QWORD *)v51 + 1) = *(_QWORD *)&dword_4F061D8;
          result = (__int64)&dword_4F077C4;
          if ( dword_4F077C4 == 2 )
          {
            result = unk_4F07778;
            if ( unk_4F07778 > 202001 )
            {
LABEL_163:
              v364 = 1;
            }
            else
            {
              v364 = 0;
              if ( unk_4F07778 > 201401 )
              {
                result = (__int64)&dword_4F077BC;
                if ( dword_4F077BC )
                {
                  result = (__int64)&qword_4F077B4;
                  if ( !(_DWORD)qword_4F077B4 )
                  {
                    v227 = v347;
                    result = 5;
                    if ( qword_4F077A8 >= 0x186A0u )
                      v227 = 5;
                    v347 = v227;
                  }
                }
              }
            }
          }
          else
          {
LABEL_107:
            v364 = 0;
          }
          goto LABEL_28;
        case 0xA3u:
          sub_8706F0(0, 0);
          result = dword_4D04884;
          v364 = dword_4D04884;
          goto LABEL_28;
        default:
          goto LABEL_19;
      }
      goto LABEL_28;
    }
    if ( word_4F06418[0] != 1 )
      goto LABEL_19;
    if ( (*(_DWORD *)&word_4D04A10 & 0x12019) != 0 )
      goto LABEL_19;
    v3 = 0;
    v7 = 0;
    if ( (unsigned __int16)sub_7BE840(0, 0) != 55 )
      goto LABEL_19;
    v37 = qword_4D03B98 + 176LL * unk_4D03B90;
    v370.m128i_i64[0] = *(_QWORD *)(v37 + 16);
    *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
    *(_BYTE *)(v37 + 5) |= 0x20u;
    *(_QWORD *)(v37 + 16) = 0;
    v38 = sub_64E550(1, 0);
    v39 = (__int64 *)v38;
    if ( *(_QWORD *)(v38 + 128) )
    {
      v40 = *(FILE **)v38;
      v41 = (unsigned __int64)v371;
      sub_685920(v371, *(FILE **)v38, 8u);
      dword_4F5FD80 = 0;
      qword_4F5FD78 = 0x100000001LL;
    }
    else
    {
      *(_BYTE *)(v38 + 120) = qword_4F5FD78 & 1 | *(_BYTE *)(v38 + 120) & 0xFE;
      v360 = (FILE *)(v38 + 64);
      v212 = sub_86E480(7u, (unsigned int *)(v38 + 64));
      v39[16] = (__int64)v212;
      *((_QWORD *)v212 + 9) = v39;
      if ( !dword_4F04C3C )
        sub_8699D0(v39[16], 21, 0);
      v213 = v39[16];
      *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
      *(_QWORD *)(v213 + 8) = qword_4F063F0;
      if ( dword_4F077C4 == 2 )
        *(_QWORD *)(v39[16] + 80) = qword_4F06BC0;
      v41 = v39[16];
      v40 = v360;
      sub_86D5C0(v41, v360);
      v45 = qword_4F04C68;
      v42 = *(unsigned int *)(*v39 + 44);
      *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 416) = v42;
      if ( dword_4F077C4 == 2 )
      {
        for ( ; qword_4D03B98 <= v37; v37 -= 176LL )
        {
          if ( *(_DWORD *)v37 )
          {
            if ( *(_DWORD *)v37 == 8 )
              break;
          }
          else
          {
            *(_BYTE *)(v37 + 5) |= 1u;
            if ( (*(_BYTE *)(v37 + 4) & 0x20) != 0 )
              break;
          }
        }
        v41 = v39[16];
        sub_86C020(v41);
        v45 = qword_4F04C68;
        if ( dword_4F077C4 != 2 || unk_4F07778 <= 202301 )
        {
          if ( dword_4D0488C
            || word_4D04898
            && (_DWORD)qword_4F077B4
            && qword_4F077A0 > 0x765Bu
            && (v41 = dword_4F063F8, v307 = sub_729F80(dword_4F063F8), v45 = qword_4F04C68, v307) )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 5) != 0 )
            {
              v40 = v360;
              v41 = 2754;
              sub_6851C0(0xAC2u, v360);
              *(_BYTE *)(*v39 + 81) |= 1u;
            }
            else
            {
              *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 13) |= 0x10u;
            }
          }
        }
      }
    }
    sub_7B8B50(v41, (unsigned int *)v40, v42, v43, v44, (__int64)v45);
    if ( dword_4D043E0 && word_4F06418[0] == 142 )
    {
      v302 = sub_5CB9F0(&v370);
      *v302 = (_QWORD *)sub_5CC970(17);
    }
    if ( v370.m128i_i64[0] )
      sub_5CEC90(v370.m128i_i64[0], (__int64)v39, 12);
LABEL_240:
    if ( !v363 || (LOBYTE(result) = v363[193], (result & 2) == 0) || v364 )
    {
      v357 = 1;
      goto LABEL_39;
    }
    v357 = 1;
    v21 = 1;
LABEL_31:
    if ( (result & 5) == 0 )
      goto LABEL_38;
    v22 = v363[195];
    if ( (v22 & 8) != 0 && v347 == 8 && !dword_4D04964 )
    {
LABEL_81:
      v347 = 5;
      v23 = 5;
      goto LABEL_37;
    }
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0x1D4BFu )
          goto LABEL_36;
        goto LABEL_80;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_36;
    }
    if ( dword_4F077C4 != 2 || qword_4F077A0 <= 0x765Bu )
      goto LABEL_36;
LABEL_80:
    if ( (v22 & 1) != 0 )
      goto LABEL_81;
LABEL_36:
    v23 = v347;
LABEL_37:
    sub_684AA0(v23, (v363[174] == 1) + 2388, &v368);
LABEL_38:
    result = qword_4F04C68[0] + 776LL * dword_4F04C58;
    *(_BYTE *)(result + 13) |= 0x10u;
    if ( !v21 )
      return result;
LABEL_39:
    a2 = 0;
  }
  if ( word_4F06418[0] != 268 )
  {
LABEL_19:
    if ( !a2 && word_4F06418[0] == 187 )
    {
      sub_7B8B50(v7, v3, v11, v10, v8, v9);
      a2 = 1;
    }
    v12 = v364;
    v13 = dword_4F077C4;
    if ( dword_4F077C4 != 2 )
    {
      if ( v364 )
        goto LABEL_22;
      if ( dword_4D0488C )
      {
        v347 = 5;
        if ( !(a1 | v357) )
          goto LABEL_23;
        goto LABEL_92;
      }
      if ( !word_4D04898 )
        goto LABEL_22;
      goto LABEL_289;
    }
    v257 = 176LL * unk_4D03B90;
    v258 = v257 + qword_4D03B98;
    v259 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
    *(_BYTE *)(v258 + 5) |= 0x80u;
    v366 = 0;
    *(_QWORD *)(v258 + 144) = v259;
    *(_QWORD *)(qword_4D03B98 + v257 + 136) = &v366;
    if ( !v12 )
    {
      if ( dword_4D0488C )
      {
        v347 = 5;
      }
      else if ( word_4D04898 )
      {
LABEL_289:
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A0 > 0x765Bu )
          {
            v293 = !sub_729F80(dword_4F063F8);
            v13 = dword_4F077C4;
            if ( !v293 )
              v347 = 5;
          }
        }
        if ( v13 == 2 )
          goto LABEL_291;
LABEL_22:
        if ( !(a1 | v357) )
        {
LABEL_23:
          ++*(_BYTE *)(qword_4F061C8 + 83LL);
          sub_86F7D0(0x6Fu, dword_4F07508);
          sub_86E9F0(a2);
          sub_7BE280(0x4Bu, 65, 0, 0, v14, v15);
          --*(_BYTE *)(qword_4F061C8 + 83LL);
          goto LABEL_24;
        }
LABEL_92:
        if ( !(unsigned int)sub_651B00(3u) )
          goto LABEL_23;
        if ( a1 )
        {
          sub_6851C0(0x20Du, dword_4F07508);
        }
        else
        {
          if ( dword_4F077C4 == 2 )
          {
LABEL_494:
            sub_6851C0(0x430u, dword_4F07508);
            goto LABEL_100;
          }
          if ( unk_4F07778 <= 202310 )
          {
            if ( unk_4F07778 > 199900 )
            {
              v46 = 5;
              if ( dword_4D04964 )
                v46 = unk_4F07471;
              sub_684AC0(v46, 0x430u);
              goto LABEL_100;
            }
            goto LABEL_494;
          }
        }
LABEL_100:
        sub_86E660(a2, 0);
        goto LABEL_24;
      }
    }
LABEL_291:
    if ( ((word_4F06418[0] - 175) & 0xFFFB) == 0 || (unsigned int)sub_679C10(2u) )
    {
      v209 = qword_4D03B98 + 176LL * unk_4D03B90;
      if ( qword_4D0495C && a1 )
        sub_6851C0(0x20Du, dword_4F07508);
      v210 = *(_QWORD *)(v209 + 16);
      if ( v210 )
      {
        sub_5CC9F0(v210);
        *(_QWORD *)(v209 + 16) = 0;
      }
      sub_86E660(a2, &v364);
      if ( !v364
        && (dword_4D0488C
         || word_4D04898 && (_DWORD)qword_4F077B4 && qword_4F077A0 > 0x765Bu && sub_729F80(dword_4F063F8)) )
      {
        v347 = 5;
      }
LABEL_24:
      result = (__int64)&dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        result = 176LL * unk_4D03B90;
        v17 = result + qword_4D03B98;
        *(_BYTE *)(v17 + 5) &= ~0x80u;
        *(_QWORD *)(v17 + 144) = 0;
        *(_QWORD *)(qword_4D03B98 + result + 136) = 0;
        if ( !v363 )
          return result;
        goto LABEL_29;
      }
LABEL_28:
      if ( !v363 )
        return result;
LABEL_29:
      result = (unsigned __int8)v363[193];
      if ( (result & 2) == 0 )
        return result;
      goto LABEL_30;
    }
    if ( dword_4F077C4 == 2 )
      goto LABEL_23;
    goto LABEL_22;
  }
LABEL_41:
  v367 = 0;
  v369 = 0;
  v24 = dword_4F07508;
  v370.m128i_i64[0] = 0;
  sub_86F7D0(0x6Fu, dword_4F07508);
  v336 = 0;
  *(_QWORD *)v371 = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
  v27 = dword_4D047EC;
  if ( dword_4D047EC )
  {
    v25 = dword_4D047E8;
    if ( dword_4D047E8 )
    {
      if ( (_DWORD)qword_4F5FD78 )
      {
        v24 = (unsigned int *)qword_4F5FD70;
        v336 = sub_86B560(qword_4F5FD68, qword_4F5FD70);
      }
    }
  }
  if ( !qword_4F04C50 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    sub_6851D0(0xB89u);
    goto LABEL_67;
  }
  v28 = *(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 152LL);
  for ( i = *(_BYTE **)(qword_4F04C50 + 32LL); *(_BYTE *)(v28 + 140) == 12; v28 = *(_QWORD *)(v28 + 160) )
    ;
  v29 = *(_QWORD *)(qword_4F04C50 + 32LL);
  v329 = *(__int64 **)(v28 + 160);
  if ( (char)i[207] < 0 )
  {
    if ( word_4F06418[0] == 96 )
    {
      v24 = &dword_4F063F8;
      v29 = 2740;
      sub_6851C0(0xAB4u, &dword_4F063F8);
    }
  }
  else if ( word_4F06418[0] == 268 )
  {
    if ( (i[198] & 0x10) != 0 )
    {
      v24 = &dword_4F063F8;
      sub_6851C0(0xE7Cu, &dword_4F063F8);
    }
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 13) & 8) != 0 )
    {
      v24 = &dword_4F063F8;
      v29 = 2741;
      sub_6851C0(0xAB5u, &dword_4F063F8);
    }
    else
    {
      v29 = (unsigned __int64)i;
      sub_71DF80((__int64)i);
    }
  }
  sub_7B8B50(v29, v24, v25, v27, v26, (__int64)&dword_4F061D8);
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  v339 = word_4F06418[0];
  v30 = i[174];
  if ( v30 == 1 )
  {
    if ( unk_4D03B90 > 0 && *(_DWORD *)qword_4D03B98 == 8 && (*(_BYTE *)(qword_4D03B98 + 180) & 0x20) != 0 )
    {
      v31 = 0;
      sub_6851C0(0x3BAu, v371);
      sub_854B40();
      v325 = 0;
      v329 = (__int64 *)sub_72C930();
      v316 = 0;
      goto LABEL_55;
    }
    if ( word_4F06418[0] != 75 )
    {
LABEL_414:
      v260 = 278;
LABEL_415:
      sub_6851C0(v260, dword_4F07508);
      v325 = 0;
      v329 = (__int64 *)sub_72C930();
      goto LABEL_54;
    }
LABEL_317:
    if ( (char)i[207] >= 0 )
      sub_86B690(0, &v367);
    else
      v370.m128i_i64[0] = 0;
    v325 = 0;
    goto LABEL_54;
  }
  if ( word_4F06418[0] == 75 )
    goto LABEL_317;
  if ( v30 == 2 )
    goto LABEL_414;
  if ( (unsigned int)sub_8D2600(v329) || (v325 = sub_8D3D40(v329)) != 0 )
  {
    if ( dword_4F077C4 != 2 )
    {
      v325 = dword_4F077C0;
      if ( dword_4F077C0 )
      {
        v325 = 1;
      }
      else
      {
        sub_6851C0(0x76u, dword_4F07508);
        v329 = (__int64 *)sub_72C930();
      }
      goto LABEL_54;
    }
    v325 = qword_4D0495C;
    if ( (_DWORD)qword_4D0495C )
    {
      v325 = sub_8D2600(v329);
      if ( v325 )
      {
        v260 = 118;
        goto LABEL_415;
      }
    }
  }
LABEL_54:
  v31 = 1;
  v316 = sub_869D30();
LABEL_55:
  if ( v339 != 75 )
    v367 = (__m128i *)sub_6D0CE0(v329, 0x78u, &v369, v370.m128i_i64);
  if ( v336 )
  {
    if ( v367 )
    {
      v34 = *(_QWORD *)(v28 + 160);
      if ( !(unsigned int)sub_731920((__int64)v367, 1, 0, v32, v33, (__int64)&dword_4F061D8) )
      {
        v35 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
        if ( !v35 )
          v35 = &dword_4F063F8;
        v340 = sub_86E480(0, v35);
        if ( (unsigned int)sub_8D2600(v34) )
        {
          v36 = qword_4F5FD78;
          *((_QWORD *)v340 + 6) = v367;
          v367 = 0;
          sub_86B010((__int64)v336, v36);
          if ( !v31 )
            goto LABEL_64;
LABEL_342:
          v214 = sub_86E480((((char)i[207] >> 7) & 2u) + 8, v371);
          v215 = (__int64)v214;
          if ( !dword_4F04C3C )
            sub_8699D0((__int64)v214, 21, v316);
          if ( v215 )
          {
            sub_854980(0, v215);
            *(_QWORD *)(v215 + 48) = v367;
            *(_QWORD *)(v215 + 72) = v369;
            goto LABEL_346;
          }
LABEL_64:
          if ( word_4F06418[0] == 75 )
            *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          goto LABEL_66;
        }
        v319 = sub_736020(v34, 0);
        v321 = sub_731250(v319);
        v333 = v367;
        v219 = sub_6E9930(56, v34);
        v222 = sub_698020(v321, v219, (__int64)v333, v220, (__int64)v321, v221);
        *((_QWORD *)v340 + 6) = v222;
        *(_BYTE *)(v222 + 25) |= 4u;
        v367 = (__m128i *)sub_73E830(v319);
      }
    }
    sub_86B010((__int64)v336, qword_4F5FD78);
  }
  if ( !v31 )
    goto LABEL_64;
  if ( !v325 )
    goto LABEL_342;
  v223 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
  if ( !v223 )
    v223 = &dword_4F063F8;
  v224 = sub_86E480(0, v223);
  v225 = v224;
  if ( !v224 )
    goto LABEL_64;
  sub_854980(0, (__int64)v224);
  *((_QWORD *)v225 + 6) = v367;
  v226 = sub_86E480(8u, v371);
  v215 = (__int64)v226;
  v33 = dword_4F04C3C;
  if ( !dword_4F04C3C )
    sub_8699D0((__int64)v226, 21, v316);
LABEL_346:
  if ( dword_4D0488C
    || word_4D04898
    && (v33 = (unsigned int)qword_4F077B4, (_DWORD)qword_4F077B4)
    && qword_4F077A0 > 0x765Bu
    && sub_729F80(dword_4F063F8)
    || (v216 = *(_QWORD *)(qword_4F04C50 + 32LL), (*(_BYTE *)(v216 + 193) & 2) == 0)
    || *(_BYTE *)(v216 + 174) == 1 )
  {
    if ( (char)i[207] < 0 )
      *(_QWORD *)(v215 + 48) = sub_695660(v370.m128i_i64[0], 0, (__int64 *)v215);
  }
  else
  {
    v217 = qword_4F04C68[0] + 776LL * dword_4F04C58;
    v218 = *(_BYTE *)(v217 + 13);
    if ( (v218 & 8) != 0 )
    {
      *(_BYTE *)(v217 + 13) = v218 | 0x10;
      sub_6851C0(0x953u, v371);
    }
    else if ( !v367 && !v369 )
    {
      *(_BYTE *)(v217 + 13) = v218 | 0x10;
    }
  }
  if ( *(_BYTE *)(v215 + 40) == 8 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 13) |= 8u;
  if ( word_4F06418[0] == 75 )
    *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
  *(_QWORD *)(v215 + 8) = *(_QWORD *)&dword_4F061D8;
LABEL_66:
  sub_7BE280(0x4Bu, 65, 0, 0, v33, (__int64)&dword_4F061D8);
LABEL_67:
  result = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  if ( v363 )
  {
    result = (unsigned __int8)v363[193];
    if ( (result & 2) != 0 && v363[174] == 1 )
    {
LABEL_30:
      v21 = v364;
      if ( v364 )
        return result;
      goto LABEL_31;
    }
  }
  return result;
}
