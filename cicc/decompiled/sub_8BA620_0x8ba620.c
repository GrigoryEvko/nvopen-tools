// Function: sub_8BA620
// Address: 0x8ba620
//
__int64 ***__fastcall sub_8BA620(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  int v6; // r14d
  __int64 v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rcx
  __int64 v12; // rsi
  unsigned int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int64 v21; // rdi
  int v22; // r8d
  int v23; // eax
  int v24; // edx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdi
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // rax
  int v34; // edx
  int v35; // eax
  char v37; // al
  int v38; // eax
  bool v39; // zf
  __int64 v40; // r8
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __m128i v43; // xmm3
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // xmm4_8
  __m128i v48; // xmm6
  __m128i v49; // xmm7
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __m128i *v56; // rax
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 *v59; // r9
  __int64 v60; // rdx
  char v61; // dl
  _QWORD *v62; // rbx
  int v63; // r13d
  __int64 *v64; // r12
  __int64 v65; // rax
  __int64 v66; // r13
  char v67; // dl
  char v68; // al
  char v69; // dl
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rax
  __int64 v75; // r13
  unsigned int v76; // r14d
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdi
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  unsigned __int64 v94; // rbx
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r9
  __int64 v102; // rdi
  _QWORD *v103; // rcx
  __int64 v104; // r8
  __int64 *v105; // r9
  __int64 *v106; // r14
  int v107; // r13d
  int v108; // eax
  int v109; // r14d
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 *v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  unsigned __int16 v116; // ax
  unsigned __int16 v117; // r14
  __int64 v118; // rdx
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rax
  __int64 v127; // rbx
  __int64 v128; // rax
  __int64 *v129; // r14
  __int64 v130; // rdx
  __int64 v131; // r8
  _QWORD *i; // rax
  __int64 v133; // rsi
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r8
  __int64 v141; // r8
  int v142; // eax
  __int64 *v143; // r11
  __int64 v144; // r12
  unsigned int *v145; // rdx
  __int64 v146; // rcx
  __int64 v147; // r8
  __int64 v148; // r9
  __int64 v149; // r13
  __int64 v150; // rsi
  __int64 v151; // rdi
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  __int64 v156; // r12
  char v157; // al
  __int64 v158; // rcx
  __int64 v159; // rax
  __int64 v160; // rcx
  __int64 v161; // r8
  __int64 v162; // r9
  unsigned int v163; // edi
  __int64 v164; // rax
  __int64 v165; // rsi
  __int64 v166; // rax
  __int64 *v167; // r11
  __int64 v168; // rax
  __int64 v169; // rdi
  __int64 v170; // rdx
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 *v175; // rdx
  __int64 v176; // rcx
  __int64 v177; // r8
  __int64 v178; // r9
  __m128i v179; // xmm1
  __m128i v180; // xmm2
  __m128i v181; // xmm3
  __int64 v182; // rbx
  __int64 v183; // rdx
  __int64 v184; // rax
  unsigned int *v185; // rsi
  __int64 v186; // rdx
  __int64 v187; // rcx
  __int64 v188; // r8
  __int64 v189; // r9
  _QWORD *v190; // rax
  __m128i *v191; // r14
  _QWORD *v192; // r12
  __int64 v193; // rsi
  __int8 v194; // dl
  __int8 v195; // dl
  __int64 v196; // rcx
  __int64 v197; // rdx
  __int64 v198; // r8
  __int64 v199; // r9
  __m128i *v200; // r8
  char v201; // al
  int v202; // eax
  __int64 v203; // r12
  bool v204; // al
  __int64 v205; // rdi
  __int64 v206; // rax
  char v207; // al
  __int64 **v208; // r12
  __m128i v209; // xmm5
  __m128i v210; // xmm6
  __m128i v211; // xmm7
  __m128i v212; // xmm5
  __m128i v213; // xmm6
  __m128i v214; // xmm7
  __int8 v215; // al
  __int8 v216; // al
  __int64 v217; // rdx
  __int64 v218; // rcx
  __int64 v219; // r8
  __int64 v220; // r9
  __int64 v221; // r12
  __int64 v222; // rbx
  __int64 *v223; // rax
  __int64 *v224; // r12
  char v225; // al
  __int64 v226; // rax
  char v227; // dl
  __m128i v228; // xmm1
  __m128i v229; // xmm2
  __m128i v230; // xmm3
  __int64 v231; // r12
  __int64 v232; // rax
  __int64 v233; // r11
  __int64 v234; // rax
  char v235; // dl
  char v236; // dl
  __m128i *v237; // rax
  char v238; // al
  unsigned __int64 v239; // rdi
  __int64 v240; // rdx
  __int64 v241; // r8
  __int64 *v242; // r9
  __int64 v243; // rcx
  __int64 *v244; // rax
  char v245; // cl
  __int64 v246; // rdx
  __int64 v247; // rcx
  __int64 v248; // r8
  __int64 v249; // r9
  __int64 *v250; // rdx
  __int64 v251; // rcx
  __int64 v252; // r8
  __int64 v253; // r9
  __int16 v254; // ax
  __int64 v255; // rax
  char v256; // al
  __int64 v257; // rdx
  __int64 v258; // rcx
  __int64 v259; // r8
  __int64 v260; // r9
  __int64 v261; // rdx
  __int64 v262; // r9
  __int64 v263; // rax
  __int64 v264; // rdi
  __int64 v265; // rax
  __int64 *v266; // r12
  __int64 v267; // rax
  __int64 v268; // rbx
  unsigned int v269; // ecx
  int v270; // eax
  unsigned int v271; // r8d
  __int64 v272; // rax
  __int64 v273; // rax
  char v274; // al
  char v275; // al
  unsigned __int64 v276; // rdi
  __int64 v277; // rax
  __int64 v278; // r8
  __int64 *v279; // r9
  __int64 v280; // rdx
  _QWORD *v281; // rcx
  __int64 v282; // rdi
  char v283; // al
  unsigned __int64 v284; // rdi
  __int64 v285; // rcx
  __int64 v286; // r8
  __int64 v287; // r9
  __int64 *v288; // r11
  unsigned __int16 v289; // ax
  __int64 v290; // rdx
  char v291; // al
  unsigned __int16 *v292; // rcx
  __int64 v293; // rdx
  __int64 v294; // rcx
  __int64 v295; // r8
  __int64 v296; // r9
  __m128i v297; // xmm5
  __m128i v298; // xmm6
  __m128i v299; // xmm7
  __int64 v300; // rax
  __int64 v301; // rdx
  __int64 v302; // rax
  __int64 v303; // rbx
  _QWORD *v304; // rax
  __int64 v305; // rbx
  __int64 v306; // rdx
  __int64 v307; // r9
  __int64 v308; // r13
  _QWORD *v309; // rbx
  __int64 v310; // rax
  bool v311; // di
  __int64 *v312; // rax
  __int64 v313; // r12
  __m128i *v314; // rax
  __int64 v315; // rax
  _BOOL4 v316; // eax
  char v317; // al
  __int64 v318; // rax
  int v319; // eax
  __int64 v320; // rdx
  __int64 v321; // rcx
  __int64 v322; // r8
  __int64 v323; // r9
  __int64 v324; // rax
  __int8 v325; // dl
  char v326; // al
  unsigned int v327; // esi
  __m128i *v328; // rax
  __int64 v329; // [rsp-10h] [rbp-320h]
  unsigned __int64 v330; // [rsp-8h] [rbp-318h]
  __int64 v331; // [rsp+8h] [rbp-308h]
  __int64 v332; // [rsp+10h] [rbp-300h]
  __int64 v333; // [rsp+10h] [rbp-300h]
  __int64 v334; // [rsp+10h] [rbp-300h]
  __int64 *v335; // [rsp+10h] [rbp-300h]
  __int64 *v336; // [rsp+10h] [rbp-300h]
  __int64 v337; // [rsp+18h] [rbp-2F8h]
  char v338; // [rsp+18h] [rbp-2F8h]
  __int64 v339; // [rsp+18h] [rbp-2F8h]
  __int64 *v340; // [rsp+18h] [rbp-2F8h]
  char v341; // [rsp+18h] [rbp-2F8h]
  __int64 v342; // [rsp+18h] [rbp-2F8h]
  __int64 v343; // [rsp+18h] [rbp-2F8h]
  int v344; // [rsp+20h] [rbp-2F0h]
  __int64 v345; // [rsp+20h] [rbp-2F0h]
  _BOOL4 v346; // [rsp+20h] [rbp-2F0h]
  unsigned int v347; // [rsp+28h] [rbp-2E8h]
  char v348; // [rsp+28h] [rbp-2E8h]
  __int64 v349; // [rsp+28h] [rbp-2E8h]
  char v350; // [rsp+28h] [rbp-2E8h]
  bool v351; // [rsp+30h] [rbp-2E0h]
  __int64 v352; // [rsp+30h] [rbp-2E0h]
  int v353; // [rsp+30h] [rbp-2E0h]
  unsigned int v354; // [rsp+38h] [rbp-2D8h]
  const __m128i *v355; // [rsp+38h] [rbp-2D8h]
  __int64 v356; // [rsp+38h] [rbp-2D8h]
  __int64 v357; // [rsp+38h] [rbp-2D8h]
  bool v358; // [rsp+38h] [rbp-2D8h]
  unsigned __int8 v359; // [rsp+40h] [rbp-2D0h]
  __int64 v360; // [rsp+40h] [rbp-2D0h]
  unsigned int v361; // [rsp+40h] [rbp-2D0h]
  __int64 v362; // [rsp+40h] [rbp-2D0h]
  unsigned int v363; // [rsp+40h] [rbp-2D0h]
  __int64 v364; // [rsp+40h] [rbp-2D0h]
  _DWORD *v365; // [rsp+48h] [rbp-2C8h]
  int v366; // [rsp+48h] [rbp-2C8h]
  __int64 *v367; // [rsp+48h] [rbp-2C8h]
  unsigned int v368; // [rsp+48h] [rbp-2C8h]
  __int64 v369; // [rsp+50h] [rbp-2C0h]
  __int64 v370; // [rsp+50h] [rbp-2C0h]
  __int64 v371; // [rsp+50h] [rbp-2C0h]
  int v372; // [rsp+50h] [rbp-2C0h]
  __int64 **v373; // [rsp+58h] [rbp-2B8h]
  const __m128i *v374; // [rsp+60h] [rbp-2B0h]
  _QWORD *v375; // [rsp+60h] [rbp-2B0h]
  __int64 v376; // [rsp+68h] [rbp-2A8h]
  __int64 v377; // [rsp+68h] [rbp-2A8h]
  __int64 *v378; // [rsp+68h] [rbp-2A8h]
  __int64 *v379; // [rsp+68h] [rbp-2A8h]
  __int64 *v380; // [rsp+68h] [rbp-2A8h]
  int v381; // [rsp+68h] [rbp-2A8h]
  __int64 v382; // [rsp+68h] [rbp-2A8h]
  unsigned int v383; // [rsp+78h] [rbp-298h] BYREF
  unsigned int v384; // [rsp+7Ch] [rbp-294h] BYREF
  __int64 *v385; // [rsp+80h] [rbp-290h] BYREF
  __int64 v386; // [rsp+88h] [rbp-288h] BYREF
  __int64 *v387; // [rsp+90h] [rbp-280h] BYREF
  __int64 v388; // [rsp+98h] [rbp-278h] BYREF
  __m128i v389[2]; // [rsp+A0h] [rbp-270h] BYREF
  __m128i v390; // [rsp+C0h] [rbp-250h] BYREF
  __m128i v391; // [rsp+D0h] [rbp-240h]
  __m128i v392; // [rsp+E0h] [rbp-230h]
  __m128i v393; // [rsp+F0h] [rbp-220h]
  __m128i v394[33]; // [rsp+100h] [rbp-210h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 60);
  v373 = qword_4D03B88;
  v7 = (int)dword_4F04C5C;
  v365 = (_DWORD *)(a1 + 140);
  v388 = *(_QWORD *)&dword_4F063F8;
  qword_4D03B88 = 0;
  sub_854430();
  if ( !*(_DWORD *)(a1 + 124) )
  {
    sub_7BDB60(1);
    *(_DWORD *)(a1 + 124) = 1;
  }
  *(_DWORD *)(a1 + 168) = sub_88D5F0();
  if ( v6 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) & 2) != 0 )
  {
    sub_6851C0(0x42Eu, v365);
    *(_DWORD *)(a1 + 52) = 1;
  }
  unk_4F072F5 = 1;
  v8 = sub_727340();
  v11 = (_QWORD *)dword_4F04C3C;
  v8[8] = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  *(_QWORD *)((char *)v8 + 140) = *(_QWORD *)(a1 + 140);
  if ( !(_DWORD)v11 )
  {
    v375 = v8;
    sub_8699D0((__int64)v8, 59, 0);
    v8 = v375;
    dword_4F06C5C = 0;
  }
  *(_QWORD *)(a1 + 336) = v8;
  sub_8B0460(a1, a3, 0, v11, v9, v10);
  v12 = *(unsigned int *)(a1 + 156);
  v13 = dword_4F06650[0];
  v14 = qword_4F061C0;
  v374 = (const __m128i *)(a1 + 248);
  *(_DWORD *)(a1 + 160) = dword_4F06650[0];
  sub_7AE700((__int64)(v14 + 3), v12, v13, 0, a1 + 248);
  *(_DWORD *)(a1 + 280) = dword_4F06650[0];
  sub_89D5D0(a1, v12);
  v16 = qword_4F04C68[0];
  v17 = (int)dword_4F04C5C;
  while ( 1 )
  {
    v18 = qword_4F04C68[0] + 776 * v17;
    v19 = *(unsigned __int8 *)(v18 + 4);
    if ( (_BYTE)v19 != 8 )
      break;
    v17 = *(int *)(v18 + 344);
    if ( (_DWORD)v17 == -1 )
      BUG();
  }
  v20 = 0xA3A0FD5C5F02A3A1LL * ((v18 - qword_4F04C68[0]) >> 3);
  *(_DWORD *)(a1 + 20) = (_BYTE)v19 == 6;
  *(_DWORD *)(a1 + 208) = v20;
  v21 = 1594008481 * (unsigned int)((v18 - v16) >> 3);
  if ( (_BYTE)v19 == 6 )
  {
    *(_QWORD *)(a1 + 240) = *(_QWORD *)(v18 + 208);
    *(_BYTE *)(a1 + 164) = *(_BYTE *)(v18 + 5) & 3;
    if ( (*(_BYTE *)(v18 + 7) & 1) != 0 )
      *(_DWORD *)(a1 + 84) = 1;
    v22 = *(_DWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v18 + 5) & 8) == 0 )
    {
      *(_DWORD *)(a1 + 200) = v20;
      if ( !v22 )
      {
        *(_DWORD *)(a1 + 204) = v20;
LABEL_15:
        v23 = v22 != 0;
        goto LABEL_16;
      }
      goto LABEL_89;
    }
    if ( !v22 )
    {
      *(_QWORD *)(a1 + 200) = -1;
LABEL_59:
      v19 = v5 + 24;
      v21 = *(_DWORD *)(a1 + 92) == 0 ? 437 : 2280;
      sub_6851C0(v21, (_DWORD *)(v5 + 24));
      v38 = *(_DWORD *)(a1 + 208);
      v39 = *(_QWORD *)(a1 + 192) == 0;
      *(_DWORD *)(a1 + 52) = 1;
      *(_DWORD *)(a1 + 204) = v38;
      *(_DWORD *)(a1 + 200) = v38;
      if ( !v39 )
        goto LABEL_18;
LABEL_60:
      *(_DWORD *)(a1 + 32) = 1;
      goto LABEL_61;
    }
    *(_DWORD *)(a1 + 200) = v20;
  }
  else
  {
    v37 = *(_BYTE *)(v18 + 4);
    if ( v37 && (unsigned __int8)(v37 - 3) > 1u )
    {
      *(_QWORD *)(a1 + 200) = -1;
      *(_DWORD *)(a1 + 16) = 0;
      goto LABEL_59;
    }
    v23 = *(_DWORD *)(a1 + 16);
    *(_DWORD *)(a1 + 200) = v20;
    v22 = v23;
    if ( !v23 )
    {
      *(_DWORD *)(a1 + 204) = v20;
      goto LABEL_16;
    }
  }
LABEL_89:
  if ( unk_4F04C48 == -1 || (*(_BYTE *)(v16 + 776LL * dword_4F04C64 + 6) & 6) == 0 )
    v21 = (unsigned int)dword_4F04C34;
  *(_DWORD *)(a1 + 204) = v21;
  v23 = 0;
  if ( (_BYTE)v19 == 6 )
    goto LABEL_15;
LABEL_16:
  *(_DWORD *)(a1 + 16) = v23;
  if ( (_DWORD)v21 == -1 )
    goto LABEL_59;
  if ( !*(_QWORD *)(a1 + 192) )
    goto LABEL_60;
LABEL_18:
  *(_QWORD *)(*(_QWORD *)(a1 + 336) + 176LL) = *(_QWORD *)(a1 + 488);
  v24 = *(_DWORD *)(a1 + 20);
  v394[0].m128i_i32[0] = 0;
  if ( v24 && !*(_DWORD *)(a1 + 16) && *(_QWORD *)(a1 + 224) > 1u )
  {
    sub_6851C0(0x309u, dword_4F07508);
    *(_DWORD *)(a1 + 52) = 1;
  }
  v25 = *(_QWORD *)(a1 + 192);
  v26 = *(_QWORD *)(v25 + 24);
  if ( v26 )
  {
    v27 = *(_QWORD *)(v26 + 24);
    if ( v27 )
    {
      v28 = *(_QWORD *)(v27 + 24);
      if ( v28 )
      {
        v29 = *(_QWORD *)(v28 + 24);
        if ( v29 )
          sub_890340(v29);
        v30 = *(_QWORD **)v28;
        if ( *(_QWORD *)v28 )
        {
          do
          {
            *(_BYTE *)(v30[1] + 83LL) |= 0x40u;
            v30 = (_QWORD *)*v30;
          }
          while ( v30 );
        }
      }
      v31 = *(_QWORD **)v27;
      if ( *(_QWORD *)v27 )
      {
        do
        {
          *(_BYTE *)(v31[1] + 83LL) |= 0x40u;
          v31 = (_QWORD *)*v31;
        }
        while ( v31 );
      }
    }
    v32 = *(_QWORD **)v26;
    if ( *(_QWORD *)v26 )
    {
      do
      {
        *(_BYTE *)(v32[1] + 83LL) |= 0x40u;
        v32 = (_QWORD *)*v32;
      }
      while ( v32 );
    }
  }
  v33 = *(_QWORD **)v25;
  if ( *(_QWORD *)v25 )
  {
    do
    {
      *(_BYTE *)(v33[1] + 83LL) |= 0x40u;
      v33 = (_QWORD *)*v33;
    }
    while ( v33 );
  }
  v34 = *(_DWORD *)(a1 + 16);
  if ( v34 )
  {
    v34 = *(_DWORD *)(a1 + 108);
    if ( v34 )
    {
      v34 = 1;
      *(_DWORD *)(a1 + 168) = *(_DWORD *)(a1 + 172) - *(_DWORD *)(a1 + 224);
    }
    else if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0 )
    {
      v34 = 1;
      *(_DWORD *)(a1 + 168) = *(_DWORD *)(a1 + 172);
    }
  }
  if ( *(_DWORD *)(a1 + 24) && !*(_QWORD *)(a1 + 240) && *(_DWORD *)(a1 + 168) > *(_DWORD *)(a1 + 176) + 1 )
    *(_DWORD *)(a1 + 24) = 0;
  v19 = *(_QWORD *)(a1 + 192);
  v21 = a1;
  sub_8992E0(a1, (__int64 *)v19, v34, v394);
  if ( !*(_DWORD *)(a1 + 32) )
  {
    v20 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v20 )
      goto LABEL_65;
    goto LABEL_43;
  }
LABEL_61:
  if ( !dword_4F04C3C )
  {
    v19 = (unsigned int)v7;
    v21 = *(_QWORD *)(*(_QWORD *)(a1 + 336) + 96LL);
    v369 = *(_QWORD *)(a1 + 336);
    sub_869FD0((_QWORD *)v21, v7);
    *(_QWORD *)(v369 + 96) = 0;
  }
  *(_QWORD *)(a1 + 336) = 0;
  if ( *(_DWORD *)(a1 + 24) )
  {
LABEL_43:
    if ( (_DWORD)v7 == -1 )
      BUG();
    v35 = *(unsigned __int8 *)(qword_4F04C68[0] + 776 * v7 + 4);
    v20 = (unsigned int)(v35 - 3);
    if ( (unsigned __int8)(v35 - 3) > 1u && (_BYTE)v35 && ((_BYTE)v35 != 6 || !dword_4D04740) && !*(_DWORD *)(a1 + 52) )
    {
      v19 = v5 + 24;
      v21 = 790;
      sub_6851C0(0x316u, (_DWORD *)(v5 + 24));
      *(_DWORD *)(a1 + 52) = 1;
    }
    if ( *(_DWORD *)(a1 + 32) )
      goto LABEL_47;
    goto LABEL_65;
  }
  if ( *(_DWORD *)(a1 + 32) )
  {
LABEL_47:
    if ( v6 )
      sub_6851C0(0x42Du, v365);
    sub_7293C0(1, &v388, (_QWORD **)(v5 + 448));
    if ( *(_DWORD *)(a1 + 124) )
    {
      sub_7BDC00();
      *(_DWORD *)(a1 + 124) = 0;
    }
    sub_8B84E0(a1);
    goto LABEL_52;
  }
LABEL_65:
  if ( word_4F06418[0] != 295 )
  {
    v75 = *(_QWORD *)a1;
    v385 = 0;
    v386 = 0;
    v383 = 0;
    *(_BYTE *)(v75 + 122) |= 4u;
    v76 = dword_4F06650[0];
    sub_866880(v390.m128i_i64, v19);
    sub_5CCA00();
    if ( word_4F06418[0] == 153 )
      sub_7B8B50((unsigned __int64)&v390, (unsigned int *)v19, v77, v78, v79, v80);
    sub_5CCA00();
    v81 = 0;
    sub_88D9B0(0, 0, v82, v83, v84, v85);
    if ( (unsigned __int16)(word_4F06418[0] - 101) > 0x32u )
    {
      v81 = dword_4D0449C;
      if ( !dword_4D0449C || word_4F06418[0] != 87 )
        goto LABEL_101;
      sub_7B8B50(0, (unsigned int *)dword_4D0449C, 87, v86, v87, v88);
      if ( word_4F06418[0] == 151 || word_4F06418[0] == 101 )
        sub_7B8B50(0, (unsigned int *)v81, v246, v247, v248, v249);
      sub_5CCA00();
      if ( dword_4F077C4 == 2 )
      {
        if ( word_4F06418[0] != 1 || (v250 = &qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
        {
          v81 = 0;
          if ( !(unsigned int)sub_7C0F00(0x50401u, 0, (__int64)v250, v251, v252, v253) )
            goto LABEL_101;
        }
LABEL_428:
        v81 = 0;
        v254 = sub_7BE840(0, 0);
        if ( (v254 & 0xFFFD) == 0x49 || v254 == 55 )
          *(_DWORD *)(a1 + 120) = 1;
        goto LABEL_101;
      }
      if ( word_4F06418[0] == 1 )
        goto LABEL_428;
LABEL_101:
      v89 = v390.m128i_i64[0];
      sub_866920(v390.m128i_i64[0]);
      if ( v76 == dword_4F06650[0] )
        goto LABEL_134;
      LODWORD(v94) = 0;
      goto LABEL_103;
    }
    v111 = 0x4000000000009LL;
    if ( !_bittest64(&v111, (unsigned int)word_4F06418[0] - 101) )
      goto LABEL_101;
    sub_7B8B50(0, 0, 0x4000000000009LL, v86, v87, v88);
    sub_5CCA00();
    if ( dword_4F077C4 == 2 )
    {
      if ( word_4F06418[0] != 1 || (v112 = &qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
      {
        v81 = 0;
        if ( !(unsigned int)sub_7C0F00(0x50401u, 0, (__int64)v112, v113, v114, v115) )
        {
          v116 = word_4F06418[0];
          goto LABEL_117;
        }
      }
    }
    else
    {
      v116 = word_4F06418[0];
      if ( word_4F06418[0] != 1 )
        goto LABEL_117;
    }
    v81 = 0;
    v116 = sub_7BE840(0, 0);
    v394[0].m128i_i16[0] = v116;
    if ( (v116 & 0xFFFD) != 0x49
      && v116 != 55
      && dword_4F077C4 == 2
      && (unk_4F07778 > 201102 || dword_4F07774 || HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9EFBu) )
    {
      v81 = 73;
      sub_668C50(v394, 73, 1);
      v116 = v394[0].m128i_i16[0];
    }
LABEL_117:
    if ( (unsigned __int16)(v116 - 17) <= 0x3Au )
    {
      v89 = v390.m128i_i64[0];
      v94 = (0x500004000000001uLL >> ((unsigned __int8)v116 - 17)) & 1;
      sub_866920(v390.m128i_i64[0]);
      if ( v76 == dword_4F06650[0] )
      {
LABEL_104:
        if ( (_DWORD)v94 )
        {
          v81 = (__int64)&v385;
          v102 = a1;
          sub_8A6CC0((__m128i *)a1, (unsigned __int64 *)&v385, &v383, 0);
          v106 = v385;
          if ( v385 )
          {
            v366 = *(_DWORD *)(a1 + 36);
            switch ( *((_BYTE *)v385 + 80) )
            {
              case 4:
              case 5:
                v166 = *(_QWORD *)(v385[12] + 80);
                goto LABEL_221;
              case 6:
                v166 = *(_QWORD *)(v385[12] + 32);
                goto LABEL_221;
              case 9:
              case 0xA:
                v166 = *(_QWORD *)(v385[12] + 56);
                goto LABEL_221;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v166 = v385[11];
LABEL_221:
                v386 = v166;
                if ( !v366 )
                  goto LABEL_222;
                goto LABEL_219;
              default:
                v386 = 0;
                v371 = 0;
                if ( !v366 )
                  goto LABEL_153;
                v166 = 0;
LABEL_219:
                v371 = v166;
                v366 = 0;
                break;
            }
          }
          else
          {
            v386 = 0;
            v371 = 0;
            v366 = 0;
          }
          goto LABEL_153;
        }
LABEL_134:
        v117 = word_4F06418[0];
        if ( unk_4D0440C && word_4F06418[0] == 179 )
        {
          v354 = dword_4F06650[0];
          v91 = dword_4F077BC;
          if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
          {
            v89 = (__int64)&dword_4F063F8;
            v81 = 2889;
            sub_684B40(&dword_4F063F8, 0xB49u);
            v117 = word_4F06418[0];
          }
        }
        else
        {
          if ( word_4F06418[0] != 238 )
          {
            LODWORD(v94) = *(_DWORD *)(a1 + 120);
            if ( (_DWORD)v94 )
            {
              *(_QWORD *)(a1 + 376) = *(_QWORD *)&dword_4F063F8;
              v390.m128i_i64[0] = sub_5CC190(1);
              sub_6446A0(v390.m128i_i64, 0);
              sub_88D9B0(1u, (unsigned int *)1, v118, v119, v120, v121);
              if ( word_4F06418[0] == 153 )
              {
                sub_6851C0(0x115u, &dword_4F063F8);
                sub_7B8B50(0x115u, &dword_4F063F8, v257, v258, v259, v260);
                *(_DWORD *)(a1 + 52) = 1;
                v390.m128i_i64[0] = sub_5CC190(1);
                sub_6446A0(v390.m128i_i64, 0);
              }
              sub_88D9B0(1u, (unsigned int *)1, v122, v123, v124, v125);
              memset(v394, 0, 0x1D8u);
              v394[9].m128i_i64[1] = (__int64)v394;
              v394[1].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
              if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
                v394[11].m128i_i8[2] |= 1u;
              v102 = (__int64)v394;
              v81 = 512;
              sub_66F9E0((__int64)v394, 0x200u, 0, 1, v389[0].m128i_i64, 0, &v384, (unsigned int *)&v387, a1 + 344);
              v106 = *(__int64 **)v389[0].m128i_i64[0];
              if ( !*(_QWORD *)v389[0].m128i_i64[0] || (*((_BYTE *)v106 + 81) & 0x20) != 0 )
                goto LABEL_151;
              v126 = *(_QWORD *)(v389[0].m128i_i64[0] + 176);
              v127 = *(_QWORD *)(v106[12] + 32);
              if ( (_DWORD)v387 )
                *(_DWORD *)(a1 + 36) = 1;
              v370 = v126;
              sub_897580(a1, v106, v127);
              *(_QWORD *)(v370 + 24) = *(_QWORD *)(v127 + 104);
              if ( !dword_4F04C3C )
              {
                if ( (_DWORD)v387 )
                {
                  *(_BYTE *)(v389[0].m128i_i64[0] + 143) |= 8u;
                }
                else
                {
                  v314 = sub_8921F0(*(_QWORD *)(a1 + 336));
                  v314[3].m128i_i8[9] |= 1u;
                }
              }
              v81 = (__int64)v106;
              v102 = a1;
              if ( (unsigned int)sub_89BFC0(a1, (__int64)v106, 0, (FILE *)(v106 + 6)) )
              {
                v81 = a1 + 344;
                v102 = *(_QWORD *)(*(_QWORD *)(v127 + 176) + 88LL);
                sub_729470(v102, (const __m128i *)(a1 + 344));
                v207 = *((_BYTE *)v106 + 80);
                v385 = v106;
                switch ( v207 )
                {
                  case 4:
                  case 5:
                    v128 = *(_QWORD *)(v106[12] + 80);
                    break;
                  case 6:
                    v128 = *(_QWORD *)(v106[12] + 32);
                    break;
                  case 9:
                  case 10:
                    v128 = *(_QWORD *)(v106[12] + 56);
                    break;
                  case 19:
                  case 20:
                  case 21:
                  case 22:
                    v128 = v106[11];
                    break;
                  default:
                    v128 = 0;
                    break;
                }
              }
              else
              {
LABEL_151:
                v385 = 0;
                v106 = 0;
                v128 = 0;
              }
              v386 = v128;
              LODWORD(v94) = 0;
              v371 = v128;
              v366 = 0;
              goto LABEL_153;
            }
            v156 = sub_67B370(v394);
            if ( v394[0].m128i_i32[0]
              && unk_4F04C48 != -1
              && (v159 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v159 + 6) & 6) != 0) )
            {
              v158 = 0;
              if ( *(_DWORD *)(a1 + 16) == v394[0].m128i_i32[0] || !*(_DWORD *)(a1 + 20) )
              {
LABEL_202:
                *(_QWORD *)(v159 + 512) = v158;
                v366 = sub_651B00(2u);
                if ( v366 )
                  goto LABEL_313;
                if ( word_4F06418[0] == 1 )
                {
                  if ( dword_4F077C4 != 2
                    || (word_4D04A10 & 0x200) == 0
                    && (v81 = 0, !(unsigned int)sub_7C0F00(0, 0, (__int64)&qword_4D04A00, v160, v161, v162))
                    || (unk_4D04A12 & 1) == 0 )
                  {
LABEL_313:
                    if ( !*(_DWORD *)(a1 + 20) || (v366 = *(_DWORD *)(a1 + 16)) != 0 )
                    {
                      sub_87E3B0((__int64)v394);
                      v81 = 1;
                      sub_898140(
                        v75,
                        1u,
                        *(_DWORD *)(a1 + 20),
                        0,
                        *(_QWORD *)(a1 + 240),
                        *(_DWORD *)(a1 + 52),
                        *(_DWORD *)(a1 + 24),
                        &v390,
                        (unsigned __int64)v394,
                        0,
                        0,
                        (_QWORD *)(a1 + 344));
                      if ( (v391.m128i_i8[1] & 0x20) != 0 )
                      {
                        *(_DWORD *)(a1 + 52) = 1;
                      }
                      else
                      {
                        v202 = *(_DWORD *)(a1 + 52);
                        if ( (v391.m128i_i8[0] & 1) != 0 )
                        {
                          if ( !v202 )
                            goto LABEL_320;
                        }
                        else if ( !v202 )
                        {
                          if ( *(_DWORD *)(a1 + 16) || *(_QWORD *)(a1 + 224) <= 1u )
                          {
LABEL_320:
                            v102 = *(_QWORD *)(v75 + 288);
                            v366 = sub_8D2310(v102);
                            if ( v366 )
                            {
                              if ( (*(_BYTE *)(v75 + 125) & 0x10) != 0 )
                              {
                                v81 = (__int64)&v390;
                                v102 = a1;
                                v366 = 0;
                                v385 = sub_893A40(a1, &v390, (__int64)v394);
                                v371 = 0;
                                v386 = v385[11];
                                goto LABEL_345;
                              }
                              v203 = *(_QWORD *)a1;
                              v389[0].m128i_i64[0] = 0;
                              if ( word_4F06418[0] != 73 && word_4F06418[0] != 163 )
                              {
                                if ( word_4F06418[0] != 55 )
                                {
                                  if ( word_4F06418[0] == 56 && (unsigned int)sub_651030(&v387) )
                                  {
                                    v325 = v394[4].m128i_i8[0];
                                    v326 = v394[4].m128i_i8[0] | 4;
                                    v394[4].m128i_i8[0] |= 4u;
                                    if ( (_DWORD)v387 )
                                    {
                                      v394[4].m128i_i8[0] = v325 | 0xC;
                                      if ( dword_4F077BC
                                        && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
                                      {
                                        v327 = 2514;
LABEL_619:
                                        sub_684B40(&dword_4F063F8, v327);
                                        v204 = (v394[4].m128i_i8[0] & 4) != 0;
                                        goto LABEL_327;
                                      }
                                    }
                                    else
                                    {
                                      v394[4].m128i_i8[0] = v326 | 0x12;
                                      if ( dword_4F077BC
                                        && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
                                      {
                                        v327 = 2513;
                                        goto LABEL_619;
                                      }
                                    }
                                    v204 = 1;
LABEL_327:
                                    v205 = *(_QWORD *)(v203 + 288);
                                    *(_BYTE *)(v203 + 122) = *(_BYTE *)(v203 + 122) & 0xFE | v204;
                                    sub_8DCB20(v205);
                                    if ( *(_DWORD *)(a1 + 16) )
                                    {
                                      if ( (v394[4].m128i_i8[0] & 4) != 0 )
                                      {
                                        if ( *(_DWORD *)(a1 + 20) )
                                          v394[4].m128i_i8[0] |= 2u;
                                      }
                                      else if ( dword_4D04964 && (v394[4].m128i_i8[1] & 1) != 0 && *(_DWORD *)(a1 + 44) )
                                      {
                                        sub_684AA0(byte_4F07472[0], 0x595u, &v390.m128i_i32[2]);
                                      }
                                    }
                                    *(__m128i *)(a1 + 440) = v394[0];
                                    sub_64BAA0(&v390, (__int64)v394, v389[0].m128i_i64, a1);
                                    if ( !dword_4F04C3C )
                                    {
                                      if ( (v394[4].m128i_i8[0] & 4) == 0 )
                                      {
                                        v328 = sub_8921F0(*(_QWORD *)(a1 + 336));
                                        v328[2].m128i_i64[0] = v394[5].m128i_i64[0];
                                        v328[3].m128i_i8[9] = (4 * (*(_BYTE *)(a1 + 16) & 1))
                                                            | v328[3].m128i_i8[9] & 0xFB;
                                        if ( *(_BYTE *)(v203 + 268) )
                                          v328[3].m128i_i8[10] |= 1u;
                                      }
                                      *(_QWORD *)(v203 + 352) = *(_QWORD *)(*(_QWORD *)(a1 + 336) + 96LL);
                                      sub_65C210(v203);
                                    }
                                    v81 = v389[0].m128i_i64[0];
                                    if ( v389[0].m128i_i64[0] )
                                    {
                                      if ( *(_BYTE *)(v389[0].m128i_i64[0] + 80) == 10 && *(_DWORD *)(a1 + 24) )
                                        *(_DWORD *)(a1 + 24) = 0;
                                      if ( !*(_DWORD *)(a1 + 16) )
                                        goto LABEL_343;
                                      if ( (v394[4].m128i_i8[0] & 4) != 0 && *(_BYTE *)(v81 + 80) == 20 )
                                      {
                                        v324 = *(_QWORD *)(*(_QWORD *)(v81 + 88) + 176LL);
                                        if ( v324 )
                                        {
                                          if ( (*(_BYTE *)(v324 + 198) & 0x20) != 0 )
                                          {
                                            sub_684AA0(7u, 0xE3Bu, &v390.m128i_i32[2]);
                                            if ( !*(_DWORD *)(a1 + 16) )
                                              goto LABEL_342;
                                          }
                                        }
                                      }
                                    }
                                    else if ( !*(_DWORD *)(a1 + 16) )
                                    {
                                      goto LABEL_343;
                                    }
                                    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 8) == 0 )
                                    {
LABEL_342:
                                      v81 = v389[0].m128i_i64[0];
                                      goto LABEL_343;
                                    }
                                    *(_DWORD *)(a1 + 52) = 1;
                                    sub_6851C0(0x3D2u, &v390.m128i_i32[2]);
                                    v81 = v389[0].m128i_i64[0];
LABEL_343:
                                    v102 = a1;
                                    v385 = (__int64 *)v81;
                                    sub_89C1A0(a1, v81, v394, &v386, (FILE *)&v390.m128i_u64[1]);
                                    if ( *(_DWORD *)(a1 + 36) )
                                    {
                                      v366 = 0;
                                      v371 = v386;
LABEL_345:
                                      if ( dword_4F04C64 == -1
                                        || (v103 = qword_4F04C68,
                                            v206 = qword_4F04C68[0] + 776LL * dword_4F04C64,
                                            (*(_BYTE *)(v206 + 7) & 1) == 0)
                                        || dword_4F04C44 == -1 && (*(_BYTE *)(v206 + 6) & 2) == 0 )
                                      {
                                        if ( (v394[4].m128i_i8[1] & 8) == 0 )
                                        {
                                          v102 = (__int64)&v394[0].m128i_i64[1];
                                          sub_87E280((_QWORD **)&v394[0].m128i_i64[1]);
                                        }
                                      }
                                      v106 = v385;
                                      goto LABEL_153;
                                    }
                                    goto LABEL_443;
                                  }
LABEL_326:
                                  v204 = (v394[4].m128i_i8[0] & 4) != 0;
                                  goto LABEL_327;
                                }
                                if ( (*(_BYTE *)(v203 + 16) & 0x10) == 0 )
                                  goto LABEL_326;
                              }
                              v394[4].m128i_i8[0] |= 4u;
                              v204 = 1;
                              goto LABEL_327;
                            }
                            if ( v391.m128i_i64[1] )
                            {
                              v225 = *(_BYTE *)(v391.m128i_i64[1] + 80);
                              if ( ((v225 - 7) & 0xFD) == 0 || v225 == 21 )
                              {
LABEL_400:
                                v81 = (__int64)&v390;
                                v102 = a1;
                                v226 = sub_89D8A0(a1, (__int64)&v390, &v386);
                                v385 = (__int64 *)v226;
                                if ( v386 )
                                {
                                  v227 = *(_BYTE *)(v226 + 80);
                                  if ( v227 == 21 )
                                  {
                                    if ( (*(_BYTE *)(*(_QWORD *)(v386 + 192) + 176LL) & 1) != 0 )
                                      goto LABEL_443;
                                  }
                                  else if ( v227 == 9 && (*(_BYTE *)(v75 + 127) & 4) == 0 )
                                  {
                                    goto LABEL_443;
                                  }
                                  v371 = v386;
                                  goto LABEL_345;
                                }
LABEL_443:
                                v366 = 0;
                                v371 = 0;
                                goto LABEL_345;
                              }
LABEL_432:
                              v366 = 1;
                              v371 = 0;
                              if ( (v391.m128i_i8[1] & 0x20) == 0 )
                              {
                                v81 = (__int64)&v390.m128i_i64[1];
                                v102 = 457;
                                sub_6851A0(0x1C9u, &v390.m128i_i32[2], *(_QWORD *)(v390.m128i_i64[0] + 8));
                              }
                              goto LABEL_345;
                            }
                            v102 = dword_4D043C8;
                            if ( dword_4D043C8 )
                              goto LABEL_400;
                            if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
                              goto LABEL_432;
                            if ( dword_4F077BC )
                            {
                              if ( !(_DWORD)qword_4F077B4 )
                              {
                                if ( qword_4F077A8 <= 0xC34Fu )
                                  goto LABEL_432;
                                goto LABEL_502;
                              }
                            }
                            else if ( !(_DWORD)qword_4F077B4 )
                            {
                              goto LABEL_432;
                            }
                            if ( qword_4F077A0 <= 0x76BFu )
                              goto LABEL_432;
LABEL_502:
                            sub_684B30(0xD7Au, &dword_4F063F8);
                            goto LABEL_400;
                          }
                          v81 = (__int64)dword_4F07508;
                          sub_6851C0(0x309u, dword_4F07508);
                          *(_DWORD *)(a1 + 52) = 1;
                        }
                      }
                      v391.m128i_i8[1] |= 0x20u;
                      v391.m128i_i64[1] = 0;
                      goto LABEL_320;
                    }
                    v102 = a1;
                    v255 = sub_609BF0(a1);
                    v385 = (__int64 *)v255;
                    v106 = (__int64 *)v255;
                    if ( !v255 || (v256 = *(_BYTE *)(v255 + 80), v256 == 20) )
                    {
                      v81 = (__int64)v106;
                      v102 = a1;
                      sub_89C1A0(a1, (__int64)v106, 0, &v386, (FILE *)(a1 + 344));
                      v106 = v385;
                      v371 = v386;
                      if ( !v386 )
                        goto LABEL_450;
                      if ( *((_BYTE *)v385 + 80) != 21 )
                        goto LABEL_438;
                    }
                    else
                    {
                      if ( v256 != 21 )
                      {
                        v371 = v386;
                        if ( !v386 )
                          goto LABEL_450;
                        goto LABEL_438;
                      }
                      v371 = v106[11];
                      v386 = v371;
                      if ( !v371 )
                      {
                        LODWORD(v94) = 0;
                        goto LABEL_153;
                      }
                    }
                    if ( (*(_BYTE *)(*(_QWORD *)(v371 + 192) + 176LL) & 1) != 0 )
                      goto LABEL_450;
LABEL_438:
                    LODWORD(v94) = *(_DWORD *)(a1 + 36);
                    if ( !(_DWORD)v94 )
                    {
                      v366 = 0;
LABEL_222:
                      v371 = 0;
                      goto LABEL_153;
                    }
LABEL_450:
                    LODWORD(v94) = 0;
                    goto LABEL_153;
                  }
                }
                else if ( word_4F06418[0] == 34
                       || word_4F06418[0] == 27
                       || dword_4F077C4 == 2
                       && (word_4F06418[0] == 33
                        || dword_4D04474 && word_4F06418[0] == 52
                        || dword_4D0485C && word_4F06418[0] == 25
                        || word_4F06418[0] == 156) )
                {
                  goto LABEL_313;
                }
                v81 = (__int64)&dword_4F063F8;
                v102 = 169;
                LODWORD(v94) = 0;
                sub_6851C0(0xA9u, &dword_4F063F8);
                v106 = v385;
                v371 = 0;
                goto LABEL_153;
              }
              *(_DWORD *)(a1 + 16) = v394[0].m128i_i32[0];
            }
            else
            {
              if ( v394[0].m128i_i32[0] != *(_DWORD *)(a1 + 16) && *(_DWORD *)(a1 + 20) )
                *(_DWORD *)(a1 + 16) = v394[0].m128i_i32[0];
              if ( v156 )
              {
                while ( 1 )
                {
                  v157 = *(_BYTE *)(v156 + 140);
                  if ( v157 != 12 )
                    break;
                  v156 = *(_QWORD *)(v156 + 160);
                }
                if ( v157 == 14 )
                  v156 = sub_7D0530(v156);
                if ( (unsigned int)sub_8D3A70(v156) )
                {
                  v81 = (__int64)qword_4F04C68;
                  v158 = *(_QWORD *)v156;
                  v159 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                  goto LABEL_202;
                }
              }
            }
            v159 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            v158 = 0;
            goto LABEL_202;
          }
          v354 = dword_4F06650[0];
        }
        *(_QWORD *)(a1 + 376) = *(_QWORD *)&dword_4F063F8;
        *(_QWORD *)(a1 + 384) = qword_4F063F0;
        sub_7B8B50(v89, (unsigned int *)v81, v90, v91, v92, v93);
        if ( dword_4F077C4 == 2 )
        {
          if ( word_4F06418[0] == 1 )
          {
            v175 = &qword_4D04A00;
            if ( (word_4D04A10 & 0x200) != 0 )
            {
LABEL_266:
              if ( (word_4D04A10 & 0x58) != 0 )
              {
                v81 = (__int64)&qword_4D04A08;
                v89 = 502;
                sub_6851C0(0x1F6u, &qword_4D04A08);
              }
              else
              {
                if ( (unk_4D04A12 & 1) != 0 )
                {
                  v81 = (__int64)&dword_4F063F8;
                  v89 = 753;
                  sub_6851C0(0x2F1u, &dword_4F063F8);
                  v228 = _mm_loadu_si128(&xmmword_4F06660[1]);
                  v229 = _mm_loadu_si128(&xmmword_4F06660[2]);
                  v230 = _mm_loadu_si128(&xmmword_4F06660[3]);
                  v390 = _mm_loadu_si128(xmmword_4F06660);
                  v391 = v228;
                  v392 = v229;
                  v393 = v230;
                  goto LABEL_364;
                }
                if ( (word_4D04A10 & 0x2000) == 0 )
                {
                  v179 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
                  v180 = _mm_loadu_si128(&xmmword_4D04A20);
                  v181 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
                  v390 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
                  v391 = v179;
                  v392 = v180;
                  v393 = v181;
                  if ( (*(_BYTE *)(qword_4D04A00 + 73) & 2) != 0 && word_4F06418[0] == 1 )
                  {
                    v89 = qword_4D04A00;
                    if ( (*(_BYTE *)(qword_4D04A00 + 73) & 2) != 0 && sub_887690(qword_4D04A00) )
                    {
                      v318 = v390.m128i_i64[0];
                      v81 = (__int64)&dword_4F063F8;
                      v89 = 3316;
                      *(_BYTE *)(v390.m128i_i64[0] + 73) &= ~2u;
                      sub_684AE0(0xCF4u, &dword_4F063F8, *(_QWORD *)(v318 + 8));
                    }
                  }
                  if ( *(_DWORD *)(a1 + 52) )
                  {
                    v391.m128i_i8[1] |= 0x20u;
                    v182 = 0;
                    v391.m128i_i64[1] = 0;
                  }
                  else
                  {
                    v81 = 0;
                    v89 = (__int64)&v390;
                    v269 = dword_4F04C5C;
                    v270 = *(_DWORD *)(a1 + 200);
                    v271 = dword_4D03F98[0];
                    dword_4D03F98[0] = 0;
                    dword_4F04C5C = v270;
                    v363 = v269;
                    v368 = v271;
                    v272 = sub_7CFB70(&v390, 0);
                    v177 = v368;
                    v176 = v363;
                    v182 = v272;
                    dword_4D03F98[0] = v368;
                    dword_4F04C5C = v363;
                  }
                  v183 = *(_QWORD *)&dword_4F063F8;
                  v184 = qword_4F063F0;
                  *(_QWORD *)(a1 + 360) = *(_QWORD *)&dword_4F063F8;
                  *(_QWORD *)(a1 + 368) = v184;
                  *(_QWORD *)(a1 + 392) = v183;
                  *(_QWORD *)(a1 + 400) = v184;
                  sub_7B8B50(v89, (unsigned int *)v81, v183, v176, v177, v178);
LABEL_274:
                  v387 = (__int64 *)sub_5CC190(6);
                  if ( v387 )
                  {
                    sub_5CF700(v387);
                    *(_QWORD *)(a1 + 400) = *(_QWORD *)&dword_4F061D8;
                  }
                  v185 = (unsigned int *)(unsigned int)dword_4F0664C;
                  sub_8975E0((const __m128i *)a1, dword_4F0664C, 1);
                  if ( word_4F06418[0] == 56 )
                    sub_7B8B50(a1, v185, v186, v187, v188, v189);
                  else
                    sub_6851C0(0x2BEu, &dword_4F063F8);
                  if ( v117 == 238 )
                  {
                    v208 = &v387;
                    if ( v387 )
                      v208 = sub_5CB9F0(&v387);
                    *v208 = (__int64 *)sub_5CC190(1);
                  }
                  if ( v182 && *(_BYTE *)(v182 + 80) == 19 && (*(_BYTE *)(*(_QWORD *)(v182 + 88) + 265LL) & 1) != 0 )
                  {
                    *(_DWORD *)(a1 + 112) = 1;
                    v192 = sub_87EF90(0x13u, (__int64)&v390);
                    *((_DWORD *)v192 + 10) = *(_DWORD *)(v182 + 40);
                    *(_QWORD *)(a1 + 432) = v192;
                    v191 = (__m128i *)v192[11];
                    v191[16].m128i_i8[9] |= 1u;
                    v191[8].m128i_i64[0] = (__int64)v387;
                    v215 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | v191[10].m128i_i8[0] & 0xF7;
                    v191[10].m128i_i8[0] = v215;
                    v216 = (16 * (*(_BYTE *)(a1 + 88) & 1)) | v215 & 0xEF;
                    v191[10].m128i_i8[0] = v216;
                    v191[10].m128i_i8[0] = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v216 & 0xDF;
                    v360 = *(_QWORD *)(v182 + 88);
                    v367 = *(__int64 **)(a1 + 192);
                    sub_88FB80(*(_QWORD *)(*(_QWORD *)(v360 + 32) + 32LL), v367[4], (__int64)&v390, v182);
                    sub_88DD80(a1, (__int64)v192, v217, v218, v219, v220);
                    v372 = 1;
                  }
                  else
                  {
                    v190 = sub_885AD0(0x13u, (__int64)&v390, *(_DWORD *)(a1 + 204), 0);
                    *(_DWORD *)(a1 + 36) = 1;
                    v191 = (__m128i *)v190[11];
                    v192 = v190;
                    v193 = (__int64)v190;
                    v182 = (__int64)v190;
                    v191[16].m128i_i8[9] |= 1u;
                    v194 = v191[10].m128i_i8[0];
                    v191[8].m128i_i64[0] = (__int64)v387;
                    v195 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | v194 & 0xF7;
                    v191[10].m128i_i8[0] = v195;
                    v196 = 16 * (*(_BYTE *)(a1 + 88) & 1u);
                    LOBYTE(v190) = (16 * (*(_BYTE *)(a1 + 88) & 1)) | v195 & 0xEF;
                    v191[10].m128i_i8[0] = (char)v190;
                    v197 = 32 * (*(_BYTE *)(a1 + 128) & 1u);
                    v191[10].m128i_i8[0] = (32 * (*(_BYTE *)(a1 + 128) & 1)) | (unsigned __int8)v190 & 0xDF;
                    v360 = *(_QWORD *)(v182 + 88);
                    v367 = *(__int64 **)(a1 + 192);
                    sub_88DD80(a1, v193, v197, v196, v198, v199);
                    v372 = 0;
                    if ( (*((_BYTE *)v192 + 81) & 0x10) != 0 )
                    {
                      if ( !*(_DWORD *)(a1 + 44) )
                      {
                        v372 = *(_DWORD *)(a1 + 48);
                        if ( !v372 )
                        {
                          v300 = sub_8788F0(*(_QWORD *)v192[8]);
                          if ( v300 )
                          {
                            v301 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v300 + 88) + 168LL) + 152LL);
                            if ( v301 )
                            {
                              if ( (*(_BYTE *)(v301 + 29) & 0x20) == 0 )
                              {
                                v302 = sub_883800(*(_QWORD *)(v300 + 96) + 192LL, *v192);
                                if ( v302 )
                                {
                                  while ( 1 )
                                  {
                                    if ( *(_BYTE *)(v302 + 80) == 19 )
                                    {
                                      v303 = *(_QWORD *)(v302 + 88);
                                      if ( *(_DWORD *)(v303 + 64) == v354 )
                                        break;
                                    }
                                    v302 = *(_QWORD *)(v302 + 32);
                                    if ( !v302 )
                                      goto LABEL_518;
                                  }
                                  *(_QWORD *)(v192[11] + 88LL) = v302;
                                  v304 = sub_878440();
                                  v304[1] = v192;
                                  *v304 = *(_QWORD *)(v303 + 96);
                                  *(_QWORD *)(v303 + 96) = v304;
                                }
                              }
                            }
                          }
LABEL_518:
                          v305 = v191[5].m128i_i64[1];
                          if ( (word_4F06418[0] & 0xFFBF) == 9 || word_4F06418[0] == 75 )
                          {
                            v182 = (__int64)v192;
                            sub_897580(a1, v192, v360);
                            sub_879080(v191, 0, (__int64)v367);
                            v361 = dword_4F07590;
                            if ( !dword_4F07590 )
                              goto LABEL_288;
LABEL_286:
                            v361 = dword_4F04C3C;
                            dword_4F04C3C = 1;
LABEL_287:
                            if ( v372 )
                            {
                              sub_8756F0(1, v182, &v390.m128i_i64[1], 0);
                              goto LABEL_289;
                            }
LABEL_288:
                            sub_8756F0(3, v182, &v390.m128i_i64[1], 0);
LABEL_289:
                            if ( *(_DWORD *)(a1 + 112) )
                              sub_89BD20(*v367, a1, v182, (_DWORD *)v192 + 12, 1, 0, 1, 8u);
                            sub_88F9D0((__int64 *)*v367, 0);
                            if ( dword_4F07590 )
                              dword_4F04C3C = v361;
                            sub_896F00(a1, (__int64)v192, (__int64)v191, 0, 0);
                            v81 = a1 + 344;
                            sub_729470(*(_QWORD *)(v191[11].m128i_i64[0] + 88), (const __m128i *)(a1 + 344));
                            v201 = *(_BYTE *)(v182 + 80);
                            v385 = (__int64 *)v182;
                            switch ( v201 )
                            {
                              case 4:
                              case 5:
                                v102 = *(_QWORD *)(*(_QWORD *)(v182 + 96) + 80LL);
                                break;
                              case 6:
                                v102 = *(_QWORD *)(*(_QWORD *)(v182 + 96) + 32LL);
                                break;
                              case 9:
                              case 10:
                                v102 = *(_QWORD *)(*(_QWORD *)(v182 + 96) + 56LL);
                                break;
                              case 19:
                              case 20:
                              case 21:
                              case 22:
                                v102 = *(_QWORD *)(v182 + 88);
                                break;
                              default:
                                v102 = 0;
                                break;
                            }
                            v386 = v102;
                            LODWORD(v94) = 0;
                            v106 = v385;
                            v366 = 0;
                            v371 = sub_892400(v102);
LABEL_153:
                            *(_QWORD *)v75 = v106;
                            if ( *(_DWORD *)(a1 + 124) )
                            {
                              sub_7BDC00();
                              *(_DWORD *)(a1 + 124) = 0;
                            }
                            v129 = v385;
                            v130 = *(_QWORD *)(a1 + 216);
                            if ( v385 )
                            {
                              v129 = 0;
                              if ( (v385[10] & 0x10FF) == 0x14 )
                              {
                                v105 = (__int64 *)*(unsigned int *)(a1 + 320);
                                if ( (_DWORD)v105 )
                                {
                                  v103 = qword_4F04C68;
                                  v129 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 584);
                                }
                              }
                              if ( !v130 )
                              {
LABEL_160:
                                v131 = v386;
                                if ( v386 )
                                {
                                  for ( i = *(_QWORD **)(v386 + 56); ; i = (_QWORD *)*i )
                                  {
                                    if ( !i )
                                    {
                                      *(_QWORD *)(v386 + 56) = *(_QWORD *)(a1 + 328);
                                      goto LABEL_166;
                                    }
                                    if ( !*i )
                                      break;
                                  }
                                  *i = *(_QWORD *)(a1 + 328);
                                  v131 = v386;
LABEL_166:
                                  sub_897580(a1, v385, v131);
                                  v133 = (__int64)v385;
                                  sub_8911B0(a1, (__int64)v385, v134, v135, v136, v137);
                                  if ( (_DWORD)v94 )
                                  {
                                    v141 = *(unsigned int *)(a1 + 52);
                                    v142 = *(_DWORD *)(a1 + 36);
                                    if ( (_DWORD)v141 )
                                    {
                                      if ( v142 )
                                      {
                                        LODWORD(v94) = 0;
                                        v144 = 0;
                                        sub_65C470(v75, v133, v138, v139, v141);
                                        sub_643EB0(v75, 0);
                                        v149 = *(_QWORD *)(a1 + 192);
                                        if ( !v149 )
                                          goto LABEL_179;
                                        goto LABEL_175;
                                      }
                                    }
                                    else if ( v142 )
                                    {
                                      v143 = v385;
                                      if ( (*((_BYTE *)v385 + 81) & 0x20) != 0 )
                                      {
                                        LODWORD(v94) = *(_DWORD *)(a1 + 68);
                                        if ( (_DWORD)v94 )
                                        {
                                          LODWORD(v94) = 0;
                                          v144 = 0;
LABEL_172:
                                          v133 = (__int64)v143;
                                          v376 = (__int64)v143;
                                          if ( (unsigned int)sub_89BFC0(a1, (__int64)v143, 1, (FILE *)(v143 + 6)) )
                                          {
                                            v133 = v376;
                                            sub_8A9BD0((const __m128i *)a1, v376);
                                          }
LABEL_174:
                                          sub_65C470(v75, v133, v138, v139, v141);
                                          sub_643EB0(v75, 0);
                                          v149 = *(_QWORD *)(a1 + 192);
                                          if ( !v149 )
                                          {
LABEL_177:
                                            if ( (_DWORD)v94 )
                                              v149 = sub_88FD30(v386, v144, 1);
                                            else
                                              v149 = v144;
LABEL_179:
                                            v150 = v371;
                                            sub_898C50(v386, (char *)v371, (__int64)v145, v146, v147, v148);
                                            if ( v149 )
                                            {
                                              v150 = v149;
                                              sub_88FD30(v386, v149, 0);
                                            }
                                            goto LABEL_181;
                                          }
LABEL_175:
                                          if ( !*(_DWORD *)(v149 + 44) )
                                          {
                                            v145 = &dword_4F066AC;
                                            *(_DWORD *)(v149 + 44) = ++dword_4F066AC;
                                          }
                                          goto LABEL_177;
                                        }
                                        v144 = 0;
                                        sub_65C470(v75, v133, v138, v139, v141);
                                        sub_643EB0(v75, 0);
                                        v149 = *(_QWORD *)(a1 + 192);
                                        if ( v149 )
                                          goto LABEL_175;
                                        goto LABEL_236;
                                      }
                                      v357 = *(_QWORD *)(*(_QWORD *)(v386 + 176) + 88LL);
                                      *(_QWORD *)(*(_QWORD *)(v357 + 168) + 160LL) = *(_QWORD *)(v386 + 104);
                                      *(_DWORD *)(*(_QWORD *)(a1 + 192) + 48LL) = ++dword_4F066AC;
                                      switch ( *((_BYTE *)v143 + 80) )
                                      {
                                        case 4:
                                        case 5:
                                          v362 = *(_QWORD *)(v143[12] + 80);
                                          break;
                                        case 6:
                                          v362 = *(_QWORD *)(v143[12] + 32);
                                          break;
                                        case 9:
                                        case 0xA:
                                          v362 = *(_QWORD *)(v143[12] + 56);
                                          break;
                                        case 0x13:
                                        case 0x14:
                                        case 0x15:
                                        case 0x16:
                                          v362 = v143[11];
                                          break;
                                        default:
                                          BUG();
                                      }
                                      v345 = (__int64)v143;
                                      v231 = *(_QWORD *)(v362 + 176);
                                      *(_BYTE *)(v231 + 81) |= 2u;
                                      v352 = *(_QWORD *)(v231 + 96);
                                      v347 = (*(_BYTE *)(v357 + 89) & 4) != 0;
                                      *(_BYTE *)(v357 + 140) = *(_BYTE *)(v362 + 264);
                                      v232 = sub_892330(v357);
                                      *(_BYTE *)(v352 + 180) |= 8u;
                                      v337 = v345;
                                      *(_QWORD *)(v352 + 120) = *(_QWORD *)(qword_4F04C68[0]
                                                                          + 776LL * dword_4F04C34
                                                                          + 224);
                                      v346 = sub_864700(
                                               *(_QWORD *)(v362 + 32),
                                               v357,
                                               0,
                                               v231,
                                               v345,
                                               v232,
                                               1,
                                               *(_DWORD *)(a1 + 24) == 0 ? 2 : 4194306);
                                      sub_854C10(*(const __m128i **)(v362 + 56));
                                      sub_7BC160(v362);
                                      v233 = v337;
                                      if ( *(_QWORD *)(v362 + 128) )
                                      {
                                        v331 = v337;
                                        v234 = *(_QWORD *)(v357 + 168);
                                        v235 = *(_BYTE *)(v234 + 109);
                                        v332 = v234;
                                        *(_BYTE *)(v234 + 109) = v235 & 0xF8;
                                        v338 = v235 & 7;
                                        sub_66A990(*(__m128i **)(v362 + 128), v357, *(_QWORD *)a1, 1, 0, 0);
                                        v233 = v331;
                                        v236 = *(_BYTE *)(v332 + 109);
                                        if ( (v236 & 7) == 0 )
                                          *(_BYTE *)(v332 + 109) = v338 | v236 & 0xF8;
                                      }
                                      v339 = v233;
                                      ++*(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720);
                                      sub_607B60(
                                        (__int64 *)v357,
                                        *(_QWORD *)a1,
                                        dword_4F04C34,
                                        0,
                                        0,
                                        v347,
                                        1u,
                                        0,
                                        *(_QWORD *)(a1 + 336),
                                        a1 + 344);
                                      v237 = *(__m128i **)(v357 + 72);
                                      if ( v237 )
                                      {
                                        *v237 = _mm_loadu_si128((const __m128i *)(a1 + 360));
                                        v237[1] = _mm_loadu_si128((const __m128i *)(a1 + 376));
                                        v237[2] = _mm_loadu_si128((const __m128i *)(a1 + 392));
                                      }
                                      v238 = sub_87D550(v339);
                                      v239 = v231;
                                      v133 = 0;
                                      v348 = *(_BYTE *)(v357 + 88);
                                      *(_BYTE *)(v357 + 143) |= 8u;
                                      *(_BYTE *)(v357 + 88) = v238 & 3 | v348 & 0xFC;
                                      --*(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720);
                                      sub_854980(v231, 0);
                                      v243 = v346;
                                      v144 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 584);
                                      if ( v346 )
                                        sub_863FE0(v239, 0, v240, v346, v241, v242);
                                      *(_BYTE *)(v352 + 180) &= ~8u;
                                      while ( word_4F06418[0] != 9 )
                                        sub_7B8B50(v239, 0, v240, v243, v241, (__int64)v242);
                                      sub_7B8B50(v239, 0, v240, v243, v241, (__int64)v242);
                                      v139 = v357;
                                      *(_BYTE *)(v362 + 265) |= 2u;
                                      v244 = v385;
                                      *(_QWORD *)(v357 + 64) = v385[6];
                                      v138 = v383;
                                      if ( v383 )
                                      {
                                        v245 = *((_BYTE *)v244 + 80);
                                        v138 = (unsigned __int8)(v245 - 4);
                                        switch ( v245 )
                                        {
                                          case 4:
                                          case 5:
                                            v139 = *(_QWORD *)(v244[12] + 80);
                                            break;
                                          case 6:
                                            v139 = *(_QWORD *)(v244[12] + 32);
                                            break;
                                          case 9:
                                          case 10:
                                            v139 = *(_QWORD *)(v244[12] + 56);
                                            break;
                                          case 19:
                                          case 20:
                                          case 21:
                                          case 22:
                                            v139 = v244[11];
                                            break;
                                          default:
                                            BUG();
                                        }
                                        if ( *(_QWORD *)(v139 + 168) )
                                        {
                                          v382 = v75;
                                          v308 = v139;
                                          v353 = v94;
                                          v309 = *(_QWORD **)(v139 + 168);
                                          v364 = v144;
                                          do
                                          {
                                            v138 = v309[1];
                                            if ( v138 != *(_QWORD *)(v308 + 176) )
                                            {
                                              v310 = *(_QWORD *)(v138 + 88);
                                              if ( (*(_BYTE *)(v310 + 178) & 1) == 0 )
                                              {
                                                v133 = *(unsigned __int8 *)(v310 + 140);
                                                v139 = *(unsigned __int8 *)(v357 + 140);
                                                v311 = (_BYTE)v133 == (unsigned __int8)v139;
                                                LOBYTE(v133) = (_BYTE)v133 == 11;
                                                if ( !((unsigned __int8)v133 | v311) && (_BYTE)v139 != 11 )
                                                  *(_BYTE *)(v310 + 140) = v139;
                                                v312 = *(__int64 **)(*(_QWORD *)(v138 + 96) + 128LL);
                                                if ( v312 )
                                                {
                                                  while ( *((_BYTE *)v312 + 16) != 2 )
                                                  {
                                                    v312 = (__int64 *)*v312;
                                                    if ( !v312 )
                                                      goto LABEL_530;
                                                  }
                                                  v313 = *(_QWORD *)(v138 + 88);
                                                  if ( (unsigned int)sub_8D23B0(v313) && (unsigned int)sub_8D3A70(v313) )
                                                  {
                                                    v133 = 0;
                                                    sub_8AD220(v313, 0);
                                                  }
                                                }
                                              }
                                            }
LABEL_530:
                                            v309 = (_QWORD *)*v309;
                                          }
                                          while ( v309 );
                                          v75 = v382;
                                          v144 = v364;
                                          LODWORD(v94) = v353;
                                        }
                                      }
LABEL_306:
                                      if ( !*(_DWORD *)(a1 + 68) )
                                        goto LABEL_174;
                                      v143 = v385;
                                      if ( *(_DWORD *)(a1 + 52) )
                                        goto LABEL_174;
                                      goto LABEL_172;
                                    }
                                    LODWORD(v94) = dword_4F04C3C;
                                    v144 = 0;
                                    if ( dword_4F04C3C )
                                    {
                                      LODWORD(v94) = 0;
                                    }
                                    else
                                    {
                                      v138 = (__int64)sub_8921F0(*(_QWORD *)(a1 + 336));
                                      v139 = *(_BYTE *)(v138 + 57) & 0xFA;
                                      *(_BYTE *)(v138 + 57) = *(_BYTE *)(v138 + 57) & 0xFA
                                                            | (4 * (*(_BYTE *)(a1 + 16) & 1) + 1);
                                    }
                                    goto LABEL_306;
                                  }
                                  v167 = v385;
                                  if ( v385 )
                                  {
                                    if ( *(_DWORD *)(a1 + 96) )
                                    {
                                      *(_BYTE *)(v386 + 265) |= 2u;
                                      goto LABEL_229;
                                    }
                                    v359 = *((_BYTE *)v385 + 80);
                                    if ( v359 == 19 )
                                    {
                                      if ( !*(_DWORD *)(a1 + 52) )
                                      {
                                        if ( *(_DWORD *)(a1 + 112) )
                                        {
                                          v263 = *(_QWORD *)(a1 + 432);
                                          switch ( *(_BYTE *)(v263 + 80) )
                                          {
                                            case 4:
                                            case 5:
                                              v264 = *(_QWORD *)(*(_QWORD *)(v263 + 96) + 80LL);
                                              break;
                                            case 6:
                                              v264 = *(_QWORD *)(*(_QWORD *)(v263 + 96) + 32LL);
                                              break;
                                            case 9:
                                            case 0xA:
                                              v264 = *(_QWORD *)(*(_QWORD *)(v263 + 96) + 56LL);
                                              break;
                                            case 0x13:
                                            case 0x14:
                                            case 0x15:
                                            case 0x16:
                                              v264 = *(_QWORD *)(v263 + 88);
                                              break;
                                            default:
                                              v264 = 0;
                                              break;
                                          }
                                          v265 = sub_892400(v264);
                                          v133 = *(_QWORD *)(a1 + 432);
                                          v371 = v265;
                                          sub_892430(a1, v133);
                                          v266 = v385;
                                          v267 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 432) + 88LL)
                                                                       + 176LL)
                                                           + 88LL);
                                          v268 = *(_QWORD *)(v267 + 160);
                                          if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v385[11] + 176) + 88LL) + 160LL) != v268 )
                                          {
                                            v133 = *(_QWORD *)(v267 + 160);
                                            v377 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v385[11] + 176) + 88LL) + 160LL);
                                            if ( !(unsigned int)sub_8D97D0(v377, v133, 0, v139, v140) )
                                            {
                                              v133 = 2638;
                                              sub_687290(
                                                8u,
                                                0xA4Eu,
                                                (_DWORD *)(*(_QWORD *)(a1 + 432) + 48LL),
                                                (__int64)v266,
                                                v268,
                                                v377);
                                            }
                                          }
                                        }
                                        else
                                        {
                                          v133 = (__int64)v385;
                                          sub_892430(a1, (__int64)v385);
                                        }
                                      }
                                    }
                                    else
                                    {
                                      if ( v359 == 6 )
                                        goto LABEL_229;
                                      v351 = v359 == 21;
                                      if ( v359 != 9 && v359 != 21 )
                                      {
                                        if ( v359 <= 0x14u )
                                        {
                                          v174 = 1182720;
                                          if ( _bittest64(&v174, v359) )
                                          {
                                            if ( (unsigned int)sub_893570((__int64)v385) )
                                            {
                                              v133 = 0;
                                              sub_643EB0(v75, 0);
                                              if ( !*(_DWORD *)(a1 + 52)
                                                || (*(_BYTE *)(v386 + 160) |= 4u, !*(_DWORD *)(a1 + 52)) )
                                              {
                                                if ( *(_DWORD *)(a1 + 36) && !dword_4D047AC && !*(_QWORD *)(a1 + 240) )
                                                  sub_8950B0((__int64)v385);
                                              }
                                            }
                                          }
                                        }
                                        goto LABEL_229;
                                      }
                                      v344 = *(_DWORD *)(a1 + 52);
                                      if ( v344 )
                                        goto LABEL_229;
                                      v221 = *(_QWORD *)a1;
                                      v394[0].m128i_i32[0] = 0;
                                      switch ( v359 )
                                      {
                                        case 9u:
                                          v356 = *(_QWORD *)(v385[12] + 56);
                                          v222 = *(_QWORD *)(v356 + 192);
                                          goto LABEL_458;
                                        case 0x15u:
                                          v356 = v385[11];
                                          v222 = *(_QWORD *)(v356 + 192);
                                          if ( v359 == 21 )
                                          {
                                            v139 = v385[11];
                                            v223 = *(__int64 **)(v356 + 88);
                                            if ( v223 )
                                            {
                                              if ( (*(_BYTE *)(v356 + 160) & 1) == 0 && v385 != v223 )
                                                goto LABEL_387;
                                            }
                                          }
LABEL_458:
                                          if ( (*(_BYTE *)(v222 + 176) & 1) != 0 && !*(_QWORD *)(a1 + 240) )
                                            goto LABEL_387;
                                          v349 = *(_QWORD *)v222;
                                          *(_QWORD *)v221 = *(_QWORD *)v222;
                                          v273 = *(_QWORD *)(v221 + 280);
                                          if ( v273 )
                                            *(_QWORD *)(v222 + 256) = v273;
                                          *(_BYTE *)(v222 + 137) = *(_BYTE *)(v221 + 268);
                                          v274 = *(_BYTE *)(v221 + 269);
                                          if ( v274 != 1 )
                                          {
                                            if ( v274 == 2
                                              || dword_4F077C4 == 2
                                              && (unk_4F07778 > 201102 || dword_4F07774)
                                              && (v139 = (__int64)qword_4F04C68,
                                                  (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 8) & 2) != 0) )
                                            {
                                              v317 = *(_BYTE *)(v222 + 88);
                                              *(_BYTE *)(v222 + 168) &= 0xF8u;
                                              *(_BYTE *)(v222 + 136) = 2;
                                              *(_BYTE *)(v222 + 88) = v317 & 0x8F | 0x10;
                                            }
                                            else
                                            {
                                              v275 = *(_BYTE *)(v222 + 88);
                                              *(_BYTE *)(v222 + 136) = 0;
                                              *(_BYTE *)(v222 + 88) = v275 & 0x8F | 0x20;
                                            }
                                          }
                                          *(_BYTE *)(v222 + 170) |= 0x10u;
                                          v140 = dword_4D047B0;
                                          if ( !dword_4D047B0 && (*(_BYTE *)(v356 + 160) & 8) == 0 )
                                            goto LABEL_387;
                                          if ( v359 == 21 )
                                          {
                                            if ( dword_4F077C4 != 2 )
                                              goto LABEL_568;
                                          }
                                          else
                                          {
                                            v276 = (unsigned __int64)v167;
                                            v340 = v167;
                                            v333 = v167[12];
                                            v277 = sub_8807C0((__int64)v167);
                                            v280 = v333;
                                            v167 = v340;
                                            v39 = dword_4F077C4 == 2;
                                            *(_QWORD *)(v333 + 48) = v277;
                                            if ( !v39 )
                                              goto LABEL_470;
                                          }
                                          v335 = v167;
                                          v343 = *(_QWORD *)(v222 + 120);
                                          v319 = sub_8D23B0(v343);
                                          v276 = v343;
                                          v167 = v335;
                                          if ( v319 )
                                          {
                                            sub_8AE000(v343);
                                            v167 = v335;
                                          }
LABEL_470:
                                          v281 = qword_4F04C68;
                                          v341 = v351
                                               | ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >> 1)
                                                ^ 1)
                                               & 1;
                                          if ( !v341 )
                                            goto LABEL_471;
LABEL_568:
                                          v334 = (__int64)v167;
                                          v342 = sub_892400(v356);
                                          v315 = sub_892350(v222);
                                          v316 = sub_864700(
                                                   *(_QWORD *)(v342 + 32),
                                                   0,
                                                   0,
                                                   v349,
                                                   v334,
                                                   v315,
                                                   1,
                                                   (*(_BYTE *)(v356 + 160) & 1) == 0 ? 2 : 4194306);
                                          v133 = v329;
                                          v276 = v330;
                                          v167 = (__int64 *)v334;
                                          v341 = v316;
LABEL_471:
                                          if ( *(_QWORD *)(v356 + 8) )
                                          {
                                            v350 = *(_BYTE *)(v221 + 124) >> 7;
                                            if ( (*(_BYTE *)(v222 + 176) & 1) != 0 )
                                            {
                                              v282 = *(_QWORD *)(v222 + 120);
                                              if ( (*(_BYTE *)(v282 + 140) & 0xFB) == 8 )
                                              {
                                                v378 = v167;
                                                v133 = dword_4F077C4 != 2;
                                                v283 = sub_8D4C10(v282, v133);
                                                v167 = v378;
                                                v344 = v283 & 1;
                                              }
                                            }
                                            v284 = v356;
                                            v379 = v167;
                                            sub_7BC160(v356);
                                            v288 = v379;
                                            v289 = word_4F06418[0];
                                            *(_QWORD *)(v221 + 288) = *(_QWORD *)(v222 + 120);
                                            v358 = v289 == 27;
                                            if ( v289 == 73 )
                                            {
                                              *(_BYTE *)(v221 + 127) |= 8u;
                                              v291 = *(_BYTE *)(v222 + 176);
                                            }
                                            else
                                            {
                                              v290 = (8 * v358) | *(_BYTE *)(v221 + 127) & 0xF7u;
                                              *(_BYTE *)(v221 + 127) = (8 * v358) | *(_BYTE *)(v221 + 127) & 0xF7;
                                              if ( v359 == 21 && (v291 = *(_BYTE *)(v222 + 176), (v291 & 1) != 0) )
                                              {
                                                LOBYTE(v290) = v290 & 8;
                                                if ( (_BYTE)v290 )
                                                {
LABEL_482:
                                                  if ( (*(_BYTE *)(v222 + 172) & 0x20) == 0 )
                                                  {
                                                    v336 = v288;
                                                    v381 = sub_5F2750(
                                                             *(_QWORD *)(v222 + 120),
                                                             v222,
                                                             v344,
                                                             v351,
                                                             (*(_BYTE *)(v222 + 170) & 0x20) != 0);
                                                    if ( !v381 )
                                                    {
                                                      v133 = *(_QWORD *)(v222 + 120);
                                                      v276 = v221;
                                                      *(_QWORD *)(v222 + 120) = sub_5F2840(
                                                                                  v221,
                                                                                  v133,
                                                                                  (__int64)&dword_4F063F8);
                                                      sub_6BBA30(v221);
LABEL_485:
                                                      if ( word_4F06418[0] != 9 )
                                                      {
                                                        v133 = (__int64)&dword_4F063F8;
                                                        v276 = 65;
                                                        sub_6851C0(0x41u, &dword_4F063F8);
                                                        while ( word_4F06418[0] != 9 )
                                                          sub_7B8B50(0x41u, &dword_4F063F8, v293, v294, v295, v296);
                                                      }
                                                      sub_7B8B50(v276, (unsigned int *)v133, v293, v294, v295, v296);
                                                      goto LABEL_489;
                                                    }
                                                    v288 = v336;
                                                    if ( (*(_BYTE *)(v222 + 176) & 1) != 0
                                                      && (*(_BYTE *)(v222 + 172) & 0x20) == 0 )
                                                    {
                                                      v133 = v222;
                                                      sub_5F2700(v221, v222, v320, v321, v322, v323);
                                                      goto LABEL_484;
                                                    }
                                                  }
LABEL_483:
                                                  *(_BYTE *)(v221 + 124) &= ~0x80u;
                                                  v133 = (__int64)(v288 + 6);
                                                  sub_638AC0(v221, v288 + 6, 2u, v358, v394, 0);
                                                  *(_BYTE *)(v221 + 124) = (v350 << 7) | *(_BYTE *)(v221 + 124) & 0x7F;
LABEL_484:
                                                  v276 = v221;
                                                  sub_649FB0(v221, v133);
                                                  v381 = 1;
                                                  goto LABEL_485;
                                                }
                                                v292 = word_4F06418;
                                                if ( word_4F06418[0] == 56 )
                                                  goto LABEL_480;
                                              }
                                              else
                                              {
                                                sub_7B8B50(v284, (unsigned int *)v133, v290, v285, v286, v287);
                                                v288 = v379;
                                                if ( (*(_BYTE *)(v221 + 127) & 8) == 0 && word_4F06418[0] == 56 )
                                                {
                                                  if ( (*(_BYTE *)(v222 + 176) & 1) == 0 )
                                                    goto LABEL_483;
LABEL_480:
                                                  v380 = v288;
                                                  sub_7B8B50(
                                                    v284,
                                                    (unsigned int *)v133,
                                                    v290,
                                                    (__int64)v292,
                                                    v286,
                                                    v287);
                                                  v291 = *(_BYTE *)(v222 + 176);
                                                  v288 = v380;
                                                  goto LABEL_481;
                                                }
                                                v291 = *(_BYTE *)(v222 + 176);
                                              }
                                            }
LABEL_481:
                                            if ( (v291 & 1) == 0 )
                                              goto LABEL_483;
                                            goto LABEL_482;
                                          }
                                          if ( *(_BYTE *)(v222 + 177) || *(_QWORD *)(a1 + 240) )
                                            goto LABEL_552;
                                          if ( (*(_BYTE *)(v221 + 10) & 0x20) != 0 )
                                            *(_BYTE *)(v222 + 172) |= 0x10u;
                                          v276 = v349;
                                          v133 = v349 + 48;
                                          v381 = sub_63BB10(v349, v349 + 48);
                                          if ( v381 )
                                          {
LABEL_552:
                                            v381 = 0;
                                          }
                                          else
                                          {
                                            if ( v359 == 21 && (*(_BYTE *)(v221 + 10) & 8) != 0 )
                                              goto LABEL_580;
                                            v133 = *(_QWORD *)(v222 + 120);
                                            v276 = v349;
                                            sub_640330(v349, v133, 0, 0);
                                          }
LABEL_489:
                                          if ( v359 != 21 )
                                            goto LABEL_490;
LABEL_580:
                                          if ( !v394[0].m128i_i32[0]
                                            && (unsigned int)sub_8B1260(
                                                               *(_QWORD *)(v222 + 120),
                                                               *(_BYTE *)(v222 + 136),
                                                               v221,
                                                               1) )
                                          {
                                            *(_QWORD *)(v222 + 120) = sub_72C930();
                                          }
                                          v133 = v221;
                                          v276 = v222;
                                          sub_6581B0(v222, v221, v381);
LABEL_490:
                                          if ( v341 )
                                            sub_863FE0(v276, v133, v280, (__int64)v281, v278, v279);
                                          sub_8CB9C0(v222);
LABEL_387:
                                          v138 = *(unsigned int *)(a1 + 68);
                                          if ( (_DWORD)v138 )
                                          {
                                            if ( !*(_DWORD *)(a1 + 52) )
                                            {
                                              v224 = v385;
                                              v133 = (__int64)v385;
                                              if ( (unsigned int)sub_89BFC0(a1, (__int64)v385, 1, (FILE *)(v385 + 6)) )
                                              {
                                                v133 = (__int64)v224;
                                                sub_8A9BD0((const __m128i *)a1, (__int64)v224);
                                              }
                                            }
                                          }
                                          break;
                                      }
                                    }
                                  }
LABEL_229:
                                  sub_65C470(v75, v133, v138, v139, v140);
                                  sub_643EB0(v75, 0);
                                  v168 = *(_QWORD *)(a1 + 192);
                                  if ( v168 )
                                  {
                                    v148 = *(unsigned int *)(v168 + 44);
                                    if ( !(_DWORD)v148 )
                                    {
                                      v150 = v371;
                                      v169 = v386;
                                      v170 = dword_4F066AC + 1;
                                      *(_DWORD *)(v168 + 44) = v170;
                                      dword_4F066AC = v170;
                                      sub_898C50(v169, (char *)v371, v170, (__int64)&dword_4F066AC, v147, v148);
LABEL_181:
                                      if ( v129 )
                                      {
                                        v150 = (__int64)v129;
                                        sub_88FD30(v386 + 296, (__int64)v129, 0);
                                      }
                                      v151 = 0;
                                      sub_5F94C0(0);
                                      if ( v385 )
                                      {
                                        if ( *((_BYTE *)v385 + 80) == 20 )
                                        {
                                          if ( v386 )
                                          {
                                            v173 = *(_QWORD *)(v386 + 176);
                                            if ( (*(_BYTE *)(v173 + 198) & 0x10) != 0
                                              && (*(_BYTE *)(v173 + 197) & 2) == 0 )
                                            {
                                              v151 = *(_QWORD *)(v173 + 152);
                                              sub_8E3700(v151);
                                            }
                                          }
                                        }
                                      }
                                      if ( v366 )
                                      {
                                        v171 = qword_4F061C8;
                                        ++*(_BYTE *)(qword_4F061C8 + 83LL);
                                        ++*(_BYTE *)(v171 + 82);
                                        sub_7BE180(v151, v150, v152, v153, v154, v155);
                                        v172 = qword_4F061C8;
                                        --*(_BYTE *)(qword_4F061C8 + 83LL);
                                        --*(_BYTE *)(v172 + 82);
                                      }
                                      *(_QWORD *)(a1 + 8) = v385;
LABEL_52:
                                      if ( *(_DWORD *)(a1 + 320) )
                                        goto LABEL_53;
LABEL_85:
                                      sub_7AEA70((const __m128i *)(a1 + 288));
                                      goto LABEL_53;
                                    }
                                  }
LABEL_236:
                                  v150 = v371;
                                  sub_898C50(v386, (char *)v371, (__int64)v145, v146, v147, v148);
                                  goto LABEL_181;
                                }
LABEL_233:
                                sub_854000(*(__m128i **)(a1 + 328));
                                v131 = v386;
                                goto LABEL_166;
                              }
                            }
                            else if ( !v130 )
                            {
                              goto LABEL_233;
                            }
                            do
                            {
                              sub_863FC0(v102, v81, v130, (__int64)v103, v104, v105);
                              v39 = (*(_QWORD *)(a1 + 216))-- == 1;
                            }
                            while ( !v39 );
                            if ( !v385 )
                              goto LABEL_233;
                            goto LABEL_160;
                          }
                          sub_7ADF70((__int64)v389, 1);
                          memset(v394, 0, 358);
                          v394[4].m128i_i8[11] = 1;
                          sub_7C6880((unsigned __int64)v389, (__int64)v394, v306, 0, (__int64)v389, v307);
                          if ( v305 )
                          {
                            v182 = (__int64)v192;
                            sub_7AEA70(v389);
LABEL_284:
                            v200 = 0;
                            goto LABEL_285;
                          }
                          v182 = (__int64)v192;
LABEL_441:
                          sub_7AE210((__int64)v389);
                          v200 = v389;
LABEL_285:
                          v355 = v200;
                          sub_897580(a1, (__int64 *)v182, v360);
                          sub_879080(v191, v355, (__int64)v367);
                          v361 = dword_4F07590;
                          if ( !dword_4F07590 )
                            goto LABEL_287;
                          goto LABEL_286;
                        }
                      }
                      v182 = (__int64)v192;
                      v372 = 0;
                      v191[4].m128i_i32[0] = v354;
                    }
                  }
                  if ( (word_4F06418[0] & 0xFFBF) == 9 || word_4F06418[0] == 75 )
                    goto LABEL_284;
                  sub_7ADF70((__int64)v389, 1);
                  memset(v394, 0, 358);
                  v394[4].m128i_i8[11] = 1;
                  sub_7C6880((unsigned __int64)v389, (__int64)v394, v261, 0, (__int64)v389, v262);
                  goto LABEL_441;
                }
              }
              v209 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v210 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v211 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v390 = _mm_loadu_si128(xmmword_4F06660);
              v391 = v209;
              v392 = v210;
              v393 = v211;
LABEL_364:
              v182 = 0;
              v391.m128i_i8[1] |= 0x20u;
              v390.m128i_i64[1] = *(_QWORD *)dword_4F07508;
              sub_7B8B50(v89, (unsigned int *)v81, (__int64)v175, v176, v177, v178);
              goto LABEL_274;
            }
          }
          v81 = 0;
          v89 = 20;
          if ( (unsigned int)sub_7C0F00(0x14u, 0, (__int64)v175, v176, v177, v178) )
          {
LABEL_265:
            v175 = &qword_4D04A00;
            goto LABEL_266;
          }
        }
        else if ( word_4F06418[0] == 1 )
        {
          goto LABEL_265;
        }
        v182 = 0;
        sub_6851C0(0x28u, &dword_4F063F8);
        v212 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v213 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v214 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v390.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v391 = v212;
        v392 = v213;
        v391.m128i_i8[1] = v212.m128i_i8[1] | 0x20;
        v390.m128i_i64[1] = *(_QWORD *)dword_4F07508;
        v393 = v214;
        goto LABEL_274;
      }
LABEL_103:
      sub_7ADF70((__int64)v394, 0);
      sub_7B8B50((unsigned __int64)v394, 0, v95, v96, v97, v98);
      v81 = v76;
      sub_7AE700((__int64)(qword_4F061C0 + 3), v76, dword_4F06650[0], 0, (__int64)v394);
      v89 = (__int64)v394;
      sub_7BC000(v89, v76, v99, v100, v89, v101);
      goto LABEL_104;
    }
    goto LABEL_101;
  }
  v40 = *(unsigned int *)(a1 + 124);
  if ( (_DWORD)v40 )
  {
    sub_7BDC00();
    *(_DWORD *)(a1 + 124) = 0;
  }
  v390.m128i_i64[0] = *(_QWORD *)&dword_4F063F8;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  sub_7B8B50(v21, (unsigned int *)v19, v20, v16, v40, v15);
  ++*(_BYTE *)(qword_4F061C8 + 64LL);
  v41 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
  v42 = _mm_loadu_si128(&xmmword_4D04A20);
  v43 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  v394[0] = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  v394[1] = v41;
  v394[2] = v42;
  v394[3] = v43;
  *(_QWORD *)(a1 + 360) = *(_QWORD *)&dword_4F063F8;
  v44 = qword_4F063F0;
  *(_QWORD *)(a1 + 368) = qword_4F063F0;
  *(_QWORD *)(a1 + 384) = v44;
  if ( !(unsigned int)sub_7BE280(1u, 40, 0, 0, v45, v46) )
    goto LABEL_69;
  if ( (v394[1].m128i_i8[0] & 0x58) != 0 )
  {
    v163 = 502;
    goto LABEL_209;
  }
  if ( (v394[1].m128i_i8[0] & 1) != 0 )
  {
    v163 = 283;
LABEL_209:
    sub_6851C0(v163, &v394[0].m128i_i32[2]);
LABEL_69:
    v47 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v48 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v49 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v394[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    v394[0].m128i_i64[0] = v47;
    v394[1].m128i_i8[1] |= 0x20u;
    v394[2] = v48;
    v394[0].m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v394[3] = v49;
    goto LABEL_70;
  }
  v107 = dword_4F04C5C;
  v108 = *(_DWORD *)(a1 + 200);
  v109 = dword_4D03F98[0];
  dword_4D03F98[0] = 0;
  dword_4F04C5C = v108;
  v110 = sub_7CFB70(v394, 0);
  if ( v110 && *(_DWORD *)(v110 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
  {
    sub_6854C0(0xBE6u, (FILE *)&v394[0].m128i_u64[1], v110);
    v297 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v298 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v299 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v394[0].m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v394[1] = v297;
    v394[2] = v298;
    v394[1].m128i_i8[1] = v297.m128i_i8[1] | 0x20;
    v394[0].m128i_i64[1] = *(_QWORD *)dword_4F07508;
    v394[3] = v299;
  }
  dword_4D03F98[0] = v109;
  dword_4F04C5C = v107;
LABEL_70:
  v50 = 702;
  v51 = 56;
  sub_7BE5B0(0x38u, 0x2BEu, 0, 0);
  if ( word_4F06418[0] == 56 )
    sub_7B8B50(0x38u, (unsigned int *)0x2BE, v52, v53, v54, v55);
  --*(_BYTE *)(qword_4F061C8 + 64LL);
  v56 = sub_6D6A30();
  v60 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 83LL);
  if ( (v394[1].m128i_i8[1] & 0x20) == 0 )
  {
    v61 = *(_BYTE *)(qword_4F04C68[0] + 776LL * *(int *)(a1 + 200) + 4);
    if ( (unsigned __int8)(v61 - 3) > 1u && v61 )
    {
      v50 = (__int64)&v390;
      v51 = 3045;
      sub_6851C0(0xBE5u, &v390);
    }
    else
    {
      v62 = *(_QWORD **)(a1 + 336);
      v63 = *(_DWORD *)(a1 + 204);
      v62[24] = v56;
      v64 = sub_885AD0(0x16u, (__int64)v394, v63, 0);
      v65 = qword_4F04C68[0] + 776LL * v63;
      if ( *(_BYTE *)(v65 + 4) )
        sub_877E90((__int64)v64, 0, *(_QWORD *)(*(_QWORD *)(v65 + 184) + 32LL));
      v66 = v64[11];
      *(_QWORD *)(v66 + 32) = *(_QWORD *)(a1 + 192);
      v67 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | *(_BYTE *)(v66 + 160) & 0xF7;
      *(_BYTE *)(v66 + 160) = v67;
      v68 = (16 * (*(_BYTE *)(a1 + 88) & 1)) | v67 & 0xEF;
      *(_BYTE *)(v66 + 160) = v68;
      v69 = *(_BYTE *)(a1 + 128);
      *(_QWORD *)(v66 + 104) = v62;
      *(_BYTE *)(v66 + 160) = (32 * (v69 & 1)) | v68 & 0xDF;
      sub_897580(a1, v64, v66);
      v50 = (__int64)v64;
      v51 = a1;
      *(_QWORD *)(*(_QWORD *)(v66 + 104) + 200LL) = *(_QWORD *)(v66 + 104);
      sub_8911B0(a1, (__int64)v64, v70, v71, v72, v73);
      v60 = v62[22];
      if ( v60 )
      {
        v74 = *(_QWORD *)(v60 + 16);
        v50 = v74 + 8;
        if ( v74 )
        {
LABEL_79:
          v51 = 3102;
          sub_6851C0(0xC1Eu, (_DWORD *)v50);
          goto LABEL_80;
        }
        if ( (*(_BYTE *)(*(_QWORD *)(*v62 + 88LL) + 160LL) & 0x20) != 0 )
        {
          v164 = *(_QWORD *)(v60 + 8);
          if ( v164 )
          {
            while ( 1 )
            {
              if ( *(_BYTE *)(v164 + 120) == 1 )
              {
                v165 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v164 + 128) + 168LL) + 32LL);
                if ( v165 )
                  break;
              }
              v164 = *(_QWORD *)(v164 + 112);
              if ( !v164 )
                goto LABEL_444;
            }
            v50 = v165 + 28;
          }
          else
          {
LABEL_444:
            v50 = (__int64)&v390;
          }
          goto LABEL_79;
        }
      }
    }
  }
LABEL_80:
  if ( *(_QWORD *)(a1 + 216) )
  {
    do
    {
      sub_863FC0(v51, v50, v60, v57, v58, v59);
      v39 = (*(_QWORD *)(a1 + 216))-- == 1;
    }
    while ( !v39 );
  }
  if ( !*(_DWORD *)(a1 + 320) )
    sub_7AEA70((const __m128i *)(a1 + 288));
  sub_7AEA70(v374);
  if ( !*(_DWORD *)(a1 + 320) )
    goto LABEL_85;
LABEL_53:
  sub_7AEA70(v374);
  qword_4D03B88 = v373;
  return &qword_4D03B88;
}
