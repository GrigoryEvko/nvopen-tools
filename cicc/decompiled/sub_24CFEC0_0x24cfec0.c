// Function: sub_24CFEC0
// Address: 0x24cfec0
//
_QWORD *__fastcall sub_24CFEC0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  const char *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  _QWORD *v10; // rdx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r15
  unsigned int v17; // eax
  __int64 v18; // r9
  __int64 v19; // r8
  char v20; // al
  __int64 *v21; // r15
  __int64 v22; // r12
  char v23; // al
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  int v29; // ecx
  char v30; // dl
  int v31; // edx
  char v32; // bl
  _QWORD *v33; // rdi
  __m128i *v34; // rax
  __m128i si128; // xmm0
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // r13
  int v41; // eax
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  int v46; // edi
  char v47; // si
  int v48; // esi
  __int64 v49; // r13
  int v50; // ecx
  __int64 v51; // rax
  unsigned __int64 v52; // r11
  __int64 v53; // rcx
  __int64 v54; // rax
  int v55; // esi
  char v56; // dl
  int v57; // eax
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rcx
  int v64; // edi
  char v65; // si
  int v66; // eax
  __int64 v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // r12
  __int64 v71; // r14
  char v72; // dh
  char v73; // al
  char v74; // bl
  __int16 v75; // ax
  __int64 v76; // rdi
  unsigned int *v77; // rsi
  __int64 v78; // r9
  unsigned int *v79; // r8
  unsigned int *v80; // rcx
  int v81; // edi
  unsigned int *v82; // rdx
  unsigned int *v83; // rax
  char v84; // bl
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // r8
  unsigned __int64 v89; // rsi
  __int64 v90; // r9
  __int64 v91; // rcx
  __int64 v92; // rax
  unsigned int **v93; // r12
  __int64 v94; // r15
  __int64 v95; // rdi
  __int64 v96; // rax
  char v97; // dh
  __int64 v98; // rsi
  __int16 v99; // cx
  __int64 v100; // r8
  char v101; // al
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned __int64 v104; // rsi
  char v105; // r12
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 v111; // rcx
  __int64 v112; // rax
  unsigned int **v113; // r13
  int v114; // r8d
  __int64 *v115; // rbx
  __int64 v116; // r12
  __int64 v117; // rdx
  int v118; // eax
  __int64 **v119; // r13
  __int64 v120; // rdx
  unsigned int v121; // r15d
  unsigned int v122; // eax
  __int64 v123; // rax
  int v124; // edx
  unsigned __int64 *v125; // rax
  __int64 v126; // rax
  __int64 v127; // r8
  __int64 v128; // r9
  __int64 v129; // rax
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  __int64 v132; // rax
  unsigned __int64 v133; // r10
  __int64 v134; // r11
  __int64 v135; // rax
  int v136; // ecx
  char v137; // dl
  int v138; // eax
  unsigned __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __int64 v141; // rcx
  int v142; // esi
  char v143; // al
  int v144; // eax
  unsigned __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  unsigned __int64 v148; // rdx
  __int64 v149; // rax
  __int64 v150; // rax
  __int64 *v151; // r15
  int *v152; // r13
  __int64 v153; // rax
  __int64 v154; // rdi
  char v155; // bl
  __int64 v156; // r12
  int v157; // r8d
  __int64 v158; // rdi
  char v159; // al
  __int64 v160; // rax
  char v161; // r9
  unsigned __int64 v162; // rax
  char v163; // al
  __int16 v164; // r10
  __int64 v165; // rax
  __int64 v166; // rdx
  unsigned int v167; // eax
  __int64 v168; // rax
  unsigned __int64 v169; // rsi
  unsigned __int64 v170; // rdx
  __int64 v171; // rax
  __int64 v172; // rax
  unsigned __int64 *v173; // rax
  int v174; // r9d
  unsigned __int64 v175; // rsi
  _QWORD *v176; // rax
  __int64 v177; // rdx
  __int64 v178; // rax
  char v179; // al
  __int64 *v180; // r10
  unsigned __int64 v181; // rbx
  int v182; // eax
  __int64 v183; // rax
  unsigned __int8 *v184; // r10
  __int64 (__fastcall *v185)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v186; // rax
  int v187; // r8d
  int v188; // edi
  __int64 v189; // rax
  unsigned __int64 v190; // r15
  unsigned int v191; // r13d
  unsigned int v192; // eax
  unsigned __int64 v193; // rax
  __int64 **v194; // r15
  unsigned int v195; // eax
  unsigned __int64 v196; // r9
  __int64 v197; // rax
  int v198; // edi
  int v199; // esi
  _QWORD *v200; // rax
  __int64 v201; // rdx
  unsigned int *v202; // rbx
  unsigned __int64 v203; // rax
  unsigned int *v204; // r13
  __int64 v205; // rdx
  unsigned int v206; // esi
  __int64 **v207; // rax
  unsigned __int64 v208; // rax
  unsigned int *v209; // rcx
  __int64 v210; // [rsp+0h] [rbp-EE0h]
  __int64 v211; // [rsp+0h] [rbp-EE0h]
  __int64 v212; // [rsp+8h] [rbp-ED8h]
  __int64 v213; // [rsp+8h] [rbp-ED8h]
  unsigned __int64 v214; // [rsp+10h] [rbp-ED0h]
  __int64 **v215; // [rsp+10h] [rbp-ED0h]
  _QWORD *v216; // [rsp+10h] [rbp-ED0h]
  unsigned __int64 v217; // [rsp+10h] [rbp-ED0h]
  int v218; // [rsp+18h] [rbp-EC8h]
  char v219; // [rsp+18h] [rbp-EC8h]
  __int64 *v220; // [rsp+18h] [rbp-EC8h]
  _QWORD *v221; // [rsp+20h] [rbp-EC0h]
  char v222; // [rsp+20h] [rbp-EC0h]
  int *v223; // [rsp+20h] [rbp-EC0h]
  unsigned __int8 v224; // [rsp+4Fh] [rbp-E91h]
  char v225; // [rsp+50h] [rbp-E90h]
  int v226; // [rsp+50h] [rbp-E90h]
  unsigned __int8 v227; // [rsp+50h] [rbp-E90h]
  unsigned __int8 *v228; // [rsp+50h] [rbp-E90h]
  unsigned __int8 *v229; // [rsp+50h] [rbp-E90h]
  _QWORD *v230; // [rsp+50h] [rbp-E90h]
  __int64 v231; // [rsp+50h] [rbp-E90h]
  unsigned __int8 *v232; // [rsp+50h] [rbp-E90h]
  int v233; // [rsp+50h] [rbp-E90h]
  __int64 v234; // [rsp+58h] [rbp-E88h]
  unsigned __int64 v235; // [rsp+58h] [rbp-E88h]
  unsigned __int64 v236; // [rsp+58h] [rbp-E88h]
  __int64 **v237; // [rsp+58h] [rbp-E88h]
  _BYTE *v238; // [rsp+58h] [rbp-E88h]
  unsigned __int64 v239; // [rsp+58h] [rbp-E88h]
  unsigned __int64 v240; // [rsp+58h] [rbp-E88h]
  int v241; // [rsp+60h] [rbp-E80h]
  int v242; // [rsp+60h] [rbp-E80h]
  __int64 v243; // [rsp+60h] [rbp-E80h]
  __int64 v244; // [rsp+60h] [rbp-E80h]
  unsigned __int64 v245; // [rsp+60h] [rbp-E80h]
  _QWORD *v246; // [rsp+68h] [rbp-E78h]
  __int64 *v247; // [rsp+68h] [rbp-E78h]
  char v248; // [rsp+70h] [rbp-E70h]
  __int64 *v249; // [rsp+70h] [rbp-E70h]
  int v250; // [rsp+70h] [rbp-E70h]
  int v251; // [rsp+70h] [rbp-E70h]
  int v252; // [rsp+70h] [rbp-E70h]
  unsigned __int64 v253; // [rsp+70h] [rbp-E70h]
  char v254; // [rsp+78h] [rbp-E68h]
  unsigned __int64 v255; // [rsp+78h] [rbp-E68h]
  __int64 v256; // [rsp+78h] [rbp-E68h]
  __int64 v257; // [rsp+78h] [rbp-E68h]
  _BYTE *v258; // [rsp+78h] [rbp-E68h]
  _BYTE *v259; // [rsp+78h] [rbp-E68h]
  __int64 v260; // [rsp+78h] [rbp-E68h]
  char v261; // [rsp+80h] [rbp-E60h]
  __int64 v262; // [rsp+88h] [rbp-E58h]
  unsigned __int64 v263; // [rsp+88h] [rbp-E58h]
  _QWORD *v264; // [rsp+90h] [rbp-E50h]
  unsigned __int64 v265; // [rsp+90h] [rbp-E50h]
  __int64 **v266; // [rsp+90h] [rbp-E50h]
  unsigned int *v267; // [rsp+90h] [rbp-E50h]
  __int64 *v268; // [rsp+98h] [rbp-E48h]
  _QWORD *v270; // [rsp+A8h] [rbp-E38h]
  __int64 v271; // [rsp+C8h] [rbp-E18h] BYREF
  _QWORD v272[4]; // [rsp+D0h] [rbp-E10h] BYREF
  __int64 v273; // [rsp+F0h] [rbp-DF0h] BYREF
  unsigned __int64 v274; // [rsp+F8h] [rbp-DE8h]
  __int64 v275; // [rsp+100h] [rbp-DE0h]
  __int64 v276; // [rsp+108h] [rbp-DD8h]
  __int64 v277; // [rsp+110h] [rbp-DD0h]
  _BYTE *v278; // [rsp+120h] [rbp-DC0h] BYREF
  __int64 v279; // [rsp+128h] [rbp-DB8h]
  _BYTE v280[64]; // [rsp+130h] [rbp-DB0h] BYREF
  __int64 *v281; // [rsp+170h] [rbp-D70h] BYREF
  __int64 v282; // [rsp+178h] [rbp-D68h]
  _BYTE v283[64]; // [rsp+180h] [rbp-D60h] BYREF
  __int64 *v284; // [rsp+1C0h] [rbp-D20h] BYREF
  __int64 v285; // [rsp+1C8h] [rbp-D18h]
  _BYTE v286[64]; // [rsp+1D0h] [rbp-D10h] BYREF
  __int64 *v287; // [rsp+210h] [rbp-CD0h] BYREF
  __int64 v288; // [rsp+218h] [rbp-CC8h]
  _BYTE v289[128]; // [rsp+220h] [rbp-CC0h] BYREF
  unsigned int *v290; // [rsp+2A0h] [rbp-C40h] BYREF
  __int64 v291; // [rsp+2A8h] [rbp-C38h]
  _BYTE v292[16]; // [rsp+2B0h] [rbp-C30h] BYREF
  __int16 v293; // [rsp+2C0h] [rbp-C20h]
  __int64 v294; // [rsp+2D0h] [rbp-C10h]
  __int64 v295; // [rsp+2D8h] [rbp-C08h]
  __int16 v296; // [rsp+2E0h] [rbp-C00h]
  _QWORD *v297; // [rsp+2E8h] [rbp-BF8h]
  void **v298; // [rsp+2F0h] [rbp-BF0h]
  void **v299; // [rsp+2F8h] [rbp-BE8h]
  __int64 v300; // [rsp+300h] [rbp-BE0h]
  int v301; // [rsp+308h] [rbp-BD8h]
  __int16 v302; // [rsp+30Ch] [rbp-BD4h]
  char v303; // [rsp+30Eh] [rbp-BD2h]
  __int64 v304; // [rsp+310h] [rbp-BD0h]
  __int64 v305; // [rsp+318h] [rbp-BC8h]
  void *v306; // [rsp+320h] [rbp-BC0h] BYREF
  void *v307; // [rsp+328h] [rbp-BB8h] BYREF
  unsigned int *v308; // [rsp+330h] [rbp-BB0h] BYREF
  const char *v309; // [rsp+338h] [rbp-BA8h]
  __int64 v310; // [rsp+340h] [rbp-BA0h] BYREF
  __int64 v311; // [rsp+348h] [rbp-B98h]
  char *v312; // [rsp+350h] [rbp-B90h]
  __int64 v313; // [rsp+358h] [rbp-B88h]
  char v314; // [rsp+360h] [rbp-B80h] BYREF
  __int64 v315; // [rsp+368h] [rbp-B78h]
  __int64 v316; // [rsp+370h] [rbp-B70h]
  __int64 *v317; // [rsp+378h] [rbp-B68h]
  __int64 v318; // [rsp+380h] [rbp-B60h]
  __int64 v319; // [rsp+388h] [rbp-B58h]
  __int16 v320; // [rsp+390h] [rbp-B50h]
  __int64 v321; // [rsp+398h] [rbp-B48h]
  void **v322; // [rsp+3A0h] [rbp-B40h]
  void **v323; // [rsp+3A8h] [rbp-B38h]
  void *v324; // [rsp+3B0h] [rbp-B30h]
  int v325; // [rsp+3B8h] [rbp-B28h] BYREF
  __int16 v326; // [rsp+3BCh] [rbp-B24h]
  char v327; // [rsp+3BEh] [rbp-B22h]
  __int64 v328; // [rsp+3C0h] [rbp-B20h]
  __int64 v329; // [rsp+3C8h] [rbp-B18h]
  void *v330; // [rsp+3D0h] [rbp-B10h] BYREF
  void *v331; // [rsp+3D8h] [rbp-B08h] BYREF
  char v332; // [rsp+3E0h] [rbp-B00h]
  char v333; // [rsp+3E1h] [rbp-AFFh]
  __int64 v334; // [rsp+3E8h] [rbp-AF8h]
  __int64 **v335; // [rsp+3F0h] [rbp-AF0h] BYREF
  unsigned __int64 v336; // [rsp+3F8h] [rbp-AE8h]
  __int64 v337; // [rsp+400h] [rbp-AE0h]
  unsigned __int64 v338; // [rsp+408h] [rbp-AD8h]
  __int64 v339; // [rsp+410h] [rbp-AD0h]
  unsigned __int64 v340; // [rsp+418h] [rbp-AC8h]
  __int64 v341; // [rsp+420h] [rbp-AC0h]
  unsigned __int64 v342; // [rsp+428h] [rbp-AB8h]
  __int64 v343; // [rsp+430h] [rbp-AB0h]
  _OWORD v344[5]; // [rsp+438h] [rbp-AA8h] BYREF
  _OWORD v345[5]; // [rsp+488h] [rbp-A58h] BYREF
  _OWORD v346[5]; // [rsp+4D8h] [rbp-A08h] BYREF
  _OWORD v347[5]; // [rsp+528h] [rbp-9B8h] BYREF
  _OWORD v348[5]; // [rsp+578h] [rbp-968h] BYREF
  _OWORD v349[5]; // [rsp+5C8h] [rbp-918h] BYREF
  _OWORD v350[5]; // [rsp+618h] [rbp-8C8h] BYREF
  _OWORD v351[125]; // [rsp+668h] [rbp-878h] BYREF
  _QWORD v352[2]; // [rsp+E38h] [rbp-A8h] BYREF
  _QWORD v353[2]; // [rsp+E48h] [rbp-98h] BYREF
  unsigned __int64 v354; // [rsp+E58h] [rbp-88h]
  __int64 v355; // [rsp+E60h] [rbp-80h]
  unsigned __int64 v356; // [rsp+E68h] [rbp-78h]
  __int64 v357; // [rsp+E70h] [rbp-70h]
  _QWORD v358[2]; // [rsp+E78h] [rbp-68h] BYREF
  _QWORD v359[2]; // [rsp+E88h] [rbp-58h] BYREF
  unsigned __int64 v360; // [rsp+E98h] [rbp-48h]
  __int64 v361; // [rsp+EA0h] [rbp-40h]

  v336 = 0;
  v337 = 0;
  v338 = 0;
  v339 = 0;
  v340 = 0;
  v341 = 0;
  v342 = 0;
  v343 = 0;
  memset(v344, 0, sizeof(v344));
  memset(v345, 0, sizeof(v345));
  memset(v346, 0, sizeof(v346));
  memset(v347, 0, sizeof(v347));
  memset(v348, 0, sizeof(v348));
  memset(v349, 0, sizeof(v349));
  memset(v350, 0, sizeof(v350));
  memset(v351, 0, sizeof(v351));
  v352[0] = 0;
  v352[1] = 0;
  v353[0] = 0;
  v353[1] = 0;
  v354 = 0;
  v355 = 0;
  v356 = 0;
  v357 = 0;
  v358[0] = 0;
  v358[1] = 0;
  v359[0] = 0;
  v359[1] = 0;
  v360 = 0;
  v361 = 0;
  if ( (_BYTE)qword_4FEDE48 && byte_4FEDD68 )
  {
    v33 = sub_CB72A0();
    v34 = (__m128i *)v33[4];
    if ( v33[3] - (_QWORD)v34 <= 0x6Eu )
    {
      sub_CB6200(
        (__int64)v33,
        "warning: Option -tsan-compound-read-before-write has no effect when -tsan-instrument-read-before-write is set.\n",
        0x6Fu);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4387F60);
      qmemcpy(&v34[6], "-write is set.\n", 15);
      *v34 = si128;
      v34[1] = _mm_load_si128((const __m128i *)&xmmword_4387F70);
      v34[2] = _mm_load_si128((const __m128i *)&xmmword_4387F80);
      v34[3] = _mm_load_si128((const __m128i *)&xmmword_4387F90);
      v34[4] = _mm_load_si128((const __m128i *)&xmmword_4387FA0);
      v34[5] = _mm_load_si128((const __m128i *)&xmmword_4387FB0);
      v33[4] += 111LL;
    }
  }
  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v7 = sub_BD5D20(a3);
  if ( v8 == 16 && !(*(_QWORD *)v7 ^ 0x646F6D2E6E617374LL | *((_QWORD *)v7 + 1) ^ 0x726F74635F656C75LL)
    || (unsigned __int8)sub_B2D610(a3, 20)
    || (v248 = sub_B2D610(a3, 10)) != 0 )
  {
    v9 = a1 + 4;
    v10 = a1 + 10;
    goto LABEL_6;
  }
  sub_24CCBB0((__int64 *)&v335, *(__int64 **)(a3 + 40), v6 + 8);
  v278 = v280;
  v287 = (__int64 *)v289;
  v281 = (__int64 *)v283;
  v284 = (__int64 *)v286;
  v288 = 0x800000000LL;
  v279 = 0x800000000LL;
  v282 = 0x800000000LL;
  v285 = 0x800000000LL;
  v224 = sub_B2D610(a3, 63);
  v12 = sub_B2BEC0(a3);
  v13 = *(_QWORD *)(a3 + 80);
  v261 = 0;
  v262 = v12;
  if ( v13 != a3 + 72 )
  {
    v264 = a1;
    while ( 1 )
    {
      if ( !v13 )
        BUG();
      v14 = *(_QWORD *)(v13 + 32);
      v15 = v13 + 24;
      if ( v14 != v13 + 24 )
        break;
LABEL_23:
      sub_24CF840((__int64 *)&v278, (__int64)&v287);
      v13 = *(_QWORD *)(v13 + 8);
      if ( a3 + 72 == v13 )
      {
        a1 = v264;
        goto LABEL_25;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v14 )
          BUG();
        v16 = v14 - 24;
        if ( (*(_BYTE *)(v14 - 17) & 0x20) != 0 && sub_B91C10(v14 - 24, 31) )
          goto LABEL_15;
        LOBYTE(v17) = sub_B46500((unsigned __int8 *)(v14 - 24));
        v19 = v17;
        v20 = *(_BYTE *)(v14 - 24);
        if ( (_BYTE)v19 )
        {
          if ( v20 != 61 && v20 != 62 )
          {
            if ( v20 != 64 && v20 != 65 && v20 != 66 )
              BUG();
LABEL_74:
            v38 = (unsigned int)v282;
            v39 = (unsigned int)v282 + 1LL;
            if ( v39 > HIDWORD(v282) )
            {
              sub_C8D5F0((__int64)&v281, v283, v39, 8u, v19, v18);
              v38 = (unsigned int)v282;
            }
            v281[v38] = v16;
            LODWORD(v282) = v282 + 1;
            goto LABEL_15;
          }
          if ( *(_BYTE *)(v14 + 48) )
            goto LABEL_74;
LABEL_67:
          v36 = (unsigned int)v279;
          v37 = (unsigned int)v279 + 1LL;
          if ( v37 > HIDWORD(v279) )
          {
            sub_C8D5F0((__int64)&v278, v280, v37, 8u, v19, v18);
            v36 = (unsigned int)v279;
          }
          *(_QWORD *)&v278[8 * v36] = v16;
          LODWORD(v279) = v279 + 1;
          goto LABEL_15;
        }
        if ( (unsigned __int8)(v20 - 61) <= 1u )
          goto LABEL_67;
        if ( v20 != 85 )
          break;
        v126 = *(_QWORD *)(v14 - 56);
        if ( !v126
          || *(_BYTE *)v126
          || *(_QWORD *)(v126 + 24) != *(_QWORD *)(v14 + 56)
          || (*(_BYTE *)(v126 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v126 + 36) - 68) > 3 )
        {
          sub_F58670(v14 - 24, (__int64 *)(v6 + 8));
          if ( *(_BYTE *)(v14 - 24) == 85 )
          {
            v129 = *(_QWORD *)(v14 - 56);
            if ( v129 )
            {
              if ( !*(_BYTE *)v129
                && *(_QWORD *)(v129 + 24) == *(_QWORD *)(v14 + 56)
                && (*(_BYTE *)(v129 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v129 + 36) - 238) <= 7
                && ((1LL << (*(_BYTE *)(v129 + 36) + 18)) & 0xAD) != 0 )
              {
                v130 = (unsigned int)v285;
                v131 = (unsigned int)v285 + 1LL;
                if ( v131 > HIDWORD(v285) )
                {
                  sub_C8D5F0((__int64)&v284, v286, v131, 8u, v127, v128);
                  v130 = (unsigned int)v285;
                }
                v284[v130] = v16;
                LODWORD(v285) = v285 + 1;
              }
            }
          }
          goto LABEL_22;
        }
LABEL_15:
        v14 = *(_QWORD *)(v14 + 8);
        if ( v15 == v14 )
          goto LABEL_23;
      }
      if ( v20 != 34 )
        goto LABEL_15;
LABEL_22:
      sub_24CF840((__int64 *)&v278, (__int64)&v287);
      v14 = *(_QWORD *)(v14 + 8);
      v261 = 1;
      if ( v15 == v14 )
        goto LABEL_23;
    }
  }
LABEL_25:
  v254 = byte_4FEE388 & v224;
  if ( ((unsigned __int8)byte_4FEE388 & v224) != 0 )
  {
    v247 = &v287[2 * (unsigned int)v288];
    if ( v287 == v247 )
    {
      v254 = 0;
      goto LABEL_26;
    }
    v151 = v287;
    v216 = a1;
    v152 = &v325;
    while ( 1 )
    {
      sub_23E3770((__int64)&v308, *v151);
      v153 = *v151;
      v154 = *(_QWORD *)(*v151 - 32);
      v155 = *(_BYTE *)*v151;
      v272[0] = v154;
      if ( *(_BYTE *)v153 != 61 )
        v153 = *(_QWORD *)(v153 - 64);
      v156 = *(_QWORD *)(v153 + 8);
      if ( (unsigned __int8)sub_BD6020(v154) )
        goto LABEL_239;
      v157 = sub_24CC3F0(v156, v262);
      if ( v157 < 0 )
        goto LABEL_239;
      v158 = *v151;
      v159 = *(_BYTE *)(*v151 + 7) & 0x20;
      if ( v155 == 62 )
        break;
      if ( !v159 )
        goto LABEL_244;
      v250 = v157;
      v160 = sub_B91C10(v158, 1);
      v157 = v250;
      if ( !v160 )
        goto LABEL_317;
      v226 = v250;
      v248 = sub_DFFEB0(v160);
      if ( !v248 )
      {
        v158 = *v151;
        v157 = v226;
        goto LABEL_244;
      }
      v293 = 257;
      sub_921880(&v308, v356, v357, (int)v272, 1, (__int64)&v290, 0);
LABEL_239:
      nullsub_61();
      v324 = &unk_49DA100;
      nullsub_63();
      if ( v308 != (unsigned int *)&v310 )
        _libc_free((unsigned __int64)v308);
      v151 += 2;
      if ( v247 == v151 )
      {
        a1 = v216;
        v254 = v248;
        goto LABEL_26;
      }
    }
    if ( v159 )
    {
      v252 = v157;
      v178 = sub_B91C10(v158, 1);
      v157 = v252;
      if ( v178 )
      {
        v179 = sub_DFFEB0(v178);
        v157 = v252;
        v248 = v179;
        if ( v179 )
        {
          v180 = &v273;
          v181 = *(_QWORD *)(*v151 - 64);
          v182 = *(unsigned __int8 *)(*(_QWORD *)(v181 + 8) + 8LL);
          if ( (unsigned int)(v182 - 17) <= 1 )
          {
            LOWORD(v277) = 257;
            v183 = sub_BCB2D0(v317);
            v184 = (unsigned __int8 *)sub_ACD640(v183, 0, 0);
            v185 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v318 + 96LL);
            if ( v185 == sub_948070 )
            {
              if ( *(_BYTE *)v181 <= 0x15u && *v184 <= 0x15u )
              {
                v228 = v184;
                v186 = sub_AD5840(v181, v184, 0);
                v184 = v228;
                goto LABEL_276;
              }
              goto LABEL_338;
            }
            v232 = v184;
            v186 = v185(v318, (_BYTE *)v181, v184);
            v184 = v232;
LABEL_276:
            if ( v186 )
            {
              v181 = v186;
              v180 = &v273;
            }
            else
            {
LABEL_338:
              v229 = v184;
              v293 = 257;
              v200 = sub_BD2C40(72, 2u);
              if ( v200 )
              {
                v201 = (__int64)v229;
                v230 = v200;
                sub_B4DE80((__int64)v200, v181, v201, (__int64)&v290, 0, 0);
                v200 = v230;
              }
              v231 = (__int64)v200;
              (*(void (__fastcall **)(__int64, _QWORD *, __int64 *, __int64, __int64))(*(_QWORD *)v319 + 16LL))(
                v319,
                v200,
                &v273,
                v315,
                v316);
              v202 = v308;
              v203 = v231;
              v180 = &v273;
              if ( v308 != &v308[4 * (unsigned int)v309] )
              {
                v223 = v152;
                v204 = &v308[4 * (unsigned int)v309];
                do
                {
                  v205 = *((_QWORD *)v202 + 1);
                  v206 = *v202;
                  v202 += 4;
                  v220 = v180;
                  sub_B99FD0(v231, v206, v205);
                  v180 = v220;
                }
                while ( v204 != v202 );
                v203 = v231;
                v152 = v223;
              }
              v181 = v203;
            }
            LOBYTE(v182) = *(_BYTE *)(*(_QWORD *)(v181 + 8) + 8LL);
          }
          if ( (_BYTE)v182 == 12 )
          {
            v233 = (int)v180;
            v293 = 257;
            v207 = (__int64 **)sub_BCE3C0(v317, 0);
            v181 = sub_24CC250((__int64 *)&v308, 0x30u, v181, v207, (__int64)&v290, 0, v273, 0);
            LODWORD(v180) = v233;
          }
          v293 = 257;
          v273 = v272[0];
          v274 = v181;
          sub_921880(&v308, v354, v355, (int)v180, 2, (__int64)&v290, 0);
          goto LABEL_239;
        }
      }
LABEL_317:
      v158 = *v151;
    }
LABEL_244:
    v161 = byte_4FEDD68;
    _BitScanReverse64(&v162, 1LL << (*(_WORD *)(v158 + 2) >> 1));
    v163 = v162 ^ 0x3F;
    if ( byte_4FEDD68 )
      v161 = v151[1] & 1;
    LOBYTE(v164) = qword_4FEDF28;
    if ( (_BYTE)qword_4FEDF28 )
      v164 = *(_WORD *)(v158 + 2) & 1;
    v219 = v164;
    v222 = v161;
    v227 = 63 - v163;
    v251 = v157;
    v165 = sub_9208B0(v262, v156);
    v291 = v166;
    v290 = (unsigned int *)((v165 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    v167 = sub_CA1930(&v290);
    if ( v227 <= 2u && (1LL << v227) % (unsigned __int64)(v167 >> 3) )
    {
      if ( v222 )
      {
        v168 = 16 * (v251 + 49LL);
        v169 = *(unsigned __int64 *)((char *)&v336 + v168);
        v170 = *(__int64 *)((char *)&v337 + v168);
LABEL_252:
        v293 = 257;
        sub_921880(&v308, v169, v170, (int)v272, 1, (__int64)&v290, 0);
        v248 = v254;
        goto LABEL_239;
      }
      v189 = v251;
      if ( v219 )
      {
        v173 = (unsigned __int64 *)&v350[v189];
        if ( v155 == 62 )
          v173 = (unsigned __int64 *)&v351[v251];
      }
      else
      {
        v173 = (unsigned __int64 *)&v346[v189];
        if ( v155 == 62 )
          v173 = (unsigned __int64 *)&v347[v251];
      }
    }
    else
    {
      if ( v222 )
      {
        v171 = 16 * (v251 + 44LL);
        v169 = *(unsigned __int64 *)((char *)&v336 + v171);
        v170 = *(__int64 *)((char *)&v337 + v171);
        goto LABEL_252;
      }
      v172 = v251;
      if ( v219 )
      {
        v173 = (unsigned __int64 *)&v348[v172];
        if ( v155 == 62 )
          v173 = (unsigned __int64 *)&v349[v251];
      }
      else
      {
        v173 = (unsigned __int64 *)&v344[v172];
        if ( v155 == 62 )
          v173 = (unsigned __int64 *)&v345[v251];
      }
    }
    v169 = *v173;
    v170 = v173[1];
    goto LABEL_252;
  }
LABEL_26:
  v225 = qword_4FEE0E8;
  if ( (_BYTE)qword_4FEE0E8 )
  {
    v249 = &v281[(unsigned int)v282];
    if ( v281 != v249 )
    {
      v21 = v281;
      v221 = a1;
      while ( 1 )
      {
        v22 = *v21;
        sub_23E3770((__int64)&v308, *v21);
        v23 = *(_BYTE *)v22;
        if ( *(_BYTE *)v22 == 61 )
        {
          v24 = *(_QWORD *)(v22 - 32);
          v234 = *(_QWORD *)(v22 + 8);
          v241 = sub_24CC3F0(v234, v262);
          if ( v241 >= 0 )
          {
            v273 = v24;
            v25 = sub_24CC1C0((__int64)&v308, (*(_WORD *)(v22 + 2) >> 7) & 7);
            v293 = 257;
            v274 = v25;
            v26 = sub_921880(
                    &v308,
                    *(&v336 + 2 * v241 + 108),
                    *(&v337 + 2 * v241 + 108),
                    (int)&v273,
                    2,
                    (__int64)&v290,
                    0);
            v293 = 257;
            v27 = v26;
            v28 = *(_QWORD *)(v26 + 8);
            if ( v234 == v28 )
            {
LABEL_41:
              sub_BD84D0(v22, v27);
              v254 = v225;
              goto LABEL_42;
            }
            v29 = *(unsigned __int8 *)(v28 + 8);
            v30 = *(_BYTE *)(v28 + 8);
            if ( (unsigned int)(v29 - 17) > 1 )
            {
              if ( (_BYTE)v29 == 14 )
                goto LABEL_156;
              goto LABEL_35;
            }
            if ( *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL) != 14 )
              goto LABEL_34;
LABEL_156:
            v114 = *(unsigned __int8 *)(v234 + 8);
            if ( (unsigned int)(v114 - 17) <= 1 )
              LOBYTE(v114) = *(_BYTE *)(**(_QWORD **)(v234 + 16) + 8LL);
            if ( (_BYTE)v114 == 12 )
            {
              v27 = sub_24CC250((__int64 *)&v308, 0x2Fu, v27, (__int64 **)v234, (__int64)&v290, 0, v272[0], 0);
              goto LABEL_41;
            }
LABEL_34:
            if ( v29 != 18 )
            {
LABEL_35:
              if ( v29 != 17 )
              {
LABEL_36:
                if ( v30 != 12 )
                  goto LABEL_40;
                v31 = *(unsigned __int8 *)(v234 + 8);
                if ( (unsigned int)(v31 - 17) <= 1 )
                  LOBYTE(v31) = *(_BYTE *)(**(_QWORD **)(v234 + 16) + 8LL);
                if ( (_BYTE)v31 == 14 )
                  v27 = sub_24CC250((__int64 *)&v308, 0x30u, v27, (__int64 **)v234, (__int64)&v290, 0, v272[0], 0);
                else
LABEL_40:
                  v27 = sub_24CC250((__int64 *)&v308, 0x31u, v27, (__int64 **)v234, (__int64)&v290, 0, v272[0], 0);
                goto LABEL_41;
              }
            }
            v30 = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
            goto LABEL_36;
          }
        }
        else
        {
          if ( v23 != 62 )
          {
            if ( v23 == 66 )
            {
              v49 = *(_QWORD *)(v22 - 64);
              v50 = sub_24CC3F0(*(_QWORD *)(*(_QWORD *)(v22 - 32) + 8LL), v262);
              if ( v50 < 0 )
                goto LABEL_42;
              v243 = *(&v337 + 10 * ((*(_WORD *)(v22 + 2) >> 4) & 0x1F) + 2 * v50 + 128);
              if ( !v243 )
                goto LABEL_42;
              v255 = *(&v336 + 10 * ((*(_WORD *)(v22 + 2) >> 4) & 0x1F) + 2 * v50 + 128);
              v51 = sub_BCD140(v317, 8 << v50);
              v52 = *(_QWORD *)(v22 - 32);
              v273 = v49;
              v293 = 257;
              v53 = v51;
              v54 = *(_QWORD *)(v52 + 8);
              if ( v53 == v54 )
              {
                v58 = v52;
              }
              else
              {
                v55 = *(unsigned __int8 *)(v54 + 8);
                v56 = *(_BYTE *)(v54 + 8);
                if ( (unsigned int)(v55 - 17) > 1 )
                {
                  if ( (_BYTE)v55 == 14 )
                    goto LABEL_297;
                }
                else
                {
                  if ( *(_BYTE *)(**(_QWORD **)(v54 + 16) + 8LL) != 14 )
                    goto LABEL_97;
LABEL_297:
                  v188 = *(unsigned __int8 *)(v53 + 8);
                  if ( (unsigned int)(v188 - 17) <= 1 )
                    LOBYTE(v188) = *(_BYTE *)(**(_QWORD **)(v53 + 16) + 8LL);
                  if ( (_BYTE)v188 == 12 )
                  {
                    v239 = v52;
                    v58 = sub_24CC250((__int64 *)&v308, 0x2Fu, v52, (__int64 **)v53, (__int64)&v290, 0, v272[0], 0);
                    v52 = v239;
                    goto LABEL_104;
                  }
LABEL_97:
                  if ( v55 == 18 )
                  {
LABEL_98:
                    v56 = *(_BYTE *)(**(_QWORD **)(v54 + 16) + 8LL);
                    goto LABEL_99;
                  }
                }
                if ( v55 == 17 )
                  goto LABEL_98;
LABEL_99:
                if ( v56 != 12 )
                  goto LABEL_103;
                v57 = *(unsigned __int8 *)(v53 + 8);
                if ( (unsigned int)(v57 - 17) <= 1 )
                  LOBYTE(v57) = *(_BYTE *)(**(_QWORD **)(v53 + 16) + 8LL);
                if ( (_BYTE)v57 == 14 )
                {
                  v240 = v52;
                  v58 = sub_24CC250((__int64 *)&v308, 0x30u, v52, (__int64 **)v53, (__int64)&v290, 0, v272[0], 0);
                  v52 = v240;
                }
                else
                {
LABEL_103:
                  v235 = v52;
                  v58 = sub_24CC250((__int64 *)&v308, 0x31u, v52, (__int64 **)v53, (__int64)&v290, 0, v272[0], 0);
                  v52 = v235;
                }
              }
LABEL_104:
              v274 = v58;
              v236 = v52;
              v59 = sub_24CC1C0((__int64)&v308, (*(_WORD *)(v22 + 2) >> 1) & 7);
              v293 = 257;
              v275 = v59;
              v60 = sub_921880(&v308, v255, v243, (int)&v273, 3, (__int64)&v290, 0);
              v293 = 257;
              v61 = v60;
              v62 = *(_QWORD *)(v60 + 8);
              v63 = *(_QWORD *)(v236 + 8);
              if ( v63 == v62 )
              {
LABEL_114:
                v67 = v61;
                goto LABEL_115;
              }
              v64 = *(unsigned __int8 *)(v62 + 8);
              v65 = *(_BYTE *)(v62 + 8);
              if ( (unsigned int)(v64 - 17) > 1 )
              {
                if ( (_BYTE)v64 == 14 )
                  goto LABEL_292;
              }
              else
              {
                if ( *(_BYTE *)(**(_QWORD **)(v62 + 16) + 8LL) != 14 )
                  goto LABEL_107;
LABEL_292:
                v187 = *(unsigned __int8 *)(v63 + 8);
                if ( (unsigned int)(v187 - 17) <= 1 )
                  LOBYTE(v187) = *(_BYTE *)(**(_QWORD **)(v63 + 16) + 8LL);
                if ( (_BYTE)v187 == 12 )
                {
                  v61 = sub_24CC250((__int64 *)&v308, 0x2Fu, v61, (__int64 **)v63, (__int64)&v290, 0, v272[0], 0);
                  goto LABEL_114;
                }
LABEL_107:
                if ( v64 == 18 )
                {
LABEL_108:
                  v65 = *(_BYTE *)(**(_QWORD **)(v62 + 16) + 8LL);
                  goto LABEL_109;
                }
              }
              if ( v64 == 17 )
                goto LABEL_108;
LABEL_109:
              if ( v65 != 12 )
                goto LABEL_113;
              v66 = *(unsigned __int8 *)(v63 + 8);
              if ( (unsigned int)(v66 - 17) <= 1 )
                LOBYTE(v66) = *(_BYTE *)(**(_QWORD **)(v63 + 16) + 8LL);
              if ( (_BYTE)v66 == 14 )
                v61 = sub_24CC250((__int64 *)&v308, 0x30u, v61, (__int64 **)v63, (__int64)&v290, 0, v272[0], 0);
              else
LABEL_113:
                v61 = sub_24CC250((__int64 *)&v308, 0x31u, v61, (__int64 **)v63, (__int64)&v290, 0, v272[0], 0);
              goto LABEL_114;
            }
            if ( v23 == 65 )
            {
              v244 = *(_QWORD *)(v22 - 96);
              v237 = *(__int64 ***)(*(_QWORD *)(v22 - 32) + 8LL);
              v218 = sub_24CC3F0((__int64)v237, v262);
              if ( v218 < 0 )
                goto LABEL_42;
              v132 = sub_BCD140(v317, 8 << v218);
              v293 = 257;
              v133 = *(_QWORD *)(v22 - 64);
              v134 = v132;
              v135 = *(_QWORD *)(v133 + 8);
              if ( v134 != v135 )
              {
                v136 = *(unsigned __int8 *)(v135 + 8);
                v137 = *(_BYTE *)(v135 + 8);
                if ( (unsigned int)(v136 - 17) > 1 )
                {
                  if ( (_BYTE)v136 == 14 )
                    goto LABEL_329;
                }
                else
                {
                  if ( *(_BYTE *)(**(_QWORD **)(v135 + 16) + 8LL) != 14 )
                    goto LABEL_206;
LABEL_329:
                  v199 = *(unsigned __int8 *)(v134 + 8);
                  if ( (unsigned int)(v199 - 17) <= 1 )
                    LOBYTE(v199) = *(_BYTE *)(**(_QWORD **)(v134 + 16) + 8LL);
                  if ( (_BYTE)v199 == 12 )
                  {
                    v256 = v134;
                    v139 = sub_24CC250((__int64 *)&v308, 0x2Fu, v133, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
                    goto LABEL_213;
                  }
LABEL_206:
                  if ( v136 == 18 )
                  {
LABEL_207:
                    v137 = *(_BYTE *)(**(_QWORD **)(v135 + 16) + 8LL);
                    goto LABEL_208;
                  }
                }
                if ( v136 == 17 )
                  goto LABEL_207;
LABEL_208:
                if ( v137 != 12 )
                  goto LABEL_212;
                v138 = *(unsigned __int8 *)(v134 + 8);
                if ( (unsigned int)(v138 - 17) <= 1 )
                  LOBYTE(v138) = *(_BYTE *)(**(_QWORD **)(v134 + 16) + 8LL);
                if ( (_BYTE)v138 == 14 )
                {
                  v256 = v134;
                  v139 = sub_24CC250((__int64 *)&v308, 0x30u, v133, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
                }
                else
                {
LABEL_212:
                  v256 = v134;
                  v139 = sub_24CC250((__int64 *)&v308, 0x31u, v133, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
                }
LABEL_213:
                v134 = v256;
                v133 = v139;
              }
              v293 = 257;
              v140 = *(_QWORD *)(v22 - 32);
              v141 = *(_QWORD *)(v140 + 8);
              if ( v134 == v141 )
                goto LABEL_225;
              v142 = *(unsigned __int8 *)(v141 + 8);
              v143 = *(_BYTE *)(v141 + 8);
              if ( (unsigned int)(v142 - 17) > 1 )
              {
                if ( (_BYTE)v142 == 14 )
                  goto LABEL_324;
              }
              else
              {
                if ( *(_BYTE *)(**(_QWORD **)(v141 + 16) + 8LL) != 14 )
                  goto LABEL_217;
LABEL_324:
                v198 = *(unsigned __int8 *)(v134 + 8);
                if ( (unsigned int)(v198 - 17) <= 1 )
                  LOBYTE(v198) = *(_BYTE *)(**(_QWORD **)(v134 + 16) + 8LL);
                if ( (_BYTE)v198 == 12 )
                {
                  v214 = v133;
                  v257 = v134;
                  v145 = sub_24CC250((__int64 *)&v308, 0x2Fu, v140, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
LABEL_224:
                  v134 = v257;
                  v133 = v214;
                  v140 = v145;
                  goto LABEL_225;
                }
LABEL_217:
                if ( v142 == 18 )
                {
LABEL_218:
                  v143 = *(_BYTE *)(**(_QWORD **)(v141 + 16) + 8LL);
                  goto LABEL_219;
                }
              }
              if ( v142 == 17 )
                goto LABEL_218;
LABEL_219:
              if ( v143 != 12 )
                goto LABEL_223;
              v144 = *(unsigned __int8 *)(v134 + 8);
              if ( (unsigned int)(v144 - 17) <= 1 )
                LOBYTE(v144) = *(_BYTE *)(**(_QWORD **)(v134 + 16) + 8LL);
              if ( (_BYTE)v144 != 14 )
              {
LABEL_223:
                v214 = v133;
                v257 = v134;
                v145 = sub_24CC250((__int64 *)&v308, 0x31u, v140, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
                goto LABEL_224;
              }
              v217 = v133;
              v260 = v134;
              v208 = sub_24CC250((__int64 *)&v308, 0x30u, v140, (__int64 **)v134, (__int64)&v290, 0, v273, 0);
              v133 = v217;
              v140 = v208;
              v134 = v260;
LABEL_225:
              v274 = v133;
              v275 = v140;
              v273 = v244;
              v215 = (__int64 **)v134;
              v258 = (_BYTE *)v133;
              v276 = sub_24CC1C0((__int64)&v308, (*(_WORD *)(v22 + 2) >> 2) & 7);
              v277 = sub_24CC1C0((__int64)&v308, (*(_WORD *)(v22 + 2) >> 5) & 7);
              v293 = 257;
              v146 = sub_921880(
                       &v308,
                       *(&v336 + 2 * v218 + 318),
                       *(&v337 + 2 * v218 + 318),
                       (int)&v273,
                       5,
                       (__int64)&v290,
                       0);
              v293 = 257;
              v245 = v146;
              v147 = sub_92B530(&v308, 0x20u, v146, v258, (__int64)&v290);
              v148 = v245;
              v259 = (_BYTE *)v147;
              if ( v237 != v215 )
              {
                v293 = 257;
                v148 = sub_24CC250((__int64 *)&v308, 0x30u, v245, v237, (__int64)&v290, 0, v272[0], 0);
              }
              LODWORD(v272[0]) = 0;
              v293 = 257;
              v238 = (_BYTE *)v148;
              v149 = sub_ACADE0(*(__int64 ***)(v22 + 8));
              v150 = sub_2466140((__int64 *)&v308, v149, v238, v272, 1, (__int64)&v290);
              v293 = 257;
              LODWORD(v272[0]) = 1;
              v67 = sub_2466140((__int64 *)&v308, v150, v259, v272, 1, (__int64)&v290);
LABEL_115:
              sub_BD84D0(v22, v67);
              sub_B43D60((_QWORD *)v22);
              v254 = v225;
              goto LABEL_42;
            }
            v254 = v225;
            if ( v23 != 64 )
              goto LABEL_42;
            v273 = sub_24CC1C0((__int64)&v308, *(_WORD *)(v22 + 2) & 7);
            if ( *(_BYTE *)(v22 + 72) )
            {
              v175 = v352[0];
              v176 = v352;
            }
            else
            {
              v175 = v353[0];
              v176 = v353;
            }
            v177 = v176[1];
            v293 = 257;
            sub_921880(&v308, v175, v177, (int)&v273, 1, (__int64)&v290, 0);
            goto LABEL_90;
          }
          v40 = *(_QWORD *)(v22 - 32);
          v41 = sub_24CC3F0(*(_QWORD *)(*(_QWORD *)(v22 - 64) + 8LL), v262);
          v242 = v41;
          if ( v41 >= 0 )
          {
            v42 = sub_BCD140(v317, 8 << v41);
            v273 = v40;
            v293 = 257;
            v43 = *(_QWORD *)(v22 - 64);
            v44 = v42;
            v45 = *(_QWORD *)(v43 + 8);
            if ( v44 == v45 )
              goto LABEL_89;
            v46 = *(unsigned __int8 *)(v45 + 8);
            v47 = *(_BYTE *)(v45 + 8);
            if ( (unsigned int)(v46 - 17) > 1 )
            {
              if ( (_BYTE)v46 == 14 )
                goto LABEL_260;
            }
            else
            {
              if ( *(_BYTE *)(**(_QWORD **)(v45 + 16) + 8LL) != 14 )
                goto LABEL_82;
LABEL_260:
              v174 = *(unsigned __int8 *)(v44 + 8);
              if ( (unsigned int)(v174 - 17) <= 1 )
                LOBYTE(v174) = *(_BYTE *)(**(_QWORD **)(v44 + 16) + 8LL);
              if ( (_BYTE)v174 == 12 )
              {
                v43 = sub_24CC250((__int64 *)&v308, 0x2Fu, v43, (__int64 **)v44, (__int64)&v290, 0, v272[0], 0);
                goto LABEL_89;
              }
LABEL_82:
              if ( v46 == 18 )
              {
LABEL_83:
                v47 = *(_BYTE *)(**(_QWORD **)(v45 + 16) + 8LL);
                goto LABEL_84;
              }
            }
            if ( v46 == 17 )
              goto LABEL_83;
LABEL_84:
            if ( v47 != 12 )
              goto LABEL_88;
            v48 = *(unsigned __int8 *)(v44 + 8);
            if ( (unsigned int)(v48 - 17) <= 1 )
              LOBYTE(v48) = *(_BYTE *)(**(_QWORD **)(v44 + 16) + 8LL);
            if ( (_BYTE)v48 == 14 )
              v43 = sub_24CC250((__int64 *)&v308, 0x30u, v43, (__int64 **)v44, (__int64)&v290, 0, v272[0], 0);
            else
LABEL_88:
              v43 = sub_24CC250((__int64 *)&v308, 0x31u, v43, (__int64 **)v44, (__int64)&v290, 0, v272[0], 0);
LABEL_89:
            v274 = v43;
            v275 = sub_24CC1C0((__int64)&v308, (*(_WORD *)(v22 + 2) >> 7) & 7);
            v293 = 257;
            sub_921880(&v308, *(&v336 + 2 * v242 + 118), *(&v337 + 2 * v242 + 118), (int)&v273, 3, (__int64)&v290, 0);
LABEL_90:
            sub_B43D60((_QWORD *)v22);
            v254 = v225;
          }
        }
LABEL_42:
        nullsub_61();
        v324 = &unk_49DA100;
        nullsub_63();
        if ( v308 != (unsigned int *)&v310 )
          _libc_free((unsigned __int64)v308);
        if ( v249 == ++v21 )
        {
          a1 = v221;
          break;
        }
      }
    }
  }
  if ( (_BYTE)qword_4FEE008 )
  {
    if ( v224 )
    {
      v115 = v284;
      v268 = &v284[(unsigned int)v285];
      if ( v284 != v268 )
      {
        v246 = a1;
        while ( 1 )
        {
          v116 = *v115;
          sub_23E3770((__int64)&v308, *v115);
          if ( *(_BYTE *)v116 != 85 )
            goto LABEL_164;
          v117 = *(_QWORD *)(v116 - 32);
          if ( !v117 )
            goto LABEL_164;
          if ( *(_BYTE *)v117
            || *(_QWORD *)(v117 + 24) != *(_QWORD *)(v116 + 80)
            || (*(_BYTE *)(v117 + 33) & 0x20) == 0
            || ((*(_DWORD *)(v117 + 36) - 243) & 0xFFFFFFFD) != 0 )
          {
            if ( *(_BYTE *)v117 )
              goto LABEL_164;
            if ( *(_QWORD *)(v117 + 24) != *(_QWORD *)(v116 + 80) )
              goto LABEL_164;
            if ( (*(_BYTE *)(v117 + 33) & 0x20) == 0 )
              goto LABEL_164;
            v118 = *(_DWORD *)(v117 + 36);
            if ( v118 != 238 && (unsigned int)(v118 - 240) > 1 )
              goto LABEL_164;
            v119 = v335;
            v293 = 257;
            v120 = *(_DWORD *)(v116 + 4) & 0x7FFFFFF;
            v272[0] = *(_QWORD *)(v116 - 32 * v120);
            v272[1] = *(_QWORD *)(v116 + 32 * (1 - v120));
            LOWORD(v277) = 257;
            v265 = *(_QWORD *)(v116 + 32 * (2 - v120));
            v121 = sub_BCB060(*(_QWORD *)(v265 + 8));
            v122 = sub_BCB060((__int64)v119);
            v272[2] = sub_24CC250(
                        (__int64 *)&v308,
                        (unsigned int)(v121 <= v122) + 38,
                        v265,
                        v119,
                        (__int64)&v273,
                        0,
                        v271,
                        0);
            v123 = *(_QWORD *)(v116 - 32);
            if ( !v123 || *(_BYTE *)v123 || *(_QWORD *)(v123 + 24) != *(_QWORD *)(v116 + 80) )
              BUG();
            v124 = *(_DWORD *)(v123 + 36);
            v125 = v359;
            if ( ((v124 - 238) & 0xFFFFFFFD) != 0 )
              v125 = v358;
            sub_921880(&v308, *v125, v125[1], (int)v272, 3, (__int64)&v290, 0);
          }
          else
          {
            v293 = 257;
            v266 = (__int64 **)sub_BCB2D0(v317);
            v190 = *(_QWORD *)(v116 + 32 * (1LL - (*(_DWORD *)(v116 + 4) & 0x7FFFFFF)));
            v191 = sub_BCB060(*(_QWORD *)(v190 + 8));
            v192 = sub_BCB060((__int64)v266);
            v193 = sub_24CC250(
                     (__int64 *)&v308,
                     (unsigned int)(v191 <= v192) + 38,
                     v190,
                     v266,
                     (__int64)&v290,
                     0,
                     v273,
                     0);
            v194 = v335;
            v293 = 257;
            v253 = v193;
            v263 = *(_QWORD *)(v116 + 32 * (2LL - (*(_DWORD *)(v116 + 4) & 0x7FFFFFF)));
            LODWORD(v266) = sub_BCB060(*(_QWORD *)(v263 + 8));
            v195 = sub_BCB060((__int64)v194);
            v196 = sub_24CC250(
                     (__int64 *)&v308,
                     (unsigned int)((unsigned int)v266 <= v195) + 38,
                     v263,
                     v194,
                     (__int64)&v290,
                     0,
                     v273,
                     0);
            v293 = 257;
            v197 = *(_QWORD *)(v116 - 32LL * (*(_DWORD *)(v116 + 4) & 0x7FFFFFF));
            v275 = v196;
            v274 = v253;
            v273 = v197;
            sub_921880(&v308, v360, v361, (int)&v273, 3, (__int64)&v290, 0);
          }
          sub_B43D60((_QWORD *)v116);
LABEL_164:
          nullsub_61();
          v324 = &unk_49DA100;
          nullsub_63();
          if ( v308 != (unsigned int *)&v310 )
            _libc_free((unsigned __int64)v308);
          if ( v268 == ++v115 )
          {
            a1 = v246;
            break;
          }
        }
      }
    }
  }
  if ( !(unsigned __int8)sub_B2D620(a3, "sanitize_thread_no_checking_at_run_time", 0x27u) || !v261 )
  {
    v32 = v254 | v261;
    if ( !((unsigned __int8)v254 | (unsigned __int8)v261) )
      goto LABEL_51;
    goto LABEL_140;
  }
  v68 = *(_QWORD *)(a3 + 80);
  if ( v68 )
    v68 -= 24;
  v69 = sub_AA4FF0(v68);
  v70 = *(_QWORD *)(a3 + 80);
  v71 = v69;
  v73 = v72;
  if ( !v71 )
    v73 = 0;
  v74 = v73;
  if ( v70 )
    v70 -= 24;
  v297 = (_QWORD *)sub_AA48A0(v70);
  v298 = &v306;
  v299 = &v307;
  v306 = &unk_49DA100;
  v307 = &unk_49DA0B0;
  LOBYTE(v75) = 1;
  HIBYTE(v75) = v74;
  v290 = (unsigned int *)v292;
  v296 = v75;
  v291 = 0x200000000LL;
  v300 = 0;
  v301 = 0;
  v302 = 512;
  v303 = 7;
  v304 = 0;
  v305 = 0;
  v294 = v70;
  v295 = v71;
  if ( v71 != v70 + 48 )
  {
    v76 = v71 - 24;
    if ( !v71 )
      v76 = 0;
    v77 = *(unsigned int **)sub_B46C60(v76);
    v308 = v77;
    if ( v77 && (sub_B96E90((__int64)&v308, (__int64)v77, 1), (v79 = v308) != 0) )
    {
      v80 = v290;
      v81 = v291;
      v82 = &v290[4 * (unsigned int)v291];
      if ( v290 != v82 )
      {
        v83 = v290;
        while ( *v83 )
        {
          v83 += 4;
          if ( v82 == v83 )
            goto LABEL_311;
        }
        *((_QWORD *)v83 + 1) = v308;
        goto LABEL_132;
      }
LABEL_311:
      if ( (unsigned int)v291 >= (unsigned __int64)HIDWORD(v291) )
      {
        if ( HIDWORD(v291) < (unsigned __int64)(unsigned int)v291 + 1 )
        {
          v267 = v308;
          sub_C8D5F0((__int64)&v290, v292, (unsigned int)v291 + 1LL, 0x10u, (__int64)v308, v78);
          v80 = v290;
          v79 = v267;
        }
        v209 = &v80[4 * (unsigned int)v291];
        *(_QWORD *)v209 = 0;
        *((_QWORD *)v209 + 1) = v79;
        v79 = v308;
        LODWORD(v291) = v291 + 1;
      }
      else
      {
        if ( v82 )
        {
          *v82 = 0;
          *((_QWORD *)v82 + 1) = v79;
          v81 = v291;
          v79 = v308;
        }
        LODWORD(v291) = v81 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v290, 0);
      v79 = v308;
    }
    if ( v79 )
LABEL_132:
      sub_B91220((__int64)&v308, (__int64)v79);
  }
  sub_24CC5B0((__int64)&v290, *(_QWORD *)(v70 + 72));
  LOWORD(v312) = 257;
  sub_921880(&v290, v340, v341, 0, 0, (__int64)&v308, 0);
  v84 = qword_4FEE1C8;
  v309 = "tsan_ignore_cleanup";
  v85 = *(_QWORD *)(a3 + 80);
  v308 = (unsigned int *)a3;
  v310 = v85;
  v311 = a3 + 72;
  v86 = sub_B2BE50(a3);
  v88 = v210;
  v321 = v86;
  v89 = 512;
  v90 = v212;
  v322 = &v330;
  v323 = &v331;
  v312 = &v314;
  v91 = 0x200000000LL;
  v330 = &unk_49DA100;
  v333 = v84;
  v313 = 0x200000000LL;
  v324 = 0;
  v325 = 0;
  v326 = 512;
  v327 = 7;
  v328 = 0;
  v329 = 0;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v331 = &unk_49DA0B0;
  v332 = 0;
  v334 = 0;
  while ( 1 )
  {
    v92 = sub_29CEC00(&v308, v89, v87, v91, v88, v90);
    v93 = (unsigned int **)v92;
    if ( !v92 )
      break;
    sub_24CC5B0(v92, a3);
    v89 = v342;
    LOWORD(v277) = 257;
    sub_921880(v93, v342, v343, 0, 0, (__int64)&v273, 0);
    v87 = v212;
  }
  nullsub_61();
  v330 = &unk_49DA100;
  nullsub_63();
  if ( v312 != &v314 )
    _libc_free((unsigned __int64)v312);
  nullsub_61();
  v306 = &unk_49DA100;
  nullsub_63();
  if ( v290 != (unsigned int *)v292 )
    _libc_free((unsigned __int64)v290);
LABEL_140:
  v32 = byte_4FEE2A8;
  if ( byte_4FEE2A8 )
  {
    v94 = a3;
    v95 = *(_QWORD *)(a3 + 80);
    if ( v95 )
      v95 -= 24;
    v96 = sub_AA4FF0(v95);
    v98 = *(_QWORD *)(a3 + 80);
    LOBYTE(v99) = 1;
    v100 = v96;
    v101 = 0;
    if ( v100 )
      v101 = v97;
    HIBYTE(v99) = v101;
    if ( v98 )
      v98 -= 24;
    sub_24CC710((__int64)&v290, v98, v100, v99);
    HIDWORD(v273) = 0;
    LOWORD(v312) = 257;
    v102 = sub_BCB2D0(v297);
    v272[0] = sub_ACD640(v102, 0, 0);
    v103 = sub_B33D10((__int64)&v290, 0x133u, 0, 0, (int)v272, 1, v273, (__int64)&v308);
    v104 = v336;
    LOWORD(v312) = 257;
    v271 = v103;
    sub_921880(&v290, v336, v337, (int)&v271, 1, (__int64)&v308, 0);
    v105 = qword_4FEE1C8;
    v309 = "tsan_cleanup";
    v106 = *(_QWORD *)(a3 + 80);
    v308 = (unsigned int *)a3;
    v310 = v106;
    v311 = a3 + 72;
    v107 = sub_B2BE50(a3);
    v109 = 512;
    v110 = 0;
    v321 = v107;
    v322 = &v330;
    v323 = &v331;
    v330 = &unk_49DA100;
    v312 = &v314;
    v111 = 0x200000000LL;
    v333 = v105;
    v313 = 0x200000000LL;
    v324 = 0;
    v325 = 0;
    v326 = 512;
    v327 = 7;
    v328 = 0;
    v329 = 0;
    v318 = 0;
    v319 = 0;
    v320 = 0;
    v331 = &unk_49DA0B0;
    v332 = 0;
    v334 = 0;
    v270 = a1;
    while ( 1 )
    {
      v112 = sub_29CEC00(&v308, v104, v108, v111, v109, v110);
      v113 = (unsigned int **)v112;
      if ( !v112 )
        break;
      sub_24CC5B0(v112, v94);
      v104 = v338;
      LOWORD(v277) = 257;
      sub_921880(v113, v338, v339, 0, 0, (__int64)&v273, 0);
      v108 = v211;
      v111 = v213;
    }
    a1 = v270;
    nullsub_61();
    v330 = &unk_49DA100;
    nullsub_63();
    if ( v312 != &v314 )
      _libc_free((unsigned __int64)v312);
    nullsub_61();
    v306 = &unk_49DA100;
    nullsub_63();
    if ( v290 != (unsigned int *)v292 )
      _libc_free((unsigned __int64)v290);
  }
  else
  {
    v32 = v254;
  }
LABEL_51:
  if ( v284 != (__int64 *)v286 )
    _libc_free((unsigned __int64)v284);
  if ( v281 != (__int64 *)v283 )
    _libc_free((unsigned __int64)v281);
  if ( v278 != v280 )
    _libc_free((unsigned __int64)v278);
  if ( v287 != (__int64 *)v289 )
    _libc_free((unsigned __int64)v287);
  v9 = a1 + 4;
  v10 = a1 + 10;
  if ( v32 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v9;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
LABEL_6:
  a1[1] = v9;
  a1[2] = 0x100000002LL;
  a1[6] = 0;
  a1[7] = v10;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  a1[4] = &qword_4F82400;
  *a1 = 1;
  return a1;
}
