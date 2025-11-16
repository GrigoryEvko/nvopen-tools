// Function: sub_2806080
// Address: 0x2806080
//
__int64 __fastcall sub_2806080(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rdi
  size_t v7; // r12
  const char *v8; // r13
  _QWORD *v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rax
  __m128i *v12; // rsi
  __int64 v13; // r12
  bool v14; // bl
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 **v21; // rdx
  __int64 v22; // rcx
  void **v23; // rax
  __int32 v24; // eax
  void **v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r12
  __int64 *v40; // r15
  __int64 v41; // rax
  __int64 v42; // rcx
  int v43; // edx
  __int64 v44; // rax
  unsigned __int8 *v45; // r10
  __int64 v46; // rdi
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r8
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdi
  unsigned __int64 v58; // rax
  int v59; // edx
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rax
  const char *v63; // rax
  unsigned __int64 v64; // rdx
  __int64 *v65; // r12
  __int64 v67; // r13
  __int64 v68; // rbx
  __int64 v69; // r14
  unsigned __int8 *v70; // r15
  __int64 v71; // r12
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r14
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __m128i v82; // xmm3
  __m128i v83; // xmm4
  __m128i v84; // xmm5
  _BYTE *v85; // r8
  unsigned __int64 *v86; // r14
  __int64 v87; // rax
  unsigned __int64 *v88; // r13
  unsigned __int64 v89; // rdi
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r12
  __int64 v93; // rbx
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rcx
  __int64 v98; // rcx
  __int64 v99; // rax
  __int64 v100; // r12
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // r14
  __int64 v107; // r8
  __int64 v108; // r9
  __m128i v109; // xmm0
  __m128i v110; // xmm1
  __m128i v111; // xmm2
  _BYTE *v112; // r8
  unsigned __int64 *v113; // r14
  unsigned __int64 *v114; // r13
  unsigned __int64 v115; // rdi
  __int64 v116; // rsi
  __int64 v117; // rcx
  __int64 v118; // rdi
  _QWORD *v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 *v126; // rax
  __int64 v127; // rsi
  char v128; // bl
  __int64 *v129; // rsi
  char *v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rcx
  __int64 v135; // r8
  __int64 v136; // r9
  __int64 *v137; // rdi
  unsigned __int32 n; // eax
  __int64 v139; // r14
  __int64 v140; // rax
  char *v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  __int64 v145; // rdi
  unsigned __int8 *v146; // rdx
  __int64 v147; // rcx
  __int64 v148; // r8
  __int64 v149; // r9
  __int64 v150; // r15
  unsigned __int8 *v151; // rax
  _QWORD *v152; // rax
  __int64 v153; // rsi
  unsigned int v154; // edx
  __int64 *v155; // rax
  unsigned __int64 v156; // rax
  __int64 v157; // r12
  int v158; // ebx
  unsigned int i; // r13d
  unsigned int v160; // esi
  __int64 *v161; // rax
  __int64 v162; // rcx
  __int64 v163; // r8
  __int64 v164; // r9
  int v165; // r12d
  __int64 v166; // rbx
  int v167; // eax
  int v168; // ecx
  unsigned int m; // eax
  _QWORD *v170; // rdx
  unsigned int v171; // eax
  unsigned __int64 *v172; // r12
  unsigned __int64 *v173; // rbx
  unsigned __int64 v174; // rdi
  __int64 v175; // rax
  __int64 *v176; // r14
  __int64 v177; // rax
  size_t v178; // r15
  __int64 v179; // rbx
  unsigned __int64 v180; // rdx
  __int64 v181; // rax
  __int64 v182; // rax
  unsigned __int64 *v183; // r12
  unsigned __int64 *v184; // rbx
  unsigned __int64 v185; // rdi
  __int64 v186; // r13
  __int64 v187; // rdx
  __int64 v188; // rax
  __int64 v189; // r9
  __int64 v190; // rcx
  __int64 v191; // rbx
  __int64 v192; // rdx
  unsigned __int8 *v193; // r14
  int v194; // r10d
  __int64 v195; // rax
  int v196; // r15d
  __int64 v197; // r8
  __int64 v198; // r12
  int v199; // eax
  int v200; // esi
  unsigned int j; // eax
  _QWORD *v202; // rdx
  unsigned int v203; // eax
  __int64 v204; // rsi
  __int64 v205; // rax
  __int64 v206; // rax
  unsigned __int8 *v207; // rax
  unsigned __int64 v208; // rax
  int v209; // edx
  __int64 v210; // rax
  bool v211; // cf
  __int64 v212; // rdx
  unsigned __int8 *v213; // rax
  unsigned __int8 *v214; // rbx
  int v215; // esi
  _QWORD *v216; // rcx
  unsigned int v217; // eax
  _QWORD *v218; // rdx
  __int64 v219; // r10
  unsigned __int8 **v220; // rax
  __int64 v221; // rsi
  __int64 v222; // rax
  __int64 v223; // rax
  unsigned __int64 v224; // r12
  __int64 v225; // rbx
  char v226; // al
  unsigned int v227; // r13d
  int v228; // r12d
  unsigned int v229; // esi
  __int64 *v230; // rax
  __int64 v231; // rcx
  __int64 v232; // r8
  __int64 v233; // r9
  __int64 v234; // rdi
  __int64 v235; // rax
  __int64 v236; // rcx
  __int64 v237; // rsi
  unsigned __int8 *v238; // r10
  unsigned __int8 *v239; // r10
  unsigned __int8 *v240; // r10
  unsigned __int8 *v241; // r10
  int v242; // edx
  int v243; // edx
  __int64 v244; // rsi
  unsigned __int8 *v245; // rsi
  unsigned __int8 *v246; // rsi
  unsigned __int8 *v247; // rsi
  unsigned __int64 v248; // rax
  __int64 v249; // r13
  int v250; // r12d
  unsigned int k; // ebx
  unsigned int v252; // esi
  __int64 *v253; // rax
  __int64 v254; // rcx
  __int64 v255; // r8
  __int64 v256; // r9
  _QWORD *v257; // rax
  __int64 v258; // rdx
  __int64 v259; // rax
  __int64 v260; // rax
  int v261; // eax
  int v262; // edi
  __int64 v263; // r14
  __int64 *v264; // r13
  __int64 *v265; // r12
  unsigned int v266; // r13d
  int v267; // r12d
  unsigned int v268; // esi
  __int64 *v269; // rax
  __int64 v270; // rcx
  __int64 v271; // r8
  __int64 v272; // r9
  unsigned __int8 *v273; // rax
  __int64 v274; // rcx
  __int64 v275; // rdx
  unsigned __int8 *v276; // rax
  unsigned __int8 *v277; // rsi
  int v278; // [rsp+Ch] [rbp-594h]
  __int64 v279; // [rsp+10h] [rbp-590h]
  __int64 v280; // [rsp+18h] [rbp-588h]
  __int64 v281; // [rsp+20h] [rbp-580h]
  __int64 v282; // [rsp+38h] [rbp-568h]
  __int64 v283; // [rsp+48h] [rbp-558h]
  __int64 v284; // [rsp+50h] [rbp-550h]
  unsigned __int64 v285; // [rsp+58h] [rbp-548h]
  __int64 v286; // [rsp+60h] [rbp-540h]
  unsigned int v287; // [rsp+68h] [rbp-538h]
  char v288; // [rsp+6Eh] [rbp-532h]
  char v289; // [rsp+6Fh] [rbp-531h]
  __int64 v290; // [rsp+70h] [rbp-530h]
  char v291; // [rsp+78h] [rbp-528h]
  __int64 v292; // [rsp+78h] [rbp-528h]
  __int64 v293; // [rsp+88h] [rbp-518h]
  __int64 v294; // [rsp+88h] [rbp-518h]
  __int64 v296; // [rsp+98h] [rbp-508h]
  __int64 v297; // [rsp+98h] [rbp-508h]
  __int64 v298; // [rsp+A0h] [rbp-500h]
  int v299; // [rsp+A0h] [rbp-500h]
  __int64 v300; // [rsp+A8h] [rbp-4F8h]
  unsigned __int64 *v301; // [rsp+A8h] [rbp-4F8h]
  __int64 v302; // [rsp+B0h] [rbp-4F0h]
  __int64 v303; // [rsp+B0h] [rbp-4F0h]
  __int64 v304; // [rsp+B0h] [rbp-4F0h]
  __int64 *v305; // [rsp+C0h] [rbp-4E0h]
  __int64 *v306; // [rsp+C0h] [rbp-4E0h]
  unsigned int v310; // [rsp+ECh] [rbp-4B4h] BYREF
  __int64 *v311; // [rsp+F0h] [rbp-4B0h] BYREF
  __int64 v312; // [rsp+F8h] [rbp-4A8h] BYREF
  __int64 v313[2]; // [rsp+100h] [rbp-4A0h] BYREF
  __int64 v314[2]; // [rsp+110h] [rbp-490h] BYREF
  __int64 *v315; // [rsp+120h] [rbp-480h]
  _QWORD *v316; // [rsp+130h] [rbp-470h] BYREF
  size_t v317; // [rsp+138h] [rbp-468h]
  _QWORD dest[2]; // [rsp+140h] [rbp-460h] BYREF
  __int64 v319; // [rsp+150h] [rbp-450h] BYREF
  __int64 v320; // [rsp+158h] [rbp-448h]
  __int64 v321; // [rsp+160h] [rbp-440h]
  __int64 v322; // [rsp+168h] [rbp-438h]
  __int64 v323; // [rsp+170h] [rbp-430h] BYREF
  __int64 v324; // [rsp+178h] [rbp-428h]
  __int64 v325; // [rsp+180h] [rbp-420h]
  unsigned int v326; // [rsp+188h] [rbp-418h]
  __m128i v327; // [rsp+190h] [rbp-410h] BYREF
  __int64 v328; // [rsp+1A0h] [rbp-400h]
  unsigned int v329; // [rsp+1B0h] [rbp-3F0h]
  unsigned __int64 v330; // [rsp+1B8h] [rbp-3E8h]
  __int64 v331; // [rsp+1C0h] [rbp-3E0h]
  __m128i v332; // [rsp+1D0h] [rbp-3D0h] BYREF
  __int64 v333; // [rsp+1E0h] [rbp-3C0h] BYREF
  int v334; // [rsp+1E8h] [rbp-3B8h]
  char v335; // [rsp+1ECh] [rbp-3B4h]
  __int64 v336; // [rsp+1F0h] [rbp-3B0h] BYREF
  char *v337; // [rsp+210h] [rbp-390h] BYREF
  unsigned __int8 *v338; // [rsp+218h] [rbp-388h]
  __int64 v339; // [rsp+220h] [rbp-380h]
  __m128i v340; // [rsp+228h] [rbp-378h] BYREF
  unsigned __int64 v341; // [rsp+238h] [rbp-368h]
  __m128i v342; // [rsp+240h] [rbp-360h]
  __m128i v343; // [rsp+250h] [rbp-350h]
  unsigned __int64 *v344; // [rsp+260h] [rbp-340h] BYREF
  __int64 v345; // [rsp+268h] [rbp-338h]
  _BYTE v346[320]; // [rsp+270h] [rbp-330h] BYREF
  char v347; // [rsp+3B0h] [rbp-1F0h]
  int v348; // [rsp+3B4h] [rbp-1ECh]
  __int64 v349; // [rsp+3B8h] [rbp-1E8h]
  __m128i v350; // [rsp+3C0h] [rbp-1E0h] BYREF
  __int64 v351; // [rsp+3D0h] [rbp-1D0h] BYREF
  __m128i v352; // [rsp+3D8h] [rbp-1C8h] BYREF
  __int64 v353; // [rsp+3E8h] [rbp-1B8h]
  __m128i v354; // [rsp+3F0h] [rbp-1B0h] BYREF
  __m128i v355; // [rsp+400h] [rbp-1A0h] BYREF
  _BYTE *v356; // [rsp+410h] [rbp-190h] BYREF
  unsigned int v357; // [rsp+418h] [rbp-188h]
  _BYTE v358[320]; // [rsp+420h] [rbp-180h] BYREF
  char v359; // [rsp+560h] [rbp-40h]
  int v360; // [rsp+564h] [rbp-3Ch]
  __int64 v361; // [rsp+568h] [rbp-38h]

  v6 = **(_QWORD **)(a3 + 32);
  if ( !v6 || (*(_BYTE *)(v6 + 7) & 0x10) == 0 )
  {
    v7 = 14;
    v350.m128i_i64[0] = 14;
    v8 = "<unnamed loop>";
    v316 = dest;
    v9 = dest;
LABEL_4:
    memcpy(v9, v8, v7);
    v7 = v350.m128i_i64[0];
    goto LABEL_5;
  }
  v63 = sub_BD5D20(v6);
  v8 = v63;
  v7 = v64;
  v316 = dest;
  if ( &v63[v64] && !v63 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v350.m128i_i64[0] = v64;
  if ( v64 > 0xF )
  {
    v316 = (_QWORD *)sub_22409D0((__int64)&v316, (unsigned __int64 *)&v350, 0);
    v9 = v316;
    dest[0] = v350.m128i_i64[0];
    goto LABEL_4;
  }
  if ( v64 == 1 )
  {
    LOBYTE(dest[0]) = *v63;
  }
  else if ( v64 )
  {
    v9 = dest;
    goto LABEL_4;
  }
LABEL_5:
  v317 = v7;
  *((_BYTE *)v316 + v7) = 0;
  sub_1049690(v314, *(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL));
  v298 = a5[9];
  v300 = a5[3];
  v305 = (__int64 *)a5[4];
  v296 = a5[2];
  v10 = sub_D4B130(a3);
  if ( !v10 )
    goto LABEL_7;
  v291 = sub_D474B0(a3);
  if ( !v291 )
    goto LABEL_7;
  v26 = sub_D47600(a3);
  v27 = v26;
  if ( !v26 )
  {
    v12 = &v332;
    v332.m128i_i64[0] = (__int64)&v333;
    v332.m128i_i64[1] = 0x400000000LL;
    sub_D46D90(a3, (__int64)&v332);
    if ( (unsigned __int8)sub_D47630(a3) )
    {
      LOBYTE(v319) = 0;
      goto LABEL_105;
    }
    if ( (__int64 *)v332.m128i_i64[0] != &v333 )
    {
      _libc_free(v332.m128i_u64[0]);
LABEL_7:
      v297 = a5[9];
      v302 = a5[3];
      v306 = (__int64 *)a5[4];
      v301 = (unsigned __int64 *)a5[2];
      if ( !sub_D47930(a3) )
      {
LABEL_93:
        *(_QWORD *)(a1 + 48) = 0;
        *(_QWORD *)(a1 + 8) = a1 + 32;
        *(_QWORD *)(a1 + 56) = a1 + 80;
        *(_QWORD *)(a1 + 16) = 0x100000002LL;
        *(_QWORD *)(a1 + 64) = 2;
        *(_DWORD *)(a1 + 72) = 0;
        *(_BYTE *)(a1 + 76) = 1;
        *(_DWORD *)(a1 + 24) = 0;
        *(_BYTE *)(a1 + 28) = 1;
        *(_QWORD *)(a1 + 32) = &qword_4F82400;
        *(_QWORD *)a1 = 1;
        goto LABEL_94;
      }
      v299 = 0;
      goto LABEL_9;
    }
    v299 = 0;
LABEL_141:
    v297 = a5[9];
    v302 = a5[3];
    v306 = (__int64 *)a5[4];
    v301 = (unsigned __int64 *)a5[2];
    if ( !sub_D47930(a3) )
    {
LABEL_92:
      if ( v299 == 1 )
        goto LABEL_14;
      goto LABEL_93;
    }
LABEL_9:
    v11 = sub_DCF3A0(v306, (char *)a3, 1);
    if ( sub_D968A0(v11) || (v12 = (__m128i *)a3, v13 = sub_DCF3A0(v306, (char *)a3, 0), v14 = sub_D968A0(v13)) )
    {
LABEL_11:
      sub_F71210(a3, v301, (__int64)v306, (unsigned __int64 *)v302, v297);
      goto LABEL_12;
    }
    if ( sub_D96A50(v13) || (v12 = (__m128i *)v13, !(unsigned __int8)sub_DBE090((__int64)v306, v13)) )
    {
      v288 = qword_4FFEB08;
      if ( (_BYTE)qword_4FFEB08 )
      {
        v286 = sub_D47840(a3);
        v140 = sub_D47930(a3);
        v290 = v140;
        if ( v286 )
        {
          if ( v140 )
          {
            sub_D33BC0((__int64)&v327, a3);
            sub_D4E470(v327.m128i_i64, v302);
            v289 = sub_DFEF30((__int64)&v327, v302, v141, v142, v143, v144);
            if ( !v289 )
            {
              v145 = **(_QWORD **)(a3 + 32);
              v313[0] = (__int64)&v332;
              v332.m128i_i64[1] = (__int64)&v336;
              v333 = 0x100000004LL;
              v338 = &v340.m128i_u8[8];
              v313[1] = (__int64)&v319;
              v292 = v145;
              v319 = 0;
              v320 = 0;
              v321 = 0;
              v322 = 0;
              v334 = 0;
              v335 = 1;
              v336 = v145;
              v332.m128i_i64[0] = 1;
              v337 = 0;
              v339 = 4;
              v340.m128i_i32[0] = 0;
              v340.m128i_i8[4] = 1;
              v311 = v313;
              v323 = 0;
              v324 = 0;
              v325 = 0;
              v326 = 0;
              v350 = (__m128i)(unsigned __int64)sub_AA4E30(v145);
              v355.m128i_i16[0] = 257;
              v351 = 0;
              v352 = 0u;
              v353 = 0;
              v354 = 0u;
              v294 = v331;
              v285 = v330;
              if ( v331 != v330 )
              {
                while ( 1 )
                {
                  v150 = *(_QWORD *)(v294 - 8);
                  if ( !v340.m128i_i8[4] )
                    break;
                  v151 = v338;
                  v147 = HIDWORD(v339);
                  v146 = &v338[8 * HIDWORD(v339)];
                  if ( v338 == v146 )
                  {
LABEL_279:
                    if ( HIDWORD(v339) >= (unsigned int)v339 )
                      break;
                    v147 = (unsigned int)++HIDWORD(v339);
                    *(_QWORD *)v146 = v150;
                    ++v337;
                  }
                  else
                  {
                    while ( v150 != *(_QWORD *)v151 )
                    {
                      v151 += 8;
                      if ( v146 == v151 )
                        goto LABEL_279;
                    }
                  }
LABEL_254:
                  if ( v335 )
                  {
                    v152 = (_QWORD *)v332.m128i_i64[1];
                    v146 = (unsigned __int8 *)(v332.m128i_i64[1] + 8LL * HIDWORD(v333));
                    if ( (unsigned __int8 *)v332.m128i_i64[1] == v146 )
                      goto LABEL_268;
                    while ( v150 != *v152 )
                    {
                      if ( v146 == (unsigned __int8 *)++v152 )
                        goto LABEL_268;
                    }
                  }
                  else if ( !sub_C8CA60((__int64)&v332, v150) )
                  {
                    goto LABEL_268;
                  }
                  v147 = *(unsigned int *)(v302 + 24);
                  v153 = *(_QWORD *)(v302 + 8);
                  if ( !(_DWORD)v147 )
                    goto LABEL_262;
                  v147 = (unsigned int)(v147 - 1);
                  v154 = v147 & (((unsigned int)v150 >> 9) ^ ((unsigned int)v150 >> 4));
                  v155 = (__int64 *)(v153 + 16LL * v154);
                  v148 = *v155;
                  if ( v150 != *v155 )
                  {
                    v261 = 1;
                    while ( v148 != -4096 )
                    {
                      v262 = v261 + 1;
                      v154 = v147 & (v261 + v154);
                      v155 = (__int64 *)(v153 + 16LL * v154);
                      v148 = *v155;
                      if ( v150 == *v155 )
                        goto LABEL_261;
                      v261 = v262;
                    }
LABEL_262:
                    v146 = (unsigned __int8 *)(v150 + 48);
                    v156 = *(_QWORD *)(v150 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v156 != v150 + 48 )
                    {
                      if ( !v156 )
                        goto LABEL_312;
                      v157 = v156 - 24;
                      if ( (unsigned int)*(unsigned __int8 *)(v156 - 24) - 30 <= 0xA )
                      {
                        v158 = sub_B46E30(v157);
                        if ( v158 )
                        {
                          for ( i = 0; i != v158; ++i )
                          {
                            v160 = i;
                            v161 = (__int64 *)sub_B46EC0(v157, v160);
                            sub_2805C60(v311, v150, v161, v162, v163, v164);
                          }
                        }
                      }
                    }
                    goto LABEL_268;
                  }
LABEL_261:
                  if ( a3 != v155[1] )
                    goto LABEL_262;
                  v186 = sub_AA5930(v150);
                  v283 = v187;
                  v282 = v150 + 48;
                  if ( v186 != v187 )
                  {
                    while ( 1 )
                    {
                      v189 = *(_QWORD *)(v186 + 8);
                      if ( *(_BYTE *)(v189 + 8) != 12 )
                        goto LABEL_324;
                      v190 = *(_QWORD *)(v186 + 40);
                      if ( v292 != v190 )
                        break;
                      v221 = *(_QWORD *)(v186 - 8);
                      v222 = 0x1FFFFFFFE0LL;
                      if ( (*(_DWORD *)(v186 + 4) & 0x7FFFFFF) != 0 )
                      {
                        v223 = 0;
                        do
                        {
                          if ( v286 == *(_QWORD *)(v221 + 32LL * *(unsigned int *)(v186 + 72) + 8 * v223) )
                          {
                            v222 = 32 * v223;
                            goto LABEL_370;
                          }
                          ++v223;
                        }
                        while ( (*(_DWORD *)(v186 + 4) & 0x7FFFFFF) != (_DWORD)v223 );
                        v222 = 0x1FFFFFFFE0LL;
                      }
LABEL_370:
                      v193 = *(unsigned __int8 **)(v221 + v222);
                      if ( v193 )
                        goto LABEL_355;
LABEL_324:
                      v188 = *(_QWORD *)(v186 + 32);
                      if ( !v188 )
                        goto LABEL_477;
                      v186 = 0;
                      if ( *(_BYTE *)(v188 - 24) == 84 )
                        v186 = v188 - 24;
                      if ( v283 == v186 )
                        goto LABEL_376;
                    }
                    v191 = *(_QWORD *)(v190 + 16);
                    if ( !v191 )
                      goto LABEL_354;
                    while ( 1 )
                    {
                      v192 = *(_QWORD *)(v191 + 24);
                      if ( (unsigned __int8)(*(_BYTE *)v192 - 30) <= 0xAu )
                        break;
                      v191 = *(_QWORD *)(v191 + 8);
                      if ( !v191 )
                        goto LABEL_354;
                    }
                    v193 = 0;
                    v194 = v322 - 1;
                    v287 = ((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4);
                    v284 = v320;
                    v195 = v150;
                    v196 = v322;
                    v197 = v195;
                    while ( 1 )
                    {
                      v198 = *(_QWORD *)(v192 + 40);
                      if ( v196 )
                      {
                        v278 = v194;
                        v279 = v190;
                        v310 = v287;
                        v280 = v189;
                        v281 = v197;
                        LODWORD(v312) = ((unsigned int)v198 >> 9) ^ ((unsigned int)v198 >> 4);
                        v199 = sub_28052C0((unsigned int *)&v312, &v310);
                        v194 = v278;
                        v197 = v281;
                        v200 = 1;
                        v189 = v280;
                        v190 = v279;
                        for ( j = v278 & v199; ; j = v278 & v203 )
                        {
                          v202 = (_QWORD *)(v284 + 16LL * j);
                          if ( v198 == *v202 && v279 == v202[1] )
                            break;
                          if ( *v202 == -4096 && v202[1] == -4096 )
                            goto LABEL_348;
                          v203 = v200 + j;
                          ++v200;
                        }
                        v204 = *(_QWORD *)(v186 - 8);
                        v205 = 0x1FFFFFFFE0LL;
                        if ( (*(_DWORD *)(v186 + 4) & 0x7FFFFFF) != 0 )
                        {
                          v206 = 0;
                          do
                          {
                            if ( v198 == *(_QWORD *)(v204 + 32LL * *(unsigned int *)(v186 + 72) + 8 * v206) )
                            {
                              v205 = 32 * v206;
                              goto LABEL_344;
                            }
                            ++v206;
                          }
                          while ( (*(_DWORD *)(v186 + 4) & 0x7FFFFFF) != (_DWORD)v206 );
                          v205 = 0x1FFFFFFFE0LL;
                        }
LABEL_344:
                        v207 = *(unsigned __int8 **)(v204 + v205);
                        if ( *v207 != 13 )
                        {
                          if ( v193 && v193 != v207 )
                          {
                            v150 = v281;
                            goto LABEL_324;
                          }
                          v193 = v207;
                        }
                      }
LABEL_348:
                      v191 = *(_QWORD *)(v191 + 8);
                      if ( !v191 )
                        break;
                      while ( 1 )
                      {
                        v192 = *(_QWORD *)(v191 + 24);
                        if ( (unsigned __int8)(*(_BYTE *)v192 - 30) <= 0xAu )
                          break;
                        v191 = *(_QWORD *)(v191 + 8);
                        if ( !v191 )
                          goto LABEL_351;
                      }
                    }
LABEL_351:
                    v150 = v197;
                    if ( !v193 )
                    {
LABEL_354:
                      v193 = (unsigned __int8 *)sub_ACADE0((__int64 **)v189);
                      if ( !v193 )
                        goto LABEL_324;
                    }
LABEL_355:
                    v208 = *(_QWORD *)(v150 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v208 == v282 )
                    {
                      v212 = 0;
                      goto LABEL_359;
                    }
                    if ( v208 )
                    {
                      v209 = *(unsigned __int8 *)(v208 - 24);
                      v210 = v208 - 24;
                      v211 = (unsigned int)(v209 - 30) < 0xB;
                      v212 = 0;
                      if ( v211 )
                        v212 = v210;
LABEL_359:
                      if ( !(unsigned __int8)sub_B19DB0((__int64)v301, (__int64)v193, v212) )
                        goto LABEL_324;
                      v213 = sub_2805590(v193, (__int64)&v323, &v350);
                      v312 = v186;
                      v214 = v213;
                      if ( v326 )
                      {
                        v215 = 1;
                        v216 = 0;
                        v217 = (v326 - 1) & (((unsigned int)v186 >> 9) ^ ((unsigned int)v186 >> 4));
                        v218 = (_QWORD *)(v324 + 16LL * v217);
                        v219 = *v218;
                        if ( v186 == *v218 )
                        {
LABEL_362:
                          v220 = (unsigned __int8 **)(v218 + 1);
LABEL_363:
                          *v220 = v214;
                          goto LABEL_324;
                        }
                        while ( v219 != -4096 )
                        {
                          if ( !v216 && v219 == -8192 )
                            v216 = v218;
                          v217 = (v326 - 1) & (v215 + v217);
                          v218 = (_QWORD *)(v324 + 16LL * v217);
                          v219 = *v218;
                          if ( v186 == *v218 )
                            goto LABEL_362;
                          ++v215;
                        }
                        if ( v216 )
                          v218 = v216;
                      }
                      else
                      {
                        v218 = 0;
                      }
                      v257 = sub_FAA5E0((__int64)&v323, &v312, v218);
                      v258 = v312;
                      v220 = (unsigned __int8 **)(v257 + 1);
                      *v220 = 0;
                      *(v220 - 1) = (unsigned __int8 *)v258;
                      goto LABEL_363;
                    }
LABEL_477:
                    BUG();
                  }
LABEL_376:
                  v224 = *(_QWORD *)(v150 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v150 + 48 == v224 )
                    goto LABEL_477;
                  if ( !v224 )
                    goto LABEL_477;
                  v225 = v224 - 24;
                  if ( (unsigned int)*(unsigned __int8 *)(v224 - 24) - 30 > 0xA )
                    goto LABEL_477;
                  v226 = *(_BYTE *)(v224 - 24);
                  if ( v226 != 31 )
                  {
                    if ( v226 != 32 )
                      goto LABEL_381;
                    v146 = sub_2805590(**(unsigned __int8 ***)(v224 - 32), (__int64)&v323, &v350);
                    if ( *v146 != 17 )
                    {
                      v248 = *(_QWORD *)(v150 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v150 + 48 != v248 )
                      {
                        if ( !v248 )
LABEL_312:
                          BUG();
                        v249 = v248 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v248 - 24) - 30 <= 0xA )
                        {
                          v250 = sub_B46E30(v249);
                          if ( v250 )
                          {
                            for ( k = 0; k != v250; ++k )
                            {
                              v252 = k;
                              v253 = (__int64 *)sub_B46EC0(v249, v252);
                              sub_2805C60(v311, v150, v253, v254, v255, v256);
                            }
                          }
                        }
                      }
                      goto LABEL_268;
                    }
                    v234 = ((*(_DWORD *)(v224 - 20) & 0x7FFFFFFu) >> 1) - 1;
                    v235 = v234 >> 2;
                    if ( v234 >> 2 )
                    {
                      v236 = *(_QWORD *)(v224 - 32);
                      v235 *= 4;
                      v149 = 2;
                      v237 = 0;
                      while ( 1 )
                      {
                        v148 = v237 + 1;
                        v241 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)v149);
                        if ( v241 && v146 == v241 )
                        {
                          v148 = v237;
                          goto LABEL_398;
                        }
                        v238 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(v149 + 2));
                        if ( v238 && v146 == v238 )
                          goto LABEL_398;
                        v148 = v237 + 3;
                        v239 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(v149 + 4));
                        if ( v239 )
                        {
                          if ( v146 == v239 )
                            break;
                        }
                        v237 += 4;
                        v240 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(2 * v237));
                        if ( v240 && v146 == v240 )
                          goto LABEL_398;
                        v149 = (unsigned int)(v149 + 8);
                        if ( v237 == v235 )
                          goto LABEL_403;
                      }
                      v148 = v237 + 2;
LABEL_398:
                      v242 = v148;
                      if ( v148 != v234 )
                      {
LABEL_399:
                        v243 = v242 + 1;
                        goto LABEL_400;
                      }
LABEL_406:
                      v243 = 0;
                      goto LABEL_400;
                    }
                    v236 = *(_QWORD *)(v224 - 32);
LABEL_403:
                    v244 = v234 - v235;
                    if ( v234 - v235 == 2 )
                    {
                      v148 = v235;
                    }
                    else
                    {
                      if ( v244 != 3 )
                      {
                        if ( v244 != 1 )
                          goto LABEL_406;
LABEL_407:
                        v245 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(2 * v235 + 2));
                        if ( !v245 || v146 != v245 || v234 == v235 )
                          goto LABEL_406;
                        v242 = v235;
                        if ( v235 != 4294967294LL )
                          goto LABEL_399;
                        v243 = 0;
LABEL_400:
                        sub_2805C60(
                          v313,
                          v150,
                          *(__int64 **)(v236 + 32LL * (unsigned int)(2 * v243 + 1)),
                          v236,
                          v148,
                          v149);
                        goto LABEL_268;
                      }
                      v148 = v235 + 1;
                      v246 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(2 * (v235 + 1)));
                      if ( v246 && v146 == v246 )
                      {
LABEL_414:
                        v242 = v235;
                        if ( v235 != v234 )
                          goto LABEL_399;
                        v243 = 0;
                        goto LABEL_400;
                      }
                    }
                    v235 = v148 + 1;
                    v247 = *(unsigned __int8 **)(v236 + 32LL * (unsigned int)(2 * (v148 + 1)));
                    if ( !v247 || v146 != v247 )
                      goto LABEL_407;
                    v235 = v148;
                    goto LABEL_414;
                  }
                  if ( (*(_DWORD *)(v224 - 20) & 0x7FFFFFF) != 3
                    || (v263 = *(_QWORD *)(v224 - 120)) == 0
                    || (v264 = *(__int64 **)(v224 - 56)) == 0
                    || (v265 = *(__int64 **)(v224 - 88)) == 0 )
                  {
LABEL_381:
                    v227 = 0;
                    v228 = sub_B46E30(v225);
                    if ( v228 )
                    {
                      do
                      {
                        v229 = v227++;
                        v230 = (__int64 *)sub_B46EC0(v225, v229);
                        sub_2805C60(v311, v150, v230, v231, v232, v233);
                      }
                      while ( v228 != v227 );
                    }
                    goto LABEL_268;
                  }
                  if ( *(_BYTE *)v263 != 82 || *(_BYTE *)(*(_QWORD *)(v263 + 8) + 8LL) != 12 )
                  {
                    v266 = 0;
                    v267 = sub_B46E30(v225);
                    if ( v267 )
                    {
                      do
                      {
                        v268 = v266++;
                        v269 = (__int64 *)sub_B46EC0(v225, v268);
                        sub_2805C60(v311, v150, v269, v270, v271, v272);
                      }
                      while ( v267 != v266 );
                    }
                    goto LABEL_268;
                  }
                  v273 = sub_2805590((unsigned __int8 *)v263, (__int64)&v323, &v350);
                  if ( v273 == (unsigned __int8 *)v263 )
                    goto LABEL_468;
                  v275 = *v273;
                  if ( (unsigned int)(v275 - 12) > 1 )
                  {
                    if ( (_BYTE)v275 == 17 )
                    {
                      if ( !sub_AD7930(v273, (__int64)&v323, v275, v274, v148) )
                      {
                        sub_2805C60(v313, v150, v265, v147, v148, v149);
                        goto LABEL_268;
                      }
LABEL_472:
                      sub_2805C60(v313, v150, v264, v147, v148, v149);
                      goto LABEL_268;
                    }
LABEL_468:
                    sub_2805FF0(&v311, v150);
                    goto LABEL_268;
                  }
                  if ( *(_BYTE *)(a3 + 84) )
                  {
                    v276 = *(unsigned __int8 **)(a3 + 64);
                    v147 = *(unsigned int *)(a3 + 76);
                    v277 = &v276[8 * v147];
                    v146 = v276;
                    if ( v276 != v277 )
                    {
                      while ( v264 != *(__int64 **)v146 )
                      {
                        v146 += 8;
                        if ( v277 == v146 )
                          goto LABEL_268;
                      }
                      goto LABEL_463;
                    }
                  }
                  else if ( sub_C8CA60(a3 + 56, (__int64)v264) )
                  {
                    if ( *(_BYTE *)(a3 + 84) )
                    {
                      v276 = *(unsigned __int8 **)(a3 + 64);
                      v147 = *(unsigned int *)(a3 + 76);
LABEL_463:
                      v146 = &v276[8 * v147];
                      while ( v146 != v276 )
                      {
                        if ( v265 == *(__int64 **)v276 )
                          goto LABEL_472;
                        v276 += 8;
                      }
                      goto LABEL_268;
                    }
                    if ( sub_C8CA60(a3 + 56, (__int64)v265) )
                      goto LABEL_472;
                  }
LABEL_268:
                  v294 -= 8;
                  if ( v285 == v294 )
                    goto LABEL_269;
                }
                sub_C8CC70((__int64)&v337, *(_QWORD *)(v294 - 8), (__int64)v146, v147, v148, v149);
                goto LABEL_254;
              }
LABEL_269:
              if ( (_DWORD)v322 )
              {
                v165 = v322 - 1;
                v166 = v320;
                v310 = ((unsigned int)v292 >> 9) ^ ((unsigned int)v292 >> 4);
                LODWORD(v312) = ((unsigned int)v290 >> 9) ^ ((unsigned int)v290 >> 4);
                v167 = sub_28052C0((unsigned int *)&v312, &v310);
                v168 = 1;
                for ( m = v165 & v167; ; m = v165 & v171 )
                {
                  v170 = (_QWORD *)(v166 + 16LL * m);
                  if ( v290 == *v170 && v292 == v170[1] )
                    break;
                  if ( *v170 == -4096 && v170[1] == -4096 )
                    goto LABEL_430;
                  v171 = v168 + m;
                  ++v168;
                }
              }
              else
              {
LABEL_430:
                v289 = v288;
              }
              sub_C7D6A0(v324, 16LL * v326, 8);
              if ( !v340.m128i_i8[4] )
                _libc_free((unsigned __int64)v338);
              sub_C7D6A0(v320, 16LL * (unsigned int)v322, 8);
              if ( !v335 )
                _libc_free(v332.m128i_u64[1]);
              v14 = v289;
            }
            if ( v330 )
              j_j___libc_free_0(v330);
            v12 = (__m128i *)(16LL * v329);
            sub_C7D6A0(v328, (__int64)v12, 8);
            if ( v14 )
              goto LABEL_11;
          }
        }
      }
    }
    goto LABEL_92;
  }
  v28 = sub_AA4FF0(v26);
  if ( !v28 )
    goto LABEL_477;
  v29 = (unsigned int)*(unsigned __int8 *)(v28 - 24) - 39;
  if ( (unsigned int)v29 <= 0x38 )
  {
    v30 = 0x100060000000001LL;
    if ( _bittest64(&v30, v29) )
      goto LABEL_7;
  }
  v31 = sub_D4B130(a3);
  if ( !sub_AA5B70(v31) )
  {
    v32 = *(_QWORD *)(v31 + 16);
    if ( !v32 )
    {
LABEL_151:
      sub_DAC210((__int64)v305, a3);
      v90 = sub_AA5930(v27);
      v92 = v91;
      v93 = v90;
      while ( v92 != v93 )
      {
        v94 = sub_ACADE0(*(__int64 ***)(v93 + 8));
        if ( (*(_BYTE *)(v93 + 7) & 0x40) != 0 )
        {
          v95 = *(_QWORD *)(v93 - 8);
          v96 = v95 + 32LL * (*(_DWORD *)(v93 + 4) & 0x7FFFFFF);
        }
        else
        {
          v96 = v93;
          v95 = v93 - 32LL * (*(_DWORD *)(v93 + 4) & 0x7FFFFFF);
        }
        for ( ; v95 != v96; v95 += 32 )
        {
          if ( *(_QWORD *)v95 )
          {
            v97 = *(_QWORD *)(v95 + 8);
            **(_QWORD **)(v95 + 16) = v97;
            if ( v97 )
              *(_QWORD *)(v97 + 16) = *(_QWORD *)(v95 + 16);
          }
          *(_QWORD *)v95 = v94;
          if ( v94 )
          {
            v98 = *(_QWORD *)(v94 + 16);
            *(_QWORD *)(v95 + 8) = v98;
            if ( v98 )
              *(_QWORD *)(v98 + 16) = v95 + 8;
            *(_QWORD *)(v95 + 16) = v94 + 16;
            *(_QWORD *)(v94 + 16) = v95;
          }
        }
        v99 = *(_QWORD *)(v93 + 32);
        if ( !v99 )
          goto LABEL_477;
        v93 = 0;
        if ( *(_BYTE *)(v99 - 24) == 84 )
          v93 = v99 - 24;
      }
      v100 = v314[0];
      v101 = sub_B2BE50(v314[0]);
      if ( sub_B6EA50(v101)
        || (v259 = sub_B2BE50(v100),
            v260 = sub_B6F970(v259),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v260 + 48LL))(v260)) )
      {
        v106 = **(_QWORD **)(a3 + 32);
        sub_D4BD20(&v327, a3, v102, v103, v104, v105);
        sub_B157E0((__int64)&v332, &v327);
        sub_B17430((__int64)&v350, (__int64)"loop-delete", (__int64)"NeverExecutes", 13, &v332, v106);
        sub_B18290((__int64)&v350, "Loop deleted because it never executes", 0x26u);
        v109 = _mm_loadu_si128(&v352);
        v344 = (unsigned __int64 *)v346;
        LODWORD(v338) = v350.m128i_i32[2];
        v110 = _mm_loadu_si128(&v354);
        v111 = _mm_loadu_si128(&v355);
        v341 = v353;
        BYTE4(v338) = v350.m128i_i8[12];
        v345 = 0x400000000LL;
        v339 = v351;
        v337 = (char *)&unk_49D9D40;
        v340 = v109;
        v342 = v110;
        v343 = v111;
        if ( v357 )
        {
          sub_2805310((__int64)&v344, (__int64)&v356, v353, v357, v107, v108);
          v112 = v356;
          v347 = v359;
          v350.m128i_i64[0] = (__int64)&unk_49D9D40;
          v348 = v360;
          v349 = v361;
          v337 = (char *)&unk_49D9D78;
          if ( v356 != &v356[80 * v357] )
          {
            v183 = (unsigned __int64 *)v356;
            v184 = (unsigned __int64 *)&v356[80 * v357];
            do
            {
              v184 -= 10;
              v185 = v184[4];
              if ( (unsigned __int64 *)v185 != v184 + 6 )
                j_j___libc_free_0(v185);
              if ( (unsigned __int64 *)*v184 != v184 + 2 )
                j_j___libc_free_0(*v184);
            }
            while ( v183 != v184 );
            v112 = v356;
          }
        }
        else
        {
          v112 = v356;
          v347 = v359;
          v348 = v360;
          v349 = v361;
          v337 = (char *)&unk_49D9D78;
        }
        if ( v112 != v358 )
          _libc_free((unsigned __int64)v112);
        if ( v327.m128i_i64[0] )
          sub_B91220((__int64)&v327, v327.m128i_i64[0]);
        sub_1049740(v314, (__int64)&v337);
        v113 = v344;
        v337 = (char *)&unk_49D9D40;
        v114 = &v344[10 * (unsigned int)v345];
        if ( v344 != v114 )
        {
          do
          {
            v114 -= 10;
            v115 = v114[4];
            if ( (unsigned __int64 *)v115 != v114 + 6 )
              j_j___libc_free_0(v115);
            if ( (unsigned __int64 *)*v114 != v114 + 2 )
              j_j___libc_free_0(*v114);
          }
          while ( v113 != v114 );
          v113 = v344;
        }
        if ( v113 != (unsigned __int64 *)v346 )
          _libc_free((unsigned __int64)v113);
      }
      sub_F77B70(a3, v296, (__int64)v305, v300, v298);
      goto LABEL_12;
    }
    while ( 1 )
    {
      v33 = *(_QWORD *)(v32 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v33 - 30) <= 0xAu )
        break;
      v32 = *(_QWORD *)(v32 + 8);
      if ( !v32 )
        goto LABEL_151;
    }
LABEL_38:
    v34 = *(_QWORD *)(v33 + 40);
    v35 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v35 == v34 + 48 || !v35 || (unsigned int)*(unsigned __int8 *)(v35 - 24) - 30 > 0xA )
      goto LABEL_477;
    if ( *(_BYTE *)(v35 - 24) == 31 && (*(_DWORD *)(v35 - 20) & 0x7FFFFFF) == 3 )
    {
      v116 = *(_QWORD *)(v35 - 120);
      if ( *(_BYTE *)v116 == 17 )
      {
        v117 = *(_QWORD *)(v35 - 56);
        if ( v117 )
        {
          v118 = *(_QWORD *)(v35 - 88);
          if ( v118 )
          {
            v119 = *(_QWORD **)(v116 + 24);
            if ( *(_DWORD *)(v116 + 32) > 0x40u )
              v119 = (_QWORD *)*v119;
            if ( !v119 )
              v117 = v118;
            if ( v31 != v117 )
            {
              while ( 1 )
              {
                v32 = *(_QWORD *)(v32 + 8);
                if ( !v32 )
                  goto LABEL_151;
                v33 = *(_QWORD *)(v32 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v33 - 30) <= 0xAu )
                  goto LABEL_38;
              }
            }
          }
        }
      }
    }
  }
  v36 = v10;
  v12 = &v332;
  v332.m128i_i64[0] = (__int64)&v333;
  v332.m128i_i64[1] = 0x400000000LL;
  sub_D46D90(a3, (__int64)&v332);
  LOBYTE(v319) = 0;
  v37 = sub_AA5930(v27);
  v293 = v38;
  v39 = v37;
  if ( v37 == v38 )
    goto LABEL_105;
  do
  {
    v40 = (__int64 *)v332.m128i_i64[0];
    v41 = 0x1FFFFFFFE0LL;
    v42 = *(_QWORD *)(v39 - 8);
    v43 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
    if ( v43 )
    {
      v44 = 0;
      do
      {
        if ( *(_QWORD *)v332.m128i_i64[0] == *(_QWORD *)(v42 + 32LL * *(unsigned int *)(v39 + 72) + 8 * v44) )
        {
          v41 = 32 * v44;
          goto LABEL_49;
        }
        ++v44;
      }
      while ( v43 != (_DWORD)v44 );
      v41 = 0x1FFFFFFFE0LL;
    }
LABEL_49:
    v45 = *(unsigned __int8 **)(v42 + v41);
    v12 = (__m128i *)(v332.m128i_i64[0] + 8);
    v46 = 8LL * v332.m128i_u32[2] - 8;
    v47 = v332.m128i_i64[0] + 8LL * v332.m128i_u32[2];
    v48 = v46 >> 5;
    v49 = v46 >> 3;
    if ( v48 <= 0 )
      goto LABEL_71;
    v50 = v332.m128i_i64[0] + 32 * v48 + 8;
    while ( 1 )
    {
      if ( !v43 )
      {
        if ( v45 != *(unsigned __int8 **)(v42 + 0x1FFFFFFFE0LL) )
          goto LABEL_138;
LABEL_136:
        if ( v45 != *(unsigned __int8 **)(v42 + 0x1FFFFFFFE0LL) )
        {
LABEL_137:
          v12 = (__m128i *)((char *)v12 + 24);
          goto LABEL_138;
        }
        goto LABEL_69;
      }
      v51 = *(unsigned int *)(v39 + 72);
      v52 = 0;
      do
      {
        if ( v12->m128i_i64[0] == *(_QWORD *)(v42 + 32LL * *(unsigned int *)(v39 + 72) + 8 * v52) )
        {
          v53 = 32 * v52;
          goto LABEL_56;
        }
        ++v52;
      }
      while ( v43 != (_DWORD)v52 );
      v53 = 0x1FFFFFFFE0LL;
LABEL_56:
      if ( v45 != *(unsigned __int8 **)(v42 + v53) )
        goto LABEL_138;
      v54 = 0;
      do
      {
        if ( v12->m128i_i64[1] == *(_QWORD *)(v42 + 32 * v51 + 8 * v54) )
        {
          if ( *(unsigned __int8 **)(v42 + 32 * v54) == v45 )
            goto LABEL_61;
LABEL_145:
          v12 = (__m128i *)((char *)v12 + 8);
          if ( (__m128i *)v47 == v12 )
            goto LABEL_74;
LABEL_139:
          v299 = (unsigned __int8)v319;
          if ( v40 != &v333 )
            _libc_free((unsigned __int64)v40);
          goto LABEL_141;
        }
        ++v54;
      }
      while ( v43 != (_DWORD)v54 );
      if ( *(unsigned __int8 **)(v42 + 0x1FFFFFFFE0LL) != v45 )
        goto LABEL_145;
LABEL_61:
      v55 = 0;
LABEL_63:
      if ( v12[1].m128i_i64[0] == *(_QWORD *)(v42 + 32 * v51 + 8 * v55) )
      {
        if ( *(unsigned __int8 **)(v42 + 32 * v55) != v45 )
          break;
        goto LABEL_65;
      }
      if ( v43 != (_DWORD)++v55 )
        goto LABEL_63;
      if ( *(unsigned __int8 **)(v42 + 0x1FFFFFFFE0LL) != v45 )
        break;
LABEL_65:
      v56 = 0;
      v57 = v42 + 32 * v51;
      while ( *(_QWORD *)(v57 + 8 * v56) != v12[1].m128i_i64[1] )
      {
        if ( v43 == (_DWORD)++v56 )
          goto LABEL_136;
      }
      if ( v45 != *(unsigned __int8 **)(v42 + 32 * v56) )
        goto LABEL_137;
LABEL_69:
      v12 += 2;
      if ( (__m128i *)v50 == v12 )
      {
        v49 = (v47 - (__int64)v12) >> 3;
LABEL_71:
        if ( v49 != 2 )
        {
          if ( v49 != 3 )
          {
            if ( v49 != 1 )
              goto LABEL_74;
            goto LABEL_210;
          }
          v120 = 0x1FFFFFFFE0LL;
          if ( v43 )
          {
            v121 = 0;
            do
            {
              if ( v12->m128i_i64[0] == *(_QWORD *)(v42 + 32LL * *(unsigned int *)(v39 + 72) + 8 * v121) )
              {
                v120 = 32 * v121;
                goto LABEL_201;
              }
              ++v121;
            }
            while ( v43 != (_DWORD)v121 );
            v120 = 0x1FFFFFFFE0LL;
          }
LABEL_201:
          if ( v45 != *(unsigned __int8 **)(v42 + v120) )
            goto LABEL_138;
          v12 = (__m128i *)((char *)v12 + 8);
        }
        v122 = 0x1FFFFFFFE0LL;
        if ( v43 )
        {
          v123 = 0;
          do
          {
            if ( v12->m128i_i64[0] == *(_QWORD *)(v42 + 32LL * *(unsigned int *)(v39 + 72) + 8 * v123) )
            {
              v122 = 32 * v123;
              goto LABEL_208;
            }
            ++v123;
          }
          while ( v43 != (_DWORD)v123 );
          v122 = 0x1FFFFFFFE0LL;
        }
LABEL_208:
        if ( v45 != *(unsigned __int8 **)(v42 + v122) )
          goto LABEL_138;
        v12 = (__m128i *)((char *)v12 + 8);
LABEL_210:
        v124 = 0x1FFFFFFFE0LL;
        if ( v43 )
        {
          v125 = 0;
          do
          {
            if ( v12->m128i_i64[0] == *(_QWORD *)(v42 + 32LL * *(unsigned int *)(v39 + 72) + 8 * v125) )
            {
              v124 = 32 * v125;
              goto LABEL_215;
            }
            ++v125;
          }
          while ( v43 != (_DWORD)v125 );
          v124 = 0x1FFFFFFFE0LL;
        }
LABEL_215:
        if ( v45 == *(unsigned __int8 **)(v42 + v124) )
          goto LABEL_74;
LABEL_138:
        if ( (__m128i *)v47 == v12 )
          goto LABEL_74;
        goto LABEL_139;
      }
    }
    if ( (__m128i *)v47 != ++v12 )
      goto LABEL_139;
LABEL_74:
    if ( *v45 > 0x1Cu )
    {
      v58 = *(_QWORD *)(v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v58 == v36 + 48 )
      {
        v60 = 0;
      }
      else
      {
        if ( !v58 )
          goto LABEL_477;
        v59 = *(unsigned __int8 *)(v58 - 24);
        v60 = 0;
        v61 = v58 - 24;
        if ( (unsigned int)(v59 - 30) < 0xB )
          v60 = v61;
      }
      v12 = (__m128i *)v45;
      if ( !(unsigned __int8)sub_D4B180(a3, v45, &v319, v60, 0, (__int64)v305) )
      {
LABEL_243:
        v40 = (__int64 *)v332.m128i_i64[0];
        goto LABEL_139;
      }
    }
    v62 = *(_QWORD *)(v39 + 32);
    if ( !v62 )
      goto LABEL_477;
    v39 = 0;
    if ( *(_BYTE *)(v62 - 24) == 84 )
      v39 = v62 - 24;
  }
  while ( v293 != v39 );
LABEL_105:
  v67 = *(_QWORD *)(a3 + 32);
  v303 = *(_QWORD *)(a3 + 40);
  if ( v67 != v303 )
  {
    do
    {
      v68 = *(_QWORD *)(*(_QWORD *)v67 + 56LL);
      v69 = *(_QWORD *)v67 + 48LL;
      if ( v69 != v68 )
      {
        while ( 1 )
        {
          v70 = (unsigned __int8 *)(v68 - 24);
          if ( !v68 )
            v70 = 0;
          if ( (unsigned __int8)sub_B46970(v70) && !sub_BD2BE0((__int64)v70) )
            break;
          v68 = *(_QWORD *)(v68 + 8);
          if ( v69 == v68 )
            goto LABEL_114;
        }
        if ( v69 != v68 )
          goto LABEL_243;
      }
LABEL_114:
      v67 += 8;
    }
    while ( v303 != v67 );
    v303 = *(_QWORD *)(a3 + 32);
  }
  v71 = *(_QWORD *)(*(_QWORD *)v303 + 72LL);
  if ( !(unsigned __int8)sub_B2D610(v71, 19) )
  {
    v128 = sub_B2D610(v71, 76);
    if ( !v128 )
    {
      sub_D33BC0((__int64)&v337, a3);
      sub_D4E470((__int64 *)&v337, v300);
      v129 = (__int64 *)v300;
      if ( !(unsigned __int8)sub_DFEF30((__int64)&v337, v300, v130, v131, v132, v133) )
      {
        v351 = a3;
        v350.m128i_i64[0] = (__int64)&v351;
        v137 = &v351;
        v350.m128i_i64[1] = 0x800000001LL;
        for ( n = 1; ; n = v177 + v179 )
        {
          if ( !n )
          {
LABEL_237:
            v128 = v291;
            goto LABEL_238;
          }
          while ( 1 )
          {
            v139 = v137[n - 1];
            v350.m128i_i32[2] = n - 1;
            v128 = sub_D4A4A0(v139, (__int64)v129, n, v134, v135, v136);
            if ( !v128 )
              break;
            n = v350.m128i_u32[2];
            v137 = (__int64 *)v350.m128i_i64[0];
            if ( !v350.m128i_i32[2] )
              goto LABEL_237;
          }
          v129 = (__int64 *)v139;
          v175 = sub_DCF3A0(v305, (char *)v139, 1);
          if ( sub_D96A50(v175) )
            break;
          v135 = *(_QWORD *)(v139 + 16);
          v176 = *(__int64 **)(v139 + 8);
          v177 = v350.m128i_u32[2];
          v134 = v350.m128i_u32[3];
          v178 = v135 - (_QWORD)v176;
          v179 = (v135 - (__int64)v176) >> 3;
          v180 = v179 + v350.m128i_u32[2];
          if ( v180 > v350.m128i_u32[3] )
          {
            v129 = &v351;
            v304 = v135;
            sub_C8D5F0((__int64)&v350, &v351, v180, 8u, v135, v136);
            v177 = v350.m128i_u32[2];
            v135 = v304;
          }
          v137 = (__int64 *)v350.m128i_i64[0];
          if ( (__int64 *)v135 != v176 )
          {
            v129 = v176;
            memmove((void *)(v350.m128i_i64[0] + 8 * v177), v176, v178);
            LODWORD(v177) = v350.m128i_i32[2];
            v137 = (__int64 *)v350.m128i_i64[0];
          }
          v350.m128i_i32[2] = v177 + v179;
        }
        v137 = (__int64 *)v350.m128i_i64[0];
LABEL_238:
        if ( v137 != &v351 )
          _libc_free((unsigned __int64)v137);
      }
      if ( v341 )
        j_j___libc_free_0(v341);
      v12 = (__m128i *)(16LL * v340.m128i_u32[2]);
      sub_C7D6A0(v339, (__int64)v12, 8);
      if ( !v128 )
        goto LABEL_243;
    }
  }
  v72 = v314[0];
  v73 = sub_B2BE50(v314[0]);
  if ( sub_B6EA50(v73)
    || (v181 = sub_B2BE50(v72),
        v182 = sub_B6F970(v181),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v182 + 48LL))(v182)) )
  {
    v78 = **(_QWORD **)(a3 + 32);
    sub_D4BD20(&v323, a3, v74, v75, v76, v77);
    sub_B157E0((__int64)&v327, &v323);
    sub_B17430((__int64)&v350, (__int64)"loop-delete", (__int64)"Invariant", 9, &v327, v78);
    sub_B18290((__int64)&v350, "Loop deleted because it is invariant", 0x24u);
    v82 = _mm_loadu_si128(&v352);
    v344 = (unsigned __int64 *)v346;
    LODWORD(v338) = v350.m128i_i32[2];
    v341 = v353;
    v83 = _mm_loadu_si128(&v354);
    BYTE4(v338) = v350.m128i_i8[12];
    v84 = _mm_loadu_si128(&v355);
    v345 = 0x400000000LL;
    v339 = v351;
    v337 = (char *)&unk_49D9D40;
    v340 = v82;
    v342 = v83;
    v343 = v84;
    if ( v357 )
    {
      sub_2805310((__int64)&v344, (__int64)&v356, v357, v79, v80, v81);
      v347 = v359;
      v350.m128i_i64[0] = (__int64)&unk_49D9D40;
      v348 = v360;
      v349 = v361;
      v337 = (char *)&unk_49D9D78;
      v85 = &v356[80 * v357];
      if ( v356 != v85 )
      {
        v172 = (unsigned __int64 *)&v356[80 * v357];
        v173 = (unsigned __int64 *)v356;
        do
        {
          v172 -= 10;
          v174 = v172[4];
          if ( (unsigned __int64 *)v174 != v172 + 6 )
            j_j___libc_free_0(v174);
          if ( (unsigned __int64 *)*v172 != v172 + 2 )
            j_j___libc_free_0(*v172);
        }
        while ( v173 != v172 );
        v85 = v356;
      }
    }
    else
    {
      v85 = v356;
      v347 = v359;
      v348 = v360;
      v349 = v361;
      v337 = (char *)&unk_49D9D78;
    }
    if ( v85 != v358 )
      _libc_free((unsigned __int64)v85);
    if ( v323 )
      sub_B91220((__int64)&v323, v323);
    sub_1049740(v314, (__int64)&v337);
    v86 = v344;
    v337 = (char *)&unk_49D9D40;
    v87 = 10LL * (unsigned int)v345;
    v88 = &v344[v87];
    if ( v344 != &v344[v87] )
    {
      do
      {
        v88 -= 10;
        v89 = v88[4];
        if ( (unsigned __int64 *)v89 != v88 + 6 )
          j_j___libc_free_0(v89);
        if ( (unsigned __int64 *)*v88 != v88 + 2 )
          j_j___libc_free_0(*v88);
      }
      while ( v86 != v88 );
      v88 = v344;
    }
    if ( v88 != (unsigned __int64 *)v346 )
      _libc_free((unsigned __int64)v88);
  }
  sub_F77B70(a3, v296, (__int64)v305, v300, v298);
  if ( (__int64 *)v332.m128i_i64[0] != &v333 )
    _libc_free(v332.m128i_u64[0]);
LABEL_12:
  v12 = (__m128i *)a3;
  sub_22D0060(*(_QWORD *)(a6 + 8), a3, (__int64)v316, v317);
  if ( a3 == *(_QWORD *)(a6 + 16) )
    *(_BYTE *)(a6 + 24) = 1;
LABEL_14:
  sub_22D0390((__int64)&v350, (__int64)v12, v15, v16, v17, v18);
  if ( a5[9] )
  {
    if ( v355.m128i_i8[12] )
    {
      v21 = (__int64 **)(v354.m128i_i64[1] + 8LL * v355.m128i_u32[1]);
      v22 = v355.m128i_u32[1];
      if ( (__int64 **)v354.m128i_i64[1] == v21 )
      {
LABEL_217:
        v24 = v355.m128i_i32[2];
      }
      else
      {
        v23 = (void **)v354.m128i_i64[1];
        while ( *v23 != &unk_4F8F810 )
        {
          if ( v21 == (__int64 **)++v23 )
            goto LABEL_217;
        }
        --v355.m128i_i32[1];
        v21 = *(__int64 ***)(v354.m128i_i64[1] + 8LL * v355.m128i_u32[1]);
        *v23 = v21;
        v22 = v355.m128i_u32[1];
        ++v354.m128i_i64[0];
        v24 = v355.m128i_i32[2];
      }
    }
    else
    {
      v126 = sub_C8CA60((__int64)&v354, (__int64)&unk_4F8F810);
      if ( v126 )
      {
        *v126 = -2;
        ++v354.m128i_i64[0];
        v22 = v355.m128i_u32[1];
        v24 = ++v355.m128i_i32[2];
      }
      else
      {
        v22 = v355.m128i_u32[1];
        v24 = v355.m128i_i32[2];
      }
    }
    if ( v24 == (_DWORD)v22 )
    {
      if ( v352.m128i_i8[4] )
      {
        v25 = (void **)v350.m128i_i64[1];
        v127 = v350.m128i_i64[1] + 8LL * HIDWORD(v351);
        v22 = HIDWORD(v351);
        v21 = (__int64 **)v350.m128i_i64[1];
        if ( v350.m128i_i64[1] == v127 )
          goto LABEL_220;
        while ( *v21 != &qword_4F82400 )
        {
          if ( (__int64 **)v127 == ++v21 )
          {
LABEL_26:
            while ( *v25 != &unk_4F8F810 )
            {
              if ( v21 == (__int64 **)++v25 )
                goto LABEL_220;
            }
            break;
          }
        }
      }
      else if ( !sub_C8CA60((__int64)&v350, (__int64)&qword_4F82400) )
      {
        goto LABEL_22;
      }
    }
    else
    {
LABEL_22:
      if ( !v352.m128i_i8[4] )
        goto LABEL_222;
      v25 = (void **)v350.m128i_i64[1];
      v22 = HIDWORD(v351);
      v21 = (__int64 **)(v350.m128i_i64[1] + 8LL * HIDWORD(v351));
      if ( (__int64 **)v350.m128i_i64[1] != v21 )
        goto LABEL_26;
LABEL_220:
      if ( (unsigned int)v22 < (unsigned int)v351 )
      {
        HIDWORD(v351) = v22 + 1;
        *v21 = (__int64 *)&unk_4F8F810;
        ++v350.m128i_i64[0];
      }
      else
      {
LABEL_222:
        sub_C8CC70((__int64)&v350, (__int64)&unk_4F8F810, (__int64)v21, v22, v19, v20);
      }
    }
  }
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v352.m128i_i64[1], (__int64)&v350);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v356, (__int64)&v354);
  if ( !v355.m128i_i8[12] )
    _libc_free(v354.m128i_u64[1]);
  if ( !v352.m128i_i8[4] )
    _libc_free(v350.m128i_u64[1]);
LABEL_94:
  v65 = v315;
  if ( v315 )
  {
    sub_FDC110(v315);
    j_j___libc_free_0((unsigned __int64)v65);
  }
  if ( v316 != dest )
    j_j___libc_free_0((unsigned __int64)v316);
  return a1;
}
