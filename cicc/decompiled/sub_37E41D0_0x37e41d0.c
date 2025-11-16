// Function: sub_37E41D0
// Address: 0x37e41d0
//
__int64 __fastcall sub_37E41D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __m128i a7)
{
  __int64 v7; // r15
  __int64 (*v10)(void); // rdx
  __int64 v11; // rax
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 (*v16)(void); // rax
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  int v20; // edx
  int v21; // ecx
  unsigned __int64 v22; // r12
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r13
  _QWORD *v28; // rax
  _QWORD *v29; // rcx
  _BYTE *v30; // rsi
  _DWORD *v31; // rdx
  _DWORD *v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rcx
  __int64 v35; // r9
  __int64 v36; // rax
  unsigned __int64 v37; // r8
  __int64 *v38; // rbx
  __int64 *v39; // r14
  __int64 *v40; // r13
  __int64 *v41; // rbx
  __m128i *v42; // rbx
  __m128i *v43; // r12
  __int8 *v44; // r14
  __int64 v45; // rax
  __m128i *v46; // r15
  __m128i *v47; // rax
  char v48; // al
  __int64 v49; // rdx
  __int64 v50; // rdi
  const void *v51; // rsi
  size_t v52; // rdx
  __int64 v53; // r8
  __int64 v54; // r9
  __m128i *v55; // rax
  unsigned int v56; // r15d
  __m128i *v57; // r15
  __m128i *v58; // rax
  char v59; // al
  __int64 v60; // rdx
  __int64 v61; // rdi
  const void *v62; // rsi
  size_t v63; // rdx
  unsigned __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // r13
  unsigned __int64 *v67; // rax
  unsigned __int64 *i; // rdx
  unsigned int v69; // r12d
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rsi
  __int64 v77; // r9
  __int64 v78; // rax
  __int64 v79; // rdx
  void *v80; // rdi
  _QWORD **v81; // r8
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // r12
  int v85; // edx
  _QWORD **v86; // rdx
  __int64 v87; // rbx
  __int64 v88; // rax
  __int64 *v89; // rax
  __int64 v90; // r8
  __int64 v91; // rsi
  _QWORD *v92; // r9
  __int64 v93; // rax
  unsigned __int64 v94; // r13
  __int64 j; // rbx
  __int64 v96; // rax
  __int64 *v97; // r12
  int *v98; // r13
  __int64 *v99; // r8
  int v100; // esi
  unsigned int v101; // edi
  __int64 *v102; // rdx
  int v103; // r9d
  __int64 v104; // r14
  _QWORD *v105; // rax
  __int64 v106; // r8
  _QWORD *v107; // rbx
  int v108; // r10d
  __int64 *v109; // rsi
  unsigned int v110; // r15d
  __int64 v111; // rcx
  __int64 *v112; // rax
  _QWORD *v113; // rdx
  __int64 v114; // rsi
  __int64 v115; // r9
  __int64 v116; // r8
  int v117; // r10d
  unsigned int v118; // edi
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rcx
  __int64 v122; // rsi
  __int64 v123; // rdi
  int v124; // r10d
  __int64 *v125; // rdx
  unsigned int v126; // edi
  __int64 *v127; // rax
  _QWORD *v128; // rcx
  __int64 *v129; // rax
  char v130; // cl
  int v131; // eax
  int v132; // ecx
  unsigned int v133; // esi
  int v134; // eax
  int v135; // eax
  _QWORD *v136; // rax
  int v137; // edx
  _QWORD *v138; // rax
  __int64 v139; // rcx
  int v140; // r11d
  __int64 *v141; // rdi
  unsigned int v142; // edx
  int v143; // edi
  unsigned int v144; // r8d
  __int64 v145; // rdx
  unsigned __int64 v146; // r12
  unsigned __int64 v147; // rdi
  unsigned __int64 v148; // rdi
  unsigned __int64 v149; // rdi
  unsigned __int64 v150; // rdi
  unsigned __int64 v151; // rdi
  unsigned __int64 v152; // rdi
  unsigned __int64 v153; // r12
  __int64 v154; // rax
  __int64 v155; // r13
  unsigned __int64 *v156; // rbx
  unsigned __int64 *v157; // r14
  unsigned __int64 v158; // rdi
  __int64 v159; // rax
  __int64 v160; // rbx
  __int64 v161; // r14
  unsigned __int64 v162; // rdi
  __int64 v163; // rax
  __int64 v164; // rbx
  __int64 v165; // r13
  unsigned __int64 v166; // r14
  unsigned __int64 v167; // rdi
  unsigned __int64 v168; // rdi
  unsigned __int64 v169; // rdi
  __int64 v170; // rbx
  unsigned __int64 v171; // r14
  unsigned __int64 v172; // rdi
  int v173; // eax
  unsigned int v174; // ecx
  __int64 v175; // rdx
  _QWORD *v176; // rax
  _QWORD *k; // rdx
  int v178; // eax
  __int64 v179; // rsi
  _DWORD *v180; // rax
  _DWORD *m; // rdx
  int v182; // eax
  _QWORD *v183; // r14
  __int64 v184; // r13
  unsigned int v185; // edx
  _QWORD *v186; // rbx
  unsigned __int64 v187; // r12
  unsigned __int64 v188; // rdi
  unsigned __int64 v189; // rdi
  int v190; // eax
  __int64 v191; // rdx
  _QWORD *v192; // rax
  _QWORD *n; // rdx
  int v194; // eax
  __int64 v195; // rsi
  _QWORD *v196; // rax
  _QWORD *ii; // rdx
  __int64 v198; // rsi
  __int64 v199; // rdi
  __int64 v200; // rax
  __int64 v201; // rbx
  __int64 v202; // r12
  unsigned __int64 v203; // rdi
  __int64 v204; // rax
  _QWORD *v205; // r14
  _QWORD *v206; // rbx
  unsigned __int64 v207; // r12
  unsigned __int64 v208; // rdi
  unsigned __int64 v209; // rdi
  _QWORD **v210; // rbx
  _QWORD **v211; // r12
  unsigned __int64 *v212; // r13
  _QWORD ***v213; // rbx
  _QWORD ***v214; // r12
  unsigned __int64 *v215; // r13
  unsigned __int64 *v216; // rbx
  unsigned __int64 *v217; // r12
  __int64 *v218; // rbx
  __int64 *v219; // r12
  unsigned __int64 v220; // rdi
  _BYTE *v221; // rbx
  unsigned __int64 v222; // r12
  unsigned __int8 v224; // al
  int v225; // r11d
  __int64 *v226; // r10
  __int64 v227; // rdx
  int v228; // r11d
  __int64 *v229; // rdi
  __int64 v230; // rcx
  __int64 v231; // rsi
  int v232; // r11d
  __int64 v233; // rdi
  int v234; // r10d
  __int64 v235; // r15
  __int64 v236; // rdx
  __int64 v237; // r15
  int v238; // r10d
  __int64 v239; // rcx
  int v240; // r10d
  __int64 v241; // r15
  __int64 v242; // rsi
  __int64 v243; // rcx
  __int64 *v244; // rbx
  unsigned __int64 v245; // rdi
  __int64 (*v246)(); // rax
  __int64 v247; // rdi
  __int64 (*v248)(); // rax
  __int64 (*v249)(); // rax
  __int64 v250; // rdi
  const char *(*v251)(); // rax
  size_t v252; // rdx
  unsigned int v253; // edx
  unsigned int v254; // eax
  int v255; // r14d
  unsigned int v256; // eax
  unsigned int v257; // edx
  unsigned int v258; // eax
  int v259; // r12d
  unsigned int v260; // eax
  unsigned int v261; // ecx
  unsigned int v262; // eax
  int v263; // r14d
  unsigned int v264; // eax
  unsigned __int64 *v265; // r13
  unsigned __int64 *v266; // rbx
  _BYTE *v267; // r13
  _BYTE *v268; // rbx
  unsigned __int64 v269; // r12
  unsigned __int64 v270; // rdi
  unsigned __int64 v271; // rdi
  int v272; // edx
  int v273; // r14d
  unsigned int v274; // eax
  unsigned int v275; // eax
  __int64 v276; // rdx
  signed __int64 v277; // r13
  int v278; // r14d
  int v279; // r13d
  unsigned int v280; // eax
  int v281; // r14d
  unsigned int v282; // eax
  __int32 v283; // [rsp+10h] [rbp-3C10h]
  __int32 v284; // [rsp+10h] [rbp-3C10h]
  __int64 v287; // [rsp+20h] [rbp-3C00h]
  __int64 v288; // [rsp+28h] [rbp-3BF8h]
  __int64 v289; // [rsp+30h] [rbp-3BF0h]
  unsigned __int64 v290; // [rsp+38h] [rbp-3BE8h]
  unsigned int v292; // [rsp+4Ch] [rbp-3BD4h]
  __int64 v294; // [rsp+60h] [rbp-3BC0h]
  unsigned int v295; // [rsp+68h] [rbp-3BB8h]
  __int64 v296; // [rsp+68h] [rbp-3BB8h]
  __int64 v297; // [rsp+70h] [rbp-3BB0h]
  unsigned int v298; // [rsp+70h] [rbp-3BB0h]
  _QWORD *v299; // [rsp+78h] [rbp-3BA8h]
  __int64 *v300; // [rsp+80h] [rbp-3BA0h]
  unsigned int v301; // [rsp+80h] [rbp-3BA0h]
  unsigned __int8 v302; // [rsp+80h] [rbp-3BA0h]
  __int64 v303; // [rsp+88h] [rbp-3B98h]
  __int64 *v304; // [rsp+88h] [rbp-3B98h]
  int *v305; // [rsp+88h] [rbp-3B98h]
  __int64 v306; // [rsp+88h] [rbp-3B98h]
  int v307; // [rsp+88h] [rbp-3B98h]
  int v308; // [rsp+9Ch] [rbp-3B84h] BYREF
  _QWORD ***v309; // [rsp+A0h] [rbp-3B80h] BYREF
  int v310; // [rsp+A8h] [rbp-3B78h]
  _QWORD **v311; // [rsp+B0h] [rbp-3B70h] BYREF
  int v312; // [rsp+B8h] [rbp-3B68h]
  _QWORD v313[4]; // [rsp+C0h] [rbp-3B60h] BYREF
  __int64 v314; // [rsp+E0h] [rbp-3B40h] BYREF
  _QWORD *v315; // [rsp+E8h] [rbp-3B38h]
  __int64 v316; // [rsp+F0h] [rbp-3B30h]
  unsigned int v317; // [rsp+F8h] [rbp-3B28h]
  unsigned __int64 v318; // [rsp+100h] [rbp-3B20h] BYREF
  __int64 v319; // [rsp+108h] [rbp-3B18h]
  __int64 v320; // [rsp+110h] [rbp-3B10h]
  unsigned int v321; // [rsp+118h] [rbp-3B08h]
  __int64 v322; // [rsp+120h] [rbp-3B00h] BYREF
  __int64 v323; // [rsp+128h] [rbp-3AF8h] BYREF
  __int64 v324; // [rsp+130h] [rbp-3AF0h]
  __int64 v325; // [rsp+138h] [rbp-3AE8h]
  unsigned int v326; // [rsp+140h] [rbp-3AE0h]
  _BYTE *v327; // [rsp+178h] [rbp-3AA8h]
  __int64 v328; // [rsp+180h] [rbp-3AA0h]
  _BYTE v329[576]; // [rsp+188h] [rbp-3A98h] BYREF
  __int64 v330; // [rsp+3C8h] [rbp-3858h] BYREF
  char v331; // [rsp+3D0h] [rbp-3850h]
  __int64 v332; // [rsp+3D8h] [rbp-3848h]
  unsigned int v333; // [rsp+3E0h] [rbp-3840h]
  __int64 v334; // [rsp+458h] [rbp-37C8h]
  __int64 v335; // [rsp+460h] [rbp-37C0h]
  __int64 v336; // [rsp+468h] [rbp-37B8h]
  __int16 v337; // [rsp+470h] [rbp-37B0h]
  _BYTE *v338; // [rsp+480h] [rbp-37A0h] BYREF
  __int64 v339; // [rsp+488h] [rbp-3798h]
  _BYTE v340[2560]; // [rsp+490h] [rbp-3790h] BYREF
  unsigned __int64 *v341; // [rsp+E90h] [rbp-2D90h] BYREF
  __int64 v342; // [rsp+E98h] [rbp-2D88h]
  _BYTE v343[4736]; // [rsp+EA0h] [rbp-2D80h] BYREF
  char *v344; // [rsp+2120h] [rbp-1B00h] BYREF
  __int64 v345; // [rsp+2128h] [rbp-1AF8h]
  _BYTE v346[6896]; // [rsp+2130h] [rbp-1AF0h] BYREF

  v7 = (__int64)a1;
  if ( !sub_B92180(*(_QWORD *)a2) )
    return 0;
  a1[49] = a4;
  a1[1] = a3;
  a1[2] = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  a1[3] = *(_QWORD *)(a2 + 32);
  v10 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v11 = 0;
  if ( v10 != sub_2DAC790 )
    v11 = v10();
  a1[4] = v11;
  v12 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
  if ( v12 == sub_2DD19D0 )
  {
    a1[5] = 0;
    BUG();
  }
  v13 = v12();
  a1[5] = v13;
  (*(void (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v13 + 256LL))(v13, a2, a1 + 7);
  v299 = a1 + 16;
  a1[6] = *(_QWORD *)(a2 + 48);
  sub_35065A0((__int64 **)a1 + 16, (__int64 *)a2);
  v14 = *(_QWORD *)(a2 + 16);
  if ( !*(_BYTE *)(a1[6] + 65LL) )
    goto LABEL_6;
  v246 = *(__int64 (**)())(*(_QWORD *)v14 + 136LL);
  if ( v246 == sub_2DD19D0 )
    BUG();
  v247 = ((__int64 (__fastcall *)(_QWORD))v246)(*(_QWORD *)(a2 + 16));
  v248 = *(__int64 (**)())(*(_QWORD *)v247 + 160LL);
  if ( v248 != sub_2FDBC00 && ((unsigned __int8 (__fastcall *)(__int64))v248)(v247) )
  {
    *(_BYTE *)(v7 + 2360) = 1;
    v249 = *(__int64 (**)())(*(_QWORD *)v14 + 144LL);
    if ( v249 == sub_2C8F680 )
      BUG();
    v250 = ((__int64 (__fastcall *)(__int64))v249)(v14);
    v251 = *(const char *(**)())(*(_QWORD *)v250 + 984LL);
    if ( v251 == sub_2FE3320 )
    {
      *(_QWORD *)(v7 + 2376) = 0;
      *(_QWORD *)(v7 + 2368) = byte_3F871B3;
    }
    else
    {
      *(_QWORD *)(v7 + 2368) = ((__int64 (__fastcall *)(__int64, __int64))v251)(v250, a2);
      *(_QWORD *)(v7 + 2376) = v276;
    }
  }
  else
  {
LABEL_6:
    *(_BYTE *)(v7 + 2360) = 0;
  }
  v15 = 0;
  v16 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 144LL);
  if ( v16 != sub_2C8F680 )
    v15 = v16();
  v17 = sub_22077B0(0x378u);
  v18 = v17;
  if ( v17 )
    sub_37D3B80(v17, a2, *(_QWORD *)(v7 + 32), *(_QWORD *)(v7 + 16), v15);
  *(_QWORD *)(v7 + 408) = v18;
  v338 = v340;
  v339 = 0x2000000000LL;
  v341 = (unsigned __int64 *)v343;
  v344 = v346;
  *(_QWORD *)(v7 + 424) = 0;
  *(_QWORD *)(v7 + 432) = 0;
  v345 = 0x800000000LL;
  v342 = 0x800000000LL;
  v19 = *(_QWORD *)(a2 + 328);
  if ( v19 == a2 + 320 )
  {
    v292 = 0;
    v22 = 0;
    v295 = 0;
  }
  else
  {
    v20 = -1;
    do
    {
      v21 = *(_DWORD *)(v19 + 24);
      v19 = *(_QWORD *)(v19 + 8);
      if ( v20 < v21 )
        v20 = v21;
    }
    while ( v19 != a2 + 320 );
    v295 = v20 + 1;
    v22 = v20 + 1;
    v292 = v20 + 1;
  }
  sub_37DD9E0(v7, a2);
  v26 = (unsigned int)v339;
  if ( (unsigned int)v339 != v22 )
  {
    v27 = 80 * v22;
    if ( (unsigned int)v339 > v22 )
    {
      v267 = &v338[v27];
      v268 = &v338[80 * (unsigned int)v339];
      while ( v267 != v268 )
      {
        v268 -= 80;
        if ( (v268[8] & 1) == 0 )
          sub_C7D6A0(*((_QWORD *)v268 + 2), 16LL * *((unsigned int *)v268 + 6), 8);
      }
      LODWORD(v339) = v22;
    }
    else
    {
      if ( HIDWORD(v339) < v22 )
      {
        sub_37E3BE0((__int64)&v338, v22, HIDWORD(v339), v23, v24, v25);
        v26 = (unsigned int)v339;
      }
      v28 = &v338[80 * v26];
      v29 = v28 + 10;
      v30 = &v338[v27];
      v31 = v28 + 10;
      if ( v28 != (_QWORD *)&v338[v27] )
      {
        while ( 1 )
        {
          if ( v28 )
          {
            *v28 = 0;
            v32 = v28 + 2;
            *(v32 - 2) = 1;
            *(v32 - 1) = 0;
            do
            {
              if ( v32 )
                *v32 = -1;
              v32 += 4;
            }
            while ( v32 != v31 );
          }
          v28 = v29;
          v31 += 20;
          if ( v30 == (_BYTE *)v29 )
            break;
          v29 += 10;
        }
      }
      LODWORD(v339) = v22;
    }
  }
  v33 = *(_QWORD *)(v7 + 400);
  LOBYTE(v324) = v324 | 1;
  v289 = v7 + 2264;
  v322 = v7 + 2264;
  v287 = v7 + 2072;
  v323 = 0;
  sub_37BEA10((__int64)&v323);
  v331 |= 1u;
  v327 = v329;
  v328 = 0x800000000LL;
  v330 = 0;
  sub_37BEA60((__int64)&v330);
  v36 = (unsigned int)v345;
  v37 = 0;
  v334 = 0;
  v335 = v7 + 2072;
  v336 = v33;
  v337 = 0;
  if ( (unsigned int)v345 == v22 )
  {
LABEL_73:
    if ( (v331 & 1) == 0 )
      goto LABEL_389;
  }
  else
  {
    v38 = (__int64 *)v344;
    v39 = (__int64 *)&v344[856 * (unsigned int)v345];
    if ( (unsigned int)v345 <= v22 )
    {
      v40 = &v322;
      v34 = v22 - (unsigned int)v345;
      v297 = v34;
      if ( HIDWORD(v345) < v22 )
      {
        if ( v344 > (char *)&v322 || v39 <= &v322 )
        {
          v38 = (__int64 *)sub_C8D7D0((__int64)&v344, (__int64)v346, v22, 0x358u, &v318, v35);
          sub_37E3FC0((__int64 **)&v344, (__int64)v38);
          v279 = v318;
          if ( v344 != v346 )
            _libc_free((unsigned __int64)v344);
          HIDWORD(v345) = v279;
          v36 = (unsigned int)v345;
          v344 = (char *)v38;
          v40 = &v322;
        }
        else
        {
          v277 = (char *)&v322 - v344;
          v38 = (__int64 *)sub_C8D7D0((__int64)&v344, (__int64)v346, v22, 0x358u, &v318, v35);
          sub_37E3FC0((__int64 **)&v344, (__int64)v38);
          v278 = v318;
          if ( v344 != v346 )
            _libc_free((unsigned __int64)v344);
          v344 = (char *)v38;
          v36 = (unsigned int)v345;
          v40 = (__int64 *)((char *)v38 + v277);
          HIDWORD(v345) = v278;
        }
      }
      v288 = v7;
      v290 = v22;
      v41 = &v38[107 * v36];
      v35 = (__int64)(v41 + 103);
      v42 = (__m128i *)(v41 + 11);
      v303 = v297;
      v43 = (__m128i *)v35;
      while ( 1 )
      {
        v44 = &v42[-6].m128i_i8[8];
        if ( v42 != (__m128i *)88 )
          break;
LABEL_71:
        v43 = (__m128i *)((char *)v43 + 856);
        v42 = (__m128i *)((char *)v42 + 856);
        if ( !--v303 )
        {
          LODWORD(v345) = v297 + v345;
          v22 = v290;
          v7 = v288;
          goto LABEL_73;
        }
      }
      v45 = *v40;
      v42[-5].m128i_i64[0] = 0;
      v46 = v42 - 4;
      v42[-6].m128i_i64[1] = v45;
      v47 = v42 - 4;
      *((_DWORD *)v44 + 4) = 1;
      v42[-5].m128i_i32[3] = 0;
      do
      {
        if ( v47 )
          v47->m128i_i32[0] = -1;
        v47 = (__m128i *)((char *)v47 + 8);
      }
      while ( v47 != v42 );
      if ( (v44[16] & 1) == 0 )
        sub_C7D6A0(v42[-4].m128i_i64[0], 8LL * v42[-4].m128i_u32[2], 4);
      v48 = v44[16] | 1;
      v44[16] = v48;
      if ( (v40[2] & 1) == 0 && *((_DWORD *)v40 + 8) > 8u )
      {
        v44[16] = v48 & 0xFE;
        if ( (v40[2] & 1) != 0 )
        {
          v50 = 64;
          LODWORD(v49) = 8;
        }
        else
        {
          v49 = *((unsigned int *)v40 + 8);
          v50 = 8 * v49;
        }
        v283 = v49;
        v42[-4].m128i_i64[0] = sub_C7D670(v50, 4);
        v42[-4].m128i_i32[2] = v283;
      }
      *((_DWORD *)v44 + 4) = v40[2] & 0xFFFFFFFE | *((_DWORD *)v44 + 4) & 1;
      v42[-5].m128i_i32[3] = *((_DWORD *)v40 + 5);
      if ( (v44[16] & 1) == 0 )
        v46 = (__m128i *)v42[-4].m128i_i64[0];
      v51 = v40 + 3;
      if ( (v40[2] & 1) == 0 )
        v51 = (const void *)v40[3];
      v52 = 64;
      if ( (v44[16] & 1) == 0 )
        v52 = 8LL * v42[-4].m128i_u32[2];
      memcpy(v46, v51, v52);
      v55 = v42 + 1;
      v42->m128i_i32[2] = 0;
      v42->m128i_i64[0] = (__int64)v42[1].m128i_i64;
      v42->m128i_i32[3] = 8;
      v56 = *((_DWORD *)v40 + 24);
      if ( !v56 || v42 == (__m128i *)(v40 + 11) )
      {
LABEL_53:
        v42[37].m128i_i64[0] = 0;
        v57 = v42 + 38;
        *((_DWORD *)v44 + 172) = 1;
        v58 = v42 + 38;
        v42[37].m128i_i32[3] = 0;
        do
        {
          if ( v58 )
            v58->m128i_i32[0] = -1;
          ++v58;
        }
        while ( v58 != v43 );
        if ( (v44[688] & 1) == 0 )
          sub_C7D6A0(v42[38].m128i_i64[0], 16LL * v42[38].m128i_u32[2], 8);
        v59 = v44[688] | 1;
        v44[688] = v59;
        if ( (v40[86] & 1) == 0 && *((_DWORD *)v40 + 176) > 8u )
        {
          v44[688] = v59 & 0xFE;
          if ( (v40[86] & 1) != 0 )
          {
            v61 = 128;
            LODWORD(v60) = 8;
          }
          else
          {
            v60 = *((unsigned int *)v40 + 176);
            v61 = 16 * v60;
          }
          v284 = v60;
          v42[38].m128i_i64[0] = sub_C7D670(v61, 8);
          v42[38].m128i_i32[2] = v284;
        }
        *((_DWORD *)v44 + 172) = v40[86] & 0xFFFFFFFE | *((_DWORD *)v44 + 172) & 1;
        v42[37].m128i_i32[3] = *((_DWORD *)v40 + 173);
        if ( (v44[688] & 1) == 0 )
          v57 = (__m128i *)v42[38].m128i_i64[0];
        v62 = v40 + 87;
        if ( (v40[86] & 1) == 0 )
          v62 = (const void *)v40[87];
        v63 = 128;
        if ( (v44[688] & 1) == 0 )
          v63 = 16LL * v42[38].m128i_u32[2];
        memcpy(v57, v62, v63);
        v42[46].m128i_i64[0] = v40[103];
        v42[46].m128i_i64[1] = v40[104];
        v42[47] = _mm_loadu_si128((const __m128i *)(v40 + 105));
        goto LABEL_71;
      }
      if ( v56 > 8 )
      {
        sub_C8D5F0((__int64)v42, &v42[1], v56, 0x48u, v53, v54);
        v55 = (__m128i *)v42->m128i_i64[0];
        v252 = 72LL * *((unsigned int *)v40 + 24);
        if ( !v252 )
          goto LABEL_411;
      }
      else
      {
        v252 = 72LL * v56;
      }
      memcpy(v55, (const void *)v40[11], v252);
LABEL_411:
      v42->m128i_i32[2] = v56;
      goto LABEL_53;
    }
    v244 = (__int64 *)&v344[856 * v22];
    while ( v244 != v39 )
    {
      while ( 1 )
      {
        v39 -= 107;
        if ( (v39[86] & 1) == 0 )
          sub_C7D6A0(v39[87], 16LL * *((unsigned int *)v39 + 176), 8);
        v245 = v39[11];
        if ( (__int64 *)v245 != v39 + 13 )
          _libc_free(v245);
        if ( (v39[2] & 1) != 0 )
          break;
        sub_C7D6A0(v39[3], 8LL * *((unsigned int *)v39 + 8), 4);
        if ( v244 == v39 )
          goto LABEL_399;
      }
    }
LABEL_399:
    LODWORD(v345) = v22;
    if ( (v331 & 1) == 0 )
    {
LABEL_389:
      sub_C7D6A0(v332, 16LL * v333, 8);
      v64 = (unsigned __int64)v327;
      if ( v327 == v329 )
        goto LABEL_76;
      goto LABEL_75;
    }
  }
  v64 = (unsigned __int64)v327;
  if ( v327 != v329 )
LABEL_75:
    _libc_free(v64);
LABEL_76:
  if ( (v324 & 1) == 0 )
    sub_C7D6A0(v325, 8LL * v326, 4);
  v65 = (unsigned int)v342;
  if ( (unsigned int)v342 != v22 )
  {
    v66 = 74 * v22;
    if ( (unsigned int)v342 > v22 )
    {
      v265 = &v341[v66];
      v266 = &v341[74 * (unsigned int)v342];
      while ( v265 != v266 )
      {
        v266 -= 74;
        if ( (unsigned __int64 *)*v266 != v266 + 2 )
          _libc_free(*v266);
      }
    }
    else
    {
      if ( HIDWORD(v342) < v22 )
      {
        sub_37C5040((__int64)&v341, v22, HIDWORD(v342), v34, v37, v35);
        v65 = (unsigned int)v342;
      }
      v67 = &v341[74 * v65];
      for ( i = &v341[v66]; i != v67; v67 += 74 )
      {
        if ( v67 )
        {
          *((_DWORD *)v67 + 2) = 0;
          *v67 = (unsigned __int64)(v67 + 2);
          *((_DWORD *)v67 + 3) = 8;
        }
      }
    }
    LODWORD(v342) = v22;
  }
  sub_37E0750(v7, a2, &v338, v292, v37, v35);
  v69 = *(_DWORD *)(*(_QWORD *)(v7 + 408) + 40LL);
  sub_37BD1C0((__int64)&v309, v295, v69, v70, v71, v72);
  sub_37BD1C0((__int64)&v311, v295, v69, v73, v74, v75);
  v76 = a2;
  sub_37D6250(v7, a2, &v311, &v309, &v338, v77);
  v78 = *(_QWORD *)(v7 + 776);
  v79 = 40LL * *(unsigned int *)(v7 + 784);
  v80 = (void *)(v78 + v79);
  if ( v78 != v78 + v79 )
  {
    v81 = v311;
    v76 = qword_5051170;
    do
    {
      if ( *(_BYTE *)(v78 + 24) )
      {
        if ( (*(_QWORD *)(v78 + 16) & 0xFFFFF00000LL) == 0 )
        {
          v82 = *(_QWORD *)(*v81[*(_DWORD *)(v78 + 16) & 0xFFFFF] + 8LL * (*(_DWORD *)(v78 + 20) >> 8));
          if ( v82 != v76 )
            *(_QWORD *)(v78 + 16) = v82;
        }
      }
      v78 += 40;
    }
    while ( v80 != (void *)v78 );
    v80 = *(void **)(v7 + 776);
    v79 = 40LL * *(unsigned int *)(v7 + 784);
  }
  if ( (unsigned __int64)v79 > 0x28 )
  {
    v76 = 0xCCCCCCCCCCCCCCCDLL * (v79 >> 3);
    qsort(v80, v76, 0x28u, (__compar_fn_t)sub_37B5D20);
  }
  v83 = *(unsigned int *)(v7 + 608);
  v84 = *(_QWORD *)(v7 + 600);
  v85 = *(_DWORD *)(v7 + 608);
  if ( v84 + 8 * v83 != v84 )
  {
    v300 = (__int64 *)(v84 + 8 * v83);
    v304 = *(__int64 **)(v7 + 600);
    do
    {
      v86 = v311;
      v87 = *v304;
      v88 = *(unsigned int *)(*v304 + 24);
      *(_DWORD *)(v7 + 416) = v88;
      v89 = (__int64 *)&v344[856 * v88];
      *(_QWORD *)(v7 + 424) = v89;
      v89[103] = v87;
      v90 = *(_QWORD *)(v7 + 408);
      v91 = *(unsigned int *)(v90 + 40);
      v92 = v86[*(int *)(v87 + 24)];
      *(_DWORD *)(v90 + 280) = *(_DWORD *)(v7 + 416);
      v93 = 0;
      LODWORD(v86) = v91;
      v76 = 8 * v91;
      if ( (_DWORD)v86 )
      {
        do
        {
          *(_QWORD *)(*(_QWORD *)(v90 + 32) + v93) = *(_QWORD *)(*v92 + v93);
          v93 += 8;
        }
        while ( v93 != v76 );
      }
      *(_DWORD *)(v7 + 420) = 1;
      v94 = *(_QWORD *)(v87 + 56);
      for ( j = v87 + 48; j != v94; v94 = *(_QWORD *)(v94 + 8) )
      {
        while ( 1 )
        {
          v76 = v94;
          sub_37E06C0(v7, v94, &v309, &v311);
          ++*(_DWORD *)(v7 + 420);
          if ( !v94 )
            BUG();
          if ( (*(_BYTE *)v94 & 4) == 0 )
            break;
          v94 = *(_QWORD *)(v94 + 8);
          if ( j == v94 )
            goto LABEL_107;
        }
        while ( (*(_BYTE *)(v94 + 44) & 8) != 0 )
          v94 = *(_QWORD *)(v94 + 8);
      }
LABEL_107:
      ++v304;
      *(_DWORD *)(*(_QWORD *)(v7 + 408) + 304LL) = 0;
    }
    while ( v300 != v304 );
    v85 = *(_DWORD *)(v7 + 608);
  }
  v314 = 0;
  v96 = 0;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  v318 = 0;
  v319 = 0;
  v320 = 0;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  LODWORD(v325) = 0;
  v298 = 0;
  v301 = 0;
  if ( !v85 )
    goto LABEL_340;
  v296 = v7;
  do
  {
    v97 = (__int64 *)&v344[856 * *(int *)(*(_QWORD *)(*(_QWORD *)(v296 + 600) + 8 * v96) + 24LL)];
    v98 = (int *)v97[11];
    v305 = &v98[18 * *((unsigned int *)v97 + 24)];
    if ( v305 != v98 )
    {
      v294 = (__int64)(v97 + 85);
      do
      {
        v130 = *((_BYTE *)v97 + 688);
        v131 = *v98;
        v308 = *v98;
        v132 = v130 & 1;
        if ( v132 )
        {
          v99 = v97 + 87;
          v100 = 7;
        }
        else
        {
          v133 = *((_DWORD *)v97 + 176);
          v99 = (__int64 *)v97[87];
          if ( !v133 )
          {
            v313[0] = 0;
            v142 = *((_DWORD *)v97 + 172);
            ++v97[85];
            v143 = (v142 >> 1) + 1;
LABEL_174:
            v144 = 3 * v133;
            goto LABEL_175;
          }
          v100 = v133 - 1;
        }
        v101 = v100 & (37 * v131);
        v102 = &v99[2 * v101];
        v103 = *(_DWORD *)v102;
        if ( v131 == *(_DWORD *)v102 )
        {
LABEL_115:
          v104 = v102[1];
          goto LABEL_116;
        }
        v225 = 1;
        v226 = 0;
        while ( v103 != -1 )
        {
          if ( !v226 && v103 == -2 )
            v226 = v102;
          v101 = v100 & (v225 + v101);
          v102 = &v99[2 * v101];
          v103 = *(_DWORD *)v102;
          if ( v131 == *(_DWORD *)v102 )
            goto LABEL_115;
          ++v225;
        }
        v144 = 24;
        v133 = 8;
        if ( !v226 )
          v226 = v102;
        v313[0] = v226;
        v142 = *((_DWORD *)v97 + 172);
        ++v97[85];
        v143 = (v142 >> 1) + 1;
        if ( !(_BYTE)v132 )
        {
          v133 = *((_DWORD *)v97 + 176);
          goto LABEL_174;
        }
LABEL_175:
        if ( v144 <= 4 * v143 )
        {
          v133 *= 2;
LABEL_344:
          sub_37C5F80(v294, v133);
          sub_37BDA60(v294, &v308, v313);
          v131 = v308;
          v142 = *((_DWORD *)v97 + 172);
          goto LABEL_177;
        }
        if ( v133 - *((_DWORD *)v97 + 173) - v143 <= v133 >> 3 )
          goto LABEL_344;
LABEL_177:
        *((_DWORD *)v97 + 172) = (2 * (v142 >> 1) + 2) | v142 & 1;
        v145 = v313[0];
        if ( *(_DWORD *)v313[0] != -1 )
          --*((_DWORD *)v97 + 173);
        *(_DWORD *)v145 = v131;
        v104 = 0;
        *(_QWORD *)(v145 + 8) = 0;
LABEL_116:
        v105 = sub_35051D0(v299, v104);
        v106 = v317;
        v107 = v105;
        if ( !v317 )
        {
          ++v314;
          goto LABEL_352;
        }
        v108 = 1;
        v109 = 0;
        v110 = ((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4);
        v111 = (v317 - 1) & v110;
        v112 = &v315[11 * v111];
        v113 = (_QWORD *)*v112;
        if ( v107 == (_QWORD *)*v112 )
          goto LABEL_118;
        while ( 1 )
        {
          if ( v113 == (_QWORD *)-4096LL )
          {
            if ( !v109 )
              v109 = v112;
            ++v314;
            v135 = v316 + 1;
            if ( 4 * ((int)v316 + 1) < 3 * v317 )
            {
              v111 = v317 >> 3;
              if ( v317 - HIDWORD(v316) - v135 > (unsigned int)v111 )
              {
LABEL_148:
                LODWORD(v316) = v135;
                if ( *v109 != -4096 )
                  --HIDWORD(v316);
                a7 = 0;
                *v109 = (__int64)v107;
                v109[1] = (__int64)(v109 + 3);
                v109[2] = 0x400000000LL;
                v136 = v109 + 6;
                v114 = (__int64)(v109 + 1);
                *(_OWORD *)(v114 + 32) = 0;
                *(_OWORD *)(v114 + 16) = 0;
                *(_DWORD *)(v114 + 40) = 0;
                *(_QWORD *)(v114 + 48) = 0;
                *(_QWORD *)(v114 + 56) = v136;
                *(_QWORD *)(v114 + 64) = v136;
                *(_QWORD *)(v114 + 72) = 0;
                goto LABEL_119;
              }
              sub_37C8550((__int64)&v314, v317);
              if ( v317 )
              {
                v234 = 1;
                v111 = 0;
                LODWORD(v235) = (v317 - 1) & v110;
                v109 = &v315[11 * (unsigned int)v235];
                v236 = *v109;
                v135 = v316 + 1;
                if ( v107 != (_QWORD *)*v109 )
                {
                  while ( v236 != -4096 )
                  {
                    if ( !v111 && v236 == -8192 )
                      v111 = (__int64)v109;
                    v106 = (unsigned int)(v234 + 1);
                    v235 = (v317 - 1) & ((_DWORD)v235 + v234);
                    v109 = &v315[11 * v235];
                    v236 = *v109;
                    if ( v107 == (_QWORD *)*v109 )
                      goto LABEL_148;
                    ++v234;
                  }
                  if ( v111 )
                    v109 = (__int64 *)v111;
                }
                goto LABEL_148;
              }
LABEL_557:
              LODWORD(v316) = v316 + 1;
              BUG();
            }
LABEL_352:
            sub_37C8550((__int64)&v314, 2 * v317);
            if ( v317 )
            {
              v106 = (__int64)v315;
              LODWORD(v227) = (v317 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
              v109 = &v315[11 * (unsigned int)v227];
              v111 = *v109;
              v135 = v316 + 1;
              if ( v107 != (_QWORD *)*v109 )
              {
                v228 = 1;
                v229 = 0;
                while ( v111 != -4096 )
                {
                  if ( v111 == -8192 && !v229 )
                    v229 = v109;
                  v227 = (v317 - 1) & ((_DWORD)v227 + v228);
                  v109 = &v315[11 * v227];
                  v111 = *v109;
                  if ( v107 == (_QWORD *)*v109 )
                    goto LABEL_148;
                  ++v228;
                }
                if ( v229 )
                  v109 = v229;
              }
              goto LABEL_148;
            }
            goto LABEL_557;
          }
          if ( v109 || v113 != (_QWORD *)-8192LL )
            v112 = v109;
          v111 = (v317 - 1) & (v108 + (_DWORD)v111);
          v113 = (_QWORD *)v315[11 * (unsigned int)v111];
          if ( v107 == v113 )
            break;
          ++v108;
          v109 = v112;
          v112 = &v315[11 * (unsigned int)v111];
        }
        v112 = &v315[11 * (unsigned int)v111];
LABEL_118:
        v114 = (__int64)(v112 + 1);
LABEL_119:
        sub_2B5C0F0((__int64)v313, v114, (unsigned int *)&v308, v111, v106);
        if ( !v321 )
        {
          ++v318;
          goto LABEL_360;
        }
        v115 = v321 - 1;
        v116 = v319;
        v117 = 1;
        v118 = v115 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
        v119 = v319 + 72LL * v118;
        v120 = 0;
        v121 = *(_QWORD *)v119;
        if ( v107 == *(_QWORD **)v119 )
          goto LABEL_121;
        while ( 2 )
        {
          if ( v121 == -4096 )
          {
            if ( !v120 )
              v120 = v119;
            ++v318;
            v137 = v320 + 1;
            if ( 4 * ((int)v320 + 1) < 3 * v321 )
            {
              if ( v321 - HIDWORD(v320) - v137 > v321 >> 3 )
              {
LABEL_157:
                LODWORD(v320) = v137;
                if ( *(_QWORD *)v120 != -4096 )
                  --HIDWORD(v320);
                *(_QWORD *)v120 = v107;
                v123 = v120 + 8;
                *(_QWORD *)(v120 + 8) = 0;
                *(_QWORD *)(v120 + 16) = v120 + 40;
                *(_QWORD *)(v120 + 24) = 4;
                *(_DWORD *)(v120 + 32) = 0;
                *(_BYTE *)(v120 + 36) = 1;
                v122 = v97[103];
                goto LABEL_160;
              }
              sub_37C8820((__int64)&v318, v321);
              if ( v321 )
              {
                v115 = v321 - 1;
                v240 = 1;
                LODWORD(v241) = v115 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
                v137 = v320 + 1;
                v242 = 0;
                v120 = v319 + 72LL * (unsigned int)v241;
                v243 = *(_QWORD *)v120;
                if ( v107 != *(_QWORD **)v120 )
                {
                  while ( v243 != -4096 )
                  {
                    if ( v243 == -8192 && !v242 )
                      v242 = v120;
                    v116 = (unsigned int)(v240 + 1);
                    v241 = (unsigned int)v115 & ((_DWORD)v241 + v240);
                    v120 = v319 + 72 * v241;
                    v243 = *(_QWORD *)v120;
                    if ( v107 == *(_QWORD **)v120 )
                      goto LABEL_157;
                    ++v240;
                  }
                  if ( v242 )
                    v120 = v242;
                }
                goto LABEL_157;
              }
LABEL_553:
              LODWORD(v320) = v320 + 1;
              BUG();
            }
LABEL_360:
            sub_37C8820((__int64)&v318, 2 * v321);
            if ( v321 )
            {
              v116 = v319;
              LODWORD(v230) = (v321 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
              v137 = v320 + 1;
              v120 = v319 + 72LL * (unsigned int)v230;
              v231 = *(_QWORD *)v120;
              if ( v107 != *(_QWORD **)v120 )
              {
                v232 = 1;
                v233 = 0;
                while ( v231 != -4096 )
                {
                  if ( v231 == -8192 && !v233 )
                    v233 = v120;
                  v115 = (unsigned int)(v232 + 1);
                  v230 = (v321 - 1) & ((_DWORD)v230 + v232);
                  v120 = v319 + 72 * v230;
                  v231 = *(_QWORD *)v120;
                  if ( v107 == *(_QWORD **)v120 )
                    goto LABEL_157;
                  ++v232;
                }
                if ( v233 )
                  v120 = v233;
              }
              goto LABEL_157;
            }
            goto LABEL_553;
          }
          if ( v121 != -8192 || v120 )
            v119 = v120;
          v118 = v115 & (v117 + v118);
          v121 = *(_QWORD *)(v319 + 72LL * v118);
          if ( v107 != (_QWORD *)v121 )
          {
            ++v117;
            v120 = v119;
            v119 = v319 + 72LL * v118;
            continue;
          }
          break;
        }
        v119 = v319 + 72LL * v118;
LABEL_121:
        v122 = v97[103];
        v123 = v119 + 8;
        if ( !*(_BYTE *)(v119 + 36) )
        {
LABEL_122:
          sub_C8CC70(v123, v122, v119, v121, v116, v115);
          goto LABEL_123;
        }
LABEL_160:
        v138 = *(_QWORD **)(v123 + 8);
        v121 = *(unsigned int *)(v123 + 20);
        v119 = (__int64)&v138[v121];
        if ( v138 == (_QWORD *)v119 )
        {
LABEL_163:
          if ( (unsigned int)v121 >= *(_DWORD *)(v123 + 16) )
            goto LABEL_122;
          *(_DWORD *)(v123 + 20) = v121 + 1;
          *(_QWORD *)v119 = v122;
          ++*(_QWORD *)v123;
          v76 = (unsigned int)v325;
          if ( (_DWORD)v325 )
            goto LABEL_124;
LABEL_165:
          ++v322;
          goto LABEL_166;
        }
        while ( *v138 != v122 )
        {
          if ( (_QWORD *)v119 == ++v138 )
            goto LABEL_163;
        }
LABEL_123:
        v76 = (unsigned int)v325;
        if ( !(_DWORD)v325 )
          goto LABEL_165;
LABEL_124:
        v124 = 1;
        v125 = 0;
        v126 = (v76 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
        v127 = (__int64 *)(v323 + 16LL * v126);
        v128 = (_QWORD *)*v127;
        if ( v107 == (_QWORD *)*v127 )
          goto LABEL_125;
        while ( 2 )
        {
          if ( v128 == (_QWORD *)-4096LL )
          {
            if ( !v125 )
              v125 = v127;
            ++v322;
            v134 = v324 + 1;
            if ( 4 * ((int)v324 + 1) < (unsigned int)(3 * v76) )
            {
              if ( (int)v76 - HIDWORD(v324) - v134 > (unsigned int)v76 >> 3 )
              {
LABEL_139:
                LODWORD(v324) = v134;
                if ( *v125 != -4096 )
                  --HIDWORD(v324);
                *v125 = (__int64)v107;
                v129 = v125 + 1;
                v125[1] = 0;
                goto LABEL_126;
              }
              sub_37C8A20((__int64)&v322, v76);
              if ( (_DWORD)v325 )
              {
                v76 = 0;
                LODWORD(v237) = (v325 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
                v238 = 1;
                v134 = v324 + 1;
                v125 = (__int64 *)(v323 + 16LL * (unsigned int)v237);
                v239 = *v125;
                if ( v107 != (_QWORD *)*v125 )
                {
                  while ( v239 != -4096 )
                  {
                    if ( !v76 && v239 == -8192 )
                      v76 = (__int64)v125;
                    v237 = ((_DWORD)v325 - 1) & (unsigned int)(v237 + v238);
                    v125 = (__int64 *)(v323 + 16 * v237);
                    v239 = *v125;
                    if ( v107 == (_QWORD *)*v125 )
                      goto LABEL_139;
                    ++v238;
                  }
                  if ( v76 )
                    v125 = (__int64 *)v76;
                }
                goto LABEL_139;
              }
LABEL_558:
              LODWORD(v324) = v324 + 1;
              BUG();
            }
LABEL_166:
            sub_37C8A20((__int64)&v322, 2 * v76);
            if ( (_DWORD)v325 )
            {
              LODWORD(v139) = (v325 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
              v134 = v324 + 1;
              v125 = (__int64 *)(v323 + 16LL * (unsigned int)v139);
              v76 = *v125;
              if ( v107 != (_QWORD *)*v125 )
              {
                v140 = 1;
                v141 = 0;
                while ( v76 != -4096 )
                {
                  if ( !v141 && v76 == -8192 )
                    v141 = v125;
                  v139 = ((_DWORD)v325 - 1) & (unsigned int)(v139 + v140);
                  v125 = (__int64 *)(v323 + 16 * v139);
                  v76 = *v125;
                  if ( v107 == (_QWORD *)*v125 )
                    goto LABEL_139;
                  ++v140;
                }
                if ( v141 )
                  v125 = v141;
              }
              goto LABEL_139;
            }
            goto LABEL_558;
          }
          if ( v125 || v128 != (_QWORD *)-8192LL )
            v127 = v125;
          v126 = (v76 - 1) & (v124 + v126);
          v128 = *(_QWORD **)(v323 + 16LL * v126);
          if ( v107 != v128 )
          {
            ++v124;
            v125 = v127;
            v127 = (__int64 *)(v323 + 16LL * v126);
            continue;
          }
          break;
        }
        v127 = (__int64 *)(v323 + 16LL * v126);
LABEL_125:
        v129 = v127 + 1;
LABEL_126:
        ++v301;
        v98 += 18;
        *v129 = v104;
      }
      while ( v305 != v98 );
    }
    v96 = ++v298;
  }
  while ( v298 < *(_DWORD *)(v296 + 608) );
  v7 = v296;
  if ( a5 < v292 && a6 < v301 )
  {
    v146 = *(_QWORD *)(v296 + 408);
    v302 = 0;
    if ( v146 )
      goto LABEL_184;
LABEL_197:
    v153 = *(_QWORD *)(v7 + 432);
    if ( v153 )
      goto LABEL_198;
    goto LABEL_238;
  }
LABEL_340:
  v76 = v292;
  v224 = sub_37E3110(
           (_QWORD *)v7,
           v292,
           (__int64)&v322,
           (__int64)&v314,
           (__int64)&v318,
           &v341,
           a7,
           (__int64)&v309,
           &v311,
           &v344,
           a2,
           a4);
  v146 = *(_QWORD *)(v7 + 408);
  v302 = v224;
  if ( v146 )
  {
LABEL_184:
    sub_C7D6A0(*(_QWORD *)(v146 + 864), 8LL * *(unsigned int *)(v146 + 880), 4);
    sub_C7D6A0(*(_QWORD *)(v146 + 832), 8LL * *(unsigned int *)(v146 + 848), 4);
    v147 = *(_QWORD *)(v146 + 296);
    if ( v147 != v146 + 312 )
      _libc_free(v147);
    v148 = *(_QWORD *)(v146 + 256);
    if ( v148 )
      j_j___libc_free_0(v148);
    sub_37B77A0(*(_QWORD *)(v146 + 224));
    sub_37B7EE0(*(_QWORD *)(v146 + 176));
    v149 = *(_QWORD *)(v146 + 112);
    if ( v149 != v146 + 128 )
      _libc_free(v149);
    v150 = *(_QWORD *)(v146 + 88);
    if ( v150 != v146 + 104 )
      _libc_free(v150);
    v151 = *(_QWORD *)(v146 + 64);
    if ( v151 )
      j_j___libc_free_0(v151);
    v152 = *(_QWORD *)(v146 + 32);
    if ( v152 != v146 + 48 )
      _libc_free(v152);
    v76 = 888;
    j_j___libc_free_0(v146);
    goto LABEL_197;
  }
  v153 = *(_QWORD *)(v7 + 432);
  if ( v153 )
  {
LABEL_198:
    sub_C7D6A0(*(_QWORD *)(v153 + 3592), 4LL * *(unsigned int *)(v153 + 3608), 4);
    v154 = *(unsigned int *)(v153 + 3576);
    if ( (_DWORD)v154 )
    {
      v155 = *(_QWORD *)(v153 + 3560);
      v306 = v155 + 112 * v154;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v155 <= 0xFFFFFFFD )
          {
            v156 = *(unsigned __int64 **)(v155 + 8);
            v157 = &v156[11 * *(unsigned int *)(v155 + 16)];
            if ( v156 != v157 )
            {
              do
              {
                v157 -= 11;
                if ( (unsigned __int64 *)*v157 != v157 + 2 )
                  _libc_free(*v157);
              }
              while ( v156 != v157 );
              v157 = *(unsigned __int64 **)(v155 + 8);
            }
            if ( v157 != (unsigned __int64 *)(v155 + 24) )
              break;
          }
          v155 += 112;
          if ( v306 == v155 )
            goto LABEL_209;
        }
        v155 += 112;
        _libc_free((unsigned __int64)v157);
      }
      while ( v306 != v155 );
LABEL_209:
      v154 = *(unsigned int *)(v153 + 3576);
    }
    sub_C7D6A0(*(_QWORD *)(v153 + 3560), 112 * v154, 8);
    v158 = *(_QWORD *)(v153 + 3472);
    if ( v158 != v153 + 3488 )
      _libc_free(v158);
    v159 = *(unsigned int *)(v153 + 3464);
    if ( (_DWORD)v159 )
    {
      v160 = *(_QWORD *)(v153 + 3448);
      v161 = v160 + 88 * v159;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v160 <= 0xFFFFFFFD )
          {
            v162 = *(_QWORD *)(v160 + 8);
            if ( v162 != v160 + 24 )
              break;
          }
          v160 += 88;
          if ( v161 == v160 )
            goto LABEL_218;
        }
        _libc_free(v162);
        v160 += 88;
      }
      while ( v161 != v160 );
LABEL_218:
      v159 = *(unsigned int *)(v153 + 3464);
    }
    sub_C7D6A0(*(_QWORD *)(v153 + 3448), 88 * v159, 8);
    v163 = *(unsigned int *)(v153 + 3432);
    if ( (_DWORD)v163 )
    {
      v164 = *(_QWORD *)(v153 + 3416);
      v165 = v164 + 88 * v163;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v164 <= 0xFFFFFFFD )
          {
            v166 = *(_QWORD *)(v164 + 56);
            while ( v166 )
            {
              sub_37B80B0(*(_QWORD *)(v166 + 24));
              v167 = v166;
              v166 = *(_QWORD *)(v166 + 16);
              j_j___libc_free_0(v167);
            }
            v168 = *(_QWORD *)(v164 + 8);
            if ( v168 != v164 + 24 )
              break;
          }
          v164 += 88;
          if ( v165 == v164 )
            goto LABEL_227;
        }
        _libc_free(v168);
        v164 += 88;
      }
      while ( v165 != v164 );
LABEL_227:
      v163 = *(unsigned int *)(v153 + 3432);
    }
    sub_C7D6A0(*(_QWORD *)(v153 + 3416), 88 * v163, 8);
    v169 = *(_QWORD *)(v153 + 3136);
    if ( v169 != v153 + 3152 )
      _libc_free(v169);
    v170 = *(_QWORD *)(v153 + 48);
    v171 = v170 + 96LL * *(unsigned int *)(v153 + 56);
    if ( v170 != v171 )
    {
      do
      {
        v171 -= 96LL;
        v172 = *(_QWORD *)(v171 + 16);
        if ( v172 != v171 + 32 )
          _libc_free(v172);
      }
      while ( v170 != v171 );
      v171 = *(_QWORD *)(v153 + 48);
    }
    if ( v171 != v153 + 64 )
      _libc_free(v171);
    v76 = 3632;
    j_j___libc_free_0(v153);
  }
LABEL_238:
  *(_QWORD *)(v7 + 408) = 0;
  *(_QWORD *)(v7 + 424) = 0;
  *(_QWORD *)(v7 + 432) = 0;
  sub_270F0C0(v7 + 440, v76);
  v173 = *(_DWORD *)(v7 + 680);
  ++*(_QWORD *)(v7 + 664);
  *(_DWORD *)(v7 + 608) = 0;
  if ( !v173 )
  {
    if ( !*(_DWORD *)(v7 + 684) )
      goto LABEL_245;
    v175 = *(unsigned int *)(v7 + 688);
    if ( (unsigned int)v175 <= 0x40 )
      goto LABEL_242;
    sub_C7D6A0(*(_QWORD *)(v7 + 672), 16LL * *(unsigned int *)(v7 + 688), 8);
    *(_DWORD *)(v7 + 688) = 0;
    goto LABEL_388;
  }
  v174 = 4 * v173;
  v175 = *(unsigned int *)(v7 + 688);
  if ( (unsigned int)(4 * v173) < 0x40 )
    v174 = 64;
  if ( (unsigned int)v175 > v174 )
  {
    v280 = v173 - 1;
    if ( v280 )
    {
      _BitScanReverse(&v280, v280);
      v281 = 1 << (33 - (v280 ^ 0x1F));
      if ( v281 < 64 )
        v281 = 64;
      if ( (_DWORD)v175 == v281 )
        goto LABEL_518;
    }
    else
    {
      v281 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(v7 + 672), 16LL * *(unsigned int *)(v7 + 688), 8);
    v282 = sub_37B8280(v281);
    *(_DWORD *)(v7 + 688) = v282;
    if ( v282 )
    {
      *(_QWORD *)(v7 + 672) = sub_C7D670(16LL * v282, 8);
LABEL_518:
      sub_37BF820(v7 + 664);
      goto LABEL_245;
    }
LABEL_388:
    *(_QWORD *)(v7 + 672) = 0;
    goto LABEL_244;
  }
LABEL_242:
  v176 = *(_QWORD **)(v7 + 672);
  for ( k = &v176[2 * v175]; k != v176; v176 += 2 )
    *v176 = -4096;
LABEL_244:
  *(_QWORD *)(v7 + 680) = 0;
LABEL_245:
  v178 = *(_DWORD *)(v7 + 712);
  ++*(_QWORD *)(v7 + 696);
  if ( v178 )
  {
    v253 = 4 * v178;
    v179 = *(unsigned int *)(v7 + 720);
    if ( (unsigned int)(4 * v178) < 0x40 )
      v253 = 64;
    if ( v253 >= (unsigned int)v179 )
    {
LABEL_248:
      v180 = *(_DWORD **)(v7 + 704);
      for ( m = &v180[2 * v179]; m != v180; v180 += 2 )
        *v180 = -1;
      goto LABEL_250;
    }
    v254 = v178 - 1;
    if ( v254 )
    {
      _BitScanReverse(&v254, v254);
      v255 = 1 << (33 - (v254 ^ 0x1F));
      if ( v255 < 64 )
        v255 = 64;
      if ( v255 == (_DWORD)v179 )
      {
LABEL_421:
        sub_37BF860(v7 + 696);
        goto LABEL_251;
      }
    }
    else
    {
      v255 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(v7 + 704), 8 * v179, 4);
    v256 = sub_37B8280(v255);
    *(_DWORD *)(v7 + 720) = v256;
    if ( !v256 )
      goto LABEL_499;
    *(_QWORD *)(v7 + 704) = sub_C7D670(8LL * v256, 4);
    goto LABEL_421;
  }
  if ( *(_DWORD *)(v7 + 716) )
  {
    v179 = *(unsigned int *)(v7 + 720);
    if ( (unsigned int)v179 <= 0x40 )
      goto LABEL_248;
    sub_C7D6A0(*(_QWORD *)(v7 + 704), 8 * v179, 4);
    *(_DWORD *)(v7 + 720) = 0;
LABEL_499:
    *(_QWORD *)(v7 + 704) = 0;
LABEL_250:
    *(_QWORD *)(v7 + 712) = 0;
  }
LABEL_251:
  sub_37B7D10(*(_QWORD *)(v7 + 744));
  *(_QWORD *)(v7 + 744) = 0;
  *(_QWORD *)(v7 + 752) = v7 + 736;
  *(_QWORD *)(v7 + 760) = v7 + 736;
  *(_QWORD *)(v7 + 768) = 0;
  *(_DWORD *)(v7 + 784) = 0;
  sub_37BF8F0(v287);
  v182 = *(_DWORD *)(v7 + 2120);
  ++*(_QWORD *)(v7 + 2104);
  if ( !v182 && !*(_DWORD *)(v7 + 2124) )
    goto LABEL_266;
  v183 = *(_QWORD **)(v7 + 2112);
  v184 = 136LL * *(unsigned int *)(v7 + 2128);
  v185 = 4 * v182;
  v186 = &v183[(unsigned __int64)v184 / 8];
  if ( (unsigned int)(4 * v182) < 0x40 )
    v185 = 64;
  if ( *(_DWORD *)(v7 + 2128) > v185 )
  {
    v307 = v182;
    do
    {
      if ( *v183 != -4096 && *v183 != -8192 )
      {
        v269 = v183[13];
        while ( v269 )
        {
          sub_37B75D0(*(_QWORD *)(v269 + 24));
          v270 = v269;
          v269 = *(_QWORD *)(v269 + 16);
          j_j___libc_free_0(v270);
        }
        v271 = v183[1];
        if ( (_QWORD *)v271 != v183 + 3 )
          _libc_free(v271);
      }
      v183 += 17;
    }
    while ( v186 != v183 );
    v272 = *(_DWORD *)(v7 + 2128);
    if ( v307 )
    {
      v273 = 64;
      if ( v307 != 1 )
      {
        _BitScanReverse(&v274, v307 - 1);
        v273 = 1 << (33 - (v274 ^ 0x1F));
        if ( v273 < 64 )
          v273 = 64;
      }
      if ( v273 == v272 )
        goto LABEL_491;
      sub_C7D6A0(*(_QWORD *)(v7 + 2112), v184, 8);
      v275 = sub_37B8280(v273);
      *(_DWORD *)(v7 + 2128) = v275;
      if ( v275 )
      {
        *(_QWORD *)(v7 + 2112) = sub_C7D670(136LL * v275, 8);
LABEL_491:
        sub_37BFB00(v7 + 2104);
        goto LABEL_266;
      }
    }
    else
    {
      if ( !v272 )
        goto LABEL_491;
      sub_C7D6A0(*(_QWORD *)(v7 + 2112), v184, 8);
      *(_DWORD *)(v7 + 2128) = 0;
    }
    *(_QWORD *)(v7 + 2112) = 0;
    goto LABEL_265;
  }
  for ( ; v186 != v183; v183 += 17 )
  {
    if ( *v183 != -4096 )
    {
      if ( *v183 != -8192 )
      {
        v187 = v183[13];
        while ( v187 )
        {
          sub_37B75D0(*(_QWORD *)(v187 + 24));
          v188 = v187;
          v187 = *(_QWORD *)(v187 + 16);
          j_j___libc_free_0(v188);
        }
        v189 = v183[1];
        if ( (_QWORD *)v189 != v183 + 3 )
          _libc_free(v189);
      }
      *v183 = -4096;
    }
  }
LABEL_265:
  *(_QWORD *)(v7 + 2120) = 0;
LABEL_266:
  v190 = *(_DWORD *)(v7 + 2152);
  ++*(_QWORD *)(v7 + 2136);
  if ( v190 )
  {
    v261 = 4 * v190;
    v191 = *(unsigned int *)(v7 + 2160);
    if ( (unsigned int)(4 * v190) < 0x40 )
      v261 = 64;
    if ( (unsigned int)v191 <= v261 )
    {
LABEL_269:
      v192 = *(_QWORD **)(v7 + 2144);
      for ( n = &v192[4 * v191]; n != v192; *((_DWORD *)v192 - 6) = -1 )
      {
        *v192 = -4096;
        v192 += 4;
      }
      goto LABEL_271;
    }
    v262 = v190 - 1;
    if ( v262 )
    {
      _BitScanReverse(&v262, v262);
      v263 = 1 << (33 - (v262 ^ 0x1F));
      if ( v263 < 64 )
        v263 = 64;
      if ( (_DWORD)v191 == v263 )
      {
LABEL_446:
        sub_37BFB40(v7 + 2136);
        goto LABEL_272;
      }
    }
    else
    {
      v263 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(v7 + 2144), 32LL * *(unsigned int *)(v7 + 2160), 8);
    v264 = sub_37B8280(v263);
    *(_DWORD *)(v7 + 2160) = v264;
    if ( !v264 )
      goto LABEL_509;
    *(_QWORD *)(v7 + 2144) = sub_C7D670(32LL * v264, 8);
    goto LABEL_446;
  }
  if ( *(_DWORD *)(v7 + 2156) )
  {
    v191 = *(unsigned int *)(v7 + 2160);
    if ( (unsigned int)v191 <= 0x40 )
      goto LABEL_269;
    sub_C7D6A0(*(_QWORD *)(v7 + 2144), 32LL * *(unsigned int *)(v7 + 2160), 8);
    *(_DWORD *)(v7 + 2160) = 0;
LABEL_509:
    *(_QWORD *)(v7 + 2144) = 0;
LABEL_271:
    *(_QWORD *)(v7 + 2152) = 0;
  }
LABEL_272:
  sub_37BFC10(v7 + 2168);
  v194 = *(_DWORD *)(v7 + 2280);
  ++*(_QWORD *)(v7 + 2264);
  if ( v194 )
  {
    v257 = 4 * v194;
    v195 = *(unsigned int *)(v7 + 2288);
    if ( (unsigned int)(4 * v194) < 0x40 )
      v257 = 64;
    if ( v257 >= (unsigned int)v195 )
    {
LABEL_275:
      v196 = *(_QWORD **)(v7 + 2272);
      for ( ii = &v196[6 * v195]; ii != v196; *(v196 - 2) = 0 )
      {
        *v196 = 0;
        v196 += 6;
        *((_BYTE *)v196 - 24) = 0;
      }
      goto LABEL_277;
    }
    v258 = v194 - 1;
    if ( v258 )
    {
      _BitScanReverse(&v258, v258);
      v259 = 1 << (33 - (v258 ^ 0x1F));
      if ( v259 < 64 )
        v259 = 64;
      if ( v259 == (_DWORD)v195 )
      {
LABEL_436:
        sub_37BFF30(v289);
        goto LABEL_278;
      }
    }
    else
    {
      v259 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(v7 + 2272), 48 * v195, 8);
    v260 = sub_37B8280(v259);
    *(_DWORD *)(v7 + 2288) = v260;
    if ( !v260 )
      goto LABEL_511;
    *(_QWORD *)(v7 + 2272) = sub_C7D670(48LL * v260, 8);
    goto LABEL_436;
  }
  if ( *(_DWORD *)(v7 + 2284) )
  {
    v195 = *(unsigned int *)(v7 + 2288);
    if ( (unsigned int)v195 <= 0x40 )
      goto LABEL_275;
    sub_C7D6A0(*(_QWORD *)(v7 + 2272), 48 * v195, 8);
    *(_DWORD *)(v7 + 2288) = 0;
LABEL_511:
    *(_QWORD *)(v7 + 2272) = 0;
LABEL_277:
    *(_QWORD *)(v7 + 2280) = 0;
  }
LABEL_278:
  v198 = (unsigned int)v325;
  v199 = v323;
  *(_DWORD *)(v7 + 2304) = 0;
  sub_C7D6A0(v199, 16 * v198, 8);
  v200 = v321;
  if ( v321 )
  {
    v201 = v319;
    v202 = v319 + 72LL * v321;
    do
    {
      while ( *(_QWORD *)v201 == -8192 || *(_QWORD *)v201 == -4096 || *(_BYTE *)(v201 + 36) )
      {
        v201 += 72;
        if ( v202 == v201 )
          goto LABEL_285;
      }
      v203 = *(_QWORD *)(v201 + 16);
      v201 += 72;
      _libc_free(v203);
    }
    while ( v202 != v201 );
LABEL_285:
    v200 = v321;
  }
  sub_C7D6A0(v319, 72 * v200, 8);
  v204 = v317;
  if ( v317 )
  {
    v205 = v315;
    v206 = &v315[11 * v317];
    do
    {
      if ( *v205 != -4096 && *v205 != -8192 )
      {
        v207 = v205[7];
        while ( v207 )
        {
          sub_37B80B0(*(_QWORD *)(v207 + 24));
          v208 = v207;
          v207 = *(_QWORD *)(v207 + 16);
          j_j___libc_free_0(v208);
        }
        v209 = v205[1];
        if ( (_QWORD *)v209 != v205 + 3 )
          _libc_free(v209);
      }
      v205 += 11;
    }
    while ( v206 != v205 );
    v204 = v317;
  }
  sub_C7D6A0((__int64)v315, 88 * v204, 8);
  v210 = v311;
  v211 = &v311[v312];
  if ( v311 != v211 )
  {
    do
    {
      v212 = *--v211;
      if ( v212 )
      {
        if ( (unsigned __int64 *)*v212 != v212 + 2 )
          _libc_free(*v212);
        j_j___libc_free_0((unsigned __int64)v212);
      }
    }
    while ( v210 != v211 );
    v211 = v311;
  }
  if ( v211 != v313 )
    _libc_free((unsigned __int64)v211);
  v213 = v309;
  v214 = &v309[v310];
  if ( v309 != v214 )
  {
    do
    {
      v215 = (unsigned __int64 *)*--v214;
      if ( v215 )
      {
        if ( (unsigned __int64 *)*v215 != v215 + 2 )
          _libc_free(*v215);
        j_j___libc_free_0((unsigned __int64)v215);
      }
    }
    while ( v213 != v214 );
    v214 = v309;
  }
  if ( v214 != &v311 )
    _libc_free((unsigned __int64)v214);
  v216 = v341;
  v217 = &v341[74 * (unsigned int)v342];
  if ( v341 != v217 )
  {
    do
    {
      v217 -= 74;
      if ( (unsigned __int64 *)*v217 != v217 + 2 )
        _libc_free(*v217);
    }
    while ( v216 != v217 );
    v217 = v341;
  }
  if ( v217 != (unsigned __int64 *)v343 )
    _libc_free((unsigned __int64)v217);
  v218 = (__int64 *)v344;
  v219 = (__int64 *)&v344[856 * (unsigned int)v345];
  if ( v344 != (char *)v219 )
  {
    do
    {
      v219 -= 107;
      if ( (v219[86] & 1) == 0 )
        sub_C7D6A0(v219[87], 16LL * *((unsigned int *)v219 + 176), 8);
      v220 = v219[11];
      if ( (__int64 *)v220 != v219 + 13 )
        _libc_free(v220);
      if ( (v219[2] & 1) == 0 )
        sub_C7D6A0(v219[3], 8LL * *((unsigned int *)v219 + 8), 4);
    }
    while ( v218 != v219 );
    v219 = (__int64 *)v344;
  }
  if ( v219 != (__int64 *)v346 )
    _libc_free((unsigned __int64)v219);
  v221 = v338;
  v222 = (unsigned __int64)&v338[80 * (unsigned int)v339];
  if ( v338 != (_BYTE *)v222 )
  {
    do
    {
      v222 -= 80LL;
      if ( (*(_BYTE *)(v222 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v222 + 16), 16LL * *(unsigned int *)(v222 + 24), 8);
    }
    while ( v221 != (_BYTE *)v222 );
    v222 = (unsigned __int64)v338;
  }
  if ( (_BYTE *)v222 != v340 )
    _libc_free(v222);
  return v302;
}
