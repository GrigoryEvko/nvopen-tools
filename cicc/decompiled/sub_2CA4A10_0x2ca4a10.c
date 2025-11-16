// Function: sub_2CA4A10
// Address: 0x2ca4a10
//
_BOOL8 __fastcall sub_2CA4A10(__int64 a1, __int64 a2, unsigned int a3, __int64 **a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 *v9; // r8
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // rax
  int v19; // esi
  int v20; // r12d
  __int64 *v21; // r10
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rsi
  int v27; // r12d
  unsigned __int8 *v28; // rax
  unsigned int v29; // edi
  unsigned __int8 **v30; // rdx
  unsigned __int8 *v31; // r10
  char v32; // al
  __int64 *v33; // rax
  int v34; // eax
  __int64 *v35; // rbx
  unsigned int v36; // eax
  __int64 *v37; // rsi
  __int64 v38; // rdi
  __int64 *v39; // rsi
  __int64 v40; // r10
  int v41; // esi
  __int64 *v42; // rdx
  int v43; // ecx
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 *v47; // rsi
  int v48; // esi
  int v49; // edx
  unsigned int v50; // eax
  unsigned int v51; // eax
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  unsigned __int64 v54; // rdi
  unsigned int v55; // eax
  _QWORD *v56; // rbx
  _QWORD *v57; // r12
  unsigned __int64 v58; // rdi
  __int64 *v60; // rbx
  unsigned int v61; // eax
  __int64 *v62; // rsi
  __int64 v63; // rdi
  __int64 *v64; // rsi
  __int64 v65; // r10
  int v66; // esi
  __int64 *v67; // rdx
  int v68; // ecx
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rbx
  unsigned int v73; // r12d
  __int64 v74; // r13
  __int64 *v75; // rax
  unsigned int v76; // esi
  int v77; // edx
  int v78; // r11d
  __int64 v79; // rax
  unsigned int v80; // ecx
  _DWORD *v81; // rdi
  int v82; // r8d
  unsigned __int64 *v83; // rdi
  __int64 v84; // r8
  unsigned int v85; // eax
  __int64 *v86; // rcx
  __int64 v87; // r10
  __m128i *v88; // rsi
  int v89; // edi
  int v90; // ecx
  unsigned int v91; // eax
  unsigned __int64 *v92; // r13
  int v93; // esi
  unsigned int v94; // r9d
  unsigned int v95; // ecx
  __int64 v96; // r12
  int v97; // edi
  __int64 v98; // rax
  __int64 v99; // rdx
  unsigned __int64 *v100; // rcx
  __int64 v101; // rbx
  __int64 v102; // r13
  __int64 v103; // r12
  unsigned __int64 v104; // rax
  __int64 v105; // rdx
  __int64 v106; // r14
  __int64 *v107; // r15
  unsigned int v108; // ebx
  __int64 *v109; // rax
  __int64 *v110; // r13
  __int64 v111; // rdx
  __int64 *v112; // rax
  int v113; // r12d
  __int64 v114; // r15
  __int64 v115; // rcx
  __int64 v116; // r14
  unsigned int v117; // ecx
  int *v118; // rax
  int v119; // r9d
  _QWORD *v120; // rcx
  _BYTE *v121; // rsi
  _QWORD *v122; // rdi
  _QWORD *v123; // rdx
  int v124; // esi
  unsigned __int64 v125; // rdi
  int v126; // r8d
  _DWORD *v127; // rdi
  int v128; // edx
  __int64 *v129; // rax
  unsigned int v130; // esi
  int *v131; // rdx
  int v132; // edi
  bool v133; // zf
  __int64 v134; // rax
  __int64 v135; // rdx
  _QWORD *v136; // r14
  __int64 v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rbx
  __int64 v140; // r15
  __int64 v141; // rax
  __int64 **v142; // rdx
  __int64 **v143; // r10
  void **v144; // r13
  int v145; // r12d
  void *v146; // rax
  void *v147; // rcx
  unsigned int v148; // eax
  void **v149; // rdi
  void *v150; // r8
  int v151; // edi
  unsigned int v152; // edx
  void **v153; // rcx
  void *v154; // rsi
  int j; // edx
  __int64 v156; // rdx
  unsigned __int64 v157; // rax
  __int64 v158; // rdx
  _BYTE *v159; // r12
  __int64 v160; // rdi
  unsigned int v161; // ecx
  __int64 *v162; // rdx
  __int64 v163; // r9
  __int64 *v164; // rax
  unsigned __int64 *v165; // r13
  unsigned __int64 v166; // r9
  unsigned __int64 v167; // rdi
  int v168; // r14d
  __int64 v169; // rdx
  unsigned __int64 v170; // rsi
  unsigned int v171; // edx
  __int64 *v172; // r8
  __int64 v173; // r10
  __int64 v174; // r8
  __int64 k; // r9
  int v176; // r8d
  __int64 v177; // rax
  __int64 v178; // r10
  unsigned __int64 v179; // rax
  int v180; // r12d
  __int64 *v181; // r14
  __int64 v182; // rcx
  __int64 v183; // rsi
  int v184; // ecx
  __int64 **v185; // r12
  __int64 v186; // rax
  _QWORD *v187; // r15
  __int64 **v188; // r14
  __int64 v189; // r11
  unsigned int v190; // eax
  __int64 *v191; // rcx
  __int64 v192; // rdi
  int v193; // esi
  __int64 *v194; // r13
  __int64 *v195; // r10
  int v196; // edx
  int v197; // esi
  void **v198; // rdx
  int v199; // eax
  int v200; // esi
  int v201; // edx
  int v202; // edx
  int v203; // esi
  _QWORD *v204; // rdi
  int v205; // eax
  __int64 v206; // rax
  __m128i *v207; // r13
  unsigned __int64 v208; // rax
  __int64 v209; // r15
  __int64 *v210; // rax
  __int64 v211; // rbx
  unsigned int ****v212; // rax
  unsigned int ***v213; // r14
  __int64 *v214; // rbx
  unsigned int *v215; // rsi
  int v216; // edi
  unsigned int v217; // edx
  unsigned int **v218; // r8
  unsigned int *v219; // r9
  unsigned int **v220; // r9
  unsigned int *v221; // rdx
  unsigned int v222; // eax
  unsigned int **v223; // r8
  unsigned int *v224; // r10
  unsigned int v225; // eax
  _QWORD *v226; // r13
  _QWORD *v227; // rbx
  unsigned __int64 *v228; // r14
  unsigned __int64 *v229; // r12
  unsigned int v230; // eax
  __int64 v231; // rbx
  __int64 v232; // r12
  unsigned __int64 v233; // rdi
  __int64 v234; // rbx
  __int64 v235; // r12
  unsigned __int64 v236; // rdi
  unsigned int v237; // eax
  int v238; // esi
  unsigned int v239; // ecx
  __int64 v240; // r14
  int v241; // edi
  __int64 v242; // rdx
  __int64 v243; // r13
  __int64 v244; // rax
  __int64 v245; // r15
  __m128i *v246; // rsi
  const __m128i *v247; // rdx
  int v248; // esi
  _QWORD *v249; // rdx
  int v250; // eax
  int v251; // edx
  int v252; // r10d
  int v253; // ebx
  unsigned int v254; // r8d
  int i; // r11d
  int v256; // r10d
  int *v257; // r11
  int v258; // ebx
  __int64 v259; // r11
  int v260; // eax
  int v261; // eax
  int v262; // edx
  __int64 *v263; // rsi
  int v264; // edi
  _QWORD *v265; // rax
  _QWORD *v266; // r15
  _QWORD *v267; // rbx
  unsigned __int64 *v268; // r14
  unsigned __int64 v269; // rdi
  __int64 v270; // rax
  __int64 v271; // r12
  __int64 v272; // r13
  unsigned __int64 v273; // rdi
  __m128i *v274; // r14
  signed __int64 v275; // rbx
  __int64 v276; // r12
  signed __int64 v277; // rdx
  unsigned __int64 m; // rbx
  __int64 v279; // rdi
  int v280; // r14d
  __m128i *v281; // rbx
  const __m128i *v282; // rdi
  int v283; // r8d
  __int64 v284; // rdi
  unsigned __int32 v285; // ecx
  __int64 *v286; // rdx
  __int64 v287; // r9
  __int64 *v288; // rax
  __int64 *v289; // rax
  __int64 v290; // rax
  __int64 **v291; // r12
  _QWORD *v292; // rbx
  _BYTE *v293; // rdi
  __int64 *v294; // r14
  _QWORD *v295; // r13
  _QWORD *v296; // rsi
  __int64 v297; // rax
  unsigned int v298; // esi
  int v299; // eax
  _QWORD *v300; // rdx
  int v301; // eax
  __int64 v302; // rax
  int v303; // r8d
  int v304; // r10d
  __int64 v305; // r13
  int v306; // esi
  _QWORD *v307; // rdx
  int v308; // eax
  int v309; // r11d
  int v310; // edi
  int v311; // r11d
  int v312; // r9d
  int v313; // edx
  int v314; // r10d
  int v315; // r13d
  unsigned int v316; // r10d
  unsigned int v317; // r11d
  int v318; // r10d
  int v319; // eax
  int v320; // eax
  int v321; // r9d
  int v322; // r9d
  __int64 v323; // rdx
  bool v324; // [rsp+7h] [rbp-659h]
  __int64 v325; // [rsp+8h] [rbp-658h]
  __int64 v326; // [rsp+18h] [rbp-648h]
  __int64 v327; // [rsp+28h] [rbp-638h]
  unsigned int v328; // [rsp+30h] [rbp-630h]
  unsigned int ***v329; // [rsp+30h] [rbp-630h]
  __int64 *v331; // [rsp+40h] [rbp-620h]
  __int64 v332; // [rsp+48h] [rbp-618h]
  int v333; // [rsp+50h] [rbp-610h]
  __int64 *v334; // [rsp+50h] [rbp-610h]
  __int64 v335; // [rsp+50h] [rbp-610h]
  int v336; // [rsp+50h] [rbp-610h]
  __int64 v337; // [rsp+50h] [rbp-610h]
  int v338; // [rsp+60h] [rbp-600h]
  __int64 v339; // [rsp+60h] [rbp-600h]
  __int64 v340; // [rsp+60h] [rbp-600h]
  int v341; // [rsp+60h] [rbp-600h]
  int v342; // [rsp+60h] [rbp-600h]
  int v343; // [rsp+68h] [rbp-5F8h]
  _QWORD *v344; // [rsp+68h] [rbp-5F8h]
  __int64 v345; // [rsp+70h] [rbp-5F0h]
  __int64 v346; // [rsp+70h] [rbp-5F0h]
  __int64 v347; // [rsp+70h] [rbp-5F0h]
  int v348; // [rsp+70h] [rbp-5F0h]
  __int64 v349; // [rsp+70h] [rbp-5F0h]
  unsigned __int64 v350; // [rsp+70h] [rbp-5F0h]
  _QWORD *v352; // [rsp+78h] [rbp-5E8h]
  __int64 v353; // [rsp+78h] [rbp-5E8h]
  __int64 v354; // [rsp+78h] [rbp-5E8h]
  __int64 *v355; // [rsp+80h] [rbp-5E0h]
  __int64 *v356; // [rsp+80h] [rbp-5E0h]
  __int64 v357; // [rsp+80h] [rbp-5E0h]
  __int64 v358; // [rsp+80h] [rbp-5E0h]
  unsigned int ***v360; // [rsp+88h] [rbp-5D8h]
  int v361; // [rsp+9Ch] [rbp-5C4h] BYREF
  _BYTE *v362; // [rsp+A0h] [rbp-5C0h] BYREF
  _BYTE *v363; // [rsp+A8h] [rbp-5B8h] BYREF
  __int64 v364; // [rsp+B0h] [rbp-5B0h] BYREF
  _QWORD *v365; // [rsp+B8h] [rbp-5A8h] BYREF
  unsigned __int64 v366; // [rsp+C0h] [rbp-5A0h] BYREF
  __int64 v367; // [rsp+C8h] [rbp-598h]
  __int64 v368; // [rsp+D0h] [rbp-590h]
  __int64 *v369; // [rsp+E0h] [rbp-580h] BYREF
  __int64 *v370; // [rsp+E8h] [rbp-578h]
  __int64 *v371; // [rsp+F0h] [rbp-570h]
  __int64 *v372; // [rsp+100h] [rbp-560h] BYREF
  __int64 *v373; // [rsp+108h] [rbp-558h]
  __int64 *v374; // [rsp+110h] [rbp-550h]
  unsigned __int64 v375; // [rsp+120h] [rbp-540h] BYREF
  __int64 v376; // [rsp+128h] [rbp-538h]
  __int64 v377; // [rsp+130h] [rbp-530h]
  __int64 v378[4]; // [rsp+140h] [rbp-520h] BYREF
  void *src; // [rsp+160h] [rbp-500h] BYREF
  __m128i *v380; // [rsp+168h] [rbp-4F8h]
  const __m128i *v381; // [rsp+170h] [rbp-4F0h]
  __int64 v382; // [rsp+180h] [rbp-4E0h] BYREF
  _QWORD *v383; // [rsp+188h] [rbp-4D8h]
  __int64 v384; // [rsp+190h] [rbp-4D0h]
  unsigned int v385; // [rsp+198h] [rbp-4C8h]
  __int64 v386; // [rsp+1A0h] [rbp-4C0h] BYREF
  _QWORD *v387; // [rsp+1A8h] [rbp-4B8h]
  __int64 v388; // [rsp+1B0h] [rbp-4B0h]
  unsigned int v389; // [rsp+1B8h] [rbp-4A8h]
  __int64 v390; // [rsp+1C0h] [rbp-4A0h] BYREF
  _QWORD *v391; // [rsp+1C8h] [rbp-498h]
  __int64 v392; // [rsp+1D0h] [rbp-490h]
  unsigned int v393; // [rsp+1D8h] [rbp-488h]
  __int64 v394; // [rsp+1E0h] [rbp-480h] BYREF
  __int64 v395; // [rsp+1E8h] [rbp-478h]
  __int64 v396; // [rsp+1F0h] [rbp-470h]
  unsigned int v397; // [rsp+1F8h] [rbp-468h]
  __int64 v398; // [rsp+200h] [rbp-460h] BYREF
  __int64 v399; // [rsp+208h] [rbp-458h]
  __int64 v400; // [rsp+210h] [rbp-450h]
  unsigned int v401; // [rsp+218h] [rbp-448h]
  __int64 v402; // [rsp+220h] [rbp-440h] BYREF
  __int64 v403; // [rsp+228h] [rbp-438h]
  __int64 v404; // [rsp+230h] [rbp-430h]
  unsigned int v405; // [rsp+238h] [rbp-428h]
  __int64 v406; // [rsp+240h] [rbp-420h] BYREF
  __int64 v407; // [rsp+248h] [rbp-418h]
  __int64 v408; // [rsp+250h] [rbp-410h]
  __int64 v409; // [rsp+258h] [rbp-408h]
  unsigned __int8 *v410; // [rsp+260h] [rbp-400h] BYREF
  __int64 v411; // [rsp+268h] [rbp-3F8h]
  __int64 v412; // [rsp+270h] [rbp-3F0h]
  __int64 v413; // [rsp+278h] [rbp-3E8h]
  __int64 v414; // [rsp+280h] [rbp-3E0h] BYREF
  _QWORD *v415; // [rsp+288h] [rbp-3D8h]
  __int64 v416; // [rsp+290h] [rbp-3D0h]
  unsigned int v417; // [rsp+298h] [rbp-3C8h]
  __m128i v418; // [rsp+2A0h] [rbp-3C0h] BYREF
  __m128i v419; // [rsp+2B0h] [rbp-3B0h] BYREF
  _BYTE v420[928]; // [rsp+2C0h] [rbp-3A0h] BYREF

  sub_27C1C30(
    (__int64)v420,
    *(__int64 **)(a1 + 184),
    *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL),
    (__int64)"BaseAddressStrengthReduce",
    1);
  v10 = *a4;
  v11 = a4[1];
  v382 = 0;
  v383 = 0;
  v384 = 0;
  v385 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  v386 = 0;
  v387 = 0;
  v388 = 0;
  v389 = 0;
  v369 = 0;
  v370 = 0;
  v371 = 0;
  v390 = 0;
  v391 = 0;
  v392 = 0;
  v393 = 0;
  v372 = 0;
  v373 = 0;
  v374 = 0;
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v402 = *v10;
      v26 = *(_QWORD *)v402;
      v406 = *(_QWORD *)v402;
      if ( byte_50122E8 && (unsigned __int8)sub_2C92820(v26, v26, v7, v8, (unsigned int)v9) )
        goto LABEL_12;
      v27 = sub_CEFE70(*(_QWORD *)(a1 + 184), v26);
      *(_DWORD *)sub_2C97130(a1 + 32, &v406) = v27;
      if ( *(_BYTE *)(sub_D95540(v406) + 8) != 14 )
      {
        v12 = &v391[4 * v393];
        v13 = sub_D95540(v406);
        v14 = v13;
        if ( v393 )
        {
          v15 = (v393 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v16 = &v391[4 * v15];
          v17 = *v16;
          if ( v14 == *v16 )
          {
LABEL_5:
            if ( v12 != v16 )
              goto LABEL_6;
LABEL_43:
            v46 = sub_D95540(v406);
            v47 = v373;
            v418.m128i_i64[0] = v46;
            if ( v373 == v374 )
            {
              sub_2C96280((__int64)&v372, v373, &v418);
            }
            else
            {
              if ( v373 )
              {
                *v373 = v46;
                v47 = v373;
              }
              v373 = v47 + 1;
            }
LABEL_6:
            v18 = sub_D95540(v406);
            v19 = v393;
            v414 = v18;
            v7 = v18;
            if ( v393 )
            {
              LODWORD(v9) = (_DWORD)v391;
              v20 = 1;
              v21 = 0;
              v8 = (v393 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v22 = &v391[4 * v8];
              v23 = *v22;
              if ( v7 == *v22 )
              {
LABEL_8:
                v24 = (__int64 *)v22[2];
                v25 = (__int64)(v22 + 1);
                if ( (__int64 *)v22[3] != v24 )
                {
                  if ( v24 )
                  {
                    v7 = v402;
                    *v24 = v402;
                    v24 = (__int64 *)v22[2];
                  }
                  v22[2] = (__int64)(v24 + 1);
                  goto LABEL_12;
                }
                goto LABEL_53;
              }
              while ( v23 != -4096 )
              {
                if ( !v21 && v23 == -8192 )
                  v21 = v22;
                v8 = (v393 - 1) & (v20 + (_DWORD)v8);
                v22 = &v391[4 * (unsigned int)v8];
                v23 = *v22;
                if ( v7 == *v22 )
                  goto LABEL_8;
                ++v20;
              }
              if ( v21 )
                v22 = v21;
              ++v390;
              v264 = v392 + 1;
              v418.m128i_i64[0] = (__int64)v22;
              if ( 4 * ((int)v392 + 1) < 3 * v393 )
              {
                if ( v393 - HIDWORD(v392) - v264 > v393 >> 3 )
                  goto LABEL_389;
                goto LABEL_393;
              }
            }
            else
            {
              ++v390;
              v418.m128i_i64[0] = 0;
            }
            v19 = 2 * v393;
LABEL_393:
            sub_2C941C0((__int64)&v390, v19);
            sub_2C90410((__int64)&v390, &v414, &v418);
            v7 = v414;
            v264 = v392 + 1;
            v22 = (__int64 *)v418.m128i_i64[0];
LABEL_389:
            LODWORD(v392) = v264;
            if ( *v22 != -4096 )
              --HIDWORD(v392);
            goto LABEL_52;
          }
          v45 = 1;
          while ( v17 != -4096 )
          {
            v321 = v45 + 1;
            v15 = (v393 - 1) & (v45 + v15);
            v16 = &v391[4 * v15];
            v17 = *v16;
            if ( v14 == *v16 )
              goto LABEL_5;
            v45 = v321;
          }
        }
        if ( v12 == &v391[4 * v393] )
          goto LABEL_43;
        goto LABEL_6;
      }
      v28 = sub_CF22E0(*(unsigned __int8 **)(***(_QWORD ***)(v402 + 8) + 24LL));
      v410 = v28;
      if ( v389 )
      {
        v29 = (v389 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v30 = (unsigned __int8 **)&v387[4 * v29];
        v31 = *v30;
        if ( v28 == *v30 )
        {
LABEL_18:
          if ( &v387[4 * v389] != v30 )
            goto LABEL_19;
        }
        else
        {
          v262 = 1;
          while ( v31 != (unsigned __int8 *)-4096LL )
          {
            v322 = v262 + 1;
            v323 = (v389 - 1) & (v29 + v262);
            v29 = v323;
            v30 = (unsigned __int8 **)&v387[4 * v323];
            v31 = *v30;
            if ( v28 == *v30 )
              goto LABEL_18;
            v262 = v322;
          }
        }
      }
      v263 = v370;
      if ( v370 == v371 )
      {
        sub_2C95F70((__int64)&v369, v370, &v410);
      }
      else
      {
        if ( v370 )
        {
          *v370 = (__int64)v28;
          v263 = v370;
        }
        v370 = v263 + 1;
      }
LABEL_19:
      v32 = sub_2C90050((__int64)&v386, (__int64 *)&v410, &v414);
      v9 = (__int64 *)&v410;
      if ( !v32 )
      {
        v48 = v389;
        v22 = (__int64 *)v414;
        ++v386;
        v49 = v388 + 1;
        v418.m128i_i64[0] = v414;
        if ( 4 * ((int)v388 + 1) >= 3 * v389 )
        {
          v48 = 2 * v389;
        }
        else if ( v389 - HIDWORD(v388) - v49 > v389 >> 3 )
        {
          goto LABEL_49;
        }
        sub_2C93F90((__int64)&v386, v48);
        sub_2C90050((__int64)&v386, (__int64 *)&v410, &v418);
        v49 = v388 + 1;
        v22 = (__int64 *)v418.m128i_i64[0];
LABEL_49:
        LODWORD(v388) = v49;
        if ( *v22 != -4096 )
          --HIDWORD(v388);
        v7 = (__int64)v410;
LABEL_52:
        *v22 = v7;
        v25 = (__int64)(v22 + 1);
        v24 = 0;
        v22[1] = 0;
        v22[2] = 0;
        v22[3] = 0;
        goto LABEL_53;
      }
      v7 = v414;
      v33 = *(__int64 **)(v414 + 16);
      v24 = *(__int64 **)(v414 + 24);
      v25 = v414 + 8;
      if ( v33 == v24 )
      {
LABEL_53:
        sub_2C90A30(v25, v24, &v402);
LABEL_12:
        if ( v11 == ++v10 )
          break;
      }
      else
      {
        if ( v33 )
        {
          v8 = v402;
          *v33 = v402;
          v33 = *(__int64 **)(v7 + 16);
        }
        ++v10;
        *(_QWORD *)(v7 + 16) = v33 + 1;
        if ( v11 == v10 )
          break;
      }
    }
  }
  v34 = sub_31C5B60(*(_QWORD *)(a1 + 208), a2, 0);
  if ( v34 <= 0 )
  {
    v343 = -1;
  }
  else
  {
    v324 = 0;
    if ( v34 < a3 )
      goto LABEL_58;
    v343 = 0;
    if ( dword_5012AC8 >= a3 )
      v343 = v34 - a3;
  }
  v35 = v369;
  v355 = v370;
  if ( v369 != v370 )
  {
    while ( 1 )
    {
      v40 = *(_QWORD *)(a1 + 200);
      if ( !v389 )
        break;
      v36 = (v389 - 1) & (((unsigned int)*v35 >> 9) ^ ((unsigned int)*v35 >> 4));
      v37 = &v387[4 * v36];
      v38 = *v37;
      if ( *v35 != *v37 )
      {
        v341 = 1;
        v42 = 0;
        while ( v38 != -4096 )
        {
          if ( v42 || v38 != -8192 )
            v37 = v42;
          v36 = (v389 - 1) & (v341 + v36);
          v38 = v387[4 * v36];
          if ( *v35 == v38 )
          {
            v37 = &v387[4 * v36];
            goto LABEL_31;
          }
          ++v341;
          v42 = v37;
          v37 = &v387[4 * v36];
        }
        if ( !v42 )
          v42 = v37;
        ++v386;
        v43 = v388 + 1;
        v418.m128i_i64[0] = (__int64)v42;
        if ( 4 * ((int)v388 + 1) < 3 * v389 )
        {
          if ( v389 - HIDWORD(v388) - v43 > v389 >> 3 )
            goto LABEL_37;
          v345 = v40;
          v41 = v389;
          goto LABEL_36;
        }
LABEL_35:
        v345 = v40;
        v41 = 2 * v389;
LABEL_36:
        sub_2C93F90((__int64)&v386, v41);
        sub_2C90050((__int64)&v386, v35, &v418);
        v42 = (__int64 *)v418.m128i_i64[0];
        v40 = v345;
        v43 = v388 + 1;
LABEL_37:
        LODWORD(v388) = v43;
        if ( *v42 != -4096 )
          --HIDWORD(v388);
        v44 = *v35;
        v39 = v42 + 1;
        v42[1] = 0;
        v42[2] = 0;
        *v42 = v44;
        v42[3] = 0;
        goto LABEL_32;
      }
LABEL_31:
      v39 = v37 + 1;
LABEL_32:
      ++v35;
      sub_2CA2C90(a1, v39, v40, a3, (__int64)&v382, (__int64)&v366);
      if ( v355 == v35 )
        goto LABEL_81;
    }
    ++v386;
    v418.m128i_i64[0] = 0;
    goto LABEL_35;
  }
LABEL_81:
  v60 = v372;
  v356 = v373;
  if ( v372 != v373 )
  {
    while ( 1 )
    {
      v65 = *(_QWORD *)(a1 + 200);
      if ( !v393 )
        break;
      v61 = (v393 - 1) & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
      v62 = &v391[4 * v61];
      v63 = *v62;
      if ( *v60 != *v62 )
      {
        v342 = 1;
        v67 = 0;
        while ( v63 != -4096 )
        {
          if ( v67 || v63 != -8192 )
            v62 = v67;
          v61 = (v393 - 1) & (v342 + v61);
          v63 = v391[4 * v61];
          if ( *v60 == v63 )
          {
            v62 = &v391[4 * v61];
            goto LABEL_84;
          }
          ++v342;
          v67 = v62;
          v62 = &v391[4 * v61];
        }
        if ( !v67 )
          v67 = v62;
        ++v390;
        v68 = v392 + 1;
        v418.m128i_i64[0] = (__int64)v67;
        if ( 4 * ((int)v392 + 1) < 3 * v393 )
        {
          if ( v393 - HIDWORD(v392) - v68 > v393 >> 3 )
            goto LABEL_90;
          v346 = v65;
          v66 = v393;
          goto LABEL_89;
        }
LABEL_88:
        v346 = v65;
        v66 = 2 * v393;
LABEL_89:
        sub_2C941C0((__int64)&v390, v66);
        sub_2C90410((__int64)&v390, v60, &v418);
        v67 = (__int64 *)v418.m128i_i64[0];
        v65 = v346;
        v68 = v392 + 1;
LABEL_90:
        LODWORD(v392) = v68;
        if ( *v67 != -4096 )
          --HIDWORD(v392);
        v69 = *v60;
        v64 = v67 + 1;
        v67[1] = 0;
        v67[2] = 0;
        *v67 = v69;
        v67[3] = 0;
        goto LABEL_85;
      }
LABEL_84:
      v64 = v62 + 1;
LABEL_85:
      ++v60;
      sub_2CA2C90(a1, v64, v65, a3, (__int64)&v382, (__int64)&v366);
      if ( v356 == v60 )
        goto LABEL_93;
    }
    ++v390;
    v418.m128i_i64[0] = 0;
    goto LABEL_88;
  }
LABEL_93:
  v70 = v366;
  v394 = 0;
  v395 = 0;
  v396 = 0;
  v397 = 0;
  v71 = (__int64)(v367 - v366) >> 3;
  v375 = 0;
  v376 = 0;
  v377 = 0;
  v398 = 0;
  v399 = 0;
  v400 = 0;
  v401 = 0;
  v328 = v71;
  if ( !(_DWORD)v71 )
  {
    v402 = 0;
    v403 = 0;
    v404 = 0;
    v405 = 0;
    v324 = 0;
    goto LABEL_508;
  }
  v72 = 0;
  v73 = 0;
  v74 = 8LL * (unsigned int)(v71 - 1);
  while ( 1 )
  {
    if ( *sub_2C93820((__int64)&v382, (__int64 *)(v70 + v72)) )
    {
      v75 = sub_2C93820((__int64)&v382, (__int64 *)(v72 + v366));
      v76 = v401;
      LODWORD(v414) = (__int64)(*(_QWORD *)(*v75 + 8) - *(_QWORD *)*v75) >> 3;
      if ( !v401 )
        goto LABEL_110;
    }
    else
    {
      LODWORD(v414) = 0;
      v76 = v401;
      if ( !v401 )
      {
LABEL_110:
        ++v398;
        v418.m128i_i64[0] = 0;
        goto LABEL_111;
      }
    }
    v77 = v414;
    v78 = 1;
    v79 = 0;
    v80 = (v76 - 1) & (37 * v414);
    v81 = (_DWORD *)(v399 + 32LL * v80);
    v82 = *v81;
    if ( (_DWORD)v414 == *v81 )
    {
LABEL_97:
      v83 = (unsigned __int64 *)(v81 + 2);
      goto LABEL_98;
    }
    while ( v82 != -1 )
    {
      if ( v82 == -2 && !v79 )
        v79 = (__int64)v81;
      v80 = (v76 - 1) & (v78 + v80);
      v81 = (_DWORD *)(v399 + 32LL * v80);
      v82 = *v81;
      if ( (_DWORD)v414 == *v81 )
        goto LABEL_97;
      ++v78;
    }
    if ( !v79 )
      v79 = (__int64)v81;
    ++v398;
    v89 = v400 + 1;
    v418.m128i_i64[0] = v79;
    if ( 4 * ((int)v400 + 1) < 3 * v76 )
    {
      if ( v76 - HIDWORD(v400) - v89 > v76 >> 3 )
        goto LABEL_364;
      goto LABEL_112;
    }
LABEL_111:
    v76 *= 2;
LABEL_112:
    sub_2C93D70((__int64)&v398, v76);
    sub_2C8FFB0((__int64)&v398, (int *)&v414, &v418);
    v77 = v414;
    v89 = v400 + 1;
    v79 = v418.m128i_i64[0];
LABEL_364:
    LODWORD(v400) = v89;
    if ( *(_DWORD *)v79 != -1 )
      --HIDWORD(v400);
    *(_DWORD *)v79 = v77;
    v83 = (unsigned __int64 *)(v79 + 8);
    *(_QWORD *)(v79 + 8) = 0;
    *(_QWORD *)(v79 + 16) = 0;
    *(_QWORD *)(v79 + 24) = 0;
LABEL_98:
    v84 = *(_QWORD *)(v366 + v72);
    if ( v385 )
    {
      v85 = (v385 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
      v86 = &v383[2 * v85];
      v87 = *v86;
      if ( v84 == *v86 )
      {
LABEL_100:
        v418.m128i_i64[0] = (__int64)&v382;
        v419.m128i_i64[0] = (__int64)v86;
        v418.m128i_i64[1] = v382;
        v419.m128i_i64[1] = (__int64)&v383[2 * v385];
        v88 = (__m128i *)v83[1];
        if ( v88 == (__m128i *)v83[2] )
          goto LABEL_116;
        goto LABEL_101;
      }
      v90 = 1;
      while ( v87 != -4096 )
      {
        v311 = v90 + 1;
        v85 = (v385 - 1) & (v90 + v85);
        v86 = &v383[2 * v85];
        v87 = *v86;
        if ( v84 == *v86 )
          goto LABEL_100;
        v90 = v311;
      }
    }
    v418.m128i_i64[0] = (__int64)&v382;
    v418.m128i_i64[1] = v382;
    v419.m128i_i64[0] = (__int64)&v383[2 * v385];
    v419.m128i_i64[1] = v419.m128i_i64[0];
    v88 = (__m128i *)v83[1];
    if ( v88 == (__m128i *)v83[2] )
    {
LABEL_116:
      sub_2C90580(v83, v88, &v418);
      goto LABEL_104;
    }
LABEL_101:
    if ( v88 )
    {
      *v88 = _mm_loadu_si128(&v418);
      v88[1] = _mm_loadu_si128(&v419);
      v88 = (__m128i *)v83[1];
    }
    v83[1] = (unsigned __int64)&v88[2];
LABEL_104:
    if ( v73 < (unsigned int)v414 )
      v73 = v414;
    if ( v74 == v72 )
      break;
    v70 = v366;
    v72 += 8;
  }
  v328 = v73;
  LODWORD(v414) = 2;
  if ( v73 > 1 )
  {
    v91 = 2;
    v92 = &v375;
    do
    {
      v93 = v401;
      if ( v401 )
      {
        v94 = v401 - 1;
        v95 = (v401 - 1) & (37 * v91);
        v96 = v399 + 32LL * v95;
        v97 = *(_DWORD *)v96;
        if ( v91 != *(_DWORD *)v96 )
        {
          v253 = *(_DWORD *)v96;
          v254 = (v401 - 1) & (37 * v91);
          for ( i = 1; ; i = v256 )
          {
            if ( v253 == -1 )
              goto LABEL_119;
            v256 = i + 1;
            v254 = v94 & (i + v254);
            v257 = (int *)(v399 + 32LL * v254);
            v253 = *v257;
            if ( *v257 == v91 )
              break;
          }
          if ( v257 == (int *)(v399 + 32LL * v401) )
            goto LABEL_119;
          v258 = 1;
          v259 = 0;
          while ( v97 != -1 )
          {
            if ( v97 == -2 && !v259 )
              v259 = v96;
            v95 = v94 & (v258 + v95);
            v96 = v399 + 32LL * v95;
            v97 = *(_DWORD *)v96;
            if ( *(_DWORD *)v96 == v91 )
              goto LABEL_123;
            ++v258;
          }
          if ( !v259 )
            v259 = v96;
          ++v398;
          v260 = v400 + 1;
          v418.m128i_i64[0] = v259;
          if ( 4 * ((int)v400 + 1) >= 3 * v401 )
          {
            v93 = 2 * v401;
          }
          else if ( v401 - HIDWORD(v400) - v260 > v401 >> 3 )
          {
LABEL_337:
            LODWORD(v400) = v260;
            if ( *(_DWORD *)v259 != -1 )
              --HIDWORD(v400);
            v261 = v414;
            *(_QWORD *)(v259 + 8) = 0;
            *(_QWORD *)(v259 + 16) = 0;
            *(_DWORD *)v259 = v261;
            *(_QWORD *)(v259 + 24) = 0;
            goto LABEL_119;
          }
          sub_2C93D70((__int64)&v398, v93);
          sub_2C8FFB0((__int64)&v398, (int *)&v414, &v418);
          v259 = v418.m128i_i64[0];
          v260 = v400 + 1;
          goto LABEL_337;
        }
        if ( v96 != v399 + 32LL * v401 )
        {
LABEL_123:
          v98 = *(_QWORD *)(v96 + 8);
          v99 = (*(_QWORD *)(v96 + 16) - v98) >> 5;
          if ( (_DWORD)v99 )
          {
            v100 = v92;
            v101 = 0;
            v102 = v96;
            v103 = (__int64)v100;
            v357 = 32LL * (unsigned int)(v99 - 1);
            while ( 1 )
            {
              sub_2C9F570(
                a1,
                **(_QWORD **)(v98 + v101 + 16),
                *(_QWORD **)(*(_QWORD *)(v98 + v101 + 16) + 8LL),
                (__int64)&v394,
                v103,
                *(_QWORD *)(a1 + 200));
              if ( v357 == v101 )
                break;
              v98 = *(_QWORD *)(v102 + 8);
              v101 += 32;
            }
            v92 = (unsigned __int64 *)v103;
          }
        }
      }
LABEL_119:
      v91 = v414 + 1;
      LODWORD(v414) = v91;
    }
    while ( v91 <= v328 );
  }
  v104 = v375;
  v402 = 0;
  v403 = 0;
  v404 = 0;
  v324 = (_DWORD)v396 != 0;
  v405 = 0;
  v105 = (__int64)(v376 - v375) >> 3;
  if ( !(_DWORD)v105 )
  {
LABEL_508:
    v406 = 0;
    v407 = 0;
    v408 = 0;
    v409 = 0;
    v410 = 0;
    v411 = 0;
    v412 = 0;
    v413 = 0;
    goto LABEL_185;
  }
  v106 = 0;
  v107 = &v402;
  v347 = 8LL * (unsigned int)(v105 - 1);
  v108 = 0;
  while ( 2 )
  {
    v410 = *(unsigned __int8 **)(v104 + v106);
    v109 = sub_2C93500((__int64)&v394, (__int64 *)&v410);
    v110 = (__int64 *)*v109;
    v111 = *(_QWORD *)*v109;
    if ( *(_QWORD *)(*v109 + 8) == v111 )
      goto LABEL_149;
    v112 = v107;
    v113 = 0;
    v114 = v106;
    v115 = 0;
    v116 = (__int64)v112;
    while ( 2 )
    {
      v123 = *(_QWORD **)(v111 + 8 * v115);
      v124 = v405;
      v414 = (__int64)v123;
      v125 = (__int64)(v123[1] - *v123) >> 3;
      LODWORD(v406) = v125;
      v126 = v125;
      if ( v108 < v125 )
        v108 = v125;
      if ( !v405 )
      {
        ++v402;
        v418.m128i_i64[0] = 0;
        goto LABEL_142;
      }
      v117 = (v405 - 1) & (37 * v125);
      v118 = (int *)(v403 + 32LL * v117);
      v119 = *v118;
      if ( (_DWORD)v125 != *v118 )
      {
        v338 = 1;
        v127 = 0;
        while ( v119 != -1 )
        {
          if ( v119 == -2 && !v127 )
            v127 = v118;
          v117 = (v405 - 1) & (v338 + v117);
          v118 = (int *)(v403 + 32LL * v117);
          v119 = *v118;
          if ( v126 == *v118 )
            goto LABEL_133;
          ++v338;
        }
        if ( !v127 )
          v127 = v118;
        ++v402;
        v128 = v404 + 1;
        v418.m128i_i64[0] = (__int64)v127;
        if ( 4 * ((int)v404 + 1) >= 3 * v405 )
        {
LABEL_142:
          v124 = 2 * v405;
        }
        else if ( v405 - HIDWORD(v404) - v128 > v405 >> 3 )
        {
          goto LABEL_144;
        }
        sub_2C93B50(v116, v124);
        sub_2C8FF10(v116, (int *)&v406, &v418);
        v126 = v406;
        v127 = (_DWORD *)v418.m128i_i64[0];
        v128 = v404 + 1;
LABEL_144:
        LODWORD(v404) = v128;
        if ( *v127 != -1 )
          --HIDWORD(v404);
        *v127 = v126;
        v121 = 0;
        v122 = v127 + 2;
        *v122 = 0;
        v122[1] = 0;
        v122[2] = 0;
        goto LABEL_147;
      }
LABEL_133:
      v120 = (_QWORD *)*((_QWORD *)v118 + 2);
      v121 = (_BYTE *)*((_QWORD *)v118 + 3);
      v122 = v118 + 2;
      if ( v120 == (_QWORD *)v121 )
      {
LABEL_147:
        sub_2C90710((__int64)v122, v121, &v414);
        goto LABEL_137;
      }
      if ( v120 )
      {
        *v120 = v123;
        v120 = (_QWORD *)*((_QWORD *)v118 + 2);
      }
      *((_QWORD *)v118 + 2) = v120 + 1;
LABEL_137:
      v111 = *v110;
      v115 = (unsigned int)++v113;
      if ( v113 != (v110[1] - *v110) >> 3 )
        continue;
      break;
    }
    v129 = (__int64 *)v116;
    v106 = v114;
    v107 = v129;
LABEL_149:
    if ( v347 != v106 )
    {
      v104 = v375;
      v106 += 8;
      continue;
    }
    break;
  }
  v406 = 0;
  v407 = 0;
  v408 = 0;
  v409 = 0;
  v410 = 0;
  v411 = 0;
  v412 = 0;
  v413 = 0;
  LODWORD(v365) = v108;
  if ( v108 <= 1 )
    goto LABEL_185;
  while ( 2 )
  {
    if ( !v405 )
      goto LABEL_184;
    v130 = (v405 - 1) & (37 * v108);
    v131 = (int *)(v403 + 32LL * v130);
    v132 = *v131;
    if ( *v131 != v108 )
    {
      for ( j = 1; ; j = v312 )
      {
        if ( v132 == -1 )
          goto LABEL_184;
        v312 = j + 1;
        v130 = (v405 - 1) & (j + v130);
        v131 = (int *)(v403 + 32LL * v130);
        v132 = *v131;
        if ( *v131 == v108 )
          break;
      }
    }
    if ( v131 == (int *)(v403 + 32LL * v405) )
      goto LABEL_184;
    v133 = (unsigned __int8)sub_2C8FF10((__int64)&v402, (int *)&v365, &v414) == 0;
    v134 = v414;
    if ( !v133 )
    {
      v135 = *(_QWORD *)(v414 + 16);
      v136 = (_QWORD *)(v414 + 8);
      goto LABEL_164;
    }
    v200 = v405;
    v418.m128i_i64[0] = v414;
    ++v402;
    v201 = v404 + 1;
    if ( 4 * ((int)v404 + 1) >= 3 * v405 )
    {
      v200 = 2 * v405;
      goto LABEL_395;
    }
    if ( v405 - HIDWORD(v404) - v201 <= v405 >> 3 )
    {
LABEL_395:
      sub_2C93B50((__int64)&v402, v200);
      sub_2C8FF10((__int64)&v402, (int *)&v365, &v418);
      v201 = v404 + 1;
      v134 = v418.m128i_i64[0];
    }
    LODWORD(v404) = v201;
    if ( *(_DWORD *)v134 != -1 )
      --HIDWORD(v404);
    v202 = (int)v365;
    v136 = (_QWORD *)(v134 + 8);
    *(_QWORD *)(v134 + 8) = 0;
    *(_QWORD *)(v134 + 16) = 0;
    *(_DWORD *)v134 = v202;
    v135 = 0;
    *(_QWORD *)(v134 + 24) = 0;
LABEL_164:
    v137 = *v136;
    v138 = (v135 - *v136) >> 3;
    if ( !(_DWORD)v138 )
      goto LABEL_184;
    v139 = 0;
    v140 = 8LL * (unsigned int)(v138 - 1);
    while ( 2 )
    {
      v141 = *(_QWORD *)(v137 + v139);
      src = 0;
      v378[0] = v141;
      v142 = *(__int64 ***)v141;
      v143 = *(__int64 ***)(v141 + 8);
      v144 = (void **)(v407 + 8LL * (unsigned int)v409);
      if ( v143 == *(__int64 ***)v141 )
      {
        v146 = 0;
      }
      else
      {
        v145 = v409 - 1;
        do
        {
          v147 = (void *)**v142;
          if ( (_DWORD)v409 )
          {
            v148 = v145 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
            v149 = (void **)(v407 + 8LL * v148);
            v150 = *v149;
            if ( v147 == *v149 )
            {
LABEL_168:
              if ( v144 != v149 )
                goto LABEL_180;
            }
            else
            {
              v151 = 1;
              while ( v150 != (void *)-4096LL )
              {
                v148 = v145 & (v151 + v148);
                v333 = v151 + 1;
                v149 = (void **)(v407 + 8LL * v148);
                v150 = *v149;
                if ( v147 == *v149 )
                  goto LABEL_168;
                v151 = v333;
              }
            }
          }
          v146 = (void *)(*v142++)[1];
          src = v146;
        }
        while ( v143 != v142 );
      }
      if ( (_DWORD)v409 )
      {
        v152 = (v409 - 1) & (((unsigned int)v146 >> 9) ^ ((unsigned int)v146 >> 4));
        v153 = (void **)(v407 + 8LL * v152);
        v154 = *v153;
        if ( *v153 == v146 )
        {
LABEL_179:
          if ( v153 != v144 )
            goto LABEL_180;
        }
        else
        {
          v184 = 1;
          while ( v154 != (void *)-4096LL )
          {
            v310 = v184 + 1;
            v152 = (v409 - 1) & (v184 + v152);
            v153 = (void **)(v407 + 8LL * v152);
            v154 = *v153;
            if ( *v153 == v146 )
              goto LABEL_179;
            v184 = v310;
          }
        }
      }
      if ( !(unsigned __int8)sub_2C8FCF0((__int64)&v410, v378, &v414) )
      {
        v248 = v413;
        v249 = (_QWORD *)v414;
        ++v410;
        v250 = v412 + 1;
        v418.m128i_i64[0] = v414;
        if ( 4 * ((int)v412 + 1) >= (unsigned int)(3 * v413) )
        {
          v248 = 2 * v413;
        }
        else if ( (int)v413 - HIDWORD(v412) - v250 > (unsigned int)v413 >> 3 )
        {
LABEL_320:
          LODWORD(v412) = v250;
          if ( *v249 != -4096 )
            --HIDWORD(v412);
          *v249 = v378[0];
          goto LABEL_223;
        }
        sub_2C92940((__int64)&v410, v248);
        sub_2C8FCF0((__int64)&v410, v378, &v418);
        v249 = (_QWORD *)v418.m128i_i64[0];
        v250 = v412 + 1;
        goto LABEL_320;
      }
LABEL_223:
      v185 = *(__int64 ***)v378[0];
      if ( *(_QWORD *)(v378[0] + 8) == *(_QWORD *)v378[0] )
        goto LABEL_235;
      v186 = v140;
      v187 = v136;
      v188 = *(__int64 ***)(v378[0] + 8);
      v189 = v186;
      while ( 2 )
      {
        while ( 2 )
        {
          v193 = v409;
          v194 = *v185;
          if ( !(_DWORD)v409 )
          {
            ++v406;
            v418.m128i_i64[0] = 0;
            goto LABEL_229;
          }
          v190 = (v409 - 1) & (((unsigned int)*v194 >> 9) ^ ((unsigned int)*v194 >> 4));
          v191 = (__int64 *)(v407 + 8LL * v190);
          v192 = *v191;
          if ( *v191 == *v194 )
          {
LABEL_226:
            if ( v188 == ++v185 )
              goto LABEL_234;
            continue;
          }
          break;
        }
        v336 = 1;
        v195 = 0;
        while ( v192 != -4096 )
        {
          if ( v192 != -8192 || v195 )
            v191 = v195;
          v190 = (v409 - 1) & (v336 + v190);
          v192 = *(_QWORD *)(v407 + 8LL * v190);
          if ( *v194 == v192 )
            goto LABEL_226;
          ++v336;
          v195 = v191;
          v191 = (__int64 *)(v407 + 8LL * v190);
        }
        if ( !v195 )
          v195 = v191;
        ++v406;
        v196 = v408 + 1;
        v418.m128i_i64[0] = (__int64)v195;
        if ( 4 * ((int)v408 + 1) >= (unsigned int)(3 * v409) )
        {
LABEL_229:
          v335 = v189;
          v193 = 2 * v409;
LABEL_230:
          sub_2C92B10((__int64)&v406, v193);
          sub_2C8FE60((__int64)&v406, v194, &v418);
          v195 = (__int64 *)v418.m128i_i64[0];
          v189 = v335;
          v196 = v408 + 1;
          goto LABEL_231;
        }
        if ( (int)v409 - HIDWORD(v408) - v196 <= (unsigned int)v409 >> 3 )
        {
          v335 = v189;
          goto LABEL_230;
        }
LABEL_231:
        LODWORD(v408) = v196;
        if ( *v195 != -4096 )
          --HIDWORD(v408);
        ++v185;
        *v195 = *v194;
        if ( v188 != v185 )
          continue;
        break;
      }
LABEL_234:
      v136 = v187;
      v140 = v189;
LABEL_235:
      if ( !(unsigned __int8)sub_2C8FE60((__int64)&v406, (__int64 *)&src, &v414) )
      {
        v197 = v409;
        v198 = (void **)v414;
        ++v406;
        v199 = v408 + 1;
        v418.m128i_i64[0] = v414;
        if ( 4 * ((int)v408 + 1) >= (unsigned int)(3 * v409) )
        {
          v197 = 2 * v409;
        }
        else if ( (int)v409 - HIDWORD(v408) - v199 > (unsigned int)v409 >> 3 )
        {
          goto LABEL_238;
        }
        sub_2C92B10((__int64)&v406, v197);
        sub_2C8FE60((__int64)&v406, (__int64 *)&src, &v418);
        v198 = (void **)v418.m128i_i64[0];
        v199 = v408 + 1;
LABEL_238:
        LODWORD(v408) = v199;
        if ( *v198 != (void *)-4096LL )
          --HIDWORD(v408);
        *v198 = src;
      }
LABEL_180:
      if ( v140 != v139 )
      {
        v137 = *v136;
        v139 += 8;
        continue;
      }
      break;
    }
LABEL_184:
    v108 = (_DWORD)v365 - 1;
    LODWORD(v365) = v108;
    if ( v108 > 1 )
      continue;
    break;
  }
LABEL_185:
  v414 = 0;
  v415 = 0;
  v416 = 0;
  v417 = 0;
  if ( dword_5012C88 )
    sub_2CA3BA0(a1, &v375, (__int64)&v394, (__int64)&v410, (__int64)&v414);
  sub_2C91AF0(*(_QWORD *)(a1 + 280));
  v156 = v376;
  *(_QWORD *)(a1 + 288) = a1 + 272;
  *(_QWORD *)(a1 + 296) = a1 + 272;
  v157 = v375;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  v418 = 0u;
  v158 = (__int64)(v156 - v157) >> 3;
  v419.m128i_i64[0] = 0;
  v419.m128i_i32[2] = 0;
  v361 = 0;
  if ( !(_DWORD)v158 )
    goto LABEL_274;
  v339 = 0;
  v159 = v420;
  v327 = 8LL * (unsigned int)(v158 - 1);
  while ( 2 )
  {
    v160 = *(_QWORD *)(v157 + v339);
    v365 = (_QWORD *)v160;
    if ( !v417 )
      goto LABEL_192;
    v161 = (v417 - 1) & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
    v162 = &v415[4 * v161];
    v163 = *v162;
    if ( v160 != *v162 )
    {
      v251 = 1;
      while ( v163 != -4096 )
      {
        v252 = v251 + 1;
        v161 = (v417 - 1) & (v251 + v161);
        v162 = &v415[4 * v161];
        v163 = *v162;
        if ( v160 == *v162 )
          goto LABEL_191;
        v251 = v252;
      }
      goto LABEL_192;
    }
LABEL_191:
    if ( v162 == &v415[4 * v417] )
      goto LABEL_192;
    if ( (unsigned __int8)sub_2C90350((__int64)&v414, (__int64 *)&v365, v378) )
    {
      v177 = *(_QWORD *)(v378[0] + 16);
      v334 = (__int64 *)(v378[0] + 8);
      goto LABEL_209;
    }
    v203 = v417;
    v204 = (_QWORD *)v378[0];
    ++v414;
    v205 = v416 + 1;
    src = (void *)v378[0];
    if ( 4 * ((int)v416 + 1) >= 3 * v417 )
    {
      v203 = 2 * v417;
      goto LABEL_479;
    }
    if ( v417 - HIDWORD(v416) - v205 <= v417 >> 3 )
    {
LABEL_479:
      sub_2C943F0((__int64)&v414, v203);
      sub_2C90350((__int64)&v414, (__int64 *)&v365, &src);
      v204 = src;
      v205 = v416 + 1;
    }
    LODWORD(v416) = v205;
    if ( *v204 != -4096 )
      --HIDWORD(v416);
    v206 = (__int64)v365;
    v204[1] = 0;
    v204[2] = 0;
    *v204 = v206;
    v334 = v204 + 1;
    v177 = 0;
    v204[3] = 0;
LABEL_209:
    v178 = *v334;
    v179 = 0xAAAAAAAAAAAAAAABLL * ((v177 - *v334) >> 3);
    if ( (_DWORD)v179 )
    {
      v353 = (__int64)v159;
      v349 = 0;
      v332 = 24LL * (unsigned int)v179;
      while ( 1 )
      {
        v180 = 0;
        memset(v378, 0, 24);
        v181 = (__int64 *)(v178 + v349);
        src = 0;
        v380 = 0;
        v381 = 0;
        sub_2C9F200(a1, (_QWORD *)(v178 + v349), (__int64)v378);
        v182 = *v181;
        v183 = 0;
        if ( v181[1] != *v181 )
        {
          do
          {
            sub_2CA1FC0(
              a1,
              *(__int64 *******)(v182 + 8 * v183),
              (__int64)v365,
              (__int64)v378,
              (__int64 **)&src,
              v353,
              (__int64)&v418,
              a5);
            v182 = *v181;
            v183 = (unsigned int)++v180;
          }
          while ( v180 != (v181[1] - *v181) >> 3 );
        }
        if ( src )
          j_j___libc_free_0((unsigned __int64)src);
        if ( v378[0] )
          j_j___libc_free_0(v378[0]);
        v349 += 24;
        if ( v332 == v349 )
          break;
        v178 = *v334;
      }
      v159 = (_BYTE *)v353;
    }
LABEL_192:
    v164 = sub_2C93500((__int64)&v394, (__int64 *)&v365);
    v165 = (unsigned __int64 *)*v164;
    v166 = *(_QWORD *)(*v164 + 8);
    v167 = *(_QWORD *)*v164;
    v352 = (_QWORD *)*v164;
    if ( v167 != v166 )
    {
      v168 = 0;
      v169 = 0;
      do
      {
        v170 = *(_QWORD *)(v167 + 8 * v169);
        if ( (_DWORD)v413 )
        {
          v171 = (v413 - 1) & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
          v172 = (__int64 *)(v411 + 8LL * v171);
          v173 = *v172;
          if ( v170 == *v172 )
          {
LABEL_196:
            if ( v172 != (__int64 *)(v411 + 8LL * (unsigned int)v413) )
            {
              sub_2CA06E0(a1, v170, (__int64)v365, &v361, v343, (__int64)v159, (__int64)&v418, a5);
              v166 = v165[1];
              v167 = *v165;
            }
          }
          else
          {
            v176 = 1;
            while ( v173 != -4096 )
            {
              v171 = (v413 - 1) & (v176 + v171);
              v348 = v176 + 1;
              v172 = (__int64 *)(v411 + 8LL * v171);
              v173 = *v172;
              if ( v170 == *v172 )
                goto LABEL_196;
              v176 = v348;
            }
          }
        }
        v169 = (unsigned int)++v168;
      }
      while ( v168 != (__int64)(v166 - v167) >> 3 );
    }
    if ( v167 )
      j_j___libc_free_0(v167);
    j_j___libc_free_0((unsigned __int64)v352);
    if ( v327 != v339 )
    {
      v157 = v375;
      v339 += 8;
      continue;
    }
    break;
  }
  if ( v361 > 0 && (int)qword_5012D68 > 2 )
  {
    v237 = v328;
    src = 0;
    v380 = 0;
    v381 = 0;
    LODWORD(v365) = v328;
    if ( v328 )
    {
      while ( 2 )
      {
        v238 = v401;
        if ( v401 )
        {
          v174 = v401 - 1;
          v239 = v174 & (37 * v237);
          v240 = v399 + 32LL * v239;
          v241 = *(_DWORD *)v240;
          if ( *(_DWORD *)v240 == v237 )
          {
            if ( v240 != 32LL * v401 + v399 )
              goto LABEL_309;
          }
          else
          {
            v315 = *(_DWORD *)v240;
            v316 = v174 & (37 * v237);
            for ( k = 1; ; k = v317 )
            {
              if ( v315 == -1 )
                goto LABEL_305;
              v317 = k + 1;
              v316 = v174 & (k + v316);
              k = v399 + 32LL * v316;
              v315 = *(_DWORD *)k;
              if ( v237 == *(_DWORD *)k )
                break;
            }
            if ( k == v399 + 32LL * v401 )
              goto LABEL_305;
            if ( v241 != v237 )
            {
              v318 = 1;
              k = 0;
              while ( v241 != -1 )
              {
                if ( v241 == -2 && !k )
                  k = v240;
                v239 = v174 & (v318 + v239);
                v240 = v399 + 32LL * v239;
                v241 = *(_DWORD *)v240;
                if ( v237 == *(_DWORD *)v240 )
                  goto LABEL_309;
                ++v318;
              }
              v174 = 2 * v401;
              if ( k )
                v240 = k;
              ++v398;
              v319 = v400 + 1;
              v378[0] = v240;
              if ( 4 * ((int)v400 + 1) >= 3 * v401 )
              {
                v238 = 2 * v401;
              }
              else if ( v401 - HIDWORD(v400) - v319 > v401 >> 3 )
              {
LABEL_491:
                LODWORD(v400) = v319;
                if ( *(_DWORD *)v240 != -1 )
                  --HIDWORD(v400);
                v320 = (int)v365;
                *(_QWORD *)(v240 + 8) = 0;
                *(_QWORD *)(v240 + 16) = 0;
                *(_DWORD *)v240 = v320;
                *(_QWORD *)(v240 + 24) = 0;
                goto LABEL_305;
              }
              sub_2C93D70((__int64)&v398, v238);
              sub_2C8FFB0((__int64)&v398, (int *)&v365, v378);
              v240 = v378[0];
              v319 = v400 + 1;
              goto LABEL_491;
            }
LABEL_309:
            v242 = *(_QWORD *)(v240 + 8);
            v243 = 0;
            v244 = (*(_QWORD *)(v240 + 16) - v242) >> 5;
            v245 = 32LL * (unsigned int)(v244 - 1);
            if ( (_DWORD)v244 )
            {
              while ( 1 )
              {
                v246 = v380;
                v247 = (const __m128i *)(v243 + v242);
                if ( v380 == v381 )
                {
                  sub_2C90580((unsigned __int64 *)&src, v380, v247);
                }
                else
                {
                  if ( v380 )
                  {
                    *v380 = _mm_loadu_si128(v247);
                    v246[1] = _mm_loadu_si128(v247 + 1);
                    v246 = v380;
                  }
                  v380 = v246 + 2;
                }
                if ( v245 == v243 )
                  break;
                v242 = *(_QWORD *)(v240 + 8);
                v243 += 32;
              }
            }
          }
        }
LABEL_305:
        v133 = (_DWORD)v365 == 1;
        v237 = (_DWORD)v365 - 1;
        LODWORD(v365) = (_DWORD)v365 - 1;
        if ( !v133 )
          continue;
        break;
      }
      v207 = v380;
      v274 = (__m128i *)src;
      v275 = (char *)v380 - (_BYTE *)src;
      v276 = a1 + 32;
      v277 = ((char *)v380 - (_BYTE *)src) >> 5;
      v208 = v277;
      if ( (_DWORD)v277 )
      {
        v358 = 32LL * (unsigned int)(v277 - 1);
        for ( m = 0; ; m += 32LL )
        {
          v279 = *(_QWORD *)(a1 + 184);
          v378[0] = *(_QWORD *)v274[m / 0x10 + 1].m128i_i64[0];
          v280 = sub_CEFE70(v279, v378[0]);
          *(_DWORD *)sub_2C97130(v276, v378) = v280;
          v274 = (__m128i *)src;
          if ( v358 == m )
            break;
        }
        v207 = v380;
        v275 = (char *)v380 - (_BYTE *)src;
        v208 = ((char *)v380 - (_BYTE *)src) >> 5;
      }
      if ( v274 != v207 )
      {
        _BitScanReverse64(&v208, v208);
        sub_2C98900((__int64)v274, v207, 2LL * (int)(63 - (v208 ^ 0x3F)), v276, v174, k);
        if ( v275 <= 512 )
        {
          sub_2C97780(v274, v207, v276);
        }
        else
        {
          v281 = v274 + 32;
          sub_2C97780(v274, v274 + 32, v276);
          if ( v207 != &v274[32] )
          {
            do
            {
              v282 = v281;
              v281 += 2;
              sub_2C97270(v282, v276);
            }
            while ( v207 != v281 );
          }
        }
        v207 = (__m128i *)src;
        v208 = ((char *)v380 - (_BYTE *)src) >> 5;
      }
      if ( (_DWORD)v208 )
      {
        v209 = a5;
        v350 = 0;
        v337 = 32LL * (unsigned int)v208;
        while ( 1 )
        {
          v210 = (__int64 *)v207[v350 / 0x10 + 1].m128i_i64[0];
          v211 = *v210;
          v212 = (unsigned int ****)v210[1];
          v340 = v211;
          v360 = v212[1];
          if ( v360 != *v212 )
            break;
LABEL_271:
          v350 += 32LL;
          if ( v337 == v350 )
            goto LABEL_272;
        }
        v213 = *v212;
        while ( 2 )
        {
          v214 = (__int64 *)*v213;
          v215 = **v213;
          if ( (_DWORD)v409 )
          {
            v216 = v409 - 1;
            v217 = (v409 - 1) & (((unsigned int)v215 >> 9) ^ ((unsigned int)v215 >> 4));
            v218 = (unsigned int **)(v407 + 8LL * v217);
            v219 = *v218;
            if ( v215 != *v218 )
            {
              v303 = 1;
              while ( v219 != (unsigned int *)-4096LL )
              {
                v304 = v303 + 1;
                v217 = v216 & (v303 + v217);
                v218 = (unsigned int **)(v407 + 8LL * v217);
                v219 = *v218;
                if ( v215 == *v218 )
                  goto LABEL_266;
                v303 = v304;
              }
              goto LABEL_269;
            }
LABEL_266:
            v220 = (unsigned int **)(v407 + 8LL * (unsigned int)v409);
            if ( v220 != v218 )
            {
              v221 = (unsigned int *)v214[1];
              v222 = v216 & (((unsigned int)v221 >> 9) ^ ((unsigned int)v221 >> 4));
              v223 = (unsigned int **)(v407 + 8LL * v222);
              v224 = *v223;
              if ( *v223 == v221 )
              {
LABEL_268:
                if ( v220 != v223 )
                  goto LABEL_269;
              }
              else
              {
                v283 = 1;
                while ( v224 != (unsigned int *)-4096LL )
                {
                  v309 = v283 + 1;
                  v222 = v216 & (v283 + v222);
                  v223 = (unsigned int **)(v407 + 8LL * v222);
                  v224 = *v223;
                  if ( *v223 == v221 )
                    goto LABEL_268;
                  v283 = v309;
                }
              }
              sub_2C9EEF0(a1, v215, (_QWORD **)v221, (__int64 *)&v362, (__int64 *)&v363, *(_QWORD *)(a1 + 200), 0);
              v284 = *v214;
              if ( *v362 != 23 || v362 == *(_BYTE **)(*(_QWORD *)(**(_QWORD **)v284 + 16LL) + 40LL) )
              {
                if ( v419.m128i_i32[2] )
                {
                  v285 = (v419.m128i_i32[2] - 1) & (((unsigned int)v284 >> 9) ^ ((unsigned int)v284 >> 4));
                  v286 = (__int64 *)(v418.m128i_i64[1] + 16LL * v285);
                  v287 = *v286;
                  if ( v284 == *v286 )
                  {
LABEL_435:
                    if ( v286 != (__int64 *)(v418.m128i_i64[1] + 16LL * v419.m128i_u32[2]) )
                    {
                      v288 = sub_2C92EC0((__int64)&v418, v214);
                      v289 = sub_DA3860(*(_QWORD **)(a1 + 184), *v288);
                      v344 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v289, v340, 0, 0);
                      v290 = v214[1];
                      v291 = *(__int64 ***)v290;
                      v354 = *(_QWORD *)v290 + 8LL * *(unsigned int *)(v290 + 8);
                      if ( v354 != *(_QWORD *)v290 )
                      {
                        v331 = v214;
                        v292 = 0;
                        v329 = v213;
                        while ( 2 )
                        {
                          v293 = v363;
                          v294 = *v291;
                          if ( *v363 == 23 )
                          {
                            if ( v363 == *(_BYTE **)(v294[2] + 40) )
                              v293 = (_BYTE *)v294[2];
                            else
                              v293 = (_BYTE *)sub_986580((__int64)v363);
                          }
                          if ( !v292 )
                          {
                            v302 = v325;
                            LOWORD(v302) = 0;
                            v325 = v302;
                            v292 = sub_F8DB90((__int64)v420, (__int64)v344, 0, (__int64)(v293 + 24), 0);
                          }
                          v295 = v292;
                          if ( !sub_D968A0(*v294) )
                          {
                            v296 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v344, *v294, 0, 0);
                            v297 = v326;
                            LOWORD(v297) = 0;
                            v326 = v297;
                            v295 = sub_F8DB90((__int64)v420, (__int64)v296, 0, v294[2] + 24, 0);
                          }
                          v364 = v294[2];
                          if ( (unsigned __int8)sub_23FDF60(v209, &v364, &v365) )
                            goto LABEL_439;
                          v298 = *(_DWORD *)(v209 + 24);
                          v299 = *(_DWORD *)(v209 + 16);
                          v300 = v365;
                          ++*(_QWORD *)v209;
                          v301 = v299 + 1;
                          v378[0] = (__int64)v300;
                          if ( 4 * v301 >= 3 * v298 )
                          {
                            v298 *= 2;
                          }
                          else if ( v298 - *(_DWORD *)(v209 + 20) - v301 > v298 >> 3 )
                          {
                            goto LABEL_449;
                          }
                          sub_CF4090(v209, v298);
                          sub_23FDF60(v209, &v364, v378);
                          v300 = (_QWORD *)v378[0];
                          v301 = *(_DWORD *)(v209 + 16) + 1;
LABEL_449:
                          *(_DWORD *)(v209 + 16) = v301;
                          if ( *v300 != -4096 )
                            --*(_DWORD *)(v209 + 20);
                          *v300 = v364;
LABEL_439:
                          ++v291;
                          sub_2C95850(v294[3], (__int64)v295, v294[2]);
                          if ( v291 == (__int64 **)v354 )
                          {
                            v305 = (__int64)v292;
                            v213 = v329;
                            v214 = v331;
                            goto LABEL_457;
                          }
                          continue;
                        }
                      }
                      v305 = 0;
LABEL_457:
                      *sub_2C92EC0((__int64)&v418, v214 + 1) = v305;
                      if ( !(unsigned __int8)sub_2C8FE60((__int64)&v406, v214 + 1, &v365) )
                      {
                        v306 = v409;
                        v307 = v365;
                        ++v406;
                        v308 = v408 + 1;
                        v378[0] = (__int64)v365;
                        if ( 4 * ((int)v408 + 1) >= (unsigned int)(3 * v409) )
                        {
                          v306 = 2 * v409;
                        }
                        else if ( (int)v409 - HIDWORD(v408) - v308 > (unsigned int)v409 >> 3 )
                        {
                          goto LABEL_460;
                        }
                        sub_2C92B10((__int64)&v406, v306);
                        sub_2C8FE60((__int64)&v406, v214 + 1, v378);
                        v307 = (_QWORD *)v378[0];
                        v308 = v408 + 1;
LABEL_460:
                        LODWORD(v408) = v308;
                        if ( *v307 != -4096 )
                          --HIDWORD(v408);
                        *v307 = v214[1];
                      }
                    }
                  }
                  else
                  {
                    v313 = 1;
                    while ( v287 != -4096 )
                    {
                      v314 = v313 + 1;
                      v285 = (v419.m128i_i32[2] - 1) & (v313 + v285);
                      v286 = (__int64 *)(v418.m128i_i64[1] + 16LL * v285);
                      v287 = *v286;
                      if ( v284 == *v286 )
                        goto LABEL_435;
                      v313 = v314;
                    }
                  }
                }
              }
            }
          }
LABEL_269:
          if ( v360 == ++v213 )
          {
            v207 = (__m128i *)src;
            goto LABEL_271;
          }
          continue;
        }
      }
LABEL_272:
      if ( v207 )
        j_j___libc_free_0((unsigned __int64)v207);
    }
  }
LABEL_274:
  if ( (_DWORD)v384 )
  {
    v265 = v383;
    v266 = &v383[2 * v385];
    if ( v383 != v266 )
    {
      while ( 1 )
      {
        v267 = v265;
        if ( *v265 != -4096 && *v265 != -8192 )
          break;
        v265 += 2;
        if ( v266 == v265 )
          goto LABEL_275;
      }
      while ( 1 )
      {
        if ( v266 == v267 )
          goto LABEL_275;
        v268 = (unsigned __int64 *)v267[1];
        v269 = *v268;
        v270 = (__int64)(v268[1] - *v268) >> 3;
        if ( !(_DWORD)v270 )
          goto LABEL_410;
        v271 = 0;
        v272 = 8LL * (unsigned int)(v270 - 1);
        while ( 1 )
        {
          v273 = *(_QWORD *)(v269 + v271);
          if ( v273 )
          {
            j_j___libc_free_0(v273);
            v268 = (unsigned __int64 *)v267[1];
          }
          if ( v272 == v271 )
            break;
          v269 = *v268;
          v271 += 8;
        }
        if ( v268 )
          break;
LABEL_413:
        v267 += 2;
        if ( v267 == v266 )
          goto LABEL_275;
        while ( *v267 == -8192 || *v267 == -4096 )
        {
          v267 += 2;
          if ( v266 == v267 )
            goto LABEL_275;
        }
      }
      v269 = *v268;
LABEL_410:
      if ( v269 )
        j_j___libc_free_0(v269);
      j_j___libc_free_0((unsigned __int64)v268);
      goto LABEL_413;
    }
  }
LABEL_275:
  sub_C7D6A0(v418.m128i_i64[1], 16LL * v419.m128i_u32[2], 8);
  v225 = v417;
  if ( v417 )
  {
    v226 = v415;
    v227 = &v415[4 * v417];
    do
    {
      if ( *v226 != -4096 && *v226 != -8192 )
      {
        v228 = (unsigned __int64 *)v226[2];
        v229 = (unsigned __int64 *)v226[1];
        if ( v228 != v229 )
        {
          do
          {
            if ( *v229 )
              j_j___libc_free_0(*v229);
            v229 += 3;
          }
          while ( v228 != v229 );
          v229 = (unsigned __int64 *)v226[1];
        }
        if ( v229 )
          j_j___libc_free_0((unsigned __int64)v229);
      }
      v226 += 4;
    }
    while ( v227 != v226 );
    v225 = v417;
  }
  sub_C7D6A0((__int64)v415, 32LL * v225, 8);
  sub_C7D6A0(v411, 8LL * (unsigned int)v413, 8);
  sub_C7D6A0(v407, 8LL * (unsigned int)v409, 8);
  v230 = v405;
  if ( v405 )
  {
    v231 = v403;
    v232 = v403 + 32LL * v405;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v231 <= 0xFFFFFFFD )
        {
          v233 = *(_QWORD *)(v231 + 8);
          if ( v233 )
            break;
        }
        v231 += 32;
        if ( v232 == v231 )
          goto LABEL_294;
      }
      v231 += 32;
      j_j___libc_free_0(v233);
    }
    while ( v232 != v231 );
LABEL_294:
    v230 = v405;
  }
  sub_C7D6A0(v403, 32LL * v230, 8);
  v50 = v401;
  if ( v401 )
  {
    v234 = v399;
    v235 = v399 + 32LL * v401;
    do
    {
      if ( *(_DWORD *)v234 <= 0xFFFFFFFD )
      {
        v236 = *(_QWORD *)(v234 + 8);
        if ( v236 )
          j_j___libc_free_0(v236);
      }
      v234 += 32;
    }
    while ( v235 != v234 );
    v50 = v401;
  }
  sub_C7D6A0(v399, 32LL * v50, 8);
  if ( v375 )
    j_j___libc_free_0(v375);
  sub_C7D6A0(v395, 16LL * v397, 8);
LABEL_58:
  if ( v372 )
    j_j___libc_free_0((unsigned __int64)v372);
  v51 = v393;
  if ( v393 )
  {
    v52 = v391;
    v53 = &v391[4 * v393];
    do
    {
      if ( *v52 != -8192 && *v52 != -4096 )
      {
        v54 = v52[1];
        if ( v54 )
          j_j___libc_free_0(v54);
      }
      v52 += 4;
    }
    while ( v53 != v52 );
    v51 = v393;
  }
  sub_C7D6A0((__int64)v391, 32LL * v51, 8);
  if ( v369 )
    j_j___libc_free_0((unsigned __int64)v369);
  v55 = v389;
  if ( v389 )
  {
    v56 = v387;
    v57 = &v387[4 * v389];
    do
    {
      if ( *v56 != -8192 && *v56 != -4096 )
      {
        v58 = v56[1];
        if ( v58 )
          j_j___libc_free_0(v58);
      }
      v56 += 4;
    }
    while ( v57 != v56 );
    v55 = v389;
  }
  sub_C7D6A0((__int64)v387, 32LL * v55, 8);
  if ( v366 )
    j_j___libc_free_0(v366);
  sub_C7D6A0((__int64)v383, 16LL * v385, 8);
  sub_27C20B0((__int64)v420);
  return v324;
}
