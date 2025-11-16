// Function: sub_1EE1050
// Address: 0x1ee1050
//
_BOOL8 __fastcall sub_1EE1050(__int64 a1, __int64 *a2, int a3)
{
  __int64 *v3; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r8
  int v15; // r9d
  unsigned int v16; // r11d
  __int64 v17; // r9
  __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r8
  __int64 v21; // rdi
  __int64 (*v22)(); // rax
  char v23; // bl
  __int64 v24; // r14
  unsigned __int64 v25; // rax
  __int64 i; // rcx
  __int64 v27; // rdi
  unsigned int v28; // ecx
  unsigned int v29; // esi
  __int64 *v30; // rdx
  __int64 v31; // r8
  int v32; // r9d
  __int64 v33; // rbx
  unsigned __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r8
  __int64 v37; // r10
  __int64 *v38; // rcx
  __int64 v39; // r14
  int v40; // r12d
  __int64 *v41; // rax
  int v42; // r8d
  int v43; // r9d
  __int64 v44; // rbx
  _QWORD *v45; // rax
  __int64 v46; // rax
  unsigned int v47; // ecx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  int v51; // r9d
  __int64 v52; // r12
  __int64 *v53; // rax
  unsigned __int64 *v54; // r14
  unsigned __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  __int64 v59; // rdi
  void (*v60)(); // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  int v64; // r9d
  __int64 v65; // r14
  unsigned __int64 v66; // rcx
  unsigned int v67; // edx
  __int64 v68; // r8
  __int64 v69; // rax
  __int64 v70; // r12
  __int64 v71; // r14
  __int64 *v72; // rdx
  __int64 *v73; // rax
  __int64 v74; // r8
  _BYTE *v75; // r9
  __int64 v76; // rcx
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 *v79; // rdx
  __int64 v80; // rbx
  __int64 *v81; // rax
  __int64 v82; // rsi
  __int64 v83; // r12
  int v84; // r13d
  __int64 *v85; // rcx
  __int64 v86; // rsi
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // r14
  __int64 v91; // r13
  __int64 v92; // rsi
  unsigned __int64 j; // rcx
  __int64 v94; // rdi
  __int64 v95; // rsi
  __int64 *v96; // rax
  __int64 v97; // r10
  __int64 v98; // r14
  int v99; // ebx
  __int64 *v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 k; // rdx
  _WORD *v104; // rax
  unsigned int v105; // r12d
  __int64 v106; // rdx
  int v107; // eax
  __int64 v108; // r12
  __int64 *v109; // rax
  unsigned int v110; // r11d
  __int64 v111; // rcx
  __int64 *v112; // rax
  char v113; // al
  char v114; // al
  __int64 v115; // rcx
  int v116; // r8d
  int v117; // r9d
  __int64 v118; // rcx
  int v119; // r8d
  _DWORD *v120; // rax
  __int64 v121; // r14
  char v122; // si
  int v123; // ecx
  int v124; // r9d
  __int64 v125; // r8
  int v126; // r11d
  unsigned __int64 v127; // rax
  unsigned int v128; // edi
  __int64 v129; // r10
  unsigned int v130; // ecx
  __int64 v131; // r11
  __int64 v132; // r12
  __int64 v133; // rdx
  unsigned __int64 jj; // rsi
  unsigned int v135; // ecx
  __int64 v136; // rdi
  unsigned int v137; // edx
  __int64 *v138; // rax
  __int64 v139; // r11
  __int64 *v140; // rdx
  __int64 *v141; // rax
  __int64 *v142; // r14
  __int64 v143; // r11
  __int64 *v144; // r8
  __int64 kk; // r13
  __int64 v146; // rdx
  __int64 *v147; // rax
  __int64 *v148; // rcx
  __int64 v149; // rax
  unsigned __int64 v150; // rdi
  __int64 v151; // rdx
  __int64 v152; // rbx
  __int64 v153; // rax
  unsigned __int64 v154; // rdx
  int v155; // eax
  __int64 v156; // r12
  __int64 v157; // rbx
  __int64 m; // r12
  _QWORD *v159; // rdx
  unsigned int v160; // ebx
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 v163; // r8
  int v164; // r9d
  int v165; // r9d
  __int64 **v166; // rax
  __int64 v167; // r14
  __int64 v168; // kr00_8
  unsigned __int64 v169; // rax
  unsigned int v170; // ebx
  __int64 v171; // r8
  __int64 v172; // rcx
  unsigned int v173; // edx
  __int16 v174; // r14
  _WORD *v175; // rdx
  __int16 *v176; // rbx
  unsigned __int16 v177; // r14
  __int64 v178; // rax
  unsigned __int16 *v179; // rax
  unsigned __int16 v180; // cx
  unsigned __int16 v181; // si
  unsigned __int64 v182; // rdx
  __int64 v183; // rax
  __int64 v184; // r13
  __int16 v185; // ax
  int v186; // ebx
  __int64 v187; // rax
  __int64 v188; // rdx
  __int64 v189; // rcx
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // r12
  __int64 v193; // rcx
  __int64 v194; // r8
  __int64 v195; // r9
  _QWORD *v196; // rdi
  __int64 v197; // rax
  __int64 v198; // rcx
  __int64 v199; // r8
  __int64 v200; // r9
  __int64 v201; // rbx
  __int64 n; // r12
  int v203; // ebx
  unsigned int v204; // ebx
  __int64 v205; // r12
  _BYTE *v206; // rax
  _QWORD *v207; // rdx
  unsigned int v208; // ebx
  __int64 v209; // rcx
  __int64 v210; // rdx
  __int64 v211; // r11
  __int64 v212; // rdi
  __int64 v213; // rdi
  _QWORD *v214; // rsi
  _QWORD *v215; // rax
  __int64 v216; // rdx
  __int64 v217; // rdi
  _QWORD *v218; // rsi
  _QWORD *v219; // rcx
  __int64 v220; // rax
  _QWORD *v221; // r9
  unsigned __int64 v222; // r14
  __int64 v223; // rax
  __int64 v224; // rdx
  unsigned __int64 v225; // rcx
  __int64 v226; // rdi
  unsigned int v227; // edx
  unsigned int v228; // esi
  __int64 *v229; // rax
  __int64 v230; // r8
  unsigned __int64 v231; // rcx
  unsigned __int64 ii; // r10
  unsigned int v233; // esi
  __int64 *v234; // rax
  __int64 v235; // r8
  unsigned __int64 v236; // rbx
  char v237; // al
  __int64 v238; // rdi
  __int64 v239; // r10
  __int64 v240; // r8
  unsigned __int64 v241; // rax
  unsigned __int64 v242; // r14
  __int64 v243; // rbx
  __int64 v244; // r15
  __int64 v245; // r12
  __int64 v246; // r8
  __int64 *v247; // r9
  __int64 v248; // r13
  __int64 *v249; // r10
  __int64 v250; // rdi
  int v251; // eax
  unsigned __int64 v252; // rax
  __int64 v253; // rcx
  unsigned int v254; // edx
  __int16 v255; // bx
  _WORD *v256; // rdx
  _WORD *v257; // rcx
  unsigned __int16 v258; // bx
  _WORD *v259; // r13
  _QWORD *v260; // r14
  __int64 *v261; // r12
  __int16 v262; // ax
  int v263; // edx
  int v264; // edx
  int v265; // r10d
  __int64 v266; // rdx
  __int64 *v267; // rax
  __int64 v268; // rdx
  _QWORD *v269; // rax
  _QWORD *v270; // r8
  __int64 v271; // rsi
  _QWORD *v272; // rdx
  _QWORD *v273; // rax
  __int64 v274; // rax
  _QWORD *v275; // r14
  __int64 v276; // rax
  __int64 v277; // rax
  unsigned __int64 v278; // rdx
  __int32 v279; // eax
  unsigned __int64 v280; // rbx
  __int64 *v281; // rdi
  __int64 v282; // rcx
  __int64 v283; // r14
  __int64 v284; // r12
  __int64 *v285; // rbx
  _QWORD *v286; // rdx
  _QWORD *v287; // rax
  _QWORD *v288; // rcx
  __int64 v289; // rdx
  _QWORD *v290; // rdx
  __int64 v291; // rcx
  __int64 v292; // r8
  __int64 v293; // r9
  __int64 v294; // r13
  __int64 v295; // rbx
  unsigned int v296; // esi
  unsigned int v297; // ecx
  __int64 v298; // r8
  __int64 *v299; // rax
  __int64 v300; // rdi
  __int64 v301; // rax
  unsigned int v302; // r12d
  __int64 v303; // rdx
  unsigned int v304; // r12d
  __int64 v305; // rdx
  _QWORD *v306; // rdi
  _DWORD *v307; // rcx
  __int64 v308; // rdx
  __int64 v309; // r9
  unsigned int v310; // esi
  unsigned int v311; // eax
  __int64 v312; // r10
  __int64 v313; // rcx
  __int64 v314; // r8
  __int64 v315; // r9
  _QWORD *v316; // rdi
  _QWORD *v317; // r12
  __int64 v318; // rsi
  _QWORD *v319; // rdi
  int v320; // eax
  int v321; // r9d
  __int64 v322; // rax
  int v323; // eax
  int v324; // r11d
  int v325; // eax
  int v326; // r10d
  __int64 v327; // rsi
  _QWORD *v328; // rcx
  _QWORD *v329; // rax
  __int64 v330; // rsi
  _QWORD *v331; // rcx
  _QWORD *v332; // rax
  __int64 v333; // [rsp-8h] [rbp-188h]
  __int64 v334; // [rsp+8h] [rbp-178h]
  _QWORD *v335; // [rsp+8h] [rbp-178h]
  __int64 *v336; // [rsp+10h] [rbp-170h]
  char v337; // [rsp+10h] [rbp-170h]
  __int64 *v338; // [rsp+18h] [rbp-168h]
  __int64 *v339; // [rsp+18h] [rbp-168h]
  __int64 *v340; // [rsp+20h] [rbp-160h]
  __int64 *v341; // [rsp+20h] [rbp-160h]
  __int64 *v342; // [rsp+20h] [rbp-160h]
  __int64 v343; // [rsp+20h] [rbp-160h]
  __int32 v344; // [rsp+20h] [rbp-160h]
  __int64 v345; // [rsp+20h] [rbp-160h]
  __int64 v346; // [rsp+20h] [rbp-160h]
  __int64 v347; // [rsp+28h] [rbp-158h]
  __int64 v348; // [rsp+28h] [rbp-158h]
  unsigned __int64 v349; // [rsp+28h] [rbp-158h]
  __int64 v350; // [rsp+28h] [rbp-158h]
  __int64 v351; // [rsp+28h] [rbp-158h]
  int v352; // [rsp+28h] [rbp-158h]
  __int64 v353; // [rsp+28h] [rbp-158h]
  unsigned __int64 v354; // [rsp+30h] [rbp-150h]
  __int64 **v355; // [rsp+30h] [rbp-150h]
  __int64 v356; // [rsp+30h] [rbp-150h]
  __int64 v357; // [rsp+30h] [rbp-150h]
  _QWORD *v358; // [rsp+30h] [rbp-150h]
  __int64 v359; // [rsp+30h] [rbp-150h]
  __int64 v360; // [rsp+30h] [rbp-150h]
  int v361; // [rsp+30h] [rbp-150h]
  __int64 *v362; // [rsp+38h] [rbp-148h]
  __int64 v363; // [rsp+38h] [rbp-148h]
  __int64 *v364; // [rsp+38h] [rbp-148h]
  __int64 v365; // [rsp+38h] [rbp-148h]
  __int64 v366; // [rsp+38h] [rbp-148h]
  bool v367; // [rsp+38h] [rbp-148h]
  __int64 v368; // [rsp+38h] [rbp-148h]
  int v369; // [rsp+38h] [rbp-148h]
  bool v370; // [rsp+40h] [rbp-140h]
  char v371; // [rsp+40h] [rbp-140h]
  __int64 v372; // [rsp+40h] [rbp-140h]
  __int64 v373; // [rsp+40h] [rbp-140h]
  __int64 v374; // [rsp+40h] [rbp-140h]
  char v375; // [rsp+48h] [rbp-138h]
  __int64 v376; // [rsp+48h] [rbp-138h]
  __int64 v377; // [rsp+48h] [rbp-138h]
  __int64 v378; // [rsp+48h] [rbp-138h]
  unsigned __int64 v379; // [rsp+48h] [rbp-138h]
  __int64 v380; // [rsp+48h] [rbp-138h]
  __int64 v381; // [rsp+48h] [rbp-138h]
  unsigned int v382; // [rsp+48h] [rbp-138h]
  unsigned __int64 v383; // [rsp+50h] [rbp-130h]
  __int64 v384; // [rsp+50h] [rbp-130h]
  __int64 v385; // [rsp+50h] [rbp-130h]
  __int64 *v386; // [rsp+50h] [rbp-130h]
  __int64 v387; // [rsp+50h] [rbp-130h]
  _QWORD *v388; // [rsp+50h] [rbp-130h]
  __int64 v389; // [rsp+50h] [rbp-130h]
  unsigned __int64 v390; // [rsp+58h] [rbp-128h]
  unsigned __int64 v391; // [rsp+58h] [rbp-128h]
  __int64 v392; // [rsp+60h] [rbp-120h]
  __int64 v393; // [rsp+68h] [rbp-118h]
  __int64 v394; // [rsp+68h] [rbp-118h]
  __int64 v395; // [rsp+68h] [rbp-118h]
  __int64 *v396; // [rsp+68h] [rbp-118h]
  __int64 v397; // [rsp+68h] [rbp-118h]
  int v398; // [rsp+78h] [rbp-108h] BYREF
  unsigned int v399; // [rsp+7Ch] [rbp-104h] BYREF
  __int64 v400; // [rsp+80h] [rbp-100h] BYREF
  __int64 v401; // [rsp+88h] [rbp-F8h]
  __int64 v402; // [rsp+90h] [rbp-F0h]
  __int16 v403; // [rsp+98h] [rbp-E8h]
  char v404; // [rsp+9Ah] [rbp-E6h]
  __int64 **v405; // [rsp+A0h] [rbp-E0h]
  __int64 *v406; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v407; // [rsp+B8h] [rbp-C8h]
  _BYTE v408[64]; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v409; // [rsp+100h] [rbp-80h] BYREF
  __int64 v410; // [rsp+110h] [rbp-70h] BYREF
  __int64 v411; // [rsp+118h] [rbp-68h]
  __int64 v412; // [rsp+120h] [rbp-60h]

  if ( a3 )
  {
    v370 = 0;
    v3 = a2;
    v392 = (__int64)&a2[(unsigned int)(a3 - 1) + 1];
    while ( 1 )
    {
      v5 = *v3;
      if ( !*v3 )
        goto LABEL_4;
      v6 = *(_QWORD **)(a1 + 576);
      v7 = *(_QWORD **)(a1 + 568);
      v393 = a1 + 560;
      if ( v6 == v7 )
      {
        v8 = &v7[*(unsigned int *)(a1 + 588)];
        if ( v7 == v8 )
        {
          v207 = *(_QWORD **)(a1 + 568);
        }
        else
        {
          do
          {
            if ( v5 == *v7 )
              break;
            ++v7;
          }
          while ( v8 != v7 );
          v207 = v8;
        }
      }
      else
      {
        v8 = &v6[*(unsigned int *)(a1 + 584)];
        v7 = sub_16CC9F0(v393, v5);
        if ( v5 == *v7 )
        {
          v61 = *(_QWORD *)(a1 + 576);
          if ( v61 == *(_QWORD *)(a1 + 568) )
            v62 = *(unsigned int *)(a1 + 588);
          else
            v62 = *(unsigned int *)(a1 + 584);
          v207 = (_QWORD *)(v61 + 8 * v62);
        }
        else
        {
          v9 = *(_QWORD *)(a1 + 576);
          if ( v9 != *(_QWORD *)(a1 + 568) )
          {
            v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(a1 + 584));
            goto LABEL_10;
          }
          v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(a1 + 588));
          v207 = v7;
        }
      }
      while ( v207 != v7 && *v7 >= 0xFFFFFFFFFFFFFFFELL )
        ++v7;
LABEL_10:
      if ( v7 != v8 )
        goto LABEL_3;
      v10 = *v3;
      v11 = *(_QWORD *)(a1 + 256);
      v12 = &v400;
      v403 = 0;
      v390 = v10;
      v400 = v11;
      v401 = 0;
      v402 = 0;
      v404 = 0;
      v405 = 0;
      v375 = sub_1EDADD0((__int64)&v400, v10);
      if ( !v375 )
        goto LABEL_3;
      if ( !v405 )
        goto LABEL_31;
      v16 = HIDWORD(v402);
      v17 = (unsigned int)v402;
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL);
      v19 = *(_QWORD *)(v18 + 16LL * (HIDWORD(v401) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      v20 = *(_QWORD *)(v18 + 16 * (v401 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v404 )
      {
        v16 = v402;
        v17 = HIDWORD(v402);
        v19 = *(_QWORD *)(v18 + 16 * (v401 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
        v20 = *(_QWORD *)(v18 + 16LL * (HIDWORD(v401) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      }
      v21 = *(_QWORD *)(a1 + 256);
      v22 = *(__int64 (**)())(*(_QWORD *)v21 + 408LL);
      if ( v22 == sub_1ED7CF0 )
        goto LABEL_16;
      v113 = ((__int64 (__fastcall *)(__int64, unsigned __int64, unsigned __int64, _QWORD, unsigned __int64, __int64, __int64 **, _QWORD))v22)(
               v21,
               v390,
               v19,
               v16,
               v20,
               v17,
               v405,
               *(_QWORD *)(a1 + 272));
      v14 = v333;
      if ( v113 )
      {
        if ( !v405 )
          goto LABEL_31;
LABEL_16:
        v23 = sub_1E17E50(v390);
        if ( v23 )
        {
          v190 = *(unsigned int *)(a1 + 672);
          if ( (unsigned int)v190 >= *(_DWORD *)(a1 + 676) )
          {
            sub_16CD150(a1 + 664, (const void *)(a1 + 680), 0, 8, v14, v15);
            v190 = *(unsigned int *)(a1 + 672);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 664) + 8 * v190) = v390;
          ++*(_DWORD *)(a1 + 672);
          sub_1ED8620(a1);
          v370 = v23;
          goto LABEL_3;
        }
        if ( v405 )
        {
          sub_1ED87E0(*(_QWORD *)(a1 + 256), v390, &v398, &v399, (int *)&v406, v409.m128i_i32);
          v24 = *(_QWORD *)(a1 + 272);
          v25 = v390;
          for ( i = *(_QWORD *)(v24 + 272); (*(_BYTE *)(v25 + 46) & 4) != 0; v25 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL )
            ;
          v27 = *(_QWORD *)(i + 368);
          v28 = *(_DWORD *)(i + 384);
          if ( v28 )
          {
            v29 = (v28 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v30 = (__int64 *)(v27 + 16LL * v29);
            v31 = *v30;
            if ( v25 == *v30 )
            {
LABEL_22:
              v32 = v398;
              v33 = v30[1];
              v34 = *(unsigned int *)(v24 + 408);
              v35 = v398 & 0x7FFFFFFF;
              v36 = v398 & 0x7FFFFFFF;
              v37 = 8 * v36;
              if ( (v398 & 0x7FFFFFFFu) < (unsigned int)v34 )
              {
                v38 = *(__int64 **)(*(_QWORD *)(v24 + 400) + 8LL * v35);
                if ( v38 )
                  goto LABEL_24;
              }
              v110 = v35 + 1;
              if ( (unsigned int)v34 >= v35 + 1 )
                goto LABEL_145;
              v191 = v110;
              if ( v110 < v34 )
              {
                *(_DWORD *)(v24 + 408) = v110;
                goto LABEL_145;
              }
              if ( v110 <= v34 )
              {
LABEL_145:
                v111 = *(_QWORD *)(v24 + 400);
              }
              else
              {
                if ( v110 > (unsigned __int64)*(unsigned int *)(v24 + 412) )
                {
                  v343 = v398 & 0x7FFFFFFF;
                  v369 = v398;
                  v387 = v110;
                  sub_16CD150(v24 + 400, (const void *)(v24 + 416), v110, 8, v36, v398);
                  v34 = *(unsigned int *)(v24 + 408);
                  v36 = v343;
                  v37 = 8 * v343;
                  v110 = v35 + 1;
                  v32 = v369;
                  v191 = v387;
                }
                v111 = *(_QWORD *)(v24 + 400);
                v214 = (_QWORD *)(v111 + 8 * v191);
                v215 = (_QWORD *)(v111 + 8 * v34);
                v216 = *(_QWORD *)(v24 + 416);
                if ( v214 != v215 )
                {
                  do
                    *v215++ = v216;
                  while ( v214 != v215 );
                  v111 = *(_QWORD *)(v24 + 400);
                }
                *(_DWORD *)(v24 + 408) = v110;
              }
              v363 = v36;
              *(_QWORD *)(v111 + v37) = sub_1DBA290(v32);
              v384 = *(_QWORD *)(*(_QWORD *)(v24 + 400) + 8 * v363);
              sub_1DBB110((_QWORD *)v24, v384);
              v38 = (__int64 *)v384;
LABEL_24:
              v383 = v33 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (_DWORD)v406 && (v39 = v38[13]) != 0 )
              {
                v40 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 248LL) + 4LL * (unsigned int)v406);
                while ( 1 )
                {
                  if ( (*(_DWORD *)(v39 + 112) & v40) != 0 )
                  {
                    v41 = (__int64 *)sub_1DB3C70((__int64 *)v39, v33);
                    if ( v41 != (__int64 *)(*(_QWORD *)v39 + 24LL * *(unsigned int *)(v39 + 8))
                      && (*(_DWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v41 >> 1) & 3) <= (*(_DWORD *)(v383 + 24) | (unsigned int)(v33 >> 1) & 3) )
                    {
                      break;
                    }
                  }
                  v39 = *(_QWORD *)(v39 + 104);
                  if ( !v39 )
                    goto LABEL_76;
                }
                v12 = &v400;
              }
              else
              {
                v364 = v38;
                v112 = (__int64 *)sub_1DB3C70(v38, v33);
                if ( v112 == (__int64 *)(*v364 + 24LL * *((unsigned int *)v364 + 2))
                  || (v13 = *(_DWORD *)(v383 + 24) | (unsigned int)(v33 >> 1) & 3,
                      (*(_DWORD *)((*v112 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v112 >> 1) & 3) > (unsigned int)v13) )
                {
LABEL_76:
                  v64 = v399;
                  v65 = *(_QWORD *)(a1 + 272);
                  v66 = *(unsigned int *)(v65 + 408);
                  v67 = v399 & 0x7FFFFFFF;
                  v68 = v399 & 0x7FFFFFFF;
                  v69 = 8 * v68;
                  if ( (v399 & 0x7FFFFFFF) < (unsigned int)v66 )
                  {
                    v70 = *(_QWORD *)(*(_QWORD *)(v65 + 400) + 8LL * v67);
                    if ( v70 )
                    {
LABEL_78:
                      v71 = v383 | 4;
                      v72 = (__int64 *)sub_1DB3C70((__int64 *)v70, v383 | 4);
                      if ( v72 == (__int64 *)(*(_QWORD *)v70 + 24LL * *(unsigned int *)(v70 + 8))
                        || (*(_DWORD *)((*v72 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v72 >> 1) & 3) > (*(_DWORD *)(v383 + 24) | 2u) )
                      {
                        BUG();
                      }
                      v395 = v72[1];
                      v73 = (__int64 *)sub_1DB3C70((__int64 *)v70, v395);
                      v76 = (__int64)v73;
                      if ( v73 == (__int64 *)(*(_QWORD *)v70 + 24LL * *(unsigned int *)(v70 + 8))
                        || (v77 = *(_DWORD *)((v395 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v395 >> 1) & 3,
                            (*(_DWORD *)((*v73 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v73 >> 1) & 3) > (unsigned int)v77)
                        || (v78 = v73[2]) == 0
                        || (*(_BYTE *)(v78 + 8) & 6) != 0 )
                      {
                        v79 = (__int64 *)sub_1DB3C70((__int64 *)v70, v33);
                        if ( v79 == (__int64 *)(*(_QWORD *)v70 + 24LL * *(unsigned int *)(v70 + 8))
                          || (*(_DWORD *)((*v79 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v79 >> 1) & 3) > (*(_DWORD *)(v383 + 24) | (unsigned int)((v33 >> 1) & 3))
                          || (v80 = v79[2]) == 0 )
                        {
                          sub_1DBEA10(*(_QWORD *)(a1 + 272), v70, v71);
                        }
                        else
                        {
                          v81 = (__int64 *)sub_1DB3C70((__int64 *)v70, v71);
                          v82 = 0;
                          if ( v81 != (__int64 *)(*(_QWORD *)v70 + 24LL * *(unsigned int *)(v70 + 8))
                            && (*(_DWORD *)((*v81 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v81 >> 1) & 3) <= (*(_DWORD *)(v383 + 24) | 2u) )
                          {
                            v82 = v81[2];
                          }
                          sub_1DB4840(v70, v82, v80);
                          if ( *(_QWORD *)(v70 + 104) )
                          {
                            v376 = v70;
                            v83 = *(_QWORD *)(v70 + 104);
                            v396 = v3;
                            v84 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 248LL) + 4LL * v409.m128i_u32[0]);
                            do
                            {
                              if ( (*(_DWORD *)(v83 + 112) & v84) != 0 )
                              {
                                v85 = (__int64 *)sub_1DB3C70((__int64 *)v83, v71);
                                v86 = 0;
                                if ( v85 != (__int64 *)(*(_QWORD *)v83 + 24LL * *(unsigned int *)(v83 + 8))
                                  && (*(_DWORD *)((*v85 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v85 >> 1) & 3) <= (*(_DWORD *)(v383 + 24) | 2u) )
                                {
                                  v86 = v85[2];
                                }
                                sub_1DB4670(v83, v86);
                              }
                              v83 = *(_QWORD *)(v83 + 104);
                            }
                            while ( v83 );
                            v3 = v396;
                            v70 = v376;
                          }
                          sub_1DB4C70(v70);
                        }
                        v89 = *(_QWORD *)(a1 + 248);
                        if ( (v399 & 0x80000000) != 0 )
                          v90 = *(_QWORD *)(*(_QWORD *)(v89 + 24) + 16LL * (v399 & 0x7FFFFFFF) + 8);
                        else
                          v90 = *(_QWORD *)(*(_QWORD *)(v89 + 272) + 8LL * v399);
                        if ( !v90 )
                          goto LABEL_118;
                        if ( (*(_BYTE *)(v90 + 4) & 8) == 0 )
                        {
LABEL_104:
                          v362 = v3;
                          v91 = v90;
LABEL_105:
                          if ( (*(_BYTE *)(v91 + 3) & 0x10) != 0 )
                            goto LABEL_116;
                          v92 = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL);
                          for ( j = *(_QWORD *)(v91 + 16);
                                (*(_BYTE *)(j + 46) & 4) != 0;
                                j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                          {
                            ;
                          }
                          v94 = *(_QWORD *)(v92 + 368);
                          v95 = *(unsigned int *)(v92 + 384);
                          if ( (_DWORD)v95 )
                          {
                            v88 = (unsigned int)(v95 - 1);
                            v87 = (unsigned int)v88 & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
                            v96 = (__int64 *)(v94 + 16 * v87);
                            v97 = *v96;
                            if ( j == *v96 )
                            {
LABEL_110:
                              v98 = v96[1];
                              v99 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 248LL)
                                              + 4LL * ((*(_DWORD *)v91 >> 8) & 0xFFF));
                              if ( v99 != -1 && *(_QWORD *)(v70 + 104) )
                              {
                                v377 = v70;
                                v108 = *(_QWORD *)(v70 + 104);
                                while ( 1 )
                                {
                                  if ( (*(_DWORD *)(v108 + 112) & v99) != 0 )
                                  {
                                    v109 = (__int64 *)sub_1DB3C70((__int64 *)v108, v98);
                                    if ( v109 != (__int64 *)(*(_QWORD *)v108 + 24LL * *(unsigned int *)(v108 + 8))
                                      && (*(_DWORD *)((*v109 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                        | (unsigned int)(*v109 >> 1) & 3) <= (*(_DWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                        + 24)
                                                                            | (unsigned int)(v98 >> 1) & 3) )
                                    {
                                      break;
                                    }
                                  }
                                  v108 = *(_QWORD *)(v108 + 104);
                                  if ( !v108 )
                                  {
                                    v70 = v377;
                                    goto LABEL_114;
                                  }
                                }
                                v70 = v377;
                              }
                              else
                              {
                                v100 = (__int64 *)sub_1DB3C70((__int64 *)v70, v98);
                                if ( v100 == (__int64 *)(*(_QWORD *)v70 + 24LL * *(unsigned int *)(v70 + 8))
                                  || (*(_DWORD *)((*v100 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v100 >> 1) & 3) > (*(_DWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v98 >> 1) & 3) )
                                {
LABEL_114:
                                  *(_BYTE *)(v91 + 4) |= 1u;
                                  v91 = *(_QWORD *)(v91 + 32);
                                  if ( !v91 )
                                    goto LABEL_117;
                                  goto LABEL_115;
                                }
                              }
LABEL_116:
                              while ( 1 )
                              {
                                v91 = *(_QWORD *)(v91 + 32);
                                if ( !v91 )
                                  break;
LABEL_115:
                                if ( (*(_BYTE *)(v91 + 4) & 8) == 0 )
                                  goto LABEL_105;
                              }
LABEL_117:
                              v3 = v362;
                              goto LABEL_118;
                            }
                            v107 = 1;
                            while ( v97 != -8 )
                            {
                              v263 = v107 + 1;
                              v87 = (unsigned int)v88 & (v107 + (_DWORD)v87);
                              v96 = (__int64 *)(v94 + 16LL * (unsigned int)v87);
                              v97 = *v96;
                              if ( *v96 == j )
                                goto LABEL_110;
                              v107 = v263;
                            }
                          }
                          v96 = (__int64 *)(v94 + 16 * v95);
                          goto LABEL_110;
                        }
                        while ( 1 )
                        {
                          v90 = *(_QWORD *)(v90 + 32);
                          if ( !v90 )
                            break;
                          if ( (*(_BYTE *)(v90 + 4) & 8) == 0 )
                            goto LABEL_104;
                        }
LABEL_118:
                        v101 = v399;
                        v102 = *(_QWORD *)(v390 + 32);
                        for ( k = v102 + 40LL * *(unsigned int *)(v390 + 40); k != v102; v102 += 40 )
                        {
                          if ( !*(_BYTE *)v102
                            && (*(_BYTE *)(v102 + 3) & 0x10) != 0
                            && *(_DWORD *)(v102 + 8) == (_DWORD)v101 )
                          {
                            *(_BYTE *)(v102 + 4) |= 1u;
                          }
                        }
                        sub_1DC0580(*(_QWORD **)(a1 + 272), v70, 0, v101, v87, v88);
LABEL_125:
                        v104 = *(_WORD **)(v390 + 16);
                      }
                      else
                      {
                        v104 = (_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL) + 576LL);
                        *(_QWORD *)(v390 + 16) = v104;
                        v203 = *(_DWORD *)(v390 + 40);
                        if ( v203 )
                        {
                          v204 = v203 - 1;
                          v205 = 40LL * v204;
                          while ( 1 )
                          {
                            v206 = (_BYTE *)(v205 + *(_QWORD *)(v390 + 32));
                            if ( !*v206 && (v206[3] & 0x10) == 0 )
                              sub_1E16C90(v390, v204, v77, v76, v74, v75);
                            v205 -= 40;
                            if ( !v204 )
                              break;
                            --v204;
                          }
                          goto LABEL_125;
                        }
                      }
                      if ( *v104 != 9 )
                        sub_1ED8E30(a1, v390);
                      goto LABEL_3;
                    }
                  }
                  v105 = v67 + 1;
                  if ( (unsigned int)v66 >= v67 + 1 )
                    goto LABEL_132;
                  v211 = v105;
                  if ( v105 < v66 )
                  {
                    *(_DWORD *)(v65 + 408) = v105;
                    goto LABEL_132;
                  }
                  if ( v105 <= v66 )
                  {
LABEL_132:
                    v106 = *(_QWORD *)(v65 + 400);
                  }
                  else
                  {
                    if ( v105 > (unsigned __int64)*(unsigned int *)(v65 + 412) )
                    {
                      v357 = v399 & 0x7FFFFFFF;
                      v382 = v399;
                      sub_16CD150(v65 + 400, (const void *)(v65 + 416), v105, 8, v68, v399);
                      v66 = *(unsigned int *)(v65 + 408);
                      v68 = v357;
                      v69 = 8 * v357;
                      v64 = v382;
                      v211 = v105;
                    }
                    v106 = *(_QWORD *)(v65 + 400);
                    v217 = *(_QWORD *)(v65 + 416);
                    v218 = (_QWORD *)(v106 + 8 * v211);
                    v219 = (_QWORD *)(v106 + 8 * v66);
                    if ( v218 != v219 )
                    {
                      do
                        *v219++ = v217;
                      while ( v218 != v219 );
                      v106 = *(_QWORD *)(v65 + 400);
                    }
                    *(_DWORD *)(v65 + 408) = v105;
                  }
                  v397 = v68;
                  *(_QWORD *)(v106 + v69) = sub_1DBA290(v64);
                  v70 = *(_QWORD *)(*(_QWORD *)(v65 + 400) + 8 * v397);
                  sub_1DBB110((_QWORD *)v65, v70);
                  goto LABEL_78;
                }
              }
              goto LABEL_31;
            }
            v264 = 1;
            while ( v31 != -8 )
            {
              v265 = v264 + 1;
              v29 = (v28 - 1) & (v29 + v264);
              v30 = (__int64 *)(v27 + 16LL * v29);
              v31 = *v30;
              if ( *v30 == v25 )
                goto LABEL_22;
              v264 = v265;
            }
          }
          v30 = (__int64 *)(v27 + 16LL * v28);
          goto LABEL_22;
        }
LABEL_31:
        if ( HIDWORD(v401) == (_DWORD)v401 )
        {
          v156 = sub_1E86160(*(_QWORD *)(a1 + 272), SHIDWORD(v401), v13, (unsigned int)v401, v14, v15);
          v157 = sub_1E85F30(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), v390);
          sub_1E86030((__int64)&v406, v156, v157);
          if ( v406 != (__int64 *)v407 )
          {
            if ( v407 )
            {
              sub_1DB4840(v156, v407, (__int64)v406);
              for ( m = *(_QWORD *)(v156 + 104); m; m = *(_QWORD *)(m + 104) )
              {
                sub_1E86030((__int64)&v409, m, v157);
                if ( v409.m128i_i64[1] && v409.m128i_i64[0] != v409.m128i_i64[1] )
                  sub_1DB4840(m, v409.m128i_i64[1], v409.m128i_i64[0]);
              }
            }
          }
          goto LABEL_202;
        }
        if ( !v405 )
        {
          v210 = (unsigned int)v401 >> 6;
          if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 304LL) + 8 * v210) & (1LL << v401)) != 0
            && *(_DWORD *)(sub_1E86160(*(_QWORD *)(a1 + 272), SHIDWORD(v401), v210, (unsigned int)v401, v14, v15) + 72) == 1 )
          {
            goto LABEL_213;
          }
          v114 = sub_1EDF9B0(a1, (__int64)&v400, v390, &v409);
          if ( !v114 )
          {
            if ( v409.m128i_i8[0] )
              goto LABEL_4;
            goto LABEL_3;
          }
          goto LABEL_203;
        }
        if ( (_BYTE)v403 )
        {
          *(_DWORD *)(a1 + 392) = 0;
          *(_BYTE *)(a1 + 396) = 0;
LABEL_35:
          if ( (unsigned __int8)sub_1EDD780(a1, (__int64)&v400) )
            goto LABEL_36;
          goto LABEL_155;
        }
        v160 = *(_DWORD *)(sub_1E86160(*(_QWORD *)(a1 + 272), SHIDWORD(v401), v13, (unsigned int)v401, v14, v15) + 8);
        if ( v160 > *(_DWORD *)(sub_1E86160(*(_QWORD *)(a1 + 272), v401, v161, v162, v163, v164) + 8) )
          sub_1EDB070((__int64)&v400);
LABEL_213:
        v166 = v405;
        *(_BYTE *)(a1 + 396) = 0;
        *(_DWORD *)(a1 + 392) = 0;
        if ( v166 )
          goto LABEL_35;
        v167 = *(_QWORD *)(a1 + 272);
        v168 = v401;
        v169 = *(unsigned int *)(v167 + 408);
        v170 = HIDWORD(v401) & 0x7FFFFFFF;
        v348 = HIDWORD(v401) & 0x7FFFFFFF;
        v171 = 8 * v348;
        if ( (HIDWORD(v401) & 0x7FFFFFFFu) < (unsigned int)v169 )
        {
          v355 = *(__int64 ***)(*(_QWORD *)(v167 + 400) + 8LL * v170);
          if ( v355 )
          {
LABEL_216:
            if ( !(unsigned __int8)sub_1E69FD0(*(_QWORD **)(a1 + 248), v168) )
            {
              v172 = *(_QWORD *)(a1 + 256);
              if ( !v172 )
                BUG();
              v341 = v3;
              v173 = *(_DWORD *)(*(_QWORD *)(v172 + 8) + 24LL * (unsigned int)v168 + 16);
              v174 = v168 * (v173 & 0xF);
              v175 = (_WORD *)(*(_QWORD *)(v172 + 56) + 2LL * (v173 >> 4));
              v176 = v175 + 1;
              v177 = *v175 + v174;
              while ( v176 )
              {
                v178 = *(_QWORD *)(a1 + 256);
                if ( !v178 )
                  BUG();
                v179 = (unsigned __int16 *)(*(_QWORD *)(v178 + 48) + 4LL * v177);
                v180 = *v179;
                v181 = v179[1];
                while ( v180 )
                {
                  v182 = (unsigned __int64)v180 >> 6;
                  v183 = 1LL << v180;
                  v180 = v181;
                  v181 = 0;
                  if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 304LL) + 8 * v182) & v183) == 0 )
                    goto LABEL_154;
                }
                v184 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 672LL) + 8LL * v177);
                if ( !v184 )
                {
                  v335 = *(_QWORD **)(a1 + 272);
                  v337 = qword_4FC4440[20];
                  v269 = (_QWORD *)sub_22077B0(104);
                  v270 = v335;
                  v184 = (__int64)v269;
                  if ( v269 )
                  {
                    *v269 = v269 + 2;
                    v269[1] = 0x200000000LL;
                    v269[8] = v269 + 10;
                    v269[9] = 0x200000000LL;
                    if ( v337 )
                    {
                      v301 = sub_22077B0(48);
                      v270 = v335;
                      if ( v301 )
                      {
                        *(_DWORD *)(v301 + 8) = 0;
                        *(_QWORD *)(v301 + 16) = 0;
                        *(_QWORD *)(v301 + 24) = v301 + 8;
                        *(_QWORD *)(v301 + 32) = v301 + 8;
                        *(_QWORD *)(v301 + 40) = 0;
                      }
                      *(_QWORD *)(v184 + 96) = v301;
                    }
                    else
                    {
                      v269[12] = 0;
                    }
                  }
                  *(_QWORD *)(v270[84] + 8LL * v177) = v184;
                  sub_1DBA8F0(v270, v184, v177);
                }
                if ( *(_DWORD *)(v184 + 8) && (unsigned __int8)sub_1DB3D00(v355, v184, *(__int64 **)v184) )
                {
LABEL_154:
                  v3 = v341;
                  goto LABEL_155;
                }
                v185 = *v176++;
                if ( v185 )
                  v177 += v185;
                else
                  v176 = 0;
              }
              v212 = *(_QWORD *)(a1 + 272);
              v409 = 0u;
              v3 = v341;
              LODWORD(v410) = 0;
              if ( (unsigned __int8)sub_1DBCD10(v212, (__int64)v355, (__int64)&v409)
                && (*(_QWORD *)(v409.m128i_i64[0] + 8LL * ((unsigned int)v168 >> 6)) & (1LL << v168)) == 0 )
              {
                _libc_free(v409.m128i_u64[0]);
                goto LABEL_155;
              }
              _libc_free(v409.m128i_u64[0]);
            }
            v213 = *(_QWORD *)(a1 + 248);
            if ( v404 )
            {
              v356 = sub_1E69D00(v213, SHIDWORD(v168));
LABEL_273:
              sub_1ED8E30(a1, v356);
              sub_1E69E80(*(_QWORD *)(a1 + 248), SHIDWORD(v401));
LABEL_36:
              if ( HIBYTE(v403) )
                sub_1E693D0(*(_QWORD *)(a1 + 248), v401, (__int64)v405);
              if ( v405 )
              {
                v44 = *(_QWORD *)(a1 + 296) + 24LL * *((unsigned __int16 *)*v405 + 12);
                if ( *(_DWORD *)(a1 + 304) == *(_DWORD *)v44 )
                {
                  if ( !*(_BYTE *)(v44 + 8) )
                    goto LABEL_41;
LABEL_230:
                  v186 = v401;
                  v187 = *(unsigned int *)(a1 + 752);
                  if ( (unsigned int)v187 >= *(_DWORD *)(a1 + 756) )
                  {
                    sub_16CD150(a1 + 744, (const void *)(a1 + 760), 0, 4, v42, v43);
                    v187 = *(unsigned int *)(a1 + 752);
                  }
                  *(_DWORD *)(*(_QWORD *)(a1 + 744) + 4 * v187) = v186;
                  ++*(_DWORD *)(a1 + 752);
                }
                else
                {
                  sub_1ED7890(a1 + 296, v405);
                  if ( *(_BYTE *)(v44 + 8) )
                    goto LABEL_230;
                }
              }
LABEL_41:
              v45 = *(_QWORD **)(a1 + 568);
              if ( *(_QWORD **)(a1 + 576) == v45 )
              {
                v159 = &v45[*(unsigned int *)(a1 + 588)];
                if ( v45 == v159 )
                {
LABEL_236:
                  v45 = v159;
                }
                else
                {
                  while ( v390 != *v45 )
                  {
                    if ( v159 == ++v45 )
                      goto LABEL_236;
                  }
                }
              }
              else
              {
                v45 = sub_16CC9F0(v393, v390);
                if ( v390 == *v45 )
                {
                  v188 = *(_QWORD *)(a1 + 576);
                  if ( v188 == *(_QWORD *)(a1 + 568) )
                    v189 = *(unsigned int *)(a1 + 588);
                  else
                    v189 = *(unsigned int *)(a1 + 584);
                  v159 = (_QWORD *)(v188 + 8 * v189);
                }
                else
                {
                  v46 = *(_QWORD *)(a1 + 576);
                  if ( v46 != *(_QWORD *)(a1 + 568) )
                    goto LABEL_44;
                  v45 = (_QWORD *)(v46 + 8LL * *(unsigned int *)(a1 + 588));
                  v159 = v45;
                }
              }
              if ( v159 == v45 )
              {
LABEL_44:
                v47 = v402;
                if ( (_DWORD)v402 )
                  goto LABEL_210;
              }
              else
              {
                *v45 = -2;
                v47 = v402;
                ++*(_DWORD *)(a1 + 592);
                if ( v47 )
LABEL_210:
                  sub_1EDCD00((_QWORD *)a1, v401, v401, v47);
              }
              sub_1EDCD00((_QWORD *)a1, SHIDWORD(v401), v401, HIDWORD(v402));
              if ( *(_DWORD *)(a1 + 392) )
              {
                v197 = sub_1E86160(*(_QWORD *)(a1 + 272), v401, v48, v49, v50, v51);
                v201 = *(_QWORD *)(v197 + 104);
                for ( n = v197; v201; v201 = *(_QWORD *)(v201 + 104) )
                {
                  if ( (*(_DWORD *)(v201 + 112) & *(_DWORD *)(a1 + 392)) != 0 )
                    sub_1DBFFB0(*(_QWORD **)(a1 + 272), v201, *(_DWORD *)(n + 112), v198, v199, v200);
                }
                sub_1DB4C70(n);
              }
              if ( *(_BYTE *)(a1 + 396) )
              {
                v192 = sub_1E86160(*(_QWORD *)(a1 + 272), v401, v48, v49, v50, v51);
                v196 = *(_QWORD **)(a1 + 272);
                if ( (unsigned __int8)sub_1DC0580(v196, v192, 0, v193, v194, v195) )
                {
                  v409.m128i_i64[0] = (__int64)&v410;
                  v409.m128i_i64[1] = 0x800000000LL;
                  sub_1DBEB50((__int64)v196, v192, (__int64)&v409);
                  if ( (__int64 *)v409.m128i_i64[0] != &v410 )
                    _libc_free(v409.m128i_u64[0]);
                }
              }
              v52 = *(_QWORD *)(a1 + 272);
              v394 = 8LL * (HIDWORD(v401) & 0x7FFFFFFF);
              v53 = (__int64 *)(*(_QWORD *)(v52 + 400) + v394);
              v54 = (unsigned __int64 *)*v53;
              if ( *v53 )
              {
                sub_1DB4CE0(*v53);
                v55 = v54[12];
                v391 = v55;
                if ( v55 )
                {
                  v56 = *(_QWORD *)(v55 + 16);
                  while ( v56 )
                  {
                    sub_1ED8B20(*(_QWORD *)(v56 + 24));
                    v57 = v56;
                    v56 = *(_QWORD *)(v56 + 16);
                    j_j___libc_free_0(v57, 56);
                  }
                  j_j___libc_free_0(v391, 48);
                }
                v58 = v54[8];
                if ( (unsigned __int64 *)v58 != v54 + 10 )
                  _libc_free(v58);
                if ( (unsigned __int64 *)*v54 != v54 + 2 )
                  _libc_free(*v54);
                j_j___libc_free_0(v54, 120);
                v53 = (__int64 *)(*(_QWORD *)(v52 + 400) + v394);
              }
              *v53 = 0;
              v59 = *(_QWORD *)(a1 + 256);
              v370 = v375;
              v60 = *(void (**)())(*(_QWORD *)v59 + 248LL);
              if ( v60 != nullsub_746 )
                ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v60)(
                  v59,
                  HIDWORD(v401),
                  (unsigned int)v401,
                  *(_QWORD *)(a1 + 240));
              goto LABEL_3;
            }
            if ( (unsigned __int8)sub_1E69E00(v213, SHIDWORD(v168)) && sub_1DBCA20(*(_QWORD *)(a1 + 272), (__int64)v355) )
            {
              v220 = sub_1E69D00(*(_QWORD *)(a1 + 248), SHIDWORD(v168));
              v221 = *(_QWORD **)(a1 + 248);
              v222 = v220;
              if ( v168 < 0 )
                v223 = *(_QWORD *)(v221[3] + 16 * v348 + 8);
              else
                v223 = *(_QWORD *)(v221[34] + 8LL * HIDWORD(v168));
              while ( v223 && ((*(_BYTE *)(v223 + 3) & 0x10) != 0 || (*(_BYTE *)(v223 + 4) & 8) != 0) )
                v223 = *(_QWORD *)(v223 + 32);
              v356 = *(_QWORD *)(v223 + 16);
              v224 = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL);
              v225 = v356;
              if ( (*(_BYTE *)(v356 + 46) & 4) != 0 )
              {
                do
                  v225 = *(_QWORD *)v225 & 0xFFFFFFFFFFFFFFF8LL;
                while ( (*(_BYTE *)(v225 + 46) & 4) != 0 );
              }
              v226 = *(_QWORD *)(v224 + 368);
              v227 = *(_DWORD *)(v224 + 384);
              if ( v227 )
              {
                v228 = (v227 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
                v229 = (__int64 *)(v226 + 16LL * v228);
                v230 = *v229;
                if ( *v229 == v225 )
                  goto LABEL_301;
                v325 = 1;
                while ( v230 != -8 )
                {
                  v326 = v325 + 1;
                  v228 = (v227 - 1) & (v325 + v228);
                  v229 = (__int64 *)(v226 + 16LL * v228);
                  v230 = *v229;
                  if ( v225 == *v229 )
                    goto LABEL_301;
                  v325 = v326;
                }
              }
              v229 = (__int64 *)(v226 + 16LL * v227);
LABEL_301:
              v231 = v222;
              for ( ii = v229[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
                    (*(_BYTE *)(v231 + 46) & 4) != 0;
                    v231 = *(_QWORD *)v231 & 0xFFFFFFFFFFFFFFF8LL )
              {
                ;
              }
              if ( v227 )
              {
                v233 = (v227 - 1) & (((unsigned int)v231 >> 9) ^ ((unsigned int)v231 >> 4));
                v234 = (__int64 *)(v226 + 16LL * v233);
                v235 = *v234;
                if ( *v234 == v231 )
                  goto LABEL_305;
                v323 = 1;
                while ( v235 != -8 )
                {
                  v324 = v323 + 1;
                  v233 = (v227 - 1) & (v323 + v233);
                  v234 = (__int64 *)(v226 + 16LL * v233);
                  v235 = *v234;
                  if ( *v234 == v231 )
                    goto LABEL_305;
                  v323 = v324;
                }
              }
              v234 = (__int64 *)(v226 + 16LL * v227);
LABEL_305:
              v349 = ii;
              v236 = v234[1] & 0xFFFFFFFFFFFFFFF8LL;
              v368 = v236 | 4;
              v237 = sub_1E69FD0(v221, v168);
              v238 = *(_QWORD *)(a1 + 272);
              v239 = v349;
              if ( v237 )
                goto LABEL_321;
              v240 = *(_QWORD *)(v238 + 272);
              v241 = v236;
              do
              {
                v241 = *(_QWORD *)(v241 + 8);
                if ( v240 + 336 == v241 )
                {
                  v242 = *(_QWORD *)(v240 + 336) & 0xFFFFFFFFFFFFFFF8LL;
                  goto LABEL_310;
                }
              }
              while ( !*(_QWORD *)(v241 + 16) );
              v242 = v241 & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_310:
              if ( v349 == v242 )
              {
LABEL_321:
                sub_1DBE8F0(v238, v168, v239);
                v253 = *(_QWORD *)(a1 + 256);
                if ( !v253 )
                  BUG();
                v386 = v3;
                v254 = *(_DWORD *)(*(_QWORD *)(v253 + 8) + 24LL * (unsigned int)v168 + 16);
                v255 = v168 * (v254 & 0xF);
                v256 = (_WORD *)(*(_QWORD *)(v253 + 56) + 2LL * (v254 >> 4));
                v257 = v256 + 1;
                v258 = *v256 + v255;
                while ( 1 )
                {
                  v259 = v257;
                  if ( !v257 )
                    break;
                  while ( 1 )
                  {
                    v260 = *(_QWORD **)(a1 + 272);
                    v261 = *(__int64 **)(v260[84] + 8LL * v258);
                    if ( !v261 )
                    {
                      v371 = qword_4FC4440[20];
                      v267 = (__int64 *)sub_22077B0(104);
                      v268 = v258;
                      v261 = v267;
                      if ( v267 )
                      {
                        *v267 = (__int64)(v267 + 2);
                        v267[1] = 0x200000000LL;
                        v267[8] = (__int64)(v267 + 10);
                        v267[9] = 0x200000000LL;
                        if ( v371 )
                        {
                          v274 = sub_22077B0(48);
                          v268 = v258;
                          if ( v274 )
                          {
                            *(_DWORD *)(v274 + 8) = 0;
                            *(_QWORD *)(v274 + 16) = 0;
                            *(_QWORD *)(v274 + 24) = v274 + 8;
                            *(_QWORD *)(v274 + 32) = v274 + 8;
                            *(_QWORD *)(v274 + 40) = 0;
                          }
                          v261[12] = v274;
                        }
                        else
                        {
                          v267[12] = 0;
                        }
                      }
                      *(_QWORD *)(v260[84] + 8 * v268) = v261;
                      sub_1DBA8F0(v260, (__int64)v261, v258);
                      v260 = *(_QWORD **)(a1 + 272);
                    }
                    ++v259;
                    sub_1DB79D0(v261, v368, v260 + 37);
                    v262 = *(v259 - 1);
                    if ( !v262 )
                      break;
                    v258 += v262;
                    if ( !v259 )
                      goto LABEL_327;
                  }
                  v257 = 0;
                }
LABEL_327:
                v3 = v386;
                goto LABEL_273;
              }
              v243 = a1;
              v244 = *(_QWORD *)(v238 + 272);
              v245 = v240 + 336;
              v246 = v243;
              v247 = v3;
              v248 = v349;
              v249 = &v400;
              while ( 1 )
              {
                v250 = 0;
                if ( (v242 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                  v250 = *(_QWORD *)((v242 & 0xFFFFFFFFFFFFFFF8LL) + 16);
                v339 = v249;
                v342 = v247;
                v350 = v246;
                v251 = sub_1E165A0(v250, v168, 0, *(_QWORD *)(v246 + 256));
                v246 = v350;
                v247 = v342;
                v249 = v339;
                if ( v251 != -1 )
                  break;
                v252 = v242 & 0xFFFFFFFFFFFFFFF8LL;
                do
                {
                  v252 = *(_QWORD *)(v252 + 8);
                  if ( v245 == v252 )
                  {
                    v242 = *(_QWORD *)(v244 + 336) & 0xFFFFFFFFFFFFFFF8LL;
                    goto LABEL_319;
                  }
                }
                while ( !*(_QWORD *)(v252 + 16) );
                v242 = v252 & 0xFFFFFFFFFFFFFFF9LL | v242 & 6;
LABEL_319:
                if ( v248 == v242 )
                {
                  v238 = *(_QWORD *)(v350 + 272);
                  v239 = v248;
                  a1 = v350;
                  v3 = v342;
                  goto LABEL_321;
                }
              }
              v3 = v342;
              a1 = v350;
              v12 = v339;
            }
LABEL_155:
            v114 = sub_1EDF9B0(a1, (__int64)v12, v390, &v399);
            if ( !v114 )
            {
              if ( (_BYTE)v403 || !v405 )
                goto LABEL_4;
              if ( !(unsigned __int8)sub_1EDC030(a1, (__int64)v12, v390, v115, v116, v117)
                && !(unsigned __int8)sub_1EDEA30(a1, (__int64)v12, v390, v118, v119) )
              {
                if ( (_BYTE)v403 )
                  goto LABEL_4;
                if ( !v405 )
                  goto LABEL_4;
                if ( **(_WORD **)(v390 + 16) != 15 )
                  goto LABEL_4;
                v120 = *(_DWORD **)(v390 + 32);
                if ( (*v120 & 0xFFF00) != 0 )
                  goto LABEL_4;
                if ( (v120[10] & 0xFFF00) != 0 )
                  goto LABEL_4;
                v121 = *(_QWORD *)(v390 + 24);
                if ( *(_BYTE *)(v121 + 180)
                  || (unsigned int)((__int64)(*(_QWORD *)(v121 + 72) - *(_QWORD *)(v121 + 64)) >> 3) != 2 )
                {
                  goto LABEL_4;
                }
                v122 = v404;
                v123 = v401;
                v124 = HIDWORD(v401);
                v125 = *(_QWORD *)(a1 + 272);
                v126 = v401;
                if ( !v404 )
                  v126 = HIDWORD(v401);
                v127 = *(unsigned int *)(v125 + 408);
                v128 = v126 & 0x7FFFFFFF;
                v129 = v126 & 0x7FFFFFFF;
                v365 = 8 * v129;
                if ( (v126 & 0x7FFFFFFFu) < (unsigned int)v127 )
                {
                  v385 = *(_QWORD *)(*(_QWORD *)(v125 + 400) + 8LL * v128);
                  if ( v385 )
                  {
LABEL_171:
                    if ( !v122 )
                      v124 = v123;
                    v130 = v124 & 0x7FFFFFFF;
                    v131 = v124 & 0x7FFFFFFF;
                    v366 = 8 * v131;
                    if ( (v124 & 0x7FFFFFFFu) < (unsigned int)v127 )
                    {
                      v132 = *(_QWORD *)(*(_QWORD *)(v125 + 400) + 8LL * v130);
                      if ( v132 )
                      {
LABEL_175:
                        v133 = *(_QWORD *)(v125 + 272);
                        for ( jj = v390; (*(_BYTE *)(jj + 46) & 4) != 0; jj = *(_QWORD *)jj & 0xFFFFFFFFFFFFFFF8LL )
                          ;
                        v135 = *(_DWORD *)(v133 + 384);
                        v136 = *(_QWORD *)(v133 + 368);
                        if ( v135 )
                        {
                          v137 = (v135 - 1) & (((unsigned int)jj >> 9) ^ ((unsigned int)jj >> 4));
                          v138 = (__int64 *)(v136 + 16LL * v137);
                          v139 = *v138;
                          if ( *v138 == jj )
                            goto LABEL_179;
                          v320 = 1;
                          while ( v139 != -8 )
                          {
                            v321 = v320 + 1;
                            v322 = (v135 - 1) & (v137 + v320);
                            v137 = v322;
                            v138 = (__int64 *)(v136 + 16 * v322);
                            v139 = *v138;
                            if ( jj == *v138 )
                              goto LABEL_179;
                            v320 = v321;
                          }
                        }
                        v138 = (__int64 *)(v136 + 16LL * v135);
LABEL_179:
                        v354 = v138[1] & 0xFFFFFFFFFFFFFFF8LL;
                        v347 = v354 | 2;
                        v140 = (__int64 *)sub_1DB3C70((__int64 *)v385, v354 | 2);
                        if ( v140 == (__int64 *)(*(_QWORD *)v385 + 24LL * *(unsigned int *)(v385 + 8))
                          || (*(_DWORD *)((*v140 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v140 >> 1) & 3) > (*(_DWORD *)(v354 + 24) | 1u) )
                        {
                          BUG();
                        }
                        if ( (*(_BYTE *)(v140[2] + 8) & 6) != 0 )
                          goto LABEL_4;
                        v367 = sub_1DB4030(
                                 v132,
                                 *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL) + 392LL)
                                           + 16LL * *(unsigned int *)(v121 + 48)),
                                 v347);
                        if ( v367 )
                          goto LABEL_4;
                        v141 = *(__int64 **)(v121 + 72);
                        v142 = *(__int64 **)(v121 + 64);
                        v340 = v141;
                        if ( v141 == v142 )
                          goto LABEL_4;
                        v338 = v3;
                        v143 = *(_QWORD *)(a1 + 272);
                        v144 = (__int64 *)v385;
                        for ( kk = 0; ; kk = v152 )
                        {
                          v152 = *v142;
                          v153 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v143 + 272) + 392LL)
                                           + 16LL * *(unsigned int *)(*v142 + 48)
                                           + 8);
                          v154 = v153 & 0xFFFFFFFFFFFFFFF8LL;
                          v155 = (v153 >> 1) & 3;
                          if ( v155 )
                            v146 = (2LL * (v155 - 1)) | v154;
                          else
                            v146 = *(_QWORD *)v154 & 0xFFFFFFFFFFFFFFF8LL | 6;
                          v336 = v144;
                          v334 = v146;
                          v147 = (__int64 *)sub_1DB3C70(v144, v146);
                          v144 = v336;
                          v148 = v147;
                          if ( v147 == (__int64 *)(*v336 + 24LL * *((unsigned int *)v336 + 2))
                            || (*(_DWORD *)((*v147 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v147 >> 1) & 3)) > (*(_DWORD *)((v334 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v334 >> 1) & 3) )
                          {
                            BUG();
                          }
                          v143 = *(_QWORD *)(a1 + 272);
                          v149 = *(_QWORD *)(v147[2] + 8);
                          v150 = *(_QWORD *)(v148[2] + 8) & 0xFFFFFFFFFFFFFFF8LL;
                          if ( v150 )
                          {
                            v151 = *(_QWORD *)(v150 + 16);
                            if ( v151 )
                            {
                              if ( **(_WORD **)(v151 + 16) == 15 )
                              {
                                v307 = *(_DWORD **)(v151 + 32);
                                if ( (*v307 & 0xFFF00) == 0
                                  && (v307[10] & 0xFFF00) == 0
                                  && v307[2] == *((_DWORD *)v336 + 28)
                                  && v307[12] == *(_DWORD *)(v132 + 112)
                                  && v152 == *(_QWORD *)(v151 + 24) )
                                {
                                  v308 = *(_QWORD *)(v132 + 64);
                                  v309 = v308 + 8LL * *(unsigned int *)(v132 + 72);
                                  if ( v308 == v309 )
                                  {
LABEL_408:
                                    v152 = kk;
                                    v367 = v375;
                                  }
                                  else
                                  {
                                    v310 = (v149 >> 1) & 3;
                                    while ( 1 )
                                    {
                                      if ( (*(_QWORD *)(*(_QWORD *)v308 + 8LL) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                                      {
                                        v311 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)v308 + 8LL) & 0xFFFFFFFFFFFFFFF8LL)
                                                         + 24)
                                             | (*(__int64 *)(*(_QWORD *)v308 + 8LL) >> 1) & 3;
                                        if ( (v310 | *(_DWORD *)(v150 + 24)) < v311 )
                                        {
                                          v312 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v143 + 272) + 392LL)
                                                           + 16LL * *(unsigned int *)(v152 + 48)
                                                           + 8);
                                          if ( v311 < (*(_DWORD *)((v312 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                     | (unsigned int)(v312 >> 1) & 3) )
                                            break;
                                        }
                                      }
                                      v308 += 8;
                                      if ( v309 == v308 )
                                        goto LABEL_408;
                                    }
                                  }
                                }
                              }
                            }
                          }
                          if ( v340 == ++v142 )
                            break;
                        }
                        v3 = v338;
                        if ( !v367 )
                          goto LABEL_4;
                        if ( v152 )
                        {
                          if ( (unsigned int)((__int64)(*(_QWORD *)(v152 + 96) - *(_QWORD *)(v152 + 88)) >> 3) > 1 )
                            goto LABEL_4;
                          v275 = (_QWORD *)sub_1DD5EE0(v152);
                          if ( v275 != (_QWORD *)(v152 + 24) )
                          {
                            v378 = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL);
                            v276 = sub_1E85F30(v378, (unsigned __int64)v275);
                            if ( sub_1DB4030(
                                   v132,
                                   v276 & 0xFFFFFFFFFFFFFFF8LL | 2,
                                   *(_QWORD *)(*(_QWORD *)(v378 + 392) + 16LL * *(unsigned int *)(v152 + 48) + 8)) )
                            {
                              goto LABEL_4;
                            }
                          }
                          v372 = *(_QWORD *)(v152 + 56);
                          v344 = *(_DWORD *)(v132 + 112);
                          v379 = (unsigned __int64)sub_1E0B640(
                                                     v372,
                                                     *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL) + 960LL,
                                                     (__int64 *)(v390 + 64),
                                                     0);
                          sub_1DD5BA0((__int64 *)(v152 + 16), v379);
                          v277 = *(_QWORD *)v379;
                          v278 = *v275 & 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v379 + 8) = v275;
                          *(_QWORD *)v379 = v278 | v277 & 7;
                          *(_QWORD *)(v278 + 8) = v379;
                          *v275 = v379 | *v275 & 7LL;
                          v409.m128i_i32[2] = v344;
                          v409.m128i_i64[0] = 0x10000000;
                          v410 = 0;
                          v411 = 0;
                          v412 = 0;
                          sub_1E1A9C0(v379, v372, &v409);
                          v279 = *(_DWORD *)(v385 + 112);
                          v409.m128i_i64[0] = 0;
                          v410 = 0;
                          v409.m128i_i32[2] = v279;
                          v411 = 0;
                          v412 = 0;
                          sub_1E1A9C0(v379, v372, &v409);
                          v280 = sub_1DC1550(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 272LL), v379, 0) & 0xFFFFFFFFFFFFFFF8LL
                               | 4;
                          sub_1DB79D0((__int64 *)v132, v280, (__int64 *)(*(_QWORD *)(a1 + 272) + 296LL));
                          v281 = *(__int64 **)(v132 + 104);
                          v282 = v379;
                          if ( v281 )
                          {
                            v283 = v132;
                            v284 = v280;
                            v285 = v281;
                            do
                            {
                              v373 = v282;
                              sub_1DB79D0(v285, v284, (__int64 *)(*(_QWORD *)(a1 + 272) + 296LL));
                              v285 = (__int64 *)v285[13];
                              v282 = v373;
                            }
                            while ( v285 );
                            v132 = v283;
                          }
                          v286 = *(_QWORD **)(a1 + 576);
                          v287 = *(_QWORD **)(a1 + 568);
                          if ( v286 == v287 )
                          {
                            v318 = *(unsigned int *)(a1 + 588);
                            v319 = &v286[v318];
                            if ( v286 == v319 )
                            {
LABEL_428:
                              v287 = &v286[v318];
                            }
                            else
                            {
                              while ( v282 != *v287 )
                              {
                                if ( v319 == ++v287 )
                                  goto LABEL_428;
                              }
                            }
                            goto LABEL_422;
                          }
                          v380 = v282;
                          v287 = sub_16CC9F0(v393, v282);
                          if ( v380 == *v287 )
                          {
                            v288 = *(_QWORD **)(a1 + 576);
                            v286 = *(_QWORD **)(a1 + 568);
                            if ( v288 == v286 )
                            {
                              v318 = *(unsigned int *)(a1 + 588);
                              goto LABEL_422;
                            }
                            v289 = *(unsigned int *)(a1 + 584);
LABEL_371:
                            v290 = &v288[v289];
                          }
                          else
                          {
                            v286 = *(_QWORD **)(a1 + 576);
                            v288 = v286;
                            if ( v286 != *(_QWORD **)(a1 + 568) )
                            {
                              v289 = *(unsigned int *)(a1 + 584);
                              v287 = &v288[v289];
                              goto LABEL_371;
                            }
                            v318 = *(unsigned int *)(a1 + 588);
                            v287 = &v286[v318];
LABEL_422:
                            v290 = &v286[v318];
                          }
                          if ( v290 != v287 )
                          {
                            *v287 = -2;
                            ++*(_DWORD *)(a1 + 592);
                          }
                        }
                        sub_1ED8E30(a1, v390);
                        v406 = (__int64 *)v408;
                        v407 = 0x800000000LL;
                        sub_1E86030((__int64)&v409, v132, v347);
                        v381 = v409.m128i_i64[1];
                        sub_1DC0B50(*(_QWORD *)(a1 + 272), v132, v354 | 4, (__int64)&v406);
                        *(_QWORD *)(v381 + 8) = 0;
                        sub_1DBC0D0(*(_QWORD **)(a1 + 272), v132, v406, (unsigned int)v407, 0, 0);
                        if ( *(_QWORD *)(v132 + 104) )
                        {
                          v294 = v354 | 4;
                          v295 = *(_QWORD *)(v132 + 104);
                          do
                          {
                            LODWORD(v407) = 0;
                            v299 = (__int64 *)sub_1DB3C70((__int64 *)v295, v354);
                            v300 = *(_QWORD *)v295 + 24LL * *(unsigned int *)(v295 + 8);
                            if ( v299 == (__int64 *)v300 )
                            {
LABEL_469:
                              sub_1DC0B50(*(_QWORD *)(a1 + 272), v295, v294, (__int64)&v406);
                              MEMORY[8] = 0;
                              BUG();
                            }
                            v296 = *(_DWORD *)(v354 + 24);
                            v297 = *(_DWORD *)((*v299 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                            if ( (unsigned __int64)(v297 | (*v299 >> 1) & 3) <= v296
                              && v354 == (v299[1] & 0xFFFFFFFFFFFFFFF8LL) )
                            {
                              if ( (__int64 *)v300 == v299 + 3 )
                                goto LABEL_469;
                              v297 = *(_DWORD *)((v299[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
                              v299 += 3;
                            }
                            v298 = 0;
                            if ( v297 <= v296 )
                              v298 = v299[2];
                            v374 = v298;
                            sub_1DC0B50(*(_QWORD *)(a1 + 272), v295, v294, (__int64)&v406);
                            *(_QWORD *)(v374 + 8) = 0;
                            sub_1DBC0D0(*(_QWORD **)(a1 + 272), v295, v406, (unsigned int)v407, 0, 0);
                            v295 = *(_QWORD *)(v295 + 104);
                          }
                          while ( v295 );
                          v3 = v338;
                        }
                        v316 = *(_QWORD **)(a1 + 272);
                        if ( (unsigned __int8)sub_1DC0580(v316, v132, 0, v291, v292, v293) )
                        {
                          v409.m128i_i64[0] = (__int64)&v410;
                          v409.m128i_i64[1] = 0x800000000LL;
                          sub_1DBEB50((__int64)v316, v132, (__int64)&v409);
                          if ( (__int64 *)v409.m128i_i64[0] != &v410 )
                            _libc_free(v409.m128i_u64[0]);
                        }
                        v317 = *(_QWORD **)(a1 + 272);
                        if ( (unsigned __int8)sub_1DC0580(v317, v385, 0, v313, v314, v315) )
                        {
                          v409.m128i_i64[0] = (__int64)&v410;
                          v409.m128i_i64[1] = 0x800000000LL;
                          sub_1DBEB50((__int64)v317, v385, (__int64)&v409);
                          if ( (__int64 *)v409.m128i_i64[0] != &v410 )
                            _libc_free(v409.m128i_u64[0]);
                        }
                        if ( v406 != (__int64 *)v408 )
                          _libc_free((unsigned __int64)v406);
                        v370 = v367;
                        goto LABEL_3;
                      }
                    }
                    v302 = v130 + 1;
                    if ( v130 + 1 > (unsigned int)v127 )
                    {
                      if ( v302 >= v127 )
                      {
                        if ( v302 <= v127 )
                          goto LABEL_391;
                        if ( v302 > (unsigned __int64)*(unsigned int *)(v125 + 412) )
                        {
                          v346 = v124 & 0x7FFFFFFF;
                          v352 = v124;
                          v360 = v125;
                          sub_16CD150(v125 + 400, (const void *)(v125 + 416), v302, 8, v125, v124);
                          v131 = v346;
                          v124 = v352;
                          v125 = v360;
                        }
                        v303 = *(_QWORD *)(v125 + 400);
                        v327 = *(_QWORD *)(v125 + 416);
                        v328 = (_QWORD *)(v303 + 8LL * v302);
                        v329 = (_QWORD *)(v303 + 8LL * *(unsigned int *)(v125 + 408));
                        if ( v328 != v329 )
                        {
                          do
                            *v329++ = v327;
                          while ( v328 != v329 );
                          v303 = *(_QWORD *)(v125 + 400);
                        }
                        *(_DWORD *)(v125 + 408) = v302;
                      }
                      else
                      {
                        *(_DWORD *)(v125 + 408) = v302;
                        v303 = *(_QWORD *)(v125 + 400);
                      }
                    }
                    else
                    {
LABEL_391:
                      v303 = *(_QWORD *)(v125 + 400);
                    }
                    v351 = v131;
                    v358 = (_QWORD *)v125;
                    *(_QWORD *)(v303 + v366) = sub_1DBA290(v124);
                    v132 = *(_QWORD *)(v358[50] + 8 * v351);
                    sub_1DBB110(v358, v132);
                    v125 = *(_QWORD *)(a1 + 272);
                    goto LABEL_175;
                  }
                }
                v304 = v128 + 1;
                if ( (unsigned int)v127 < v128 + 1 )
                {
                  if ( v304 >= v127 )
                  {
                    if ( v304 <= v127 )
                      goto LABEL_394;
                    if ( v304 > (unsigned __int64)*(unsigned int *)(v125 + 412) )
                    {
                      v353 = v126 & 0x7FFFFFFF;
                      v361 = v126;
                      v389 = *(_QWORD *)(a1 + 272);
                      sub_16CD150(v125 + 400, (const void *)(v125 + 416), v304, 8, v125, SHIDWORD(v401));
                      v129 = v353;
                      v126 = v361;
                      v125 = v389;
                    }
                    v305 = *(_QWORD *)(v125 + 400);
                    v330 = *(_QWORD *)(v125 + 416);
                    v331 = (_QWORD *)(v305 + 8LL * v304);
                    v332 = (_QWORD *)(v305 + 8LL * *(unsigned int *)(v125 + 408));
                    if ( v331 != v332 )
                    {
                      do
                        *v332++ = v330;
                      while ( v331 != v332 );
                      v305 = *(_QWORD *)(v125 + 400);
                    }
                    *(_DWORD *)(v125 + 408) = v304;
                  }
                  else
                  {
                    *(_DWORD *)(v125 + 408) = v304;
                    v305 = *(_QWORD *)(v125 + 400);
                  }
                }
                else
                {
LABEL_394:
                  v305 = *(_QWORD *)(v125 + 400);
                }
                v359 = v129;
                v388 = (_QWORD *)v125;
                *(_QWORD *)(v305 + v365) = sub_1DBA290(v126);
                v306 = v388;
                v385 = *(_QWORD *)(v388[50] + 8 * v359);
                sub_1DBB110(v306, v385);
                v125 = *(_QWORD *)(a1 + 272);
                v122 = v404;
                v124 = HIDWORD(v401);
                v123 = v401;
                v127 = *(unsigned int *)(v125 + 408);
                goto LABEL_171;
              }
LABEL_202:
              sub_1ED8E30(a1, v390);
              v114 = v375;
            }
LABEL_203:
            v370 = v114;
            goto LABEL_3;
          }
        }
        v208 = v170 + 1;
        if ( (unsigned int)v169 >= v208 )
          goto LABEL_259;
        v266 = v208;
        if ( v208 < v169 )
        {
          *(_DWORD *)(v167 + 408) = v208;
          goto LABEL_259;
        }
        if ( v208 <= v169 )
        {
LABEL_259:
          v209 = *(_QWORD *)(v167 + 400);
        }
        else
        {
          if ( v208 > (unsigned __int64)*(unsigned int *)(v167 + 412) )
          {
            v345 = 8LL * (HIDWORD(v401) & 0x7FFFFFFF);
            sub_16CD150(v167 + 400, (const void *)(v167 + 416), v208, 8, 8 * HIDWORD(v401), v165);
            v169 = *(unsigned int *)(v167 + 408);
            v171 = v345;
            v266 = v208;
          }
          v209 = *(_QWORD *)(v167 + 400);
          v271 = *(_QWORD *)(v167 + 416);
          v272 = (_QWORD *)(v209 + 8 * v266);
          v273 = (_QWORD *)(v209 + 8 * v169);
          if ( v272 != v273 )
          {
            do
              *v273++ = v271;
            while ( v272 != v273 );
            v209 = *(_QWORD *)(v167 + 400);
          }
          *(_DWORD *)(v167 + 408) = v208;
        }
        *(_QWORD *)(v209 + v171) = sub_1DBA290(SHIDWORD(v168));
        v355 = *(__int64 ***)(*(_QWORD *)(v167 + 400) + 8 * v348);
        sub_1DBB110((_QWORD *)v167, (__int64)v355);
        goto LABEL_216;
      }
LABEL_3:
      *v3 = 0;
LABEL_4:
      if ( ++v3 == (__int64 *)v392 )
        return v370;
    }
  }
  return 0;
}
