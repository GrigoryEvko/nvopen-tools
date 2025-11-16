// Function: sub_2591C20
// Address: 0x2591c20
//
__int64 __fastcall sub_2591C20(__int64 a1, _QWORD *a2)
{
  __int16 v2; // ax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  int v8; // esi
  unsigned __int64 *v9; // rax
  int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rax
  _BYTE *v19; // rax
  __m128i v20; // xmm0
  __int64 v21; // rax
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // r14
  _BYTE *v24; // r13
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __int64 v27; // r13
  __m128i *v28; // r12
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  char v31; // al
  unsigned int v32; // r12d
  __int64 v33; // rdx
  _BYTE *v34; // rcx
  unsigned __int64 v35; // rax
  char v37; // dl
  __int64 v38; // r13
  int v39; // eax
  unsigned __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r14
  __int64 v43; // r13
  __int64 v44; // rbx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rax
  char v48; // al
  __int64 *v49; // rdi
  char v50; // r15
  __int64 *v51; // rdi
  unsigned __int8 v52; // r13
  __m128i v53; // rax
  __int64 v54; // rax
  unsigned __int8 *v55; // rax
  unsigned __int64 v56; // rdi
  __int16 v57; // bx
  unsigned __int64 v58; // rsi
  __m128i v59; // rax
  __int32 v60; // eax
  __m128i v61; // rax
  char v62; // al
  __int64 v63; // rcx
  __int64 v64; // r14
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // r15
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  char v73; // r14
  __int64 v74; // r10
  unsigned __int8 **v75; // r15
  __int64 v76; // rbx
  __int64 v77; // rax
  char v78; // r9
  unsigned __int64 v79; // rcx
  unsigned __int8 *v80; // rbx
  unsigned __int8 *v81; // r8
  _BYTE *v82; // rax
  _BYTE *v83; // r8
  __int64 v84; // rax
  char v85; // al
  unsigned __int8 v86; // al
  __int64 *v87; // rax
  __int64 *v88; // rdx
  unsigned int v89; // ebx
  __int64 *i; // rdx
  __int64 v91; // rbx
  unsigned __int64 *v92; // r14
  __int64 v93; // rbx
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // r15
  __m128i v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 v99; // rsi
  const __m128i *v100; // rcx
  unsigned __int64 v101; // rax
  unsigned __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rdx
  __int64 v106; // rcx
  unsigned __int8 v107; // r13
  __m128i v108; // rax
  unsigned __int64 v109; // rax
  unsigned __int64 v110; // r13
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __m128i v116; // rax
  unsigned __int8 *v117; // rax
  unsigned int v118; // ebx
  bool v119; // al
  __int64 v120; // rax
  __int64 v121; // rax
  unsigned int v122; // edi
  __int64 v123; // rsi
  unsigned int v124; // eax
  __int64 *v125; // rdx
  unsigned __int64 *v126; // r14
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rax
  unsigned __int64 *v130; // r13
  unsigned __int64 *v131; // rsi
  char v132; // al
  unsigned __int64 v133; // rsi
  __int64 v134; // rdi
  _BYTE *v135; // r8
  __int64 v136; // rdx
  unsigned __int64 *v137; // rbx
  __int64 v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rax
  __int64 v141; // r13
  __int64 v142; // rax
  unsigned __int64 *v143; // r13
  __int64 v144; // rsi
  char v145; // al
  unsigned __int64 v146; // r14
  __int64 v147; // rsi
  unsigned __int8 v148; // al
  char v149; // al
  unsigned __int64 v150; // r14
  __int64 v151; // rsi
  unsigned __int8 v152; // al
  char v153; // al
  unsigned __int64 v154; // r14
  __int64 v155; // rsi
  unsigned __int8 v156; // al
  char v157; // al
  unsigned __int64 v158; // r14
  unsigned __int8 v159; // al
  __m128i v160; // rax
  __int64 v161; // rdi
  __int64 v162; // rsi
  __int64 v163; // rbx
  __int64 v164; // rax
  char v165; // al
  __int64 v166; // rax
  __int64 v167; // rax
  __int64 v168; // rsi
  __int64 v169; // rbx
  __int64 v170; // r14
  __int64 v171; // rbx
  __int64 v172; // rax
  __int64 v173; // r9
  unsigned __int8 *v174; // rax
  __int64 v175; // rdx
  __int64 v176; // rcx
  __int64 *v177; // rbx
  __int64 *v178; // r15
  __int64 v179; // rdx
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // r8
  unsigned __int64 v183; // r15
  __int64 *v184; // r9
  int v185; // eax
  unsigned int v186; // ecx
  __int64 v187; // rdx
  __int64 v188; // rax
  __int8 *v189; // r14
  __int64 v190; // rbx
  unsigned __int64 v191; // rsi
  __int64 v192; // rax
  __int64 v193; // r15
  __int64 v194; // rbx
  __int64 v195; // rax
  __int64 v196; // r13
  __int64 v197; // rbx
  __int64 v198; // rdx
  __int64 v199; // rdi
  __int64 v200; // rcx
  __int64 v201; // rdx
  __int64 v202; // rax
  _QWORD *v203; // rsi
  __int64 v204; // rdi
  _QWORD *v205; // r9
  __int64 v206; // r10
  __int64 v207; // rdi
  _QWORD *v208; // rdi
  bool v209; // al
  unsigned __int64 v210; // rax
  int v211; // ecx
  __int64 v212; // rax
  __int64 v213; // rax
  unsigned __int64 v214; // rcx
  __m128i *v215; // rdx
  __m128i *v216; // rax
  __int64 v217; // rdx
  __int64 v218; // rax
  __int64 v219; // r11
  __int64 v220; // rsi
  unsigned int v221; // esi
  __int64 *v222; // rdi
  __int64 v223; // r9
  __int64 v224; // rax
  unsigned int v225; // esi
  unsigned int v226; // eax
  unsigned int v227; // edx
  unsigned int v228; // ecx
  unsigned int v229; // ecx
  __int64 v230; // rdx
  __int64 v231; // rbx
  __int64 v232; // rax
  unsigned __int64 v233; // rdx
  unsigned __int64 v234; // rcx
  __m128i *v235; // rdx
  __m128i *v236; // rax
  __int64 v237; // rdx
  __int64 v238; // rcx
  __int64 v239; // r8
  __int64 v240; // r9
  unsigned __int64 v241; // rax
  unsigned __int64 v242; // r13
  __int64 v243; // rdx
  __int64 v244; // rax
  int v245; // edx
  __int64 v246; // rax
  __int64 v247; // rdx
  __int64 v248; // rcx
  __int64 v249; // r8
  __int64 v250; // r9
  int v251; // edi
  __int64 v252; // rsi
  int j; // edi
  int v254; // r10d
  __int64 v255; // rdi
  unsigned __int64 v256; // r13
  __int64 v257; // [rsp-10h] [rbp-790h]
  __int64 v258; // [rsp-10h] [rbp-790h]
  unsigned __int64 v259; // [rsp-10h] [rbp-790h]
  __int64 v260; // [rsp-8h] [rbp-788h]
  _QWORD *v261; // [rsp+10h] [rbp-770h]
  unsigned __int8 **v262; // [rsp+18h] [rbp-768h]
  unsigned __int64 v263; // [rsp+30h] [rbp-750h]
  _BYTE *v264; // [rsp+50h] [rbp-730h]
  unsigned __int64 v265; // [rsp+58h] [rbp-728h]
  _BYTE *v266; // [rsp+58h] [rbp-728h]
  unsigned __int8 *v267; // [rsp+58h] [rbp-728h]
  unsigned __int64 v268; // [rsp+58h] [rbp-728h]
  unsigned int v269; // [rsp+60h] [rbp-720h]
  char v270; // [rsp+68h] [rbp-718h]
  char v271; // [rsp+70h] [rbp-710h]
  unsigned __int64 *v272; // [rsp+70h] [rbp-710h]
  unsigned __int64 *v273; // [rsp+70h] [rbp-710h]
  unsigned __int64 v274; // [rsp+78h] [rbp-708h]
  __int64 v275; // [rsp+78h] [rbp-708h]
  unsigned __int64 v276; // [rsp+80h] [rbp-700h]
  unsigned __int64 *v277; // [rsp+80h] [rbp-700h]
  __int64 v278; // [rsp+80h] [rbp-700h]
  bool v279; // [rsp+80h] [rbp-700h]
  _BYTE *v280; // [rsp+88h] [rbp-6F8h]
  char v281; // [rsp+90h] [rbp-6F0h]
  unsigned __int8 **v282; // [rsp+90h] [rbp-6F0h]
  unsigned __int8 *v283; // [rsp+98h] [rbp-6E8h]
  __int64 v284; // [rsp+98h] [rbp-6E8h]
  unsigned __int8 **v285; // [rsp+98h] [rbp-6E8h]
  __int64 v286; // [rsp+98h] [rbp-6E8h]
  __int64 *v287; // [rsp+98h] [rbp-6E8h]
  __int8 *v288; // [rsp+98h] [rbp-6E8h]
  _QWORD *v289; // [rsp+A8h] [rbp-6D8h]
  int v291; // [rsp+B8h] [rbp-6C8h]
  char v292; // [rsp+BFh] [rbp-6C1h]
  char v294; // [rsp+DEh] [rbp-6A2h] BYREF
  char v295; // [rsp+DFh] [rbp-6A1h] BYREF
  __m128i v296; // [rsp+E0h] [rbp-6A0h] BYREF
  unsigned __int64 v297; // [rsp+F0h] [rbp-690h]
  __int64 v298; // [rsp+F8h] [rbp-688h]
  unsigned __int8 *v299; // [rsp+100h] [rbp-680h]
  __int64 v300; // [rsp+108h] [rbp-678h]
  unsigned __int64 v301; // [rsp+110h] [rbp-670h]
  __int64 v302; // [rsp+118h] [rbp-668h]
  unsigned __int64 v303; // [rsp+120h] [rbp-660h]
  __int64 v304; // [rsp+128h] [rbp-658h]
  __m128i v305; // [rsp+130h] [rbp-650h] BYREF
  __int64 v306; // [rsp+140h] [rbp-640h]
  __m128i v307; // [rsp+150h] [rbp-630h] BYREF
  _BYTE *v308[6]; // [rsp+160h] [rbp-620h] BYREF
  __m128i v309; // [rsp+190h] [rbp-5F0h] BYREF
  __int64 v310; // [rsp+1A0h] [rbp-5E0h] BYREF
  __int64 v311; // [rsp+1A8h] [rbp-5D8h]
  unsigned __int64 *v312; // [rsp+1B0h] [rbp-5D0h]
  __int64 v313; // [rsp+1B8h] [rbp-5C8h]
  _QWORD v314[2]; // [rsp+1C0h] [rbp-5C0h] BYREF
  __int16 v315; // [rsp+1D0h] [rbp-5B0h]
  __m128i v316; // [rsp+1E0h] [rbp-5A0h] BYREF
  __int64 v317; // [rsp+1F0h] [rbp-590h] BYREF
  __int64 v318; // [rsp+1F8h] [rbp-588h]
  unsigned __int64 *v319; // [rsp+200h] [rbp-580h]
  __int64 v320; // [rsp+208h] [rbp-578h]
  _QWORD v321[2]; // [rsp+210h] [rbp-570h] BYREF
  __int16 v322; // [rsp+220h] [rbp-560h]
  __int64 v323; // [rsp+230h] [rbp-550h] BYREF
  __int64 v324; // [rsp+238h] [rbp-548h]
  __int64 *v325; // [rsp+240h] [rbp-540h] BYREF
  unsigned int v326; // [rsp+248h] [rbp-538h]
  __m128i *v327; // [rsp+280h] [rbp-500h] BYREF
  __int64 v328; // [rsp+288h] [rbp-4F8h]
  _BYTE v329[96]; // [rsp+290h] [rbp-4F0h] BYREF
  void *v330; // [rsp+2F0h] [rbp-490h]
  void *v331; // [rsp+2F8h] [rbp-488h]
  __int16 v332; // [rsp+300h] [rbp-480h]
  __int64 v333; // [rsp+308h] [rbp-478h]
  __int64 v334; // [rsp+310h] [rbp-470h]
  __int64 v335; // [rsp+318h] [rbp-468h]
  __int64 v336; // [rsp+320h] [rbp-460h]
  _BYTE *v337; // [rsp+328h] [rbp-458h] BYREF
  __int64 v338; // [rsp+330h] [rbp-450h]
  _BYTE v339[192]; // [rsp+338h] [rbp-448h] BYREF
  char v340; // [rsp+3F8h] [rbp-388h]
  unsigned __int64 v341; // [rsp+400h] [rbp-380h] BYREF
  __int64 v342; // [rsp+408h] [rbp-378h]
  _BYTE v343[384]; // [rsp+410h] [rbp-370h] BYREF
  _BYTE *v344; // [rsp+590h] [rbp-1F0h] BYREF
  __int64 v345; // [rsp+598h] [rbp-1E8h]
  _BYTE v346[384]; // [rsp+5A0h] [rbp-1E0h] BYREF
  char v347[8]; // [rsp+720h] [rbp-60h] BYREF
  int v348; // [rsp+728h] [rbp-58h] BYREF
  unsigned __int64 v349; // [rsp+730h] [rbp-50h]
  int *v350; // [rsp+738h] [rbp-48h]
  int *v351; // [rsp+740h] [rbp-40h]
  __int64 v352; // [rsp+748h] [rbp-38h]

  v333 = 0;
  v334 = 0;
  v330 = &unk_4A171B8;
  v2 = *(_WORD *)(a1 + 104);
  v335 = 0;
  v332 = v2;
  v336 = 0;
  v331 = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v7 = *(unsigned int *)(a1 + 136);
  LODWORD(v336) = v7;
  if ( (_DWORD)v7 )
  {
    v166 = sub_C7D670(24 * v7, 8);
    v4 = *(_QWORD *)(a1 + 120);
    v334 = v166;
    v3 = v166;
    v335 = *(_QWORD *)(a1 + 128);
    v167 = 0;
    v168 = 24LL * (unsigned int)v336;
    do
    {
      *(__m128i *)(v3 + v167) = _mm_loadu_si128((const __m128i *)(v4 + v167));
      *(_QWORD *)(v3 + v167 + 16) = *(_QWORD *)(v4 + v167 + 16);
      v167 += 24;
    }
    while ( v168 != v167 );
  }
  else
  {
    v334 = 0;
    v335 = 0;
  }
  v8 = *(_DWORD *)(a1 + 152);
  v337 = v339;
  v338 = 0x800000000LL;
  if ( v8 )
    sub_2539BB0((__int64)&v337, a1 + 144, v3, v4, v5, v6);
  v340 = *(_BYTE *)(a1 + 352);
  v289 = (_QWORD *)(a1 + 72);
  v323 = 0;
  v280 = (_BYTE *)sub_250D070((_QWORD *)(a1 + 72));
  v9 = (unsigned __int64 *)&v325;
  v324 = 1;
  do
  {
    *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != (unsigned __int64 *)&v327 );
  v348 = 0;
  v327 = (__m128i *)v329;
  v328 = 0x400000000LL;
  v341 = (unsigned __int64)v343;
  v344 = v346;
  v350 = &v348;
  v351 = &v348;
  v345 = 0x1000000000LL;
  v349 = 0;
  v352 = 0;
  v342 = 0x1000000000LL;
  LOBYTE(v317) = 3;
  v316.m128i_i64[1] = sub_2509740(v289);
  v316.m128i_i64[0] = (__int64)v280;
  v10 = 0;
  sub_25379E0((__int64)&v341, &v316, v11, v12, v13, v14);
  LODWORD(v17) = v342;
  while ( 2 )
  {
    v18 = (unsigned int)v17;
    v17 = (unsigned int)(v17 - 1);
    v19 = (_BYTE *)(v341 + 24 * v18);
    v20 = _mm_loadu_si128((const __m128i *)(v19 - 24));
    LODWORD(v342) = v17;
    v305 = v20;
    v21 = *((_QWORD *)v19 - 1);
    v22 = v20.m128i_i64[0];
    v292 = v21;
    v306 = v21;
    if ( v352 )
    {
      sub_253CDA0((__int64)v347, &v305);
      if ( !v37 )
      {
        LODWORD(v17) = v342;
        goto LABEL_15;
      }
    }
    else
    {
      v23 = (unsigned __int64)v344;
      v24 = &v344[24 * (unsigned int)v345];
      if ( v344 != v24 )
      {
        v25 = (unsigned __int64)v344;
        while ( *(_OWORD *)&v20 != *(_OWORD *)v25 || v292 != *(_BYTE *)(v25 + 16) )
        {
          v25 += 24LL;
          if ( v24 == (_BYTE *)v25 )
            goto LABEL_99;
        }
        if ( v24 != (_BYTE *)v25 )
          goto LABEL_15;
LABEL_99:
        if ( (unsigned int)v345 > 0xFuLL )
        {
          while ( 2 )
          {
            while ( 2 )
            {
              while ( 2 )
              {
                v99 = sub_253B830((__int64)v347, (unsigned __int64 *)v23);
                if ( !v98 )
                {
                  v23 += 24LL;
                  if ( v24 == (_BYTE *)v23 )
                    goto LABEL_135;
                  goto LABEL_132;
                }
LABEL_131:
                v100 = (const __m128i *)v23;
                v23 += 24LL;
                sub_253CCC0((__int64)v347, v99, v98, v100);
                if ( v24 == (_BYTE *)v23 )
                  goto LABEL_135;
LABEL_132:
                if ( !v352 )
                  continue;
                break;
              }
              v98 = (__int64)v351;
              if ( *((_QWORD *)v351 + 4) == *(_QWORD *)v23 )
              {
                v101 = *((_QWORD *)v351 + 5);
                v102 = *(_QWORD *)(v23 + 8);
                if ( v101 == v102 )
                {
                  if ( *((_BYTE *)v351 + 48) >= *(_BYTE *)(v23 + 16) )
                    continue;
                  goto LABEL_130;
                }
              }
              else
              {
                if ( *((_QWORD *)v351 + 4) < *(_QWORD *)v23 )
                {
LABEL_130:
                  v99 = 0;
                  goto LABEL_131;
                }
                if ( *((_QWORD *)v351 + 4) != *(_QWORD *)v23 )
                  continue;
                v101 = *((_QWORD *)v351 + 5);
                v102 = *(_QWORD *)(v23 + 8);
              }
              break;
            }
            if ( v102 <= v101 )
              continue;
            goto LABEL_130;
          }
        }
        goto LABEL_100;
      }
      if ( (unsigned int)v345 <= 0xFuLL )
      {
LABEL_100:
        sub_25379E0((__int64)&v344, &v305, v17, 0, v15, v16);
        goto LABEL_41;
      }
LABEL_135:
      LODWORD(v345) = 0;
      sub_253CDA0((__int64)v347, &v305);
    }
LABEL_41:
    v291 = v10 + 1;
    if ( v10 >= (int)qword_4FEF748 )
      goto LABEL_67;
    v38 = *(_QWORD *)(v20.m128i_i64[0] + 8);
    if ( *(_BYTE *)(v38 + 8) == 14 )
    {
      v55 = sub_BD3990((unsigned __int8 *)v20.m128i_i64[0], (__int64)&v305);
      v47 = sub_250C3F0((unsigned __int64)v55, v38);
      if ( v20.m128i_i64[0] != v47 && v47 )
        goto LABEL_56;
    }
    else
    {
      v39 = *(unsigned __int8 *)v20.m128i_i64[0];
      if ( (unsigned __int8)v39 <= 0x1Cu )
        goto LABEL_65;
      v40 = (unsigned int)(v39 - 34);
      if ( (unsigned __int8)(v39 - 34) > 0x33u )
        goto LABEL_71;
      v41 = 0x8000000000041LL;
      if ( !_bittest64(&v41, v40) )
        goto LABEL_71;
      v42 = *(_QWORD *)(v20.m128i_i64[0] - 32);
      if ( !v42 || *(_BYTE *)v42 )
        goto LABEL_71;
      if ( (*(_BYTE *)(v42 + 2) & 1) != 0 )
      {
        sub_B2C6D0(*(_QWORD *)(v20.m128i_i64[0] - 32), (__int64)&v305, v40, 0x8000000000041LL);
        v43 = *(_QWORD *)(v42 + 96);
        v44 = v43 + 40LL * *(_QWORD *)(v42 + 104);
        if ( (*(_BYTE *)(v42 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v42, (__int64)&v305, v105, v106);
          v43 = *(_QWORD *)(v42 + 96);
        }
      }
      else
      {
        v43 = *(_QWORD *)(v42 + 96);
        v44 = v43 + 40LL * *(_QWORD *)(v42 + 104);
      }
      if ( v43 != v44 )
      {
        while ( !(unsigned __int8)sub_B2D750(v43) )
        {
          v43 += 40;
          if ( v44 == v43 )
            goto LABEL_70;
        }
        v46 = *(_DWORD *)(v20.m128i_i64[0] + 4) & 0x7FFFFFF;
        v47 = *(_QWORD *)(v20.m128i_i64[0] + 32 * (*(unsigned int *)(v43 + 32) - v46));
        if ( v47 )
        {
          if ( v20.m128i_i64[0] != v47 )
          {
LABEL_56:
            v316.m128i_i64[0] = v47;
            v316.m128i_i64[1] = v20.m128i_i64[1];
            v48 = v292;
            goto LABEL_57;
          }
        }
      }
    }
LABEL_70:
    v39 = *(unsigned __int8 *)v20.m128i_i64[0];
    if ( (unsigned __int8)v39 <= 0x1Cu )
      goto LABEL_65;
LABEL_71:
    v283 = (unsigned __int8 *)v305.m128i_i64[1];
    v281 = v306;
    if ( (unsigned __int8)(v39 - 82) <= 1u )
    {
      v56 = *(_QWORD *)(v20.m128i_i64[0] - 64);
      v57 = *(_WORD *)(v20.m128i_i64[0] + 2);
      v276 = *(_QWORD *)(v20.m128i_i64[0] - 32);
      v309.m128i_i64[0] = (__int64)&v310;
      v58 = *(_QWORD *)(a1 + 80);
      v307.m128i_i64[0] = (__int64)v308;
      v274 = v56;
      v294 = 0;
      v307.m128i_i64[1] = 0x300000000LL;
      v309.m128i_i64[1] = 0x300000000LL;
      v59.m128i_i64[0] = sub_250D2C0(v56, v58);
      v316 = v59;
      if ( (unsigned __int8)sub_2526B50((__int64)a2, &v316, a1, (__int64)&v307, 1u, &v294, 1u) )
      {
        v60 = v307.m128i_i32[2];
      }
      else
      {
        v103 = 0;
        v307.m128i_i32[2] = 0;
        if ( !v307.m128i_i32[3] )
        {
          sub_C8D5F0((__int64)&v307, v308, 1u, 0x10u, v15, v16);
          v103 = 16LL * v307.m128i_u32[2];
        }
        v104 = v307.m128i_i64[0];
        *(_QWORD *)(v307.m128i_i64[0] + v103) = v56;
        *(_QWORD *)(v104 + v103 + 8) = v283;
        v60 = ++v307.m128i_i32[2];
      }
      if ( !v60 )
        goto LABEL_59;
      v61.m128i_i64[0] = sub_250D2C0(v276, *(_QWORD *)(a1 + 80));
      v316 = v61;
      v62 = sub_2526B50((__int64)a2, &v316, a1, (__int64)&v309, 1u, &v294, 1u);
      v15 = v257;
      v16 = v260;
      if ( !v62 )
      {
        v309.m128i_i32[2] = 0;
        sub_25592F0((__int64)&v309, v276, (__int64)v283, v63, v257, v260);
      }
      if ( v309.m128i_i32[2] )
      {
        v261 = (_QWORD *)sub_BD5C60(v56);
        v64 = a2[26];
        if ( *(_BYTE *)v20.m128i_i64[0] <= 0x1Cu )
        {
          v70 = 0;
          v67 = 0;
          v69 = 0;
          v66 = 0;
        }
        else
        {
          v65 = sub_B43CB0(v20.m128i_i64[0]);
          v66 = v65;
          if ( v65 )
          {
            v67 = sub_2554D30(*(_QWORD *)(v64 + 240), v65, 0);
            v284 = sub_2555710(*(_QWORD *)(a2[26] + 240LL), v66, 0);
            v68 = sub_255E580(*(_QWORD *)(v64 + 240), v66, 0);
            v69 = v284;
            v70 = v20.m128i_i64[0];
            v66 = v68;
          }
          else
          {
            v70 = v20.m128i_i64[0];
            v67 = 0;
            v69 = 0;
          }
          v64 = a2[26];
        }
        v71 = *(_QWORD *)(v64 + 104);
        v320 = v70;
        v322 = 257;
        v316.m128i_i64[0] = v71;
        v72 = 16LL * v307.m128i_u32[2];
        v316.m128i_i64[1] = v69;
        v318 = v67;
        v262 = (unsigned __int8 **)(v307.m128i_i64[0] + v72);
        v49 = (__int64 *)v309.m128i_i64[0];
        v317 = 0;
        v319 = (unsigned __int64 *)v66;
        v321[0] = 0;
        v321[1] = 0;
        if ( v307.m128i_i64[0] != v307.m128i_i64[0] + v72 )
        {
          v285 = (unsigned __int8 **)v307.m128i_i64[0];
          v73 = v281;
          v74 = v309.m128i_i64[0];
          v269 = v57 & 0x3F;
          while ( 1 )
          {
            v282 = (unsigned __int8 **)(v74 + 16LL * v309.m128i_u32[2]);
            if ( (unsigned __int8 **)v74 == v282 )
              goto LABEL_231;
            v75 = (unsigned __int8 **)v74;
            do
            {
              v80 = *v285;
              if ( (unsigned int)**v285 - 12 <= 1 || (v81 = *v75, (unsigned int)**v75 - 12 <= 1) )
              {
                v76 = sub_25096F0(v289);
                v77 = sub_ACA8A0(*(__int64 ***)(v20.m128i_i64[0] + 8));
                v78 = v73;
                v258 = v76;
                v79 = v77;
                goto LABEL_87;
              }
              if ( v81 == v80 )
              {
                v267 = *v75;
                if ( (unsigned __int8)sub_B535D0(v269) || (v165 = sub_B53600(v269), v81 = v267, v165) )
                {
                  v162 = (unsigned __int8)sub_B535D0(v269);
                  v161 = sub_BCB2A0(v261);
LABEL_229:
                  v163 = sub_ACD640(v161, v162, 0);
                  v164 = sub_25096F0(v289);
                  v78 = v73;
                  v258 = v164;
                  v79 = v163;
                  goto LABEL_87;
                }
              }
              v265 = (unsigned __int64)v81;
              v264 = (_BYTE *)sub_250C3F0((unsigned __int64)v80, *(_QWORD *)(v274 + 8));
              v82 = (_BYTE *)sub_250C3F0(v265, *(_QWORD *)(v276 + 8));
              v83 = (_BYTE *)v265;
              if ( !v264
                || !v82
                || (v263 = v269 | v263 & 0xFFFFFF0000000000LL,
                    v84 = sub_10197D0(v263, v264, v82, &v316),
                    v83 = (_BYTE *)v265,
                    !v84)
                || v20.m128i_i64[0] == v84 )
              {
                v266 = v83;
                v85 = sub_B52830(v269);
                if ( !v85 )
                  goto LABEL_224;
                v15 = (unsigned __int64)v266;
                v86 = *v80;
                if ( *v266 != 20 && v86 != 20 )
                {
                  v49 = (__int64 *)v309.m128i_i64[0];
                  v50 = 0;
                  goto LABEL_60;
                }
                if ( v86 != 20 )
                  v15 = (unsigned __int64)v80;
                v160.m128i_i64[0] = sub_250D2C0(v15, 0);
                v296 = v160;
                v85 = sub_258F340(a2, a1, &v296, 0, &v295, 0, 0);
                if ( !v85 )
                {
LABEL_224:
                  v49 = (__int64 *)v309.m128i_i64[0];
                  v50 = v85;
                  goto LABEL_60;
                }
                v161 = sub_BCB2A0(v261);
                v162 = v269 == 33;
                goto LABEL_229;
              }
              v268 = v84;
              v224 = sub_25096F0(v289);
              v78 = v73;
              v258 = v224;
              v79 = v268;
LABEL_87:
              v75 += 2;
              sub_258BA20(a1, (__int64)a2, (_BYTE *)(a1 + 88), v79, 0, v78, v258);
            }
            while ( v282 != v75 );
            v74 = v309.m128i_i64[0];
LABEL_231:
            v285 += 2;
            if ( v262 == v285 )
            {
              v49 = (__int64 *)v74;
              break;
            }
          }
        }
        v50 = 1;
      }
      else
      {
LABEL_59:
        v49 = (__int64 *)v309.m128i_i64[0];
        v50 = 1;
      }
LABEL_60:
      if ( v49 != &v310 )
        _libc_free((unsigned __int64)v49);
      v51 = (__int64 *)v307.m128i_i64[0];
      if ( (_BYTE **)v307.m128i_i64[0] != v308 )
        goto LABEL_63;
      goto LABEL_64;
    }
    if ( v39 != 84 )
    {
      if ( v39 == 86 )
      {
        v309.m128i_i8[0] = 0;
        v116.m128i_i64[0] = sub_250D2C0(*(_QWORD *)(v20.m128i_i64[0] - 96), 0);
        v316 = v116;
        v117 = sub_2527570((__int64)a2, &v316, a1, &v309);
        v300 = v46;
        v299 = v117;
        if ( !(_BYTE)v46 )
          goto LABEL_58;
        if ( !v299 )
          goto LABEL_329;
        v45 = *v299;
        if ( (_BYTE)v45 == 12 || (_DWORD)v45 == 13 )
          goto LABEL_58;
        if ( (_BYTE)v45 != 17 )
        {
LABEL_329:
          if ( v20.m128i_i64[0] == sub_250D070(v289) )
          {
            v316.m128i_i64[0] = *(_QWORD *)(v20.m128i_i64[0] - 64);
            v316.m128i_i64[1] = (__int64)v283;
            LOBYTE(v317) = v281;
            sub_25379E0((__int64)&v341, &v316, v237, v238, v239, v240);
            v246 = *(_QWORD *)(v20.m128i_i64[0] - 32);
            v316.m128i_i64[1] = (__int64)v283;
            v316.m128i_i64[0] = v246;
            LOBYTE(v317) = v281;
            sub_25379E0((__int64)&v341, &v316, v247, v248, v249, v250);
            LODWORD(v17) = v342;
            v10 = v291;
          }
          else
          {
            sub_250D230((unsigned __int64 *)&v316, v20.m128i_u64[0], 1, 0);
            v241 = sub_2527850((__int64)a2, &v316, a1, &v309, v281);
            v297 = v241;
            v242 = v241;
            v298 = v243;
            if ( !(_BYTE)v243 )
              goto LABEL_58;
            if ( !v241 )
              goto LABEL_65;
            v244 = sub_25096F0(v289);
            sub_258BA20(a1, (__int64)a2, (_BYTE *)(a1 + 88), v242, v283, v281, v244);
            LODWORD(v17) = v342;
            v10 = v291;
          }
          goto LABEL_15;
        }
        v118 = *((_DWORD *)v299 + 8);
        if ( v118 <= 0x40 )
          v119 = *((_QWORD *)v299 + 3) == 0;
        else
          v119 = v118 == (unsigned int)sub_C444A0((__int64)(v299 + 24));
        if ( v119 )
          v120 = *(_QWORD *)(v20.m128i_i64[0] - 32);
        else
          v120 = *(_QWORD *)(v20.m128i_i64[0] - 64);
        v316.m128i_i64[0] = v120;
        v316.m128i_i64[1] = (__int64)v283;
        v48 = v281;
LABEL_57:
        LOBYTE(v317) = v48;
        sub_25379E0((__int64)&v341, &v316, v46, v45, v15, v16);
        goto LABEL_58;
      }
      if ( v39 != 61 )
      {
        v296.m128i_i8[0] = 0;
        v87 = &v317;
        v88 = &v317;
        v89 = *(_DWORD *)(v20.m128i_i64[0] + 4) & 0x7FFFFFF;
        v316.m128i_i64[0] = (__int64)&v317;
        v316.m128i_i64[1] = 0x800000000LL;
        if ( v89 )
        {
          if ( v89 > 8uLL )
          {
            sub_C8D5F0((__int64)&v316, &v317, v89, 8u, v15, v16);
            v88 = (__int64 *)v316.m128i_i64[0];
            v87 = (__int64 *)(v316.m128i_i64[0] + 8LL * v316.m128i_u32[2]);
          }
          for ( i = &v88[v89]; i != v87; ++v87 )
          {
            if ( v87 )
              *v87 = 0;
          }
          v316.m128i_i32[2] = v89;
        }
        v91 = 32LL * (*(_DWORD *)(v20.m128i_i64[0] + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v20.m128i_i64[0] + 7) & 0x40) != 0 )
        {
          v16 = *(_QWORD *)(v20.m128i_i64[0] - 8);
          v277 = (unsigned __int64 *)(v16 + v91);
        }
        else
        {
          v277 = (unsigned __int64 *)v20.m128i_i64[0];
          v16 = v20.m128i_i64[0] - v91;
        }
        if ( v277 == (unsigned __int64 *)v16 )
          goto LABEL_339;
        v92 = (unsigned __int64 *)v16;
        v93 = 0;
        v271 = 0;
        do
        {
          v95 = *v92;
          v96.m128i_i64[0] = sub_250D2C0(*v92, *(_QWORD *)(a1 + 80));
          v309 = v96;
          v94 = sub_2527850((__int64)a2, &v309, a1, &v296, 1u);
          v304 = v97;
          v303 = v94;
          if ( !(_BYTE)v97 )
          {
            v22 = v20.m128i_i64[0];
            v51 = (__int64 *)v316.m128i_i64[0];
            v50 = 1;
            goto LABEL_125;
          }
          if ( !v94 )
            v94 = v95;
          *(_QWORD *)(v93 + v316.m128i_i64[0]) = v94;
          v51 = (__int64 *)v316.m128i_i64[0];
          v92 += 4;
          v271 |= *(_QWORD *)(v316.m128i_i64[0] + v93) != v95;
          v93 += 8;
        }
        while ( v277 != v92 );
        v50 = v271;
        v22 = v20.m128i_i64[0];
        if ( v271 )
        {
          v169 = a2[26];
          v170 = sub_B43CB0(v20.m128i_i64[0]);
          v275 = sub_2554D30(*(_QWORD *)(v169 + 240), v170, 0);
          v278 = sub_2555710(*(_QWORD *)(a2[26] + 240LL), v170, 0);
          v171 = sub_255E580(*(_QWORD *)(v169 + 240), v170, 0);
          v172 = sub_B43CC0(v20.m128i_i64[0]);
          v315 = 257;
          v311 = v275;
          v309.m128i_i64[0] = v172;
          v309.m128i_i64[1] = v278;
          v310 = 0;
          v312 = (unsigned __int64 *)v171;
          v313 = v20.m128i_i64[0];
          v314[0] = 0;
          v314[1] = 0;
          v174 = sub_1020E00(
                   (unsigned __int8 *)v20.m128i_i64[0],
                   (unsigned __int8 **)v316.m128i_i64[0],
                   v316.m128i_u32[2],
                   &v309,
                   v278,
                   v173);
          if ( (unsigned __int8 *)v20.m128i_i64[0] != v174 && v174 )
          {
            v307.m128i_i64[0] = (__int64)v174;
            v307.m128i_i64[1] = (__int64)v283;
            LOBYTE(v308[0]) = v281;
            sub_25379E0((__int64)&v341, &v307, v175, v176, v15, v16);
            v51 = (__int64 *)v316.m128i_i64[0];
            goto LABEL_125;
          }
LABEL_339:
          v51 = (__int64 *)v316.m128i_i64[0];
          v50 = 0;
        }
LABEL_125:
        if ( v51 != &v317 )
LABEL_63:
          _libc_free((unsigned __int64)v51);
LABEL_64:
        if ( !v50 )
          goto LABEL_65;
        goto LABEL_58;
      }
      v319 = v321;
      v312 = v314;
      v309 = 0u;
      v310 = 0;
      v311 = 0;
      v313 = 0x400000000LL;
      v316 = 0u;
      v317 = 0;
      v318 = 0;
      v320 = 0x400000000LL;
      v295 = 0;
      v50 = sub_2531360((__int64)a2, v20.m128i_i64[0], (__int64)&v309, (__int64)&v316, a1, &v295, 1);
      if ( !v50 )
      {
LABEL_155:
        v50 = 0;
LABEL_156:
        if ( v319 != v321 )
          _libc_free((unsigned __int64)v319);
        sub_C7D6A0(v316.m128i_i64[1], 8LL * (unsigned int)v318, 8);
        if ( v312 != v314 )
          _libc_free((unsigned __int64)v312);
        sub_C7D6A0(v309.m128i_i64[1], 8LL * (unsigned int)v311, 8);
        goto LABEL_64;
      }
      v121 = a2[26];
      v122 = *(_DWORD *)(v121 + 184);
      v123 = *(_QWORD *)(v121 + 168);
      if ( !v122 )
        goto LABEL_191;
      v114 = v122 - 1;
      v124 = v114 & (((unsigned __int32)v20.m128i_i32[0] >> 9) ^ ((unsigned __int32)v20.m128i_i32[0] >> 4));
      v125 = (__int64 *)(v123 + 8LL * v124);
      v113 = *v125;
      if ( v20.m128i_i64[0] != *v125 )
      {
        v245 = 1;
        while ( v113 != -4096 )
        {
          v115 = (unsigned int)(v245 + 1);
          v124 = v114 & (v245 + v124);
          v125 = (__int64 *)(v123 + 8LL * v124);
          v113 = *v125;
          if ( v20.m128i_i64[0] == *v125 )
            goto LABEL_173;
          v245 = v115;
        }
        goto LABEL_191;
      }
LABEL_173:
      if ( v125 == (__int64 *)(v123 + 8LL * v122) )
        goto LABEL_191;
      v126 = v319;
      v308[0] = &v295;
      v127 = 8LL * (unsigned int)v320;
      v272 = &v319[(unsigned __int64)v127 / 8];
      v128 = v127 >> 3;
      v129 = v127 >> 5;
      v307.m128i_i64[0] = (__int64)a2;
      v307.m128i_i64[1] = a1;
      if ( v129 )
      {
        v130 = &v319[4 * v129];
        while ( 1 )
        {
          v133 = *v126;
          if ( !*v126 )
            goto LABEL_179;
          if ( *(_BYTE *)v133 == 85 )
            break;
          v134 = v307.m128i_i64[0];
          v135 = v308[0];
          v136 = v307.m128i_i64[1];
          if ( *(_BYTE *)v133 != 62 )
            goto LABEL_310;
          if ( (*(_BYTE *)(v133 + 7) & 0x40) != 0 )
            v131 = *(unsigned __int64 **)(v133 - 8);
          else
            v131 = (unsigned __int64 *)(v133 - 32LL * (*(_DWORD *)(v133 + 4) & 0x7FFFFFF));
          v132 = sub_2522C50(v307.m128i_i64[0], v131, v307.m128i_i64[1], 0, v308[0], 0, 1);
          v113 = v260;
LABEL_178:
          if ( !v132 )
            goto LABEL_190;
LABEL_179:
          if ( !(unsigned __int8)sub_253BBA0(v307.m128i_i64, v126[1]) )
          {
            ++v126;
            goto LABEL_190;
          }
          if ( !(unsigned __int8)sub_253BBA0(v307.m128i_i64, v126[2]) )
          {
            v126 += 2;
            goto LABEL_190;
          }
          if ( !(unsigned __int8)sub_253BBA0(v307.m128i_i64, v126[3]) )
          {
            v126 += 3;
            goto LABEL_190;
          }
          v126 += 4;
          if ( v130 == v126 )
          {
            v128 = v272 - v126;
            goto LABEL_366;
          }
        }
        v218 = *(_QWORD *)(v133 - 32);
        if ( v218
          && !*(_BYTE *)v218
          && *(_QWORD *)(v218 + 24) == *(_QWORD *)(v133 + 80)
          && (*(_BYTE *)(v218 + 33) & 0x20) != 0
          && *(_DWORD *)(v218 + 36) == 11 )
        {
          goto LABEL_179;
        }
        v134 = v307.m128i_i64[0];
        v135 = v308[0];
        v136 = v307.m128i_i64[1];
LABEL_310:
        v132 = sub_251BFD0(v134, v133, v136, 0, v135, 0, 1, 0);
        goto LABEL_178;
      }
LABEL_366:
      if ( v128 != 2 )
      {
        if ( v128 != 3 )
        {
          if ( v128 == 1 )
            goto LABEL_369;
          goto LABEL_191;
        }
        if ( (unsigned __int8)sub_253BBA0(v307.m128i_i64, *v126) )
        {
          ++v126;
          goto LABEL_388;
        }
LABEL_190:
        if ( v272 != v126 )
          goto LABEL_155;
        goto LABEL_191;
      }
LABEL_388:
      if ( !(unsigned __int8)sub_253BBA0(v307.m128i_i64, *v126) )
        goto LABEL_190;
      ++v126;
LABEL_369:
      if ( !(unsigned __int8)sub_253BBA0(v307.m128i_i64, *v126) )
        goto LABEL_190;
LABEL_191:
      v137 = v312;
      v296.m128i_i8[0] = v281 & 1;
      v308[0] = a2;
      v138 = 8LL * (unsigned int)v313;
      v273 = &v312[(unsigned __int64)v138 / 8];
      v139 = v138 >> 3;
      v140 = v138 >> 5;
      v307.m128i_i64[0] = (__int64)&v296;
      v307.m128i_i64[1] = a1;
      if ( v140 )
      {
        v141 = 4 * v140;
        v142 = a1;
        v143 = &v312[v141];
        while ( 1 )
        {
          v158 = *v137;
          v144 = *(_QWORD *)(v142 + 72) & 0xFFFFFFFFFFFFFFFCLL;
          if ( (*(_QWORD *)(v142 + 72) & 3LL) == 3 )
            v144 = *(_QWORD *)(v144 + 24);
          v159 = *(_BYTE *)v144;
          if ( *(_BYTE *)v144 )
          {
            if ( v159 == 22 )
            {
              v144 = *(_QWORD *)(v144 + 24);
            }
            else if ( v159 <= 0x1Cu )
            {
              v144 = 0;
            }
            else
            {
              v144 = sub_B43CB0(v144);
            }
          }
          v145 = sub_250C180(v158, v144);
          *(_BYTE *)v307.m128i_i64[0] &= v145;
          if ( !(unsigned __int8)sub_252BB70((__int64)v308[0], v307.m128i_i64[1], v158, 1) )
            goto LABEL_253;
          v146 = v137[1];
          v147 = *(_QWORD *)(v307.m128i_i64[1] + 72) & 0xFFFFFFFFFFFFFFFCLL;
          if ( (*(_QWORD *)(v307.m128i_i64[1] + 72) & 3LL) == 3 )
            v147 = *(_QWORD *)(v147 + 24);
          v148 = *(_BYTE *)v147;
          if ( *(_BYTE *)v147 )
          {
            if ( v148 == 22 )
            {
              v147 = *(_QWORD *)(v147 + 24);
            }
            else if ( v148 <= 0x1Cu )
            {
              v147 = 0;
            }
            else
            {
              v147 = sub_B43CB0(v147);
            }
          }
          v149 = sub_250C180(v146, v147);
          *(_BYTE *)v307.m128i_i64[0] &= v149;
          if ( !(unsigned __int8)sub_252BB70((__int64)v308[0], v307.m128i_i64[1], v146, 1) )
          {
            ++v137;
            goto LABEL_253;
          }
          v150 = v137[2];
          v151 = *(_QWORD *)(v307.m128i_i64[1] + 72) & 0xFFFFFFFFFFFFFFFCLL;
          if ( (*(_QWORD *)(v307.m128i_i64[1] + 72) & 3LL) == 3 )
            v151 = *(_QWORD *)(v151 + 24);
          v152 = *(_BYTE *)v151;
          if ( *(_BYTE *)v151 )
          {
            if ( v152 == 22 )
            {
              v151 = *(_QWORD *)(v151 + 24);
            }
            else if ( v152 <= 0x1Cu )
            {
              v151 = 0;
            }
            else
            {
              v151 = sub_B43CB0(v151);
            }
          }
          v153 = sub_250C180(v150, v151);
          *(_BYTE *)v307.m128i_i64[0] &= v153;
          if ( !(unsigned __int8)sub_252BB70((__int64)v308[0], v307.m128i_i64[1], v150, 1) )
          {
            v137 += 2;
            goto LABEL_253;
          }
          v154 = v137[3];
          v155 = *(_QWORD *)(v307.m128i_i64[1] + 72) & 0xFFFFFFFFFFFFFFFCLL;
          if ( (*(_QWORD *)(v307.m128i_i64[1] + 72) & 3LL) == 3 )
            v155 = *(_QWORD *)(v155 + 24);
          v156 = *(_BYTE *)v155;
          if ( *(_BYTE *)v155 )
          {
            if ( v156 == 22 )
            {
              v155 = *(_QWORD *)(v155 + 24);
            }
            else if ( v156 <= 0x1Cu )
            {
              v155 = 0;
            }
            else
            {
              v155 = sub_B43CB0(v155);
            }
          }
          v157 = sub_250C180(v154, v155);
          *(_BYTE *)v307.m128i_i64[0] &= v157;
          if ( !(unsigned __int8)sub_252BB70((__int64)v308[0], v307.m128i_i64[1], v154, 1) )
          {
            v137 += 3;
            goto LABEL_253;
          }
          v137 += 4;
          if ( v143 == v137 )
          {
            v139 = v273 - v137;
            break;
          }
          v142 = v307.m128i_i64[1];
        }
      }
      switch ( v139 )
      {
        case 2LL:
LABEL_352:
          if ( (unsigned __int8)sub_254CAE0((__int64)&v307, *v137) )
          {
            ++v137;
LABEL_347:
            if ( (unsigned __int8)sub_254CAE0((__int64)&v307, *v137) )
            {
LABEL_254:
              v177 = (__int64 *)&v312[(unsigned int)v313];
              if ( v312 != (unsigned __int64 *)v177 )
              {
                v270 = v50;
                v178 = (__int64 *)v312;
                do
                {
                  while ( 1 )
                  {
                    v179 = *v178;
                    v307.m128i_i64[1] = (__int64)v283;
                    v307.m128i_i64[0] = v179;
                    if ( !v296.m128i_i8[0] )
                      break;
                    ++v178;
                    LOBYTE(v308[0]) = v281;
                    sub_25379E0((__int64)&v341, &v307, v179, v113, v114, v115);
                    if ( v177 == v178 )
                      goto LABEL_259;
                  }
                  ++v178;
                  LOBYTE(v308[0]) = 2;
                  sub_25379E0((__int64)&v341, &v307, v179, v113, v114, v115);
                }
                while ( v177 != v178 );
LABEL_259:
                v22 = v20.m128i_i64[0];
                v50 = v270;
              }
              if ( (v281 & 1 & (v296.m128i_i8[0] ^ 1)) != 0 )
              {
                v50 = v281 & 1 & (v296.m128i_i8[0] ^ 1);
                v180 = sub_25096F0(v289);
                sub_258BA20(a1, (__int64)a2, (_BYTE *)(a1 + 88), v22, v283, 1, v180);
              }
              goto LABEL_156;
            }
          }
          break;
        case 3LL:
          if ( (unsigned __int8)sub_254CAE0((__int64)&v307, *v137) )
          {
            ++v137;
            goto LABEL_352;
          }
          break;
        case 1LL:
          goto LABEL_347;
        default:
          goto LABEL_254;
      }
LABEL_253:
      if ( v273 != v137 )
        goto LABEL_155;
      goto LABEL_254;
    }
    if ( v20.m128i_i64[0] != sub_250D070(v289) )
    {
      v309.m128i_i8[0] = 0;
      sub_250D230((unsigned __int64 *)&v316, v20.m128i_u64[0], 1, 0);
      v109 = sub_2527850((__int64)a2, &v316, a1, &v309, v281);
      v301 = v109;
      v110 = v109;
      v302 = v111;
      if ( !(_BYTE)v111 )
        goto LABEL_58;
      if ( !v109 )
        goto LABEL_65;
      v112 = sub_25096F0(v289);
      sub_258BA20(a1, (__int64)a2, (_BYTE *)(a1 + 88), v110, (unsigned __int8 *)v20.m128i_i64[0], v281, v112);
      v15 = v259;
      v16 = v260;
      LODWORD(v17) = v342;
      v10 = v291;
      goto LABEL_15;
    }
    v181 = sub_B43CB0(v20.m128i_i64[0]);
    v309.m128i_i32[2] = 0;
    v309.m128i_i64[0] = v181;
    v183 = v181;
    if ( (v324 & 1) != 0 )
    {
      v184 = (__int64 *)&v325;
      v185 = 3;
    }
    else
    {
      v225 = v326;
      v184 = v325;
      v185 = v326 - 1;
      if ( !v326 )
      {
        v226 = v324;
        ++v323;
        v316.m128i_i64[0] = 0;
        v227 = ((unsigned int)v324 >> 1) + 1;
        goto LABEL_321;
      }
    }
    v186 = v185 & (((unsigned int)v183 >> 9) ^ ((unsigned int)v183 >> 4));
    v187 = (__int64)&v184[2 * v186];
    v182 = *(_QWORD *)v187;
    if ( v183 != *(_QWORD *)v187 )
    {
      v251 = 1;
      v252 = 0;
      while ( v182 != -4096 )
      {
        if ( v182 == -8192 && !v252 )
          v252 = v187;
        v186 = v185 & (v251 + v186);
        v187 = (__int64)&v184[2 * v186];
        v182 = *(_QWORD *)v187;
        if ( v183 == *(_QWORD *)v187 )
          goto LABEL_265;
        ++v251;
      }
      v226 = v324;
      if ( v252 )
        v187 = v252;
      ++v323;
      v316.m128i_i64[0] = v187;
      v227 = ((unsigned int)v324 >> 1) + 1;
      if ( (v324 & 1) != 0 )
      {
        v228 = 12;
        v225 = 4;
LABEL_322:
        if ( v228 <= 4 * v227 )
        {
          v225 *= 2;
        }
        else
        {
          v229 = v225 - HIDWORD(v324) - v227;
          v230 = v183;
          if ( v229 > v225 >> 3 )
          {
LABEL_324:
            v231 = v316.m128i_i64[0];
            LODWORD(v324) = (2 * (v226 >> 1) + 2) | v226 & 1;
            if ( *(_QWORD *)v316.m128i_i64[0] != -4096 )
              --HIDWORD(v324);
            *(_QWORD *)v316.m128i_i64[0] = v230;
            *(_DWORD *)(v231 + 8) = v309.m128i_i32[2];
            v232 = (unsigned int)v328;
            v316 = (__m128i)v183;
            v233 = (unsigned int)v328 + 1LL;
            LOBYTE(v317) = 0;
            if ( v233 > HIDWORD(v328) )
            {
              v256 = (unsigned __int64)v327;
              if ( v327 > &v316 || &v316 >= (__m128i *)((char *)v327 + 24 * (unsigned int)v328) )
              {
                sub_C8D5F0((__int64)&v327, v329, v233, 0x18u, v182, (__int64)v184);
                v234 = (unsigned __int64)v327;
                v232 = (unsigned int)v328;
                v235 = &v316;
              }
              else
              {
                sub_C8D5F0((__int64)&v327, v329, v233, 0x18u, v182, (__int64)v184);
                v234 = (unsigned __int64)v327;
                v232 = (unsigned int)v328;
                v235 = (__m128i *)((char *)&v316 + (_QWORD)v327 - v256);
              }
            }
            else
            {
              v234 = (unsigned __int64)v327;
              v235 = &v316;
            }
            v236 = (__m128i *)(v234 + 24 * v232);
            *v236 = _mm_loadu_si128(v235);
            v236[1].m128i_i64[0] = v235[1].m128i_i64[0];
            v188 = (unsigned int)v328;
            LODWORD(v328) = v328 + 1;
            *(_DWORD *)(v231 + 8) = v188;
            goto LABEL_266;
          }
        }
        sub_2577930((__int64)&v323, v225);
        sub_2568670((__int64)&v323, v309.m128i_i64, &v316);
        v230 = v309.m128i_i64[0];
        v226 = v324;
        goto LABEL_324;
      }
      v225 = v326;
LABEL_321:
      v228 = 3 * v225;
      goto LABEL_322;
    }
LABEL_265:
    v188 = *(unsigned int *)(v187 + 8);
LABEL_266:
    v189 = &v327->m128i_i8[24 * v188];
    if ( !*((_QWORD *)v189 + 1) )
    {
      sub_250D230((unsigned __int64 *)&v316, v183, 4, 0);
      *((_QWORD *)v189 + 1) = sub_251BBC0((__int64)a2, v316.m128i_i64[0], v316.m128i_i64[1], a1, 2, 0, 1);
    }
    v190 = a2[26];
    v191 = sub_B43CB0(v20.m128i_i64[0]);
    v192 = sub_255E1F0(*(_QWORD *)(v190 + 240), v191, 0);
    v279 = 1;
    v193 = v192;
    if ( v192 )
    {
      v194 = *(_QWORD *)(v20.m128i_i64[0] + 40);
      v195 = sub_E387E0(v192, v194);
      v193 = v195;
      if ( v195 )
        v279 = **(_QWORD **)(v195 + 8) == v194;
      else
        v279 = 0;
    }
    if ( (*(_DWORD *)(v20.m128i_i64[0] + 4) & 0x7FFFFFF) == 0 )
      goto LABEL_58;
    v196 = 0;
    v197 = 8LL * (*(_DWORD *)(v20.m128i_i64[0] + 4) & 0x7FFFFFF);
    while ( 1 )
    {
      v198 = *(_QWORD *)(v20.m128i_i64[0] - 8);
      v199 = *((_QWORD *)v189 + 1);
      v200 = *(_QWORD *)(v198 + 32LL * *(unsigned int *)(v20.m128i_i64[0] + 72) + v196);
      if ( v199 )
      {
        v286 = *(_QWORD *)(v198 + 32LL * *(unsigned int *)(v20.m128i_i64[0] + 72) + v196);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v199 + 168LL))(
               v199,
               v200,
               *(_QWORD *)(v20.m128i_i64[0] + 40)) )
        {
          v189[16] = 1;
          goto LABEL_297;
        }
        v198 = *(_QWORD *)(v20.m128i_i64[0] - 8);
        v200 = v286;
      }
      v201 = *(_QWORD *)(v198 + 4 * v196);
      if ( v201 && v20.m128i_i64[0] == v201 )
        goto LABEL_297;
      if ( !v279 || *(_BYTE *)v201 <= 0x1Cu )
        goto LABEL_291;
      if ( !v193 )
        goto LABEL_65;
      v202 = *(_QWORD *)(v201 + 40);
      if ( *(_DWORD *)(v193 + 72) )
        break;
      v203 = *(_QWORD **)(v193 + 88);
      v204 = 8LL * *(unsigned int *)(v193 + 96);
      v205 = &v203[(unsigned __int64)v204 / 8];
      v206 = v204 >> 3;
      v207 = v204 >> 5;
      if ( !v207 )
        goto LABEL_334;
      v208 = &v203[4 * v207];
      do
      {
        if ( v202 == *v203 )
          goto LABEL_289;
        if ( v202 == v203[1] )
        {
          v209 = v205 != v203 + 1;
          goto LABEL_290;
        }
        if ( v202 == v203[2] )
        {
          v209 = v205 != v203 + 2;
          goto LABEL_290;
        }
        if ( v202 == v203[3] )
        {
          v209 = v205 != v203 + 3;
          goto LABEL_290;
        }
        v203 += 4;
      }
      while ( v208 != v203 );
      v206 = v205 - v203;
LABEL_334:
      if ( v206 == 2 )
        goto LABEL_362;
      if ( v206 != 3 )
      {
        if ( v206 == 1 )
          goto LABEL_337;
        goto LABEL_291;
      }
      if ( v202 == *v203 )
        goto LABEL_289;
      ++v203;
LABEL_362:
      if ( v202 == *v203 )
        goto LABEL_289;
      ++v203;
LABEL_337:
      if ( v202 == *v203 )
      {
LABEL_289:
        v209 = v205 != v203;
        goto LABEL_290;
      }
LABEL_291:
      v210 = *(_QWORD *)(v200 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v210 == v200 + 48 )
      {
        v212 = 0;
      }
      else
      {
        if ( !v210 )
          BUG();
        v211 = *(unsigned __int8 *)(v210 - 24);
        v212 = v210 - 24;
        if ( (unsigned int)(v211 - 30) >= 0xB )
          v212 = 0;
      }
      v316.m128i_i64[1] = v212;
      v316.m128i_i64[0] = v201;
      LOBYTE(v317) = v281;
      v213 = (unsigned int)v342;
      v214 = v341;
      v215 = &v316;
      v16 = (unsigned int)v342 + 1LL;
      if ( v16 > HIDWORD(v342) )
      {
        if ( v341 > (unsigned __int64)&v316 || (unsigned __int64)&v316 >= v341 + 24LL * (unsigned int)v342 )
        {
          sub_C8D5F0((__int64)&v341, v343, (unsigned int)v342 + 1LL, 0x18u, v15, v16);
          v214 = v341;
          v213 = (unsigned int)v342;
          v215 = &v316;
        }
        else
        {
          v288 = &v316.m128i_i8[-v341];
          sub_C8D5F0((__int64)&v341, v343, (unsigned int)v342 + 1LL, 0x18u, v15, v16);
          v214 = v341;
          v213 = (unsigned int)v342;
          v215 = (__m128i *)&v288[v341];
        }
      }
      v216 = (__m128i *)(v214 + 24 * v213);
      *v216 = _mm_loadu_si128(v215);
      v217 = v215[1].m128i_i64[0];
      LODWORD(v342) = v342 + 1;
      v216[1].m128i_i64[0] = v217;
LABEL_297:
      v196 += 8;
      if ( v197 == v196 )
        goto LABEL_58;
    }
    v219 = *(_QWORD *)(v193 + 64);
    v220 = *(unsigned int *)(v193 + 80);
    v287 = (__int64 *)(v219 + 8 * v220);
    if ( !(_DWORD)v220 )
      goto LABEL_291;
    v15 = (unsigned int)(v220 - 1);
    v221 = v15 & (((unsigned int)v202 >> 9) ^ ((unsigned int)v202 >> 4));
    v222 = (__int64 *)(v219 + 8LL * v221);
    v223 = *v222;
    if ( v202 != *v222 )
    {
      for ( j = 1; ; j = v254 )
      {
        if ( v223 == -4096 )
          goto LABEL_291;
        v254 = j + 1;
        v255 = (unsigned int)v15 & (v221 + j);
        v221 = v255;
        v222 = (__int64 *)(v219 + 8 * v255);
        v223 = *v222;
        if ( v202 == *v222 )
          break;
      }
    }
    v209 = v287 != v222;
LABEL_290:
    if ( !v209 )
      goto LABEL_291;
LABEL_65:
    if ( v280 != (_BYTE *)v22 )
    {
      v52 = v306;
      v53.m128i_i64[0] = sub_250D2C0(v22, 0);
      v316 = v53;
      if ( !(unsigned __int8)sub_258BE70(a1, (__int64)a2, &v316, v52) )
        goto LABEL_67;
LABEL_58:
      LODWORD(v17) = v342;
      v10 = v291;
LABEL_15:
      if ( !(_DWORD)v17 )
      {
        v26 = (unsigned __int64)v327;
        v27 = a1;
        v28 = (__m128i *)((char *)v327 + 24 * (unsigned int)v328);
        if ( v327 != v28 )
        {
          do
          {
            while ( !*(_BYTE *)(v26 + 16) )
            {
              v26 += 24LL;
              if ( v28 == (__m128i *)v26 )
                goto LABEL_21;
            }
            v29 = *(_QWORD *)(v26 + 8);
            v26 += 24LL;
            sub_250ED80((__int64)a2, v29, a1, 1);
          }
          while ( v28 != (__m128i *)v26 );
        }
LABEL_21:
        v30 = v341;
        if ( (_BYTE *)v341 != v343 )
          goto LABEL_22;
        goto LABEL_23;
      }
      continue;
    }
    break;
  }
  if ( *v280 == 22 )
  {
    v107 = v306;
    v108.m128i_i64[0] = sub_250D2C0((unsigned __int64)v280, 0);
    v316 = v108;
    if ( (unsigned __int8)sub_258BE70(a1, (__int64)a2, &v316, v107) )
      goto LABEL_58;
  }
  if ( v20.m128i_i64[1] != sub_2509740(v289) )
  {
LABEL_67:
    v54 = sub_25096F0(v289);
    sub_258BA20(a1, (__int64)a2, (_BYTE *)(a1 + 88), v22, (unsigned __int8 *)v20.m128i_i64[1], v292, v54);
    LODWORD(v17) = v342;
    v10 = v291;
    goto LABEL_15;
  }
  v27 = a1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 128LL))(a1);
  v30 = v341;
  if ( (_BYTE *)v341 != v343 )
LABEL_22:
    _libc_free(v30);
LABEL_23:
  sub_253AA90(v349);
  if ( v344 != v346 )
    _libc_free((unsigned __int64)v344);
  if ( v327 != (__m128i *)v329 )
    _libc_free((unsigned __int64)v327);
  if ( (v324 & 1) == 0 )
    sub_C7D6A0((__int64)v325, 16LL * v326, 8);
  v31 = *(_BYTE *)(v27 + 105);
  v32 = 0;
  if ( v31 == HIBYTE(v332) )
  {
    if ( !v31 )
      goto LABEL_302;
    if ( v340 != *(_BYTE *)(v27 + 352) || (unsigned int)v338 != (unsigned __int64)*(unsigned int *)(v27 + 152) )
      goto LABEL_37;
    v33 = *(_QWORD *)(v27 + 144);
    v34 = &v337[24 * (unsigned int)v338];
    if ( v337 == v34 )
    {
LABEL_302:
      v32 = 1;
      goto LABEL_37;
    }
    v35 = (unsigned __int64)v337;
    while ( *(_QWORD *)v35 == *(_QWORD *)v33
         && *(_QWORD *)(v35 + 8) == *(_QWORD *)(v33 + 8)
         && *(_BYTE *)(v35 + 16) == *(_BYTE *)(v33 + 16) )
    {
      v35 += 24LL;
      v33 += 24;
      if ( v34 == (_BYTE *)v35 )
        goto LABEL_302;
    }
    v32 = 0;
  }
LABEL_37:
  v330 = &unk_4A171B8;
  if ( v337 != v339 )
    _libc_free((unsigned __int64)v337);
  sub_C7D6A0(v334, 24LL * (unsigned int)v336, 8);
  return v32;
}
