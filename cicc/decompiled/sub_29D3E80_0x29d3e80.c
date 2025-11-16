// Function: sub_29D3E80
// Address: 0x29d3e80
//
__int64 __fastcall sub_29D3E80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  const __m128i *v4; // r15
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __m128i *v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  const __m128i *v23; // rdi
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  const __m128i *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __m128i *v33; // rcx
  const __m128i *v34; // rdx
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r13
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r15
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // r14
  __int64 v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // r8
  _QWORD *v46; // rdi
  _QWORD *v47; // rsi
  bool v48; // al
  __int64 v49; // r9
  __int64 *v50; // r15
  unsigned __int64 v51; // r13
  unsigned __int64 v52; // r12
  __int64 v53; // r11
  unsigned __int64 v54; // r14
  __int64 v55; // rdx
  unsigned __int64 v56; // rdi
  int v57; // eax
  __m128i *v58; // rcx
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rax
  _QWORD *v64; // rdx
  __int64 v65; // rbx
  __int64 v66; // r12
  __int64 v67; // r14
  __int64 v68; // rdx
  __int64 v69; // rax
  _QWORD *v70; // rdi
  _QWORD *v71; // rsi
  bool v72; // al
  unsigned __int64 v73; // r13
  __int64 *v74; // r14
  unsigned __int64 v75; // r8
  _QWORD *v76; // rdi
  _QWORD *v77; // rsi
  bool v78; // al
  __int64 v79; // r9
  unsigned __int64 v80; // r10
  __int64 v81; // rdx
  unsigned __int64 v82; // rsi
  int v83; // eax
  __m128i *v84; // rcx
  unsigned __int64 v85; // r8
  int v86; // r11d
  _QWORD *v87; // rdi
  _QWORD *v88; // rsi
  bool v89; // al
  int v90; // edi
  __int64 v91; // rsi
  __int64 v92; // rcx
  __int64 *v93; // r8
  int v94; // ecx
  unsigned int v95; // edx
  __int64 *v96; // rdi
  __int64 v97; // r9
  __int64 v98; // rax
  __int64 v99; // rsi
  __int64 *v100; // r12
  int v101; // edx
  unsigned int v102; // eax
  __int64 *v103; // rcx
  __int64 v104; // rdi
  __int64 v105; // rdi
  __int64 v106; // rcx
  int v107; // ecx
  unsigned int v108; // edx
  __int64 *v109; // rsi
  int v110; // r10d
  __int64 v111; // rdi
  __int64 v112; // rdx
  __int64 v113; // rdi
  __int64 *v114; // rsi
  int v115; // eax
  unsigned int v116; // edx
  __int64 *v117; // rcx
  __int64 v118; // r11
  __int64 *v119; // rbx
  __int64 v120; // rax
  __int64 *k; // r12
  __int64 v122; // rsi
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 *v125; // r12
  __int64 v126; // rbx
  __int64 v127; // rcx
  int v128; // esi
  __int64 v129; // rdi
  int v130; // esi
  unsigned int v131; // edx
  __int64 *v132; // rax
  __int64 v133; // rcx
  unsigned __int64 v134; // rdx
  _QWORD *v135; // r11
  __int64 v136; // rcx
  __int64 *v137; // rbx
  __int64 *v138; // r13
  __int64 *v139; // r14
  __int64 *v140; // r13
  _QWORD *v141; // r12
  __int64 v142; // rbx
  _BYTE *v143; // rsi
  __int64 v144; // rsi
  _QWORD *v145; // rax
  unsigned __int64 *v146; // rax
  unsigned __int64 *v147; // rdi
  __int64 v148; // rdx
  _QWORD *v149; // r14
  _QWORD *v150; // rbx
  _QWORD *v151; // r13
  _QWORD *v152; // rbx
  __int64 v153; // r15
  _QWORD *v154; // rax
  _QWORD *v155; // rdx
  _QWORD *v156; // r12
  __int64 v157; // rcx
  _QWORD *v158; // rax
  _QWORD *v159; // rdx
  _BYTE *v160; // r13
  __int64 v161; // r12
  __int64 *v162; // rdx
  int v163; // eax
  __int64 *v164; // rdi
  __int64 v165; // rcx
  _QWORD *v166; // r12
  __int64 v167; // rbx
  __int64 *v168; // rax
  __int64 v169; // r13
  __int64 *v170; // rax
  __int64 v171; // rax
  int v172; // r11d
  int v173; // r11d
  __int64 *v174; // r12
  __int64 *v175; // rbx
  int v176; // edx
  __int64 v177; // rbx
  __int64 v178; // rax
  __int64 v179; // rdx
  __int64 v180; // rcx
  __int64 v181; // r8
  __int64 v182; // r9
  __int64 v183; // rdx
  __int64 v184; // rcx
  __int64 v185; // r8
  __int64 v186; // r9
  _QWORD *v187; // r12
  _QWORD *v188; // r14
  void (__fastcall *v189)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v190; // rax
  __int64 v191; // r15
  __int64 *v192; // rax
  __int64 *v193; // rdx
  __int64 v194; // rbx
  __int64 *v195; // rax
  char v196; // dl
  unsigned __int64 v197; // rax
  const __m128i *v199; // r14
  __m128i *v200; // rax
  __int64 v201; // rdx
  unsigned __int64 v202; // r8
  const __m128i *v203; // r12
  __m128i *v204; // rax
  __int64 v205; // rdx
  __int64 *v206; // r14
  __int64 v207; // rdi
  int v208; // r8d
  unsigned int v209; // r12d
  unsigned int v210; // edx
  _QWORD *v211; // rax
  __int64 v212; // rcx
  _QWORD **v213; // rax
  __int64 v214; // rcx
  _QWORD **v215; // r14
  _QWORD **v216; // r13
  __int64 *v217; // r13
  __int64 *v218; // r14
  __int64 v219; // rdi
  __int64 v220; // rax
  __int64 v221; // rax
  unsigned int v222; // eax
  __int64 v223; // rdx
  char v224; // al
  unsigned __int64 v225; // rdi
  unsigned __int64 v226; // rdi
  __int64 *v227; // rax
  unsigned int v228; // ecx
  _QWORD *v229; // rdi
  __int64 v230; // rsi
  unsigned int v231; // eax
  int v232; // eax
  unsigned __int64 v233; // rax
  unsigned __int64 v234; // rax
  int v235; // ebx
  __int64 v236; // r12
  _QWORD *v237; // rax
  _QWORD *j; // rdx
  __int64 v239; // r10
  unsigned int v240; // r11d
  int m; // r9d
  __int64 *v242; // r9
  int v243; // r11d
  _QWORD *v244; // r9
  int v245; // ecx
  unsigned int v246; // r8d
  int v247; // edx
  _QWORD *v248; // rax
  unsigned __int64 v249; // r10
  unsigned int v250; // r11d
  int v251; // edi
  int v252; // r11d
  unsigned __int64 *v253; // rdi
  int v254; // ecx
  int v255; // r10d
  int v256; // r10d
  unsigned int v257; // r15d
  unsigned __int64 v258; // rdi
  int v259; // ecx
  int v260; // ecx
  int i; // esi
  int v262; // r10d
  int v263; // eax
  int v264; // r8d
  int v265; // r8d
  __int64 v266; // r9
  unsigned int v267; // r12d
  __int64 v268; // rdi
  _QWORD *v269; // rcx
  int v270; // r8d
  int v271; // r8d
  __int64 v272; // r9
  unsigned int v273; // r12d
  __int64 v274; // rdi
  unsigned int v275; // r10d
  signed __int64 v276; // r14
  signed __int64 v277; // r12
  _QWORD *v278; // rsi
  int v279; // r12d
  __int64 v280; // rcx
  int v281; // edi
  unsigned int v282; // r10d
  int v285; // [rsp+44h] [rbp-71Ch]
  unsigned __int64 *v286; // [rsp+48h] [rbp-718h]
  _QWORD *v287; // [rsp+50h] [rbp-710h]
  __int64 v288; // [rsp+50h] [rbp-710h]
  __int64 v289; // [rsp+58h] [rbp-708h]
  void *dest; // [rsp+A8h] [rbp-6B8h]
  unsigned __int64 *v291; // [rsp+C0h] [rbp-6A0h]
  __int64 v292; // [rsp+D0h] [rbp-690h]
  __int64 v293; // [rsp+D0h] [rbp-690h]
  __int64 *v294; // [rsp+D0h] [rbp-690h]
  unsigned __int64 v295; // [rsp+D0h] [rbp-690h]
  __int64 v296; // [rsp+D8h] [rbp-688h]
  int v297; // [rsp+D8h] [rbp-688h]
  __int64 v298; // [rsp+E0h] [rbp-680h]
  __int64 v299; // [rsp+E0h] [rbp-680h]
  _QWORD *v300; // [rsp+E0h] [rbp-680h]
  __int64 *v301; // [rsp+E0h] [rbp-680h]
  _QWORD *v302; // [rsp+E0h] [rbp-680h]
  __m128i *v304; // [rsp+F0h] [rbp-670h]
  __int64 v305; // [rsp+F8h] [rbp-668h]
  __int64 v306; // [rsp+100h] [rbp-660h]
  __int64 v307; // [rsp+100h] [rbp-660h]
  __int64 *v308; // [rsp+100h] [rbp-660h]
  int v309; // [rsp+100h] [rbp-660h]
  _QWORD *v310; // [rsp+100h] [rbp-660h]
  _QWORD *v311; // [rsp+100h] [rbp-660h]
  _QWORD *v312; // [rsp+118h] [rbp-648h] BYREF
  __int64 v313; // [rsp+120h] [rbp-640h] BYREF
  _QWORD *v314; // [rsp+128h] [rbp-638h]
  __int64 v315; // [rsp+130h] [rbp-630h]
  __int64 v316; // [rsp+138h] [rbp-628h]
  __int64 *v317; // [rsp+140h] [rbp-620h]
  __int64 v318; // [rsp+148h] [rbp-618h]
  __int64 v319; // [rsp+150h] [rbp-610h] BYREF
  __int64 v320; // [rsp+158h] [rbp-608h]
  __int64 v321; // [rsp+160h] [rbp-600h]
  __int64 v322; // [rsp+168h] [rbp-5F8h]
  unsigned __int64 *v323; // [rsp+170h] [rbp-5F0h]
  __int64 v324; // [rsp+178h] [rbp-5E8h]
  __m128i *v325; // [rsp+180h] [rbp-5E0h] BYREF
  __int64 v326; // [rsp+188h] [rbp-5D8h]
  _BYTE v327[48]; // [rsp+190h] [rbp-5D0h] BYREF
  __int64 *v328; // [rsp+1C0h] [rbp-5A0h] BYREF
  __int64 v329; // [rsp+1C8h] [rbp-598h]
  _BYTE v330[48]; // [rsp+1D0h] [rbp-590h] BYREF
  __int64 v331[16]; // [rsp+200h] [rbp-560h] BYREF
  __int64 v332; // [rsp+280h] [rbp-4E0h] BYREF
  __int64 *v333; // [rsp+288h] [rbp-4D8h]
  __int64 v334; // [rsp+290h] [rbp-4D0h]
  int v335; // [rsp+298h] [rbp-4C8h]
  char v336; // [rsp+29Ch] [rbp-4C4h]
  _QWORD v337[8]; // [rsp+2A0h] [rbp-4C0h] BYREF
  unsigned __int64 v338; // [rsp+2E0h] [rbp-480h] BYREF
  unsigned __int64 v339; // [rsp+2E8h] [rbp-478h]
  unsigned __int64 v340; // [rsp+2F0h] [rbp-470h]
  char v341[8]; // [rsp+300h] [rbp-460h] BYREF
  unsigned __int64 v342; // [rsp+308h] [rbp-458h]
  char v343; // [rsp+31Ch] [rbp-444h]
  _BYTE v344[64]; // [rsp+320h] [rbp-440h] BYREF
  unsigned __int64 v345; // [rsp+360h] [rbp-400h]
  __int64 v346; // [rsp+368h] [rbp-3F8h]
  unsigned __int64 v347; // [rsp+370h] [rbp-3F0h]
  char v348[8]; // [rsp+380h] [rbp-3E0h] BYREF
  unsigned __int64 v349; // [rsp+388h] [rbp-3D8h]
  char v350; // [rsp+39Ch] [rbp-3C4h]
  char v351[64]; // [rsp+3A0h] [rbp-3C0h] BYREF
  const __m128i *v352; // [rsp+3E0h] [rbp-380h]
  const __m128i *v353; // [rsp+3E8h] [rbp-378h]
  unsigned __int64 v354; // [rsp+3F0h] [rbp-370h]
  char v355[8]; // [rsp+3F8h] [rbp-368h] BYREF
  unsigned __int64 v356; // [rsp+400h] [rbp-360h]
  char v357; // [rsp+414h] [rbp-34Ch]
  char v358[64]; // [rsp+418h] [rbp-348h] BYREF
  const __m128i *v359; // [rsp+458h] [rbp-308h]
  const __m128i *v360; // [rsp+460h] [rbp-300h]
  __int64 v361; // [rsp+468h] [rbp-2F8h]
  unsigned __int64 *v362; // [rsp+470h] [rbp-2F0h] BYREF
  unsigned __int64 v363; // [rsp+478h] [rbp-2E8h]
  unsigned __int64 v364; // [rsp+480h] [rbp-2E0h] BYREF
  char v365; // [rsp+48Ch] [rbp-2D4h]
  _BYTE v366[64]; // [rsp+490h] [rbp-2D0h] BYREF
  unsigned __int64 v367; // [rsp+4D0h] [rbp-290h]
  __int64 v368; // [rsp+4D8h] [rbp-288h]
  __int64 v369; // [rsp+4E0h] [rbp-280h]
  __int64 v370; // [rsp+680h] [rbp-E0h]
  __int64 v371; // [rsp+688h] [rbp-D8h]
  __int64 v372; // [rsp+690h] [rbp-D0h]
  __int64 v373; // [rsp+698h] [rbp-C8h]
  char v374; // [rsp+6A0h] [rbp-C0h]
  __int64 v375; // [rsp+6A8h] [rbp-B8h]
  char *v376; // [rsp+6B0h] [rbp-B0h]
  __int64 v377; // [rsp+6B8h] [rbp-A8h]
  int v378; // [rsp+6C0h] [rbp-A0h]
  char v379; // [rsp+6C4h] [rbp-9Ch]
  char v380; // [rsp+6C8h] [rbp-98h] BYREF
  __int16 v381; // [rsp+708h] [rbp-58h]
  _QWORD *v382; // [rsp+710h] [rbp-50h]
  _QWORD *v383; // [rsp+718h] [rbp-48h]
  __int64 v384; // [rsp+720h] [rbp-40h]

  v286 = *(unsigned __int64 **)(a1 + 80);
  if ( v286 != *(unsigned __int64 **)(a1 + 72) )
  {
    v291 = *(unsigned __int64 **)(a1 + 72);
    v3 = 0;
    v4 = (const __m128i *)&v362;
    while ( 1 )
    {
      v5 = *v291;
      v338 = 0;
      memset(v331, 0, 0x78u);
      v331[1] = (__int64)&v331[4];
      v334 = 0x100000008LL;
      v337[0] = v5;
      v362 = (unsigned __int64 *)v5;
      LODWORD(v331[2]) = 8;
      BYTE4(v331[3]) = 1;
      v333 = v337;
      v339 = 0;
      v340 = 0;
      v335 = 0;
      v336 = 1;
      v332 = 1;
      LOBYTE(v364) = 0;
      sub_29D3BC0((__int64)&v338, v4);
      sub_C8CF70((__int64)v4, v366, 8, (__int64)&v331[4], (__int64)v331);
      v6 = v331[12];
      memset(&v331[12], 0, 24);
      v367 = v6;
      v368 = v331[13];
      v369 = v331[14];
      sub_C8CF70((__int64)v341, v344, 8, (__int64)v337, (__int64)&v332);
      v7 = v338;
      v338 = 0;
      v345 = v7;
      v8 = v339;
      v339 = 0;
      v346 = v8;
      v9 = v340;
      v340 = 0;
      v347 = v9;
      sub_C8CF70((__int64)v348, v351, 8, (__int64)v344, (__int64)v341);
      v10 = v345;
      v345 = 0;
      v352 = (const __m128i *)v10;
      v11 = v346;
      v346 = 0;
      v353 = (const __m128i *)v11;
      v12 = v347;
      v347 = 0;
      v354 = v12;
      sub_C8CF70((__int64)v355, v358, 8, (__int64)v366, (__int64)v4);
      v16 = v367;
      v367 = 0;
      v359 = (const __m128i *)v16;
      v17 = v368;
      v368 = 0;
      v360 = (const __m128i *)v17;
      v18 = v369;
      v369 = 0;
      v361 = v18;
      if ( v345 )
        j_j___libc_free_0(v345);
      if ( !v343 )
        _libc_free(v342);
      if ( v367 )
        j_j___libc_free_0(v367);
      if ( !v365 )
        _libc_free(v363);
      if ( v338 )
        j_j___libc_free_0(v338);
      if ( !v336 )
        _libc_free((unsigned __int64)v333);
      if ( v331[12] )
        j_j___libc_free_0(v331[12]);
      if ( !BYTE4(v331[3]) )
        _libc_free(v331[1]);
      sub_C8CD80((__int64)&v332, (__int64)v337, (__int64)v348, v13, v14, v15);
      v23 = v353;
      v24 = (unsigned __int64)v352;
      v338 = 0;
      v339 = 0;
      v340 = 0;
      v25 = (char *)v353 - (char *)v352;
      if ( v353 == v352 )
      {
        v25 = 0;
        v27 = 0;
      }
      else
      {
        if ( v25 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_451;
        v26 = sub_22077B0((char *)v353 - (char *)v352);
        v23 = v353;
        v24 = (unsigned __int64)v352;
        v27 = v26;
      }
      v338 = v27;
      v339 = v27;
      v340 = v27 + v25;
      if ( (const __m128i *)v24 != v23 )
      {
        v20 = (__m128i *)v27;
        v28 = (const __m128i *)v24;
        do
        {
          if ( v20 )
          {
            *v20 = _mm_loadu_si128(v28);
            v21 = v28[1].m128i_i64[0];
            v20[1].m128i_i64[0] = v21;
          }
          v28 = (const __m128i *)((char *)v28 + 24);
          v20 = (__m128i *)((char *)v20 + 24);
        }
        while ( v28 != v23 );
        v27 += 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v24) >> 3) + 24;
      }
      v339 = v27;
      sub_C8CD80((__int64)v341, (__int64)v344, (__int64)v355, (__int64)v20, v21, v22);
      v23 = v360;
      v24 = (unsigned __int64)v359;
      v345 = 0;
      v346 = 0;
      v347 = 0;
      v31 = (char *)v360 - (char *)v359;
      if ( v360 == v359 )
      {
        v31 = 0;
        v32 = 0;
      }
      else
      {
        if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_451:
          sub_4261EA(v23, v24, v19);
        v32 = sub_22077B0((char *)v360 - (char *)v359);
        v23 = v360;
        v24 = (unsigned __int64)v359;
      }
      v345 = v32;
      v346 = v32;
      v347 = v32 + v31;
      if ( (const __m128i *)v24 == v23 )
      {
        v35 = v32;
      }
      else
      {
        v33 = (__m128i *)v32;
        v34 = (const __m128i *)v24;
        do
        {
          if ( v33 )
          {
            *v33 = _mm_loadu_si128(v34);
            v29 = v34[1].m128i_i64[0];
            v33[1].m128i_i64[0] = v29;
          }
          v34 = (const __m128i *)((char *)v34 + 24);
          v33 = (__m128i *)((char *)v33 + 24);
        }
        while ( v34 != v23 );
        v35 = v345;
        v32 += 8 * (((unsigned __int64)&v34[-2].m128i_u64[1] - v24) >> 3) + 24;
      }
      v36 = v339;
      v37 = v338;
      v346 = v32;
      v304 = (__m128i *)v4;
      v38 = v339 - v338;
      if ( v339 - v338 == v32 - v35 )
        goto LABEL_252;
      do
      {
LABEL_38:
        v39 = *(_QWORD *)(v36 - 24);
        if ( *(_DWORD *)(v39 + 16) == 1 )
          goto LABEL_239;
        v315 = 0;
        v325 = (__m128i *)v327;
        v326 = 0x200000000LL;
        v316 = 0;
        v317 = &v319;
        v318 = 0;
        v314 = 0;
        v40 = *(unsigned __int64 **)(v39 + 8);
        v313 = 0;
        v41 = *v40;
        v42 = *(_QWORD *)(*v40 + 16);
        if ( v42 )
        {
          while ( 1 )
          {
            v43 = *(_QWORD *)(v42 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v43 - 30) <= 0xAu )
              break;
            v42 = *(_QWORD *)(v42 + 8);
            if ( !v42 )
              goto LABEL_63;
          }
LABEL_43:
          v44 = *(_QWORD *)(v43 + 40);
          v331[0] = v44;
          v362 = (unsigned __int64 *)v44;
          v45 = *(unsigned int *)(v39 + 72);
          if ( (_DWORD)v45 )
          {
            v105 = *(_QWORD *)(v39 + 64);
            v106 = *(unsigned int *)(v39 + 80);
            v49 = v105 + 8 * v106;
            if ( !(_DWORD)v106 )
              goto LABEL_47;
            v107 = v106 - 1;
            v108 = v107 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
            v109 = (__int64 *)(v105 + 8LL * v108);
            v45 = *v109;
            if ( v44 != *v109 )
            {
              for ( i = 1; ; i = v262 )
              {
                if ( v45 == -4096 )
                  goto LABEL_47;
                v262 = i + 1;
                v108 = v107 & (i + v108);
                v109 = (__int64 *)(v105 + 8LL * v108);
                v45 = *v109;
                if ( v44 == *v109 )
                  break;
              }
            }
            v48 = v49 != (_QWORD)v109;
          }
          else
          {
            v46 = *(_QWORD **)(v39 + 88);
            v47 = &v46[*(unsigned int *)(v39 + 96)];
            v48 = v47 != sub_29D3470(v46, (__int64)v47, v304->m128i_i64);
          }
          if ( v48 )
            sub_29D3C00((__int64)&v313, v331);
LABEL_47:
          while ( 1 )
          {
            v42 = *(_QWORD *)(v42 + 8);
            if ( !v42 )
              break;
            v43 = *(_QWORD *)(v42 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v43 - 30) <= 0xAu )
              goto LABEL_43;
          }
          if ( &v317[(unsigned int)v318] != v317 )
          {
            v306 = v39;
            v50 = &v317[(unsigned int)v318];
            v51 = (unsigned __int64)v317;
            v52 = v41;
            do
            {
              v59 = *(_QWORD *)v51;
              v60 = *(_QWORD *)(*(_QWORD *)v51 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v60 == *(_QWORD *)v51 + 48LL )
                goto LABEL_467;
              if ( !v60 )
                goto LABEL_469;
              if ( (unsigned int)*(unsigned __int8 *)(v60 - 24) - 30 > 0xA )
                goto LABEL_467;
              v61 = *(_QWORD *)(v60 - 56);
              if ( v61 && v52 == v61 )
              {
                v54 = v52;
                v53 = 0;
              }
              else
              {
                v53 = v52;
                v54 = 0;
              }
              v55 = (unsigned int)v326;
              v56 = (unsigned __int64)v325;
              v57 = v326;
              v58 = (__m128i *)((char *)v325 + 24 * (unsigned int)v326);
              if ( (unsigned int)v326 >= (unsigned __int64)HIDWORD(v326) )
              {
                v364 = v53;
                v363 = v54;
                v199 = v304;
                v362 = (unsigned __int64 *)v59;
                if ( HIDWORD(v326) < (unsigned __int64)(unsigned int)v326 + 1 )
                {
                  if ( v325 > v304 || v58 <= v304 )
                  {
                    v199 = v304;
                    sub_C8D5F0((__int64)&v325, v327, (unsigned int)v326 + 1LL, 0x18u, v45, v49);
                    v56 = (unsigned __int64)v325;
                    v55 = (unsigned int)v326;
                  }
                  else
                  {
                    v276 = (char *)v304 - (char *)v325;
                    sub_C8D5F0((__int64)&v325, v327, (unsigned int)v326 + 1LL, 0x18u, v45, v49);
                    v56 = (unsigned __int64)v325;
                    v55 = (unsigned int)v326;
                    v199 = (__m128i *)((char *)v325 + v276);
                  }
                }
                v200 = (__m128i *)(v56 + 24 * v55);
                *v200 = _mm_loadu_si128(v199);
                v201 = v199[1].m128i_i64[0];
                LODWORD(v326) = v326 + 1;
                v200[1].m128i_i64[0] = v201;
              }
              else
              {
                if ( v58 )
                {
                  v58[1].m128i_i64[0] = v53;
                  v58->m128i_i64[0] = v59;
                  v58->m128i_i64[1] = v54;
                  v57 = v326;
                }
                LODWORD(v326) = v57 + 1;
              }
              v51 += 8LL;
            }
            while ( v50 != (__int64 *)v51 );
            v39 = v306;
          }
        }
LABEL_63:
        ++v313;
        if ( !(_DWORD)v315 )
        {
          if ( !HIDWORD(v315) )
            goto LABEL_69;
          v62 = (unsigned int)v316;
          if ( (unsigned int)v316 > 0x40 )
          {
            sub_C7D6A0((__int64)v314, 8LL * (unsigned int)v316, 8);
            v314 = 0;
            v315 = 0;
            LODWORD(v316) = 0;
            goto LABEL_69;
          }
LABEL_66:
          v63 = v314;
          v64 = &v314[v62];
          if ( v314 != v64 )
          {
            do
              *v63++ = -4096;
            while ( v64 != v63 );
          }
          v315 = 0;
          goto LABEL_69;
        }
        v228 = 4 * v315;
        v62 = (unsigned int)v316;
        if ( (unsigned int)(4 * v315) < 0x40 )
          v228 = 64;
        if ( v228 >= (unsigned int)v316 )
          goto LABEL_66;
        v229 = v314;
        v230 = (unsigned int)v316;
        if ( (_DWORD)v315 == 1 )
        {
          v236 = 1024;
          v235 = 128;
LABEL_334:
          sub_C7D6A0((__int64)v314, v230 * 8, 8);
          LODWORD(v316) = v235;
          v237 = (_QWORD *)sub_C7D670(v236, 8);
          v315 = 0;
          v314 = v237;
          for ( j = &v237[(unsigned int)v316]; j != v237; ++v237 )
          {
            if ( v237 )
              *v237 = -4096;
          }
          goto LABEL_69;
        }
        _BitScanReverse(&v231, v315 - 1);
        v232 = 1 << (33 - (v231 ^ 0x1F));
        if ( v232 < 64 )
          v232 = 64;
        if ( v232 != (_DWORD)v316 )
        {
          v233 = (4 * v232 / 3u + 1) | ((unsigned __int64)(4 * v232 / 3u + 1) >> 1);
          v234 = ((v233 | (v233 >> 2)) >> 4)
               | v233
               | (v233 >> 2)
               | ((((v233 | (v233 >> 2)) >> 4) | v233 | (v233 >> 2)) >> 8);
          v235 = (v234 | (v234 >> 16)) + 1;
          v236 = 8 * ((v234 | (v234 >> 16)) + 1);
          goto LABEL_334;
        }
        v315 = 0;
        v278 = &v314[v230];
        do
        {
          if ( v229 )
            *v229 = -4096;
          ++v229;
        }
        while ( v278 != v229 );
LABEL_69:
        LODWORD(v318) = 0;
        v65 = *(_QWORD *)(v39 + 8);
        v66 = v65 + 8LL * *(unsigned int *)(v39 + 16);
        if ( v65 == v66 )
          goto LABEL_119;
        do
        {
          while ( 1 )
          {
            v67 = *(_QWORD *)(*(_QWORD *)v65 + 16LL);
            if ( v67 )
              break;
LABEL_78:
            v65 += 8;
            if ( v66 == v65 )
              goto LABEL_79;
          }
          do
          {
            v68 = *(_QWORD *)(v67 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v68 - 30) <= 0xAu )
            {
LABEL_74:
              v69 = *(_QWORD *)(v68 + 40);
              v331[0] = v69;
              v362 = (unsigned __int64 *)v69;
              if ( *(_DWORD *)(v39 + 72) )
              {
                v91 = *(_QWORD *)(v39 + 64);
                v92 = *(unsigned int *)(v39 + 80);
                v93 = (__int64 *)(v91 + 8 * v92);
                if ( (_DWORD)v92 )
                {
                  v94 = v92 - 1;
                  v95 = v94 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
                  v96 = (__int64 *)(v91 + 8LL * v95);
                  v97 = *v96;
                  if ( v69 == *v96 )
                  {
LABEL_107:
                    v72 = v93 != v96;
                    goto LABEL_76;
                  }
                  v90 = 1;
                  while ( v97 != -4096 )
                  {
                    v110 = v90 + 1;
                    v111 = v94 & (v95 + v90);
                    v95 = v111;
                    v96 = (__int64 *)(v91 + 8 * v111);
                    v97 = *v96;
                    if ( v69 == *v96 )
                      goto LABEL_107;
                    v90 = v110;
                  }
                }
              }
              else
              {
                v70 = *(_QWORD **)(v39 + 88);
                v71 = &v70[*(unsigned int *)(v39 + 96)];
                v72 = v71 != sub_29D3470(v70, (__int64)v71, v304->m128i_i64);
LABEL_76:
                if ( v72 )
                  goto LABEL_77;
              }
              sub_29D3C00((__int64)&v313, v331);
LABEL_77:
              while ( 1 )
              {
                v67 = *(_QWORD *)(v67 + 8);
                if ( !v67 )
                  goto LABEL_78;
                v68 = *(_QWORD *)(v67 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v68 - 30) <= 0xAu )
                  goto LABEL_74;
              }
            }
            v67 = *(_QWORD *)(v67 + 8);
          }
          while ( v67 );
          v65 += 8;
        }
        while ( v66 != v65 );
LABEL_79:
        v73 = (unsigned __int64)v317;
        v74 = &v317[(unsigned int)v318];
        if ( v317 != v74 )
        {
          do
          {
            v79 = *(_QWORD *)v73;
            v85 = *(_QWORD *)(*(_QWORD *)v73 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v85 == *(_QWORD *)v73 + 48LL )
              goto LABEL_467;
            if ( !v85 )
LABEL_469:
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v85 - 24) - 30 > 0xA )
LABEL_467:
              BUG();
            v80 = *(_QWORD *)(v85 - 56);
            v362 = (unsigned __int64 *)v80;
            v86 = *(_DWORD *)(v39 + 72);
            if ( v86 )
            {
              v98 = *(unsigned int *)(v39 + 80);
              v99 = *(_QWORD *)(v39 + 64);
              v100 = (__int64 *)(v99 + 8 * v98);
              if ( !(_DWORD)v98 )
                goto LABEL_98;
              v101 = v98 - 1;
              v102 = (v98 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
              v103 = (__int64 *)(v99 + 8LL * v102);
              v104 = *v103;
              if ( v80 != *v103 )
              {
                v260 = 1;
                while ( v104 != -4096 )
                {
                  v102 = v101 & (v260 + v102);
                  v309 = v260 + 1;
                  v103 = (__int64 *)(v99 + 8LL * v102);
                  v104 = *v103;
                  if ( v80 == *v103 )
                    goto LABEL_110;
                  v260 = v309;
                }
LABEL_98:
                v80 = 0;
                goto LABEL_81;
              }
LABEL_110:
              v89 = v100 != v103;
            }
            else
            {
              v87 = *(_QWORD **)(v39 + 88);
              v88 = &v87[*(unsigned int *)(v39 + 96)];
              v89 = v88 != sub_29D3470(v87, (__int64)v88, v304->m128i_i64);
            }
            if ( !v89 )
              goto LABEL_98;
LABEL_81:
            if ( (*(_DWORD *)(v85 - 20) & 0x7FFFFFF) == 1 || (v75 = *(_QWORD *)(v85 - 88)) == 0 )
            {
LABEL_86:
              v75 = 0;
              goto LABEL_87;
            }
            v362 = (unsigned __int64 *)v75;
            if ( !v86 )
            {
              v76 = *(_QWORD **)(v39 + 88);
              v77 = &v76[*(unsigned int *)(v39 + 96)];
              v78 = v77 != sub_29D3470(v76, (__int64)v77, v304->m128i_i64);
              goto LABEL_85;
            }
            v112 = *(unsigned int *)(v39 + 80);
            v113 = *(_QWORD *)(v39 + 64);
            v114 = (__int64 *)(v113 + 8 * v112);
            if ( !(_DWORD)v112 )
              goto LABEL_371;
            v115 = v112 - 1;
            v116 = (v112 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v117 = (__int64 *)(v113 + 8LL * v116);
            v118 = *v117;
            if ( v75 != *v117 )
            {
              v259 = 1;
              while ( v118 != -4096 )
              {
                v279 = v259 + 1;
                v280 = v115 & (v116 + v259);
                v116 = v280;
                v117 = (__int64 *)(v113 + 8 * v280);
                v118 = *v117;
                if ( v75 == *v117 )
                  goto LABEL_118;
                v259 = v279;
              }
LABEL_371:
              v117 = v114;
            }
LABEL_118:
            v78 = v114 != v117;
LABEL_85:
            if ( !v78 )
              goto LABEL_86;
LABEL_87:
            v81 = (unsigned int)v326;
            v82 = (unsigned __int64)v325;
            v83 = v326;
            v84 = (__m128i *)((char *)v325 + 24 * (unsigned int)v326);
            if ( (unsigned int)v326 >= (unsigned __int64)HIDWORD(v326) )
            {
              v364 = v75;
              v202 = (unsigned int)v326 + 1LL;
              v203 = v304;
              v362 = (unsigned __int64 *)v79;
              v363 = v80;
              if ( HIDWORD(v326) < v202 )
              {
                if ( v325 > v304 || v84 <= v304 )
                {
                  v203 = v304;
                  sub_C8D5F0((__int64)&v325, v327, (unsigned int)v326 + 1LL, 0x18u, v202, v79);
                  v82 = (unsigned __int64)v325;
                  v81 = (unsigned int)v326;
                }
                else
                {
                  v277 = (char *)v304 - (char *)v325;
                  sub_C8D5F0((__int64)&v325, v327, (unsigned int)v326 + 1LL, 0x18u, v202, v79);
                  v82 = (unsigned __int64)v325;
                  v81 = (unsigned int)v326;
                  v203 = (__m128i *)((char *)v325 + v277);
                }
              }
              v204 = (__m128i *)(v82 + 24 * v81);
              *v204 = _mm_loadu_si128(v203);
              v205 = v203[1].m128i_i64[0];
              LODWORD(v326) = v326 + 1;
              v204[1].m128i_i64[0] = v205;
            }
            else
            {
              if ( v84 )
              {
                v84[1].m128i_i64[0] = v75;
                v84->m128i_i64[0] = v79;
                v84->m128i_i64[1] = v80;
                v83 = v326;
              }
              LODWORD(v326) = v83 + 1;
            }
            v73 += 8LL;
          }
          while ( v74 != (__int64 *)v73 );
        }
LABEL_119:
        v321 = 0;
        v328 = (__int64 *)v330;
        v329 = 0x600000000LL;
        v322 = 0;
        v323 = (unsigned __int64 *)&v325;
        v324 = 0;
        v320 = 0;
        v119 = *(__int64 **)(v39 + 8);
        v120 = *(unsigned int *)(v39 + 16);
        v319 = 0;
        for ( k = &v119[v120]; v119 != k; sub_29D3C00((__int64)&v319, k) )
          --k;
        BYTE4(v331[0]) = 0;
        v122 = (__int64)v304;
        v362 = &v364;
        v363 = 0x1000000000LL;
        v381 = 0;
        v372 = a2;
        v370 = 0;
        v371 = 0;
        v373 = 0;
        v374 = 0;
        v375 = 0;
        v376 = &v380;
        v377 = 8;
        v378 = 0;
        v379 = 1;
        v382 = 0;
        v383 = 0;
        v384 = 0;
        sub_3194260(&v325, v304, &v328, "irr", 3, v331[0]);
        if ( a3 )
        {
          v125 = v328;
          v126 = (unsigned int)v329;
          v127 = **(_QWORD **)(v39 + 8);
          v128 = *(_DWORD *)(a3 + 24);
          v129 = *(_QWORD *)(a3 + 8);
          if ( !v128 )
            goto LABEL_383;
          v130 = v128 - 1;
          v131 = v130 & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
          v132 = (__int64 *)(v129 + 16LL * v131);
          v124 = *v132;
          if ( *v132 != v127 )
          {
            v263 = 1;
            while ( v124 != -4096 )
            {
              v123 = (unsigned int)(v263 + 1);
              v131 = v130 & (v263 + v131);
              v132 = (__int64 *)(v129 + 16LL * v131);
              v124 = *v132;
              if ( v127 == *v132 )
                goto LABEL_124;
              v263 = v123;
            }
LABEL_383:
            v305 = 0;
            goto LABEL_127;
          }
LABEL_124:
          v305 = v132[1];
          if ( v305 && v127 == **(_QWORD **)(v305 + 32) )
            v305 = *(_QWORD *)v305;
LABEL_127:
          *(_QWORD *)(a3 + 136) += 160LL;
          v133 = *(_QWORD *)(a3 + 56);
          v134 = ((v133 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 160;
          if ( *(_QWORD *)(a3 + 64) >= v134 && v133 )
          {
            *(_QWORD *)(a3 + 56) = v134;
            v135 = (_QWORD *)((v133 + 7) & 0xFFFFFFFFFFFFFFF8LL);
          }
          else
          {
            v135 = (_QWORD *)sub_9D1E70(a3 + 56, 160, 160, 3);
          }
          memset(v135, 0, 0xA0u);
          v136 = v305;
          v135[9] = 8;
          v135[8] = v135 + 11;
          *((_BYTE *)v135 + 84) = 1;
          v331[0] = (__int64)v135;
          if ( v305 )
          {
            *v135 = v305;
            v122 = *(_QWORD *)(v305 + 16);
            if ( v122 == *(_QWORD *)(v305 + 24) )
            {
              v310 = v135;
              sub_D4C7F0(v305 + 8, (_BYTE *)v122, v331);
              v135 = v310;
            }
            else
            {
              if ( v122 )
              {
                *(_QWORD *)v122 = v331[0];
                v122 = *(_QWORD *)(v305 + 16);
              }
              v122 += 8;
              *(_QWORD *)(v305 + 16) = v122;
            }
          }
          else
          {
            v122 = *(_QWORD *)(a3 + 40);
            if ( v122 == *(_QWORD *)(a3 + 48) )
            {
              v311 = v135;
              sub_D4C7F0(a3 + 32, (_BYTE *)v122, v331);
              v135 = v311;
            }
            else
            {
              if ( v122 )
              {
                *(_QWORD *)v122 = v135;
                v122 = *(_QWORD *)(a3 + 40);
              }
              v122 += 8;
              *(_QWORD *)(a3 + 40) = v122;
            }
          }
          v137 = &v125[v126];
          v138 = v135;
          if ( v125 != v137 )
          {
            do
            {
              v122 = *v125++;
              sub_D4F330(v138, v122, a3);
            }
            while ( v137 != v125 );
            v135 = v138;
          }
          v139 = *(__int64 **)(v39 + 88);
          v140 = &v139[*(unsigned int *)(v39 + 96)];
          if ( v139 != v140 )
          {
            dest = (void *)v39;
            v141 = v135;
            v292 = (__int64)(v135 + 4);
            v298 = (__int64)(v135 + 7);
            while ( 1 )
            {
              v142 = *v139;
              v331[0] = *v139;
              v143 = (_BYTE *)v141[5];
              if ( v143 == (_BYTE *)v141[6] )
              {
                sub_9319A0(v292, v143, v331);
                v144 = v331[0];
              }
              else
              {
                if ( v143 )
                {
                  *(_QWORD *)v143 = v142;
                  v143 = (_BYTE *)v141[5];
                }
                v141[5] = v143 + 8;
                v144 = v142;
              }
              if ( *((_BYTE *)v141 + 84) )
              {
                v145 = (_QWORD *)v141[8];
                v136 = *((unsigned int *)v141 + 19);
                v134 = (unsigned __int64)&v145[v136];
                if ( v145 != (_QWORD *)v134 )
                {
                  while ( v144 != *v145 )
                  {
                    if ( (_QWORD *)v134 == ++v145 )
                      goto LABEL_202;
                  }
LABEL_149:
                  v122 = *(unsigned int *)(a3 + 24);
                  v123 = *(_QWORD *)(a3 + 8);
                  if ( !(_DWORD)v122 )
                    goto LABEL_194;
                  goto LABEL_150;
                }
LABEL_202:
                if ( (unsigned int)v136 < *((_DWORD *)v141 + 18) )
                {
                  v136 = (unsigned int)(v136 + 1);
                  *((_DWORD *)v141 + 19) = v136;
                  *(_QWORD *)v134 = v144;
                  ++v141[7];
                  goto LABEL_149;
                }
              }
              sub_C8CC70(v298, v144, v134, v136, v123, v124);
              v122 = *(unsigned int *)(a3 + 24);
              v123 = *(_QWORD *)(a3 + 8);
              if ( !(_DWORD)v122 )
              {
LABEL_194:
                if ( !v305 )
                {
                  ++*(_QWORD *)a3;
                  goto LABEL_196;
                }
                goto LABEL_152;
              }
LABEL_150:
              v124 = (unsigned int)(v122 - 1);
              v136 = (unsigned int)v124 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
              v146 = (unsigned __int64 *)(v123 + 16 * v136);
              v134 = *v146;
              v147 = v146;
              if ( *v146 != v142 )
              {
                v249 = *v146;
                v250 = v136;
                v251 = 1;
                while ( v249 != -4096 )
                {
                  v250 = v124 & (v251 + v250);
                  v297 = v251 + 1;
                  v147 = (unsigned __int64 *)(v123 + 16LL * v250);
                  v249 = *v147;
                  if ( v142 == *v147 )
                    goto LABEL_151;
                  v251 = v297;
                }
                if ( v305 )
                  goto LABEL_152;
LABEL_355:
                v252 = 1;
                v253 = 0;
                while ( v134 != -4096 )
                {
                  if ( v134 == -8192 && !v253 )
                    v253 = v146;
                  v136 = (unsigned int)v124 & (v252 + (_DWORD)v136);
                  v146 = (unsigned __int64 *)(v123 + 16LL * (unsigned int)v136);
                  v134 = *v146;
                  if ( v142 == *v146 )
                    goto LABEL_201;
                  ++v252;
                }
                v254 = *(_DWORD *)(a3 + 16);
                if ( v253 )
                  v146 = v253;
                ++*(_QWORD *)a3;
                v134 = (unsigned int)(v254 + 1);
                if ( 4 * (int)v134 >= (unsigned int)(3 * v122) )
                {
LABEL_196:
                  v122 = (unsigned int)(2 * v122);
                  sub_D4F150(a3, v122);
                  v172 = *(_DWORD *)(a3 + 24);
                  if ( !v172 )
                    goto LABEL_468;
                  v173 = v172 - 1;
                  v124 = *(_QWORD *)(a3 + 8);
                  v136 = v173 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
                  v134 = (unsigned int)(*(_DWORD *)(a3 + 16) + 1);
                  v146 = (unsigned __int64 *)(v124 + 16 * v136);
                  v123 = *v146;
                  if ( *v146 != v142 )
                  {
                    v281 = 1;
                    v122 = 0;
                    while ( v123 != -4096 )
                    {
                      if ( !v122 && v123 == -8192 )
                        v122 = (__int64)v146;
                      v136 = v173 & (unsigned int)(v136 + v281);
                      v146 = (unsigned __int64 *)(v124 + 16 * v136);
                      v123 = *v146;
                      if ( v142 == *v146 )
                        goto LABEL_198;
                      ++v281;
                    }
                    if ( v122 )
                      v146 = (unsigned __int64 *)v122;
                  }
                }
                else
                {
                  v136 = (unsigned int)(v122 - *(_DWORD *)(a3 + 20) - v134);
                  if ( (unsigned int)v136 <= (unsigned int)v122 >> 3 )
                  {
                    sub_D4F150(a3, v122);
                    v255 = *(_DWORD *)(a3 + 24);
                    if ( !v255 )
                      goto LABEL_468;
                    v256 = v255 - 1;
                    v123 = *(_QWORD *)(a3 + 8);
                    v122 = 1;
                    v257 = v256 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
                    v134 = (unsigned int)(*(_DWORD *)(a3 + 16) + 1);
                    v136 = 0;
                    v146 = (unsigned __int64 *)(v123 + 16LL * v257);
                    v258 = *v146;
                    if ( v142 != *v146 )
                    {
                      while ( v258 != -4096 )
                      {
                        if ( v258 == -8192 && !v136 )
                          v136 = (__int64)v146;
                        v124 = (unsigned int)(v122 + 1);
                        v122 = v256 & (v257 + (unsigned int)v122);
                        v257 = v122;
                        v146 = (unsigned __int64 *)(v123 + 16LL * (unsigned int)v122);
                        v258 = *v146;
                        if ( v142 == *v146 )
                          goto LABEL_198;
                        v122 = (unsigned int)v124;
                      }
                      if ( v136 )
                        v146 = (unsigned __int64 *)v136;
                    }
                  }
                }
LABEL_198:
                *(_DWORD *)(a3 + 16) = v134;
                if ( *v146 != -4096 )
                  --*(_DWORD *)(a3 + 20);
                *v146 = v142;
                v146[1] = 0;
LABEL_201:
                v146[1] = (unsigned __int64)v141;
                goto LABEL_152;
              }
LABEL_151:
              if ( v305 == v147[1] )
              {
                if ( v134 != v142 )
                  goto LABEL_355;
                goto LABEL_201;
              }
LABEL_152:
              if ( v140 == ++v139 )
              {
                v39 = (__int64)dest;
                v135 = v141;
                break;
              }
            }
          }
          v296 = **(_QWORD **)(v39 + 8);
          v148 = a3 + 32;
          if ( v305 )
            v148 = v305 + 8;
          v149 = *(_QWORD **)(v148 + 8);
          v150 = *(_QWORD **)v148;
          v307 = v148;
          if ( v149 != *(_QWORD **)v148 )
          {
            v299 = v39;
            v151 = *(_QWORD **)v148;
            v152 = v135;
            v153 = (__int64)(v135 + 7);
            while ( 1 )
            {
              v123 = (__int64)v151;
              if ( v152 == (_QWORD *)*v151 )
                goto LABEL_191;
              v122 = **(_QWORD **)(*v151 + 32LL);
              if ( *((_BYTE *)v152 + 84) )
              {
                v154 = (_QWORD *)v152[8];
                v155 = &v154[*((unsigned int *)v152 + 19)];
                if ( v154 == v155 )
                  goto LABEL_191;
                while ( v122 != *v154 )
                {
                  if ( v155 == ++v154 )
                    goto LABEL_191;
                }
LABEL_164:
                v156 = v149 - 1;
                if ( v151 == v149 - 1 )
                {
LABEL_172:
                  v39 = v299;
                  v135 = v152;
                  v150 = (_QWORD *)v123;
                  v160 = *(_BYTE **)(v307 + 8);
                  goto LABEL_173;
                }
                while ( 1 )
                {
                  v157 = *v156;
                  v149 = v156;
                  if ( v152 == (_QWORD *)*v156 )
                    goto LABEL_190;
                  v122 = **(_QWORD **)(v157 + 32);
                  if ( *((_BYTE *)v152 + 84) )
                  {
                    v158 = (_QWORD *)v152[8];
                    v159 = &v158[*((unsigned int *)v152 + 19)];
                    if ( v158 == v159 )
                      goto LABEL_190;
                    while ( v122 != *v158 )
                    {
                      if ( v159 == ++v158 )
                        goto LABEL_190;
                    }
                    goto LABEL_171;
                  }
                  v293 = v123;
                  v170 = sub_C8CA60(v153, v122);
                  v123 = v293;
                  if ( !v170 )
                    break;
LABEL_171:
                  if ( v151 == --v156 )
                    goto LABEL_172;
                }
                v157 = *v156;
LABEL_190:
                v171 = *v151;
                *v151 = v157;
                *v156 = v171;
LABEL_191:
                v123 = (__int64)++v151;
                if ( v149 == v151 )
                {
LABEL_192:
                  v135 = v152;
                  v39 = v299;
                  v150 = (_QWORD *)v123;
                  v160 = *(_BYTE **)(v307 + 8);
                  goto LABEL_173;
                }
              }
              else
              {
                v227 = sub_C8CA60(v153, v122);
                v123 = (__int64)v151;
                if ( v227 )
                  goto LABEL_164;
                v123 = (__int64)++v151;
                if ( v149 == v151 )
                  goto LABEL_192;
              }
            }
          }
          v160 = *(_BYTE **)v148;
LABEL_173:
          v331[0] = (__int64)&v331[2];
          v331[1] = 0x800000000LL;
          v161 = (v160 - (_BYTE *)v150) >> 3;
          if ( (unsigned __int64)(v160 - (_BYTE *)v150) > 0x40 )
          {
            v122 = (__int64)&v331[2];
            v302 = v135;
            sub_C8D5F0((__int64)v331, &v331[2], (v160 - (_BYTE *)v150) >> 3, 8u, v123, v124);
            v162 = (__int64 *)v331[0];
            v163 = v331[1];
            v135 = v302;
            v164 = (__int64 *)(v331[0] + 8LL * LODWORD(v331[1]));
          }
          else
          {
            v162 = &v331[2];
            v163 = 0;
            v164 = &v331[2];
          }
          if ( v160 != (_BYTE *)v150 )
          {
            v122 = (__int64)v150;
            v300 = v135;
            memmove(v164, v150, v160 - (_BYTE *)v150);
            v162 = (__int64 *)v331[0];
            v163 = v331[1];
            v135 = v300;
          }
          LODWORD(v331[1]) = v161 + v163;
          v165 = (unsigned int)(v161 + v163);
          if ( v150 != *(_QWORD **)(v307 + 8) )
          {
            *(_QWORD *)(v307 + 8) = v150;
            v165 = LODWORD(v331[1]);
          }
          v301 = &v162[v165];
          if ( v162 == v301 )
            goto LABEL_205;
          v166 = v135;
          v308 = v162;
          v289 = (__int64)(v135 + 1);
          while ( 1 )
          {
            v167 = *v308;
            v168 = *(__int64 **)(*v308 + 32);
            v169 = *v168;
            if ( v296 == *v168 )
              break;
            v312 = (_QWORD *)*v308;
            *(_QWORD *)v167 = v166;
            v122 = v166[2];
            if ( v122 == v166[3] )
            {
              sub_D4C7F0(v289, (_BYTE *)v122, &v312);
            }
            else
            {
              if ( v122 )
              {
                *(_QWORD *)v122 = v312;
                v122 = v166[2];
              }
              v122 += 8;
              v166[2] = v122;
            }
LABEL_184:
            if ( v301 == ++v308 )
            {
              v301 = (__int64 *)v331[0];
LABEL_205:
              if ( v301 != &v331[2] )
                _libc_free((unsigned __int64)v301);
              nullsub_188();
              if ( v305 )
                nullsub_188();
              goto LABEL_209;
            }
          }
          v294 = *(__int64 **)(v167 + 40);
          if ( v168 == v294 )
          {
LABEL_295:
            v213 = *(_QWORD ***)(v167 + 8);
            v214 = *(_QWORD *)(v167 + 24);
            *(_QWORD *)(v167 + 8) = 0;
            v215 = *(_QWORD ***)(v167 + 16);
            *(_QWORD *)(v167 + 24) = 0;
            v295 = (unsigned __int64)v213;
            v288 = v214;
            *(_QWORD *)(v167 + 16) = 0;
            if ( v213 == v215 )
              goto LABEL_397;
            v216 = v213;
            do
            {
              while ( 1 )
              {
                v312 = *v216;
                *v312 = v166;
                v122 = v166[2];
                if ( v122 != v166[3] )
                  break;
                ++v216;
                sub_D4C7F0(v289, (_BYTE *)v122, &v312);
                if ( v215 == v216 )
                  goto LABEL_302;
              }
              if ( v122 )
              {
                *(_QWORD *)v122 = v312;
                v122 = v166[2];
              }
              v122 += 8;
              ++v216;
              v166[2] = v122;
            }
            while ( v215 != v216 );
LABEL_302:
            v217 = *(__int64 **)(v167 + 16);
            if ( *(__int64 **)(v167 + 8) == v217 )
            {
LABEL_397:
              *(_BYTE *)(v167 + 152) = 1;
            }
            else
            {
              v218 = *(__int64 **)(v167 + 8);
              do
              {
                v219 = *v218++;
                sub_D47BB0(v219, v122);
              }
              while ( v217 != v218 );
              *(_BYTE *)(v167 + 152) = 1;
              v220 = *(_QWORD *)(v167 + 8);
              if ( *(_QWORD *)(v167 + 16) != v220 )
                *(_QWORD *)(v167 + 16) = v220;
            }
            v221 = *(_QWORD *)(v167 + 32);
            if ( v221 != *(_QWORD *)(v167 + 40) )
              *(_QWORD *)(v167 + 40) = v221;
            ++*(_QWORD *)(v167 + 56);
            if ( *(_BYTE *)(v167 + 84) )
            {
              *(_QWORD *)v167 = 0;
            }
            else
            {
              v222 = 4 * (*(_DWORD *)(v167 + 76) - *(_DWORD *)(v167 + 80));
              v223 = *(unsigned int *)(v167 + 72);
              if ( v222 < 0x20 )
                v222 = 32;
              if ( (unsigned int)v223 > v222 )
              {
                sub_C8C990(v167 + 56, v122);
              }
              else
              {
                v122 = 0xFFFFFFFFLL;
                memset(*(void **)(v167 + 64), -1, 8 * v223);
              }
              v224 = *(_BYTE *)(v167 + 84);
              *(_QWORD *)v167 = 0;
              if ( !v224 )
                _libc_free(*(_QWORD *)(v167 + 64));
            }
            v225 = *(_QWORD *)(v167 + 32);
            if ( v225 )
            {
              v122 = *(_QWORD *)(v167 + 48) - v225;
              j_j___libc_free_0(v225);
            }
            v226 = *(_QWORD *)(v167 + 8);
            if ( v226 )
            {
              v122 = *(_QWORD *)(v167 + 24) - v226;
              j_j___libc_free_0(v226);
            }
            if ( v295 )
            {
              v122 = v288 - v295;
              j_j___libc_free_0(v295);
            }
            goto LABEL_184;
          }
          v287 = v166;
          v206 = *(__int64 **)(*v308 + 32);
          while ( 2 )
          {
            v122 = *(unsigned int *)(a3 + 24);
            v207 = *(_QWORD *)(a3 + 8);
            if ( !(_DWORD)v122 )
              goto LABEL_287;
            v208 = v122 - 1;
            v209 = ((unsigned int)v169 >> 9) ^ ((unsigned int)v169 >> 4);
            v210 = (v122 - 1) & v209;
            v211 = (_QWORD *)(v207 + 16LL * v210);
            v212 = *v211;
            if ( v169 == *v211 )
            {
              if ( v211[1] == v167 )
              {
LABEL_292:
                v211[1] = v287;
LABEL_293:
                if ( v294 == ++v206 )
                {
LABEL_294:
                  v166 = v287;
                  goto LABEL_295;
                }
LABEL_288:
                v169 = *v206;
                continue;
              }
LABEL_287:
              if ( v294 == ++v206 )
                goto LABEL_294;
              goto LABEL_288;
            }
            break;
          }
          v239 = *v211;
          v240 = (v122 - 1) & (((unsigned int)v169 >> 9) ^ ((unsigned int)v169 >> 4));
          for ( m = 1; ; m = v285 )
          {
            if ( v239 == -4096 )
              goto LABEL_287;
            v240 = v208 & (m + v240);
            v285 = m + 1;
            v242 = (__int64 *)(v207 + 16LL * v240);
            v239 = *v242;
            if ( v169 == *v242 )
              break;
          }
          if ( v167 != v242[1] )
            goto LABEL_287;
          v243 = 1;
          v244 = 0;
          while ( v212 != -4096 )
          {
            if ( !v244 && v212 == -8192 )
              v244 = v211;
            v210 = v208 & (v243 + v210);
            v211 = (_QWORD *)(v207 + 16LL * v210);
            v212 = *v211;
            if ( v169 == *v211 )
              goto LABEL_292;
            ++v243;
          }
          v245 = *(_DWORD *)(a3 + 16);
          v246 = 2 * v122;
          if ( v244 )
            v211 = v244;
          ++*(_QWORD *)a3;
          v247 = v245 + 1;
          if ( 4 * (v245 + 1) >= (unsigned int)(3 * v122) )
          {
            v122 = v246;
            sub_D4F150(a3, v246);
            v264 = *(_DWORD *)(a3 + 24);
            if ( !v264 )
              goto LABEL_468;
            v265 = v264 - 1;
            v266 = *(_QWORD *)(a3 + 8);
            v267 = v265 & v209;
            v247 = *(_DWORD *)(a3 + 16) + 1;
            v211 = (_QWORD *)(v266 + 16LL * v267);
            v268 = *v211;
            if ( v169 != *v211 )
            {
              v122 = 1;
              v269 = 0;
              while ( v268 != -4096 )
              {
                if ( v268 == -8192 && !v269 )
                  v269 = v211;
                v282 = v122 + 1;
                v122 = v265 & (v267 + (unsigned int)v122);
                v267 = v122;
                v211 = (_QWORD *)(v266 + 16LL * (unsigned int)v122);
                v268 = *v211;
                if ( v169 == *v211 )
                  goto LABEL_349;
                v122 = v282;
              }
LABEL_394:
              if ( v269 )
                v211 = v269;
            }
          }
          else if ( (int)v122 - *(_DWORD *)(a3 + 20) - v247 <= (unsigned int)v122 >> 3 )
          {
            sub_D4F150(a3, v122);
            v270 = *(_DWORD *)(a3 + 24);
            if ( !v270 )
            {
LABEL_468:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v271 = v270 - 1;
            v272 = *(_QWORD *)(a3 + 8);
            v122 = 1;
            v273 = v271 & v209;
            v247 = *(_DWORD *)(a3 + 16) + 1;
            v269 = 0;
            v211 = (_QWORD *)(v272 + 16LL * v273);
            v274 = *v211;
            if ( v169 != *v211 )
            {
              while ( v274 != -4096 )
              {
                if ( !v269 && v274 == -8192 )
                  v269 = v211;
                v275 = v122 + 1;
                v122 = v271 & (v273 + (unsigned int)v122);
                v273 = v122;
                v211 = (_QWORD *)(v272 + 16LL * (unsigned int)v122);
                v274 = *v211;
                if ( v169 == *v211 )
                  goto LABEL_349;
                v122 = v275;
              }
              goto LABEL_394;
            }
          }
LABEL_349:
          *(_DWORD *)(a3 + 16) = v247;
          if ( *v211 != -4096 )
            --*(_DWORD *)(a3 + 20);
          *v211 = v169;
          v248 = v211 + 1;
          *v248 = 0;
          *v248 = v287;
          goto LABEL_293;
        }
LABEL_209:
        v174 = v328;
        v175 = &v328[(unsigned int)v329];
        if ( v328 != v175 )
        {
          do
          {
            v122 = *v174++;
            sub_E3BB90(a1, v122, v39);
          }
          while ( v175 != v174 );
          v175 = v328;
        }
        v176 = *(_DWORD *)(v39 + 20);
        v177 = *v175;
        v178 = 0;
        *(_DWORD *)(v39 + 16) = 0;
        if ( !v176 )
        {
          v122 = v39 + 24;
          sub_C8D5F0(v39 + 8, (const void *)(v39 + 24), 1u, 8u, v123, v124);
          v178 = 8LL * *(unsigned int *)(v39 + 16);
        }
        *(_QWORD *)(*(_QWORD *)(v39 + 8) + v178) = v177;
        ++*(_DWORD *)(v39 + 16);
        *(_DWORD *)(v39 + 184) = 0;
        nullsub_317();
        if ( *(_QWORD *)v39 )
          nullsub_317();
        sub_FFCE90((__int64)v304, v122, v179, v180, v181, v182);
        sub_FFD870((__int64)v304, v122, v183, v184, v185, v186);
        sub_FFBC40((__int64)v304, v122);
        v187 = v383;
        v188 = v382;
        if ( v383 != v382 )
        {
          do
          {
            v189 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v188[7];
            *v188 = &unk_49E5048;
            if ( v189 )
              v189(v188 + 5, v188 + 5, 3);
            *v188 = &unk_49DB368;
            v190 = v188[3];
            if ( v190 != 0 && v190 != -4096 && v190 != -8192 )
              sub_BD60C0(v188 + 1);
            v188 += 9;
          }
          while ( v187 != v188 );
          v188 = v382;
        }
        if ( v188 )
          j_j___libc_free_0((unsigned __int64)v188);
        if ( !v379 )
          _libc_free((unsigned __int64)v376);
        if ( v362 != &v364 )
          _libc_free((unsigned __int64)v362);
        if ( v323 != (unsigned __int64 *)&v325 )
          _libc_free((unsigned __int64)v323);
        sub_C7D6A0(v320, 8LL * (unsigned int)v322, 8);
        if ( v328 != (__int64 *)v330 )
          _libc_free((unsigned __int64)v328);
        if ( v317 != &v319 )
          _libc_free((unsigned __int64)v317);
        sub_C7D6A0((__int64)v314, 8LL * (unsigned int)v316, 8);
        if ( v325 != (__m128i *)v327 )
          _libc_free((unsigned __int64)v325);
        v36 = v339;
        v3 = 1;
LABEL_239:
        while ( 1 )
        {
          v191 = *(_QWORD *)(v36 - 24);
          if ( *(_BYTE *)(v36 - 8) )
            break;
          v192 = *(__int64 **)(v191 + 32);
          *(_BYTE *)(v36 - 8) = 1;
          *(_QWORD *)(v36 - 16) = v192;
          if ( *(__int64 **)(v191 + 40) != v192 )
            goto LABEL_241;
LABEL_247:
          v339 -= 24LL;
          v37 = v338;
          v36 = v339;
          if ( v339 == v338 )
            goto LABEL_251;
        }
        while ( 1 )
        {
          while ( 1 )
          {
            v192 = *(__int64 **)(v36 - 16);
            if ( *(__int64 **)(v191 + 40) == v192 )
              goto LABEL_247;
LABEL_241:
            v193 = v192 + 1;
            *(_QWORD *)(v36 - 16) = v192 + 1;
            v194 = *v192;
            if ( v336 )
              break;
LABEL_249:
            sub_C8CC70((__int64)&v332, v194, (__int64)v193, v38, v29, v30);
            if ( v196 )
              goto LABEL_250;
          }
          v195 = v333;
          v38 = HIDWORD(v334);
          v193 = &v333[HIDWORD(v334)];
          if ( v333 == v193 )
            break;
          while ( v194 != *v195 )
          {
            if ( v193 == ++v195 )
              goto LABEL_277;
          }
        }
LABEL_277:
        if ( HIDWORD(v334) >= (unsigned int)v334 )
          goto LABEL_249;
        ++HIDWORD(v334);
        *v193 = v194;
        ++v332;
LABEL_250:
        v362 = (unsigned __int64 *)v194;
        LOBYTE(v364) = 0;
        sub_29D3BC0((__int64)&v338, v304);
        v37 = v338;
        v36 = v339;
LABEL_251:
        v35 = v345;
        v38 = v36 - v37;
      }
      while ( v36 - v37 != v346 - v345 );
LABEL_252:
      if ( v36 != v37 )
      {
        v197 = v35;
        do
        {
          v38 = *(_QWORD *)v197;
          if ( *(_QWORD *)v37 != *(_QWORD *)v197 )
            goto LABEL_38;
          v38 = *(unsigned __int8 *)(v37 + 16);
          if ( (_BYTE)v38 != *(_BYTE *)(v197 + 16) )
            goto LABEL_38;
          if ( (_BYTE)v38 )
          {
            v38 = *(_QWORD *)(v197 + 8);
            if ( *(_QWORD *)(v37 + 8) != v38 )
              goto LABEL_38;
          }
          v37 += 24LL;
          v197 += 24LL;
        }
        while ( v37 != v36 );
      }
      v4 = v304;
      if ( v35 )
        j_j___libc_free_0(v35);
      if ( !v343 )
        _libc_free(v342);
      if ( v338 )
        j_j___libc_free_0(v338);
      if ( !v336 )
        _libc_free((unsigned __int64)v333);
      if ( v359 )
        j_j___libc_free_0((unsigned __int64)v359);
      if ( !v357 )
        _libc_free(v356);
      if ( v352 )
        j_j___libc_free_0((unsigned __int64)v352);
      if ( !v350 )
        _libc_free(v349);
      if ( v286 == ++v291 )
        return v3;
    }
  }
  return 0;
}
