// Function: sub_331C5B0
// Address: 0x331c5b0
//
__int64 __fastcall sub_331C5B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 v3; // r12
  const __m128i *v4; // rax
  __int64 v5; // r14
  __m128i v6; // xmm1
  __int64 v7; // rcx
  unsigned __int64 v8; // r15
  unsigned __int32 v9; // ebx
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int16 v16; // r11
  unsigned __int8 v17; // al
  char v18; // r11
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // rax
  __int64 v23; // rdx
  __int16 v24; // r8
  __int16 v25; // cx
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned __int16 v28; // ax
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // r9
  unsigned __int16 *v32; // rsi
  __int64 v33; // r10
  __int64 v34; // rsi
  char v35; // al
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r15
  __int16 v42; // ax
  __int64 v43; // rax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rdi
  __int64 (*v47)(); // rax
  __int64 v48; // rsi
  _QWORD *v49; // r14
  __int64 v50; // r13
  __int64 v51; // rbx
  __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  __int64 v54; // r8
  _QWORD *v55; // r9
  unsigned __int16 v56; // dx
  __int64 v57; // rdx
  bool v58; // al
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rcx
  unsigned __int64 v62; // r8
  __int64 v63; // rax
  __int64 *v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rbx
  __int16 v67; // ax
  __int64 v68; // rdx
  bool v69; // al
  __int64 v70; // rax
  unsigned __int64 *v71; // rax
  unsigned int v72; // edi
  unsigned __int64 v73; // r11
  unsigned __int64 v74; // rbx
  unsigned __int64 v75; // r15
  __int64 v76; // r12
  __int64 v77; // r14
  int v78; // eax
  int v79; // edx
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 *v82; // rax
  int v83; // eax
  __int64 v84; // r10
  int v85; // r11d
  int v86; // edx
  __int64 v87; // rsi
  __int64 v88; // r9
  __int64 v89; // r8
  _QWORD *v90; // rcx
  __int64 v91; // rax
  __int64 v92; // r9
  __int64 v93; // rsi
  unsigned int v94; // edx
  __int64 v95; // rax
  __int64 v96; // r8
  unsigned __int64 v97; // rdx
  __int64 *v98; // rax
  _DWORD *v99; // rdi
  __int64 v100; // r8
  _QWORD *v101; // rax
  __int64 v102; // r8
  __int64 v103; // rsi
  unsigned __int64 v104; // rdx
  __int64 v105; // r9
  unsigned __int64 v106; // rax
  __int64 *v107; // rdx
  __int64 *v108; // rbx
  _QWORD *v109; // r12
  __int64 v110; // r13
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rdx
  __int16 v114; // ax
  __int64 v115; // rdx
  int v116; // eax
  __int64 v117; // rcx
  unsigned __int16 v118; // r10
  __int64 v119; // rax
  char v120; // al
  _QWORD *v121; // r9
  unsigned int v122; // eax
  __int64 v123; // r9
  __int64 v124; // r8
  unsigned __int64 v125; // rdx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rax
  __int16 v129; // ax
  int v130; // r15d
  __int64 v131; // rax
  _QWORD *v132; // r14
  __int64 v133; // r13
  __int64 v134; // r15
  __int64 v135; // r10
  __int64 v136; // r11
  int v137; // eax
  unsigned __int64 v138; // rdx
  __int64 *v139; // rbx
  __int64 *v140; // r12
  __int64 v141; // rdi
  unsigned int v142; // ecx
  __int64 v143; // rsi
  _QWORD *v144; // rdi
  __int64 *v145; // rbx
  __int64 *v146; // r12
  __int64 v147; // rsi
  __int64 v148; // rdi
  int v149; // edx
  __int64 v150; // r8
  int v151; // edx
  __int64 v152; // r8
  int v153; // r9d
  __int64 v154; // r13
  __int64 v155; // rax
  __int64 v156; // r14
  __int64 v157; // r15
  __int64 v158; // rax
  __int64 v159; // rsi
  __int64 v160; // r14
  unsigned int v161; // ebx
  __int64 v162; // rax
  unsigned __int16 v163; // cx
  __int64 v164; // rsi
  __int64 v165; // rax
  __int64 v166; // r15
  unsigned int v167; // edx
  __int64 v168; // r14
  unsigned int v169; // eax
  __int64 v170; // rt0
  int v171; // ebx
  __int64 v172; // rax
  __int64 j; // rax
  char v174; // al
  __int64 v175; // r13
  __int64 v176; // r14
  int v177; // eax
  __int64 *v178; // rdx
  __int64 v179; // rsi
  _QWORD *v180; // rcx
  __int64 v181; // rax
  unsigned int v182; // edx
  bool v183; // al
  unsigned __int16 *v184; // rax
  __int64 v185; // r14
  int v186; // ecx
  __int64 v187; // r15
  int v188; // r14d
  int v189; // edx
  int v190; // r15d
  __int64 v191; // r13
  __int64 v192; // r8
  __int64 v193; // rax
  __int64 v194; // rdx
  __int16 v195; // ax
  unsigned __int64 v196; // rax
  __int64 v197; // rdx
  __int64 v198; // rax
  __int64 v199; // rdx
  __int64 v200; // rcx
  __int64 v201; // rax
  __int64 v202; // rdx
  __int64 v203; // rcx
  unsigned int v204; // r15d
  __int64 v205; // rax
  _QWORD *v206; // rdi
  __int64 v207; // rdx
  unsigned __int64 v208; // rdx
  unsigned __int64 *v209; // rax
  unsigned int v210; // esi
  __int64 v211; // rdx
  __int64 v212; // rdx
  unsigned __int64 v213; // rdi
  __int64 v214; // rax
  __int64 v215; // r15
  __int64 v216; // rax
  __int64 v217; // r8
  __int64 v218; // rax
  int v219; // ecx
  __int64 v220; // rax
  unsigned __int64 v221; // rdx
  __int64 v222; // rax
  __int64 v223; // rcx
  unsigned __int64 v224; // rdx
  __int64 v225; // rax
  _DWORD *v226; // rsi
  __int64 v227; // rdx
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 v230; // rcx
  __int16 v231; // dx
  __int64 v232; // rcx
  __int64 v233; // rax
  __int128 v234; // rax
  __m128i v235; // rax
  __m128i *v236; // rdx
  __int16 v237; // ax
  __m128i v238; // rax
  unsigned __int64 v239; // rax
  __int64 v240; // rdx
  int v241; // ecx
  __int64 v242; // rax
  __int8 v243; // cl
  __int64 v244; // rax
  __int64 v245; // rax
  __int64 v246; // r15
  __int64 v247; // rax
  __int64 v248; // rax
  __int64 v249; // rdx
  __int64 v250; // rdx
  __int64 v251; // r15
  __int64 v252; // rax
  __int64 v253; // r8
  __int64 v254; // rax
  __int64 v255; // rax
  unsigned __int64 v256; // rdx
  __int64 v257; // rdx
  unsigned int *v258; // rax
  unsigned __int16 *v259; // rax
  __int64 v260; // rsi
  __int64 v261; // rdx
  unsigned int *v262; // rax
  unsigned __int16 *v263; // rax
  __int64 v264; // rcx
  __int64 v265; // r8
  __int64 v266; // rax
  __int16 v267; // bx
  __m128i v268; // xmm4
  __int64 v269; // rax
  int v270; // r9d
  int v271; // edx
  int v272; // edx
  __int64 v273; // rdi
  int v274; // edx
  _QWORD *v275; // rax
  __int64 v276; // r15
  __int64 v277; // r14
  int v278; // eax
  __int64 v279; // rdi
  int v280; // edx
  __int64 v281; // r14
  unsigned int v282; // edx
  __int64 v283; // rsi
  unsigned __int64 v284; // r15
  __int64 v285; // r8
  unsigned __int64 *v286; // rdi
  __int64 v287; // rcx
  __int64 v288; // rcx
  __int64 v289; // rcx
  unsigned __int16 *v290; // rax
  unsigned __int64 v291; // rsi
  _BYTE **v292; // rax
  __int64 v293; // rsi
  _BYTE *v294; // rdx
  __int64 v295; // rcx
  unsigned __int64 v296; // rax
  __int64 v297; // rax
  __int64 v298; // rdx
  __int64 v299; // rdi
  __int64 v300; // rax
  __m128i v301; // rax
  __int64 v302; // rsi
  _QWORD *v303; // [rsp+8h] [rbp-498h]
  _QWORD *v304; // [rsp+8h] [rbp-498h]
  __int64 v305; // [rsp+10h] [rbp-490h]
  unsigned __int64 v306; // [rsp+18h] [rbp-488h]
  int v307; // [rsp+18h] [rbp-488h]
  unsigned __int64 v308; // [rsp+18h] [rbp-488h]
  __int64 v309; // [rsp+20h] [rbp-480h]
  __int64 v310; // [rsp+20h] [rbp-480h]
  int v311; // [rsp+20h] [rbp-480h]
  __int64 v312; // [rsp+28h] [rbp-478h]
  __int64 v313; // [rsp+30h] [rbp-470h]
  _QWORD *v314; // [rsp+30h] [rbp-470h]
  _QWORD *v315; // [rsp+30h] [rbp-470h]
  unsigned __int64 v316; // [rsp+30h] [rbp-470h]
  unsigned __int32 v317; // [rsp+38h] [rbp-468h]
  int v318; // [rsp+38h] [rbp-468h]
  int v319; // [rsp+38h] [rbp-468h]
  __int64 v320; // [rsp+40h] [rbp-460h]
  __int64 v321; // [rsp+40h] [rbp-460h]
  __int64 v322; // [rsp+40h] [rbp-460h]
  unsigned __int64 v323; // [rsp+60h] [rbp-440h]
  unsigned __int64 v324; // [rsp+60h] [rbp-440h]
  int v325; // [rsp+60h] [rbp-440h]
  unsigned int v326; // [rsp+60h] [rbp-440h]
  __int64 v327; // [rsp+68h] [rbp-438h]
  __int64 v328; // [rsp+68h] [rbp-438h]
  char v329; // [rsp+78h] [rbp-428h]
  unsigned __int32 v330; // [rsp+78h] [rbp-428h]
  unsigned int v331; // [rsp+78h] [rbp-428h]
  unsigned __int32 v332; // [rsp+78h] [rbp-428h]
  __int64 v333; // [rsp+80h] [rbp-420h]
  __int64 i; // [rsp+80h] [rbp-420h]
  _QWORD *v335; // [rsp+80h] [rbp-420h]
  _QWORD *v336; // [rsp+80h] [rbp-420h]
  unsigned __int64 v337; // [rsp+80h] [rbp-420h]
  unsigned int v338; // [rsp+80h] [rbp-420h]
  __int64 v339; // [rsp+80h] [rbp-420h]
  __int64 v340; // [rsp+88h] [rbp-418h]
  __int64 v341; // [rsp+90h] [rbp-410h]
  unsigned __int64 v342; // [rsp+90h] [rbp-410h]
  __int64 v343; // [rsp+90h] [rbp-410h]
  unsigned __int64 v344; // [rsp+90h] [rbp-410h]
  __int64 v345; // [rsp+90h] [rbp-410h]
  int v346; // [rsp+90h] [rbp-410h]
  __int64 v347; // [rsp+90h] [rbp-410h]
  __int16 v348; // [rsp+90h] [rbp-410h]
  __int64 v349; // [rsp+90h] [rbp-410h]
  __int64 v350; // [rsp+90h] [rbp-410h]
  __int64 v351; // [rsp+90h] [rbp-410h]
  __int64 v352; // [rsp+98h] [rbp-408h]
  __int64 v353; // [rsp+A0h] [rbp-400h]
  __int64 v354; // [rsp+A0h] [rbp-400h]
  __int64 v355; // [rsp+A0h] [rbp-400h]
  __int64 *v356; // [rsp+A0h] [rbp-400h]
  __int64 v357; // [rsp+A0h] [rbp-400h]
  __int64 v358; // [rsp+A0h] [rbp-400h]
  int v359; // [rsp+A0h] [rbp-400h]
  unsigned int v360; // [rsp+A0h] [rbp-400h]
  __int16 v361; // [rsp+A8h] [rbp-3F8h]
  __int64 v362; // [rsp+A8h] [rbp-3F8h]
  unsigned __int16 v363; // [rsp+A8h] [rbp-3F8h]
  __int64 v364; // [rsp+A8h] [rbp-3F8h]
  char v365; // [rsp+A8h] [rbp-3F8h]
  int v366; // [rsp+A8h] [rbp-3F8h]
  __int64 v367; // [rsp+A8h] [rbp-3F8h]
  unsigned __int8 v368; // [rsp+B0h] [rbp-3F0h]
  int v369; // [rsp+B0h] [rbp-3F0h]
  __int64 (__fastcall *v370)(__int64); // [rsp+B0h] [rbp-3F0h]
  __int64 v371; // [rsp+B0h] [rbp-3F0h]
  __int64 v372; // [rsp+B0h] [rbp-3F0h]
  unsigned __int32 v373; // [rsp+B0h] [rbp-3F0h]
  unsigned __int32 v374; // [rsp+B0h] [rbp-3F0h]
  __int64 v375; // [rsp+B8h] [rbp-3E8h]
  unsigned __int64 v376; // [rsp+B8h] [rbp-3E8h]
  int v377; // [rsp+B8h] [rbp-3E8h]
  int v378; // [rsp+B8h] [rbp-3E8h]
  unsigned int v379; // [rsp+C0h] [rbp-3E0h]
  __int64 v380; // [rsp+C0h] [rbp-3E0h]
  __int64 v381; // [rsp+C0h] [rbp-3E0h]
  __int64 v382; // [rsp+C0h] [rbp-3E0h]
  __int64 v383; // [rsp+C0h] [rbp-3E0h]
  __int64 v384; // [rsp+C0h] [rbp-3E0h]
  int v385; // [rsp+C0h] [rbp-3E0h]
  __int64 v386; // [rsp+C0h] [rbp-3E0h]
  __int64 v387; // [rsp+C0h] [rbp-3E0h]
  int v388; // [rsp+C0h] [rbp-3E0h]
  __int64 v389; // [rsp+C0h] [rbp-3E0h]
  int v390; // [rsp+C0h] [rbp-3E0h]
  unsigned __int64 v391; // [rsp+C0h] [rbp-3E0h]
  char v392; // [rsp+C0h] [rbp-3E0h]
  __int64 v393; // [rsp+C8h] [rbp-3D8h]
  __int64 v394; // [rsp+D0h] [rbp-3D0h]
  __int64 v395; // [rsp+D0h] [rbp-3D0h]
  unsigned int v396; // [rsp+D0h] [rbp-3D0h]
  int v397; // [rsp+D0h] [rbp-3D0h]
  __int64 v398; // [rsp+F0h] [rbp-3B0h]
  __m128i v399; // [rsp+110h] [rbp-390h] BYREF
  __int64 v400; // [rsp+120h] [rbp-380h]
  __int64 v401; // [rsp+128h] [rbp-378h]
  __int64 v402; // [rsp+130h] [rbp-370h]
  __int64 v403; // [rsp+138h] [rbp-368h]
  __int64 v404; // [rsp+140h] [rbp-360h] BYREF
  int v405; // [rsp+148h] [rbp-358h]
  __int64 v406; // [rsp+150h] [rbp-350h] BYREF
  __int64 v407; // [rsp+158h] [rbp-348h]
  __int64 v408[8]; // [rsp+160h] [rbp-340h] BYREF
  _QWORD v409[8]; // [rsp+1A0h] [rbp-300h] BYREF
  __int64 v410; // [rsp+1E0h] [rbp-2C0h] BYREF
  __int64 v411; // [rsp+1E8h] [rbp-2B8h]
  _BYTE *v412; // [rsp+220h] [rbp-280h] BYREF
  __int64 v413; // [rsp+228h] [rbp-278h]
  _BYTE v414[64]; // [rsp+230h] [rbp-270h] BYREF
  __int64 v415; // [rsp+270h] [rbp-230h] BYREF
  _QWORD *v416; // [rsp+278h] [rbp-228h] BYREF
  __int64 v417; // [rsp+280h] [rbp-220h]
  _QWORD v418[9]; // [rsp+288h] [rbp-218h] BYREF
  __m128i v419; // [rsp+2D0h] [rbp-1D0h] BYREF
  __int64 v420; // [rsp+2E0h] [rbp-1C0h]
  __int64 *v421; // [rsp+2E8h] [rbp-1B8h]
  __int64 v422; // [rsp+2F0h] [rbp-1B0h]
  _BYTE v423[32]; // [rsp+2F8h] [rbp-1A8h] BYREF
  _QWORD *v424; // [rsp+318h] [rbp-188h]
  __int64 v425; // [rsp+320h] [rbp-180h]
  _QWORD v426[3]; // [rsp+328h] [rbp-178h] BYREF
  __int128 src; // [rsp+340h] [rbp-160h] BYREF
  __int64 v428; // [rsp+350h] [rbp-150h] BYREF
  _BYTE *v429; // [rsp+358h] [rbp-148h] BYREF
  __int64 v430; // [rsp+360h] [rbp-140h]
  _BYTE v432[136]; // [rsp+3D0h] [rbp-D0h] BYREF
  __int64 v433; // [rsp+458h] [rbp-48h]
  __m128i *v434; // [rsp+460h] [rbp-40h]

  v2 = a1;
  v3 = a2;
  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4[2].m128i_i64[1];
  v6 = _mm_loadu_si128(v4 + 5);
  v7 = v4[5].m128i_i64[0];
  v8 = v4[3].m128i_u64[0];
  v379 = v4[3].m128i_u32[0];
  v9 = v4[5].m128i_u32[2];
  v10 = *(_DWORD *)(v5 + 24);
  v399 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v375 = v7;
  if ( v10 == 234 )
  {
    if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
      goto LABEL_3;
    v25 = *(_WORD *)(a2 + 32);
    if ( (v25 & 0x380) != 0 )
      goto LABEL_4;
    v26 = a1[1];
    v27 = *(_QWORD *)(**(_QWORD **)(v5 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v5 + 40) + 8LL);
    v28 = *(_WORD *)v27;
    v29 = *(_QWORD *)(v27 + 8);
    if ( *((_BYTE *)v2 + 33) || (v30 = *(_QWORD *)(a2 + 112), (*(_BYTE *)(v30 + 37) & 0xF) != 0) || (v25 & 8) != 0 )
    {
      v36 = 1;
      if ( v28 != 1 && (!v28 || (v36 = v28, !*(_QWORD *)(v26 + 8LL * v28 + 112))) || *(_BYTE *)(v26 + 500 * v36 + 6713) )
      {
LABEL_3:
        if ( !*((_DWORD *)v2 + 7) || (*(_WORD *)(v3 + 32) & 0x380) != 0 )
          goto LABEL_4;
        goto LABEL_10;
      }
      v30 = *(_QWORD *)(a2 + 112);
    }
    v31 = *v2;
    v32 = (unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16LL * v379);
    v33 = *((_QWORD *)v32 + 1);
    v370 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v26 + 256LL);
    v34 = *v32;
    if ( v370 == sub_2FE2F90 )
      v35 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v26 + 248LL))(
              v26,
              v34,
              v33,
              v28,
              v29,
              v31,
              v30);
    else
      v35 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64, __int64, __int64))v370)(
              v26,
              v34,
              v33,
              v28,
              v29,
              v31,
              v30);
    if ( v35 )
    {
      v48 = *(_QWORD *)(v3 + 80);
      v49 = *(_QWORD **)(v5 + 40);
      v50 = *v2;
      v51 = *(_QWORD *)(v3 + 112);
      *(_QWORD *)v432 = v48;
      if ( v48 )
        sub_B96E90((__int64)v432, v48, 1);
      *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
      result = sub_33F3F90(
                 v50,
                 v399.m128i_i32[0],
                 v399.m128i_i32[2],
                 (unsigned int)v432,
                 *v49,
                 v49[1],
                 v6.m128i_i64[0],
                 v6.m128i_i64[1],
                 v51);
      if ( *(_QWORD *)v432 )
      {
        v394 = result;
        sub_B91220((__int64)v432, *(__int64 *)v432);
        return v394;
      }
      return result;
    }
    v10 = *(_DWORD *)(v5 + 24);
  }
  if ( v10 != 51 )
    goto LABEL_3;
  if ( (*(_WORD *)(v3 + 32) & 0x380) == 0 )
  {
    if ( (*(_BYTE *)(v3 + 32) & 8) == 0 )
      return v399.m128i_i64[0];
    if ( *((_DWORD *)v2 + 7) )
    {
LABEL_10:
      if ( (*(_BYTE *)(*(_QWORD *)(v3 + 112) + 37LL) & 0xF) == 0 )
      {
        v16 = sub_33E0440(*v2, v6.m128i_i64[0], v6.m128i_i64[1]);
        if ( HIBYTE(v16) )
        {
          v368 = v16;
          v17 = sub_2EAC4F0(*(_QWORD *)(v3 + 112));
          v18 = v368;
          if ( v17 < v368 )
          {
            v19 = *(_QWORD *)(v3 + 112);
            if ( (*(_QWORD *)(v19 + 8) & ~(-1LL << v368)) == 0 )
            {
              v20 = *(_QWORD *)(v3 + 80);
              v21 = *v2;
              *(__m128i *)v432 = _mm_loadu_si128((const __m128i *)(v19 + 40));
              v22 = *(unsigned __int16 *)(v3 + 96);
              v23 = *(_QWORD *)(v3 + 104);
              *(__m128i *)&v432[16] = _mm_loadu_si128((const __m128i *)(v19 + 56));
              v24 = *(_WORD *)(v19 + 32);
              *(_QWORD *)&src = v20;
              if ( v20 )
              {
                v329 = v368;
                v333 = v22;
                v340 = v23;
                v341 = v19;
                v361 = v24;
                v369 = v21;
                sub_325F5D0((__int64 *)&src);
                v18 = v329;
                v22 = v333;
                v23 = v340;
                v19 = v341;
                v24 = v361;
                LODWORD(v21) = v369;
              }
              DWORD2(src) = *(_DWORD *)(v3 + 72);
              v8 = v8 & 0xFFFFFFFF00000000LL | v379;
              sub_33F5040(
                v21,
                v399.m128i_i32[0],
                v399.m128i_i32[2],
                (unsigned int)&src,
                v5,
                v8,
                v6.m128i_i64[0],
                v6.m128i_i64[1],
                *(_OWORD *)v19,
                *(_QWORD *)(v19 + 16),
                v22,
                v23,
                v18,
                v24,
                (__int64)v432);
              if ( (_QWORD)src )
                sub_B91220((__int64)&src, src);
            }
          }
        }
      }
    }
  }
LABEL_4:
  result = sub_32BD840(v2, v3);
  if ( result )
    return result;
  v37 = v3;
  result = sub_3284560(v2, v3, v12, v13, v14, v15);
  if ( result )
    return result;
  if ( (*(_WORD *)(v3 + 32) & 0x380) != 0 )
    goto LABEL_32;
  if ( !*((_DWORD *)v2 + 7) || (v37 = v3, sub_33644B0(v408, v3, *v2, v38), !v408[0]) || *(_DWORD *)(v408[0] + 24) == 51 )
  {
    v112 = *(_QWORD *)(v3 + 40);
    goto LABEL_121;
  }
  v52 = *v2;
  v412 = v414;
  v413 = 0x800000000LL;
  v421 = (__int64 *)v423;
  v422 = 0x400000000LL;
  v434 = &v419;
  v424 = v426;
  v433 = 0;
  memset(v432, 0, sizeof(v432));
  v419 = 0u;
  v420 = 0;
  v425 = 0;
  v426[0] = 0;
  v426[1] = 1;
  sub_33644B0(v409, v3, v52, 0);
  if ( !v409[0] || *(_DWORD *)(v409[0] + 24LL) == 51 )
    goto LABEL_175;
  v56 = *(_WORD *)(v3 + 96);
  v54 = *(_QWORD *)(v3 + 104);
  LOWORD(v415) = v56;
  v416 = (_QWORD *)v54;
  if ( !v56 )
  {
    v353 = v54;
    v400 = sub_3007260((__int64)&v415);
    v401 = v57;
    if ( v400 )
    {
      LOWORD(src) = 0;
      *((_QWORD *)&src + 1) = v353;
      v58 = sub_3007100((__int64)&src);
      v54 = v353;
      if ( !v58 )
      {
        v411 = v353;
        LOWORD(v410) = 0;
        v402 = sub_3007260((__int64)&v410);
        v403 = v59;
        goto LABEL_71;
      }
    }
LABEL_175:
    v365 = 0;
    goto LABEL_176;
  }
  if ( v56 == 1 || (unsigned __int16)(v56 - 504) <= 7u )
    goto LABEL_431;
  v53 = *(_QWORD *)&byte_444C4A0[16 * v56 - 16];
  if ( !v53 || (unsigned __int16)(v56 - 176) <= 0x34u || v56 == 270 || (unsigned __int16)(v56 - 229) <= 0x1Fu )
    goto LABEL_175;
  v174 = byte_444C4A0[16 * v56 - 8];
  LOWORD(v410) = v56;
  v411 = v54;
  LOBYTE(v403) = v174;
  v402 = v53;
LABEL_71:
  v415 = v402;
  LOBYTE(v416) = v403;
  v62 = (unsigned __int64)(sub_CA1930(&v415) + 7) >> 3;
  v63 = (unsigned int)v433;
  if ( (_DWORD)v433 )
  {
    v337 = v62;
    *(_QWORD *)&src = v432;
    *((_QWORD *)&src + 1) = &v429;
    v428 = 0x400000000LL;
    sub_32B0350((__int64)&src, 0, v60, v61, v62, (__int64)&v429);
    v62 = v337;
    goto LABEL_260;
  }
  if ( HIDWORD(v433) == 8 )
  {
    HIDWORD(v428) = 4;
    v178 = (__int64 *)&v432[8];
    *((_QWORD *)&src + 1) = &v429;
    *(_QWORD *)&src = v432;
    do
    {
      if ( *v178 > 0 )
        break;
      v63 = (unsigned int)(v63 + 1);
      v178 += 2;
    }
    while ( (_DWORD)v63 != 8 );
    LODWORD(v428) = 1;
    v429 = v432;
    v430 = (v63 << 32) | 8;
LABEL_260:
    sub_331C500((__int64)&src, 0, v62);
    v55 = &v429;
    if ( *((_BYTE ***)&src + 1) != &v429 )
      _libc_free(*((unsigned __int64 *)&src + 1));
    goto LABEL_78;
  }
  if ( HIDWORD(v433) )
  {
    v64 = (__int64 *)&v432[8];
    do
    {
      if ( *v64 > 0 )
        break;
      LODWORD(v63) = v63 + 1;
      v64 += 2;
    }
    while ( HIDWORD(v433) != (_DWORD)v63 );
  }
  LODWORD(src) = v63;
  HIDWORD(v433) = sub_325EF20((__int64)v432, (unsigned int *)&src, HIDWORD(v433), 0, v62);
LABEL_78:
  v330 = v9;
  v323 = v8;
  for ( i = v3; ; i = v66 )
  {
    v65 = *(__int64 **)(i + 40);
    v66 = *v65;
    if ( *(_DWORD *)(*v65 + 24) != 299 )
      goto LABEL_84;
    v67 = *(_WORD *)(v66 + 96);
    v68 = *(_QWORD *)(v66 + 104);
    LOWORD(src) = v67;
    *((_QWORD *)&src + 1) = v68;
    if ( v67 )
      v69 = (unsigned __int16)(v67 - 176) <= 0x34u;
    else
      v69 = sub_3007100((__int64)&src);
    if ( v69 )
    {
      v9 = v330;
      v8 = v323;
      goto LABEL_175;
    }
    v70 = *(_QWORD *)(v66 + 56);
    if ( !v70
      || *(_QWORD *)(v70 + 32)
      || (*(_BYTE *)(*(_QWORD *)(v66 + 112) + 37LL) & 0xF) != 0
      || (v195 = *(_WORD *)(v66 + 32), (v195 & 8) != 0)
      || (v195 & 0x380) != 0
      || (sub_33644B0(&v410, v66, *v2, v53), !(unsigned __int8)sub_3364290(v409, &v410, *v2, &v406)) )
    {
LABEL_84:
      v9 = v330;
      v8 = v323;
      goto LABEL_85;
    }
    v196 = *(_QWORD *)(v66 + 104);
    LOWORD(v415) = *(_WORD *)(v66 + 96);
    v416 = (_QWORD *)v196;
    *(_QWORD *)&src = sub_2D5B750((unsigned __int16 *)&v415);
    *((_QWORD *)&src + 1) = v197;
    v198 = sub_CA1930(&src);
    v415 = (__int64)v432;
    v316 = (unsigned __int64)(v198 + 7) >> 3;
    v416 = v418;
    v417 = 0x400000000LL;
    if ( (_DWORD)v433 )
    {
      sub_32B0350((__int64)&v415, v406, v199, v200, v54, (__int64)v55);
      v201 = HIDWORD(v433);
    }
    else
    {
      v201 = HIDWORD(v433);
      v202 = 0;
      do
      {
        v203 = (unsigned int)v202;
        if ( HIDWORD(v433) == (_DWORD)v202 )
          break;
        ++v202;
        v54 = 16 * v202;
      }
      while ( v406 >= *(_QWORD *)(v432 + 16 * v202 + 6) );
      v418[0] = v432;
      LODWORD(v417) = 1;
      v418[1] = (v203 << 32) | HIDWORD(v433);
    }
    v204 = v433;
    *(_QWORD *)&src = v432;
    *((_QWORD *)&src + 1) = &v429;
    v428 = 0x400000000LL;
    v53 = (unsigned int)v201;
    if ( (_DWORD)v433 )
    {
      v205 = (unsigned int)v201 | (unsigned __int64)(v201 << 32);
      v429 = &v432[8];
    }
    else
    {
      v429 = v432;
      v205 = (unsigned int)v201 | (unsigned __int64)(v201 << 32);
    }
    v430 = v205;
    if ( (_DWORD)v417 && (v206 = v416, *((_DWORD *)v416 + 3) < *((_DWORD *)v416 + 2)) )
    {
      v207 = (unsigned int)v417;
      v291 = (unsigned __int64)&v416[2 * (unsigned int)v417 - 2];
      if ( *(_DWORD *)(v291 + 12) == HIDWORD(v430) && v429 == *(_BYTE **)v291 )
        goto LABEL_292;
    }
    else
    {
      if ( HIDWORD(v430) >= (unsigned int)v430 )
        goto LABEL_292;
      v206 = v416;
      v207 = (unsigned int)v417;
    }
    if ( *(_QWORD *)(v206[2 * v207 - 2] + 16LL * HIDWORD(v206[2 * v207 - 1])) < (signed __int64)(v406 + v316) )
    {
      v9 = v330;
      v8 = v323;
      goto LABEL_403;
    }
LABEL_292:
    *(_QWORD *)&src = v432;
    *((_QWORD *)&src + 1) = &v429;
    v428 = 0x400000000LL;
    if ( (_DWORD)v433 )
      break;
    v212 = (unsigned int)v417;
    v210 = 1;
    v292 = &v429;
    v429 = v432;
    v430 = v53;
    LODWORD(v428) = 1;
    if ( !(_DWORD)v417 )
      goto LABEL_415;
LABEL_299:
    v53 = (unsigned __int64)v416;
    if ( *((_DWORD *)v416 + 3) >= *((_DWORD *)v416 + 2) )
      goto LABEL_414;
    v213 = (unsigned __int64)&v416[2 * v212 - 2];
    v212 = 16LL * v210;
    v53 = *(unsigned int *)(v213 + 12);
    v214 = (__int64)v292 + v212 - 16;
    if ( (_DWORD)v53 != *(_DWORD *)(v214 + 12) || *(_QWORD *)v213 != *(_QWORD *)v214 )
    {
      if ( (_DWORD)v53 )
      {
LABEL_399:
        *(_DWORD *)(v213 + 12) = v53 - 1;
        goto LABEL_400;
      }
      goto LABEL_405;
    }
LABEL_302:
    if ( *((_BYTE ***)&src + 1) != &v429 )
      _libc_free(*((unsigned __int64 *)&src + 1));
    v215 = v406;
    v216 = (unsigned int)v433;
    v217 = v406 + v316;
    if ( (_DWORD)v433 )
    {
      v339 = v406 + v316;
      *(_QWORD *)&src = v432;
      *((_QWORD *)&src + 1) = &v429;
      v428 = 0x400000000LL;
      sub_32B0350((__int64)&src, v406, v212, v53, v217, (__int64)v55);
      v217 = v339;
    }
    else
    {
      if ( HIDWORD(v433) != 8 )
      {
        v218 = 0;
        do
        {
          v219 = v218;
          if ( HIDWORD(v433) == (_DWORD)v218 )
            break;
          ++v218;
        }
        while ( v406 >= *(_QWORD *)(v432 + 16 * v218 + 6) );
        LODWORD(src) = v219;
        HIDWORD(v433) = sub_325EF20((__int64)v432, (unsigned int *)&src, HIDWORD(v433), v406, v217);
        goto LABEL_310;
      }
      *(_QWORD *)&src = v432;
      v294 = v432;
      *((_QWORD *)&src + 1) = &v429;
      HIDWORD(v428) = 4;
      do
      {
        if ( v406 < *((_QWORD *)v294 + 1) )
          break;
        v216 = (unsigned int)(v216 + 1);
        v294 += 16;
      }
      while ( (_DWORD)v216 != 8 );
      v429 = v432;
      LODWORD(v428) = 1;
      v430 = (v216 << 32) | 8;
    }
    sub_331C500((__int64)&src, v215, v217);
    if ( *((_BYTE ***)&src + 1) != &v429 )
      _libc_free(*((unsigned __int64 *)&src + 1));
LABEL_310:
    v220 = (unsigned int)v413;
    v53 = HIDWORD(v413);
    v221 = (unsigned int)v413 + 1LL;
    if ( v221 > HIDWORD(v413) )
    {
      sub_C8D5F0((__int64)&v412, v414, v221, 8u, v54, (__int64)v55);
      v220 = (unsigned int)v413;
    }
    *(_QWORD *)&v412[8 * v220] = v66;
    LODWORD(v413) = v413 + 1;
    if ( v416 != v418 )
      _libc_free((unsigned __int64)v416);
  }
  v430 = v53;
  v429 = &v432[8];
  for ( LODWORD(v428) = 1; ; LODWORD(v428) = v428 + 1 )
  {
    v210 = v428;
    v292 = (_BYTE **)*((_QWORD *)&src + 1);
    v211 = (unsigned int)(v428 - 1);
    if ( v204 <= (unsigned int)v211 )
      break;
    v55 = (_QWORD *)((*(_QWORD *)(*(_QWORD *)(*((_QWORD *)&src + 1) + 16 * v211)
                                + 8LL * *(unsigned int *)(*((_QWORD *)&src + 1) + 16 * v211 + 12))
                    & 0x3FLL)
                   + 1);
    v308 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)&src + 1) + 16 * v211)
                     + 8LL * *(unsigned int *)(*((_QWORD *)&src + 1) + 16 * v211 + 12))
         & 0xFFFFFFFFFFFFFFC0LL;
    v208 = (unsigned int)v428 + 1LL;
    if ( v208 > HIDWORD(v428) )
    {
      v304 = v55;
      sub_C8D5F0((__int64)&src + 8, &v429, v208, 0x10u, v54, (__int64)v55);
      v292 = (_BYTE **)*((_QWORD *)&src + 1);
      v55 = v304;
    }
    v209 = (unsigned __int64 *)&v292[2 * (unsigned int)v428];
    *v209 = v308;
    v209[1] = (unsigned __int64)v55;
  }
  v212 = (unsigned int)v417;
  v53 = (unsigned __int64)v432;
  if ( (_DWORD)v417 )
    goto LABEL_299;
LABEL_414:
  if ( !v210 )
    goto LABEL_302;
LABEL_415:
  v53 = *((unsigned int *)v292 + 2);
  if ( *((_DWORD *)v292 + 3) >= (unsigned int)v53 )
    goto LABEL_302;
  v213 = (unsigned __int64)&v416[2 * (unsigned int)v212 - 2];
  LODWORD(v53) = *(_DWORD *)(v213 + 12);
  if ( !(_DWORD)v53 )
  {
LABEL_405:
    v293 = v415;
    goto LABEL_406;
  }
  if ( (_DWORD)v212 && *((_DWORD *)v416 + 3) < *((_DWORD *)v416 + 2) )
    goto LABEL_399;
  v293 = v415;
  if ( !*(_DWORD *)(v415 + 136) )
    goto LABEL_399;
LABEL_406:
  sub_F03AD0((unsigned int *)&v416, *(_DWORD *)(v293 + 136));
LABEL_400:
  v206 = v416;
  v53 = v406;
  v212 = (__int64)&v416[2 * (unsigned int)v417 - 2];
  if ( *(_QWORD *)(*(_QWORD *)v212 + 16LL * *(unsigned int *)(v212 + 12) + 8) > v406 )
    goto LABEL_302;
  v54 = *((_QWORD *)&src + 1);
  v55 = &v429;
  v9 = v330;
  v8 = v323;
  if ( *((_BYTE ***)&src + 1) != &v429 )
  {
    _libc_free(*((unsigned __int64 *)&src + 1));
    v206 = v416;
  }
LABEL_403:
  if ( v206 != v418 )
    _libc_free((unsigned __int64)v206);
LABEL_85:
  if ( !(_DWORD)v413 )
    goto LABEL_175;
  v313 = v5;
  v317 = v9;
  v71 = *(unsigned __int64 **)(i + 40);
  v309 = v3;
  v306 = v8;
  v72 = *((_DWORD *)v71 + 2);
  v73 = v71[1];
  v74 = *v71;
  *(_QWORD *)&src = &v428;
  v75 = v73;
  v342 = v74;
  *((_QWORD *)&src + 1) = 0x800000000LL;
  v331 = v72;
  v327 = v72;
  v76 = 8LL * (unsigned int)(v413 - 1);
  do
  {
    v77 = *(_QWORD *)&v412[v76];
    v324 = v72 | v75 & 0xFFFFFFFF00000000LL;
    v75 = v324;
    v78 = sub_3268F30(v2, v77, v74, v324, v54, (__int64)v55);
    v54 = sub_33EC3F0(
            *v2,
            v77,
            v78,
            v79,
            *(_QWORD *)(*(_QWORD *)(v77 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v77 + 40) + 48LL),
            *(_OWORD *)(*(_QWORD *)(v77 + 40) + 80LL),
            *(_OWORD *)(*(_QWORD *)(v77 + 40) + 120LL));
    v80 = DWORD2(src);
    v55 = (_QWORD *)(v320 & 0xFFFFFFFF00000000LL);
    v320 &= 0xFFFFFFFF00000000LL;
    v81 = DWORD2(src) + 1LL;
    if ( v81 > HIDWORD(src) )
    {
      v303 = v55;
      v305 = v54;
      sub_C8D5F0((__int64)&src, &v428, v81, 0x10u, v54, (__int64)v55);
      v80 = DWORD2(src);
      v55 = v303;
      v54 = v305;
    }
    v82 = (__int64 *)(src + 16 * v80);
    *v82 = v54;
    v82[1] = (__int64)v55;
    ++DWORD2(src);
    *(_QWORD *)&v412[v76] = v54;
    v76 -= 8;
  }
  while ( v76 != -8 );
  v3 = v309;
  v9 = v317;
  v5 = v313;
  v8 = v306;
  v83 = sub_3268F30(v2, v309, v342, v324, v54, (__int64)v55);
  v84 = *v2;
  v85 = v83;
  v325 = v86;
  v321 = *(_QWORD *)(v309 + 112);
  if ( (*(_BYTE *)(v309 + 33) & 4) != 0 )
  {
    v87 = *(_QWORD *)(v309 + 80);
    v88 = *(_QWORD *)(v309 + 104);
    v89 = *(unsigned __int16 *)(v309 + 96);
    v90 = *(_QWORD **)(v309 + 40);
    v415 = v87;
    if ( v87 )
    {
      v307 = v83;
      v310 = v89;
      v312 = v88;
      v314 = v90;
      v318 = v84;
      sub_B96E90((__int64)&v415, v87, 1);
      v85 = v307;
      v89 = v310;
      v88 = v312;
      v90 = v314;
      LODWORD(v84) = v318;
    }
    LODWORD(v416) = *(_DWORD *)(v3 + 72);
    v91 = sub_33F49B0(v84, v85, v325, (unsigned int)&v415, v90[5], v90[6], v90[10], v90[11], v89, v88, v321);
    v93 = v415;
    v322 = v91;
    v326 = v94;
    if ( v415 )
      goto LABEL_94;
  }
  else
  {
    v179 = *(_QWORD *)(v309 + 80);
    v180 = *(_QWORD **)(v309 + 40);
    v415 = v179;
    if ( v179 )
    {
      v311 = v83;
      v315 = v180;
      v319 = v84;
      sub_B96E90((__int64)&v415, v179, 1);
      v85 = v311;
      v180 = v315;
      LODWORD(v84) = v319;
    }
    LODWORD(v416) = *(_DWORD *)(v3 + 72);
    v181 = sub_33F3F90(v84, v85, v325, (unsigned int)&v415, v180[5], v180[6], v180[10], v180[11], v321);
    v93 = v415;
    v322 = v181;
    v326 = v182;
    if ( v415 )
LABEL_94:
      sub_B91220((__int64)&v415, v93);
  }
  v95 = DWORD2(src);
  v96 = v326;
  v97 = DWORD2(src) + 1LL;
  if ( v97 > HIDWORD(src) )
  {
    sub_C8D5F0((__int64)&src, &v428, v97, 0x10u, v326, v92);
    v95 = DWORD2(src);
    v96 = v326;
  }
  v98 = (__int64 *)(src + 16 * v95);
  v98[1] = v96;
  *v98 = v322;
  v99 = (_DWORD *)src;
  ++DWORD2(src);
  v100 = src + 16LL * DWORD2(src);
  if ( !((16LL * DWORD2(src)) >> 6) )
  {
    v101 = (_QWORD *)src;
LABEL_320:
    v223 = v100 - (_QWORD)v101;
    if ( v100 - (_QWORD)v101 != 32 )
    {
      if ( v223 != 48 )
      {
        if ( v223 != 16 )
        {
LABEL_323:
          v224 = DWORD2(src) + 1LL;
          if ( (_QWORD)src == v100 )
          {
            v285 = v327;
            if ( HIDWORD(src) < v224 )
            {
              sub_C8D5F0((__int64)&src, &v428, v224, 0x10u, v327, v92);
              v99 = (_DWORD *)src;
              v285 = v327;
            }
            v286 = (unsigned __int64 *)&v99[4 * DWORD2(src)];
            v286[1] = v285;
            *v286 = v342;
            ++DWORD2(src);
          }
          else
          {
            if ( HIDWORD(src) < v224 )
            {
              sub_C8D5F0((__int64)&src, &v428, v224, 0x10u, v100, v92);
              v99 = (_DWORD *)src;
            }
            v225 = DWORD2(src);
            v226 = v99;
            v227 = 4LL * DWORD2(src);
            if ( &v99[v227] )
            {
              *(__m128i *)&v99[v227] = _mm_loadu_si128((const __m128i *)&v99[v227 - 4]);
              v225 = DWORD2(src);
              v99 = (_DWORD *)src;
            }
            v228 = 4 * v225;
            if ( v226 != &v99[v228 - 4] )
              memmove(v226 + 4, v226, (char *)&v99[v228 - 4] - (char *)v226);
            ++DWORD2(src);
            *(_QWORD *)v226 = v342;
            v226[2] = v331;
          }
          goto LABEL_106;
        }
LABEL_386:
        v289 = *(_QWORD *)(*v101 + 40LL);
        if ( v342 == *(_QWORD *)v289 && v331 == *(_DWORD *)(v289 + 8) )
          goto LABEL_105;
        goto LABEL_323;
      }
      v287 = *(_QWORD *)(*v101 + 40LL);
      if ( v342 == *(_QWORD *)v287 && *(_DWORD *)(v287 + 8) == v331 )
        goto LABEL_105;
      v101 += 2;
    }
    v288 = *(_QWORD *)(*v101 + 40LL);
    if ( v342 == *(_QWORD *)v288 && v331 == *(_DWORD *)(v288 + 8) )
      goto LABEL_105;
    v101 += 2;
    goto LABEL_386;
  }
  v101 = (_QWORD *)src;
  while ( 1 )
  {
    v92 = *(_QWORD *)(*v101 + 40LL);
    if ( v342 == *(_QWORD *)v92 && *(_DWORD *)(v92 + 8) == v331 )
      break;
    v92 = *(_QWORD *)(v101[2] + 40LL);
    if ( v342 == *(_QWORD *)v92 && *(_DWORD *)(v92 + 8) == v331 )
    {
      v101 += 2;
      break;
    }
    v92 = *(_QWORD *)(v101[4] + 40LL);
    if ( v342 == *(_QWORD *)v92 && *(_DWORD *)(v92 + 8) == v331 )
    {
      v101 += 4;
      break;
    }
    v92 = *(_QWORD *)(v101[6] + 40LL);
    if ( v342 == *(_QWORD *)v92 && *(_DWORD *)(v92 + 8) == v331 )
    {
      v101 += 6;
      break;
    }
    v101 += 8;
    if ( (_QWORD *)(src + ((16LL * DWORD2(src)) >> 6 << 6)) == v101 )
      goto LABEL_320;
  }
LABEL_105:
  if ( (_QWORD *)v100 == v101 )
    goto LABEL_323;
LABEL_106:
  v102 = *v2;
  v103 = *(_QWORD *)(i + 80);
  v415 = v103;
  if ( v103 )
  {
    v343 = v102;
    sub_B96E90((__int64)&v415, v103, 1);
    v102 = v343;
  }
  LODWORD(v416) = *(_DWORD *)(i + 72);
  v105 = sub_3402E70(v102, &v415, &src);
  v106 = v104;
  if ( v415 )
  {
    v354 = v105;
    v344 = v104;
    sub_B91220((__int64)&v415, v415);
    v106 = v344;
    v105 = v354;
  }
  v415 = v105;
  v355 = v105;
  v416 = (_QWORD *)v106;
  sub_32EB790((__int64)v2, v3, &v415, 1, 1);
  v55 = (_QWORD *)v355;
  if ( *(_DWORD *)(v355 + 24) != 328 )
  {
    v415 = v355;
    sub_32B3B20((__int64)(v2 + 71), &v415);
    v55 = (_QWORD *)v355;
    if ( *(int *)(v355 + 88) < 0 )
    {
      *(_DWORD *)(v355 + 88) = *((_DWORD *)v2 + 12);
      v229 = *((unsigned int *)v2 + 12);
      if ( v229 + 1 > (unsigned __int64)*((unsigned int *)v2 + 13) )
      {
        sub_C8D5F0((__int64)(v2 + 5), v2 + 7, v229 + 1, 8u, v54, v355);
        v229 = *((unsigned int *)v2 + 12);
        v55 = (_QWORD *)v355;
      }
      *(_QWORD *)(v2[5] + 8 * v229) = v55;
      ++*((_DWORD *)v2 + 12);
    }
  }
  v107 = (__int64 *)v55[5];
  v53 = (unsigned __int64)(v2 + 71);
  v345 = (__int64)(v2 + 71);
  v356 = &v107[5 * *((unsigned int *)v55 + 16)];
  if ( v107 != v356 )
  {
    v332 = v9;
    v108 = (__int64 *)v55[5];
    v328 = v3;
    v109 = v2;
    do
    {
      v110 = *v108;
      if ( *(_DWORD *)(*v108 + 24) != 328 )
      {
        v415 = *v108;
        sub_32B3B20(v345, &v415);
        if ( *(int *)(v110 + 88) < 0 )
        {
          *(_DWORD *)(v110 + 88) = *((_DWORD *)v109 + 12);
          v111 = *((unsigned int *)v109 + 12);
          v53 = *((unsigned int *)v109 + 13);
          if ( v111 + 1 > v53 )
          {
            sub_C8D5F0((__int64)(v109 + 5), v109 + 7, v111 + 1, 8u, v54, (__int64)v55);
            v111 = *((unsigned int *)v109 + 12);
          }
          *(_QWORD *)(v109[5] + 8 * v111) = v110;
          ++*((_DWORD *)v109 + 12);
        }
      }
      v108 += 5;
    }
    while ( v356 != v108 );
    v2 = v109;
    v9 = v332;
    v3 = v328;
  }
  if ( *(_DWORD *)(i + 24) != 328 )
  {
    v415 = i;
    sub_32B3B20((__int64)(v2 + 71), &v415);
    if ( *(int *)(i + 88) < 0 )
    {
      *(_DWORD *)(i + 88) = *((_DWORD *)v2 + 12);
      v222 = *((unsigned int *)v2 + 12);
      if ( v222 + 1 > (unsigned __int64)*((unsigned int *)v2 + 13) )
      {
        sub_C8D5F0((__int64)(v2 + 5), v2 + 7, v222 + 1, 8u, v54, (__int64)v55);
        v222 = *((unsigned int *)v2 + 12);
      }
      v53 = i;
      *(_QWORD *)(v2[5] + 8 * v222) = i;
      ++*((_DWORD *)v2 + 12);
    }
  }
  if ( (__int64 *)src != &v428 )
    _libc_free(src);
  v365 = 1;
LABEL_176:
  if ( (_DWORD)v433 )
    sub_32AEA30((__int64)v432, (char *)sub_325D570, 0, v53, v54, (unsigned __int64)v55);
  v419.m128i_i64[0] = 0;
  if ( v421 != &v421[(unsigned int)v422] )
  {
    v373 = v9;
    v138 = (unsigned __int64)v421;
    v139 = &v421[(unsigned int)v422];
    v357 = v3;
    v140 = v421;
    while ( 1 )
    {
      v141 = *v140;
      v142 = (unsigned int)((__int64)((__int64)v140 - v138) >> 3) >> 7;
      v143 = 4096LL << v142;
      if ( v142 >= 0x1E )
        v143 = 0x40000000000LL;
      ++v140;
      sub_C7D6A0(v141, v143, 16);
      if ( v139 == v140 )
        break;
      v138 = (unsigned __int64)v421;
    }
    v9 = v373;
    v3 = v357;
  }
  v144 = v424;
  if ( v424 != &v424[2 * (unsigned int)v425] )
  {
    v374 = v9;
    v145 = &v424[2 * (unsigned int)v425];
    v358 = v3;
    v146 = v424;
    do
    {
      v147 = v146[1];
      v148 = *v146;
      v146 += 2;
      sub_C7D6A0(v148, v147, 16);
    }
    while ( v145 != v146 );
    v9 = v374;
    v3 = v358;
    v144 = v424;
  }
  if ( v144 != v426 )
    _libc_free((unsigned __int64)v144);
  if ( v421 != (__int64 *)v423 )
    _libc_free((unsigned __int64)v421);
  if ( v412 != v414 )
    _libc_free((unsigned __int64)v412);
  if ( v365 )
    return v3;
  v37 = v3;
  v150 = sub_3268F30(v2, v3, **(_QWORD **)(v3 + 40), *(_QWORD *)(*(_QWORD *)(v3 + 40) + 8LL), v54, (__int64)v55);
  v112 = *(_QWORD *)(v3 + 40);
  if ( *(_QWORD *)v112 != v150 || *(_DWORD *)(v112 + 8) != v149 )
  {
    sub_32F7110((__int64)v2, v3, v150, v149);
    return v3;
  }
LABEL_121:
  v399.m128i_i64[0] = *(_QWORD *)v112;
  v399.m128i_i32[2] = *(_DWORD *)(v112 + 8);
  if ( (*(_BYTE *)(v3 + 33) & 4) == 0 || (*(_WORD *)(v3 + 32) & 0x380) != 0 )
    goto LABEL_32;
  v113 = 16LL * v379 + *(_QWORD *)(v5 + 48);
  v114 = *(_WORD *)v113;
  v115 = *(_QWORD *)(v113 + 8);
  *(_WORD *)v432 = v114;
  *(_QWORD *)&v432[8] = v115;
  if ( v114 )
  {
    if ( (unsigned __int16)(v114 - 2) <= 7u
      || (unsigned __int16)(v114 - 17) <= 0x6Cu
      || (unsigned __int16)(v114 - 176) <= 0x1Fu )
    {
      goto LABEL_127;
    }
    goto LABEL_32;
  }
  if ( !sub_3007070((__int64)v432) )
    goto LABEL_32;
LABEL_127:
  v116 = *(_DWORD *)(v5 + 24);
  if ( (v116 == 11 || v116 == 35) && (*(_BYTE *)(v5 + 32) & 8) != 0 )
  {
LABEL_32:
    v39 = v5;
    if ( !*(_BYTE *)sub_2E79000(*(__int64 **)(*v2 + 40LL)) )
      v39 = sub_33CF6D0(v5, v8 & 0xFFFFFFFF00000000LL | v379);
    if ( *(_DWORD *)(v39 + 24) == 298 )
    {
      v40 = *(_QWORD *)(v39 + 40);
      if ( v375 == *(_QWORD *)(v40 + 40) && *(_DWORD *)(v40 + 48) == v9 )
      {
        v129 = *(_WORD *)(v3 + 96);
        if ( v129 == *(_WORD *)(v39 + 96)
          && (*(_QWORD *)(v39 + 104) == *(_QWORD *)(v3 + 104) || v129)
          && (*(_WORD *)(v3 + 32) & 0x380) == 0 )
        {
          if ( (unsigned __int8)sub_3287C60(v3) )
          {
            v130 = sub_2EAC1E0(*(_QWORD *)(v39 + 112));
            if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v3 + 112)) == v130 )
            {
              if ( (unsigned __int8)sub_33CFB90(&v399, v39, 1, 2) )
                return v399.m128i_i64[0];
            }
          }
        }
      }
    }
    result = sub_3287C80(v2, v3);
    v362 = result;
    if ( result )
      return result;
    v41 = v399.m128i_i64[0];
    if ( *(_DWORD *)(v399.m128i_i64[0] + 24) != 299 )
      goto LABEL_46;
    v42 = *(_WORD *)(v3 + 32);
    if ( (v42 & 0x380) != 0 )
      goto LABEL_46;
    v371 = *(_QWORD *)(v3 + 112);
    if ( (*(_BYTE *)(v371 + 37) & 0xF) != 0
      || (v42 & 8) != 0
      || (*(_WORD *)(v399.m128i_i64[0] + 32) & 0x380) != 0
      || !(unsigned __int8)sub_3287C60(v399.m128i_i64[0])
      || !*((_DWORD *)v2 + 7) )
    {
      goto LABEL_46;
    }
    v43 = *(_QWORD *)(v41 + 40);
    if ( v375 != *(_QWORD *)(v43 + 80)
      || *(_DWORD *)(v43 + 88) != v9
      || *(_QWORD *)(v43 + 40) != v5
      || *(_DWORD *)(v43 + 48) != v379
      || (v237 = *(_WORD *)(v3 + 96), v237 != *(_WORD *)(v41 + 96))
      || *(_QWORD *)(v41 + 104) != *(_QWORD *)(v3 + 104) && !v237 )
    {
LABEL_45:
      v44 = *(_QWORD *)(v41 + 56);
      if ( !v44 )
        goto LABEL_46;
      if ( *(_QWORD *)(v44 + 32) )
        goto LABEL_46;
      if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v41 + 40) + 80LL) + 24LL) == 51 )
        goto LABEL_46;
      v388 = sub_2EAC1E0(*(_QWORD *)(v3 + 112));
      if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v41 + 112)) != v388 )
        goto LABEL_46;
      v230 = *(_QWORD *)(v3 + 104);
      v348 = *(_WORD *)(v3 + 96);
      LOWORD(src) = v348;
      v389 = v230;
      *((_QWORD *)&src + 1) = v230;
      if ( !sub_3280200((__int64)&src) )
      {
        v231 = *(_WORD *)(v41 + 96);
        *(_QWORD *)&v432[8] = *(_QWORD *)(v41 + 104);
        *(_WORD *)v432 = v231;
        if ( !sub_3280200((__int64)v432) )
        {
          sub_33644B0(&src, v3, *v2, v232);
          sub_33644B0(v432, v41, *v2, v295);
          v296 = *(_QWORD *)(v41 + 104);
          LOWORD(v415) = *(_WORD *)(v41 + 96);
          v416 = (_QWORD *)v296;
          v297 = sub_2D5B750((unsigned __int16 *)&v415);
          v299 = v298;
          v351 = v297;
          LOWORD(v298) = *(_WORD *)(v3 + 96);
          v419.m128i_i64[0] = v297;
          v300 = *(_QWORD *)(v3 + 104);
          v419.m128i_i64[1] = v299;
          LOWORD(v412) = v298;
          v413 = v300;
          v301.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v412);
          v302 = *v2;
          v419 = v301;
          if ( !(unsigned __int8)sub_3364440(&src, v302, v301.m128i_i64[0], v432, v351, &v419) )
            goto LABEL_46;
          v236 = &v419;
          v419 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v41 + 40));
LABEL_345:
          sub_32EB790((__int64)v2, v41, v236->m128i_i64, 1, 1);
          return v3;
        }
      }
      v233 = *(_QWORD *)(v41 + 40);
      if ( v375 == *(_QWORD *)(v233 + 80) && *(_DWORD *)(v233 + 88) == v9 )
      {
        LOWORD(src) = v348;
        *((_QWORD *)&src + 1) = v389;
        *(_QWORD *)&v234 = sub_3285A00((unsigned __int16 *)&src);
        *(_OWORD *)v432 = v234;
        WORD4(v234) = *(_WORD *)(v41 + 96);
        v416 = *(_QWORD **)(v41 + 104);
        LOWORD(v415) = WORD4(v234);
        v235.m128i_i64[0] = sub_3285A00((unsigned __int16 *)&v415);
        v419 = v235;
        if ( (!v235.m128i_i8[8] || v432[8]) && v419.m128i_i64[0] <= *(_QWORD *)v432 )
        {
          v236 = (__m128i *)v432;
          *(__m128i *)v432 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v41 + 40));
          goto LABEL_345;
        }
      }
LABEL_46:
      v45 = *(_DWORD *)(v5 + 24);
      if ( v45 == 216 || v45 == 230 )
      {
        v131 = *(_QWORD *)(v5 + 56);
        if ( v131 )
        {
          if ( !*(_QWORD *)(v131 + 32)
            && (*(_WORD *)(v3 + 32) & 0x380) == 0
            && (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v2[1]
                                                                                                 + 688LL))(
                 v2[1],
                 *(unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v5 + 40) + 48LL)
                                     + 16LL * *(unsigned int *)(*(_QWORD *)(v5 + 40) + 8LL)),
                 *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v5 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v5 + 40) + 8LL)
                           + 8),
                 *(unsigned __int16 *)(v3 + 96),
                 *(_QWORD *)(v3 + 104),
                 *((unsigned __int8 *)v2 + 33)) )
          {
            v132 = *(_QWORD **)(v5 + 40);
            v133 = *v2;
            *(_QWORD *)v432 = *(_QWORD *)(v3 + 80);
            v134 = *(_QWORD *)(v3 + 112);
            v135 = *(unsigned __int16 *)(v3 + 96);
            v136 = *(_QWORD *)(v3 + 104);
            if ( *(_QWORD *)v432 )
            {
              v380 = *(unsigned __int16 *)(v3 + 96);
              v393 = *(_QWORD *)(v3 + 104);
              sub_325F5D0((__int64 *)v432);
              v135 = v380;
              v136 = v393;
            }
            *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
            v381 = sub_33F49B0(
                     v133,
                     v399.m128i_i32[0],
                     v399.m128i_i32[2],
                     (unsigned int)v432,
                     *v132,
                     v132[1],
                     v375,
                     v9 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                     v135,
                     v136,
                     v134);
            sub_9C6650(v432);
            return v381;
          }
        }
      }
      if ( !*((_BYTE *)v2 + 34)
        || (v46 = v2[1], v47 = *(__int64 (**)())(*(_QWORD *)v46 + 272LL), v47 == sub_2FE2FB0)
        || ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v47)(
             v46,
             *(unsigned __int16 *)(v3 + 96),
             *(_QWORD *)(v3 + 104)) )
      {
        while ( (unsigned __int8)sub_32FE970((__int64)v2, v3) )
        {
          if ( *(_DWORD *)(v3 + 24) != 299 )
            return v3;
        }
      }
      if ( *((int *)v2 + 6) <= 2
        || !(unsigned __int8)sub_3312A90(v2, v3) && (*((int *)v2 + 6) <= 2 || !(unsigned __int8)sub_3312210(v2, v3)) )
      {
        v137 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 40) + 40LL) + 24LL);
        if ( v137 == 36 || v137 == 12 )
        {
          result = sub_327C770(v2, v3);
          if ( result )
            return result;
        }
        if ( !*((_DWORD *)v2 + 7)
          || (*(_BYTE *)(*(_QWORD *)(v3 + 112) + 37LL) & 0xF) != 0
          || (*(_BYTE *)(v3 + 32) & 8) != 0 )
        {
          return sub_32BDCD0(v2, v3);
        }
        v158 = *(_QWORD *)(v3 + 40);
        v159 = *(_QWORD *)(v3 + 80);
        v160 = *(_QWORD *)(v158 + 40);
        v161 = *(_DWORD *)(v158 + 48);
        v404 = v159;
        if ( v159 )
          sub_B96E90((__int64)&v404, v159, 1);
        v405 = *(_DWORD *)(v3 + 72);
        v162 = *(_QWORD *)(v160 + 48) + 16LL * v161;
        v163 = *(_WORD *)v162;
        v164 = *(_QWORD *)(v162 + 8);
        *(_WORD *)v432 = v163;
        *(_QWORD *)&v432[8] = v164;
        if ( v163 )
        {
          if ( (unsigned __int16)(v163 - 2) > 7u )
            goto LABEL_220;
        }
        else
        {
          v183 = sub_30070A0((__int64)v432);
          v163 = 0;
          if ( !v183 )
            goto LABEL_220;
        }
        if ( *(_DWORD *)(v160 + 24) != 187 )
          goto LABEL_220;
        v165 = *(_QWORD *)(v160 + 40);
        v166 = *(_QWORD *)v165;
        v167 = *(_DWORD *)(v165 + 8);
        v168 = *(_QWORD *)(v165 + 40);
        LODWORD(v165) = *(_DWORD *)(v165 + 48);
        v406 = 0;
        LODWORD(v407) = 0;
        v410 = 0;
        LODWORD(v411) = 0;
        v396 = v165;
        if ( *(_DWORD *)(v166 + 24) != 190 )
        {
          if ( *(_DWORD *)(v168 + 24) != 190 )
            goto LABEL_220;
          v169 = v167;
          v167 = v396;
          v170 = v168;
          v168 = v166;
          v166 = v170;
          v396 = v169;
        }
        v406 = v168;
        v171 = 1;
        LODWORD(v407) = v396;
        v172 = *(_QWORD *)(v166 + 40);
        v410 = *(_QWORD *)v172;
        LODWORD(v411) = *(_DWORD *)(v172 + 8);
        for ( j = *(_QWORD *)(v166 + 56); j; j = *(_QWORD *)(j + 32) )
        {
          if ( *(_DWORD *)(j + 8) == v167 )
          {
            if ( !v171 )
              goto LABEL_220;
            v171 = 0;
          }
        }
        if ( v171 == 1 )
        {
LABEL_220:
          if ( v404 )
            sub_B91220((__int64)&v404, v404);
          result = v362;
          if ( v362 )
            return result;
          return sub_32BDCD0(v2, v3);
        }
        LOWORD(src) = v163;
        *((_QWORD *)&src + 1) = v164;
        if ( !v163 )
        {
          v238.m128i_i64[0] = sub_3007260((__int64)&src);
          v419 = v238;
          goto LABEL_358;
        }
        if ( v163 != 1 && (unsigned __int16)(v163 - 504) > 7u )
        {
          v242 = 16LL * (v163 - 1);
          v243 = byte_444C4A0[v242 + 8];
          v244 = *(_QWORD *)&byte_444C4A0[v242];
          v419.m128i_i8[8] = v243;
          v419.m128i_i64[0] = v244;
LABEL_358:
          *(_QWORD *)v432 = v419.m128i_i64[0];
          v432[8] = v419.m128i_i8[8];
          v239 = sub_CA1930(v432);
          v240 = *(_QWORD *)(*(_QWORD *)(v166 + 40) + 40LL);
          v241 = *(_DWORD *)(v240 + 24);
          if ( v241 == 11 || v241 == 35 )
          {
            v391 = v239 >> 1;
            v376 = (unsigned int)(v239 >> 1);
            if ( sub_D94970(*(_QWORD *)(v240 + 96) + 24LL, (_QWORD *)v376) )
            {
              if ( *(_DWORD *)(v168 + 24) != 214 )
                goto LABEL_362;
              if ( !(unsigned __int8)sub_3286E00(&v406) )
                goto LABEL_362;
              v245 = *(_QWORD *)(v168 + 40);
              v246 = *(_QWORD *)v245;
              v247 = *(unsigned int *)(v245 + 8);
              v360 = v247;
              v248 = *(_QWORD *)(v246 + 48) + 16 * v247;
              v249 = *(_QWORD *)(v248 + 8);
              LOWORD(v248) = *(_WORD *)v248;
              *(_QWORD *)&v432[8] = v249;
              *(_WORD *)v432 = v248;
              if ( !sub_32801C0((__int64)v432) )
                goto LABEL_362;
              *(_QWORD *)&src = sub_3262090(v246, v360);
              *((_QWORD *)&src + 1) = v250;
              if ( v376 < sub_CA1930(&src) )
                goto LABEL_362;
              v251 = v410;
              if ( *(_DWORD *)(v410 + 24) != 214 )
                goto LABEL_362;
              if ( !(unsigned __int8)sub_3286E00(&v410) )
                goto LABEL_362;
              v252 = *(_QWORD *)(v251 + 40);
              v253 = *(_QWORD *)v252;
              v254 = *(unsigned int *)(v252 + 8);
              v338 = v254;
              v255 = *(_QWORD *)(v253 + 48) + 16 * v254;
              v256 = *(_QWORD *)(v255 + 8);
              v349 = v253;
              LOWORD(v415) = *(_WORD *)v255;
              v416 = (_QWORD *)v256;
              if ( !sub_32801C0((__int64)&v415) )
                goto LABEL_362;
              v412 = (_BYTE *)sub_3262090(v349, v338);
              v413 = v257;
              if ( v376 >= sub_CA1930(&v412) )
              {
                v258 = *(unsigned int **)(v168 + 40);
                if ( *(_DWORD *)(*(_QWORD *)v258 + 24LL) == 234 )
                  v259 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v258 + 48LL) + 16LL * v258[2]);
                else
                  v259 = (unsigned __int16 *)(*(_QWORD *)(v168 + 48) + 16LL * v396);
                v260 = *v259;
                v261 = *((_QWORD *)v259 + 1);
                v262 = *(unsigned int **)(v251 + 40);
                if ( *(_DWORD *)(*(_QWORD *)v262 + 24LL) == 234 )
                {
                  v290 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v262 + 48LL) + 16LL * v262[2]);
                  v265 = *((_QWORD *)v290 + 1);
                  v264 = *v290;
                }
                else
                {
                  v263 = (unsigned __int16 *)(*(_QWORD *)(v251 + 48) + 16LL * (unsigned int)v411);
                  v264 = *v263;
                  v265 = *((_QWORD *)v263 + 1);
                }
                if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)v2[1] + 344LL))(
                       v2[1],
                       v260,
                       v261,
                       v264,
                       v265) )
                {
                  v266 = *(_QWORD *)(v3 + 112);
                  v267 = *(_WORD *)(v266 + 32);
                  *(__m128i *)v432 = _mm_loadu_si128((const __m128i *)(v266 + 40));
                  v268 = _mm_loadu_si128((const __m128i *)(v266 + 56));
                  v269 = *v2;
                  *(__m128i *)&v432[16] = v268;
                  v270 = sub_327FC40(*(_QWORD **)(v269 + 64), v391);
                  v377 = v271;
                  v366 = v270;
                  v406 = sub_33FAF80(*v2, 214, (unsigned int)&v404, v270, v271, v270, *(_OWORD *)*(_QWORD *)(v168 + 40));
                  LODWORD(v407) = v272;
                  v398 = sub_33FAF80(*v2, 214, (unsigned int)&v404, v366, v377, v366, *(_OWORD *)*(_QWORD *)(v251 + 40));
                  v273 = *v2;
                  v410 = v398;
                  LODWORD(v411) = v274;
                  v275 = *(_QWORD **)(v3 + 40);
                  v276 = v275[11];
                  v277 = v275[10];
                  v278 = sub_33F4560(
                           v273,
                           *v275,
                           v275[1],
                           (unsigned int)&v404,
                           v406,
                           v407,
                           v277,
                           v276,
                           *(_OWORD *)*(_QWORD *)(v3 + 112),
                           *(_QWORD *)(*(_QWORD *)(v3 + 112) + 16LL),
                           *(_BYTE *)(*(_QWORD *)(v3 + 112) + 34LL),
                           v267,
                           (__int64)v432);
                  v279 = *v2;
                  v397 = v278;
                  LOBYTE(v416) = 0;
                  v378 = v280;
                  v415 = (unsigned int)v391 >> 3;
                  v350 = v415;
                  v281 = sub_3409320(v279, v277, v276, (unsigned int)v391 >> 3, (_DWORD)v416, (unsigned int)&v404, 0);
                  v283 = *(_QWORD *)(v3 + 112);
                  v284 = v282 | v276 & 0xFFFFFFFF00000000LL;
                  v392 = *(_BYTE *)(v283 + 34);
                  v367 = *v2;
                  sub_327C6E0((__int64)&src, (__int64 *)v283, v350);
                  v362 = sub_33F4560(
                           v367,
                           v397,
                           v378,
                           (unsigned int)&v404,
                           v410,
                           v411,
                           v281,
                           v284,
                           src,
                           v428,
                           v392,
                           v267,
                           (__int64)v432);
                }
              }
              else
              {
LABEL_362:
                v362 = 0;
              }
            }
          }
          goto LABEL_220;
        }
LABEL_431:
        BUG();
      }
      return v3;
    }
    v390 = sub_2EAC1E0(v371);
    if ( (unsigned int)sub_2EAC1E0(*(_QWORD *)(v41 + 112)) != v390 )
    {
      if ( !*((_DWORD *)v2 + 7) )
        goto LABEL_46;
      goto LABEL_45;
    }
    return v399.m128i_i64[0];
  }
  v117 = *(_QWORD *)(v3 + 104);
  v118 = *(_WORD *)(v3 + 96);
  if ( (unsigned int)(v116 - 213) > 2 )
    goto LABEL_135;
  v119 = *(_QWORD *)(**(_QWORD **)(v5 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v5 + 40) + 8LL);
  if ( *(_WORD *)v119 != v118 || *(_QWORD *)(v119 + 8) != v117 && !v118 )
    goto LABEL_135;
  v37 = 299;
  v335 = *(_QWORD **)(v5 + 40);
  v363 = *(_WORD *)(v3 + 96);
  v372 = *(_QWORD *)(v3 + 104);
  v120 = sub_328A020(v2[1], 0x12Bu, v118, v117, 0);
  v121 = v335;
  if ( v120 )
  {
    v175 = *v2;
    v176 = *(_QWORD *)(v3 + 112);
    *(_QWORD *)v432 = *(_QWORD *)(v3 + 80);
    if ( *(_QWORD *)v432 )
    {
      sub_325F5D0((__int64 *)v432);
      v121 = v335;
    }
    *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
    v384 = sub_33F3F90(
             v175,
             v399.m128i_i32[0],
             v399.m128i_i32[2],
             (unsigned int)v432,
             *v121,
             v121[1],
             v375,
             v9 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             v176);
    sub_9C6650(v432);
    return v384;
  }
  v117 = v372;
  v118 = v363;
LABEL_135:
  *(_WORD *)v432 = v118;
  *(_QWORD *)&v432[8] = v117;
  v364 = sub_32844A0((unsigned __int16 *)v432, v37);
  v122 = sub_3263630(v5, v379);
  v124 = v364;
  v419.m128i_i32[2] = v122;
  if ( v122 > 0x40 )
  {
    sub_C43690((__int64)&v419, 0, 0);
    v124 = v364;
  }
  else
  {
    v419.m128i_i64[0] = 0;
  }
  if ( (_DWORD)v124 )
  {
    if ( (unsigned int)v124 > 0x40 )
    {
      sub_C43C90(&v419, 0, v124);
    }
    else
    {
      v125 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v124);
      if ( v419.m128i_i32[2] > 0x40u )
        *(_QWORD *)v419.m128i_i64[0] |= v125;
      else
        v419.m128i_i64[0] |= v125;
    }
  }
  sub_32B3E80((__int64)v2, v5, 1, 0, v124, v123);
  v8 = v379 | v8 & 0xFFFFFFFF00000000LL;
  if ( (unsigned __int8)sub_32D08B0((__int64)v2, v5, v8, (int)&v419) )
  {
    if ( *(_DWORD *)(v3 + 24) )
      sub_32B3E80((__int64)v2, v3, 1, 0, v126, v127);
    v128 = v3;
    goto LABEL_145;
  }
  v152 = sub_34494D0(v2[1], v5, v8, &v419, *v2, 0);
  v153 = v151;
  if ( v152 )
  {
    v154 = *v2;
    v155 = *(_QWORD *)(v3 + 112);
    v156 = *(unsigned __int16 *)(v3 + 96);
    *(_QWORD *)v432 = *(_QWORD *)(v3 + 80);
    v157 = *(_QWORD *)(v3 + 104);
    if ( *(_QWORD *)v432 )
    {
      v359 = v152;
      v382 = v155;
      v346 = v151;
      sub_325F5D0((__int64 *)v432);
      v153 = v346;
      LODWORD(v152) = v359;
      v155 = v382;
    }
    *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
    v383 = sub_33F49B0(
             v154,
             v399.m128i_i32[0],
             v399.m128i_i32[2],
             (unsigned int)v432,
             v152,
             v153,
             v375,
             v9 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             v156,
             v157,
             v155);
    sub_9C6650(v432);
    v128 = v383;
    goto LABEL_145;
  }
  v177 = *(_DWORD *)(v5 + 24);
  if ( v177 != 11 && v177 != 35 || (*(_BYTE *)(v5 + 32) & 8) != 0 )
  {
LABEL_255:
    sub_969240(v419.m128i_i64);
    goto LABEL_32;
  }
  v336 = (_QWORD *)(*(_QWORD *)(v5 + 96) + 24LL);
  sub_9865C0((__int64)v432, (__int64)v336);
  sub_325F530(v432, v419.m128i_i64);
  DWORD2(src) = *(_DWORD *)&v432[8];
  *(_QWORD *)&src = *(_QWORD *)v432;
  if ( sub_AAD8B0((__int64)&src, v336) )
  {
    sub_969240((__int64 *)&src);
    goto LABEL_255;
  }
  v184 = (unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16LL * v379);
  v185 = *((_QWORD *)v184 + 1);
  v186 = *v184;
  v187 = *v2;
  *(_QWORD *)v432 = *(_QWORD *)(v3 + 80);
  if ( *(_QWORD *)v432 )
  {
    v385 = v186;
    sub_325F5D0((__int64 *)v432);
    v186 = v385;
  }
  *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
  v188 = sub_34007B0(v187, (unsigned int)&src, (unsigned int)v432, v186, v185, 0, 0);
  v190 = v189;
  sub_9C6650(v432);
  v191 = *v2;
  v192 = *(_QWORD *)(v3 + 112);
  v193 = *(unsigned __int16 *)(v3 + 96);
  v194 = *(_QWORD *)(v3 + 104);
  *(_QWORD *)v432 = *(_QWORD *)(v3 + 80);
  if ( *(_QWORD *)v432 )
  {
    v347 = v193;
    v352 = v194;
    v386 = v192;
    sub_325F5D0((__int64 *)v432);
    v193 = v347;
    v194 = v352;
    v192 = v386;
  }
  *(_DWORD *)&v432[8] = *(_DWORD *)(v3 + 72);
  v387 = sub_33F49B0(
           v191,
           v399.m128i_i32[0],
           v399.m128i_i32[2],
           (unsigned int)v432,
           v188,
           v190,
           v375,
           v9 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL,
           v193,
           v194,
           v192);
  sub_9C6650(v432);
  sub_969240((__int64 *)&src);
  v128 = v387;
LABEL_145:
  v395 = v128;
  sub_969240(v419.m128i_i64);
  return v395;
}
