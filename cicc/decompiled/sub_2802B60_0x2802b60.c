// Function: sub_2802B60
// Address: 0x2802b60
//
__int64 __fastcall sub_2802B60(_QWORD *a1)
{
  _QWORD *v1; // r15
  __int64 v2; // rdx
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  const __m128i *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // rsi
  const __m128i *v20; // rdi
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rcx
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  const __m128i *v28; // rcx
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  _QWORD *v36; // r14
  __int64 v37; // rcx
  __int64 v38; // r15
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rbx
  __int64 *v42; // rax
  char v43; // dl
  unsigned __int64 v44; // rdx
  __int64 v46; // rsi
  __int64 *v47; // rax
  __int64 *v48; // rbx
  int v49; // r13d
  _QWORD *v50; // r12
  __int64 v51; // r14
  __int64 v52; // rbx
  __int64 v53; // r15
  int v54; // r12d
  __int64 v55; // r13
  __int64 v56; // rsi
  unsigned int v57; // ebx
  _QWORD *v58; // r12
  _QWORD *v59; // r15
  unsigned __int64 v60; // rdx
  _QWORD *v61; // rax
  _QWORD *v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // rsi
  __int64 v65; // rax
  _QWORD *v66; // rdx
  __int64 v67; // r8
  __int64 v68; // rdi
  unsigned int v69; // eax
  _QWORD *v70; // rbx
  _QWORD *v71; // r12
  unsigned __int64 v72; // rdx
  _QWORD *v73; // rax
  _QWORD *v74; // rcx
  __int64 v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // rax
  _QWORD *v78; // rdx
  __int64 v79; // r8
  __int64 v80; // rdi
  unsigned int v81; // eax
  unsigned int v82; // eax
  _QWORD *v83; // r15
  __int64 v84; // rbx
  char v85; // al
  __int64 v86; // r12
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 *v91; // r14
  __int64 v92; // rdx
  __int64 v93; // r9
  __m128i *v94; // r13
  __m128i *v95; // r12
  __int64 v96; // rdi
  _QWORD *v97; // rax
  __int64 v98; // rax
  __int64 *v99; // rdx
  unsigned int v100; // eax
  char v101; // al
  unsigned int v102; // r12d
  _QWORD *v103; // rbx
  _QWORD *v104; // r15
  unsigned __int64 v105; // rcx
  _QWORD *v106; // rax
  _QWORD *v107; // rdx
  __int64 v108; // rdi
  __int64 v109; // rsi
  __int128 v110; // rax
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // rdi
  __int64 *v114; // r15
  unsigned __int64 *v115; // rbx
  __int64 v116; // rdx
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 *v119; // rdi
  __m128i v120; // xmm0
  __m128i v121; // xmm5
  __int64 v122; // rax
  __m128i v123; // xmm6
  __m128i v124; // xmm7
  __m128i v125; // xmm0
  __m128i v126; // xmm3
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // r8
  __int64 *v130; // r12
  __int64 v131; // rax
  _QWORD *v132; // rax
  __int64 *v133; // rax
  __int64 v134; // rdx
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int64 v137; // r8
  __int64 v138; // r9
  _QWORD *v139; // r12
  __int64 v140; // rax
  __int64 *v141; // rax
  __int64 v142; // rdx
  __int64 v143; // rax
  _QWORD *v144; // rax
  __int64 v145; // rax
  __int64 v146; // r12
  __int64 v147; // rsi
  __int64 v148; // r12
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  __int64 v152; // r8
  __int64 v153; // r9
  __m128i v154; // xmm4
  __m128i si128; // xmm5
  __m128i v156; // xmm6
  int v157; // r12d
  unsigned __int64 *v158; // r13
  unsigned __int64 *v159; // r13
  __int64 v160; // r8
  unsigned __int64 *v161; // r12
  unsigned __int64 v162; // rdi
  _QWORD *v163; // r13
  _QWORD *v164; // r12
  unsigned __int64 v165; // rsi
  _QWORD *v166; // rax
  _QWORD *v167; // rdi
  __int64 v168; // rcx
  __int64 v169; // rdx
  __int64 v170; // rax
  _QWORD *v171; // rdi
  __int64 v172; // rax
  __int64 v173; // rcx
  __int64 v174; // rsi
  __int64 v175; // rdi
  __int64 *v176; // rax
  __int64 *v177; // rax
  unsigned int v178; // r8d
  unsigned int v179; // edx
  __int64 v180; // r10
  unsigned int v181; // esi
  __int64 *v182; // rdx
  __int64 *v183; // rsi
  unsigned __int64 v184; // rdx
  unsigned __int64 v185; // rax
  __m128i *v186; // rax
  __m128i *v187; // rcx
  unsigned __int64 v188; // rdx
  __int64 v189; // rax
  __int64 v190; // rax
  __m128i *v191; // rax
  __int64 v192; // rsi
  int v193; // r14d
  __int64 v194; // r12
  __m128i *v195; // rbx
  unsigned __int64 *v196; // r12
  unsigned __int64 v197; // rdi
  __int64 v198; // [rsp+30h] [rbp-E10h]
  __int64 v199; // [rsp+38h] [rbp-E08h]
  __int64 *v200; // [rsp+40h] [rbp-E00h]
  __int64 v201; // [rsp+78h] [rbp-DC8h]
  unsigned __int64 *v202; // [rsp+78h] [rbp-DC8h]
  char v203; // [rsp+96h] [rbp-DAAh]
  unsigned __int8 v204; // [rsp+97h] [rbp-DA9h]
  unsigned int v205; // [rsp+98h] [rbp-DA8h]
  _QWORD *v206; // [rsp+98h] [rbp-DA8h]
  __int64 *v207; // [rsp+A0h] [rbp-DA0h]
  unsigned int v208; // [rsp+B8h] [rbp-D88h]
  __int64 v209; // [rsp+C0h] [rbp-D80h]
  __int64 v210; // [rsp+C0h] [rbp-D80h]
  __int64 *v211; // [rsp+C0h] [rbp-D80h]
  char v212; // [rsp+C0h] [rbp-D80h]
  __int64 *v213; // [rsp+C8h] [rbp-D78h]
  __int64 v214; // [rsp+C8h] [rbp-D78h]
  char v215; // [rsp+C8h] [rbp-D78h]
  __int64 *v216; // [rsp+C8h] [rbp-D78h]
  _QWORD *v217; // [rsp+D0h] [rbp-D70h]
  unsigned int v218; // [rsp+D0h] [rbp-D70h]
  __int64 v219; // [rsp+E0h] [rbp-D60h]
  unsigned int v220; // [rsp+E0h] [rbp-D60h]
  __int64 v221; // [rsp+E0h] [rbp-D60h]
  int v222; // [rsp+E0h] [rbp-D60h]
  _QWORD *v223; // [rsp+E0h] [rbp-D60h]
  __int64 *v224; // [rsp+E8h] [rbp-D58h]
  __int64 v225; // [rsp+E8h] [rbp-D58h]
  __m128i *v226; // [rsp+E8h] [rbp-D58h]
  __int64 v227; // [rsp+F0h] [rbp-D50h] BYREF
  __int64 v228; // [rsp+F8h] [rbp-D48h]
  __int64 v229; // [rsp+100h] [rbp-D40h] BYREF
  __int64 *v230; // [rsp+108h] [rbp-D38h]
  unsigned int v231; // [rsp+110h] [rbp-D30h]
  unsigned int v232; // [rsp+114h] [rbp-D2Ch]
  char v233; // [rsp+11Ch] [rbp-D24h]
  char v234[64]; // [rsp+120h] [rbp-D20h] BYREF
  unsigned __int64 v235; // [rsp+160h] [rbp-CE0h] BYREF
  __int64 v236; // [rsp+168h] [rbp-CD8h]
  unsigned __int64 v237; // [rsp+170h] [rbp-CD0h]
  char v238[8]; // [rsp+180h] [rbp-CC0h] BYREF
  unsigned __int64 v239; // [rsp+188h] [rbp-CB8h]
  char v240; // [rsp+19Ch] [rbp-CA4h]
  char v241[64]; // [rsp+1A0h] [rbp-CA0h] BYREF
  unsigned __int64 v242; // [rsp+1E0h] [rbp-C60h]
  unsigned __int64 v243; // [rsp+1E8h] [rbp-C58h]
  unsigned __int64 v244; // [rsp+1F0h] [rbp-C50h]
  __int16 v245; // [rsp+200h] [rbp-C40h] BYREF
  char v246; // [rsp+202h] [rbp-C3Eh]
  int v247; // [rsp+204h] [rbp-C3Ch]
  char v248; // [rsp+208h] [rbp-C38h]
  __int64 v249; // [rsp+210h] [rbp-C30h]
  int v250; // [rsp+218h] [rbp-C28h]
  __int64 v251; // [rsp+220h] [rbp-C20h]
  int v252; // [rsp+228h] [rbp-C18h]
  int v253; // [rsp+230h] [rbp-C10h]
  __int64 v254; // [rsp+238h] [rbp-C08h]
  __int64 v255; // [rsp+240h] [rbp-C00h]
  __int64 v256; // [rsp+248h] [rbp-BF8h]
  unsigned int v257; // [rsp+250h] [rbp-BF0h]
  __int64 v258; // [rsp+258h] [rbp-BE8h]
  __int64 v259; // [rsp+260h] [rbp-BE0h]
  __int64 v260; // [rsp+268h] [rbp-BD8h]
  __int64 v261; // [rsp+270h] [rbp-BD0h]
  __int64 v262; // [rsp+278h] [rbp-BC8h]
  __int64 v263; // [rsp+280h] [rbp-BC0h]
  __m128i v264; // [rsp+290h] [rbp-BB0h] BYREF
  _QWORD v265[15]; // [rsp+2A0h] [rbp-BA0h] BYREF
  char v266[8]; // [rsp+320h] [rbp-B20h] BYREF
  unsigned __int64 v267; // [rsp+328h] [rbp-B18h]
  char v268; // [rsp+33Ch] [rbp-B04h]
  char v269[64]; // [rsp+340h] [rbp-B00h] BYREF
  const __m128i *v270; // [rsp+380h] [rbp-AC0h]
  unsigned __int64 v271; // [rsp+388h] [rbp-AB8h]
  __int64 v272; // [rsp+390h] [rbp-AB0h]
  char v273[8]; // [rsp+398h] [rbp-AA8h] BYREF
  unsigned __int64 v274; // [rsp+3A0h] [rbp-AA0h]
  char v275; // [rsp+3B4h] [rbp-A8Ch]
  char v276[64]; // [rsp+3B8h] [rbp-A88h] BYREF
  const __m128i *v277; // [rsp+3F8h] [rbp-A48h]
  const __m128i *v278; // [rsp+400h] [rbp-A40h]
  __int64 v279; // [rsp+408h] [rbp-A38h]
  __int64 v280; // [rsp+410h] [rbp-A30h] BYREF
  char *v281; // [rsp+418h] [rbp-A28h]
  __int64 v282; // [rsp+420h] [rbp-A20h]
  int v283; // [rsp+428h] [rbp-A18h]
  char v284; // [rsp+42Ch] [rbp-A14h]
  char v285; // [rsp+430h] [rbp-A10h] BYREF
  _OWORD v286[26]; // [rsp+530h] [rbp-910h] BYREF
  char v287; // [rsp+6D0h] [rbp-770h]
  int v288; // [rsp+6D4h] [rbp-76Ch]
  __int64 v289; // [rsp+6D8h] [rbp-768h]
  unsigned __int64 v290; // [rsp+6E0h] [rbp-760h] BYREF
  unsigned __int64 v291; // [rsp+6E8h] [rbp-758h]
  __int64 v292; // [rsp+6F0h] [rbp-750h] BYREF
  __m128i v293; // [rsp+6F8h] [rbp-748h] BYREF
  __m128i v294; // [rsp+708h] [rbp-738h] BYREF
  __m128i v295; // [rsp+718h] [rbp-728h] BYREF
  __m128i v296; // [rsp+728h] [rbp-718h] BYREF
  __int64 v297; // [rsp+738h] [rbp-708h]
  unsigned __int64 v298; // [rsp+740h] [rbp-700h] BYREF
  unsigned __int64 v299; // [rsp+748h] [rbp-6F8h]
  __int64 v300; // [rsp+750h] [rbp-6F0h]
  char v301; // [rsp+880h] [rbp-5C0h]
  int v302; // [rsp+884h] [rbp-5BCh]
  __int64 v303; // [rsp+888h] [rbp-5B8h]
  __m128i *v304; // [rsp+890h] [rbp-5B0h] BYREF
  unsigned __int64 v305; // [rsp+898h] [rbp-5A8h]
  _BYTE v306[16]; // [rsp+8A0h] [rbp-5A0h] BYREF
  _BYTE v307[64]; // [rsp+8B0h] [rbp-590h] BYREF
  unsigned __int64 v308; // [rsp+8F0h] [rbp-550h]
  unsigned __int64 v309; // [rsp+8F8h] [rbp-548h]
  __int64 v310; // [rsp+900h] [rbp-540h]
  __m128i v311; // [rsp+AA0h] [rbp-3A0h] BYREF
  const char *v312; // [rsp+AB0h] [rbp-390h]
  __int64 v313; // [rsp+AB8h] [rbp-388h]
  _QWORD v314[3]; // [rsp+AC0h] [rbp-380h] BYREF
  int v315; // [rsp+AD8h] [rbp-368h]
  __int64 v316; // [rsp+AE0h] [rbp-360h]
  __int64 v317; // [rsp+AE8h] [rbp-358h]
  __int64 v318; // [rsp+AF0h] [rbp-350h]
  __int64 v319; // [rsp+AF8h] [rbp-348h]
  unsigned __int128 v320; // [rsp+B00h] [rbp-340h]
  __int64 v321; // [rsp+B10h] [rbp-330h]
  __int64 v322; // [rsp+B18h] [rbp-328h]
  __int64 v323; // [rsp+B20h] [rbp-320h]
  char *v324; // [rsp+B28h] [rbp-318h]
  __int64 v325; // [rsp+B30h] [rbp-310h]
  int v326; // [rsp+B38h] [rbp-308h]
  char v327; // [rsp+B3Ch] [rbp-304h]
  char v328; // [rsp+B40h] [rbp-300h] BYREF
  __int64 v329; // [rsp+BC0h] [rbp-280h]
  __int64 v330; // [rsp+BC8h] [rbp-278h]
  __int64 v331; // [rsp+BD0h] [rbp-270h]
  int v332; // [rsp+BD8h] [rbp-268h]
  char *v333; // [rsp+BE0h] [rbp-260h]
  __int64 v334; // [rsp+BE8h] [rbp-258h]
  char v335; // [rsp+BF0h] [rbp-250h] BYREF
  __int64 v336; // [rsp+C20h] [rbp-220h]
  __int64 v337; // [rsp+C28h] [rbp-218h]
  __int64 v338; // [rsp+C30h] [rbp-210h]
  int v339; // [rsp+C38h] [rbp-208h]
  __int64 v340; // [rsp+C40h] [rbp-200h]
  char *v341; // [rsp+C48h] [rbp-1F8h]
  __int64 v342; // [rsp+C50h] [rbp-1F0h]
  int v343; // [rsp+C58h] [rbp-1E8h]
  char v344; // [rsp+C5Ch] [rbp-1E4h]
  char v345; // [rsp+C60h] [rbp-1E0h] BYREF
  __int64 v346; // [rsp+C70h] [rbp-1D0h]
  __int64 v347; // [rsp+C78h] [rbp-1C8h]
  __int64 v348; // [rsp+C80h] [rbp-1C0h]
  __int64 v349; // [rsp+C88h] [rbp-1B8h]
  __int64 v350; // [rsp+C90h] [rbp-1B0h]
  __int64 v351; // [rsp+C98h] [rbp-1A8h]
  __int16 v352; // [rsp+CA0h] [rbp-1A0h]
  char v353; // [rsp+CA2h] [rbp-19Eh]
  char *v354; // [rsp+CA8h] [rbp-198h]
  __int64 v355; // [rsp+CB0h] [rbp-190h]
  char v356; // [rsp+CB8h] [rbp-188h] BYREF
  __int64 v357; // [rsp+CD8h] [rbp-168h]
  __int64 v358; // [rsp+CE0h] [rbp-160h]
  __int16 v359; // [rsp+CE8h] [rbp-158h]
  __int64 v360; // [rsp+CF0h] [rbp-150h]
  _QWORD *v361; // [rsp+CF8h] [rbp-148h]
  void **v362; // [rsp+D00h] [rbp-140h]
  __int64 v363; // [rsp+D08h] [rbp-138h]
  int v364; // [rsp+D10h] [rbp-130h]
  __int16 v365; // [rsp+D14h] [rbp-12Ch]
  char v366; // [rsp+D16h] [rbp-12Ah]
  __int64 v367; // [rsp+D18h] [rbp-128h]
  __int64 v368; // [rsp+D20h] [rbp-120h]
  _QWORD v369[3]; // [rsp+D28h] [rbp-118h] BYREF
  __m128i v370; // [rsp+D40h] [rbp-100h]
  __m128i v371; // [rsp+D50h] [rbp-F0h]
  __m128i v372; // [rsp+D60h] [rbp-E0h]
  __m128i v373; // [rsp+D70h] [rbp-D0h]
  __int64 v374; // [rsp+D80h] [rbp-C0h]
  void *v375; // [rsp+D88h] [rbp-B8h] BYREF
  char v376[16]; // [rsp+D90h] [rbp-B0h] BYREF
  __int64 v377; // [rsp+DA0h] [rbp-A0h]
  __int64 v378; // [rsp+DA8h] [rbp-98h]
  char *v379; // [rsp+DB0h] [rbp-90h]
  __int64 v380; // [rsp+DB8h] [rbp-88h]
  char v381; // [rsp+DC0h] [rbp-80h] BYREF
  const char *v382; // [rsp+E00h] [rbp-40h]

  v1 = a1;
  v2 = a1[2];
  v200 = *(__int64 **)(v2 + 40);
  if ( *(__int64 **)(v2 + 32) != v200 )
  {
    v207 = *(__int64 **)(v2 + 32);
    v204 = 0;
    while ( 1 )
    {
      v3 = *v207;
      v298 = 0;
      memset(v286, 0, 0x78u);
      v292 = 0x100000008LL;
      *((_QWORD *)&v286[0] + 1) = &v286[2];
      v293.m128i_i64[1] = v3;
      v311.m128i_i64[0] = v3;
      LODWORD(v286[1]) = 8;
      BYTE12(v286[1]) = 1;
      v291 = (unsigned __int64)&v293.m128i_u64[1];
      v299 = 0;
      v300 = 0;
      v293.m128i_i32[0] = 0;
      v293.m128i_i8[4] = 1;
      v290 = 1;
      LOBYTE(v312) = 0;
      sub_2802B20((__int64)&v298, &v311);
      sub_C8CF70((__int64)&v311, v314, 8, (__int64)&v286[2], (__int64)v286);
      v4 = *(_QWORD *)&v286[6];
      memset(&v286[6], 0, 24);
      v320 = __PAIR128__(*((unsigned __int64 *)&v286[6] + 1), v4);
      v321 = *(_QWORD *)&v286[7];
      sub_C8CF70((__int64)&v304, v307, 8, (__int64)&v293.m128i_i64[1], (__int64)&v290);
      v5 = v298;
      v298 = 0;
      v308 = v5;
      v6 = v299;
      v299 = 0;
      v309 = v6;
      v7 = v300;
      v300 = 0;
      v310 = v7;
      sub_C8CF70((__int64)v266, v269, 8, (__int64)v307, (__int64)&v304);
      v8 = v308;
      v308 = 0;
      v270 = (const __m128i *)v8;
      v9 = v309;
      v309 = 0;
      v271 = v9;
      v10 = v310;
      v310 = 0;
      v272 = v10;
      sub_C8CF70((__int64)v273, v276, 8, (__int64)v314, (__int64)&v311);
      v14 = (const __m128i *)*((_QWORD *)&v320 + 1);
      v277 = (const __m128i *)v320;
      v320 = 0u;
      v278 = v14;
      v15 = v321;
      v321 = 0;
      v279 = v15;
      if ( v308 )
        j_j___libc_free_0(v308);
      if ( !v306[12] )
        _libc_free(v305);
      if ( (_QWORD)v320 )
        j_j___libc_free_0(v320);
      if ( !BYTE4(v313) )
        _libc_free(v311.m128i_u64[1]);
      if ( v298 )
        j_j___libc_free_0(v298);
      if ( !v293.m128i_i8[4] )
        _libc_free(v291);
      if ( *(_QWORD *)&v286[6] )
        j_j___libc_free_0(*(unsigned __int64 *)&v286[6]);
      if ( !BYTE12(v286[1]) )
        _libc_free(*((unsigned __int64 *)&v286[0] + 1));
      sub_C8CD80((__int64)&v229, (__int64)v234, (__int64)v266, v11, v12, v13);
      v19 = v271;
      v20 = v270;
      v235 = 0;
      v236 = 0;
      v237 = 0;
      v21 = v271 - (_QWORD)v270;
      if ( (const __m128i *)v271 == v270 )
      {
        v21 = 0;
        v23 = 0;
      }
      else
      {
        if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_313;
        v22 = sub_22077B0(v271 - (_QWORD)v270);
        v19 = v271;
        v20 = v270;
        v23 = v22;
      }
      v235 = v23;
      v236 = v23;
      v237 = v23 + v21;
      if ( v20 != (const __m128i *)v19 )
      {
        v24 = (__m128i *)v23;
        v25 = v20;
        do
        {
          if ( v24 )
          {
            *v24 = _mm_loadu_si128(v25);
            v17 = v25[1].m128i_i64[0];
            v24[1].m128i_i64[0] = v17;
          }
          v25 = (const __m128i *)((char *)v25 + 24);
          v24 = (__m128i *)((char *)v24 + 24);
        }
        while ( v25 != (const __m128i *)v19 );
        v23 += 8 * ((unsigned __int64)((char *)&v25[-2].m128i_u64[1] - (char *)v20) >> 3) + 24;
      }
      v20 = (const __m128i *)v238;
      v236 = v23;
      sub_C8CD80((__int64)v238, (__int64)v241, (__int64)v273, v23, v17, v18);
      v28 = v278;
      v19 = (unsigned __int64)v277;
      v242 = 0;
      v243 = 0;
      v244 = 0;
      v29 = (char *)v278 - (char *)v277;
      if ( v278 == v277 )
      {
        v31 = 0;
      }
      else
      {
        if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_313:
          sub_4261EA(v20, v19, v16);
        v30 = sub_22077B0((char *)v278 - (char *)v277);
        v28 = v278;
        v19 = (unsigned __int64)v277;
        v31 = v30;
      }
      v242 = v31;
      v32 = (__m128i *)v31;
      v243 = v31;
      v244 = v31 + v29;
      if ( v28 != (const __m128i *)v19 )
      {
        v33 = (const __m128i *)v19;
        do
        {
          if ( v32 )
          {
            *v32 = _mm_loadu_si128(v33);
            v26 = v33[1].m128i_i64[0];
            v32[1].m128i_i64[0] = v26;
          }
          v33 = (const __m128i *)((char *)v33 + 24);
          v32 = (__m128i *)((char *)v32 + 24);
        }
        while ( v33 != v28 );
        v32 = (__m128i *)(v31 + 8 * (((unsigned __int64)&v33[-2].m128i_u64[1] - v19) >> 3) + 24);
      }
      v34 = v236;
      v35 = v235;
      v243 = (unsigned __int64)v32;
      v36 = v1;
      v37 = v236 - v235;
      if ( (__m128i *)(v236 - v235) == (__m128i *)((char *)v32 - v31) )
        goto LABEL_52;
      do
      {
LABEL_38:
        v217 = *(_QWORD **)(v34 - 24);
        if ( v217[2] != v217[1] )
          goto LABEL_39;
        v46 = *v36;
        v280 = 0;
        v281 = &v285;
        v282 = 32;
        v283 = 0;
        v284 = 1;
        sub_30AB790(v217, v46, &v280);
        v252 = 0;
        v245 = 0;
        v253 = 0;
        v254 = 0;
        v255 = 0;
        v256 = 0;
        v257 = 0;
        v258 = 0;
        v259 = 0;
        v260 = 0;
        v261 = 0;
        v262 = 0;
        v263 = 0;
        v47 = (__int64 *)v217[5];
        v251 = 0;
        v48 = (__int64 *)v217[4];
        v246 = 0;
        v247 = 0;
        v248 = 0;
        v249 = 0;
        v250 = 0;
        v213 = v47;
        if ( v48 == v47 )
        {
          v57 = 1;
          LOBYTE(v49) = 0;
          goto LABEL_103;
        }
        v224 = v48;
        v49 = 0;
        v50 = v36;
        do
        {
          v51 = *v224;
          v52 = *(_QWORD *)(*v224 + 56);
          v53 = *v224 + 48;
          if ( v52 == v53 )
            goto LABEL_94;
          v219 = *v224;
          v36 = v50;
          v54 = v49;
          v55 = *v224 + 48;
          do
          {
            while ( 1 )
            {
              if ( !v52 )
                goto LABEL_316;
              LOBYTE(v53) = *(_BYTE *)(v52 - 24) == 85 || *(_BYTE *)(v52 - 24) == 34;
              if ( !(_BYTE)v53 )
                goto LABEL_84;
              v56 = *(_QWORD *)(v52 - 56);
              if ( v56 && !*(_BYTE *)v56 && *(_QWORD *)(v56 + 24) == *(_QWORD *)(v52 + 56) )
              {
                if ( *(_DWORD *)(v56 + 36) == 286 )
                  goto LABEL_96;
                if ( !(unsigned __int8)sub_DF9C30((__int64 *)v36[4], (_BYTE *)v56) )
                  break;
              }
              v54 = v53;
LABEL_84:
              v52 = *(_QWORD *)(v52 + 8);
              if ( v55 == v52 )
                goto LABEL_93;
            }
            v52 = *(_QWORD *)(v52 + 8);
          }
          while ( v55 != v52 );
LABEL_93:
          v49 = v54;
          v50 = v36;
          v51 = v219;
LABEL_94:
          sub_30ABD80(&v245, v51, v50[4], &v280, 0, 0);
          ++v224;
        }
        while ( v213 != v224 );
        v36 = v50;
        if ( v250 )
          goto LABEL_96;
        v57 = v249;
        if ( !(_DWORD)v249 )
          v57 = 1;
LABEL_103:
        v58 = sub_C52410();
        v59 = v58 + 1;
        v60 = sub_C959E0();
        v61 = (_QWORD *)v58[2];
        if ( v61 )
        {
          v62 = v58 + 1;
          do
          {
            while ( 1 )
            {
              v63 = v61[2];
              v64 = v61[3];
              if ( v60 <= v61[4] )
                break;
              v61 = (_QWORD *)v61[3];
              if ( !v64 )
                goto LABEL_108;
            }
            v62 = v61;
            v61 = (_QWORD *)v61[2];
          }
          while ( v63 );
LABEL_108:
          if ( v59 != v62 && v60 >= v62[4] )
            v59 = v62;
        }
        if ( v59 == (_QWORD *)((char *)sub_C52410() + 8) )
          goto LABEL_119;
        v65 = v59[7];
        if ( !v65 )
          goto LABEL_119;
        v66 = v59 + 6;
        do
        {
          while ( 1 )
          {
            v67 = *(_QWORD *)(v65 + 16);
            v68 = *(_QWORD *)(v65 + 24);
            if ( *(_DWORD *)(v65 + 32) >= dword_4FFE8C8 )
              break;
            v65 = *(_QWORD *)(v65 + 24);
            if ( !v68 )
              goto LABEL_117;
          }
          v66 = (_QWORD *)v65;
          v65 = *(_QWORD *)(v65 + 16);
        }
        while ( v67 );
LABEL_117:
        if ( v59 + 6 == v66 || dword_4FFE8C8 < *((_DWORD *)v66 + 8) || (v69 = qword_4FFE948, *((int *)v66 + 9) <= 0) )
LABEL_119:
          v69 = sub_DFB600(v36[4]);
        v208 = 1;
        if ( v69 >= v57 )
          v208 = v69 / v57;
        v70 = sub_C52410();
        v71 = v70 + 1;
        v72 = sub_C959E0();
        v73 = (_QWORD *)v70[2];
        if ( v73 )
        {
          v74 = v70 + 1;
          do
          {
            while ( 1 )
            {
              v75 = v73[2];
              v76 = v73[3];
              if ( v72 <= v73[4] )
                break;
              v73 = (_QWORD *)v73[3];
              if ( !v76 )
                goto LABEL_127;
            }
            v74 = v73;
            v73 = (_QWORD *)v73[2];
          }
          while ( v75 );
LABEL_127:
          if ( v74 != v71 && v72 >= v74[4] )
            v71 = v74;
        }
        if ( v71 == (_QWORD *)((char *)sub_C52410() + 8) )
          goto LABEL_138;
        v77 = v71[7];
        if ( !v77 )
          goto LABEL_138;
        v78 = v71 + 6;
        do
        {
          while ( 1 )
          {
            v79 = *(_QWORD *)(v77 + 16);
            v80 = *(_QWORD *)(v77 + 24);
            if ( *(_DWORD *)(v77 + 32) >= dword_4FFE708 )
              break;
            v77 = *(_QWORD *)(v77 + 24);
            if ( !v80 )
              goto LABEL_136;
          }
          v78 = (_QWORD *)v77;
          v77 = *(_QWORD *)(v77 + 16);
        }
        while ( v79 );
LABEL_136:
        if ( v78 == v71 + 6 || dword_4FFE708 < *((_DWORD *)v78 + 8) || (v81 = qword_4FFE788, *((int *)v78 + 9) <= 0) )
LABEL_138:
          v81 = sub_DFB660(v36[4]);
        if ( v208 <= v81 )
        {
          v82 = sub_DBB070(v36[3], (__int64)v217, 0);
          if ( !v82 || v82 >= v208 + 1 )
          {
            v304 = (__m128i *)v306;
            v305 = 0x1000000000LL;
            v199 = v217[5];
            if ( v217[4] != v199 )
            {
              v209 = v217[4];
              v83 = v36;
              v205 = 0;
              v220 = 0;
              v203 = v49;
              while ( 1 )
              {
                v84 = *(_QWORD *)(*(_QWORD *)v209 + 56LL);
                v225 = *(_QWORD *)v209 + 48LL;
                if ( v84 == v225 )
                  goto LABEL_166;
                while ( 1 )
                {
LABEL_145:
                  if ( !v84 )
                    goto LABEL_316;
                  v85 = *(_BYTE *)(v84 - 24);
                  if ( v85 != 61 )
                  {
                    if ( v85 != 62 )
                      goto LABEL_165;
                    v163 = sub_C52410();
                    v164 = v163 + 1;
                    v165 = sub_C959E0();
                    v166 = (_QWORD *)v163[2];
                    if ( v166 )
                    {
                      v167 = v163 + 1;
                      do
                      {
                        while ( 1 )
                        {
                          v168 = v166[2];
                          v169 = v166[3];
                          if ( v165 <= v166[4] )
                            break;
                          v166 = (_QWORD *)v166[3];
                          if ( !v169 )
                            goto LABEL_231;
                        }
                        v167 = v166;
                        v166 = (_QWORD *)v166[2];
                      }
                      while ( v168 );
LABEL_231:
                      if ( v167 != v164 && v165 >= v167[4] )
                        v164 = v167;
                    }
                    if ( v164 == (_QWORD *)((char *)sub_C52410() + 8) )
                      goto LABEL_163;
                    v170 = v164[7];
                    if ( !v170 )
                      goto LABEL_163;
                    v171 = v164 + 6;
                    do
                    {
                      if ( *(_DWORD *)(v170 + 32) < dword_4FFE9A8 )
                      {
                        v170 = *(_QWORD *)(v170 + 24);
                      }
                      else
                      {
                        v171 = (_QWORD *)v170;
                        v170 = *(_QWORD *)(v170 + 16);
                      }
                    }
                    while ( v170 );
                    if ( v171 == v164 + 6
                      || dword_4FFE9A8 < *((_DWORD *)v171 + 8)
                      || (v101 = qword_4FFEA28, *((int *)v171 + 9) <= 0) )
                    {
LABEL_163:
                      v101 = sub_DFB690(v83[4]);
                    }
                    if ( !v101 )
                      goto LABEL_165;
                  }
                  v86 = *(_QWORD *)(v84 - 56);
                  v87 = *(_QWORD *)(v86 + 8);
                  if ( (unsigned int)*(unsigned __int8 *)(v87 + 8) - 17 <= 1 )
                    v87 = **(_QWORD **)(v87 + 16);
                  if ( !sub_DFB6C0(v83[4], *(_DWORD *)(v87 + 8) >> 8) )
                    goto LABEL_165;
                  ++v220;
                  if ( (unsigned __int8)sub_D48480((__int64)v217, v86, v88, v89) )
                    goto LABEL_165;
                  v91 = sub_DD8400(v83[3], v86);
                  if ( *((_WORD *)v91 + 12) != 8 )
                    goto LABEL_165;
                  v92 = (unsigned int)v305;
                  v93 = (__int64)v304;
                  ++v205;
                  v201 = v84 - 24;
                  v94 = &v304[2 * (unsigned int)v305];
                  if ( v304 == v94 )
                    break;
                  v95 = v304;
                  while ( 1 )
                  {
                    v97 = sub_DCC810((__int64 *)v83[3], (__int64)v91, v95->m128i_i64[0], 0, 0);
                    if ( !*((_WORD *)v97 + 12) )
                    {
                      v98 = v97[4];
                      v99 = *(__int64 **)(v98 + 24);
                      v100 = *(_DWORD *)(v98 + 32);
                      if ( v100 > 0x40 )
                      {
                        v96 = *v99;
LABEL_156:
                        v214 = v96;
                        if ( (unsigned int)sub_DFB3D0((_QWORD *)v83[4]) > (__int64)abs64(v96) )
                          break;
                        goto LABEL_157;
                      }
                      if ( v100 )
                      {
                        v96 = (__int64)((_QWORD)v99 << (64 - (unsigned __int8)v100)) >> (64 - (unsigned __int8)v100);
                        goto LABEL_156;
                      }
                      v214 = 0;
                      if ( (unsigned int)sub_DFB3D0((_QWORD *)v83[4]) )
                        break;
                    }
LABEL_157:
                    v95 += 2;
                    if ( v94 == v95 )
                    {
                      v92 = (unsigned int)v305;
                      v94 = v304;
                      v186 = &v304[2 * (unsigned int)v305];
                      goto LABEL_271;
                    }
                  }
                  v172 = v95->m128i_i64[1];
                  if ( !v172 )
                  {
                    v95[1].m128i_i64[1] = v201;
                    v95->m128i_i64[1] = v201;
                    v95[1].m128i_i8[0] = *(_BYTE *)(v84 - 24) == 62;
                    v84 = *(_QWORD *)(v84 + 8);
                    if ( v225 != v84 )
                      continue;
                    goto LABEL_166;
                  }
                  v173 = *(_QWORD *)(v172 + 40);
                  v174 = *(_QWORD *)(v84 + 16);
                  if ( v173 != v174 )
                  {
                    v175 = v83[1];
                    v176 = *(__int64 **)(*(_QWORD *)(v173 + 72) + 80LL);
                    if ( v176 )
                    {
                      v177 = v176 - 3;
                      if ( (__int64 *)v173 != v177 )
                      {
                        if ( (__int64 *)v174 != v177 )
                        {
                          v178 = *(_DWORD *)(v175 + 32);
                          v179 = *(_DWORD *)(v173 + 44) + 1;
                          v176 = 0;
                          if ( v179 < v178 )
                            goto LABEL_250;
LABEL_251:
                          if ( v174 )
                            goto LABEL_252;
                          v180 = 0;
                          v181 = 0;
                          goto LABEL_253;
                        }
                        goto LABEL_260;
                      }
                    }
                    else
                    {
                      if ( !v174 )
                        BUG();
                      v178 = *(_DWORD *)(v175 + 32);
                      v179 = *(_DWORD *)(v173 + 44) + 1;
                      if ( v178 > v179 )
                      {
LABEL_250:
                        v176 = *(__int64 **)(*(_QWORD *)(v175 + 24) + 8LL * v179);
                        goto LABEL_251;
                      }
LABEL_252:
                      v180 = (unsigned int)(*(_DWORD *)(v174 + 44) + 1);
                      v181 = *(_DWORD *)(v174 + 44) + 1;
LABEL_253:
                      v182 = 0;
                      if ( v181 < v178 )
                        v182 = *(__int64 **)(*(_QWORD *)(v175 + 24) + 8 * v180);
                      while ( v176 != v182 )
                      {
                        if ( *((_DWORD *)v176 + 4) < *((_DWORD *)v182 + 4) )
                        {
                          v183 = v176;
                          v176 = v182;
                          v182 = v183;
                        }
                        v176 = (__int64 *)v176[1];
                      }
                      v177 = (__int64 *)*v182;
LABEL_260:
                      if ( (__int64 *)v173 != v177 )
                      {
                        v184 = v177[6] & 0xFFFFFFFFFFFFFFF8LL;
                        if ( (__int64 *)v184 == v177 + 6 )
                        {
                          v185 = 0;
                        }
                        else
                        {
                          if ( !v184 )
LABEL_316:
                            BUG();
                          v185 = v184 - 24;
                          if ( (unsigned int)*(unsigned __int8 *)(v184 - 24) - 30 >= 0xB )
                            v185 = 0;
                        }
                        v95->m128i_i64[1] = v185;
                      }
                    }
                  }
                  if ( *(_BYTE *)(v84 - 24) != 62 || v214 )
                  {
LABEL_165:
                    v84 = *(_QWORD *)(v84 + 8);
                    if ( v225 == v84 )
                      goto LABEL_166;
                  }
                  else
                  {
                    v95[1].m128i_i8[0] = 1;
                    v84 = *(_QWORD *)(v84 + 8);
                    if ( v225 == v84 )
                      goto LABEL_166;
                  }
                }
                v186 = &v304[2 * (unsigned int)v305];
LABEL_271:
                v311.m128i_i64[0] = (__int64)v91;
                v187 = &v311;
                v313 = v84 - 24;
                v311.m128i_i64[1] = v84 - 24;
                LOBYTE(v312) = *(_BYTE *)(v84 - 24) == 62;
                v188 = v92 + 1;
                if ( v188 > HIDWORD(v305) )
                {
                  if ( v94 > &v311 || v186 <= &v311 )
                  {
                    sub_C8D5F0((__int64)&v304, v306, v188, 0x20u, v90, v93);
                    v187 = &v311;
                    v186 = &v304[2 * (unsigned int)v305];
                  }
                  else
                  {
                    sub_C8D5F0((__int64)&v304, v306, v188, 0x20u, v90, v93);
                    v187 = (__m128i *)((char *)v304 + (char *)&v311 - (char *)v94);
                    v186 = &v304[2 * (unsigned int)v305];
                  }
                }
                *v186 = _mm_loadu_si128(v187);
                v186[1] = _mm_loadu_si128(v187 + 1);
                LODWORD(v305) = v305 + 1;
                v84 = *(_QWORD *)(v84 + 8);
                if ( v225 != v84 )
                  goto LABEL_145;
LABEL_166:
                v209 += 8;
                if ( v199 == v209 )
                {
                  LOBYTE(v49) = v203;
                  v102 = v305;
                  v36 = v83;
                  goto LABEL_168;
                }
              }
            }
            v205 = 0;
            v102 = 0;
            v220 = 0;
LABEL_168:
            v103 = sub_C52410();
            v104 = v103 + 1;
            v105 = sub_C959E0();
            v106 = (_QWORD *)v103[2];
            if ( v106 )
            {
              v107 = v103 + 1;
              do
              {
                while ( 1 )
                {
                  v108 = v106[2];
                  v109 = v106[3];
                  if ( v105 <= v106[4] )
                    break;
                  v106 = (_QWORD *)v106[3];
                  if ( !v109 )
                    goto LABEL_173;
                }
                v107 = v106;
                v106 = (_QWORD *)v106[2];
              }
              while ( v108 );
LABEL_173:
              if ( v107 != v104 && v105 >= v107[4] )
                v104 = v107;
            }
            if ( v104 == (_QWORD *)((char *)sub_C52410() + 8) )
              goto LABEL_304;
            *(_QWORD *)&v110 = v104[7];
            if ( !(_QWORD)v110 )
              goto LABEL_304;
            v111 = (unsigned int)dword_4FFE7E8;
            *((_QWORD *)&v110 + 1) = v104 + 6;
            do
            {
              while ( 1 )
              {
                v112 = *(_QWORD *)(v110 + 16);
                v113 = *(_QWORD *)(v110 + 24);
                if ( *(_DWORD *)(v110 + 32) >= dword_4FFE7E8 )
                  break;
                *(_QWORD *)&v110 = *(_QWORD *)(v110 + 24);
                if ( !v113 )
                  goto LABEL_182;
              }
              *((_QWORD *)&v110 + 1) = v110;
              *(_QWORD *)&v110 = *(_QWORD *)(v110 + 16);
            }
            while ( v112 );
LABEL_182:
            if ( *((_QWORD **)&v110 + 1) == v104 + 6
              || dword_4FFE7E8 < *(_DWORD *)(*((_QWORD *)&v110 + 1) + 32LL)
              || (v218 = qword_4FFE868, *(int *)(*((_QWORD *)&v110 + 1) + 36LL) <= 0) )
            {
LABEL_304:
              v218 = sub_DFB630((__int64 *)v36[4], v220, v205, v102, v49 & 1);
            }
            v114 = (__int64 *)v304;
            v226 = &v304[2 * (unsigned int)v305];
            if ( v304 != v226 )
            {
              v215 = 0;
              v115 = &v290;
              while ( 1 )
              {
                while ( v218 <= 1 )
                {
LABEL_194:
                  v117 = *(_QWORD *)(v114[1] + 40);
                  v118 = sub_AA4E30(v117);
                  v119 = (__int64 *)v36[3];
                  v312 = "prefaddr";
                  v324 = &v328;
                  v311.m128i_i64[0] = (__int64)v119;
                  v311.m128i_i64[1] = v118;
                  v333 = &v335;
                  v334 = 0x200000000LL;
                  LOBYTE(v313) = 1;
                  memset(v314, 0, sizeof(v314));
                  v315 = 0;
                  v316 = 0;
                  v317 = 0;
                  v318 = 0;
                  v319 = 0;
                  v320 = 0u;
                  v321 = 0;
                  v322 = 0;
                  v323 = 0;
                  v325 = 16;
                  v326 = 0;
                  v327 = 1;
                  v329 = 0;
                  v330 = 0;
                  v331 = 0;
                  v332 = 0;
                  v336 = 0;
                  v337 = 0;
                  v338 = 0;
                  v339 = 0;
                  v340 = 0;
                  v341 = &v345;
                  v264.m128i_i64[0] = (__int64)&v311;
                  *((_QWORD *)&v286[1] + 1) = sub_27BFDD0;
                  v120 = _mm_load_si128(&v264);
                  v352 = 1;
                  v265[1] = *(_QWORD *)&v286[2];
                  v290 = (unsigned __int64)&unk_49E5698;
                  v121 = _mm_loadu_si128((const __m128i *)((char *)v286 + 8));
                  *(_QWORD *)&v286[2] = sub_27BFD20;
                  *(__m128i *)((char *)v286 + 8) = v120;
                  v291 = (unsigned __int64)&unk_49D94D0;
                  v264 = v121;
                  v342 = 2;
                  v343 = 0;
                  v344 = 1;
                  v346 = 0;
                  v347 = 0;
                  v348 = 0;
                  v349 = 0;
                  v350 = 0;
                  v351 = 0;
                  v353 = 0;
                  *(_QWORD *)&v286[0] = &unk_49DA0D8;
                  v265[0] = 0;
                  v292 = v118;
                  v293 = (__m128i)(unsigned __int64)v118;
                  v294 = 0u;
                  v295 = 0u;
                  v296 = 0u;
                  LOWORD(v297) = 257;
                  v122 = sub_B2BE50(*v119);
                  v123 = _mm_loadu_si128(&v293);
                  v360 = v122;
                  v124 = _mm_loadu_si128(&v294);
                  v361 = v369;
                  v125 = _mm_loadu_si128(&v295);
                  v362 = &v375;
                  v126 = _mm_loadu_si128(&v296);
                  v365 = 512;
                  v359 = 0;
                  v355 = 0x200000000LL;
                  v369[2] = v292;
                  v354 = &v356;
                  v363 = 0;
                  v364 = 0;
                  v366 = 7;
                  v367 = 0;
                  v368 = 0;
                  v357 = 0;
                  v358 = 0;
                  v369[0] = &unk_49E5698;
                  v369[1] = &unk_49D94D0;
                  v374 = v297;
                  v375 = &unk_49DA0D8;
                  v377 = 0;
                  v370 = v123;
                  v371 = v124;
                  v372 = v125;
                  v373 = v126;
                  if ( *((_QWORD *)&v286[1] + 1) )
                  {
                    (*((void (__fastcall **)(char *, char *, __int64))&v286[1] + 1))(v376, (char *)v286 + 8, 2);
                    v378 = *(_QWORD *)&v286[2];
                    v377 = *((_QWORD *)&v286[1] + 1);
                  }
                  v290 = (unsigned __int64)&unk_49E5698;
                  v291 = (unsigned __int64)&unk_49D94D0;
                  nullsub_63();
                  nullsub_63();
                  sub_B32BF0(v286);
                  if ( v265[0] )
                    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v265[0])(&v264, &v264, 3);
                  v130 = (__int64 *)v36[3];
                  v379 = &v381;
                  v380 = 0x800000000LL;
                  v382 = byte_3F871B3;
                  v210 = sub_D33D80((_QWORD *)*v114, (__int64)v130, v127, v128, v129);
                  v221 = v36[3];
                  v131 = sub_D95540(**(_QWORD **)(*v114 + 32));
                  v132 = sub_DA2C50(v221, v131, v208, 0);
                  v290 = (unsigned __int64)&v292;
                  v292 = (__int64)v132;
                  v293.m128i_i64[0] = v210;
                  v291 = 0x200000002LL;
                  v133 = sub_DC8BD0(v130, (__int64)v115, 0, 0);
                  if ( (__int64 *)v290 != &v292 )
                  {
                    v211 = v133;
                    _libc_free(v290);
                    v133 = v211;
                  }
                  v134 = *v114;
                  v291 = 0x200000002LL;
                  v290 = (unsigned __int64)&v292;
                  v292 = v134;
                  v293.m128i_i64[0] = (__int64)v133;
                  v139 = sub_DC7EB0(v130, (__int64)v115, 0, 0);
                  if ( (__int64 *)v290 != &v292 )
                    _libc_free(v290);
                  v212 = sub_F80610((__int64)&v311, (__int64)v139, v135, v136, v137, v138);
                  if ( !v212 )
                  {
                    sub_27C20B0((__int64)&v311);
                    goto LABEL_192;
                  }
                  v140 = sub_D95540((__int64)v139);
                  if ( (unsigned int)*(unsigned __int8 *)(v140 + 8) - 17 <= 1 )
                    v140 = **(_QWORD **)(v140 + 16);
                  v222 = *(_DWORD *)(v140 + 8) >> 8;
                  v141 = (__int64 *)sub_AA48A0(v117);
                  v142 = sub_BCE3C0(v141, v222);
                  v143 = v198;
                  LOWORD(v143) = 0;
                  v198 = v143;
                  v223 = sub_F8DB90((__int64)&v311, (__int64)v139, v142, v114[1] + 24, 0);
                  sub_23D0AB0((__int64)&v264, v114[1], 0, 0, 0);
                  v144 = (_QWORD *)sub_AA48A0(v117);
                  v145 = sub_BCB2D0(v144);
                  v293.m128i_i16[4] = 257;
                  v146 = v145;
                  *(_QWORD *)&v286[0] = v223;
                  v147 = *((unsigned __int8 *)v114 + 16);
                  HIDWORD(v228) = 0;
                  *((_QWORD *)&v286[0] + 1) = sub_AD64C0(v145, v147, 0);
                  *(_QWORD *)&v286[1] = sub_AD64C0(v146, 3, 0);
                  *((_QWORD *)&v286[1] + 1) = sub_AD64C0(v146, 1, 0);
                  v227 = v223[1];
                  sub_B33D10((__int64)&v264, 0x11Eu, (__int64)&v227, 1, (int)v286, 4, v228, (__int64)v115);
                  v148 = *(_QWORD *)v36[5];
                  v216 = (__int64 *)v36[5];
                  v149 = sub_B2BE50(v148);
                  if ( !sub_B6EA50(v149) )
                  {
                    v189 = sub_B2BE50(v148);
                    v190 = sub_B6F970(v189);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v190 + 48LL))(v190) )
                      goto LABEL_219;
                  }
                  sub_B174A0((__int64)v115, (__int64)"loop-data-prefetch", (__int64)"Prefetched", 10, v114[3]);
                  sub_B18290((__int64)v115, "prefetched memory access", 0x18u);
                  v154 = _mm_loadu_si128(&v293);
                  si128 = _mm_load_si128((const __m128i *)&v294.m128i_u64[1]);
                  v156 = _mm_load_si128((const __m128i *)&v295.m128i_u64[1]);
                  DWORD2(v286[0]) = v291;
                  v157 = v297;
                  v158 = (unsigned __int64 *)v296.m128i_i64[1];
                  *(__m128i *)((char *)&v286[1] + 8) = v154;
                  BYTE12(v286[0]) = BYTE4(v291);
                  v286[3] = si128;
                  *(_QWORD *)&v286[1] = v292;
                  v286[4] = v156;
                  *(_QWORD *)&v286[0] = &unk_49D9D40;
                  *((_QWORD *)&v286[2] + 1) = v294.m128i_i64[0];
                  *(_QWORD *)&v286[5] = &v286[6];
                  *((_QWORD *)&v286[5] + 1) = 0x400000000LL;
                  if ( (_DWORD)v297 )
                  {
                    v191 = (__m128i *)&v286[6];
                    v192 = (unsigned int)v297;
                    if ( (unsigned int)v297 > 4 )
                    {
                      sub_11F02D0((__int64)&v286[5], (unsigned int)v297, v150, v151, v152, v153);
                      v191 = *(__m128i **)&v286[5];
                      v192 = (unsigned int)v297;
                    }
                    v158 = (unsigned __int64 *)(v296.m128i_i64[1] + 80 * v192);
                    if ( (unsigned __int64 *)v296.m128i_i64[1] != v158 )
                    {
                      v206 = v36;
                      v193 = v157;
                      v194 = v296.m128i_i64[1];
                      v202 = v115;
                      v195 = v191;
                      do
                      {
                        if ( v195 )
                        {
                          v195->m128i_i64[0] = (__int64)v195[1].m128i_i64;
                          sub_28025B0(v195->m128i_i64, *(_BYTE **)v194, *(_QWORD *)v194 + *(_QWORD *)(v194 + 8));
                          v195[2].m128i_i64[0] = (__int64)v195[3].m128i_i64;
                          sub_28025B0(
                            v195[2].m128i_i64,
                            *(_BYTE **)(v194 + 32),
                            *(_QWORD *)(v194 + 32) + *(_QWORD *)(v194 + 40));
                          v195[4] = _mm_loadu_si128((const __m128i *)(v194 + 64));
                        }
                        v194 += 80;
                        v195 += 5;
                      }
                      while ( v158 != (unsigned __int64 *)v194 );
                      v158 = (unsigned __int64 *)v296.m128i_i64[1];
                      DWORD2(v286[5]) = v193;
                      v287 = v301;
                      v36 = v206;
                      v115 = v202;
                      v288 = v302;
                      v289 = v303;
                      *(_QWORD *)&v286[0] = &unk_49D9D78;
                      v290 = (unsigned __int64)&unk_49D9D40;
                      if ( v296.m128i_i64[1] != v296.m128i_i64[1] + 80LL * (unsigned int)v297 )
                      {
                        v196 = (unsigned __int64 *)(v296.m128i_i64[1] + 80LL * (unsigned int)v297);
                        do
                        {
                          v196 -= 10;
                          v197 = v196[4];
                          if ( (unsigned __int64 *)v197 != v196 + 6 )
                            j_j___libc_free_0(v197);
                          if ( (unsigned __int64 *)*v196 != v196 + 2 )
                            j_j___libc_free_0(*v196);
                        }
                        while ( v158 != v196 );
                        v158 = (unsigned __int64 *)v296.m128i_i64[1];
                      }
                      goto LABEL_208;
                    }
                    DWORD2(v286[5]) = v157;
                  }
                  v287 = v301;
                  v288 = v302;
                  v289 = v303;
                  *(_QWORD *)&v286[0] = &unk_49D9D78;
LABEL_208:
                  if ( v158 != &v298 )
                    _libc_free((unsigned __int64)v158);
                  sub_1049740(v216, (__int64)v286);
                  v159 = *(unsigned __int64 **)&v286[5];
                  *(_QWORD *)&v286[0] = &unk_49D9D40;
                  v160 = 80LL * DWORD2(v286[5]);
                  v161 = (unsigned __int64 *)(*(_QWORD *)&v286[5] + v160);
                  if ( *(_QWORD *)&v286[5] != *(_QWORD *)&v286[5] + v160 )
                  {
                    do
                    {
                      v161 -= 10;
                      v162 = v161[4];
                      if ( (unsigned __int64 *)v162 != v161 + 6 )
                        j_j___libc_free_0(v162);
                      if ( (unsigned __int64 *)*v161 != v161 + 2 )
                        j_j___libc_free_0(*v161);
                    }
                    while ( v159 != v161 );
                    v161 = *(unsigned __int64 **)&v286[5];
                  }
                  if ( v161 != (unsigned __int64 *)&v286[6] )
                    _libc_free((unsigned __int64)v161);
LABEL_219:
                  nullsub_61();
                  v265[14] = &unk_49DA100;
                  nullsub_63();
                  if ( (_QWORD *)v264.m128i_i64[0] != v265 )
                    _libc_free(v264.m128i_u64[0]);
                  v114 += 4;
                  sub_27C20B0((__int64)&v311);
                  v215 = v212;
                  if ( v226 == (__m128i *)v114 )
                  {
LABEL_222:
                    v204 |= v215;
                    v226 = v304;
                    goto LABEL_223;
                  }
                }
                *(_QWORD *)&v110 = sub_D33D80((_QWORD *)*v114, v36[3], *((__int64 *)&v110 + 1), v111, v112);
                if ( !*(_WORD *)(v110 + 24) )
                {
                  v116 = *(_QWORD *)(v110 + 32);
                  *(_QWORD *)&v110 = *(_QWORD *)(v116 + 24);
                  *((_QWORD *)&v110 + 1) = *(unsigned int *)(v116 + 32);
                  if ( DWORD2(v110) > 0x40 )
                  {
                    *(_QWORD *)&v110 = *(_QWORD *)v110;
                  }
                  else
                  {
                    if ( !DWORD2(v110) )
                      goto LABEL_192;
                    v111 = (unsigned int)(64 - DWORD2(v110));
                    *(_QWORD *)&v110 = (__int64)((_QWORD)v110 << (64 - BYTE8(v110))) >> (64 - BYTE8(v110));
                  }
                  v110 = (__int64)v110;
                  if ( v218 <= (unsigned int)abs64(v110) )
                    goto LABEL_194;
                }
LABEL_192:
                v114 += 4;
                if ( v226 == (__m128i *)v114 )
                  goto LABEL_222;
              }
            }
LABEL_223:
            if ( v226 != (__m128i *)v306 )
              _libc_free((unsigned __int64)v226);
          }
        }
LABEL_96:
        sub_C7D6A0(v255, 24LL * v257, 8);
        if ( !v284 )
          _libc_free((unsigned __int64)v281);
        v34 = v236;
LABEL_39:
        while ( 2 )
        {
          v38 = *(_QWORD *)(v34 - 24);
          if ( !*(_BYTE *)(v34 - 8) )
          {
            v39 = *(__int64 **)(v38 + 8);
            *(_BYTE *)(v34 - 8) = 1;
            *(_QWORD *)(v34 - 16) = v39;
            if ( *(__int64 **)(v38 + 16) != v39 )
              goto LABEL_41;
LABEL_47:
            v236 -= 24;
            v35 = v235;
            v34 = v236;
            if ( v236 == v235 )
              goto LABEL_51;
            continue;
          }
          break;
        }
        while ( 1 )
        {
          v39 = *(__int64 **)(v34 - 16);
          if ( *(__int64 **)(v38 + 16) == v39 )
            goto LABEL_47;
LABEL_41:
          v40 = v39 + 1;
          *(_QWORD *)(v34 - 16) = v39 + 1;
          v41 = *v39;
          if ( v233 )
          {
            v42 = v230;
            v37 = v232;
            v40 = &v230[v232];
            if ( v230 != v40 )
            {
              while ( v41 != *v42 )
              {
                if ( v40 == ++v42 )
                  goto LABEL_77;
              }
              continue;
            }
LABEL_77:
            if ( v232 < v231 )
              break;
          }
          sub_C8CC70((__int64)&v229, v41, (__int64)v40, v37, v26, v27);
          if ( v43 )
            goto LABEL_50;
        }
        ++v232;
        *v40 = v41;
        ++v229;
LABEL_50:
        v311.m128i_i64[0] = v41;
        LOBYTE(v312) = 0;
        sub_2802B20((__int64)&v235, &v311);
        v35 = v235;
        v34 = v236;
LABEL_51:
        v31 = v242;
        v37 = v34 - v35;
      }
      while ( v34 - v35 != v243 - v242 );
LABEL_52:
      if ( v35 != v34 )
      {
        v44 = v31;
        do
        {
          v37 = *(_QWORD *)v44;
          if ( *(_QWORD *)v35 != *(_QWORD *)v44 )
            goto LABEL_38;
          v37 = *(unsigned __int8 *)(v35 + 16);
          if ( (_BYTE)v37 != *(_BYTE *)(v44 + 16) )
            goto LABEL_38;
          if ( (_BYTE)v37 )
          {
            v37 = *(_QWORD *)(v44 + 8);
            if ( *(_QWORD *)(v35 + 8) != v37 )
              goto LABEL_38;
          }
          v35 += 24LL;
          v44 += 24LL;
        }
        while ( v35 != v34 );
      }
      v1 = v36;
      if ( v31 )
        j_j___libc_free_0(v31);
      if ( !v240 )
        _libc_free(v239);
      if ( v235 )
        j_j___libc_free_0(v235);
      if ( !v233 )
        _libc_free((unsigned __int64)v230);
      if ( v277 )
        j_j___libc_free_0((unsigned __int64)v277);
      if ( !v275 )
        _libc_free(v274);
      if ( v270 )
        j_j___libc_free_0((unsigned __int64)v270);
      if ( !v268 )
        _libc_free(v267);
      if ( v200 == ++v207 )
        return v204;
    }
  }
  return 0;
}
