// Function: sub_3150D70
// Address: 0x3150d70
//
unsigned __int64 *__fastcall sub_3150D70(unsigned __int64 *a1, _BYTE *a2, int a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __m128i v7; // kr00_16
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  _QWORD *v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  _QWORD *v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rax
  __int64 v67; // r9
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  char v72; // bl
  __int64 v73; // rax
  _QWORD *v74; // rax
  _QWORD *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  _QWORD *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  _QWORD *v98; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  _QWORD *v104; // rax
  _QWORD *v105; // rax
  char v106; // bl
  __int64 v107; // rax
  _QWORD *v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  _QWORD *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rcx
  __int64 v121; // r8
  __int64 v122; // r9
  _QWORD *v123; // rax
  _QWORD *v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // r8
  __int64 v128; // r9
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rdx
  __int64 v133; // rdx
  __int64 *v134; // rdx
  __int64 v135; // rax
  _QWORD *v136; // rax
  _QWORD *v137; // rax
  _QWORD *v138; // rax
  __int64 v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rcx
  __int64 v142; // r8
  __int64 v143; // r9
  _QWORD *v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rcx
  __int64 v147; // r8
  __int64 v148; // r9
  _QWORD *v149; // rax
  __int64 v150; // rcx
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  __int64 v156; // r9
  _QWORD *v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rcx
  __int64 v160; // r8
  __int64 v161; // r9
  _QWORD *v162; // rax
  _QWORD *v163; // rax
  __int64 v164; // rdx
  __int64 v165; // rcx
  __int64 v166; // r8
  __int64 v167; // r9
  __int64 v168; // rdx
  __int64 v169; // rcx
  __int64 v170; // r8
  __int64 v171; // r9
  _QWORD *v172; // rax
  _QWORD *v173; // rax
  int v174; // [rsp+Ch] [rbp-1374h]
  __int64 v175; // [rsp+10h] [rbp-1370h]
  int v176; // [rsp+10h] [rbp-1370h]
  __int8 v177; // [rsp+18h] [rbp-1368h]
  _QWORD *v178; // [rsp+18h] [rbp-1368h]
  __int8 v179; // [rsp+18h] [rbp-1368h]
  __int16 v180; // [rsp+18h] [rbp-1368h]
  __int16 v181; // [rsp+18h] [rbp-1368h]
  char v182; // [rsp+20h] [rbp-1360h]
  _QWORD *v183; // [rsp+20h] [rbp-1360h]
  __int64 v184; // [rsp+28h] [rbp-1358h]
  __int64 v185; // [rsp+34h] [rbp-134Ch]
  int v186; // [rsp+3Ch] [rbp-1344h]
  __int64 v187; // [rsp+40h] [rbp-1340h]
  int v188; // [rsp+48h] [rbp-1338h]
  __int64 v189; // [rsp+4Ch] [rbp-1334h]
  int v190; // [rsp+54h] [rbp-132Ch]
  __int64 v191; // [rsp+58h] [rbp-1328h]
  int v192; // [rsp+60h] [rbp-1320h]
  __int64 v193; // [rsp+64h] [rbp-131Ch]
  int v194; // [rsp+6Ch] [rbp-1314h]
  __int64 v195; // [rsp+70h] [rbp-1310h]
  int v196; // [rsp+78h] [rbp-1308h]
  __int64 v197; // [rsp+7Ch] [rbp-1304h]
  int v198; // [rsp+84h] [rbp-12FCh]
  __int64 v199; // [rsp+88h] [rbp-12F8h]
  int v200; // [rsp+90h] [rbp-12F0h]
  __int64 v201; // [rsp+94h] [rbp-12ECh]
  int v202; // [rsp+9Ch] [rbp-12E4h]
  __int64 v203; // [rsp+A0h] [rbp-12E0h]
  int v204; // [rsp+A8h] [rbp-12D8h]
  __int64 v205; // [rsp+ACh] [rbp-12D4h]
  int v206; // [rsp+B4h] [rbp-12CCh]
  __int64 v207; // [rsp+B8h] [rbp-12C8h]
  int v208; // [rsp+C0h] [rbp-12C0h]
  __int64 v209; // [rsp+C4h] [rbp-12BCh]
  int v210; // [rsp+CCh] [rbp-12B4h]
  __int64 v211; // [rsp+D0h] [rbp-12B0h]
  int v212; // [rsp+D8h] [rbp-12A8h]
  __int64 v213; // [rsp+DCh] [rbp-12A4h]
  int v214; // [rsp+E4h] [rbp-129Ch]
  __int64 v215; // [rsp+E8h] [rbp-1298h]
  int v216; // [rsp+F0h] [rbp-1290h]
  __int64 v217; // [rsp+F4h] [rbp-128Ch]
  int v218; // [rsp+FCh] [rbp-1284h]
  unsigned __int64 v219; // [rsp+100h] [rbp-1280h] BYREF
  __int64 v220; // [rsp+108h] [rbp-1278h]
  _QWORD *v221; // [rsp+110h] [rbp-1270h] BYREF
  __int64 v222; // [rsp+118h] [rbp-1268h]
  __int64 v223; // [rsp+120h] [rbp-1260h]
  __int64 v224; // [rsp+128h] [rbp-1258h]
  __int64 v225; // [rsp+130h] [rbp-1250h]
  __m128i v226; // [rsp+140h] [rbp-1240h] BYREF
  __int64 v227; // [rsp+150h] [rbp-1230h]
  __int64 v228; // [rsp+158h] [rbp-1228h]
  __int64 v229; // [rsp+160h] [rbp-1220h]
  __m128i v230; // [rsp+170h] [rbp-1210h] BYREF
  __int64 v231; // [rsp+180h] [rbp-1200h] BYREF
  __int64 *v232; // [rsp+188h] [rbp-11F8h]
  __int64 v233; // [rsp+190h] [rbp-11F0h]
  int v234; // [rsp+1A0h] [rbp-11E0h]
  int v235; // [rsp+1B0h] [rbp-11D0h]
  unsigned __int64 v236[275]; // [rsp+1B8h] [rbp-11C8h] BYREF
  __int64 v237; // [rsp+A50h] [rbp-930h]
  int v238; // [rsp+A58h] [rbp-928h]
  __m128i v239; // [rsp+A60h] [rbp-920h] BYREF
  __int64 v240; // [rsp+A70h] [rbp-910h]
  __int64 *v241; // [rsp+A78h] [rbp-908h]
  __int64 *v242; // [rsp+A80h] [rbp-900h]
  __int64 v243; // [rsp+A88h] [rbp-8F8h]
  int v244; // [rsp+A90h] [rbp-8F0h]
  int v245; // [rsp+A98h] [rbp-8E8h] BYREF
  __int64 v246; // [rsp+AA0h] [rbp-8E0h]
  int *v247; // [rsp+AA8h] [rbp-8D8h]
  int *v248; // [rsp+AB0h] [rbp-8D0h]
  __int64 v249; // [rsp+AB8h] [rbp-8C8h]
  int v250; // [rsp+AC8h] [rbp-8B8h] BYREF
  __int64 v251; // [rsp+AD0h] [rbp-8B0h]
  int *v252; // [rsp+AD8h] [rbp-8A8h]
  int *v253; // [rsp+AE0h] [rbp-8A0h]
  __int64 v254; // [rsp+AE8h] [rbp-898h]
  char v255; // [rsp+AF0h] [rbp-890h]
  __int64 v256; // [rsp+1340h] [rbp-40h]
  int v257; // [rsp+1348h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v5 = (_QWORD *)sub_22077B0(0x10u);
  if ( v5 )
    *v5 = &unk_4A0CDF8;
  v239.m128i_i64[0] = (__int64)v5;
  sub_314D9D0(a1, (unsigned __int64 *)&v239);
  sub_23501E0(v239.m128i_i64);
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
    *v6 = &unk_4A0E678;
  v239.m128i_i64[0] = (__int64)v6;
  sub_314D9D0(a1, (unsigned __int64 *)&v239);
  sub_23501E0(v239.m128i_i64);
  if ( a3 != 2 )
  {
    if ( a3 == 1 )
    {
      v226 = 0u;
      v227 = 0;
      v228 = 0;
      v229 = 0;
      v239.m128i_i8[0] = 0;
      sub_3150CD0((unsigned __int64 *)&v226, v239.m128i_i8);
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      v239.m128i_i8[0] = 0;
      sub_314D8C0((unsigned __int64 *)&v230, v239.m128i_i8);
      sub_314D980((unsigned __int64 *)&v230);
      v108 = (_QWORD *)sub_22077B0(0x10u);
      if ( v108 )
        *v108 = &unk_4A121F8;
      v221 = v108;
      sub_2354930((__int64)&v239, &v221, 1, 0, 0, 0);
      sub_233F7D0((__int64 *)&v221);
      sub_2353940((unsigned __int64 *)&v230, v239.m128i_i64);
      sub_233F7F0((__int64)&v239.m128i_i64[1]);
      sub_233F7D0(v239.m128i_i64);
      sub_234D2B0((__int64)&v239, v230.m128i_i64, 0, 0);
      sub_235A8B0((unsigned __int64 *)&v226, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_234A9E0(&v239, (unsigned __int64 *)&v226);
      sub_2357280(a1, v239.m128i_i64);
      sub_233F000(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      sub_234A970((__int64)&v226);
      sub_314DB20(a1);
      sub_314DAD0(a1);
      LOBYTE(v187) = 0;
      HIDWORD(v187) = 1;
      LOBYTE(v188) = 0;
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      sub_F10C20((__int64)&v239, v187, v188);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v109, v110, v111, v112);
      sub_233BCC0((__int64)&v239);
      sub_28448C0(&v221, 1, 0);
      v181 = (__int16)v221;
      v113 = sub_22077B0(0x10u);
      if ( v113 )
      {
        *(_WORD *)(v113 + 8) = v181;
        *(_QWORD *)v113 = &unk_4A124B8;
      }
      v226.m128i_i64[0] = v113;
      sub_2354930((__int64)&v239, &v226, 0, 0, 0, 0);
      sub_233F7D0(v226.m128i_i64);
      sub_2353940((unsigned __int64 *)&v230, v239.m128i_i64);
      sub_233F7F0((__int64)&v239.m128i_i64[1]);
      sub_233F7D0(v239.m128i_i64);
      LOBYTE(v189) = 0;
      HIDWORD(v189) = 1;
      LOBYTE(v190) = 0;
      sub_F10C20((__int64)&v239, v189, v190);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v114, v115, v116, v117);
      sub_233BCC0((__int64)&v239);
      sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
      sub_23571D0(a1, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      v118 = (_QWORD *)sub_22077B0(0x10u);
      if ( v118 )
        *v118 = &unk_4A0E5B8;
      v239.m128i_i64[0] = (__int64)v118;
      sub_314D9D0(a1, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      sub_291E720(&v239, 0);
      sub_23A2000((unsigned __int64 *)&v230, v239.m128i_i8);
      v226.m128i_i16[6] = 0;
      v226.m128i_i32[2] = (int)&loc_1000000;
      v226.m128i_i64[0] = 256;
      sub_2339E50((__int64)&v239, 256, v226.m128i_i64[1] & 0xFFFFFFFFFFFFLL);
      sub_314D7D0((unsigned __int64 *)&v230, v239.m128i_i64, v119, v120, v121, v122);
      sub_2341D90((__int64)&v239);
      sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
      sub_23571D0(a1, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      v239.m128i_i8[0] = 1;
      sub_314DA70(a1, v239.m128i_i8);
      v226.m128i_i64[0] = 0x100010000000005LL;
      v226.m128i_i64[1] = 0x1000101000000LL;
      v239 = 0u;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v227 = 0;
      sub_29744D0((__int64)&v230, &v226);
      sub_23A1F80((unsigned __int64 *)&v239, v230.m128i_i64);
      v230.m128i_i8[0] = 0;
      sub_314D8C0((unsigned __int64 *)&v239, v230.m128i_i8);
      v230.m128i_i8[0] = 0;
      sub_23A2060((unsigned __int64 *)&v239, v230.m128i_i8);
      sub_234AAB0((__int64)&v230, v239.m128i_i64, 0);
      sub_23571D0(a1, v230.m128i_i64);
      sub_233EFE0(v230.m128i_i64);
      sub_233F7F0((__int64)&v239);
      v123 = (_QWORD *)sub_22077B0(0x10u);
      if ( v123 )
        *v123 = &unk_4A0CF78;
      v239.m128i_i64[0] = (__int64)v123;
      sub_314D9D0(a1, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      v124 = (_QWORD *)sub_22077B0(0x10u);
      if ( v124 )
        *v124 = &unk_4A0ED38;
      v239.m128i_i64[0] = (__int64)v124;
      sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      LOBYTE(v191) = 0;
      HIDWORD(v191) = 1;
      LOBYTE(v192) = 0;
      sub_F10C20((__int64)&v239, v191, v192);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v125, v126, v127, v128);
      sub_233BCC0((__int64)&v239);
      sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
      sub_23571D0(a1, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      v226 = 0u;
      v227 = 0;
      v228 = 0;
      v229 = 0;
      v239.m128i_i8[0] = 0;
      sub_3150CD0((unsigned __int64 *)&v226, v239.m128i_i8);
      v219 = __PAIR64__(qword_4FFDDA8[8], qword_4FFDE88[8]);
      LOWORD(v220) = 0;
      sub_2356430((__int64)&v230, (__int64 *)&v219, 1, 0, 0);
      v129 = v230.m128i_i64[0];
      v230.m128i_i64[0] = 0;
      v242 = 0;
      v239.m128i_i64[0] = v129;
      v243 = 0;
      v239.m128i_i64[1] = v230.m128i_i64[1];
      v230.m128i_i64[1] = 0;
      v240 = v231;
      v231 = 0;
      v241 = v232;
      v232 = 0;
      v244 = v234;
      v130 = sub_22077B0(0x40u);
      if ( v130 )
      {
        *(_QWORD *)(v130 + 40) = 0;
        *(_QWORD *)(v130 + 48) = 0;
        *(_QWORD *)v130 = &unk_4A11E38;
        v131 = v239.m128i_i64[0];
        v239.m128i_i64[0] = 0;
        *(_QWORD *)(v130 + 8) = v131;
        v132 = v239.m128i_i64[1];
        v239.m128i_i64[1] = 0;
        *(_QWORD *)(v130 + 16) = v132;
        v133 = v240;
        v240 = 0;
        *(_QWORD *)(v130 + 24) = v133;
        v134 = v241;
        v241 = 0;
        *(_QWORD *)(v130 + 32) = v134;
        *(_DWORD *)(v130 + 56) = v244;
      }
      v221 = (_QWORD *)v130;
      LOWORD(v222) = 0;
      sub_233F7F0((__int64)&v239.m128i_i64[1]);
      sub_233F7D0(v239.m128i_i64);
      sub_235A8B0((unsigned __int64 *)&v226, (__int64 *)&v221);
      sub_233EFE0((__int64 *)&v221);
      sub_233F7F0((__int64)&v230.m128i_i64[1]);
      sub_233F7D0(v230.m128i_i64);
      sub_234A9E0(&v239, (unsigned __int64 *)&v226);
      sub_2357280(a1, v239.m128i_i64);
      sub_233F000(v239.m128i_i64);
      sub_234A970((__int64)&v226);
      v239 = 0u;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v230.m128i_i16[0] = 1;
      sub_314D860((unsigned __int64 *)&v239, v230.m128i_i16);
      v182 = a2[5];
      v135 = sub_22077B0(0x10u);
      if ( v135 )
      {
        *(_BYTE *)(v135 + 8) = v182;
        *(_QWORD *)v135 = &unk_4A11DB8;
      }
      v230.m128i_i64[0] = v135;
      sub_314D790((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
      if ( v230.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v230.m128i_i64[0] + 8LL))(v230.m128i_i64[0]);
      v136 = (_QWORD *)sub_22077B0(0x10u);
      if ( v136 )
        *v136 = &unk_4A0FC78;
      v230.m128i_i64[0] = (__int64)v136;
      sub_314D790((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
      if ( v230.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v230.m128i_i64[0] + 8LL))(v230.m128i_i64[0]);
      v227 = 0;
      v226.m128i_i64[0] = 0x100010000000008LL;
      v226.m128i_i64[1] = 0x1000101010001LL;
      sub_29744D0((__int64)&v230, &v226);
      sub_23A1F80((unsigned __int64 *)&v239, v230.m128i_i64);
      sub_234AAB0((__int64)&v230, v239.m128i_i64, 0);
      sub_23571D0(a1, v230.m128i_i64);
      sub_233EFE0(v230.m128i_i64);
      sub_233F7F0((__int64)&v239);
      sub_23A0BA0((__int64)&v239, 0);
      sub_23A2670(a1, (__int64)&v239);
      sub_233AAF0((__int64)&v239);
      v137 = (_QWORD *)sub_22077B0(0x10u);
      if ( v137 )
        *v137 = &unk_4A0E578;
      v239.m128i_i64[0] = (__int64)v137;
      sub_314D9D0(a1, (unsigned __int64 *)&v239);
      sub_23501E0(v239.m128i_i64);
      v239 = 0u;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v138 = (_QWORD *)sub_22077B0(0x10u);
      if ( v138 )
        *v138 = &unk_4A10F78;
      v230.m128i_i64[0] = (__int64)v138;
      sub_314D790((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
      if ( v230.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v230.m128i_i64[0] + 8LL))(v230.m128i_i64[0]);
      v230.m128i_i8[0] = 0;
      sub_23A2060((unsigned __int64 *)&v239, v230.m128i_i8);
      sub_234AAB0((__int64)&v230, v239.m128i_i64, 0);
      sub_23571D0(a1, v230.m128i_i64);
      sub_233EFE0(v230.m128i_i64);
      sub_233F7F0((__int64)&v239);
      v226 = 0u;
      v227 = 0;
      v228 = 0;
      v229 = 0;
      v139 = sub_22077B0(0x10u);
      if ( v139 )
      {
        *(_DWORD *)(v139 + 8) = 2;
        *(_QWORD *)v139 = &unk_4A0EA78;
      }
      v239.m128i_i64[0] = v139;
      sub_3150C90((unsigned __int64 *)&v226, (unsigned __int64 *)&v239);
      sub_233F000(v239.m128i_i64);
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      v239.m128i_i8[0] = 0;
      sub_23A2060((unsigned __int64 *)&v230, v239.m128i_i8);
      LOBYTE(v193) = 0;
      HIDWORD(v193) = 1;
      LOBYTE(v194) = 0;
      sub_F10C20((__int64)&v239, v193, v194);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v140, v141, v142, v143);
      sub_233BCC0((__int64)&v239);
      sub_234D2B0((__int64)&v239, v230.m128i_i64, 0, 0);
      sub_235A8B0((unsigned __int64 *)&v226, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_234A9E0(&v239, (unsigned __int64 *)&v226);
      sub_2357280(a1, v239.m128i_i64);
      sub_233F000(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      sub_234A970((__int64)&v226);
      v144 = (_QWORD *)sub_22077B0(0x10u);
      if ( v144 )
        *v144 = &unk_4A0F1B8;
      v239.m128i_i64[0] = (__int64)v144;
      v239.m128i_i8[8] = 0;
      sub_23571D0(a1, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      v226 = 0u;
      v227 = 0;
      v228 = 0;
      v229 = 0;
      v239 = 0u;
      v240 = 0;
      v241 = 0;
      sub_235A960((unsigned __int64 *)&v226, v239.m128i_i64);
      sub_233F0C0(v239.m128i_i64);
      LOBYTE(v195) = 0;
      HIDWORD(v195) = 1;
      LOBYTE(v196) = 0;
      sub_F10C20((__int64)&v230, v195, v196);
      sub_314D600((__int64)&v239, (__int64)&v230, v145, v146, v147, v148);
      v256 = v237;
      v257 = v238;
      v149 = (_QWORD *)sub_22077B0(0x8F8u);
      if ( v149 )
      {
        v183 = v149;
        *v149 = &unk_4A11978;
        sub_314D600((__int64)(v149 + 1), (__int64)&v239, (__int64)&unk_4A11978, v150, v151, v152);
        v149 = v183;
        v183[285] = v256;
        *((_DWORD *)v183 + 572) = v257;
      }
      v221 = v149;
      LOWORD(v222) = 0;
      sub_233BCC0((__int64)&v239);
      sub_235A8B0((unsigned __int64 *)&v226, (__int64 *)&v221);
      sub_233EFE0((__int64 *)&v221);
      sub_233BCC0((__int64)&v230);
      sub_234A9E0(&v239, (unsigned __int64 *)&v226);
      sub_2357280(a1, v239.m128i_i64);
      sub_233F000(v239.m128i_i64);
      sub_234A970((__int64)&v226);
      v226.m128i_i64[0] = 256;
      v226.m128i_i16[6] = 0;
      v226.m128i_i32[2] = 0;
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      sub_2339E50((__int64)&v239, 256, v226.m128i_i64[1] & 0xFFFFFFFFFFFFLL);
      sub_314D7D0((unsigned __int64 *)&v230, v239.m128i_i64, v153, v154, v155, v156);
      sub_2341D90((__int64)&v239);
      v157 = (_QWORD *)sub_22077B0(0x10u);
      if ( v157 )
        *v157 = &unk_4A0F4B8;
      v239.m128i_i64[0] = (__int64)v157;
      sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
      sub_233EFE0(v239.m128i_i64);
      LOBYTE(v197) = 0;
      HIDWORD(v197) = 1;
      LOBYTE(v198) = 0;
      sub_F10C20((__int64)&v239, v197, v198);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v158, v159, v160, v161);
      sub_233BCC0((__int64)&v239);
      v162 = (_QWORD *)sub_22077B0(0x10u);
      if ( v162 )
        *v162 = &unk_4A117B8;
      v239.m128i_i64[0] = (__int64)v162;
      sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      sub_291E720(&v239, 0);
      sub_23A2000((unsigned __int64 *)&v230, v239.m128i_i8);
      sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
      sub_23571D0(a1, v239.m128i_i64);
      sub_233EFE0(v239.m128i_i64);
      sub_233F7F0((__int64)&v230);
      v239.m128i_i8[0] = 0;
      sub_314DA10(a1, v239.m128i_i8);
      sub_23A0BA0((__int64)&v239, 0);
      sub_23A2670(a1, (__int64)&v239);
      sub_233AAF0((__int64)&v239);
      v226.m128i_i64[0] = 0x100010000000008LL;
      v226.m128i_i64[1] = 0x1000101010001LL;
      v239 = 0u;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v227 = 0;
      sub_29744D0((__int64)&v230, &v226);
      sub_23A1F80((unsigned __int64 *)&v239, v230.m128i_i64);
      v230.m128i_i8[0] = 0;
      sub_314D8C0((unsigned __int64 *)&v239, v230.m128i_i8);
      sub_234AAB0((__int64)&v230, v239.m128i_i64, 0);
      sub_23571D0(a1, v230.m128i_i64);
      sub_233EFE0(v230.m128i_i64);
      sub_233F7F0((__int64)&v239);
      v163 = (_QWORD *)sub_22077B0(0x10u);
      if ( v163 )
        *v163 = &unk_4A0D3B8;
      v239.m128i_i64[0] = (__int64)v163;
      sub_314D9D0(a1, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      LOBYTE(v199) = 0;
      HIDWORD(v199) = 1;
      LOBYTE(v200) = 0;
      v230 = 0u;
      v231 = 0;
      v232 = 0;
      v233 = 0;
      sub_F10C20((__int64)&v239, v199, v200);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v164, v165, v166, v167);
      sub_233BCC0((__int64)&v239);
      v226.m128i_i16[4] = 1;
      v226.m128i_i64[0] = __PAIR64__(qword_4FFDDA8[8], qword_4FFDE88[8]);
      sub_2356430((__int64)&v239, v226.m128i_i64, 1, 0, 0);
      sub_2353940((unsigned __int64 *)&v230, v239.m128i_i64);
      sub_233F7F0((__int64)&v239.m128i_i64[1]);
      sub_233F7D0(v239.m128i_i64);
      v226.m128i_i8[0] = 0;
      v226.m128i_i32[1] = 1;
      v226.m128i_i8[8] = 0;
      sub_F10C20((__int64)&v239, v226.m128i_i64[0], v226.m128i_i32[2]);
      sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v168, v169, v170, v171);
      sub_233BCC0((__int64)&v239);
      v239.m128i_i8[0] = 0;
      sub_314D8C0((unsigned __int64 *)&v230, v239.m128i_i8);
      v172 = (_QWORD *)sub_22077B0(0x10u);
      if ( v172 )
        *v172 = &unk_4A11738;
      v239.m128i_i64[0] = (__int64)v172;
      sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
      if ( v239.m128i_i64[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
      v173 = (_QWORD *)sub_22077B0(0x10u);
      if ( v173 )
        *v173 = &unk_4A10D38;
      v239.m128i_i64[0] = (__int64)v173;
      sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
      sub_233EFE0(v239.m128i_i64);
      goto LABEL_88;
    }
    if ( a3 )
      BUG();
    v231 = 0;
    v230.m128i_i64[0] = 0x100010000000004LL;
    v230.m128i_i64[1] = 0x1000101000000LL;
    sub_29744D0((__int64)&v239, &v230);
    v7 = v239;
    v184 = v240;
    v8 = sub_22077B0(0x20u);
    if ( v8 )
    {
      *(__m128i *)(v8 + 8) = v7;
      *(_QWORD *)(v8 + 24) = v184;
      *(_QWORD *)v8 = &unk_4A11BB8;
    }
    v226.m128i_i64[0] = v8;
    v226.m128i_i8[8] = 0;
    sub_23571D0(a1, v226.m128i_i64);
    sub_233EFE0(v226.m128i_i64);
    sub_23A0BA0((__int64)&v239, 0);
    sub_23A2670(a1, (__int64)&v239);
    sub_233AAF0((__int64)&v239);
    LOBYTE(v201) = 0;
    HIDWORD(v201) = 1;
    LOBYTE(v202) = 0;
    v230 = 0u;
    v231 = 0;
    v232 = 0;
    v233 = 0;
    sub_F10C20((__int64)&v239, v201, v202);
    sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v9, v10, v11, v12);
    sub_233BCC0((__int64)&v239);
    v239.m128i_i8[0] = 0;
    sub_314D8C0((unsigned __int64 *)&v230, v239.m128i_i8);
    sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
    sub_23571D0(a1, v239.m128i_i64);
    sub_233EFE0(v239.m128i_i64);
    sub_233F7F0((__int64)&v230);
    sub_23A0BA0((__int64)&v239, 0);
    sub_23A2670(a1, (__int64)&v239);
    sub_233AAF0((__int64)&v239);
    v13 = (_QWORD *)sub_22077B0(0x10u);
    if ( v13 )
      *v13 = &unk_4A0E5F8;
    v239.m128i_i64[0] = (__int64)v13;
    sub_314D9D0(a1, (unsigned __int64 *)&v239);
    if ( v239.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
    v14 = (_QWORD *)sub_22077B0(0x10u);
    if ( v14 )
      *v14 = &unk_4A30EE0;
    v239.m128i_i64[0] = (__int64)v14;
    sub_314D9D0(a1, (unsigned __int64 *)&v239);
    if ( v239.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
    v230 = 0u;
    v231 = 0;
    v232 = 0;
    v233 = 0;
    v15 = (_QWORD *)sub_22077B0(0x10u);
    if ( v15 )
      *v15 = &unk_4A10F78;
    v239.m128i_i64[0] = (__int64)v15;
    sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
    sub_233EFE0(v239.m128i_i64);
    LOBYTE(v203) = 0;
    HIDWORD(v203) = 1;
    LOBYTE(v204) = 0;
    sub_F10C20((__int64)&v239, v203, v204);
    sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v16, v17, v18, v19);
    sub_233BCC0((__int64)&v239);
    v226.m128i_i64[0] = 0x100010000000005LL;
    v226.m128i_i64[1] = 0x1000101000001LL;
    v227 = 0;
    sub_29744D0((__int64)&v239, &v226);
    sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
    LOBYTE(v205) = 0;
    HIDWORD(v205) = 1;
    LOBYTE(v206) = 0;
    sub_F10C20((__int64)&v239, v205, v206);
    sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v20, v21, v22, v23);
    sub_233BCC0((__int64)&v239);
    sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
    sub_23571D0(a1, v239.m128i_i64);
    sub_233EFE0(v239.m128i_i64);
    sub_233F7F0((__int64)&v230);
    v24 = (_QWORD *)sub_22077B0(0x10u);
    if ( v24 )
      *v24 = &unk_4A0D3B8;
    v239.m128i_i64[0] = (__int64)v24;
    sub_314D9D0(a1, (unsigned __int64 *)&v239);
    if ( v239.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
    v25 = (_QWORD *)sub_22077B0(0x10u);
    if ( v25 )
      *v25 = &unk_4A0FFF8;
    v239.m128i_i64[0] = (__int64)v25;
    v239.m128i_i8[8] = 0;
    sub_23571D0(a1, v239.m128i_i64);
    sub_233EFE0(v239.m128i_i64);
    v239 = 0u;
    v240 = 0;
    v241 = 0;
    v242 = 0;
    v26 = sub_22077B0(0x10u);
    v27 = v26;
    if ( v26 )
    {
      *(_BYTE *)(v26 + 8) = 0;
      *(_QWORD *)v26 = &unk_4A0EC78;
    }
    v230.m128i_i64[0] = v26;
    if ( v239.m128i_i64[1] == v240 )
    {
      sub_235A6C0((unsigned __int64 *)&v239, (char *)v239.m128i_i64[1], &v230);
      v27 = v230.m128i_i64[0];
    }
    else
    {
      if ( v239.m128i_i64[1] )
      {
        *(_QWORD *)v239.m128i_i64[1] = v26;
        v239.m128i_i64[1] += 8;
LABEL_31:
        v28 = (_QWORD *)sub_22077B0(0x10u);
        if ( v28 )
          *v28 = &unk_4A10D38;
        v230.m128i_i64[0] = (__int64)v28;
        v230.m128i_i16[4] = 0;
        sub_235A8B0((unsigned __int64 *)&v239, v230.m128i_i64);
        sub_233EFE0(v230.m128i_i64);
        v29 = sub_22077B0(0x10u);
        if ( v29 )
        {
          *(_DWORD *)(v29 + 8) = 2;
          *(_QWORD *)v29 = &unk_4A0EA78;
        }
        v230.m128i_i64[0] = v29;
        sub_3150C90((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
        sub_233F000(v230.m128i_i64);
        sub_234A9E0(&v230, (unsigned __int64 *)&v239);
        sub_2357280(a1, v230.m128i_i64);
        sub_233F000(v230.m128i_i64);
        sub_234A970((__int64)&v239);
        v30 = (_QWORD *)sub_22077B0(0x10u);
        if ( v30 )
          *v30 = &unk_4A0E5B8;
        v239.m128i_i64[0] = (__int64)v30;
        sub_314D9D0(a1, (unsigned __int64 *)&v239);
        if ( v239.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
        v239.m128i_i32[2] = 0;
        v241 = &v239.m128i_i64[1];
        v242 = &v239.m128i_i64[1];
        v247 = &v245;
        v248 = &v245;
        v252 = &v250;
        v253 = &v250;
        v240 = 0;
        v243 = 0;
        v245 = 0;
        v246 = 0;
        v249 = 0;
        v250 = 0;
        v251 = 0;
        v254 = 0;
        v255 = 0;
        sub_2358990(a1, (__int64)&v239);
        sub_233A870(&v239);
        v31 = (_QWORD *)sub_22077B0(0x10u);
        if ( v31 )
          *v31 = &unk_4A0ED38;
        v239.m128i_i64[0] = (__int64)v31;
        v239.m128i_i8[8] = 0;
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        v32 = sub_22077B0(0x10u);
        if ( v32 )
        {
          *(_BYTE *)(v32 + 8) = 0;
          *(_QWORD *)v32 = &unk_4A0EC78;
        }
        v239.m128i_i64[0] = v32;
        sub_2357280(a1, v239.m128i_i64);
        sub_233F000(v239.m128i_i64);
        v230 = 0u;
        v231 = 0;
        v232 = 0;
        v233 = 0;
        v239.m128i_i8[0] = 0;
        sub_23A2060((unsigned __int64 *)&v230, v239.m128i_i8);
        v33 = (_QWORD *)sub_22077B0(0x10u);
        if ( v33 )
          *v33 = &unk_4A0F1B8;
        v239.m128i_i64[0] = (__int64)v33;
        sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
        sub_233EFE0(v239.m128i_i64);
        v226.m128i_i64[0] = 0x100010000000001LL;
        v226.m128i_i64[1] = 0x1000101000000LL;
        v227 = 0;
        sub_29744D0((__int64)&v239, &v226);
        sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
        v34 = (_QWORD *)sub_22077B0(0x10u);
        if ( v34 )
          *v34 = &unk_4A0FD78;
        v239.m128i_i64[0] = (__int64)v34;
        sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
        sub_233EFE0(v239.m128i_i64);
        v35 = sub_22077B0(0x10u);
        if ( v35 )
        {
          *(_BYTE *)(v35 + 8) = 1;
          *(_QWORD *)v35 = &unk_4A11F78;
        }
        v239 = (__m128i)(unsigned __int64)v35;
        v221 = 0;
        v240 = 0;
        v241 = 0;
        v242 = 0;
        v243 = 0;
        v244 = 0;
        v36 = (_QWORD *)sub_22077B0(0x10u);
        if ( v36 )
          *v36 = &unk_4A0B640;
        v226.m128i_i64[0] = (__int64)v36;
        sub_314D790(&v239.m128i_u64[1], (unsigned __int64 *)&v226);
        if ( v226.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v226.m128i_i64[0] + 8LL))(v226.m128i_i64[0]);
        v37 = (_QWORD *)sub_22077B0(0x10u);
        if ( v37 )
          *v37 = &unk_4A0B680;
        v226.m128i_i64[0] = (__int64)v37;
        sub_314D790(&v239.m128i_u64[1], (unsigned __int64 *)&v226);
        if ( v226.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v226.m128i_i64[0] + 8LL))(v226.m128i_i64[0]);
        sub_233F7D0((__int64 *)&v221);
        sub_2353940((unsigned __int64 *)&v230, v239.m128i_i64);
        sub_233F7F0((__int64)&v239.m128i_i64[1]);
        sub_233F7D0(v239.m128i_i64);
        sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        sub_233F7F0((__int64)&v230);
        sub_23A0BA0((__int64)&v239, 0);
        sub_23A2670(a1, (__int64)&v239);
        sub_233AAF0((__int64)&v239);
        LOBYTE(v207) = 0;
        HIDWORD(v207) = 1;
        LOBYTE(v208) = 0;
        v230 = 0u;
        v231 = 0;
        v232 = 0;
        v233 = 0;
        sub_F10C20((__int64)&v239, v207, v208);
        sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v38, v39, v40, v41);
        sub_233BCC0((__int64)&v239);
        v226.m128i_i64[0] = 0x100010000000007LL;
        v226.m128i_i64[1] = 0x1000101000000LL;
        v227 = 0;
        sub_29744D0((__int64)&v239, &v226);
        sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
        sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        sub_233F7F0((__int64)&v230);
        v239.m128i_i8[0] = 0;
        sub_314DA10(a1, v239.m128i_i8);
        sub_291E720(&v230, 0);
        v177 = v230.m128i_i8[0];
        v42 = sub_22077B0(0x10u);
        if ( v42 )
        {
          *(_BYTE *)(v42 + 8) = v177;
          *(_QWORD *)v42 = &unk_4A11C38;
        }
        v239.m128i_i64[0] = v42;
        v239.m128i_i8[8] = 0;
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        v43 = sub_22077B0(0x10u);
        if ( v43 )
        {
          *(_BYTE *)(v43 + 8) = 1;
          *(_QWORD *)v43 = &unk_4A0CDB8;
        }
        v239.m128i_i64[0] = v43;
        sub_314D9D0(a1, (unsigned __int64 *)&v239);
        if ( v239.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
        LOBYTE(v209) = 0;
        HIDWORD(v209) = 1;
        LOBYTE(v210) = 0;
        sub_F10C20((__int64)&v230, v209, v210);
        sub_314D600((__int64)&v239, (__int64)&v230, v44, v45, v46, v47);
        v256 = v237;
        v257 = v238;
        v48 = (_QWORD *)sub_22077B0(0x8F8u);
        if ( v48 )
        {
          v178 = v48;
          *v48 = &unk_4A11978;
          sub_314D600((__int64)(v48 + 1), (__int64)&v239, (__int64)&unk_4A11978, v49, v50, v51);
          v48 = v178;
          v178[285] = v256;
          *((_DWORD *)v178 + 572) = v257;
        }
        v226.m128i_i64[0] = (__int64)v48;
        v226.m128i_i8[8] = 0;
        sub_233BCC0((__int64)&v239);
        sub_23571D0(a1, v226.m128i_i64);
        sub_233EFE0(v226.m128i_i64);
        sub_233BCC0((__int64)&v230);
        v221 = 0;
        v222 = 0;
        v223 = 0;
        v224 = 0;
        v225 = 0;
        v239 = 0u;
        v240 = 0;
        v241 = 0;
        sub_235A960((unsigned __int64 *)&v221, v239.m128i_i64);
        if ( v239.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
        LOBYTE(v211) = 0;
        HIDWORD(v211) = 1;
        LOBYTE(v212) = 0;
        v226 = 0u;
        v227 = 0;
        v228 = 0;
        v229 = 0;
        sub_F10C20((__int64)&v239, v211, v212);
        sub_2353C90((unsigned __int64 *)&v226, (__int64)&v239, v52, v53, v54, v55);
        sub_233BCC0((__int64)&v239);
        sub_297B2F0((__int64)&v239, 1);
        v179 = v239.m128i_i8[0];
        v175 = v239.m128i_i64[1];
        v56 = sub_22077B0(0x18u);
        if ( v56 )
        {
          *(_BYTE *)(v56 + 8) = v179;
          *(_QWORD *)v56 = &unk_4A11BF8;
          *(_QWORD *)(v56 + 16) = v175;
        }
        v230.m128i_i64[0] = v56;
        sub_314D790((unsigned __int64 *)&v226, (unsigned __int64 *)&v230);
        sub_233EFE0(v230.m128i_i64);
        v230.m128i_i64[0] = (__int64)&v231;
        v230.m128i_i64[1] = 0x600000000LL;
        v235 = 0;
        memset(v236, 0, 48);
        sub_28448C0(&v219, 1, 0);
        sub_2332320((__int64)&v230, 0, v57, v58, v59, v60);
        v180 = v219;
        v61 = sub_22077B0(0x10u);
        if ( v61 )
        {
          *(_WORD *)(v61 + 8) = v180;
          *(_QWORD *)v61 = &unk_4A124B8;
        }
        v239.m128i_i64[0] = v61;
        sub_314DB70(v236, (unsigned __int64 *)&v239);
        sub_233F7D0(v239.m128i_i64);
        v176 = qword_4FFDE88[8];
        v174 = qword_4FFDDA8[8];
        sub_2332320((__int64)&v230, 0, v62, v63, v64, v65);
        v66 = sub_22077B0(0x18u);
        if ( v66 )
        {
          *(_DWORD *)(v66 + 8) = v176;
          *(_QWORD *)v66 = &unk_4A12478;
          *(_DWORD *)(v66 + 12) = v174;
          *(_WORD *)(v66 + 16) = 1;
        }
        v239.m128i_i64[0] = v66;
        sub_314DB70(v236, (unsigned __int64 *)&v239);
        sub_233F7D0(v239.m128i_i64);
        sub_23A20C0((__int64)&v239, (__int64)&v230, 1, 0, 0, v67);
        sub_2353940((unsigned __int64 *)&v226, v239.m128i_i64);
        sub_233F7F0((__int64)&v239.m128i_i64[1]);
        sub_233F7D0(v239.m128i_i64);
        WORD2(v220) = 0;
        LODWORD(v220) = 0;
        v219 = 0;
        sub_2339E50((__int64)&v239, 0, v220 & 0xFFFFFFFFFFFFLL);
        sub_314D7D0((unsigned __int64 *)&v226, v239.m128i_i64, v68, v69, v70, v71);
        sub_2341D90((__int64)&v239);
        v239 = (__m128i)0x10001000100uLL;
        v240 = 0x300000000LL;
        LODWORD(v241) = 0;
        sub_2353C00((unsigned __int64 *)&v226, &v239);
        sub_234D2B0((__int64)&v239, v226.m128i_i64, 0, 0);
        sub_235A8B0((unsigned __int64 *)&v221, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        sub_234A9E0(&v239, (unsigned __int64 *)&v221);
        sub_2357280(a1, v239.m128i_i64);
        sub_233F000(v239.m128i_i64);
        sub_2337B30((unsigned __int64 *)&v230);
        sub_233F7F0((__int64)&v226);
        sub_234A970((__int64)&v221);
        v239 = 0u;
        v240 = 0;
        v241 = 0;
        v242 = 0;
        sub_291E720(&v230, 0);
        sub_23A2000((unsigned __int64 *)&v239, v230.m128i_i8);
        v230.m128i_i16[0] = 1;
        sub_314D860((unsigned __int64 *)&v239, v230.m128i_i16);
        v72 = a2[5];
        v73 = sub_22077B0(0x10u);
        if ( v73 )
        {
          *(_BYTE *)(v73 + 8) = v72;
          *(_QWORD *)v73 = &unk_4A11DB8;
        }
        v230.m128i_i64[0] = v73;
        sub_314D790((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
        if ( v230.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v230.m128i_i64[0] + 8LL))(v230.m128i_i64[0]);
        v74 = (_QWORD *)sub_22077B0(0x10u);
        if ( v74 )
          *v74 = &unk_4A0FC78;
        v230.m128i_i64[0] = (__int64)v74;
        sub_314D790((unsigned __int64 *)&v239, (unsigned __int64 *)&v230);
        sub_233EFE0(v230.m128i_i64);
        sub_234AAB0((__int64)&v230, v239.m128i_i64, 0);
        sub_23571D0(a1, v230.m128i_i64);
        sub_233EFE0(v230.m128i_i64);
        sub_233F7F0((__int64)&v239);
        v75 = (_QWORD *)sub_22077B0(0x10u);
        if ( v75 )
          *v75 = &unk_4A0E2B8;
        v239.m128i_i64[0] = (__int64)v75;
        sub_314D9D0(a1, (unsigned __int64 *)&v239);
        if ( v239.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
        LOBYTE(v213) = 0;
        HIDWORD(v213) = 1;
        LOBYTE(v214) = 0;
        v230 = 0u;
        v231 = 0;
        v232 = 0;
        v233 = 0;
        sub_F10C20((__int64)&v239, v213, v214);
        sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v76, v77, v78, v79);
        sub_233BCC0((__int64)&v239);
        sub_27DC820((__int64)&v239, -1);
        sub_2354380((unsigned __int64 *)&v230, v239.m128i_i64);
        sub_233B480((__int64)&v239, (__int64)&v239, v80, v81, v82, v83);
        LOBYTE(v215) = 0;
        HIDWORD(v215) = 1;
        LOBYTE(v216) = 0;
        sub_F10C20((__int64)&v239, v215, v216);
        sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v84, v85, v86, v87);
        sub_233BCC0((__int64)&v239);
        v239.m128i_i8[0] = 0;
        sub_23A2060((unsigned __int64 *)&v230, v239.m128i_i8);
        LOBYTE(v217) = 0;
        HIDWORD(v217) = 1;
        LOBYTE(v218) = 0;
        sub_F10C20((__int64)&v239, v217, v218);
        sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v88, v89, v90, v91);
        sub_233BCC0((__int64)&v239);
        v92 = (_QWORD *)sub_22077B0(0x10u);
        if ( v92 )
          *v92 = &unk_4A117B8;
        v239.m128i_i64[0] = (__int64)v92;
        sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
        sub_233EFE0(v239.m128i_i64);
        v226.m128i_i64[0] = 0x100010000000002LL;
        v226.m128i_i64[1] = 0x1000101000001LL;
        v227 = 0;
        sub_29744D0((__int64)&v239, &v226);
        sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
        sub_291E720(&v239, 0);
        sub_23A2000((unsigned __int64 *)&v230, v239.m128i_i8);
        sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        sub_233F7F0((__int64)&v230);
        v93 = sub_22077B0(0x10u);
        if ( v93 )
        {
          *(_BYTE *)(v93 + 8) = 0;
          *(_QWORD *)v93 = &unk_4A0E8B8;
        }
        v239.m128i_i64[0] = v93;
        sub_314D9D0(a1, (unsigned __int64 *)&v239);
        sub_23501E0(v239.m128i_i64);
        LOBYTE(v221) = 0;
        HIDWORD(v221) = 1;
        LOBYTE(v222) = 0;
        v230 = 0u;
        v231 = 0;
        v232 = 0;
        v233 = 0;
        sub_F10C20((__int64)&v239, (__int64)v221, v222);
        sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v94, v95, v96, v97);
        sub_233BCC0((__int64)&v239);
        v226.m128i_i64[0] = 0x100010000000004LL;
        v226.m128i_i64[1] = 0x1000101010000LL;
        v227 = 0;
        sub_29744D0((__int64)&v239, &v226);
        sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
LABEL_88:
        sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
        sub_23571D0(a1, v239.m128i_i64);
        sub_233EFE0(v239.m128i_i64);
        goto LABEL_89;
      }
      v239.m128i_i64[1] = 8;
    }
    if ( v27 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
    goto LABEL_31;
  }
  v239.m128i_i8[0] = 0;
  sub_314DA10(a1, v239.m128i_i8);
  if ( !*a2 )
  {
    v106 = a2[5];
    v107 = sub_22077B0(0x10u);
    if ( v107 )
    {
      *(_BYTE *)(v107 + 8) = v106;
      *(_QWORD *)v107 = &unk_4A11DB8;
    }
    v239.m128i_i64[0] = v107;
    v239.m128i_i8[8] = 0;
    sub_23571D0(a1, v239.m128i_i64);
    sub_233EFE0(v239.m128i_i64);
  }
  sub_314DB20(a1);
  sub_314DAD0(a1);
  sub_23A0BA0((__int64)&v239, 0);
  sub_23A2670(a1, (__int64)&v239);
  sub_233AAF0((__int64)&v239);
  v226.m128i_i64[0] = 0x100000000000001LL;
  v226.m128i_i64[1] = 0x1000001000000LL;
  v230 = 0u;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v227 = 0;
  sub_29744D0((__int64)&v239, &v226);
  sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
  sub_314D980((unsigned __int64 *)&v230);
  LOBYTE(v185) = 0;
  HIDWORD(v185) = 1;
  LOBYTE(v186) = 0;
  sub_F10C20((__int64)&v239, v185, v186);
  sub_2353C90((unsigned __int64 *)&v230, (__int64)&v239, v100, v101, v102, v103);
  sub_233BCC0((__int64)&v239);
  v104 = (_QWORD *)sub_22077B0(0x10u);
  if ( v104 )
    *v104 = &unk_4A0ED38;
  v239.m128i_i64[0] = (__int64)v104;
  sub_314D790((unsigned __int64 *)&v230, (unsigned __int64 *)&v239);
  if ( v239.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
  v227 = 0;
  v226.m128i_i64[0] = 0x100000000000001LL;
  v226.m128i_i64[1] = 0x1000001000000LL;
  sub_29744D0((__int64)&v239, &v226);
  sub_23A1F80((unsigned __int64 *)&v230, v239.m128i_i64);
  sub_234AAB0((__int64)&v239, v230.m128i_i64, 0);
  sub_23571D0(a1, v239.m128i_i64);
  sub_233EFE0(v239.m128i_i64);
  v105 = (_QWORD *)sub_22077B0(0x10u);
  if ( v105 )
    *v105 = &unk_4A0E2B8;
  v239.m128i_i64[0] = (__int64)v105;
  sub_314D9D0(a1, (unsigned __int64 *)&v239);
  if ( v239.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v239.m128i_i64[0] + 8LL))(v239.m128i_i64[0]);
LABEL_89:
  sub_233F7F0((__int64)&v230);
  v98 = (_QWORD *)sub_22077B0(0x10u);
  if ( v98 )
    *v98 = &unk_4A0E538;
  v239.m128i_i64[0] = (__int64)v98;
  sub_314D9D0(a1, (unsigned __int64 *)&v239);
  sub_23501E0(v239.m128i_i64);
  return a1;
}
