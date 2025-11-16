// Function: sub_340D5E0
// Address: 0x340d5e0
//
__int64 __fastcall sub_340D5E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        unsigned __int8 a9,
        unsigned __int8 a10,
        __int64 a11,
        __int16 a12,
        __int128 a13,
        __int64 a14,
        __int128 a15,
        __int64 a16,
        const __m128i *a17)
{
  _QWORD *v17; // r15
  int v19; // eax
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 *v24; // rdi
  __int64 v25; // rax
  __m128i v26; // xmm4
  __m128i *v27; // rsi
  __m128i *v28; // rax
  __m128i v29; // xmm6
  __m128i *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __m128i v33; // xmm5
  __m128i *v34; // rsi
  char v35; // bl
  __int64 v36; // rsi
  __int64 v37; // r14
  __int64 *v38; // rdi
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int); // r12
  __int64 v40; // rax
  int v41; // edx
  unsigned __int16 v42; // ax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int16 v45; // si
  __int64 v46; // rax
  __int64 *v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  unsigned __int64 v52; // rdi
  unsigned int v53; // r14d
  __int64 v54; // rax
  __int64 v55; // rax
  void (***v56)(); // rdi
  void (*v57)(); // rax
  _WORD *v58; // rsi
  unsigned __int8 *v59; // rbx
  __int64 v60; // rbx
  unsigned int v61; // r12d
  unsigned __int64 v63; // rbx
  bool v64; // zf
  _DWORD *v65; // r13
  __int64 v66; // rax
  __int64 v67; // r12
  __int64 *v68; // rax
  __int64 v69; // rax
  char v70; // cl
  int v71; // eax
  __int64 v72; // rax
  bool v73; // r12
  unsigned __int8 v74; // dl
  __int64 v75; // rsi
  unsigned __int16 v76; // ax
  unsigned int v77; // edx
  unsigned int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __m128i v83; // xmm4
  unsigned __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r13
  __int64 v87; // r12
  char v88; // al
  __int16 v89; // bx
  __int16 v90; // r10
  __int16 v91; // r15
  unsigned __int64 v92; // rcx
  unsigned __int8 *v93; // rax
  __int64 v94; // rdx
  __m128i *v95; // r8
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r9
  unsigned __int64 v99; // rdx
  __m128i **v100; // rax
  __int64 v101; // rax
  unsigned __int64 v102; // r9
  unsigned __int64 v103; // rdx
  __m128i **v104; // rax
  unsigned __int32 v105; // eax
  __m128i v106; // xmm0
  __int64 v107; // rax
  __int64 v108; // rdx
  unsigned __int64 v109; // rdx
  __int64 v110; // rdx
  int v111; // eax
  __int64 v112; // rdx
  __int64 v113; // r13
  __int64 v114; // r12
  unsigned __int8 *v115; // rax
  __int64 v116; // rdx
  unsigned __int8 *v117; // rax
  unsigned __int64 v118; // rdx
  __m128i *v119; // r8
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // r9
  unsigned __int64 v123; // rdx
  __m128i **v124; // rax
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rdx
  unsigned __int64 v128; // rdx
  __int64 v129; // rdx
  int v130; // eax
  __m128i v131; // xmm5
  __int64 v132; // rsi
  unsigned __int8 v133; // bl
  __int64 v134; // r13
  int v135; // eax
  __int64 v136; // rdx
  __int128 v137; // [rsp+0h] [rbp-1510h]
  __int128 v138; // [rsp+0h] [rbp-1510h]
  __m128i *v139; // [rsp+10h] [rbp-1500h]
  unsigned __int64 v140; // [rsp+18h] [rbp-14F8h]
  __int64 v141; // [rsp+18h] [rbp-14F8h]
  unsigned int v142; // [rsp+20h] [rbp-14F0h]
  __int64 v143; // [rsp+38h] [rbp-14D8h]
  __int64 v144; // [rsp+40h] [rbp-14D0h]
  char v145; // [rsp+50h] [rbp-14C0h]
  __int64 v146; // [rsp+50h] [rbp-14C0h]
  __int64 *v147; // [rsp+58h] [rbp-14B8h]
  __int64 v148; // [rsp+60h] [rbp-14B0h]
  __int64 v149; // [rsp+68h] [rbp-14A8h]
  __int64 v150; // [rsp+70h] [rbp-14A0h]
  unsigned __int8 v151; // [rsp+78h] [rbp-1498h]
  __int64 v152; // [rsp+78h] [rbp-1498h]
  __int64 v153; // [rsp+80h] [rbp-1490h]
  unsigned __int64 v154; // [rsp+80h] [rbp-1490h]
  __int64 v155; // [rsp+88h] [rbp-1488h]
  unsigned __int64 v156; // [rsp+90h] [rbp-1480h]
  unsigned __int64 v157; // [rsp+90h] [rbp-1480h]
  unsigned __int8 v158; // [rsp+9Bh] [rbp-1475h]
  unsigned int v159; // [rsp+9Ch] [rbp-1474h]
  __int16 v160; // [rsp+9Ch] [rbp-1474h]
  __int64 v161; // [rsp+A0h] [rbp-1470h]
  unsigned __int8 (__fastcall *v162)(_DWORD *, unsigned __int64 *, _QWORD, __int64 **, _QWORD, _QWORD, __m128i *); // [rsp+A0h] [rbp-1470h]
  __int64 v163; // [rsp+A8h] [rbp-1468h]
  char v164; // [rsp+A8h] [rbp-1468h]
  unsigned int v165; // [rsp+A8h] [rbp-1468h]
  unsigned __int8 v166; // [rsp+B0h] [rbp-1460h]
  bool v167; // [rsp+B8h] [rbp-1458h]
  unsigned __int64 v168; // [rsp+B8h] [rbp-1458h]
  unsigned __int64 v169; // [rsp+B8h] [rbp-1458h]
  __m128i v170; // [rsp+C0h] [rbp-1450h] BYREF
  unsigned __int8 *v171; // [rsp+D0h] [rbp-1440h]
  __int64 v172; // [rsp+D8h] [rbp-1438h]
  __int64 v173; // [rsp+E0h] [rbp-1430h]
  __int64 v174; // [rsp+E8h] [rbp-1428h]
  __int64 v175; // [rsp+F0h] [rbp-1420h]
  __int64 v176; // [rsp+F8h] [rbp-1418h]
  __m128i v177; // [rsp+100h] [rbp-1410h]
  __m128i v178; // [rsp+110h] [rbp-1400h]
  __m128i v179; // [rsp+120h] [rbp-13F0h]
  __m128i v180; // [rsp+130h] [rbp-13E0h]
  __m128i *v181; // [rsp+140h] [rbp-13D0h]
  __int64 v182; // [rsp+148h] [rbp-13C8h]
  unsigned __int8 *v183; // [rsp+150h] [rbp-13C0h]
  __int64 v184; // [rsp+158h] [rbp-13B8h]
  __m128i *v185; // [rsp+160h] [rbp-13B0h]
  __int64 v186; // [rsp+168h] [rbp-13A8h]
  __m128i v187; // [rsp+170h] [rbp-13A0h] BYREF
  __int64 v188; // [rsp+180h] [rbp-1390h]
  __int64 v189; // [rsp+188h] [rbp-1388h]
  __m128i v190; // [rsp+190h] [rbp-1380h] BYREF
  __int64 v191; // [rsp+1A0h] [rbp-1370h]
  __int64 v192; // [rsp+1A8h] [rbp-1368h]
  __int64 v193; // [rsp+1B0h] [rbp-1360h]
  __int64 v194; // [rsp+1B8h] [rbp-1358h]
  __int64 v195; // [rsp+1C0h] [rbp-1350h]
  __int64 v196; // [rsp+1C8h] [rbp-1348h]
  __m128i v197; // [rsp+1D0h] [rbp-1340h] BYREF
  __int64 v198; // [rsp+1E0h] [rbp-1330h]
  __m128i v199; // [rsp+1F0h] [rbp-1320h] BYREF
  __int64 v200; // [rsp+200h] [rbp-1310h]
  unsigned __int64 v201; // [rsp+210h] [rbp-1300h] BYREF
  __int64 v202; // [rsp+218h] [rbp-12F8h]
  __int64 v203; // [rsp+220h] [rbp-12F0h]
  __int128 v204; // [rsp+230h] [rbp-12E0h]
  __int64 v205; // [rsp+240h] [rbp-12D0h]
  __int128 v206; // [rsp+250h] [rbp-12C0h] BYREF
  __int64 v207; // [rsp+260h] [rbp-12B0h]
  unsigned __int64 v208; // [rsp+270h] [rbp-12A0h] BYREF
  __m128i *v209; // [rsp+278h] [rbp-1298h]
  __m128i v210; // [rsp+280h] [rbp-1290h]
  unsigned __int8 **v211; // [rsp+290h] [rbp-1280h] BYREF
  __int64 v212; // [rsp+298h] [rbp-1278h]
  unsigned __int8 *v213; // [rsp+2A0h] [rbp-1270h] BYREF
  __m128i v214; // [rsp+320h] [rbp-11F0h] BYREF
  __m128i v215; // [rsp+330h] [rbp-11E0h] BYREF
  __m128i v216[7]; // [rsp+340h] [rbp-11D0h] BYREF
  __int64 *v217; // [rsp+3B0h] [rbp-1160h] BYREF
  __int64 v218; // [rsp+3B8h] [rbp-1158h]
  __int64 v219; // [rsp+3C0h] [rbp-1150h] BYREF
  unsigned __int64 v220; // [rsp+3C8h] [rbp-1148h]
  __int64 v221; // [rsp+3D0h] [rbp-1140h]
  __int64 v222; // [rsp+3D8h] [rbp-1138h]
  __int64 v223; // [rsp+3E0h] [rbp-1130h]
  unsigned __int64 v224; // [rsp+3E8h] [rbp-1128h] BYREF
  __m128i *v225; // [rsp+3F0h] [rbp-1120h]
  __int64 v226; // [rsp+3F8h] [rbp-1118h]
  _QWORD *v227; // [rsp+400h] [rbp-1110h]
  __int64 v228; // [rsp+408h] [rbp-1108h] BYREF
  int v229; // [rsp+410h] [rbp-1100h]
  __int64 v230; // [rsp+418h] [rbp-10F8h]
  _BYTE *v231; // [rsp+420h] [rbp-10F0h]
  __int64 v232; // [rsp+428h] [rbp-10E8h]
  _BYTE v233[1792]; // [rsp+430h] [rbp-10E0h] BYREF
  _BYTE *v234; // [rsp+B30h] [rbp-9E0h]
  __int64 v235; // [rsp+B38h] [rbp-9D8h]
  _BYTE v236[512]; // [rsp+B40h] [rbp-9D0h] BYREF
  _BYTE *v237; // [rsp+D40h] [rbp-7D0h]
  __int64 v238; // [rsp+D48h] [rbp-7C8h]
  _BYTE v239[1792]; // [rsp+D50h] [rbp-7C0h] BYREF
  _BYTE *v240; // [rsp+1450h] [rbp-C0h]
  __int64 v241; // [rsp+1458h] [rbp-B8h]
  _BYTE v242[64]; // [rsp+1460h] [rbp-B0h] BYREF
  __int64 v243; // [rsp+14A0h] [rbp-70h]
  __int64 v244; // [rsp+14A8h] [rbp-68h]
  int v245; // [rsp+14B0h] [rbp-60h]
  char v246; // [rsp+14D0h] [rbp-40h]

  v17 = (_QWORD *)a1;
  v172 = a3;
  v171 = (unsigned __int8 *)a2;
  v166 = a10;
  v19 = *(_DWORD *)(a8 + 24);
  v170.m128i_i64[0] = a5;
  v170.m128i_i64[1] = a6;
  if ( v19 != 35 && v19 != 11 )
    goto LABEL_3;
  v60 = *(_QWORD *)(a8 + 96);
  v61 = *(_DWORD *)(v60 + 32);
  if ( v61 <= 0x40 )
  {
    if ( !*(_QWORD *)(v60 + 24) )
      return (__int64)v171;
    v63 = *(_QWORD *)(v60 + 24);
  }
  else
  {
    if ( v61 == (unsigned int)sub_C444A0(v60 + 24) )
      return (__int64)v171;
    v63 = **(_QWORD **)(v60 + 24);
  }
  v155 = v170.m128i_i64[0];
  v64 = *(_DWORD *)(a7 + 24) == 51;
  v149 = v170.m128i_i64[1];
  v153 = *((_QWORD *)&a7 + 1);
  v150 = a7;
  v158 = a9;
  v197 = _mm_loadu_si128((const __m128i *)&a13);
  v198 = a14;
  v199 = _mm_loadu_si128((const __m128i *)&a15);
  v200 = a16;
  if ( v64 )
  {
    v59 = v171;
  }
  else
  {
    v65 = *(_DWORD **)(a1 + 16);
    v163 = a7;
    v66 = sub_2E79000(*(__int64 **)(a1 + 40));
    v67 = *(_QWORD *)(a1 + 40);
    v201 = 0;
    v148 = v66;
    v68 = *(__int64 **)(a1 + 64);
    v202 = 0;
    v147 = v68;
    v69 = *(_QWORD *)(v67 + 48);
    v203 = 0;
    v144 = v69;
    v70 = sub_33CC5F0((__int64 *)v67, a1);
    v167 = *(_DWORD *)(v170.m128i_i64[0] + 24) == 15 || *(_DWORD *)(v170.m128i_i64[0] + 24) == 39;
    if ( v167 )
    {
      v71 = *(_DWORD *)(v170.m128i_i64[0] + 96);
      if ( v71 < 0 )
      {
        v167 = v71 < -*(_DWORD *)(v144 + 32);
        v145 = v167;
      }
      else
      {
        v145 = 1;
      }
      v143 = v170.m128i_i64[0];
    }
    else
    {
      v145 = 0;
      v143 = 0;
    }
    v75 = v163;
    v164 = v70;
    v76 = sub_33E0440(a1, v75, v153);
    v151 = a9;
    if ( HIBYTE(v76) )
    {
      if ( a9 >= (unsigned __int8)v76 )
        LOBYTE(v76) = a9;
      v151 = v76;
    }
    v77 = v65[134250];
    if ( !v164 )
      v77 = v65[134249];
    v159 = v77;
    v162 = *(unsigned __int8 (__fastcall **)(_DWORD *, unsigned __int64 *, _QWORD, __int64 **, _QWORD, _QWORD, __m128i *))(*(_QWORD *)v65 + 1984LL);
    v214.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)v67 + 120LL);
    v165 = sub_2EAC1E0((__int64)&v199);
    v78 = sub_2EAC1E0((__int64)&v197);
    v217 = (__int64 *)v63;
    LOBYTE(v218) = v145;
    *(_DWORD *)((char *)&v218 + 2) = 0;
    BYTE1(v218) = a9;
    BYTE6(v218) = v151;
    if ( v162(v65, &v201, v159, &v217, v78, v165, &v214) )
    {
      if ( v167 )
      {
        v132 = sub_3007410(v201, v147, v79, v80, v81, v82);
        v133 = sub_AE5020(v148, v132);
        v134 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v67 + 16) + 200LL))(*(_QWORD *)(v67 + 16));
        if ( (!(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v134 + 544LL))(v134, v67)
           || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v134 + 536LL))(v134, v67))
          && *(_BYTE *)(v148 + 17)
          && v133 > *(_BYTE *)(v148 + 16) )
        {
          v133 = *(_BYTE *)(v148 + 16);
        }
        if ( a9 < v133 )
        {
          v135 = *(_DWORD *)(v143 + 96);
          v136 = *(_QWORD *)(v144 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v144 + 32) + v135);
          if ( *(_BYTE *)(v136 + 16) < v133 )
          {
            *(_BYTE *)(v136 + 16) = v133;
            v82 = *(_QWORD *)(v144 + 8);
            if ( (*(_BYTE *)(v82 + 40LL * (unsigned int)(*(_DWORD *)(v144 + 32) + v135) + 20) & 0xFD) == 0 )
              sub_2E76F70(v144, v133);
          }
          v158 = v133;
        }
      }
      v209 = 0;
      v83 = _mm_loadu_si128(a17 + 1);
      v214.m128i_i64[0] = (__int64)&v215;
      v208 = 0;
      v160 = 4 * (v166 != 0);
      v211 = &v213;
      v212 = 0x800000000LL;
      v214.m128i_i64[1] = 0x800000000LL;
      v218 = 0x800000000LL;
      v84 = v201;
      v217 = &v219;
      v210 = v83;
      v85 = (__int64)(v202 - v201) >> 4;
      v142 = v85;
      if ( (_DWORD)v85 )
      {
        v86 = 0;
        v87 = 0;
        v146 = 16LL * (unsigned int)(v85 - 1);
        while ( 1 )
        {
          v106 = _mm_loadu_si128((const __m128i *)(v84 + v86));
          v190 = v106;
          if ( v106.m128i_i16[0] )
          {
            if ( v106.m128i_i16[0] == 1 || (unsigned __int16)(v106.m128i_i16[0] - 504) <= 7u )
LABEL_157:
              BUG();
            v108 = 16LL * (v106.m128i_u16[0] - 1);
            v107 = *(_QWORD *)&byte_444C4A0[v108];
            LOBYTE(v108) = byte_444C4A0[v108 + 8];
          }
          else
          {
            v107 = sub_3007260((__int64)&v190);
            v193 = v107;
            v194 = v108;
          }
          BYTE8(v206) = v108;
          *(_QWORD *)&v206 = v107;
          v168 = (unsigned __int64)sub_CA1930(&v206) >> 3;
          v109 = v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v199.m128i_i8[0] & 4) != 0 )
            {
              *((_QWORD *)&v206 + 1) = v87 + v199.m128i_i64[1];
              BYTE4(v207) = BYTE4(v200);
              *(_QWORD *)&v206 = v109 | 4;
              LODWORD(v207) = *(_DWORD *)(v109 + 12);
            }
            else
            {
              *(_QWORD *)&v206 = v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              *((_QWORD *)&v206 + 1) = v87 + v199.m128i_i64[1];
              BYTE4(v207) = BYTE4(v200);
              v112 = *(_QWORD *)(v109 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v112 + 8) - 17 <= 1 )
                v112 = **(_QWORD **)(v112 + 16);
              LODWORD(v207) = *(_DWORD *)(v112 + 8) >> 8;
            }
          }
          else
          {
            BYTE4(v207) = 0;
            *(_QWORD *)&v206 = 0;
            *((_QWORD *)&v206 + 1) = v87 + v199.m128i_i64[1];
            LODWORD(v207) = v200;
          }
          v88 = sub_2EAC1F0((__int64 *)&v206, v168, (__int64)v147, v148);
          LOBYTE(v89) = v151;
          HIBYTE(v89) = 1;
          v90 = v160 | 0x10;
          if ( !v88 )
            v90 = 4 * (v166 != 0);
          v91 = v90;
          v92 = v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v199.m128i_i8[0] & 4) != 0 )
            {
              *((_QWORD *)&v204 + 1) = v87 + v199.m128i_i64[1];
              BYTE4(v205) = BYTE4(v200);
              *(_QWORD *)&v204 = v92 | 4;
              LODWORD(v205) = *(_DWORD *)(v92 + 12);
            }
            else
            {
              *((_QWORD *)&v204 + 1) = v87 + v199.m128i_i64[1];
              v110 = *(_QWORD *)(v92 + 8);
              *(_QWORD *)&v204 = v199.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              v111 = *(unsigned __int8 *)(v110 + 8);
              BYTE4(v205) = BYTE4(v200);
              if ( (unsigned int)(v111 - 17) <= 1 )
                v110 = **(_QWORD **)(v110 + 16);
              LODWORD(v205) = *(_DWORD *)(v110 + 8) >> 8;
            }
          }
          else
          {
            *((_QWORD *)&v204 + 1) = v87 + v199.m128i_i64[1];
            *(_QWORD *)&v204 = 0;
            LODWORD(v205) = v200;
            BYTE4(v205) = 0;
          }
          LOBYTE(v189) = 0;
          v188 = v87;
          v93 = sub_3409320((_QWORD *)a1, v150, v153, v87, 0, a4, v106, 0);
          v185 = sub_33F1F00(
                   (__int64 *)a1,
                   v190.m128i_u32[0],
                   v190.m128i_i64[1],
                   a4,
                   (__int64)v171,
                   v172,
                   (__int64)v93,
                   v94,
                   v204,
                   v205,
                   v89,
                   v91,
                   (__int64)&v208,
                   0);
          v95 = v185;
          v96 = (unsigned int)v212;
          v98 = (unsigned int)v97;
          v186 = v97;
          v99 = (unsigned int)v212 + 1LL;
          if ( v99 > HIDWORD(v212) )
          {
            v141 = v98;
            sub_C8D5F0((__int64)&v211, &v213, v99, 0x10u, (__int64)v185, v98);
            v96 = (unsigned int)v212;
            v95 = v185;
            v98 = v141;
          }
          v100 = (__m128i **)&v211[2 * v96];
          v100[1] = (__m128i *)v98;
          *v100 = v95;
          v101 = v214.m128i_u32[2];
          v102 = v156 & 0xFFFFFFFF00000000LL | 1;
          LODWORD(v212) = v212 + 1;
          v103 = v214.m128i_u32[2] + 1LL;
          v156 = v102;
          if ( v103 > v214.m128i_u32[3] )
          {
            v139 = v95;
            v140 = v102;
            sub_C8D5F0((__int64)&v214, &v215, v103, 0x10u, (__int64)v95, v102);
            v101 = v214.m128i_u32[2];
            v95 = v139;
            v102 = v140;
          }
          v104 = (__m128i **)(v214.m128i_i64[0] + 16 * v101);
          *v104 = v95;
          v104[1] = (__m128i *)v102;
          v87 += (unsigned int)v168;
          v105 = ++v214.m128i_i32[2];
          if ( v86 == v146 )
            break;
          v84 = v201;
          v86 += 16;
        }
        *((_QWORD *)&v137 + 1) = v105;
        v17 = (_QWORD *)a1;
        v113 = 0;
        *(_QWORD *)&v137 = v214.m128i_i64[0];
        v114 = 0;
        v115 = sub_33FC220((_QWORD *)a1, 2, a4, 1, 0, v102, v137);
        LODWORD(v218) = 0;
        v154 = (unsigned __int64)v115;
        v183 = v115;
        v184 = v116;
        v157 = (unsigned int)v116 | v172 & 0xFFFFFFFF00000000LL;
        do
        {
          v187 = _mm_loadu_si128((const __m128i *)(v201 + v113 * 8));
          if ( v187.m128i_i16[0] )
          {
            if ( v187.m128i_i16[0] == 1 || (unsigned __int16)(v187.m128i_i16[0] - 504) <= 7u )
              goto LABEL_157;
            v127 = 16LL * (v187.m128i_u16[0] - 1);
            v126 = *(_QWORD *)&byte_444C4A0[v127];
            LOBYTE(v127) = byte_444C4A0[v127 + 8];
          }
          else
          {
            v126 = sub_3007260((__int64)&v187);
            v195 = v126;
            v196 = v127;
          }
          v190.m128i_i8[8] = v127;
          v190.m128i_i64[0] = v126;
          v169 = (unsigned __int64)sub_CA1930(&v190) >> 3;
          v128 = v197.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v197.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v197.m128i_i8[0] & 4) != 0 )
            {
              *((_QWORD *)&v206 + 1) = v114 + v197.m128i_i64[1];
              BYTE4(v207) = BYTE4(v198);
              *(_QWORD *)&v206 = v128 | 4;
              LODWORD(v207) = *(_DWORD *)(v128 + 12);
            }
            else
            {
              *(_QWORD *)&v206 = v197.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              *((_QWORD *)&v206 + 1) = v114 + v197.m128i_i64[1];
              BYTE4(v207) = BYTE4(v198);
              v129 = *(_QWORD *)(v128 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v129 + 8) - 17 <= 1 )
                v129 = **(_QWORD **)(v129 + 16);
              LODWORD(v207) = *(_DWORD *)(v129 + 8) >> 8;
            }
          }
          else
          {
            BYTE4(v207) = 0;
            *(_QWORD *)&v206 = 0;
            *((_QWORD *)&v206 + 1) = v114 + v197.m128i_i64[1];
            LODWORD(v207) = v198;
          }
          LOBYTE(v192) = 0;
          v191 = v114;
          v117 = sub_3409320((_QWORD *)a1, v155, v149, v114, 0, a4, v106, 0);
          v181 = sub_33F4560(
                   (_QWORD *)a1,
                   v154,
                   v157,
                   a4,
                   (unsigned __int64)v211[v113],
                   (unsigned __int64)v211[v113 + 1],
                   (unsigned __int64)v117,
                   v118,
                   v206,
                   v207,
                   v158,
                   v160,
                   (__int64)&v208);
          v119 = v181;
          v120 = (unsigned int)v218;
          v182 = v121;
          v122 = (unsigned int)v121;
          v123 = (unsigned int)v218 + 1LL;
          if ( v123 > HIDWORD(v218) )
          {
            v152 = v122;
            sub_C8D5F0((__int64)&v217, &v219, v123, 0x10u, (__int64)v181, v122);
            v120 = (unsigned int)v218;
            v119 = v181;
            v122 = v152;
          }
          v124 = (__m128i **)&v217[2 * v120];
          v113 += 2;
          *v124 = v119;
          v124[1] = (__m128i *)v122;
          v114 += (unsigned int)v169;
          v125 = (unsigned int)(v218 + 1);
          LODWORD(v218) = v218 + 1;
        }
        while ( 2LL * v142 != v113 );
      }
      else
      {
        sub_33FC220((_QWORD *)a1, 2, a4, 1, 0, v82, (unsigned __int64)&v215);
        v125 = 0;
        LODWORD(v218) = 0;
      }
      *((_QWORD *)&v138 + 1) = v125;
      *(_QWORD *)&v138 = v217;
      v59 = sub_33FC220(v17, 2, a4, 1, 0, v122, v138);
      if ( v217 != &v219 )
        _libc_free((unsigned __int64)v217);
      if ( (__m128i *)v214.m128i_i64[0] != &v215 )
        _libc_free(v214.m128i_u64[0]);
      if ( v211 != &v213 )
        _libc_free((unsigned __int64)v211);
    }
    else
    {
      v59 = 0;
    }
    if ( v201 )
      j_j___libc_free_0(v201);
  }
  if ( !v59 )
  {
LABEL_3:
    v20 = v17[1];
    if ( v20 )
    {
      v21 = *(__int64 (**)())(*(_QWORD *)v20 + 48LL);
      if ( v21 != sub_33C7CF0 )
      {
        v59 = (unsigned __int8 *)((__int64 (__fastcall *)(__int64, _QWORD *, __int64, unsigned __int8 *, __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD, __int64))v21)(
                                   v20,
                                   v17,
                                   a4,
                                   v171,
                                   v172,
                                   a9,
                                   v170.m128i_i64[0],
                                   v170.m128i_i64[1],
                                   a7,
                                   *((_QWORD *)&a7 + 1),
                                   a8,
                                   *((_QWORD *)&a8 + 1),
                                   v166,
                                   a13,
                                   *((_QWORD *)&a13 + 1),
                                   a14,
                                   a15,
                                   *((_QWORD *)&a15 + 1),
                                   a16);
        if ( v59 )
          return (__int64)v59;
      }
    }
    v22 = sub_2EAC1E0((__int64)&a13);
    sub_33C8580(v17[2], v22);
    v23 = sub_2EAC1E0((__int64)&a15);
    sub_33C8580(v17[2], v23);
    v24 = (__int64 *)v17[8];
    v208 = 0;
    v209 = 0;
    v210.m128i_i64[0] = 0;
    v214 = 0u;
    v215 = 0u;
    v216[0] = 0u;
    v25 = sub_BCE3C0(v24, 0);
    v26 = _mm_load_si128(&v170);
    v27 = v209;
    v215.m128i_i64[1] = v25;
    v180 = v26;
    v214.m128i_i64[1] = v170.m128i_i64[0];
    v215.m128i_i32[0] = v26.m128i_i32[2];
    v28 = (__m128i *)v210.m128i_i64[0];
    if ( v209 == (__m128i *)v210.m128i_i64[0] )
    {
      sub_332CDC0(&v208, v209, &v214);
      v131 = _mm_loadu_si128((const __m128i *)&a7);
      v30 = v209;
      v214.m128i_i64[1] = a7;
      v178 = v131;
      v215.m128i_i32[0] = v131.m128i_i32[2];
      if ( v209 != (__m128i *)v210.m128i_i64[0] )
      {
        if ( !v209 )
          goto LABEL_10;
        goto LABEL_9;
      }
    }
    else
    {
      if ( v209 )
      {
        *v209 = _mm_load_si128(&v214);
        v27[1] = _mm_load_si128(&v215);
        v27[2] = _mm_load_si128(v216);
        v27 = v209;
        v28 = (__m128i *)v210.m128i_i64[0];
      }
      v29 = _mm_loadu_si128((const __m128i *)&a7);
      v30 = v27 + 3;
      v209 = v30;
      v214.m128i_i64[1] = a7;
      v179 = v29;
      v215.m128i_i32[0] = v29.m128i_i32[2];
      if ( v30 != v28 )
      {
LABEL_9:
        *v30 = _mm_load_si128(&v214);
        v30[1] = _mm_load_si128(&v215);
        v30[2] = _mm_load_si128(v216);
        v30 = v209;
LABEL_10:
        v209 = v30 + 3;
LABEL_11:
        v31 = sub_2E79000((__int64 *)v17[5]);
        v32 = sub_AE4420(v31, v17[8], 0);
        v33 = _mm_loadu_si128((const __m128i *)&a8);
        v34 = v209;
        v215.m128i_i64[1] = v32;
        v177 = v33;
        v214.m128i_i64[1] = a8;
        v215.m128i_i32[0] = v33.m128i_i32[2];
        if ( v209 == (__m128i *)v210.m128i_i64[0] )
        {
          sub_332CDC0(&v208, v209, &v214);
        }
        else
        {
          if ( v209 )
          {
            *v209 = _mm_load_si128(&v214);
            v34[1] = _mm_load_si128(&v215);
            v34[2] = _mm_load_si128(v216);
            v34 = v209;
          }
          v209 = v34 + 3;
        }
        v246 = 0;
        v220 = 0xFFFFFFFF00000020LL;
        v231 = v233;
        v232 = 0x2000000000LL;
        v235 = 0x2000000000LL;
        v238 = 0x2000000000LL;
        v240 = v242;
        v241 = 0x400000000LL;
        v237 = v239;
        v35 = HIBYTE(a12);
        v217 = 0;
        v218 = 0;
        v219 = 0;
        v221 = 0;
        v222 = 0;
        v223 = 0;
        v224 = 0;
        v225 = 0;
        v226 = 0;
        v227 = v17;
        v228 = 0;
        v229 = 0;
        v230 = 0;
        v234 = v236;
        v243 = 0;
        v244 = 0;
        v245 = 0;
        if ( HIBYTE(a12) )
        {
          v35 = a12;
        }
        else
        {
          v72 = v17[2];
          v73 = 0;
          if ( *(_QWORD *)(v72 + 529040) )
          {
            v161 = *(_QWORD *)(v72 + 529040);
            if ( strlen((const char *)v161) == 7 )
            {
              if ( *(_DWORD *)v161 != 1835885933
                || *(_WORD *)(v161 + 4) != 30319
                || (v130 = 0, *(_BYTE *)(v161 + 6) != 101) )
              {
                v130 = 1;
              }
              v73 = v130 == 0;
            }
          }
          if ( a11 )
          {
            v74 = sub_34B9CE0(a11);
            if ( (*(_WORD *)(a11 + 2) & 3u) - 1 <= 1 )
              v35 = sub_34B9AF0(a11, *v17, v73 & v74);
            if ( v228 )
              sub_B91220((__int64)&v228, v228);
          }
        }
        v36 = *(_QWORD *)a4;
        v228 = v36;
        if ( v36 )
          sub_B96E90((__int64)&v228, v36, 1);
        v229 = *(_DWORD *)(a4 + 8);
        v176 = v172;
        v175 = (__int64)v171;
        v37 = v17[2];
        v217 = (__int64 *)v171;
        v38 = (__int64 *)v17[5];
        LODWORD(v218) = v172;
        v39 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v37 + 32LL);
        v40 = sub_2E79000(v38);
        if ( v39 == sub_2D42F30 )
        {
          v41 = sub_AE2980(v40, 0)[1];
          v42 = 2;
          if ( v41 != 1 )
          {
            v42 = 3;
            if ( v41 != 2 )
            {
              v42 = 4;
              if ( v41 != 4 )
              {
                v42 = 5;
                if ( v41 != 8 )
                {
                  v42 = 6;
                  if ( v41 != 16 )
                  {
                    v42 = 7;
                    if ( v41 != 32 )
                    {
                      v42 = 8;
                      if ( v41 != 64 )
                        v42 = 9 * (v41 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v42 = v39(v37, v40, 0);
        }
        v172 = sub_33EED90((__int64)v17, *(const char **)(v17[2] + 529040LL), v42, 0);
        v171 = (unsigned __int8 *)v43;
        v44 = *(_QWORD *)(v170.m128i_i64[0] + 48) + 16LL * v170.m128i_u32[2];
        v45 = *(_WORD *)v44;
        v46 = *(_QWORD *)(v44 + 8);
        LOWORD(v211) = v45;
        v47 = (__int64 *)v17[8];
        v212 = v46;
        v51 = sub_3007410((__int64)&v211, v47, v170.m128i_i64[1], v48, v49, v50);
        v52 = v224;
        v53 = *(_DWORD *)(v17[2] + 533004LL);
        v219 = v51;
        v173 = v172;
        v222 = v172;
        v174 = (__int64)v171;
        LODWORD(v221) = v53;
        LODWORD(v223) = (_DWORD)v171;
        v224 = v208;
        v54 = (__int64)((__int64)v209->m128i_i64 - v208) >> 4;
        v225 = v209;
        v208 = 0;
        v209 = 0;
        HIDWORD(v220) = -1431655765 * v54;
        v55 = v210.m128i_i64[0];
        v210.m128i_i64[0] = 0;
        v226 = v55;
        if ( v52 )
          j_j___libc_free_0(v52);
        v56 = (void (***)())v227[2];
        v57 = **v56;
        if ( v57 != nullsub_1688 )
          ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v57)(v56, v227[5], v53, &v224);
        v58 = (_WORD *)v17[2];
        BYTE2(v220) = v35;
        LOBYTE(v220) = v220 & 0xDF;
        sub_3377410((__int64)&v211, v58, (__int64)&v217);
        v59 = v213;
        if ( v240 != v242 )
          _libc_free((unsigned __int64)v240);
        if ( v237 != v239 )
          _libc_free((unsigned __int64)v237);
        if ( v234 != v236 )
          _libc_free((unsigned __int64)v234);
        if ( v231 != v233 )
          _libc_free((unsigned __int64)v231);
        if ( v228 )
          sub_B91220((__int64)&v228, v228);
        if ( v224 )
          j_j___libc_free_0(v224);
        if ( v208 )
          j_j___libc_free_0(v208);
        return (__int64)v59;
      }
    }
    sub_332CDC0(&v208, v30, &v214);
    goto LABEL_11;
  }
  return (__int64)v59;
}
