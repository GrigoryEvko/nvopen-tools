// Function: sub_24ECC60
// Address: 0x24ecc60
//
void __fastcall sub_24ECC60(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdx
  __int64 *v25; // rsi
  unsigned __int8 **v26; // rax
  unsigned __int8 *v27; // rbx
  unsigned __int8 *v28; // rax
  const char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // r13
  __int64 v39; // rsi
  _QWORD *v40; // rdi
  __int64 v41; // r8
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // r14
  __int64 v44; // r12
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int8 *v51; // rax
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // edx
  __int64 v55; // r12
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r13
  __int64 v60; // rax
  int v61; // r13d
  __int64 v62; // rax
  __int64 v63; // rdx
  int v64; // eax
  unsigned __int8 *v65; // r14
  __int64 v66; // rax
  unsigned __int8 *v67; // r13
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 *v70; // rcx
  __int32 v71; // edx
  __int64 *v72; // rax
  unsigned __int64 *v73; // r14
  __int64 v74; // rsi
  __int64 v75; // r13
  __int64 v76; // r12
  __int64 v77; // r13
  _QWORD *v78; // rax
  __int64 v79; // r14
  __int64 v80; // r15
  __int64 v81; // r13
  __int64 v82; // rdx
  unsigned int v83; // esi
  unsigned __int8 *v84; // r12
  __int64 *v85; // rax
  __int64 **v86; // r14
  __int64 v87; // rax
  unsigned __int8 *v88; // rbx
  __int64 v89; // rdx
  __int64 v90; // rdx
  __int16 *v91; // rbx
  __int16 *v92; // r12
  __int64 v93; // rax
  unsigned __int64 *v94; // rdx
  __int64 v95; // r14
  unsigned __int64 *v96; // r13
  unsigned __int64 v97; // rbx
  __int64 v98; // r15
  bool v99; // zf
  _QWORD *v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int8 v103; // cl
  __int64 v104; // rsi
  _QWORD *v105; // r12
  _QWORD *v106; // rbx
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rax
  unsigned int v110; // eax
  _QWORD *v111; // rbx
  _QWORD *v112; // r15
  __int64 v113; // rsi
  char *v114; // rax
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 v118; // rax
  __int64 *v119; // [rsp+20h] [rbp-500h]
  unsigned __int8 **v120; // [rsp+28h] [rbp-4F8h]
  unsigned __int64 *v121; // [rsp+60h] [rbp-4C0h]
  __int64 v122; // [rsp+70h] [rbp-4B0h]
  __int64 v123; // [rsp+70h] [rbp-4B0h]
  __int64 v125; // [rsp+80h] [rbp-4A0h]
  __int64 v127; // [rsp+90h] [rbp-490h]
  __int64 v128; // [rsp+98h] [rbp-488h]
  unsigned __int8 **v130; // [rsp+A8h] [rbp-478h]
  __int64 v132; // [rsp+C8h] [rbp-458h] BYREF
  __m128i v133[2]; // [rsp+D0h] [rbp-450h] BYREF
  __int16 v134; // [rsp+F0h] [rbp-430h]
  __m128i v135[2]; // [rsp+100h] [rbp-420h] BYREF
  char v136; // [rsp+120h] [rbp-400h]
  char v137; // [rsp+121h] [rbp-3FFh]
  __m128i v138; // [rsp+130h] [rbp-3F0h] BYREF
  __m128i *v139; // [rsp+140h] [rbp-3E0h]
  __int16 v140; // [rsp+150h] [rbp-3D0h]
  __m128i v141; // [rsp+160h] [rbp-3C0h] BYREF
  __int64 v142; // [rsp+170h] [rbp-3B0h] BYREF
  __int64 v143; // [rsp+178h] [rbp-3A8h]
  __int64 v144; // [rsp+180h] [rbp-3A0h]
  unsigned int *v145[2]; // [rsp+1B0h] [rbp-370h] BYREF
  _BYTE v146[32]; // [rsp+1C0h] [rbp-360h] BYREF
  __int64 v147; // [rsp+1E0h] [rbp-340h]
  __int64 v148; // [rsp+1E8h] [rbp-338h]
  __int16 v149; // [rsp+1F0h] [rbp-330h]
  __int64 v150; // [rsp+1F8h] [rbp-328h]
  void **v151; // [rsp+200h] [rbp-320h]
  void **v152; // [rsp+208h] [rbp-318h]
  __int64 v153; // [rsp+210h] [rbp-310h]
  int v154; // [rsp+218h] [rbp-308h]
  __int16 v155; // [rsp+21Ch] [rbp-304h]
  char v156; // [rsp+21Eh] [rbp-302h]
  __int64 v157; // [rsp+220h] [rbp-300h]
  __int64 v158; // [rsp+228h] [rbp-2F8h]
  void *v159; // [rsp+230h] [rbp-2F0h] BYREF
  void *v160; // [rsp+238h] [rbp-2E8h] BYREF
  __m128i v161; // [rsp+240h] [rbp-2E0h] BYREF
  __int64 v162; // [rsp+250h] [rbp-2D0h] BYREF
  __int64 v163; // [rsp+258h] [rbp-2C8h]
  __int64 i; // [rsp+260h] [rbp-2C0h]
  __int64 v165; // [rsp+270h] [rbp-2B0h]
  __int64 v166; // [rsp+278h] [rbp-2A8h]
  __int64 v167; // [rsp+280h] [rbp-2A0h]
  __int64 v168; // [rsp+288h] [rbp-298h]
  void **v169; // [rsp+290h] [rbp-290h]
  void **v170; // [rsp+298h] [rbp-288h]
  __int64 v171; // [rsp+2A0h] [rbp-280h]
  int v172; // [rsp+2A8h] [rbp-278h]
  __int16 v173; // [rsp+2ACh] [rbp-274h]
  char v174; // [rsp+2AEh] [rbp-272h]
  __int64 v175; // [rsp+2B0h] [rbp-270h]
  __int64 v176; // [rsp+2B8h] [rbp-268h]
  void *v177; // [rsp+2C0h] [rbp-260h] BYREF
  void *v178; // [rsp+2C8h] [rbp-258h] BYREF
  __m128i v179; // [rsp+2D0h] [rbp-250h] BYREF
  _BYTE v180[32]; // [rsp+2E0h] [rbp-240h] BYREF
  __int64 v181; // [rsp+300h] [rbp-220h]
  __int64 v182; // [rsp+308h] [rbp-218h]
  __int16 v183; // [rsp+310h] [rbp-210h]
  __int64 v184; // [rsp+318h] [rbp-208h]
  void **v185; // [rsp+320h] [rbp-200h]
  void **v186; // [rsp+328h] [rbp-1F8h]
  __int64 v187; // [rsp+330h] [rbp-1F0h]
  int v188; // [rsp+338h] [rbp-1E8h]
  __int16 v189; // [rsp+33Ch] [rbp-1E4h]
  char v190; // [rsp+33Eh] [rbp-1E2h]
  __int64 v191; // [rsp+340h] [rbp-1E0h]
  __int64 v192; // [rsp+348h] [rbp-1D8h]
  void *v193; // [rsp+350h] [rbp-1D0h] BYREF
  void *v194; // [rsp+358h] [rbp-1C8h] BYREF
  __m128i v195; // [rsp+370h] [rbp-1B0h] BYREF
  __m128i *v196; // [rsp+380h] [rbp-1A0h]
  __int64 *v197; // [rsp+388h] [rbp-198h]
  __int64 v198; // [rsp+390h] [rbp-190h]
  _BYTE *v199; // [rsp+398h] [rbp-188h]
  __int64 v200; // [rsp+3A0h] [rbp-180h]
  _BYTE v201[32]; // [rsp+3A8h] [rbp-178h] BYREF
  __int16 *v202; // [rsp+3C8h] [rbp-158h]
  __int64 v203; // [rsp+3D0h] [rbp-150h]
  __int16 v204; // [rsp+3D8h] [rbp-148h] BYREF
  __int64 v205; // [rsp+3E0h] [rbp-140h]
  void **v206; // [rsp+3E8h] [rbp-138h]
  _QWORD *v207; // [rsp+3F0h] [rbp-130h]
  __int64 v208; // [rsp+3F8h] [rbp-128h]
  int v209; // [rsp+400h] [rbp-120h]
  __int16 v210; // [rsp+404h] [rbp-11Ch]
  char v211; // [rsp+406h] [rbp-11Ah]
  __int64 v212; // [rsp+408h] [rbp-118h]
  __int64 v213; // [rsp+410h] [rbp-110h]
  void *v214; // [rsp+418h] [rbp-108h] BYREF
  _QWORD v215[4]; // [rsp+420h] [rbp-100h] BYREF
  _QWORD *v216; // [rsp+440h] [rbp-E0h]
  __int64 v217; // [rsp+448h] [rbp-D8h]
  unsigned int v218; // [rsp+450h] [rbp-D0h]
  _QWORD *v219; // [rsp+460h] [rbp-C0h]
  unsigned int v220; // [rsp+470h] [rbp-B0h]
  char v221; // [rsp+478h] [rbp-A8h]
  __int64 v222; // [rsp+488h] [rbp-98h]
  __int64 v223; // [rsp+490h] [rbp-90h]
  _BYTE *v224; // [rsp+498h] [rbp-88h]
  __int64 v225; // [rsp+4A0h] [rbp-80h]
  _BYTE v226[120]; // [rsp+4A8h] [rbp-78h] BYREF

  sub_B2D470(a2, 36);
  sub_B2D520(a2, 22);
  sub_B2D520(a2, 43);
  v6 = (__int64 *)sub_B2BE50(a2);
  v7 = sub_BCE3C0(v6, 0);
  v8 = *(_QWORD *)(*a3 - 32LL * (*(_DWORD *)(*a3 + 4) & 0x7FFFFFF));
  v153 = 0;
  v150 = sub_BD5C60(v8);
  v151 = &v159;
  v152 = &v160;
  v155 = 512;
  v149 = 0;
  v145[0] = (unsigned int *)v146;
  v159 = &unk_49DA100;
  v145[1] = (unsigned int *)0x200000000LL;
  v9 = v8;
  v154 = 0;
  v160 = &unk_49DA0B0;
  v156 = 7;
  v157 = 0;
  v158 = 0;
  v147 = 0;
  v148 = 0;
  sub_D5F1F0((__int64)v145, v8);
  v11 = *(_QWORD *)(*(_QWORD *)(v8 + 40) + 72LL);
  v12 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
  v13 = *(_QWORD *)(v8 + 32 * (2 - v12));
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
  {
    v14 = *(_QWORD *)(v13 + 24);
    if ( (*(_BYTE *)(v11 + 2) & 1) == 0 )
      goto LABEL_3;
  }
  else
  {
    v14 = **(_QWORD **)(v13 + 24);
    if ( (*(_BYTE *)(v11 + 2) & 1) == 0 )
      goto LABEL_3;
  }
  sub_B2C6D0(v11, v9, v12, v10);
LABEL_3:
  v15 = *(_QWORD *)(v11 + 96);
  LOWORD(v198) = 257;
  v16 = sub_10E0940((__int64 *)v145, v15 + 40LL * (unsigned int)v14, v7, (__int64)&v195);
  LOWORD(v198) = 259;
  v195.m128i_i64[0] = (__int64)"async.ctx.frameptr";
  v17 = a3[45];
  v18 = sub_BCB2B0(v6);
  v19 = sub_94B2B0(v145, v18, v16, v17, (__int64)&v195);
  v195 = (__m128i)6uLL;
  v20 = v19;
  v21 = a3[39];
  if ( v21 )
  {
    v196 = (__m128i *)a3[39];
    if ( v21 != -4096 && v21 != -8192 )
      sub_BD73F0((__int64)&v195);
  }
  else
  {
    v196 = 0;
  }
  sub_BD84D0(*a3, v20);
  a3[39] = (__int64)v196;
  sub_D68D70(&v195);
  v24 = *((unsigned int *)a3 + 32);
  v119 = *(__int64 **)(a2 + 64);
  if ( v24 > *(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v24, 8u, v22, v23);
    v24 = *((unsigned int *)a3 + 32);
  }
  v25 = &v132;
  v125 = 0;
  v26 = (unsigned __int8 **)a3[15];
  v130 = v26;
  v120 = &v26[v24];
  if ( v26 != v120 )
  {
    do
    {
      v132 = v125;
      v27 = *v130;
      v28 = sub_BD3990(*(unsigned __int8 **)&(*v130)[32 * (2LL - (*((_DWORD *)*v130 + 1) & 0x7FFFFFF))], (__int64)v25);
      v29 = sub_BD5D20((__int64)v28);
      if ( v30 == 36 )
      {
        v31 = *(_QWORD *)v29 ^ 0x5F74666977735F5FLL | *((_QWORD *)v29 + 1) ^ 0x65725F636E797361LL;
        if ( !v31 )
        {
          v31 = *((_QWORD *)v29 + 2) ^ 0x6F72705F656D7573LL | *((_QWORD *)v29 + 3) ^ 0x6E6F635F7463656ALL;
          if ( !v31 && *((_DWORD *)v29 + 8) == 1954047348 )
          {
            v114 = (char *)&unk_43889C4;
LABEL_130:
            v141.m128i_i64[0] = (__int64)"_";
            v134 = 267;
            LOWORD(v144) = 259;
            v133[0].m128i_i64[0] = (__int64)&v132;
            v137 = 1;
            v135[0].m128i_i64[0] = (__int64)v114;
            v136 = 3;
            sub_9C6370(&v138, v135, v133, (__int64)&v132, v32, v33);
            sub_9C6370(&v195, &v138, &v141, v115, v116, v117);
            goto LABEL_13;
          }
        }
      }
      else if ( v30 == 32 )
      {
        v31 = *(_QWORD *)v29 ^ 0x5F74666977735F5FLL | *((_QWORD *)v29 + 1) ^ 0x65725F636E797361LL;
        if ( !v31 )
        {
          v31 = *((_QWORD *)v29 + 2) ^ 0x7465675F656D7573LL | *((_QWORD *)v29 + 3) ^ 0x747865746E6F635FLL;
          v114 = "TY";
          if ( !v31 )
            goto LABEL_130;
        }
      }
      LOWORD(i) = 267;
      v180[17] = 1;
      v161.m128i_i64[0] = (__int64)&v132;
      v179.m128i_i64[0] = (__int64)".resume.";
      v180[16] = 3;
      sub_9C6370(&v195, &v179, &v161, v31, v32, v33);
LABEL_13:
      v128 = sub_24E5090(a2, (__int64)a3, &v195, v119, (__int64)v27);
      v36 = *(unsigned int *)(a4 + 8);
      if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v36 + 1, 8u, v34, v35);
        v36 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v36) = v128;
      v37 = v122;
      ++*(_DWORD *)(a4 + 8);
      v38 = (_QWORD *)*((_QWORD *)v27 + 5);
      v39 = (__int64)(v27 + 24);
      LOWORD(v37) = 0;
      LOWORD(v198) = 257;
      v40 = v38;
      v122 = v37;
      v38 += 6;
      v41 = sub_AA8550(v40, (__int64 *)v27 + 3, v37, (__int64)&v195, 0);
      v42 = *v38 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v42 == v38 )
        goto LABEL_131;
      if ( !v42 )
        BUG();
      v43 = v42 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
      {
LABEL_131:
        v44 = -32;
        v43 = 0;
      }
      else
      {
        v44 = v42 - 56;
      }
      v127 = v41;
      v195.m128i_i64[0] = (__int64)"coro.return";
      LOWORD(v198) = 259;
      v45 = sub_B2BE50(a2);
      v46 = sub_22077B0(0x50u);
      v47 = v46;
      if ( v46 )
      {
        v39 = v45;
        sub_AA4D50(v46, v45, (__int64)&v195, a2, v127);
        if ( *(_QWORD *)(v43 - 32) )
        {
          v48 = *(_QWORD *)(v43 - 24);
          **(_QWORD **)(v43 - 16) = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = *(_QWORD *)(v43 - 16);
        }
        *(_QWORD *)(v43 - 32) = v47;
        v49 = *(_QWORD *)(v47 + 16);
        *(_QWORD *)(v43 - 24) = v49;
        if ( v49 )
          *(_QWORD *)(v49 + 16) = v43 - 24;
        *(_QWORD *)(v43 - 16) = v47 + 16;
        *(_QWORD *)(v47 + 16) = v44;
      }
      else if ( *(_QWORD *)(v43 - 32) )
      {
        v118 = *(_QWORD *)(v43 - 24);
        **(_QWORD **)(v43 - 16) = v118;
        if ( v118 )
          *(_QWORD *)(v118 + 16) = *(_QWORD *)(v43 - 16);
        *(_QWORD *)(v43 - 32) = 0;
      }
      v50 = sub_AA48A0(v47);
      v165 = v47;
      v168 = v50;
      v169 = &v177;
      v170 = &v178;
      v161.m128i_i64[0] = (__int64)&v162;
      v177 = &unk_49DA100;
      LOWORD(v167) = 0;
      v161.m128i_i64[1] = 0x200000000LL;
      v178 = &unk_49DA0B0;
      v171 = 0;
      v172 = 0;
      v173 = 512;
      v174 = 7;
      v175 = 0;
      v176 = 0;
      v166 = v47 + 48;
      v51 = sub_BD3990(*(unsigned __int8 **)&v27[32 * (3LL - (*((_DWORD *)v27 + 1) & 0x7FFFFFF))], v39);
      v54 = *v27;
      v55 = (__int64)v51;
      if ( v54 == 40 )
      {
        v56 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v27);
      }
      else
      {
        v56 = -32;
        if ( v54 != 85 )
        {
          if ( v54 != 34 )
            BUG();
          v56 = -96;
        }
      }
      if ( (v27[7] & 0x80u) != 0 )
      {
        v57 = sub_BD2BC0((__int64)v27);
        v59 = v57 + v58;
        v60 = 0;
        if ( (v27[7] & 0x80u) != 0 )
          v60 = sub_BD2BC0((__int64)v27);
        if ( (unsigned int)((v59 - v60) >> 4) )
        {
          if ( (v27[7] & 0x80u) == 0 )
            BUG();
          v61 = *(_DWORD *)(sub_BD2BC0((__int64)v27) + 8);
          if ( (v27[7] & 0x80u) == 0 )
            BUG();
          v62 = sub_BD2BC0((__int64)v27);
          v56 -= 32LL * (unsigned int)(*(_DWORD *)(v62 + v63 - 4) - v61);
        }
      }
      v64 = *((_DWORD *)v27 + 1);
      v65 = &v27[v56];
      v141.m128i_i64[0] = (__int64)&v142;
      v66 = 32LL * (v64 & 0x7FFFFFF);
      v141.m128i_i64[1] = 0x800000000LL;
      v67 = &v27[-v66];
      v68 = v56 + v66;
      v69 = v68 >> 5;
      if ( (unsigned __int64)v68 > 0x100 )
      {
        sub_C8D5F0((__int64)&v141, &v142, v68 >> 5, 8u, v52, v53);
        v70 = (__int64 *)v141.m128i_i64[0];
        v71 = v141.m128i_i32[2];
        v72 = (__int64 *)(v141.m128i_i64[0] + 8LL * v141.m128i_u32[2]);
      }
      else
      {
        v70 = &v142;
        v71 = 0;
        v72 = &v142;
      }
      if ( v67 != v65 )
      {
        do
        {
          if ( v72 )
            *v72 = *(_QWORD *)v67;
          v67 += 32;
          ++v72;
        }
        while ( v65 != v67 );
        v70 = (__int64 *)v141.m128i_i64[0];
        v71 = v141.m128i_i32[2];
      }
      v73 = (unsigned __int64 *)(v70 + 4);
      v141.m128i_i32[2] = v69 + v71;
      v74 = *((_QWORD *)v27 + 6);
      v75 = (unsigned int)(v69 + v71) - 4LL;
      v195.m128i_i64[0] = v74;
      if ( v74 )
        sub_B96E90((__int64)&v195, v74, 1);
      v76 = sub_24E5F20(v195.m128i_i64, v55, a5, v73, v75, (__int64)&v161);
      if ( v195.m128i_i64[0] )
        sub_B91220((__int64)&v195, v195.m128i_i64[0]);
      v77 = v168;
      LOWORD(v198) = 257;
      v78 = sub_BD2C40(72, 0);
      v79 = (__int64)v78;
      if ( v78 )
        sub_B4BB80((__int64)v78, v77, 0, 0, 0, 0);
      (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v170 + 2))(v170, v79, &v195, v166, v167);
      v80 = v161.m128i_i64[0];
      v81 = v161.m128i_i64[0] + 16LL * v161.m128i_u32[2];
      if ( v161.m128i_i64[0] != v81 )
      {
        do
        {
          v82 = *(_QWORD *)(v80 + 8);
          v83 = *(_DWORD *)v80;
          v80 += 16;
          sub_B99FD0(v79, v83, v82);
        }
        while ( v81 != v80 );
      }
      v199 = v201;
      v200 = 0x400000000LL;
      v202 = &v204;
      v224 = v226;
      v195 = 0u;
      v196 = 0;
      v197 = 0;
      v198 = 0;
      v203 = 0x800000000LL;
      v225 = 0x800000000LL;
      v226[64] = 1;
      sub_29F2700(v76, &v195, 0, 0, 1, 0);
      v84 = sub_BD3990(*(unsigned __int8 **)&v27[32 * (1LL - (*((_DWORD *)v27 + 1) & 0x7FFFFFF))], (__int64)&v195);
      v85 = (__int64 *)sub_B2BE50(*(_QWORD *)(*((_QWORD *)v27 + 5) + 72LL));
      v86 = (__int64 **)sub_BCE3C0(v85, 0);
      v184 = sub_BD5C60((__int64)v84);
      v179.m128i_i64[0] = (__int64)v180;
      v193 = &unk_49DA100;
      v186 = &v194;
      v189 = 512;
      v183 = 0;
      v179.m128i_i64[1] = 0x200000000LL;
      v194 = &unk_49DA0B0;
      v185 = &v193;
      v187 = 0;
      v188 = 0;
      v190 = 7;
      v191 = 0;
      v192 = 0;
      v181 = 0;
      v182 = 0;
      sub_D5F1F0((__int64)&v179, (__int64)v84);
      v140 = 257;
      v25 = (__int64 *)sub_10E0940(v179.m128i_i64, v128, (__int64)v86, (__int64)&v138);
      sub_BD84D0((__int64)v84, (__int64)v25);
      sub_B43D60(v84);
      v87 = sub_ACADE0(v86);
      v88 = &v27[32 * (1LL - (*((_DWORD *)v27 + 1) & 0x7FFFFFF))];
      if ( *(_QWORD *)v88 )
      {
        v89 = *((_QWORD *)v88 + 1);
        **((_QWORD **)v88 + 2) = v89;
        if ( v89 )
          *(_QWORD *)(v89 + 16) = *((_QWORD *)v88 + 2);
      }
      *(_QWORD *)v88 = v87;
      if ( v87 )
      {
        v90 = *(_QWORD *)(v87 + 16);
        *((_QWORD *)v88 + 1) = v90;
        if ( v90 )
        {
          v25 = (__int64 *)(v88 + 8);
          *(_QWORD *)(v90 + 16) = v88 + 8;
        }
        *((_QWORD *)v88 + 2) = v87 + 16;
        *(_QWORD *)(v87 + 16) = v88;
      }
      nullsub_61();
      v193 = &unk_49DA100;
      nullsub_63();
      if ( (_BYTE *)v179.m128i_i64[0] != v180 )
        _libc_free(v179.m128i_u64[0]);
      if ( v224 != v226 )
        _libc_free((unsigned __int64)v224);
      v91 = v202;
      v92 = &v202[12 * (unsigned int)v203];
      if ( v202 != v92 )
      {
        do
        {
          v93 = *((_QWORD *)v92 - 1);
          v92 -= 12;
          if ( v93 != -4096 && v93 != 0 && v93 != -8192 )
            sub_BD60C0(v92);
        }
        while ( v91 != v92 );
        v92 = v202;
      }
      if ( v92 != &v204 )
        _libc_free((unsigned __int64)v92);
      if ( v199 != v201 )
        _libc_free((unsigned __int64)v199);
      if ( (__int64 *)v141.m128i_i64[0] != &v142 )
        _libc_free(v141.m128i_u64[0]);
      nullsub_61();
      v177 = &unk_49DA100;
      nullsub_63();
      if ( (__int64 *)v161.m128i_i64[0] != &v162 )
        _libc_free(v161.m128i_u64[0]);
      ++v130;
      ++v125;
    }
    while ( v120 != v130 );
  }
  sub_24E3E60((__int64)&v179, a2);
  v94 = (unsigned __int64 *)a3[15];
  v121 = &v94[*((unsigned int *)a3 + 32)];
  if ( v121 != v94 )
  {
    v95 = 0;
    v96 = (unsigned __int64 *)a3[15];
    do
    {
      v135[0].m128i_i64[0] = v95;
      v97 = *v96;
      v98 = *(_QWORD *)(*(_QWORD *)a4 + 8 * v95);
      v140 = 2819;
      v138.m128i_i64[0] = (__int64)"resume.";
      v139 = v135;
      v123 = sub_C996C0("BaseCloner", 10, 0, 0);
      v195.m128i_i64[1] = a2;
      v195.m128i_i64[0] = (__int64)&unk_4A16A60;
      v196 = &v138;
      v99 = *((_DWORD *)a3 + 70) == 3;
      v197 = a3;
      LODWORD(v198) = v99 + 3;
      v205 = sub_B2BE50(a2);
      v199 = v201;
      v206 = &v214;
      v210 = 512;
      v207 = v215;
      v204 = 0;
      v214 = &unk_49DA100;
      v200 = 0x200000000LL;
      v215[0] = &unk_49DA0B0;
      v208 = 0;
      v215[1] = a5;
      v209 = 0;
      v211 = 7;
      v212 = 0;
      v213 = 0;
      v202 = 0;
      v203 = 0;
      v215[2] = &v179;
      v215[3] = 0;
      v218 = 128;
      v100 = (_QWORD *)sub_C7D670(0x2000, 8);
      v217 = 0;
      v216 = v100;
      v161.m128i_i64[1] = 2;
      v102 = (__int64)&v100[8 * (unsigned __int64)v218];
      v161.m128i_i64[0] = (__int64)&unk_49DD7B0;
      v162 = 0;
      v163 = -4096;
      for ( i = 0; (_QWORD *)v102 != v100; v100 += 8 )
      {
        if ( v100 )
        {
          v103 = v161.m128i_i8[8];
          v100[2] = 0;
          v100[3] = -4096;
          *v100 = &unk_49DD7B0;
          v100[1] = v103 & 6;
          v101 = i;
          v100[4] = i;
        }
      }
      v221 = 0;
      v222 = v98;
      v223 = 0;
      v224 = (_BYTE *)v97;
      sub_24EA1C0((__int64)&v195, (__int64)&unk_49DD7B0, v102, v101);
      v195.m128i_i64[0] = (__int64)&unk_4A16A60;
      if ( v221 )
      {
        v110 = v220;
        v221 = 0;
        if ( v220 )
        {
          v111 = v219;
          v112 = &v219[2 * v220];
          do
          {
            if ( *v111 != -4096 && *v111 != -8192 )
            {
              v113 = v111[1];
              if ( v113 )
                sub_B91220((__int64)(v111 + 1), v113);
            }
            v111 += 2;
          }
          while ( v112 != v111 );
          v110 = v220;
        }
        sub_C7D6A0((__int64)v219, 16LL * v110, 8);
      }
      v104 = v218;
      if ( v218 )
      {
        v105 = v216;
        v141.m128i_i64[1] = 2;
        v142 = 0;
        v106 = &v216[8 * (unsigned __int64)v218];
        v143 = -4096;
        v141.m128i_i64[0] = (__int64)&unk_49DD7B0;
        v161.m128i_i64[0] = (__int64)&unk_49DD7B0;
        v107 = -4096;
        v144 = 0;
        v161.m128i_i64[1] = 2;
        v162 = 0;
        v163 = -8192;
        i = 0;
        while ( 1 )
        {
          v108 = v105[3];
          if ( v108 != v107 )
          {
            v107 = v163;
            if ( v108 != v163 )
            {
              v109 = v105[7];
              if ( v109 != -4096 && v109 != 0 && v109 != -8192 )
              {
                sub_BD60C0(v105 + 5);
                v108 = v105[3];
              }
              v107 = v108;
            }
          }
          *v105 = &unk_49DB368;
          if ( v107 != 0 && v107 != -4096 && v107 != -8192 )
            sub_BD60C0(v105 + 1);
          v105 += 8;
          if ( v106 == v105 )
            break;
          v107 = v143;
        }
        v161.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v163 != -4096 && v163 != 0 && v163 != -8192 )
          sub_BD60C0(&v161.m128i_i64[1]);
        v141.m128i_i64[0] = (__int64)&unk_49DB368;
        if ( v143 != 0 && v143 != -4096 && v143 != -8192 )
          sub_BD60C0(&v141.m128i_i64[1]);
        v104 = v218;
      }
      sub_C7D6A0((__int64)v216, v104 << 6, 8);
      nullsub_61();
      v214 = &unk_49DA100;
      nullsub_63();
      if ( v199 != v201 )
        _libc_free((unsigned __int64)v199);
      if ( v123 )
        sub_C9AF60(v123);
      ++v95;
      ++v96;
    }
    while ( v121 != v96 );
  }
  if ( !v180[12] )
    _libc_free(v179.m128i_u64[1]);
  nullsub_61();
  v159 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v145[0] != v146 )
    _libc_free((unsigned __int64)v145[0]);
}
