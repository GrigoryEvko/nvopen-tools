// Function: sub_20CD3E0
// Address: 0x20cd3e0
//
unsigned __int64 __fastcall sub_20CD3E0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // r13
  unsigned int v16; // ebx
  int v17; // ebx
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  __int64 **v21; // r15
  _QWORD *v22; // rax
  __int64 v23; // r12
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // r13
  _QWORD *v30; // rax
  _QWORD *v31; // r12
  unsigned __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rsi
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rdx
  _QWORD *v60; // r13
  __int64 v61; // rdx
  __int64 v62; // rcx
  bool v63; // al
  __int64 v64; // r10
  unsigned __int8 v65; // r9
  int v66; // r8d
  _QWORD *v67; // rax
  __int64 v68; // r15
  __int64 *v69; // r13
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rsi
  unsigned __int8 *v73; // rsi
  __int16 v74; // dx
  __int64 v75; // rax
  bool v76; // zf
  _QWORD *v77; // rax
  _QWORD *v78; // r13
  unsigned __int64 *v79; // r15
  __int64 v80; // rax
  unsigned __int64 v81; // rsi
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  __int64 v84; // r13
  _QWORD *v85; // rax
  _QWORD *v86; // r15
  unsigned __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  unsigned __int8 *v93; // rsi
  __int64 v94; // rsi
  int v95; // eax
  __int64 v96; // rax
  int v97; // edx
  __int64 v98; // rdx
  __int64 *v99; // rax
  __int64 v100; // rcx
  unsigned __int64 v101; // rdx
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rax
  unsigned __int8 *v106; // rsi
  __int64 *v107; // rsi
  __int64 **v108; // r13
  __int64 v109; // rax
  _QWORD *v110; // r12
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  double v114; // xmm4_8
  double v115; // xmm5_8
  unsigned __int64 result; // rax
  _QWORD *v117; // rax
  __int64 v118; // rax
  __int64 v119; // r13
  __int64 *v120; // r12
  __int64 v121; // rax
  __int64 v122; // rcx
  __int64 v123; // r12
  __int64 v124; // rsi
  unsigned __int8 *v125; // rsi
  __int64 v126; // rax
  __int64 v127; // r13
  __int64 *v128; // r12
  __int64 v129; // rax
  __int64 v130; // rcx
  __int64 v131; // r12
  __int64 v132; // rsi
  unsigned __int8 *v133; // rsi
  __int64 v134; // rax
  unsigned __int64 *v135; // r15
  __int64 v136; // rax
  unsigned __int64 v137; // rcx
  __int64 v138; // rsi
  __int64 v139; // rax
  __int64 v140; // r10
  __int64 *v141; // r15
  __int64 v142; // rsi
  __int64 v143; // rax
  __int64 v144; // rsi
  __int64 v145; // r15
  unsigned __int8 *v146; // rsi
  unsigned __int64 *v147; // r13
  __int64 v148; // rax
  unsigned __int64 v149; // rcx
  __int64 v150; // rsi
  unsigned __int8 *v151; // rsi
  __int64 *v152; // r12
  __int64 v153; // rax
  __int64 v154; // rcx
  __int64 v155; // r12
  __int64 v156; // rsi
  unsigned __int8 *v157; // rsi
  __int64 *v158; // r12
  __int64 v159; // rax
  __int64 v160; // rcx
  __int64 v161; // r12
  __int64 v162; // rsi
  unsigned __int8 *v163; // rsi
  __int64 **v164; // [rsp+8h] [rbp-198h]
  __int64 v165; // [rsp+18h] [rbp-188h]
  __int64 v166; // [rsp+20h] [rbp-180h]
  __int64 v167; // [rsp+28h] [rbp-178h]
  char v168; // [rsp+28h] [rbp-178h]
  __int64 v169; // [rsp+28h] [rbp-178h]
  __int64 v170; // [rsp+30h] [rbp-170h]
  unsigned int v171; // [rsp+30h] [rbp-170h]
  __int64 v172; // [rsp+30h] [rbp-170h]
  __int64 v173; // [rsp+38h] [rbp-168h]
  __int64 v174; // [rsp+40h] [rbp-160h]
  _QWORD *v175; // [rsp+48h] [rbp-158h]
  __int16 v176; // [rsp+48h] [rbp-158h]
  __int64 v177; // [rsp+48h] [rbp-158h]
  __int64 v178; // [rsp+48h] [rbp-158h]
  __int64 v179; // [rsp+48h] [rbp-158h]
  __int64 v180; // [rsp+48h] [rbp-158h]
  __int64 v181; // [rsp+50h] [rbp-150h]
  unsigned __int64 *v182; // [rsp+50h] [rbp-150h]
  __int64 ***v183; // [rsp+58h] [rbp-148h]
  __int64 v184; // [rsp+58h] [rbp-148h]
  __int64 v185; // [rsp+60h] [rbp-140h]
  unsigned __int64 *v186; // [rsp+60h] [rbp-140h]
  __int64 v187; // [rsp+60h] [rbp-140h]
  _QWORD *v188; // [rsp+68h] [rbp-138h]
  __int64 v189; // [rsp+78h] [rbp-128h]
  __int64 *v190; // [rsp+88h] [rbp-118h] BYREF
  __int64 v191[2]; // [rsp+90h] [rbp-110h] BYREF
  __int16 v192; // [rsp+A0h] [rbp-100h]
  __int64 v193[2]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v194; // [rsp+C0h] [rbp-E0h]
  unsigned __int8 *v195[2]; // [rsp+D0h] [rbp-D0h] BYREF
  __int16 v196; // [rsp+E0h] [rbp-C0h]
  __int64 *v197[2]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v198; // [rsp+100h] [rbp-A0h]
  __int64 v199; // [rsp+108h] [rbp-98h]
  __int64 v200; // [rsp+118h] [rbp-88h]
  __int64 *v201; // [rsp+120h] [rbp-80h] BYREF
  _QWORD *v202; // [rsp+128h] [rbp-78h]
  __int64 *v203; // [rsp+130h] [rbp-70h]
  __int64 v204; // [rsp+138h] [rbp-68h]
  __int64 v205; // [rsp+140h] [rbp-60h]
  int v206; // [rsp+148h] [rbp-58h]
  __int64 v207; // [rsp+150h] [rbp-50h]
  __int64 v208; // [rsp+158h] [rbp-48h]

  v183 = *(__int64 ****)(a2 - 72);
  v185 = *(_QWORD *)(a2 - 48);
  v189 = *(_QWORD *)(a2 - 24);
  v175 = *(_QWORD **)(a2 + 40);
  v11 = v175[7];
  v12 = sub_16498A0(a2);
  v13 = *(__int64 **)(a2 + 48);
  v201 = 0;
  v204 = v12;
  v14 = *(_QWORD **)(a2 + 40);
  v205 = 0;
  v202 = v14;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v203 = (__int64 *)(a2 + 24);
  v197[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v197, (__int64)v13, 2);
    if ( v201 )
      sub_161E7C0((__int64)&v201, (__int64)v201);
    v201 = v197[0];
    if ( v197[0] )
      sub_1623210((__int64)v197, (unsigned __int8 *)v197[0], (__int64)&v201);
  }
  v15 = v204;
  v16 = *(_DWORD *)(*(_QWORD *)(a1 + 160) + 104LL);
  v197[0] = (__int64 *)"partword.cmpxchg.end";
  LOWORD(v198) = 259;
  v17 = v16 >> 3;
  v181 = sub_157FBF0(v175, (__int64 *)(a2 + 24), (__int64)v197);
  v197[0] = (__int64 *)"partword.cmpxchg.failure";
  LOWORD(v198) = 259;
  v188 = (_QWORD *)sub_22077B0(64);
  if ( v188 )
    sub_157FB60(v188, v15, (__int64)v197, v11, v181);
  v197[0] = (__int64 *)"partword.cmpxchg.loop";
  LOWORD(v198) = 259;
  v18 = (_QWORD *)sub_22077B0(64);
  v174 = (__int64)v18;
  if ( v18 )
    sub_157FB60(v18, v15, (__int64)v197, v11, (__int64)v188);
  v173 = v175[5];
  v19 = (_QWORD *)((v173 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v173 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v19 = 0;
  sub_15F20C0(v19);
  v20 = *(__int64 **)(a2 - 48);
  v202 = v175;
  v203 = v175 + 5;
  sub_20CB200(v197, (__int64 *)&v201, a2, *v20, v183, v17, *(double *)a3.m128_u64, a4, a5);
  v194 = 257;
  v21 = (__int64 **)v197[0];
  v166 = v198;
  v192 = 257;
  v184 = v199;
  v165 = v200;
  if ( v197[0] != *(__int64 **)v189 )
  {
    if ( *(_BYTE *)(v189 + 16) > 0x10u )
    {
      v196 = 257;
      v189 = sub_15FDBD0(37, v189, (__int64)v197[0], (__int64)v195, 0);
      if ( v202 )
      {
        v158 = v203;
        sub_157E9D0((__int64)(v202 + 5), v189);
        v159 = *(_QWORD *)(v189 + 24);
        v160 = *v158;
        *(_QWORD *)(v189 + 32) = v158;
        v160 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v189 + 24) = v160 | v159 & 7;
        *(_QWORD *)(v160 + 8) = v189 + 24;
        *v158 = *v158 & 7 | (v189 + 24);
      }
      sub_164B780(v189, v191);
      if ( v201 )
      {
        v190 = v201;
        sub_1623A60((__int64)&v190, (__int64)v201, 2);
        v161 = v189 + 48;
        v162 = *(_QWORD *)(v189 + 48);
        if ( v162 )
          sub_161E7C0(v161, v162);
        v163 = (unsigned __int8 *)v190;
        *(_QWORD *)(v189 + 48) = v190;
        if ( v163 )
          sub_1623210((__int64)&v190, v163, v161);
      }
    }
    else
    {
      v189 = sub_15A46C0(37, (__int64 ***)v189, (__int64 **)v197[0], 0);
    }
  }
  if ( *(_BYTE *)(v189 + 16) > 0x10u || *(_BYTE *)(v184 + 16) > 0x10u )
  {
    v196 = 257;
    v126 = sub_15FB440(23, (__int64 *)v189, v184, (__int64)v195, 0);
    v167 = v126;
    v127 = v126;
    if ( v202 )
    {
      v128 = v203;
      sub_157E9D0((__int64)(v202 + 5), v126);
      v129 = *(_QWORD *)(v127 + 24);
      v130 = *v128;
      *(_QWORD *)(v127 + 32) = v128;
      v130 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v127 + 24) = v130 | v129 & 7;
      *(_QWORD *)(v130 + 8) = v127 + 24;
      *v128 = *v128 & 7 | (v127 + 24);
    }
    sub_164B780(v167, v193);
    if ( v201 )
    {
      v190 = v201;
      sub_1623A60((__int64)&v190, (__int64)v201, 2);
      v131 = v167 + 48;
      v132 = *(_QWORD *)(v167 + 48);
      if ( v132 )
        sub_161E7C0(v131, v132);
      v133 = (unsigned __int8 *)v190;
      *(_QWORD *)(v167 + 48) = v190;
      if ( v133 )
        sub_1623210((__int64)&v190, v133, v131);
    }
  }
  else
  {
    v167 = sub_15A2D50((__int64 *)v189, v184, 0, 0, *(double *)a3.m128_u64, a4, a5);
  }
  v194 = 257;
  v192 = 257;
  if ( v21 != *(__int64 ***)v185 )
  {
    if ( *(_BYTE *)(v185 + 16) > 0x10u )
    {
      v196 = 257;
      v185 = sub_15FDBD0(37, v185, (__int64)v21, (__int64)v195, 0);
      if ( v202 )
      {
        v152 = v203;
        sub_157E9D0((__int64)(v202 + 5), v185);
        v153 = *(_QWORD *)(v185 + 24);
        v154 = *v152;
        *(_QWORD *)(v185 + 32) = v152;
        v154 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v185 + 24) = v154 | v153 & 7;
        *(_QWORD *)(v154 + 8) = v185 + 24;
        *v152 = *v152 & 7 | (v185 + 24);
      }
      sub_164B780(v185, v191);
      if ( v201 )
      {
        v190 = v201;
        sub_1623A60((__int64)&v190, (__int64)v201, 2);
        v155 = v185 + 48;
        v156 = *(_QWORD *)(v185 + 48);
        if ( v156 )
          sub_161E7C0(v155, v156);
        v157 = (unsigned __int8 *)v190;
        *(_QWORD *)(v185 + 48) = v190;
        if ( v157 )
          sub_1623210((__int64)&v190, v157, v155);
      }
    }
    else
    {
      v185 = sub_15A46C0(37, (__int64 ***)v185, v21, 0);
    }
  }
  if ( *(_BYTE *)(v185 + 16) > 0x10u || *(_BYTE *)(v184 + 16) > 0x10u )
  {
    v196 = 257;
    v118 = sub_15FB440(23, (__int64 *)v185, v184, (__int64)v195, 0);
    v170 = v118;
    v119 = v118;
    if ( v202 )
    {
      v120 = v203;
      sub_157E9D0((__int64)(v202 + 5), v118);
      v121 = *(_QWORD *)(v119 + 24);
      v122 = *v120;
      *(_QWORD *)(v119 + 32) = v120;
      v122 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v119 + 24) = v122 | v121 & 7;
      *(_QWORD *)(v122 + 8) = v119 + 24;
      *v120 = *v120 & 7 | (v119 + 24);
    }
    sub_164B780(v170, v193);
    if ( v201 )
    {
      v190 = v201;
      sub_1623A60((__int64)&v190, (__int64)v201, 2);
      v123 = v170 + 48;
      v124 = *(_QWORD *)(v170 + 48);
      if ( v124 )
        sub_161E7C0(v123, v124);
      v125 = (unsigned __int8 *)v190;
      *(_QWORD *)(v170 + 48) = v190;
      if ( v125 )
        sub_1623210((__int64)&v190, v125, v123);
    }
  }
  else
  {
    v170 = sub_15A2D50((__int64 *)v185, v184, 0, 0, *(double *)a3.m128_u64, a4, a5);
  }
  v196 = 257;
  v22 = sub_1648A60(64, 1u);
  v23 = (__int64)v22;
  if ( v22 )
    sub_15F9210((__int64)v22, (__int64)v21, v166, 0, 0, 0);
  if ( v202 )
  {
    v24 = v203;
    sub_157E9D0((__int64)(v202 + 5), v23);
    v25 = *(_QWORD *)(v23 + 24);
    v26 = *v24;
    *(_QWORD *)(v23 + 32) = v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v23 + 24) = v26 | v25 & 7;
    *(_QWORD *)(v26 + 8) = v23 + 24;
    *v24 = *v24 & 7 | (v23 + 24);
  }
  sub_164B780(v23, (__int64 *)v195);
  if ( v201 )
  {
    v193[0] = (__int64)v201;
    sub_1623A60((__int64)v193, (__int64)v201, 2);
    v27 = *(_QWORD *)(v23 + 48);
    if ( v27 )
      sub_161E7C0(v23 + 48, v27);
    v28 = (unsigned __int8 *)v193[0];
    *(_QWORD *)(v23 + 48) = v193[0];
    if ( v28 )
      sub_1623210((__int64)v193, v28, v23 + 48);
  }
  *(_WORD *)(v23 + 18) = *(_WORD *)(a2 + 18) & 1 | *(_WORD *)(v23 + 18) & 0xFFFE;
  v196 = 257;
  v29 = sub_1281C00((__int64 *)&v201, v23, v165, (__int64)v195);
  v196 = 257;
  v30 = sub_1648A60(56, 1u);
  v31 = v30;
  if ( v30 )
    sub_15F8320((__int64)v30, v174, 0);
  if ( v202 )
  {
    v186 = (unsigned __int64 *)v203;
    sub_157E9D0((__int64)(v202 + 5), (__int64)v31);
    v32 = *v186;
    v33 = v31[3] & 7LL;
    v31[4] = v186;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    v31[3] = v32 | v33;
    *(_QWORD *)(v32 + 8) = v31 + 3;
    *v186 = *v186 & 7 | (unsigned __int64)(v31 + 3);
  }
  sub_164B780((__int64)v31, (__int64 *)v195);
  if ( v201 )
  {
    v193[0] = (__int64)v201;
    sub_1623A60((__int64)v193, (__int64)v201, 2);
    v34 = v31[6];
    v35 = (__int64)(v31 + 6);
    if ( v34 )
    {
      sub_161E7C0((__int64)(v31 + 6), v34);
      v35 = (__int64)(v31 + 6);
    }
    v36 = (unsigned __int8 *)v193[0];
    v31[6] = v193[0];
    if ( v36 )
      sub_1623210((__int64)v193, v36, v35);
  }
  v194 = 257;
  v202 = (_QWORD *)v174;
  v203 = (__int64 *)(v174 + 40);
  v196 = 257;
  v37 = sub_1648B60(64);
  v38 = v37;
  if ( v37 )
  {
    v187 = v37;
    sub_15F1EA0(v37, (__int64)v21, 53, 0, 0, 0);
    *(_DWORD *)(v38 + 56) = 2;
    sub_164B780(v38, (__int64 *)v195);
    sub_1648880(v38, *(_DWORD *)(v38 + 56), 1);
  }
  else
  {
    v187 = 0;
  }
  if ( v202 )
  {
    v39 = v203;
    sub_157E9D0((__int64)(v202 + 5), v38);
    v40 = *(_QWORD *)(v38 + 24);
    v41 = *v39;
    *(_QWORD *)(v38 + 32) = v39;
    v41 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v38 + 24) = v41 | v40 & 7;
    *(_QWORD *)(v41 + 8) = v38 + 24;
    *v39 = *v39 & 7 | (v38 + 24);
  }
  sub_164B780(v187, v193);
  v46 = (__int64)v201;
  if ( v201 )
  {
    v191[0] = (__int64)v201;
    sub_1623A60((__int64)v191, (__int64)v201, 2);
    v47 = *(_QWORD *)(v38 + 48);
    if ( v47 )
      sub_161E7C0(v38 + 48, v47);
    v46 = v191[0];
    *(_QWORD *)(v38 + 48) = v191[0];
    if ( v46 )
      sub_1623210((__int64)v191, (unsigned __int8 *)v46, v38 + 48);
  }
  v48 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  if ( v48 == *(_DWORD *)(v38 + 56) )
  {
    sub_15F55D0(v38, v46, v42, v43, v44, v45);
    v48 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  }
  v49 = (v48 + 1) & 0xFFFFFFF;
  v50 = v49 | *(_DWORD *)(v38 + 20) & 0xF0000000;
  *(_DWORD *)(v38 + 20) = v50;
  if ( (v50 & 0x40000000) != 0 )
    v51 = *(_QWORD *)(v38 - 8);
  else
    v51 = v187 - 24 * v49;
  v52 = (__int64 *)(v51 + 24LL * (unsigned int)(v49 - 1));
  if ( *v52 )
  {
    v53 = v52[1];
    v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v54 = v53;
    if ( v53 )
    {
      v46 = *(_QWORD *)(v53 + 16) & 3LL;
      *(_QWORD *)(v53 + 16) = v46 | v54;
    }
  }
  *v52 = v29;
  if ( v29 )
  {
    v55 = *(_QWORD *)(v29 + 8);
    v46 = v29 + 8;
    v52[1] = v55;
    if ( v55 )
      *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
    v52[2] = v46 | v52[2] & 3;
    *(_QWORD *)(v29 + 8) = v52;
  }
  v56 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  v57 = (unsigned int)(v56 - 1);
  if ( (*(_BYTE *)(v38 + 23) & 0x40) != 0 )
    v58 = *(_QWORD *)(v38 - 8);
  else
    v58 = v187 - 24 * v56;
  v59 = 3LL * *(unsigned int *)(v38 + 56);
  *(_QWORD *)(v58 + 8 * v57 + 24LL * *(unsigned int *)(v38 + 56) + 8) = v175;
  v194 = 257;
  if ( *(_BYTE *)(v167 + 16) > 0x10u )
    goto LABEL_147;
  v60 = (_QWORD *)v38;
  if ( !sub_1593BB0(v167, v46, v59, v58) )
  {
    if ( *(_BYTE *)(v38 + 16) <= 0x10u )
    {
      v46 = v167;
      v60 = (_QWORD *)sub_15A2D10((__int64 *)v38, v167, *(double *)a3.m128_u64, a4, a5);
      goto LABEL_68;
    }
LABEL_147:
    v196 = 257;
    v134 = sub_15FB440(27, (__int64 *)v38, v167, (__int64)v195, 0);
    v60 = (_QWORD *)v134;
    if ( v202 )
    {
      v135 = (unsigned __int64 *)v203;
      sub_157E9D0((__int64)(v202 + 5), v134);
      v136 = v60[3];
      v137 = *v135;
      v60[4] = v135;
      v137 &= 0xFFFFFFFFFFFFFFF8LL;
      v60[3] = v137 | v136 & 7;
      *(_QWORD *)(v137 + 8) = v60 + 3;
      *v135 = *v135 & 7 | (unsigned __int64)(v60 + 3);
    }
    sub_164B780((__int64)v60, v193);
    v46 = (__int64)v201;
    if ( v201 )
    {
      v191[0] = (__int64)v201;
      sub_1623A60((__int64)v191, (__int64)v201, 2);
      v138 = v60[6];
      if ( v138 )
        sub_161E7C0((__int64)(v60 + 6), v138);
      v46 = v191[0];
      v60[6] = v191[0];
      if ( v46 )
        sub_1623210((__int64)v191, (unsigned __int8 *)v46, (__int64)(v60 + 6));
    }
  }
LABEL_68:
  v194 = 257;
  if ( *(_BYTE *)(v170 + 16) > 0x10u )
    goto LABEL_154;
  v63 = sub_1593BB0(v170, v46, v61, v62);
  v64 = v38;
  if ( v63 )
    goto LABEL_72;
  if ( *(_BYTE *)(v38 + 16) > 0x10u )
  {
LABEL_154:
    v196 = 257;
    v139 = sub_15FB440(27, (__int64 *)v38, v170, (__int64)v195, 0);
    v140 = v139;
    if ( v202 )
    {
      v141 = v203;
      v178 = v139;
      sub_157E9D0((__int64)(v202 + 5), v139);
      v140 = v178;
      v142 = *v141;
      v143 = *(_QWORD *)(v178 + 24);
      *(_QWORD *)(v178 + 32) = v141;
      v142 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v178 + 24) = v142 | v143 & 7;
      *(_QWORD *)(v142 + 8) = v178 + 24;
      *v141 = *v141 & 7 | (v178 + 24);
    }
    v179 = v140;
    sub_164B780(v140, v193);
    v64 = v179;
    if ( v201 )
    {
      v191[0] = (__int64)v201;
      sub_1623A60((__int64)v191, (__int64)v201, 2);
      v64 = v179;
      v144 = *(_QWORD *)(v179 + 48);
      v145 = v179 + 48;
      if ( v144 )
      {
        sub_161E7C0(v179 + 48, v144);
        v64 = v179;
      }
      v146 = (unsigned __int8 *)v191[0];
      *(_QWORD *)(v64 + 48) = v191[0];
      if ( v146 )
      {
        v180 = v64;
        sub_1623210((__int64)v191, v146, v145);
        v64 = v180;
      }
    }
  }
  else
  {
    v64 = sub_15A2D10((__int64 *)v38, v170, *(double *)a3.m128_u64, a4, a5);
  }
LABEL_72:
  v164 = (__int64 **)v64;
  v65 = *(_WORD *)(a2 + 18);
  v168 = *(_BYTE *)(a2 + 56);
  v66 = (*(unsigned __int16 *)(a2 + 18) >> 2) & 7;
  v196 = 257;
  v176 = v66;
  v171 = v65 >> 5;
  v67 = sub_1648A60(64, 3u);
  v68 = (__int64)v67;
  if ( v67 )
    sub_15F99E0((__int64)v67, v166, v164, (__int64)v60, v176, v171, v168, 0);
  if ( v202 )
  {
    v69 = v203;
    sub_157E9D0((__int64)(v202 + 5), v68);
    v70 = *(_QWORD *)(v68 + 24);
    v71 = *v69;
    *(_QWORD *)(v68 + 32) = v69;
    v71 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v68 + 24) = v71 | v70 & 7;
    *(_QWORD *)(v71 + 8) = v68 + 24;
    *v69 = *v69 & 7 | (v68 + 24);
  }
  sub_164B780(v68, (__int64 *)v195);
  if ( v201 )
  {
    v193[0] = (__int64)v201;
    sub_1623A60((__int64)v193, (__int64)v201, 2);
    v72 = *(_QWORD *)(v68 + 48);
    if ( v72 )
      sub_161E7C0(v68 + 48, v72);
    v73 = (unsigned __int8 *)v193[0];
    *(_QWORD *)(v68 + 48) = v193[0];
    if ( v73 )
      sub_1623210((__int64)v193, v73, v68 + 48);
  }
  v74 = *(_WORD *)(a2 + 18) & 1 | *(_WORD *)(v68 + 18) & 0xFFFE;
  *(_WORD *)(v68 + 18) = v74;
  *(_WORD *)(v68 + 18) = v74 & 0x8000 | v74 & 0x7EFF | ((*(_BYTE *)(a2 + 19) & 1) << 8);
  v196 = 257;
  LODWORD(v193[0]) = 0;
  v177 = sub_12A9E60((__int64 *)&v201, v68, (__int64)v193, 1, (__int64)v195);
  v196 = 257;
  LODWORD(v193[0]) = 1;
  v75 = sub_12A9E60((__int64 *)&v201, v68, (__int64)v193, 1, (__int64)v195);
  v76 = (*(_BYTE *)(a2 + 19) & 1) == 0;
  v172 = v75;
  v196 = 257;
  if ( v76 )
  {
    v117 = sub_1648A60(56, 3u);
    v78 = v117;
    if ( v117 )
      sub_15F83E0((__int64)v117, v181, (__int64)v188, v172, 0);
  }
  else
  {
    v77 = sub_1648A60(56, 1u);
    v78 = v77;
    if ( v77 )
      sub_15F8320((__int64)v77, v181, 0);
  }
  if ( v202 )
  {
    v79 = (unsigned __int64 *)v203;
    sub_157E9D0((__int64)(v202 + 5), (__int64)v78);
    v80 = v78[3];
    v81 = *v79;
    v78[4] = v79;
    v81 &= 0xFFFFFFFFFFFFFFF8LL;
    v78[3] = v81 | v80 & 7;
    *(_QWORD *)(v81 + 8) = v78 + 3;
    *v79 = *v79 & 7 | (unsigned __int64)(v78 + 3);
  }
  sub_164B780((__int64)v78, (__int64 *)v195);
  if ( v201 )
  {
    v193[0] = (__int64)v201;
    sub_1623A60((__int64)v193, (__int64)v201, 2);
    v82 = v78[6];
    if ( v82 )
      sub_161E7C0((__int64)(v78 + 6), v82);
    v83 = (unsigned __int8 *)v193[0];
    v78[6] = v193[0];
    if ( v83 )
      sub_1623210((__int64)v193, v83, (__int64)(v78 + 6));
  }
  v196 = 257;
  v202 = v188;
  v203 = v188 + 5;
  v84 = sub_1281C00((__int64 *)&v201, v177, v165, (__int64)v195);
  v196 = 257;
  v169 = sub_12AA0C0((__int64 *)&v201, 0x21u, (_BYTE *)v38, v84, (__int64)v195);
  v196 = 257;
  v85 = sub_1648A60(56, 3u);
  v86 = v85;
  if ( v85 )
    sub_15F83E0((__int64)v85, v174, v181, v169, 0);
  if ( v202 )
  {
    v182 = (unsigned __int64 *)v203;
    sub_157E9D0((__int64)(v202 + 5), (__int64)v86);
    v87 = *v182;
    v88 = v86[3] & 7LL;
    v86[4] = v182;
    v87 &= 0xFFFFFFFFFFFFFFF8LL;
    v86[3] = v87 | v88;
    *(_QWORD *)(v87 + 8) = v86 + 3;
    *v182 = *v182 & 7 | (unsigned __int64)(v86 + 3);
  }
  sub_164B780((__int64)v86, (__int64 *)v195);
  v93 = (unsigned __int8 *)v201;
  if ( v201 )
  {
    v193[0] = (__int64)v201;
    sub_1623A60((__int64)v193, (__int64)v201, 2);
    v94 = v86[6];
    v89 = (__int64)(v86 + 6);
    if ( v94 )
    {
      sub_161E7C0((__int64)(v86 + 6), v94);
      v89 = (__int64)(v86 + 6);
    }
    v93 = (unsigned __int8 *)v193[0];
    v86[6] = v193[0];
    if ( v93 )
      sub_1623210((__int64)v193, v93, v89);
  }
  v95 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  if ( v95 == *(_DWORD *)(v38 + 56) )
  {
    sub_15F55D0(v38, (__int64)v93, v89, v90, v91, v92);
    v95 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  }
  v96 = (v95 + 1) & 0xFFFFFFF;
  v97 = v96 | *(_DWORD *)(v38 + 20) & 0xF0000000;
  *(_DWORD *)(v38 + 20) = v97;
  if ( (v97 & 0x40000000) != 0 )
    v98 = *(_QWORD *)(v38 - 8);
  else
    v98 = v187 - 24 * v96;
  v99 = (__int64 *)(v98 + 24LL * (unsigned int)(v96 - 1));
  if ( *v99 )
  {
    v100 = v99[1];
    v101 = v99[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v101 = v100;
    if ( v100 )
      *(_QWORD *)(v100 + 16) = *(_QWORD *)(v100 + 16) & 3LL | v101;
  }
  *v99 = v84;
  if ( v84 )
  {
    v102 = *(_QWORD *)(v84 + 8);
    v99[1] = v102;
    if ( v102 )
      *(_QWORD *)(v102 + 16) = (unsigned __int64)(v99 + 1) | *(_QWORD *)(v102 + 16) & 3LL;
    v99[2] = (v84 + 8) | v99[2] & 3;
    *(_QWORD *)(v84 + 8) = v99;
  }
  v103 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
  v104 = (unsigned int)(v103 - 1);
  if ( (*(_BYTE *)(v38 + 23) & 0x40) != 0 )
    v105 = *(_QWORD *)(v38 - 8);
  else
    v105 = v187 - 24 * v103;
  *(_QWORD *)(v105 + 8 * v104 + 24LL * *(unsigned int *)(v38 + 56) + 8) = v188;
  v106 = *(unsigned __int8 **)(a2 + 48);
  v202 = *(_QWORD **)(a2 + 40);
  v195[0] = v106;
  v203 = (__int64 *)(a2 + 24);
  if ( v106 )
  {
    sub_1623A60((__int64)v195, (__int64)v106, 2);
    v107 = v201;
    if ( !v201 )
      goto LABEL_116;
  }
  else
  {
    v107 = v201;
    if ( !v201 )
      goto LABEL_118;
  }
  sub_161E7C0((__int64)&v201, (__int64)v107);
LABEL_116:
  v201 = (__int64 *)v195[0];
  if ( v195[0] )
    sub_1623210((__int64)v195, v195[0], (__int64)&v201);
LABEL_118:
  v192 = 257;
  v194 = 257;
  v108 = (__int64 **)v197[1];
  v109 = sub_156E320((__int64 *)&v201, v177, v184, (__int64)v191, 0);
  v110 = (_QWORD *)v109;
  if ( v108 != *(__int64 ***)v109 )
  {
    if ( *(_BYTE *)(v109 + 16) > 0x10u )
    {
      v196 = 257;
      v110 = (_QWORD *)sub_15FDBD0(36, v109, (__int64)v108, (__int64)v195, 0);
      if ( v202 )
      {
        v147 = (unsigned __int64 *)v203;
        sub_157E9D0((__int64)(v202 + 5), (__int64)v110);
        v148 = v110[3];
        v149 = *v147;
        v110[4] = v147;
        v149 &= 0xFFFFFFFFFFFFFFF8LL;
        v110[3] = v149 | v148 & 7;
        *(_QWORD *)(v149 + 8) = v110 + 3;
        *v147 = *v147 & 7 | (unsigned __int64)(v110 + 3);
      }
      sub_164B780((__int64)v110, v193);
      if ( v201 )
      {
        v190 = v201;
        sub_1623A60((__int64)&v190, (__int64)v201, 2);
        v150 = v110[6];
        if ( v150 )
          sub_161E7C0((__int64)(v110 + 6), v150);
        v151 = (unsigned __int8 *)v190;
        v110[6] = v190;
        if ( v151 )
          sub_1623210((__int64)&v190, v151, (__int64)(v110 + 6));
      }
    }
    else
    {
      v110 = (_QWORD *)sub_15A46C0(36, (__int64 ***)v109, v108, 0);
    }
  }
  v111 = sub_1599EF0(*(__int64 ***)a2);
  v196 = 257;
  LODWORD(v193[0]) = 0;
  v112 = sub_17FE490((__int64 *)&v201, v111, (__int64)v110, v193, 1, (__int64 *)v195);
  v196 = 257;
  LODWORD(v193[0]) = 1;
  v113 = sub_17FE490((__int64 *)&v201, v112, v172, v193, 1, (__int64 *)v195);
  sub_164D160(a2, v113, a3, a4, a5, a6, v114, v115, a9, a10);
  result = sub_15F20C0((_QWORD *)a2);
  if ( v201 )
    return sub_161E7C0((__int64)&v201, (__int64)v201);
  return result;
}
