// Function: sub_1975210
// Address: 0x1975210
//
__int64 __fastcall sub_1975210(__int64 *a1)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  _QWORD *v13; // rdi
  unsigned int v14; // r14d
  _QWORD *v16; // r15
  _QWORD *v17; // rdi
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r15
  __m128i v27; // xmm4
  __m128i v28; // xmm5
  _BYTE *v29; // r8
  _BYTE *v30; // r15
  _QWORD *v31; // rbx
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r14
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r15
  __m128i v39; // xmm2
  __m128i v40; // xmm3
  _BYTE *v41; // r8
  _BYTE *v42; // r15
  _QWORD *v43; // rbx
  _QWORD *v44; // rdi
  __int64 *v45; // r14
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // r15
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  _BYTE *v51; // r8
  _BYTE *v52; // r15
  _QWORD *v53; // rbx
  _QWORD *v54; // rdi
  __int64 *v55; // r14
  __int64 v56; // rax
  __int64 v57; // r15
  __m128i v58; // xmm6
  __m128i v59; // xmm7
  _BYTE *v60; // r8
  _QWORD *v61; // rbx
  _QWORD *v62; // rdi
  _QWORD *v63; // r15
  _QWORD *v64; // r12
  _QWORD *v65; // rdi
  _QWORD *v66; // r15
  _QWORD *v67; // r12
  _QWORD *v68; // rdi
  int v69; // r15d
  __int64 v70; // rax
  unsigned int v71; // r15d
  __int64 v72; // r8
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // r15
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rsi
  unsigned __int8 v79; // cl
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 *v89; // r8
  __int64 *v90; // rdi
  __int64 v91; // r15
  __int64 v92; // rbx
  __int64 v93; // rax
  unsigned __int64 v94; // rsi
  __int64 v95; // rdx
  unsigned __int8 v96; // r15
  __int64 v97; // rcx
  _QWORD *v98; // r15
  _QWORD *v99; // r12
  _QWORD *v100; // rdi
  __int64 *v101; // r12
  __int64 v102; // rax
  __int64 v103; // rsi
  __int64 v104; // r15
  __m128i v105; // xmm6
  __m128i v106; // xmm7
  _BYTE *v107; // r8
  _QWORD *v108; // r12
  _QWORD *v109; // rbx
  _QWORD *v110; // rdi
  _QWORD *v111; // rbx
  _QWORD *v112; // rdi
  __int64 v113; // rax
  __int64 v114; // r15
  __m128i v115; // xmm2
  __m128i v116; // xmm3
  _BYTE *v117; // r8
  _QWORD *v118; // rbx
  _QWORD *v119; // rdi
  __int64 *v120; // r12
  __int64 v121; // rax
  __int64 v122; // rsi
  __int64 v123; // r13
  char *v124; // rsi
  size_t v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // rax
  _QWORD *v130; // r15
  _QWORD *v131; // r12
  _QWORD *v132; // rdi
  __int64 v133; // rax
  __int64 v134; // rax
  _QWORD *v135; // r15
  _QWORD *v136; // r12
  _QWORD *v137; // rdi
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rsi
  __int64 v146; // r13
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rax
  __int64 *v151; // r14
  __int64 v152; // rax
  __int64 v153; // rsi
  __int64 v154; // r12
  __m128i v155; // xmm2
  __m128i v156; // xmm3
  _BYTE *v157; // r8
  __int64 v158; // rax
  _QWORD *v159; // r12
  _QWORD *v160; // rbx
  _QWORD *v161; // rdi
  _BYTE *v162; // r15
  _QWORD *v163; // rbx
  _QWORD *v164; // rdi
  __int64 v165; // rax
  __int64 v166; // rax
  __int64 *v167; // r12
  __int64 v168; // rax
  __int64 v169; // rsi
  __int64 v170; // r15
  __m128i v171; // xmm4
  __m128i v172; // xmm5
  _BYTE *v173; // r8
  _QWORD *v174; // r12
  _QWORD *v175; // rbx
  _QWORD *v176; // rdi
  _QWORD *v177; // rbx
  _QWORD *v178; // rdi
  __int64 v179; // rax
  __int64 v180; // rax
  __int64 v181; // [rsp+8h] [rbp-628h]
  unsigned __int8 v182; // [rsp+8h] [rbp-628h]
  __int64 *v183; // [rsp+10h] [rbp-620h]
  __int64 v184; // [rsp+20h] [rbp-610h]
  __int64 v185; // [rsp+28h] [rbp-608h]
  unsigned __int8 v186; // [rsp+40h] [rbp-5F0h]
  __int64 *v187; // [rsp+40h] [rbp-5F0h]
  __int64 v188; // [rsp+40h] [rbp-5F0h]
  __int64 *v189; // [rsp+40h] [rbp-5F0h]
  __int64 *v190; // [rsp+40h] [rbp-5F0h]
  __int64 *v191; // [rsp+40h] [rbp-5F0h]
  __int64 v192; // [rsp+48h] [rbp-5E8h]
  __int64 v193; // [rsp+48h] [rbp-5E8h]
  __m128i v194; // [rsp+50h] [rbp-5E0h]
  _BYTE v195[16]; // [rsp+60h] [rbp-5D0h] BYREF
  void (__fastcall *v196)(__int64 *, _BYTE *, __int64); // [rsp+70h] [rbp-5C0h]
  unsigned __int8 (__fastcall *v197)(_QWORD); // [rsp+78h] [rbp-5B8h]
  __m128i v198; // [rsp+80h] [rbp-5B0h] BYREF
  _BYTE v199[16]; // [rsp+90h] [rbp-5A0h] BYREF
  void (__fastcall *v200)(__int64 *, _BYTE *, __int64); // [rsp+A0h] [rbp-590h]
  unsigned __int8 (__fastcall *v201)(_QWORD); // [rsp+A8h] [rbp-588h]
  __m128i v202; // [rsp+B0h] [rbp-580h] BYREF
  _BYTE v203[16]; // [rsp+C0h] [rbp-570h] BYREF
  void (__fastcall *v204)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-560h]
  unsigned __int8 (__fastcall *v205)(_QWORD); // [rsp+D8h] [rbp-558h]
  __m128i v206; // [rsp+E0h] [rbp-550h]
  _BYTE v207[16]; // [rsp+F0h] [rbp-540h] BYREF
  void (__fastcall *v208)(__int64 *, __int64 *, __int64); // [rsp+100h] [rbp-530h]
  unsigned __int8 (__fastcall *v209)(_QWORD); // [rsp+108h] [rbp-528h]
  __m128i v210; // [rsp+110h] [rbp-520h] BYREF
  _BYTE v211[16]; // [rsp+120h] [rbp-510h] BYREF
  void (__fastcall *v212)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-500h]
  __int64 (__fastcall *v213)(_QWORD, _QWORD); // [rsp+138h] [rbp-4F8h]
  _BYTE *v214; // [rsp+140h] [rbp-4F0h] BYREF
  __int64 v215; // [rsp+148h] [rbp-4E8h]
  _BYTE v216[64]; // [rsp+150h] [rbp-4E0h] BYREF
  _BYTE *v217; // [rsp+190h] [rbp-4A0h] BYREF
  __int64 v218; // [rsp+198h] [rbp-498h]
  _BYTE v219[64]; // [rsp+1A0h] [rbp-490h] BYREF
  __m128i v220; // [rsp+1E0h] [rbp-450h] BYREF
  _BYTE v221[16]; // [rsp+1F0h] [rbp-440h] BYREF
  void (__fastcall *v222)(_QWORD, _QWORD, _QWORD); // [rsp+200h] [rbp-430h]
  __int64 (__fastcall *v223)(_QWORD, _QWORD); // [rsp+208h] [rbp-428h]
  __m128i v224; // [rsp+210h] [rbp-420h]
  _BYTE v225[16]; // [rsp+220h] [rbp-410h] BYREF
  __int64 v226; // [rsp+230h] [rbp-400h]
  unsigned __int8 (__fastcall *v227)(_QWORD); // [rsp+238h] [rbp-3F8h]
  __m128i v228; // [rsp+240h] [rbp-3F0h] BYREF
  __int64 v229; // [rsp+250h] [rbp-3E0h] BYREF
  __m128i v230; // [rsp+258h] [rbp-3D8h]
  unsigned __int8 (__fastcall *v231)(_QWORD); // [rsp+268h] [rbp-3C8h]
  __int64 v232; // [rsp+270h] [rbp-3C0h]
  __m128i v233; // [rsp+278h] [rbp-3B8h]
  __int64 v234; // [rsp+288h] [rbp-3A8h]
  char v235; // [rsp+290h] [rbp-3A0h]
  _BYTE *v236; // [rsp+298h] [rbp-398h] BYREF
  __int64 v237; // [rsp+2A0h] [rbp-390h]
  _BYTE v238[352]; // [rsp+2A8h] [rbp-388h] BYREF
  char v239; // [rsp+408h] [rbp-228h]
  int v240; // [rsp+40Ch] [rbp-224h]
  __int64 v241; // [rsp+410h] [rbp-220h]
  __m128i v242; // [rsp+420h] [rbp-210h] BYREF
  __int64 v243; // [rsp+430h] [rbp-200h] BYREF
  __m128i v244; // [rsp+438h] [rbp-1F8h] BYREF
  unsigned __int8 (__fastcall *v245)(_QWORD); // [rsp+448h] [rbp-1E8h]
  __m128i v246; // [rsp+450h] [rbp-1E0h] BYREF
  __int64 v247; // [rsp+460h] [rbp-1D0h] BYREF
  __int64 v248; // [rsp+468h] [rbp-1C8h]
  void (__fastcall *v249)(__int64 *, __int64 *, __int64); // [rsp+470h] [rbp-1C0h]
  _BYTE *v250; // [rsp+478h] [rbp-1B8h] BYREF
  unsigned int v251; // [rsp+480h] [rbp-1B0h]
  _BYTE v252[352]; // [rsp+488h] [rbp-1A8h] BYREF
  char v253; // [rsp+5E8h] [rbp-48h]
  int v254; // [rsp+5ECh] [rbp-44h]
  __int64 v255; // [rsp+5F0h] [rbp-40h]

  v2 = sub_13FC520(a1[1]);
  v3 = sub_13FCB50(a1[1]);
  v4 = sub_13F9E70(a1[1]);
  if ( v4 != v3
    || (v18 = v4, v19 = sub_13F9E70(*a1), v19 != sub_13FCB50(*a1))
    || *(_BYTE *)(sub_157EBA0(v18) + 16) != 26
    || (v20 = sub_13FCB50(*a1), *(_BYTE *)(sub_157EBA0(v20) + 16) != 26) )
  {
    v5 = (__int64 *)a1[6];
    v6 = sub_15E0530(*v5);
    if ( !sub_1602790(v6) )
    {
      v33 = sub_15E0530(*v5);
      v34 = sub_16033E0(v33);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v34 + 48LL))(v34) )
        return 1;
    }
    v7 = **(_QWORD **)(*a1 + 32);
    sub_13FD840(&v217, *a1);
    sub_15C9090((__int64)&v220, &v217);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"ExitingNotLatch", 15, &v220, v7);
    sub_15CAB20(
      (__int64)&v242,
      "Loops where the latch is not the exiting block cannot be interchange currently.",
      0x4Fu);
    v8 = _mm_loadu_si128(&v244);
    v9 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v8;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v9;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v16 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v10 = &v250[88 * v251];
      if ( v250 != (_BYTE *)v10 )
      {
        do
        {
          v10 -= 11;
          v17 = (_QWORD *)v10[4];
          if ( v17 != v10 + 6 )
            j_j___libc_free_0(v17, v10[6] + 1LL);
          if ( (_QWORD *)*v10 != v10 + 2 )
            j_j___libc_free_0(*v10, v10[2] + 1LL);
        }
        while ( v16 != v10 );
        v10 = v250;
        if ( v250 == v252 )
          goto LABEL_9;
        goto LABEL_8;
      }
    }
    else
    {
      v10 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    if ( v10 == (_QWORD *)v252 )
    {
LABEL_9:
      if ( v217 )
        sub_161E7C0((__int64)&v217, (__int64)v217);
      sub_143AA50(v5, (__int64)&v228);
      v11 = v236;
      v228.m128i_i64[0] = (__int64)&unk_49ECF68;
      v12 = &v236[88 * (unsigned int)v237];
      if ( v236 != (_BYTE *)v12 )
      {
        do
        {
          v12 -= 11;
          v13 = (_QWORD *)v12[4];
          if ( v13 != v12 + 6 )
            j_j___libc_free_0(v13, v12[6] + 1LL);
          if ( (_QWORD *)*v12 != v12 + 2 )
            j_j___libc_free_0(*v12, v12[2] + 1LL);
        }
        while ( v11 != v12 );
        v12 = v236;
      }
      if ( v12 != (_QWORD *)v238 )
        _libc_free((unsigned __int64)v12);
      return 1;
    }
LABEL_8:
    _libc_free((unsigned __int64)v10);
    goto LABEL_9;
  }
  v21 = a1[1];
  v214 = v216;
  v215 = 0x800000000LL;
  v217 = v219;
  v218 = 0x800000000LL;
  if ( !sub_13FCB50(v21)
    || !sub_13FC470(v21)
    || (v186 = sub_19745D0((__int64)a1, v21, (__int64)&v214, (__int64)&v217)) == 0 )
  {
    v35 = (__int64 *)a1[6];
    v36 = sub_15E0530(*v35);
    if ( !sub_1602790(v36) )
    {
      v126 = sub_15E0530(*v35);
      v127 = sub_16033E0(v126);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v127 + 48LL))(v127) )
      {
LABEL_88:
        v14 = 1;
        goto LABEL_63;
      }
    }
    v37 = a1[1];
    v38 = **(_QWORD **)(v37 + 32);
    sub_13FD840(&v210, v37);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"UnsupportedPHIInner", 19, &v220, v38);
    sub_15CAB20(
      (__int64)&v242,
      "Only inner loops with induction or reduction PHI nodes can be interchange currently.",
      0x54u);
    v39 = _mm_loadu_si128(&v244);
    v40 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v39;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v40;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v41 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v66 = &v250[88 * v251];
      if ( v250 == (_BYTE *)v66 )
        goto LABEL_75;
      v67 = v250;
      do
      {
        v66 -= 11;
        v68 = (_QWORD *)v66[4];
        if ( v68 != v66 + 6 )
          j_j___libc_free_0(v68, v66[6] + 1LL);
        if ( (_QWORD *)*v66 != v66 + 2 )
          j_j___libc_free_0(*v66, v66[2] + 1LL);
      }
      while ( v67 != v66 );
    }
    else
    {
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    v41 = v250;
LABEL_75:
    if ( v41 != v252 )
      _libc_free((unsigned __int64)v41);
    if ( v210.m128i_i64[0] )
      sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
    sub_143AA50(v35, (__int64)&v228);
    v42 = v236;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v43 = &v236[88 * (unsigned int)v237];
    if ( v236 != (_BYTE *)v43 )
    {
      do
      {
        v43 -= 11;
        v44 = (_QWORD *)v43[4];
        if ( v44 != v43 + 6 )
          j_j___libc_free_0(v44, v43[6] + 1LL);
        if ( (_QWORD *)*v43 != v43 + 2 )
          j_j___libc_free_0(*v43, v43[2] + 1LL);
      }
      while ( v42 != (_BYTE *)v43 );
      v42 = v236;
    }
    if ( v42 != v238 )
      _libc_free((unsigned __int64)v42);
    goto LABEL_88;
  }
  if ( (_DWORD)v215 != 1 )
  {
    v45 = (__int64 *)a1[6];
    v46 = sub_15E0530(*v45);
    if ( !sub_1602790(v46) )
    {
      v128 = sub_15E0530(*v45);
      v129 = sub_16033E0(v128);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v129 + 48LL))(v129) )
        goto LABEL_108;
    }
    v47 = a1[1];
    v48 = **(_QWORD **)(v47 + 32);
    sub_13FD840(&v210, v47);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"MultiInductionInner", 19, &v220, v48);
    sub_15CAB20((__int64)&v242, "Only inner loops with 1 induction variable can be interchanged currently.", 0x49u);
    v49 = _mm_loadu_si128(&v244);
    v50 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v49;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v50;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v51 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v63 = &v250[88 * v251];
      if ( v250 == (_BYTE *)v63 )
        goto LABEL_95;
      v64 = v250;
      do
      {
        v63 -= 11;
        v65 = (_QWORD *)v63[4];
        if ( v65 != v63 + 6 )
          j_j___libc_free_0(v65, v63[6] + 1LL);
        if ( (_QWORD *)*v63 != v63 + 2 )
          j_j___libc_free_0(*v63, v63[2] + 1LL);
      }
      while ( v64 != v63 );
    }
    else
    {
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    v51 = v250;
LABEL_95:
    if ( v51 != v252 )
      _libc_free((unsigned __int64)v51);
    if ( v210.m128i_i64[0] )
      sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
    sub_143AA50(v45, (__int64)&v228);
    v52 = v236;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v53 = &v236[88 * (unsigned int)v237];
    if ( v236 == (_BYTE *)v53 )
      goto LABEL_106;
    do
    {
      v53 -= 11;
      v54 = (_QWORD *)v53[4];
      if ( v54 != v53 + 6 )
        j_j___libc_free_0(v54, v53[6] + 1LL);
      if ( (_QWORD *)*v53 != v53 + 2 )
        j_j___libc_free_0(*v53, v53[2] + 1LL);
    }
    while ( v52 != (_BYTE *)v53 );
    goto LABEL_105;
  }
  if ( (_DWORD)v218 )
    *((_BYTE *)a1 + 56) = 1;
  v22 = *(_QWORD *)v214;
  v23 = *a1;
  v185 = *a1;
  LODWORD(v215) = 0;
  v184 = v22;
  LODWORD(v218) = 0;
  v24 = v185;
  if ( !sub_13FCB50(v23)
    || !sub_13FC470(v185)
    || (v24 = v185, v14 = sub_19745D0((__int64)a1, v185, (__int64)&v214, (__int64)&v217), !(_BYTE)v14) )
  {
    v55 = (__int64 *)a1[6];
    v56 = sub_15E0530(*v55);
    if ( !sub_1602790(v56) )
    {
      v138 = sub_15E0530(*v55);
      v139 = sub_16033E0(v138);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v139 + 48LL))(v139, v24) )
      {
LABEL_108:
        v14 = v186;
        goto LABEL_63;
      }
    }
    v57 = **(_QWORD **)(*a1 + 32);
    sub_13FD840(&v210, *a1);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"UnsupportedPHIOuter", 19, &v220, v57);
    sub_15CAB20(
      (__int64)&v242,
      "Only outer loops with induction or reduction PHI nodes can be interchanged currently.",
      0x55u);
    v58 = _mm_loadu_si128(&v244);
    v59 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v58;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v59;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v60 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v135 = &v250[88 * v251];
      if ( v250 == (_BYTE *)v135 )
        goto LABEL_115;
      v136 = v250;
      do
      {
        v135 -= 11;
        v137 = (_QWORD *)v135[4];
        if ( v137 != v135 + 6 )
          j_j___libc_free_0(v137, v135[6] + 1LL);
        if ( (_QWORD *)*v135 != v135 + 2 )
          j_j___libc_free_0(*v135, v135[2] + 1LL);
      }
      while ( v136 != v135 );
    }
    else
    {
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    v60 = v250;
LABEL_115:
    if ( v60 != v252 )
      _libc_free((unsigned __int64)v60);
    if ( v210.m128i_i64[0] )
      sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
    sub_143AA50(v55, (__int64)&v228);
    v52 = v236;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v61 = &v236[88 * (unsigned int)v237];
    if ( v236 == (_BYTE *)v61 )
      goto LABEL_106;
    do
    {
      v61 -= 11;
      v62 = (_QWORD *)v61[4];
      if ( v62 != v61 + 6 )
        j_j___libc_free_0(v62, v61[6] + 1LL);
      if ( (_QWORD *)*v61 != v61 + 2 )
        j_j___libc_free_0(*v61, v61[2] + 1LL);
    }
    while ( v52 != (_BYTE *)v61 );
LABEL_105:
    v52 = v236;
LABEL_106:
    if ( v52 != v238 )
      _libc_free((unsigned __int64)v52);
    goto LABEL_108;
  }
  if ( (_DWORD)v218 )
  {
    v190 = (__int64 *)a1[6];
    v113 = sub_15E0530(*v190);
    if ( !sub_1602790(v113) )
    {
      v142 = sub_15E0530(*v190);
      v143 = sub_16033E0(v142);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v143 + 48LL))(v143) )
        goto LABEL_63;
    }
    v114 = **(_QWORD **)(*a1 + 32);
    sub_13FD840(&v210, *a1);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"ReductionsOuter", 15, &v220, v114);
    sub_15CAB20((__int64)&v242, "Outer loops with reductions cannot be interchangeed currently.", 0x3Eu);
    v115 = _mm_loadu_si128(&v244);
    v116 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v115;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v116;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v117 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v130 = &v250[88 * v251];
      if ( v250 == (_BYTE *)v130 )
        goto LABEL_263;
      v131 = v250;
      do
      {
        v130 -= 11;
        v132 = (_QWORD *)v130[4];
        if ( v132 != v130 + 6 )
          j_j___libc_free_0(v132, v130[6] + 1LL);
        if ( (_QWORD *)*v130 != v130 + 2 )
          j_j___libc_free_0(*v130, v130[2] + 1LL);
      }
      while ( v131 != v130 );
    }
    else
    {
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    v117 = v250;
LABEL_263:
    if ( v117 != v252 )
      _libc_free((unsigned __int64)v117);
    if ( v210.m128i_i64[0] )
      sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
    sub_143AA50(v190, (__int64)&v228);
    v30 = v236;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v118 = &v236[88 * (unsigned int)v237];
    if ( v236 == (_BYTE *)v118 )
      goto LABEL_61;
    do
    {
      v118 -= 11;
      v119 = (_QWORD *)v118[4];
      if ( v119 != v118 + 6 )
        j_j___libc_free_0(v119, v118[6] + 1LL);
      if ( (_QWORD *)*v118 != v118 + 2 )
        j_j___libc_free_0(*v118, v118[2] + 1LL);
    }
    while ( v30 != (_BYTE *)v118 );
    goto LABEL_60;
  }
  if ( (_DWORD)v215 != 1 )
  {
    v187 = (__int64 *)a1[6];
    v25 = sub_15E0530(*v187);
    if ( !sub_1602790(v25) )
    {
      v133 = sub_15E0530(*v187);
      v134 = sub_16033E0(v133);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v134 + 48LL))(v134) )
        goto LABEL_63;
    }
    v26 = **(_QWORD **)(*a1 + 32);
    sub_13FD840(&v210, *a1);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"MultiIndutionOuter", 18, &v220, v26);
    sub_15CAB20((__int64)&v242, "Only outer loops with 1 induction variable can be interchanged currently.", 0x49u);
    v27 = _mm_loadu_si128(&v244);
    v28 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
    v228.m128i_i32[2] = v242.m128i_i32[2];
    v230 = v27;
    v228.m128i_i8[12] = v242.m128i_i8[12];
    v233 = v28;
    v229 = v243;
    v231 = v245;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v232 = v246.m128i_i64[0];
    v235 = (char)v249;
    if ( (_BYTE)v249 )
      v234 = v248;
    v236 = v238;
    v237 = 0x400000000LL;
    if ( v251 )
    {
      sub_1974F80((__int64)&v236, (__int64)&v250);
      v29 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      v98 = &v250[88 * v251];
      if ( v250 == (_BYTE *)v98 )
        goto LABEL_50;
      v99 = v250;
      do
      {
        v98 -= 11;
        v100 = (_QWORD *)v98[4];
        if ( v100 != v98 + 6 )
          j_j___libc_free_0(v100, v98[6] + 1LL);
        if ( (_QWORD *)*v98 != v98 + 2 )
          j_j___libc_free_0(*v98, v98[2] + 1LL);
      }
      while ( v99 != v98 );
    }
    else
    {
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    v29 = v250;
LABEL_50:
    if ( v29 != v252 )
      _libc_free((unsigned __int64)v29);
    if ( v210.m128i_i64[0] )
      sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
    sub_143AA50(v187, (__int64)&v228);
    v30 = v236;
    v228.m128i_i64[0] = (__int64)&unk_49ECF68;
    v31 = &v236[88 * (unsigned int)v237];
    if ( v236 == (_BYTE *)v31 )
      goto LABEL_61;
    do
    {
      v31 -= 11;
      v32 = (_QWORD *)v31[4];
      if ( v32 != v31 + 6 )
        j_j___libc_free_0(v32, v31[6] + 1LL);
      if ( (_QWORD *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31, v31[2] + 1LL);
    }
    while ( v30 != (_BYTE *)v31 );
    goto LABEL_60;
  }
  v69 = *(_DWORD *)(v184 + 20);
  v70 = sub_13FC520(a1[1]);
  v71 = v69 & 0xFFFFFFF;
  v72 = v184;
  v73 = v70;
  if ( !v71 )
    goto LABEL_151;
  v74 = v71;
  v75 = 0;
  v76 = v73;
  v188 = 8 * v74;
  do
  {
    if ( (*(_BYTE *)(v184 + 23) & 0x40) != 0 )
      v77 = *(_QWORD *)(v184 - 8);
    else
      v77 = v184 - 24LL * (*(_DWORD *)(v184 + 20) & 0xFFFFFFF);
    v78 = *(_QWORD *)(v77 + 3 * v75);
    v79 = *(_BYTE *)(v78 + 16);
    if ( v79 > 0x10u
      && (v79 <= 0x17u
       || v76 == *(_QWORD *)(v75 + v77 + 24LL * *(unsigned int *)(v184 + 56) + 8) && !sub_13FC1A0(*a1, v78)) )
    {
      v101 = (__int64 *)a1[6];
      v14 = (unsigned __int8)v14;
      v102 = sub_15E0530(*v101);
      if ( !sub_1602790(v102) )
      {
        v140 = sub_15E0530(*v101);
        v141 = sub_16033E0(v140);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v141 + 48LL))(v141) )
          goto LABEL_63;
      }
      v103 = a1[1];
      v104 = **(_QWORD **)(v103 + 32);
      sub_13FD840(&v210, v103);
      sub_15C9090((__int64)&v220, &v210);
      sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"UnsupportedStructureInner", 25, &v220, v104);
      sub_15CAB20((__int64)&v242, "Inner loop structure not understood currently.", 0x2Eu);
      v105 = _mm_loadu_si128(&v244);
      v106 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
      v228.m128i_i32[2] = v242.m128i_i32[2];
      v230 = v105;
      v228.m128i_i8[12] = v242.m128i_i8[12];
      v233 = v106;
      v229 = v243;
      v231 = v245;
      v228.m128i_i64[0] = (__int64)&unk_49ECF68;
      v232 = v246.m128i_i64[0];
      v235 = (char)v249;
      if ( (_BYTE)v249 )
        v234 = v248;
      v236 = v238;
      v237 = 0x400000000LL;
      if ( v251 )
        sub_1974F80((__int64)&v236, (__int64)&v250);
      v107 = v250;
      v239 = v253;
      v240 = v254;
      v241 = v255;
      v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v242.m128i_i64[0] = (__int64)&unk_49ECF68;
      if ( v250 != &v250[88 * v251] )
      {
        v189 = v101;
        v108 = &v250[88 * v251];
        v109 = v250;
        do
        {
          v108 -= 11;
          v110 = (_QWORD *)v108[4];
          if ( v110 != v108 + 6 )
            j_j___libc_free_0(v110, v108[6] + 1LL);
          if ( (_QWORD *)*v108 != v108 + 2 )
            j_j___libc_free_0(*v108, v108[2] + 1LL);
        }
        while ( v109 != v108 );
        v101 = v189;
        v107 = v250;
      }
      if ( v107 != v252 )
        _libc_free((unsigned __int64)v107);
      if ( v210.m128i_i64[0] )
        sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
      sub_143AA50(v101, (__int64)&v228);
      v30 = v236;
      v228.m128i_i64[0] = (__int64)&unk_49ECF68;
      v111 = &v236[88 * (unsigned int)v237];
      if ( v236 == (_BYTE *)v111 )
        goto LABEL_61;
      do
      {
        v111 -= 11;
        v112 = (_QWORD *)v111[4];
        if ( v112 != v111 + 6 )
          j_j___libc_free_0(v112, v111[6] + 1LL);
        if ( (_QWORD *)*v111 != v111 + 2 )
          j_j___libc_free_0(*v111, v111[2] + 1LL);
      }
      while ( v30 != (_BYTE *)v111 );
      goto LABEL_60;
    }
    v75 += 8;
  }
  while ( v188 != v75 );
  v72 = v184;
  v14 = (unsigned __int8)v14;
LABEL_151:
  v192 = v72;
  v80 = sub_13FA090(a1[1]);
  v81 = sub_157F280(v80);
  v83 = v82;
  v84 = v81;
  if ( v81 == v83 )
  {
LABEL_160:
    v88 = 3LL * *(unsigned int *)(v192 + 56);
    if ( (*(_BYTE *)(v192 + 23) & 0x40) != 0 )
    {
      v89 = *(__int64 **)(v192 - 8);
      if ( v2 != v89[v88 + 1] )
        goto LABEL_162;
    }
    else
    {
      v89 = (__int64 *)(v192 - 24LL * (*(_DWORD *)(v192 + 20) & 0xFFFFFFF));
      if ( v2 != v89[v88 + 1] )
      {
LABEL_162:
        v193 = *v89;
        if ( *v89 )
        {
LABEL_163:
          if ( *(_BYTE *)(v193 + 16) > 0x17u )
          {
            sub_1580910(&v242);
            v200 = 0;
            v198 = v242;
            if ( v244.m128i_i64[1] )
            {
              ((void (__fastcall *)(_BYTE *, __int64 *, __int64))v244.m128i_i64[1])(v199, &v243, 2);
              v201 = v245;
              v200 = (void (__fastcall *)(__int64 *, _BYTE *, __int64))v244.m128i_i64[1];
            }
            v230.m128i_i64[1] = 0;
            v228 = v198;
            if ( v200 )
            {
              v200(&v229, v199, 2);
              v231 = v201;
              v230.m128i_i64[1] = (__int64)v200;
            }
            v196 = 0;
            v194 = v228;
            if ( v230.m128i_i64[1] )
            {
              ((void (__fastcall *)(_BYTE *, __int64 *, __int64))v230.m128i_i64[1])(v195, &v229, 2);
              v197 = v231;
              v196 = (void (__fastcall *)(__int64 *, _BYTE *, __int64))v230.m128i_i64[1];
              if ( v230.m128i_i64[1] )
                ((void (__fastcall *)(__int64 *, __int64 *, __int64))v230.m128i_i64[1])(&v229, &v229, 3);
            }
            v208 = 0;
            v206 = v246;
            if ( v249 )
            {
              v249((__int64 *)v207, &v247, 2);
              v209 = (unsigned __int8 (__fastcall *)(_QWORD))v250;
              v208 = v249;
            }
            v230.m128i_i64[1] = 0;
            v228 = v206;
            if ( v208 )
            {
              v208(&v229, (__int64 *)v207, 2);
              v231 = v209;
              v230.m128i_i64[1] = (__int64)v208;
            }
            v204 = 0;
            v202 = v228;
            if ( v230.m128i_i64[1] )
            {
              ((void (__fastcall *)(_BYTE *, __int64 *, __int64))v230.m128i_i64[1])(v203, &v229, 2);
              v205 = v231;
              v204 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v230.m128i_i64[1];
              if ( v230.m128i_i64[1] )
                ((void (__fastcall *)(__int64 *, __int64 *, __int64))v230.m128i_i64[1])(&v229, &v229, 3);
            }
            v230.m128i_i64[1] = 0;
            v228 = v194;
            if ( v196 )
            {
              v196(&v229, v195, 2);
              v231 = v197;
              v230.m128i_i64[1] = (__int64)v196;
            }
            v212 = 0;
            v210 = v202;
            if ( v204 )
            {
              v204(v211, v203, 2);
              v213 = (__int64 (__fastcall *)(_QWORD, _QWORD))v205;
              v212 = v204;
            }
            v222 = 0;
            v220 = v210;
            if ( v212 )
            {
              v212(v221, v211, 2);
              v223 = v213;
              v222 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v212;
            }
            v226 = 0;
            v224 = v228;
            if ( v230.m128i_i64[1] )
            {
              ((void (__fastcall *)(_BYTE *, __int64 *, __int64))v230.m128i_i64[1])(v225, &v229, 2);
              v227 = v231;
              v226 = v230.m128i_i64[1];
            }
            if ( v212 )
              v212(v211, v211, 3);
            if ( v230.m128i_i64[1] )
              ((void (__fastcall *)(__int64 *, __int64 *, __int64))v230.m128i_i64[1])(&v229, &v229, 3);
            if ( v204 )
              v204(v203, v203, 3);
            if ( v208 )
              v208((__int64 *)v207, (__int64 *)v207, 3);
            if ( v196 )
              v196((__int64 *)v195, v195, 3);
            if ( v200 )
              v200((__int64 *)v199, v199, 3);
            sub_A17130((__int64)&v247);
            sub_A17130((__int64)&v243);
            v206 = v220;
            sub_1974F30((__int64)v207, (__int64)v221);
            v90 = (__int64 *)v211;
            v210 = v224;
            sub_1974F30((__int64)v211, (__int64)v225);
            v183 = a1;
LABEL_199:
            v244.m128i_i64[1] = 0;
            v242 = v210;
            if ( v212 )
            {
              v90 = &v243;
              v212(&v243, v211, 2);
              v245 = (unsigned __int8 (__fastcall *)(_QWORD))v213;
              v244.m128i_i64[1] = (__int64)v212;
            }
            v91 = v206.m128i_i64[0];
            v230.m128i_i64[1] = 0;
            v228 = v206;
            if ( v208 )
            {
              v90 = &v229;
              v208(&v229, (__int64 *)v207, 2);
              v91 = v228.m128i_i64[0];
              v92 = v242.m128i_i64[0];
              v231 = v209;
              v230.m128i_i64[1] = (__int64)v208;
              if ( v208 )
              {
                v90 = &v229;
                v208(&v229, &v229, 3);
              }
            }
            else
            {
              v92 = v242.m128i_i64[0];
            }
            if ( v244.m128i_i64[1] )
            {
              v90 = &v243;
              ((void (__fastcall *)(__int64 *, __int64 *, __int64))v244.m128i_i64[1])(&v243, &v243, 3);
            }
            if ( v91 != v92 )
            {
              v244.m128i_i64[1] = 0;
              v242 = v206;
              v93 = (__int64)v208;
              if ( v208 )
              {
                v90 = &v243;
                v208(&v243, (__int64 *)v207, 2);
                v245 = v209;
                v93 = (__int64)v208;
                v244.m128i_i64[1] = (__int64)v208;
              }
              v94 = *(_QWORD *)v242.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              for ( v242.m128i_i64[0] = v94; ; v242.m128i_i64[0] = v94 )
              {
                v95 = v94 - 24;
                if ( v94 )
                  v94 -= 24LL;
                if ( !v93 )
                  goto LABEL_286;
                v90 = &v243;
                v96 = ((__int64 (__fastcall *)(__int64 *, unsigned __int64))v245)(&v243, v94);
                if ( v96 )
                  break;
                v93 = v244.m128i_i64[1];
                v94 = *(_QWORD *)v242.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
              }
              v97 = v242.m128i_i64[0];
              if ( v242.m128i_i64[0] )
                v97 = v242.m128i_i64[0] - 24;
              if ( v244.m128i_i64[1] )
              {
                v181 = v97;
                v90 = &v243;
                ((void (__fastcall *)(__int64 *, __int64 *, __int64))v244.m128i_i64[1])(&v243, &v243, 3);
                v97 = v181;
              }
              if ( (unsigned __int8)(*(_BYTE *)(v97 + 16) - 26) <= 0x32u )
              {
                v95 = 0x6000C00000001LL;
                if ( _bittest64(&v95, (unsigned int)*(unsigned __int8 *)(v97 + 16) - 26) )
                {
                  while ( 1 )
                  {
                    v206.m128i_i64[0] = *(_QWORD *)v206.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                    v94 = v206.m128i_i64[0];
                    if ( v206.m128i_i64[0] )
                      v94 = v206.m128i_i64[0] - 24;
                    if ( !v208 )
                      break;
                    v90 = (__int64 *)v207;
                    if ( v209(v207) )
                      goto LABEL_199;
                  }
LABEL_286:
                  sub_4263D6(v90, v94, v95);
                }
              }
              v182 = v96;
              if ( sub_15F41F0(v97, v193) )
              {
                v14 = 0;
                sub_A17130((__int64)v211);
                sub_A17130((__int64)v207);
                sub_A17130((__int64)v225);
                sub_A17130((__int64)v221);
              }
              else
              {
                v151 = (__int64 *)v183[6];
                v152 = sub_15E0530(*v151);
                if ( sub_1602790(v152)
                  || (v165 = sub_15E0530(*v151),
                      v166 = sub_16033E0(v165),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v166 + 48LL))(v166)) )
                {
                  v153 = v183[1];
                  v154 = **(_QWORD **)(v153 + 32);
                  sub_13FD840(&v198, v153);
                  sub_15C9090((__int64)&v202, &v198);
                  sub_15CA540(
                    (__int64)&v242,
                    (__int64)"loop-interchange",
                    (__int64)"UnsupportedInsBetweenInduction",
                    30,
                    &v202,
                    v154);
                  sub_15CAB20(
                    (__int64)&v242,
                    "Found unsupported instruction between induction variable increment and branch.",
                    0x4Eu);
                  v155 = _mm_loadu_si128(&v244);
                  v156 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
                  v228.m128i_i32[2] = v242.m128i_i32[2];
                  v230 = v155;
                  v228.m128i_i8[12] = v242.m128i_i8[12];
                  v233 = v156;
                  v229 = v243;
                  v231 = v245;
                  v228.m128i_i64[0] = (__int64)&unk_49ECF68;
                  v232 = v246.m128i_i64[0];
                  v235 = (char)v249;
                  if ( (_BYTE)v249 )
                    v234 = v248;
                  v236 = v238;
                  v237 = 0x400000000LL;
                  if ( v251 )
                    sub_1974F80((__int64)&v236, (__int64)&v250);
                  v157 = v250;
                  v239 = v253;
                  v240 = v254;
                  v241 = v255;
                  v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
                  v242.m128i_i64[0] = (__int64)&unk_49ECF68;
                  v158 = 88LL * v251;
                  v159 = &v250[v158];
                  if ( v250 != &v250[v158] )
                  {
                    v160 = v250;
                    do
                    {
                      v159 -= 11;
                      v161 = (_QWORD *)v159[4];
                      if ( v161 != v159 + 6 )
                        j_j___libc_free_0(v161, v159[6] + 1LL);
                      if ( (_QWORD *)*v159 != v159 + 2 )
                        j_j___libc_free_0(*v159, v159[2] + 1LL);
                    }
                    while ( v160 != v159 );
                    v157 = v250;
                  }
                  if ( v157 != v252 )
                    _libc_free((unsigned __int64)v157);
                  if ( v198.m128i_i64[0] )
                    sub_161E7C0((__int64)&v198, v198.m128i_i64[0]);
                  sub_143AA50(v151, (__int64)&v228);
                  v162 = v236;
                  v228.m128i_i64[0] = (__int64)&unk_49ECF68;
                  v163 = &v236[88 * (unsigned int)v237];
                  if ( v236 != (_BYTE *)v163 )
                  {
                    do
                    {
                      v163 -= 11;
                      v164 = (_QWORD *)v163[4];
                      if ( v164 != v163 + 6 )
                        j_j___libc_free_0(v164, v163[6] + 1LL);
                      if ( (_QWORD *)*v163 != v163 + 2 )
                        j_j___libc_free_0(*v163, v163[2] + 1LL);
                    }
                    while ( v162 != (_BYTE *)v163 );
                    v162 = v236;
                  }
                  if ( v162 != v238 )
                    _libc_free((unsigned __int64)v162);
                }
                sub_A17130((__int64)v211);
                sub_A17130((__int64)v207);
                sub_A17130((__int64)v225);
                sub_A17130((__int64)v221);
                v14 = v182;
              }
              goto LABEL_63;
            }
            sub_A17130((__int64)v211);
            sub_A17130((__int64)v207);
            sub_A17130((__int64)v225);
            sub_A17130((__int64)v221);
            v167 = (__int64 *)v183[6];
            v168 = sub_15E0530(*v167);
            if ( !sub_1602790(v168) )
            {
              v179 = sub_15E0530(*v167);
              v180 = sub_16033E0(v179);
              if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v180 + 48LL))(v180) )
                goto LABEL_63;
            }
            v169 = v183[1];
            v170 = **(_QWORD **)(v169 + 32);
            sub_13FD840(&v210, v169);
            sub_15C9090((__int64)&v220, &v210);
            sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"NoIndutionVariable", 18, &v220, v170);
            sub_15CAB20((__int64)&v242, "Did not find the induction variable.", 0x24u);
            v171 = _mm_loadu_si128(&v244);
            v172 = _mm_loadu_si128((const __m128i *)&v246.m128i_u64[1]);
            v228.m128i_i32[2] = v242.m128i_i32[2];
            v230 = v171;
            v228.m128i_i8[12] = v242.m128i_i8[12];
            v233 = v172;
            v229 = v243;
            v231 = v245;
            v228.m128i_i64[0] = (__int64)&unk_49ECF68;
            v232 = v246.m128i_i64[0];
            v235 = (char)v249;
            if ( (_BYTE)v249 )
              v234 = v248;
            v236 = v238;
            v237 = 0x400000000LL;
            if ( v251 )
              sub_1974F80((__int64)&v236, (__int64)&v250);
            v173 = v250;
            v239 = v253;
            v240 = v254;
            v241 = v255;
            v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
            v242.m128i_i64[0] = (__int64)&unk_49ECF68;
            if ( v250 != &v250[88 * v251] )
            {
              v191 = v167;
              v174 = &v250[88 * v251];
              v175 = v250;
              do
              {
                v174 -= 11;
                v176 = (_QWORD *)v174[4];
                if ( v176 != v174 + 6 )
                  j_j___libc_free_0(v176, v174[6] + 1LL);
                if ( (_QWORD *)*v174 != v174 + 2 )
                  j_j___libc_free_0(*v174, v174[2] + 1LL);
              }
              while ( v175 != v174 );
              v167 = v191;
              v173 = v250;
            }
            if ( v173 != v252 )
              _libc_free((unsigned __int64)v173);
            if ( v210.m128i_i64[0] )
              sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
            sub_143AA50(v167, (__int64)&v228);
            v30 = v236;
            v228.m128i_i64[0] = (__int64)&unk_49ECF68;
            v177 = &v236[88 * (unsigned int)v237];
            if ( v236 == (_BYTE *)v177 )
            {
LABEL_61:
              if ( v30 != v238 )
                _libc_free((unsigned __int64)v30);
              goto LABEL_63;
            }
            do
            {
              v177 -= 11;
              v178 = (_QWORD *)v177[4];
              if ( v178 != v177 + 6 )
                j_j___libc_free_0(v178, v177[6] + 1LL);
              if ( (_QWORD *)*v177 != v177 + 2 )
                j_j___libc_free_0(*v177, v177[2] + 1LL);
            }
            while ( v30 != (_BYTE *)v177 );
LABEL_60:
            v30 = v236;
            goto LABEL_61;
          }
          v120 = (__int64 *)a1[6];
          v144 = sub_15E0530(*v120);
          if ( !sub_1602790(v144) )
          {
            v147 = sub_15E0530(*v120);
            v148 = sub_16033E0(v147);
            if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v148 + 48LL))(v148) )
              goto LABEL_63;
          }
          v145 = a1[1];
          v146 = **(_QWORD **)(v145 + 32);
          sub_13FD840(&v210, v145);
          sub_15C9090((__int64)&v220, &v210);
          sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"NoIncrementInInner", 18, &v220, v146);
          v124 = "The inner loop does not increment the induction variable.";
          v125 = 57;
LABEL_276:
          sub_15CAB20((__int64)&v242, v124, v125);
          sub_18980B0((__int64)&v228, (__int64)&v242);
          v241 = v255;
          v228.m128i_i64[0] = (__int64)&unk_49ECFC8;
          v242.m128i_i64[0] = (__int64)&unk_49ECF68;
          sub_1897B80((__int64)&v250);
          if ( v210.m128i_i64[0] )
            sub_161E7C0((__int64)&v210, v210.m128i_i64[0]);
          sub_143AA50(v120, (__int64)&v228);
          v228.m128i_i64[0] = (__int64)&unk_49ECF68;
          sub_1897B80((__int64)&v236);
          goto LABEL_63;
        }
LABEL_319:
        BUG();
      }
    }
    v193 = v89[3];
    if ( v193 )
      goto LABEL_163;
    goto LABEL_319;
  }
  while ( 1 )
  {
    v85 = *(_DWORD *)(v84 + 20) & 0xFFFFFFF;
    if ( (unsigned int)v85 > 1 )
      break;
    v86 = (*(_BYTE *)(v84 + 23) & 0x40) != 0 ? *(_QWORD *)(v84 - 8) : v84 - 24 * v85;
    if ( *(_BYTE *)(*(_QWORD *)v86 + 16LL) <= 0x17u )
      break;
    v87 = *(_QWORD *)(v84 + 32);
    if ( !v87 )
      BUG();
    v84 = 0;
    if ( *(_BYTE *)(v87 - 8) == 77 )
      v84 = v87 - 24;
    if ( v83 == v84 )
      goto LABEL_160;
  }
  v120 = (__int64 *)a1[6];
  v121 = sub_15E0530(*v120);
  if ( sub_1602790(v121)
    || (v149 = sub_15E0530(*v120),
        v150 = sub_16033E0(v149),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v150 + 48LL))(v150)) )
  {
    v122 = a1[1];
    v123 = **(_QWORD **)(v122 + 32);
    sub_13FD840(&v210, v122);
    sub_15C9090((__int64)&v220, &v210);
    sub_15CA540((__int64)&v242, (__int64)"loop-interchange", (__int64)"NoLCSSAPHIOuterInner", 20, &v220, v123);
    v124 = "Only inner loops with LCSSA PHIs can be interchange currently.";
    v125 = 62;
    goto LABEL_276;
  }
LABEL_63:
  if ( v217 != v219 )
    _libc_free((unsigned __int64)v217);
  if ( v214 != v216 )
    _libc_free((unsigned __int64)v214);
  return v14;
}
