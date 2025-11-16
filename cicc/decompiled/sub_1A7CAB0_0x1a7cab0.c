// Function: sub_1A7CAB0
// Address: 0x1a7cab0
//
__int64 __fastcall sub_1A7CAB0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        bool *a4,
        __int64 a5,
        _QWORD *a6,
        __m128 a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 *a15)
{
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r10
  __int64 v18; // r15
  unsigned int v19; // ecx
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  _QWORD *v24; // rdi
  _QWORD *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rax
  signed __int64 v29; // rdx
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  char *v34; // r15
  char *v35; // r14
  char *v36; // rdi
  __int64 v37; // r8
  __int64 v38; // r9
  _QWORD *v39; // rbx
  _QWORD *v40; // r14
  _QWORD *v41; // rdi
  int v42; // ebx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r13
  int v46; // r13d
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 v54; // r14
  _QWORD *v55; // rdi
  bool v56; // cc
  __int64 *v57; // r15
  __int64 *v58; // rdx
  __int64 *v59; // r14
  __int64 v60; // rcx
  __int64 **v61; // rdx
  __int64 *v62; // rsi
  unsigned __int64 v63; // rcx
  __int64 v64; // rcx
  __int64 v65; // rcx
  __int64 v66; // r12
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rax
  double v70; // xmm4_8
  double v71; // xmm5_8
  __int64 v72; // r13
  __int64 v73; // r12
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rdx
  int v77; // ecx
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r13
  unsigned int v81; // ebx
  __int64 v82; // rcx
  __int64 v83; // rdx
  _QWORD *v84; // rax
  __int64 v85; // rdi
  unsigned __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rdi
  __int64 v90; // rbx
  __int64 v91; // rdx
  __int64 v92; // r14
  int v93; // eax
  __int64 v94; // rax
  int v95; // edx
  __int64 v96; // rbx
  __int64 v97; // r14
  __int64 v98; // r13
  __int64 v99; // r15
  int v100; // ebx
  int v101; // ebx
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rsi
  __int64 v107; // r15
  _QWORD *v108; // rax
  __int64 v109; // r12
  __int64 *v110; // rbx
  __int64 v111; // r13
  __int64 v112; // r15
  __int64 v113; // rdi
  __int64 v114; // rcx
  __int64 v115; // r8
  int v116; // eax
  __int64 v117; // rax
  int v118; // edi
  __int64 v119; // rdi
  __int64 *v120; // rax
  __int64 v121; // r8
  unsigned __int64 v122; // rdi
  __int64 v123; // rdi
  __int64 v124; // rdi
  __int64 v125; // rax
  int v126; // edi
  __int64 v127; // rdi
  __int64 **v128; // rax
  __int64 *v129; // r8
  unsigned __int64 v130; // rdi
  __int64 v131; // r8
  __int64 v132; // rcx
  int v133; // eax
  __int64 v134; // rax
  int v135; // edx
  __int64 v136; // rdx
  __int64 *v137; // rax
  __int64 v138; // rcx
  unsigned __int64 v139; // rsi
  __int64 v140; // rcx
  __int64 v141; // rdx
  __int64 v142; // rcx
  _QWORD *v143; // rdx
  __int64 *v144; // rax
  __int64 v145; // rcx
  unsigned __int64 v146; // rsi
  __int64 v147; // rcx
  __int64 i; // r14
  __int64 v149; // rdi
  unsigned __int64 v150; // rax
  __int64 *v151; // rax
  __int64 v152; // rsi
  unsigned __int64 v153; // rcx
  __int64 v154; // rcx
  __int64 v155; // r14
  __m128i *v156; // rax
  __m128i *v157; // rbx
  __int64 v158; // rsi
  __m128i *v159; // r14
  __int64 v160; // rbx
  unsigned __int64 *v161; // rcx
  double v162; // xmm4_8
  double v163; // xmm5_8
  unsigned __int64 *v164; // rcx
  unsigned __int64 v165; // rdx
  double v166; // xmm4_8
  double v167; // xmm5_8
  __int64 v168; // rsi
  unsigned __int8 *v169; // rsi
  __int64 v170; // r14
  __int64 v171; // rbx
  __int64 v172; // rax
  __int64 v173; // r15
  __int64 v174; // rax
  __int64 v175; // rsi
  __int64 v176; // rdx
  __int64 v177; // rax
  int v178; // edx
  __int64 v179; // rdx
  __int64 *v180; // rax
  __int64 v181; // rcx
  unsigned __int64 v182; // rsi
  __int64 v183; // rcx
  __int64 v184; // rax
  __int64 v185; // rsi
  __int64 v186; // rdi
  unsigned int v187; // eax
  __int64 v188; // [rsp+8h] [rbp-468h]
  __int64 v190; // [rsp+20h] [rbp-450h]
  __int64 *v192; // [rsp+38h] [rbp-438h]
  __int64 v194; // [rsp+48h] [rbp-428h]
  __int64 *v195; // [rsp+50h] [rbp-420h]
  __int64 v196; // [rsp+50h] [rbp-420h]
  __int64 v197; // [rsp+50h] [rbp-420h]
  __int64 v198; // [rsp+58h] [rbp-418h]
  __int64 v199; // [rsp+58h] [rbp-418h]
  __int64 v200; // [rsp+60h] [rbp-410h]
  __int64 v201; // [rsp+60h] [rbp-410h]
  _QWORD *v203; // [rsp+68h] [rbp-408h]
  __int64 v204; // [rsp+68h] [rbp-408h]
  unsigned int v205; // [rsp+68h] [rbp-408h]
  __int64 v206; // [rsp+70h] [rbp-400h]
  __int64 v207; // [rsp+70h] [rbp-400h]
  __int64 v208; // [rsp+70h] [rbp-400h]
  __int64 v209; // [rsp+70h] [rbp-400h]
  __int64 v211; // [rsp+78h] [rbp-3F8h]
  const char *v212; // [rsp+80h] [rbp-3F0h] BYREF
  __int64 v213; // [rsp+88h] [rbp-3E8h]
  __int64 v214; // [rsp+90h] [rbp-3E0h]
  __m128 v215; // [rsp+98h] [rbp-3D8h]
  __int64 v216; // [rsp+A8h] [rbp-3C8h]
  __int64 v217; // [rsp+B0h] [rbp-3C0h]
  __m128 v218; // [rsp+B8h] [rbp-3B8h]
  __int64 v219; // [rsp+C8h] [rbp-3A8h]
  char v220; // [rsp+D0h] [rbp-3A0h]
  _BYTE *v221; // [rsp+D8h] [rbp-398h] BYREF
  __int64 v222; // [rsp+E0h] [rbp-390h]
  _BYTE v223[352]; // [rsp+E8h] [rbp-388h] BYREF
  char v224; // [rsp+248h] [rbp-228h]
  int v225; // [rsp+24Ch] [rbp-224h]
  __int64 v226; // [rsp+250h] [rbp-220h]
  __m128i v227; // [rsp+260h] [rbp-210h] BYREF
  __int64 v228; // [rsp+270h] [rbp-200h]
  __m128i v229; // [rsp+278h] [rbp-1F8h] BYREF
  __int64 v230; // [rsp+288h] [rbp-1E8h]
  __int64 v231; // [rsp+290h] [rbp-1E0h]
  __m128i v232; // [rsp+298h] [rbp-1D8h] BYREF
  __int64 v233; // [rsp+2A8h] [rbp-1C8h]
  char v234; // [rsp+2B0h] [rbp-1C0h]
  char *v235; // [rsp+2B8h] [rbp-1B8h] BYREF
  unsigned int v236; // [rsp+2C0h] [rbp-1B0h]
  char v237; // [rsp+2C8h] [rbp-1A8h] BYREF
  char v238; // [rsp+428h] [rbp-48h]
  int v239; // [rsp+42Ch] [rbp-44h]
  __int64 v240; // [rsp+430h] [rbp-40h]

  if ( !a1 )
    BUG();
  v15 = *(_QWORD *)(a1 + 32);
  v16 = a1;
  v194 = 0;
  v195 = 0;
  v198 = a1 | 4;
  while ( 1 )
  {
    v17 = v15 - 24;
    if ( !v15 )
      v17 = 0;
    v18 = v17;
    if ( a2 == v17 )
      break;
    if ( (unsigned __int8)sub_15F3040(v17) || sub_15F3330(v18) )
      goto LABEL_7;
    v25 = (_QWORD *)v18;
    if ( *(_BYTE *)(v18 + 16) == 54 && ((unsigned __int8)sub_15F3040(v16) || sub_15F3330(v16)) )
    {
      v26 = sub_15F2050(v18);
      v200 = sub_1632FA0(v26);
      sub_141EB40(&v227, (__int64 *)v18);
      if ( (sub_134F0E0(a6, v198, (__int64)&v227) & 2) != 0
        || !(unsigned __int8)sub_13F86A0(
                               *(_QWORD *)(v18 - 24),
                               1 << (*(unsigned __int16 *)(v18 + 18) >> 1) >> 1,
                               v200,
                               v18,
                               0) )
      {
        goto LABEL_7;
      }
    }
    v27 = 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
    v28 = (_QWORD *)(v18 - v27);
    if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
    {
      v28 = *(_QWORD **)(v18 - 8);
      v25 = &v28[(unsigned __int64)v27 / 8];
    }
    v29 = 0xAAAAAAAAAAAAAAABLL * (v27 >> 3);
    if ( !(v29 >> 2) )
    {
LABEL_40:
      if ( v29 == 2 )
      {
LABEL_92:
        if ( v16 != *v28 )
        {
          v28 += 3;
          if ( v16 != *v28 )
            goto LABEL_19;
          goto LABEL_44;
        }
      }
      else
      {
        if ( v29 != 3 )
        {
          if ( v29 != 1 || v16 != *v28 )
            goto LABEL_19;
LABEL_44:
          if ( v25 == v28 )
            goto LABEL_19;
LABEL_7:
          if ( !(unsigned __int8)sub_15F34B0(v18) )
            return 0;
          v19 = *(unsigned __int8 *)(v18 + 16) - 24;
          if ( v19 > 0x1C || ((1LL << v19) & 0x1C019800) == 0 )
            return 0;
          if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
          {
            v21 = *(_QWORD **)(v18 - 8);
            v22 = v21[3];
            if ( v16 != *v21 )
            {
LABEL_13:
              if ( v22 != v16 )
                return 0;
LABEL_14:
              v23 = *(_QWORD *)(v18 + 8);
              if ( v23 )
              {
                if ( !*(_QWORD *)(v23 + 8) )
                {
                  v24 = sub_1648700(v23);
                  if ( *((_BYTE *)v24 + 16) == 25 )
                  {
                    v195 = (__int64 *)sub_1A7A5C0((__int64)v24, v16);
                    if ( v195 )
                    {
                      v194 = v18;
                      goto LABEL_19;
                    }
                  }
                }
              }
              return 0;
            }
          }
          else
          {
            v31 = (_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
            v22 = v31[3];
            if ( v16 != *v31 )
              goto LABEL_13;
          }
          if ( v22 == v16 )
            return 0;
          goto LABEL_14;
        }
        if ( v16 != *v28 )
        {
          v28 += 3;
          goto LABEL_92;
        }
      }
LABEL_34:
      if ( v25 == v28 )
        goto LABEL_19;
      goto LABEL_7;
    }
    v30 = &v28[12 * (v29 >> 2)];
    while ( 1 )
    {
      if ( v16 == *v28 )
        goto LABEL_34;
      if ( v16 == v28[3] )
      {
        if ( v25 == v28 + 3 )
          goto LABEL_19;
        goto LABEL_7;
      }
      if ( v16 == v28[6] )
      {
        if ( v25 == v28 + 6 )
          goto LABEL_19;
        goto LABEL_7;
      }
      if ( v16 == v28[9] )
        break;
      v28 += 12;
      if ( v30 == v28 )
      {
        v29 = 0xAAAAAAAAAAAAAAABLL * (v25 - v28);
        goto LABEL_40;
      }
    }
    if ( v25 != v28 + 9 )
      goto LABEL_7;
LABEL_19:
    v15 = *(_QWORD *)(v15 + 8);
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 1 )
  {
    v32 = *(_QWORD *)(a2 - 24);
    if ( v32 != v16 && *(_BYTE *)(v32 + 16) != 9 && !v195 )
    {
      v195 = 0;
      v186 = sub_1A7A5C0(0, v16);
      if ( !v186 )
      {
        v187 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( v187 )
          v186 = *(_QWORD *)(a2 - 24LL * v187);
        if ( !sub_1A7A490(v186, v16, a2) )
          return 0;
        v195 = (__int64 *)sub_1A7A5C0(a2, v16);
        if ( !v195 )
          return 0;
      }
    }
  }
  v190 = *(_QWORD *)(a2 + 40);
  v201 = *(_QWORD *)(v190 + 56);
  v33 = sub_15E0530(*a15);
  if ( sub_1602790(v33)
    || (v78 = sub_15E0530(*a15),
        v79 = sub_16033E0(v78),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v79 + 48LL))(v79)) )
  {
    sub_15CA3B0((__int64)&v227, (__int64)"tailcallelim", (__int64)"tailcall-recursion", 18, v16);
    sub_15CAB20((__int64)&v227, "transforming tail recursion into loop", 0x25u);
    a7 = (__m128)_mm_loadu_si128(&v229);
    a8 = _mm_loadu_si128(&v232);
    LODWORD(v213) = v227.m128i_i32[2];
    v215 = a7;
    BYTE4(v213) = v227.m128i_i8[12];
    v218 = (__m128)a8;
    v214 = v228;
    v216 = v230;
    v212 = (const char *)&unk_49ECF68;
    v217 = v231;
    v220 = v234;
    if ( v234 )
      v219 = v233;
    v221 = v223;
    v222 = 0x400000000LL;
    if ( v236 )
    {
      sub_1A7B480((__int64)&v221, (__int64)&v235);
      v35 = v235;
      v224 = v238;
      v225 = v239;
      v226 = v240;
      v212 = (const char *)&unk_49ECF98;
      v227.m128i_i64[0] = (__int64)&unk_49ECF68;
      v34 = &v235[88 * v236];
      if ( v235 != v34 )
      {
        do
        {
          v34 -= 88;
          v36 = (char *)*((_QWORD *)v34 + 4);
          if ( v36 != v34 + 48 )
            j_j___libc_free_0(v36, *((_QWORD *)v34 + 6) + 1LL);
          if ( *(char **)v34 != v34 + 16 )
            j_j___libc_free_0(*(_QWORD *)v34, *((_QWORD *)v34 + 2) + 1LL);
        }
        while ( v35 != v34 );
        v34 = v235;
      }
    }
    else
    {
      v34 = v235;
      v224 = v238;
      v225 = v239;
      v226 = v240;
      v212 = (const char *)&unk_49ECF98;
    }
    if ( v34 != &v237 )
      _libc_free((unsigned __int64)v34);
    sub_143AA50(a15, (__int64)&v212);
    v39 = v221;
    v212 = (const char *)&unk_49ECF68;
    v40 = &v221[88 * (unsigned int)v222];
    if ( v221 != (_BYTE *)v40 )
    {
      do
      {
        v40 -= 11;
        v41 = (_QWORD *)v40[4];
        if ( v41 != v40 + 6 )
          j_j___libc_free_0(v41, v40[6] + 1LL);
        if ( (_QWORD *)*v40 != v40 + 2 )
          j_j___libc_free_0(*v40, v40[2] + 1LL);
      }
      while ( v39 != v40 );
      v40 = v221;
    }
    if ( v40 != (_QWORD *)v223 )
      _libc_free((unsigned __int64)v40);
  }
  if ( *a3 )
    goto LABEL_82;
  v50 = *(_QWORD *)(v201 + 80);
  LOWORD(v228) = 257;
  if ( v50 )
    v50 -= 24;
  *a3 = v50;
  v51 = sub_15E0530(v201);
  v203 = (_QWORD *)sub_22077B0(64);
  if ( v203 )
    sub_157FB60(v203, v51, (__int64)&v227, v201, v50);
  sub_164B7C0((__int64)v203, *a3);
  v52 = *a3;
  v227.m128i_i64[0] = (__int64)"tailrecurse";
  LOWORD(v228) = 259;
  sub_164B780(v52, v227.m128i_i64);
  v53 = 1;
  v54 = *a3;
  v55 = sub_1648A60(56, 1u);
  if ( v55 )
  {
    v53 = v54;
    sub_15F8590((__int64)v55, v54, (__int64)v203);
  }
  v56 = (*(_WORD *)(v16 + 18) & 3u) - 1 <= 1;
  *a4 = v56;
  if ( !v56 )
  {
    v199 = *(_QWORD *)(*a3 + 48);
    goto LABEL_103;
  }
  v170 = *(_QWORD *)(*a3 + 48);
  v171 = *a3 + 40;
  v172 = v203[6];
  v199 = v170;
  if ( v171 != v170 )
  {
    v173 = v172 - 24;
    if ( !v172 )
      v173 = 0;
    do
    {
      v174 = v170;
      v170 = *(_QWORD *)(v170 + 8);
      if ( *(_BYTE *)(v174 - 8) == 53 && *(_BYTE *)(*(_QWORD *)(v174 - 48) + 16LL) == 13 )
      {
        v53 = v173;
        sub_15F22F0((_QWORD *)(v174 - 24), v173);
      }
    }
    while ( v171 != v170 );
    v199 = *(_QWORD *)(*a3 + 48);
LABEL_103:
    if ( !v199 )
      goto LABEL_105;
  }
  v199 -= 24;
LABEL_105:
  if ( (*(_BYTE *)(v201 + 18) & 1) != 0 )
  {
    sub_15E08E0(v201, v53);
    v57 = *(__int64 **)(v201 + 88);
    if ( (*(_BYTE *)(v201 + 18) & 1) != 0 )
      sub_15E08E0(v201, v53);
    v58 = *(__int64 **)(v201 + 88);
  }
  else
  {
    v57 = *(__int64 **)(v201 + 88);
    v58 = v57;
  }
  v192 = &v58[5 * *(_QWORD *)(v201 + 96)];
  if ( v192 != v57 )
  {
    v59 = v57;
    v188 = v16;
    do
    {
      v212 = sub_1649960((__int64)v59);
      LOWORD(v228) = 773;
      v213 = v68;
      v227.m128i_i64[0] = (__int64)&v212;
      v227.m128i_i64[1] = (__int64)".tr";
      v206 = *v59;
      v69 = sub_1648B60(64);
      v72 = v69;
      if ( v69 )
      {
        v73 = v69;
        sub_15F1EA0(v69, v206, 53, 0, 0, v199);
        *(_DWORD *)(v72 + 56) = 2;
        sub_164B780(v72, v227.m128i_i64);
        sub_1648880(v72, *(_DWORD *)(v72 + 56), 1);
      }
      else
      {
        v73 = 0;
      }
      sub_164D160((__int64)v59, v72, a7, *(double *)a8.m128i_i64, a9, a10, v70, v71, a13, a14);
      v75 = *(_DWORD *)(v72 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v75 == *(_DWORD *)(v72 + 56) )
      {
        sub_15F55D0(v72, v72, v75, v74, v37, v38);
        LODWORD(v75) = *(_DWORD *)(v72 + 20) & 0xFFFFFFF;
      }
      v76 = ((_DWORD)v75 + 1) & 0xFFFFFFF;
      v77 = v76 | *(_DWORD *)(v72 + 20) & 0xF0000000;
      *(_DWORD *)(v72 + 20) = v77;
      if ( (v77 & 0x40000000) != 0 )
        v60 = *(_QWORD *)(v72 - 8);
      else
        v60 = v73 - 24 * v76;
      v61 = (__int64 **)(v60 + 24LL * (unsigned int)(v76 - 1));
      if ( *v61 )
      {
        v62 = v61[1];
        v63 = (unsigned __int64)v61[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v63 = v62;
        if ( v62 )
          v62[2] = v62[2] & 3 | v63;
      }
      *v61 = v59;
      v64 = v59[1];
      v61[1] = (__int64 *)v64;
      if ( v64 )
        *(_QWORD *)(v64 + 16) = (unsigned __int64)(v61 + 1) | *(_QWORD *)(v64 + 16) & 3LL;
      v61[2] = (__int64 *)((unsigned __int64)(v59 + 1) | (unsigned __int64)v61[2] & 3);
      v59[1] = (__int64)v61;
      v65 = *(_DWORD *)(v72 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v72 + 23) & 0x40) != 0 )
        v66 = *(_QWORD *)(v72 - 8);
      else
        v66 = v73 - 24 * v65;
      *(_QWORD *)(v66 + 8LL * (unsigned int)(v65 - 1) + 24LL * *(unsigned int *)(v72 + 56) + 8) = v203;
      v67 = *(unsigned int *)(a5 + 8);
      if ( (unsigned int)v67 >= *(_DWORD *)(a5 + 12) )
      {
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v37, v38);
        v67 = *(unsigned int *)(a5 + 8);
      }
      v59 += 5;
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v67) = v72;
      ++*(_DWORD *)(a5 + 8);
    }
    while ( v59 != v192 );
    v16 = v188;
    if ( !*a4 )
      goto LABEL_84;
    goto LABEL_83;
  }
LABEL_82:
  if ( !*a4 )
    goto LABEL_84;
LABEL_83:
  if ( (*(_WORD *)(v16 + 18) & 3u) - 1 > 1 )
    return 0;
LABEL_84:
  v42 = *(_DWORD *)(v16 + 20) & 0xFFFFFFF;
  if ( *(char *)(v16 + 23) >= 0 )
    goto LABEL_129;
  v43 = sub_1648A40(v16);
  v45 = v43 + v44;
  if ( *(char *)(v16 + 23) >= 0 )
  {
    if ( (unsigned int)(v45 >> 4) )
LABEL_290:
      BUG();
LABEL_129:
    v49 = 0;
    goto LABEL_130;
  }
  if ( !(unsigned int)((v45 - sub_1648A40(v16)) >> 4) )
    goto LABEL_129;
  if ( *(char *)(v16 + 23) >= 0 )
    goto LABEL_290;
  v46 = *(_DWORD *)(sub_1648A40(v16) + 8);
  if ( *(char *)(v16 + 23) >= 0 )
    BUG();
  v47 = sub_1648A40(v16);
  v49 = *(_DWORD *)(v47 + v48 - 4) - v46;
LABEL_130:
  v80 = 0;
  v81 = v42 - 1 - v49;
  if ( v81 )
  {
    v82 = v81;
    do
    {
      v90 = *(_QWORD *)(*(_QWORD *)a5 + 8 * v80);
      v91 = v80 - (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
      v92 = *(_QWORD *)(v16 + 24 * v91);
      v93 = *(_DWORD *)(v90 + 20) & 0xFFFFFFF;
      if ( v93 == *(_DWORD *)(v90 + 56) )
      {
        v207 = v82;
        sub_15F55D0(*(_QWORD *)(*(_QWORD *)a5 + 8 * v80), a5, v91, v82, v37, v38);
        v82 = v207;
        v93 = *(_DWORD *)(v90 + 20) & 0xFFFFFFF;
      }
      v94 = (v93 + 1) & 0xFFFFFFF;
      v95 = v94 | *(_DWORD *)(v90 + 20) & 0xF0000000;
      *(_DWORD *)(v90 + 20) = v95;
      if ( (v95 & 0x40000000) != 0 )
        v83 = *(_QWORD *)(v90 - 8);
      else
        v83 = v90 - 24 * v94;
      v84 = (_QWORD *)(v83 + 24LL * (unsigned int)(v94 - 1));
      if ( *v84 )
      {
        v85 = v84[1];
        v86 = v84[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v86 = v85;
        if ( v85 )
        {
          v37 = *(_QWORD *)(v85 + 16) & 3LL;
          *(_QWORD *)(v85 + 16) = v37 | v86;
        }
      }
      *v84 = v92;
      if ( v92 )
      {
        v87 = *(_QWORD *)(v92 + 8);
        v37 = v92 + 8;
        v84[1] = v87;
        if ( v87 )
        {
          v38 = (__int64)(v84 + 1);
          *(_QWORD *)(v87 + 16) = (unsigned __int64)(v84 + 1) | *(_QWORD *)(v87 + 16) & 3LL;
        }
        v84[2] = v37 | v84[2] & 3LL;
        *(_QWORD *)(v92 + 8) = v84;
      }
      v88 = *(_DWORD *)(v90 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v90 + 23) & 0x40) != 0 )
        v89 = *(_QWORD *)(v90 - 8);
      else
        v89 = v90 - 24 * v88;
      ++v80;
      *(_QWORD *)(v89 + 8LL * (unsigned int)(v88 - 1) + 24LL * *(unsigned int *)(v90 + 56) + 8) = v190;
    }
    while ( v82 != v80 );
  }
  if ( v195 )
  {
    v96 = *a3;
    v97 = *(_QWORD *)(*a3 + 8);
    if ( v97 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v97) + 16) - 25) > 9u )
      {
        v97 = *(_QWORD *)(v97 + 8);
        if ( !v97 )
          goto LABEL_183;
      }
      v98 = *(_QWORD *)(v96 + 48);
      v99 = v97;
      LOWORD(v228) = 259;
      if ( v98 )
        v98 -= 24;
      v100 = 0;
      v227.m128i_i64[0] = (__int64)"accumulator.tr";
      while ( 1 )
      {
        v99 = *(_QWORD *)(v99 + 8);
        if ( !v99 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v99) + 16) - 25) <= 9u )
        {
          v99 = *(_QWORD *)(v99 + 8);
          ++v100;
          if ( !v99 )
            goto LABEL_157;
        }
      }
LABEL_157:
      v101 = v100 + 2;
    }
    else
    {
LABEL_183:
      v98 = *(_QWORD *)(v96 + 48);
      if ( v98 )
      {
        v98 -= 24;
        v101 = 1;
        v97 = 0;
      }
      else
      {
        v97 = 0;
        v101 = 1;
      }
      v227.m128i_i64[0] = (__int64)"accumulator.tr";
      LOWORD(v228) = 259;
    }
    v208 = *v195;
    v102 = sub_1648B60(64);
    v106 = v208;
    v107 = v102;
    if ( v102 )
    {
      v209 = v102;
      sub_15F1EA0(v102, v106, 53, 0, 0, v98);
      *(_DWORD *)(v107 + 56) = v101;
      sub_164B780(v107, v227.m128i_i64);
      v106 = *(unsigned int *)(v107 + 56);
      sub_1648880(v107, v106, 1);
    }
    else
    {
      v209 = 0;
    }
    if ( v97 )
    {
      v108 = sub_1648700(v97);
      v204 = v16;
      v109 = v107 + 8;
      v110 = v195;
      v111 = v107;
      v112 = v97;
LABEL_166:
      v113 = *(_QWORD *)(v201 + 80);
      v114 = v108[5];
      v115 = *(unsigned int *)(v111 + 56);
      if ( v113 )
        v113 -= 24;
      v116 = *(_DWORD *)(v111 + 20) & 0xFFFFFFF;
      if ( v113 == v114 )
      {
        if ( v116 == (_DWORD)v115 )
        {
          v197 = v114;
          sub_15F55D0(v111, v106, v103, v114, v115, v105);
          v114 = v197;
          v116 = *(_DWORD *)(v111 + 20) & 0xFFFFFFF;
        }
        v125 = (v116 + 1) & 0xFFFFFFF;
        v126 = v125 | *(_DWORD *)(v111 + 20) & 0xF0000000;
        *(_DWORD *)(v111 + 20) = v126;
        if ( (v126 & 0x40000000) != 0 )
          v127 = *(_QWORD *)(v111 - 8);
        else
          v127 = v209 - 24 * v125;
        v128 = (__int64 **)(v127 + 24LL * (unsigned int)(v125 - 1));
        if ( *v128 )
        {
          v129 = v128[1];
          v130 = (unsigned __int64)v128[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v130 = v129;
          if ( v129 )
          {
            v105 = v129[2] & 3;
            v129[2] = v105 | v130;
          }
        }
        *v128 = v110;
        v131 = v110[1];
        v128[1] = (__int64 *)v131;
        if ( v131 )
        {
          v105 = (unsigned __int64)(v128 + 1) | *(_QWORD *)(v131 + 16) & 3LL;
          *(_QWORD *)(v131 + 16) = v105;
        }
        v128[2] = (__int64 *)((unsigned __int64)v128[2] & 3 | (unsigned __int64)(v110 + 1));
        v110[1] = (__int64)v128;
      }
      else
      {
        if ( v116 == (_DWORD)v115 )
        {
          v196 = v114;
          sub_15F55D0(v111, v106, v103, v114, v115, v105);
          v114 = v196;
          v116 = *(_DWORD *)(v111 + 20) & 0xFFFFFFF;
        }
        v117 = (v116 + 1) & 0xFFFFFFF;
        v118 = v117 | *(_DWORD *)(v111 + 20) & 0xF0000000;
        *(_DWORD *)(v111 + 20) = v118;
        if ( (v118 & 0x40000000) != 0 )
          v119 = *(_QWORD *)(v111 - 8);
        else
          v119 = v209 - 24 * v117;
        v120 = (__int64 *)(v119 + 24LL * (unsigned int)(v117 - 1));
        if ( *v120 )
        {
          v121 = v120[1];
          v122 = v120[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v122 = v121;
          if ( v121 )
          {
            v105 = *(_QWORD *)(v121 + 16) & 3LL;
            *(_QWORD *)(v121 + 16) = v105 | v122;
          }
        }
        *v120 = v111;
        v123 = *(_QWORD *)(v111 + 8);
        v120[1] = v123;
        if ( v123 )
        {
          v105 = (__int64)(v120 + 1);
          *(_QWORD *)(v123 + 16) = (unsigned __int64)(v120 + 1) | *(_QWORD *)(v123 + 16) & 3LL;
        }
        v120[2] = v109 | v120[2] & 3;
        *(_QWORD *)(v111 + 8) = v120;
      }
      v124 = *(_DWORD *)(v111 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v111 + 23) & 0x40) != 0 )
        v104 = *(_QWORD *)(v111 - 8);
      else
        v104 = v209 - 24 * v124;
      *(_QWORD *)(v104 + 8LL * (unsigned int)(v124 - 1) + 24LL * *(unsigned int *)(v111 + 56) + 8) = v114;
      while ( 1 )
      {
        v112 = *(_QWORD *)(v112 + 8);
        if ( !v112 )
          break;
        v108 = sub_1648700(v112);
        v106 = *((unsigned __int8 *)v108 + 16);
        v103 = (unsigned int)(v106 - 25);
        if ( (unsigned __int8)(v106 - 25) <= 9u )
          goto LABEL_166;
      }
      v107 = v111;
      v16 = v204;
    }
    v132 = *(unsigned int *)(v107 + 56);
    v133 = *(_DWORD *)(v107 + 20) & 0xFFFFFFF;
    if ( v194 )
    {
      if ( v133 == (_DWORD)v132 )
      {
        sub_15F55D0(v107, v106, v103, v132, v104, v105);
        v133 = *(_DWORD *)(v107 + 20) & 0xFFFFFFF;
      }
      v134 = (v133 + 1) & 0xFFFFFFF;
      v135 = v134 | *(_DWORD *)(v107 + 20) & 0xF0000000;
      *(_DWORD *)(v107 + 20) = v135;
      if ( (v135 & 0x40000000) != 0 )
        v136 = *(_QWORD *)(v107 - 8);
      else
        v136 = v209 - 24 * v134;
      v137 = (__int64 *)(v136 + 24LL * (unsigned int)(v134 - 1));
      if ( *v137 )
      {
        v138 = v137[1];
        v139 = v137[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v139 = v138;
        if ( v138 )
          *(_QWORD *)(v138 + 16) = v139 | *(_QWORD *)(v138 + 16) & 3LL;
      }
      *v137 = v194;
      v140 = *(_QWORD *)(v194 + 8);
      v137[1] = v140;
      if ( v140 )
        *(_QWORD *)(v140 + 16) = (unsigned __int64)(v137 + 1) | *(_QWORD *)(v140 + 16) & 3LL;
      v137[2] = v137[2] & 3 | (v194 + 8);
      *(_QWORD *)(v194 + 8) = v137;
      v141 = *(_DWORD *)(v107 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v107 + 23) & 0x40) != 0 )
        v142 = *(_QWORD *)(v107 - 8);
      else
        v142 = v209 - 24 * v141;
      *(_QWORD *)(v142 + 8LL * (unsigned int)(v141 - 1) + 24LL * *(unsigned int *)(v107 + 56) + 8) = v190;
      if ( (*(_BYTE *)(v194 + 23) & 0x40) != 0 )
        v143 = *(_QWORD **)(v194 - 8);
      else
        v143 = (_QWORD *)(v194 - 24LL * (*(_DWORD *)(v194 + 20) & 0xFFFFFFF));
      v144 = &v143[3 * (*v143 != v16)];
      if ( *v144 )
      {
        v145 = v144[1];
        v146 = v144[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v146 = v145;
        if ( v145 )
          *(_QWORD *)(v145 + 16) = v146 | *(_QWORD *)(v145 + 16) & 3LL;
      }
      *v144 = v107;
      v147 = *(_QWORD *)(v107 + 8);
      v144[1] = v147;
      if ( v147 )
        *(_QWORD *)(v147 + 16) = (unsigned __int64)(v144 + 1) | *(_QWORD *)(v147 + 16) & 3LL;
      v144[2] = v144[2] & 3 | (v107 + 8);
      *(_QWORD *)(v107 + 8) = v144;
    }
    else
    {
      v205 = *(_DWORD *)(a2 + 20);
      v175 = v205;
      v176 = v205 & 0xFFFFFFF;
      if ( (v205 & 0xFFFFFFF) != 0 )
      {
        v176 = -3LL * (unsigned int)v176;
        v175 = *(_QWORD *)(a2 + 8 * v176);
        v194 = v175;
      }
      if ( v133 == (_DWORD)v132 )
      {
        sub_15F55D0(v107, v175, v176, v132, v104, v105);
        v133 = *(_DWORD *)(v107 + 20) & 0xFFFFFFF;
      }
      v177 = (v133 + 1) & 0xFFFFFFF;
      v178 = v177 | *(_DWORD *)(v107 + 20) & 0xF0000000;
      *(_DWORD *)(v107 + 20) = v178;
      if ( (v178 & 0x40000000) != 0 )
        v179 = *(_QWORD *)(v107 - 8);
      else
        v179 = v209 - 24 * v177;
      v180 = (__int64 *)(v179 + 24LL * (unsigned int)(v177 - 1));
      if ( *v180 )
      {
        v181 = v180[1];
        v182 = v180[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v182 = v181;
        if ( v181 )
          *(_QWORD *)(v181 + 16) = v182 | *(_QWORD *)(v181 + 16) & 3LL;
      }
      *v180 = v194;
      if ( v194 )
      {
        v183 = *(_QWORD *)(v194 + 8);
        v180[1] = v183;
        if ( v183 )
          *(_QWORD *)(v183 + 16) = (unsigned __int64)(v180 + 1) | *(_QWORD *)(v183 + 16) & 3LL;
        v180[2] = (v194 + 8) | v180[2] & 3;
        *(_QWORD *)(v194 + 8) = v180;
      }
      v184 = *(_DWORD *)(v107 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v107 + 23) & 0x40) != 0 )
        v185 = *(_QWORD *)(v107 - 8);
      else
        v185 = v209 - 24 * v184;
      *(_QWORD *)(v185 + 8LL * (unsigned int)(v184 - 1) + 24LL * *(unsigned int *)(v107 + 56) + 8) = v190;
    }
    for ( i = *(_QWORD *)(v201 + 80); v201 + 72 != i; i = *(_QWORD *)(i + 8) )
    {
      v149 = i - 24;
      if ( !i )
        v149 = 0;
      v150 = sub_157EBA0(v149);
      if ( *(_BYTE *)(v150 + 16) == 25 )
      {
        v151 = (__int64 *)(v150 - 24LL * (*(_DWORD *)(v150 + 20) & 0xFFFFFFF));
        if ( *v151 )
        {
          v152 = v151[1];
          v153 = v151[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v153 = v152;
          if ( v152 )
            *(_QWORD *)(v152 + 16) = *(_QWORD *)(v152 + 16) & 3LL | v153;
        }
        *v151 = v107;
        v154 = *(_QWORD *)(v107 + 8);
        v151[1] = v154;
        if ( v154 )
          *(_QWORD *)(v154 + 16) = (unsigned __int64)(v151 + 1) | *(_QWORD *)(v154 + 16) & 3LL;
        v151[2] = (v107 + 8) | v151[2] & 3;
        *(_QWORD *)(v107 + 8) = v151;
      }
    }
  }
  v155 = *a3;
  v156 = (__m128i *)sub_1648A60(56, 1u);
  v157 = v156;
  if ( v156 )
    sub_15F8320((__int64)v156, v155, a2);
  v158 = *(_QWORD *)(v16 + 48);
  v159 = v157 + 3;
  v227.m128i_i64[0] = v158;
  if ( v158 )
  {
    sub_1623A60((__int64)&v227, v158, 2);
    if ( v159 == &v227 )
    {
      if ( v227.m128i_i64[0] )
        sub_161E7C0((__int64)&v227, v227.m128i_i64[0]);
      goto LABEL_234;
    }
    v168 = v157[3].m128i_i64[0];
    if ( !v168 )
    {
LABEL_239:
      v169 = (unsigned __int8 *)v227.m128i_i64[0];
      v157[3].m128i_i64[0] = v227.m128i_i64[0];
      if ( v169 )
        sub_1623210((__int64)&v227, v169, (__int64)v157[3].m128i_i64);
      goto LABEL_234;
    }
LABEL_238:
    sub_161E7C0((__int64)v157[3].m128i_i64, v168);
    goto LABEL_239;
  }
  if ( v159 != &v227 )
  {
    v168 = v157[3].m128i_i64[0];
    if ( v168 )
      goto LABEL_238;
  }
LABEL_234:
  v160 = a2;
  sub_157EA20(v190 + 40, a2);
  v161 = *(unsigned __int64 **)(a2 + 32);
  v211 = *(_QWORD *)(a2 + 24);
  *v161 = v211 & 0xFFFFFFFFFFFFFFF8LL | *v161 & 7;
  *(_QWORD *)((v211 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v161;
  *(_QWORD *)(v160 + 24) &= 7uLL;
  *(_QWORD *)(v160 + 32) = 0;
  sub_164BEC0(
    v160,
    v211,
    v211 & 0xFFFFFFFFFFFFFFF8LL,
    (__int64)v161,
    a7,
    *(double *)a8.m128i_i64,
    a9,
    a10,
    v162,
    v163,
    a13,
    a14);
  sub_157EA20(v190 + 40, v16);
  v164 = *(unsigned __int64 **)(v16 + 32);
  v165 = *(_QWORD *)(v16 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  *v164 = v165 | *v164 & 7;
  *(_QWORD *)(v165 + 8) = v164;
  *(_QWORD *)(v16 + 24) &= 7uLL;
  *(_QWORD *)(v16 + 32) = 0;
  sub_164BEC0(v16, v16, v165, (__int64)v164, a7, *(double *)a8.m128i_i64, a9, a10, v166, v167, a13, a14);
  return 1;
}
