// Function: sub_1FA0CA0
// Address: 0x1fa0ca0
//
__int64 *__fastcall sub_1FA0CA0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v7; // rax
  char v8; // r8
  __int64 *v9; // rcx
  __int64 v10; // r15
  unsigned int v11; // r14d
  char *v12; // rax
  __int64 v13; // rsi
  char v14; // dl
  const void **v15; // rax
  __int64 *v16; // rdx
  __int64 *result; // rax
  unsigned int v18; // r11d
  __int64 v19; // r13
  __int64 v20; // rsi
  __int128 *v21; // r14
  __int64 *v22; // r12
  int v23; // eax
  __int64 *v24; // rbx
  __int64 *v25; // rdx
  __int64 v26; // rsi
  __int16 v27; // ax
  _QWORD *v28; // rax
  __int64 v29; // rdi
  __int64 (*v30)(); // r9
  unsigned __int8 *v31; // rax
  int v32; // eax
  __int64 *v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // r15
  __int64 v36; // rdx
  unsigned __int64 v37; // r15
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned int v43; // eax
  __int64 *v44; // r12
  __int128 v45; // rax
  __int64 *v46; // rdx
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rax
  char v59; // r15
  __int64 v60; // rsi
  __int64 v61; // r9
  __int64 *v62; // rcx
  __int64 *v63; // r14
  __int64 v64; // r14
  __int64 v65; // rdx
  __int64 v66; // r15
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // r14
  __int64 v70; // rsi
  __int64 *v71; // r12
  int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // rsi
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 *v77; // r10
  __int64 v78; // rdx
  __m128i v79; // rax
  bool v80; // r14
  __m128i v81; // xmm0
  __int64 v82; // r15
  __int64 v83; // rax
  __int8 v84; // dl
  __int64 v85; // rax
  char v86; // r14
  __int64 v87; // rax
  __int64 *v88; // rax
  unsigned __int8 (__fastcall *v89)(__int64, __int64, __int64, __int64 *, __int64 *); // r11
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rdx
  int v93; // r14d
  int v94; // eax
  __int64 *v95; // r14
  __int64 v96; // r12
  unsigned int v97; // r13d
  int v98; // eax
  __int64 *v99; // rbx
  __int64 *v100; // rdx
  __int64 v101; // rax
  __int64 v102; // r13
  __int64 *v103; // r15
  const void **v104; // rdx
  const void **v105; // r14
  unsigned __int64 v106; // rdx
  unsigned int v107; // ecx
  __int64 *v108; // r14
  __int64 v109; // rdx
  __int64 v110; // r13
  __int64 *v111; // r12
  int v112; // eax
  __int64 *v113; // rbx
  __int64 *v114; // rdx
  const void **v115; // r8
  __int64 *v116; // rcx
  __int64 *v117; // rdi
  __int64 *v118; // r14
  __int64 v119; // r14
  __int64 v120; // rdx
  __int64 v121; // r15
  __int128 v122; // [rsp-20h] [rbp-130h]
  __int64 v123; // [rsp+8h] [rbp-108h]
  int v124; // [rsp+10h] [rbp-100h]
  __int64 v125; // [rsp+10h] [rbp-100h]
  __int64 v126; // [rsp+18h] [rbp-F8h]
  __int64 v127; // [rsp+18h] [rbp-F8h]
  int v128; // [rsp+20h] [rbp-F0h]
  __int64 v129; // [rsp+28h] [rbp-E8h]
  __int64 v130; // [rsp+28h] [rbp-E8h]
  __int64 *v131; // [rsp+30h] [rbp-E0h]
  __int64 v132; // [rsp+38h] [rbp-D8h]
  __int64 v133; // [rsp+40h] [rbp-D0h]
  __int64 v134; // [rsp+40h] [rbp-D0h]
  unsigned __int8 (__fastcall *v135)(__int64, __int64, __int64, __int64 *, __int64 *); // [rsp+40h] [rbp-D0h]
  const void **v136; // [rsp+40h] [rbp-D0h]
  __int64 v137; // [rsp+48h] [rbp-C8h]
  __m128i v138; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v139; // [rsp+60h] [rbp-B0h]
  __int64 *v140; // [rsp+68h] [rbp-A8h]
  __int64 v141; // [rsp+70h] [rbp-A0h]
  __int64 v142; // [rsp+78h] [rbp-98h]
  unsigned int v143; // [rsp+80h] [rbp-90h] BYREF
  const void **v144; // [rsp+88h] [rbp-88h]
  __int64 v145; // [rsp+90h] [rbp-80h] BYREF
  int v146; // [rsp+98h] [rbp-78h]
  __m128i v147; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE *v148; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v149; // [rsp+B8h] [rbp-58h]
  _BYTE v150[80]; // [rsp+C0h] [rbp-50h] BYREF

  v7 = *(__int64 **)(a2 + 32);
  v8 = *(_BYTE *)(a1 + 24);
  v9 = (__int64 *)v7[1];
  v10 = *v7;
  v11 = *((_DWORD *)v7 + 2);
  v12 = *(char **)(a2 + 40);
  v140 = v9;
  v13 = *(_QWORD *)(a1 + 8);
  v14 = *v12;
  LOBYTE(v9) = *(_BYTE *)(a1 + 25);
  v15 = (const void **)*((_QWORD *)v12 + 1);
  LOBYTE(v143) = v14;
  v16 = *(__int64 **)a1;
  v144 = v15;
  result = (__int64 *)sub_1F7F730(a2, v13, v16, (char)v9, v8, a3, *(double *)a4.m128i_i64, a5);
  if ( result )
    return result;
  v18 = *(unsigned __int16 *)(v10 + 24);
  v19 = v10;
  if ( (unsigned __int16)(v18 - 142) > 2u )
  {
    if ( (_WORD)v18 == 145 )
    {
      v67 = sub_1F84730((_QWORD *)a1, v10, *(double *)a3.m128i_i64, a4, a5);
      if ( v67 )
      {
        if ( v67 != v10 )
        {
          v69 = **(_QWORD **)(v10 + 32);
          v149 = v68;
          v148 = (_BYTE *)v67;
          sub_1F994A0(a1, v10, (__int64 *)&v148, 1, 1);
          sub_1F81BC0(a1, v69);
        }
        return (__int64 *)a2;
      }
      v18 = *(unsigned __int16 *)(v10 + 24);
      if ( v18 == 145 )
      {
        v70 = *(_QWORD *)(a2 + 72);
        v71 = *(__int64 **)a1;
        v139 = (__int64 *)&v148;
        v148 = (_BYTE *)v70;
        if ( v70 )
          sub_1623A60((__int64)&v148, v70, 2);
        v72 = *(_DWORD *)(a2 + 64);
        v24 = v139;
        LODWORD(v149) = v72;
        result = (__int64 *)sub_1D321C0(
                              v71,
                              **(_QWORD **)(v10 + 32),
                              *(_QWORD *)(*(_QWORD *)(v10 + 32) + 8LL),
                              (__int64)v139,
                              v143,
                              v144,
                              *(double *)a3.m128i_i64,
                              *(double *)a4.m128i_i64,
                              *(double *)a5.m128i_i64);
        v26 = (__int64)v148;
        if ( v148 )
          goto LABEL_7;
        return result;
      }
    }
    if ( v18 == 118 )
    {
      v28 = *(_QWORD **)(v10 + 32);
      if ( *(_WORD *)(*v28 + 24LL) == 145 && *(_WORD *)(v28[5] + 24LL) == 10 )
      {
        v29 = *(_QWORD *)(a1 + 8);
        v30 = *(__int64 (**)())(*(_QWORD *)v29 + 800LL);
        if ( v30 == sub_1D12DF0
          || (v31 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(*v28 + 32LL) + 40LL)
                                      + 16LL * *(unsigned int *)(*(_QWORD *)(*v28 + 32LL) + 8LL)),
              !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v30)(
                 v29,
                 *v31,
                 *((_QWORD *)v31 + 1),
                 *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + 16LL * v11),
                 *(_QWORD *)(*(_QWORD *)(v10 + 40) + 16LL * v11 + 8))) )
        {
          v145 = *(_QWORD *)(a2 + 72);
          if ( v145 )
          {
            v140 = &v145;
            sub_1F6CA20(&v145);
          }
          v32 = *(_DWORD *)(a2 + 64);
          v33 = *(__int64 **)a1;
          v140 = &v145;
          v146 = v32;
          v34 = *(__int64 **)(**(_QWORD **)(v10 + 32) + 32LL);
          v35 = v34[1];
          v141 = sub_1D321C0(
                   v33,
                   *v34,
                   v35,
                   (__int64)&v145,
                   v143,
                   v144,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64);
          v142 = v36;
          v37 = (unsigned int)v36 | v35 & 0xFFFFFFFF00000000LL;
          v38 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v19 + 32) + 40LL) + 88LL) + 24LL;
          sub_13A38D0((__int64)&v147, v38);
          v43 = sub_1D159A0((char *)&v143, v38, v39, v40, v41, v42, v124, v126, v128, v129);
          sub_16A5C50((__int64)&v148, (const void **)&v147, v43);
          sub_1F6C9E0(v147.m128i_i64, (__int64 *)&v148);
          sub_135E100((__int64 *)&v148);
          v44 = *(__int64 **)a1;
          *(_QWORD *)&v45 = sub_1D38970(
                              (__int64)v44,
                              (__int64)&v147,
                              (__int64)&v145,
                              v143,
                              v144,
                              0,
                              a3,
                              *(double *)a4.m128i_i64,
                              a5,
                              0);
          v139 = sub_1D332F0(
                   v44,
                   118,
                   (__int64)&v145,
                   v143,
                   v144,
                   0,
                   *(double *)a3.m128i_i64,
                   *(double *)a4.m128i_i64,
                   a5,
                   v141,
                   v37,
                   v45);
          v140 = v46;
          sub_135E100(v147.m128i_i64);
          sub_17CD270(&v145);
          return v139;
        }
      }
    }
    if ( *(_BYTE *)(a1 + 25) )
    {
      v27 = *(_WORD *)(v10 + 24);
      if ( v27 != 185 )
        goto LABEL_12;
      if ( (*(_BYTE *)(v10 + 27) & 0xC) != 0 )
        goto LABEL_33;
      if ( !(_BYTE)v143 )
        return 0;
      if ( (unsigned __int8)(v143 - 14) <= 0x5Fu )
        return 0;
      if ( (*(_WORD *)(v10 + 26) & 0x380) != 0 )
        return 0;
      v134 = 16LL * v11;
      v73 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + v134);
      if ( !(_BYTE)v73
        || (unsigned __int8)*(_WORD *)(*(_QWORD *)(a1 + 8) + 2 * (v73 + 115LL * (unsigned __int8)v143 + 16104)) >> 4 )
      {
        return 0;
      }
      v148 = v150;
      v149 = 0x400000000LL;
      if ( sub_1D18C00(v10, 1, v11) )
      {
        v139 = (__int64 *)&v148;
        goto LABEL_61;
      }
      v123 = *(_QWORD *)(a1 + 8);
      v139 = (__int64 *)&v148;
      if ( (unsigned __int8)sub_1F6D830(v143, (__int64)v144, a2, v10, v11, 144, (__int64)&v148, v123) )
      {
LABEL_61:
        v74 = *(_QWORD *)(v10 + 104);
        v75 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + 16LL * v11);
        v76 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + v134 + 8);
        v77 = *(__int64 **)a1;
        v78 = *(_QWORD *)(v10 + 32);
        v147.m128i_i64[0] = *(_QWORD *)(a2 + 72);
        if ( v147.m128i_i64[0] )
        {
          v125 = v75;
          v127 = v76;
          v130 = v78;
          v131 = v77;
          v138.m128i_i64[0] = (__int64)&v147;
          sub_1F6CA20(v147.m128i_i64);
          v75 = v125;
          v76 = v127;
          v78 = v130;
          v77 = v131;
        }
        v147.m128i_i32[2] = *(_DWORD *)(a2 + 64);
        v79.m128i_i64[0] = sub_1D2B590(
                             v77,
                             1,
                             (__int64)&v147,
                             v143,
                             (__int64)v144,
                             v74,
                             *(_OWORD *)v78,
                             *(_QWORD *)(v78 + 40),
                             *(_QWORD *)(v78 + 48),
                             v75,
                             v76);
        v138 = v79;
        sub_17CD270(v147.m128i_i64);
        sub_1FA0970(
          (__int64 **)a1,
          (__int64)v139,
          v10,
          v11,
          v138.m128i_i64[0],
          v138.m128i_i64[1],
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          0x90u);
        v80 = sub_1D18C00(v10, 1, v11);
        v81 = _mm_load_si128(&v138);
        v139 = (__int64 *)&v147;
        v147 = v81;
        sub_1F994A0(a1, a2, v147.m128i_i64, 1, 1);
        if ( v80 )
        {
          sub_1D44C70(*(_QWORD *)a1, v10, 1, v138.m128i_i64[0], 1u);
        }
        else
        {
          v115 = *(const void ***)(*(_QWORD *)(v10 + 40) + v134 + 8);
          v116 = (__int64 *)*(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + v134);
          v117 = v139;
          v118 = *(__int64 **)a1;
          v140 = v139;
          v136 = v115;
          v139 = v116;
          sub_1F80610((__int64)v117, v10);
          v119 = sub_1D309E0(
                   v118,
                   145,
                   (__int64)v140,
                   (__int64)v139,
                   v136,
                   0,
                   *(double *)v81.m128i_i64,
                   *(double *)a4.m128i_i64,
                   *(double *)a5.m128i_i64,
                   *(_OWORD *)&v138);
          v121 = v120;
          sub_17CD270(v140);
          sub_1F9A400(a1, v19, v119, v121, v138.m128i_i64[0], 1, 1);
        }
        result = (__int64 *)a2;
        if ( v148 != v150 )
        {
          v139 = 0;
          v140 = (__int64 *)a2;
          _libc_free((unsigned __int64)v148);
          return v140;
        }
        return result;
      }
      if ( v148 != v150 )
        _libc_free((unsigned __int64)v148);
    }
    v27 = *(_WORD *)(v10 + 24);
    if ( v27 != 185 )
      goto LABEL_12;
    if ( (*(_BYTE *)(v10 + 27) & 0xC) == 0 )
      return 0;
LABEL_33:
    if ( (*(_WORD *)(v10 + 26) & 0x380) != 0 )
      return 0;
    if ( sub_1D18C00(v10, 1, v11) )
    {
      v56 = *(unsigned __int8 *)(v10 + 88);
      v57 = *(_QWORD *)(v10 + 96);
      v58 = v56;
      v59 = (*(_BYTE *)(v10 + 27) >> 2) & 3;
      if ( !*(_BYTE *)(a1 + 24)
        || (_BYTE)v143
        && (_BYTE)v56
        && (((int)*(unsigned __int16 *)(*(_QWORD *)(a1 + 8) + 2 * (v56 + 115LL * (unsigned __int8)v143 + 16104)) >> (4 * v59))
          & 0xF) == 0 )
      {
        v60 = *(_QWORD *)(a2 + 72);
        v61 = *(_QWORD *)(v19 + 104);
        v62 = *(__int64 **)(v19 + 32);
        v63 = *(__int64 **)a1;
        v139 = (__int64 *)&v148;
        v148 = (_BYTE *)v60;
        if ( v60 )
        {
          v133 = v58;
          v137 = v57;
          v138.m128i_i64[0] = v61;
          v140 = v62;
          sub_1F6CA20((__int64 *)&v148);
          v58 = v133;
          v57 = v137;
          v61 = v138.m128i_i64[0];
          v62 = v140;
        }
        LODWORD(v149) = *(_DWORD *)(a2 + 64);
        v64 = sub_1D2B590(v63, v59, (__int64)v139, v143, (__int64)v144, v61, *(_OWORD *)v62, v62[5], v62[6], v58, v57);
        v66 = v65;
        if ( v148 )
          sub_161E7C0((__int64)v139, (__int64)v148);
        v148 = (_BYTE *)v64;
        v149 = v66;
        sub_1F994A0(a1, a2, v139, 1, 1);
        sub_1D44C70(*(_QWORD *)a1, v19, 1, v64, 1u);
        return (__int64 *)a2;
      }
    }
    v27 = *(_WORD *)(v19 + 24);
LABEL_12:
    if ( v27 != 137 )
      return 0;
    if ( (_BYTE)v143 )
    {
      if ( (unsigned __int8)(v143 - 14) > 0x5Fu )
        goto LABEL_23;
    }
    else if ( !sub_1F58D20((__int64)&v143) )
    {
      goto LABEL_23;
    }
    if ( !*(_BYTE *)(a1 + 24) )
    {
      v82 = *(_QWORD *)(a1 + 8);
      v83 = *(_QWORD *)(**(_QWORD **)(v19 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v19 + 32) + 8LL);
      v84 = *(_BYTE *)v83;
      v147.m128i_i64[1] = *(_QWORD *)(v83 + 8);
      v147.m128i_i8[0] = v84;
      v85 = *(_QWORD *)(v19 + 40) + 16LL * v11;
      v86 = *(_BYTE *)v85;
      v87 = *(_QWORD *)(v85 + 8);
      v139 = (__int64 *)v147.m128i_i64[1];
      v132 = v87;
      v88 = *(__int64 **)a1;
      v89 = *(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64 *, __int64 *))(*(_QWORD *)v82 + 264LL);
      v140 = (__int64 *)v147.m128i_i64[0];
      v90 = v88[4];
      v135 = v89;
      v138.m128i_i64[0] = v88[6];
      v91 = sub_1E0A0C0(v90);
      if ( v86 != v135(v82, v91, v138.m128i_i64[0], v140, v139) || !v86 && v132 != v92 )
      {
        if ( (_BYTE)v143 )
          v93 = sub_1F6C8D0(v143);
        else
          v93 = sub_1F58D40((__int64)&v143);
        if ( v147.m128i_i8[0] )
          v94 = sub_1F6C8D0(v147.m128i_i8[0]);
        else
          v94 = sub_1F58D40((__int64)&v147);
        if ( v94 == v93 )
        {
          v95 = *(__int64 **)a1;
          v96 = *(_QWORD *)(v19 + 32);
          v139 = (__int64 *)&v148;
          v97 = *(_DWORD *)(*(_QWORD *)(v96 + 80) + 84LL);
          v148 = *(_BYTE **)(a2 + 72);
          if ( v148 )
            sub_1F6CA20((__int64 *)&v148);
          v98 = *(_DWORD *)(a2 + 64);
          v99 = v139;
          LODWORD(v149) = v98;
          v139 = sub_1F81070(
                   v95,
                   (__int64)v139,
                   v143,
                   v144,
                   *(_QWORD *)v96,
                   *(__int16 **)(v96 + 8),
                   (__m128)a3,
                   *(double *)a4.m128i_i64,
                   a5,
                   *(_OWORD *)(v96 + 40),
                   v97);
          v140 = v100;
          sub_17CD270(v99);
          return v139;
        }
        else
        {
          v101 = sub_1F7DF20(&v147);
          v102 = *(_QWORD *)(v19 + 32);
          v103 = *(__int64 **)a1;
          v105 = v104;
          v106 = v101;
          v139 = (__int64 *)&v148;
          v107 = *(_DWORD *)(*(_QWORD *)(v102 + 80) + 84LL);
          v148 = *(_BYTE **)(a2 + 72);
          if ( v148 )
          {
            v138.m128i_i64[0] = v101;
            LODWORD(v140) = v107;
            sub_1F6CA20((__int64 *)&v148);
            v106 = v138.m128i_i64[0];
            v107 = (unsigned int)v140;
          }
          LODWORD(v149) = *(_DWORD *)(a2 + 64);
          v108 = sub_1F81070(
                   v103,
                   (__int64)v139,
                   v106,
                   v105,
                   *(_QWORD *)v102,
                   *(__int16 **)(v102 + 8),
                   (__m128)a3,
                   *(double *)a4.m128i_i64,
                   a5,
                   *(_OWORD *)(v102 + 40),
                   v107);
          v110 = v109;
          sub_17CD270(v139);
          v111 = *(__int64 **)a1;
          v148 = *(_BYTE **)(a2 + 72);
          if ( v148 )
            sub_1F6CA20(v139);
          v112 = *(_DWORD *)(a2 + 64);
          v113 = v139;
          LODWORD(v149) = v112;
          v139 = (__int64 *)sub_1D321C0(
                              v111,
                              (__int64)v108,
                              v110,
                              (__int64)v139,
                              v143,
                              v144,
                              *(double *)a3.m128i_i64,
                              *(double *)a4.m128i_i64,
                              *(double *)a5.m128i_i64);
          v140 = v114;
          sub_17CD270(v113);
          return v139;
        }
      }
      return 0;
    }
LABEL_23:
    v47 = *(_QWORD *)(a2 + 72);
    v139 = (__int64 *)&v148;
    v148 = (_BYTE *)v47;
    if ( v47 )
      sub_1623A60((__int64)&v148, v47, 2);
    v48 = *(_DWORD *)(a2 + 64);
    v24 = v139;
    LODWORD(v149) = v48;
    v49 = *(_QWORD *)a1;
    LODWORD(v140) = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v19 + 32) + 80LL) + 84LL);
    v50 = sub_1D38BB0(v49, 0, (__int64)v139, v143, v144, 0, a3, *(double *)a4.m128i_i64, a5, 0);
    v52 = v51;
    v53 = v50;
    v54 = sub_1D38BB0(*(_QWORD *)a1, 1, (__int64)v139, v143, v144, 0, a3, *(double *)a4.m128i_i64, a5, 0);
    *((_QWORD *)&v122 + 1) = v52;
    *(_QWORD *)&v122 = v53;
    result = sub_1F87CB0(
               (_QWORD *)a1,
               (__int64)v139,
               **(_QWORD **)(v19 + 32),
               *(_QWORD *)(*(_QWORD *)(v19 + 32) + 8LL),
               *(_QWORD *)(*(_QWORD *)(v19 + 32) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v19 + 32) + 48LL),
               a3,
               *(double *)a4.m128i_i64,
               a5,
               v54,
               v55,
               v122,
               (unsigned int)v140,
               1);
    v26 = (__int64)v148;
    if ( result )
    {
      if ( !v148 )
        return result;
LABEL_7:
      v139 = v25;
      v140 = result;
      sub_161E7C0((__int64)v24, v26);
      return v140;
    }
    if ( v148 )
    {
      sub_161E7C0((__int64)v139, (__int64)v148);
      return 0;
    }
    return 0;
  }
  v20 = *(_QWORD *)(a2 + 72);
  v21 = *(__int128 **)(v10 + 32);
  v139 = (__int64 *)&v148;
  v22 = *(__int64 **)a1;
  v148 = (_BYTE *)v20;
  if ( v20 )
  {
    sub_1623A60((__int64)&v148, v20, 2);
    v18 = *(unsigned __int16 *)(v10 + 24);
  }
  v23 = *(_DWORD *)(a2 + 64);
  v24 = v139;
  LODWORD(v149) = v23;
  result = (__int64 *)sub_1D309E0(
                        v22,
                        v18,
                        (__int64)v139,
                        v143,
                        v144,
                        0,
                        *(double *)a3.m128i_i64,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        *v21);
  v26 = (__int64)v148;
  if ( v148 )
    goto LABEL_7;
  return result;
}
