// Function: sub_1FAA420
// Address: 0x1faa420
//
__int64 __fastcall sub_1FAA420(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r13
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  unsigned __int64 v11; // rcx
  unsigned int v12; // edi
  __int16 v13; // dx
  __int64 v14; // rax
  char v15; // r15
  const void **v16; // rax
  __int16 v17; // r8
  bool v18; // r14
  __int64 v19; // rax
  unsigned int v20; // ecx
  bool v21; // dl
  __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // r10
  char v25; // r14
  unsigned int v26; // r14d
  bool v27; // al
  __int64 v28; // r12
  __int64 v29; // rsi
  __int64 v30; // r12
  bool v32; // al
  __int64 *v33; // rax
  unsigned __int64 v34; // r14
  __int64 v35; // r9
  unsigned int v36; // ecx
  char v37; // al
  bool v38; // zf
  __int64 v39; // rsi
  __int64 *v40; // rax
  __int64 v41; // rsi
  unsigned int v42; // r14d
  bool v43; // al
  __int64 v44; // rsi
  __int64 *v45; // r12
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 *v48; // rax
  __int64 v49; // rsi
  __int64 *v50; // r12
  __int64 *v51; // rax
  unsigned __int64 v52; // r14
  char v53; // al
  char v54; // cl
  char v55; // cl
  __int64 v56; // rdx
  unsigned int v57; // eax
  char v58; // cl
  __int64 v59; // rdx
  unsigned int v60; // edx
  __int64 v61; // r15
  int v62; // eax
  unsigned __int64 v63; // r15
  __int64 v64; // rsi
  unsigned int v65; // eax
  __int64 *v66; // r15
  char v67; // r8
  __int64 v68; // rbx
  unsigned __int8 *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  const void **v72; // rdx
  __int128 v73; // rax
  __int128 v74; // rax
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  char v77; // al
  char v78; // r14
  __int64 v79; // rsi
  __int64 *v80; // rax
  __int64 v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rbx
  unsigned __int8 *v84; // rax
  __int64 v85; // rax
  unsigned int v86; // eax
  const void **v87; // rdx
  __int128 v88; // rax
  __int64 v89; // rax
  unsigned __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rsi
  __int64 *v93; // r15
  __int64 v94; // r13
  __int128 v95; // rax
  __int128 v96; // kr10_16
  __int64 v97; // rsi
  __int64 *v98; // r12
  __int64 v99; // r13
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 *v102; // r15
  __int128 v103; // rax
  __int64 v104; // r13
  __int64 *v105; // r12
  __int64 v106; // rax
  __int64 v107; // r15
  __int64 v108; // rbx
  __int64 v109; // rbx
  __int64 v110; // rax
  _QWORD *v111; // rax
  __int64 *v112; // r15
  __int128 v113; // rax
  __int64 *v114; // r12
  __int64 *v115; // r12
  unsigned __int64 v116; // rdx
  unsigned __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // rbx
  __int64 *v120; // r14
  __int64 v121; // rcx
  __int64 v122; // rdx
  __int64 *v123; // rdx
  __int64 v124; // r12
  __int64 v125; // r15
  __int64 v126; // rax
  __int128 v127; // [rsp-10h] [rbp-110h]
  char v128; // [rsp+8h] [rbp-F8h]
  __int64 v129; // [rsp+8h] [rbp-F8h]
  __int16 v130; // [rsp+10h] [rbp-F0h]
  __int64 v131; // [rsp+10h] [rbp-F0h]
  bool v132; // [rsp+18h] [rbp-E8h]
  char v133; // [rsp+18h] [rbp-E8h]
  __int16 v134; // [rsp+18h] [rbp-E8h]
  char v135; // [rsp+18h] [rbp-E8h]
  unsigned int v136; // [rsp+18h] [rbp-E8h]
  bool v137; // [rsp+18h] [rbp-E8h]
  __int64 v138; // [rsp+18h] [rbp-E8h]
  __int64 v139; // [rsp+18h] [rbp-E8h]
  __int64 v140; // [rsp+20h] [rbp-E0h]
  __int64 v141; // [rsp+20h] [rbp-E0h]
  __int128 v142; // [rsp+20h] [rbp-E0h]
  __int64 v143; // [rsp+20h] [rbp-E0h]
  __int64 v144; // [rsp+20h] [rbp-E0h]
  __int64 v145; // [rsp+30h] [rbp-D0h]
  __int64 v146; // [rsp+30h] [rbp-D0h]
  _QWORD *v147; // [rsp+30h] [rbp-D0h]
  __int128 v148; // [rsp+30h] [rbp-D0h]
  __int64 v149; // [rsp+40h] [rbp-C0h]
  __int128 v150; // [rsp+40h] [rbp-C0h]
  __int64 v151; // [rsp+40h] [rbp-C0h]
  __int64 v152; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v153; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v154; // [rsp+48h] [rbp-B8h]
  __int64 v155; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v156; // [rsp+58h] [rbp-A8h]
  char v157; // [rsp+58h] [rbp-A8h]
  char v158; // [rsp+58h] [rbp-A8h]
  __int64 *v159; // [rsp+58h] [rbp-A8h]
  _QWORD *v160; // [rsp+60h] [rbp-A0h]
  __int128 v161; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v162; // [rsp+60h] [rbp-A0h]
  __int128 v163; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v164; // [rsp+68h] [rbp-98h]
  __int64 v165; // [rsp+68h] [rbp-98h]
  unsigned int v166; // [rsp+70h] [rbp-90h] BYREF
  const void **v167; // [rsp+78h] [rbp-88h]
  unsigned __int64 v168; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v169; // [rsp+88h] [rbp-78h]
  const void *v170; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v171; // [rsp+98h] [rbp-68h]
  __int64 v172; // [rsp+A0h] [rbp-60h] BYREF
  int v173; // [rsp+A8h] [rbp-58h]
  __int64 v174; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v175; // [rsp+B8h] [rbp-48h]
  __int64 v176; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v177; // [rsp+C8h] [rbp-38h]

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)v7;
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v11 = *(_QWORD *)(v7 + 40);
  v12 = *(_DWORD *)(v7 + 48);
  v164 = v9.m128i_u64[1];
  v13 = *(_WORD *)(*(_QWORD *)v7 + 24LL);
  v156 = v11;
  v155 = *(unsigned int *)(v7 + 8);
  v140 = 16 * v155;
  v14 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16 * v155;
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  v153 = v10.m128i_u64[1];
  LOBYTE(v166) = v15;
  v167 = v16;
  if ( v13 == 48 || (v17 = *(_WORD *)(v11 + 24), v17 == 48) )
  {
    v29 = *(_QWORD *)(a2 + 72);
    v30 = *(_QWORD *)a1;
    v176 = v29;
    if ( v29 )
      sub_1623A60((__int64)&v176, v29, 2);
    v177 = *(_DWORD *)(v6 + 64);
    v28 = sub_1D38BB0(v30, 0, (__int64)&v176, v166, v167, 0, v9, *(double *)v10.m128i_i64, a5, 0);
    if ( v176 )
      sub_161E7C0((__int64)&v176, v176);
    return v28;
  }
  v169 = 1;
  v168 = 0;
  v171 = 1;
  v170 = 0;
  if ( v15 )
  {
    if ( (unsigned __int8)(v15 - 14) > 0x5Fu )
      goto LABEL_5;
LABEL_25:
    v33 = sub_1FA8C50(a1, a2, *(double *)v9.m128i_i64, *(double *)v10.m128i_i64, a5);
    if ( v33 )
      goto LABEL_26;
    v135 = sub_1D1A8B0(v8, (__int64)&v168);
    v37 = sub_1D1A8B0(v156, (__int64)&v170);
    v38 = ((unsigned __int8)v37 & (unsigned __int8)v135) == 0;
    v133 = v37 & v135;
    v25 = v37;
    if ( v38 )
    {
      if ( !sub_1D23600(*(_QWORD *)a1, v9.m128i_i64[0]) )
        goto LABEL_14;
      goto LABEL_13;
    }
    v24 = *(_QWORD **)a1;
    goto LABEL_53;
  }
  v130 = v13;
  v134 = v17;
  v32 = sub_1F58D20((__int64)&v166);
  v17 = v134;
  v13 = v130;
  if ( v32 )
    goto LABEL_25;
LABEL_5:
  v18 = v13 == 10 || v13 == 32;
  if ( !v18 )
  {
    if ( v17 == 10 || v17 == 32 )
    {
      v21 = 0;
      v22 = *(_QWORD *)(v156 + 88);
      v23 = v22 + 24;
      goto LABEL_49;
    }
    goto LABEL_35;
  }
  v19 = *(_QWORD *)(v8 + 88);
  v20 = *(_DWORD *)(v19 + 32);
  if ( v20 <= 0x40 )
  {
    v169 = *(_DWORD *)(v19 + 32);
    v168 = *(_QWORD *)(v19 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v20);
    v21 = v17 == 32 || v17 == 10;
    if ( v21 )
    {
      v22 = *(_QWORD *)(v156 + 88);
      v18 = (*(_BYTE *)(v8 + 26) & 8) != 0;
      v23 = v22 + 24;
      goto LABEL_49;
    }
LABEL_35:
    v25 = 0;
    v133 = 0;
    if ( !sub_1D23600(*(_QWORD *)a1, v9.m128i_i64[0]) )
    {
LABEL_36:
      v33 = sub_1F77C50((__int64 **)a1, v6, *(double *)v9.m128i_i64, *(double *)v10.m128i_i64, a5);
      if ( !v33 )
      {
        v149 = v156;
        v131 = v12;
        v34 = v12 | v153 & 0xFFFFFFFF00000000LL;
        v154 = v34;
        if ( !(unsigned __int8)sub_1F70310(v156, v34, 1u) || !(unsigned __int8)sub_1D208B0(*(_QWORD *)a1, v156, v34) )
          goto LABEL_38;
        v78 = 0;
        goto LABEL_101;
      }
LABEL_26:
      v26 = v171;
      v28 = (__int64)v33;
      goto LABEL_27;
    }
    goto LABEL_13;
  }
  sub_16A51C0((__int64)&v168, v19 + 24);
  v21 = *(_WORD *)(v156 + 24) == 10 || *(_WORD *)(v156 + 24) == 32;
  if ( !v21 )
    goto LABEL_35;
  v22 = *(_QWORD *)(v156 + 88);
  v23 = v22 + 24;
  v18 = (*(_BYTE *)(v8 + 26) & 8) != 0;
  if ( v171 > 0x40 )
  {
LABEL_9:
    v132 = v21;
    sub_16A51C0((__int64)&v170, v23);
    v21 = v132;
    goto LABEL_10;
  }
LABEL_49:
  v36 = *(_DWORD *)(v22 + 32);
  if ( v36 > 0x40 )
    goto LABEL_9;
  v171 = *(_DWORD *)(v22 + 32);
  v170 = (const void *)(*(_QWORD *)(v22 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v36));
LABEL_10:
  v24 = *(_QWORD **)a1;
  v133 = (*(_BYTE *)(v156 + 26) & 8) != 0;
  if ( v21 )
  {
    v25 = (*(_BYTE *)(v156 + 26) & 8) != 0 || v18;
    if ( v25 )
    {
      v164 = v155 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( !sub_1D23600((__int64)v24, v8) )
        goto LABEL_15;
      goto LABEL_13;
    }
LABEL_53:
    v39 = *(_QWORD *)(v6 + 72);
    v176 = v39;
    if ( v39 )
    {
      v160 = v24;
      sub_1623A60((__int64)&v176, v39, 2);
      v24 = v160;
    }
    v177 = *(_DWORD *)(v6 + 64);
    v40 = sub_1D32920(
            v24,
            0x36u,
            (__int64)&v176,
            v166,
            (__int64)v167,
            v8,
            *(double *)v9.m128i_i64,
            *(double *)v10.m128i_i64,
            a5,
            v156);
    v41 = v176;
    v28 = (__int64)v40;
    if ( !v176 )
      goto LABEL_57;
LABEL_56:
    sub_161E7C0((__int64)&v176, v41);
LABEL_57:
    v26 = v171;
    goto LABEL_27;
  }
  v25 = 1;
  v164 = v155 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  if ( sub_1D23600((__int64)v24, v8) )
  {
LABEL_13:
    v153 = v12 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( !sub_1D23600(*(_QWORD *)a1, v156) )
    {
      v49 = *(_QWORD *)(v6 + 72);
      v50 = *(__int64 **)a1;
      v176 = v49;
      if ( v49 )
        sub_1623A60((__int64)&v176, v49, 2);
      v177 = *(_DWORD *)(v6 + 64);
      v51 = sub_1D332F0(
              v50,
              54,
              (__int64)&v176,
              v166,
              v167,
              0,
              *(double *)v9.m128i_i64,
              *(double *)v10.m128i_i64,
              a5,
              v156,
              v153,
              __PAIR128__(v155 | v164 & 0xFFFFFFFF00000000LL, v8));
LABEL_76:
      v41 = v176;
      v28 = (__int64)v51;
      if ( !v176 )
        goto LABEL_57;
      goto LABEL_56;
    }
LABEL_14:
    if ( v25 )
      goto LABEL_15;
    goto LABEL_36;
  }
LABEL_15:
  v26 = v171;
  if ( v171 <= 0x40 )
    v27 = v170 == 0;
  else
    v27 = v26 == (unsigned int)sub_16A57B0((__int64)&v170);
  if ( v27 )
  {
    v28 = v156;
    goto LABEL_27;
  }
  if ( v26 > 0x40 )
  {
    if ( (unsigned int)sub_16A57B0((__int64)&v170) != v26 - 1 )
      goto LABEL_60;
LABEL_70:
    v28 = v8;
    goto LABEL_27;
  }
  if ( v170 == (const void *)1 )
    goto LABEL_70;
LABEL_60:
  v33 = sub_1F77C50((__int64 **)a1, v6, *(double *)v9.m128i_i64, *(double *)v10.m128i_i64, a5);
  if ( v33 )
    goto LABEL_26;
  v42 = v171;
  if ( v171 <= 0x40 )
    v43 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v171) == (_QWORD)v170;
  else
    v43 = v42 == (unsigned int)sub_16A58F0((__int64)&v170);
  if ( v43 )
  {
    v44 = *(_QWORD *)(v6 + 72);
    v176 = v44;
    if ( v44 )
      sub_1623A60((__int64)&v176, v44, 2);
    v45 = *(__int64 **)a1;
    v177 = *(_DWORD *)(v6 + 64);
    v46 = sub_1D38BB0((__int64)v45, 0, (__int64)&v176, v166, v167, 0, v9, *(double *)v10.m128i_i64, a5, 0);
    v48 = sub_1D332F0(
            v45,
            53,
            (__int64)&v176,
            v166,
            v167,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v10.m128i_i64,
            a5,
            v46,
            v47,
            __PAIR128__(v155 | v164 & 0xFFFFFFFF00000000LL, v8));
    goto LABEL_67;
  }
  v149 = v156;
  v131 = v12;
  v52 = v12 | v153 & 0xFFFFFFFF00000000LL;
  v154 = v52;
  v53 = sub_1F70310(v156, v52, 1u);
  if ( v53 )
  {
    v128 = v53;
    v77 = sub_1D208B0(*(_QWORD *)a1, v156, v52);
    v54 = v128;
    v78 = v77;
    if ( !v77 )
      goto LABEL_81;
LABEL_101:
    if ( v15 )
    {
      if ( (unsigned __int8)(v15 - 14) <= 0x5Fu )
        goto LABEL_103;
    }
    else if ( sub_1F58D20((__int64)&v166) )
    {
LABEL_103:
      if ( *(int *)(a1 + 16) > 2 )
      {
        v54 = v78;
        goto LABEL_81;
      }
    }
    v79 = *(_QWORD *)(v6 + 72);
    v176 = v79;
    if ( v79 )
      sub_1623A60((__int64)&v176, v79, 2);
    v177 = *(_DWORD *)(v6 + 64);
    v80 = sub_1F70B90((__int64 **)a1, v149, v154, (__int64)&v176, v9, *(double *)v10.m128i_i64, a5);
    v81 = *(_QWORD *)(a1 + 8);
    v138 = (__int64)v80;
    v83 = v82;
    v84 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + v140);
    v158 = *(_BYTE *)(a1 + 25);
    v146 = *((_QWORD *)v84 + 1);
    v152 = *v84;
    v85 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
    v86 = sub_1F40B60(v81, v152, v146, v85, v158);
    *(_QWORD *)&v88 = sub_1D323C0(
                        *(__int64 **)a1,
                        v138,
                        v83,
                        (__int64)&v176,
                        v86,
                        v87,
                        *(double *)v9.m128i_i64,
                        *(double *)v10.m128i_i64,
                        *(double *)a5.m128i_i64);
    v51 = sub_1D332F0(
            *(__int64 **)a1,
            122,
            (__int64)&v176,
            v166,
            v167,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v10.m128i_i64,
            a5,
            v8,
            v155 | v164 & 0xFFFFFFFF00000000LL,
            v88);
    goto LABEL_76;
  }
  v54 = 1;
LABEL_81:
  if ( v133 == 1 || !v54 )
    goto LABEL_38;
  v55 = v171;
  v175 = v171;
  if ( v171 <= 0x40 )
  {
    v56 = (__int64)v170;
LABEL_85:
    v174 = ~v56 & (0xFFFFFFFFFFFFFFFFLL >> -v55);
    goto LABEL_86;
  }
  sub_16A4FD0((__int64)&v174, &v170);
  v55 = v175;
  if ( v175 <= 0x40 )
  {
    v56 = v174;
    goto LABEL_85;
  }
  sub_16A8F40(&v174);
LABEL_86:
  sub_16A7400((__int64)&v174);
  v57 = v175;
  v175 = 0;
  v177 = v57;
  v176 = v174;
  if ( v57 > 0x40 )
  {
    v129 = v174;
    v137 = (unsigned int)sub_16A5940((__int64)&v176) == 1;
    if ( v129 )
    {
      j_j___libc_free_0_0(v129);
      if ( v175 > 0x40 )
      {
        if ( v174 )
          j_j___libc_free_0_0(v174);
      }
    }
    if ( v137 )
    {
LABEL_89:
      v58 = v171;
      v175 = v171;
      if ( v171 > 0x40 )
      {
        sub_16A4FD0((__int64)&v174, &v170);
        v58 = v175;
        if ( v175 > 0x40 )
        {
          sub_16A8F40(&v174);
          goto LABEL_92;
        }
        v59 = v174;
      }
      else
      {
        v59 = (__int64)v170;
      }
      v174 = ~v59 & (0xFFFFFFFFFFFFFFFFLL >> -v58);
LABEL_92:
      sub_16A7400((__int64)&v174);
      v60 = v175;
      v61 = v174;
      v175 = 0;
      v177 = v60;
      v176 = v174;
      if ( v60 > 0x40 )
      {
        v136 = v60 - 1 - sub_16A57B0((__int64)&v176);
        if ( v61 )
        {
          j_j___libc_free_0_0(v61);
          if ( v175 > 0x40 )
          {
            if ( v174 )
              j_j___libc_free_0_0(v174);
          }
        }
      }
      else
      {
        v62 = 64;
        if ( v174 )
        {
          _BitScanReverse64(&v63, v174);
          v62 = v63 ^ 0x3F;
        }
        v136 = 63 - v62;
      }
      v64 = *(_QWORD *)(v6 + 72);
      v176 = v64;
      if ( v64 )
        sub_1623A60((__int64)&v176, v64, 2);
      v65 = *(_DWORD *)(v6 + 64);
      v66 = *(__int64 **)a1;
      v67 = *(_BYTE *)(a1 + 25);
      v68 = *(_QWORD *)(a1 + 8);
      v177 = v65;
      v69 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + v140);
      v157 = v67;
      v145 = *((_QWORD *)v69 + 1);
      v151 = *v69;
      v70 = sub_1E0A0C0(v66[4]);
      v71 = sub_1F40B60(v68, v151, v145, v70, v157);
      *(_QWORD *)&v73 = sub_1D38BB0(
                          (__int64)v66,
                          v136,
                          (__int64)&v176,
                          v71,
                          v72,
                          0,
                          v9,
                          *(double *)v10.m128i_i64,
                          a5,
                          0);
      *(_QWORD *)&v74 = sub_1D332F0(
                          v66,
                          122,
                          (__int64)&v176,
                          v166,
                          v167,
                          0,
                          *(double *)v9.m128i_i64,
                          *(double *)v10.m128i_i64,
                          a5,
                          v8,
                          v155 | v164 & 0xFFFFFFFF00000000LL,
                          v73);
      v161 = v74;
      v75 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v176, v166, v167, 0, v9, *(double *)v10.m128i_i64, a5, 0);
      v48 = sub_1D332F0(
              v66,
              53,
              (__int64)&v176,
              v166,
              v167,
              0,
              *(double *)v9.m128i_i64,
              *(double *)v10.m128i_i64,
              a5,
              v75,
              v76,
              v161);
LABEL_67:
      v41 = v176;
      v28 = (__int64)v48;
      if ( !v176 )
        goto LABEL_57;
      goto LABEL_56;
    }
  }
  else if ( v174 && (v174 & (v174 - 1)) == 0 )
  {
    goto LABEL_89;
  }
LABEL_38:
  if ( *(_WORD *)(v8 + 24) == 122 )
  {
    v154 = v131 | v154 & 0xFFFFFFFF00000000LL;
    if ( (unsigned __int8)sub_1F70310(v156, v154, 1u)
      && (unsigned __int8)sub_1F70310(
                            *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v8 + 32) + 48LL),
                            1u) )
    {
      v101 = *(_QWORD *)(v8 + 32);
      v102 = *(__int64 **)a1;
      v176 = *(_QWORD *)(v6 + 72);
      if ( v176 )
      {
        v141 = v101;
        sub_1F6CA20(&v176);
        v101 = v141;
      }
      v177 = *(_DWORD *)(v6 + 64);
      *(_QWORD *)&v103 = sub_1D332F0(
                           v102,
                           122,
                           (__int64)&v176,
                           v166,
                           v167,
                           0,
                           *(double *)v9.m128i_i64,
                           *(double *)v10.m128i_i64,
                           a5,
                           v156,
                           v154,
                           *(_OWORD *)(v101 + 40));
      v142 = v103;
      sub_17CD270(&v176);
      if ( (unsigned __int8)sub_1F70310(v142, DWORD2(v142), 0) )
      {
        v104 = *(_QWORD *)(v8 + 32);
        v105 = *(__int64 **)a1;
        v176 = *(_QWORD *)(v6 + 72);
        if ( v176 )
          sub_1F6CA20(&v176);
        v177 = *(_DWORD *)(v6 + 64);
        v28 = (__int64)sub_1D332F0(
                         v105,
                         54,
                         (__int64)&v176,
                         v166,
                         v167,
                         0,
                         *(double *)v9.m128i_i64,
                         *(double *)v10.m128i_i64,
                         a5,
                         *(_QWORD *)v104,
                         *(_QWORD *)(v104 + 8),
                         v142);
        sub_17CD270(&v176);
        v26 = v171;
        goto LABEL_27;
      }
    }
    if ( *(_WORD *)(v8 + 24) == 122 )
    {
      if ( (unsigned __int8)sub_1F70310(
                              *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL),
                              *(_QWORD *)(*(_QWORD *)(v8 + 32) + 48LL),
                              0) )
      {
        v89 = *(_QWORD *)(v8 + 48);
        if ( v89 )
        {
          if ( !*(_QWORD *)(v89 + 32) )
          {
            v90 = v156;
            v156 = v8;
            v91 = v131;
LABEL_131:
            v92 = *(_QWORD *)(v6 + 72);
            v93 = *(__int64 **)a1;
            v94 = *(_QWORD *)(v156 + 32);
            v176 = v92;
            if ( v92 )
            {
              v162 = v90;
              v165 = v91;
              sub_1623A60((__int64)&v176, v92, 2);
              v91 = v165;
              v90 = v162;
            }
            *((_QWORD *)&v127 + 1) = v91;
            *(_QWORD *)&v127 = v90;
            v177 = *(_DWORD *)(v6 + 64);
            *(_QWORD *)&v95 = sub_1D332F0(
                                v93,
                                54,
                                (__int64)&v176,
                                v166,
                                v167,
                                0,
                                *(double *)v9.m128i_i64,
                                *(double *)v10.m128i_i64,
                                a5,
                                *(_QWORD *)v94,
                                *(_QWORD *)(v94 + 8),
                                v127);
            v96 = v95;
            if ( v176 )
            {
              v163 = v95;
              sub_161E7C0((__int64)&v176, v176);
              v96 = v163;
            }
            v97 = *(_QWORD *)(v6 + 72);
            v98 = *(__int64 **)a1;
            v99 = *(_QWORD *)(v156 + 32);
            v176 = v97;
            if ( v97 )
              sub_1623A60((__int64)&v176, v97, 2);
            v177 = *(_DWORD *)(v6 + 64);
            v51 = sub_1D332F0(
                    v98,
                    122,
                    (__int64)&v176,
                    v166,
                    v167,
                    0,
                    *(double *)v9.m128i_i64,
                    *(double *)v10.m128i_i64,
                    a5,
                    v96,
                    *((unsigned __int64 *)&v96 + 1),
                    *(_OWORD *)(v99 + 40));
            goto LABEL_76;
          }
        }
      }
    }
  }
  if ( *(_WORD *)(v156 + 24) == 122 )
  {
    if ( (unsigned __int8)sub_1F70310(
                            *(_QWORD *)(*(_QWORD *)(v156 + 32) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v156 + 32) + 48LL),
                            0) )
    {
      v100 = *(_QWORD *)(v156 + 48);
      if ( v100 )
      {
        if ( !*(_QWORD *)(v100 + 32) )
        {
          v90 = v8;
          v91 = v155;
          goto LABEL_131;
        }
      }
    }
  }
  *(_QWORD *)&v150 = v156;
  *((_QWORD *)&v150 + 1) = v131 | v154 & 0xFFFFFFFF00000000LL;
  if ( !sub_1D23600(*(_QWORD *)a1, v156)
    || *(_WORD *)(v8 + 24) != 52
    || !sub_1D23600(*(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL)) )
  {
    goto LABEL_42;
  }
  v106 = *(_QWORD *)(v8 + 48);
  if ( v106 && !*(_QWORD *)(v106 + 32) )
    goto LABEL_162;
  v107 = *(_QWORD *)(v6 + 48);
  if ( !v107 )
    goto LABEL_167;
  v143 = v6;
  do
  {
    v108 = *(_QWORD *)(v107 + 16);
    if ( *(_WORD *)(v108 + 24) == 52 )
    {
      if ( sub_1D23600(*(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(v108 + 32) + 40LL)) )
        goto LABEL_161;
      v109 = *(_QWORD *)(v108 + 48);
      if ( v109 )
      {
        while ( 1 )
        {
          v110 = *(_QWORD *)(v109 + 16);
          if ( *(_WORD *)(v110 + 24) == 52 )
          {
            if ( sub_1D23600(*(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(v110 + 32) + 40LL)) )
              break;
          }
          v109 = *(_QWORD *)(v109 + 32);
          if ( !v109 )
            goto LABEL_153;
        }
LABEL_161:
        v6 = v143;
LABEL_162:
        v111 = *(_QWORD **)(v8 + 32);
        goto LABEL_163;
      }
    }
LABEL_153:
    v107 = *(_QWORD *)(v107 + 32);
  }
  while ( v107 );
  v6 = v143;
LABEL_167:
  if ( !*(_QWORD *)(v156 + 48) )
  {
LABEL_42:
    v176 = *(_QWORD *)(v6 + 72);
    if ( v176 )
      sub_1F6CA20(&v176);
    v177 = *(_DWORD *)(v6 + 64);
    v28 = (__int64)sub_1F82ED0(
                     (__int64 *)a1,
                     0x36u,
                     (__int64)&v176,
                     v8,
                     v155 | v164 & 0xFFFFFFFF00000000LL,
                     *(double *)v9.m128i_i64,
                     *(double *)v10.m128i_i64,
                     a5,
                     v35,
                     v150);
    if ( v176 )
      sub_161E7C0((__int64)&v176, v176);
    if ( !v28 )
      v28 = 0;
    goto LABEL_57;
  }
  v118 = v6;
  v119 = *(_QWORD *)(v156 + 48);
  v120 = (__int64 *)a1;
  v121 = v118;
  while ( 1 )
  {
    v122 = *(_QWORD *)(v119 + 16);
    if ( v121 != v122 && *(_WORD *)(v122 + 24) == 54 )
      break;
LABEL_169:
    v119 = *(_QWORD *)(v119 + 32);
    if ( !v119 )
    {
      a1 = (__int64)v120;
      v6 = v121;
      goto LABEL_42;
    }
  }
  v111 = *(_QWORD **)(v8 + 32);
  v123 = *(__int64 **)(v122 + 32);
  v124 = *v111;
  v125 = *v123;
  if ( v156 == *v123 && *((_DWORD *)v123 + 2) == v12 )
    v125 = v123[5];
  if ( v124 != v125 )
  {
    if ( *(_WORD *)(v125 + 24) == 52 )
    {
      v139 = v121;
      v126 = sub_1D23600(*v120, *(_QWORD *)(*(_QWORD *)(v125 + 32) + 40LL));
      v121 = v139;
      if ( v126 )
      {
        if ( v124 == **(_QWORD **)(v125 + 32) )
        {
          a1 = (__int64)v120;
          v111 = *(_QWORD **)(v8 + 32);
          v6 = v139;
          goto LABEL_163;
        }
      }
    }
    goto LABEL_169;
  }
  a1 = (__int64)v120;
  v6 = v121;
LABEL_163:
  v112 = *(__int64 **)a1;
  v147 = v111;
  v159 = *(__int64 **)a1;
  sub_1F80610((__int64)&v176, v150);
  *(_QWORD *)&v113 = sub_1D332F0(
                       v112,
                       54,
                       (__int64)&v176,
                       v166,
                       v167,
                       0,
                       *(double *)v9.m128i_i64,
                       *(double *)v10.m128i_i64,
                       a5,
                       v147[5],
                       v147[6],
                       v150);
  v114 = *(__int64 **)a1;
  v148 = v113;
  v144 = *(_QWORD *)(v8 + 32);
  sub_1F80610((__int64)&v174, v8);
  v115 = sub_1D332F0(
           v114,
           54,
           (__int64)&v174,
           v166,
           v167,
           0,
           *(double *)v9.m128i_i64,
           *(double *)v10.m128i_i64,
           a5,
           *(_QWORD *)v144,
           *(_QWORD *)(v144 + 8),
           v150);
  v117 = v116;
  v172 = *(_QWORD *)(v6 + 72);
  if ( v172 )
    sub_1F6CA20(&v172);
  v173 = *(_DWORD *)(v6 + 64);
  v28 = (__int64)sub_1D332F0(
                   v159,
                   52,
                   (__int64)&v172,
                   v166,
                   v167,
                   0,
                   *(double *)v9.m128i_i64,
                   *(double *)v10.m128i_i64,
                   a5,
                   (__int64)v115,
                   v117,
                   v148);
  sub_17CD270(&v172);
  sub_17CD270(&v174);
  sub_17CD270(&v176);
  v26 = v171;
LABEL_27:
  if ( v26 > 0x40 && v170 )
    j_j___libc_free_0_0(v170);
  if ( v169 > 0x40 && v168 )
    j_j___libc_free_0_0(v168);
  return v28;
}
