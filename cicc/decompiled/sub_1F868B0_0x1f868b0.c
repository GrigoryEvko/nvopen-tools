// Function: sub_1F868B0
// Address: 0x1f868b0
//
__int64 __fastcall sub_1F868B0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  int v8; // r13d
  __int64 v9; // rax
  char v10; // r14
  const void **v11; // rax
  int v12; // ecx
  unsigned __int16 v13; // ax
  unsigned __int8 *v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r11
  unsigned int v19; // r10d
  int v20; // edx
  __int64 v21; // r9
  unsigned int *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  const void *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rax
  __m128i v30; // xmm2
  __int64 v31; // r14
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 *v35; // r15
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 *v38; // r15
  const void *v39; // r12
  __int64 v40; // rdx
  __int64 v41; // r13
  _QWORD *v42; // rax
  __int64 result; // rax
  bool v44; // al
  unsigned int *v45; // r14
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  unsigned int *v49; // r9
  __int64 v50; // rsi
  __int64 *v51; // r15
  const void **v52; // r8
  __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 *v55; // r14
  __int64 v56; // rdx
  __int64 v57; // r15
  __int64 *v58; // r13
  __int64 v59; // rsi
  __int64 v60; // rdi
  __int64 (*v61)(); // rax
  const __m128i *v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rax
  char v65; // r14
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rax
  char v69; // r8
  const void **v70; // rax
  bool v71; // al
  bool v72; // al
  __int64 v73; // rsi
  __int64 *v74; // r15
  const void **v75; // r8
  __int64 v76; // rcx
  __int64 v77; // rsi
  __int64 *v78; // r14
  unsigned __int64 v79; // rdx
  unsigned __int64 v80; // r15
  __int64 *v81; // r13
  __int64 v82; // rcx
  __int64 v83; // rdi
  bool (__fastcall *v84)(__int64, __int64, unsigned __int8); // rax
  __int128 v85; // rax
  __int64 v86; // r14
  __int64 v87; // r12
  char v88; // al
  __int64 (*v89)(); // rax
  char v90; // al
  char v91; // al
  __int64 *v92; // r14
  unsigned int v93; // edx
  __int64 *v94; // r14
  unsigned int v95; // edx
  __int64 *v96; // r15
  __int64 v97; // rsi
  __int64 *v98; // r15
  __int64 v99; // rdx
  const void *v100; // r12
  __int64 v101; // rdx
  __int64 v102; // r13
  __int128 v103; // [rsp-10h] [rbp-F0h]
  char v104; // [rsp+Fh] [rbp-D1h]
  char v105; // [rsp+Fh] [rbp-D1h]
  __int64 v106; // [rsp+10h] [rbp-D0h]
  __int64 v107; // [rsp+10h] [rbp-D0h]
  __int64 v108; // [rsp+10h] [rbp-D0h]
  __int64 v109; // [rsp+10h] [rbp-D0h]
  __int64 v110; // [rsp+20h] [rbp-C0h]
  unsigned int *v111; // [rsp+20h] [rbp-C0h]
  const void **v112; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v113; // [rsp+20h] [rbp-C0h]
  unsigned __int8 v114; // [rsp+20h] [rbp-C0h]
  __int64 v115; // [rsp+20h] [rbp-C0h]
  __int64 v116; // [rsp+20h] [rbp-C0h]
  __int64 *v117; // [rsp+30h] [rbp-B0h]
  _QWORD *v118; // [rsp+30h] [rbp-B0h]
  unsigned __int8 v119; // [rsp+30h] [rbp-B0h]
  const void **v120; // [rsp+30h] [rbp-B0h]
  __int64 v121; // [rsp+30h] [rbp-B0h]
  const void **v122; // [rsp+30h] [rbp-B0h]
  __int64 v123; // [rsp+30h] [rbp-B0h]
  __int64 v124; // [rsp+30h] [rbp-B0h]
  unsigned int v125; // [rsp+30h] [rbp-B0h]
  unsigned int v126; // [rsp+30h] [rbp-B0h]
  __int64 *v127; // [rsp+30h] [rbp-B0h]
  __int64 v128; // [rsp+38h] [rbp-A8h]
  const void *s1; // [rsp+40h] [rbp-A0h]
  __int64 s1c; // [rsp+40h] [rbp-A0h]
  const void **s1a; // [rsp+40h] [rbp-A0h]
  __int64 s1b; // [rsp+40h] [rbp-A0h]
  __int64 s1d; // [rsp+40h] [rbp-A0h]
  unsigned __int64 s1_8; // [rsp+48h] [rbp-98h]
  unsigned __int64 s1_8a; // [rsp+48h] [rbp-98h]
  unsigned int v136; // [rsp+70h] [rbp-70h] BYREF
  const void **v137; // [rsp+78h] [rbp-68h]
  unsigned int v138; // [rsp+80h] [rbp-60h] BYREF
  const void **v139; // [rsp+88h] [rbp-58h]
  char v140[8]; // [rsp+90h] [rbp-50h] BYREF
  const void **v141; // [rsp+98h] [rbp-48h]
  __int64 v142; // [rsp+A0h] [rbp-40h] BYREF
  int v143; // [rsp+A8h] [rbp-38h]

  v5 = *(unsigned int **)(a2 + 32);
  v6 = *(_QWORD *)v5;
  v7 = *((_QWORD *)v5 + 5);
  v8 = *(_DWORD *)(*(_QWORD *)v5 + 56LL);
  v9 = *(_QWORD *)(*(_QWORD *)v5 + 40LL) + 16LL * v5[2];
  v10 = *(_BYTE *)v9;
  v11 = *(const void ***)(v9 + 8);
  LOBYTE(v136) = v10;
  v137 = v11;
  if ( !v8 )
    return 0;
  v12 = *(unsigned __int16 *)(v6 + 24);
  v13 = *(_WORD *)(v6 + 24);
  if ( (_WORD)v12 == 145 )
    return 0;
  v16 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v6 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(v6 + 32) + 8LL));
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  v19 = (unsigned __int8)v17;
  if ( (unsigned int)(v12 - 142) <= 1 || v12 == 127 )
    goto LABEL_5;
  if ( v12 == 144 )
  {
    if ( !*(_BYTE *)(a1 + 25) )
      goto LABEL_5;
    v83 = *(_QWORD *)(a1 + 8);
    v84 = *(bool (__fastcall **)(__int64, __int64, unsigned __int8))(*(_QWORD *)v83 + 1136LL);
    if ( v84 == sub_1F6BB70 )
    {
      if ( !(_BYTE)v17 || !*(_QWORD *)(v83 + 8LL * (unsigned __int8)v17 + 120) )
        goto LABEL_13;
      goto LABEL_5;
    }
    v114 = *v16;
    v126 = (unsigned __int8)v17;
    s1d = *((_QWORD *)v16 + 1);
    v90 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v84)(
            v83,
            *(unsigned __int16 *)(a2 + 24),
            (unsigned __int8)v17,
            v18);
    v18 = s1d;
    v19 = v126;
    v17 = v114;
    if ( v90 )
      goto LABEL_5;
    v13 = *(_WORD *)(v6 + 24);
  }
  v20 = v13;
  if ( v13 != 145 )
    goto LABEL_8;
  v60 = *(_QWORD *)(a1 + 8);
  v61 = *(__int64 (**)())(*(_QWORD *)v60 + 824LL);
  if ( v61 == sub_1D12E00 )
  {
    if ( !(_BYTE)v17 )
      goto LABEL_13;
  }
  else
  {
    v113 = v17;
    v125 = v19;
    s1b = v18;
    v88 = ((__int64 (__fastcall *)(__int64, _QWORD, const void **, _QWORD, __int64))v61)(v60, v136, v137, v19, v18);
    v60 = *(_QWORD *)(a1 + 8);
    v18 = s1b;
    v17 = v113;
    if ( v88 )
    {
      v89 = *(__int64 (**)())(*(_QWORD *)v60 + 800LL);
      if ( v89 != sub_1D12DF0 )
      {
        v91 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, const void **))v89)(v60, v125, s1b, v136, v137);
        v18 = s1b;
        v17 = v113;
        if ( v91 )
          goto LABEL_7;
        v60 = *(_QWORD *)(a1 + 8);
      }
    }
    if ( !(_BYTE)v17 )
      goto LABEL_7;
  }
  if ( !*(_QWORD *)(v60 + 8LL * (unsigned __int8)v17 + 120) )
    goto LABEL_7;
LABEL_5:
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) <= 0x5Fu )
    {
LABEL_7:
      v20 = *(unsigned __int16 *)(v6 + 24);
      v13 = *(_WORD *)(v6 + 24);
LABEL_8:
      if ( (unsigned __int16)(v13 - 122) <= 2u || v20 == 118 )
      {
        v21 = *(_QWORD *)(v7 + 32);
        v22 = *(unsigned int **)(v6 + 32);
        if ( *((_QWORD *)v22 + 5) == *(_QWORD *)(v21 + 40) && v22[12] == *(_DWORD *)(v21 + 48) )
        {
          v73 = *(_QWORD *)(v6 + 72);
          v74 = *(__int64 **)a1;
          v75 = *(const void ***)(*(_QWORD *)(*(_QWORD *)v22 + 40LL) + 16LL * v22[2] + 8);
          v76 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v22 + 40LL) + 16LL * v22[2]);
          v142 = v73;
          if ( v73 )
          {
            v108 = v76;
            v112 = v75;
            v123 = v21;
            sub_1623A60((__int64)&v142, v73, 2);
            v76 = v108;
            v75 = v112;
            v21 = v123;
          }
          v77 = *(unsigned __int16 *)(a2 + 24);
          v143 = *(_DWORD *)(v6 + 64);
          v78 = sub_1D332F0(
                  v74,
                  v77,
                  (__int64)&v142,
                  v76,
                  v75,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  a5,
                  *(_QWORD *)v22,
                  *((_QWORD *)v22 + 1),
                  *(_OWORD *)v21);
          v80 = v79;
          if ( v142 )
            sub_161E7C0((__int64)&v142, v142);
          sub_1F81BC0(a1, (__int64)v78);
          v81 = *(__int64 **)a1;
          v82 = *(_QWORD *)(v6 + 32);
          v142 = *(_QWORD *)(a2 + 72);
          if ( v142 )
          {
            v124 = v82;
            sub_1623A60((__int64)&v142, v142, 2);
            v82 = v124;
          }
          v143 = *(_DWORD *)(a2 + 64);
          result = (__int64)sub_1D332F0(
                              v81,
                              *(unsigned __int16 *)(v6 + 24),
                              (__int64)&v142,
                              v136,
                              v137,
                              0,
                              *(double *)a3.m128i_i64,
                              *(double *)a4.m128i_i64,
                              a5,
                              (__int64)v78,
                              v80,
                              *(_OWORD *)(v82 + 40));
          v59 = v142;
          if ( !v142 )
            return result;
LABEL_53:
          v121 = result;
          sub_161E7C0((__int64)&v142, v59);
          return v121;
        }
      }
      if ( (v20 == 158 || v20 == 111) && *(int *)(a1 + 16) <= 1 )
      {
        v62 = *(const __m128i **)(v7 + 32);
        a3 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 32));
        v63 = v62->m128i_i64[0];
        a4 = _mm_loadu_si128(v62);
        v64 = *(_QWORD *)(**(_QWORD **)(v6 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v6 + 32) + 8LL);
        v65 = *(_BYTE *)v64;
        v122 = *(const void ***)(v64 + 8);
        v139 = v122;
        v66 = v62->m128i_u32[2];
        v67 = *(_QWORD *)(a2 + 72);
        LOBYTE(v138) = v65;
        v68 = *(_QWORD *)(v63 + 40) + 16 * v66;
        v69 = *(_BYTE *)v68;
        v70 = *(const void ***)(v68 + 8);
        v142 = v67;
        s1a = v70;
        v140[0] = v69;
        v141 = v70;
        if ( v67 )
        {
          v104 = v69;
          sub_1623A60((__int64)&v142, v67, 2);
          v69 = v104;
        }
        v143 = *(_DWORD *)(a2 + 64);
        if ( v65 )
        {
          v71 = (unsigned __int8)(v65 - 14) <= 0x47u || (unsigned __int8)(v65 - 2) <= 5u;
        }
        else
        {
          v105 = v69;
          v71 = sub_1F58CF0((__int64)&v138);
          v69 = v105;
        }
        if ( v71 )
        {
          if ( v69 )
          {
            v72 = (unsigned __int8)(v69 - 14) <= 0x47u || (unsigned __int8)(v69 - 2) <= 5u;
          }
          else
          {
            v72 = sub_1F58CF0((__int64)v140);
            v69 = 0;
          }
          if ( v72 && v69 == v65 && (s1a == v122 || v65) )
          {
            *(_QWORD *)&v85 = sub_1D332F0(
                                *(__int64 **)a1,
                                *(unsigned __int16 *)(a2 + 24),
                                (__int64)&v142,
                                v138,
                                v139,
                                0,
                                *(double *)a3.m128i_i64,
                                *(double *)a4.m128i_i64,
                                a5,
                                a3.m128i_i64[0],
                                a3.m128i_u64[1],
                                *(_OWORD *)&a4);
            v86 = v85;
            v87 = sub_1D309E0(
                    *(__int64 **)a1,
                    *(unsigned __int16 *)(v6 + 24),
                    (__int64)&v142,
                    v136,
                    v137,
                    0,
                    *(double *)a3.m128i_i64,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    v85);
            sub_1F81BC0(a1, v86);
            v59 = v142;
            result = v87;
            if ( !v142 )
              return result;
            goto LABEL_53;
          }
        }
        if ( v142 )
          sub_161E7C0((__int64)&v142, v142);
      }
LABEL_13:
      if ( *(_WORD *)(v6 + 24) != 110 )
        return 0;
      if ( *(int *)(a1 + 16) > 2 )
        return 0;
      v23 = *(_QWORD *)(v6 + 48);
      if ( !v23 )
        return 0;
      if ( *(_QWORD *)(v23 + 32) )
        return 0;
      v24 = *(_QWORD *)(v7 + 48);
      if ( !v24 )
        return 0;
      if ( *(_QWORD *)(v24 + 32) )
        return 0;
      s1 = (const void *)sub_1F80640(v6);
      v26 = v25;
      v27 = (const void *)sub_1F80640(v7);
      if ( v26 != v28 || 4 * v26 && memcmp(s1, v27, 4 * v26) )
        return 0;
      v29 = *(_QWORD *)(v6 + 32);
      v30 = _mm_loadu_si128((const __m128i *)(v29 + 40));
      v31 = *(_QWORD *)(v29 + 40);
      s1_8 = v30.m128i_u64[1];
      if ( *(_WORD *)(a2 + 24) == 120 && *(_WORD *)(v31 + 24) != 48 )
      {
        if ( *(_BYTE *)(a1 + 25) )
          goto LABEL_24;
        v92 = *(__int64 **)a1;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
          sub_1F6CA20(&v142);
        v143 = *(_DWORD *)(a2 + 64);
        v31 = sub_1D38BB0((__int64)v92, 0, (__int64)&v142, v136, v137, 0, a3, *(double *)a4.m128i_i64, v30, 0);
        s1_8 = v93 | v30.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        sub_17CD270(&v142);
        v29 = *(_QWORD *)(v6 + 32);
      }
      v32 = *(_QWORD *)(v7 + 32);
      if ( *(_QWORD *)(v29 + 40) == *(_QWORD *)(v32 + 40) && *(_DWORD *)(v29 + 48) == *(_DWORD *)(v32 + 48) && v31 )
      {
        v96 = *(__int64 **)a1;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
        {
          v109 = v29;
          v115 = v32;
          sub_1F6CA20(&v142);
          v29 = v109;
          v32 = v115;
        }
        v97 = *(unsigned __int16 *)(a2 + 24);
        v143 = *(_DWORD *)(a2 + 64);
        v98 = sub_1D332F0(
                v96,
                v97,
                (__int64)&v142,
                v136,
                v137,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                v30,
                *(_QWORD *)v29,
                *(_QWORD *)(v29 + 8),
                *(_OWORD *)v32);
        v116 = v99;
        sub_17CD270(&v142);
        sub_1F81BC0(a1, (__int64)v98);
        v127 = *(__int64 **)a1;
        v100 = (const void *)sub_1F80640(v6);
        v102 = v101;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
          sub_1F6CA20(&v142);
        v143 = *(_DWORD *)(a2 + 64);
        v42 = sub_1D41320(
                (__int64)v127,
                v136,
                v137,
                (__int64)&v142,
                (__int64)v98,
                v116,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                v30,
                v31,
                s1_8,
                v100,
                v102);
LABEL_33:
        v118 = v42;
        sub_17CD270(&v142);
        return (__int64)v118;
      }
LABEL_24:
      v33 = *(_QWORD *)v29;
      s1_8a = *(unsigned int *)(v29 + 8) | s1_8 & 0xFFFFFFFF00000000LL;
      if ( *(_WORD *)(a2 + 24) == 120 && *(_WORD *)(v33 + 24) != 48 )
      {
        if ( *(_BYTE *)(a1 + 25) )
          return 0;
        v94 = *(__int64 **)a1;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
          sub_1F6CA20(&v142);
        v143 = *(_DWORD *)(a2 + 64);
        v33 = sub_1D38BB0((__int64)v94, 0, (__int64)&v142, v136, v137, 0, a3, *(double *)a4.m128i_i64, v30, 0);
        s1_8a = v95 | s1_8a & 0xFFFFFFFF00000000LL;
        sub_17CD270(&v142);
        v29 = *(_QWORD *)(v6 + 32);
      }
      v34 = *(_QWORD *)(v7 + 32);
      if ( *(_QWORD *)v29 == *(_QWORD *)v34 && *(_DWORD *)(v29 + 8) == *(_DWORD *)(v34 + 8) && v33 )
      {
        v35 = *(__int64 **)a1;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
        {
          v106 = v29;
          v110 = v34;
          sub_1F6CA20(&v142);
          v29 = v106;
          v34 = v110;
        }
        v36 = *(unsigned __int16 *)(a2 + 24);
        v143 = *(_DWORD *)(a2 + 64);
        v117 = sub_1D332F0(
                 v35,
                 v36,
                 (__int64)&v142,
                 v136,
                 v137,
                 0,
                 *(double *)a3.m128i_i64,
                 *(double *)a4.m128i_i64,
                 v30,
                 *(_QWORD *)(v29 + 40),
                 *(_QWORD *)(v29 + 48),
                 *(_OWORD *)(v34 + 40));
        v128 = v37;
        sub_17CD270(&v142);
        sub_1F81BC0(a1, (__int64)v117);
        v38 = *(__int64 **)a1;
        v39 = (const void *)sub_1F80640(v6);
        v41 = v40;
        v142 = *(_QWORD *)(a2 + 72);
        if ( v142 )
          sub_1F6CA20(&v142);
        v143 = *(_DWORD *)(a2 + 64);
        v42 = sub_1D41320(
                (__int64)v38,
                v136,
                v137,
                (__int64)&v142,
                v33,
                s1_8a,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                v30,
                (__int64)v117,
                v128,
                v39,
                v41);
        goto LABEL_33;
      }
      return 0;
    }
  }
  else
  {
    v119 = v17;
    s1c = v18;
    v44 = sub_1F58D20((__int64)&v136);
    v18 = s1c;
    v17 = v119;
    if ( v44 )
      goto LABEL_7;
  }
  v45 = *(unsigned int **)(v7 + 32);
  v46 = *(_QWORD *)(*(_QWORD *)v45 + 40LL) + 16LL * v45[2];
  if ( *(_BYTE *)v46 != (_BYTE)v17 || *(_QWORD *)(v46 + 8) != v18 && !(_BYTE)v17 )
    goto LABEL_7;
  if ( *(_BYTE *)(a1 + 24) )
  {
    v47 = *(_QWORD *)(a1 + 8);
    if ( (_BYTE)v17 != 1 && (!(_BYTE)v17 || !*(_QWORD *)(v47 + 8LL * (unsigned __int8)v17 + 120)) )
      goto LABEL_7;
    v48 = *(unsigned __int16 *)(a2 + 24);
    if ( (unsigned int)v48 > 0x102 || *(_BYTE *)(v48 + 259 * v17 + v47 + 2422) )
      goto LABEL_7;
  }
  v49 = *(unsigned int **)(v6 + 32);
  v50 = *(_QWORD *)(v6 + 72);
  v51 = *(__int64 **)a1;
  v52 = *(const void ***)(*(_QWORD *)(*(_QWORD *)v49 + 40LL) + 16LL * v49[2] + 8);
  v53 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v49 + 40LL) + 16LL * v49[2]);
  v142 = v50;
  if ( v50 )
  {
    v107 = v53;
    v111 = v49;
    v120 = v52;
    sub_1623A60((__int64)&v142, v50, 2);
    v53 = v107;
    v49 = v111;
    v52 = v120;
  }
  v54 = *(unsigned __int16 *)(a2 + 24);
  v143 = *(_DWORD *)(v6 + 64);
  v55 = sub_1D332F0(
          v51,
          v54,
          (__int64)&v142,
          v53,
          v52,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          *(_QWORD *)v49,
          *((_QWORD *)v49 + 1),
          *(_OWORD *)v45);
  v57 = v56;
  if ( v142 )
    sub_161E7C0((__int64)&v142, v142);
  sub_1F81BC0(a1, (__int64)v55);
  v58 = *(__int64 **)a1;
  v142 = *(_QWORD *)(a2 + 72);
  if ( v142 )
    sub_1623A60((__int64)&v142, v142, 2);
  v143 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v103 + 1) = v57;
  *(_QWORD *)&v103 = v55;
  result = sub_1D309E0(
             v58,
             *(unsigned __int16 *)(v6 + 24),
             (__int64)&v142,
             v136,
             v137,
             0,
             *(double *)a3.m128i_i64,
             *(double *)a4.m128i_i64,
             *(double *)a5.m128i_i64,
             v103);
  v59 = v142;
  if ( v142 )
    goto LABEL_53;
  return result;
}
