// Function: sub_1FA3B00
// Address: 0x1fa3b00
//
__int64 __fastcall sub_1FA3B00(_QWORD **a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v7; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // r15
  char *v11; // rax
  __int64 v12; // rsi
  char v13; // dl
  const void **v14; // rax
  char v15; // cl
  __int64 v16; // rsi
  char v17; // r8
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r12
  __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i v25; // rax
  __int64 v26; // r14
  const __m128i *v27; // roff
  __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // r9
  __int64 v37; // r8
  unsigned int v38; // edx
  __int64 *v39; // r10
  __int64 v40; // r11
  unsigned int v41; // ecx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // edx
  __int128 v45; // rax
  unsigned __int16 v46; // ax
  _QWORD *v47; // rdx
  __int64 *v48; // rax
  __int64 *v49; // r10
  __int64 v50; // rsi
  __int32 v51; // eax
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned __int8 v55; // r8
  __int64 v56; // r9
  unsigned __int64 v57; // r10
  unsigned __int8 v58; // al
  __int64 v59; // rax
  __int64 v60; // r10
  __int64 *v61; // r15
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __m128 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  unsigned int v72; // eax
  __int128 v73; // rax
  __m128i v74; // rax
  bool v75; // al
  __m128i v76; // xmm1
  __int64 v77; // r10
  __int64 *v78; // r12
  const void **v79; // r8
  __int64 v80; // rcx
  __int32 v81; // eax
  __int64 v82; // rdx
  __int64 *v83; // rax
  __m128i v84; // xmm3
  __int32 v85; // ecx
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rax
  _DWORD *v89; // rsi
  bool v90; // dl
  int v91; // eax
  __int64 v92; // r9
  __int64 v93; // rdx
  __int128 v94; // rax
  unsigned int v95; // eax
  const void **v96; // rdx
  const void **v97; // r10
  unsigned int v98; // r11d
  __int64 *v99; // rax
  __int16 *v100; // rdx
  const void ***v101; // rax
  __m128i v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // r9
  unsigned __int64 v111; // rax
  const void **v112; // rdx
  __int64 *v113; // rax
  __int64 v114; // rdx
  bool v115; // al
  __int64 v116; // [rsp+0h] [rbp-140h]
  __int128 v117; // [rsp+0h] [rbp-140h]
  __int64 v118; // [rsp+8h] [rbp-138h]
  int v119; // [rsp+10h] [rbp-130h]
  int v120; // [rsp+10h] [rbp-130h]
  __int64 v121; // [rsp+18h] [rbp-128h]
  __int64 v122; // [rsp+18h] [rbp-128h]
  __int64 v123; // [rsp+18h] [rbp-128h]
  __int64 v124; // [rsp+18h] [rbp-128h]
  int v125; // [rsp+20h] [rbp-120h]
  int v126; // [rsp+20h] [rbp-120h]
  __int64 v127; // [rsp+28h] [rbp-118h]
  bool v128; // [rsp+28h] [rbp-118h]
  __int64 v129; // [rsp+28h] [rbp-118h]
  unsigned int v130; // [rsp+30h] [rbp-110h]
  __int64 v131; // [rsp+30h] [rbp-110h]
  __int64 v132; // [rsp+38h] [rbp-108h]
  unsigned int v133; // [rsp+40h] [rbp-100h]
  char v134; // [rsp+40h] [rbp-100h]
  __int128 v135; // [rsp+40h] [rbp-100h]
  int v136; // [rsp+40h] [rbp-100h]
  bool v137; // [rsp+40h] [rbp-100h]
  __m128i v138; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v139; // [rsp+60h] [rbp-E0h]
  __m128i v140; // [rsp+70h] [rbp-D0h]
  __m128 v141; // [rsp+80h] [rbp-C0h]
  __int128 v142; // [rsp+90h] [rbp-B0h]
  unsigned int v143; // [rsp+A0h] [rbp-A0h] BYREF
  const void **v144; // [rsp+A8h] [rbp-98h]
  __int64 v145; // [rsp+B0h] [rbp-90h] BYREF
  int v146; // [rsp+B8h] [rbp-88h]
  __int64 v147[2]; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v148; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v149; // [rsp+E0h] [rbp-60h] BYREF
  _BYTE v150[80]; // [rsp+F0h] [rbp-50h] BYREF

  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  v9 = *v7;
  v10 = v7[1];
  LODWORD(v142) = *((_DWORD *)v7 + 2);
  v11 = *(char **)(a2 + 40);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = *v11;
  v14 = (const void **)*((_QWORD *)v11 + 1);
  v145 = v12;
  LOBYTE(v143) = v13;
  v144 = v14;
  if ( v12 )
    sub_1623A60((__int64)&v145, v12, 2);
  v15 = *((_BYTE *)a1 + 25);
  v16 = (__int64)a1[1];
  v17 = *((_BYTE *)a1 + 24);
  v18 = *a1;
  v146 = *(_DWORD *)(a2 + 64);
  v19 = sub_1F7F730(a2, v16, v18, v15, v17, a3, *(double *)a4.m128i_i64, a5);
  if ( v19 )
  {
    v20 = v19;
    goto LABEL_5;
  }
  v22 = *(_WORD *)(v9 + 24);
  if ( ((v22 - 142) & 0xFFFD) == 0 )
  {
    v20 = sub_1D309E0(
            *a1,
            142,
            (__int64)&v145,
            v143,
            v144,
            0,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            *(_OWORD *)*(_QWORD *)(v9 + 32));
    goto LABEL_5;
  }
  if ( v22 != 145 )
  {
    v23 = (__int64)a1[1];
LABEL_11:
    v24 = sub_1FA3590(
            *a1,
            a1,
            v23,
            v143,
            (__int64)v144,
            *((_BYTE *)a1 + 24),
            (__m128)a3,
            *(double *)a4.m128i_i64,
            a5,
            a2,
            v8,
            v10,
            2,
            0x8Eu);
    if ( v24 )
      goto LABEL_12;
    v24 = sub_1FA1A20((__int64 *)a1, a2, a3, a4, a5);
    if ( v24 )
      goto LABEL_12;
    v24 = sub_1FA0740(*a1, (__int64)a1, (__int64)a1[1], v143, (__int64)v144, *((_BYTE *)a1 + 24), a2, v9, v142, 2);
    if ( v24 )
      goto LABEL_12;
    v46 = *(_WORD *)(v9 + 24);
    if ( (unsigned __int16)(v46 - 118) <= 2u )
    {
      v47 = *(_QWORD **)(v9 + 32);
      if ( *(_WORD *)(*v47 + 24LL) == 185
        && *(_WORD *)(v47[5] + 24LL) == 10
        && !*((_BYTE *)a1 + 24)
        && sub_1F6C830((__int64)a1[1], v46, v143) )
      {
        v58 = *(_BYTE *)(v57 + 88);
        if ( v58 )
        {
          if ( v55
            && (*(_BYTE *)(v56 + 2 * (v58 + 115LL * v55 + 16104) + 1) & 0xF) == 0
            && ((*(_BYTE *)(v57 + 27) ^ 0xC) & 0xC) != 0
            && (*(_WORD *)(v57 + 26) & 0x380) == 0 )
          {
            v149.m128i_i64[0] = (__int64)v150;
            v140.m128i_i64[0] = (__int64)&v149;
            v149.m128i_i64[1] = 0x400000000LL;
            v59 = *(_QWORD *)(v9 + 32);
            v141.m128_u64[0] = v57;
            if ( (unsigned __int8)sub_1F6D830(
                                    v143,
                                    (__int64)v144,
                                    v9,
                                    *(_QWORD *)v59,
                                    *(_DWORD *)(v59 + 8),
                                    142,
                                    (__int64)&v149,
                                    v56) )
            {
              v60 = v141.m128_u64[0];
              v61 = *a1;
              v62 = *(_QWORD *)(v141.m128_u64[0] + 104);
              v63 = *(unsigned __int8 *)(v141.m128_u64[0] + 88);
              v64 = *(_QWORD *)(v141.m128_u64[0] + 96);
              v65 = *(_QWORD *)(v141.m128_u64[0] + 32);
              v148.m128i_i64[0] = *(_QWORD *)(v141.m128_u64[0] + 72);
              if ( v148.m128i_i64[0] )
              {
                v131 = v63;
                v132 = v64;
                v138.m128i_i64[0] = v62;
                v139.m128i_i64[0] = v65;
                sub_1F6CA20(v148.m128i_i64);
                v63 = v131;
                v64 = v132;
                v62 = v138.m128i_i64[0];
                v65 = v139.m128i_i64[0];
                v60 = v141.m128_u64[0];
              }
              v148.m128i_i32[2] = *(_DWORD *)(v60 + 64);
              v122 = v60;
              v66.m128_u64[0] = sub_1D2B590(
                                  v61,
                                  2,
                                  (__int64)&v148,
                                  v143,
                                  (__int64)v144,
                                  v62,
                                  *(_OWORD *)v65,
                                  *(_QWORD *)(v65 + 40),
                                  *(_QWORD *)(v65 + 48),
                                  v63,
                                  v64);
              v141 = v66;
              v139.m128i_i64[0] = v66.m128_u64[0];
              sub_17CD270(v148.m128i_i64);
              v67 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 40LL) + 88LL) + 24LL;
              sub_13A38D0((__int64)v147, v67);
              v72 = sub_1D159A0((char *)&v143, v67, v68, v69, v70, v71, v119, v122, v125, v127);
              sub_16A5B10((__int64)&v148, v147, v72);
              sub_1F6C9E0(v147, v148.m128i_i64);
              sub_135E100(v148.m128i_i64);
              v138.m128i_i64[0] = (__int64)*a1;
              *(_QWORD *)&v73 = sub_1D38970(
                                  v138.m128i_i64[0],
                                  (__int64)v147,
                                  (__int64)&v145,
                                  v143,
                                  v144,
                                  0,
                                  a3,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  0);
              v74.m128i_i64[0] = (__int64)sub_1D332F0(
                                            (__int64 *)v138.m128i_i64[0],
                                            *(unsigned __int16 *)(v9 + 24),
                                            (__int64)&v145,
                                            v143,
                                            v144,
                                            0,
                                            *(double *)a3.m128i_i64,
                                            *(double *)a4.m128i_i64,
                                            a5,
                                            v141.m128_i64[0],
                                            v141.m128_u64[1],
                                            v73);
              v138 = v74;
              sub_1FA0970(
                a1,
                v140.m128i_i64[0],
                **(_QWORD **)(v9 + 32),
                *(_DWORD *)(*(_QWORD *)(v9 + 32) + 8LL),
                v141.m128_i64[0],
                v141.m128_i64[1],
                (__m128)a3,
                *(double *)a4.m128i_i64,
                a5,
                0x8Eu);
              v128 = sub_1D18C00(v9, 1, v142);
              v75 = sub_1D18C00(v123, 1, 0);
              v76 = _mm_load_si128(&v138);
              v140.m128i_i8[0] = v75;
              v148 = v76;
              sub_1F994A0((__int64)a1, a2, v148.m128i_i64, 1, 1);
              v77 = v123;
              if ( !v128 )
              {
                v101 = (const void ***)(*(_QWORD *)(v9 + 40) + 16LL * (unsigned int)v142);
                v102.m128i_i64[0] = sub_1D309E0(
                                      *a1,
                                      145,
                                      (__int64)&v145,
                                      *(unsigned __int8 *)v101,
                                      v101[1],
                                      0,
                                      *(double *)a3.m128i_i64,
                                      *(double *)v76.m128i_i64,
                                      *(double *)a5.m128i_i64,
                                      *(_OWORD *)&v138);
                v148 = v102;
                sub_1F994A0((__int64)a1, v9, v148.m128i_i64, 1, 1);
                v77 = v123;
              }
              v78 = *a1;
              if ( v140.m128i_i8[0] )
              {
                sub_1D44C70((__int64)*a1, v77, 1, v139.m128i_i64[0], 1u);
              }
              else
              {
                v79 = *(const void ***)(*(_QWORD *)(v77 + 40) + 8LL);
                v80 = **(unsigned __int8 **)(v77 + 40);
                v148.m128i_i64[0] = *(_QWORD *)(v77 + 72);
                if ( v148.m128i_i64[0] )
                {
                  v138.m128i_i64[0] = v80;
                  v140.m128i_i64[0] = v77;
                  *(_QWORD *)&v142 = v79;
                  sub_1F6CA20(v148.m128i_i64);
                  v80 = v138.m128i_i64[0];
                  v77 = v140.m128i_i64[0];
                  v79 = (const void **)v142;
                }
                v81 = *(_DWORD *)(v77 + 64);
                v140.m128i_i64[0] = v77;
                v148.m128i_i32[2] = v81;
                *(_QWORD *)&v142 = sub_1D309E0(
                                     v78,
                                     145,
                                     (__int64)&v148,
                                     v80,
                                     v79,
                                     0,
                                     *(double *)a3.m128i_i64,
                                     *(double *)v76.m128i_i64,
                                     *(double *)a5.m128i_i64,
                                     *(_OWORD *)&v141);
                *((_QWORD *)&v142 + 1) = v82;
                sub_17CD270(v148.m128i_i64);
                sub_1F9A400((__int64)a1, v140.m128i_i64[0], v142, *((__int64 *)&v142 + 1), v139.m128i_i64[0], 1, 1);
              }
              v20 = a2;
              sub_135E100(v147);
              if ( (_BYTE *)v149.m128i_i64[0] != v150 )
                _libc_free(v149.m128i_u64[0]);
              goto LABEL_5;
            }
            if ( (_BYTE *)v149.m128i_i64[0] != v150 )
              _libc_free(v149.m128i_u64[0]);
          }
        }
      }
    }
    v24 = (__int64)sub_1F776B0(a2, *a1, *((_BYTE *)a1 + 24), a3, *(double *)a4.m128i_i64, a5);
    if ( v24 )
      goto LABEL_12;
    if ( *(_WORD *)(v9 + 24) != 137 )
    {
LABEL_36:
      if ( (!*((_BYTE *)a1 + 24) || sub_1F6C830((__int64)a1[1], 0x8Fu, v143))
        && (unsigned __int8)sub_1D1F9F0((__int64)*a1, v8, v10, 0) )
      {
        *((_QWORD *)&v117 + 1) = v10;
        *(_QWORD *)&v117 = v8;
        v20 = sub_1D309E0(
                *a1,
                143,
                (__int64)&v145,
                v143,
                v144,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                v117);
      }
      else
      {
        v48 = sub_1F77270(a1, a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        v49 = 0;
        if ( v48 )
          v49 = v48;
        v20 = (__int64)v49;
      }
      goto LABEL_5;
    }
    v83 = *(__int64 **)(v9 + 32);
    a5 = _mm_loadu_si128((const __m128i *)v83);
    v84 = _mm_loadu_si128((const __m128i *)(v83 + 5));
    v85 = *(_DWORD *)(v83[10] + 84);
    v86 = *v83;
    v140 = a5;
    v87 = *((unsigned int *)v83 + 2);
    v139 = v84;
    v141.m128_i32[0] = v85;
    v88 = *(_QWORD *)(v86 + 40) + 16 * v87;
    LOBYTE(v86) = *(_BYTE *)v88;
    v148.m128i_i64[1] = *(_QWORD *)(v88 + 8);
    v148.m128i_i8[0] = v86;
    v134 = v86;
    v138.m128i_i64[0] = (__int64)&v143;
    if ( !sub_1F7E0D0((__int64)&v143) || *((_BYTE *)a1 + 24) )
    {
LABEL_85:
      if ( (unsigned int)sub_1F701D0(v9, v142) == 1 )
        *(_QWORD *)&v142 = sub_1D389D0((__int64)*a1, (__int64)&v145, v143, v144, 0, 0, a3, *(double *)a4.m128i_i64, a5);
      else
        *(_QWORD *)&v142 = sub_1D395A0(
                             (__int64)*a1,
                             1,
                             (__int64)&v145,
                             v143,
                             v144,
                             v92,
                             *(double *)a3.m128i_i64,
                             a4,
                             a5,
                             *(_OWORD *)&v148);
      *((_QWORD *)&v142 + 1) = v93;
      *(_QWORD *)&v94 = sub_1D38BB0((__int64)*a1, 0, (__int64)&v145, v143, v144, 0, a3, *(double *)a4.m128i_i64, a5, 0);
      v135 = v94;
      v24 = (__int64)sub_1F87CB0(
                       a1,
                       (__int64)&v145,
                       v140.m128i_u64[0],
                       v140.m128i_u64[1],
                       v139.m128i_i64[0],
                       v139.m128i_i64[1],
                       a3,
                       *(double *)a4.m128i_i64,
                       a5,
                       v142,
                       *((__int64 *)&v142 + 1),
                       v94,
                       v141.m128_u32[0],
                       1);
      if ( !v24 )
      {
        if ( !sub_1F7E0D0(v138.m128i_i64[0])
          && !(*(unsigned __int8 (__fastcall **)(_QWORD *, _QWORD, const void **))(*a1[1] + 712LL))(a1[1], v143, v144) )
        {
          v95 = sub_1F6C7D0((__int64)*a1, (__int64)a1[1], v148.m128i_u32[0], v148.m128i_i64[1]);
          v97 = v96;
          v98 = v95;
          if ( !*((_BYTE *)a1 + 24) || sub_1F6C830((__int64)a1[1], 0x89u, v148.m128i_u8[0]) )
          {
            v99 = sub_1F81070(
                    *a1,
                    (__int64)&v145,
                    v98,
                    v97,
                    v140.m128i_u64[0],
                    (__int16 *)v140.m128i_i64[1],
                    (__m128)a3,
                    *(double *)a4.m128i_i64,
                    a5,
                    *(_OWORD *)&v139,
                    v141.m128_u32[0]);
            v20 = (__int64)sub_1F810E0(
                             *a1,
                             (__int64)&v145,
                             v143,
                             v144,
                             (unsigned __int64)v99,
                             v100,
                             (__m128)a3,
                             *(double *)a4.m128i_i64,
                             a5,
                             v142,
                             v135,
                             *((__int64 *)&v135 + 1));
            goto LABEL_5;
          }
        }
        goto LABEL_36;
      }
LABEL_12:
      v20 = v24;
      goto LABEL_5;
    }
    v89 = a1[1];
    v149 = _mm_loadu_si128(&v148);
    if ( v134 )
    {
      if ( (unsigned __int8)(v134 - 14) > 0x5Fu )
      {
        v90 = (unsigned __int8)(v134 - 8) <= 5u || (unsigned __int8)(v134 - 86) <= 0x17u;
        goto LABEL_82;
      }
    }
    else
    {
      v127 = (__int64)v89;
      v137 = sub_1F58CD0((__int64)&v149);
      v115 = sub_1F58D20((__int64)&v149);
      v90 = v137;
      if ( !v115 )
      {
LABEL_82:
        if ( v90 )
          v91 = v89[16];
        else
          v91 = v89[15];
LABEL_84:
        if ( v91 == 2 )
        {
          v149.m128i_i32[0] = sub_1F6C7D0((__int64)*a1, (__int64)v89, v148.m128i_u32[0], v148.m128i_i64[1]);
          v149.m128i_i64[1] = v103;
          v136 = sub_1D159A0((char *)v138.m128i_i64[0], (__int64)v89, v103, v104, v105, v106, v119, v121, v125, v127);
          if ( v136 == (unsigned int)sub_1D159A0(
                                       v149.m128i_i8,
                                       (__int64)v89,
                                       v107,
                                       v108,
                                       v109,
                                       v110,
                                       v120,
                                       v124,
                                       v126,
                                       v129) )
          {
            v20 = (__int64)sub_1F81070(
                             *a1,
                             (__int64)&v145,
                             v143,
                             v144,
                             v140.m128i_u64[0],
                             (__int16 *)v140.m128i_i64[1],
                             (__m128)a3,
                             *(double *)a4.m128i_i64,
                             a5,
                             *(_OWORD *)&v139,
                             v141.m128_u32[0]);
            goto LABEL_5;
          }
          v111 = sub_1F7DF20(&v148);
          if ( v149.m128i_i8[0] == (_BYTE)v111 && (v149.m128i_i8[0] || (const void **)v149.m128i_i64[1] == v112) )
          {
            v113 = sub_1F81070(
                     *a1,
                     (__int64)&v145,
                     v111,
                     v112,
                     v140.m128i_u64[0],
                     (__int16 *)v140.m128i_i64[1],
                     (__m128)a3,
                     *(double *)a4.m128i_i64,
                     a5,
                     *(_OWORD *)&v139,
                     v141.m128_u32[0]);
            v20 = sub_1D322C0(
                    *a1,
                    (__int64)v113,
                    v114,
                    (__int64)&v145,
                    v143,
                    v144,
                    *(double *)a3.m128i_i64,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64);
            goto LABEL_5;
          }
        }
        goto LABEL_85;
      }
    }
    v91 = v89[17];
    goto LABEL_84;
  }
  v25.m128i_i64[0] = sub_1F84730(a1, v9, *(double *)a3.m128i_i64, a4, a5);
  if ( !v25.m128i_i64[0] )
  {
    v27 = *(const __m128i **)(v9 + 32);
    a3 = _mm_loadu_si128(v27);
    v28 = v27->m128i_i64[0];
    v29 = v27->m128i_u32[2];
    v141 = (__m128)a3;
    v127 = v28;
    v130 = v29;
    v133 = sub_1F701D0(v28, v29);
    v138.m128i_i32[0] = sub_1F701D0(v9, v142);
    v139.m128i_i32[0] = sub_1D159C0((__int64)&v143, (unsigned int)v142, v30, v31, v32, v33);
    v140.m128i_i64[0] = a3.m128i_i64[1];
    v34 = sub_1D23330((__int64)*a1, a3.m128i_i64[0], a3.m128i_i64[1], 0);
    v36 = v139.m128i_u32[0];
    v37 = v133;
    if ( v133 != v139.m128i_i32[0] )
    {
      v38 = v133 - v138.m128i_i32[0];
      v39 = *a1;
      if ( v133 >= v139.m128i_i32[0] )
      {
        if ( v34 <= v38 )
        {
          v40 = (unsigned int)v142;
          if ( !*((_BYTE *)a1 + 24) )
          {
LABEL_27:
            if ( v133 > v139.m128i_i32[0] )
            {
              *(_QWORD *)&v142 = v40;
              v140.m128i_i64[0] = (__int64)v39;
              sub_1F80610((__int64)&v149, v8);
              v127 = sub_1D309E0(
                       (__int64 *)v140.m128i_i64[0],
                       145,
                       (__int64)&v149,
                       v143,
                       v144,
                       0,
                       *(double *)a3.m128i_i64,
                       *(double *)a4.m128i_i64,
                       *(double *)a5.m128i_i64,
                       *(_OWORD *)&v141);
              v130 = v44;
              sub_17CD270(v149.m128i_i64);
              v39 = *a1;
              v40 = v142;
              v37 = v118;
            }
            goto LABEL_29;
          }
          goto LABEL_22;
        }
        v54 = sub_1D309E0(
                v39,
                145,
                (__int64)&v145,
                v143,
                v144,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(_OWORD *)&v141);
      }
      else
      {
        if ( v34 <= v38 )
        {
          if ( !*((_BYTE *)a1 + 24) )
          {
            v40 = (unsigned int)v142;
            goto LABEL_50;
          }
LABEL_22:
          v40 = (unsigned int)v142;
          v23 = (__int64)a1[1];
          v41 = 1;
          v42 = *(unsigned __int8 *)(*(_QWORD *)(v9 + 40) + 16LL * (unsigned int)v142);
          if ( (_BYTE)v42 != 1 )
          {
            if ( !(_BYTE)v42 )
              goto LABEL_11;
            v41 = (unsigned __int8)v42;
            if ( !*(_QWORD *)(v23 + 8 * v42 + 120) )
              goto LABEL_11;
          }
          v43 = v41;
          v35 = 129LL * v41;
          if ( *(_BYTE *)(v23 + v43 + 2 * v35 + 2570) )
            goto LABEL_11;
          v39 = *a1;
          if ( v133 >= v139.m128i_i32[0] )
            goto LABEL_27;
LABEL_50:
          v50 = *(_QWORD *)(v9 + 72);
          v140.m128i_i64[0] = (__int64)&v149;
          v149.m128i_i64[0] = v50;
          if ( v50 )
          {
            v139.m128i_i64[0] = (__int64)v39;
            *(_QWORD *)&v142 = v40;
            sub_1623A60((__int64)&v149, v50, 2);
            v39 = (__int64 *)v139.m128i_i64[0];
            v40 = v142;
          }
          v51 = *(_DWORD *)(v9 + 64);
          *(_QWORD *)&v142 = v40;
          v149.m128i_i32[2] = v51;
          v52 = sub_1D309E0(
                  v39,
                  144,
                  v140.m128i_i64[0],
                  v143,
                  v144,
                  0,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64,
                  *(_OWORD *)&v141);
          v36 = v116;
          v40 = v142;
          v127 = v52;
          v130 = v53;
          if ( v149.m128i_i64[0] )
          {
            sub_161E7C0(v140.m128i_i64[0], v149.m128i_i64[0]);
            v39 = *a1;
            v40 = v142;
          }
          else
          {
            v39 = *a1;
          }
          goto LABEL_29;
        }
        v54 = sub_1D309E0(
                v39,
                142,
                (__int64)&v145,
                v143,
                v144,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                *(_OWORD *)&v141);
      }
      v20 = v54;
      goto LABEL_5;
    }
    v35 = v133 - v138.m128i_i32[0];
    if ( (unsigned int)v35 < v34 )
    {
      v20 = v141.m128_u64[0];
      goto LABEL_5;
    }
    if ( !*((_BYTE *)a1 + 24) )
    {
      v39 = *a1;
      v40 = (unsigned int)v142;
LABEL_29:
      *(_QWORD *)&v142 = v39;
      *(_QWORD *)&v45 = sub_1D2EF30(
                          v39,
                          *(unsigned __int8 *)(*(_QWORD *)(v9 + 40) + 16 * v40),
                          *(_QWORD *)(*(_QWORD *)(v9 + 40) + 16 * v40 + 8),
                          v35,
                          v37,
                          v36);
      v141.m128_u64[0] = v127;
      v141.m128_u64[1] = v130 | v141.m128_u64[1] & 0xFFFFFFFF00000000LL;
      v20 = (__int64)sub_1D332F0(
                       (__int64 *)v142,
                       148,
                       (__int64)&v145,
                       v143,
                       v144,
                       0,
                       *(double *)a3.m128i_i64,
                       *(double *)a4.m128i_i64,
                       a5,
                       v127,
                       v141.m128_u64[1],
                       v45);
      goto LABEL_5;
    }
    goto LABEL_22;
  }
  if ( v9 != v25.m128i_i64[0] )
  {
    v26 = **(_QWORD **)(v9 + 32);
    v149 = v25;
    sub_1F994A0((__int64)a1, v9, v149.m128i_i64, 1, 1);
    sub_1F81BC0((__int64)a1, v26);
  }
  v20 = a2;
LABEL_5:
  if ( v145 )
    sub_161E7C0((__int64)&v145, v145);
  return v20;
}
