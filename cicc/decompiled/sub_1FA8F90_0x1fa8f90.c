// Function: sub_1FA8F90
// Address: 0x1fa8f90
//
__int64 __fastcall sub_1FA8F90(_QWORD **a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r14
  unsigned __int64 v11; // r15
  __m128i v12; // xmm0
  __int64 v13; // rax
  char v14; // dl
  const void **v15; // rax
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // r8d
  int v21; // r9d
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdi
  __int64 v28; // rdx
  int v29; // eax
  bool v30; // al
  unsigned int v31; // eax
  int v32; // r8d
  int v33; // r9d
  unsigned int v34; // edx
  __int64 v35; // r9
  __int64 v36; // r8
  __int64 v37; // r8
  unsigned __int64 v38; // rcx
  char v39; // al
  __int64 v40; // rsi
  __int64 *v41; // r12
  unsigned int v42; // ecx
  unsigned __int64 v43; // rdx
  __int128 v44; // rax
  __int64 v45; // rdx
  int v46; // ecx
  int v47; // r8d
  int v48; // r9d
  unsigned __int16 v49; // ax
  int v50; // edx
  int v51; // eax
  __int64 *v52; // rdx
  __int64 v53; // rax
  __int16 v54; // ax
  __int64 v55; // rax
  __int64 v56; // rcx
  _QWORD *v57; // rax
  __int64 v58; // rsi
  _QWORD *v59; // rdi
  unsigned int v60; // ecx
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdx
  int v67; // ecx
  int v68; // r8d
  int v69; // r9d
  __int64 v70; // rax
  __int64 v71; // r12
  __int64 v72; // rax
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rax
  bool v76; // al
  int v77; // ecx
  int v78; // r8d
  int v79; // r9d
  __int64 *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int128 v83; // rax
  __int64 *v84; // rdi
  __int64 v85; // rax
  __int64 *v86; // rax
  unsigned __int64 v87; // rdx
  __int64 *v88; // r12
  __int128 v89; // rax
  __int64 *v90; // r13
  __int128 v91; // rax
  __int64 *v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rdx
  __int64 v95; // rsi
  __int64 v96; // rcx
  __int64 v97; // r9
  int v98; // eax
  bool v99; // r10
  const __m128i *v100; // r8
  __int64 v101; // rdx
  int v102; // ecx
  int v103; // r8d
  int v104; // r9d
  __int64 v105; // rax
  __int64 *v106; // r12
  int v107; // ecx
  int v108; // r8d
  int v109; // r9d
  __int64 v110; // rax
  __int64 v111; // rax
  _QWORD *v112; // r12
  __int128 v113; // [rsp-10h] [rbp-E0h]
  unsigned int v114; // [rsp+14h] [rbp-BCh]
  __int64 v115; // [rsp+18h] [rbp-B8h]
  unsigned int v116; // [rsp+20h] [rbp-B0h]
  char v117; // [rsp+20h] [rbp-B0h]
  char v118; // [rsp+20h] [rbp-B0h]
  unsigned int v119; // [rsp+20h] [rbp-B0h]
  int v120; // [rsp+20h] [rbp-B0h]
  int v121; // [rsp+20h] [rbp-B0h]
  __int64 v122; // [rsp+30h] [rbp-A0h]
  unsigned int v123; // [rsp+38h] [rbp-98h]
  __int64 v124; // [rsp+40h] [rbp-90h]
  __m128i v125; // [rsp+40h] [rbp-90h]
  unsigned int v126; // [rsp+50h] [rbp-80h]
  unsigned __int64 v127; // [rsp+58h] [rbp-78h]
  __int64 v128; // [rsp+58h] [rbp-78h]
  __int64 v129; // [rsp+58h] [rbp-78h]
  __int64 v130; // [rsp+58h] [rbp-78h]
  unsigned int v131; // [rsp+60h] [rbp-70h] BYREF
  const void **v132; // [rsp+68h] [rbp-68h]
  __int64 v133; // [rsp+70h] [rbp-60h] BYREF
  int v134; // [rsp+78h] [rbp-58h]
  unsigned __int64 v135; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v136; // [rsp+88h] [rbp-48h]
  unsigned __int64 v137; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v138; // [rsp+98h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_QWORD *)v7;
  v10 = *(_QWORD *)v7;
  v11 = *(_QWORD *)(v7 + 8);
  v12 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v123 = *(_DWORD *)(v7 + 8);
  v127 = *(_QWORD *)(v7 + 40);
  v126 = *(_DWORD *)(v7 + 48);
  v13 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v123;
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  v133 = v8;
  LOBYTE(v131) = v14;
  v132 = v15;
  if ( v8 )
  {
    sub_1623A60((__int64)&v133, v8, 2);
    v14 = v131;
  }
  v134 = *(_DWORD *)(a2 + 64);
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_5;
  }
  else if ( !sub_1F58D20((__int64)&v131) )
  {
    goto LABEL_5;
  }
  result = (__int64)sub_1FA8C50((__int64)a1, a2, *(double *)v12.m128i_i64, *(double *)a4.m128i_i64, a5);
  if ( result )
    goto LABEL_9;
  if ( (unsigned __int8)sub_1D16620(v127, (__int64 *)a2) )
    goto LABEL_15;
LABEL_5:
  v16 = (__int64)*a1;
  if ( v127 == v9 && v123 == v126 )
  {
    result = sub_1F6EEE0(
               (__int64)&v133,
               (__int64)a1[1],
               v131,
               v132,
               (__int64)*a1,
               *((_BYTE *)a1 + 24),
               v12,
               *(double *)a4.m128i_i64,
               a5);
    goto LABEL_9;
  }
  if ( sub_1D23600(v16, v10) && sub_1D23600((__int64)*a1, v12.m128i_i64[0]) )
  {
    result = (__int64)sub_1D32920(
                        *a1,
                        0x35u,
                        (__int64)&v133,
                        v131,
                        (__int64)v132,
                        v9,
                        *(double *)v12.m128i_i64,
                        *(double *)a4.m128i_i64,
                        a5,
                        v127);
    goto LABEL_9;
  }
  result = (__int64)sub_1F77C50(a1, a2, *(double *)v12.m128i_i64, *(double *)a4.m128i_i64, a5);
  if ( !result )
  {
    v22 = *(unsigned __int16 *)(v127 + 24);
    if ( (v22 == 10 || v22 == 32) && (*(_BYTE *)(v127 + 26) & 8) == 0 )
    {
      v40 = *(_QWORD *)(v127 + 88);
      v41 = *a1;
      v42 = *(_DWORD *)(v40 + 32);
      v136 = v42;
      if ( v42 > 0x40 )
      {
        sub_16A4FD0((__int64)&v135, (const void **)(v40 + 24));
        LOBYTE(v42) = v136;
        if ( v136 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v135);
LABEL_44:
          sub_16A7400((__int64)&v135);
          v138 = v136;
          v136 = 0;
          v137 = v135;
          *(_QWORD *)&v44 = sub_1D38970(
                              (__int64)v41,
                              (__int64)&v137,
                              (__int64)&v133,
                              v131,
                              v132,
                              0,
                              v12,
                              *(double *)a4.m128i_i64,
                              a5,
                              0);
          result = (__int64)sub_1D332F0(
                              v41,
                              52,
                              (__int64)&v133,
                              v131,
                              v132,
                              0,
                              *(double *)v12.m128i_i64,
                              *(double *)a4.m128i_i64,
                              a5,
                              v10,
                              v11,
                              v44);
          if ( v138 > 0x40 && v137 )
          {
            v129 = result;
            j_j___libc_free_0_0(v137);
            result = v129;
          }
          if ( v136 > 0x40 && v135 )
          {
            v130 = result;
            j_j___libc_free_0_0(v135);
            result = v130;
          }
          goto LABEL_9;
        }
        v43 = v135;
      }
      else
      {
        v43 = *(_QWORD *)(v40 + 24);
      }
      v135 = ~v43 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v42);
      goto LABEL_44;
    }
    v23 = sub_1D1ADA0(v10, v11, v18, v19, v20, v21);
    if ( !v23 )
      goto LABEL_50;
    v27 = *(_QWORD *)(v23 + 88);
    v28 = *(unsigned int *)(v27 + 32);
    if ( (unsigned int)v28 <= 0x40 )
    {
      v30 = *(_QWORD *)(v27 + 24) == 0;
    }
    else
    {
      v116 = *(_DWORD *)(v27 + 32);
      v29 = sub_16A57B0(v27 + 24);
      v28 = v116;
      v30 = v116 == v29;
    }
    if ( !v30 )
      goto LABEL_50;
    v31 = sub_1D159C0((__int64)&v131, v11, v28, v24, v25, v26);
    v34 = v31;
    if ( (unsigned int)*(unsigned __int16 *)(v127 + 24) - 123 <= 1 )
    {
      v119 = v31;
      v55 = sub_1D1ADA0(
              *(_QWORD *)(*(_QWORD *)(v127 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v127 + 32) + 48LL),
              v31,
              v127,
              v32,
              v33);
      v34 = v119;
      if ( v55 )
      {
        v56 = *(_QWORD *)(v55 + 88);
        v57 = *(_QWORD **)(v56 + 24);
        if ( *(_DWORD *)(v56 + 32) > 0x40u )
          v57 = (_QWORD *)*v57;
        if ( (_QWORD *)(v119 - 1) == v57 )
        {
          v58 = (unsigned int)(*(_WORD *)(v127 + 24) == 123) + 123;
          if ( !*((_BYTE *)a1 + 24)
            || ((v59 = a1[1], v60 = 1, (_BYTE)v131 == 1)
             || (_BYTE)v131 && (v60 = (unsigned __int8)v131, v59[(unsigned __int8)v131 + 15]))
            && !*((_BYTE *)v59 + 259 * v60 + (unsigned int)v58 + 2422) )
          {
            result = (__int64)sub_1D332F0(
                                *a1,
                                v58,
                                (__int64)&v133,
                                v131,
                                v132,
                                0,
                                *(double *)v12.m128i_i64,
                                *(double *)a4.m128i_i64,
                                a5,
                                **(_QWORD **)(v127 + 32),
                                *(_QWORD *)(*(_QWORD *)(v127 + 32) + 8LL),
                                *(_OWORD *)(*(_QWORD *)(v127 + 32) + 40LL));
            goto LABEL_9;
          }
        }
      }
    }
    if ( (*(_BYTE *)(a2 + 80) & 2) != 0 )
    {
LABEL_15:
      result = v10;
      goto LABEL_9;
    }
    v136 = v34;
    v35 = (__int64)*a1;
    v36 = 1LL << ((unsigned __int8)v34 - 1);
    if ( v34 > 0x40 )
    {
      v114 = v34 - 1;
      v115 = 1LL << ((unsigned __int8)v34 - 1);
      v122 = (__int64)*a1;
      sub_16A4EF0((__int64)&v135, 0, 0);
      v34 = v136;
      v35 = v122;
      v36 = v115;
      if ( v136 > 0x40 )
      {
        *(_QWORD *)(v135 + 8LL * (v114 >> 6)) |= v115;
        v34 = v136;
        if ( v136 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v135);
          v38 = v135;
          v34 = v136;
          v35 = v122;
          goto LABEL_31;
        }
        v37 = v135;
LABEL_30:
        v38 = ~v37 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v34);
        v135 = v38;
LABEL_31:
        v138 = v34;
        v137 = v38;
        v136 = 0;
        v39 = sub_1D1F940(v35, v12.m128i_i64[0], v12.m128i_i64[1], (__int64)&v137, 0);
        if ( v138 > 0x40 && v137 )
        {
          v117 = v39;
          j_j___libc_free_0_0(v137);
          v39 = v117;
        }
        if ( v136 > 0x40 && v135 )
        {
          v118 = v39;
          j_j___libc_free_0_0(v135);
          v39 = v118;
        }
        if ( v39 )
        {
          if ( (*(_BYTE *)(a2 + 80) & 4) == 0 )
          {
            result = v12.m128i_i64[0];
            goto LABEL_9;
          }
          goto LABEL_15;
        }
LABEL_50:
        if ( (unsigned __int8)sub_1F709E0(v10, v11) )
        {
          *((_QWORD *)&v113 + 1) = v11;
          *(_QWORD *)&v113 = v10;
          result = (__int64)sub_1D332F0(
                              *a1,
                              120,
                              (__int64)&v133,
                              v131,
                              v132,
                              0,
                              *(double *)v12.m128i_i64,
                              *(double *)a4.m128i_i64,
                              a5,
                              v127,
                              v126 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                              v113);
          goto LABEL_9;
        }
        v49 = *(_WORD *)(v127 + 24);
        if ( v49 == 53 )
        {
          v74 = sub_1D1ADA0(**(_QWORD **)(v127 + 32), *(_QWORD *)(*(_QWORD *)(v127 + 32) + 8LL), v45, v46, v47, v48);
          if ( v74 )
          {
            v75 = *(_QWORD *)(v74 + 88);
            if ( *(_DWORD *)(v75 + 32) <= 0x40u )
            {
              v76 = *(_QWORD *)(v75 + 24) == 0;
            }
            else
            {
              v120 = *(_DWORD *)(v75 + 32);
              v76 = v120 == (unsigned int)sub_16A57B0(v75 + 24);
            }
            if ( v76 )
            {
              result = (__int64)sub_1D332F0(
                                  *a1,
                                  52,
                                  (__int64)&v133,
                                  v131,
                                  v132,
                                  0,
                                  *(double *)v12.m128i_i64,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  v10,
                                  v11,
                                  *(_OWORD *)(*(_QWORD *)(v127 + 32) + 40LL));
              goto LABEL_9;
            }
          }
          v49 = *(_WORD *)(v127 + 24);
        }
        v50 = v49;
        if ( v49 == 53 )
        {
          v61 = *(_QWORD *)(v127 + 32);
          if ( *(_QWORD *)v61 == v9 && *(_DWORD *)(v61 + 8) == v123 )
            goto LABEL_120;
          v51 = *(unsigned __int16 *)(v9 + 24);
          if ( (_WORD)v51 != 52 )
            goto LABEL_55;
        }
        else
        {
          v51 = *(unsigned __int16 *)(v9 + 24);
          if ( (_WORD)v51 != 52 )
          {
            if ( v50 != 52 )
            {
LABEL_55:
              if ( v51 == 53 )
              {
                v52 = *(__int64 **)(v9 + 32);
                v53 = v52[5];
                if ( *(_WORD *)(v53 + 24) == 53 )
                {
                  v85 = *(_QWORD *)(v53 + 32);
                  if ( v127 == *(_QWORD *)(v85 + 40) && *(_DWORD *)(v85 + 48) == v126 )
                  {
                    result = (__int64)sub_1D332F0(
                                        *a1,
                                        53,
                                        (__int64)&v133,
                                        v131,
                                        v132,
                                        0,
                                        *(double *)v12.m128i_i64,
                                        *(double *)a4.m128i_i64,
                                        a5,
                                        *v52,
                                        v52[1],
                                        *(_OWORD *)v85);
                    goto LABEL_9;
                  }
                }
              }
              goto LABEL_57;
            }
LABEL_106:
            a4 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v127 + 32) + 40LL));
            if ( (unsigned __int8)sub_1F70310(v10, v11, 1u)
              && (unsigned __int8)sub_1F70310(a4.m128i_i64[0], a4.m128i_u32[2], 1u) )
            {
              v86 = sub_1D332F0(
                      *a1,
                      53,
                      (__int64)&v133,
                      v131,
                      v132,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)a4.m128i_i64,
                      a5,
                      v10,
                      v11,
                      *(_OWORD *)&a4);
              result = (__int64)sub_1D332F0(
                                  *a1,
                                  53,
                                  (__int64)&v133,
                                  v131,
                                  v132,
                                  0,
                                  *(double *)v12.m128i_i64,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  (__int64)v86,
                                  v87,
                                  *(_OWORD *)*(_QWORD *)(v127 + 32));
              goto LABEL_9;
            }
            v51 = *(unsigned __int16 *)(v9 + 24);
            if ( v51 != 52 )
              goto LABEL_55;
            v61 = *(_QWORD *)(v9 + 32);
            v62 = *(_QWORD *)(v61 + 40);
LABEL_79:
            v63 = *(unsigned __int16 *)(v62 + 24);
            if ( (unsigned int)(v63 - 52) <= 1 )
            {
              v64 = *(_QWORD *)(v62 + 32);
              if ( v127 == *(_QWORD *)v64 && *(_DWORD *)(v64 + 8) == v126 )
              {
                result = (__int64)sub_1D332F0(
                                    *a1,
                                    v63,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    *(_QWORD *)v61,
                                    *(_QWORD *)(v61 + 8),
                                    *(_OWORD *)(v64 + 40));
                goto LABEL_9;
              }
            }
            if ( *(_WORD *)(v62 + 24) == 52 )
            {
              v65 = *(_QWORD *)(v62 + 32);
              if ( v127 == *(_QWORD *)(v65 + 40) && *(_DWORD *)(v65 + 48) == v126 )
              {
                result = (__int64)sub_1D332F0(
                                    *a1,
                                    52,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    *(_QWORD *)v61,
                                    *(_QWORD *)(v61 + 8),
                                    *(_OWORD *)v65);
                goto LABEL_9;
              }
            }
LABEL_57:
            v54 = *(_WORD *)(v127 + 24);
            if ( v54 == 53 )
            {
              if ( sub_1D18C00(v127, 1, v126) )
              {
                v88 = *a1;
                *(_QWORD *)&v89 = sub_1D332F0(
                                    *a1,
                                    53,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    *(_QWORD *)(*(_QWORD *)(v127 + 32) + 40LL),
                                    *(_QWORD *)(*(_QWORD *)(v127 + 32) + 48LL),
                                    *(_OWORD *)*(_QWORD *)(v127 + 32));
                result = (__int64)sub_1D332F0(
                                    v88,
                                    52,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    v10,
                                    v11,
                                    v89);
                goto LABEL_9;
              }
              v54 = *(_WORD *)(v127 + 24);
            }
            if ( v54 == 54 && sub_1D18C00(v127, 1, v126) )
            {
              v80 = *(__int64 **)(v127 + 32);
              v81 = *v80;
              if ( *(_WORD *)(*v80 + 24) == 53 )
              {
                if ( (unsigned __int8)sub_1F6D200(
                                        **(_QWORD **)(v81 + 32),
                                        *(_QWORD *)(*(_QWORD *)(v81 + 32) + 8LL),
                                        v81,
                                        v77,
                                        v78,
                                        v79) )
                {
                  *(_QWORD *)&v83 = sub_1D332F0(
                                      *a1,
                                      54,
                                      (__int64)&v133,
                                      v131,
                                      v132,
                                      0,
                                      *(double *)v12.m128i_i64,
                                      *(double *)a4.m128i_i64,
                                      a5,
                                      *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v127 + 32) + 32LL) + 40LL),
                                      *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v127 + 32) + 32LL) + 48LL),
                                      *(_OWORD *)(*(_QWORD *)(v127 + 32) + 40LL));
                  goto LABEL_116;
                }
                v80 = *(__int64 **)(v127 + 32);
              }
              v82 = v80[5];
              if ( *(_WORD *)(v82 + 24) == 53
                && (unsigned __int8)sub_1F6D200(
                                      **(_QWORD **)(v82 + 32),
                                      *(_QWORD *)(*(_QWORD *)(v82 + 32) + 8LL),
                                      v81,
                                      v77,
                                      v78,
                                      v79) )
              {
                *(_QWORD *)&v83 = sub_1D332F0(
                                    *a1,
                                    54,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    **(_QWORD **)(v127 + 32),
                                    *(_QWORD *)(*(_QWORD *)(v127 + 32) + 8LL),
                                    *(_OWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v127 + 32) + 40LL) + 32LL) + 40LL));
LABEL_116:
                v84 = *a1;
LABEL_117:
                result = (__int64)sub_1D332F0(
                                    v84,
                                    52,
                                    (__int64)&v133,
                                    v131,
                                    v132,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)a4.m128i_i64,
                                    a5,
                                    v10,
                                    v11,
                                    v83);
                goto LABEL_9;
              }
            }
            if ( *(_WORD *)(v9 + 24) == 48 )
              goto LABEL_15;
            result = v127;
            if ( *(_WORD *)(v127 + 24) == 48 )
              goto LABEL_9;
            result = (__int64)sub_1F787C0(a2, *a1, v12, *(double *)a4.m128i_i64, a5);
            if ( result )
              goto LABEL_9;
            result = (__int64)sub_1F7E7D0(
                                a2,
                                *a1,
                                v66,
                                v67,
                                v68,
                                v69,
                                *(double *)v12.m128i_i64,
                                *(double *)a4.m128i_i64,
                                a5);
            if ( result )
              goto LABEL_9;
            if ( sub_1F6C880((__int64)a1[1], 0x79u, v131) && *(_WORD *)(v9 + 24) == 120 && *(_WORD *)(v127 + 24) == 123 )
            {
              v92 = *(__int64 **)(v9 + 32);
              v100 = *(const __m128i **)(v127 + 32);
              v93 = *v92;
              v94 = *((unsigned int *)v92 + 2);
              v95 = v92[5];
              v96 = v100->m128i_i64[0];
              v97 = v100->m128i_u32[2];
              a5 = _mm_loadu_si128(v100);
              v98 = *((_DWORD *)v92 + 12);
              v99 = v93 == v100->m128i_i64[0];
              LOBYTE(v100) = (_DWORD)v94 == (_DWORD)v97;
              if ( (_DWORD)v94 == (_DWORD)v97 && v99 && v127 == v95 && v98 == v126 )
                goto LABEL_167;
              if ( v127 == v93 )
              {
                LOBYTE(v96) = v95 == v96;
                LOBYTE(v94) = (_DWORD)v94 == v126;
                if ( ((unsigned __int8)v94 & (unsigned __int8)v96) != 0 && v98 == (_DWORD)v97 )
                {
LABEL_167:
                  v121 = sub_1D159C0((__int64)&v131, v95, v94, v96, (__int64)v100, v97);
                  v105 = sub_1D1ADA0(
                           *(_QWORD *)(*(_QWORD *)(v127 + 32) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v127 + 32) + 48LL),
                           v101,
                           v102,
                           v103,
                           v104);
                  if ( v105 )
                  {
                    if ( sub_13A38F0(*(_QWORD *)(v105 + 88) + 24LL, (_QWORD *)(unsigned int)(v121 - 1)) )
                    {
                      v106 = *a1;
                      v137 = *(_QWORD *)(a2 + 72);
                      if ( v137 )
                        sub_1F6CA20((__int64 *)&v137);
                      v138 = *(_DWORD *)(a2 + 64);
                      v124 = sub_1D309E0(
                               v106,
                               121,
                               (__int64)&v137,
                               v131,
                               v132,
                               0,
                               *(double *)v12.m128i_i64,
                               *(double *)a4.m128i_i64,
                               *(double *)a5.m128i_i64,
                               *(_OWORD *)&a5);
                      sub_17CD270((__int64 *)&v137);
                      result = v124;
                      goto LABEL_9;
                    }
                  }
                }
              }
            }
            v70 = sub_1F6CCC0(v9);
            v71 = v70;
            if ( v70 )
            {
              if ( *((_BYTE *)a1 + 24) )
              {
                if ( *(_WORD *)(v127 + 24) != 148 )
                  goto LABEL_97;
                goto LABEL_140;
              }
              if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, __int64))(*a1[1] + 1048LL))(a1[1], v70) )
              {
                v72 = sub_1F6CCC0(v127);
                if ( v72 )
                {
                  if ( *(_QWORD *)(v72 + 88) == *(_QWORD *)(v71 + 88) )
                  {
                    result = sub_1D38BB0(
                               (__int64)*a1,
                               *(_QWORD *)(v71 + 96) - *(_QWORD *)(v72 + 96),
                               (__int64)&v133,
                               v131,
                               v132,
                               0,
                               v12,
                               *(double *)a4.m128i_i64,
                               a5,
                               0);
                    goto LABEL_9;
                  }
                }
              }
            }
            v73 = *(unsigned __int16 *)(v127 + 24);
            if ( v73 != 148 )
            {
              if ( *((_BYTE *)a1 + 24) != 1 && v73 == 124 && sub_1D18C00(v127, 1, v126) )
              {
                v125 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v127 + 32) + 40LL));
                v110 = sub_1D1ADA0(v125.m128i_i64[0], v125.m128i_u32[2], v125.m128i_i64[1], v107, v108, v109);
                if ( v110 )
                {
                  v111 = *(_QWORD *)(v110 + 88);
                  v112 = *(_QWORD **)(v111 + 24);
                  if ( *(_DWORD *)(v111 + 32) > 0x40u )
                    v112 = (_QWORD *)*v112;
                  if ( (_QWORD *)((unsigned int)sub_1F701D0(v127, v126) - 1) == v112 )
                  {
                    *(_QWORD *)&v83 = sub_1D332F0(
                                        *a1,
                                        123,
                                        (__int64)&v133,
                                        v131,
                                        v132,
                                        0,
                                        *(double *)v12.m128i_i64,
                                        *(double *)a4.m128i_i64,
                                        a5,
                                        **(_QWORD **)(v127 + 32),
                                        *(_QWORD *)(*(_QWORD *)(v127 + 32) + 8LL),
                                        *(_OWORD *)&v125);
                    goto LABEL_116;
                  }
                }
              }
LABEL_97:
              result = 0;
              goto LABEL_9;
            }
LABEL_140:
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v127 + 32) + 40LL) + 88LL) == 2 )
            {
              v90 = *a1;
              *(_QWORD *)&v91 = sub_1D38BB0(
                                  (__int64)*a1,
                                  1,
                                  (__int64)&v133,
                                  v131,
                                  v132,
                                  0,
                                  v12,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  0);
              *(_QWORD *)&v83 = sub_1D332F0(
                                  v90,
                                  118,
                                  (__int64)&v133,
                                  v131,
                                  v132,
                                  0,
                                  *(double *)v12.m128i_i64,
                                  *(double *)a4.m128i_i64,
                                  a5,
                                  **(_QWORD **)(v127 + 32),
                                  *(_QWORD *)(*(_QWORD *)(v127 + 32) + 8LL),
                                  v91);
              v84 = *a1;
              goto LABEL_117;
            }
            goto LABEL_97;
          }
        }
        v61 = *(_QWORD *)(v9 + 32);
        if ( v127 != *(_QWORD *)v61 || *(_DWORD *)(v61 + 8) != v126 )
        {
          v62 = *(_QWORD *)(v61 + 40);
          if ( v127 == v62 && *(_DWORD *)(v61 + 48) == v126 )
          {
            result = *(_QWORD *)v61;
            goto LABEL_9;
          }
          if ( v50 != 52 )
            goto LABEL_79;
          goto LABEL_106;
        }
LABEL_120:
        result = *(_QWORD *)(v61 + 40);
        goto LABEL_9;
      }
    }
    else
    {
      v135 = 0;
    }
    v37 = v135 | v36;
    goto LABEL_30;
  }
LABEL_9:
  if ( v133 )
  {
    v128 = result;
    sub_161E7C0((__int64)&v133, v133);
    return v128;
  }
  return result;
}
