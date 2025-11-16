// Function: sub_1FC08F0
// Address: 0x1fc08f0
//
__int64 __fastcall sub_1FC08F0(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // r13
  unsigned int v11; // r14d
  __m128i v12; // xmm1
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int8 v15; // r12
  const void **v16; // rax
  __int64 v17; // r12
  __int64 *v18; // rax
  __int16 v20; // ax
  __int64 v21; // r9
  __int64 v22; // rdx
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rsi
  int v27; // edi
  __int64 v28; // rdx
  __int64 *v29; // rcx
  int v30; // r8d
  int v31; // r9d
  __int64 *v32; // rax
  __int64 *v33; // rax
  int v34; // edx
  int v35; // edx
  __int128 v36; // rax
  __int64 *v37; // r12
  __int128 *v38; // rbx
  __int64 *v39; // rax
  unsigned __int64 v40; // rdx
  const __m128i *v41; // roff
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // r8
  const void ***v45; // rax
  __int128 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r13
  __int64 *v56; // r12
  __int64 *v57; // rax
  unsigned __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rdx
  int v63; // edx
  __int64 v64; // rax
  __int64 *v65; // rdi
  __int64 *v66; // r12
  __int128 *v67; // rbx
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  int v70; // [rsp-10h] [rbp-100h]
  __int128 v71; // [rsp-10h] [rbp-100h]
  int v72; // [rsp-8h] [rbp-F8h]
  __int128 v73; // [rsp+0h] [rbp-F0h]
  unsigned int v74; // [rsp+10h] [rbp-E0h]
  __m128i v75; // [rsp+10h] [rbp-E0h]
  __int64 v76; // [rsp+28h] [rbp-C8h]
  int v77; // [rsp+30h] [rbp-C0h]
  __m128i v78; // [rsp+30h] [rbp-C0h]
  __int64 *v79; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v80; // [rsp+60h] [rbp-90h]
  __m128i v81; // [rsp+60h] [rbp-90h]
  __int64 *v82; // [rsp+60h] [rbp-90h]
  unsigned int v83; // [rsp+70h] [rbp-80h] BYREF
  const void **v84; // [rsp+78h] [rbp-78h]
  __int64 v85; // [rsp+80h] [rbp-70h] BYREF
  int v86; // [rsp+88h] [rbp-68h]
  __int64 v87[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 **v88; // [rsp+A0h] [rbp-50h] BYREF
  int v89; // [rsp+A8h] [rbp-48h]
  char v90; // [rsp+ACh] [rbp-44h]
  __int64 v91; // [rsp+B0h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = *(_QWORD *)v7;
  v11 = *(_DWORD *)(v7 + 8);
  v12 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v13 = *(_QWORD *)(v7 + 40);
  v77 = *(_DWORD *)(v7 + 48);
  v14 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v11;
  v80 = v13;
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  v85 = v8;
  LOBYTE(v83) = v15;
  v84 = v16;
  if ( v8 )
    sub_1623A60((__int64)&v85, v8, 2);
  v86 = *(_DWORD *)(a2 + 64);
  if ( v15 )
  {
    if ( (unsigned __int8)(v15 - 14) > 0x5Fu )
      goto LABEL_5;
  }
  else if ( !sub_1F58D20((__int64)&v83) )
  {
    goto LABEL_5;
  }
  v18 = sub_1FA8C50((__int64)a1, a2, *(double *)v9.m128i_i64, *(double *)v12.m128i_i64, a5);
  if ( v18 )
    goto LABEL_12;
  if ( (unsigned __int8)sub_1D16620(v80, (__int64 *)a2) )
    goto LABEL_14;
  if ( (unsigned __int8)sub_1D16620(v10, (__int64 *)a2) )
  {
LABEL_20:
    v17 = v12.m128i_i64[0];
    goto LABEL_15;
  }
LABEL_5:
  if ( *(_WORD *)(v10 + 24) == 48 )
    goto LABEL_14;
  if ( *(_WORD *)(v80 + 24) == 48 )
    goto LABEL_20;
  if ( sub_1D23600((__int64)*a1, v9.m128i_i64[0]) )
  {
    if ( sub_1D23600((__int64)*a1, v12.m128i_i64[0]) )
    {
      v17 = (__int64)sub_1D32920(
                       *a1,
                       0x34u,
                       (__int64)&v85,
                       v83,
                       (__int64)v84,
                       v10,
                       *(double *)v9.m128i_i64,
                       *(double *)v12.m128i_i64,
                       a5,
                       v80);
      goto LABEL_15;
    }
    v32 = sub_1D332F0(
            *a1,
            52,
            (__int64)&v85,
            v83,
            v84,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v12.m128i_i64,
            a5,
            v12.m128i_i64[0],
            v12.m128i_u64[1],
            *(_OWORD *)&v9);
    goto LABEL_35;
  }
  if ( sub_1D185B0(v12.m128i_i64[0]) )
  {
LABEL_14:
    v17 = v9.m128i_i64[0];
    goto LABEL_15;
  }
  if ( (unsigned __int8)sub_1F70310(v12.m128i_i64[0], v12.m128i_u32[2], 1u) )
  {
    v20 = *(_WORD *)(v10 + 24);
    if ( v20 == 53 )
    {
      if ( (unsigned __int8)sub_1F70310(**(_QWORD **)(v10 + 32), *(_QWORD *)(*(_QWORD *)(v10 + 32) + 8LL), 1u) )
      {
        v37 = *a1;
        v38 = *(__int128 **)(v10 + 32);
        v39 = sub_1D332F0(
                *a1,
                52,
                (__int64)&v85,
                v83,
                v84,
                0,
                *(double *)v9.m128i_i64,
                *(double *)v12.m128i_i64,
                a5,
                v12.m128i_i64[0],
                v12.m128i_u64[1],
                *v38);
        v17 = (__int64)sub_1D332F0(
                         v37,
                         53,
                         (__int64)&v85,
                         v83,
                         v84,
                         0,
                         *(double *)v9.m128i_i64,
                         *(double *)v12.m128i_i64,
                         a5,
                         (__int64)v39,
                         v40,
                         *(__int128 *)((char *)v38 + 40));
        goto LABEL_15;
      }
      v20 = *(_WORD *)(v10 + 24);
    }
    if ( v20 == 142 )
    {
      if ( sub_1D18C00(v10, 1, v11) )
      {
        if ( (unsigned __int8)sub_1F706D0(v12.m128i_i64[0], v12.m128i_u32[2]) )
        {
          v41 = *(const __m128i **)(v10 + 32);
          a5 = _mm_loadu_si128(v41);
          v42 = v41->m128i_i64[0];
          v43 = v41->m128i_u32[2];
          v76 = v41->m128i_i64[0];
          v74 = v41->m128i_u32[2];
          if ( (!*((_BYTE *)a1 + 24)
             || sub_1F6C830((__int64)a1[1], 0x78u, *(_BYTE *)(*(_QWORD *)(v42 + 40) + 16 * v43))
             && sub_1F6C830(v44, 0x8Fu, v15))
            && (unsigned int)sub_1F701D0(v76, v74) == 1 )
          {
            v45 = (const void ***)(*(_QWORD *)(v76 + 40) + 16LL * v74);
            *(_QWORD *)&v46 = sub_1D3C080(
                                *a1,
                                (__int64)&v85,
                                a5.m128i_i64[0],
                                a5.m128i_u64[1],
                                *(unsigned __int8 *)v45,
                                v45[1],
                                v9,
                                *(double *)v12.m128i_i64,
                                a5);
            v17 = sub_1D309E0(
                    *a1,
                    143,
                    (__int64)&v85,
                    v83,
                    v84,
                    0,
                    *(double *)v9.m128i_i64,
                    *(double *)v12.m128i_i64,
                    *(double *)a5.m128i_i64,
                    v46);
            goto LABEL_15;
          }
        }
      }
      v20 = *(_WORD *)(v10 + 24);
    }
    if ( v20 == 119 )
    {
      v33 = *(__int64 **)(v10 + 32);
      v34 = *(unsigned __int16 *)(*v33 + 24);
      if ( v34 == 14 || v34 == 36 )
      {
        v35 = *(unsigned __int16 *)(v33[5] + 24);
        if ( v35 == 10 || v35 == 32 )
        {
          if ( (unsigned __int8)sub_1D206E0((__int64)*a1, *v33, v33[1], v33[5], v33[6]) )
          {
            *(_QWORD *)&v36 = sub_1D332F0(
                                *a1,
                                52,
                                (__int64)&v85,
                                v83,
                                v84,
                                0,
                                *(double *)v9.m128i_i64,
                                *(double *)v12.m128i_i64,
                                a5,
                                v12.m128i_i64[0],
                                v12.m128i_u64[1],
                                *(_OWORD *)(*(_QWORD *)(v10 + 32) + 40LL));
            v32 = sub_1D332F0(
                    *a1,
                    52,
                    (__int64)&v85,
                    v83,
                    v84,
                    0,
                    *(double *)v9.m128i_i64,
                    *(double *)v12.m128i_i64,
                    a5,
                    **(_QWORD **)(v10 + 32),
                    *(_QWORD *)(*(_QWORD *)(v10 + 32) + 8LL),
                    v36);
LABEL_35:
            v17 = (__int64)v32;
            goto LABEL_15;
          }
        }
      }
    }
  }
  v18 = sub_1F77C50(a1, a2, *(double *)v9.m128i_i64, *(double *)v12.m128i_i64, a5);
  if ( v18
    || (v18 = sub_1F82ED0(
                (__int64 *)a1,
                0x34u,
                (__int64)&v85,
                v9.m128i_i64[0],
                v9.m128i_u64[1],
                *(double *)v9.m128i_i64,
                *(double *)v12.m128i_i64,
                a5,
                v21,
                *(_OWORD *)&v12),
        v24 = v70,
        v25 = v72,
        v18) )
  {
LABEL_12:
    v17 = (__int64)v18;
    goto LABEL_15;
  }
  if ( *(_WORD *)(v10 + 24) == 53
    && (unsigned __int8)sub_1F6D200(
                          **(_QWORD **)(v10 + 32),
                          *(_QWORD *)(*(_QWORD *)(v10 + 32) + 8LL),
                          v22,
                          v23,
                          v70,
                          v72) )
  {
    v61 = sub_1D332F0(
            *a1,
            53,
            (__int64)&v85,
            v83,
            v84,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v12.m128i_i64,
            a5,
            v12.m128i_i64[0],
            v12.m128i_u64[1],
            *(_OWORD *)(*(_QWORD *)(v10 + 32) + 40LL));
    goto LABEL_71;
  }
  if ( *(_WORD *)(v80 + 24) == 53
    && (unsigned __int8)sub_1F6D200(
                          **(_QWORD **)(v80 + 32),
                          *(_QWORD *)(*(_QWORD *)(v80 + 32) + 8LL),
                          v22,
                          v23,
                          v24,
                          v25) )
  {
    v61 = sub_1D332F0(
            *a1,
            53,
            (__int64)&v85,
            v83,
            v84,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v12.m128i_i64,
            a5,
            v9.m128i_i64[0],
            v9.m128i_u64[1],
            *(_OWORD *)(*(_QWORD *)(v80 + 32) + 40LL));
LABEL_71:
    v17 = (__int64)v61;
    goto LABEL_15;
  }
  v26 = *(unsigned __int16 *)(v80 + 24);
  if ( (_DWORD)v26 == 53 )
  {
    v47 = *(_QWORD *)(v80 + 32);
    if ( *(_QWORD *)(v47 + 40) == v10 && *(_DWORD *)(v47 + 48) == v11 )
    {
      v17 = *(_QWORD *)v47;
      goto LABEL_15;
    }
    v27 = *(unsigned __int16 *)(v10 + 24);
    if ( v27 == 53 )
    {
      v59 = *(_QWORD *)(v10 + 32);
      if ( *(_QWORD *)(v59 + 40) == v80 && v77 == *(_DWORD *)(v59 + 48) )
        goto LABEL_64;
    }
    v48 = *(_QWORD *)(v47 + 40);
    v49 = *(_QWORD *)(v80 + 32);
    if ( *(_WORD *)(v48 + 24) == 52 )
    {
      v62 = *(_QWORD *)(v48 + 32);
      if ( *(_QWORD *)v62 == v10 && *(_DWORD *)(v62 + 8) == v11 )
      {
        v61 = sub_1D332F0(
                *a1,
                53,
                (__int64)&v85,
                v83,
                v84,
                0,
                *(double *)v9.m128i_i64,
                *(double *)v12.m128i_i64,
                a5,
                *(_QWORD *)v47,
                *(_QWORD *)(v47 + 8),
                *(_OWORD *)(v62 + 40));
        goto LABEL_71;
      }
      if ( *(_QWORD *)(v62 + 40) == v10 && *(_DWORD *)(v62 + 48) == v11 )
      {
        v61 = sub_1D332F0(
                *a1,
                53,
                (__int64)&v85,
                v83,
                v84,
                0,
                *(double *)v9.m128i_i64,
                *(double *)v12.m128i_i64,
                a5,
                *(_QWORD *)v47,
                *(_QWORD *)(v47 + 8),
                *(_OWORD *)v62);
        goto LABEL_71;
      }
    }
    v50 = *(_QWORD *)v47;
    if ( *(_WORD *)(v50 + 24) != 53 )
      goto LABEL_57;
    v60 = *(_QWORD *)(v50 + 32);
    if ( *(_QWORD *)(v60 + 40) != v10 )
      goto LABEL_57;
  }
  else
  {
    v27 = *(unsigned __int16 *)(v10 + 24);
    if ( v27 == 53 )
    {
      v59 = *(_QWORD *)(v10 + 32);
      if ( *(_QWORD *)(v59 + 40) == v80 && v77 == *(_DWORD *)(v59 + 48) )
      {
LABEL_64:
        v17 = *(_QWORD *)v59;
        goto LABEL_15;
      }
    }
    if ( (_DWORD)v26 != 52 )
      goto LABEL_32;
    v49 = *(_QWORD *)(v80 + 32);
    if ( *(_WORD *)(*(_QWORD *)v49 + 24LL) != 53 )
      goto LABEL_32;
    v60 = *(_QWORD *)(*(_QWORD *)v49 + 32LL);
    if ( *(_QWORD *)(v60 + 40) != v10 )
      goto LABEL_32;
  }
  if ( *(_DWORD *)(v60 + 48) == v11 )
  {
    v17 = (__int64)sub_1D332F0(
                     *a1,
                     v26,
                     (__int64)&v85,
                     v83,
                     v84,
                     0,
                     *(double *)v9.m128i_i64,
                     *(double *)v12.m128i_i64,
                     a5,
                     *(_QWORD *)v60,
                     *(_QWORD *)(v60 + 8),
                     *(_OWORD *)(v49 + 40));
    goto LABEL_15;
  }
LABEL_57:
  if ( (_DWORD)v26 == 53 && v27 == 53 )
  {
    v51 = *(_QWORD *)(v10 + 32);
    v78 = _mm_loadu_si128((const __m128i *)v49);
    v73 = (__int128)_mm_loadu_si128((const __m128i *)(v49 + 40));
    v81 = _mm_loadu_si128((const __m128i *)v51);
    v75 = _mm_loadu_si128((const __m128i *)(v51 + 40));
    if ( (unsigned __int8)sub_1F70310(v81.m128i_i64[0], v81.m128i_u32[2], 0)
      || (unsigned __int8)sub_1F70310(v78.m128i_i64[0], v78.m128i_u32[2], 0) )
    {
      v52 = *a1;
      sub_1F80610((__int64)&v88, v12.m128i_i64[0]);
      v53 = sub_1D332F0(
              v52,
              52,
              (__int64)&v88,
              v83,
              v84,
              0,
              *(double *)v9.m128i_i64,
              *(double *)v12.m128i_i64,
              a5,
              v75.m128i_i64[0],
              v75.m128i_u64[1],
              v73);
      v55 = v54;
      v56 = v53;
      v79 = *a1;
      sub_1F80610((__int64)v87, v9.m128i_i64[0]);
      v57 = sub_1D332F0(
              v79,
              52,
              (__int64)v87,
              v83,
              v84,
              0,
              *(double *)v9.m128i_i64,
              *(double *)v12.m128i_i64,
              a5,
              v81.m128i_i64[0],
              v81.m128i_u64[1],
              *(_OWORD *)&v78);
      *((_QWORD *)&v71 + 1) = v55;
      *(_QWORD *)&v71 = v56;
      v17 = (__int64)sub_1D332F0(
                       v52,
                       53,
                       (__int64)&v85,
                       v83,
                       v84,
                       0,
                       *(double *)v9.m128i_i64,
                       *(double *)v12.m128i_i64,
                       a5,
                       (__int64)v57,
                       v58,
                       v71);
      sub_17CD270(v87);
      sub_17CD270((__int64 *)&v88);
      goto LABEL_15;
    }
  }
LABEL_32:
  v29 = sub_1F787C0(a2, *a1, v9, *(double *)v12.m128i_i64, a5);
  if ( v29 )
    goto LABEL_33;
  v29 = sub_1F7E7D0(a2, *a1, v28, 0, v30, v31, *(double *)v9.m128i_i64, *(double *)v12.m128i_i64, a5);
  if ( v29 )
    goto LABEL_33;
  if ( (unsigned __int8)sub_1FB1D70((__int64)a1, a2, 0) )
  {
    v17 = a2;
    goto LABEL_15;
  }
  v88 = *(__int64 ***)(a2 + 72);
  if ( v88 )
    sub_1F6CA20((__int64 *)&v88);
  v89 = *(_DWORD *)(a2 + 64);
  v82 = sub_1F806E0(
          (__int64)a1,
          v9.m128i_i64[0],
          v9.m128i_u32[2],
          v12.m128i_i64[0],
          v12.m128i_u32[2],
          (__int64)&v88,
          v9,
          *(double *)v12.m128i_i64,
          a5,
          1);
  sub_17CD270((__int64 *)&v88);
  if ( v82 )
  {
    v17 = (__int64)v82;
    goto LABEL_15;
  }
  if ( (!*((_BYTE *)a1 + 24) || sub_1F6C830((__int64)a1[1], 0x77u, v15))
    && (unsigned __int8)sub_1D206E0((__int64)*a1, v9.m128i_i64[0], v9.m128i_i64[1], v12.m128i_i64[0], v12.m128i_i64[1]) )
  {
    v17 = (__int64)sub_1D332F0(
                     *a1,
                     119,
                     (__int64)&v85,
                     v83,
                     v84,
                     0,
                     *(double *)v9.m128i_i64,
                     *(double *)v12.m128i_i64,
                     a5,
                     v9.m128i_i64[0],
                     v9.m128i_u64[1],
                     *(_OWORD *)&v12);
    goto LABEL_15;
  }
  if ( sub_1D18970(v9.m128i_i64[0]) && (unsigned __int8)sub_1F706D0(v12.m128i_i64[0], v12.m128i_u32[2]) )
  {
    v66 = *a1;
    v67 = *(__int128 **)(v10 + 32);
    v68 = sub_1D38BB0((__int64)v66, 0, (__int64)&v85, v83, v84, 0, v9, *(double *)v12.m128i_i64, a5, 0);
    v17 = (__int64)sub_1D332F0(
                     v66,
                     53,
                     (__int64)&v85,
                     v83,
                     v84,
                     0,
                     *(double *)v9.m128i_i64,
                     *(double *)v12.m128i_i64,
                     a5,
                     v68,
                     v69,
                     *v67);
    goto LABEL_15;
  }
  v29 = sub_1F7EA80(
          (__int64)a1,
          v9.m128i_i64[0],
          v9.m128i_u64[1],
          v12.m128i_i64[0],
          v12.m128i_u64[1],
          a2,
          v9,
          *(double *)v12.m128i_i64,
          a5);
  if ( v29 )
  {
LABEL_33:
    v17 = (__int64)v29;
    goto LABEL_15;
  }
  v17 = (__int64)sub_1F7EA80(
                   (__int64)a1,
                   v12.m128i_i64[0],
                   v12.m128i_u64[1],
                   v9.m128i_i64[0],
                   v9.m128i_u64[1],
                   a2,
                   v9,
                   *(double *)v12.m128i_i64,
                   a5);
  if ( !v17 )
  {
    v63 = *((_DWORD *)a1 + 4);
    v64 = (__int64)*a1;
    v88 = a1;
    v65 = a1[1];
    v90 = 0;
    v89 = v63;
    v91 = v64;
    v18 = (__int64 *)(*(__int64 (__fastcall **)(__int64 *, __int64, __int64 ***))(*v65 + 1112))(v65, a2, &v88);
    goto LABEL_12;
  }
LABEL_15:
  if ( v85 )
    sub_161E7C0((__int64)&v85, v85);
  return v17;
}
