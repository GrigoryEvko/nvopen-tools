// Function: sub_1784060
// Address: 0x1784060
//
__int64 __fastcall sub_1784060(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  unsigned __int8 *v12; // rsi
  __m128 v13; // xmm0
  __m128i v14; // xmm1
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // r13
  __int64 **v30; // rbx
  __int64 v31; // rsi
  char v32; // al
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rcx
  unsigned __int8 v39; // al
  unsigned int v40; // r12d
  __int64 v41; // rax
  __int64 v42; // r12
  _QWORD *v43; // r13
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // r14
  __int64 v50; // rax
  int v51; // eax
  int v52; // eax
  __int64 **v53; // r10
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rsi
  unsigned __int8 *v58; // r12
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  _QWORD *v62; // rax
  unsigned int v63; // r12d
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // [rsp-10h] [rbp-A0h]
  int v67; // [rsp+0h] [rbp-90h]
  int v68; // [rsp+4h] [rbp-8Ch]
  __int64 v69; // [rsp+8h] [rbp-88h]
  __int64 v70; // [rsp+10h] [rbp-80h] BYREF
  __int16 v71; // [rsp+20h] [rbp-70h]
  __m128 v72; // [rsp+30h] [rbp-60h] BYREF
  __m128i v73; // [rsp+40h] [rbp-50h]
  __int64 v74; // [rsp+50h] [rbp-40h]

  v11 = a2;
  v74 = a2;
  v12 = *(unsigned __int8 **)(a2 - 24);
  v13 = (__m128)_mm_loadu_si128(a1 + 167);
  v14 = _mm_loadu_si128(a1 + 168);
  v15 = *(_QWORD **)(v11 - 48);
  v72 = v13;
  v73 = v14;
  v16 = sub_13E0AC0(v15, v12, (__int64 *)&v72);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v11 + 8);
    if ( v17 )
    {
      v18 = a1->m128i_i64[0];
      v19 = v16;
      do
      {
        v20 = sub_1648700(v17);
        sub_170B990(v18, (__int64)v20);
        v17 = *(_QWORD *)(v17 + 8);
      }
      while ( v17 );
      if ( v11 == v19 )
        v19 = sub_1599EF0(*(__int64 ***)v11);
      sub_164D160(v11, v19, v13, *(double *)v14.m128i_i64, a5, a6, v21, v22, a9, a10);
      return v11;
    }
    return 0;
  }
  v24 = (__int64)sub_1707490((__int64)a1, (unsigned __int8 *)v11, *(double *)v13.m128_u64, *(double *)v14.m128i_i64, a5);
  if ( v24 )
    return v24;
  v24 = sub_1783D50(a1->m128i_i64, (_BYTE *)v11, v25, v26, v13, *(double *)v14.m128i_i64, a5, a6, v27, v28, a9, a10);
  if ( v24 )
    return v24;
  v24 = (__int64)sub_1780430(v11, a1->m128i_i64[1], *(double *)v13.m128_u64, *(double *)v14.m128i_i64, a5);
  if ( v24 )
    return v24;
  v29 = *(_QWORD *)(v11 - 24);
  v30 = *(__int64 ***)v11;
  v31 = a1[166].m128i_i64[1];
  v69 = *(_QWORD *)(v11 - 48);
  v32 = sub_14BDFF0(v29, v31, 1u, 0, a1[165].m128i_i64[0], v11, a1[166].m128i_i64[0]);
  v33 = v66;
  if ( v32 )
  {
    v34 = sub_15A04A0(v30);
    v35 = a1->m128i_i64[1];
    v73.m128i_i16[0] = 257;
    if ( *(_BYTE *)(v29 + 16) > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
    {
      v36 = (__int64)sub_170A2B0(v35, 11, (__int64 *)v29, v34, (__int64 *)&v72, 0, 0);
    }
    else
    {
      v36 = sub_15A2B30((__int64 *)v29, v34, 0, 0, *(double *)v13.m128_u64, *(double *)v14.m128i_i64, a5);
      v37 = sub_14DBA30(v36, *(_QWORD *)(v35 + 96), 0);
      if ( v37 )
        v36 = v37;
    }
    v73.m128i_i16[0] = 257;
    return sub_15FB440(26, (__int64 *)v69, v36, (__int64)&v72, 0);
  }
  v38 = v69;
  v39 = *(_BYTE *)(v69 + 16);
  if ( v39 == 13 )
  {
    v40 = *(_DWORD *)(v69 + 32);
    v41 = v69;
    if ( v40 > 0x40 )
      goto LABEL_22;
    if ( *(_QWORD *)(v69 + 24) != 1 )
      goto LABEL_29;
LABEL_23:
    v42 = a1->m128i_i64[1];
    v73.m128i_i16[0] = 257;
    v71 = 257;
    if ( *(_BYTE *)(v29 + 16) > 0x10u || *(_BYTE *)(v69 + 16) > 0x10u )
    {
      v43 = sub_177F2B0(v42, 33, v29, v69, &v70);
    }
    else
    {
      v43 = (_QWORD *)sub_15A37B0(0x21u, (_QWORD *)v29, (_QWORD *)v69, 0);
      v44 = sub_14DBA30((__int64)v43, *(_QWORD *)(v42 + 96), 0);
      if ( v44 )
        v43 = (_QWORD *)v44;
    }
    return sub_15FDE70(v43, (__int64)v30, (__int64)&v72, 0);
  }
  v38 = v69;
  v33 = *(_QWORD *)v69;
  if ( *(_BYTE *)(*(_QWORD *)v69 + 8LL) != 16 || v39 > 0x10u )
    goto LABEL_29;
  v41 = sub_15A1020((_BYTE *)v69, v31, v33, v69);
  if ( !v41 || *(_BYTE *)(v41 + 16) != 13 )
  {
    v63 = 0;
    v68 = *(_DWORD *)(*(_QWORD *)v69 + 32LL);
    while ( v68 != v63 )
    {
      v31 = v63;
      v64 = sub_15A0A60(v69, v63);
      if ( !v64 )
        goto LABEL_29;
      v38 = *(unsigned __int8 *)(v64 + 16);
      if ( (_BYTE)v38 != 9 )
      {
        if ( (_BYTE)v38 != 13 )
          goto LABEL_29;
        v38 = *(unsigned int *)(v64 + 32);
        if ( (unsigned int)v38 <= 0x40 )
        {
          if ( *(_QWORD *)(v64 + 24) != 1 )
            goto LABEL_29;
        }
        else
        {
          v67 = *(_DWORD *)(v64 + 32);
          v65 = sub_16A57B0(v64 + 24);
          v38 = (unsigned int)(v67 - 1);
          if ( v65 != (_DWORD)v38 )
            goto LABEL_29;
        }
      }
      ++v63;
    }
    goto LABEL_23;
  }
  v40 = *(_DWORD *)(v41 + 32);
  if ( v40 <= 0x40 )
  {
    if ( *(_QWORD *)(v41 + 24) != 1 )
      goto LABEL_29;
    goto LABEL_23;
  }
LABEL_22:
  if ( (unsigned int)sub_16A57B0(v41 + 24) == v40 - 1 )
    goto LABEL_23;
LABEL_29:
  if ( (unsigned __int8)sub_177F910((unsigned int *)v29, v31, v33, v38) )
  {
    v45 = a1->m128i_i64[1];
    v73.m128i_i16[0] = 257;
    if ( *(_BYTE *)(v69 + 16) > 0x10u || *(_BYTE *)(v29 + 16) > 0x10u )
    {
      v46 = (__int64)sub_177F2B0(v45, 36, v69, v29, (__int64 *)&v72);
    }
    else
    {
      v46 = sub_15A37B0(0x24u, (_QWORD *)v69, (_QWORD *)v29, 0);
      v47 = sub_14DBA30(v46, *(_QWORD *)(v45 + 96), 0);
      if ( v47 )
        v46 = v47;
    }
    v48 = a1->m128i_i64[1];
    v73.m128i_i16[0] = 257;
    if ( *(_BYTE *)(v69 + 16) > 0x10u || *(_BYTE *)(v29 + 16) > 0x10u )
    {
      v49 = (__int64)sub_170A2B0(v48, 13, (__int64 *)v69, v29, (__int64 *)&v72, 0, 0);
    }
    else
    {
      v49 = sub_15A2B60((__int64 *)v69, v29, 0, 0, *(double *)v13.m128_u64, *(double *)v14.m128i_i64, a5);
      v50 = sub_14DBA30(v49, *(_QWORD *)(v48 + 96), 0);
      if ( v50 )
        v49 = v50;
    }
    v73.m128i_i16[0] = 257;
    return sub_14EDD70(v46, (_QWORD *)v69, v49, (__int64)&v72, 0, 0);
  }
  else
  {
    v51 = *(unsigned __int8 *)(v29 + 16);
    if ( (unsigned __int8)v51 > 0x17u )
    {
      v52 = v51 - 24;
    }
    else
    {
      if ( (_BYTE)v51 != 5 )
        return 0;
      v52 = *(unsigned __int16 *)(v29 + 18);
    }
    if ( v52 != 38 )
      return 0;
    v53 = (*(_BYTE *)(v29 + 23) & 0x40) != 0
        ? *(__int64 ***)(v29 - 8)
        : (__int64 **)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
    if ( !*v53 || !sub_17287D0(**v53, 1) )
      return 0;
    v73.m128i_i16[0] = 257;
    v54 = a1->m128i_i64[1];
    v55 = sub_15A04A0(v30);
    if ( *(_BYTE *)(v69 + 16) > 0x10u || *(_BYTE *)(v55 + 16) > 0x10u )
    {
      v57 = 32;
      v58 = sub_177F2B0(v54, 32, v69, v55, (__int64 *)&v72);
    }
    else
    {
      v56 = sub_15A37B0(0x20u, (_QWORD *)v69, (_QWORD *)v55, 0);
      v57 = *(_QWORD *)(v54 + 96);
      v58 = (unsigned __int8 *)v56;
      v59 = sub_14DBA30(v56, v57, 0);
      if ( v59 )
        v58 = (unsigned __int8 *)v59;
    }
    v73.m128i_i16[0] = 257;
    v62 = (_QWORD *)sub_15A06D0(v30, v57, v60, v61);
    return sub_14EDD70((__int64)v58, v62, v69, (__int64)&v72, 0, 0);
  }
}
