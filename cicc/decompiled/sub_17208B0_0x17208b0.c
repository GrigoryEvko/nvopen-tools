// Function: sub_17208B0
// Address: 0x17208b0
//
__int64 __fastcall sub_17208B0(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int8 *v10; // r15
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // rdx
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rcx
  __int64 v28; // r13
  unsigned __int64 v29; // rax
  __int64 v30; // rcx
  __int64 *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 ***v34; // r14
  void *v35; // rax
  void *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdx
  unsigned __int8 *v41; // rax
  __int64 **v42; // r14
  __int64 v43; // r13
  unsigned __int8 *v44; // rax
  unsigned __int64 v45; // r10
  __int64 v46; // rax
  __int64 *v47; // [rsp+0h] [rbp-90h]
  __int64 **v48; // [rsp+8h] [rbp-88h]
  __int64 **v49; // [rsp+8h] [rbp-88h]
  __int64 *v50; // [rsp+8h] [rbp-88h]
  __int64 v51; // [rsp+10h] [rbp-80h]
  unsigned __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+28h] [rbp-68h] BYREF
  __m128 v54; // [rsp+30h] [rbp-60h] BYREF
  __m128i v55; // [rsp+40h] [rbp-50h]
  __int64 v56; // [rsp+50h] [rbp-40h]

  v10 = (unsigned __int8 *)a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v56 = a2;
  v54 = v12;
  v55 = v13;
  v14 = sub_15F24E0(a2);
  v15 = sub_13D6AE0(*(unsigned __int8 **)(a2 - 48), *(unsigned __int8 **)(a2 - 24), v14, &v54);
  if ( v15 )
  {
    if ( *(_QWORD *)(a2 + 8) )
    {
      v17 = v15;
      sub_17205C0(a1->m128i_i64[0], a2);
      if ( a2 == v17 )
        v17 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v17, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v18, v19, a9, a10);
      return (__int64)v10;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_170D400(a1, a2, v16, (__m128i)v12, *(double *)v13.m128i_i64, a5) )
    return (__int64)v10;
  v21 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)a2,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
  if ( v21 )
    return v21;
  v21 = sub_1713A90(
          a1->m128i_i64,
          (_BYTE *)a2,
          v12,
          *(double *)v13.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          v22,
          v23,
          a9,
          a10);
  if ( v21 )
    return v21;
  v28 = *(_QWORD *)(a2 - 48);
  v29 = *(_QWORD *)(a2 - 24);
  v54.m128_u64[1] = (unsigned __int64)&v53;
  v52 = v29;
  if ( (unsigned __int8)sub_171FB50((__int64)&v54, v28, (__int64)&v53, v27) )
  {
    v31 = (__int64 *)v52;
    v55.m128i_i16[0] = 257;
    v32 = v53;
LABEL_21:
    v10 = (unsigned __int8 *)sub_15FB440(14, v31, v32, (__int64)&v54, 0);
    sub_15F2530(v10, a2, 1);
    return (__int64)v10;
  }
  if ( *(_BYTE *)(v52 + 16) > 0x10u )
  {
    v54.m128_u64[1] = (unsigned __int64)&v53;
    if ( (unsigned __int8)sub_171FB50((__int64)&v54, v52, (__int64)&v53, v30) )
    {
      v32 = v53;
      v55.m128i_i16[0] = 257;
      v31 = (__int64 *)v28;
      goto LABEL_21;
    }
  }
  if ( *(_BYTE *)(v28 + 16) != 66 )
  {
LABEL_18:
    v24 = (__int64)sub_1707FD0(a1, (unsigned __int8 *)a2, v28, v52);
    if ( v24 )
      return sub_170E100(
               a1->m128i_i64,
               a2,
               v24,
               v12,
               *(double *)v13.m128i_i64,
               *(double *)a5.m128i_i64,
               a6,
               v25,
               v26,
               a9,
               a10);
    if ( sub_15F24A0(a2) && sub_15F24C0(a2) )
    {
      v54 = (__m128)a1->m128i_u64[1];
      v24 = sub_171BFC0((__int64 *)&v54, a2, (__m128i)v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
      if ( v24 )
        return sub_170E100(
                 a1->m128i_i64,
                 a2,
                 v24,
                 v12,
                 *(double *)v13.m128i_i64,
                 *(double *)a5.m128i_i64,
                 a6,
                 v25,
                 v26,
                 a9,
                 a10);
    }
    return 0;
  }
  v33 = *(_QWORD *)v28;
  v34 = *(__int64 ****)(v28 - 24);
  v51 = *(_QWORD *)v28;
  if ( *(_BYTE *)(v52 + 16) != 14 )
    goto LABEL_50;
  v48 = *v34;
  if ( *(_BYTE *)(v33 + 8) == 16 )
    v33 = **(_QWORD **)(v33 + 16);
  if ( *((_BYTE *)v48 + 8) == 16 )
    v48 = (__int64 **)*v48[2];
  switch ( *(_BYTE *)(v33 + 8) )
  {
    case 0:
    case 6:
      v35 = sub_16982C0();
      break;
    case 1:
      v35 = sub_1698260();
      break;
    case 2:
      v35 = sub_1698270();
      break;
    case 3:
      v35 = sub_1698280();
      break;
    case 4:
      v35 = sub_16982A0();
      break;
    case 5:
      v35 = sub_1698290();
      break;
  }
  if ( (unsigned int)sub_16982D0((__int64)v35) >= *((_DWORD *)v48 + 2) >> 8
    && (v45 = sub_15A40D0(v52, *v34, 0), (v46 = *(_QWORD *)(v28 + 8)) != 0)
    && !*(_QWORD *)(v46 + 8)
    && (v50 = (__int64 *)v45, v52 == sub_15A3F70(v45, *(__int64 ***)a2, 0))
    && (unsigned int)sub_171CA60(a1->m128i_i64, (__int64)v34, v50, a2) == 2 )
  {
    v39 = a1->m128i_i64[1];
    v55.m128i_i16[0] = 259;
    v40 = (__int64)v50;
    v54.m128_u64[0] = (unsigned __int64)"addconv";
  }
  else
  {
LABEL_50:
    if ( *(_BYTE *)(v52 + 16) != 66 )
      goto LABEL_18;
    v47 = *(__int64 **)(v52 - 24);
    v49 = *v34;
    if ( *(_BYTE *)(v51 + 8) == 16 )
      v51 = **(_QWORD **)(v51 + 16);
    if ( *((_BYTE *)v49 + 8) == 16 )
      v49 = (__int64 **)*v49[2];
    switch ( *(_BYTE *)(v51 + 8) )
    {
      case 1:
        v36 = sub_1698260();
        break;
      case 2:
        v36 = sub_1698270();
        break;
      case 3:
        v36 = sub_1698280();
        break;
      case 4:
        v36 = sub_16982A0();
        break;
      case 5:
        v36 = sub_1698290();
        break;
      case 6:
        v36 = sub_16982C0();
        break;
      default:
        BUG();
    }
    if ( (unsigned int)sub_16982D0((__int64)v36) < *((_DWORD *)v49 + 2) >> 8 || (__int64 **)*v47 != *v34 )
      goto LABEL_18;
    v37 = *(_QWORD *)(v28 + 8);
    if ( !v37 || *(_QWORD *)(v37 + 8) )
    {
      v38 = *(_QWORD *)(v52 + 8);
      if ( !v38 || *(_QWORD *)(v38 + 8) )
        goto LABEL_18;
    }
    if ( (unsigned int)sub_171CA60(a1->m128i_i64, (__int64)v34, v47, a2) != 2 )
      goto LABEL_18;
    v39 = a1->m128i_i64[1];
    v55.m128i_i16[0] = 259;
    v40 = (__int64)v47;
    v54.m128_u64[0] = (unsigned __int64)"addconv";
  }
  v41 = sub_17094A0(
          v39,
          (__int64)v34,
          v40,
          (__int64 *)&v54,
          0,
          1,
          *(double *)v12.m128_u64,
          *(double *)v13.m128i_i64,
          *(double *)a5.m128i_i64);
  v42 = *(__int64 ***)a2;
  v55.m128i_i16[0] = 257;
  v43 = (__int64)v41;
  v44 = (unsigned __int8 *)sub_1648A60(56, 1u);
  v10 = v44;
  if ( v44 )
    sub_15FCE10((__int64)v44, v43, (__int64)v42, (__int64)&v54, 0);
  return (__int64)v10;
}
