// Function: sub_1783F70
// Address: 0x1783f70
//
_QWORD *__fastcall sub_1783F70(
        const __m128i *a1,
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
  __m128 v10; // xmm0
  __m128i v11; // xmm1
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // r13
  _QWORD *v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  _OWORD v21[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v22; // [rsp+20h] [rbp-30h]

  v10 = (__m128)_mm_loadu_si128(a1 + 167);
  v11 = _mm_loadu_si128(a1 + 168);
  v22 = a2;
  v21[0] = v10;
  v21[1] = v11;
  v12 = sub_15F24E0(a2);
  v13 = sub_13D1D50(*(unsigned __int8 **)(a2 - 48), *(_QWORD *)(a2 - 24), v12, v21);
  if ( !v13 )
    return sub_1707490((__int64)a1, (unsigned __int8 *)a2, *(double *)v10.m128_u64, *(double *)v11.m128i_i64, a5);
  v14 = *(_QWORD *)(a2 + 8);
  if ( !v14 )
    return 0;
  v15 = a1->m128i_i64[0];
  v16 = v13;
  do
  {
    v17 = sub_1648700(v14);
    sub_170B990(v15, (__int64)v17);
    v14 = *(_QWORD *)(v14 + 8);
  }
  while ( v14 );
  if ( a2 == v16 )
    v16 = sub_1599EF0(*(__int64 ***)a2);
  sub_164D160(a2, v16, v10, *(double *)v11.m128i_i64, a5, a6, v18, v19, a9, a10);
  return (_QWORD *)a2;
}
