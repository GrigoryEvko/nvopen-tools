// Function: sub_C80E20
// Address: 0xc80e20
//
__int64 __fastcall sub_C80E20(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  bool v6; // r14
  __int64 *v8; // rax
  __m128i v9; // [rsp+0h] [rbp-C0h] BYREF
  __m128i v10; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v11; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v12; // [rsp+30h] [rbp-90h] BYREF
  __m128i v13; // [rsp+40h] [rbp-80h]
  __m128i v14; // [rsp+50h] [rbp-70h]
  _QWORD v15[12]; // [rsp+60h] [rbp-60h] BYREF

  sub_C80240((__int64)&v9, a1, a2, a3);
  v4 = _mm_loadu_si128(&v10);
  v5 = _mm_loadu_si128(&v11);
  v12 = _mm_loadu_si128(&v9);
  v13 = v4;
  v14 = v5;
  sub_C801E0((__int64)v15, (__int64)a1, a2);
  if ( sub_C80200(&v9, v15) )
    return 0;
  v6 = 0;
  if ( v10.m128i_i64[1] > 2uLL && sub_C80220(*(_BYTE *)v10.m128i_i64[0], a3) )
  {
    v6 = *(_BYTE *)(v10.m128i_i64[0] + 1) == *(_BYTE *)v10.m128i_i64[0];
    if ( a3 <= 1 )
      goto LABEL_8;
  }
  else if ( a3 <= 1 )
  {
LABEL_4:
    if ( sub_C80220(*(_BYTE *)v10.m128i_i64[0], a3) )
      return v10.m128i_i64[0];
    return 0;
  }
  if ( !v10.m128i_i64[1] || *(_BYTE *)(v10.m128i_i64[0] + v10.m128i_i64[1] - 1) != 58 )
  {
LABEL_8:
    if ( v6 )
      goto LABEL_9;
    goto LABEL_4;
  }
LABEL_9:
  v8 = sub_C803D0(v12.m128i_i64);
  if ( sub_C80200(v8, v15) || !sub_C80220(*(_BYTE *)v13.m128i_i64[0], a3) )
    return v10.m128i_i64[0];
  return (__int64)a1;
}
