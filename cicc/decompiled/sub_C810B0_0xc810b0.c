// Function: sub_C810B0
// Address: 0xc810b0
//
__int64 __fastcall sub_C810B0(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  bool v6; // r12
  __int64 v7; // rax
  __int64 *v9; // rax
  __int64 *v10; // rax
  __m128i v11; // [rsp+0h] [rbp-C0h] BYREF
  __m128i v12; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v13; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v14; // [rsp+30h] [rbp-90h] BYREF
  __m128i v15; // [rsp+40h] [rbp-80h]
  __m128i v16; // [rsp+50h] [rbp-70h]
  _QWORD v17[12]; // [rsp+60h] [rbp-60h] BYREF

  sub_C80240((__int64)&v11, a1, a2, a3);
  v4 = _mm_loadu_si128(&v12);
  v5 = _mm_loadu_si128(&v13);
  v14 = _mm_loadu_si128(&v11);
  v15 = v4;
  v16 = v5;
  sub_C801E0((__int64)v17, (__int64)a1, a2);
  if ( sub_C80200(&v11, v17) )
    return 0;
  v6 = 0;
  if ( v12.m128i_i64[1] > 2uLL && sub_C80220(*(_BYTE *)v12.m128i_i64[0], a3) )
  {
    v7 = v12.m128i_i64[0];
    v6 = *(_BYTE *)(v12.m128i_i64[0] + 1) == *(_BYTE *)v12.m128i_i64[0];
    if ( a3 <= 1 )
      goto LABEL_8;
  }
  else
  {
    v7 = v12.m128i_i64[0];
    if ( a3 <= 1 )
      goto LABEL_4;
  }
  if ( !v12.m128i_i64[1] || *(_BYTE *)(v7 + v12.m128i_i64[1] - 1) != 58 )
  {
LABEL_8:
    if ( v6 )
    {
      v9 = sub_C803D0(v14.m128i_i64);
      if ( !sub_C80200(v9, v17) && sub_C80220(*(_BYTE *)v15.m128i_i64[0], a3) )
        return v15.m128i_i64[0];
      return 0;
    }
    goto LABEL_4;
  }
  v10 = sub_C803D0(v14.m128i_i64);
  if ( !sub_C80200(v10, v17) && sub_C80220(*(_BYTE *)v15.m128i_i64[0], a3) )
    return v15.m128i_i64[0];
  if ( v6 )
    return 0;
LABEL_4:
  if ( !sub_C80220(*(_BYTE *)v12.m128i_i64[0], a3) )
    return 0;
  return v12.m128i_i64[0];
}
