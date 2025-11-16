// Function: sub_2366BC0
// Address: 0x2366bc0
//
__int64 __fastcall sub_2366BC0(const __m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  unsigned int v8; // r12d
  __m128i v10; // [rsp+0h] [rbp-60h]
  unsigned __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-48h]
  unsigned __int64 i; // [rsp+20h] [rbp-40h]

  v3 = a1[1].m128i_i64[1];
  v11 = 0;
  v4 = a1[1].m128i_i64[0];
  v12 = 0;
  i = 0;
  v10 = _mm_loadu_si128(a1);
  v5 = v3 - v4;
  if ( v3 == v4 )
  {
    v7 = 0;
  }
  else
  {
    if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v6 = sub_22077B0(v3 - v4);
    v3 = a1[1].m128i_i64[1];
    v4 = a1[1].m128i_i64[0];
    v7 = v6;
  }
  v11 = v7;
  v12 = v7;
  for ( i = v7 + v5; v3 != v4; v7 += 40LL )
  {
    if ( v7 )
    {
      *(__m128i *)v7 = _mm_loadu_si128((const __m128i *)v4);
      sub_23667F0((__m128i **)(v7 + 16), (const __m128i **)(v4 + 16), a3);
    }
    v4 += 40;
  }
  v12 = v7;
  v8 = 0;
  if ( v10.m128i_i64[1] == 16 )
    LOBYTE(v8) = (*(_QWORD *)v10.m128i_i64[0] ^ 0x6572702D706F6F6CLL
                | *(_QWORD *)(v10.m128i_i64[0] + 8) ^ 0x6E6F697461636964LL) == 0;
  sub_234A6B0(&v11);
  return v8;
}
