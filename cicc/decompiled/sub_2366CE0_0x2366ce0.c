// Function: sub_2366CE0
// Address: 0x2366ce0
//
__int64 __fastcall sub_2366CE0(const __m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int64 *v11; // rdi
  __m128i v13; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 i; // [rsp+20h] [rbp-40h]

  v3 = a1[1].m128i_i64[1];
  v14 = 0;
  v4 = a1[1].m128i_i64[0];
  v15 = 0;
  i = 0;
  v13 = _mm_loadu_si128(a1);
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
  v14 = v7;
  v15 = v7;
  for ( i = v7 + v5; v3 != v4; v7 += 40 )
  {
    if ( v7 )
    {
      *(__m128i *)v7 = _mm_loadu_si128((const __m128i *)v4);
      sub_23667F0((__m128i **)(v7 + 16), (const __m128i **)(v4 + 16), a3);
    }
    v4 += 40;
  }
  v15 = v7;
  v8 = sub_C931B0(v13.m128i_i64, "simple-loop-unswitch", 0x14u, 0);
  v9 = v15;
  v10 = v14;
  LOBYTE(v3) = v8 != -1;
  if ( v15 != v14 )
  {
    do
    {
      v11 = (unsigned __int64 *)(v10 + 16);
      v10 += 40LL;
      sub_234A6B0(v11);
    }
    while ( v9 != v10 );
    v10 = v14;
  }
  if ( v10 )
    j_j___libc_free_0(v10);
  return (unsigned int)v3;
}
