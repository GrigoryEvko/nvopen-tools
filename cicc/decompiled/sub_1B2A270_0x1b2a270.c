// Function: sub_1B2A270
// Address: 0x1b2a270
//
__int64 __fastcall sub_1B2A270(__int64 *a1, unsigned __int64 a2)
{
  __int8 *v3; // r8
  __m128i *v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // [rsp+8h] [rbp-68h] BYREF
  __m128i *v12; // [rsp+10h] [rbp-60h]
  __int64 v13; // [rsp+18h] [rbp-58h]
  __m128i v14; // [rsp+20h] [rbp-50h] BYREF
  __int64 v15[2]; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v16[6]; // [rsp+40h] [rbp-30h] BYREF

  v11 = a2;
  if ( a2 )
  {
    v9 = a2;
    v3 = &v14.m128i_i8[5];
    do
    {
      *--v3 = v9 % 0xA + 48;
      v10 = v9;
      v9 /= 0xAu;
    }
    while ( v10 > 9 );
  }
  else
  {
    v14.m128i_i8[4] = 48;
    v3 = &v14.m128i_i8[4];
  }
  v15[0] = (__int64)v16;
  sub_1B29E50(v15, v3, (__int64)v14.m128i_i64 + 5);
  v4 = (__m128i *)sub_2241130(v15, 0, 0, "llvm.ssa.copy.", 14);
  v12 = &v14;
  if ( (__m128i *)v4->m128i_i64[0] == &v4[1] )
  {
    v14 = _mm_loadu_si128(v4 + 1);
  }
  else
  {
    v12 = (__m128i *)v4->m128i_i64[0];
    v14.m128i_i64[0] = v4[1].m128i_i64[0];
  }
  v13 = v4->m128i_i64[1];
  v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
  v5 = (_QWORD *)v15[0];
  v4->m128i_i64[1] = 0;
  v4[1].m128i_i8[0] = 0;
  if ( v5 != v16 )
    j_j___libc_free_0(v5, v16[0] + 1LL);
  v6 = sub_15E1360(*a1, 197, (__int64)&v11);
  v7 = sub_1632190((__int64)a1, (__int64)v12, v13, v6);
  if ( v12 != &v14 )
    j_j___libc_free_0(v12, v14.m128i_i64[0] + 1);
  return v7;
}
