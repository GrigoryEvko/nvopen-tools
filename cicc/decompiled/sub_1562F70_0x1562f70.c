// Function: sub_1562F70
// Address: 0x1562f70
//
__int64 __fastcall sub_1562F70(__m128i *a1, __int64 a2, int a3)
{
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 *v5; // r13
  __int64 v6; // rsi
  __int64 v7; // [rsp+8h] [rbp-38h] BYREF
  __int64 v8[5]; // [rsp+18h] [rbp-28h] BYREF

  a1->m128i_i64[0] = 0;
  a1[1].m128i_i32[0] = 0;
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i64[0] = (__int64)a1[1].m128i_i64;
  a1[2].m128i_i64[1] = (__int64)a1[1].m128i_i64;
  a1[3].m128i_i64[0] = 0;
  a1[3].m128i_i64[1] = 0;
  a1[4].m128i_i64[0] = 0;
  a1[4].m128i_i64[1] = 0;
  a1[5].m128i_i64[0] = 0;
  a1[5].m128i_i64[1] = 0;
  v7 = a2;
  v8[0] = sub_15601E0(&v7, a3);
  v3 = (__int64 *)sub_155EE30(v8);
  result = sub_155EE40(v8);
  if ( v3 != (__int64 *)result )
  {
    v5 = (__int64 *)result;
    do
    {
      v6 = *v3++;
      result = (__int64)sub_1562E30(a1, v6);
    }
    while ( v5 != v3 );
  }
  return result;
}
