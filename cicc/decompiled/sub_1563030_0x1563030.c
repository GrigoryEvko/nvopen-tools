// Function: sub_1563030
// Address: 0x1563030
//
__int64 __fastcall sub_1563030(__m128i *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 v5; // rsi
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

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
  v6[0] = a2;
  v2 = (__int64 *)sub_155EE30(v6);
  result = sub_155EE40(v6);
  if ( v2 != (__int64 *)result )
  {
    v4 = (__int64 *)result;
    do
    {
      v5 = *v2++;
      result = (__int64)sub_1562E30(a1, v5);
    }
    while ( v4 != v2 );
  }
  return result;
}
