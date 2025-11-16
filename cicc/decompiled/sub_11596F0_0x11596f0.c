// Function: sub_11596F0
// Address: 0x11596f0
//
unsigned __int8 *__fastcall sub_11596F0(__m128i *a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  char v3; // al
  unsigned __int8 *v4; // rax
  unsigned __int8 *result; // rax
  _OWORD v6[2]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v7; // [rsp+20h] [rbp-40h]
  unsigned __int8 *v8; // [rsp+28h] [rbp-38h]
  __m128i v9; // [rsp+30h] [rbp-30h]
  __int64 v10; // [rsp+40h] [rbp-20h]

  v2 = a1[10].m128i_i64[0];
  v6[0] = _mm_loadu_si128(a1 + 6);
  v7 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v10 = v2;
  v8 = a2;
  v6[1] = _mm_loadu_si128(a1 + 7);
  v9 = _mm_loadu_si128(a1 + 9);
  v3 = sub_B45210((__int64)a2);
  v4 = sub_1009300(*((__int64 **)a2 - 8), *((_BYTE **)a2 - 4), v3, (__int64 *)v6, 0, 1);
  if ( v4 )
    return sub_F162A0((__int64)a1, (__int64)a2, (__int64)v4);
  result = sub_F0F270((__int64)a1, a2);
  if ( !result )
    return (unsigned __int8 *)sub_F11DB0(a1->m128i_i64, a2);
  return result;
}
