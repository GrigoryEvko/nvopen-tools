// Function: sub_3777990
// Address: 0x3777990
//
__m128i *__fastcall sub_3777990(__m128i *a1, __int64 *a2, unsigned __int64 a3, unsigned __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  int v11; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a3 + 80);
  v10 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v10, v8, 1);
  v11 = *(_DWORD *)(a3 + 72);
  sub_3777810(a1, a2, a3, a4, (__int64)&v10, a5);
  if ( v10 )
    sub_B91220((__int64)&v10, v10);
  return a1;
}
