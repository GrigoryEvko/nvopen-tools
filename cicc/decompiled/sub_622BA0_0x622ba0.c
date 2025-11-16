// Function: sub_622BA0
// Address: 0x622ba0
//
__int64 __fastcall sub_622BA0(unsigned int a1, const __m128i *a2, _WORD *a3)
{
  int v5; // ebx
  int v6; // [rsp-6Ch] [rbp-6Ch] BYREF
  __m128i v7; // [rsp-68h] [rbp-68h] BYREF
  __m128i v8; // [rsp-58h] [rbp-58h] BYREF
  __m128i v9[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( a1 > 0x10 )
    return 0;
  sub_620DE0(a3, 0);
  v7 = _mm_loadu_si128(a2);
  sub_620DE0(v9, 0xFFu);
  if ( a1 )
  {
    v5 = 8 * (a1 - 1);
    do
    {
      v8 = _mm_loadu_si128(v9);
      sub_6213D0((__int64)&v8, (__int64)&v7);
      sub_621410((__int64)&v8, v5, &v6);
      v5 -= 8;
      sub_6213B0((__int64)a3, (__int64)&v8);
      sub_6214E0(v7.m128i_i16, 8, 0, 0);
    }
    while ( v5 != -8 );
  }
  return 1;
}
