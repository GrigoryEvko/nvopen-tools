// Function: sub_14205E0
// Address: 0x14205e0
//
bool *__fastcall sub_14205E0(bool *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __m128i v7; // [rsp+10h] [rbp-30h]
  __int64 v8; // [rsp+20h] [rbp-20h]

  if ( *(_BYTE *)a4 )
  {
    v6.m128i_i64[0] = 0;
    v6.m128i_i64[1] = -1;
    v7 = 0u;
    v8 = 0;
  }
  else
  {
    v8 = *(_QWORD *)(a4 + 40);
    v6 = _mm_loadu_si128((const __m128i *)(a4 + 8));
    v7 = _mm_loadu_si128((const __m128i *)(a4 + 24));
  }
  sub_14203A0(a1, a2, &v6, a3, a5);
  return a1;
}
