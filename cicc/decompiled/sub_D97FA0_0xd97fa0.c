// Function: sub_D97FA0
// Address: 0xd97fa0
//
__int64 __fastcall sub_D97FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 a6, __int128 a7)
{
  __int64 v8; // [rsp-10h] [rbp-20h]
  __m128i v9; // [rsp+0h] [rbp-10h] BYREF

  v9 = _mm_loadu_si128((const __m128i *)&a7);
  sub_D97D90(a1, a2, a3, a4, a5, a6, (__int64 **)&v9, 1);
  return v8;
}
