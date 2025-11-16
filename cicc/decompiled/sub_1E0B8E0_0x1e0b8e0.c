// Function: sub_1E0B8E0
// Address: 0x1e0b8e0
//
__int64 __fastcall sub_1E0B8E0(
        __int64 a1,
        unsigned __int16 a2,
        int a3,
        int a4,
        int a5,
        int a6,
        __int128 a7,
        __int64 a8,
        unsigned __int8 a9,
        int a10,
        char a11)
{
  int v13; // r14d
  __int64 v14; // r13
  __int128 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+20h] [rbp-40h]

  v19 = a8;
  v13 = a9;
  v18 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
  v14 = sub_145CBF0((__int64 *)(a1 + 120), 80, 16);
  sub_1E342C0(v14, a2, a3, a4, a5, a6, v18, v19, v13, a10, a11);
  return v14;
}
