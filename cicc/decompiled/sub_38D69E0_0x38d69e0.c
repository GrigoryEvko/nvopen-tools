// Function: sub_38D69E0
// Address: 0x38d69e0
//
__int64 __fastcall sub_38D69E0(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        int a5,
        int a6,
        unsigned __int8 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v10; // rbx
  __m128i v11; // xmm0
  __m128i v13; // [rsp+0h] [rbp-50h] BYREF
  int v14; // [rsp+14h] [rbp-3Ch]
  int v15; // [rsp+18h] [rbp-38h]
  int v16; // [rsp+1Ch] [rbp-34h]

  v14 = a4;
  v10 = a9;
  v15 = a5;
  v16 = a6;
  v13 = _mm_loadu_si128((const __m128i *)&a8);
  sub_39120A0(a1);
  a9 = v10;
  v11 = _mm_load_si128(&v13);
  return sub_38DC620(a1, a2, a3, v14, v15, (unsigned __int8)v16, a7, v11.m128i_i32[0], v11.m128i_i32[2], v10);
}
