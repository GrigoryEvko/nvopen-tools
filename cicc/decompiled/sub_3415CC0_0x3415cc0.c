// Function: sub_3415CC0
// Address: 0x3415cc0
//
__int64 __fastcall sub_3415CC0(
        const __m128i *a1,
        __int64 a2,
        int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __m128i *v8; // rax
  int v9; // edx
  __int64 v10; // r9
  __m128i v12[3]; // [rsp+0h] [rbp-30h] BYREF

  v8 = sub_33ED250((__int64)a1, a4, a5);
  v12[0] = _mm_loadu_si128((const __m128i *)&a7);
  return sub_3415C10(a1, a2, a3, (unsigned __int64)v8, v9, v10, (unsigned __int64 *)v12, 1);
}
