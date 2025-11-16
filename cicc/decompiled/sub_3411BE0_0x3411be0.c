// Function: sub_3411BE0
// Address: 0x3411be0
//
unsigned __int8 *__fastcall sub_3411BE0(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        unsigned __int16 *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  unsigned int *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __m128i v12[3]; // [rsp+0h] [rbp-30h] BYREF

  v12[0] = _mm_loadu_si128((const __m128i *)&a7);
  v8 = (unsigned int *)sub_33E5830(a1, a4, a5);
  return sub_3411630(a1, a2, a3, v8, v9, v10, *(_OWORD *)&_mm_load_si128(v12));
}
