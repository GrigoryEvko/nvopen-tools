// Function: sub_14636A0
// Address: 0x14636a0
//
__int64 *__fastcall sub_14636A0(
        __int64 *src,
        __int64 **a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __int64 a8,
        _QWORD *a9,
        _QWORD *a10,
        __int64 *a11,
        __int64 a12)
{
  __int64 v14; // rax
  __int64 *v15; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // [rsp+8h] [rbp-38h]

  v14 = ((((char *)a2 - (char *)src) >> 3) + 1) / 2;
  v20 = 8 * v14;
  v15 = &src[v14];
  if ( v14 <= a4 )
  {
    sub_1462F20(src, (__int64 **)&src[v14], a3, a4, a5, 8 * v14, a6, a7);
    sub_1462F20(v15, a2, a3, v17, v18, v19, a6, a7);
  }
  else
  {
    sub_14636A0(src, (__int64)a9, (int)a10, (__int64)a11, a12);
    sub_14636A0(v15, (__int64)a9, (int)a10, (__int64)a11, a12);
  }
  return sub_1463050(
           (__int64 **)src,
           (__int64 **)v15,
           (__int64)a2,
           v20 >> 3,
           ((char *)a2 - (char *)v15) >> 3,
           (__int64 **)a3,
           a4,
           a9,
           a10,
           a11,
           a12);
}
