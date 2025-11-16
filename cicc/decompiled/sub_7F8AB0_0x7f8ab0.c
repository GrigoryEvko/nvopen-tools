// Function: sub_7F8AB0
// Address: 0x7f8ab0
//
_QWORD *__fastcall sub_7F8AB0(
        char *src,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __m128i *a11)
{
  __int64 v12; // rdi

  v12 = *a2;
  if ( !*a2 )
  {
    sub_7F8110(src, (__int64)a2, a3, a4, a5, a6, a7, a8, a9, a10);
    v12 = *a2;
  }
  return sub_7F88E0(v12, a11);
}
