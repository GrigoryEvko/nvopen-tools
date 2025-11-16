// Function: sub_2B71BE0
// Address: 0x2b71be0
//
__int64 __fastcall sub_2B71BE0(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        char a4,
        __int64 a5,
        const __m128i *a6,
        _DWORD *a7,
        const void *a8,
        __int64 a9,
        unsigned int *a10,
        __int64 a11,
        int a12)
{
  __int64 result; // rax

  result = sub_2B70420(a1, a2, a3, a4 == 0 ? 3 : 0, a5, a6, a7, a8, a9, a10, a11);
  if ( result )
  {
    if ( a12 )
      *(_DWORD *)(result + 432) = a12;
  }
  return result;
}
