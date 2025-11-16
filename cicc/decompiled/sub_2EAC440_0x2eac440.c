// Function: sub_2EAC440
// Address: 0x2eac440
//
__int64 __fastcall sub_2EAC440(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        char a4,
        const __m128i *a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        char a9,
        char a10,
        unsigned __int8 a11)
{
  unsigned __int64 v11; // rdx

  if ( a3 == -1 || a3 == 0xBFFFFFFFFFFFFFFELL )
  {
    v11 = 0;
  }
  else if ( (a3 & 0x4000000000000000LL) != 0 )
  {
    v11 = 8 * (((unsigned __int64)(unsigned int)(8 * a3) << 29) | 0x21) + 4;
  }
  else
  {
    v11 = ((unsigned __int64)(unsigned int)(8 * a3) << 32) | 1;
  }
  return sub_2EAC3E0(a1, a2, v11, a4, a5, a6, a7, a8, a9, a10, a11);
}
