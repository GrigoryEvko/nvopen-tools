// Function: sub_1807990
// Address: 0x1807990
//
unsigned __int64 __fastcall sub_1807990(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        double a7,
        double a8,
        double a9,
        __int64 a10,
        unsigned __int8 a11,
        char a12,
        unsigned int a13)
{
  if ( (((_DWORD)a10 - 8) & 0xFFFFFFF7) != 0 && (((_DWORD)a10 - 32) & 0xFFFFFFDF) != 0 && (_DWORD)a10 != 128
    || a5 < a6 && a5 && a5 < (unsigned int)a10 >> 3 )
  {
    return sub_18074C0(a1, a2, a3, a4, a10, a11, a7, a8, a9, a12, a13);
  }
  else
  {
    return sub_1806810(a1, a2, a3, a4, a10, a11, a7, a8, a9, 0, a12, a13);
  }
}
