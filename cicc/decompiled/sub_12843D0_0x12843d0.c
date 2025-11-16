// Function: sub_12843D0
// Address: 0x12843d0
//
__int64 __fastcall sub_12843D0(
        __int64 *a1,
        _DWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        __int64 a10,
        _BYTE *a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15)
{
  if ( (_DWORD)a10 == 1 )
    return sub_1282050(a1, a2, 0, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
  else
    return sub_1280F50(a1, a7, (unsigned __int64)a11, a12, a15 & 1);
}
