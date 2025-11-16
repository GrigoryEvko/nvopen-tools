// Function: sub_1034B30
// Address: 0x1034b30
//
__int64 __fastcall sub_1034B30(
        __int64 a1,
        __int64 a2,
        char a3,
        _QWORD *a4,
        __int16 a5,
        __int64 a6,
        _BYTE *a7,
        int *a8,
        char a9,
        __int64 *a10)
{
  unsigned __int64 v12; // r14
  int v13; // r15d
  __int64 result; // rax

  if ( !a7 || *a7 != 61 )
    return sub_1033860(a1, a2, a3, a4, a5, a6, (__int64)a7, a8, a9, a10);
  v12 = sub_102F4D0(a1, (__int64)a7, a6);
  v13 = v12 & 7;
  if ( v13 == 2 )
    return v12;
  result = sub_1033860(a1, a2, a3, a4, a5, a6, (__int64)a7, a8, a9, a10);
  if ( (result & 7) != 2 && v13 == 3 && v12 >> 61 == 1 )
    return v12;
  return result;
}
