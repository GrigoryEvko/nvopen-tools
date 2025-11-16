// Function: sub_141C340
// Address: 0x141c340
//
__int64 __fastcall sub_141C340(
        __int64 a1,
        __m128i *a2,
        unsigned __int8 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        unsigned __int8 a8)
{
  unsigned __int64 v10; // r14
  int v11; // r15d
  __int64 result; // rax

  if ( !a6 || *(_BYTE *)(a6 + 16) != 54 )
    return sub_141B2C0(a1, a2, a3, a4, a5, a6, a7, a8);
  v10 = sub_14173C0(a1, a6, a5);
  v11 = v10 & 7;
  if ( v11 == 2 )
    return v10;
  result = sub_141B2C0(a1, a2, a3, a4, a5, a6, a7, a8);
  if ( (result & 7) != 2 && v11 == 3 && v10 >> 61 == 1 )
    return v10;
  return result;
}
