// Function: sub_266DE30
// Address: 0x266de30
//
__int64 __fastcall sub_266DE30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        int a7,
        int a8,
        int a9,
        __int64 a10,
        __int64 a11,
        __int16 a12)
{
  *a5 = a4;
  *(_BYTE *)(a1 + 24) = *(_BYTE *)(a1 + 24) & 0xFC | 2;
  *(_QWORD *)a1 = a10;
  *(_QWORD *)(a1 + 8) = a11;
  *(_WORD *)(a1 + 16) = a12;
  return a1;
}
