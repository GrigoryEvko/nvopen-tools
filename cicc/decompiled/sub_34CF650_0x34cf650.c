// Function: sub_34CF650
// Address: 0x34cf650
//
__int64 __fastcall sub_34CF650(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        _BYTE *a9,
        __int64 a10,
        int a11,
        int a12,
        unsigned int a13)
{
  bool v13; // zf
  __int64 result; // rax

  sub_23CEB70(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  v13 = (_BYTE)qword_503AEC8 == 0;
  *(_QWORD *)a1 = &unk_4A379F8;
  *(_DWORD *)(a1 + 632) = a11;
  *(_DWORD *)(a1 + 636) = a12;
  result = a13;
  *(_DWORD *)(a1 + 648) = a13;
  if ( !v13 )
    *(_BYTE *)(a1 + 877) |= 2u;
  if ( (_BYTE)qword_503ADE8 )
    *(_BYTE *)(a1 + 877) |= 4u;
  return result;
}
