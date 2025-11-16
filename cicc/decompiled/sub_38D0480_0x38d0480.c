// Function: sub_38D0480
// Address: 0x38d0480
//
__int64 __fastcall sub_38D0480(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
    return sub_38D0300(a1, a2, 0, a3);
  else
    return sub_38D01D0((__int64)a1, a2, 0, a3);
}
