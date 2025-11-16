// Function: sub_29ABCE0
// Address: 0x29abce0
//
__int64 __fastcall sub_29ABCE0(__int64 a1)
{
  _QWORD *v2; // rdi
  unsigned int v3; // eax

  v2 = (_QWORD *)sub_AA48A0(**(_QWORD **)(a1 + 88));
  v3 = *(_DWORD *)(a1 + 112);
  if ( v3 <= 1 )
    return sub_BCB120(v2);
  if ( v3 == 2 )
    return sub_BCB2A0(v2);
  return sub_BCB2C0(v2);
}
