// Function: sub_8D0B70
// Address: 0x8d0b70
//
__int64 __fastcall sub_8D0B70(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  _QWORD *v5; // rdi
  __int64 v6; // r8
  __int64 v7; // r9

  if ( (*(_BYTE *)(a1 + 81) & 0x20) != 0 )
    return 0;
  v5 = (_QWORD *)sub_880F80(a1);
  if ( (_QWORD *)qword_4D03FF0 == v5 )
    return 0;
  sub_8D0A80(v5, a2, v3, v4, v6, v7);
  return 1;
}
