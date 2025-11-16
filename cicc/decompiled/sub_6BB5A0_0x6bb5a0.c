// Function: sub_6BB5A0
// Address: 0x6bb5a0
//
__int64 __fastcall sub_6BB5A0(__int64 a1, int a2)
{
  _QWORD *v2; // r8

  if ( qword_4D03C50 )
  {
    v2 = *(_QWORD **)(qword_4D03C50 + 136LL);
    if ( v2 )
    {
      if ( *v2 )
        return sub_6E1C80(*(_QWORD *)(qword_4D03C50 + 136LL));
    }
  }
  if ( word_4F06418[0] == 73 && dword_4D04428 | a2 )
    return sub_6BA760(a1, 0);
  return sub_6A2C00(a1, 1u);
}
