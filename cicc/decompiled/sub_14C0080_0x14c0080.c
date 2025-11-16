// Function: sub_14C0080
// Address: 0x14c0080
//
__int64 __fastcall sub_14C0080(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r8

  if ( *(_BYTE *)(a1 + 16) != 35 )
    return 0;
  v3 = *(_QWORD *)(a1 - 48);
  if ( a2 == v3 )
  {
    v3 = *(_QWORD *)(a1 - 24);
  }
  else if ( a2 != *(_QWORD *)(a1 - 24) )
  {
    return 0;
  }
  return sub_14BE170(v3, 0, a3);
}
