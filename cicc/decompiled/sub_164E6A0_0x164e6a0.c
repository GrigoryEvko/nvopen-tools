// Function: sub_164E6A0
// Address: 0x164e6a0
//
char __fastcall sub_164E6A0(__int64 a1, _QWORD *a2)
{
  char result; // al

  if ( *(_DWORD *)(a1 + 24) <= 0x40u )
  {
    if ( *(_QWORD *)(a1 + 16) == *a2 )
      return 1;
  }
  else
  {
    result = sub_16A5220(a1 + 16, a2);
    if ( result )
      return result;
  }
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return *(_QWORD *)a1 == a2[2];
  else
    return sub_16A5220(a1, a2 + 2);
}
