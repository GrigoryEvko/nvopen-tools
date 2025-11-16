// Function: sub_987E80
// Address: 0x987e80
//
char __fastcall sub_987E80(__int64 a1, _QWORD *a2)
{
  char result; // al

  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a1 != *a2 )
      return 0;
  }
  else
  {
    result = sub_C43C50(a1, a2);
    if ( !result )
      return result;
  }
  if ( *(_DWORD *)(a1 + 24) <= 0x40u )
    return *(_QWORD *)(a1 + 16) == a2[2];
  else
    return sub_C43C50(a1 + 16, a2 + 2);
}
