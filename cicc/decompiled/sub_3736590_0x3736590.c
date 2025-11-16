// Function: sub_3736590
// Address: 0x3736590
//
char __fastcall sub_3736590(_QWORD *a1)
{
  char result; // al

  result = 1;
  if ( *(_DWORD *)(a1[10] + 32LL) != 2 )
  {
    result = *(_BYTE *)(a1[26] + 3769LL);
    if ( result )
      return a1[51] == 0;
  }
  return result;
}
