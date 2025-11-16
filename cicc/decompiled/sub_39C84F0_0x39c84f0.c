// Function: sub_39C84F0
// Address: 0x39c84f0
//
char __fastcall sub_39C84F0(_QWORD *a1)
{
  char result; // al

  result = 1;
  if ( *(_DWORD *)(a1[10] + 36LL) != 2 )
  {
    result = *(_BYTE *)(a1[25] + 4513LL);
    if ( result )
      return a1[77] == 0;
  }
  return result;
}
