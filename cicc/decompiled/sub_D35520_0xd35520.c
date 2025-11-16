// Function: sub_D35520
// Address: 0xd35520
//
char __fastcall sub_D35520(__int64 a1)
{
  char result; // al

  result = sub_D354F0(a1);
  if ( !result )
    return (unsigned int)(*(_DWORD *)(a1 + 8) - 1) <= 1;
  return result;
}
