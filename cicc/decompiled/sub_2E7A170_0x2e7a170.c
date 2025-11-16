// Function: sub_2E7A170
// Address: 0x2e7a170
//
char __fastcall sub_2E7A170(__int64 a1)
{
  char result; // al

  result = *(_BYTE *)(a1 + 9);
  if ( !result )
    return sub_AC2F10(*(_BYTE **)a1);
  return result;
}
