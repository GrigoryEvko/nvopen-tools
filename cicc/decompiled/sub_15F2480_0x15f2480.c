// Function: sub_15F2480
// Address: 0x15f2480
//
char __fastcall sub_15F2480(__int64 a1)
{
  unsigned __int8 v1; // dl
  char result; // al

  v1 = *(_BYTE *)(a1 + 17);
  result = v1 >> 7;
  if ( ((v1 ^ 0x7E) & 0x7E) != 0 )
    return 0;
  return result;
}
