// Function: sub_B45190
// Address: 0xb45190
//
char __fastcall sub_B45190(__int64 a1)
{
  unsigned __int8 v1; // dl
  char result; // al

  v1 = *(_BYTE *)(a1 + 1);
  result = v1 >> 7;
  if ( ((v1 ^ 0x7E) & 0x7E) != 0 )
    return 0;
  return result;
}
