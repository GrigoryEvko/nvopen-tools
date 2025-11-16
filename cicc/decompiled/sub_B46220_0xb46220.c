// Function: sub_B46220
// Address: 0xb46220
//
char __fastcall sub_B46220(__int64 a1, __int64 a2)
{
  char result; // al

  result = sub_B46130(a1, a2, 0);
  if ( result )
    return ((*(_BYTE *)(a2 + 1) ^ *(_BYTE *)(a1 + 1)) & 0xFE) == 0;
  return result;
}
