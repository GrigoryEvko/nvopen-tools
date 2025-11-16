// Function: sub_15F41F0
// Address: 0x15f41f0
//
char __fastcall sub_15F41F0(__int64 a1, __int64 a2)
{
  char result; // al

  result = sub_15F40E0(a1, a2);
  if ( result )
    return ((*(_BYTE *)(a2 + 17) ^ *(_BYTE *)(a1 + 17)) & 0xFE) == 0;
  return result;
}
