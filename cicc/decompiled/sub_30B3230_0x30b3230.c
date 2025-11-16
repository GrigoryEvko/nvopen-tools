// Function: sub_30B3230
// Address: 0x30b3230
//
char __fastcall sub_30B3230(char *a1, __int64 a2, __int64 a3)
{
  char result; // al

  result = *a1;
  if ( !*a1 || *(_DWORD *)(a2 + 56) != 4 )
    return sub_30B1A10(a3, a2) != 0;
  return result;
}
