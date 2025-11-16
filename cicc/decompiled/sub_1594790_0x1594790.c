// Function: sub_1594790
// Address: 0x1594790
//
char __fastcall sub_1594790(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  char result; // al

  v2 = *(_DWORD *)(a1 + 8) >> 8;
  result = sub_1642F90(a1, 1);
  if ( result )
    return (unsigned __int64)(a2 + 1) <= 2;
  if ( v2 > 0x3F )
    return 1;
  if ( a2 >= -(1LL << ((unsigned __int8)v2 - 1)) )
    return a2 <= (1LL << ((unsigned __int8)v2 - 1)) - 1;
  return result;
}
