// Function: sub_1776630
// Address: 0x1776630
//
char __fastcall sub_1776630(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // cl
  unsigned __int8 v3; // cl
  char result; // al

  if ( a2 == a1 )
    return 1;
  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 <= 0x17u )
    return 0;
  v3 = v2 - 35;
  if ( v3 > 0x2Au )
    return 0;
  result = ((0x43FFE23FFFFuLL >> v3) & 1) == 0;
  if ( ((0x43FFE23FFFFuLL >> v3) & 1) == 0 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) > 0x17u )
    return sub_15F40E0(a1, a2);
  return result;
}
