// Function: sub_D2F730
// Address: 0xd2f730
//
__int64 __fastcall sub_D2F730(_BYTE *a1, _BYTE *a2)
{
  unsigned __int8 v2; // cl
  __int64 result; // rax

  if ( a2 == a1 )
    return 1;
  if ( *a1 <= 0x1Cu )
    return 0;
  v2 = *a1 - 42;
  if ( v2 > 0x2Au )
    return 0;
  result = ((0x43FFE23FFFFuLL >> v2) & 1) == 0;
  if ( ((0x43FFE23FFFFuLL >> v2) & 1) == 0 )
    return 0;
  if ( *a2 > 0x1Cu )
    return sub_B46130((__int64)a1, (__int64)a2, 0);
  return result;
}
