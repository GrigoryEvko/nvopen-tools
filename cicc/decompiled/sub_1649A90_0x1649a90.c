// Function: sub_1649A90
// Address: 0x1649a90
//
__int64 __fastcall sub_1649A90(__int64 a1)
{
  char v1; // al
  unsigned int v3; // r8d

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 17 )
    return sub_15E02D0(a1);
  v3 = 0;
  if ( v1 == 53 )
    return (*(_WORD *)(a1 + 18) >> 6) & 1;
  return v3;
}
