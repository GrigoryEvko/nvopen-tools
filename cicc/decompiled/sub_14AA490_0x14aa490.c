// Function: sub_14AA490
// Address: 0x14aa490
//
unsigned __int64 __fastcall sub_14AA490(__int64 a1)
{
  unsigned __int8 v1; // cl
  unsigned __int64 result; // rax
  unsigned __int16 v3; // cx

  v1 = *(_BYTE *)(a1 + 16);
  result = 0;
  if ( v1 <= 0x17u )
  {
    if ( v1 == 5 )
    {
      v3 = *(_WORD *)(a1 + 18);
      if ( v3 <= 0x17u )
        return ((unsigned __int64)&loc_80A800 >> v3) & 1;
    }
  }
  else if ( v1 <= 0x2Fu )
  {
    return (0x80A800000000uLL >> v1) & 1;
  }
  return result;
}
