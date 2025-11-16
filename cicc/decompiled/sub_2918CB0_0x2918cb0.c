// Function: sub_2918CB0
// Address: 0x2918cb0
//
unsigned __int64 __fastcall sub_2918CB0(__int64 a1)
{
  unsigned __int8 v1; // cl
  unsigned __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 8);
  result = 1;
  if ( v1 > 3u && v1 != 5 && (v1 & 0xF5) != 4 )
  {
    result = 0;
    if ( v1 <= 0x14u )
      return (0x160400uLL >> v1) & 1;
  }
  return result;
}
