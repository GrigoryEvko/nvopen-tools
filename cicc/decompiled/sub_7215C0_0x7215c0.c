// Function: sub_7215C0
// Address: 0x7215c0
//
_BOOL8 __fastcall sub_7215C0(unsigned __int8 *a1)
{
  unsigned __int8 v1; // r12
  _BOOL8 result; // rax

  v1 = *a1;
  if ( !unk_4F07598 || (result = 1, v1 != 92) )
  {
    if ( !unk_4F07594 )
      return v1 == 47;
    if ( !isalpha(v1) || (result = 1, a1[1] != 58) )
    {
      if ( v1 != 92 )
        return v1 == 47;
      result = 1;
      if ( a1[1] != 92 )
        return v1 == 47;
    }
  }
  return result;
}
