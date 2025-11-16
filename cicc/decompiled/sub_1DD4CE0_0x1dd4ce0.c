// Function: sub_1DD4CE0
// Address: 0x1dd4ce0
//
__int64 __fastcall sub_1DD4CE0(__int64 a1)
{
  __int64 result; // rax
  __int16 v2; // dx

  result = *(unsigned __int16 *)(a1 + 46);
  v2 = *(_WORD *)(a1 + 46) & 4;
  if ( (result & 8) != 0 )
  {
    if ( !v2 )
    {
      sub_1E16440();
      result = *(unsigned __int16 *)(a1 + 46);
      if ( (result & 4) != 0 && (result & 8) == 0 )
        return sub_1E16420();
    }
  }
  else if ( v2 )
  {
    return sub_1E16420();
  }
  return result;
}
