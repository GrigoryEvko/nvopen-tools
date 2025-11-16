// Function: sub_25375B0
// Address: 0x25375b0
//
__int64 __fastcall sub_25375B0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 97);
  if ( (*(_BYTE *)(a1 + 97) & 3) == 3 )
  {
    result = (unsigned __int8)byte_4FEF4C8;
    if ( !byte_4FEF4C8 )
    {
      result = sub_2207590((__int64)&byte_4FEF4C8);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4C8);
    }
  }
  else if ( (result & 2) != 0 )
  {
    result = (unsigned __int8)byte_4FEF4C0;
    if ( !byte_4FEF4C0 )
    {
      result = sub_2207590((__int64)&byte_4FEF4C0);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4C0);
    }
  }
  else if ( (result & 1) != 0 )
  {
    result = (unsigned __int8)byte_4FEF4B8;
    if ( !byte_4FEF4B8 )
    {
      result = sub_2207590((__int64)&byte_4FEF4B8);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4B8);
    }
  }
  return result;
}
