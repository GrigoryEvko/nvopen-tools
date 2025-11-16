// Function: sub_2537500
// Address: 0x2537500
//
__int64 __fastcall sub_2537500(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 97);
  if ( (*(_BYTE *)(a1 + 97) & 3) == 3 )
  {
    result = (unsigned __int8)byte_4FEF4E0;
    if ( !byte_4FEF4E0 )
    {
      result = sub_2207590((__int64)&byte_4FEF4E0);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4E0);
    }
  }
  else if ( (result & 2) != 0 )
  {
    result = (unsigned __int8)byte_4FEF4D8;
    if ( !byte_4FEF4D8 )
    {
      result = sub_2207590((__int64)&byte_4FEF4D8);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4D8);
    }
  }
  else if ( (result & 1) != 0 )
  {
    result = (unsigned __int8)byte_4FEF4D0;
    if ( !byte_4FEF4D0 )
    {
      result = sub_2207590((__int64)&byte_4FEF4D0);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF4D0);
    }
  }
  return result;
}
