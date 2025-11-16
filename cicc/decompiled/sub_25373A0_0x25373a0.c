// Function: sub_25373A0
// Address: 0x25373a0
//
__int64 __fastcall sub_25373A0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 97);
  if ( (*(_BYTE *)(a1 + 97) & 3) == 3 )
  {
    result = (unsigned __int8)byte_4FEF498;
    if ( !byte_4FEF498 )
    {
      result = sub_2207590((__int64)&byte_4FEF498);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF498);
    }
  }
  else if ( (result & 2) != 0 )
  {
    result = (unsigned __int8)byte_4FEF490;
    if ( !byte_4FEF490 )
    {
      result = sub_2207590((__int64)&byte_4FEF490);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF490);
    }
  }
  else if ( (result & 1) != 0 )
  {
    result = (unsigned __int8)byte_4FEF488;
    if ( !byte_4FEF488 )
    {
      result = sub_2207590((__int64)&byte_4FEF488);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF488);
    }
  }
  return result;
}
