// Function: sub_2537450
// Address: 0x2537450
//
__int64 __fastcall sub_2537450(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 97);
  if ( (*(_BYTE *)(a1 + 97) & 3) == 3 )
  {
    result = (unsigned __int8)byte_4FEF480;
    if ( !byte_4FEF480 )
    {
      result = sub_2207590((__int64)&byte_4FEF480);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF480);
    }
  }
  else if ( (result & 2) != 0 )
  {
    result = (unsigned __int8)byte_4FEF478;
    if ( !byte_4FEF478 )
    {
      result = sub_2207590((__int64)&byte_4FEF478);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF478);
    }
  }
  else if ( (result & 1) != 0 )
  {
    result = (unsigned __int8)byte_4FEF470;
    if ( !byte_4FEF470 )
    {
      result = sub_2207590((__int64)&byte_4FEF470);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF470);
    }
  }
  return result;
}
