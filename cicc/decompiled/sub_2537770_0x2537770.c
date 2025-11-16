// Function: sub_2537770
// Address: 0x2537770
//
__int64 __fastcall sub_2537770(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 100);
  if ( (unsigned __int8)v1 == 255 || (*(_DWORD *)(a1 + 100) & 0xFC) == 0xFC )
  {
    result = (unsigned __int8)byte_4FEF468;
    if ( !byte_4FEF468 )
    {
      result = sub_2207590((__int64)&byte_4FEF468);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF468);
    }
  }
  else if ( (*(_DWORD *)(a1 + 100) & 0xEC) == 0xEC )
  {
    result = (unsigned __int8)byte_4FEF460;
    if ( !byte_4FEF460 )
    {
      result = sub_2207590((__int64)&byte_4FEF460);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF460);
    }
  }
  else if ( (*(_DWORD *)(a1 + 100) & 0xDC) == 0xDC )
  {
    result = (unsigned __int8)byte_4FEF458;
    if ( !byte_4FEF458 )
    {
      result = sub_2207590((__int64)&byte_4FEF458);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF458);
    }
  }
  else
  {
    result = (unsigned __int8)v1 & 0xCC;
    if ( (_DWORD)result == 204 )
    {
      result = (unsigned __int8)byte_4FEF450;
      if ( !byte_4FEF450 )
      {
        result = sub_2207590((__int64)&byte_4FEF450);
        if ( (_DWORD)result )
          return sub_2207640((__int64)&byte_4FEF450);
      }
    }
  }
  return result;
}
