// Function: sub_2537160
// Address: 0x2537160
//
__int64 __fastcall sub_2537160(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 296) )
  {
    result = (unsigned __int8)byte_4FEF358;
    if ( !byte_4FEF358 )
    {
      result = sub_2207590((__int64)&byte_4FEF358);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF358);
    }
  }
  else
  {
    result = (unsigned __int8)byte_4FEF350;
    if ( !byte_4FEF350 )
    {
      result = sub_2207590((__int64)&byte_4FEF350);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF350);
    }
  }
  return result;
}
