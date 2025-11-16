// Function: sub_2536F30
// Address: 0x2536f30
//
__int64 __fastcall sub_2536F30(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 184) )
  {
    result = (unsigned __int8)byte_4FEF5C0;
    if ( !byte_4FEF5C0 )
    {
      result = sub_2207590((__int64)&byte_4FEF5C0);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF5C0);
    }
  }
  else
  {
    result = (unsigned __int8)byte_4FEF5B8;
    if ( !byte_4FEF5B8 )
    {
      result = sub_2207590((__int64)&byte_4FEF5B8);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF5B8);
    }
  }
  return result;
}
