// Function: sub_2537710
// Address: 0x2537710
//
__int64 __fastcall sub_2537710(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 100);
  if ( (unsigned __int8)v1 == 255 || (result = (unsigned __int8)v1 & 0xFC, (_DWORD)result == 252) )
  {
    result = (unsigned __int8)byte_4FEF448;
    if ( !byte_4FEF448 )
    {
      result = sub_2207590((__int64)&byte_4FEF448);
      if ( (_DWORD)result )
        return sub_2207640((__int64)&byte_4FEF448);
    }
  }
  return result;
}
