// Function: sub_E8ACF0
// Address: 0xe8acf0
//
__int64 __fastcall sub_E8ACF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_E98E00();
  if ( (_DWORD)result )
  {
    if ( !*(_BYTE *)(a1 + 304) )
    {
      if ( !*(_BYTE *)(a1 + 305) )
        return result;
      return sub_E7C210((__int64 *)a1, a2, 0);
    }
    result = sub_E7C210((__int64 *)a1, a2, 1u);
    if ( *(_BYTE *)(a1 + 305) )
      return sub_E7C210((__int64 *)a1, a2, 0);
  }
  return result;
}
