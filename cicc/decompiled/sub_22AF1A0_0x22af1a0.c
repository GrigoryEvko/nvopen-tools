// Function: sub_22AF1A0
// Address: 0x22af1a0
//
__int64 __fastcall sub_22AF1A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(_WORD *)(a1 + 2) & 0x3F;
  if ( (*(_WORD *)(a1 + 2) & 0x3Fu) <= 0x27 )
  {
    v2 = 0xCC00000C0CLL;
    if ( _bittest64(&v2, result) )
      return sub_B52F50(result);
  }
  return result;
}
