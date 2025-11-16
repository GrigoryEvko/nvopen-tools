// Function: sub_BCBCB0
// Address: 0xbcbcb0
//
unsigned __int64 __fastcall sub_BCBCB0(__int64 a1)
{
  unsigned __int64 v1; // rcx
  __int64 v2; // rdx
  unsigned __int64 result; // rax

  v1 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned __int8)v1 > 0xCu || (v2 = 4143, result = 1, !_bittest64(&v2, v1)) )
  {
    result = 0;
    if ( (unsigned __int8)v1 <= 0x13u )
      return (0x84050uLL >> v1) & 1;
  }
  return result;
}
