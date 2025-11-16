// Function: sub_E345B0
// Address: 0xe345b0
//
__int64 __fastcall sub_E345B0(_BYTE *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return 3;
  v2 = 0x8000000000041LL;
  if ( !_bittest64(&v2, (unsigned int)(unsigned __int8)*a1 - 34) )
    return 3;
  result = (unsigned int)sub_B49240((__int64)a1) - 142;
  if ( (unsigned int)result > 2 )
    return 3;
  return result;
}
