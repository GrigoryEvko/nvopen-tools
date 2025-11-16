// Function: sub_BC8730
// Address: 0xbc8730
//
__int64 __fastcall sub_BC8730(unsigned __int8 *a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rcx

  v1 = 0;
  if ( (a1[7] & 0x20) != 0 )
    v1 = sub_B91C10((__int64)a1, 2);
  result = sub_BC86C0(v1);
  if ( !(_BYTE)result && (unsigned __int8)(*a1 - 34) <= 0x33u )
  {
    v3 = 0x8000000000041LL;
    if ( _bittest64(&v3, (unsigned int)*a1 - 34) )
      return (unsigned int)sub_BC8680(v1) ^ 1;
  }
  return result;
}
