// Function: sub_F06FC0
// Address: 0xf06fc0
//
_BOOL8 __fastcall sub_F06FC0(unsigned __int8 *a1)
{
  unsigned __int64 v1; // rdx
  _BOOL8 result; // rax
  __int64 v3; // rcx

  v1 = *a1;
  result = 0;
  if ( (unsigned __int8)v1 <= 0x36u )
  {
    v3 = 0x40540000000000LL;
    if ( _bittest64(&v3, v1) )
      return (a1[1] & 2) != 0;
  }
  return result;
}
