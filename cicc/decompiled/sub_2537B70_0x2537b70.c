// Function: sub_2537B70
// Address: 0x2537b70
//
__int64 __fastcall sub_2537B70(unsigned __int8 **a1, __int64 a2)
{
  unsigned __int8 **v2; // rdx
  __int64 result; // rax

  v2 = *(unsigned __int8 ***)(a2 + 24);
  result = 1;
  if ( *(_BYTE *)v2 == 62 )
  {
    result = **a1;
    if ( !(_BYTE)result )
      return sub_25282F0((__int64)a1[1], *(v2 - 4), (__int64)a1[2]);
  }
  return result;
}
