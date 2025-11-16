// Function: sub_1D139C0
// Address: 0x1d139c0
//
__int64 __fastcall sub_1D139C0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax

  v2 = *(unsigned int *)(a2 + 516);
  if ( (unsigned int)v2 <= 0x1E )
  {
    v3 = 1610614920;
    if ( _bittest64(&v3, v2) )
      return sub_1560180(a1 + 112, 17);
  }
  result = sub_1560180(a1 + 112, 34);
  if ( !(_BYTE)result )
    return sub_1560180(a1 + 112, 17);
  return result;
}
