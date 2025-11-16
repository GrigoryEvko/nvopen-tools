// Function: sub_10FD2B0
// Address: 0x10fd2b0
//
__int64 __fastcall sub_10FD2B0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_BYTE **)(a2 - 32);
  if ( *v3 > 0x15u )
    return 0;
  *a1[1] = v3;
  result = 1;
  if ( *v3 <= 0x15u )
  {
    if ( *v3 != 5 )
      return (unsigned int)sub_AD6CA0((__int64)v3) ^ 1;
    return 0;
  }
  return result;
}
