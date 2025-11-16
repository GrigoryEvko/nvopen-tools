// Function: sub_10A4EC0
// Address: 0x10a4ec0
//
__int64 __fastcall sub_10A4EC0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rax
  _BYTE *v5; // r8

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 42 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
  **a1 = v4;
  v5 = *(_BYTE **)(a2 - 32);
  if ( *v5 > 0x15u )
    return 0;
  *a1[1] = v5;
  result = 1;
  if ( *v5 > 0x15u )
    return result;
  if ( *v5 == 5 )
    return 0;
  return (unsigned int)sub_AD6CA0((__int64)v5) ^ 1;
}
