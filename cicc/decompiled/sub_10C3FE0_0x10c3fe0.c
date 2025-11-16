// Function: sub_10C3FE0
// Address: 0x10c3fe0
//
__int64 __fastcall sub_10C3FE0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 58 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( !v4 )
    return 0;
  **a1 = v4;
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 )
    return 0;
  *a1[1] = v5;
  return 1;
}
