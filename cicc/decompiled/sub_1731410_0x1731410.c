// Function: sub_1731410
// Address: 0x1731410
//
__int64 __fastcall sub_1731410(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 51 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    if ( !v7 )
      return 0;
    **a1 = v7;
    v6 = *(_QWORD *)(a2 - 24);
    if ( !v6 )
      return 0;
  }
  else
  {
    if ( v4 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 27 )
      return 0;
    v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v5 )
      return 0;
    **a1 = v5;
    v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v6 )
      return 0;
  }
  *a1[1] = v6;
  return 1;
}
