// Function: sub_17317C0
// Address: 0x17317c0
//
__int64 __fastcall sub_17317C0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    if ( v4 )
    {
      **a1 = v4;
      v5 = *(_QWORD *)(a2 - 24);
      if ( v5 )
      {
        *a1[1] = v5;
        return 1;
      }
    }
    return 0;
  }
  if ( v2 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v6 )
    return 0;
  **a1 = v6;
  v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v7 )
    return 0;
  *a1[1] = v7;
  return 1;
}
