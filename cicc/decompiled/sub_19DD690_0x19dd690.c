// Function: sub_19DD690
// Address: 0x19dd690
//
__int64 __fastcall sub_19DD690(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5)
{
  char v5; // al
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  v5 = *(_BYTE *)(a3 + 16);
  if ( *(_BYTE *)(a2 + 16) == 35 )
  {
    if ( v5 != 35 )
    {
      if ( v5 != 5 || *(_WORD *)(a3 + 18) != 11 )
        return 0;
      goto LABEL_12;
    }
LABEL_6:
    v7 = *(_QWORD *)(a3 - 48);
    if ( !v7 )
      return 0;
    *a4 = v7;
    v8 = *(_QWORD *)(a3 - 24);
    if ( !v8 )
      return 0;
    *a5 = v8;
    return 1;
  }
  if ( v5 == 39 )
    goto LABEL_6;
  if ( v5 != 5 || *(_WORD *)(a3 + 18) != 15 )
    return 0;
LABEL_12:
  v9 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  if ( !v9 )
    return 0;
  *a4 = v9;
  v10 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  if ( !v10 )
    return 0;
  *a5 = v10;
  return 1;
}
