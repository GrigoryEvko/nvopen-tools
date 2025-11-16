// Function: sub_1731E70
// Address: 0x1731e70
//
__int64 __fastcall sub_1731E70(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rcx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 51 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    v5 = *(_QWORD *)(a2 - 24);
    if ( v4 == *(_QWORD *)a1 && v5 )
      goto LABEL_14;
    if ( !v4 || *(_QWORD *)a1 != v5 )
      return 0;
LABEL_8:
    **(_QWORD **)(a1 + 8) = v4;
    return 1;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 27 )
    return 0;
  v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_QWORD *)a1 == v5 && v4 )
    goto LABEL_8;
  if ( !v5 || *(_QWORD *)a1 != v4 )
    return 0;
LABEL_14:
  **(_QWORD **)(a1 + 8) = v5;
  return 1;
}
