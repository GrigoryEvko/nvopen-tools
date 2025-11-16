// Function: sub_1737FA0
// Address: 0x1737fa0
//
char __fastcall sub_1737FA0(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  char result; // al
  char v6; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  __int64 v11; // rdx

  v4 = *(_QWORD *)(a2 + 8);
  if ( !v4 || *(_QWORD *)(v4 + 8) )
    return 0;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 == 50 )
  {
    result = sub_171DA10(a1, *(_QWORD *)(a2 - 48), a3, a4);
    v10 = *(_QWORD *)(a2 - 24);
    if ( !result || !v10 )
    {
      result = sub_171DA10(a1, v10, v8, v9);
      if ( !result )
        return 0;
      v11 = *(_QWORD *)(a2 - 48);
      if ( !v11 )
        return 0;
LABEL_11:
      *a1[2] = v11;
      return result;
    }
  }
  else
  {
    if ( v6 != 5 || *(_WORD *)(a2 + 18) != 26 )
      return 0;
    result = sub_14B2B20(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !result )
    {
      v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      goto LABEL_16;
    }
    v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v10 )
    {
LABEL_16:
      result = sub_14B2B20(a1, v10);
      if ( !result )
        return 0;
      v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( !v11 )
        return 0;
      goto LABEL_11;
    }
  }
  *a1[2] = v10;
  return result;
}
