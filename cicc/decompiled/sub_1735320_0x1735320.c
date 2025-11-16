// Function: sub_1735320
// Address: 0x1735320
//
char __fastcall sub_1735320(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  char result; // al
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 52 )
  {
    result = sub_171DA10(a1, *(_QWORD *)(a2 - 48), a3, a4);
    v9 = *(_QWORD *)(a2 - 24);
    if ( !result || !v9 )
    {
      result = sub_171DA10(a1, v9, v7, v8);
      if ( !result )
        return 0;
      v10 = *(_QWORD *)(a2 - 48);
      if ( !v10 )
        return 0;
LABEL_8:
      *a1[2] = v10;
      return result;
    }
  }
  else
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    result = sub_14B2B20(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !result )
    {
      v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      goto LABEL_13;
    }
    v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v9 )
    {
LABEL_13:
      result = sub_14B2B20(a1, v9);
      if ( !result )
        return 0;
      v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( !v10 )
        return 0;
      goto LABEL_8;
    }
  }
  *a1[2] = v9;
  return result;
}
