// Function: sub_1735440
// Address: 0x1735440
//
char __fastcall sub_1735440(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 50 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( !v6 || (v7 = *a1, *v7 = v6, !sub_171DA10(a1 + 1, *(_QWORD *)(a2 - 24), (__int64)v7, a4)) )
    {
      v8 = *(_QWORD *)(a2 - 24);
      if ( v8 )
      {
        v9 = *a1;
        **a1 = v8;
        return sub_171DA10(a1 + 1, *(_QWORD *)(a2 - 48), (__int64)v9, a4);
      }
      return 0;
    }
    return 1;
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(_QWORD *)(a2 - 24 * v10);
  if ( v11 )
  {
    **a1 = v11;
    if ( !sub_14B2B20(a1 + 1, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    {
      v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      goto LABEL_13;
    }
    return 1;
  }
LABEL_13:
  v12 = *(_QWORD *)(a2 + 24 * (1 - v10));
  if ( !v12 )
    return 0;
  **a1 = v12;
  return sub_14B2B20(a1 + 1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
}
