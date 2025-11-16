// Function: sub_1731230
// Address: 0x1731230
//
__int64 __fastcall sub_1731230(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v5; // rax
  char v7; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // rax

  v5 = *(_QWORD *)(a2 + 8);
  if ( !v5 || *(_QWORD *)(v5 + 8) )
    return 0;
  v7 = *(_BYTE *)(a2 + 16);
  if ( v7 == 51 )
  {
    v13 = *(_QWORD *)(a2 - 48);
    if ( !v13 )
      return 0;
    **a1 = v13;
    v11 = *(_BYTE **)(a2 - 24);
    v14 = v11[16];
    if ( v14 != 13 )
    {
      LOBYTE(v4) = v14 <= 0x10u && *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16;
      if ( (_BYTE)v4 )
      {
        v15 = sub_15A1020(v11, a2, *(_QWORD *)v11, a4);
        if ( v15 )
        {
          if ( *(_BYTE *)(v15 + 16) == 13 )
          {
            *a1[1] = v15 + 24;
            return v4;
          }
        }
      }
      return 0;
    }
LABEL_16:
    v4 = 1;
    *a1[1] = v11 + 24;
    return v4;
  }
  if ( v7 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 27 )
    return 0;
  v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v9 )
    return 0;
  **a1 = v9;
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(_BYTE **)(a2 + 24 * (1 - v10));
  if ( v11[16] == 13 )
    goto LABEL_16;
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
  {
    v12 = sub_15A1020(v11, a2, v10, a4);
    if ( v12 )
    {
      if ( *(_BYTE *)(v12 + 16) == 13 )
      {
        v4 = 1;
        *a1[1] = v12 + 24;
        return v4;
      }
    }
  }
  return 0;
}
