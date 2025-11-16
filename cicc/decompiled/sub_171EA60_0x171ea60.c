// Function: sub_171EA60
// Address: 0x171ea60
//
__int64 __fastcall sub_171EA60(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  char v5; // al
  __int64 v8; // rax
  _BYTE *v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 47 )
  {
    v8 = *(_QWORD *)(a2 - 48);
    if ( v8 )
    {
      **a1 = v8;
      v9 = *(_BYTE **)(a2 - 24);
      v10 = v9[16];
      if ( v10 == 13 )
        goto LABEL_8;
      LOBYTE(v4) = v10 <= 0x10u && *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16;
      if ( (_BYTE)v4 )
      {
        v11 = sub_15A1020(v9, a2, *(_QWORD *)v9, a4);
        if ( v11 )
        {
          if ( *(_BYTE *)(v11 + 16) == 13 )
          {
            *a1[1] = v11 + 24;
            return v4;
          }
        }
      }
    }
    return 0;
  }
  if ( v5 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 23 )
    return 0;
  v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v12 )
    return 0;
  **a1 = v12;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v9 = *(_BYTE **)(a2 + 24 * (1 - v13));
  if ( v9[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
    {
      v14 = sub_15A1020(v9, a2, v13, a4);
      if ( v14 )
      {
        if ( *(_BYTE *)(v14 + 16) == 13 )
        {
          v4 = 1;
          *a1[1] = v14 + 24;
          return v4;
        }
      }
    }
    return 0;
  }
LABEL_8:
  *a1[1] = v9 + 24;
  return 1;
}
