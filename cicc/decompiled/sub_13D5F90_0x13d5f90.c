// Function: sub_13D5F90
// Address: 0x13d5f90
//
__int64 __fastcall sub_13D5F90(_QWORD **a1, __int64 a2)
{
  unsigned int v2; // r12d
  char v3; // al
  __int64 v6; // rax
  _BYTE *v7; // rdi
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 == 50 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( v6 )
    {
      **a1 = v6;
      v7 = *(_BYTE **)(a2 - 24);
      v8 = v7[16];
      if ( v8 == 13 )
        goto LABEL_8;
      LOBYTE(v2) = v8 <= 0x10u && *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16;
      if ( (_BYTE)v2 )
      {
        v9 = sub_15A1020(v7);
        if ( v9 )
        {
          if ( *(_BYTE *)(v9 + 16) == 13 )
          {
            *a1[1] = v9 + 24;
            return v2;
          }
        }
      }
    }
    return 0;
  }
  if ( v3 != 5 )
    return 0;
  if ( *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v10 )
    return 0;
  **a1 = v10;
  v7 = *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( v7[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
    {
      v11 = sub_15A1020(v7);
      if ( v11 )
      {
        if ( *(_BYTE *)(v11 + 16) == 13 )
        {
          v2 = 1;
          *a1[1] = v11 + 24;
          return v2;
        }
      }
    }
    return 0;
  }
LABEL_8:
  *a1[1] = v7 + 24;
  return 1;
}
