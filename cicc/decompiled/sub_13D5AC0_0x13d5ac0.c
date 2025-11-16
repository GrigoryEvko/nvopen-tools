// Function: sub_13D5AC0
// Address: 0x13d5ac0
//
char __fastcall sub_13D5AC0(_QWORD *a1, __int64 a2)
{
  char v3; // al
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rcx
  char v11; // si
  __int64 v12; // r13
  char v13; // dl
  unsigned int v14; // ebx
  __int64 v15; // rax
  unsigned int v16; // r13d
  bool v17; // al
  int v18; // r14d
  unsigned int v19; // r15d
  __int64 v20; // rax
  char v21; // cl
  unsigned int v22; // esi
  bool v23; // al
  int v24; // [rsp+Ch] [rbp-34h]

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 != 52 )
  {
    if ( v3 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v10 = *(_QWORD *)(a2 - 24 * v9);
    v11 = *(_BYTE *)(v10 + 16);
    v12 = *(_QWORD *)(a2 + 24 * (1 - v9));
    v13 = *(_BYTE *)(v12 + 16);
    if ( v11 == 50 )
    {
      if ( *(_QWORD *)(v10 - 48) != *a1 && *a1 != *(_QWORD *)(v10 - 24) )
        goto LABEL_29;
    }
    else if ( v11 != 5
           || *(_WORD *)(v10 + 18) != 26
           || *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)) != *a1
           && *a1 != *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF))) )
    {
      goto LABEL_29;
    }
    if ( v13 == 13 )
    {
      v14 = *(_DWORD *)(v12 + 32);
      if ( v14 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) == *(_QWORD *)(v12 + 24);
      else
        return v14 == (unsigned int)sub_16A58F0(v12 + 24);
    }
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
    {
      v15 = sub_15A1020(*(_QWORD *)(a2 + 24 * (1 - v9)));
      if ( v15 && *(_BYTE *)(v15 + 16) == 13 )
      {
        v16 = *(_DWORD *)(v15 + 32);
        if ( v16 <= 0x40 )
          v17 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *(_QWORD *)(v15 + 24);
        else
          v17 = v16 == (unsigned int)sub_16A58F0(v15 + 24);
        if ( !v17 )
        {
LABEL_28:
          v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          v12 = *(_QWORD *)(a2 + 24 * (1 - v9));
          v13 = *(_BYTE *)(v12 + 16);
          goto LABEL_29;
        }
        return 1;
      }
      v18 = *(_QWORD *)(*(_QWORD *)v12 + 32LL);
      if ( !v18 )
        return 1;
      v19 = 0;
      while ( 1 )
      {
        v20 = sub_15A0A60(v12, v19);
        if ( !v20 )
          break;
        v21 = *(_BYTE *)(v20 + 16);
        if ( v21 != 9 )
        {
          if ( v21 != 13 )
            goto LABEL_28;
          v22 = *(_DWORD *)(v20 + 32);
          if ( v22 <= 0x40 )
          {
            v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *(_QWORD *)(v20 + 24);
          }
          else
          {
            v24 = *(_DWORD *)(v20 + 32);
            v23 = v24 == (unsigned int)sub_16A58F0(v20 + 24);
          }
          if ( !v23 )
            goto LABEL_28;
        }
        if ( v18 == ++v19 )
          return 1;
      }
      v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v12 = *(_QWORD *)(a2 + 24 * (1 - v9));
      v13 = *(_BYTE *)(v12 + 16);
    }
LABEL_29:
    if ( v13 == 50 )
    {
      if ( *(_QWORD *)(v12 - 48) == *a1 || *a1 == *(_QWORD *)(v12 - 24) )
        return sub_13CC690(*(_QWORD *)(a2 - 24 * v9));
    }
    else if ( v13 == 5
           && *(_WORD *)(v12 + 18) == 26
           && (*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)) == *a1
            || *a1 == *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)))) )
    {
      return sub_13CC690(*(_QWORD *)(a2 - 24 * v9));
    }
    return 0;
  }
  v5 = *(_QWORD *)(a2 - 48);
  v6 = *(_BYTE *)(v5 + 16);
  if ( v6 == 50 )
  {
    if ( *(_QWORD *)(v5 - 48) != *a1 && *a1 != *(_QWORD *)(v5 - 24) )
      goto LABEL_8;
  }
  else if ( v6 != 5
         || *(_WORD *)(v5 + 18) != 26
         || *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)) != *a1
         && *a1 != *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))) )
  {
    goto LABEL_8;
  }
  if ( (unsigned __int8)sub_13CC520(*(_QWORD *)(a2 - 24)) )
    return 1;
LABEL_8:
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(_BYTE *)(v7 + 16);
  if ( v8 == 50 )
  {
    if ( *(_QWORD *)(v7 - 48) != *a1 && *a1 != *(_QWORD *)(v7 - 24) )
      return 0;
  }
  else if ( v8 != 5
         || *(_WORD *)(v7 + 18) != 26
         || *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)) != *a1
         && *a1 != *(_QWORD *)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF))) )
  {
    return 0;
  }
  return sub_13CC520(*(_QWORD *)(a2 - 48));
}
