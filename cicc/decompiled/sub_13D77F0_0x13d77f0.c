// Function: sub_13D77F0
// Address: 0x13d77f0
//
char __fastcall sub_13D77F0(__int64 *a1, __int64 a2)
{
  char v3; // al
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rbx
  unsigned int v10; // r12d
  unsigned int v11; // r13d
  bool v12; // al
  __int64 v13; // rax
  unsigned int v14; // r12d
  bool v15; // al
  __int64 v16; // rax
  unsigned int v17; // ebx
  int v18; // r12d
  unsigned int v19; // r14d
  __int64 v20; // rax
  char v21; // cl
  unsigned int v22; // esi
  int v23; // r12d
  unsigned int v24; // r14d
  __int64 v25; // rax
  char v26; // dl
  unsigned int v27; // r15d
  int v28; // [rsp+Ch] [rbp-34h]

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 == 52 )
  {
    v5 = *a1;
    if ( *a1 == *(_QWORD *)(a2 - 48) )
    {
      if ( (unsigned __int8)sub_13CC520(*(_QWORD *)(a2 - 24)) )
        return 1;
      v5 = *a1;
    }
    if ( v5 == *(_QWORD *)(a2 - 24) )
      return sub_13CC520(*(_QWORD *)(a2 - 48));
    return 0;
  }
  if ( v3 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v6 = *a1;
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
  if ( *a1 != *(_QWORD *)(a2 - 24 * v7) )
    goto LABEL_9;
  if ( *(_BYTE *)(v8 + 16) != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 )
      goto LABEL_9;
    v13 = sub_15A1020(*(_QWORD *)(a2 + 24 * (1 - v7)));
    if ( v13 && *(_BYTE *)(v13 + 16) == 13 )
    {
      v14 = *(_DWORD *)(v13 + 32);
      if ( v14 <= 0x40 )
        v15 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) == *(_QWORD *)(v13 + 24);
      else
        v15 = v14 == (unsigned int)sub_16A58F0(v13 + 24);
      if ( v15 )
        return 1;
LABEL_26:
      v6 = *a1;
      v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
      goto LABEL_9;
    }
    v18 = *(_QWORD *)(*(_QWORD *)v8 + 32LL);
    if ( v18 )
    {
      v19 = 0;
      while ( 1 )
      {
        v20 = sub_15A0A60(v8, v19);
        if ( !v20 )
          break;
        v21 = *(_BYTE *)(v20 + 16);
        if ( v21 != 9 )
        {
          if ( v21 != 13 )
            goto LABEL_26;
          v22 = *(_DWORD *)(v20 + 32);
          if ( v22 <= 0x40 )
          {
            if ( *(_QWORD *)(v20 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) )
              break;
          }
          else
          {
            v28 = *(_DWORD *)(v20 + 32);
            if ( v28 != (unsigned int)sub_16A58F0(v20 + 24) )
              break;
          }
        }
        if ( v18 == ++v19 )
          return 1;
      }
      v6 = *a1;
      v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
      goto LABEL_9;
    }
    return 1;
  }
  v11 = *(_DWORD *)(v8 + 32);
  if ( v11 <= 0x40 )
    v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)(v8 + 24);
  else
    v12 = v11 == (unsigned int)sub_16A58F0(v8 + 24);
  if ( v12 )
    return 1;
LABEL_9:
  if ( v6 != v8 )
    return 0;
  v9 = *(_QWORD *)(a2 - 24 * v7);
  if ( *(_BYTE *)(v9 + 16) == 13 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *(_QWORD *)(v9 + 24);
    else
      return v10 == (unsigned int)sub_16A58F0(v9 + 24);
  }
  if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
    return 0;
  v16 = sub_15A1020(v9);
  if ( !v16 || *(_BYTE *)(v16 + 16) != 13 )
  {
    v23 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
    if ( v23 )
    {
      v24 = 0;
      while ( 1 )
      {
        v25 = sub_15A0A60(v9, v24);
        if ( !v25 )
          break;
        v26 = *(_BYTE *)(v25 + 16);
        if ( v26 != 9 )
        {
          if ( v26 != 13 )
            return 0;
          v27 = *(_DWORD *)(v25 + 32);
          if ( v27 <= 0x40 )
          {
            if ( *(_QWORD *)(v25 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v27) )
              return 0;
          }
          else if ( v27 != (unsigned int)sub_16A58F0(v25 + 24) )
          {
            return 0;
          }
        }
        if ( v23 == ++v24 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) == *(_QWORD *)(v16 + 24);
  else
    return v17 == (unsigned int)sub_16A58F0(v16 + 24);
}
