// Function: sub_13D52E0
// Address: 0x13d52e0
//
bool __fastcall sub_13D52E0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // r13
  unsigned __int8 v5; // al
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned int v12; // r14d
  bool v13; // al
  __int64 v14; // rax
  unsigned int v15; // r13d
  unsigned int v16; // r15d
  int v17; // r14d
  __int64 v18; // rax
  char v19; // dl
  unsigned int v20; // r15d
  int v21; // r14d
  __int64 v22; // rax
  char v23; // dl
  bool v24; // al
  int v25; // [rsp+Ch] [rbp-34h]
  int v26; // [rsp+Ch] [rbp-34h]

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 37 )
  {
    v4 = *(_QWORD *)(a2 - 48);
    v5 = *(_BYTE *)(v4 + 16);
    if ( v5 == 13 )
    {
      v6 = *(_DWORD *)(v4 + 32);
      if ( v6 <= 0x40 )
        v7 = *(_QWORD *)(v4 + 24) == 0;
      else
        v7 = v6 == (unsigned int)sub_16A57B0(v4 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 16 || v5 > 0x10u )
        return 0;
      v8 = sub_15A1020(*(_QWORD *)(a2 - 48));
      if ( !v8 || *(_BYTE *)(v8 + 16) != 13 )
      {
        v16 = 0;
        v17 = *(_QWORD *)(*(_QWORD *)v4 + 32LL);
        if ( v17 )
        {
          while ( 1 )
          {
            v18 = sub_15A0A60(v4, v16);
            if ( !v18 )
              break;
            v19 = *(_BYTE *)(v18 + 16);
            if ( v19 != 9 )
            {
              if ( v19 != 13 )
                return 0;
              if ( *(_DWORD *)(v18 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v18 + 24) )
                  return 0;
              }
              else
              {
                v25 = *(_DWORD *)(v18 + 32);
                if ( v25 != (unsigned int)sub_16A57B0(v18 + 24) )
                  return 0;
              }
            }
            if ( v17 == ++v16 )
              return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
          }
          return 0;
        }
        return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
      }
      v9 = *(_DWORD *)(v8 + 32);
      if ( v9 <= 0x40 )
        v7 = *(_QWORD *)(v8 + 24) == 0;
      else
        v7 = v9 == (unsigned int)sub_16A57B0(v8 + 24);
    }
    if ( !v7 )
      return 0;
    return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 13 )
    return 0;
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(_QWORD *)(a2 - 24 * v10);
  if ( *(_BYTE *)(v11 + 16) == 13 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 > 0x40 )
    {
      v13 = v12 == (unsigned int)sub_16A57B0(v11 + 24);
      goto LABEL_19;
    }
    if ( !*(_QWORD *)(v11 + 24) )
      return *(_QWORD *)(a2 + 24 * (1 - v10)) == *(_QWORD *)(a1 + 8);
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
    return 0;
  v14 = sub_15A1020(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v14 || *(_BYTE *)(v14 + 16) != 13 )
  {
    v20 = 0;
    v21 = *(_QWORD *)(*(_QWORD *)v11 + 32LL);
    if ( v21 )
    {
      while ( 1 )
      {
        v22 = sub_15A0A60(v11, v20);
        if ( !v22 )
          return 0;
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 != 9 )
        {
          if ( v23 != 13 )
            return 0;
          if ( *(_DWORD *)(v22 + 32) <= 0x40u )
          {
            v24 = *(_QWORD *)(v22 + 24) == 0;
          }
          else
          {
            v26 = *(_DWORD *)(v22 + 32);
            v24 = v26 == (unsigned int)sub_16A57B0(v22 + 24);
          }
          if ( !v24 )
            return 0;
        }
        if ( v21 == ++v20 )
          goto LABEL_20;
      }
    }
    goto LABEL_20;
  }
  v15 = *(_DWORD *)(v14 + 32);
  if ( v15 <= 0x40 )
    v13 = *(_QWORD *)(v14 + 24) == 0;
  else
    v13 = v15 == (unsigned int)sub_16A57B0(v14 + 24);
LABEL_19:
  if ( !v13 )
    return 0;
LABEL_20:
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  return *(_QWORD *)(a2 + 24 * (1 - v10)) == *(_QWORD *)(a1 + 8);
}
