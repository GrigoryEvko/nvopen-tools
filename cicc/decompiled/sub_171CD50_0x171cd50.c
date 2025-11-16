// Function: sub_171CD50
// Address: 0x171cd50
//
bool __fastcall sub_171CD50(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  unsigned int v9; // r12d
  __int64 v10; // rax
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // r12d
  unsigned int v15; // r13d
  __int64 v16; // rax
  char v17; // dl
  unsigned int v18; // r14d
  unsigned int v19; // r13d
  int v20; // r12d
  __int64 v21; // rax
  char v22; // dl
  unsigned int v23; // r14d

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 35 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( !v6 )
      return 0;
    **a1 = v6;
    v7 = *(_QWORD *)(a2 - 24);
    v8 = *(_BYTE *)(v7 + 16);
    if ( v8 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 && v8 <= 0x10u )
      {
        v10 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *(_QWORD *)v7, a4);
        if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
        {
          v19 = 0;
          v20 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
          if ( v20 )
          {
            while ( 1 )
            {
              v21 = sub_15A0A60(v7, v19);
              if ( !v21 )
                break;
              v22 = *(_BYTE *)(v21 + 16);
              if ( v22 != 9 )
              {
                if ( v22 != 13 )
                  return 0;
                v23 = *(_DWORD *)(v21 + 32);
                if ( v23 <= 0x40 )
                {
                  if ( *(_QWORD *)(v21 + 24) != 1 )
                    return 0;
                }
                else if ( (unsigned int)sub_16A57B0(v21 + 24) != v23 - 1 )
                {
                  return 0;
                }
              }
              if ( v20 == ++v19 )
                return 1;
            }
            return 0;
          }
          return 1;
        }
        goto LABEL_13;
      }
      return 0;
    }
  }
  else
  {
    if ( v4 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 11 )
      return 0;
    v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v12 )
      return 0;
    **a1 = v12;
    v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v7 = *(_QWORD *)(a2 + 24 * (1 - v13));
    if ( *(_BYTE *)(v7 + 16) != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
      {
        v10 = sub_15A1020(*(_BYTE **)(a2 + 24 * (1 - v13)), a2, v13, a4);
        if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
        {
          v14 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
          if ( v14 )
          {
            v15 = 0;
            while ( 1 )
            {
              v16 = sub_15A0A60(v7, v15);
              if ( !v16 )
                break;
              v17 = *(_BYTE *)(v16 + 16);
              if ( v17 != 9 )
              {
                if ( v17 != 13 )
                  return 0;
                v18 = *(_DWORD *)(v16 + 32);
                if ( v18 <= 0x40 )
                {
                  if ( *(_QWORD *)(v16 + 24) != 1 )
                    return 0;
                }
                else if ( (unsigned int)sub_16A57B0(v16 + 24) != v18 - 1 )
                {
                  return 0;
                }
              }
              if ( v14 == ++v15 )
                return 1;
            }
            return 0;
          }
          return 1;
        }
LABEL_13:
        v11 = *(_DWORD *)(v10 + 32);
        if ( v11 <= 0x40 )
          return *(_QWORD *)(v10 + 24) == 1;
        else
          return v11 - 1 == (unsigned int)sub_16A57B0(v10 + 24);
      }
      return 0;
    }
  }
  v9 = *(_DWORD *)(v7 + 32);
  if ( v9 <= 0x40 )
    return *(_QWORD *)(v7 + 24) == 1;
  else
    return v9 - 1 == (unsigned int)sub_16A57B0(v7 + 24);
}
