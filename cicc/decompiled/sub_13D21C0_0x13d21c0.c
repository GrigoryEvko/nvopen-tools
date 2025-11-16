// Function: sub_13D21C0
// Address: 0x13d21c0
//
char __fastcall sub_13D21C0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned int v13; // r8d
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 v17; // rsi
  int v18; // r12d
  unsigned int v19; // r14d
  __int64 v20; // rax
  char v21; // dl
  unsigned int v22; // edx
  __int64 v23; // rdi
  unsigned int v24; // r15d
  unsigned int v25; // r14d
  int v26; // r12d
  __int64 v27; // rax
  char v28; // dl
  unsigned int v29; // edx
  __int64 v30; // rdi
  unsigned int v31; // r15d

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 52 )
  {
    if ( v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v14 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v14 )
      return 0;
    **a1 = v14;
    v15 = 1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v5 = *(_QWORD *)(a2 + 24 * v15);
    if ( *(_BYTE *)(v5 + 16) == 13 )
    {
      v16 = *(_DWORD *)(v5 + 32);
      v17 = *(_QWORD *)(v5 + 24);
      v9 = v16 - 1;
      if ( v16 > 0x40 )
      {
        if ( (*(_QWORD *)(v17 + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
          return (unsigned int)sub_16A58A0(v5 + 24) == v9;
        return 0;
      }
      return v17 == 1LL << v9;
    }
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
      return 0;
    v10 = sub_15A1020(*(_QWORD *)(a2 + 24 * v15));
    if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)v5 + 32LL);
      if ( v18 )
      {
        v19 = 0;
        while ( 1 )
        {
          v20 = sub_15A0A60(v5, v19);
          if ( !v20 )
            break;
          v21 = *(_BYTE *)(v20 + 16);
          if ( v21 != 9 )
          {
            if ( v21 != 13 )
              return 0;
            v22 = *(_DWORD *)(v20 + 32);
            v23 = *(_QWORD *)(v20 + 24);
            v24 = v22 - 1;
            if ( v22 <= 0x40 )
            {
              if ( v23 != 1LL << v24 )
                return 0;
            }
            else if ( (*(_QWORD *)(v23 + 8LL * (v24 >> 6)) & (1LL << v24)) == 0
                   || (unsigned int)sub_16A58A0(v20 + 24) != v24 )
            {
              return 0;
            }
          }
          if ( v18 == ++v19 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    return sub_13CFF40((__int64 *)(v10 + 24), a2, v11, v12, v13);
  }
  v4 = *(_QWORD *)(a2 - 48);
  if ( !v4 )
    return 0;
  **a1 = v4;
  v5 = *(_QWORD *)(a2 - 24);
  v6 = *(_BYTE *)(v5 + 16);
  if ( v6 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 || v6 > 0x10u )
      return 0;
    v10 = sub_15A1020(*(_QWORD *)(a2 - 24));
    if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
    {
      v25 = 0;
      v26 = *(_QWORD *)(*(_QWORD *)v5 + 32LL);
      if ( v26 )
      {
        while ( 1 )
        {
          v27 = sub_15A0A60(v5, v25);
          if ( !v27 )
            break;
          v28 = *(_BYTE *)(v27 + 16);
          if ( v28 != 9 )
          {
            if ( v28 != 13 )
              return 0;
            v29 = *(_DWORD *)(v27 + 32);
            v30 = *(_QWORD *)(v27 + 24);
            v31 = v29 - 1;
            if ( v29 <= 0x40 )
            {
              if ( v30 != 1LL << v31 )
                return 0;
            }
            else if ( (*(_QWORD *)(v30 + 8LL * (v31 >> 6)) & (1LL << v31)) == 0
                   || v31 != (unsigned int)sub_16A58A0(v27 + 24) )
            {
              return 0;
            }
          }
          if ( v26 == ++v25 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    return sub_13CFF40((__int64 *)(v10 + 24), a2, v11, v12, v13);
  }
  v7 = *(_DWORD *)(v5 + 32);
  v8 = *(_QWORD *)(v5 + 24);
  v9 = v7 - 1;
  if ( v7 > 0x40 )
  {
    if ( (*(_QWORD *)(v8 + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
      return (unsigned int)sub_16A58A0(v5 + 24) == v9;
    return 0;
  }
  return v8 == 1LL << v9;
}
