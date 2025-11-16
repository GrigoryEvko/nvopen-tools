// Function: sub_176E260
// Address: 0x176e260
//
bool __fastcall sub_176E260(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  char v5; // al
  __int64 v7; // r14
  char v8; // al
  __int64 v9; // r13
  __int64 v10; // r15
  bool v11; // al
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int *v15; // r12
  unsigned int v16; // r13d
  __int64 v17; // rax
  __int64 *v18; // r15
  unsigned __int8 v19; // al
  unsigned int v20; // r13d
  bool v21; // al
  unsigned __int8 v22; // al
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // r13d
  bool v27; // al
  unsigned int v28; // r13d
  __int64 v29; // rax
  unsigned int v30; // r13d
  __int64 v31; // rax
  int v32; // eax
  bool v33; // al
  unsigned int v34; // [rsp+8h] [rbp-38h]
  int v35; // [rsp+Ch] [rbp-34h]
  int v36; // [rsp+Ch] [rbp-34h]
  int v37; // [rsp+Ch] [rbp-34h]

  v4 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 48 )
  {
    if ( v5 != 5 )
      return 0;
    if ( *(_WORD *)(v4 + 18) != 24 )
      return 0;
    v13 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    a2 = *(_QWORD *)(v4 - 24 * v13);
    if ( !(unsigned __int8)sub_17252E0(a1 + 16, a2, 4 * v13, a4) )
      return 0;
    v14 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    v15 = *(unsigned int **)(v4 + 24 * (1 - v14));
    if ( v15 )
    {
      if ( *((_BYTE *)v15 + 16) == 13 )
        goto LABEL_17;
      if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 )
        return 0;
      goto LABEL_29;
    }
LABEL_42:
    BUG();
  }
  v7 = *(_QWORD *)(v4 - 48);
  v8 = *(_BYTE *)(v7 + 16);
  if ( v8 != 37 )
  {
    if ( v8 != 5 || *(_WORD *)(v7 + 18) != 13 )
      return 0;
    v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    v10 = *(_QWORD *)(v7 - 24 * v9);
    if ( *(_BYTE *)(v10 + 16) == 13 )
    {
      if ( *(_DWORD *)(v10 + 32) <= 0x40u )
      {
        v11 = *(_QWORD *)(v10 + 24) == 0;
      }
      else
      {
        v35 = *(_DWORD *)(v10 + 32);
        v11 = v35 == (unsigned int)sub_16A57B0(v10 + 24);
      }
      if ( !v11 )
        return 0;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
        return 0;
      v25 = sub_15A1020(*(_BYTE **)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)), a2, 4 * v9, a4);
      if ( v25 && *(_BYTE *)(v25 + 16) == 13 )
      {
        v26 = *(_DWORD *)(v25 + 32);
        if ( v26 <= 0x40 )
          v27 = *(_QWORD *)(v25 + 24) == 0;
        else
          v27 = v26 == (unsigned int)sub_16A57B0(v25 + 24);
        if ( !v27 )
          return 0;
      }
      else
      {
        v37 = *(_QWORD *)(*(_QWORD *)v10 + 32LL);
        if ( v37 )
        {
          v30 = 0;
          do
          {
            a2 = v30;
            v31 = sub_15A0A60(v10, v30);
            if ( !v31 )
              return 0;
            a4 = *(unsigned __int8 *)(v31 + 16);
            if ( (_BYTE)a4 != 9 )
            {
              if ( (_BYTE)a4 != 13 )
                return 0;
              a4 = *(unsigned int *)(v31 + 32);
              if ( (unsigned int)a4 <= 0x40 )
              {
                v33 = *(_QWORD *)(v31 + 24) == 0;
              }
              else
              {
                v34 = *(_DWORD *)(v31 + 32);
                v32 = sub_16A57B0(v31 + 24);
                a4 = v34;
                v33 = v34 == v32;
              }
              if ( !v33 )
                return 0;
            }
          }
          while ( v37 != ++v30 );
        }
      }
      v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    }
    v12 = *(_QWORD *)(v7 + 24 * (1 - v9));
    if ( !v12 )
      return 0;
    goto LABEL_25;
  }
  v18 = *(__int64 **)(v7 - 48);
  v19 = *((_BYTE *)v18 + 16);
  if ( v19 == 13 )
  {
    v20 = *((_DWORD *)v18 + 8);
    if ( v20 <= 0x40 )
      v21 = v18[3] == 0;
    else
      v21 = v20 == (unsigned int)sub_16A57B0((__int64)(v18 + 3));
LABEL_23:
    if ( !v21 )
      return 0;
    goto LABEL_24;
  }
  if ( *(_BYTE *)(*v18 + 8) != 16 || v19 > 0x10u )
    return 0;
  v24 = sub_15A1020(*(_BYTE **)(v7 - 48), a2, *v18, a4);
  if ( v24 && *(_BYTE *)(v24 + 16) == 13 )
  {
    v21 = sub_13D01C0(v24 + 24);
    goto LABEL_23;
  }
  v36 = *(_QWORD *)(*v18 + 32);
  if ( v36 )
  {
    v28 = 0;
    do
    {
      a2 = v28;
      v29 = sub_15A0A60((__int64)v18, v28);
      if ( !v29 )
        return 0;
      a4 = *(unsigned __int8 *)(v29 + 16);
      if ( (_BYTE)a4 != 9 && ((_BYTE)a4 != 13 || !sub_13D01C0(v29 + 24)) )
        return 0;
    }
    while ( v36 != ++v28 );
  }
LABEL_24:
  v12 = *(_QWORD *)(v7 - 24);
  if ( !v12 )
    return 0;
LABEL_25:
  **(_QWORD **)(a1 + 24) = v12;
  v15 = *(unsigned int **)(v4 - 24);
  if ( !v15 )
    goto LABEL_42;
  v22 = *((_BYTE *)v15 + 16);
  if ( v22 == 13 )
    goto LABEL_17;
  v14 = *(_QWORD *)v15;
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) != 16 || v22 > 0x10u )
    return 0;
LABEL_29:
  v23 = sub_15A1020(v15, a2, v14, a4);
  v15 = (unsigned int *)v23;
  if ( !v23 || *(_BYTE *)(v23 + 16) != 13 )
    return 0;
LABEL_17:
  v16 = v15[8];
  if ( v16 > 0x40 )
  {
    if ( v16 - (unsigned int)sub_16A57B0((__int64)(v15 + 6)) <= 0x40 )
    {
      v17 = **((_QWORD **)v15 + 3);
      return *(_QWORD *)(a1 + 32) == v17;
    }
    return 0;
  }
  v17 = *((_QWORD *)v15 + 3);
  return *(_QWORD *)(a1 + 32) == v17;
}
