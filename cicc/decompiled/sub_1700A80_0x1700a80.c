// Function: sub_1700A80
// Address: 0x1700a80
//
bool __fastcall sub_1700A80(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v4; // dl
  char v5; // al
  __int16 v6; // ax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v12; // r13
  __int64 v13; // r15
  unsigned int *v14; // rdx
  unsigned __int8 v15; // al
  unsigned int v16; // r15d
  bool v17; // al
  __int64 v18; // rdx
  int v19; // eax
  _BYTE *v20; // rcx
  unsigned int v21; // r13d
  bool v22; // al
  __int64 v23; // rdi
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // r13
  unsigned int v27; // r15d
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // r13d
  unsigned int v32; // r13d
  __int64 v33; // rax
  int v34; // eax
  bool v35; // al
  __int64 v36; // rcx
  unsigned int v37; // r13d
  __int64 v38; // rax
  int v39; // eax
  bool v40; // al
  int v41; // [rsp+0h] [rbp-40h]
  int v42; // [rsp+0h] [rbp-40h]
  int v43; // [rsp+4h] [rbp-3Ch]
  int v44; // [rsp+4h] [rbp-3Ch]
  _BYTE *v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]
  __int64 v47; // [rsp+8h] [rbp-38h]
  __int64 v48; // [rsp+8h] [rbp-38h]

  v2 = a2;
  v4 = *(_BYTE *)(a2 + 24);
LABEL_2:
  v5 = *(_BYTE *)(a1 + 16);
  if ( v4 )
  {
    while ( v5 == 50 )
    {
      v12 = *(_QWORD *)(a1 - 48);
      v13 = v12;
      if ( !v12 )
        goto LABEL_7;
      v14 = *(unsigned int **)(a1 - 24);
      v15 = *((_BYTE *)v14 + 16);
      if ( v15 != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 || v15 > 0x10u )
        {
LABEL_42:
          v23 = v12;
          if ( !v12 )
            goto LABEL_7;
LABEL_28:
          v24 = *(_QWORD *)(a1 - 24);
          if ( !v24 )
            goto LABEL_7;
LABEL_29:
          a2 = v2;
          if ( !(unsigned __int8)sub_1700A80(v23, v2) )
            return 0;
          v4 = *(_BYTE *)(v2 + 24);
          a1 = v24;
          goto LABEL_2;
        }
        v45 = *(_BYTE **)(a1 - 24);
        v29 = sub_15A1020(v45, a2, (__int64)v14, *(_QWORD *)v14);
        v30 = (__int64)v45;
        if ( v29 && *(_BYTE *)(v29 + 16) == 13 )
        {
LABEL_57:
          v31 = *(_DWORD *)(v29 + 32);
          if ( v31 <= 0x40 )
            v22 = *(_QWORD *)(v29 + 24) == 1;
          else
            v22 = v31 - 1 == (unsigned int)sub_16A57B0(v29 + 24);
LABEL_25:
          if ( v22 )
          {
LABEL_26:
            *(_BYTE *)(v2 + 25) = 1;
            v4 = *(_BYTE *)(v2 + 24);
            a1 = v13;
            goto LABEL_2;
          }
        }
        else
        {
          v43 = *(_QWORD *)(*(_QWORD *)v45 + 32LL);
          if ( !v43 )
            goto LABEL_26;
          v32 = 0;
          while ( 1 )
          {
            v46 = v30;
            v33 = sub_15A0A60(v30, v32);
            v30 = v46;
            if ( !v33 )
              break;
            a2 = *(unsigned __int8 *)(v33 + 16);
            if ( (_BYTE)a2 != 9 )
            {
              if ( (_BYTE)a2 != 13 )
                break;
              a2 = *(unsigned int *)(v33 + 32);
              if ( (unsigned int)a2 <= 0x40 )
              {
                v35 = *(_QWORD *)(v33 + 24) == 1;
              }
              else
              {
                v41 = *(_DWORD *)(v33 + 32);
                v34 = sub_16A57B0(v33 + 24);
                v30 = v46;
                a2 = (unsigned int)(v41 - 1);
                v35 = (_DWORD)a2 == v34;
              }
              if ( !v35 )
                break;
            }
            if ( v43 == ++v32 )
              goto LABEL_26;
          }
        }
        goto LABEL_59;
      }
      v16 = v14[8];
      if ( v16 <= 0x40 )
        v17 = *((_QWORD *)v14 + 3) == 1;
      else
        v17 = v16 - 1 == (unsigned int)sub_16A57B0((__int64)(v14 + 6));
      if ( !v17 )
        goto LABEL_42;
      *(_BYTE *)(v2 + 25) = 1;
      v5 = *(_BYTE *)(v12 + 16);
      a1 = v12;
    }
    if ( v5 != 5 )
      goto LABEL_31;
    v18 = *(unsigned __int16 *)(a1 + 18);
    v6 = *(_WORD *)(a1 + 18);
    if ( (_WORD)v18 != 26 )
      goto LABEL_6;
    v19 = *(_DWORD *)(a1 + 20);
    a2 = -3LL * (v19 & 0xFFFFFFF);
    v13 = *(_QWORD *)(a1 - 24LL * (v19 & 0xFFFFFFF));
    if ( !v13 )
      goto LABEL_50;
    v20 = *(_BYTE **)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( v20[16] == 13 )
    {
      v21 = *((_DWORD *)v20 + 8);
      if ( v21 <= 0x40 )
        v22 = *((_QWORD *)v20 + 3) == 1;
      else
        v22 = v21 - 1 == (unsigned int)sub_16A57B0((__int64)(v20 + 24));
      goto LABEL_25;
    }
    if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) != 16 )
      goto LABEL_48;
    v47 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    v29 = sub_15A1020(v20, a2, v18, (__int64)v20);
    v36 = v47;
    if ( v29 && *(_BYTE *)(v29 + 16) == 13 )
      goto LABEL_57;
    v44 = *(_QWORD *)(*(_QWORD *)v47 + 32LL);
    if ( !v44 )
      goto LABEL_26;
    v37 = 0;
    while ( 1 )
    {
      v48 = v36;
      v38 = sub_15A0A60(v36, v37);
      v36 = v48;
      if ( !v38 )
        break;
      a2 = *(unsigned __int8 *)(v38 + 16);
      if ( (_BYTE)a2 != 9 )
      {
        if ( (_BYTE)a2 != 13 )
          break;
        a2 = *(unsigned int *)(v38 + 32);
        if ( (unsigned int)a2 <= 0x40 )
        {
          v40 = *(_QWORD *)(v38 + 24) == 1;
        }
        else
        {
          v42 = *(_DWORD *)(v38 + 32);
          v39 = sub_16A57B0(v38 + 24);
          v36 = v48;
          a2 = (unsigned int)(v42 - 1);
          v40 = (_DWORD)a2 == v39;
        }
        if ( !v40 )
          break;
      }
      if ( v44 == ++v37 )
        goto LABEL_26;
    }
LABEL_59:
    v5 = *(_BYTE *)(a1 + 16);
    if ( v5 == 50 )
    {
      v12 = *(_QWORD *)(a1 - 48);
      goto LABEL_42;
    }
    if ( v5 != 5 )
      goto LABEL_31;
    LOWORD(v18) = *(_WORD *)(a1 + 18);
LABEL_48:
    v6 = v18;
    if ( (_WORD)v18 != 26 )
      goto LABEL_6;
LABEL_49:
    v19 = *(_DWORD *)(a1 + 20);
LABEL_50:
    v28 = v19 & 0xFFFFFFF;
    v23 = *(_QWORD *)(a1 - 24 * v28);
    if ( v23 )
    {
      v24 = *(_QWORD *)(a1 + 24 * (1 - v28));
      if ( v24 )
        goto LABEL_29;
    }
    goto LABEL_7;
  }
  if ( v5 == 51 )
  {
    v23 = *(_QWORD *)(a1 - 48);
    if ( !v23 )
      goto LABEL_7;
    goto LABEL_28;
  }
  if ( v5 == 5 )
  {
    v6 = *(_WORD *)(a1 + 18);
    if ( v6 != 27 )
    {
LABEL_6:
      if ( v6 != 24 )
        goto LABEL_7;
      v25 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( !v25 )
        goto LABEL_7;
      v26 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v26 + 16) != 13 )
        goto LABEL_7;
LABEL_34:
      v27 = *(_DWORD *)(v26 + 32);
      if ( v27 > 0x40 )
      {
        if ( v27 - (unsigned int)sub_16A57B0(v26 + 24) > 0x40 )
          goto LABEL_7;
        v7 = **(_QWORD **)(v26 + 24);
      }
      else
      {
        v7 = *(_QWORD *)(v26 + 24);
      }
      a1 = v25;
      if ( *(_QWORD *)v2 )
        goto LABEL_8;
      goto LABEL_37;
    }
    goto LABEL_49;
  }
LABEL_31:
  if ( v5 == 48 )
  {
    v25 = *(_QWORD *)(a1 - 48);
    if ( v25 )
    {
      v26 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v26 + 16) == 13 )
        goto LABEL_34;
    }
  }
LABEL_7:
  v7 = 0;
  if ( !*(_QWORD *)v2 )
LABEL_37:
    *(_QWORD *)v2 = a1;
LABEL_8:
  v8 = *(unsigned int *)(v2 + 16);
  if ( v7 >= v8 )
    return 0;
  v9 = *(_QWORD *)(v2 + 8);
  v10 = 1LL << v7;
  if ( (unsigned int)v8 <= 0x40 )
    *(_QWORD *)(v2 + 8) = v9 | v10;
  else
    *(_QWORD *)(v9 + 8LL * ((unsigned int)v7 >> 6)) |= v10;
  return *(_QWORD *)v2 == a1;
}
