// Function: sub_13CEFA0
// Address: 0x13cefa0
//
__int64 __fastcall sub_13CEFA0(_BYTE *a1, _BYTE *a2, char a3)
{
  unsigned __int8 v3; // al
  __int64 v4; // r15
  _BYTE *v5; // r12
  unsigned __int8 v7; // al
  unsigned int v8; // r13d
  bool v9; // al
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int8 v13; // al
  unsigned int v14; // r13d
  bool v15; // al
  unsigned __int8 v16; // al
  unsigned int v17; // r13d
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // r13d
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned int v23; // r13d
  int v24; // eax
  __int64 v25; // rax
  unsigned int v26; // r13d
  int v27; // r13d
  unsigned int v28; // r14d
  __int64 v29; // rax
  char v30; // dl
  bool v31; // al
  unsigned int v32; // r13d
  __int64 v33; // rax
  char v34; // dl
  unsigned int v35; // r14d
  int v37; // eax
  __int64 **v38; // rbx
  __int64 *v39; // rax
  __int64 v40; // rdi
  unsigned int v41; // r13d
  __int64 v42; // rax
  char v43; // dl
  unsigned int v44; // r14d
  int v45; // [rsp+8h] [rbp-38h]
  int v46; // [rsp+8h] [rbp-38h]
  int v47; // [rsp+8h] [rbp-38h]
  int v48; // [rsp+8h] [rbp-38h]

  v3 = a2[16];
  if ( v3 == 9 )
    return (__int64)a2;
  v4 = *(_QWORD *)a1;
  v5 = a1;
  if ( v3 <= 0x10u )
  {
    if ( (unsigned __int8)sub_1593BB0(a2) )
      return sub_1599EF0(v4);
    v7 = a2[16];
    if ( v7 == 13 )
    {
      v8 = *((_DWORD *)a2 + 8);
      if ( v8 <= 0x40 )
        v9 = *((_QWORD *)a2 + 3) == 0;
      else
        v9 = v8 == (unsigned int)sub_16A57B0(a2 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
        goto LABEL_10;
      v19 = sub_15A1020(a2);
      if ( !v19 || *(_BYTE *)(v19 + 16) != 13 )
      {
        v27 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
        if ( !v27 )
          return sub_1599EF0(v4);
        v28 = 0;
        while ( 1 )
        {
          v29 = sub_15A0A60(a2, v28);
          if ( !v29 )
            break;
          v30 = *(_BYTE *)(v29 + 16);
          if ( v30 != 9 )
          {
            if ( v30 != 13 )
              break;
            if ( *(_DWORD *)(v29 + 32) <= 0x40u )
            {
              v31 = *(_QWORD *)(v29 + 24) == 0;
            }
            else
            {
              v46 = *(_DWORD *)(v29 + 32);
              v31 = v46 == (unsigned int)sub_16A57B0(v29 + 24);
            }
            if ( !v31 )
              break;
          }
          if ( v27 == ++v28 )
            return sub_1599EF0(v4);
        }
LABEL_9:
        v7 = a2[16];
LABEL_10:
        if ( v7 <= 0x10u && *(_BYTE *)(v4 + 8) == 16 )
        {
          v45 = *(_QWORD *)(v4 + 32);
          if ( v45 )
          {
            v10 = 0;
            while ( 1 )
            {
              v11 = sub_15A0A60(a2, v10);
              v12 = v11;
              if ( v11 )
              {
                if ( (unsigned __int8)sub_1593BB0(v11) || *(_BYTE *)(v12 + 16) == 9 )
                  return sub_1599EF0(v4);
              }
              if ( v45 == ++v10 )
                goto LABEL_18;
            }
          }
        }
        goto LABEL_18;
      }
      v20 = *(_DWORD *)(v19 + 32);
      if ( v20 <= 0x40 )
        v9 = *(_QWORD *)(v19 + 24) == 0;
      else
        v9 = v20 == (unsigned int)sub_16A57B0(v19 + 24);
    }
    if ( v9 )
      return sub_1599EF0(v4);
    goto LABEL_9;
  }
LABEL_18:
  v13 = a1[16];
  if ( v13 == 9 )
    return sub_15A06D0(v4);
  if ( v13 > 0x10u )
    goto LABEL_26;
  if ( (unsigned __int8)sub_1593BB0(a1) )
    return sub_15A06D0(*(_QWORD *)a1);
  if ( a1[16] == 13 )
  {
    v14 = *((_DWORD *)a1 + 8);
    if ( v14 <= 0x40 )
      v15 = *((_QWORD *)a1 + 3) == 0;
    else
      v15 = v14 == (unsigned int)sub_16A57B0(a1 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
      goto LABEL_26;
    v22 = sub_15A1020(a1);
    if ( !v22 || *(_BYTE *)(v22 + 16) != 13 )
    {
      v48 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v48 )
        return sub_15A06D0(*(_QWORD *)a1);
      v41 = 0;
      while ( 1 )
      {
        v42 = sub_15A0A60(a1, v41);
        if ( !v42 )
          goto LABEL_26;
        v43 = *(_BYTE *)(v42 + 16);
        if ( v43 != 9 )
        {
          if ( v43 != 13 )
            goto LABEL_26;
          v44 = *(_DWORD *)(v42 + 32);
          if ( v44 <= 0x40 )
          {
            if ( *(_QWORD *)(v42 + 24) )
              goto LABEL_26;
          }
          else if ( v44 != (unsigned int)sub_16A57B0(v42 + 24) )
          {
            goto LABEL_26;
          }
        }
        if ( v48 == ++v41 )
          return sub_15A06D0(*(_QWORD *)a1);
      }
    }
    v23 = *(_DWORD *)(v22 + 32);
    if ( v23 <= 0x40 )
      v15 = *(_QWORD *)(v22 + 24) == 0;
    else
      v15 = v23 == (unsigned int)sub_16A57B0(v22 + 24);
  }
  if ( v15 )
    return sub_15A06D0(*(_QWORD *)a1);
LABEL_26:
  if ( a1 != a2 )
  {
    v16 = a2[16];
    if ( v16 == 13 )
    {
      v17 = *((_DWORD *)a2 + 8);
      if ( v17 <= 0x40 )
        v18 = *((_QWORD *)a2 + 3) == 1;
      else
        v18 = v17 - 1 == (unsigned int)sub_16A57B0(a2 + 24);
LABEL_30:
      if ( v18 )
        goto LABEL_46;
      goto LABEL_43;
    }
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 || v16 > 0x10u )
      goto LABEL_43;
    v25 = sub_15A1020(a2);
    if ( v25 && *(_BYTE *)(v25 + 16) == 13 )
    {
      v26 = *(_DWORD *)(v25 + 32);
      if ( v26 <= 0x40 )
      {
        v18 = *(_QWORD *)(v25 + 24) == 1;
        goto LABEL_30;
      }
      if ( (unsigned int)sub_16A57B0(v25 + 24) == v26 - 1 )
        goto LABEL_46;
    }
    else
    {
      v47 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
      if ( !v47 )
        goto LABEL_46;
      v32 = 0;
      while ( 1 )
      {
        v33 = sub_15A0A60(a2, v32);
        if ( !v33 )
          break;
        v34 = *(_BYTE *)(v33 + 16);
        if ( v34 != 9 )
        {
          if ( v34 != 13 )
            break;
          v35 = *(_DWORD *)(v33 + 32);
          if ( !(v35 <= 0x40 ? *(_QWORD *)(v33 + 24) == 1 : v35 - 1 == (unsigned int)sub_16A57B0(v33 + 24)) )
            break;
        }
        if ( v47 == ++v32 )
          goto LABEL_46;
      }
    }
LABEL_43:
    v21 = v4;
    if ( *(_BYTE *)(v4 + 8) == 16 )
      v21 = **(_QWORD **)(v4 + 16);
    if ( (unsigned __int8)sub_1642F90(v21, 1) )
    {
LABEL_46:
      if ( !a3 )
        return sub_15A06D0(v4);
      return (__int64)v5;
    }
    v24 = (unsigned __int8)a2[16];
    if ( (unsigned __int8)v24 > 0x17u )
    {
      v37 = v24 - 24;
    }
    else
    {
      if ( (_BYTE)v24 != 5 )
        return 0;
      v37 = *((unsigned __int16 *)a2 + 9);
    }
    if ( v37 == 37 )
    {
      v38 = (a2[23] & 0x40) != 0
          ? (__int64 **)*((_QWORD *)a2 - 1)
          : (__int64 **)&a2[-24 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
      v39 = *v38;
      if ( *v38 )
      {
        v40 = *v39;
        if ( *(_BYTE *)(*v39 + 8) == 16 )
          v40 = **(_QWORD **)(v40 + 16);
        if ( (unsigned __int8)sub_1642F90(v40, 1) )
          goto LABEL_46;
      }
    }
    return 0;
  }
  if ( !a3 )
    return sub_15A06D0(v4);
  return sub_15A0680(v4, 1, 0);
}
