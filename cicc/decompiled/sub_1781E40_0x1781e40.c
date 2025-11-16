// Function: sub_1781E40
// Address: 0x1781e40
//
__int64 __fastcall sub_1781E40(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _BYTE *v6; // r13
  __int64 v7; // r12
  __int64 v8; // r14
  int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 v17; // r14
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  __int64 *v20; // r13
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r8
  unsigned __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 *v26; // rax
  unsigned int v27; // ebx
  bool v28; // al
  __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdi
  unsigned __int64 v33; // rsi
  __int64 v34; // rsi
  unsigned __int64 v35; // rax
  unsigned int v36; // r13d
  bool v37; // al
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // ebx
  __int64 v41; // rsi
  unsigned __int64 v42; // rdi
  __int64 v43; // rax
  unsigned int v44; // ebx
  unsigned int v45; // r14d
  __int64 v46; // rax
  unsigned int v47; // ebx
  unsigned int v49; // r14d
  __int64 v50; // rax
  char v51; // dl
  unsigned int v52; // r13d
  _QWORD *v54; // [rsp+0h] [rbp-60h]
  int v55; // [rsp+Ch] [rbp-54h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  _QWORD *v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+20h] [rbp-40h]
  int v59; // [rsp+20h] [rbp-40h]
  int v60; // [rsp+20h] [rbp-40h]

  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) != 79 )
    return 0;
  v6 = *(_BYTE **)(v4 - 48);
  v7 = a2;
  if ( v6[16] > 0x10u )
    goto LABEL_44;
  if ( !sub_1593BB0(*(_QWORD *)(v4 - 48), a2, a3, a4) )
  {
    if ( v6[16] == 13 )
    {
      v27 = *((_DWORD *)v6 + 8);
      if ( v27 <= 0x40 )
        v28 = *((_QWORD *)v6 + 3) == 0;
      else
        v28 = v27 == (unsigned int)sub_16A57B0((__int64)(v6 + 24));
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
        goto LABEL_44;
      v39 = sub_15A1020(v6, a2, a3, a4);
      if ( !v39 || *(_BYTE *)(v39 + 16) != 13 )
      {
        v59 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
        if ( !v59 )
          goto LABEL_5;
        v45 = 0;
        while ( 1 )
        {
          a2 = v45;
          v46 = sub_15A0A60((__int64)v6, v45);
          if ( !v46 )
            goto LABEL_44;
          a3 = *(unsigned __int8 *)(v46 + 16);
          if ( (_BYTE)a3 != 9 )
          {
            if ( (_BYTE)a3 != 13 )
              goto LABEL_44;
            v47 = *(_DWORD *)(v46 + 32);
            if ( !(v47 <= 0x40 ? *(_QWORD *)(v46 + 24) == 0 : v47 == (unsigned int)sub_16A57B0(v46 + 24)) )
              goto LABEL_44;
          }
          if ( v59 == ++v45 )
            goto LABEL_5;
        }
      }
      v40 = *(_DWORD *)(v39 + 32);
      if ( v40 <= 0x40 )
        v28 = *(_QWORD *)(v39 + 24) == 0;
      else
        v28 = v40 == (unsigned int)sub_16A57B0(v39 + 24);
    }
    if ( v28 )
      goto LABEL_5;
LABEL_44:
    v29 = *(_QWORD *)(v4 - 24);
    if ( *(_BYTE *)(v29 + 16) <= 0x10u )
    {
      if ( sub_1593BB0(*(_QWORD *)(v4 - 24), a2, a3, a4) )
      {
LABEL_46:
        v8 = -48;
        v9 = 1;
        goto LABEL_6;
      }
      if ( *(_BYTE *)(v29 + 16) == 13 )
      {
        v36 = *(_DWORD *)(v29 + 32);
        if ( v36 <= 0x40 )
          v37 = *(_QWORD *)(v29 + 24) == 0;
        else
          v37 = v36 == (unsigned int)sub_16A57B0(v29 + 24);
        if ( !v37 )
          return 0;
        v8 = -48;
        v9 = 1;
        goto LABEL_6;
      }
      if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
      {
        v43 = sub_15A1020((_BYTE *)v29, a2, v30, v31);
        if ( !v43 || *(_BYTE *)(v43 + 16) != 13 )
        {
          v60 = *(_QWORD *)(*(_QWORD *)v29 + 32LL);
          if ( v60 )
          {
            v49 = 0;
            while ( 1 )
            {
              v50 = sub_15A0A60(v29, v49);
              if ( !v50 )
                return 0;
              v51 = *(_BYTE *)(v50 + 16);
              if ( v51 != 9 )
              {
                if ( v51 != 13 )
                  return 0;
                v52 = *(_DWORD *)(v50 + 32);
                if ( !(v52 <= 0x40 ? *(_QWORD *)(v50 + 24) == 0 : v52 == (unsigned int)sub_16A57B0(v50 + 24)) )
                  return 0;
              }
              if ( v60 == ++v49 )
                goto LABEL_46;
            }
          }
          goto LABEL_46;
        }
        v44 = *(_DWORD *)(v43 + 32);
        if ( v44 <= 0x40 )
        {
          if ( !*(_QWORD *)(v43 + 24) )
            goto LABEL_46;
        }
        else if ( v44 == (unsigned int)sub_16A57B0(v43 + 24) )
        {
          goto LABEL_46;
        }
      }
    }
    return 0;
  }
LABEL_5:
  v8 = -24;
  v9 = 2;
LABEL_6:
  v10 = *(_QWORD *)(v4 + v8);
  v11 = *(_QWORD *)(v7 - 24);
  if ( v10 )
  {
    if ( v11 )
    {
      v12 = *(_QWORD *)(v7 - 16);
      v13 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v13 = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
    }
    *(_QWORD *)(v7 - 24) = v10;
    v14 = *(_QWORD *)(v10 + 8);
    *(_QWORD *)(v7 - 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = (v7 - 16) | *(_QWORD *)(v14 + 16) & 3LL;
    *(_QWORD *)(v7 - 8) = (v10 + 8) | *(_QWORD *)(v7 - 8) & 3LL;
    *(_QWORD *)(v10 + 8) = v7 - 24;
  }
  else if ( v11 )
  {
    v41 = *(_QWORD *)(v7 - 16);
    v42 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v42 = v41;
    if ( v41 )
      *(_QWORD *)(v41 + 16) = v42 | *(_QWORD *)(v41 + 16) & 3LL;
    *(_QWORD *)(v7 - 24) = 0;
  }
  if ( !*(_QWORD *)(v4 + 8) )
  {
    v38 = *(_QWORD *)(*(_QWORD *)(v4 - 72) + 8LL);
    if ( v38 )
    {
      if ( !*(_QWORD *)(v38 + 8) )
        return 1;
    }
  }
  v55 = v9;
  v57 = (_QWORD *)(v7 + 24);
  v15 = *(_QWORD *)(v7 + 40);
  v16 = *(__int64 **)(v4 - 72);
  v58 = v8;
  v54 = *(_QWORD **)(v15 + 48);
  v56 = *v16;
  do
  {
    if ( v57 == v54 )
      break;
    v17 = 0;
    v18 = *v57 & 0xFFFFFFFFFFFFFFF8LL;
    v57 = (_QWORD *)v18;
    if ( v18 )
      v17 = v18 - 24;
    if ( !(unsigned __int8)sub_14AE440(v17) )
      break;
    v19 = 3LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
    {
      v20 = *(__int64 **)(v17 - 8);
      v21 = &v20[v19];
    }
    else
    {
      v21 = (__int64 *)v17;
      v20 = (__int64 *)(v17 - v19 * 8);
    }
    for ( ; v20 != v21; v20 += 3 )
    {
      while ( v4 == *v20 )
      {
        v22 = *(_QWORD *)(v4 + v58);
        if ( !v22 )
        {
          if ( v4 )
          {
            v34 = v20[1];
            v35 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v35 = v34;
            if ( v34 )
              *(_QWORD *)(v34 + 16) = *(_QWORD *)(v34 + 16) & 3LL | v35;
            *v20 = 0;
          }
          goto LABEL_34;
        }
        if ( v4 )
        {
          v23 = v20[1];
          v24 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v24 = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
        }
        *v20 = v22;
LABEL_31:
        v25 = *(_QWORD *)(v22 + 8);
        v20[1] = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v25 + 16) & 3LL;
        v20[2] = (v22 + 8) | v20[2] & 3;
        *(_QWORD *)(v22 + 8) = v20;
LABEL_34:
        v20 += 3;
        sub_170B990(*a1, v17);
        if ( v20 == v21 )
          goto LABEL_35;
      }
      if ( v16 == (__int64 *)*v20 )
      {
        if ( v55 == 1 )
          v22 = sub_15A0600(v56);
        else
          v22 = sub_15A0640(v56);
        if ( *v20 )
        {
          v32 = v20[1];
          v33 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v33 = v32;
          if ( v32 )
            *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
        }
        *v20 = v22;
        if ( !v22 )
          goto LABEL_34;
        goto LABEL_31;
      }
    }
LABEL_35:
    if ( v4 == v17 )
    {
      if ( v16 == (__int64 *)v4 )
        return 1;
      v26 = v16;
      v4 = 0;
    }
    else
    {
      v26 = (__int64 *)v4;
      if ( v16 == (__int64 *)v17 )
        v16 = 0;
      else
        v26 = (__int64 *)((unsigned __int64)v16 | v4);
    }
  }
  while ( v26 );
  return 1;
}
