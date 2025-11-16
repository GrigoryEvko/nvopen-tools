// Function: sub_1728300
// Address: 0x1728300
//
__int64 __fastcall sub_1728300(__int64 a1, _DWORD *a2, __int64 **a3)
{
  char v4; // dl
  __int64 v5; // rdx
  char v6; // cl
  __int64 *v7; // r14
  __int64 v8; // r15
  unsigned int v9; // r8d
  __int64 v11; // rcx
  char v12; // al
  __int64 v13; // rbx
  unsigned int v14; // r15d
  unsigned int v15; // eax
  __int64 *v16; // rcx
  __int64 v17; // r14
  unsigned int v18; // r15d
  int v19; // eax
  unsigned int v20; // eax
  int v21; // eax
  unsigned int v22; // eax
  unsigned int v23; // r15d
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 *v27; // [rsp+10h] [rbp-40h]
  char v28; // [rsp+10h] [rbp-40h]
  char v29; // [rsp+18h] [rbp-38h]
  __int64 *v30; // [rsp+18h] [rbp-38h]
  unsigned int v31; // [rsp+18h] [rbp-38h]
  unsigned __int8 v32; // [rsp+18h] [rbp-38h]

  v4 = *(_BYTE *)(a1 + 16);
  switch ( v4 )
  {
    case 50:
      v16 = *(__int64 **)(a1 - 48);
      if ( !v16 )
        return 0;
      v17 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v17 + 16) != 13 )
        return 0;
      break;
    case 5:
      if ( *(_WORD *)(a1 + 18) != 26 )
        goto LABEL_4;
      v16 = *(__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( !v16 )
        goto LABEL_4;
      v17 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v17 + 16) != 13 )
        goto LABEL_4;
      break;
    case 47:
      goto LABEL_15;
    default:
      goto LABEL_51;
  }
  v18 = *(_DWORD *)(v17 + 32);
  if ( v18 > 0x40 )
  {
    v27 = v16;
    v29 = *(_BYTE *)(a1 + 16);
    v19 = sub_16A57B0(v17 + 24);
    v4 = v29;
    v16 = v27;
    if ( v18 - v19 > 0x40 || **(_QWORD **)(v17 + 24) != 255 )
      goto LABEL_28;
  }
  else if ( *(_QWORD *)(v17 + 24) != 255 )
  {
    goto LABEL_28;
  }
  v30 = v16;
  LOBYTE(v20) = sub_1642F90(*v16, 32);
  v9 = v20;
  if ( (_BYTE)v20 )
  {
    *a3 = v30;
    *a2 = 0;
    return v9;
  }
  v4 = *(_BYTE *)(a1 + 16);
LABEL_28:
  if ( v4 == 47 )
  {
LABEL_15:
    v11 = *(_QWORD *)(a1 - 48);
    v12 = *(_BYTE *)(v11 + 16);
    if ( v12 == 50 )
    {
      v7 = *(__int64 **)(v11 - 48);
      if ( !v7 )
        goto LABEL_18;
      v8 = *(_QWORD *)(v11 - 24);
      if ( *(_BYTE *)(v8 + 16) != 13 )
        goto LABEL_18;
    }
    else
    {
      if ( v12 != 5 )
        goto LABEL_18;
      if ( *(_WORD *)(v11 + 18) != 26 )
        goto LABEL_18;
      v7 = *(__int64 **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      if ( !v7 )
        goto LABEL_18;
      v8 = *(_QWORD *)(v11 + 24 * (1LL - (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v8 + 16) != 13 )
        goto LABEL_18;
    }
    v4 = 47;
    v26 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v26 + 16) != 13 )
      goto LABEL_18;
LABEL_48:
    v31 = *(_DWORD *)(v8 + 32);
    if ( v31 > 0x40 )
    {
      v28 = v4;
      v21 = sub_16A57B0(v8 + 24);
      v4 = v28;
      if ( v31 - v21 > 0x40 || **(_QWORD **)(v8 + 24) != 255 )
        goto LABEL_50;
    }
    else if ( *(_QWORD *)(v8 + 24) != 255 )
    {
LABEL_50:
      if ( v4 != 47 )
      {
LABEL_51:
        if ( v4 != 5 )
          return 0;
        goto LABEL_11;
      }
      v11 = *(_QWORD *)(a1 - 48);
LABEL_18:
      v7 = (__int64 *)v11;
      if ( !v11 )
        return 0;
      v13 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v13 + 16) != 13 )
        return 0;
      goto LABEL_20;
    }
    LOBYTE(v22) = sub_1642F90(*v7, 32);
    v9 = v22;
    if ( (_BYTE)v22 )
    {
      v23 = *(_DWORD *)(v26 + 32);
      if ( v23 <= 0x40 )
      {
        v24 = *(_QWORD *)(v26 + 24);
        if ( v24 != 8 )
        {
          if ( v24 != 16 )
            goto LABEL_59;
          goto LABEL_71;
        }
        goto LABEL_66;
      }
      v32 = v22;
      v25 = sub_16A57B0(v26 + 24);
      v9 = v32;
      if ( v23 - v25 <= 0x40 )
      {
        v24 = **(_QWORD **)(v26 + 24);
        if ( v24 != 8 )
        {
          if ( v24 != 16 )
          {
LABEL_59:
            if ( v24 == 24 )
              goto LABEL_23;
            goto LABEL_60;
          }
LABEL_71:
          *a3 = v7;
          *a2 = 2;
          return v9;
        }
LABEL_66:
        *a3 = v7;
        *a2 = 1;
        return v9;
      }
    }
LABEL_60:
    v4 = *(_BYTE *)(a1 + 16);
    goto LABEL_50;
  }
  if ( v4 != 5 )
    goto LABEL_50;
LABEL_4:
  if ( *(_WORD *)(a1 + 18) == 23 )
  {
    v5 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 50 )
    {
      v7 = *(__int64 **)(v5 - 48);
      if ( v7 )
      {
        v8 = *(_QWORD *)(v5 - 24);
        if ( *(_BYTE *)(v8 + 16) == 13 )
        {
LABEL_10:
          v4 = 5;
          v26 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          if ( *(_BYTE *)(v26 + 16) != 13 )
            goto LABEL_11;
          goto LABEL_48;
        }
      }
    }
    else if ( v6 == 5 && *(_WORD *)(v5 + 18) == 26 )
    {
      v7 = *(__int64 **)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      if ( v7 )
      {
        v8 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v8 + 16) == 13 )
          goto LABEL_10;
      }
    }
  }
LABEL_11:
  if ( *(_WORD *)(a1 + 18) != 23 )
    return 0;
  v7 = *(__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( !v7 )
    return 0;
  v13 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v13 + 16) != 13 )
    return 0;
LABEL_20:
  v14 = *(_DWORD *)(v13 + 32);
  if ( v14 > 0x40 )
  {
    if ( v14 - (unsigned int)sub_16A57B0(v13 + 24) > 0x40 || **(_QWORD **)(v13 + 24) != 24 )
      return 0;
  }
  else if ( *(_QWORD *)(v13 + 24) != 24 )
  {
    return 0;
  }
  LOBYTE(v15) = sub_1642F90(*v7, 32);
  v9 = v15;
  if ( (_BYTE)v15 )
  {
LABEL_23:
    *a3 = v7;
    *a2 = 3;
    return v9;
  }
  return 0;
}
