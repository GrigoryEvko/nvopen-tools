// Function: sub_164AA50
// Address: 0x164aa50
//
__int64 __fastcall sub_164AA50(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v3; // al
  __int64 *v4; // r12
  __int64 *v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v7; // rax
  char v8; // dl
  __int16 v9; // ax
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *v12; // r15
  __int64 *i; // r13
  __int64 v14; // rdi
  unsigned int v15; // ebx
  __int64 *v16; // rsi
  __int64 *v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-88h] BYREF
  __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v26; // [rsp+18h] [rbp-78h]
  __int64 *v27; // [rsp+20h] [rbp-70h]
  __int64 v28; // [rsp+28h] [rbp-68h]
  int v29; // [rsp+30h] [rbp-60h]
  _QWORD v30[11]; // [rsp+38h] [rbp-58h] BYREF

  v1 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v1;
  v29 = 0;
  v26 = v30;
  v27 = v30;
  v28 = 0x100000004LL;
  v30[0] = a1;
  v25 = 1;
LABEL_4:
  v3 = *(_BYTE *)(v1 + 16);
  if ( v3 > 0x17u )
  {
LABEL_5:
    if ( v3 != 56 )
    {
      switch ( v3 )
      {
        case 0x47u:
        case 0x48u:
LABEL_8:
          if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
            goto LABEL_9;
          goto LABEL_19;
        case 0x4Eu:
          v18 = v1 | 4;
          v19 = v1 & 0xFFFFFFFFFFFFFFF8LL;
          break;
        case 0x1Du:
          v18 = v1 & 0xFFFFFFFFFFFFFFFBLL;
          v19 = v1 & 0xFFFFFFFFFFFFFFF8LL;
          break;
        default:
          goto LABEL_26;
      }
      v24 = v18;
      if ( !v19 )
        goto LABEL_26;
      v20 = sub_14AF150(&v24);
      if ( !v20 )
      {
        v21 = (__int64 *)((v24 & 0xFFFFFFFFFFFFFFF8LL) - 72);
        if ( (v24 & 4) != 0 )
          v21 = (__int64 *)((v24 & 0xFFFFFFFFFFFFFFF8LL) - 24);
        v22 = *v21;
        if ( *(_BYTE *)(v22 + 16) )
          goto LABEL_26;
        v23 = *(_DWORD *)(v22 + 36);
        if ( v23 != 115 && v23 != 203 && v23 != 3660 )
          goto LABEL_26;
        v20 = *(_QWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL)
                        - 24LL * (*(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
      }
      v1 = v20;
      goto LABEL_11;
    }
    goto LABEL_20;
  }
  while ( v3 == 5 )
  {
    v9 = *(_WORD *)(v1 + 18);
    switch ( v9 )
    {
      case ' ':
LABEL_20:
        v10 = 3LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
        v11 = (__int64 *)(v1 - v10 * 8);
        if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
          v11 = *(__int64 **)(v1 - 8);
        v12 = v11 + 3;
        for ( i = &v11[v10]; i != v12; v12 += 3 )
        {
          v14 = *v12;
          if ( *(_BYTE *)(*v12 + 16) != 13 )
            goto LABEL_26;
          v15 = *(_DWORD *)(v14 + 32);
          if ( v15 <= 0x40 )
          {
            if ( *(_QWORD *)(v14 + 24) )
              goto LABEL_26;
          }
          else if ( v15 != (unsigned int)sub_16A57B0(v14 + 24) )
          {
            goto LABEL_26;
          }
        }
        v1 = *v11;
        v5 = v26;
        if ( v27 != v26 )
          goto LABEL_12;
        break;
      case '/':
        goto LABEL_8;
      case '0':
        if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
        {
LABEL_9:
          v4 = *(__int64 **)(v1 - 8);
          goto LABEL_10;
        }
LABEL_19:
        v4 = (__int64 *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
LABEL_10:
        v1 = *v4;
LABEL_11:
        v5 = v26;
        if ( v27 != v26 )
          goto LABEL_12;
        break;
      default:
        goto LABEL_26;
    }
    v16 = &v5[HIDWORD(v28)];
    if ( v5 != v16 )
    {
      v17 = 0;
      while ( v1 != *v5 )
      {
        if ( *v5 == -2 )
          v17 = v5;
        if ( v16 == ++v5 )
        {
          if ( !v17 )
            goto LABEL_46;
          *v17 = v1;
          --v29;
          ++v25;
          goto LABEL_4;
        }
      }
      return v1;
    }
LABEL_46:
    if ( HIDWORD(v28) < (unsigned int)v28 )
    {
      ++HIDWORD(v28);
      *v16 = v1;
      ++v25;
      goto LABEL_4;
    }
LABEL_12:
    sub_16CCBA0(&v25, v1);
    v6 = (unsigned __int64)v27;
    v7 = v26;
    if ( !v8 )
      goto LABEL_27;
    v3 = *(_BYTE *)(v1 + 16);
    if ( v3 > 0x17u )
      goto LABEL_5;
  }
  if ( v3 == 1 )
    __asm { jmp     rax }
LABEL_26:
  v6 = (unsigned __int64)v27;
  v7 = v26;
LABEL_27:
  if ( v7 != (__int64 *)v6 )
    _libc_free(v6);
  return v1;
}
