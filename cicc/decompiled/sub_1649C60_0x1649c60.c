// Function: sub_1649C60
// Address: 0x1649c60
//
__int64 __fastcall sub_1649C60(__int64 a1)
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
  __int64 v21; // [rsp+8h] [rbp-88h] BYREF
  __int64 v22; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-78h]
  __int64 *v24; // [rsp+20h] [rbp-70h]
  __int64 v25; // [rsp+28h] [rbp-68h]
  int v26; // [rsp+30h] [rbp-60h]
  _QWORD v27[11]; // [rsp+38h] [rbp-58h] BYREF

  v1 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v1;
  v26 = 0;
  v23 = v27;
  v24 = v27;
  v25 = 0x100000004LL;
  v27[0] = a1;
  v22 = 1;
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
      v21 = v18;
      if ( !v19 )
        goto LABEL_26;
      v20 = sub_14AF150(&v21);
      if ( !v20 )
        goto LABEL_26;
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
        v5 = v23;
        if ( v24 != v23 )
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
        v5 = v23;
        if ( v24 != v23 )
          goto LABEL_12;
        break;
      default:
        goto LABEL_26;
    }
    v16 = &v5[HIDWORD(v25)];
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
          --v26;
          ++v22;
          goto LABEL_4;
        }
      }
      return v1;
    }
LABEL_46:
    if ( HIDWORD(v25) < (unsigned int)v25 )
    {
      ++HIDWORD(v25);
      *v16 = v1;
      ++v22;
      goto LABEL_4;
    }
LABEL_12:
    sub_16CCBA0(&v22, v1);
    v6 = (unsigned __int64)v24;
    v7 = v23;
    if ( !v8 )
      goto LABEL_27;
    v3 = *(_BYTE *)(v1 + 16);
    if ( v3 > 0x17u )
      goto LABEL_5;
  }
  if ( v3 == 1 )
    __asm { jmp     rax }
LABEL_26:
  v6 = (unsigned __int64)v24;
  v7 = v23;
LABEL_27:
  if ( (__int64 *)v6 != v7 )
    _libc_free(v6);
  return v1;
}
