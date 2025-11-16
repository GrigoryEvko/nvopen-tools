// Function: sub_1649F00
// Address: 0x1649f00
//
__int64 __fastcall sub_1649F00(__int64 a1)
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
  __int64 *v13; // r13
  __int64 v14; // rdi
  unsigned int v15; // ebx
  __int64 *v16; // rsi
  __int64 *v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+18h] [rbp-88h] BYREF
  __int64 v22; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v23; // [rsp+28h] [rbp-78h]
  __int64 *v24; // [rsp+30h] [rbp-70h]
  __int64 v25; // [rsp+38h] [rbp-68h]
  int v26; // [rsp+40h] [rbp-60h]
  _QWORD v27[11]; // [rsp+48h] [rbp-58h] BYREF

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
    if ( v3 == 56 )
      goto LABEL_21;
    if ( v3 != 71 && v3 != 72 )
    {
      if ( v3 == 78 )
      {
        v18 = v1 | 4;
        v19 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( v3 != 29 )
          goto LABEL_18;
        v18 = v1 & 0xFFFFFFFFFFFFFFFBLL;
        v19 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v21 = v18;
      if ( !v19 )
        goto LABEL_18;
      v20 = sub_14AF150(&v21);
      if ( !v20 )
        goto LABEL_18;
      v1 = v20;
LABEL_11:
      v5 = v23;
      if ( v24 == v23 )
        goto LABEL_29;
      goto LABEL_12;
    }
LABEL_8:
    if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
      v4 = *(__int64 **)(v1 - 8);
    else
      v4 = (__int64 *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
    v1 = *v4;
    goto LABEL_11;
  }
  while ( 1 )
  {
    if ( v3 != 5 )
      goto LABEL_18;
    v9 = *(_WORD *)(v1 + 18);
    if ( v9 != 32 )
    {
      if ( v9 != 47 && v9 != 48 )
        goto LABEL_18;
      goto LABEL_8;
    }
LABEL_21:
    v10 = 3LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
    v11 = (__int64 *)(v1 - v10 * 8);
    if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
      v11 = *(__int64 **)(v1 - 8);
    v12 = v11 + 3;
    v13 = &v11[v10];
    if ( v11 + 3 != &v11[v10] )
      break;
LABEL_28:
    v1 = *v11;
    v5 = v23;
    if ( v24 == v23 )
    {
LABEL_29:
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
    }
LABEL_12:
    sub_16CCBA0(&v22, v1);
    v6 = (unsigned __int64)v24;
    v7 = v23;
    if ( !v8 )
      goto LABEL_19;
    v3 = *(_BYTE *)(v1 + 16);
    if ( v3 > 0x17u )
      goto LABEL_5;
  }
  while ( 1 )
  {
    v14 = *v12;
    if ( *(_BYTE *)(*v12 + 16) != 13 )
      break;
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
    {
      if ( *(_QWORD *)(v14 + 24) )
        break;
    }
    else if ( v15 != (unsigned int)sub_16A57B0(v14 + 24) )
    {
      break;
    }
    v12 += 3;
    if ( v13 == v12 )
      goto LABEL_28;
  }
LABEL_18:
  v6 = (unsigned __int64)v24;
  v7 = v23;
LABEL_19:
  if ( (__int64 *)v6 != v7 )
    _libc_free(v6);
  return v1;
}
