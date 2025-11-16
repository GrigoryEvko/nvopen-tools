// Function: sub_164A190
// Address: 0x164a190
//
__int64 __fastcall sub_164A190(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v3; // al
  __int64 *v4; // r12
  __int64 *v5; // rax
  unsigned __int64 v6; // rdi
  __int64 *v7; // rax
  char v8; // dl
  __int16 v9; // ax
  __int64 v10; // rcx
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 *v13; // rcx
  __int64 *v14; // rsi
  __int64 *v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-68h] BYREF
  __int64 v20; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v21; // [rsp+18h] [rbp-58h]
  __int64 *v22; // [rsp+20h] [rbp-50h]
  __int64 v23; // [rsp+28h] [rbp-48h]
  int v24; // [rsp+30h] [rbp-40h]
  _QWORD v25[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v1;
  v24 = 0;
  v21 = v25;
  v22 = v25;
  v23 = 0x100000004LL;
  v25[0] = a1;
  v20 = 1;
LABEL_4:
  v3 = *(_BYTE *)(v1 + 16);
  if ( v3 > 0x17u )
  {
LABEL_5:
    if ( v3 == 56 )
      goto LABEL_21;
    if ( v3 == 71 || v3 == 72 )
    {
LABEL_8:
      if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
        v4 = *(__int64 **)(v1 - 8);
      else
        v4 = (__int64 *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
      v1 = *v4;
    }
    else
    {
      if ( v3 == 78 )
      {
        v16 = v1 | 4;
        v17 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( v3 != 29 )
          goto LABEL_18;
        v16 = v1 & 0xFFFFFFFFFFFFFFFBLL;
        v17 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v19 = v16;
      if ( !v17 )
        goto LABEL_18;
      v18 = sub_14AF150(&v19);
      if ( !v18 )
        goto LABEL_18;
      v1 = v18;
    }
    v5 = v21;
    if ( v22 == v21 )
      goto LABEL_30;
    goto LABEL_12;
  }
  while ( v3 == 5 )
  {
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
    if ( v11 + 3 != v13 )
    {
      while ( *(_BYTE *)(*v12 + 16) == 13 )
      {
        v12 += 3;
        if ( v13 == v12 )
          goto LABEL_28;
      }
      goto LABEL_18;
    }
LABEL_28:
    if ( (*(_BYTE *)(v1 + 17) & 2) == 0 )
      goto LABEL_18;
    v1 = *v11;
    v5 = v21;
    if ( v22 == v21 )
    {
LABEL_30:
      v14 = &v5[HIDWORD(v23)];
      if ( v5 != v14 )
      {
        v15 = 0;
        while ( v1 != *v5 )
        {
          if ( *v5 == -2 )
            v15 = v5;
          if ( v14 == ++v5 )
          {
            if ( !v15 )
              goto LABEL_45;
            *v15 = v1;
            --v24;
            ++v20;
            goto LABEL_4;
          }
        }
        return v1;
      }
LABEL_45:
      if ( HIDWORD(v23) < (unsigned int)v23 )
      {
        ++HIDWORD(v23);
        *v14 = v1;
        ++v20;
        goto LABEL_4;
      }
    }
LABEL_12:
    sub_16CCBA0(&v20, v1);
    v6 = (unsigned __int64)v22;
    v7 = v21;
    if ( !v8 )
      goto LABEL_19;
    v3 = *(_BYTE *)(v1 + 16);
    if ( v3 > 0x17u )
      goto LABEL_5;
  }
  if ( v3 == 1 )
    __asm { jmp     rax }
LABEL_18:
  v6 = (unsigned __int64)v22;
  v7 = v21;
LABEL_19:
  if ( (__int64 *)v6 == v7 )
    return v1;
  _libc_free(v6);
  return v1;
}
