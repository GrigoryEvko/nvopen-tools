// Function: sub_28569D0
// Address: 0x28569d0
//
__int64 __fastcall sub_28569D0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r8
  __int16 v4; // ax
  __int64 v5; // r12
  __int64 v7; // r12
  void *v8; // r9
  signed __int64 v9; // r12
  __int64 v10; // r8
  _BYTE *v11; // rax
  _BYTE *v12; // rdi
  __int64 v13; // r12
  const void *v14; // r10
  signed __int64 v15; // r12
  __int64 v16; // r9
  _BYTE *v17; // rdi
  const void *v18; // [rsp+8h] [rbp-98h]
  void *src; // [rsp+10h] [rbp-90h]
  void *srca; // [rsp+10h] [rbp-90h]
  int srcb; // [rsp+10h] [rbp-90h]
  int v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  _BYTE *v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+28h] [rbp-78h]
  _BYTE v27[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = *a1;
  v4 = *(_WORD *)(*a1 + 24);
  if ( v4 == 15 )
  {
    v5 = *(_QWORD *)(v3 - 8);
    if ( *(_BYTE *)v5 > 3u )
      return 0;
    *a1 = (__int64)sub_DA2C50((__int64)a2, *(_QWORD *)(v5 + 8), 0, 0);
    return v5;
  }
  if ( v4 != 5 )
  {
    v5 = 0;
    if ( v4 != 8 )
      return v5;
    v13 = *(_QWORD *)(v3 + 40);
    v14 = *(const void **)(v3 + 32);
    v25 = v27;
    v15 = 8 * v13;
    v26 = 0x800000000LL;
    v16 = v15 >> 3;
    if ( (unsigned __int64)v15 > 0x40 )
    {
      v18 = v14;
      srca = (void *)v3;
      sub_C8D5F0((__int64)&v25, v27, v15 >> 3, 8u, v3, v16);
      v16 = v15 >> 3;
      v3 = (__int64)srca;
      v14 = v18;
      v17 = &v25[8 * (unsigned int)v26];
    }
    else
    {
      v17 = v27;
      if ( !v15 )
        goto LABEL_18;
    }
    srcb = v16;
    v24 = v3;
    memcpy(v17, v14, v15);
    v17 = v25;
    LODWORD(v15) = v26;
    LODWORD(v16) = srcb;
    v3 = v24;
LABEL_18:
    v23 = v3;
    LODWORD(v26) = v16 + v15;
    v5 = sub_28569D0(v17, a2);
    if ( v5 )
      *a1 = (__int64)sub_DBFF60((__int64)a2, (unsigned int *)&v25, *(_QWORD *)(v23 + 48), 0);
    goto LABEL_10;
  }
  v7 = *(_QWORD *)(v3 + 40);
  v8 = *(void **)(v3 + 32);
  v25 = v27;
  v9 = 8 * v7;
  v26 = 0x800000000LL;
  v10 = v9 >> 3;
  if ( (unsigned __int64)v9 > 0x40 )
  {
    src = v8;
    sub_C8D5F0((__int64)&v25, v27, v9 >> 3, 8u, v10, (__int64)v8);
    v10 = v9 >> 3;
    v8 = src;
    v12 = &v25[8 * (unsigned int)v26];
  }
  else
  {
    v11 = v27;
    if ( !v9 )
      goto LABEL_8;
    v12 = v27;
  }
  v22 = v10;
  memcpy(v12, v8, v9);
  v11 = v25;
  LODWORD(v9) = v26;
  LODWORD(v10) = v22;
LABEL_8:
  LODWORD(v26) = v10 + v9;
  v5 = sub_28569D0(&v11[8 * (unsigned int)(v10 + v9) - 8], a2);
  if ( v5 )
    *a1 = (__int64)sub_DC7EB0(a2, (__int64)&v25, 0, 0);
LABEL_10:
  if ( v25 != v27 )
    _libc_free((unsigned __int64)v25);
  return v5;
}
