// Function: sub_AE94E0
// Address: 0xae94e0
//
void __fastcall sub_AE94E0(__int64 a1)
{
  unsigned __int64 v1; // rsi
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // rdi
  __int64 *v20; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v21; // [rsp-B0h] [rbp-B0h]
  _BYTE v22[48]; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 *v23; // [rsp-78h] [rbp-78h] BYREF
  __int64 v24; // [rsp-70h] [rbp-70h]
  _BYTE v25[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return;
  v1 = 38;
  v2 = sub_B91C10(a1, 38);
  v3 = v2;
  if ( v2 )
  {
    v5 = sub_AE94B0(v2);
    v3 = v6;
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      goto LABEL_25;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      return;
    v5 = 0;
  }
  v1 = 38;
  v7 = sub_B91C10(a1, 38);
  if ( !v7 )
  {
LABEL_25:
    v20 = (__int64 *)v22;
    v21 = 0x600000000LL;
    if ( v3 == v5 )
      return;
    goto LABEL_17;
  }
  v8 = *(_QWORD *)(v7 + 8);
  v1 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v8 & 4) == 0 )
    v1 = 0;
  sub_B967C0(&v20, v1);
  if ( v3 == v5 )
  {
    if ( !(_DWORD)v21 )
    {
      v12 = v20;
      if ( v20 == (__int64 *)v22 )
        return;
      goto LABEL_15;
    }
    v24 = 0x600000000LL;
    v23 = (__int64 *)v25;
    goto LABEL_10;
  }
LABEL_17:
  v13 = v5;
  v14 = 0;
  v23 = (__int64 *)v25;
  v24 = 0x600000000LL;
  do
  {
    v13 = *(_QWORD *)(v13 + 8);
    ++v14;
  }
  while ( v13 != v3 );
  v15 = (__int64 *)v25;
  if ( v14 > 6 )
  {
    v1 = (unsigned __int64)v25;
    sub_C8D5F0(&v23, v25, v14, 8);
    v15 = &v23[(unsigned int)v24];
  }
  do
  {
    v16 = *(_QWORD *)(v5 + 24);
    *v15++ = v16;
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v13 != v5 );
  v17 = v23;
  LODWORD(v24) = v24 + v14;
  v18 = &v23[(unsigned int)v24];
  if ( v23 != v18 )
  {
    do
    {
      v19 = *v17++;
      sub_B43D60(v19, v1, v16, v4);
    }
    while ( v18 != v17 );
  }
LABEL_10:
  v9 = v20;
  v10 = &v20[(unsigned int)v21];
  if ( v10 != v20 )
  {
    do
    {
      v11 = *v9++;
      sub_B14290(v11);
    }
    while ( v10 != v9 );
  }
  if ( v23 != (__int64 *)v25 )
    _libc_free(v23, v1);
  v12 = v20;
  if ( v20 != (__int64 *)v22 )
LABEL_15:
    _libc_free(v12, v1);
}
