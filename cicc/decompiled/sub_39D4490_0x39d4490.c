// Function: sub_39D4490
// Address: 0x39d4490
//
__int64 __fastcall sub_39D4490(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int v6; // r14d
  unsigned int *v7; // r13
  unsigned int *v8; // r12
  unsigned int *v9; // rax
  unsigned int *v10; // rdx
  unsigned int *v11; // rcx
  int v12; // r9d
  __int64 v13; // r14
  unsigned int *v14; // rdi
  int v15; // r13d
  unsigned int *v16; // rax
  unsigned int *v17; // rsi
  unsigned int *v18; // rdi
  unsigned int *v19; // rcx
  unsigned int *v20; // rdx
  unsigned int *v21; // rax
  unsigned int *v23; // [rsp+0h] [rbp-80h] BYREF
  __int64 v24; // [rsp+8h] [rbp-78h]
  _BYTE v25[32]; // [rsp+10h] [rbp-70h] BYREF
  unsigned int *v26; // [rsp+30h] [rbp-50h] BYREF
  __int64 v27; // [rsp+38h] [rbp-48h]
  _DWORD v28[16]; // [rsp+40h] [rbp-40h] BYREF

  v6 = 1;
  if ( (unsigned int)((__int64)(a2[12] - a2[11]) >> 3) <= 1 )
    return v6;
  v7 = (unsigned int *)a2[15];
  v8 = (unsigned int *)a2[14];
  if ( v7 == v8 )
    return v6;
  v24 = 0x800000000LL;
  v9 = (unsigned int *)v25;
  v23 = (unsigned int *)v25;
  if ( (unsigned __int64)((char *)v7 - (char *)v8) > 0x20 )
  {
    sub_16CD150((__int64)&v23, v25, v7 - v8, 4, a5, a6);
    v9 = &v23[(unsigned int)v24];
  }
  v10 = v8;
  v11 = (unsigned int *)((char *)v9 + (char *)v7 - (char *)v8);
  do
  {
    if ( v9 )
      *v9 = *v10;
    ++v9;
    ++v10;
  }
  while ( v11 != v9 );
  LODWORD(v24) = v7 - v8 + v24;
  sub_1953BB0(v23, &v23[(unsigned int)v24]);
  v13 = (unsigned int)v24;
  v26 = v28;
  v14 = v28;
  v27 = 0x800000000LL;
  v15 = v24;
  if ( (unsigned int)v24 > 8 )
  {
    sub_16CD150((__int64)&v26, v28, (unsigned int)v24, 4, (int)&v26, v12);
    v14 = v26;
    LODWORD(v27) = v15;
    v16 = &v26[v13];
    if ( v16 != v26 )
      goto LABEL_11;
  }
  else
  {
    v16 = &v28[(unsigned int)v24];
    LODWORD(v27) = v24;
    if ( v16 != v28 )
    {
      do
      {
LABEL_11:
        if ( v14 )
          *v14 = -1;
        ++v14;
      }
      while ( v16 != v14 );
      v14 = v26;
      v17 = &v26[(unsigned int)v27];
      goto LABEL_15;
    }
  }
  v17 = v14;
LABEL_15:
  sub_1953BB0(v14, v17);
  v18 = v23;
  v19 = &v23[(unsigned int)v24];
  if ( v19 == v23 )
  {
LABEL_25:
    v6 = 1;
  }
  else
  {
    v20 = v26;
    v21 = v23;
    while ( *v20 == *v21 )
    {
      ++v21;
      ++v20;
      if ( v19 == v21 )
        goto LABEL_25;
    }
    v6 = 0;
  }
  if ( v26 != v28 )
  {
    _libc_free((unsigned __int64)v26);
    v18 = v23;
  }
  if ( v18 != (unsigned int *)v25 )
    _libc_free((unsigned __int64)v18);
  return v6;
}
