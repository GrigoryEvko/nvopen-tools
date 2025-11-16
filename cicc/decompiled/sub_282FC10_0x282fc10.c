// Function: sub_282FC10
// Address: 0x282fc10
//
__int64 __fastcall sub_282FC10(__int64 a1, int a2, int a3)
{
  char *v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  char *v6; // r10
  char *v7; // r12
  char *v8; // r13
  __int64 v9; // rbx
  char **v10; // rcx
  char *v11; // rax
  char *v12; // rsi
  size_t v13; // r14
  size_t v14; // rdx
  char *v15; // rcx
  char *v16; // rdx
  char v17; // al
  char *v18; // rax
  char v19; // si
  char v20; // al
  unsigned int v21; // r12d
  char *v23; // rsi
  size_t v24; // rax
  char *v25; // rax
  char v26; // cl
  char *v27; // rcx
  char *v28; // [rsp+0h] [rbp-60h]
  char *v29; // [rsp+8h] [rbp-58h]
  char **v30; // [rsp+8h] [rbp-58h]
  char *srca; // [rsp+10h] [rbp-50h]
  char *srcb; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+10h] [rbp-50h]
  __int64 *v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+20h] [rbp-40h]

  v3 = (char *)0xAAAAAAAAAAAAAAABLL;
  v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  if ( !(_DWORD)v4 )
    return 1;
  v34 = (__int64 *)a1;
  v5 = *(_QWORD *)a1;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v35 = 24LL * (unsigned int)v4;
  while ( 1 )
  {
    v10 = (char **)(v9 + v5);
    v11 = v10[1];
    v12 = *v10;
    v13 = v11 - *v10;
    if ( v13 <= v6 - v8 )
      break;
    if ( v13 )
    {
      if ( (v13 & 0x8000000000000000LL) != 0LL )
        sub_4261EA(a1, v12, v3);
      srca = *v10;
      a1 = sub_22077B0(v10[1] - *v10);
      v27 = (char *)memcpy((void *)a1, srca, v13);
    }
    else
    {
      v27 = 0;
    }
    if ( v8 )
    {
      a1 = (__int64)v8;
      srcb = v27;
      j_j___libc_free_0((unsigned __int64)v8);
      v27 = srcb;
    }
    v6 = &v27[v13];
    v8 = v27;
    v7 = &v27[v13];
LABEL_8:
    if ( v8 != v7 )
      goto LABEL_9;
LABEL_29:
    v9 += 24;
    v25 = &v8[a3];
    v3 = &v8[a2];
    v26 = *v3;
    *v3 = *v25;
    *v25 = v26;
    if ( v35 == v9 )
      goto LABEL_30;
LABEL_20:
    v5 = *v34;
  }
  v14 = v7 - v8;
  if ( v13 <= v7 - v8 )
  {
    if ( v13 )
    {
      a1 = (__int64)v8;
      src = v6;
      memmove(v8, v12, v10[1] - *v10);
      v6 = src;
    }
    v7 = &v8[v13];
    goto LABEL_8;
  }
  if ( v14 )
  {
    a1 = (__int64)v8;
    v28 = v6;
    v30 = v10;
    memmove(v8, v12, v14);
    v6 = v28;
    v14 = v7 - v8;
    v11 = v30[1];
    v12 = *v30;
  }
  v23 = &v12[v14];
  v24 = v11 - v23;
  if ( v24 )
  {
    a1 = (__int64)v7;
    v29 = v6;
    memmove(v7, v23, v24);
    v6 = v29;
    v7 = &v8[v13];
    goto LABEL_8;
  }
  v7 = &v8[v13];
  if ( v8 == &v8[v13] )
    goto LABEL_29;
LABEL_9:
  v15 = v8;
  v16 = v8;
  do
  {
    v17 = *v16;
    if ( *v16 == 60 )
      break;
    if ( v17 == 62 || v17 == 42 )
    {
LABEL_21:
      v21 = 0;
      goto LABEL_22;
    }
    ++v16;
  }
  while ( v7 != v16 );
  v18 = &v8[a3];
  v3 = &v8[a2];
  v19 = *v3;
  a1 = (unsigned __int8)*v18;
  *v3 = a1;
  *v18 = v19;
  do
  {
    v20 = *v15;
    if ( *v15 == 60 )
      break;
    if ( v20 == 62 || v20 == 42 )
      goto LABEL_21;
    ++v15;
  }
  while ( v7 != v15 );
  v9 += 24;
  if ( v35 != v9 )
    goto LABEL_20;
LABEL_30:
  v21 = 1;
LABEL_22:
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  return v21;
}
