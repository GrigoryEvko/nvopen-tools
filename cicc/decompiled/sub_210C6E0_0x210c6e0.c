// Function: sub_210C6E0
// Address: 0x210c6e0
//
__int64 *__fastcall sub_210C6E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 i; // r12
  int v6; // eax
  void *v7; // r9
  __int64 v8; // r15
  __int64 **v9; // r13
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r14
  const char *v13; // rsi
  size_t v14; // rdx
  const char *v15; // rdi
  __int64 v16; // r14
  int v17; // eax
  void *v18; // r9
  __int64 *v19; // r12
  void *v20; // rdx
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r12
  __int64 *v25; // rax
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 *v31; // [rsp+30h] [rbp-40h]
  const char *v32; // [rsp+30h] [rbp-40h]
  void *s2a; // [rsp+38h] [rbp-38h]
  void *s2b; // [rsp+38h] [rbp-38h]
  void *s2; // [rsp+38h] [rbp-38h]

  v28 = (a3 - 1) / 2;
  v27 = a3 & 1;
  if ( a2 >= v28 )
  {
    v9 = (__int64 **)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_29;
    v8 = a2;
    goto LABEL_32;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = (__int64 **)(a1 + 16 * (i + 1));
    v31 = *v9;
    v10 = sub_1649960(**(v9 - 1));
    v12 = v11;
    v13 = v10;
    v15 = sub_1649960(*v31);
    v7 = (void *)v14;
    if ( v14 <= v12 )
      break;
    if ( v12 )
    {
      s2b = (void *)v14;
      v6 = memcmp(v15, v13, v12);
      v7 = s2b;
      if ( v6 )
        goto LABEL_13;
LABEL_6:
      if ( (unsigned __int64)v7 >= v12 )
        goto LABEL_8;
      goto LABEL_7;
    }
LABEL_8:
    *(_QWORD *)(a1 + 8 * i) = *v9;
    if ( v8 >= v28 )
      goto LABEL_15;
LABEL_9:
    ;
  }
  if ( !v14 || (s2a = (void *)v14, v6 = memcmp(v15, v13, v14), v7 = s2a, !v6) )
  {
    if ( v7 == (void *)v12 )
      goto LABEL_8;
    goto LABEL_6;
  }
LABEL_13:
  if ( v6 < 0 )
  {
LABEL_7:
    --v8;
    v9 = (__int64 **)(a1 + 8 * v8);
    goto LABEL_8;
  }
  *(_QWORD *)(a1 + 8 * i) = *v9;
  if ( v8 < v28 )
    goto LABEL_9;
LABEL_15:
  if ( !v27 )
  {
LABEL_32:
    if ( (a3 - 2) / 2 == v8 )
    {
      v25 = *(__int64 **)(a1 + 8 * (2 * v8 + 2) - 8);
      v8 = 2 * v8 + 1;
      *v9 = v25;
      v9 = (__int64 **)(a1 + 8 * v8);
    }
  }
  v16 = (v8 - 1) / 2;
  if ( v8 <= a2 )
    goto LABEL_29;
  while ( 2 )
  {
    v9 = (__int64 **)(a1 + 8 * v16);
    v19 = *v9;
    v32 = sub_1649960(*a4);
    s2 = v20;
    v21 = sub_1649960(*v19);
    v18 = s2;
    v23 = v22;
    if ( (unsigned __int64)s2 >= v22 )
    {
      if ( v22 )
      {
        v17 = memcmp(v21, v32, v22);
        v18 = s2;
        if ( v17 )
          goto LABEL_27;
      }
      if ( v18 == (void *)v23 )
        break;
      goto LABEL_21;
    }
    if ( !s2 )
      break;
    v17 = memcmp(v21, v32, (size_t)s2);
    v18 = s2;
    if ( !v17 )
    {
LABEL_21:
      if ( (unsigned __int64)v18 <= v23 )
        break;
      goto LABEL_22;
    }
LABEL_27:
    if ( v17 < 0 )
    {
LABEL_22:
      *(_QWORD *)(a1 + 8 * v8) = *v9;
      v8 = v16;
      if ( a2 >= v16 )
        goto LABEL_29;
      v16 = (v16 - 1) / 2;
      continue;
    }
    break;
  }
  v9 = (__int64 **)(a1 + 8 * v8);
LABEL_29:
  *v9 = a4;
  return a4;
}
