// Function: sub_1A91360
// Address: 0x1a91360
//
__int64 __fastcall sub_1A91360(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // eax
  void *v6; // rcx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 *v9; // r13
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r12
  const char *v13; // rsi
  size_t v14; // rdx
  const char *v15; // rdi
  __int64 v16; // r14
  __int64 v17; // r13
  int v18; // eax
  __int64 *v19; // rbx
  __int64 v20; // r12
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r15
  const char *v24; // rsi
  size_t v25; // rdx
  const char *v26; // rdi
  size_t v27; // r12
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+38h] [rbp-48h]
  void *s2; // [rsp+40h] [rbp-40h]
  void *s2a; // [rsp+40h] [rbp-40h]
  __int64 i; // [rsp+48h] [rbp-38h]
  __int64 v38; // [rsp+48h] [rbp-38h]

  v30 = a3 & 1;
  v31 = (a3 - 1) / 2;
  if ( a2 < v31 )
  {
    for ( i = a2; ; i = v7 )
    {
      v7 = 2 * (i + 1);
      v8 = (__int64 *)(a1 + 16 * (i + 1));
      v9 = (__int64 *)(a1 + 8 * (v7 - 1));
      v34 = *v8;
      v10 = sub_1649960(*(_QWORD *)(*v9 + 40));
      v12 = v11;
      v13 = v10;
      v15 = sub_1649960(*(_QWORD *)(v34 + 40));
      v6 = (void *)v14;
      if ( v14 <= v12 )
      {
        if ( v14 )
        {
          s2 = (void *)v14;
          v5 = memcmp(v15, v13, v14);
          v6 = s2;
          if ( v5 )
            goto LABEL_13;
        }
        if ( v6 == (void *)v12 )
          goto LABEL_8;
      }
      else
      {
        if ( !v12 )
          goto LABEL_8;
        s2a = (void *)v14;
        v5 = memcmp(v15, v13, v12);
        v6 = s2a;
        if ( v5 )
        {
LABEL_13:
          if ( v5 < 0 )
          {
            --v7;
            v8 = v9;
          }
          goto LABEL_8;
        }
      }
      if ( (unsigned __int64)v6 < v12 )
      {
        --v7;
        v8 = v9;
      }
LABEL_8:
      *(_QWORD *)(a1 + 8 * i) = *v8;
      if ( v7 >= v31 )
      {
        if ( v30 )
          goto LABEL_17;
        goto LABEL_33;
      }
    }
  }
  v8 = (__int64 *)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v7 = a2;
LABEL_33:
    if ( (a3 - 2) / 2 == v7 )
    {
      v7 = 2 * v7 + 1;
      *v8 = *(_QWORD *)(a1 + 8 * v7);
      v8 = (__int64 *)(a1 + 8 * v7);
    }
LABEL_17:
    if ( v7 > a2 )
    {
      v16 = (v7 - 1) / 2;
      v38 = a1;
      v17 = v7;
      while ( 1 )
      {
        v19 = (__int64 *)(v38 + 8 * v16);
        v20 = *v19;
        v21 = sub_1649960(*(_QWORD *)(a4 + 40));
        v23 = v22;
        v24 = v21;
        v26 = sub_1649960(*(_QWORD *)(v20 + 40));
        v27 = v25;
        if ( v23 >= v25 )
        {
          if ( !v25 || (v18 = memcmp(v26, v24, v25)) == 0 )
          {
            if ( v23 == v27 )
              goto LABEL_29;
LABEL_22:
            if ( v23 <= v27 )
              goto LABEL_29;
            goto LABEL_23;
          }
        }
        else
        {
          if ( !v23 )
            goto LABEL_29;
          v18 = memcmp(v26, v24, v23);
          if ( !v18 )
            goto LABEL_22;
        }
        if ( v18 >= 0 )
        {
LABEL_29:
          v8 = (__int64 *)(v38 + 8 * v17);
          break;
        }
LABEL_23:
        *(_QWORD *)(v38 + 8 * v17) = *v19;
        v17 = v16;
        if ( a2 >= v16 )
        {
          v8 = (__int64 *)(v38 + 8 * v16);
          break;
        }
        v16 = (v16 - 1) / 2;
      }
    }
  }
  *v8 = a4;
  return a4;
}
