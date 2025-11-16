// Function: sub_2C70E10
// Address: 0x2c70e10
//
__int64 *__fastcall sub_2C70E10(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 i; // r13
  __int64 **v5; // r15
  __int64 v6; // rbx
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r12
  const char *v10; // rax
  void *v11; // rdx
  void *v12; // r10
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // r14
  __int64 *v17; // r13
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // r12
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r13
  size_t v24; // rdx
  int v25; // eax
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  const char *s2; // [rsp+30h] [rbp-50h]
  __int64 *v32; // [rsp+40h] [rbp-40h]
  void *v33; // [rsp+40h] [rbp-40h]
  const char *v34; // [rsp+40h] [rbp-40h]

  v28 = a3 & 1;
  v29 = (a3 - 1) / 2;
  if ( a2 >= v29 )
  {
    v5 = (__int64 **)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_24;
    v6 = a2;
    goto LABEL_27;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v5 = (__int64 **)(a1 + 16 * (i + 1));
    v32 = *v5;
    v7 = sub_BD5D20(**(_QWORD **)(a1 + 8 * (v6 - 1)));
    v9 = v8;
    s2 = v7;
    v10 = sub_BD5D20(*v32);
    v12 = v11;
    v13 = (unsigned __int64)v11 <= v9;
    v14 = v9;
    if ( v13 )
      v14 = (size_t)v12;
    if ( v14 && (v33 = v12, v15 = memcmp(v10, s2, v14), v12 = v33, v15) )
    {
      if ( v15 < 0 )
        v5 = (__int64 **)(a1 + 8 * --v6);
    }
    else if ( v12 != (void *)v9 && (unsigned __int64)v12 < v9 )
    {
      v5 = (__int64 **)(a1 + 8 * --v6);
    }
    *(_QWORD *)(a1 + 8 * i) = *v5;
    if ( v6 >= v29 )
      break;
  }
  if ( !v28 )
  {
LABEL_27:
    if ( (a3 - 2) / 2 == v6 )
    {
      v6 = 2 * v6 + 1;
      *v5 = *(__int64 **)(a1 + 8 * v6);
      v5 = (__int64 **)(a1 + 8 * v6);
    }
  }
  v16 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v5 = (__int64 **)(a1 + 8 * v16);
      v17 = *v5;
      v18 = sub_BD5D20(*a4);
      v20 = v19;
      v34 = v18;
      v21 = sub_BD5D20(*v17);
      v23 = v22;
      v13 = v22 <= v20;
      v24 = v20;
      if ( v13 )
        v24 = v23;
      if ( v24 && (v25 = memcmp(v21, v34, v24)) != 0 )
      {
        if ( v25 >= 0 )
          goto LABEL_23;
      }
      else if ( v23 == v20 || v23 >= v20 )
      {
LABEL_23:
        v5 = (__int64 **)(a1 + 8 * v6);
        break;
      }
      *(_QWORD *)(a1 + 8 * v6) = *v5;
      v6 = v16;
      if ( a2 >= v16 )
        break;
      v16 = (v16 - 1) / 2;
    }
  }
LABEL_24:
  *v5 = a4;
  return a4;
}
