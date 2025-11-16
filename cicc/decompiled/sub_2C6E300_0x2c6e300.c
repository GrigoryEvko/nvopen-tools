// Function: sub_2C6E300
// Address: 0x2c6e300
//
__int64 __fastcall sub_2C6E300(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r13
  void **v5; // r15
  __int64 v6; // rbx
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r12
  const char *v10; // rax
  void *v11; // rdx
  void *v12; // r8
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  const char *v20; // rax
  size_t v21; // rdx
  size_t v22; // r12
  const char *v23; // rax
  size_t v24; // rdx
  size_t v25; // r13
  size_t v26; // rdx
  int v27; // eax
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  const char *s2; // [rsp+28h] [rbp-58h]
  void *v34; // [rsp+40h] [rbp-40h]
  void *v35; // [rsp+40h] [rbp-40h]
  const char *v36; // [rsp+40h] [rbp-40h]

  v30 = a3 & 1;
  v31 = (a3 - 1) / 2;
  if ( a2 < v31 )
  {
    for ( i = a2; ; i = v6 )
    {
      v6 = 2 * (i + 1);
      v5 = (void **)(a1 + 16 * (i + 1));
      v34 = *v5;
      v7 = sub_BD5D20(*(_QWORD *)(a1 + 8 * (v6 - 1)));
      v9 = v8;
      s2 = v7;
      v10 = sub_BD5D20((__int64)v34);
      v12 = v11;
      v13 = (unsigned __int64)v11 <= v9;
      v14 = v9;
      if ( v13 )
        v14 = (size_t)v12;
      if ( v14 && (v35 = v12, v15 = memcmp(v10, s2, v14), v12 = v35, v15) )
      {
        if ( v15 < 0 )
          v5 = (void **)(a1 + 8 * --v6);
        *(_QWORD *)(a1 + 8 * i) = *v5;
        if ( v6 >= v31 )
          goto LABEL_15;
      }
      else
      {
        if ( v12 != (void *)v9 && (unsigned __int64)v12 < v9 )
          v5 = (void **)(a1 + 8 * --v6);
        *(_QWORD *)(a1 + 8 * i) = *v5;
        if ( v6 >= v31 )
        {
LABEL_15:
          if ( v30 )
            goto LABEL_16;
          goto LABEL_28;
        }
      }
    }
  }
  v5 = (void **)(a1 + 8 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_25;
  v6 = a2;
LABEL_28:
  if ( (a3 - 2) / 2 == v6 )
  {
    v6 = 2 * v6 + 1;
    *v5 = *(void **)(a1 + 8 * v6);
    v5 = (void **)(a1 + 8 * v6);
  }
LABEL_16:
  if ( v6 > a2 )
  {
    v16 = v6;
    v17 = (v6 - 1) / 2;
    v18 = v16;
    while ( 1 )
    {
      v5 = (void **)(a1 + 8 * v17);
      v19 = (__int64)*v5;
      v20 = sub_BD5D20(a4);
      v22 = v21;
      v36 = v20;
      v23 = sub_BD5D20(v19);
      v25 = v24;
      v13 = v24 <= v22;
      v26 = v22;
      if ( v13 )
        v26 = v25;
      if ( v26 && (v27 = memcmp(v23, v36, v26)) != 0 )
      {
        if ( v27 >= 0 )
          goto LABEL_24;
      }
      else if ( v25 == v22 || v25 >= v22 )
      {
LABEL_24:
        v5 = (void **)(a1 + 8 * v18);
        break;
      }
      *(_QWORD *)(a1 + 8 * v18) = *v5;
      v18 = v17;
      if ( a2 >= v17 )
        break;
      v17 = (v17 - 1) / 2;
    }
  }
LABEL_25:
  *v5 = (void *)a4;
  return a4;
}
