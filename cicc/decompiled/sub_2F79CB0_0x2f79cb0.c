// Function: sub_2F79CB0
// Address: 0x2f79cb0
//
__int64 *__fastcall sub_2F79CB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 i; // r13
  __int64 v5; // r15
  __int64 **v6; // r14
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r12
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r9
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // r9
  __int64 *v17; // r13
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // r12
  const char *v21; // rax
  __int64 v22; // r9
  size_t v23; // rdx
  size_t v24; // r13
  size_t v25; // rdx
  int v26; // eax
  __int64 *v28; // rax
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  const char *s2; // [rsp+30h] [rbp-40h]
  const char *s2a; // [rsp+30h] [rbp-40h]
  __int64 *v35; // [rsp+38h] [rbp-38h]
  size_t v36; // [rsp+38h] [rbp-38h]
  __int64 v37; // [rsp+38h] [rbp-38h]

  v30 = a3 & 1;
  v31 = (a3 - 1) / 2;
  if ( a2 < v31 )
  {
    for ( i = a2; ; i = v5 )
    {
      v5 = 2 * (i + 1);
      v6 = (__int64 **)(a1 + 16 * (i + 1));
      v35 = *v6;
      v7 = sub_BD5D20(**(v6 - 1));
      v9 = v8;
      s2 = v7;
      v10 = sub_BD5D20(*v35);
      v12 = v11;
      v13 = v11 <= v9;
      v14 = v9;
      if ( v13 )
        v14 = v12;
      if ( !v14 )
        break;
      v36 = v12;
      v15 = memcmp(v10, s2, v14);
      v12 = v36;
      if ( !v15 )
        break;
      if ( v15 < 0 )
        goto LABEL_5;
      *(_QWORD *)(a1 + 8 * i) = *v6;
      if ( v5 >= v31 )
      {
LABEL_14:
        if ( v30 )
          goto LABEL_15;
        goto LABEL_26;
      }
LABEL_7:
      ;
    }
    if ( v12 != v9 && v12 < v9 )
    {
LABEL_5:
      --v5;
      v6 = (__int64 **)(a1 + 8 * v5);
    }
    *(_QWORD *)(a1 + 8 * i) = *v6;
    if ( v5 >= v31 )
      goto LABEL_14;
    goto LABEL_7;
  }
  v6 = (__int64 **)(a1 + 8 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_23;
  v5 = a2;
LABEL_26:
  if ( (a3 - 2) / 2 == v5 )
  {
    v28 = *(__int64 **)(a1 + 8 * (2 * v5 + 2) - 8);
    v5 = 2 * v5 + 1;
    *v6 = v28;
    v6 = (__int64 **)(a1 + 8 * v5);
  }
LABEL_15:
  v16 = (v5 - 1) / 2;
  if ( v5 > a2 )
  {
    while ( 1 )
    {
      v6 = (__int64 **)(a1 + 8 * v16);
      v37 = v16;
      v17 = *v6;
      v18 = sub_BD5D20(*a4);
      v20 = v19;
      s2a = v18;
      v21 = sub_BD5D20(*v17);
      v22 = v37;
      v13 = v23 <= v20;
      v24 = v23;
      v25 = v20;
      if ( v13 )
        v25 = v24;
      if ( v25 && (v26 = memcmp(v21, s2a, v25), v22 = v37, v26) )
      {
        if ( v26 >= 0 )
          goto LABEL_22;
      }
      else if ( v24 == v20 || v24 >= v20 )
      {
LABEL_22:
        v6 = (__int64 **)(a1 + 8 * v5);
        break;
      }
      *(_QWORD *)(a1 + 8 * v5) = *v6;
      v5 = v22;
      if ( a2 >= v22 )
        break;
      v16 = (v22 - 1) / 2;
    }
  }
LABEL_23:
  *v6 = a4;
  return a4;
}
