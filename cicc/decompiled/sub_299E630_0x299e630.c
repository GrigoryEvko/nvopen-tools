// Function: sub_299E630
// Address: 0x299e630
//
_QWORD *__fastcall sub_299E630(_QWORD *a1, unsigned int *a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // rcx
  __int64 v8; // r14
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rax
  void *v12; // rdi
  unsigned __int64 v13; // rax
  size_t v14; // r12
  __int64 v15; // r12
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rax
  size_t v19; // r10
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  size_t v23; // rbx
  unsigned __int64 v24; // rbx
  size_t v25; // rbx
  size_t v26; // [rsp+8h] [rbp-48h]
  size_t v27; // [rsp+8h] [rbp-48h]
  const void *v28; // [rsp+10h] [rbp-40h]

  v6 = a1 + 3;
  *a1 = a1 + 3;
  a1[1] = 0;
  a1[2] = 64;
  v8 = *(_QWORD *)a2;
  v9 = 0;
  v10 = *a3;
  v11 = *(_QWORD *)(*(_QWORD *)a2 + 40LL);
  v28 = v6;
  if ( v11 >= *a3 )
  {
    v12 = (void *)v6;
    v13 = v11 / v10;
    v14 = v13;
    if ( v13 > 0x40 )
    {
      sub_C8D290((__int64)a1, v6, v13, 1u, a5, a6);
      v12 = (void *)(*a1 + a1[1]);
    }
    memset(v12, 241, v14);
    v9 = v14 + a1[1];
    a1[1] = v9;
    v8 = *(_QWORD *)a2;
  }
  v15 = v8 + 56LL * a2[2];
  while ( v8 != v15 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v8 + 40) / v10;
      if ( v18 != v9 )
      {
        if ( v18 >= v9 )
        {
          v23 = v18 - v9;
          if ( v18 > a1[2] )
          {
            sub_C8D290((__int64)a1, v28, *(_QWORD *)(v8 + 40) / v10, 1u, a5, a6);
            v9 = a1[1];
          }
          memset((void *)(*a1 + v9), 242, v23);
          v24 = a1[1] + v23;
          a1[1] = v24;
          v9 = v24;
        }
        else
        {
          a1[1] = v18;
          v9 = v18;
        }
      }
      v16 = *(_QWORD *)(v8 + 8);
      v19 = v16 / v10;
      v20 = v9 + v16 / v10;
      if ( v20 != v9 )
        break;
LABEL_8:
      v17 = v16 % v10;
      if ( !v17 )
        goto LABEL_9;
LABEL_20:
      if ( v9 + 1 > a1[2] )
      {
        sub_C8D290((__int64)a1, v28, v9 + 1, 1u, a5, a6);
        v9 = a1[1];
      }
      v8 += 56;
      *(_BYTE *)(*a1 + v9) = v17;
      v9 = a1[1] + 1LL;
      a1[1] = v9;
      if ( v8 == v15 )
        goto LABEL_23;
    }
    if ( __CFADD__(v9, v19) )
    {
      a1[1] = v20;
      v16 = *(_QWORD *)(v8 + 8);
      v9 += v19;
      goto LABEL_8;
    }
    if ( v20 > a1[2] )
    {
      v27 = *(_QWORD *)(v8 + 8) / v10;
      sub_C8D290((__int64)a1, v28, v20, 1u, a5, a6);
      v9 = a1[1];
      v19 = v27;
    }
    if ( v16 >= v10 )
    {
      v26 = v19;
      memset((void *)(*a1 + v9), 0, v19);
      v9 = a1[1];
      v19 = v26;
    }
    v9 += v19;
    a1[1] = v9;
    v17 = *(_QWORD *)(v8 + 8) % v10;
    if ( v17 )
      goto LABEL_20;
LABEL_9:
    v8 += 56;
  }
LABEL_23:
  v21 = a3[2] / v10;
  if ( v21 != v9 )
  {
    if ( v21 >= v9 )
    {
      v25 = v21 - v9;
      if ( v21 > a1[2] )
      {
        sub_C8D290((__int64)a1, v28, a3[2] / v10, 1u, a5, a6);
        v9 = a1[1];
      }
      memset((void *)(*a1 + v9), 243, v25);
      a1[1] += v25;
    }
    else
    {
      a1[1] = v21;
    }
  }
  return a1;
}
