// Function: sub_B41CA0
// Address: 0xb41ca0
//
__int64 __fastcall sub_B41CA0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // r12
  __int64 v7; // r14
  unsigned __int64 v8; // r13
  bool v9; // cc
  __int64 v10; // r14
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 v16; // r12
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 v22; // rcx
  __int64 v23; // r13
  unsigned __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // r14
  unsigned __int64 v27; // r13
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 *v29; // [rsp+10h] [rbp-50h]
  __int64 *v30; // [rsp+18h] [rbp-48h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 *i; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v29 = a2;
  v28 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v30 = a2;
    goto LABEL_19;
  }
  while ( 2 )
  {
    --v28;
    v6 = &a1[result >> 4];
    v7 = *v6;
    v8 = *(_QWORD *)(sub_B3FBB0(a4, a1[1]) + 784);
    v9 = v8 <= *(_QWORD *)(sub_B3FBB0(a4, v7) + 784);
    v10 = *(v29 - 1);
    if ( v9 )
    {
      v19 = *(_QWORD *)(sub_B3FBB0(a4, a1[1]) + 784);
      if ( v19 <= *(_QWORD *)(sub_B3FBB0(a4, v10) + 784) )
      {
        v26 = *(v29 - 1);
        v27 = *(_QWORD *)(sub_B3FBB0(a4, *v6) + 784);
        v9 = v27 <= *(_QWORD *)(sub_B3FBB0(a4, v26) + 784);
        v12 = *a1;
        if ( !v9 )
        {
          *a1 = *(v29 - 1);
          *(v29 - 1) = v12;
          v13 = *a1;
          v14 = a1[1];
          goto LABEL_7;
        }
        goto LABEL_6;
      }
LABEL_17:
      v14 = *a1;
      v13 = a1[1];
      a1[1] = *a1;
      *a1 = v13;
      goto LABEL_7;
    }
    v11 = *(_QWORD *)(sub_B3FBB0(a4, *v6) + 784);
    if ( v11 <= *(_QWORD *)(sub_B3FBB0(a4, v10) + 784) )
    {
      v23 = *(v29 - 1);
      v24 = *(_QWORD *)(sub_B3FBB0(a4, a1[1]) + 784);
      if ( v24 > *(_QWORD *)(sub_B3FBB0(a4, v23) + 784) )
      {
        v25 = *a1;
        *a1 = *(v29 - 1);
        *(v29 - 1) = v25;
        v13 = *a1;
        v14 = a1[1];
        goto LABEL_7;
      }
      goto LABEL_17;
    }
    v12 = *a1;
LABEL_6:
    *a1 = *v6;
    *v6 = v12;
    v13 = *a1;
    v14 = a1[1];
LABEL_7:
    v15 = v29;
    for ( i = a1 + 1; ; ++i )
    {
      v30 = i;
      v31 = *(_QWORD *)(sub_B3FBB0(a4, v14) + 784);
      if ( v31 > *(_QWORD *)(sub_B3FBB0(a4, v13) + 784) )
        goto LABEL_8;
      do
      {
        v16 = *--v15;
        v17 = *(_QWORD *)(sub_B3FBB0(a4, *a1) + 784);
      }
      while ( v17 > *(_QWORD *)(sub_B3FBB0(a4, v16) + 784) );
      if ( i >= v15 )
        break;
      v18 = *i;
      *i = *v15;
      *v15 = v18;
LABEL_8:
      v13 = *a1;
      v14 = i[1];
    }
    sub_B41CA0(i, v29, v28, a4);
    result = (char *)i - (char *)a1;
    if ( (char *)i - (char *)a1 > 128 )
    {
      if ( v28 )
      {
        v29 = i;
        continue;
      }
LABEL_19:
      v20 = result >> 3;
      v21 = ((result >> 3) - 2) >> 1;
      sub_B41AE0((__int64)a1, v21, result >> 3, a1[v21], a4);
      do
      {
        --v21;
        sub_B41AE0((__int64)a1, v21, v20, a1[v21], a4);
      }
      while ( v21 );
      do
      {
        v22 = *--v30;
        *v30 = *a1;
        result = sub_B41AE0((__int64)a1, 0, v30 - a1, v22, a4);
      }
      while ( (char *)v30 - (char *)a1 > 8 );
    }
    return result;
  }
}
