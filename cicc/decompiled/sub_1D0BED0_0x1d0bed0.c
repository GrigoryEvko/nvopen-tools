// Function: sub_1D0BED0
// Address: 0x1d0bed0
//
__int64 __fastcall sub_1D0BED0(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r9
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  char *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // r13
  __int64 *v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 *v21; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v17 = a2;
    goto LABEL_23;
  }
  v7 = (__int64 *)(a1 + 8);
  v21 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v8 = *((_QWORD *)a1 + 1);
    v9 = *(v4 - 1);
    --v6;
    v10 = &a1[8 * (result >> 4)];
    v11 = *(_QWORD *)a1;
    v12 = *(_QWORD *)v10;
    if ( v8 >= *(_QWORD *)v10 )
    {
      if ( v8 < v9 )
        goto LABEL_7;
      if ( v12 < v9 )
      {
LABEL_17:
        *(_QWORD *)a1 = v9;
        v13 = v11;
        *(v4 - 1) = v11;
        v8 = *(_QWORD *)a1;
        v11 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_QWORD *)a1 = v12;
      *(_QWORD *)v10 = v11;
      v11 = *((_QWORD *)a1 + 1);
      v8 = *(_QWORD *)a1;
      v13 = *(v4 - 1);
      goto LABEL_8;
    }
    if ( v12 < v9 )
      goto LABEL_21;
    if ( v8 < v9 )
      goto LABEL_17;
LABEL_7:
    *(_QWORD *)a1 = v8;
    *((_QWORD *)a1 + 1) = v11;
    v13 = *(v4 - 1);
LABEL_8:
    v14 = v21;
    v15 = v7;
    v16 = v4;
    while ( 1 )
    {
      v17 = v15;
      if ( v11 < v8 )
        goto LABEL_14;
      for ( --v16; v13 > v8; --v16 )
        v13 = *(v16 - 1);
      if ( v15 >= v16 )
        break;
      *v15 = v13;
      v13 = *(v16 - 1);
      *v16 = v11;
      v8 = *(_QWORD *)a1;
LABEL_14:
      v11 = *v14;
      ++v15;
      ++v14;
    }
    sub_1D0BED0(v15, v4, v6);
    result = (char *)v15 - a1;
    if ( (char *)v15 - a1 > 128 )
    {
      if ( v6 )
      {
        v4 = v15;
        continue;
      }
LABEL_23:
      v18 = result >> 3;
      v19 = ((result >> 3) - 2) >> 1;
      sub_1D0B970((__int64)a1, v19, result >> 3, *(_QWORD *)&a1[8 * v19]);
      do
      {
        --v19;
        sub_1D0B970((__int64)a1, v19, v18, *(_QWORD *)&a1[8 * v19]);
      }
      while ( v19 );
      do
      {
        v20 = *--v17;
        *v17 = *(_QWORD *)a1;
        result = sub_1D0B970((__int64)a1, 0, ((char *)v17 - a1) >> 3, v20);
      }
      while ( (char *)v17 - a1 > 8 );
    }
    return result;
  }
}
