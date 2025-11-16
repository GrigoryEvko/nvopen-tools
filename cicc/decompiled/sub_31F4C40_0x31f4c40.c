// Function: sub_31F4C40
// Address: 0x31f4c40
//
__int64 __fastcall sub_31F4C40(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r10
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 v8; // r13
  __int64 v9; // r8
  char *v10; // rdi
  __int64 v11; // rax
  unsigned __int16 v12; // si
  unsigned __int16 v13; // cx
  unsigned __int16 v14; // dx
  __int64 v15; // rdx
  __int64 *v16; // rdi
  __int64 *v17; // r13
  unsigned __int16 v18; // si
  __int64 *v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 *v24; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_23;
  }
  v7 = (__int64 *)(a1 + 8);
  v24 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v8 = *(v4 - 1);
    v9 = *(_QWORD *)a1;
    --v6;
    v10 = &a1[8 * (result >> 4)];
    v11 = *((_QWORD *)a1 + 1);
    v12 = *(_WORD *)(*(_QWORD *)v8 + 20LL);
    v13 = *(_WORD *)(*(_QWORD *)v11 + 20LL);
    v14 = *(_WORD *)(**(_QWORD **)v10 + 20LL);
    if ( v13 >= v14 )
    {
      if ( v13 < v12 )
        goto LABEL_7;
      if ( v14 < v12 )
      {
LABEL_17:
        *(_QWORD *)a1 = v8;
        v15 = v9;
        *(v4 - 1) = v9;
        v11 = *(_QWORD *)a1;
        v9 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_QWORD *)a1 = *(_QWORD *)v10;
      *(_QWORD *)v10 = v9;
      v15 = *(v4 - 1);
      v11 = *(_QWORD *)a1;
      v9 = *((_QWORD *)a1 + 1);
      goto LABEL_8;
    }
    if ( v14 < v12 )
      goto LABEL_21;
    if ( v13 < v12 )
      goto LABEL_17;
LABEL_7:
    *(_QWORD *)a1 = v11;
    *((_QWORD *)a1 + 1) = v9;
    v15 = *(v4 - 1);
LABEL_8:
    v16 = v24;
    v17 = v7;
    v18 = *(_WORD *)(*(_QWORD *)v11 + 20LL);
    v19 = v4;
    while ( 1 )
    {
      v20 = v17;
      if ( *(_WORD *)(*(_QWORD *)v9 + 20LL) < v18 )
        goto LABEL_14;
      for ( --v19; *(_WORD *)(*(_QWORD *)v15 + 20LL) > v18; --v19 )
        v15 = *(v19 - 1);
      if ( v17 >= v19 )
        break;
      *v17 = v15;
      v15 = *(v19 - 1);
      *v19 = v9;
      v18 = *(_WORD *)(**(_QWORD **)a1 + 20LL);
LABEL_14:
      v9 = *v16;
      ++v17;
      ++v16;
    }
    sub_31F4C40(v17, v4, v6);
    result = (char *)v17 - a1;
    if ( (char *)v17 - a1 > 128 )
    {
      if ( v6 )
      {
        v4 = v17;
        continue;
      }
LABEL_23:
      v21 = result >> 3;
      v22 = ((result >> 3) - 2) >> 1;
      sub_31F3DE0((__int64)a1, v22, result >> 3, *(_QWORD *)&a1[8 * v22]);
      do
      {
        --v22;
        sub_31F3DE0((__int64)a1, v22, v21, *(_QWORD *)&a1[8 * v22]);
      }
      while ( v22 );
      do
      {
        v23 = *--v20;
        *v20 = *(_QWORD *)a1;
        result = sub_31F3DE0((__int64)a1, 0, ((char *)v20 - a1) >> 3, v23);
      }
      while ( (char *)v20 - a1 > 8 );
    }
    return result;
  }
}
