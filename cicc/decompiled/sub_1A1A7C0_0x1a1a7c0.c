// Function: sub_1A1A7C0
// Address: 0x1a1a7c0
//
__int64 __fastcall sub_1A1A7C0(char *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // r10
  __int64 v7; // r12
  __int64 v9; // r15
  __int64 v10; // rdi
  char *v11; // r11
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // rsi
  __int64 *v18; // r14
  unsigned __int64 v19; // rcx
  __int64 *v20; // rax
  __int64 *v21; // r15
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rcx
  __int64 *v25; // [rsp-48h] [rbp-48h]
  __int64 *v26; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v5 = a2;
  v7 = a3;
  if ( !a3 )
  {
    v21 = a2;
    goto LABEL_23;
  }
  v26 = (__int64 *)(a1 + 8);
  v25 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v9 = *(v5 - 1);
    v10 = *(_QWORD *)a1;
    --v7;
    v11 = &a1[8 * (result >> 4)];
    v12 = *((_QWORD *)a1 + 1);
    v13 = *(_QWORD *)(v9 + 32);
    v14 = *(_QWORD *)(v12 + 32);
    v15 = *(_QWORD *)(*(_QWORD *)v11 + 32LL);
    if ( v14 >= v15 )
    {
      if ( v14 < v13 )
        goto LABEL_7;
      if ( v15 < v13 )
      {
LABEL_17:
        *(_QWORD *)a1 = v9;
        v16 = v10;
        *(v5 - 1) = v10;
        v12 = *(_QWORD *)a1;
        v10 = *((_QWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_QWORD *)a1 = *(_QWORD *)v11;
      *(_QWORD *)v11 = v10;
      v16 = *(v5 - 1);
      v12 = *(_QWORD *)a1;
      v10 = *((_QWORD *)a1 + 1);
      goto LABEL_8;
    }
    if ( v15 < v13 )
      goto LABEL_21;
    if ( v14 < v13 )
      goto LABEL_17;
LABEL_7:
    *(_QWORD *)a1 = v12;
    *((_QWORD *)a1 + 1) = v10;
    v16 = *(v5 - 1);
LABEL_8:
    v17 = v25;
    v18 = v26;
    v19 = *(_QWORD *)(v12 + 32);
    v20 = v5;
    while ( 1 )
    {
      v21 = v18;
      if ( *(_QWORD *)(v10 + 32) < v19 )
        goto LABEL_14;
      for ( --v20; *(_QWORD *)(v16 + 32) > v19; --v20 )
        v16 = *(v20 - 1);
      if ( v18 >= v20 )
        break;
      *v18 = v16;
      v16 = *(v20 - 1);
      *v20 = v10;
      v19 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
LABEL_14:
      v10 = *v17;
      ++v18;
      ++v17;
    }
    sub_1A1A7C0(v18, v5, v7, a4);
    result = (char *)v18 - a1;
    if ( (char *)v18 - a1 > 128 )
    {
      if ( v7 )
      {
        v5 = v18;
        continue;
      }
LABEL_23:
      v22 = result >> 3;
      v23 = ((result >> 3) - 2) >> 1;
      sub_1A1A2F0((__int64)a1, v23, result >> 3, *(_QWORD *)&a1[8 * v23]);
      do
      {
        --v23;
        sub_1A1A2F0((__int64)a1, v23, v22, *(_QWORD *)&a1[8 * v23]);
      }
      while ( v23 );
      do
      {
        v24 = *--v21;
        *v21 = *(_QWORD *)a1;
        result = sub_1A1A2F0((__int64)a1, 0, ((char *)v21 - a1) >> 3, v24);
      }
      while ( (char *)v21 - a1 > 8 );
    }
    return result;
  }
}
