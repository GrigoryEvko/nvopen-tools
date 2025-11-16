// Function: sub_1F2EB90
// Address: 0x1f2eb90
//
__int64 __fastcall sub_1F2EB90(__int64 a1, unsigned int **a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int **v4; // r9
  __int64 v6; // rbx
  unsigned int **v7; // r12
  int *v8; // r13
  unsigned int *v9; // rdi
  unsigned int **v10; // r10
  unsigned int *v11; // rax
  int v12; // esi
  int v13; // ecx
  int v14; // edx
  unsigned int *v15; // rdx
  unsigned int **v16; // rsi
  int v17; // ecx
  unsigned int **v18; // r13
  unsigned int **v19; // rax
  unsigned int **v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r12
  unsigned int *v23; // rcx
  unsigned int **v24; // [rsp-40h] [rbp-40h]

  result = (__int64)a2 - a1;
  if ( (__int64)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_23;
  }
  v7 = (unsigned int **)(a1 + 8);
  v24 = (unsigned int **)(a1 + 16);
  while ( 2 )
  {
    v8 = (int *)*(v4 - 1);
    v9 = *(unsigned int **)a1;
    --v6;
    v10 = (unsigned int **)(a1 + 8 * (result >> 4));
    v11 = *(unsigned int **)(a1 + 8);
    v12 = *v8;
    v13 = *v11;
    v14 = **v10;
    if ( (int)*v11 >= v14 )
    {
      if ( v13 < v12 )
        goto LABEL_7;
      if ( v14 < v12 )
      {
LABEL_17:
        *(_QWORD *)a1 = v8;
        v15 = v9;
        *(v4 - 1) = v9;
        v11 = *(unsigned int **)a1;
        v9 = *(unsigned int **)(a1 + 8);
        goto LABEL_8;
      }
LABEL_21:
      *(_QWORD *)a1 = *v10;
      *v10 = v9;
      v15 = *(v4 - 1);
      v11 = *(unsigned int **)a1;
      v9 = *(unsigned int **)(a1 + 8);
      goto LABEL_8;
    }
    if ( v14 < v12 )
      goto LABEL_21;
    if ( v13 < v12 )
      goto LABEL_17;
LABEL_7:
    *(_QWORD *)a1 = v11;
    *(_QWORD *)(a1 + 8) = v9;
    v15 = *(v4 - 1);
LABEL_8:
    v16 = v24;
    v17 = *v11;
    v18 = v7;
    v19 = v4;
    while ( 1 )
    {
      v20 = v18;
      if ( (int)*v9 < v17 )
        goto LABEL_14;
      for ( --v19; (int)*v15 > v17; --v19 )
        v15 = *(v19 - 1);
      if ( v19 <= v18 )
        break;
      *v18 = v15;
      v15 = *(v19 - 1);
      *v19 = v9;
      v17 = **(_DWORD **)a1;
LABEL_14:
      v9 = *v16;
      ++v18;
      ++v16;
    }
    sub_1F2EB90(v18, v4, v6);
    result = (__int64)v18 - a1;
    if ( (__int64)v18 - a1 > 128 )
    {
      if ( v6 )
      {
        v4 = v18;
        continue;
      }
LABEL_23:
      v21 = result >> 3;
      v22 = ((result >> 3) - 2) >> 1;
      sub_1F2E890(a1, v22, result >> 3, *(unsigned int **)(a1 + 8 * v22));
      do
      {
        --v22;
        sub_1F2E890(a1, v22, v21, *(unsigned int **)(a1 + 8 * v22));
      }
      while ( v22 );
      do
      {
        v23 = *--v20;
        *v20 = *(unsigned int **)a1;
        result = sub_1F2E890(a1, 0, ((__int64)v20 - a1) >> 3, v23);
      }
      while ( (__int64)v20 - a1 > 8 );
    }
    return result;
  }
}
