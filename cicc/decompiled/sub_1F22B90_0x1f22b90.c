// Function: sub_1F22B90
// Address: 0x1f22b90
//
__int64 __fastcall sub_1F22B90(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r8
  __int64 v5; // r14
  __int64 *v7; // rbx
  __int64 v8; // rdi
  __int64 *v9; // r10
  __int64 v10; // r9
  unsigned int v11; // ecx
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 v14; // rsi
  __int64 v15; // r10
  __int64 *v16; // rdi
  __int64 *v17; // rdx
  int v18; // r11d
  __int64 *v19; // r12
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rcx

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  v4 = a2;
  v5 = a3;
  if ( !a3 )
  {
    v19 = a2;
    goto LABEL_23;
  }
  v7 = a1 + 2;
  while ( 2 )
  {
    v8 = a1[1];
    --v5;
    v9 = &a1[result >> 4];
    v10 = *a1;
    v11 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v8 >> 1) & 3;
    v12 = *(_DWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v9 >> 1) & 3;
    v13 = *(_DWORD *)((*(v4 - 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(v4 - 1) >> 1) & 3;
    if ( v11 >= v12 )
    {
      if ( v11 < v13 )
        goto LABEL_7;
      if ( v12 < v13 )
      {
LABEL_17:
        *a1 = *(v4 - 1);
        v14 = v10;
        *(v4 - 1) = v10;
        v10 = a1[1];
        goto LABEL_8;
      }
LABEL_21:
      *a1 = *v9;
      *v9 = v10;
      v14 = *(v4 - 1);
      v10 = a1[1];
      goto LABEL_8;
    }
    if ( v12 < v13 )
      goto LABEL_21;
    if ( v11 < v13 )
      goto LABEL_17;
LABEL_7:
    *a1 = v8;
    a1[1] = v10;
    v14 = *(v4 - 1);
LABEL_8:
    v15 = *a1;
    v16 = v7;
    v17 = v4;
    v18 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    while ( 1 )
    {
      v19 = v16 - 1;
      v20 = v18 | (v15 >> 1) & 3;
      if ( (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10 >> 1) & 3) < v20 )
        goto LABEL_14;
      --v17;
      if ( (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v14 >> 1) & 3) > v20 )
      {
        do
          v21 = *--v17;
        while ( v20 < (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) );
      }
      if ( v17 <= v19 )
        break;
      *(v16 - 1) = *v17;
      v14 = *(v17 - 1);
      *v17 = v10;
      v15 = *a1;
      v18 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_14:
      v10 = *v16++;
    }
    sub_1F22B90(v16 - 1, v4, v5);
    result = (char *)v19 - (char *)a1;
    if ( (char *)v19 - (char *)a1 > 128 )
    {
      if ( v5 )
      {
        v4 = v16 - 1;
        continue;
      }
LABEL_23:
      v22 = result >> 3;
      v23 = ((result >> 3) - 2) >> 1;
      sub_1F21560((__int64)a1, v23, result >> 3, a1[v23]);
      do
      {
        --v23;
        sub_1F21560((__int64)a1, v23, v22, a1[v23]);
      }
      while ( v23 );
      do
      {
        v24 = *--v19;
        *v19 = *a1;
        result = sub_1F21560((__int64)a1, 0, v19 - a1, v24);
      }
      while ( (char *)v19 - (char *)a1 > 8 );
    }
    return result;
  }
}
