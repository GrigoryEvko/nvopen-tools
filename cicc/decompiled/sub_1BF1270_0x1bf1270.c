// Function: sub_1BF1270
// Address: 0x1bf1270
//
__int64 __fastcall sub_1BF1270(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v9; // r12
  _QWORD *v10; // r15
  _QWORD *v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // rdx
  _QWORD *v18; // rdx

  v4 = *(_QWORD **)(a3 + 16);
  v5 = *(_QWORD **)(a3 + 8);
  if ( v4 == v5 )
  {
    v6 = &v5[*(unsigned int *)(a3 + 28)];
    if ( v5 == v6 )
    {
      v18 = *(_QWORD **)(a3 + 8);
    }
    else
    {
      do
      {
        if ( a2 == *v5 )
          break;
        ++v5;
      }
      while ( v6 != v5 );
      v18 = v6;
    }
    goto LABEL_35;
  }
  v6 = &v4[*(unsigned int *)(a3 + 24)];
  v5 = sub_16CC9F0(a3, a2);
  if ( a2 == *v5 )
  {
    v15 = *(_QWORD *)(a3 + 16);
    if ( v15 == *(_QWORD *)(a3 + 8) )
      v16 = *(unsigned int *)(a3 + 28);
    else
      v16 = *(unsigned int *)(a3 + 24);
    v18 = (_QWORD *)(v15 + 8 * v16);
    goto LABEL_35;
  }
  v7 = *(_QWORD *)(a3 + 16);
  if ( v7 == *(_QWORD *)(a3 + 8) )
  {
    v5 = (_QWORD *)(v7 + 8LL * *(unsigned int *)(a3 + 28));
    v18 = v5;
LABEL_35:
    while ( v18 != v5 && *v5 >= 0xFFFFFFFFFFFFFFFELL )
      ++v5;
    goto LABEL_5;
  }
  v5 = (_QWORD *)(v7 + 8LL * *(unsigned int *)(a3 + 24));
LABEL_5:
  if ( v5 != v6 )
    return 0;
  v9 = *(_QWORD *)(a2 + 8);
  if ( !v9 )
    return 0;
  v10 = *(_QWORD **)(a1 + 72);
  while ( 1 )
  {
    v13 = sub_1648700(v9)[5];
    v12 = *(_QWORD **)(a1 + 64);
    if ( v10 != v12 )
      break;
    v14 = *(unsigned int *)(a1 + 84);
    v11 = &v10[v14];
    if ( v10 == v11 )
    {
      v17 = v10;
    }
    else
    {
      do
      {
        if ( v13 == *v12 )
          break;
        ++v12;
      }
      while ( v11 != v12 );
      v17 = &v10[v14];
    }
LABEL_22:
    while ( v17 != v12 )
    {
      if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_12;
      ++v12;
    }
    if ( v11 == v12 )
      return 1;
LABEL_13:
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      return 0;
  }
  v11 = &v10[*(unsigned int *)(a1 + 80)];
  v12 = sub_16CC9F0(a1 + 56, v13);
  if ( v13 == *v12 )
  {
    v10 = *(_QWORD **)(a1 + 72);
    if ( v10 == *(_QWORD **)(a1 + 64) )
      v17 = &v10[*(unsigned int *)(a1 + 84)];
    else
      v17 = &v10[*(unsigned int *)(a1 + 80)];
    goto LABEL_22;
  }
  v10 = *(_QWORD **)(a1 + 72);
  if ( v10 == *(_QWORD **)(a1 + 64) )
  {
    v12 = &v10[*(unsigned int *)(a1 + 84)];
    v17 = v12;
    goto LABEL_22;
  }
  v12 = &v10[*(unsigned int *)(a1 + 80)];
LABEL_12:
  if ( v11 != v12 )
    goto LABEL_13;
  return 1;
}
