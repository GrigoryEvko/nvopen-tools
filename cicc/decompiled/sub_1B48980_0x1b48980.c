// Function: sub_1B48980
// Address: 0x1b48980
//
__int64 *__fastcall sub_1B48980(__int64 a1, __int64 a2)
{
  __int64 *v4; // rsi
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 *i; // rbx
  __int64 *result; // rax
  __int64 v10; // rcx
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v4 = *(__int64 **)(a1 + 16);
  v5 = *(__int64 **)(a1 + 8);
  if ( v4 == v5 )
    v6 = *(unsigned int *)(a1 + 28);
  else
    v6 = *(unsigned int *)(a1 + 24);
  v7 = &v4[v6];
  for ( i = v4; v7 != i; ++i )
  {
    if ( (unsigned __int64)*i < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  while ( v4 == v5 )
  {
    result = &v4[*(unsigned int *)(a1 + 28)];
    if ( i == result )
      return result;
LABEL_8:
    v10 = *i;
    for ( ++i; v7 != i; ++i )
    {
      if ( (unsigned __int64)*i < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    v11 = *(_QWORD **)(a2 + 16);
    v12 = *(_QWORD **)(a2 + 8);
    if ( v11 == v12 )
    {
      v13 = &v12[*(unsigned int *)(a2 + 28)];
      if ( v12 == v13 )
      {
        v15 = *(_QWORD **)(a2 + 8);
      }
      else
      {
        do
        {
          if ( v10 == *v12 )
            break;
          ++v12;
        }
        while ( v13 != v12 );
        v15 = v13;
      }
    }
    else
    {
      v22 = v10;
      v13 = &v11[*(unsigned int *)(a2 + 24)];
      v12 = sub_16CC9F0(a2, v10);
      v10 = v22;
      if ( v22 == *v12 )
      {
        v17 = *(_QWORD *)(a2 + 16);
        if ( v17 == *(_QWORD *)(a2 + 8) )
        {
          v4 = *(__int64 **)(a1 + 16);
          v5 = *(__int64 **)(a1 + 8);
          v15 = (_QWORD *)(v17 + 8LL * *(unsigned int *)(a2 + 28));
          goto LABEL_18;
        }
        v15 = (_QWORD *)(v17 + 8LL * *(unsigned int *)(a2 + 24));
      }
      else
      {
        v14 = *(_QWORD *)(a2 + 16);
        if ( v14 == *(_QWORD *)(a2 + 8) )
        {
          v4 = *(__int64 **)(a1 + 16);
          v5 = *(__int64 **)(a1 + 8);
          v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a2 + 28));
          v15 = v12;
          goto LABEL_18;
        }
        v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a2 + 24));
        v15 = v12;
      }
      v4 = *(__int64 **)(a1 + 16);
      v5 = *(__int64 **)(a1 + 8);
    }
LABEL_18:
    while ( v15 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
      ++v12;
    if ( v12 != v13 )
      continue;
    if ( v5 == v4 )
    {
      v20 = *(unsigned int *)(a1 + 28);
      v21 = &v5[v20];
      if ( v5 == &v5[v20] )
      {
LABEL_45:
        v5 = &v4[v20];
        v19 = &v4[v20];
      }
      else
      {
        while ( v10 != *v5 )
        {
          if ( v21 == ++v5 )
            goto LABEL_45;
        }
        v19 = &v4[v20];
      }
    }
    else
    {
      v23 = v10;
      v16 = sub_16CC9F0(a1, v10);
      v4 = *(__int64 **)(a1 + 16);
      v5 = v16;
      if ( v23 == *v16 )
      {
        if ( v4 == *(__int64 **)(a1 + 8) )
          v18 = *(unsigned int *)(a1 + 28);
        else
          v18 = *(unsigned int *)(a1 + 24);
        v19 = &v4[v18];
      }
      else
      {
        if ( v4 != *(__int64 **)(a1 + 8) )
          goto LABEL_23;
        v5 = &v4[*(unsigned int *)(a1 + 28)];
        v19 = v5;
      }
    }
    if ( v19 != v5 )
    {
      *v5 = -2;
      v4 = *(__int64 **)(a1 + 16);
      ++*(_DWORD *)(a1 + 32);
      v5 = *(__int64 **)(a1 + 8);
      continue;
    }
    v4 = *(__int64 **)(a1 + 16);
LABEL_23:
    v5 = *(__int64 **)(a1 + 8);
  }
  result = &v4[*(unsigned int *)(a1 + 24)];
  if ( i != result )
    goto LABEL_8;
  return result;
}
