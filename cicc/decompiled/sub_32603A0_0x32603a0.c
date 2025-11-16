// Function: sub_32603A0
// Address: 0x32603a0
//
__int64 *__fastcall sub_32603A0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v9; // rdi
  __int64 *v11; // rax
  __int64 *v12; // rdx
  __int64 *v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 i; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  int j; // ecx
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 *result; // rax
  char v25; // dl
  __int64 v26; // r12
  __int64 v27; // rax

  v7 = (unsigned int)a4;
  v9 = *(_QWORD *)a1;
  if ( *(_BYTE *)(v9 + 28) )
  {
    v11 = *(__int64 **)(v9 + 8);
    v12 = &v11[*(unsigned int *)(v9 + 20)];
    if ( v11 == v12 )
      goto LABEL_16;
    while ( a3 != *v11 )
    {
      if ( v12 == ++v11 )
        goto LABEL_16;
    }
  }
  else if ( !sub_C8CA60(v9, a3) )
  {
    goto LABEL_16;
  }
  **(_BYTE **)(a1 + 8) = 1;
  **(_BYTE **)(a1 + 16) = 1;
  v13 = *(__int64 **)(a1 + 24);
  v14 = *((_DWORD *)v13 + 2);
  if ( v14 )
  {
    v15 = *v13;
    v16 = v14;
    for ( i = 0; i != v16; ++i )
    {
      v14 = i;
      if ( *(_QWORD *)(v15 + 16 * i) == a3 )
      {
        v18 = 4 * i;
        goto LABEL_11;
      }
      v14 = i + 1;
    }
    v18 = 4 * v16;
  }
  else
  {
    v18 = 0;
  }
LABEL_11:
  v19 = *(_QWORD *)(a1 + 32);
  v20 = (unsigned int)(a2 + 1);
  for ( j = v20; *(_DWORD *)(v19 + 8) > (unsigned int)v20; j = v20 )
  {
    v22 = *(_QWORD *)v19 + 16 * v20;
    if ( *(_DWORD *)(v22 + 8) == v14 )
    {
      *(_DWORD *)(v22 + 8) = v7;
      v19 = *(_QWORD *)(a1 + 32);
    }
    v20 = (unsigned int)(j + 1);
  }
  a4 = **(_QWORD **)(a1 + 40);
  *(_DWORD *)(a4 + 4LL * (unsigned int)v7) += *(_DWORD *)(a4 + v18);
  v12 = **(__int64 ***)(a1 + 40);
  *(_DWORD *)((char *)v12 + v18) = 0;
  --**(_DWORD **)(a1 + 48);
LABEL_16:
  v23 = *(_QWORD *)(a1 + 56);
  if ( !*(_BYTE *)(v23 + 28) )
    goto LABEL_23;
  result = *(__int64 **)(v23 + 8);
  a4 = *(unsigned int *)(v23 + 20);
  v12 = &result[a4];
  if ( result == v12 )
  {
LABEL_22:
    if ( (unsigned int)a4 < *(_DWORD *)(v23 + 16) )
    {
      *(_DWORD *)(v23 + 20) = a4 + 1;
      *v12 = a3;
      ++*(_QWORD *)v23;
LABEL_24:
      ++*(_DWORD *)(**(_QWORD **)(a1 + 40) + 4 * v7);
      v26 = *(_QWORD *)(a1 + 32);
      v27 = *(unsigned int *)(v26 + 8);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v26 + 12) )
      {
        sub_C8D5F0(v26, (const void *)(v26 + 16), v27 + 1, 0x10u, a5, a6);
        v27 = *(unsigned int *)(v26 + 8);
      }
      result = (__int64 *)(*(_QWORD *)v26 + 16 * v27);
      *result = a3;
      result[1] = v7;
      ++*(_DWORD *)(v26 + 8);
      return result;
    }
LABEL_23:
    result = sub_C8CC70(v23, a3, (__int64)v12, a4, a5, a6);
    if ( !v25 )
      return result;
    goto LABEL_24;
  }
  while ( a3 != *result )
  {
    if ( v12 == ++result )
      goto LABEL_22;
  }
  return result;
}
