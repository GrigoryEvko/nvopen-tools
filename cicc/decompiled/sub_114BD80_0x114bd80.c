// Function: sub_114BD80
// Address: 0x114bd80
//
__int64 __fastcall sub_114BD80(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  _QWORD *v5; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // esi
  __int64 v13; // r9
  _QWORD *v14; // r11
  int v15; // r13d
  unsigned int v16; // edx
  _QWORD *v17; // r8
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rbx
  int v21; // eax
  int v22; // ecx
  unsigned int v23; // edx
  __int64 v24; // rdi
  int v25; // r10d
  int v26; // eax
  int v27; // ecx
  int v28; // r10d
  unsigned int v29; // edx
  __int64 v30; // rdi

  result = *(unsigned int *)(a1 + 16);
  if ( !(_DWORD)result )
  {
    v5 = *(_QWORD **)(a1 + 32);
    v7 = (__int64)&v5[*(unsigned int *)(a1 + 40)];
    result = (__int64)sub_1149E10(v5, v7, a2);
    if ( v7 == result )
      return sub_114A990(a1, *a2, v8, v9, v10, v11);
    return result;
  }
  v12 = *(_DWORD *)(a1 + 24);
  if ( !v12 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v14 = 0;
  v15 = 1;
  v16 = (v12 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v17 = (_QWORD *)(v13 + 8LL * v16);
  v18 = *v17;
  if ( *a2 == *v17 )
    return result;
  while ( v18 != -4096 )
  {
    if ( v18 != -8192 || v14 )
      v17 = v14;
    v16 = (v12 - 1) & (v15 + v16);
    v18 = *(_QWORD *)(v13 + 8LL * v16);
    if ( *a2 == v18 )
      return result;
    ++v15;
    v14 = v17;
    v17 = (_QWORD *)(v13 + 8LL * v16);
  }
  if ( !v14 )
    v14 = v17;
  v19 = result + 1;
  ++*(_QWORD *)a1;
  if ( 4 * v19 >= 3 * v12 )
  {
LABEL_18:
    sub_CF4090(a1, 2 * v12);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v17 = *(_QWORD **)(a1 + 8);
      v23 = (v21 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v14 = &v17[v23];
      v24 = *v14;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v14 )
        goto LABEL_12;
      v25 = 1;
      v13 = 0;
      while ( v24 != -4096 )
      {
        if ( v24 == -8192 && !v13 )
          v13 = (__int64)v14;
        v23 = v22 & (v25 + v23);
        v14 = &v17[v23];
        v24 = *v14;
        if ( *a2 == *v14 )
          goto LABEL_12;
        ++v25;
      }
LABEL_22:
      if ( v13 )
        v14 = (_QWORD *)v13;
      goto LABEL_12;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v12 - *(_DWORD *)(a1 + 20) - v19 <= v12 >> 3 )
  {
    sub_CF4090(a1, v12);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v17 = *(_QWORD **)(a1 + 8);
      v13 = 0;
      v28 = 1;
      v29 = (v26 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v14 = &v17[v29];
      v30 = *v14;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v14 == *a2 )
        goto LABEL_12;
      while ( v30 != -4096 )
      {
        if ( !v13 && v30 == -8192 )
          v13 = (__int64)v14;
        v29 = v27 & (v28 + v29);
        v14 = &v17[v29];
        v30 = *v14;
        if ( *a2 == *v14 )
          goto LABEL_12;
        ++v28;
      }
      goto LABEL_22;
    }
    goto LABEL_43;
  }
LABEL_12:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v20 = *a2;
  *v14 = v20;
  result = *(unsigned int *)(a1 + 40);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, (__int64)v17, v13);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v20;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
