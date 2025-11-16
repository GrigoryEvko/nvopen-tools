// Function: sub_1F6EC10
// Address: 0x1f6ec10
//
__int64 *__fastcall sub_1F6EC10(__int64 *a1, int a2, __int64 a3, unsigned int a4, __int64 a5, int a6)
{
  __int64 v7; // r13
  __int64 v10; // r15
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 i; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  int j; // ecx
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 *result; // rax
  unsigned int v27; // r8d
  char v28; // dl
  __int64 *v29; // rsi
  __int64 *v30; // rcx
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rsi
  _QWORD *v35; // rdx
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v7 = a4;
  v10 = *a1;
  v11 = *(_QWORD **)(*a1 + 16);
  v12 = *(_QWORD **)(*a1 + 8);
  if ( v11 == v12 )
  {
    v13 = &v12[*(unsigned int *)(v10 + 28)];
    if ( v12 == v13 )
    {
      v35 = *(_QWORD **)(*a1 + 8);
    }
    else
    {
      do
      {
        if ( a3 == *v12 )
          break;
        ++v12;
      }
      while ( v13 != v12 );
      v35 = v13;
    }
  }
  else
  {
    v36 = &v11[*(unsigned int *)(v10 + 24)];
    v12 = sub_16CC9F0(*a1, a3);
    v13 = v36;
    if ( a3 == *v12 )
    {
      v33 = *(_QWORD *)(v10 + 16);
      if ( v33 == *(_QWORD *)(v10 + 8) )
        v34 = *(unsigned int *)(v10 + 28);
      else
        v34 = *(unsigned int *)(v10 + 24);
      v35 = (_QWORD *)(v33 + 8 * v34);
    }
    else
    {
      v14 = *(_QWORD *)(v10 + 16);
      if ( v14 != *(_QWORD *)(v10 + 8) )
      {
        v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(v10 + 24));
        goto LABEL_5;
      }
      v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(v10 + 28));
      v35 = v12;
    }
  }
  while ( v35 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
    ++v12;
LABEL_5:
  if ( v13 != v12 )
  {
    *(_BYTE *)a1[1] = 1;
    *(_BYTE *)a1[2] = 1;
    v15 = (__int64 *)a1[3];
    v16 = *((_DWORD *)v15 + 2);
    if ( v16 )
    {
      v17 = *v15;
      v18 = v16;
      for ( i = 0; i != v18; ++i )
      {
        v16 = i;
        if ( *(_QWORD *)(v17 + 16 * i) == a3 )
        {
          v20 = 4 * i;
          goto LABEL_11;
        }
        v16 = i + 1;
      }
      v20 = 4 * v18;
    }
    else
    {
      v20 = 0;
    }
LABEL_11:
    v21 = a1[4];
    v22 = (unsigned int)(a2 + 1);
    for ( j = v22; (unsigned int)v22 < *(_DWORD *)(v21 + 8); j = v22 )
    {
      v24 = *(_QWORD *)v21 + 16 * v22;
      if ( *(_DWORD *)(v24 + 8) == v16 )
      {
        *(_DWORD *)(v24 + 8) = v7;
        v21 = a1[4];
      }
      v22 = (unsigned int)(j + 1);
    }
    *(_DWORD *)(*(_QWORD *)a1[5] + 4LL * (unsigned int)v7) += *(_DWORD *)(*(_QWORD *)a1[5] + v20);
    *(_DWORD *)(*(_QWORD *)a1[5] + v20) = 0;
    --*(_DWORD *)a1[6];
  }
  v25 = a1[7];
  result = *(__int64 **)(v25 + 8);
  if ( *(__int64 **)(v25 + 16) != result )
    goto LABEL_17;
  v29 = &result[*(unsigned int *)(v25 + 28)];
  v27 = *(_DWORD *)(v25 + 28);
  if ( result == v29 )
  {
LABEL_44:
    if ( v27 < *(_DWORD *)(v25 + 24) )
    {
      *(_DWORD *)(v25 + 28) = ++v27;
      *v29 = a3;
      ++*(_QWORD *)v25;
LABEL_36:
      ++*(_DWORD *)(*(_QWORD *)a1[5] + 4 * v7);
      v31 = a1[4];
      v32 = *(unsigned int *)(v31 + 8);
      if ( (unsigned int)v32 >= *(_DWORD *)(v31 + 12) )
      {
        sub_16CD150(v31, (const void *)(v31 + 16), 0, 16, v27, a6);
        v32 = *(unsigned int *)(v31 + 8);
      }
      result = (__int64 *)(*(_QWORD *)v31 + 16 * v32);
      *result = a3;
      result[1] = v7;
      ++*(_DWORD *)(v31 + 8);
      return result;
    }
LABEL_17:
    result = sub_16CCBA0(v25, a3);
    if ( !v28 )
      return result;
    goto LABEL_36;
  }
  v30 = 0;
  while ( a3 != *result )
  {
    if ( *result == -2 )
      v30 = result;
    if ( v29 == ++result )
    {
      if ( !v30 )
        goto LABEL_44;
      *v30 = a3;
      --*(_DWORD *)(v25 + 32);
      ++*(_QWORD *)v25;
      goto LABEL_36;
    }
  }
  return result;
}
