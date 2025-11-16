// Function: sub_25FCE60
// Address: 0x25fce60
//
__int64 *__fastcall sub_25FCE60(__int64 a1)
{
  _QWORD *v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rax
  unsigned int v5; // ecx
  __int64 v6; // r12
  unsigned __int64 v7; // r12
  unsigned __int64 i; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // r15
  __int64 *result; // rax
  unsigned __int64 v17; // r12
  unsigned __int64 j; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 *v24; // rbx
  __int64 *k; // r12
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // rbx
  __int64 *v31; // r13
  __int64 v32; // rdi
  unsigned int v33; // ecx
  __int64 v34; // rsi
  _QWORD *v35; // [rsp+8h] [rbp-38h]
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v35 = &v2[v3];
  if ( v2 != v35 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      if ( v5 >= 0x1E )
        v6 = 0x40000000000LL;
      v7 = *v2 + v6;
      if ( *v2 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v7 = *(_QWORD *)a1;
      for ( i = ((*v2 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 256; v7 >= i; i += 256LL )
      {
        v9 = *(_QWORD *)(i - 16);
        v10 = i - 256;
        if ( v9 != i )
          _libc_free(v9);
        sub_C7D6A0(*(_QWORD *)(v10 + 216), 8LL * *(unsigned int *)(v10 + 232), 8);
        v11 = *(_QWORD *)(v10 + 168);
        if ( v11 != i - 72 )
          j_j___libc_free_0(v11);
        v12 = *(_QWORD *)(v10 + 104);
        if ( v12 != i - 136 )
          _libc_free(v12);
        v13 = *(_QWORD *)(v10 + 88);
        if ( v13 != i - 152 )
          _libc_free(v13);
        sub_C7D6A0(*(_QWORD *)(v10 + 64), 8LL * *(unsigned int *)(v10 + 80), 8);
      }
      if ( v35 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v14 = *(_QWORD **)(a1 + 64);
  v15 = 2LL * *(unsigned int *)(a1 + 72);
  result = &v14[v15];
  v36 = &v14[v15];
  if ( &v14[v15] != v14 )
  {
    do
    {
      v17 = *v14 + v14[1];
      for ( j = ((*v14 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 256;
            v17 >= j;
            result = (__int64 *)sub_C7D6A0(*(_QWORD *)(v20 + 64), 8LL * *(unsigned int *)(v20 + 80), 8) )
      {
        v19 = *(_QWORD *)(j - 16);
        v20 = j - 256;
        if ( v19 != j )
          _libc_free(v19);
        sub_C7D6A0(*(_QWORD *)(v20 + 216), 8LL * *(unsigned int *)(v20 + 232), 8);
        v21 = *(_QWORD *)(v20 + 168);
        if ( v21 != j - 72 )
          j_j___libc_free_0(v21);
        v22 = *(_QWORD *)(v20 + 104);
        if ( v22 != j - 136 )
          _libc_free(v22);
        v23 = *(_QWORD *)(v20 + 88);
        if ( v23 != j - 152 )
          _libc_free(v23);
        j += 256LL;
      }
      v14 += 2;
    }
    while ( v36 != v14 );
    v24 = *(__int64 **)(a1 + 64);
    for ( k = &v24[2 * *(unsigned int *)(a1 + 72)]; k != v24; result = (__int64 *)sub_C7D6A0(v27, v26, 16) )
    {
      v26 = v24[1];
      v27 = *v24;
      v24 += 2;
    }
  }
  v28 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v28 )
  {
    result = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v29 = *result;
    v30 = &result[v28];
    v31 = result + 1;
    *(_QWORD *)a1 = *result;
    *(_QWORD *)(a1 + 8) = v29 + 4096;
    if ( v30 != result + 1 )
    {
      while ( 1 )
      {
        v32 = *v31;
        v33 = (unsigned int)(v31 - result) >> 7;
        v34 = 4096LL << v33;
        if ( v33 >= 0x1E )
          v34 = 0x40000000000LL;
        ++v31;
        result = (__int64 *)sub_C7D6A0(v32, v34, 16);
        if ( v30 == v31 )
          break;
        result = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
  return result;
}
