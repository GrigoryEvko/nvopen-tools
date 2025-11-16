// Function: sub_B83600
// Address: 0xb83600
//
__int64 *__fastcall sub_B83600(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // rbx
  bool v7; // cf
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 i; // r15
  __int64 v12; // rdi
  _QWORD *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  _QWORD *v17; // r14
  __int64 v18; // r15
  __int64 *result; // rax
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rbx
  __int64 v22; // rdi
  _QWORD *v23; // r15
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 *v27; // rbx
  __int64 *j; // r12
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 *v33; // rbx
  __int64 *v34; // r14
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 v37; // rsi
  __int64 *v38; // [rsp+8h] [rbp-38h]
  _QWORD *v39; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v38 = &v2[v3];
  if ( v2 != v38 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      v7 = v5 < 0x1E;
      v8 = *v2;
      if ( !v7 )
        v6 = 0x40000000000LL;
      v9 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v8 + v6;
      if ( v8 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v10 = *(_QWORD *)a1;
      for ( i = v9 + 176; v10 >= i; i += 176LL )
      {
        v12 = *(_QWORD *)(i - 24);
        v13 = (_QWORD *)(i - 176);
        if ( v12 != i - 8 )
          _libc_free(v12, v3);
        v14 = v13[15];
        if ( v14 != i - 40 )
          _libc_free(v14, v3);
        v15 = v13[11];
        if ( v15 != i - 72 )
          _libc_free(v15, v3);
        v16 = v13[1];
        if ( v16 != i - 152 )
          _libc_free(v16, v3);
      }
      if ( v38 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v17 = *(_QWORD **)(a1 + 64);
  v18 = 2LL * *(unsigned int *)(a1 + 72);
  result = &v17[v18];
  v39 = &v17[v18];
  if ( &v17[v18] != v17 )
  {
    do
    {
      v20 = *v17 + v17[1];
      v21 = (*v17 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v21 += 176LL;
        if ( v20 < v21 )
          break;
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 - 24);
          v23 = (_QWORD *)(v21 - 176);
          if ( v22 != v21 - 8 )
            _libc_free(v22, v3);
          v24 = v23[15];
          if ( v24 != v21 - 40 )
            _libc_free(v24, v3);
          v25 = v23[11];
          if ( v25 != v21 - 72 )
            _libc_free(v25, v3);
          v26 = v23[1];
          result = (__int64 *)(v21 - 152);
          if ( v26 == v21 - 152 )
            break;
          result = (__int64 *)_libc_free(v26, v3);
          v21 += 176LL;
          if ( v20 < v21 )
            goto LABEL_30;
        }
      }
LABEL_30:
      v17 += 2;
    }
    while ( v39 != v17 );
    v27 = *(__int64 **)(a1 + 64);
    for ( j = &v27[2 * *(unsigned int *)(a1 + 72)]; j != v27; result = (__int64 *)sub_C7D6A0(v30, v29, 16) )
    {
      v29 = v27[1];
      v30 = *v27;
      v27 += 2;
    }
  }
  v31 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v31 )
  {
    result = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v32 = *result;
    v33 = &result[v31];
    v34 = result + 1;
    *(_QWORD *)a1 = *result;
    *(_QWORD *)(a1 + 8) = v32 + 4096;
    if ( v33 != result + 1 )
    {
      while ( 1 )
      {
        v35 = *v34;
        v36 = (unsigned int)(v34 - result) >> 7;
        v37 = 4096LL << v36;
        if ( v36 >= 0x1E )
          v37 = 0x40000000000LL;
        ++v34;
        result = (__int64 *)sub_C7D6A0(v35, v37, 16);
        if ( v33 == v34 )
          break;
        result = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
  return result;
}
