// Function: sub_25FD180
// Address: 0x25fd180
//
__int64 *__fastcall sub_25FD180(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 v7; // rbx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r14
  _QWORD *v13; // r14
  __int64 v14; // r15
  __int64 *result; // rax
  unsigned __int64 v16; // r12
  _QWORD *i; // r15
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rbx
  __int64 *v20; // rbx
  __int64 *j; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 *v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rdi
  unsigned int v29; // ecx
  __int64 v30; // rsi
  __int64 *v31; // [rsp+8h] [rbp-38h]
  _QWORD *v32; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v31 = &v2[v3];
  if ( v2 != v31 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = *v2;
      v6 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v7 = 4096LL << v6;
      if ( v6 >= 0x1E )
        v7 = 0x40000000000LL;
      v8 = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v9 = v5 + v7;
      if ( v5 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v9 = *(_QWORD *)a1;
      v10 = v8 + 304;
      while ( v9 >= v10 )
      {
        v11 = *(_QWORD *)(v10 - 104);
        v12 = v10 - 304;
        if ( v11 != v10 - 88 )
          _libc_free(v11);
        v10 += 304LL;
        sub_C7D6A0(*(_QWORD *)(v12 + 176), 16LL * *(unsigned int *)(v12 + 192), 8);
        sub_C7D6A0(*(_QWORD *)(v12 + 144), 16LL * *(unsigned int *)(v12 + 160), 8);
        sub_C7D6A0(*(_QWORD *)(v12 + 104), 16LL * *(unsigned int *)(v12 + 120), 8);
        sub_C7D6A0(*(_QWORD *)(v12 + 72), 8LL * *(unsigned int *)(v12 + 88), 4);
        sub_C7D6A0(*(_QWORD *)(v12 + 40), 8LL * *(unsigned int *)(v12 + 56), 4);
      }
      if ( v31 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v13 = *(_QWORD **)(a1 + 64);
  v14 = 2LL * *(unsigned int *)(a1 + 72);
  result = &v13[v14];
  v32 = &v13[v14];
  if ( &v13[v14] != v13 )
  {
    do
    {
      v16 = *v13 + v13[1];
      result = (__int64 *)((*v13 + 7LL) & 0xFFFFFFFFFFFFFFF8LL);
      for ( i = result + 38;
            v16 >= (unsigned __int64)i;
            result = (__int64 *)sub_C7D6A0(v19[5], 8LL * *((unsigned int *)v19 + 14), 4) )
      {
        v18 = *(i - 13);
        v19 = i - 38;
        if ( (_QWORD *)v18 != i - 11 )
          _libc_free(v18);
        i += 38;
        sub_C7D6A0(v19[22], 16LL * *((unsigned int *)v19 + 48), 8);
        sub_C7D6A0(v19[18], 16LL * *((unsigned int *)v19 + 40), 8);
        sub_C7D6A0(v19[13], 16LL * *((unsigned int *)v19 + 30), 8);
        sub_C7D6A0(v19[9], 8LL * *((unsigned int *)v19 + 22), 4);
      }
      v13 += 2;
    }
    while ( v32 != v13 );
    v20 = *(__int64 **)(a1 + 64);
    for ( j = &v20[2 * *(unsigned int *)(a1 + 72)]; j != v20; result = (__int64 *)sub_C7D6A0(v23, v22, 16) )
    {
      v22 = v20[1];
      v23 = *v20;
      v20 += 2;
    }
  }
  v24 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v24 )
  {
    result = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v25 = *result;
    v26 = &result[v24];
    v27 = result + 1;
    *(_QWORD *)a1 = *result;
    *(_QWORD *)(a1 + 8) = v25 + 4096;
    if ( v26 != result + 1 )
    {
      while ( 1 )
      {
        v28 = *v27;
        v29 = (unsigned int)(v27 - result) >> 7;
        v30 = 4096LL << v29;
        if ( v29 >= 0x1E )
          v30 = 0x40000000000LL;
        ++v27;
        result = (__int64 *)sub_C7D6A0(v28, v30, 16);
        if ( v26 == v27 )
          break;
        result = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
  return result;
}
