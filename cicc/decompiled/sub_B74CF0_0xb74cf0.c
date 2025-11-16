// Function: sub_B74CF0
// Address: 0xb74cf0
//
__int64 *__fastcall sub_B74CF0(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rax
  unsigned int v5; // ecx
  __int64 v6; // rdx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdi
  _QWORD *v12; // r13
  __int64 v13; // r15
  __int64 *result; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 *v20; // rbx
  __int64 *i; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 v28; // rdi
  unsigned int v29; // ecx
  __int64 v30; // rsi
  _QWORD *v31; // [rsp+8h] [rbp-38h]
  _QWORD *v32; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v31 = &v2[v3];
  if ( v2 != v31 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      if ( v5 >= 0x1E )
        v6 = 0x40000000000LL;
      v7 = *v2 + v6;
      v8 = (*v2 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v2 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v7 = *(_QWORD *)a1;
      while ( 1 )
      {
        v8 += 48LL;
        if ( v7 < v8 )
          break;
        while ( 1 )
        {
          v9 = v8 - 48;
          if ( *(_DWORD *)(v8 - 8) > 0x40u )
          {
            v10 = *(_QWORD *)(v9 + 32);
            if ( v10 )
              j_j___libc_free_0_0(v10);
          }
          if ( *(_DWORD *)(v9 + 24) <= 0x40u )
            break;
          v11 = *(_QWORD *)(v9 + 16);
          if ( !v11 )
            break;
          j_j___libc_free_0_0(v11);
          v8 += 48LL;
          if ( v7 < v8 )
            goto LABEL_14;
        }
      }
LABEL_14:
      if ( v31 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v12 = *(_QWORD **)(a1 + 64);
  v13 = 2LL * *(unsigned int *)(a1 + 72);
  result = &v12[v13];
  v32 = &v12[v13];
  if ( &v12[v13] != v12 )
  {
    do
    {
      v15 = *v12 + v12[1];
      v16 = (*v12 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v16 += 48LL;
        if ( v15 < v16 )
          break;
        while ( 1 )
        {
          v17 = v16 - 48;
          if ( *(_DWORD *)(v16 - 8) > 0x40u )
          {
            v18 = *(_QWORD *)(v17 + 32);
            if ( v18 )
              result = (__int64 *)j_j___libc_free_0_0(v18);
          }
          if ( *(_DWORD *)(v17 + 24) <= 0x40u )
            break;
          v19 = *(_QWORD *)(v17 + 16);
          if ( !v19 )
            break;
          result = (__int64 *)j_j___libc_free_0_0(v19);
          v16 += 48LL;
          if ( v15 < v16 )
            goto LABEL_25;
        }
      }
LABEL_25:
      v12 += 2;
    }
    while ( v32 != v12 );
    v20 = *(__int64 **)(a1 + 64);
    for ( i = &v20[2 * *(unsigned int *)(a1 + 72)]; i != v20; result = (__int64 *)sub_C7D6A0(v23, v22, 16) )
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
