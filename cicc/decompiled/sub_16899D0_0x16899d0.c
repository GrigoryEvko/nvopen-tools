// Function: sub_16899D0
// Address: 0x16899d0
//
void __fastcall sub_16899D0(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rsi
  _QWORD *v4; // r13
  __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 v7; // rax
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // r13
  _QWORD *v13; // r15
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 *v24; // r12
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rdi

  v2 = *(_QWORD **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v6 = (unsigned int)(((__int64)v2 - v5) >> 3) >> 7;
      v7 = 4096LL << v6;
      if ( v6 >= 0x1E )
        v7 = 0x40000000000LL;
      v8 = *v2 + v7;
      v9 = (*v2 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v2 == *(_QWORD *)(v5 + 8 * v3 - 8) )
        v8 = *(_QWORD *)a1;
      while ( 1 )
      {
        v9 += 40LL;
        if ( v8 < v9 )
          break;
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 - 40);
          if ( v10 == v9 - 24 )
            break;
          v11 = *(_QWORD *)(v9 - 24);
          v9 += 40LL;
          j_j___libc_free_0(v10, v11 + 1);
          if ( v8 < v9 )
            goto LABEL_10;
        }
      }
LABEL_10:
      if ( v4 == ++v2 )
        break;
      v5 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v12 = *(_QWORD **)(a1 + 64);
  v13 = &v12[2 * *(unsigned int *)(a1 + 72)];
  if ( v13 != v12 )
  {
    do
    {
      v14 = *v12 + v12[1];
      v15 = (*v12 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v15 += 40LL;
        if ( v14 < v15 )
          break;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v15 - 40);
          if ( v16 == v15 - 24 )
            break;
          v17 = *(_QWORD *)(v15 - 24);
          v15 += 40LL;
          j_j___libc_free_0(v16, v17 + 1);
          if ( v14 < v15 )
            goto LABEL_17;
        }
      }
LABEL_17:
      v12 += 2;
    }
    while ( v13 != v12 );
    v18 = *(unsigned __int64 **)(a1 + 64);
    v19 = &v18[2 * *(unsigned int *)(a1 + 72)];
    while ( v18 != v19 )
    {
      v20 = *v18;
      v18 += 2;
      _libc_free(v20);
    }
  }
  v21 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v23 = *v22;
    v24 = &v22[v21];
    v25 = v22 + 1;
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 8) = v23 + 4096;
    while ( v24 != v25 )
    {
      v26 = *v25++;
      _libc_free(v26);
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
