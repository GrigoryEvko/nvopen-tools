// Function: sub_3510940
// Address: 0x3510940
//
void __fastcall sub_3510940(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v3; // rsi
  _QWORD *v4; // r13
  __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 v7; // rax
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  _QWORD *v11; // r13
  _QWORD *v12; // r15
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 *v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rsi

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
        v9 += 64LL;
        if ( v8 < v9 )
          break;
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 - 64);
          if ( v10 == v9 - 48 )
            break;
          _libc_free(v10);
          v9 += 64LL;
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
  v11 = *(_QWORD **)(a1 + 64);
  v12 = &v11[2 * *(unsigned int *)(a1 + 72)];
  if ( v12 != v11 )
  {
    do
    {
      v13 = *v11 + v11[1];
      v14 = (*v11 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v14 += 64LL;
        if ( v13 < v14 )
          break;
        while ( 1 )
        {
          v15 = *(_QWORD *)(v14 - 64);
          if ( v15 == v14 - 48 )
            break;
          _libc_free(v15);
          v14 += 64LL;
          if ( v13 < v14 )
            goto LABEL_17;
        }
      }
LABEL_17:
      v11 += 2;
    }
    while ( v12 != v11 );
    v16 = *(__int64 **)(a1 + 64);
    v17 = &v16[2 * *(unsigned int *)(a1 + 72)];
    while ( v17 != v16 )
    {
      v18 = v16[1];
      v19 = *v16;
      v16 += 2;
      sub_C7D6A0(v19, v18, 16);
    }
  }
  v20 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v20 )
  {
    v21 = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v22 = *v21;
    v23 = &v21[v20];
    v24 = v21 + 1;
    *(_QWORD *)a1 = *v21;
    *(_QWORD *)(a1 + 8) = v22 + 4096;
    if ( v23 != v21 + 1 )
    {
      while ( 1 )
      {
        v25 = *v24;
        v26 = (unsigned int)(v24 - v21) >> 7;
        v27 = 4096LL << v26;
        if ( v26 >= 0x1E )
          v27 = 0x40000000000LL;
        ++v24;
        sub_C7D6A0(v25, v27, 16);
        if ( v23 == v24 )
          break;
        v21 = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
