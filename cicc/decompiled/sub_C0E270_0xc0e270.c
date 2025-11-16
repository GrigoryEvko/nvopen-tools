// Function: sub_C0E270
// Address: 0xc0e270
//
void __fastcall sub_C0E270(__int64 a1)
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
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 *v25; // rbx
  __int64 *v26; // r13
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 v29; // rsi

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
    v18 = *(__int64 **)(a1 + 64);
    v19 = &v18[2 * *(unsigned int *)(a1 + 72)];
    while ( v19 != v18 )
    {
      v20 = v18[1];
      v21 = *v18;
      v18 += 2;
      sub_C7D6A0(v21, v20, 16);
    }
  }
  v22 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v22 )
  {
    v23 = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v24 = *v23;
    v25 = &v23[v22];
    v26 = v23 + 1;
    *(_QWORD *)a1 = *v23;
    *(_QWORD *)(a1 + 8) = v24 + 4096;
    if ( v25 != v23 + 1 )
    {
      while ( 1 )
      {
        v27 = *v26;
        v28 = (unsigned int)(v26 - v23) >> 7;
        v29 = 4096LL << v28;
        if ( v28 >= 0x1E )
          v29 = 0x40000000000LL;
        ++v26;
        sub_C7D6A0(v27, v29, 16);
        if ( v25 == v26 )
          break;
        v23 = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
