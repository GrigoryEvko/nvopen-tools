// Function: sub_1DE3B40
// Address: 0x1de3b40
//
void __fastcall sub_1DE3B40(__int64 a1)
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
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rbx
  __int64 v21; // rdx
  unsigned __int64 *v22; // r12
  unsigned __int64 *v23; // rbx
  unsigned __int64 v24; // rdi

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
    v16 = *(unsigned __int64 **)(a1 + 64);
    v17 = &v16[2 * *(unsigned int *)(a1 + 72)];
    while ( v16 != v17 )
    {
      v18 = *v16;
      v16 += 2;
      _libc_free(v18);
    }
  }
  v19 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v21 = *v20;
    v22 = &v20[v19];
    v23 = v20 + 1;
    *(_QWORD *)a1 = v21;
    *(_QWORD *)(a1 + 8) = v21 + 4096;
    while ( v22 != v23 )
    {
      v24 = *v23++;
      _libc_free(v24);
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
