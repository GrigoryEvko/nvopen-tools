// Function: sub_16CCF40
// Address: 0x16ccf40
//
void __fastcall sub_16CCF40(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 *v5; // r8
  __int64 *v6; // rsi
  __int64 *v7; // rdi
  int v8; // edx
  int v9; // eax
  int v10; // edx
  unsigned int v11; // eax
  int v12; // eax
  size_t v13; // rdx
  int v14; // eax
  int v15; // eax
  int v16; // edx
  int v17; // eax
  unsigned int v18; // r9d
  __int64 v19; // rsi
  __int64 *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r10
  __int64 *v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r8
  int v27; // eax
  int v28; // edx
  int v29; // eax
  int v30; // edx
  int v31; // eax

  if ( a1 == a2 )
    return;
  v3 = *(__int64 **)(a1 + 16);
  v5 = *(__int64 **)(a1 + 8);
  v6 = *(__int64 **)(a2 + 16);
  v7 = *(__int64 **)(a2 + 8);
  if ( v5 == v3 )
  {
    v11 = *(_DWORD *)(a1 + 28);
    if ( v6 != v7 )
    {
      if ( 8LL * v11 )
        memmove(v7, v5, 8LL * v11);
      v27 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a2 + 24) = *(_DWORD *)(a1 + 24);
      v28 = *(_DWORD *)(a1 + 28);
      *(_DWORD *)(a1 + 24) = v27;
      v29 = *(_DWORD *)(a2 + 28);
      *(_DWORD *)(a2 + 28) = v28;
      v30 = *(_DWORD *)(a1 + 32);
      *(_DWORD *)(a1 + 28) = v29;
      v31 = *(_DWORD *)(a2 + 32);
      *(_DWORD *)(a2 + 32) = v30;
      *(_DWORD *)(a1 + 32) = v31;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 16) = *(_QWORD *)(a2 + 8);
      return;
    }
    v18 = *(_DWORD *)(a1 + 28);
    if ( *(_DWORD *)(a2 + 28) <= v11 )
      v18 = *(_DWORD *)(a2 + 28);
    v19 = v18;
    v20 = &v7[v19];
    if ( v19 * 8 )
    {
      do
      {
        v21 = *v5;
        v22 = *v7++;
        *v5++ = v22;
        *(v7 - 1) = v21;
      }
      while ( v20 != v7 );
      v5 = *(__int64 **)(a1 + 8);
      v11 = *(_DWORD *)(a1 + 28);
      v7 = *(__int64 **)(a2 + 8);
      v23 = &v5[v19];
    }
    else
    {
      v23 = v5;
    }
    v24 = &v7[v19];
    if ( v11 <= v18 )
    {
      v26 = *(unsigned int *)(a2 + 28);
      if ( v24 == &v7[v26] )
        goto LABEL_20;
      memmove(v23, v24, 8 * v26 - v19 * 8);
      v11 = *(_DWORD *)(a1 + 28);
    }
    else
    {
      v25 = v11;
      if ( v23 != &v5[v25] )
      {
        memmove(&v7[v19], v23, v25 * 8 - v19 * 8);
        v11 = *(_DWORD *)(a1 + 28);
        LODWORD(v26) = *(_DWORD *)(a2 + 28);
LABEL_20:
        *(_DWORD *)(a1 + 28) = v26;
        goto LABEL_5;
      }
    }
    LODWORD(v26) = *(_DWORD *)(a2 + 28);
    goto LABEL_20;
  }
  if ( v6 != v7 )
  {
    *(_QWORD *)(a1 + 16) = v6;
    v8 = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = v3;
    v9 = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v8;
    v10 = *(_DWORD *)(a2 + 28);
    *(_DWORD *)(a2 + 24) = v9;
    v11 = *(_DWORD *)(a1 + 28);
    *(_DWORD *)(a1 + 28) = v10;
LABEL_5:
    *(_DWORD *)(a2 + 28) = v11;
    v12 = *(_DWORD *)(a1 + 32);
    *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a2 + 32) = v12;
    return;
  }
  v13 = 8LL * *(unsigned int *)(a2 + 28);
  if ( v13 )
    memmove(v5, v6, v13);
  v14 = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 24) = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v14;
  v15 = *(_DWORD *)(a1 + 28);
  *(_DWORD *)(a1 + 28) = *(_DWORD *)(a2 + 28);
  v16 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a2 + 28) = v15;
  v17 = *(_DWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 32) = v16;
  *(_DWORD *)(a2 + 32) = v17;
  *(_QWORD *)(a2 + 16) = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a1 + 8);
}
