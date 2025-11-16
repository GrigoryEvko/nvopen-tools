// Function: sub_2DF6700
// Address: 0x2df6700
//
__int64 __fastcall sub_2DF6700(__int64 a1, unsigned int a2)
{
  __int64 v2; // r9
  __int64 v6; // rsi
  __int64 v7; // rcx
  unsigned int v8; // r13d
  __int64 v9; // r8
  __int64 v10; // rcx
  int v11; // r10d
  __int64 *v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rcx
  unsigned __int64 *v17; // rcx
  __int64 v18; // rdi
  __int64 result; // rax
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rdx
  unsigned int v29; // eax
  _QWORD *v30; // rax
  _QWORD *v31; // rsi
  __int64 v32; // r12
  int v33; // edx
  int v34; // ecx
  __int64 v35; // r13
  __int64 v36; // rcx
  __int64 *v37; // r9
  __int64 v38; // rdx

  v2 = a1 + 8;
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = a2 - 1;
  if ( !v8 )
  {
    v25 = *(unsigned int *)(v6 + 164);
    v26 = *(_DWORD *)(v7 + 12) + 1;
    if ( (_DWORD)v25 != v26 )
    {
      do
      {
        v27 = v26;
        v28 = v26++ - 1;
        *(_QWORD *)(v6 + 8 * v28 + 8) = *(_QWORD *)(v6 + 8 * v27 + 8);
        *(_QWORD *)(v6 + 8 * v28 + 80) = *(_QWORD *)(v6 + 8 * v27 + 80);
      }
      while ( (_DWORD)v25 != v26 );
      v26 = *(_DWORD *)(v6 + 164);
    }
    v29 = v26 - 1;
    *(_DWORD *)(v6 + 164) = v29;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v29;
    if ( v29 )
      goto LABEL_8;
    *(_DWORD *)(v6 + 160) = 0;
    memset((void *)v6, 0, 0xA0u);
    v30 = (_QWORD *)v6;
    do
    {
      *v30 = 0;
      v30 += 2;
      *(v30 - 1) = 0;
    }
    while ( v30 != (_QWORD *)(v6 + 64) );
    v31 = (_QWORD *)(v6 + 160);
    do
    {
      *v30 = 0;
      v30 += 3;
      *((_BYTE *)v30 - 16) = 0;
      *(v30 - 1) = 0;
    }
    while ( v31 != v30 );
    v32 = *(_QWORD *)a1;
    v33 = *(_DWORD *)(a1 + 20);
    v34 = *(_DWORD *)(*(_QWORD *)a1 + 160LL);
    result = *(unsigned int *)(*(_QWORD *)a1 + 164LL);
    *(_DWORD *)(a1 + 16) = 0;
    if ( v34 )
    {
      v32 += 8;
      v35 = (unsigned int)result;
      v36 = 0;
      if ( v33 )
      {
LABEL_25:
        v38 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(v38 + v36) = v32;
        *(_QWORD *)(v38 + v36 + 8) = v35;
        ++*(_DWORD *)(a1 + 16);
        return result;
      }
    }
    else
    {
      v35 = (unsigned int)result;
      v36 = 0;
      if ( v33 )
        goto LABEL_25;
    }
    result = sub_C8D5F0(v2, (const void *)(a1 + 24), 1u, 0x10u, v25, v2);
    v36 = 16LL * *(unsigned int *)(a1 + 16);
    goto LABEL_25;
  }
  v9 = 16LL * v8;
  v10 = v9 + v7;
  v11 = *(_DWORD *)(v10 + 8);
  v12 = *(__int64 **)v10;
  if ( v11 == 1 )
  {
    v23 = *(__int64 **)(v6 + 168);
    v24 = *v23;
    *v12 = *v23;
    *v23 = (__int64)v12;
    sub_2DF6700(a1, v8, v12, v24, v9, v2);
  }
  else
  {
    v13 = *(_DWORD *)(v10 + 12) + 1;
    if ( v11 != v13 )
    {
      do
      {
        v14 = v13;
        v15 = v13++ - 1;
        v12[v15] = v12[v14];
        v12[v15 + 12] = v12[v14 + 12];
      }
      while ( v11 != v13 );
      v10 = v9 + *(_QWORD *)(a1 + 8);
      v13 = *(_DWORD *)(v10 + 8);
    }
    *(_DWORD *)(v10 + 8) = v13 - 1;
    v16 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
    v17 = (unsigned __int64 *)(*(_QWORD *)v16 + 8LL * *(unsigned int *)(v16 + 12));
    v18 = v13 - 2;
    *v17 = v18 | *v17 & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v9 + 12) == v13 - 1 )
    {
      sub_2DF4670(a1, v8, v12[v18 + 12]);
      sub_F03D40(v37, v8);
    }
  }
LABEL_8:
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
    {
      v20 = 16LL * a2;
      v21 = result + v20;
      v22 = *(_QWORD *)(*(_QWORD *)(result + 16LL * v8) + 8LL * *(unsigned int *)(result + 16LL * v8 + 12));
      *(_QWORD *)v21 = v22 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v21 + 8) = (v22 & 0x3F) + 1;
      result = *(_QWORD *)(a1 + 8);
      *(_DWORD *)(result + v20 + 12) = 0;
    }
  }
  return result;
}
