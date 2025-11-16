// Function: sub_2E1B190
// Address: 0x2e1b190
//
__int64 __fastcall sub_2E1B190(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // r12
  _QWORD *v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // r14d
  __int64 v12; // r8
  __int64 v13; // rsi
  int v14; // r9d
  _QWORD *v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdx
  unsigned __int64 *v20; // rdx
  __int64 v21; // rsi
  __int64 result; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // rcx
  unsigned int v31; // eax
  _QWORD *v32; // rax
  _QWORD *v33; // r13
  int v34; // edx
  int v35; // ecx
  __int64 v36; // r14
  __int64 v37; // rcx
  __int64 v38; // rdx

  v8 = (__int64 *)(a1 + 8);
  v9 = *(_QWORD **)a1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2 - 1;
  if ( !v11 )
  {
    v27 = *((unsigned int *)v9 + 49);
    v28 = *(_DWORD *)(v10 + 12) + 1;
    if ( (_DWORD)v27 != v28 )
    {
      do
      {
        v29 = v28;
        v30 = v28++ - 1;
        v9[v30 + 1] = v9[v29 + 1];
        v9[v30 + 12] = v9[v29 + 12];
      }
      while ( (_DWORD)v27 != v28 );
      v28 = *((_DWORD *)v9 + 49);
    }
    v31 = v28 - 1;
    *((_DWORD *)v9 + 49) = v31;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v31;
    if ( v31 )
      goto LABEL_8;
    *((_DWORD *)v9 + 48) = 0;
    memset(v9, 0, 0xC0u);
    v32 = v9 + 16;
    do
    {
      *v9 = 0;
      v9 += 2;
      *(v9 - 1) = 0;
    }
    while ( v32 != v9 );
    v33 = *(_QWORD **)a1;
    v34 = *(_DWORD *)(a1 + 20);
    v35 = *(_DWORD *)(*(_QWORD *)a1 + 192LL);
    result = *(unsigned int *)(*(_QWORD *)a1 + 196LL);
    *(_DWORD *)(a1 + 16) = 0;
    if ( v35 )
    {
      ++v33;
      v36 = (unsigned int)result;
      v37 = 0;
      if ( v34 )
      {
LABEL_23:
        v38 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(v38 + v37) = v33;
        *(_QWORD *)(v38 + v37 + 8) = v36;
        ++*(_DWORD *)(a1 + 16);
        return result;
      }
    }
    else
    {
      v36 = (unsigned int)result;
      v37 = 0;
      if ( v34 )
        goto LABEL_23;
    }
    result = sub_C8D5F0((__int64)v8, (const void *)(a1 + 24), 1u, 0x10u, v27, a6);
    v37 = 16LL * *(unsigned int *)(a1 + 16);
    goto LABEL_23;
  }
  v12 = 16LL * v11;
  v13 = v12 + v10;
  v14 = *(_DWORD *)(v13 + 8);
  v15 = *(_QWORD **)v13;
  if ( v14 == 1 )
  {
    v26 = (_QWORD *)v9[25];
    *v15 = *v26;
    *v26 = v15;
    sub_2E1B190(a1, v11);
  }
  else
  {
    v16 = *(_DWORD *)(v13 + 12) + 1;
    if ( v14 != v16 )
    {
      do
      {
        v17 = v16;
        v18 = v16++ - 1;
        v15[v18] = v15[v17];
        v15[v18 + 12] = v15[v17 + 12];
      }
      while ( v14 != v16 );
      v13 = v12 + *(_QWORD *)(a1 + 8);
      v16 = *(_DWORD *)(v13 + 8);
    }
    *(_DWORD *)(v13 + 8) = v16 - 1;
    v19 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
    v20 = (unsigned __int64 *)(*(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 12));
    v21 = v16 - 2;
    *v20 = v21 | *v20 & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v12 + 12) == v16 - 1 )
    {
      sub_2E1A5E0(a1, v11, v15[v21 + 12]);
      sub_F03D40(v8, v11);
    }
  }
LABEL_8:
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
    {
      v23 = 16LL * a2;
      v24 = result + v23;
      v25 = *(_QWORD *)(*(_QWORD *)(result + 16LL * v11) + 8LL * *(unsigned int *)(result + 16LL * v11 + 12));
      *(_QWORD *)v24 = v25 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v24 + 8) = (v25 & 0x3F) + 1;
      result = *(_QWORD *)(a1 + 8);
      *(_DWORD *)(result + v23 + 12) = 0;
    }
  }
  return result;
}
