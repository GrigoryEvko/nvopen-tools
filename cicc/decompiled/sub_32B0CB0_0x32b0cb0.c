// Function: sub_32B0CB0
// Address: 0x32b0cb0
//
__int64 __fastcall sub_32B0CB0(__int64 a1, unsigned int a2)
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
  __int64 v16; // r10
  __int64 v17; // rcx
  unsigned __int64 *v18; // rdi
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
  __int64 v30; // r12
  int v31; // edi
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 *v34; // r9
  __int64 v35; // rcx

  v2 = a1 + 8;
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = a2 - 1;
  if ( !v8 )
  {
    v25 = *(unsigned int *)(v6 + 140);
    v26 = *(_DWORD *)(v7 + 12) + 1;
    if ( (_DWORD)v25 != v26 )
    {
      do
      {
        v27 = v26;
        v28 = v26++ - 1;
        *(_QWORD *)(v6 + 8 * v28 + 8) = *(_QWORD *)(v6 + 8 * v27 + 8);
        *(_QWORD *)(v6 + 8 * v28 + 72) = *(_QWORD *)(v6 + 8 * v27 + 72);
      }
      while ( (_DWORD)v25 != v26 );
      v26 = *(_DWORD *)(v6 + 140);
    }
    v29 = v26 - 1;
    *(_DWORD *)(v6 + 140) = v29;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v29;
    if ( v29 )
      goto LABEL_8;
    result = 0;
    *(_DWORD *)(v6 + 136) = 0;
    memset((void *)v6, 0, 0x88u);
    v30 = *(_QWORD *)a1;
    v31 = *(_DWORD *)(*(_QWORD *)a1 + 136LL);
    v32 = *(unsigned int *)(*(_QWORD *)a1 + 140LL);
    *(_DWORD *)(a1 + 16) = 0;
    if ( v31 )
    {
      v30 += 8;
      v33 = 0;
      if ( *(_DWORD *)(a1 + 20) )
      {
LABEL_21:
        v35 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(v35 + v33) = v30;
        *(_QWORD *)(v35 + v33 + 8) = v32;
        ++*(_DWORD *)(a1 + 16);
        return result;
      }
    }
    else
    {
      v33 = 0;
      if ( *(_DWORD *)(a1 + 20) )
        goto LABEL_21;
    }
    result = sub_C8D5F0(v2, (const void *)(a1 + 24), 1u, 0x10u, v25, v2);
    v33 = 16LL * *(unsigned int *)(a1 + 16);
    goto LABEL_21;
  }
  v9 = 16LL * v8;
  v10 = v9 + v7;
  v11 = *(_DWORD *)(v10 + 8);
  v12 = *(__int64 **)v10;
  if ( v11 == 1 )
  {
    v23 = *(__int64 **)(v6 + 144);
    v24 = *v23;
    *v12 = *v23;
    *v23 = (__int64)v12;
    sub_32B0CB0(a1, v8, v12, v24, v9, v2);
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
    v16 = v13 - 2;
    *(_DWORD *)(v10 + 8) = v13 - 1;
    v17 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
    v18 = (unsigned __int64 *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 12));
    *v18 = v16 | *v18 & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v9 + 12) == v13 - 1 )
    {
      sub_325DE80(a1, v8, v12[v16 + 12]);
      sub_F03D40(v34, v8);
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
