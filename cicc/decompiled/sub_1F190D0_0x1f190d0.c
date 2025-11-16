// Function: sub_1F190D0
// Address: 0x1f190d0
//
void __fastcall sub_1F190D0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r12
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned int v12; // r14d
  __int64 v13; // r8
  __int64 v14; // rcx
  int v15; // r9d
  _QWORD *v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rcx
  unsigned __int64 *v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // rax
  int v28; // r8d
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // r13
  int v34; // edi
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // rcx

  v8 = a1 + 8;
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = a2 - 1;
  if ( !v12 )
  {
    v28 = *(_DWORD *)(v10 + 188);
    v29 = *(_DWORD *)(v11 + 12) + 1;
    if ( v28 != v29 )
    {
      do
      {
        v30 = v29;
        v31 = v29++ - 1;
        *(_QWORD *)(v10 + 8 * v31 + 8) = *(_QWORD *)(v10 + 8 * v30 + 8);
        *(_QWORD *)(v10 + 8 * v31 + 96) = *(_QWORD *)(v10 + 8 * v30 + 96);
      }
      while ( v28 != v29 );
      v29 = *(_DWORD *)(v10 + 188);
    }
    v32 = v29 - 1;
    *(_DWORD *)(v10 + 188) = v32;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v32;
    if ( v32 )
      goto LABEL_8;
    *(_DWORD *)(v10 + 184) = 0;
    memset((void *)v10, 0, 0xB8u);
    v33 = *(_QWORD *)a1;
    v34 = *(_DWORD *)(*(_QWORD *)a1 + 184LL);
    v35 = *(unsigned int *)(*(_QWORD *)a1 + 188LL);
    *(_DWORD *)(a1 + 16) = 0;
    if ( v34 )
    {
      v33 += 8;
      v36 = 0;
      if ( *(_DWORD *)(a1 + 20) )
      {
LABEL_21:
        v37 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(v37 + v36) = v33;
        *(_QWORD *)(v37 + v36 + 8) = v35;
        ++*(_DWORD *)(a1 + 16);
        return;
      }
    }
    else
    {
      v36 = 0;
      if ( *(_DWORD *)(a1 + 20) )
        goto LABEL_21;
    }
    sub_16CD150(v8, (const void *)(a1 + 24), 0, 16, v28, a6);
    v36 = 16LL * *(unsigned int *)(a1 + 16);
    goto LABEL_21;
  }
  v13 = 16LL * v12;
  v14 = v13 + v11;
  v15 = *(_DWORD *)(v14 + 8);
  v16 = *(_QWORD **)v14;
  if ( v15 == 1 )
  {
    v27 = *(_QWORD **)(v10 + 192);
    *v16 = *v27;
    *v27 = v16;
    sub_1F190D0(a1, v12);
  }
  else
  {
    v17 = *(_DWORD *)(v14 + 12) + 1;
    if ( v15 != v17 )
    {
      do
      {
        v18 = v17;
        v19 = v17++ - 1;
        v16[v19] = v16[v18];
        v16[v19 + 12] = v16[v18 + 12];
      }
      while ( v15 != v17 );
      v14 = v13 + *(_QWORD *)(a1 + 8);
      v17 = *(_DWORD *)(v14 + 8);
    }
    *(_DWORD *)(v14 + 8) = v17 - 1;
    v20 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
    v21 = (unsigned __int64 *)(*(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 12));
    v22 = v17 - 2;
    *v21 = v22 | *v21 & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v13 + 12) == v17 - 1 )
    {
      sub_1F18EF0(a1, v12, v16[v22 + 12]);
      sub_39460A0(v8, v12);
    }
  }
LABEL_8:
  if ( *(_DWORD *)(a1 + 16) )
  {
    v23 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(v23 + 12) < *(_DWORD *)(v23 + 8) )
    {
      v24 = 16LL * a2;
      v25 = v23 + v24;
      v26 = *(_QWORD *)(*(_QWORD *)(v23 + 16LL * v12) + 8LL * *(unsigned int *)(v23 + 16LL * v12 + 12));
      *(_QWORD *)v25 = v26 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v25 + 8) = (v26 & 0x3F) + 1;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + v24 + 12) = 0;
    }
  }
}
