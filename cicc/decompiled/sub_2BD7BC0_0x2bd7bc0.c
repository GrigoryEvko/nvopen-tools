// Function: sub_2BD7BC0
// Address: 0x2bd7bc0
//
__int64 __fastcall sub_2BD7BC0(__int64 a1, char a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v8; // r12d
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *k; // rdx
  unsigned int v15; // ecx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  int v18; // r13d
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // r13
  int v28; // eax
  unsigned int v29; // ecx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *i; // rdx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *m; // rdx
  unsigned int v38; // eax
  _QWORD *v39; // rdi
  int v40; // r14d
  _QWORD *v41; // rax
  unsigned int v42; // eax
  _QWORD *v43; // rax
  __int64 v44; // rdx
  _QWORD *j; // rdx
  __int64 v46[6]; // [rsp+0h] [rbp-30h] BYREF

  v8 = sub_2BD7690(*(_QWORD *)(a1 + 24), *(_QWORD *)a1, **(_QWORD **)(a1 + 8), *(_QWORD *)(a1 + 16), a3, a6, a7);
  if ( a2 )
  {
    v20 = *(_QWORD *)(a1 + 32);
    v21 = *(_QWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    v23 = **(_QWORD **)(a1 + 8);
    v24 = *(_QWORD *)(v20 + 32);
    v25 = *(unsigned int *)(v20 + 40);
    v46[1] = v24;
    v46[0] = v24 + 8 * v25;
    v26 = sub_2BD71E0(v21, v46, v23, v22, a3);
    v27 = *(_QWORD *)(a1 + 32);
    v8 |= v26;
    v28 = *(_DWORD *)(v27 + 16);
    ++*(_QWORD *)v27;
    if ( v28 )
    {
      v29 = 4 * v28;
      v30 = *(unsigned int *)(v27 + 24);
      if ( (unsigned int)(4 * v28) < 0x40 )
        v29 = 64;
      if ( (unsigned int)v30 <= v29 )
      {
LABEL_25:
        v31 = *(_QWORD **)(v27 + 8);
        for ( i = &v31[v30]; i != v31; ++v31 )
          *v31 = -4096;
        goto LABEL_27;
      }
      v38 = v28 - 1;
      if ( v38 )
      {
        _BitScanReverse(&v38, v38);
        v39 = *(_QWORD **)(v27 + 8);
        v40 = 1 << (33 - (v38 ^ 0x1F));
        if ( v40 < 64 )
          v40 = 64;
        if ( (_DWORD)v30 == v40 )
        {
          *(_QWORD *)(v27 + 16) = 0;
          v41 = &v39[v30];
          do
          {
            if ( v39 )
              *v39 = -4096;
            ++v39;
          }
          while ( v41 != v39 );
          goto LABEL_28;
        }
      }
      else
      {
        v39 = *(_QWORD **)(v27 + 8);
        v40 = 64;
      }
      sub_C7D6A0((__int64)v39, 8 * v30, 8);
      v42 = sub_2B149A0(v40);
      *(_DWORD *)(v27 + 24) = v42;
      if ( v42 )
      {
        v43 = (_QWORD *)sub_C7D670(8LL * v42, 8);
        v44 = *(unsigned int *)(v27 + 24);
        *(_QWORD *)(v27 + 16) = 0;
        *(_QWORD *)(v27 + 8) = v43;
        for ( j = &v43[v44]; j != v43; ++v43 )
        {
          if ( v43 )
            *v43 = -4096;
        }
        goto LABEL_28;
      }
    }
    else
    {
      if ( !*(_DWORD *)(v27 + 20) )
        goto LABEL_28;
      v30 = *(unsigned int *)(v27 + 24);
      if ( (unsigned int)v30 <= 0x40 )
        goto LABEL_25;
      sub_C7D6A0(*(_QWORD *)(v27 + 8), 8 * v30, 8);
      *(_DWORD *)(v27 + 24) = 0;
    }
    *(_QWORD *)(v27 + 8) = 0;
LABEL_27:
    *(_QWORD *)(v27 + 16) = 0;
LABEL_28:
    *(_DWORD *)(v27 + 40) = 0;
  }
  v9 = *(_QWORD *)a1;
  v10 = *(_DWORD *)(v9 + 16);
  ++*(_QWORD *)v9;
  if ( !v10 )
  {
    if ( !*(_DWORD *)(v9 + 20) )
      goto LABEL_8;
    v11 = *(unsigned int *)(v9 + 24);
    if ( (unsigned int)v11 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(v9 + 8), 8 * v11, 8);
      *(_QWORD *)(v9 + 8) = 0;
      *(_QWORD *)(v9 + 16) = 0;
      *(_DWORD *)(v9 + 24) = 0;
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  v15 = 4 * v10;
  v11 = *(unsigned int *)(v9 + 24);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v15 = 64;
  if ( (unsigned int)v11 <= v15 )
  {
LABEL_5:
    v12 = *(_QWORD **)(v9 + 8);
    for ( k = &v12[v11]; k != v12; ++v12 )
      *v12 = -4096;
    *(_QWORD *)(v9 + 16) = 0;
    goto LABEL_8;
  }
  v16 = v10 - 1;
  if ( v16 )
  {
    _BitScanReverse(&v16, v16);
    v17 = *(_QWORD **)(v9 + 8);
    v18 = 1 << (33 - (v16 ^ 0x1F));
    if ( v18 < 64 )
      v18 = 64;
    if ( (_DWORD)v11 == v18 )
    {
      *(_QWORD *)(v9 + 16) = 0;
      v19 = &v17[v11];
      do
      {
        if ( v17 )
          *v17 = -4096;
        ++v17;
      }
      while ( v19 != v17 );
      goto LABEL_8;
    }
  }
  else
  {
    v17 = *(_QWORD **)(v9 + 8);
    v18 = 64;
  }
  sub_C7D6A0((__int64)v17, 8 * v11, 8);
  v33 = ((((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
       | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
       | (4 * v18 / 3u + 1)
       | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 16;
  v34 = (v33
       | (((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
       | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
       | (4 * v18 / 3u + 1)
       | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(v9 + 24) = v34;
  v35 = (_QWORD *)sub_C7D670(8 * v34, 8);
  v36 = *(unsigned int *)(v9 + 24);
  *(_QWORD *)(v9 + 16) = 0;
  *(_QWORD *)(v9 + 8) = v35;
  for ( m = &v35[v36]; m != v35; ++v35 )
  {
    if ( v35 )
      *v35 = -4096;
  }
LABEL_8:
  *(_DWORD *)(v9 + 40) = 0;
  return v8;
}
