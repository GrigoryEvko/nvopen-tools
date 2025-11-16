// Function: sub_31D7970
// Address: 0x31d7970
//
void __fastcall sub_31D7970(unsigned __int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  char v16; // al
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned int v28; // ecx
  unsigned int v29; // eax
  _QWORD *v30; // rdi
  int v31; // ebx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *j; // rdx
  __int64 *v37; // rax
  __int64 v38; // rcx
  __int64 *v39; // rbx
  __int64 *v40; // r14
  __int64 v41; // rdi
  unsigned int v42; // ecx
  __int64 v43; // rsi
  __int64 *v44; // rbx
  __int64 v45; // rsi
  __int64 v46; // rdi
  _QWORD *v47; // rax
  __int64 *v48; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      a2 = 16 * v4;
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v4, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v28 = 4 * v3;
  a2 = 64;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v28 = 64;
  if ( v28 >= (unsigned int)v4 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 8);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v29 = v3 - 1;
  if ( !v29 )
  {
    v30 = *(_QWORD **)(a1 + 8);
    v31 = 64;
LABEL_50:
    sub_C7D6A0((__int64)v30, 16 * v4, 8);
    a2 = 8;
    v32 = ((((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
             | (4 * v31 / 3u + 1)
             | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 16;
    v33 = (v32
         | (((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
             | (4 * v31 / 3u + 1)
             | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v33;
    v34 = (_QWORD *)sub_C7D670(16 * v33, 8);
    v35 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v34;
    for ( j = &v34[2 * v35]; j != v34; v34 += 2 )
    {
      if ( v34 )
        *v34 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v29, v29);
  v30 = *(_QWORD **)(a1 + 8);
  v31 = 1 << (33 - (v29 ^ 0x1F));
  if ( v31 < 64 )
    v31 = 64;
  if ( (_DWORD)v4 != v31 )
    goto LABEL_50;
  *(_QWORD *)(a1 + 16) = 0;
  v47 = &v30[2 * (unsigned int)v4];
  do
  {
    if ( v30 )
      *v30 = -4096;
    v30 += 2;
  }
  while ( v47 != v30 );
LABEL_7:
  v7 = *(__int64 **)(a1 + 32);
  v48 = *(__int64 **)(a1 + 40);
  if ( v7 != v48 )
  {
    do
    {
      v8 = *v7;
      v9 = *(__int64 **)(*v7 + 16);
      if ( *(__int64 **)(*v7 + 8) == v9 )
      {
        *(_BYTE *)(v8 + 152) = 1;
      }
      else
      {
        v10 = *(__int64 **)(*v7 + 8);
        do
        {
          v11 = *v10++;
          sub_2EA4EF0(v11, a2);
        }
        while ( v9 != v10 );
        *(_BYTE *)(v8 + 152) = 1;
        v12 = *(_QWORD *)(v8 + 8);
        if ( v12 != *(_QWORD *)(v8 + 16) )
          *(_QWORD *)(v8 + 16) = v12;
      }
      v13 = *(_QWORD *)(v8 + 32);
      if ( v13 != *(_QWORD *)(v8 + 40) )
        *(_QWORD *)(v8 + 40) = v13;
      ++*(_QWORD *)(v8 + 56);
      if ( *(_BYTE *)(v8 + 84) )
      {
        *(_QWORD *)v8 = 0;
      }
      else
      {
        v14 = 4 * (*(_DWORD *)(v8 + 76) - *(_DWORD *)(v8 + 80));
        v15 = *(unsigned int *)(v8 + 72);
        if ( v14 < 0x20 )
          v14 = 32;
        if ( (unsigned int)v15 > v14 )
        {
          sub_C8C990(v8 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v8 + 64), -1, 8 * v15);
        }
        v16 = *(_BYTE *)(v8 + 84);
        *(_QWORD *)v8 = 0;
        if ( !v16 )
          _libc_free(*(_QWORD *)(v8 + 64));
      }
      v17 = *(_QWORD *)(v8 + 32);
      if ( v17 )
      {
        a2 = *(_QWORD *)(v8 + 48) - v17;
        j_j___libc_free_0(v17);
      }
      v18 = *(_QWORD *)(v8 + 8);
      if ( v18 )
      {
        a2 = *(_QWORD *)(v8 + 24) - v18;
        j_j___libc_free_0(v18);
      }
      ++v7;
    }
    while ( v48 != v7 );
    v19 = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a1 + 40) != v19 )
      *(_QWORD *)(a1 + 40) = v19;
  }
  v20 = *(__int64 **)(a1 + 120);
  v21 = &v20[2 * *(unsigned int *)(a1 + 128)];
  while ( v21 != v20 )
  {
    v22 = v20[1];
    v23 = *v20;
    v20 += 2;
    sub_C7D6A0(v23, v22, 16);
  }
  *(_DWORD *)(a1 + 128) = 0;
  v24 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v24 )
    goto LABEL_32;
  *(_QWORD *)(a1 + 136) = 0;
  v37 = *(__int64 **)(a1 + 72);
  v38 = *v37;
  v39 = &v37[v24];
  v40 = v37 + 1;
  *(_QWORD *)(a1 + 56) = *v37;
  for ( *(_QWORD *)(a1 + 64) = v38 + 4096; v39 != v40; v37 = *(__int64 **)(a1 + 72) )
  {
    v41 = *v40;
    v42 = (unsigned int)(v40 - v37) >> 7;
    v43 = 4096LL << v42;
    if ( v42 >= 0x1E )
      v43 = 0x40000000000LL;
    ++v40;
    sub_C7D6A0(v41, v43, 16);
  }
  *(_DWORD *)(a1 + 80) = 1;
  sub_C7D6A0(*v37, 4096, 16);
  v44 = *(__int64 **)(a1 + 120);
  v25 = (unsigned __int64)&v44[2 * *(unsigned int *)(a1 + 128)];
  if ( v44 != (__int64 *)v25 )
  {
    do
    {
      v45 = v44[1];
      v46 = *v44;
      v44 += 2;
      sub_C7D6A0(v46, v45, 16);
    }
    while ( (__int64 *)v25 != v44 );
LABEL_32:
    v25 = *(_QWORD *)(a1 + 120);
  }
  if ( v25 != a1 + 136 )
    _libc_free(v25);
  v26 = *(_QWORD *)(a1 + 72);
  if ( v26 != a1 + 88 )
    _libc_free(v26);
  v27 = *(_QWORD *)(a1 + 32);
  if ( v27 )
    j_j___libc_free_0(v27);
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
  j_j___libc_free_0(a1);
}
