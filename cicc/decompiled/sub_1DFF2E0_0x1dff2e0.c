// Function: sub_1DFF2E0
// Address: 0x1dff2e0
//
void __fastcall sub_1DFF2E0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _DWORD *v4; // rax
  _DWORD *i; // rdx
  __int64 v6; // rax
  int v7; // r14d
  _QWORD *v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // r13
  __int64 v12; // r14
  unsigned int v13; // ecx
  _DWORD *v14; // rdi
  unsigned int v15; // eax
  int v16; // eax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  int v19; // ebx
  __int64 v20; // r13
  _DWORD *v21; // rax
  __int64 v22; // rdx
  _DWORD *j; // rdx
  __int64 v24; // r15
  unsigned int v25; // edx
  int v26; // ebx
  unsigned int v27; // r14d
  unsigned int v28; // eax
  _QWORD *v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *k; // rdx
  _DWORD *v35; // rax
  _QWORD *v36; // rax

  v2 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 64));
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v13 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v3 <= v13 )
  {
LABEL_4:
    v4 = *(_DWORD **)(a1 + 64);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -1;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_7;
  }
  v14 = *(_DWORD **)(a1 + 64);
  v15 = v2 - 1;
  if ( !v15 )
  {
    v20 = 1024;
    v19 = 128;
LABEL_31:
    j___libc_free_0(v14);
    *(_DWORD *)(a1 + 80) = v19;
    v21 = (_DWORD *)sub_22077B0(v20);
    v22 = *(unsigned int *)(a1 + 80);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 64) = v21;
    for ( j = &v21[2 * v22]; j != v21; v21 += 2 )
    {
      if ( v21 )
        *v21 = -1;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v15, v15);
  v16 = 1 << (33 - (v15 ^ 0x1F));
  if ( v16 < 64 )
    v16 = 64;
  if ( (_DWORD)v3 != v16 )
  {
    v17 = (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
        | (4 * v16 / 3u + 1)
        | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)
        | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
          | (4 * v16 / 3u + 1)
          | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4);
    v18 = (v17 >> 8) | v17;
    v19 = (v18 | (v18 >> 16)) + 1;
    v20 = 8 * ((v18 | (v18 >> 16)) + 1);
    goto LABEL_31;
  }
  *(_QWORD *)(a1 + 72) = 0;
  v35 = &v14[2 * v3];
  do
  {
    if ( v14 )
      *v14 = -1;
    v14 += 2;
  }
  while ( v35 != v14 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 88);
  if ( v6 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v6;
  v7 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v7 || *(_DWORD *)(a1 + 132) )
  {
    v8 = *(_QWORD **)(a1 + 120);
    v9 = 4 * v7;
    v10 = *(unsigned int *)(a1 + 136);
    v11 = &v8[2 * v10];
    if ( (unsigned int)(4 * v7) < 0x40 )
      v9 = 64;
    if ( (unsigned int)v10 <= v9 )
    {
      for ( ; v8 != v11; v8 += 2 )
      {
        if ( *v8 != -8 )
        {
          if ( *v8 != -16 )
          {
            v12 = v8[1];
            if ( v12 )
            {
              _libc_free(*(_QWORD *)(v12 + 48));
              _libc_free(*(_QWORD *)(v12 + 24));
              j_j___libc_free_0(v12, 72);
            }
          }
          *v8 = -8;
        }
      }
LABEL_21:
      *(_QWORD *)(a1 + 128) = 0;
      goto LABEL_22;
    }
    do
    {
      if ( *v8 != -16 && *v8 != -8 )
      {
        v24 = v8[1];
        if ( v24 )
        {
          _libc_free(*(_QWORD *)(v24 + 48));
          _libc_free(*(_QWORD *)(v24 + 24));
          j_j___libc_free_0(v24, 72);
        }
      }
      v8 += 2;
    }
    while ( v8 != v11 );
    v25 = *(_DWORD *)(a1 + 136);
    if ( !v7 )
    {
      if ( v25 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 120));
        *(_QWORD *)(a1 + 120) = 0;
        *(_QWORD *)(a1 + 128) = 0;
        *(_DWORD *)(a1 + 136) = 0;
        goto LABEL_22;
      }
      goto LABEL_21;
    }
    v26 = 64;
    v27 = v7 - 1;
    if ( v27 )
    {
      _BitScanReverse(&v28, v27);
      v26 = 1 << (33 - (v28 ^ 0x1F));
      if ( v26 < 64 )
        v26 = 64;
    }
    v29 = *(_QWORD **)(a1 + 120);
    if ( v25 == v26 )
    {
      *(_QWORD *)(a1 + 128) = 0;
      v36 = &v29[2 * v25];
      do
      {
        if ( v29 )
          *v29 = -8;
        v29 += 2;
      }
      while ( v36 != v29 );
    }
    else
    {
      j___libc_free_0(v29);
      v30 = ((((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
               | (4 * v26 / 3u + 1)
               | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
             | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
           | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 16;
      v31 = (v30
           | (((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
               | (4 * v26 / 3u + 1)
               | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
             | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
           | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 136) = v31;
      v32 = (_QWORD *)sub_22077B0(16 * v31);
      v33 = *(unsigned int *)(a1 + 136);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 120) = v32;
      for ( k = &v32[2 * v33]; k != v32; v32 += 2 )
      {
        if ( v32 )
          *v32 = -8;
      }
    }
  }
LABEL_22:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_21EB2D0(a1);
  sub_1DFEBA0((_QWORD *)a1);
}
