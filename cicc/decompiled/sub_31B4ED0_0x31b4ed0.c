// Function: sub_31B4ED0
// Address: 0x31b4ed0
//
__int64 __fastcall sub_31B4ED0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  int v6; // eax
  unsigned int v7; // ecx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  __int64 v11; // r13
  __int64 v12; // r14
  _QWORD *v13; // r12
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // r12d
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *j; // rdx
  _QWORD *v26; // rax

  v5 = *(_QWORD *)(a1 + 96);
  *(_BYTE *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 96) = v5 + 1;
  if ( v5 >= qword_50357A8 && qword_50357A8 != -1 )
    return 0;
  v6 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_10;
    v8 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v8 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 64), 8 * v8, 8);
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_10;
    }
    goto LABEL_7;
  }
  v7 = 4 * v6;
  v8 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v8 <= v7 )
  {
LABEL_7:
    v9 = *(_QWORD **)(a1 + 64);
    for ( i = &v9[v8]; i != v9; ++v9 )
      *v9 = -4096;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_10;
  }
  v18 = v6 - 1;
  if ( !v18 )
  {
    v19 = *(_QWORD **)(a1 + 64);
    v20 = 64;
LABEL_29:
    sub_C7D6A0((__int64)v19, 8 * v8, 8);
    v21 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
    v22 = (v21
         | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
             | (4 * v20 / 3u + 1)
             | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
           | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 80) = v22;
    v23 = (_QWORD *)sub_C7D670(8 * v22, 8);
    v24 = *(unsigned int *)(a1 + 80);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 64) = v23;
    for ( j = &v23[v24]; j != v23; ++v23 )
    {
      if ( v23 )
        *v23 = -4096;
    }
    goto LABEL_10;
  }
  _BitScanReverse(&v18, v18);
  v19 = *(_QWORD **)(a1 + 64);
  v20 = 1 << (33 - (v18 ^ 0x1F));
  if ( v20 < 64 )
    v20 = 64;
  if ( (_DWORD)v8 != v20 )
    goto LABEL_29;
  *(_QWORD *)(a1 + 72) = 0;
  v26 = &v19[v8];
  do
  {
    if ( v19 )
      *v19 = -4096;
    ++v19;
  }
  while ( v26 != v19 );
LABEL_10:
  sub_31BE9C0(*(_QWORD *)(a1 + 48));
  v11 = *(_QWORD *)(a1 + 104);
  v12 = v11 + 8LL * *(unsigned int *)(a1 + 112);
  while ( v11 != v12 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD **)(v12 - 8);
      v12 -= 8;
      if ( !v13 )
        break;
      v14 = v13[17];
      if ( (_QWORD *)v14 != v13 + 19 )
        _libc_free(v14);
      v15 = v13[8];
      if ( (_QWORD *)v15 != v13 + 10 )
        _libc_free(v15);
      v16 = v13[2];
      if ( (_QWORD *)v16 != v13 + 4 )
        _libc_free(v16);
      j_j___libc_free_0((unsigned __int64)v13);
      if ( v11 == v12 )
        goto LABEL_20;
    }
  }
LABEL_20:
  *(_DWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  sub_31B2AF0(a1, a2, a3, 0, 0, 0);
  sub_31B45E0(a1);
  sub_31B3660(a1);
  return *(unsigned __int8 *)(a1 + 40);
}
