// Function: sub_2FAF320
// Address: 0x2faf320
//
__int64 __fastcall sub_2FAF320(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rcx
  _QWORD *v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  _QWORD *v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned int v19; // ebx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  __int64 v22; // rbx
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *i; // rdx
  __int64 v27; // r14
  unsigned __int64 v28; // rax
  __int64 result; // rax
  __int64 j; // rbx
  __int64 v31; // r12

  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  v7 = *(unsigned int *)(a3 + 56);
  v8 = (_QWORD *)sub_2207820(112 * v7 + 8);
  if ( v8 )
  {
    *v8 = v7;
    v11 = v8;
    v12 = v8 + 1;
    if ( v7 )
    {
      v13 = v8 + 1;
      v14 = &v11[14 * v7 + 1];
      do
      {
        v15 = v13 + 5;
        *v13 = 0;
        v13 += 14;
        *(v13 - 13) = 0;
        *(v13 - 11) = v15;
        *((_DWORD *)v13 - 20) = 0;
        *((_DWORD *)v13 - 19) = 4;
        *(v13 - 1) = 0;
      }
      while ( v14 != v13 );
    }
  }
  else
  {
    v12 = 0;
  }
  v16 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = v12;
  if ( v16 )
  {
    v17 = v16 + 112LL * *(_QWORD *)(v16 - 8);
    while ( v16 != v17 )
    {
      v17 -= 112;
      v18 = *(_QWORD *)(v17 + 24);
      if ( v18 != v17 + 40 )
        _libc_free(v18);
    }
    j_j_j___libc_free_0_0(v16 - 8);
  }
  *(_DWORD *)(a1 + 232) = 0;
  v19 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
  if ( v19 < *(_DWORD *)(a1 + 280) >> 2 || v19 > *(_DWORD *)(a1 + 280) )
  {
    v20 = (__int64)_libc_calloc(v19, 1u);
    if ( !v20 && (v19 || (v20 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    v21 = *(_QWORD *)(a1 + 272);
    *(_QWORD *)(a1 + 272) = v20;
    if ( v21 )
      _libc_free(v21);
    *(_DWORD *)(a1 + 280) = v19;
  }
  v22 = (__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3;
  v23 = *(unsigned int *)(a1 + 144);
  if ( (unsigned int)v22 != v23 )
  {
    if ( (unsigned int)v22 >= v23 )
    {
      if ( (unsigned int)v22 > (unsigned __int64)*(unsigned int *)(a1 + 148) )
      {
        sub_C8D5F0(
          a1 + 136,
          (const void *)(a1 + 152),
          (unsigned int)((__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3),
          8u,
          v9,
          v10);
        v23 = *(unsigned int *)(a1 + 144);
      }
      v24 = *(_QWORD *)(a1 + 136);
      v25 = (_QWORD *)(v24 + 8 * v23);
      for ( i = (_QWORD *)(v24 + 8LL * (unsigned int)v22); i != v25; ++v25 )
      {
        if ( v25 )
          *v25 = 0;
      }
    }
    *(_DWORD *)(a1 + 144) = v22;
  }
  v27 = a2 + 320;
  v28 = sub_2E3A080((__int64)a4);
  result = sub_2FAF2F0(a1, v28);
  for ( j = *(_QWORD *)(v27 + 8); v27 != j; j = *(_QWORD *)(j + 8) )
  {
    v31 = *(unsigned int *)(j + 24);
    result = sub_2E39EA0(a4, j);
    *(_QWORD *)(*(_QWORD *)(a1 + 136) + 8 * v31) = result;
  }
  return result;
}
