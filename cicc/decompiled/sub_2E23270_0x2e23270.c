// Function: sub_2E23270
// Address: 0x2e23270
//
__int64 __fastcall sub_2E23270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // r15
  __int64 *v9; // r13
  __int64 *i; // rax
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rsi
  __int64 *v14; // r13
  __int64 *v15; // r14
  __int64 v16; // rsi
  __int64 v17; // rdi
  const void *v18; // rsi
  __int64 v19; // r13
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  int v22; // r14d
  __int64 v23; // r15
  char *v24; // rsi
  __int64 v25; // rdx
  _QWORD *v26; // r13
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi

  v8 = *(__int64 **)(a1 + 24);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  v9 = &v8[*(unsigned int *)(a1 + 32)];
  if ( v8 != v9 )
  {
    for ( i = v8; ; i = *(__int64 **)(a1 + 24) )
    {
      v11 = *v8;
      v12 = (unsigned int)(v8 - i) >> 7;
      v13 = 4096LL << v12;
      if ( v12 >= 0x1E )
        v13 = 0x40000000000LL;
      ++v8;
      sub_C7D6A0(v11, v13, 16);
      if ( v9 == v8 )
        break;
    }
  }
  v14 = *(__int64 **)(a1 + 72);
  v15 = &v14[2 * *(unsigned int *)(a1 + 80)];
  while ( v15 != v14 )
  {
    v16 = v14[1];
    v17 = *v14;
    v14 += 2;
    sub_C7D6A0(v17, v16, 16);
  }
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 88) = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 96);
  if ( a1 + 24 != a2 + 24 )
  {
    v18 = *(const void **)(a2 + 24);
    v19 = a2 + 40;
    if ( v18 == (const void *)(a2 + 40) )
    {
      v20 = *(unsigned int *)(a2 + 32);
      v21 = *(unsigned int *)(a1 + 32);
      v22 = *(_DWORD *)(a2 + 32);
      if ( v20 <= v21 )
      {
        if ( *(_DWORD *)(a2 + 32) )
          memmove(*(void **)(a1 + 24), v18, 8 * v20);
        goto LABEL_17;
      }
      if ( v20 > *(unsigned int *)(a1 + 36) )
      {
        *(_DWORD *)(a1 + 32) = 0;
        sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v20, 8u, a5, a6);
        v21 = 0;
        v25 = 8LL * *(unsigned int *)(a2 + 32);
        v24 = *(char **)(a2 + 24);
        if ( v24 == &v24[v25] )
          goto LABEL_17;
      }
      else
      {
        v23 = 8 * v21;
        v24 = (char *)(a2 + 40);
        if ( *(_DWORD *)(a1 + 32) )
        {
          memmove(*(void **)(a1 + 24), v24, 8 * v21);
          v19 = *(_QWORD *)(a2 + 24);
          v20 = *(unsigned int *)(a2 + 32);
          v21 = v23;
          v24 = (char *)(v19 + v23);
        }
        v25 = 8 * v20;
        if ( v24 == (char *)(v25 + v19) )
          goto LABEL_17;
      }
      memcpy((void *)(v21 + *(_QWORD *)(a1 + 24)), v24, v25 - v21);
LABEL_17:
      *(_DWORD *)(a1 + 32) = v22;
      *(_DWORD *)(a2 + 32) = 0;
      goto LABEL_18;
    }
    v34 = *(_QWORD *)(a1 + 24);
    if ( v34 != a1 + 40 )
    {
      _libc_free(v34);
      v18 = *(const void **)(a2 + 24);
    }
    *(_QWORD *)(a1 + 24) = v18;
    *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 36) = *(_DWORD *)(a2 + 36);
    *(_QWORD *)(a2 + 24) = v19;
    *(_QWORD *)(a2 + 32) = 0;
  }
LABEL_18:
  if ( a2 + 72 != a1 + 72 )
  {
    if ( *(_DWORD *)(a2 + 80) )
    {
      v35 = *(_QWORD *)(a1 + 72);
      if ( v35 != a1 + 88 )
        _libc_free(v35);
      *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
      *(_DWORD *)(a1 + 80) = *(_DWORD *)(a2 + 80);
      *(_DWORD *)(a1 + 84) = *(_DWORD *)(a2 + 84);
      *(_QWORD *)(a2 + 72) = a2 + 88;
      *(_DWORD *)(a2 + 84) = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 80) = 0;
    }
  }
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 88) = 0;
  *(_DWORD *)(a2 + 32) = 0;
  *(_DWORD *)(a2 + 80) = 0;
  v26 = *(_QWORD **)(a1 + 120);
  while ( v26 )
  {
    v27 = v26;
    v26 = (_QWORD *)*v26;
    sub_2E22AE0(v27);
  }
  v28 = *(_QWORD *)(a1 + 104);
  if ( v28 != a1 + 152 )
    j_j___libc_free_0(v28);
  *(__m128i *)(a1 + 136) = _mm_loadu_si128((const __m128i *)(a2 + 136));
  v29 = *(_QWORD *)(a2 + 104);
  if ( v29 == a2 + 152 )
  {
    *(_QWORD *)(a1 + 104) = a1 + 152;
    *(_QWORD *)(a1 + 152) = *(_QWORD *)(a2 + 152);
  }
  else
  {
    *(_QWORD *)(a1 + 104) = v29;
  }
  v30 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 112) = v30;
  v31 = *(_QWORD *)(a2 + 120);
  *(_QWORD *)(a1 + 120) = v31;
  *(_QWORD *)(a1 + 128) = *(_QWORD *)(a2 + 128);
  if ( v31 )
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * (*(int *)(v31 + 8) % v30)) = a1 + 120;
  *(_QWORD *)(a2 + 144) = 0;
  *(_QWORD *)(a2 + 112) = 1;
  *(_QWORD *)(a2 + 152) = 0;
  *(_QWORD *)(a2 + 104) = a2 + 152;
  *(_QWORD *)(a2 + 120) = 0;
  *(_QWORD *)(a2 + 128) = 0;
  sub_2E22B80(*(_QWORD *)(a1 + 176));
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 200) = 0;
  if ( *(_QWORD *)(a2 + 176) )
  {
    *(_DWORD *)(a1 + 168) = *(_DWORD *)(a2 + 168);
    v32 = *(_QWORD *)(a2 + 176);
    *(_QWORD *)(a1 + 176) = v32;
    *(_QWORD *)(a1 + 184) = *(_QWORD *)(a2 + 184);
    *(_QWORD *)(a1 + 192) = *(_QWORD *)(a2 + 192);
    *(_QWORD *)(v32 + 8) = a1 + 168;
    *(_QWORD *)(a1 + 200) = *(_QWORD *)(a2 + 200);
    *(_QWORD *)(a2 + 176) = 0;
    *(_QWORD *)(a2 + 184) = a2 + 168;
    *(_QWORD *)(a2 + 192) = a2 + 168;
    *(_QWORD *)(a2 + 200) = 0;
  }
  return a1;
}
