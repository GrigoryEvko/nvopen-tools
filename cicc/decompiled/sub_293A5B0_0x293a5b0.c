// Function: sub_293A5B0
// Address: 0x293a5b0
//
__int64 __fastcall sub_293A5B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  unsigned __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r8
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  int v19; // ebx
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0xA0u, v22, a6);
  v12 = *(_QWORD *)a1;
  v21 = v8;
  v13 = v8;
  v14 = 160LL * *(unsigned int *)(a1 + 8);
  v15 = *(_QWORD *)a1 + v14;
  if ( *(_QWORD *)a1 != v15 )
  {
    do
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = *(_QWORD *)v12;
        *(__m128i *)(v13 + 8) = _mm_loadu_si128((const __m128i *)(v12 + 8));
        *(_QWORD *)(v13 + 24) = *(_QWORD *)(v12 + 24);
        *(__m128i *)(v13 + 32) = _mm_loadu_si128((const __m128i *)(v12 + 32));
        *(__m128i *)(v13 + 48) = _mm_loadu_si128((const __m128i *)(v12 + 48));
        *(_BYTE *)(v13 + 64) = *(_BYTE *)(v12 + 64);
        v16 = *(_QWORD *)(v12 + 72);
        *(_DWORD *)(v13 + 88) = 0;
        *(_QWORD *)(v13 + 72) = v16;
        *(_QWORD *)(v13 + 80) = v13 + 96;
        *(_DWORD *)(v13 + 92) = 8;
        if ( *(_DWORD *)(v12 + 88) )
          sub_293A290(v13 + 80, (char **)(v12 + 80), v9, v10, v14, v11);
      }
      v12 += 160LL;
      v13 += 160;
    }
    while ( v15 != v12 );
    v17 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 160LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 160LL;
        v18 = *(_QWORD *)(v15 + 80);
        if ( v18 != v15 + 96 )
          _libc_free(v18);
      }
      while ( v15 != v17 );
      v15 = *(_QWORD *)a1;
    }
  }
  v19 = v22[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_DWORD *)(a1 + 12) = v19;
  *(_QWORD *)a1 = v21;
  return v21;
}
