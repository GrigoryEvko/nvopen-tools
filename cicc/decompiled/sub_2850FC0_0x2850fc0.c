// Function: sub_2850FC0
// Address: 0x2850fc0
//
void __fastcall sub_2850FC0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  int v19; // ebx
  unsigned __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x70u, v21, a6);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v11;
    do
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = *(_QWORD *)v12;
        *(__m128i *)(v14 + 8) = _mm_loadu_si128((const __m128i *)(v12 + 8));
        *(_BYTE *)(v14 + 24) = *(_BYTE *)(v12 + 24);
        v15 = *(_QWORD *)(v12 + 32);
        *(_DWORD *)(v14 + 48) = 0;
        *(_QWORD *)(v14 + 32) = v15;
        *(_QWORD *)(v14 + 40) = v14 + 56;
        *(_DWORD *)(v14 + 52) = 4;
        v16 = *(unsigned int *)(v12 + 48);
        if ( (_DWORD)v16 )
        {
          v20 = v12;
          sub_28502F0(v14 + 40, (char **)(v12 + 40), v16, v8, v9, v10);
          v12 = v20;
        }
        *(_QWORD *)(v14 + 88) = *(_QWORD *)(v12 + 88);
        *(__m128i *)(v14 + 96) = _mm_loadu_si128((const __m128i *)(v12 + 96));
      }
      v12 += 112LL;
      v14 += 112;
    }
    while ( v13 != v12 );
    v17 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 112LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 112LL;
        v18 = *(_QWORD *)(v13 + 40);
        if ( v18 != v13 + 56 )
          _libc_free(v18);
      }
      while ( v13 != v17 );
      v13 = *(_QWORD *)a1;
    }
  }
  v19 = v21[0];
  if ( v6 != v13 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v19;
}
