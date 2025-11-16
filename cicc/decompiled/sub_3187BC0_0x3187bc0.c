// Function: sub_3187BC0
// Address: 0x3187bc0
//
void __fastcall sub_3187BC0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned __int64 v16; // r15
  void (__fastcall *v17)(unsigned __int64, unsigned __int64, __int64); // rax
  int v18; // r15d
  unsigned __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v19, a6);
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v6;
    do
    {
      if ( v9 )
      {
        v10 = *(_QWORD *)v7;
        v11 = _mm_loadu_si128((const __m128i *)(v9 + 8));
        *(_QWORD *)(v9 + 24) = 0;
        *(_QWORD *)v9 = v10;
        v12 = _mm_loadu_si128((const __m128i *)(v7 + 8));
        *(__m128i *)(v7 + 8) = v11;
        *(__m128i *)(v9 + 8) = v12;
        v13 = *(_QWORD *)(v7 + 24);
        *(_QWORD *)(v7 + 24) = 0;
        v14 = *(_QWORD *)(v9 + 32);
        *(_QWORD *)(v9 + 24) = v13;
        v15 = *(_QWORD *)(v7 + 32);
        *(_QWORD *)(v7 + 32) = v14;
        *(_QWORD *)(v9 + 32) = v15;
      }
      v7 += 40LL;
      v9 += 40;
    }
    while ( v8 != v7 );
    v16 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v17 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v8 - 16);
        v8 -= 40LL;
        if ( v17 )
          v17(v8 + 8, v8 + 8, 3);
      }
      while ( v8 != v16 );
      v8 = *(_QWORD *)a1;
    }
  }
  v18 = v19[0];
  if ( a1 + 16 != v8 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v18;
}
