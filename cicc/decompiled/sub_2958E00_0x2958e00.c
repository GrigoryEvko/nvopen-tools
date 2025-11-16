// Function: sub_2958E00
// Address: 0x2958e00
//
__int64 __fastcall sub_2958E00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  __m128i v9; // xmm0
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // r15
  int v14; // r15d
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v16 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v17, a6);
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = v16;
    do
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = *(_QWORD *)v6;
        *(_QWORD *)(v8 + 8) = *(_QWORD *)(v6 + 8);
        v9 = _mm_loadu_si128((const __m128i *)(v6 + 16));
        *(_QWORD *)(v6 + 8) = 0;
        *(__m128i *)(v8 + 16) = v9;
        *(_QWORD *)(v8 + 32) = *(_QWORD *)(v6 + 32);
        *(__m128i *)(v8 + 40) = _mm_loadu_si128((const __m128i *)(v6 + 40));
        *(__m128i *)(v8 + 56) = _mm_loadu_si128((const __m128i *)(v6 + 56));
        *(_QWORD *)(v8 + 72) = *(_QWORD *)(v6 + 72);
      }
      v6 += 80LL;
      v8 += 80;
    }
    while ( v7 != v6 );
    v10 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v11 = *(_QWORD *)(v7 - 72);
        v7 -= 80LL;
        if ( v11 )
        {
          if ( (v11 & 4) != 0 )
          {
            v12 = (unsigned __int64 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
            v13 = (unsigned __int64)v12;
            if ( v12 )
            {
              if ( (unsigned __int64 *)*v12 != v12 + 2 )
                _libc_free(*v12);
              j_j___libc_free_0(v13);
            }
          }
        }
      }
      while ( v10 != v7 );
      v7 = *(_QWORD *)a1;
    }
  }
  v14 = v17[0];
  if ( a1 + 16 != v7 )
    _libc_free(v7);
  *(_DWORD *)(a1 + 12) = v14;
  *(_QWORD *)a1 = v16;
  return v16;
}
