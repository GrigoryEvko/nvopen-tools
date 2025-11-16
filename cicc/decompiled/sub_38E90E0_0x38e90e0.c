// Function: sub_38E90E0
// Address: 0x38e90e0
//
void __fastcall sub_38E90E0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rsi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r12
  __m128i v12; // xmm2
  int v13; // eax
  unsigned __int64 v14; // r13
  int v15; // eax
  bool v16; // cc
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __m128i v23; // xmm3
  unsigned int v24; // eax
  __int64 v25; // rax
  unsigned __int64 v26; // [rsp+8h] [rbp-38h]
  unsigned __int64 v27; // [rsp+8h] [rbp-38h]
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 12);
  LODWORD(v8) = v6;
  v9 = 40 * v6;
  v10 = v5 + 40 * v6;
  if ( v10 == a2 )
  {
    if ( (unsigned int)v6 >= (unsigned int)v7 )
    {
      v28 = a3;
      sub_38E8F60(a1, 0);
      a3 = v28;
      LODWORD(v8) = *(_DWORD *)(a1 + 8);
      a2 = *(_QWORD *)a1 + 40LL * (unsigned int)v8;
    }
    if ( a2 )
    {
      v23 = _mm_loadu_si128((const __m128i *)(a3 + 8));
      *(_DWORD *)a2 = *(_DWORD *)a3;
      *(__m128i *)(a2 + 8) = v23;
      v24 = *(_DWORD *)(a3 + 32);
      *(_DWORD *)(a2 + 32) = v24;
      if ( v24 > 0x40 )
        sub_16A4FD0(a2 + 24, (const void **)(a3 + 24));
      else
        *(_QWORD *)(a2 + 24) = *(_QWORD *)(a3 + 24);
      LODWORD(v8) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + 1;
  }
  else
  {
    if ( v6 >= v7 )
    {
      v19 = a2 - v5;
      v27 = a3;
      sub_38E8F60(a1, 0);
      v5 = *(_QWORD *)a1;
      a3 = v27;
      v8 = *(unsigned int *)(a1 + 8);
      a2 = *(_QWORD *)a1 + v19;
      v9 = 40 * v8;
      v10 = *(_QWORD *)a1 + 40 * v8;
    }
    v11 = v5 + v9 - 40;
    if ( v10 )
    {
      v12 = _mm_loadu_si128((const __m128i *)(v11 + 8));
      *(_DWORD *)v10 = *(_DWORD *)v11;
      *(__m128i *)(v10 + 8) = v12;
      v13 = *(_DWORD *)(v11 + 32);
      *(_DWORD *)(v11 + 32) = 0;
      *(_DWORD *)(v10 + 32) = v13;
      *(_QWORD *)(v10 + 24) = *(_QWORD *)(v11 + 24);
      v8 = *(unsigned int *)(a1 + 8);
      v10 = *(_QWORD *)a1 + 40 * v8;
      v11 = v10 - 40;
    }
    v14 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v11 - a2) >> 3);
    if ( (__int64)(v11 - a2) > 0 )
    {
      do
      {
        v15 = *(_DWORD *)(v11 - 40);
        v10 -= 40;
        v11 -= 40;
        v16 = *(_DWORD *)(v10 + 32) <= 0x40u;
        *(_DWORD *)v10 = v15;
        *(__m128i *)(v10 + 8) = _mm_loadu_si128((const __m128i *)(v11 + 8));
        if ( !v16 )
        {
          v17 = *(_QWORD *)(v10 + 24);
          if ( v17 )
          {
            v26 = a3;
            j_j___libc_free_0_0(v17);
            a3 = v26;
          }
        }
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v11 + 24);
        *(_DWORD *)(v10 + 32) = *(_DWORD *)(v11 + 32);
        *(_DWORD *)(v11 + 32) = 0;
        --v14;
      }
      while ( v14 );
      LODWORD(v8) = *(_DWORD *)(a1 + 8);
    }
    v18 = (unsigned int)(v8 + 1);
    *(_DWORD *)(a1 + 8) = v18;
    if ( a3 >= a2 && a3 < *(_QWORD *)a1 + 40 * v18 )
      a3 += 40LL;
    v16 = *(_DWORD *)(a2 + 32) <= 0x40u;
    *(_DWORD *)a2 = *(_DWORD *)a3;
    *(__m128i *)(a2 + 8) = _mm_loadu_si128((const __m128i *)(a3 + 8));
    if ( v16 && *(_DWORD *)(a3 + 32) <= 0x40u )
    {
      v20 = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(a2 + 24) = v20;
      v21 = *(unsigned int *)(a3 + 32);
      *(_DWORD *)(a2 + 32) = v21;
      v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
      if ( (unsigned int)v21 > 0x40 )
      {
        v25 = (unsigned int)((unsigned __int64)(v21 + 63) >> 6) - 1;
        *(_QWORD *)(v20 + 8 * v25) &= v22;
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v22 & v20;
      }
    }
    else
    {
      sub_16A51C0(a2 + 24, a3 + 24);
    }
  }
}
