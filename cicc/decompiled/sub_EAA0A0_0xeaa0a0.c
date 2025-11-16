// Function: sub_EAA0A0
// Address: 0xeaa0a0
//
__int64 __fastcall sub_EAA0A0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  unsigned __int64 v15; // rbx
  __int64 v16; // r12
  __m128i v17; // xmm2
  int v18; // eax
  unsigned __int64 v19; // r13
  int v20; // eax
  bool v21; // cc
  __int64 v22; // rdi
  __int64 v23; // rax
  __m128i v25; // xmm3
  unsigned int v26; // eax
  __int64 v27; // rax
  unsigned __int64 v28; // r14
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // [rsp+8h] [rbp-38h]
  unsigned __int64 v32; // [rsp+8h] [rbp-38h]
  unsigned __int64 v33; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)a1;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(unsigned int *)(a1 + 12);
  v11 = 5 * v9;
  LODWORD(v12) = v9;
  v13 = v9 + 1;
  v14 = 8 * v11;
  v15 = v8 + v14;
  if ( a2 == v8 + v14 )
  {
    if ( v13 > v10 )
    {
      if ( v8 > a3 || v15 <= a3 )
      {
        v33 = a3;
        sub_EA9FB0(a1, v13, a3, v14, v10, a6);
        v8 = *(_QWORD *)a1;
        a3 = v33;
        v12 = *(unsigned int *)(a1 + 8);
        v15 = *(_QWORD *)a1 + 40 * v12;
      }
      else
      {
        v30 = a3 - v8;
        sub_EA9FB0(a1, v13, a3 - v8, v14, v10, a6);
        v8 = *(_QWORD *)a1;
        v12 = *(unsigned int *)(a1 + 8);
        a3 = *(_QWORD *)a1 + v30;
        v15 = *(_QWORD *)a1 + 40 * v12;
      }
    }
    if ( v15 )
    {
      v25 = _mm_loadu_si128((const __m128i *)(a3 + 8));
      *(_DWORD *)v15 = *(_DWORD *)a3;
      *(__m128i *)(v15 + 8) = v25;
      v26 = *(_DWORD *)(a3 + 32);
      *(_DWORD *)(v15 + 32) = v26;
      if ( v26 > 0x40 )
        sub_C43780(v15 + 24, (const void **)(a3 + 24));
      else
        *(_QWORD *)(v15 + 24) = *(_QWORD *)(a3 + 24);
      v8 = *(_QWORD *)a1;
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
    }
    v27 = (unsigned int)(v12 + 1);
    *(_DWORD *)(a1 + 8) = v27;
    return v8 + 40 * v27 - 40;
  }
  else
  {
    if ( v13 > v10 )
    {
      v28 = a2 - v8;
      if ( v8 > a3 || v15 <= a3 )
      {
        v32 = a3;
        sub_EA9FB0(a1, v13, a3, v14, v10, a6);
        v8 = *(_QWORD *)a1;
        a3 = v32;
        v12 = *(unsigned int *)(a1 + 8);
        a2 = *(_QWORD *)a1 + v28;
        v14 = 40 * v12;
        v15 = *(_QWORD *)a1 + 40 * v12;
      }
      else
      {
        v29 = a3 - v8;
        sub_EA9FB0(a1, v13, a3 - v8, v14, v10, a6);
        v8 = *(_QWORD *)a1;
        v12 = *(unsigned int *)(a1 + 8);
        a3 = *(_QWORD *)a1 + v29;
        a2 = *(_QWORD *)a1 + v28;
        v14 = 40 * v12;
        v15 = *(_QWORD *)a1 + 40 * v12;
      }
    }
    v16 = v8 + v14 - 40;
    if ( v15 )
    {
      v17 = _mm_loadu_si128((const __m128i *)(v16 + 8));
      *(_DWORD *)v15 = *(_DWORD *)v16;
      *(__m128i *)(v15 + 8) = v17;
      v18 = *(_DWORD *)(v16 + 32);
      *(_DWORD *)(v16 + 32) = 0;
      *(_DWORD *)(v15 + 32) = v18;
      *(_QWORD *)(v15 + 24) = *(_QWORD *)(v16 + 24);
      v8 = *(_QWORD *)a1;
      v12 = *(unsigned int *)(a1 + 8);
      v15 = *(_QWORD *)a1 + 40 * v12;
      v16 = v15 - 40;
    }
    v19 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v16 - a2) >> 3);
    if ( (__int64)(v16 - a2) > 0 )
    {
      do
      {
        v20 = *(_DWORD *)(v16 - 40);
        v15 -= 40LL;
        v16 -= 40;
        v21 = *(_DWORD *)(v15 + 32) <= 0x40u;
        *(_DWORD *)v15 = v20;
        *(__m128i *)(v15 + 8) = _mm_loadu_si128((const __m128i *)(v16 + 8));
        if ( !v21 )
        {
          v22 = *(_QWORD *)(v15 + 24);
          if ( v22 )
          {
            v31 = a3;
            j_j___libc_free_0_0(v22);
            a3 = v31;
          }
        }
        *(_QWORD *)(v15 + 24) = *(_QWORD *)(v16 + 24);
        *(_DWORD *)(v15 + 32) = *(_DWORD *)(v16 + 32);
        *(_DWORD *)(v16 + 32) = 0;
        --v19;
      }
      while ( v19 );
      v8 = *(_QWORD *)a1;
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
    }
    v23 = (unsigned int)(v12 + 1);
    *(_DWORD *)(a1 + 8) = v23;
    if ( a2 <= a3 && a3 < v8 + 40 * v23 )
      a3 += 40LL;
    v21 = *(_DWORD *)(a2 + 32) <= 0x40u;
    *(_DWORD *)a2 = *(_DWORD *)a3;
    *(__m128i *)(a2 + 8) = _mm_loadu_si128((const __m128i *)(a3 + 8));
    if ( v21 && *(_DWORD *)(a3 + 32) <= 0x40u )
    {
      *(_QWORD *)(a2 + 24) = *(_QWORD *)(a3 + 24);
      *(_DWORD *)(a2 + 32) = *(_DWORD *)(a3 + 32);
      return a2;
    }
    else
    {
      sub_C43990(a2 + 24, a3 + 24);
      return a2;
    }
  }
}
