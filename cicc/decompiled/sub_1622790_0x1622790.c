// Function: sub_1622790
// Address: 0x1622790
//
__int64 __fastcall sub_1622790(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i *v7; // r14
  unsigned int *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  int *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 v16; // rax

  v2 = *a1;
  v3 = *(unsigned int *)(a2 + 8);
  v4 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( v4 != *a1 )
  {
    do
    {
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v3 )
      {
        sub_16CD150(a2, a2 + 16, 0, 16);
        LODWORD(v3) = *(_DWORD *)(a2 + 8);
      }
      v5 = *(_QWORD *)a2 + 16LL * (unsigned int)v3;
      if ( v5 )
      {
        *(_DWORD *)v5 = *(_DWORD *)v2;
        *(_QWORD *)(v5 + 8) = *(_QWORD *)(v2 + 8);
        LODWORD(v3) = *(_DWORD *)(a2 + 8);
      }
      v3 = (unsigned int)(v3 + 1);
      v2 += 16;
      *(_DWORD *)(a2 + 8) = v3;
    }
    while ( v4 != v2 );
  }
  v6 = 16 * v3;
  v7 = *(__m128i **)a2;
  v8 = (unsigned int *)(*(_QWORD *)a2 + v6);
  v9 = v6 >> 4;
  if ( v6 )
  {
    while ( 1 )
    {
      v10 = 4 * v9;
      v11 = sub_2207800(16 * v9, &unk_435FF63);
      v12 = (int *)v11;
      if ( v11 )
        break;
      v9 >>= 1;
      if ( !v9 )
        goto LABEL_15;
    }
    v13 = v11 + v10 * 4;
    v14 = v11 + 16;
    *(__m128i *)(v14 - 16) = _mm_loadu_si128(v7);
    if ( v13 == v14 )
    {
      v16 = (__int64)v12;
    }
    else
    {
      do
      {
        v15 = _mm_loadu_si128((const __m128i *)(v14 - 16));
        v14 += 16;
        *(__m128i *)(v14 - 16) = v15;
      }
      while ( v13 != v14 );
      v16 = (__int64)&v12[v10 - 4];
    }
    v7->m128i_i32[0] = *(_DWORD *)v16;
    v7->m128i_i64[1] = *(_QWORD *)(v16 + 8);
    sub_16226C0(v7->m128i_i32, v8, v12, v9);
  }
  else
  {
LABEL_15:
    v10 = 0;
    v12 = 0;
    sub_161DEF0(v7->m128i_i8, v8);
  }
  return j_j___libc_free_0(v12, v10 * 4);
}
