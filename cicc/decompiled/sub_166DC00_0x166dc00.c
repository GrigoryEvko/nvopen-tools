// Function: sub_166DC00
// Address: 0x166dc00
//
void __fastcall sub_166DC00(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  const __m128i *v8; // rdx
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // r15
  const __m128i *v11; // rax
  __m128i *v12; // rdx
  __m128i v13; // xmm0
  const __m128i *v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rdi

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v3 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v3
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v5 = a2;
  v6 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v6 >= a2 )
    v5 = v6;
  if ( v5 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v7 = malloc(48 * v5);
  if ( !v7 )
    sub_16BD1C0("Allocation failed");
  v8 = *(const __m128i **)a1;
  v9 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v9 - (_QWORD)v8;
    v11 = v8 + 2;
    v12 = (__m128i *)v7;
    do
    {
      if ( v12 )
      {
        v13 = _mm_loadu_si128(v11 - 2);
        v12[1].m128i_i64[0] = (__int64)v12[2].m128i_i64;
        *v12 = v13;
        v14 = (const __m128i *)v11[-1].m128i_i64[0];
        if ( v14 == v11 )
        {
          v12[2] = _mm_loadu_si128(v11);
        }
        else
        {
          v12[1].m128i_i64[0] = (__int64)v14;
          v12[2].m128i_i64[0] = v11->m128i_i64[0];
        }
        v12[1].m128i_i64[1] = v11[-1].m128i_i64[1];
        v11[-1].m128i_i64[0] = (__int64)v11;
        v11[-1].m128i_i64[1] = 0;
        v11->m128i_i8[0] = 0;
      }
      v12 += 3;
      v11 += 3;
    }
    while ( v12 != (__m128i *)(v7 + 16 * (3 * ((0xAAAAAAAAAAAAAABLL * ((v10 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 3)) );
    v9 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
    if ( v15 != *(_QWORD *)a1 )
    {
      do
      {
        v15 -= 48;
        v16 = *(_QWORD *)(v15 + 16);
        if ( v16 != v15 + 32 )
          j_j___libc_free_0(v16, *(_QWORD *)(v15 + 32) + 1LL);
      }
      while ( v9 != v15 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v5;
}
