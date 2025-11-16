// Function: sub_1DB49A0
// Address: 0x1db49a0
//
void __fastcall sub_1DB49A0(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r15
  const __m128i *v4; // r12
  const __m128i *v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // rbx
  int v8; // r8d
  int v9; // r9d
  __m128i *v10; // r15
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 96);
  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(const __m128i **)(v2 + 24);
  v5 = (const __m128i *)(v2 + 8);
  if ( v4 == (const __m128i *)(v2 + 8) )
  {
    LODWORD(v7) = 0;
  }
  else
  {
    v6 = *(_QWORD *)(v2 + 24);
    v7 = 0;
    do
    {
      ++v7;
      v6 = sub_220EF30(v6);
    }
    while ( v5 != (const __m128i *)v6 );
    if ( v7 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v3 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), v7 + v3, 24, v8, v9);
      v3 = *(unsigned int *)(a1 + 8);
    }
    v10 = (__m128i *)(*(_QWORD *)a1 + 24 * v3);
    do
    {
      if ( v10 )
      {
        *v10 = _mm_loadu_si128(v4 + 2);
        v10[1].m128i_i64[0] = v4[3].m128i_i64[0];
      }
      v10 = (__m128i *)((char *)v10 + 24);
      v4 = (const __m128i *)sub_220EF30(v4);
    }
    while ( v5 != v4 );
    LODWORD(v3) = *(_DWORD *)(a1 + 8);
    v2 = *(_QWORD *)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 8) = v3 + v7;
  if ( v2 )
  {
    v11 = v2;
    sub_1DB3580(*(_QWORD *)(v2 + 16));
    j_j___libc_free_0(v11, 48);
  }
}
