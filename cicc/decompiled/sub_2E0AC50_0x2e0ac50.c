// Function: sub_2E0AC50
// Address: 0x2e0ac50
//
void __fastcall sub_2E0AC50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r9
  unsigned int v7; // r14d
  const __m128i *v8; // r15
  const __m128i *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rbx
  unsigned __int64 v12; // rdx
  __m128i *v13; // r14
  int v14; // ebx
  __int64 v15; // rax
  unsigned int v16; // [rsp+8h] [rbp-38h]
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 96);
  v7 = *(_DWORD *)(a1 + 8);
  v8 = *(const __m128i **)(v6 + 24);
  v9 = (const __m128i *)(v6 + 8);
  v16 = *(_DWORD *)(a1 + 12);
  if ( v8 == (const __m128i *)(v6 + 8) )
  {
    v12 = v7;
    LODWORD(v11) = 0;
    if ( v16 >= v7 )
    {
      *(_QWORD *)(a1 + 96) = 0;
      goto LABEL_14;
    }
  }
  else
  {
    v10 = *(_QWORD *)(v6 + 24);
    v11 = 0;
    do
    {
      ++v11;
      v10 = sub_220EF30(v10);
    }
    while ( v9 != (const __m128i *)v10 );
    v12 = v11 + v7;
    if ( v12 <= v16 )
    {
      v13 = (__m128i *)(*(_QWORD *)a1 + 24LL * v7);
      goto LABEL_6;
    }
  }
  sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 0x18u, a5, v6);
  v15 = *(unsigned int *)(a1 + 8);
  v13 = (__m128i *)(*(_QWORD *)a1 + 24 * v15);
  if ( v8 == v9 )
  {
    v6 = *(_QWORD *)(a1 + 96);
    v14 = v15 + v11;
    goto LABEL_10;
  }
  do
  {
LABEL_6:
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(v8 + 2);
      v13[1].m128i_i64[0] = v8[3].m128i_i64[0];
    }
    v13 = (__m128i *)((char *)v13 + 24);
    v8 = (const __m128i *)sub_220EF30((__int64)v8);
  }
  while ( v9 != v8 );
  v6 = *(_QWORD *)(a1 + 96);
  v14 = *(_DWORD *)(a1 + 8) + v11;
LABEL_10:
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)(a1 + 96) = 0;
  if ( v6 )
  {
LABEL_14:
    v17 = v6;
    sub_2E094A0(*(_QWORD *)(v6 + 16));
    j_j___libc_free_0(v17);
  }
}
