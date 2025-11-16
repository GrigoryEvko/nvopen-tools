// Function: sub_A3B600
// Address: 0xa3b600
//
void __fastcall sub_A3B600(__m128i *a1)
{
  const void *v1; // r14
  __m128i *v2; // rbx
  size_t v3; // r13
  size_t v4; // r12
  size_t v5; // rdx
  int v6; // eax
  __m128i v7; // xmm0

  v1 = (const void *)a1->m128i_i64[0];
  v2 = a1;
  v3 = a1->m128i_u64[1];
  while ( 1 )
  {
    v4 = v2[-1].m128i_u64[1];
    v5 = v3;
    if ( v4 <= v3 )
      v5 = v2[-1].m128i_u64[1];
    if ( !v5 )
      break;
    v6 = memcmp(v1, (const void *)v2[-1].m128i_i64[0], v5);
    if ( !v6 )
      break;
    if ( v6 >= 0 )
      goto LABEL_8;
LABEL_10:
    v7 = _mm_loadu_si128(--v2);
    v2[1] = v7;
  }
  if ( v4 != v3 && v4 > v3 )
    goto LABEL_10;
LABEL_8:
  v2->m128i_i64[1] = v3;
  v2->m128i_i64[0] = (__int64)v1;
}
