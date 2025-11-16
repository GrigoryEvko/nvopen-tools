// Function: sub_2D3F0A0
// Address: 0x2d3f0a0
//
char __fastcall sub_2D3F0A0(__m128i *a1, __m128i *a2)
{
  __int32 v2; // eax
  __int32 v3; // ecx
  __int32 v4; // eax
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int32 v8; // edx
  __int64 *m128i_i64; // rbx
  __m128i *v10; // r12
  __int64 v11; // r14
  __int64 v12; // r13
  __int32 v13; // edx
  _QWORD v15[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = a2->m128i_i32[2];
  v3 = a1->m128i_i32[2];
  v15[0] = -2;
  v15[1] = -2;
  a2->m128i_i32[2] = v3 & 0xFFFFFFFE | v2 & 1;
  a1->m128i_i32[2] = v2 & 0xFFFFFFFE | a1->m128i_i32[2] & 1;
  v4 = a1->m128i_i32[3];
  a1->m128i_i32[3] = a2->m128i_i32[3];
  a2->m128i_i32[3] = v4;
  if ( (a1->m128i_i8[8] & 1) == 0 )
  {
    if ( (a2->m128i_i8[8] & 1) == 0 )
    {
      v7 = a1[1].m128i_i64[0];
      a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
      v13 = a2[1].m128i_i32[2];
      a2[1].m128i_i64[0] = v7;
      LODWORD(v7) = a1[1].m128i_i32[2];
      a1[1].m128i_i32[2] = v13;
      a2[1].m128i_i32[2] = v7;
      return v7;
    }
    v5 = a2;
    a2 = a1;
    a1 = v5;
    goto LABEL_4;
  }
  if ( (a2->m128i_i8[8] & 1) == 0 )
  {
LABEL_4:
    a2->m128i_i8[8] |= 1u;
    v6 = a2[1].m128i_i64[0];
    v7 = 16;
    v8 = a2[1].m128i_i32[2];
    do
    {
      *(__m128i *)((char *)a2 + v7) = _mm_loadu_si128((__m128i *)((char *)a1 + v7));
      v7 += 16;
    }
    while ( v7 != 80 );
    a1->m128i_i8[8] &= ~1u;
    a1[1].m128i_i64[0] = v6;
    a1[1].m128i_i32[2] = v8;
    return v7;
  }
  m128i_i64 = a1[1].m128i_i64;
  v10 = a2 + 1;
  do
  {
    v11 = *m128i_i64;
    v12 = m128i_i64[1];
    v7 = v12 & *m128i_i64;
    if ( v7 != -1 )
      LOBYTE(v7) = !sub_2D27C10(m128i_i64, v15);
    if ( v10->m128i_i64[0] != -1 || v10->m128i_i64[1] != -1 )
      LOBYTE(v7) = sub_2D27C10(v10, v15);
    *(__m128i *)m128i_i64 = _mm_loadu_si128(v10);
    v10->m128i_i64[0] = v11;
    v10->m128i_i64[1] = v12;
    m128i_i64 += 2;
    ++v10;
  }
  while ( &a2[5] != v10 );
  return v7;
}
