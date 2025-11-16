// Function: sub_1346D40
// Address: 0x1346d40
//
int __fastcall sub_1346D40(__int64 a1, __int64 *a2)
{
  __int64 v3; // r9
  _QWORD *v4; // rdx
  __m128i *v5; // rax
  __int64 *v6; // rsi
  __m128i *v7; // rcx
  __int64 v8; // r8
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rsi
  __int64 v12; // rcx
  __m128i v14; // [rsp+0h] [rbp-70h] BYREF
  __m128i v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+20h] [rbp-50h]
  __m128i v17; // [rsp+30h] [rbp-40h] BYREF
  __m128i v18; // [rsp+40h] [rbp-30h] BYREF
  __int64 v19; // [rsp+50h] [rbp-20h]
  _BYTE v20[24]; // [rsp+58h] [rbp-18h] BYREF

  if ( pthread_mutex_trylock(&stru_4F96CA0) )
  {
    sub_130AD90((__int64)&unk_4F96C60);
    byte_4F96CC8 = 1;
  }
  ++qword_4F96C98;
  if ( a1 != qword_4F96C90 )
  {
    ++qword_4F96C88;
    qword_4F96C90 = a1;
  }
  v3 = *a2;
  v4 = a2 + 1;
  v5 = &v17;
  if ( (*a2 & 1) == 0 )
  {
    v5 = &v17;
    v6 = a2 + 1;
    v7 = &v17;
    do
    {
      v8 = *v6;
      v7 = (__m128i *)((char *)v7 + 8);
      ++v6;
      v7[-1].m128i_i64[1] = v8;
    }
    while ( v7 != (__m128i *)v20 );
    if ( v3 == *a2 )
    {
      v14 = _mm_loadu_si128(&v17);
      v16 = v19;
      v15 = _mm_loadu_si128(&v18);
    }
  }
  v9 = _mm_loadu_si128(&v14);
  v10 = _mm_loadu_si128(&v15);
  LOBYTE(v16) = 0;
  v17 = v9;
  v19 = v16;
  v18 = v10;
  v11 = (*a2)++;
  do
  {
    v12 = v5->m128i_i64[0];
    v5 = (__m128i *)((char *)v5 + 8);
    *v4++ = v12;
  }
  while ( v5 != (__m128i *)v20 );
  *a2 = v11 + 2;
  --dword_4F96DA0;
  sub_1313A20(a1);
  byte_4F96CC8 = 0;
  return pthread_mutex_unlock(&stru_4F96CA0);
}
