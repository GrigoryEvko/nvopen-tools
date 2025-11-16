// Function: sub_33ECD10
// Address: 0x33ecd10
//
__int64 __fastcall sub_33ECD10(unsigned __int16 a1)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  __int64 v5; // rcx
  __m128i *v6; // r12
  const __m128i *v7; // rax
  __m128i *v8; // rdx
  __int64 m128i_i64; // rax
  int v10; // ebx
  __m128i v11; // [rsp+0h] [rbp-30h] BYREF

  if ( !byte_5039410 && (unsigned int)sub_2207590((__int64)&byte_5039410) )
  {
    qword_5039420 = 0;
    qword_5039428 = 0;
    qword_5039430 = 0;
    v3 = sub_22077B0(0x1120u);
    v4 = qword_5039420;
    v5 = qword_5039428;
    v6 = (__m128i *)v3;
    v7 = (const __m128i *)qword_5039420;
    if ( qword_5039428 != qword_5039420 )
    {
      v8 = v6;
      do
      {
        if ( v8 )
          *v8 = _mm_loadu_si128(v7);
        ++v7;
        ++v8;
      }
      while ( (const __m128i *)v5 != v7 );
    }
    if ( v4 )
      j_j___libc_free_0(v4);
    m128i_i64 = (__int64)v6[274].m128i_i64;
    qword_5039420 = (__int64)v6;
    v10 = 0;
    qword_5039428 = (__int64)v6;
    for ( qword_5039430 = (__int64)v6[274].m128i_i64; ; m128i_i64 = qword_5039430 )
    {
      v11.m128i_i16[0] = v10;
      v11.m128i_i64[1] = 0;
      if ( v6 == (__m128i *)m128i_i64 )
      {
        ++v10;
        sub_33ECB90((unsigned __int64 *)&qword_5039420, v6, &v11);
        if ( v10 == 274 )
          goto LABEL_18;
      }
      else
      {
        if ( v6 )
        {
          *v6 = _mm_loadu_si128(&v11);
          v6 = (__m128i *)qword_5039428;
        }
        ++v10;
        qword_5039428 = (__int64)v6[1].m128i_i64;
        if ( v10 == 274 )
        {
LABEL_18:
          __cxa_atexit((void (*)(void *))sub_33C85E0, &qword_5039420, &qword_4A427C0);
          sub_2207640((__int64)&byte_5039410);
          return qword_5039420 + 16LL * a1;
        }
      }
      v6 = (__m128i *)qword_5039428;
    }
  }
  return qword_5039420 + 16LL * a1;
}
