// Function: sub_E00EB0
// Address: 0xe00eb0
//
__m128i *__fastcall sub_E00EB0(__m128i *a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // dl
  char v14; // bl
  __int64 v15; // rdx
  char v16; // dl
  __m128i v17; // xmm1
  __int64 v19; // rax
  char v20; // dl
  __m128i v21; // xmm3
  __m128i v22; // [rsp+10h] [rbp-50h] BYREF
  __m128i v23; // [rsp+20h] [rbp-40h] BYREF

  v9 = *a2;
  if ( *a2 )
    v9 = sub_E00740(v9);
  v22.m128i_i64[0] = v9;
  v10 = a2[1];
  if ( v10 )
    v10 = sub_E00750(v10, a3);
  v11 = a2[2];
  v22.m128i_i64[1] = v10;
  v23.m128i_i64[0] = v11;
  v23.m128i_i64[1] = a2[3];
  v12 = sub_9208B0(a5, a4);
  v14 = v13;
  v15 = v12;
  if ( sub_9208B0(a5, a4) == ((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL) && v16 == v14 )
  {
    v19 = sub_9208B0(a5, a4);
    if ( v20 )
    {
      v21 = _mm_loadu_si128(&v23);
      *a1 = _mm_loadu_si128(&v22);
      a1[1] = v21;
    }
    else
    {
      sub_E00CC0(a1, &v22, (unsigned __int64)(v19 + 7) >> 3);
    }
  }
  else
  {
    v17 = _mm_loadu_si128(&v23);
    *a1 = _mm_loadu_si128(&v22);
    a1[1] = v17;
  }
  return a1;
}
