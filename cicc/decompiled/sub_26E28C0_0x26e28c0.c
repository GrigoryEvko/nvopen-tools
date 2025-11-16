// Function: sub_26E28C0
// Address: 0x26e28c0
//
__m128i *__fastcall sub_26E28C0(__m128i *a1, __int64 a2, const __m128i **a3, const __m128i **a4, char a5)
{
  __m128i *v7; // r12
  const __m128i *v8; // rcx
  const __m128i *v9; // r8
  __m128i *v10; // rdx
  __int64 v11; // rax
  const __m128i *v12; // rax
  const __m128i *v13; // rcx
  const __m128i *v14; // r8
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  const __m128i *v17; // rdi
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  __m128i *v21; // [rsp+0h] [rbp-A0h]
  char v22; // [rsp+Ch] [rbp-94h] BYREF
  __m128i *v23; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-80h] BYREF
  const __m128i *v25; // [rsp+30h] [rbp-70h] BYREF
  const __m128i *v26; // [rsp+38h] [rbp-68h]
  __int8 *v27; // [rsp+40h] [rbp-60h]
  const __m128i *v28; // [rsp+50h] [rbp-50h] BYREF
  __m128i *v29; // [rsp+58h] [rbp-48h]
  __m128i *v30; // [rsp+60h] [rbp-40h]

  v7 = a1;
  a1->m128i_i64[0] = (__int64)a1[3].m128i_i64;
  a1->m128i_i64[1] = 1;
  a1[1].m128i_i64[0] = 0;
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i32[0] = 1065353216;
  a1[2].m128i_i64[1] = 0;
  a1[3].m128i_i64[0] = 0;
  v8 = a4[1];
  v22 = a5;
  v9 = *a4;
  v23 = a1;
  v24[0] = &v22;
  v24[1] = a2;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v10 = (__m128i *)((char *)v8 - (char *)v9);
  if ( v8 == v9 )
  {
    a1 = 0;
  }
  else
  {
    if ( (unsigned __int64)v10 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_26;
    v21 = (__m128i *)((char *)v8 - (char *)v9);
    v11 = sub_22077B0((char *)v8 - (char *)v9);
    v8 = a4[1];
    v9 = *a4;
    v10 = v21;
    a1 = (__m128i *)v11;
  }
  v10 = (__m128i *)((char *)v10 + (_QWORD)a1);
  v28 = a1;
  v29 = a1;
  v30 = v10;
  if ( v8 != v9 )
  {
    v10 = a1;
    v12 = v9;
    do
    {
      if ( v10 )
      {
        *v10 = _mm_loadu_si128(v12);
        a2 = v12[1].m128i_i64[0];
        v10[1].m128i_i64[0] = a2;
      }
      v12 = (const __m128i *)((char *)v12 + 24);
      v10 = (__m128i *)((char *)v10 + 24);
    }
    while ( v8 != v12 );
    a1 = (__m128i *)((char *)a1 + 8 * ((unsigned __int64)((char *)&v8[-2].m128i_u64[1] - (char *)v9) >> 3) + 24);
  }
  v13 = a3[1];
  v14 = *a3;
  v29 = a1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v15 = (char *)v13 - (char *)v14;
  if ( v13 != v14 )
  {
    if ( v15 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v16 = sub_22077B0((char *)v13 - (char *)v14);
      v13 = a3[1];
      v14 = *a3;
      v17 = (const __m128i *)v16;
      goto LABEL_13;
    }
LABEL_26:
    sub_4261EA(a1, a2, v10);
  }
  v17 = 0;
LABEL_13:
  v25 = v17;
  v26 = v17;
  v27 = &v17->m128i_i8[v15];
  if ( v13 != v14 )
  {
    v18 = (__m128i *)v17;
    v19 = v14;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v19);
        v18[1].m128i_i64[0] = v19[1].m128i_i64[0];
      }
      v19 = (const __m128i *)((char *)v19 + 24);
      v18 = (__m128i *)((char *)v18 + 24);
    }
    while ( v13 != v19 );
    v17 = (const __m128i *)((char *)v17 + 8 * ((unsigned __int64)((char *)&v13[-2].m128i_u64[1] - (char *)v14) >> 3)
                                        + 24);
  }
  v26 = v17;
  sub_26E2170(
    &v25,
    &v28,
    (unsigned __int8 (__fastcall *)(unsigned __int64, __int64, unsigned __int64 *))sub_26E7BE0,
    (unsigned __int64)v24,
    (void (__fastcall *)(__int64, __int64, __int64))sub_26E2B70,
    (__int64)&v23);
  if ( v25 )
    j_j___libc_free_0((unsigned __int64)v25);
  if ( v28 )
    j_j___libc_free_0((unsigned __int64)v28);
  return v7;
}
