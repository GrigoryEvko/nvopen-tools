// Function: sub_B39560
// Address: 0xb39560
//
unsigned __int64 __fastcall sub_B39560(__m128i **a1, const char *a2, __int64 a3)
{
  __m128i *v4; // r13
  unsigned __int64 result; // rax
  __int64 v6; // rcx
  size_t v7; // r14
  const void *v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rdx
  size_t v11; // r14
  char *v12; // rax
  __m128i *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  __m128i *v16; // rdi
  char *v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v22; // [rsp+28h] [rbp-48h]
  __m128i v23; // [rsp+30h] [rbp-40h] BYREF

  v4 = a1[1];
  if ( v4 == a1[2] )
    return (unsigned __int64)sub_B39060(a1, a1[1], a2, a3);
  v21 = &v23;
  result = strlen(a2);
  v6 = a3;
  v20 = result;
  v7 = result;
  if ( result > 0xF )
  {
    v15 = sub_22409D0(&v21, &v20, 0);
    v6 = a3;
    v21 = (__m128i *)v15;
    v16 = (__m128i *)v15;
    v23.m128i_i64[0] = v20;
    goto LABEL_15;
  }
  if ( result == 1 )
  {
    v23.m128i_i8[0] = *a2;
    goto LABEL_5;
  }
  if ( result )
  {
    v16 = &v23;
LABEL_15:
    v19 = v6;
    memcpy(v16, a2, v7);
    result = v20;
    v6 = v19;
    v22 = v20;
    v21->m128i_i8[v20] = 0;
    if ( v4 )
      goto LABEL_6;
    goto LABEL_16;
  }
LABEL_5:
  v22 = result;
  v23.m128i_i8[result] = 0;
  if ( v4 )
  {
LABEL_6:
    v8 = *(const void **)v6;
    v9 = *(unsigned int *)(v6 + 8);
    v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
    if ( v21 == &v23 )
    {
      v4[1] = _mm_load_si128(&v23);
    }
    else
    {
      v4->m128i_i64[0] = (__int64)v21;
      v4[1].m128i_i64[0] = v23.m128i_i64[0];
    }
    result = v22;
    v10 = 8 * v9;
    v21 = &v23;
    v4[2].m128i_i64[0] = 0;
    v11 = v10;
    v4->m128i_i64[1] = result;
    v4[2].m128i_i64[1] = 0;
    v22 = 0;
    v23.m128i_i8[0] = 0;
    v4[3].m128i_i64[0] = 0;
    if ( v10 )
    {
      v12 = (char *)sub_22077B0(v10);
      v4[2].m128i_i64[0] = (__int64)v12;
      v4[3].m128i_i64[0] = (__int64)&v12[v11];
      v18 = &v12[v11];
      result = (unsigned __int64)memcpy(v12, v8, v11);
      v13 = v21;
      v14 = (__int64)v18;
    }
    else
    {
      v13 = &v23;
      v14 = 0;
    }
    v4[2].m128i_i64[1] = v14;
    goto LABEL_11;
  }
LABEL_16:
  v13 = v21;
LABEL_11:
  if ( v13 != &v23 )
    result = j_j___libc_free_0(v13, v23.m128i_i64[0] + 1);
  a1[1] = (__m128i *)((char *)a1[1] + 56);
  return result;
}
