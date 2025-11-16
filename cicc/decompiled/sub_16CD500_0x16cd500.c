// Function: sub_16CD500
// Address: 0x16cd500
//
int __fastcall sub_16CD500(const __m128i *a1)
{
  __m128i *v1; // r13
  __m128i *v2; // rax
  size_t v3; // rax
  __m128i *v4; // r12
  __m128i *v5; // rbx
  unsigned __int64 v6; // rax
  bool v7; // cf
  __m128i *v8; // rax
  __m128i *v9; // rbx
  __int64 v10; // rdx
  size_t v11; // r8
  size_t v12; // r9
  _OWORD *v13; // rdi
  size_t v14; // rdx
  __int64 v15; // rdx
  size_t v16; // rdx
  size_t v18; // [rsp+10h] [rbp-70h]
  __m128i v19; // [rsp+20h] [rbp-60h] BYREF
  void *s1; // [rsp+30h] [rbp-50h]
  size_t n; // [rsp+38h] [rbp-48h]
  _OWORD src[4]; // [rsp+40h] [rbp-40h] BYREF

  v1 = (__m128i *)&a1[2];
  v2 = (__m128i *)a1[1].m128i_i64[0];
  s1 = src;
  v19 = _mm_loadu_si128(a1);
  if ( v2 == &a1[2] )
  {
    src[0] = _mm_loadu_si128(a1 + 2);
  }
  else
  {
    s1 = v2;
    *(_QWORD *)&src[0] = a1[2].m128i_i64[0];
  }
  v3 = a1[1].m128i_u64[1];
  v4 = (__m128i *)&a1[2];
  a1[1].m128i_i64[0] = (__int64)v1;
  a1[1].m128i_i64[1] = 0;
  v5 = (__m128i *)a1;
  n = v3;
  a1[2].m128i_i8[0] = 0;
  v6 = a1[-3].m128i_u64[0];
  v7 = v19.m128i_i64[0] < v6;
  if ( v19.m128i_i64[0] == v6 )
    goto LABEL_10;
LABEL_4:
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = (__m128i *)v4[-4].m128i_i64[0];
      v9 = v4 - 3;
      v4[-2] = _mm_loadu_si128(v4 - 5);
      if ( v8 == &v4[-3] )
      {
        v16 = v4[-4].m128i_u64[1];
        if ( v16 )
        {
          if ( v16 == 1 )
            v1->m128i_i8[0] = v9->m128i_i8[0];
          else
            memcpy(v1, &v4[-3], v16);
          v16 = v4[-4].m128i_u64[1];
          v1 = (__m128i *)v9[2].m128i_i64[0];
        }
        v9[2].m128i_i64[1] = v16;
        v1->m128i_i8[v16] = 0;
        v1 = (__m128i *)v4[-4].m128i_i64[0];
        goto LABEL_9;
      }
      if ( v4 == v1 )
        break;
      v9[2].m128i_i64[0] = (__int64)v8;
      v10 = v4->m128i_i64[0];
      v4[-1].m128i_i64[1] = v4[-4].m128i_i64[1];
      v4->m128i_i64[0] = v4[-3].m128i_i64[0];
      if ( !v1 )
        goto LABEL_31;
      v4[-4].m128i_i64[0] = (__int64)v1;
      v9->m128i_i64[0] = v10;
LABEL_9:
      v4 -= 3;
      v9[-1].m128i_i64[1] = 0;
      v1->m128i_i8[0] = 0;
      v1 = (__m128i *)v9[-1].m128i_i64[0];
      v5 = v9 - 2;
      v6 = v4[-5].m128i_u64[0];
      v7 = v19.m128i_i64[0] < v6;
      if ( v19.m128i_i64[0] != v6 )
        goto LABEL_4;
LABEL_10:
      v6 = v4[-5].m128i_u64[1];
      v7 = v19.m128i_i64[1] < v6;
      if ( v19.m128i_i64[1] != v6 )
        goto LABEL_4;
      v11 = n;
      v12 = v4[-4].m128i_u64[1];
      v13 = s1;
      v14 = v12;
      if ( n <= v12 )
        v14 = n;
      if ( v14 )
      {
        v18 = v4[-4].m128i_u64[1];
        LODWORD(v6) = memcmp(s1, (const void *)v4[-4].m128i_i64[0], v14);
        v13 = s1;
        v12 = v18;
        v11 = n;
        if ( (_DWORD)v6 )
          goto LABEL_17;
      }
      v6 = v11 - v12;
      if ( (__int64)(v11 - v12) >= 0x80000000LL )
        goto LABEL_18;
      if ( (__int64)v6 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_17:
        if ( (v6 & 0x80000000) == 0LL )
        {
LABEL_18:
          *v5 = _mm_load_si128(&v19);
          if ( v13 != src )
            goto LABEL_19;
LABEL_33:
          if ( v11 )
          {
            if ( v11 == 1 )
            {
              LODWORD(v6) = LOBYTE(src[0]);
              v1->m128i_i8[0] = src[0];
            }
            else
            {
              LODWORD(v6) = (unsigned int)memcpy(v1, src, v11);
            }
            v11 = n;
            v1 = (__m128i *)v5[1].m128i_i64[0];
          }
          v5[1].m128i_i64[1] = v11;
          v1->m128i_i8[v11] = 0;
          v1 = (__m128i *)s1;
          goto LABEL_22;
        }
      }
    }
    v9[2].m128i_i64[0] = (__int64)v8;
    v4[-1].m128i_i64[1] = v4[-4].m128i_i64[1];
    v4->m128i_i64[0] = v4[-3].m128i_i64[0];
LABEL_31:
    v4[-4].m128i_i64[0] = (__int64)v4[-3].m128i_i64;
    v1 = v4 - 3;
    goto LABEL_9;
  }
  v13 = s1;
  v11 = n;
  *v5 = _mm_load_si128(&v19);
  if ( v13 == src )
    goto LABEL_33;
LABEL_19:
  v6 = *(_QWORD *)&src[0];
  if ( v4 == v1 )
  {
    v5[1].m128i_i64[0] = (__int64)v13;
    v5[1].m128i_i64[1] = v11;
    v5[2].m128i_i64[0] = v6;
  }
  else
  {
    v15 = v5[2].m128i_i64[0];
    v5[1].m128i_i64[0] = (__int64)v13;
    v5[1].m128i_i64[1] = v11;
    v5[2].m128i_i64[0] = v6;
    if ( v1 )
    {
      s1 = v1;
      *(_QWORD *)&src[0] = v15;
      goto LABEL_22;
    }
  }
  s1 = src;
  v1 = (__m128i *)src;
LABEL_22:
  n = 0;
  v1->m128i_i8[0] = 0;
  if ( s1 != src )
    LODWORD(v6) = j_j___libc_free_0(s1, *(_QWORD *)&src[0] + 1LL);
  return v6;
}
