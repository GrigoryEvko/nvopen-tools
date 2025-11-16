// Function: sub_29F55B0
// Address: 0x29f55b0
//
void __fastcall sub_29F55B0(__m128i *a1)
{
  __m128i *v1; // r15
  __m128i *v2; // r13
  size_t v3; // r14
  __int64 v4; // rax
  __m128i *v5; // rbx
  size_t v6; // r8
  size_t v7; // rdx
  unsigned __int64 *v8; // rcx
  __m128i **v9; // r12
  int v10; // eax
  signed __int64 v11; // rax
  signed __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  size_t v19; // [rsp+8h] [rbp-78h]
  size_t n; // [rsp+10h] [rbp-70h]
  size_t na; // [rsp+10h] [rbp-70h]
  unsigned __int64 *s1; // [rsp+18h] [rbp-68h]
  unsigned __int64 *s1a; // [rsp+18h] [rbp-68h]
  __m128i *v24; // [rsp+20h] [rbp-60h]
  size_t v25; // [rsp+28h] [rbp-58h]
  __m128i v26; // [rsp+30h] [rbp-50h] BYREF
  __m128i *v27; // [rsp+40h] [rbp-40h]

  v1 = a1 + 1;
  v2 = (__m128i *)a1->m128i_i64[0];
  v24 = &v26;
  if ( (__m128i *)a1->m128i_i64[0] == &a1[1] )
  {
    v2 = &v26;
    v26 = _mm_loadu_si128(a1 + 1);
  }
  else
  {
    v24 = (__m128i *)a1->m128i_i64[0];
    v26.m128i_i64[0] = a1[1].m128i_i64[0];
  }
  v3 = a1->m128i_u64[1];
  v4 = a1[2].m128i_i64[0];
  a1->m128i_i64[0] = (__int64)v1;
  v5 = a1 + 1;
  a1->m128i_i64[1] = 0;
  v25 = v3;
  a1[1].m128i_i8[0] = 0;
  v27 = (__m128i *)v4;
  while ( 1 )
  {
    v6 = v5[-3].m128i_u64[0];
    v7 = v3;
    v8 = (unsigned __int64 *)v5[-4].m128i_i64[1];
    v9 = (__m128i **)&v5[-1];
    if ( v6 <= v3 )
      v7 = v5[-3].m128i_u64[0];
    if ( !v7 )
    {
      v11 = v3 - v6;
      if ( (__int64)(v3 - v6) < 0x80000000LL )
        goto LABEL_9;
      goto LABEL_13;
    }
    v19 = v5[-3].m128i_u64[0];
    n = v7;
    s1 = (unsigned __int64 *)v5[-4].m128i_i64[1];
    v10 = memcmp(v2, s1, v7);
    v8 = s1;
    v7 = n;
    v6 = v19;
    if ( !v10 )
      break;
    if ( v10 >= 0 )
      goto LABEL_12;
    v15 = (__m128i *)((char *)v5 - 40);
    if ( s1 == &v5[-3].m128i_u64[1] )
    {
LABEL_33:
      if ( v6 )
      {
        if ( v6 == 1 )
          v1->m128i_i8[0] = v15->m128i_i8[0];
        else
          memcpy(v1, v15, v6);
        v6 = v15[-1].m128i_u64[1];
        v1 = (__m128i *)v15[1].m128i_i64[1];
      }
      v15[2].m128i_i64[0] = v6;
      v1->m128i_i8[v6] = 0;
      v1 = (__m128i *)v15[-1].m128i_i64[0];
      goto LABEL_28;
    }
LABEL_25:
    if ( v1 == v5 )
    {
      v18 = v15[-1].m128i_i64[1];
      v15[1].m128i_i64[1] = (__int64)v8;
      v15[2].m128i_i64[0] = v18;
      v15[2].m128i_i64[1] = v15->m128i_i64[0];
    }
    else
    {
      v16 = v15[-1].m128i_i64[1];
      v17 = v15[2].m128i_i64[1];
      v15[1].m128i_i64[1] = (__int64)v8;
      v15[2].m128i_i64[0] = v16;
      v15[2].m128i_i64[1] = v15->m128i_i64[0];
      if ( v1 )
      {
        v15[-1].m128i_i64[0] = (__int64)v1;
        v15->m128i_i64[0] = v17;
        goto LABEL_28;
      }
    }
    v15[-1].m128i_i64[0] = (__int64)v15;
    v1 = v15;
LABEL_28:
    v15[-1].m128i_i64[1] = 0;
    v5 = v15;
    v1->m128i_i8[0] = 0;
    v2 = v24;
    v3 = v25;
    v15[3].m128i_i64[1] = v15[1].m128i_i64[0];
    v1 = (__m128i *)v15[-1].m128i_i64[0];
  }
  v11 = v3 - v19;
  if ( (__int64)(v3 - v19) >= 0x80000000LL )
    goto LABEL_12;
LABEL_9:
  if ( v11 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v11 < 0 )
  {
LABEL_24:
    v15 = (__m128i *)((char *)v5 - 40);
    if ( v8 == &v5[-3].m128i_u64[1] )
      goto LABEL_33;
    goto LABEL_25;
  }
  if ( !v7 )
    goto LABEL_13;
LABEL_12:
  na = v6;
  s1a = v8;
  LODWORD(v12) = memcmp(v8, v2, v7);
  v8 = s1a;
  v6 = na;
  if ( !(_DWORD)v12 )
  {
LABEL_13:
    v12 = v6 - v3;
    if ( (__int64)(v6 - v3) >= 0x80000000LL )
      goto LABEL_23;
    if ( v12 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_16;
  }
  if ( (int)v12 < 0 )
    goto LABEL_16;
LABEL_23:
  if ( (unsigned __int64)v27 < v5[-2].m128i_i64[1] )
    goto LABEL_24;
LABEL_16:
  if ( v2 == &v26 )
  {
    if ( v3 )
    {
      if ( v3 == 1 )
        v1->m128i_i8[0] = v26.m128i_i8[0];
      else
        memcpy(v1, &v26, v3);
      v3 = v25;
      v1 = *v9;
    }
    v9[1] = (__m128i *)v3;
    v1->m128i_i8[v3] = 0;
    v1 = v24;
  }
  else
  {
    v13 = v26.m128i_i64[0];
    if ( v1 == v5 )
    {
      *v9 = v2;
      v9[1] = (__m128i *)v3;
      v5->m128i_i64[0] = v13;
    }
    else
    {
      v14 = v5->m128i_i64[0];
      *v9 = v2;
      v9[1] = (__m128i *)v3;
      v5->m128i_i64[0] = v13;
      if ( v1 )
      {
        v24 = v1;
        v26.m128i_i64[0] = v14;
        goto LABEL_20;
      }
    }
    v24 = &v26;
    v1 = &v26;
  }
LABEL_20:
  v1->m128i_i8[0] = 0;
  v9[4] = v27;
  if ( v24 != &v26 )
    j_j___libc_free_0((unsigned __int64)v24);
}
