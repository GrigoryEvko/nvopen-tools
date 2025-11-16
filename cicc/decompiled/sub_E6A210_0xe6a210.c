// Function: sub_E6A210
// Address: 0xe6a210
//
__m128i *__fastcall sub_E6A210(_QWORD *a1, __m128i *a2)
{
  size_t v2; // rbx
  _QWORD *v3; // r14
  _QWORD *v4; // rax
  char v5; // si
  size_t v6; // r13
  size_t v7; // r12
  const void *v8; // r15
  signed __int64 v9; // rax
  size_t v10; // rdx
  _QWORD *v11; // r10
  const void *v12; // r12
  __m128i *v13; // r15
  int v14; // eax
  __int64 v16; // rdi
  __int64 v17; // rax
  size_t v18; // r14
  size_t v19; // rdx
  int v20; // eax
  unsigned int v21; // edi
  __int64 v22; // rbx
  _QWORD *v23; // [rsp+10h] [rbp-50h]
  __m128i *v25; // [rsp+20h] [rbp-40h]
  __m128i *s1; // [rsp+28h] [rbp-38h]
  _QWORD *s1a; // [rsp+28h] [rbp-38h]
  _QWORD *s1b; // [rsp+28h] [rbp-38h]

  v25 = (__m128i *)sub_22077B0(72);
  v25[2].m128i_i64[0] = (__int64)v25[3].m128i_i64;
  if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
  {
    v25[3] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v25[2].m128i_i64[0] = a2->m128i_i64[0];
    v25[3].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v2 = a2->m128i_u64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  a2->m128i_i64[1] = 0;
  v25[2].m128i_i64[1] = v2;
  v25[4].m128i_i64[0] = 0;
  a2[1].m128i_i8[0] = 0;
  v3 = (_QWORD *)a1[2];
  v23 = a1 + 1;
  if ( !v3 )
  {
    if ( v23 != (_QWORD *)a1[3] )
    {
      v3 = a1 + 1;
      v13 = (__m128i *)v25[2].m128i_i64[0];
      goto LABEL_29;
    }
    v11 = a1 + 1;
    goto LABEL_43;
  }
  s1 = (__m128i *)v25[2].m128i_i64[0];
  while ( 1 )
  {
    v6 = v3[5];
    v7 = v2;
    v8 = (const void *)v3[4];
    if ( v6 <= v2 )
      v7 = v3[5];
    if ( v7 )
    {
      LODWORD(v9) = memcmp(s1, (const void *)v3[4], v7);
      if ( (_DWORD)v9 )
        goto LABEL_13;
    }
    v9 = v2 - v6;
    if ( (__int64)(v2 - v6) >= 0x80000000LL )
      break;
    if ( v9 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_13:
      if ( (int)v9 >= 0 )
        break;
    }
    v4 = (_QWORD *)v3[2];
    v5 = 1;
    if ( !v4 )
      goto LABEL_15;
LABEL_6:
    v3 = v4;
  }
  v4 = (_QWORD *)v3[3];
  v5 = 0;
  if ( v4 )
    goto LABEL_6;
LABEL_15:
  v10 = v7;
  v11 = v3;
  v12 = v8;
  v13 = s1;
  if ( !v5 )
    goto LABEL_16;
  if ( (_QWORD *)a1[3] == v3 )
  {
    v11 = v3;
    goto LABEL_26;
  }
LABEL_29:
  v17 = sub_220EF80(v3);
  v11 = v3;
  v6 = *(_QWORD *)(v17 + 40);
  v12 = *(const void **)(v17 + 32);
  v3 = (_QWORD *)v17;
  v10 = v6;
  if ( v2 <= v6 )
    v10 = v2;
LABEL_16:
  if ( v10 )
  {
    s1a = v11;
    v14 = memcmp(v12, v13, v10);
    v11 = s1a;
    if ( v14 )
    {
LABEL_21:
      if ( v14 < 0 )
        goto LABEL_25;
      goto LABEL_22;
    }
  }
  if ( (__int64)(v6 - v2) > 0x7FFFFFFF )
  {
LABEL_22:
    if ( v13 != &v25[3] )
      j_j___libc_free_0(v13, v25[3].m128i_i64[0] + 1);
    j_j___libc_free_0(v25, 72);
    return (__m128i *)v3;
  }
  if ( (__int64)(v6 - v2) >= (__int64)0xFFFFFFFF80000000LL )
  {
    v14 = v6 - v2;
    goto LABEL_21;
  }
LABEL_25:
  if ( !v11 )
  {
    v3 = 0;
    goto LABEL_22;
  }
LABEL_26:
  v16 = 1;
  if ( v23 == v11 )
    goto LABEL_27;
  v18 = v11[5];
  v19 = v18;
  if ( v2 <= v18 )
    v19 = v2;
  if ( v19 && (s1b = v11, v20 = memcmp(v13, (const void *)v11[4], v19), v11 = s1b, (v21 = v20) != 0) )
  {
LABEL_40:
    v16 = v21 >> 31;
  }
  else
  {
    v22 = v2 - v18;
    v16 = 0;
    if ( v22 <= 0x7FFFFFFF )
    {
      if ( v22 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v21 = v22;
        goto LABEL_40;
      }
LABEL_43:
      v16 = 1;
    }
  }
LABEL_27:
  sub_220F040(v16, v25, v11, v23);
  ++a1[5];
  return v25;
}
