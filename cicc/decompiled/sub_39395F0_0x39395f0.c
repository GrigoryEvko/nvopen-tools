// Function: sub_39395F0
// Address: 0x39395f0
//
const __m128i *****__fastcall sub_39395F0(const __m128i *****a1, const __m128i ***a2)
{
  __int64 v2; // rdi
  const __m128i ****v3; // rax
  __int64 v4; // rdx
  const __m128i ****v5; // r12
  char *v6; // r13
  const __m128i **v7; // rbx
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  const __m128i ***v10; // r15
  const __m128i *i; // r14
  const __m128i ***v12; // r14
  char *v13; // rbx
  const __m128i **v14; // r13
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  const __m128i ***v17; // r15
  const __m128i *j; // r14
  __m128i *v19; // rax
  const __m128i ***v22; // [rsp+18h] [rbp-38h]
  unsigned __int64 v23; // [rsp+18h] [rbp-38h]

  v2 = 48;
  v22 = a2;
  v3 = (const __m128i ****)sub_22077B0(0x30u);
  v5 = v3;
  if ( !v3 )
    goto LABEL_19;
  v6 = (char *)a2[1];
  v7 = *a2;
  *v3 = 0;
  v3[1] = 0;
  v3[2] = 0;
  v8 = v6 - (char *)v7;
  if ( v6 == (char *)v7 )
  {
    v10 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_22;
    v2 = v6 - (char *)v7;
    v9 = sub_22077B0(v6 - (char *)v7);
    v8 = v6 - (char *)v7;
    v10 = (const __m128i ***)v9;
    v6 = (char *)a2[1];
    v7 = *a2;
  }
  *v5 = v10;
  v5[1] = v10;
  for ( v5[2] = (const __m128i ***)((char *)v10 + v8); v6 != (char *)v7; v10 += 3 )
  {
    if ( v10 )
    {
      v10[1] = (const __m128i **)v10;
      *v10 = (const __m128i **)v10;
      v10[2] = 0;
      for ( i = *v7; i != (const __m128i *)v7; i = (const __m128i *)i->m128i_i64[0] )
      {
        a2 = v10;
        v2 = sub_22077B0(0x20u);
        *(__m128i *)(v2 + 16) = _mm_loadu_si128(i + 1);
        sub_2208C80((_QWORD *)v2, (__int64)v10);
        v10[2] = (const __m128i **)((char *)v10[2] + 1);
      }
    }
    v7 += 3;
  }
  v12 = v22;
  v5[1] = v10;
  v5[3] = 0;
  v13 = (char *)v22[4];
  v14 = v22[3];
  v5[4] = 0;
  v5[5] = 0;
  v15 = v13 - (char *)v14;
  if ( v13 == (char *)v14 )
  {
    v17 = 0;
    goto LABEL_13;
  }
  if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_22:
    sub_4261EA(v2, a2, v4);
  v23 = v13 - (char *)v14;
  v16 = sub_22077B0(v13 - (char *)v14);
  v13 = (char *)v12[4];
  v14 = v12[3];
  v15 = v23;
  v17 = (const __m128i ***)v16;
LABEL_13:
  v5[3] = v17;
  v5[4] = v17;
  for ( v5[5] = (const __m128i ***)((char *)v17 + v15); v13 != (char *)v14; v17 += 3 )
  {
    if ( v17 )
    {
      v17[1] = (const __m128i **)v17;
      *v17 = (const __m128i **)v17;
      v17[2] = 0;
      for ( j = *v14; v14 != (const __m128i **)j; j = (const __m128i *)j->m128i_i64[0] )
      {
        v19 = (__m128i *)sub_22077B0(0x20u);
        v19[1] = _mm_loadu_si128(j + 1);
        sub_2208C80(v19, (__int64)v17);
        v17[2] = (const __m128i **)((char *)v17[2] + 1);
      }
    }
    v14 += 3;
  }
  v5[4] = v17;
LABEL_19:
  *a1 = v5;
  return a1;
}
