// Function: sub_2397CE0
// Address: 0x2397ce0
//
_QWORD *__fastcall sub_2397CE0(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  const __m128i *v4; // rcx
  const __m128i *v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rbx
  __m128i *v8; // rax
  __m128i *v9; // rcx
  char *v10; // r14
  char *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rbx
  _QWORD *v14; // rax
  _QWORD *v15; // r14
  const __m128i *v16; // rcx
  unsigned __int64 v17; // rbx
  __m128i *v18; // rax
  __m128i *v19; // rcx
  char *v20; // r15
  char *v21; // r13
  __int64 v22; // rax
  _QWORD *v23; // rbx
  __int64 v24; // rsi
  const __m128i *v26; // [rsp+8h] [rbp-88h]
  const __m128i *v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  __int64 v29; // [rsp+18h] [rbp-78h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  __m128i *v31; // [rsp+28h] [rbp-68h]
  __m128i *v32; // [rsp+30h] [rbp-60h]
  __int8 *v33; // [rsp+38h] [rbp-58h]
  char *v34; // [rsp+40h] [rbp-50h]
  char *v35; // [rsp+48h] [rbp-48h]
  __int8 *i; // [rsp+50h] [rbp-40h]

  v2 = (_QWORD *)a1;
  v3 = *a2;
  v4 = (const __m128i *)a2[4];
  v31 = 0;
  v5 = (const __m128i *)a2[3];
  v32 = 0;
  v28 = v3;
  v6 = a2[1];
  v33 = 0;
  v29 = v6;
  v30 = a2[2];
  v7 = (char *)v4 - (char *)v5;
  if ( v4 == v5 )
  {
    v8 = 0;
  }
  else
  {
    if ( v7 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_42;
    a1 = (char *)v4 - (char *)v5;
    v8 = (__m128i *)sub_22077B0((char *)v4 - (char *)v5);
    v4 = (const __m128i *)a2[4];
    v5 = (const __m128i *)a2[3];
  }
  v31 = v8;
  v32 = v8;
  v33 = &v8->m128i_i8[v7];
  if ( v4 == v5 )
  {
    v9 = v8;
  }
  else
  {
    v9 = (__m128i *)((char *)v8 + (char *)v4 - (char *)v5);
    do
    {
      if ( v8 )
        *v8 = _mm_loadu_si128(v5);
      ++v8;
      ++v5;
    }
    while ( v8 != v9 );
  }
  v10 = (char *)a2[7];
  v11 = (char *)a2[6];
  v32 = v9;
  v34 = 0;
  v35 = 0;
  i = 0;
  v5 = (const __m128i *)(v10 - v11);
  if ( v10 == v11 )
  {
    v13 = 0;
  }
  else
  {
    if ( (unsigned __int64)v5 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_42;
    v26 = (const __m128i *)(v10 - v11);
    v12 = sub_22077B0(v10 - v11);
    v10 = (char *)a2[7];
    v11 = (char *)a2[6];
    v5 = v26;
    v13 = v12;
  }
  v34 = (char *)v13;
  v35 = (char *)v13;
  for ( i = &v5->m128i_i8[v13]; v10 != v11; v13 += 16 )
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = *(_QWORD *)v11;
      a2 = (__int64 *)*((_QWORD *)v11 + 1);
      *(_QWORD *)(v13 + 8) = a2;
      if ( a2 )
        sub_B96E90(v13 + 8, (__int64)a2, 1);
    }
    v11 += 16;
  }
  a1 = 80;
  v35 = (char *)v13;
  v14 = (_QWORD *)sub_22077B0(0x50u);
  v15 = v14;
  if ( v14 )
  {
    v16 = v32;
    v14[4] = 0;
    v5 = v31;
    v14[5] = 0;
    v14[6] = 0;
    *v14 = &unk_4A0AFE8;
    v14[1] = v28;
    v14[2] = v29;
    v14[3] = v30;
    v17 = (char *)v16 - (char *)v5;
    if ( v16 == v5 )
    {
      v18 = 0;
    }
    else
    {
      if ( v17 > 0x7FFFFFFFFFFFFFF0LL )
        goto LABEL_42;
      a1 = (char *)v16 - (char *)v5;
      v18 = (__m128i *)sub_22077B0((char *)v16 - (char *)v5);
      v16 = v32;
      v5 = v31;
    }
    v15[4] = v18;
    v15[5] = v18;
    v15[6] = (char *)v18 + v17;
    if ( v16 == v5 )
    {
      v19 = v18;
    }
    else
    {
      v19 = (__m128i *)((char *)v18 + (char *)v16 - (char *)v5);
      do
      {
        if ( v18 )
          *v18 = _mm_loadu_si128(v5);
        ++v18;
        ++v5;
      }
      while ( v19 != v18 );
    }
    v20 = v35;
    v21 = v34;
    v15[5] = v19;
    v15[7] = 0;
    v15[8] = 0;
    v15[9] = 0;
    v5 = (const __m128i *)(v20 - v21);
    if ( v20 == v21 )
    {
      v23 = 0;
      goto LABEL_29;
    }
    if ( (unsigned __int64)v5 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v27 = (const __m128i *)(v20 - v21);
      v22 = sub_22077B0(v20 - v21);
      v20 = v35;
      v21 = v34;
      v5 = v27;
      v23 = (_QWORD *)v22;
LABEL_29:
      v15[7] = v23;
      v15[8] = v23;
      for ( v15[9] = (char *)v5 + (_QWORD)v23; v20 != v21; v23 += 2 )
      {
        if ( v23 )
        {
          *v23 = *(_QWORD *)v21;
          v24 = *((_QWORD *)v21 + 1);
          v23[1] = v24;
          if ( v24 )
            sub_B96E90((__int64)(v23 + 1), v24, 1);
        }
        v21 += 16;
      }
      v15[8] = v23;
      goto LABEL_35;
    }
LABEL_42:
    sub_4261EA(a1, a2, v5);
  }
LABEL_35:
  *v2 = v15;
  sub_2DD06B0(&v28);
  return v2;
}
