// Function: sub_30FC6E0
// Address: 0x30fc6e0
//
__int64 __fastcall sub_30FC6E0(unsigned __int64 *a1, const __m128i *a2, _QWORD *a3)
{
  __int64 v5; // rsi
  const __m128i *v6; // r12
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int8 *v10; // rcx
  bool v11; // zf
  __int64 *v12; // rcx
  __int64 *v13; // r13
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  __int64 *v16; // rdi
  _QWORD *v17; // r8
  __int64 v18; // rax
  _BYTE *v19; // rsi
  _BYTE *v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  char *v24; // rdi
  size_t v25; // r14
  char *v26; // rax
  __int64 v27; // rax
  __m128i *v28; // r14
  const __m128i *i; // r13
  __int32 v30; // edx
  __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  const __m128i *v34; // rdx
  __m128i *v35; // rcx
  const __m128i *v36; // rax
  __m128i *v37; // rdx
  __int64 v38; // rsi
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  _QWORD *v42; // [rsp+8h] [rbp-58h]
  __m128i *v43; // [rsp+8h] [rbp-58h]
  _QWORD *v44; // [rsp+8h] [rbp-58h]
  _QWORD *v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  _QWORD *v47; // [rsp+10h] [rbp-50h]
  unsigned __int64 v48; // [rsp+18h] [rbp-48h]
  __m128i *v50; // [rsp+28h] [rbp-38h]

  v5 = 0x199999999999999LL;
  v6 = (const __m128i *)a1[1];
  v7 = *a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v6->m128i_i64 - *a1) >> 4);
  if ( v8 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v6->m128i_i64 - v7) >> 4);
  v48 = v9 - 0x3333333333333333LL * ((__int64)((__int64)v6->m128i_i64 - v7) >> 4);
  v10 = &a2->m128i_i8[-v7];
  if ( __CFADD__(v9, v8) )
  {
    v40 = 0x7FFFFFFFFFFFFFD0LL;
    v48 = 0x199999999999999LL;
  }
  else
  {
    if ( v9 == 0x3333333333333333LL * ((__int64)((__int64)v6->m128i_i64 - v7) >> 4) )
    {
      v50 = 0;
      goto LABEL_7;
    }
    if ( v48 <= 0x199999999999999LL )
      v5 = v9 - 0x3333333333333333LL * ((__int64)((__int64)v6->m128i_i64 - v7) >> 4);
    v48 = v5;
    v40 = 80 * v5;
  }
  v44 = a3;
  v41 = sub_22077B0(v40);
  v10 = &a2->m128i_i8[-v7];
  a3 = v44;
  v50 = (__m128i *)v41;
LABEL_7:
  v11 = &v10[(_QWORD)v50] == 0;
  v12 = (__int64 *)&v10[(_QWORD)v50];
  v13 = v12;
  if ( !v11 )
  {
    v14 = (_BYTE *)*a3;
    v15 = a3[1];
    v16 = v12;
    *v12 = (__int64)(v12 + 2);
    v45 = a3;
    sub_30FA730(v12, v14, (__int64)&v14[v15]);
    v17 = v45;
    v13[5] = 0;
    v13[6] = 0;
    v18 = v45[4];
    v19 = (_BYTE *)v45[5];
    v13[7] = 0;
    v13[4] = v18;
    v20 = (_BYTE *)v45[6];
    v21 = v20 - v19;
    if ( v20 == v19 )
    {
      v25 = 0;
      v23 = 0;
      v24 = 0;
    }
    else
    {
      if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v16, v19, v21);
      v42 = v45;
      v46 = v45[6] - (_QWORD)v19;
      v22 = sub_22077B0(v21);
      v17 = v42;
      v23 = v46;
      v24 = (char *)v22;
      v20 = (_BYTE *)v42[6];
      v19 = (_BYTE *)v42[5];
      v25 = v20 - v19;
    }
    v13[5] = (__int64)v24;
    v13[6] = (__int64)v24;
    v13[7] = (__int64)&v24[v23];
    if ( v20 != v19 )
    {
      v47 = v17;
      v26 = (char *)memmove(v24, v19, v25);
      v17 = v47;
      v24 = v26;
    }
    v27 = v17[8];
    v13[6] = (__int64)&v24[v25];
    v13[8] = v27;
    v13[9] = v17[9];
  }
  if ( a2 == (const __m128i *)v7 )
  {
    v28 = v50;
  }
  else
  {
    v28 = v50;
    for ( i = (const __m128i *)(v7 + 16); ; i += 5 )
    {
      if ( v28 )
      {
        v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
        v34 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v34 == i )
        {
          v28[1] = _mm_loadu_si128(i);
        }
        else
        {
          v28->m128i_i64[0] = (__int64)v34;
          v28[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v28->m128i_i64[1] = i[-1].m128i_i64[1];
        v30 = i[1].m128i_i32[0];
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v28[2].m128i_i32[0] = v30;
        v28[2].m128i_i32[1] = i[1].m128i_i32[1];
        v28[2].m128i_i64[1] = i[1].m128i_i64[1];
        v28[3].m128i_i64[0] = i[2].m128i_i64[0];
        v28[3].m128i_i64[1] = i[2].m128i_i64[1];
        v31 = i[3].m128i_i64[0];
        i[2].m128i_i64[1] = 0;
        i[2].m128i_i64[0] = 0;
        i[1].m128i_i64[1] = 0;
        v28[4].m128i_i64[0] = v31;
        v28[4].m128i_i64[1] = i[3].m128i_i64[1];
      }
      v32 = i[1].m128i_u64[1];
      if ( v32 )
        j_j___libc_free_0(v32);
      v33 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v33 != i )
        j_j___libc_free_0(v33);
      v28 += 5;
      if ( a2 == &i[4] )
        break;
    }
  }
  v35 = v28 + 5;
  if ( a2 != v6 )
  {
    v36 = a2;
    v37 = v28 + 5;
    do
    {
      v37->m128i_i64[0] = (__int64)v37[1].m128i_i64;
      if ( (const __m128i *)v36->m128i_i64[0] == &v36[1] )
      {
        v37[1] = _mm_loadu_si128(v36 + 1);
      }
      else
      {
        v37->m128i_i64[0] = v36->m128i_i64[0];
        v37[1].m128i_i64[0] = v36[1].m128i_i64[0];
      }
      v38 = v36->m128i_i64[1];
      v36 += 5;
      v37 += 5;
      v37[-5].m128i_i64[1] = v38;
      v37[-3].m128i_i32[0] = v36[-3].m128i_i32[0];
      v37[-3].m128i_i32[1] = v36[-3].m128i_i32[1];
      v37[-3].m128i_i64[1] = v36[-3].m128i_i64[1];
      v37[-2].m128i_i64[0] = v36[-2].m128i_i64[0];
      v37[-2].m128i_i64[1] = v36[-2].m128i_i64[1];
      v37[-1].m128i_i64[0] = v36[-1].m128i_i64[0];
      v37[-1].m128i_i64[1] = v36[-1].m128i_i64[1];
    }
    while ( v36 != v6 );
    v35 += 5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v36 - (char *)a2 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
         + 5;
  }
  if ( v7 )
  {
    v43 = v35;
    j_j___libc_free_0(v7);
    v35 = v43;
  }
  *a1 = (unsigned __int64)v50;
  a1[1] = (unsigned __int64)v35;
  a1[2] = (unsigned __int64)&v50[5 * v48];
  return 80 * v48;
}
