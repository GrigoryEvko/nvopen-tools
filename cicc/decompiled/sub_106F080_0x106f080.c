// Function: sub_106F080
// Address: 0x106f080
//
__int64 __fastcall sub_106F080(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 v4; // rdx
  const __m128i *v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r12
  bool v10; // cf
  unsigned __int64 v11; // r12
  __int8 *v12; // rcx
  char *v13; // rcx
  __int64 v14; // rax
  __m128i v15; // xmm4
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  char *v19; // rax
  char *v20; // rdi
  const void *v21; // rsi
  __int64 v22; // rdx
  size_t v23; // rdx
  char *v24; // rax
  _BYTE *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // r12
  const __m128i *i; // r14
  const __m128i *v29; // rcx
  __int64 v30; // rdi
  const __m128i *v31; // rdi
  __int64 v32; // r8
  const __m128i *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __m128i v36; // xmm1
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 v40; // rcx
  const __m128i *v41; // rcx
  __int64 result; // rax
  __int64 v43; // rax
  __int64 *v44; // [rsp+0h] [rbp-60h]
  __int64 *v45; // [rsp+8h] [rbp-58h]
  char *v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+8h] [rbp-58h]
  __int64 *v48; // [rsp+8h] [rbp-58h]
  char *v49; // [rsp+10h] [rbp-50h]
  size_t v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  _QWORD *v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+28h] [rbp-38h]

  v4 = 0x13B13B13B13B13BLL;
  v5 = *(const __m128i **)(a1 + 8);
  v6 = *(_QWORD *)a1;
  v52 = (_QWORD *)a1;
  v7 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v5->m128i_i64 - *(_QWORD *)a1) >> 3);
  if ( v7 == 0x13B13B13B13B13BLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v5->m128i_i64 - v6) >> 3);
  v10 = __CFADD__(v7, v8);
  v11 = v7 + v8;
  v51 = v11;
  v12 = &a2->m128i_i8[-v6];
  if ( v10 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v51 = 0x13B13B13B13B13BLL;
  }
  else
  {
    if ( !v11 )
    {
      v53 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x13B13B13B13B13BLL )
      v4 = v11;
    v51 = v4;
    a1 = 104 * v4;
  }
  v48 = a3;
  v43 = sub_22077B0(a1);
  v12 = &a2->m128i_i8[-v6];
  a3 = v48;
  v53 = v43;
LABEL_7:
  v13 = &v12[v53];
  if ( v13 )
  {
    v14 = *a3;
    v15 = _mm_loadu_si128((const __m128i *)(a3 + 1));
    *((_QWORD *)v13 + 6) = 0;
    *((_QWORD *)v13 + 7) = 0;
    *(_QWORD *)v13 = v14;
    v16 = a3[3];
    *((_QWORD *)v13 + 8) = 0;
    *((_QWORD *)v13 + 3) = v16;
    LOBYTE(v16) = *((_BYTE *)a3 + 32);
    *(__m128i *)(v13 + 8) = v15;
    v13[32] = v16;
    *((_QWORD *)v13 + 5) = a3[5];
    v17 = a3[7] - a3[6];
    v18 = v17;
    if ( v17 )
    {
      if ( v17 < 0 )
        sub_4261EA(a1, a2, v4);
      v45 = a3;
      v49 = v13;
      v19 = (char *)sub_22077B0(v17);
      a3 = v45;
      v13 = v49;
      v20 = v19;
      v21 = (const void *)v45[6];
      v22 = v45[7];
      *((_QWORD *)v49 + 6) = v19;
      *((_QWORD *)v49 + 7) = v19;
      *((_QWORD *)v49 + 8) = &v19[v18];
      v23 = v22 - (_QWORD)v21;
      if ( v23 )
      {
        v44 = v45;
        v46 = v49;
        v50 = v23;
        v24 = (char *)memmove(v19, v21, v23);
        a3 = v44;
        v13 = v46;
        v23 = v50;
        v20 = v24;
      }
    }
    else
    {
      v23 = 0;
      v20 = 0;
    }
    v25 = (_BYTE *)a3[9];
    *((_QWORD *)v13 + 7) = &v20[v23];
    v26 = a3[10];
    *((_QWORD *)v13 + 9) = v13 + 88;
    sub_106E550((__int64 *)v13 + 9, v25, (__int64)&v25[v26]);
  }
  if ( a2 == (const __m128i *)v6 )
  {
    v27 = v53;
  }
  else
  {
    v27 = v53;
    for ( i = (const __m128i *)(v6 + 88); ; i = (const __m128i *)((char *)i + 104) )
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = i[-6].m128i_i64[1];
        *(__m128i *)(v27 + 8) = _mm_loadu_si128(i - 5);
        *(_QWORD *)(v27 + 24) = i[-4].m128i_i64[0];
        *(_BYTE *)(v27 + 32) = i[-4].m128i_i8[8];
        *(_QWORD *)(v27 + 40) = i[-3].m128i_i64[0];
        *(_QWORD *)(v27 + 48) = i[-3].m128i_i64[1];
        *(_QWORD *)(v27 + 56) = i[-2].m128i_i64[0];
        *(_QWORD *)(v27 + 64) = i[-2].m128i_i64[1];
        i[-2].m128i_i64[1] = 0;
        i[-2].m128i_i64[0] = 0;
        i[-3].m128i_i64[1] = 0;
        *(_QWORD *)(v27 + 72) = v27 + 88;
        v29 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v29 )
        {
          *(__m128i *)(v27 + 88) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v27 + 72) = v29;
          *(_QWORD *)(v27 + 88) = i->m128i_i64[0];
        }
        *(_QWORD *)(v27 + 80) = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v31 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i != v31 )
          j_j___libc_free_0(v31, i->m128i_i64[0] + 1);
      }
      v30 = i[-3].m128i_i64[1];
      if ( v30 )
        j_j___libc_free_0(v30, i[-2].m128i_i64[1] - v30);
      v27 += 104;
      if ( a2 == &i[1] )
        break;
    }
  }
  v32 = v27 + 104;
  if ( a2 != v5 )
  {
    v33 = a2;
    v34 = v27 + 104;
    do
    {
      v36 = _mm_loadu_si128((const __m128i *)&v33->m128i_u64[1]);
      *(_QWORD *)v34 = v33->m128i_i64[0];
      v37 = v33[1].m128i_i64[1];
      *(__m128i *)(v34 + 8) = v36;
      *(_QWORD *)(v34 + 24) = v37;
      *(_BYTE *)(v34 + 32) = v33[2].m128i_i8[0];
      *(_QWORD *)(v34 + 40) = v33[2].m128i_i64[1];
      v38 = v33[3].m128i_i64[0];
      v33[3].m128i_i64[0] = 0;
      *(_QWORD *)(v34 + 48) = v38;
      v39 = v33[3].m128i_i64[1];
      v33[3].m128i_i64[1] = 0;
      *(_QWORD *)(v34 + 56) = v39;
      v40 = v33[4].m128i_i64[0];
      v33[4].m128i_i64[0] = 0;
      *(_QWORD *)(v34 + 64) = v40;
      *(_QWORD *)(v34 + 72) = v34 + 88;
      v41 = (const __m128i *)v33[4].m128i_i64[1];
      if ( v41 == (const __m128i *)&v33[5].m128i_u64[1] )
      {
        *(__m128i *)(v34 + 88) = _mm_loadu_si128((const __m128i *)((char *)v33 + 88));
      }
      else
      {
        *(_QWORD *)(v34 + 72) = v41;
        *(_QWORD *)(v34 + 88) = v33[5].m128i_i64[1];
      }
      v35 = v33[5].m128i_i64[0];
      v33 = (const __m128i *)((char *)v33 + 104);
      v34 += 104;
      *(_QWORD *)(v34 - 24) = v35;
    }
    while ( v33 != v5 );
    v32 += 104
         * (((0xEC4EC4EC4EC4EC5LL * ((unsigned __int64)((char *)v33 - (char *)a2 - 104) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
          + 1);
  }
  if ( v6 )
  {
    v47 = v32;
    j_j___libc_free_0(v6, v52[2] - v6);
    v32 = v47;
  }
  *v52 = v53;
  result = v53 + 104 * v51;
  v52[1] = v32;
  v52[2] = result;
  return result;
}
