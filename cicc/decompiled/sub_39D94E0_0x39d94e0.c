// Function: sub_39D94E0
// Address: 0x39d94e0
//
__int64 __fastcall sub_39D94E0(unsigned __int64 a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v3; // rcx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int8 *v10; // rbx
  bool v11; // zf
  char *v12; // rbx
  char *v13; // r15
  __int64 v14; // rax
  __m128i v15; // xmm3
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __m128i *v20; // rbx
  __int64 v21; // r12
  const __m128i *v22; // r15
  __int64 v23; // r12
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r14
  __m128i *v26; // r12
  const __m128i *v27; // rax
  __m128i *v28; // rdx
  __int64 v29; // rcx
  __m128i v30; // xmm0
  __int64 v31; // rcx
  __int64 result; // rax
  __int64 v33; // [rsp+0h] [rbp-60h]
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  const __m128i *v38; // [rsp+28h] [rbp-38h]

  v3 = 0x2AAAAAAAAAAAAAALL;
  v34 = (_QWORD *)a1;
  v38 = *(const __m128i **)(a1 + 8);
  v35 = *(_QWORD *)a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v38->m128i_i64 - *(_QWORD *)a1) >> 4);
  if ( v5 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v38->m128i_i64 - *(_QWORD *)a1) >> 4);
  v8 = __CFADD__(v6, v5);
  v9 = v6 - 0x5555555555555555LL * (((__int64)v38->m128i_i64 - *(_QWORD *)a1) >> 4);
  v33 = v9;
  v10 = &a2->m128i_i8[-v35];
  if ( v8 )
  {
    a1 = 0x7FFFFFFFFFFFFFE0LL;
    v33 = 0x2AAAAAAAAAAAAAALL;
  }
  else
  {
    if ( !v9 )
    {
      v36 = 0;
      goto LABEL_7;
    }
    if ( v9 <= 0x2AAAAAAAAAAAAAALL )
      v3 = v6 - 0x5555555555555555LL * (((__int64)v38->m128i_i64 - *(_QWORD *)a1) >> 4);
    v33 = v3;
    a1 = 48 * v3;
  }
  v36 = sub_22077B0(a1);
LABEL_7:
  v11 = &v10[v36] == 0;
  v12 = &v10[v36];
  v13 = v12;
  if ( !v11 )
  {
    v14 = a3[1].m128i_i64[0];
    v15 = _mm_loadu_si128(a3);
    *((_QWORD *)v12 + 3) = 0;
    v16 = a3[1].m128i_i64[1];
    *((_QWORD *)v12 + 4) = 0;
    *((_QWORD *)v12 + 2) = v14;
    v17 = a3[2].m128i_i64[0];
    *((_QWORD *)v12 + 5) = 0;
    *(__m128i *)v12 = v15;
    v18 = v17 - v16;
    if ( v17 == v16 )
    {
      v20 = 0;
    }
    else
    {
      if ( v18 > 0x7FFFFFFFFFFFFFE0LL )
        sub_4261EA(a1, v18, v16);
      v19 = sub_22077B0(v18);
      v16 = a3[1].m128i_i64[1];
      v20 = (__m128i *)v19;
      v17 = a3[2].m128i_i64[0];
    }
    *((_QWORD *)v13 + 3) = v20;
    *((_QWORD *)v13 + 4) = v20;
    *((_QWORD *)v13 + 5) = (char *)v20 + v18;
    if ( v17 != v16 )
    {
      v21 = v16;
      do
      {
        if ( v20 )
        {
          v37 = v17;
          v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
          sub_39CF630(v20->m128i_i64, *(_BYTE **)v21, *(_QWORD *)v21 + *(_QWORD *)(v21 + 8));
          v17 = v37;
          v20[2] = _mm_loadu_si128((const __m128i *)(v21 + 32));
        }
        v21 += 48;
        v20 += 3;
      }
      while ( v17 != v21 );
    }
    *((_QWORD *)v13 + 4) = v20;
  }
  v22 = (const __m128i *)v35;
  v23 = v36;
  while ( v22 != a2 )
  {
    while ( v23 )
    {
      *(__m128i *)v23 = _mm_loadu_si128(v22);
      *(_QWORD *)(v23 + 16) = v22[1].m128i_i64[0];
      *(_QWORD *)(v23 + 24) = v22[1].m128i_i64[1];
      *(_QWORD *)(v23 + 32) = v22[2].m128i_i64[0];
      *(_QWORD *)(v23 + 40) = v22[2].m128i_i64[1];
      v22[2].m128i_i64[1] = 0;
      v22[2].m128i_i64[0] = 0;
      v22[1].m128i_i64[1] = 0;
LABEL_20:
      v22 += 3;
      v23 += 48;
      if ( v22 == a2 )
        goto LABEL_29;
    }
    v24 = (unsigned __int64 *)v22[2].m128i_i64[0];
    v25 = (unsigned __int64 *)v22[1].m128i_i64[1];
    if ( v24 != v25 )
    {
      do
      {
        if ( (unsigned __int64 *)*v25 != v25 + 2 )
          j_j___libc_free_0(*v25);
        v25 += 6;
      }
      while ( v24 != v25 );
      v25 = (unsigned __int64 *)v22[1].m128i_i64[1];
    }
    if ( !v25 )
      goto LABEL_20;
    v22 += 3;
    v23 = 48;
    j_j___libc_free_0((unsigned __int64)v25);
  }
LABEL_29:
  v26 = (__m128i *)(v23 + 48);
  if ( a2 != v38 )
  {
    v27 = a2;
    v28 = v26;
    do
    {
      v29 = v27[1].m128i_i64[0];
      v30 = _mm_loadu_si128(v27);
      v28 += 3;
      v27 += 3;
      v28[-2].m128i_i64[0] = v29;
      v31 = v27[-2].m128i_i64[1];
      v28[-3] = v30;
      v28[-2].m128i_i64[1] = v31;
      v28[-1].m128i_i64[0] = v27[-1].m128i_i64[0];
      v28[-1].m128i_i64[1] = v27[-1].m128i_i64[1];
    }
    while ( v27 != v38 );
    v26 += 3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v27 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
         + 3;
  }
  if ( v35 )
    j_j___libc_free_0(v35);
  *v34 = v36;
  result = v36 + 48 * v33;
  v34[1] = v26;
  v34[2] = result;
  return result;
}
