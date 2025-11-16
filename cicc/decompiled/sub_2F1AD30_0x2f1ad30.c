// Function: sub_2F1AD30
// Address: 0x2f1ad30
//
void __fastcall sub_2F1AD30(__int64 a1, unsigned __int64 a2)
{
  const __m128i *v2; // r15
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int8 *v10; // rax
  __int64 v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17; // [rsp+10h] [rbp-50h]
  __m128i *v18; // [rsp+18h] [rbp-48h]
  const __m128i *v19; // [rsp+28h] [rbp-38h]

  if ( !a2 )
    return;
  v2 = *(const __m128i **)a1;
  v19 = *(const __m128i **)(a1 + 8);
  v3 = (__int64)v19->m128i_i64 - *(_QWORD *)a1;
  v17 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 4);
  if ( a2 <= 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v19) >> 4) )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = a2;
    do
    {
      if ( v4 )
      {
        *(_QWORD *)(v4 + 32) = 0;
        *(_OWORD *)(v4 + 16) = 0;
        *(_QWORD *)(v4 + 40) = 0;
        *(_QWORD *)(v4 + 24) = 0;
        *(_OWORD *)v4 = 0;
      }
      v4 += 48;
      --v5;
    }
    while ( v5 );
    *(_QWORD *)(a1 + 8) = &v19[3 * a2];
    return;
  }
  if ( 0x2AAAAAAAAAAAAAALL - v17 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v6 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v19->m128i_i64 - *(_QWORD *)a1) >> 4);
  if ( a2 >= v17 )
    v6 = a2;
  v7 = __CFADD__(v17, v6);
  v8 = v17 + v6;
  if ( v7 )
  {
    v14 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_38:
    v15 = sub_22077B0(v14);
    v2 = *(const __m128i **)a1;
    v18 = (__m128i *)v15;
    v19 = *(const __m128i **)(a1 + 8);
    v16 = v15 + v14;
    goto LABEL_15;
  }
  if ( v8 )
  {
    if ( v8 > 0x2AAAAAAAAAAAAAALL )
      v8 = 0x2AAAAAAAAAAAAAALL;
    v14 = 48 * v8;
    goto LABEL_38;
  }
  v16 = 0;
  v18 = 0;
LABEL_15:
  v9 = a2;
  v10 = &v18->m128i_i8[v3];
  do
  {
    if ( v10 )
    {
      *((_QWORD *)v10 + 4) = 0;
      *((_OWORD *)v10 + 1) = 0;
      *((_QWORD *)v10 + 5) = 0;
      *((_QWORD *)v10 + 3) = 0;
      *(_OWORD *)v10 = 0;
    }
    v10 += 48;
    --v9;
  }
  while ( v9 );
  if ( v19 != v2 )
  {
    v11 = (__int64)v18;
    while ( 1 )
    {
      while ( v11 )
      {
        *(__m128i *)v11 = _mm_loadu_si128(v2);
        *(_QWORD *)(v11 + 16) = v2[1].m128i_i64[0];
        *(_QWORD *)(v11 + 24) = v2[1].m128i_i64[1];
        *(_QWORD *)(v11 + 32) = v2[2].m128i_i64[0];
        *(_QWORD *)(v11 + 40) = v2[2].m128i_i64[1];
        v2[2].m128i_i64[1] = 0;
        v2[2].m128i_i64[0] = 0;
        v2[1].m128i_i64[1] = 0;
LABEL_22:
        v2 += 3;
        v11 += 48;
        if ( v2 == v19 )
          goto LABEL_31;
      }
      v12 = (unsigned __int64 *)v2[2].m128i_i64[0];
      v13 = (unsigned __int64 *)v2[1].m128i_i64[1];
      if ( v12 != v13 )
      {
        do
        {
          if ( (unsigned __int64 *)*v13 != v13 + 2 )
            j_j___libc_free_0(*v13);
          v13 += 6;
        }
        while ( v12 != v13 );
        v13 = (unsigned __int64 *)v2[1].m128i_i64[1];
      }
      if ( !v13 )
        goto LABEL_22;
      v2 += 3;
      v11 = 48;
      j_j___libc_free_0((unsigned __int64)v13);
      if ( v2 == v19 )
      {
LABEL_31:
        v2 = *(const __m128i **)a1;
        break;
      }
    }
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  *(_QWORD *)a1 = v18;
  *(_QWORD *)(a1 + 8) = &v18[3 * v17 + 3 * a2];
  *(_QWORD *)(a1 + 16) = v16;
}
