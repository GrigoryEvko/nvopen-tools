// Function: sub_2F08FF0
// Address: 0x2f08ff0
//
void __fastcall sub_2F08FF0(__int64 a1, unsigned __int64 **a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v5; // r14
  __int64 v6; // rax
  signed __int64 v7; // rdi
  __m128i *v8; // r13
  __int64 v9; // rdx
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // r12
  unsigned __int64 *v12; // rsi
  unsigned __int64 *v13; // rdi
  __int8 *v14; // r14
  __int64 v15; // rax
  __m128i *v16; // r14
  __m128i *i; // r12
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // r12
  unsigned __int64 *v21; // rsi
  unsigned __int64 *v22; // rdi
  const __m128i *v23; // rbx
  unsigned __int64 *v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v26; // [rsp+8h] [rbp-48h]
  signed __int64 v27; // [rsp+10h] [rbp-40h]

  if ( a2 != (unsigned __int64 **)a1 )
  {
    v2 = a2[1];
    v3 = *a2;
    v5 = *(unsigned __int64 **)a1;
    v6 = *(_QWORD *)(a1 + 16);
    v7 = (char *)v2 - (char *)*a2;
    v27 = v7;
    if ( v6 - (__int64)v5 < (unsigned __int64)v7 )
    {
      if ( v7 )
      {
        if ( (unsigned __int64)v7 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(v7, a2, v6 - (_QWORD)v5);
        v24 = a2[1];
        v15 = sub_22077B0(v7);
        v2 = v24;
        v16 = (__m128i *)v15;
      }
      else
      {
        v16 = 0;
      }
      for ( i = v16; v2 != v3; i += 3 )
      {
        if ( i )
        {
          v25 = v2;
          i->m128i_i64[0] = (__int64)i[1].m128i_i64;
          sub_2F07250(i->m128i_i64, (_BYTE *)*v3, *v3 + v3[1]);
          v2 = v25;
          i[2] = _mm_loadu_si128((const __m128i *)v3 + 2);
        }
        v3 += 6;
      }
      v18 = *(unsigned __int64 **)(a1 + 8);
      v19 = *(unsigned __int64 **)a1;
      if ( v18 != *(unsigned __int64 **)a1 )
      {
        do
        {
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            j_j___libc_free_0(*v19);
          v19 += 6;
        }
        while ( v18 != v19 );
        v19 = *(unsigned __int64 **)a1;
      }
      if ( v19 )
        j_j___libc_free_0((unsigned __int64)v19);
      *(_QWORD *)a1 = v16;
      v14 = &v16->m128i_i8[v7];
      *(_QWORD *)(a1 + 16) = v14;
      goto LABEL_13;
    }
    v8 = *(__m128i **)(a1 + 8);
    v9 = (char *)v8 - (char *)v5;
    if ( v7 > (unsigned __int64)((char *)v8 - (char *)v5) )
    {
      v20 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 4);
      if ( v9 > 0 )
      {
        do
        {
          v21 = v3;
          v22 = v5;
          v3 += 6;
          v5 += 6;
          sub_2240AE0(v22, v21);
          *((__m128i *)v5 - 1) = _mm_loadu_si128((const __m128i *)v3 - 1);
          --v20;
        }
        while ( v20 );
        v2 = a2[1];
        v3 = *a2;
        v8 = *(__m128i **)(a1 + 8);
        v5 = *(unsigned __int64 **)a1;
        v9 = (__int64)v8->m128i_i64 - *(_QWORD *)a1;
      }
      v23 = (const __m128i *)((char *)v3 + v9);
      v14 = (char *)v5 + v27;
      if ( v23 == (const __m128i *)v2 )
        goto LABEL_13;
      do
      {
        if ( v8 )
        {
          v26 = v2;
          v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
          sub_2F07250(v8->m128i_i64, v23->m128i_i64[0], v23->m128i_i64[0] + v23->m128i_i64[1]);
          v2 = v26;
          v8[2] = _mm_loadu_si128(v23 + 2);
        }
        v23 += 3;
        v8 += 3;
      }
      while ( v23 != (const __m128i *)v2 );
    }
    else
    {
      if ( v7 <= 0 )
        goto LABEL_11;
      v10 = v5;
      v11 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 4);
      do
      {
        v12 = v3;
        v13 = v10;
        v3 += 6;
        v10 += 6;
        sub_2240AE0(v13, v12);
        *((__m128i *)v10 - 1) = _mm_loadu_si128((const __m128i *)v3 - 1);
        --v11;
      }
      while ( v11 );
      v5 = (unsigned __int64 *)((char *)v5 + v27);
      while ( v8 != (__m128i *)v5 )
      {
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5);
        v5 += 6;
LABEL_11:
        ;
      }
    }
    v14 = (__int8 *)(*(_QWORD *)a1 + v27);
LABEL_13:
    *(_QWORD *)(a1 + 8) = v14;
  }
}
