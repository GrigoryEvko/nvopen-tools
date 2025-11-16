// Function: sub_39CFCA0
// Address: 0x39cfca0
//
void __fastcall sub_39CFCA0(unsigned __int64 *a1, unsigned __int64 a2, __int8 *a3)
{
  unsigned __int64 **v3; // r14
  unsigned __int64 *v4; // r8
  unsigned __int64 v5; // rbx
  signed __int64 v6; // rcx
  __m128i *v7; // r12
  __int64 v8; // rax
  unsigned __int64 **v9; // r15
  unsigned __int64 v10; // r13
  __int64 v11; // r14
  __m128i v12; // xmm2
  unsigned __int64 **v13; // rsi
  __int64 v14; // rdi
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  __m128i *v19; // r15
  const __m128i *v20; // r12
  unsigned __int64 v21; // rbx
  __m128i *v22; // r14
  __int64 v23; // rbx
  const __m128i *v24; // r13
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // r12
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r15
  unsigned __int64 v29; // r15
  unsigned __int64 *v30; // rbx
  unsigned __int64 *v31; // r12
  __m128i v32; // xmm3
  const __m128i *v33; // r14
  unsigned __int64 v34; // rbx
  __m128i *v35; // r15
  __int64 v36; // rbx
  const __m128i *v37; // r13
  unsigned __int64 *v38; // [rsp+8h] [rbp-58h]
  __m128i *v39; // [rsp+10h] [rbp-50h]
  signed __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v41; // [rsp+20h] [rbp-40h]
  const __m128i *v42; // [rsp+28h] [rbp-38h]

  v41 = a1;
  if ( (unsigned __int64 *)a2 != a1 )
  {
    v3 = (unsigned __int64 **)a2;
    v4 = *(unsigned __int64 **)a2;
    v5 = *a1;
    v42 = *(const __m128i **)(a2 + 8);
    v6 = (signed __int64)v42->m128i_i64 - *(_QWORD *)a2;
    v40 = v6;
    if ( a1[2] - *a1 < v6 )
    {
      if ( v6 )
      {
        if ( (unsigned __int64)v6 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_68:
          sub_4261EA(a1, a2, a3);
        a1 = (unsigned __int64 *)((char *)v42 - *(_QWORD *)a2);
        v38 = *(unsigned __int64 **)a2;
        v18 = sub_22077B0(v6);
        v4 = v38;
        v39 = (__m128i *)v18;
      }
      else
      {
        v39 = 0;
      }
      v19 = v39;
      v20 = (const __m128i *)v4;
      if ( v42 != (const __m128i *)v4 )
      {
        do
        {
          if ( v19 )
          {
            *v19 = _mm_loadu_si128(v20);
            v19[1].m128i_i64[0] = v20[1].m128i_i64[0];
            v21 = v20[2].m128i_i64[0] - v20[1].m128i_i64[1];
            v19[1].m128i_i64[1] = 0;
            v19[2].m128i_i64[0] = 0;
            v19[2].m128i_i64[1] = 0;
            if ( v21 )
            {
              if ( v21 > 0x7FFFFFFFFFFFFFE0LL )
                goto LABEL_68;
              a1 = (unsigned __int64 *)v21;
              v22 = (__m128i *)sub_22077B0(v21);
            }
            else
            {
              v21 = 0;
              v22 = 0;
            }
            v19[1].m128i_i64[1] = (__int64)v22;
            v19[2].m128i_i64[0] = (__int64)v22;
            v19[2].m128i_i64[1] = (__int64)v22->m128i_i64 + v21;
            v23 = v20[2].m128i_i64[0];
            if ( v23 != v20[1].m128i_i64[1] )
            {
              v24 = (const __m128i *)v20[1].m128i_i64[1];
              do
              {
                if ( v22 )
                {
                  a1 = (unsigned __int64 *)v22;
                  v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
                  a2 = v24->m128i_i64[0];
                  sub_39CF630(v22->m128i_i64, v24->m128i_i64[0], v24->m128i_i64[0] + v24->m128i_i64[1]);
                  v22[2] = _mm_loadu_si128(v24 + 2);
                }
                v24 += 3;
                v22 += 3;
              }
              while ( (const __m128i *)v23 != v24 );
            }
            v19[2].m128i_i64[0] = (__int64)v22;
          }
          v19 += 3;
          v20 += 3;
        }
        while ( v42 != v20 );
      }
      v25 = v41[1];
      v26 = *v41;
      if ( v25 != *v41 )
      {
        do
        {
          v27 = *(unsigned __int64 **)(v26 + 32);
          v28 = *(unsigned __int64 **)(v26 + 24);
          if ( v27 != v28 )
          {
            do
            {
              if ( (unsigned __int64 *)*v28 != v28 + 2 )
                j_j___libc_free_0(*v28);
              v28 += 6;
            }
            while ( v27 != v28 );
            v28 = *(unsigned __int64 **)(v26 + 24);
          }
          if ( v28 )
            j_j___libc_free_0((unsigned __int64)v28);
          v26 += 48LL;
        }
        while ( v25 != v26 );
        v26 = *v41;
      }
      if ( v26 )
        j_j___libc_free_0(v26);
      v17 = (unsigned __int64)v39->m128i_u64 + v40;
      *v41 = (unsigned __int64)v39;
      v41[2] = (unsigned __int64)v39->m128i_u64 + v40;
      goto LABEL_18;
    }
    v7 = (__m128i *)a1[1];
    v8 = (__int64)v7->m128i_i64 - v5;
    a3 = &v7->m128i_i8[-v5];
    if ( v6 > (unsigned __int64)v7 - v5 )
    {
      a2 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 4);
      v29 = a2;
      if ( v8 > 0 )
      {
        v30 = (unsigned __int64 *)(v5 + 24);
        v31 = v4 + 3;
        do
        {
          v32 = _mm_loadu_si128((const __m128i *)(v31 - 3));
          a2 = (unsigned __int64)v31;
          a1 = v30;
          v31 += 6;
          v30 += 6;
          *(__m128i *)(v30 - 9) = v32;
          *(v30 - 7) = *(v31 - 7);
          sub_39CF9D0((__int64)a1, (unsigned __int64 **)a2);
          --v29;
        }
        while ( v29 );
        v4 = *v3;
        v7 = (__m128i *)v41[1];
        v5 = *v41;
        v42 = (const __m128i *)v3[1];
        a3 = &v7->m128i_i8[-*v41];
      }
      v33 = (const __m128i *)&a3[(_QWORD)v4];
      v17 = v40 + v5;
      if ( (const __m128i *)&a3[(_QWORD)v4] == v42 )
        goto LABEL_18;
      do
      {
        if ( v7 )
        {
          *v7 = _mm_loadu_si128(v33);
          v7[1].m128i_i64[0] = v33[1].m128i_i64[0];
          v34 = v33[2].m128i_i64[0] - v33[1].m128i_i64[1];
          v7[1].m128i_i64[1] = 0;
          v7[2].m128i_i64[0] = 0;
          v7[2].m128i_i64[1] = 0;
          if ( v34 )
          {
            if ( v34 > 0x7FFFFFFFFFFFFFE0LL )
              goto LABEL_68;
            a1 = (unsigned __int64 *)v34;
            v35 = (__m128i *)sub_22077B0(v34);
          }
          else
          {
            v35 = 0;
          }
          v7[1].m128i_i64[1] = (__int64)v35;
          v7[2].m128i_i64[0] = (__int64)v35;
          v7[2].m128i_i64[1] = (__int64)v35->m128i_i64 + v34;
          v36 = v33[2].m128i_i64[0];
          if ( v36 != v33[1].m128i_i64[1] )
          {
            v37 = (const __m128i *)v33[1].m128i_i64[1];
            do
            {
              if ( v35 )
              {
                a1 = (unsigned __int64 *)v35;
                v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
                a2 = v37->m128i_i64[0];
                sub_39CF630(v35->m128i_i64, v37->m128i_i64[0], v37->m128i_i64[0] + v37->m128i_i64[1]);
                v35[2] = _mm_loadu_si128(v37 + 2);
              }
              v37 += 3;
              v35 += 3;
            }
            while ( (const __m128i *)v36 != v37 );
          }
          v7[2].m128i_i64[0] = (__int64)v35;
        }
        v33 += 3;
        v7 += 3;
      }
      while ( v33 != v42 );
    }
    else
    {
      if ( v6 > 0 )
      {
        v9 = (unsigned __int64 **)(v4 + 3);
        v10 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 4);
        v11 = v5 + 24;
        do
        {
          v12 = _mm_loadu_si128((const __m128i *)(v9 - 3));
          v13 = v9;
          v14 = v11;
          v9 += 6;
          v11 += 48;
          *(__m128i *)(v11 - 72) = v12;
          *(_QWORD *)(v11 - 56) = *(v9 - 7);
          sub_39CF9D0(v14, v13);
          --v10;
        }
        while ( v10 );
        v5 += v40;
      }
      for ( ; v7 != (__m128i *)v5; v5 += 48LL )
      {
        v15 = *(unsigned __int64 **)(v5 + 32);
        v16 = *(unsigned __int64 **)(v5 + 24);
        if ( v15 != v16 )
        {
          do
          {
            if ( (unsigned __int64 *)*v16 != v16 + 2 )
              j_j___libc_free_0(*v16);
            v16 += 6;
          }
          while ( v15 != v16 );
          v16 = *(unsigned __int64 **)(v5 + 24);
        }
        if ( v16 )
          j_j___libc_free_0((unsigned __int64)v16);
      }
    }
    v17 = *v41 + v40;
LABEL_18:
    v41[1] = v17;
  }
}
