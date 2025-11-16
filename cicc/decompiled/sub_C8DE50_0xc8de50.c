// Function: sub_C8DE50
// Address: 0xc8de50
//
void __fastcall sub_C8DE50(__int64 a1, const __m128i *a2)
{
  const __m128i *v3; // r15
  const __m128i *v4; // r10
  const __m128i *v5; // r8
  bool v6; // cf
  const __m128i *v7; // rbx
  size_t v8; // r12
  __m128i v9; // xmm1
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __m128i *v12; // r12
  __m128i *i; // rdi
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __m128i *v16; // rax
  size_t v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  size_t v20; // rcx
  size_t v21; // rdx
  signed __int64 v22; // rax
  const __m128i *v23; // rax
  const __m128i *v24; // rax
  __m128i *v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdx
  size_t v28; // r12
  __int64 v29; // rdi
  size_t v30; // [rsp-80h] [rbp-80h]
  const __m128i *v32; // [rsp-70h] [rbp-70h]
  __m128i v33; // [rsp-68h] [rbp-68h] BYREF
  const __m128i *v34; // [rsp-58h] [rbp-58h]
  size_t v35; // [rsp-50h] [rbp-50h]
  _OWORD v36[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( (const __m128i *)a1 != a2 && a2 != (const __m128i *)(a1 + 48) )
  {
    v3 = (const __m128i *)(a1 + 80);
    while ( 1 )
    {
      v4 = v3 - 2;
      v5 = v3;
      v6 = v3[-2].m128i_i64[0] < *(_QWORD *)a1;
      if ( v3[-2].m128i_i64[0] != *(_QWORD *)a1 )
        break;
      v19 = *(_QWORD *)(a1 + 8);
      v6 = v3[-2].m128i_i64[1] < v19;
      if ( v3[-2].m128i_i64[1] != v19 )
        break;
      v8 = v3[-1].m128i_u64[1];
      v20 = *(_QWORD *)(a1 + 24);
      v7 = (const __m128i *)v3[-1].m128i_i64[0];
      v21 = v20;
      if ( v8 <= v20 )
        v21 = v3[-1].m128i_u64[1];
      if ( v21
        && (v30 = *(_QWORD *)(a1 + 24),
            LODWORD(v22) = memcmp((const void *)v3[-1].m128i_i64[0], *(const void **)(a1 + 16), v21),
            v4 = v3 - 2,
            v20 = v30,
            v5 = v3,
            (_DWORD)v22) )
      {
LABEL_29:
        if ( (int)v22 >= 0 )
          goto LABEL_30;
LABEL_7:
        v9 = _mm_loadu_si128(v3 - 2);
        v34 = (const __m128i *)v36;
        v33 = v9;
        if ( v7 == v3 )
        {
          v36[0] = _mm_loadu_si128(v3);
        }
        else
        {
          v10 = v3->m128i_i64[0];
          v34 = v7;
          *(_QWORD *)&v36[0] = v10;
        }
        v35 = v8;
        v32 = v3 + 1;
        v3[-1].m128i_i64[0] = (__int64)v3;
        v3[-1].m128i_i64[1] = 0;
        v3->m128i_i8[0] = 0;
        v11 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v4->m128i_i64 - a1) >> 4);
        if ( (__int64)v4->m128i_i64 - a1 > 0 )
        {
          v12 = (__m128i *)&v3[-3];
          for ( i = (__m128i *)v5; ; i = (__m128i *)v12[2].m128i_i64[0] )
          {
            v16 = (__m128i *)v12[-1].m128i_i64[0];
            v12[1] = _mm_loadu_si128(v12 - 2);
            if ( v16 == v12 )
            {
              v17 = v12[-1].m128i_u64[1];
              if ( v17 )
              {
                if ( v17 == 1 )
                  i->m128i_i8[0] = v12->m128i_i8[0];
                else
                  memcpy(i, v12, v17);
                v17 = v12[-1].m128i_u64[1];
              }
              v18 = v12[2].m128i_i64[0];
              v12[2].m128i_i64[1] = v17;
              *(_BYTE *)(v18 + v17) = 0;
            }
            else
            {
              if ( i == &v12[3] )
              {
                v12[2].m128i_i64[0] = (__int64)v16;
                v12[2].m128i_i64[1] = v12[-1].m128i_i64[1];
                v12[3].m128i_i64[0] = v12->m128i_i64[0];
              }
              else
              {
                v12[2].m128i_i64[0] = (__int64)v16;
                v14 = v12[3].m128i_i64[0];
                v12[2].m128i_i64[1] = v12[-1].m128i_i64[1];
                v12[3].m128i_i64[0] = v12->m128i_i64[0];
                if ( i )
                {
                  v12[-1].m128i_i64[0] = (__int64)i;
                  v12->m128i_i64[0] = v14;
                  goto LABEL_14;
                }
              }
              v12[-1].m128i_i64[0] = (__int64)v12;
            }
LABEL_14:
            v15 = (_BYTE *)v12[-1].m128i_i64[0];
            v12 -= 3;
            v12[2].m128i_i64[1] = 0;
            *v15 = 0;
            if ( !--v11 )
            {
              v8 = v35;
              break;
            }
          }
        }
        v24 = v34;
        v25 = *(__m128i **)(a1 + 16);
        *(__m128i *)a1 = _mm_load_si128(&v33);
        if ( v24 == (const __m128i *)v36 )
        {
          if ( v8 )
          {
            if ( v8 == 1 )
            {
              v25->m128i_i8[0] = v36[0];
              v28 = v35;
              v29 = *(_QWORD *)(a1 + 16);
              *(_QWORD *)(a1 + 24) = v35;
              *(_BYTE *)(v29 + v28) = 0;
              v25 = (__m128i *)v34;
              goto LABEL_39;
            }
            memcpy(v25, v36, v8);
            v8 = v35;
            v25 = *(__m128i **)(a1 + 16);
          }
          *(_QWORD *)(a1 + 24) = v8;
          v25->m128i_i8[v8] = 0;
          v25 = (__m128i *)v34;
        }
        else
        {
          v26 = *(_QWORD *)&v36[0];
          if ( v25 == (__m128i *)(a1 + 32) )
          {
            *(_QWORD *)(a1 + 16) = v24;
            *(_QWORD *)(a1 + 24) = v8;
            *(_QWORD *)(a1 + 32) = v26;
LABEL_50:
            v34 = (const __m128i *)v36;
            v25 = (__m128i *)v36;
            goto LABEL_39;
          }
          v27 = *(_QWORD *)(a1 + 32);
          *(_QWORD *)(a1 + 16) = v24;
          *(_QWORD *)(a1 + 24) = v8;
          *(_QWORD *)(a1 + 32) = v26;
          if ( !v25 )
            goto LABEL_50;
          v34 = v25;
          *(_QWORD *)&v36[0] = v27;
        }
LABEL_39:
        v35 = 0;
        v25->m128i_i8[0] = 0;
        if ( v34 != (const __m128i *)v36 )
          j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
        v3 += 3;
        if ( a2 == v32 )
          return;
      }
      else
      {
        v22 = v8 - v20;
        if ( (__int64)(v8 - v20) < 0x80000000LL )
        {
          if ( v22 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_7;
          goto LABEL_29;
        }
LABEL_30:
        sub_C8DB60(v4);
        v23 = v3 + 1;
        v3 += 3;
        if ( a2 == v23 )
          return;
      }
    }
    if ( v6 )
    {
      v7 = (const __m128i *)v3[-1].m128i_i64[0];
      v8 = v3[-1].m128i_u64[1];
      goto LABEL_7;
    }
    goto LABEL_30;
  }
}
