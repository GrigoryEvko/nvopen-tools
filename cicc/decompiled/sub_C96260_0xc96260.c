// Function: sub_C96260
// Address: 0xc96260
//
void __fastcall sub_C96260(__int64 a1, __m128i *a2)
{
  __m128i *v2; // r15
  __int64 v4; // rdx
  __m128i *v5; // rax
  __m128i *p_src; // r9
  __m128i *v7; // rdi
  __int64 v8; // rax
  size_t v9; // rdx
  __m128i v10; // xmm0
  unsigned __int64 v11; // r13
  __m128i *i; // rbx
  __int64 v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // rax
  __m128i *v16; // r12
  size_t v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __m128i *v21; // rdi
  __int64 v22; // r10
  __int64 v23; // rsi
  __m128i *v24; // rdi
  __m128i *v25; // [rsp+0h] [rbp-70h]
  __m128i *v26; // [rsp+0h] [rbp-70h]
  __m128i *v28; // [rsp+10h] [rbp-60h]
  size_t n; // [rsp+18h] [rbp-58h]
  __m128i src; // [rsp+20h] [rbp-50h] BYREF
  __m128i v31; // [rsp+30h] [rbp-40h]

  if ( (__m128i *)a1 != a2 )
  {
    v2 = (__m128i *)(a1 + 48);
    if ( a2 != (__m128i *)(a1 + 48) )
    {
      while ( 1 )
      {
        v4 = v2[2].m128i_i64[1];
        v5 = v2;
        v2 += 3;
        if ( *(_QWORD *)(a1 + 40) < v4 )
          break;
        sub_C96010(v5);
LABEL_27:
        if ( a2 == v2 )
          return;
      }
      p_src = &src;
      v7 = v2 - 2;
      v28 = &src;
      if ( (__m128i *)v2[-3].m128i_i64[0] == &v2[-2] )
      {
        src = _mm_loadu_si128(v2 - 2);
      }
      else
      {
        v28 = (__m128i *)v2[-3].m128i_i64[0];
        src.m128i_i64[0] = v2[-2].m128i_i64[0];
      }
      v8 = (__int64)v5->m128i_i64 - a1;
      v9 = v2[-3].m128i_u64[1];
      v10 = _mm_loadu_si128(v2 - 1);
      v2[-3].m128i_i64[0] = (__int64)v7;
      n = v9;
      v2[-3].m128i_i64[1] = 0;
      v11 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 4);
      v2[-2].m128i_i8[0] = 0;
      v31 = v10;
      if ( v8 > 0 )
      {
        for ( i = v2 - 5; ; v7 = (__m128i *)i[2].m128i_i64[0] )
        {
          v16 = (__m128i *)i[-1].m128i_i64[0];
          if ( i == v16 )
          {
            v17 = i[-1].m128i_u64[1];
            if ( v17 )
            {
              if ( v17 == 1 )
              {
                v7->m128i_i8[0] = i->m128i_i8[0];
              }
              else
              {
                v25 = p_src;
                memcpy(v7, i, v17);
                p_src = v25;
              }
            }
            v18 = v16[-1].m128i_i64[1];
            v19 = v16[2].m128i_i64[0];
            v16[2].m128i_i64[1] = v18;
            *(_BYTE *)(v19 + v18) = 0;
          }
          else
          {
            if ( v7 == &i[3] )
            {
              v20 = i[-1].m128i_i64[1];
              i[2].m128i_i64[0] = (__int64)v16;
              i[2].m128i_i64[1] = v20;
              i[3].m128i_i64[0] = i->m128i_i64[0];
            }
            else
            {
              v13 = i[-1].m128i_i64[1];
              v14 = i[3].m128i_i64[0];
              i[2].m128i_i64[0] = (__int64)v16;
              i[2].m128i_i64[1] = v13;
              i[3].m128i_i64[0] = i->m128i_i64[0];
              if ( v7 )
              {
                i[-1].m128i_i64[0] = (__int64)v7;
                i->m128i_i64[0] = v14;
                goto LABEL_11;
              }
            }
            i[-1].m128i_i64[0] = (__int64)i;
          }
LABEL_11:
          v15 = (_BYTE *)i[-1].m128i_i64[0];
          i[-1].m128i_i64[1] = 0;
          i -= 3;
          *v15 = 0;
          i[7].m128i_i64[0] = i[4].m128i_i64[0];
          i[7].m128i_i64[1] = i[4].m128i_i64[1];
          if ( !--v11 )
          {
            v9 = n;
            break;
          }
        }
      }
      v21 = *(__m128i **)a1;
      if ( v28 != p_src )
      {
        v22 = src.m128i_i64[0];
        if ( v21 == (__m128i *)(a1 + 16) )
        {
          *(_QWORD *)a1 = v28;
          *(_QWORD *)(a1 + 8) = v9;
          *(_QWORD *)(a1 + 16) = v22;
        }
        else
        {
          v23 = *(_QWORD *)(a1 + 16);
          *(_QWORD *)a1 = v28;
          *(_QWORD *)(a1 + 8) = v9;
          *(_QWORD *)(a1 + 16) = v22;
          if ( v21 )
          {
            v28 = v21;
            src.m128i_i64[0] = v23;
            goto LABEL_25;
          }
        }
        v28 = p_src;
        p_src = &src;
        v21 = &src;
LABEL_25:
        v21->m128i_i8[0] = 0;
        *(__m128i *)(a1 + 32) = v31;
        if ( v28 != p_src )
          j_j___libc_free_0(v28, src.m128i_i64[0] + 1);
        goto LABEL_27;
      }
      if ( v9 )
      {
        if ( v9 == 1 )
        {
          v21->m128i_i8[0] = src.m128i_i8[0];
          v24 = *(__m128i **)a1;
          *(_QWORD *)(a1 + 8) = n;
          v24->m128i_i8[n] = 0;
          v21 = v28;
          goto LABEL_25;
        }
        v26 = p_src;
        memcpy(v21, p_src, v9);
        v9 = n;
        v21 = *(__m128i **)a1;
        p_src = v26;
      }
      *(_QWORD *)(a1 + 8) = v9;
      v21->m128i_i8[v9] = 0;
      v21 = v28;
      goto LABEL_25;
    }
  }
}
