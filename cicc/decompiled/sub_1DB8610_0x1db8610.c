// Function: sub_1DB8610
// Address: 0x1db8610
//
__int64 __fastcall sub_1DB8610(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __m128i *v9; // r12
  unsigned __int64 v11; // r14
  __int64 v12; // r11
  unsigned int v13; // r10d
  __int64 v14; // r9
  __m128i *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // rsi
  __int64 v19; // rdx
  const __m128i *v20; // rdx
  __int64 v21; // r10
  __m128i *v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rax
  __int8 *v25; // r12
  unsigned __int64 v26; // rcx
  __m128i *i; // r13
  __int64 *v28; // rsi
  unsigned int v29; // edi
  size_t v30; // rbx
  __int64 v31; // rdx
  __m128i v32; // xmm3
  __int64 v33; // rax
  __m128i *v34; // rdi
  __int64 v35; // rdx
  unsigned __int64 *v36; // rsi
  size_t v37; // r12
  char *v38; // rax
  char *v39; // rax
  int v40; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+8h] [rbp-78h]
  unsigned __int64 v42; // [rsp+10h] [rbp-70h]
  unsigned __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 *v45; // [rsp+28h] [rbp-58h] BYREF
  __m128i v46; // [rsp+30h] [rbp-50h] BYREF
  __int64 v47; // [rsp+40h] [rbp-40h]
  char v48; // [rsp+48h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a1 + 96) )
  {
    sub_1DB82B0(a1, a2, a3, a4, a5, a6, a7, a8);
    return *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 8);
    v45 = (__int64 *)a1;
    v12 = *(_QWORD *)a1;
    v47 = a8;
    v13 = v11;
    v14 = 24 * v11;
    v46 = _mm_loadu_si128((const __m128i *)&a7);
    v15 = (__m128i *)(v12 + 24 * v11);
    v44 = v46.m128i_i64[1];
    v16 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(24 * v11) >> 3);
    if ( 24 * v11 )
    {
      v9 = (__m128i *)v12;
      LODWORD(a5) = *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)a7 >> 1) & 3;
      do
      {
        while ( 1 )
        {
          v17 = v16 >> 1;
          v18 = &v9->m128i_i64[(v16 >> 1) + (v16 & 0xFFFFFFFFFFFFFFFELL)];
          if ( (unsigned int)a5 < (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) )
            break;
          v9 = (__m128i *)(v18 + 3);
          v16 = v16 - v17 - 1;
          if ( v16 <= 0 )
            goto LABEL_9;
        }
        v16 >>= 1;
      }
      while ( v17 > 0 );
LABEL_9:
      if ( (__m128i *)v12 != v9
        && v47 == v9[-1].m128i_i64[1]
        && (unsigned int)a5 >= (*(_DWORD *)((v9[-2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                              | (unsigned int)(v9[-2].m128i_i64[1] >> 1) & 3)
        && (unsigned int)a5 <= (*(_DWORD *)((v9[-1].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                              | (unsigned int)(v9[-1].m128i_i64[0] >> 1) & 3) )
      {
        v9 = (__m128i *)((char *)v9 - 24);
        sub_1DB37E0(&v45, v9, v46.m128i_i64[1]);
        return (__int64)v9;
      }
    }
    else
    {
      v9 = (__m128i *)v12;
    }
    if ( v15 == v9 )
    {
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 12) )
      {
        sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, a5, v14);
        v9 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      }
      v32 = _mm_loadu_si128(&v46);
      v9[1].m128i_i64[0] = v47;
      *v9 = v32;
      v33 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v33;
      return *(_QWORD *)a1 + 24 * v33 - 24;
    }
    else
    {
      v19 = v9[1].m128i_i64[0];
      if ( v47 == v19
        && (v26 = v46.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL,
            a5 = (v46.m128i_i64[1] >> 1) & 3,
            (*(_DWORD *)((v9->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v9->m128i_i64[0] >> 1) & 3) <= ((unsigned int)a5 | *(_DWORD *)((v46.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24))) )
      {
        for ( i = v9; ; i = (__m128i *)((char *)i - 24) )
        {
          if ( (__m128i *)v12 == i )
          {
            v9->m128i_i64[0] = a7;
            v30 = 0;
            v31 = *v45;
            if ( (__m128i *)(*v45 + 24LL * *((unsigned int *)v45 + 2)) != v9 )
            {
              v30 = *v45 + 24LL * *((unsigned int *)v45 + 2) - (_QWORD)v9;
              v40 = a5;
              v42 = v26;
              memmove(i, v9, v30);
              v31 = *v45;
              LODWORD(a5) = v40;
              v26 = v42;
            }
            *((_DWORD *)v45 + 2) = -1431655765 * ((__int64)((__int64)i->m128i_i64 + v30 - v31) >> 3);
            goto LABEL_31;
          }
          v28 = &i[-2].m128i_i64[1];
          v29 = ((__int64)a7 >> 1) & 3 | *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24);
          if ( v29 > (*(_DWORD *)((i[-2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                    | (unsigned int)(i[-2].m128i_i64[1] >> 1) & 3) )
            break;
        }
        if ( v29 <= (*(_DWORD *)((v28[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v28[1] >> 1) & 3)
          && v19 == v28[2] )
        {
          v34 = i;
          i = (__m128i *)((char *)i - 24);
          v28[1] = v9->m128i_i64[1];
        }
        else
        {
          i->m128i_i64[0] = a7;
          v34 = (__m128i *)((char *)i + 24);
          v28[4] = v9->m128i_i64[1];
        }
        v35 = *(_QWORD *)a1;
        v36 = &v9[1].m128i_u64[1];
        v37 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8) - ((_QWORD)v9 + 24);
        if ( (unsigned __int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8)) != v36 )
        {
          v41 = a5;
          v43 = v26;
          v38 = (char *)memmove(v34, v36, v37);
          v35 = *(_QWORD *)a1;
          LODWORD(a5) = v41;
          v26 = v43;
          v34 = (__m128i *)v38;
        }
        v39 = &v34->m128i_i8[v37];
        v9 = i;
        *(_DWORD *)(a1 + 8) = -1431655765 * ((__int64)&v39[-v35] >> 3);
LABEL_31:
        if ( (*(_DWORD *)(v26 + 24) | (unsigned int)a5) > (*(_DWORD *)((v9->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                         | (unsigned int)(v9->m128i_i64[1] >> 1) & 3) )
          sub_1DB37E0(&v45, v9, v44);
      }
      else
      {
        if ( v11 >= *(unsigned int *)(a1 + 12) )
        {
          v25 = &v9->m128i_i8[-v12];
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, a5, v14);
          v12 = *(_QWORD *)a1;
          v9 = (__m128i *)&v25[*(_QWORD *)a1];
          v13 = *(_DWORD *)(a1 + 8);
          v14 = 24LL * v13;
          v15 = (__m128i *)(*(_QWORD *)a1 + v14);
        }
        v20 = (const __m128i *)(v12 + v14 - 24);
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v20);
          v15[1].m128i_i64[0] = v20[1].m128i_i64[0];
          v12 = *(_QWORD *)a1;
          v13 = *(_DWORD *)(a1 + 8);
          v14 = 24LL * v13;
          v20 = (const __m128i *)(*(_QWORD *)a1 + v14 - 24);
        }
        if ( v20 != v9 )
        {
          memmove((void *)(v12 + v14 - ((char *)v20 - (char *)v9)), v9, (char *)v20 - (char *)v9);
          v13 = *(_DWORD *)(a1 + 8);
        }
        v21 = v13 + 1;
        v22 = &v46;
        *(_DWORD *)(a1 + 8) = v21;
        if ( v9 <= &v46 && (unsigned __int64)&v46 < *(_QWORD *)a1 + 24 * v21 )
          v22 = (__m128i *)&v48;
        v23 = _mm_loadu_si128(v22);
        v24 = v22[1].m128i_i64[0];
        *v9 = v23;
        v9[1].m128i_i64[0] = v24;
      }
    }
  }
  return (__int64)v9;
}
