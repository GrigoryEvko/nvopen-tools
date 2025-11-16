// Function: sub_29B9D00
// Address: 0x29b9d00
//
unsigned __int64 **__fastcall sub_29B9D00(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 **result; // rax
  unsigned __int64 **v4; // r15
  unsigned __int64 *v6; // rcx
  __int64 v7; // rsi
  unsigned __int64 **v8; // rdx
  unsigned __int64 *v9; // r13
  unsigned __int64 *v10; // r12
  _BYTE *v11; // r8
  _BYTE *v12; // r9
  _BYTE *v13; // r10
  signed __int64 v14; // r8
  unsigned __int64 **v15; // rdx
  char *v16; // rdi
  unsigned __int64 *v17; // rdx
  unsigned __int64 **v18; // rsi
  __m128i *v19; // rsi
  __m128i *v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 *v22; // rdi
  _BYTE *v23; // r11
  size_t v24; // rsi
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  bool v27; // cf
  unsigned __int64 v28; // rax
  const void *v29; // r10
  unsigned __int64 v30; // r11
  size_t v31; // rax
  size_t v32; // rdx
  char *v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  _BYTE *v36; // rsi
  size_t n; // [rsp+10h] [rbp-90h]
  void *v38; // [rsp+18h] [rbp-88h]
  void *src; // [rsp+20h] [rbp-80h]
  void *srca; // [rsp+20h] [rbp-80h]
  void *srcb; // [rsp+20h] [rbp-80h]
  void *srcc; // [rsp+20h] [rbp-80h]
  unsigned __int64 v43; // [rsp+28h] [rbp-78h]
  size_t v44; // [rsp+28h] [rbp-78h]
  const void *v45; // [rsp+28h] [rbp-78h]
  _BYTE *v46; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v47; // [rsp+30h] [rbp-70h]
  char *v48; // [rsp+40h] [rbp-60h]
  _BYTE *v49; // [rsp+40h] [rbp-60h]
  char *v50; // [rsp+48h] [rbp-58h]
  char *v51; // [rsp+48h] [rbp-58h]
  char *v52; // [rsp+48h] [rbp-58h]
  signed __int64 desta; // [rsp+50h] [rbp-50h]
  char *dest; // [rsp+50h] [rbp-50h]
  unsigned __int64 **v55; // [rsp+58h] [rbp-48h]
  __m128i v56; // [rsp+60h] [rbp-40h] BYREF

  result = (unsigned __int64 **)a2[8];
  v4 = (unsigned __int64 **)a2[7];
  v47 = a1 + 7;
  v55 = result;
  if ( v4 != result )
  {
    while ( 1 )
    {
LABEL_2:
      v6 = *v4;
      result = (unsigned __int64 **)a1[7];
      v7 = (__int64)a1;
      v8 = (unsigned __int64 **)a1[8];
      if ( *v4 != a2 )
        v7 = (__int64)*v4;
      if ( result == v8 )
      {
LABEL_21:
        v10 = v4[1];
LABEL_22:
        if ( a2 == (unsigned __int64 *)*v10 )
          *v10 = (unsigned __int64)a1;
        if ( a2 == (unsigned __int64 *)v10[1] )
          v10[1] = (unsigned __int64)a1;
        result = (unsigned __int64 **)v4[1];
        v56.m128i_i64[0] = v7;
        v19 = (__m128i *)a1[8];
        v56.m128i_i64[1] = (__int64)result;
        if ( v19 == (__m128i *)a1[9] )
        {
          result = (unsigned __int64 **)sub_29B79B0(v47, v19, &v56);
        }
        else
        {
          if ( v19 )
          {
            *v19 = _mm_loadu_si128(&v56);
            v19 = (__m128i *)a1[8];
          }
          a1[8] = (unsigned __int64)&v19[1];
        }
        v6 = *v4;
        if ( *v4 == a2 || v6 == a1 )
          goto LABEL_15;
        result = (unsigned __int64 **)v4[1];
        v56.m128i_i64[0] = (__int64)a1;
        v56.m128i_i64[1] = (__int64)result;
        v20 = (__m128i *)v6[8];
        if ( v20 == (__m128i *)v6[9] )
        {
          result = (unsigned __int64 **)sub_29B79B0(v6 + 7, v20, &v56);
        }
        else
        {
          if ( v20 )
          {
            *v20 = _mm_loadu_si128(&v56);
            v20 = (__m128i *)v6[8];
          }
          v6[8] = (unsigned __int64)&v20[1];
        }
        goto LABEL_14;
      }
      while ( (unsigned __int64 *)v7 != *result )
      {
        result += 2;
        if ( v8 == result )
          goto LABEL_21;
      }
      v9 = result[1];
      v10 = v4[1];
      if ( !v9 )
        goto LABEL_22;
      v11 = (_BYTE *)v10[3];
      v12 = (_BYTE *)v10[2];
      v13 = (_BYTE *)v9[3];
      if ( v11 != v12 )
        break;
LABEL_15:
      if ( a2 != v6 )
      {
        result = (unsigned __int64 **)v6[7];
        v16 = (char *)v6[8];
        while ( result != (unsigned __int64 **)v16 )
        {
          v17 = *result;
          v18 = result;
          result += 2;
          if ( a2 == v17 )
          {
            if ( result != (unsigned __int64 **)v16 )
            {
              v21 = (v16 - (char *)result) >> 4;
              if ( v16 - (char *)result > 0 )
              {
                do
                {
                  v22 = *result;
                  v18 += 2;
                  result += 2;
                  *(v18 - 2) = v22;
                  *(v18 - 1) = *(result - 1);
                  --v21;
                }
                while ( v21 );
                v16 = (char *)v6[8];
              }
            }
            v4 += 2;
            v6[8] = (unsigned __int64)(v16 - 16);
            if ( v55 != v4 )
              goto LABEL_2;
            return result;
          }
        }
      }
      v4 += 2;
      if ( v55 == v4 )
        return result;
    }
    v14 = v11 - v12;
    if ( v9[4] - (unsigned __int64)v13 >= v14 )
    {
      desta = v14;
      memmove((void *)v9[3], (const void *)v10[2], v14);
      v9[3] += desta;
      result = (unsigned __int64 **)v10[2];
      v15 = (unsigned __int64 **)v10[3];
LABEL_12:
      if ( result != v15 )
        v10[3] = (unsigned __int64)result;
LABEL_14:
      v6 = *v4;
      goto LABEL_15;
    }
    v23 = (_BYTE *)v9[2];
    v24 = v13 - v23;
    v25 = (v13 - v23) >> 3;
    if ( v14 >> 3 > 0xFFFFFFFFFFFFFFFLL - v25 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v26 = v14 >> 3;
    if ( v14 >> 3 < v25 )
      v26 = (v13 - v23) >> 3;
    v27 = __CFADD__(v25, v26);
    v28 = v25 + v26;
    if ( v27 )
    {
      v34 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v28 )
      {
        v48 = 0;
        dest = 0;
        goto LABEL_48;
      }
      if ( v28 > 0xFFFFFFFFFFFFFFFLL )
        v28 = 0xFFFFFFFFFFFFFFFLL;
      v34 = 8 * v28;
    }
    srcc = (void *)v14;
    v49 = (_BYTE *)v9[3];
    v46 = (_BYTE *)v10[2];
    v52 = (char *)v34;
    v35 = sub_22077B0(v34);
    v13 = v49;
    v23 = (_BYTE *)v9[2];
    v12 = v46;
    dest = (char *)v35;
    v36 = v49;
    v14 = (signed __int64)srcc;
    v48 = &v52[v35];
    v24 = v36 - v23;
LABEL_48:
    v50 = &dest[v24 + v14];
    if ( v13 == v23 )
    {
      srcb = v23;
      v45 = v13;
      memcpy(&dest[v24], v12, v14);
      v29 = v45;
      v32 = 0;
      v30 = (unsigned __int64)srcb;
      v31 = v9[3] - (_QWORD)v45;
      if ( (const void *)v9[3] == v45 )
      {
LABEL_51:
        v33 = &v50[v32];
        if ( !v30 )
        {
LABEL_52:
          v9[3] = (unsigned __int64)v33;
          v9[2] = (unsigned __int64)dest;
          v9[4] = (unsigned __int64)v48;
          result = (unsigned __int64 **)v10[2];
          v15 = (unsigned __int64 **)v10[3];
          goto LABEL_12;
        }
LABEL_54:
        v51 = v33;
        j_j___libc_free_0(v30);
        v33 = v51;
        goto LABEL_52;
      }
    }
    else
    {
      v38 = v13;
      v43 = (unsigned __int64)v23;
      n = v14;
      src = v12;
      memmove(dest, v23, v24);
      memcpy(&dest[v24], src, n);
      v29 = v38;
      v30 = v43;
      v31 = v9[3] - (_QWORD)v38;
      if ( v38 == (void *)v9[3] )
      {
        v33 = &v50[v31];
        goto LABEL_54;
      }
    }
    srca = (void *)v30;
    v44 = v31;
    memcpy(v50, v29, v31);
    v30 = (unsigned __int64)srca;
    v32 = v44;
    goto LABEL_51;
  }
  return result;
}
