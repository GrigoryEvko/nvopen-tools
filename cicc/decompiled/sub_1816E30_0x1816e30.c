// Function: sub_1816E30
// Address: 0x1816e30
//
_OWORD *__fastcall sub_1816E30(__int64 a1)
{
  char *v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __m128i *v8; // rax
  __int64 v9; // rcx
  _OWORD *result; // rax
  _OWORD *v11; // r13
  __int64 v12; // rcx
  __m128i *v13; // rax
  __int64 v14; // rcx
  __m128i *v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  __m128i *v19; // rax
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  _QWORD *v22; // r14
  _OWORD *v23; // rbx
  _BYTE *v24; // rdi
  size_t v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // r12
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rdx
  size_t v31; // rdx
  _QWORD *v32; // [rsp+40h] [rbp-130h] BYREF
  __int64 v33; // [rsp+48h] [rbp-128h]
  _QWORD v34[2]; // [rsp+50h] [rbp-120h] BYREF
  _BYTE *v35; // [rsp+60h] [rbp-110h] BYREF
  __int64 v36; // [rsp+68h] [rbp-108h]
  _QWORD v37[2]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE *v38; // [rsp+80h] [rbp-F0h] BYREF
  unsigned __int64 v39; // [rsp+88h] [rbp-E8h]
  _QWORD v40[2]; // [rsp+90h] [rbp-E0h] BYREF
  __m128i *v41; // [rsp+A0h] [rbp-D0h]
  unsigned __int64 v42; // [rsp+A8h] [rbp-C8h]
  __m128i v43; // [rsp+B0h] [rbp-C0h] BYREF
  _QWORD v44[2]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i *v46; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v47; // [rsp+E8h] [rbp-88h]
  __m128i v48; // [rsp+F0h] [rbp-80h] BYREF
  __m128i *v49; // [rsp+100h] [rbp-70h] BYREF
  __int64 v50; // [rsp+108h] [rbp-68h]
  __m128i v51; // [rsp+110h] [rbp-60h] BYREF
  _OWORD *v52; // [rsp+120h] [rbp-50h] BYREF
  size_t n; // [rsp+128h] [rbp-48h]
  _OWORD src[4]; // [rsp+130h] [rbp-40h] BYREF

  v1 = (char *)sub_1649960(a1);
  if ( v1 )
  {
    v32 = v34;
    sub_1814770((__int64 *)&v32, v1, (__int64)&v1[v2]);
  }
  else
  {
    LOBYTE(v34[0]) = 0;
    v32 = v34;
    v33 = 0;
  }
  v35 = v37;
  sub_1814770((__int64 *)&v35, "dfs$", (__int64)"");
  v52 = src;
  sub_1814C60((__int64 *)&v52, v35, (__int64)&v35[v36]);
  sub_2241490(&v52, v32, v33, v3);
  v49 = (__m128i *)&v52;
  v51.m128i_i16[0] = 260;
  sub_164B780(a1, (__int64 *)&v49);
  if ( v52 != src )
    j_j___libc_free_0(v52, *(_QWORD *)&src[0] + 1LL);
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_BYTE **)(v4 + 88);
  v6 = *(_QWORD *)(v4 + 96);
  v38 = v40;
  sub_1814C60((__int64 *)&v38, v5, (__int64)&v5[v6]);
  sub_8FD6D0((__int64)&v52, ".symver ", &v32);
  if ( n == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_62;
  v8 = (__m128i *)sub_2241490(&v52, ",", 1, v7);
  v41 = &v43;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v43 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v41 = (__m128i *)v8->m128i_i64[0];
    v43.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8[1].m128i_i8[0] = 0;
  v42 = v9;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v8->m128i_i64[1] = 0;
  if ( v52 != src )
    j_j___libc_free_0(v52, *(_QWORD *)&src[0] + 1LL);
  result = (_OWORD *)sub_22416F0(&v38, v41, 0, v42);
  v11 = result;
  if ( result != (_OWORD *)-1LL )
  {
    sub_8FD6D0((__int64)v44, ".symver ", &v35);
    v13 = (__m128i *)sub_2241490(v44, v32, v33, v12);
    v46 = &v48;
    if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
    {
      v48 = _mm_loadu_si128(v13 + 1);
    }
    else
    {
      v46 = (__m128i *)v13->m128i_i64[0];
      v48.m128i_i64[0] = v13[1].m128i_i64[0];
    }
    v14 = v13->m128i_i64[1];
    v13[1].m128i_i8[0] = 0;
    v47 = v14;
    v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
    v13->m128i_i64[1] = 0;
    if ( v47 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v15 = (__m128i *)sub_2241490(&v46, ",", 1, v14);
      v49 = &v51;
      if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
      {
        v51 = _mm_loadu_si128(v15 + 1);
      }
      else
      {
        v49 = (__m128i *)v15->m128i_i64[0];
        v51.m128i_i64[0] = v15[1].m128i_i64[0];
      }
      v50 = v15->m128i_i64[1];
      v16 = v50;
      v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
      v17 = v35;
      v15->m128i_i64[1] = 0;
      v18 = v36;
      v15[1].m128i_i8[0] = 0;
      v19 = (__m128i *)sub_2241490(&v49, v17, v18, v16);
      v52 = src;
      if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
      {
        src[0] = _mm_loadu_si128(v19 + 1);
      }
      else
      {
        v52 = (_OWORD *)v19->m128i_i64[0];
        *(_QWORD *)&src[0] = v19[1].m128i_i64[0];
      }
      n = v19->m128i_u64[1];
      v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
      v19->m128i_i64[1] = 0;
      v20 = v39;
      v19[1].m128i_i8[0] = 0;
      v21 = v20 - (_QWORD)v11;
      if ( v42 <= v20 - (unsigned __int64)v11 )
        v21 = v42;
      if ( (unsigned __int64)v11 > v20 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
      sub_2241130(&v38, v11, v21, v52, n);
      if ( v52 != src )
        j_j___libc_free_0(v52, *(_QWORD *)&src[0] + 1LL);
      if ( v49 != &v51 )
        j_j___libc_free_0(v49, v51.m128i_i64[0] + 1);
      if ( v46 != &v48 )
        j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
      if ( (__int64 *)v44[0] != &v45 )
        j_j___libc_free_0(v44[0], v45 + 1);
      v22 = *(_QWORD **)(a1 + 40);
      v52 = src;
      v23 = v22 + 13;
      if ( v38 )
      {
        sub_1814770((__int64 *)&v52, v38, (__int64)&v38[v39]);
        v24 = (_BYTE *)v22[11];
        result = v24;
        if ( v52 != src )
        {
          v25 = n;
          v26 = *(_QWORD *)&src[0];
          if ( v23 == (_OWORD *)v24 )
          {
            v22[11] = v52;
            v22[12] = v25;
            v22[13] = v26;
          }
          else
          {
            v27 = v22[13];
            v22[11] = v52;
            v22[12] = v25;
            v22[13] = v26;
            if ( result )
            {
              v52 = result;
              *(_QWORD *)&src[0] = v27;
LABEL_43:
              n = 0;
              *(_BYTE *)result = 0;
              if ( v52 != src )
                result = (_OWORD *)j_j___libc_free_0(v52, *(_QWORD *)&src[0] + 1LL);
              v28 = v22[12];
              if ( v28 )
              {
                result = (_OWORD *)v22[11];
                if ( *((_BYTE *)result + v28 - 1) != 10 )
                {
                  v29 = v28 + 1;
                  if ( result == v23 )
                    v30 = 15;
                  else
                    v30 = v22[13];
                  if ( v29 > v30 )
                  {
                    sub_2240BB0(v22 + 11, v22[12], 0, 0, 1);
                    result = (_OWORD *)v22[11];
                  }
                  *((_BYTE *)result + v28) = 10;
                  result = (_OWORD *)v22[11];
                  v22[12] = v29;
                  *((_BYTE *)result + v28 + 1) = 0;
                }
              }
              goto LABEL_11;
            }
          }
          v52 = src;
          result = src;
          goto LABEL_43;
        }
        v31 = n;
        if ( n )
        {
          if ( n == 1 )
            *v24 = src[0];
          else
            memcpy(v24, src, n);
          v31 = n;
          v24 = (_BYTE *)v22[11];
        }
      }
      else
      {
        LOBYTE(src[0]) = 0;
        v24 = (_BYTE *)v22[11];
        v31 = 0;
      }
      v22[12] = v31;
      v24[v31] = 0;
      result = v52;
      goto LABEL_43;
    }
LABEL_62:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_11:
  if ( v41 != &v43 )
    result = (_OWORD *)j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
  if ( v38 != (_BYTE *)v40 )
    result = (_OWORD *)j_j___libc_free_0(v38, v40[0] + 1LL);
  if ( v35 != (_BYTE *)v37 )
    result = (_OWORD *)j_j___libc_free_0(v35, v37[0] + 1LL);
  if ( v32 != v34 )
    return (_OWORD *)j_j___libc_free_0(v32, v34[0] + 1LL);
  return result;
}
