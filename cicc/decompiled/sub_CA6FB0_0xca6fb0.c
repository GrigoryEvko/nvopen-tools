// Function: sub_CA6FB0
// Address: 0xca6fb0
//
_QWORD *__fastcall sub_CA6FB0(_QWORD *a1, char *a2, __int64 a3, unsigned __int64 a4)
{
  char *v4; // rbx
  char *v5; // r15
  char v6; // r14
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r14d
  unsigned __int64 v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // r8
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  char *v21; // rsi
  char v22; // dl
  __int64 v23; // rcx
  __int64 v24; // r8
  __m128i *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // rax
  __int64 v28; // rdx
  __m128i *v29; // rax
  __int64 v30; // rcx
  __m128i *v31; // rax
  __m128i *v32; // rsi
  __m128i *v33; // rdx
  __m128i *v34; // rax
  __int64 v35; // rcx
  __m128i *v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rsi
  unsigned __int64 v39; // [rsp+18h] [rbp-C8h]
  char v40; // [rsp+27h] [rbp-B9h]
  _QWORD *v41; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-A8h]
  _QWORD v43[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v44[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v45[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v46; // [rsp+70h] [rbp-70h] BYREF
  __int64 v47; // [rsp+78h] [rbp-68h]
  __m128i v48; // [rsp+80h] [rbp-60h] BYREF
  __m128i *v49; // [rsp+90h] [rbp-50h] BYREF
  __int64 v50; // [rsp+98h] [rbp-48h]
  __m128i v51; // [rsp+A0h] [rbp-40h] BYREF

  v4 = &a2[a3];
  v40 = a4;
  *a1 = a1 + 2;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  if ( &a2[a3] == a2 )
    return a1;
  v5 = a2;
  while ( 1 )
  {
    v6 = *v5;
    if ( *v5 == 92 )
    {
      while ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
      {
        sub_2241490(a1, "\\\\", 2, a4);
LABEL_34:
        if ( v4 == ++v5 )
          return a1;
        v6 = *v5;
        if ( *v5 != 92 )
          goto LABEL_4;
      }
LABEL_113:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_4:
    switch ( v6 )
    {
      case 34:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\\"", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 0:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\0", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 7:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\a", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 8:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\b", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 9:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\t", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 10:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\n", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 11:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\v", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 12:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\f", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 13:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\r", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
      case 27:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\e", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_113;
    }
    if ( (unsigned __int8)v6 <= 0x1Fu )
    {
      v20 = v6;
      v21 = &v51.m128i_i8[1];
      v22 = a0123456789abcd_10[v6 & 0xF];
      do
      {
        --v21;
        v20 >>= 4;
        *v21 = v22;
        v22 = 49;
      }
      while ( v20 );
      v41 = v43;
      sub_CA64F0((__int64 *)&v41, v21, (__int64)v51.m128i_i64 + 1);
      v44[0] = v45;
      sub_2240A50(v44, 2 - v42, 48, v23, v24);
      v25 = (__m128i *)sub_2241130(v44, 0, 0, "\\x", 2);
      v46 = &v48;
      if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
      {
        v48 = _mm_loadu_si128(v25 + 1);
      }
      else
      {
        v46 = (__m128i *)v25->m128i_i64[0];
        v48.m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v47 = v25->m128i_i64[1];
      v26 = v47;
      v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
      v25->m128i_i64[1] = 0;
      v25[1].m128i_i8[0] = 0;
      v27 = (__m128i *)sub_2241490(&v46, v41, v42, v26);
      v49 = &v51;
      if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
      {
        v51 = _mm_loadu_si128(v27 + 1);
      }
      else
      {
        v49 = (__m128i *)v27->m128i_i64[0];
        v51.m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v50 = v27->m128i_i64[1];
      v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
      v27->m128i_i64[1] = 0;
      v27[1].m128i_i8[0] = 0;
      sub_2241490(a1, v49, v50, &v51);
      a4 = (unsigned __int64)&v51;
      if ( v49 != &v51 )
        j_j___libc_free_0(v49, v51.m128i_i64[0] + 1);
      if ( v46 != &v48 )
        j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
      if ( (_QWORD *)v44[0] != v45 )
        j_j___libc_free_0(v44[0], v45[0] + 1LL);
      if ( v41 != v43 )
        j_j___libc_free_0(v41, v43[0] + 1LL);
      goto LABEL_34;
    }
    if ( v6 >= 0 )
    {
      v17 = a1[1];
      v18 = *a1;
      v19 = v17 + 1;
      if ( a1 + 2 == (_QWORD *)*a1 )
        a4 = 15;
      else
        a4 = a1[2];
      if ( v19 > a4 )
      {
        sub_2240BB0(a1, v17, 0, 0, 1);
        v18 = *a1;
        v19 = v17 + 1;
      }
      *(_BYTE *)(v18 + v17) = v6;
      a1[1] = v19;
      *(_BYTE *)(*a1 + v17 + 1) = 0;
      goto LABEL_34;
    }
    v7 = sub_CA5E90(v5, v4 - v5);
    v12 = v7;
    v39 = HIDWORD(v7);
    if ( !HIDWORD(v7) )
      break;
    switch ( (_DWORD)v7 )
    {
      case 0x85:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_113;
        sub_2241490(a1, "\\N", 2, v9);
        break;
      case 0xA0:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_113;
        sub_2241490(a1, "\\_", 2, v9);
        break;
      case 0x2028:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_113;
        sub_2241490(a1, "\\L", 2, v9);
        break;
      case 0x2029:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_113;
        sub_2241490(a1, "\\P", 2, v9);
        break;
      default:
        if ( v40 || !sub_CA1970(v7) )
        {
          v13 = v12;
          if ( v12 )
          {
            v14 = (__m128i *)&v51.m128i_i8[1];
            do
            {
              v14 = (__m128i *)((char *)v14 - 1);
              v28 = v13 & 0xF;
              v13 >>= 4;
              v14->m128i_i8[0] = a0123456789abcd_10[v28];
            }
            while ( v13 );
          }
          else
          {
            v51.m128i_i8[0] = 48;
            v14 = &v51;
          }
          v41 = v43;
          sub_CA64F0((__int64 *)&v41, v14, (__int64)v51.m128i_i64 + 1);
          if ( v42 <= 2 )
          {
            v44[0] = v45;
            sub_2240A50(v44, 2 - v42, 48, v45, v15);
            v34 = (__m128i *)sub_2241130(v44, 0, 0, "\\x", 2);
            v46 = &v48;
            if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
            {
              v48 = _mm_loadu_si128(v34 + 1);
            }
            else
            {
              v46 = (__m128i *)v34->m128i_i64[0];
              v48.m128i_i64[0] = v34[1].m128i_i64[0];
            }
            v47 = v34->m128i_i64[1];
            v35 = v47;
            v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
            v34->m128i_i64[1] = 0;
            v34[1].m128i_i8[0] = 0;
            v31 = (__m128i *)sub_2241490(&v46, v41, v42, v35);
            v49 = &v51;
            v32 = (__m128i *)v31->m128i_i64[0];
            v33 = v31 + 1;
            if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
              goto LABEL_102;
          }
          else
          {
            if ( v42 > 4 )
            {
              if ( v42 > 8 )
              {
LABEL_28:
                if ( v41 != v43 )
                  j_j___libc_free_0(v41, v43[0] + 1LL);
                break;
              }
              v44[0] = v45;
              sub_2240A50(v44, 8 - v42, 48, v45, v15);
              v29 = (__m128i *)sub_2241130(v44, 0, 0, "\\U", 2);
              v46 = &v48;
              if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
              {
                v48 = _mm_loadu_si128(v29 + 1);
              }
              else
              {
                v46 = (__m128i *)v29->m128i_i64[0];
                v48.m128i_i64[0] = v29[1].m128i_i64[0];
              }
              v47 = v29->m128i_i64[1];
              v30 = v47;
              v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
              v29->m128i_i64[1] = 0;
              v29[1].m128i_i8[0] = 0;
              v31 = (__m128i *)sub_2241490(&v46, v41, v42, v30);
              v49 = &v51;
              v32 = (__m128i *)v31->m128i_i64[0];
              v33 = v31 + 1;
              if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
                goto LABEL_92;
LABEL_102:
              v51 = _mm_loadu_si128(v31 + 1);
              goto LABEL_93;
            }
            v44[0] = v45;
            sub_2240A50(v44, 4 - v42, 48, v45, v15);
            v36 = (__m128i *)sub_2241130(v44, 0, 0, "\\u", 2);
            v46 = &v48;
            if ( (__m128i *)v36->m128i_i64[0] == &v36[1] )
            {
              v48 = _mm_loadu_si128(v36 + 1);
            }
            else
            {
              v46 = (__m128i *)v36->m128i_i64[0];
              v48.m128i_i64[0] = v36[1].m128i_i64[0];
            }
            v47 = v36->m128i_i64[1];
            v37 = v47;
            v36->m128i_i64[0] = (__int64)v36[1].m128i_i64;
            v36->m128i_i64[1] = 0;
            v36[1].m128i_i8[0] = 0;
            v31 = (__m128i *)sub_2241490(&v46, v41, v42, v37);
            v49 = &v51;
            v32 = (__m128i *)v31->m128i_i64[0];
            v33 = v31 + 1;
            if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
              goto LABEL_102;
          }
LABEL_92:
          v49 = v32;
          v51.m128i_i64[0] = v31[1].m128i_i64[0];
LABEL_93:
          v50 = v31->m128i_i64[1];
          v31->m128i_i64[0] = (__int64)v33;
          v31->m128i_i64[1] = 0;
          v31[1].m128i_i8[0] = 0;
          sub_2241490(a1, v49, v50, &v51);
          a4 = (unsigned __int64)&v51;
          if ( v49 != &v51 )
            j_j___libc_free_0(v49, v51.m128i_i64[0] + 1);
          if ( v46 != &v48 )
            j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
          if ( (_QWORD *)v44[0] != v45 )
            j_j___libc_free_0(v44[0], v45[0] + 1LL);
          goto LABEL_28;
        }
        if ( v39 > 0x3FFFFFFFFFFFFFFFLL - a1[1] )
          goto LABEL_113;
        sub_2241490(a1, v5, (unsigned int)v39, v39);
        break;
    }
    v5 += (unsigned int)(v39 - 1) + 1;
    if ( v4 == v5 )
      return a1;
  }
  v50 = 0;
  v49 = (__m128i *)&v51.m128i_u64[1];
  v51.m128i_i64[0] = 4;
  sub_CA6BA0(0xFFFDu, &v49, v8, v9, v10, v11);
  v38 = a1[1];
  sub_2241130(a1, v38, 0, v49, v50);
  if ( v49 != (__m128i *)&v51.m128i_u64[1] )
    _libc_free(v49, v38);
  return a1;
}
