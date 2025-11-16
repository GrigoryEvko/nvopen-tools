// Function: sub_16C97A0
// Address: 0x16c97a0
//
__int64 __fastcall sub_16C97A0(
        __int64 a1,
        __int64 *a2,
        char *a3,
        unsigned __int64 a4,
        _BYTE *a5,
        __int64 a6,
        _QWORD *a7)
{
  __int64 v7; // rdx
  __int64 v8; // rdx
  size_t v9; // rcx
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rcx
  char *v13; // r13
  char v14; // r12
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  char *v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r13
  unsigned __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  _BYTE *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rsi
  size_t v37; // rdx
  unsigned __int64 v41; // [rsp+28h] [rbp-168h]
  __int64 v42; // [rsp+28h] [rbp-168h]
  char *v43; // [rsp+30h] [rbp-160h] BYREF
  unsigned __int64 v44; // [rsp+38h] [rbp-158h]
  _QWORD v45[2]; // [rsp+40h] [rbp-150h] BYREF
  _QWORD v46[2]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v47; // [rsp+60h] [rbp-130h]
  _QWORD v48[2]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v49; // [rsp+80h] [rbp-110h]
  __m128i *v50; // [rsp+90h] [rbp-100h] BYREF
  __int64 v51; // [rsp+98h] [rbp-F8h]
  __m128i v52; // [rsp+A0h] [rbp-F0h] BYREF
  _QWORD *v53; // [rsp+B0h] [rbp-E0h] BYREF
  size_t n; // [rsp+B8h] [rbp-D8h]
  _QWORD src[2]; // [rsp+C0h] [rbp-D0h] BYREF
  _BYTE *v56; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+D8h] [rbp-B8h]
  _BYTE v58[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v56 = v58;
  v43 = a3;
  v44 = a4;
  v57 = 0x800000000LL;
  if ( a7 )
  {
    v7 = a7[1];
    if ( v7 )
      sub_2241130(a7, 0, v7, byte_3F871B3, 0);
  }
  if ( !(unsigned __int8)sub_16C9490(a2, (__int64)a5, a6, (__int64)&v56) )
  {
    *(_QWORD *)a1 = a1 + 16;
    if ( a5 )
    {
      sub_16C9290((__int64 *)a1, a5, (__int64)&a5[a6]);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
    goto LABEL_30;
  }
  v8 = *(_QWORD *)v56;
  v50 = &v52;
  sub_16C9290((__int64 *)&v50, a5, v8);
  if ( !v44 )
    goto LABEL_26;
  while ( 1 )
  {
    LOBYTE(v53) = 92;
    v11 = sub_16D20C0(&v43, &v53, 1, 0);
    if ( v11 == -1 )
      break;
    v12 = v11 + 1;
    if ( v11 + 1 > v44 )
      v12 = v44;
    v13 = &v43[v12];
    if ( v11 )
    {
      if ( v11 > v44 )
        v11 = v44;
      if ( 0x3FFFFFFFFFFFFFFFLL - v51 < v11 )
        goto LABEL_45;
    }
    v41 = v44 - v12;
    sub_2241490(&v50, v43, v11, v12);
    if ( !v41 )
      goto LABEL_69;
    v43 = v13;
    v44 = v41;
    v14 = *v13;
    if ( *v13 == 110 )
    {
      v24 = v51;
      v32 = (__int64)v50;
      v33 = 15;
      if ( v50 != &v52 )
        v33 = v52.m128i_i64[0];
      v27 = v51 + 1;
      if ( v51 + 1 > v33 )
      {
        sub_2240BB0(&v50, v51, 0, 0, 1);
        v32 = (__int64)v50;
      }
      *(_BYTE *)(v32 + v24) = 10;
      goto LABEL_42;
    }
    if ( v14 > 110 )
    {
      if ( v14 == 116 )
      {
        v24 = v51;
        v25 = (__int64)v50;
        v26 = 15;
        if ( v50 != &v52 )
          v26 = v52.m128i_i64[0];
        v27 = v51 + 1;
        if ( v51 + 1 > v26 )
        {
          sub_2240BB0(&v50, v51, 0, 0, 1);
          v25 = (__int64)v50;
        }
        *(_BYTE *)(v25 + v24) = 9;
LABEL_42:
        v51 = v27;
        v50->m128i_i8[v24 + 1] = 0;
        v28 = v44;
        if ( !v44 )
          goto LABEL_26;
      }
      else
      {
LABEL_46:
        v29 = v51;
        v30 = (__int64)v50;
        v9 = 15;
        if ( v50 != &v52 )
          v9 = v52.m128i_i64[0];
        v31 = v51 + 1;
        if ( v51 + 1 > v9 )
        {
          v42 = v51 + 1;
          sub_2240BB0(&v50, v51, 0, 0, 1);
          v30 = (__int64)v50;
          v31 = v42;
        }
        *(_BYTE *)(v30 + v29) = v14;
        v51 = v31;
        v50->m128i_i8[v29 + 1] = 0;
        v28 = v44;
        if ( !v44 )
          goto LABEL_26;
      }
      v18 = v28 - 1;
      ++v43;
      v44 = v18;
      goto LABEL_25;
    }
    if ( (unsigned __int8)(v14 - 48) > 9u )
      goto LABEL_46;
    v15 = sub_16D24E0(&v43, "0123456789", 10, 0);
    v16 = v44;
    if ( v15 )
    {
      if ( v15 > v44 )
        v15 = v44;
      v16 = v44 - v15;
    }
    v45[1] = v15;
    v44 = v16;
    v45[0] = v43;
    v17 = v43;
    v43 += v15;
    if ( !(unsigned __int8)sub_16D2B80(v17, v15, 10, &v53)
      && v53 == (_QWORD *)(unsigned int)v53
      && (unsigned int)v57 > (unsigned int)v53 )
    {
      v22 = &v56[16 * (_QWORD)v53];
      v23 = v22[1];
      if ( v23 > 0x3FFFFFFFFFFFFFFFLL - v51 )
        goto LABEL_45;
      sub_2241490(&v50, *v22, v23, v9);
      goto LABEL_24;
    }
    if ( a7 && !a7[1] )
    {
      v46[0] = "invalid backreference string '";
      v46[1] = v45;
      v47 = 1283;
      v48[0] = v46;
      v48[1] = "'";
      v49 = 770;
      sub_16E2FC0(&v53, v48);
      v34 = (_BYTE *)*a7;
      if ( v53 == src )
      {
        v37 = n;
        if ( n )
        {
          if ( n == 1 )
            *v34 = src[0];
          else
            memcpy(v34, src, n);
          v37 = n;
          v34 = (_BYTE *)*a7;
        }
        a7[1] = v37;
        v34[v37] = 0;
        v34 = v53;
        goto LABEL_66;
      }
      v9 = n;
      v35 = src[0];
      if ( v34 == (_BYTE *)(a7 + 2) )
      {
        *a7 = v53;
        a7[1] = v9;
        a7[2] = v35;
      }
      else
      {
        v36 = a7[2];
        *a7 = v53;
        a7[1] = v9;
        a7[2] = v35;
        if ( v34 )
        {
          v53 = v34;
          src[0] = v36;
          goto LABEL_66;
        }
      }
      v53 = src;
      v34 = src;
LABEL_66:
      n = 0;
      *v34 = 0;
      if ( v53 != src )
        j_j___libc_free_0(v53, src[0] + 1LL);
    }
LABEL_24:
    v18 = v44;
LABEL_25:
    if ( !v18 )
      goto LABEL_26;
  }
  v11 = v44;
  if ( 0x3FFFFFFFFFFFFFFFLL - v51 < v44 )
LABEL_45:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v50, v43, v44, v10);
LABEL_69:
  if ( v44 != v11 && a7 && !a7[1] )
    sub_2241130(a7, 0, 0, "replacement string contained trailing backslash", 47);
LABEL_26:
  v19 = *(_QWORD *)v56 + *((_QWORD *)v56 + 1);
  v20 = (unsigned __int64)&a5[a6 - v19];
  if ( v20 > 0x3FFFFFFFFFFFFFFFLL - v51 )
    goto LABEL_45;
  sub_2241490(&v50, v19, v20, v9);
  *(_QWORD *)a1 = a1 + 16;
  if ( v50 == &v52 )
  {
    *(__m128i *)(a1 + 16) = _mm_load_si128(&v52);
  }
  else
  {
    *(_QWORD *)a1 = v50;
    *(_QWORD *)(a1 + 16) = v52.m128i_i64[0];
  }
  *(_QWORD *)(a1 + 8) = v51;
LABEL_30:
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  return a1;
}
