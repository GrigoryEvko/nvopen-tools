// Function: sub_1569190
// Address: 0x1569190
//
__int64 __fastcall sub_1569190(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // rax
  _BYTE *v5; // rdi
  __int64 v6; // rax
  size_t v7; // rdx
  __int64 v8; // rcx
  __int64 result; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // r8
  size_t v12; // r9
  _QWORD *v13; // rax
  _BYTE *v14; // r9
  size_t v15; // r8
  _QWORD *v16; // rax
  __m128i *v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rcx
  __m128i *v21; // rax
  __m128i *v22; // rdx
  __m128i *v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rdi
  size_t n; // [rsp+0h] [rbp-120h]
  _BYTE *src; // [rsp+8h] [rbp-118h]
  size_t v33; // [rsp+10h] [rbp-110h]
  unsigned __int8 v34; // [rsp+18h] [rbp-108h]
  _BYTE *v35; // [rsp+18h] [rbp-108h]
  __m128i *v36; // [rsp+20h] [rbp-100h]
  __int64 v37; // [rsp+28h] [rbp-F8h]
  __m128i v38; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD *v39; // [rsp+40h] [rbp-E0h] BYREF
  size_t v40; // [rsp+48h] [rbp-D8h]
  _QWORD v41[2]; // [rsp+50h] [rbp-D0h] BYREF
  __m128i *v42; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+68h] [rbp-B8h]
  __m128i v44; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v45; // [rsp+80h] [rbp-A0h] BYREF
  size_t v46; // [rsp+88h] [rbp-98h]
  _QWORD v47[2]; // [rsp+90h] [rbp-90h] BYREF
  const char *v48; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v49; // [rsp+A8h] [rbp-78h]
  _BYTE v50[112]; // [rsp+B0h] [rbp-70h] BYREF

  v50[1] = 1;
  v48 = "clang.arc.retainAutoreleasedReturnValueMarker";
  v50[0] = 3;
  v2 = sub_1632310(a1, &v48);
  if ( !v2 )
    return 0;
  v3 = v2;
  v4 = sub_161F530(v2, 0);
  if ( !v4 )
    return 0;
  v5 = *(_BYTE **)(v4 - 8LL * *(unsigned int *)(v4 + 8));
  if ( !v5 || *v5 )
    return 0;
  v48 = v50;
  v49 = 0x400000000LL;
  v6 = sub_161E970(v5);
  v46 = v7;
  v45 = (_QWORD *)v6;
  sub_16D2730(&v45, &v48, "#", 1, 0xFFFFFFFFLL, 1);
  result = 0;
  if ( (_DWORD)v49 == 2 )
  {
    v10 = (unsigned __int64)v48;
    v11 = (_BYTE *)*((_QWORD *)v48 + 2);
    if ( !v11 )
    {
      LOBYTE(v47[0]) = 0;
      v45 = v47;
      v46 = 0;
LABEL_15:
      v14 = *(_BYTE **)v10;
      if ( !*(_QWORD *)v10 )
      {
        LOBYTE(v41[0]) = 0;
        v39 = v41;
        v40 = 0;
        goto LABEL_20;
      }
      v15 = *(_QWORD *)(v10 + 8);
      v39 = v41;
      v42 = (__m128i *)v15;
      if ( v15 > 0xF )
      {
        n = v15;
        src = v14;
        v27 = sub_22409D0(&v39, &v42, 0);
        v14 = src;
        v39 = (_QWORD *)v27;
        v28 = (_QWORD *)v27;
        v15 = n;
        v41[0] = v42;
      }
      else
      {
        if ( v15 == 1 )
        {
          LOBYTE(v41[0]) = *v14;
          v16 = v41;
          goto LABEL_19;
        }
        if ( !v15 )
        {
          v16 = v41;
          goto LABEL_19;
        }
        v28 = v41;
      }
      memcpy(v28, v14, v15);
      v15 = (size_t)v42;
      v16 = v39;
LABEL_19:
      v40 = v15;
      *((_BYTE *)v16 + v15) = 0;
      if ( v40 == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
LABEL_20:
      v17 = (__m128i *)sub_2241490(&v39, ";", 1, v8);
      v42 = &v44;
      if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
      {
        v44 = _mm_loadu_si128(v17 + 1);
      }
      else
      {
        v42 = (__m128i *)v17->m128i_i64[0];
        v44.m128i_i64[0] = v17[1].m128i_i64[0];
      }
      v43 = v17->m128i_i64[1];
      v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
      v17->m128i_i64[1] = 0;
      v17[1].m128i_i8[0] = 0;
      v18 = 15;
      v19 = 15;
      if ( v42 != &v44 )
        v19 = v44.m128i_i64[0];
      v20 = v43 + v46;
      if ( v43 + v46 <= v19 )
        goto LABEL_28;
      if ( v45 != v47 )
        v18 = v47[0];
      if ( v20 <= v18 )
      {
        v21 = (__m128i *)sub_2241130(&v45, 0, 0, v42, v43);
        v22 = v21 + 1;
        v36 = &v38;
        v23 = (__m128i *)v21->m128i_i64[0];
        if ( (__m128i *)v21->m128i_i64[0] != &v21[1] )
          goto LABEL_29;
      }
      else
      {
LABEL_28:
        v21 = (__m128i *)sub_2241490(&v42, v45, v46, v20);
        v22 = v21 + 1;
        v36 = &v38;
        v23 = (__m128i *)v21->m128i_i64[0];
        if ( (__m128i *)v21->m128i_i64[0] != &v21[1] )
        {
LABEL_29:
          v36 = v23;
          v38.m128i_i64[0] = v21[1].m128i_i64[0];
LABEL_30:
          v37 = v21->m128i_i64[1];
          v21->m128i_i64[0] = (__int64)v22;
          v21->m128i_i64[1] = 0;
          v21[1].m128i_i8[0] = 0;
          if ( v42 != &v44 )
            j_j___libc_free_0(v42, v44.m128i_i64[0] + 1);
          if ( v39 != v41 )
            j_j___libc_free_0(v39, v41[0] + 1LL);
          if ( v45 != v47 )
            j_j___libc_free_0(v45, v47[0] + 1LL);
          v24 = sub_161FF10(*a1, v36, v37);
          v25 = *a1;
          v45 = (_QWORD *)v24;
          v26 = sub_1627350(v25, &v45, 1, 0, 1);
          sub_1623BA0(v3, 0, v26);
          if ( v36 != &v38 )
            j_j___libc_free_0(v36, v38.m128i_i64[0] + 1);
          result = 1;
          goto LABEL_6;
        }
      }
      v38 = _mm_loadu_si128(v21 + 1);
      goto LABEL_30;
    }
    v12 = *((_QWORD *)v48 + 3);
    v45 = v47;
    v42 = (__m128i *)v12;
    if ( v12 > 0xF )
    {
      v33 = v12;
      v35 = v11;
      v29 = sub_22409D0(&v45, &v42, 0);
      v11 = v35;
      v12 = v33;
      v45 = (_QWORD *)v29;
      v30 = (_QWORD *)v29;
      v47[0] = v42;
    }
    else
    {
      if ( v12 == 1 )
      {
        LOBYTE(v47[0]) = *v11;
        v13 = v47;
LABEL_14:
        v46 = v12;
        *((_BYTE *)v13 + v12) = 0;
        v10 = (unsigned __int64)v48;
        goto LABEL_15;
      }
      if ( !v12 )
      {
        v13 = v47;
        goto LABEL_14;
      }
      v30 = v47;
    }
    memcpy(v30, v11, v12);
    v12 = (size_t)v42;
    v13 = v45;
    goto LABEL_14;
  }
LABEL_6:
  if ( v48 != v50 )
  {
    v34 = result;
    _libc_free((unsigned __int64)v48);
    return v34;
  }
  return result;
}
