// Function: sub_C984A0
// Address: 0xc984a0
//
__int64 __fastcall sub_C984A0(__int64 a1)
{
  __int64 v2; // r13
  _OWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  _OWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  _OWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 result; // rax
  _QWORD *v22; // rdi
  __int64 v23; // r9
  _QWORD *v24; // rdi
  __int64 v25; // r8
  size_t v26; // rdx
  size_t v27; // rdx
  __m128i *v28; // rdi
  __int64 v29; // r9
  size_t v30; // rdx
  void *v31; // [rsp+20h] [rbp-D0h] BYREF
  size_t v32; // [rsp+28h] [rbp-C8h]
  __m128i v33; // [rsp+30h] [rbp-C0h] BYREF
  void *dest; // [rsp+40h] [rbp-B0h] BYREF
  size_t v35; // [rsp+48h] [rbp-A8h]
  __m128i v36; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v37; // [rsp+60h] [rbp-90h] BYREF
  size_t n; // [rsp+68h] [rbp-88h]
  _QWORD src[4]; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int16 v40; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v41; // [rsp+98h] [rbp-58h]
  size_t v42; // [rsp+A0h] [rbp-50h]
  _OWORD v43[4]; // [rsp+A8h] [rbp-48h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = (_OWORD *)*(int *)(*(_QWORD *)(a1 + 8) + 16616LL);
  v40 = 3;
  v41 = v3;
  sub_C6B410(v2, (unsigned __int8 *)"pid", 3u);
  sub_C6C710(v2, &v40, v4);
  sub_C6AE10(v2);
  sub_C6BC50(&v40);
  v6 = **(_OWORD ***)(a1 + 16);
  v5 = *(_QWORD *)a1;
  v40 = 3;
  v41 = v6;
  sub_C6B410(v5, (unsigned __int8 *)"tid", 3u);
  sub_C6C710(v5, &v40, v7);
  sub_C6AE10(v5);
  sub_C6BC50(&v40);
  v8 = *(_QWORD *)a1;
  v9 = (_OWORD *)(**(_QWORD **)(a1 + 24) + **(_QWORD **)(a1 + 32));
  v40 = 3;
  v41 = v9;
  sub_C6B410(v8, (unsigned __int8 *)"ts", 2u);
  sub_C6C710(v8, &v40, v10);
  sub_C6AE10(v8);
  sub_C6BC50(&v40);
  v11 = *(_QWORD *)(a1 + 40);
  dest = &v36;
  v12 = *(_QWORD *)a1;
  sub_C95D30((__int64 *)&dest, *(_BYTE **)(v11 + 16), *(_QWORD *)(v11 + 16) + *(_QWORD *)(v11 + 24));
  v40 = 6;
  if ( (unsigned __int8)sub_C6A630((char *)dest, v35, 0) )
    goto LABEL_2;
  sub_C6B0E0((__int64 *)&v37, (__int64)dest, v35);
  v22 = dest;
  if ( v37 == src )
  {
    v27 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v27 = n;
      v22 = dest;
    }
    v35 = v27;
    *((_BYTE *)v22 + v27) = 0;
    v22 = v37;
    goto LABEL_19;
  }
  if ( dest == &v36 )
  {
    dest = v37;
    v35 = n;
    v36.m128i_i64[0] = src[0];
  }
  else
  {
    v23 = v36.m128i_i64[0];
    dest = v37;
    v35 = n;
    v36.m128i_i64[0] = src[0];
    if ( v22 )
    {
      v37 = v22;
      src[0] = v23;
      goto LABEL_19;
    }
  }
  v37 = src;
  v22 = src;
LABEL_19:
  n = 0;
  *(_BYTE *)v22 = 0;
  if ( v37 != src )
    j_j___libc_free_0(v37, src[0] + 1LL);
LABEL_2:
  v41 = v43;
  if ( dest == &v36 )
  {
    v43[0] = _mm_load_si128(&v36);
  }
  else
  {
    v41 = dest;
    *(_QWORD *)&v43[0] = v36.m128i_i64[0];
  }
  dest = &v36;
  v42 = v35;
  v35 = 0;
  v36.m128i_i8[0] = 0;
  sub_C6B410(v12, (unsigned __int8 *)"cat", 3u);
  sub_C6C710(v12, &v40, v13);
  sub_C6AE10(v12);
  sub_C6BC50(&v40);
  if ( dest != &v36 )
    j_j___libc_free_0(dest, v36.m128i_i64[0] + 1);
  v14 = *(_QWORD *)a1;
  LOWORD(v37) = 5;
  n = (size_t)"e";
  src[0] = 1;
  if ( !(unsigned __int8)sub_C6A630("e", 1, 0) )
  {
    sub_C6B0E0((__int64 *)&v31, (__int64)"e", 1u);
    v40 = 6;
    if ( (unsigned __int8)sub_C6A630((char *)v31, v32, 0) )
    {
LABEL_22:
      v41 = v43;
      if ( v31 == &v33 )
      {
        v43[0] = _mm_load_si128(&v33);
      }
      else
      {
        v41 = v31;
        *(_QWORD *)&v43[0] = v33.m128i_i64[0];
      }
      v31 = &v33;
      v42 = v32;
      v32 = 0;
      v33.m128i_i8[0] = 0;
      sub_C6BC50((unsigned __int16 *)&v37);
      sub_C6A4F0((__int64)&v37, &v40);
      sub_C6BC50(&v40);
      if ( v31 != &v33 )
        j_j___libc_free_0(v31, v33.m128i_i64[0] + 1);
      goto LABEL_7;
    }
    sub_C6B0E0((__int64 *)&dest, (__int64)v31, v32);
    v28 = (__m128i *)v31;
    if ( dest == &v36 )
    {
      v30 = v35;
      if ( v35 )
      {
        if ( v35 == 1 )
          *(_BYTE *)v31 = v36.m128i_i8[0];
        else
          memcpy(v31, &v36, v35);
        v30 = v35;
        v28 = (__m128i *)v31;
      }
      v32 = v30;
      v28->m128i_i8[v30] = 0;
      v28 = (__m128i *)dest;
      goto LABEL_51;
    }
    if ( v31 == &v33 )
    {
      v31 = dest;
      v32 = v35;
      v33.m128i_i64[0] = v36.m128i_i64[0];
    }
    else
    {
      v29 = v33.m128i_i64[0];
      v31 = dest;
      v32 = v35;
      v33.m128i_i64[0] = v36.m128i_i64[0];
      if ( v28 )
      {
        dest = v28;
        v36.m128i_i64[0] = v29;
        goto LABEL_51;
      }
    }
    dest = &v36;
    v28 = &v36;
LABEL_51:
    v35 = 0;
    v28->m128i_i8[0] = 0;
    if ( dest != &v36 )
      j_j___libc_free_0(dest, v36.m128i_i64[0] + 1);
    goto LABEL_22;
  }
LABEL_7:
  sub_C6B410(v14, (unsigned __int8 *)"ph", 2u);
  sub_C6C710(v14, (unsigned __int16 *)&v37, v15);
  sub_C6AE10(v14);
  sub_C6BC50((unsigned __int16 *)&v37);
  v16 = *(_QWORD *)a1;
  v40 = 3;
  v41 = 0;
  sub_C6B410(v16, (unsigned __int8 *)"id", 2u);
  sub_C6C710(v16, &v40, v17);
  sub_C6AE10(v16);
  sub_C6BC50(&v40);
  v18 = *(_QWORD *)(a1 + 40);
  dest = &v36;
  v19 = *(_QWORD *)a1;
  sub_C95D30((__int64 *)&dest, *(_BYTE **)(v18 + 16), *(_QWORD *)(v18 + 16) + *(_QWORD *)(v18 + 24));
  v40 = 6;
  if ( (unsigned __int8)sub_C6A630((char *)dest, v35, 0) )
    goto LABEL_8;
  sub_C6B0E0((__int64 *)&v37, (__int64)dest, v35);
  v24 = dest;
  if ( v37 == src )
  {
    v26 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v26 = n;
      v24 = dest;
    }
    v35 = v26;
    *((_BYTE *)v24 + v26) = 0;
    v24 = v37;
    goto LABEL_30;
  }
  if ( dest == &v36 )
  {
    dest = v37;
    v35 = n;
    v36.m128i_i64[0] = src[0];
  }
  else
  {
    v25 = v36.m128i_i64[0];
    dest = v37;
    v35 = n;
    v36.m128i_i64[0] = src[0];
    if ( v24 )
    {
      v37 = v24;
      src[0] = v25;
      goto LABEL_30;
    }
  }
  v37 = src;
  v24 = src;
LABEL_30:
  n = 0;
  *(_BYTE *)v24 = 0;
  if ( v37 != src )
    j_j___libc_free_0(v37, src[0] + 1LL);
LABEL_8:
  v41 = v43;
  if ( dest == &v36 )
  {
    v43[0] = _mm_load_si128(&v36);
  }
  else
  {
    v41 = dest;
    *(_QWORD *)&v43[0] = v36.m128i_i64[0];
  }
  dest = &v36;
  v42 = v35;
  v35 = 0;
  v36.m128i_i8[0] = 0;
  sub_C6B410(v19, (unsigned __int8 *)"name", 4u);
  sub_C6C710(v19, &v40, v20);
  sub_C6AE10(v19);
  result = sub_C6BC50(&v40);
  if ( dest != &v36 )
    return j_j___libc_free_0(dest, v36.m128i_i64[0] + 1);
  return result;
}
