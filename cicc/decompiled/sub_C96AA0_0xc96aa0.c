// Function: sub_C96AA0
// Address: 0xc96aa0
//
__int64 __fastcall sub_C96AA0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  size_t v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  size_t v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  size_t v18; // rdx
  size_t v19; // rdx
  void *dest; // [rsp+0h] [rbp-90h] BYREF
  size_t v21; // [rsp+8h] [rbp-88h]
  __m128i v22; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v23; // [rsp+20h] [rbp-70h] BYREF
  size_t n; // [rsp+28h] [rbp-68h]
  _QWORD src[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int16 v26; // [rsp+40h] [rbp-50h] BYREF
  _OWORD *v27; // [rsp+48h] [rbp-48h]
  size_t v28; // [rsp+50h] [rbp-40h]
  _OWORD v29[3]; // [rsp+58h] [rbp-38h] BYREF

  v2 = *a1;
  v3 = *(_QWORD *)(*a1 + 56LL);
  if ( !v3 )
    goto LABEL_2;
  v11 = a1[1];
  dest = &v22;
  sub_C95D30((__int64 *)&dest, *(_BYTE **)(v2 + 48), *(_QWORD *)(v2 + 48) + v3);
  v26 = 6;
  if ( !(unsigned __int8)sub_C6A630((char *)dest, v21, 0) )
  {
    sub_C6B0E0((__int64 *)&v23, (__int64)dest, v21);
    v14 = dest;
    if ( v23 == src )
    {
      v18 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v18 = n;
        v14 = dest;
      }
      v21 = v18;
      *((_BYTE *)v14 + v18) = 0;
      v14 = v23;
      goto LABEL_24;
    }
    if ( dest == &v22 )
    {
      dest = v23;
      v21 = n;
      v22.m128i_i64[0] = src[0];
    }
    else
    {
      v15 = v22.m128i_i64[0];
      dest = v23;
      v21 = n;
      v22.m128i_i64[0] = src[0];
      if ( v14 )
      {
        v23 = v14;
        src[0] = v15;
        goto LABEL_24;
      }
    }
    v23 = src;
    v14 = src;
LABEL_24:
    n = 0;
    *(_BYTE *)v14 = 0;
    if ( v23 != src )
      j_j___libc_free_0(v23, src[0] + 1LL);
  }
  v27 = v29;
  if ( dest == &v22 )
  {
    v29[0] = _mm_load_si128(&v22);
  }
  else
  {
    v27 = dest;
    *(_QWORD *)&v29[0] = v22.m128i_i64[0];
  }
  v22.m128i_i8[0] = 0;
  v12 = v21;
  dest = &v22;
  v21 = 0;
  v28 = v12;
  sub_C6B410(v11, "detail", 6u);
  sub_C6C710(v11, &v26, v13);
  sub_C6AE10(v11);
  sub_C6BC50(&v26);
  if ( dest != &v22 )
    j_j___libc_free_0(dest, v22.m128i_i64[0] + 1);
  v2 = *a1;
LABEL_2:
  v4 = *(_QWORD *)(v2 + 88);
  if ( !v4 )
    goto LABEL_3;
  v8 = a1[1];
  dest = &v22;
  sub_C95D30((__int64 *)&dest, *(_BYTE **)(v2 + 80), *(_QWORD *)(v2 + 80) + v4);
  v26 = 6;
  if ( !(unsigned __int8)sub_C6A630((char *)dest, v21, 0) )
  {
    sub_C6B0E0((__int64 *)&v23, (__int64)dest, v21);
    v16 = dest;
    if ( v23 == src )
    {
      v19 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v19 = n;
        v16 = dest;
      }
      v21 = v19;
      *((_BYTE *)v16 + v19) = 0;
      v16 = v23;
      goto LABEL_30;
    }
    if ( dest == &v22 )
    {
      dest = v23;
      v21 = n;
      v22.m128i_i64[0] = src[0];
    }
    else
    {
      v17 = v22.m128i_i64[0];
      dest = v23;
      v21 = n;
      v22.m128i_i64[0] = src[0];
      if ( v16 )
      {
        v23 = v16;
        src[0] = v17;
        goto LABEL_30;
      }
    }
    v23 = src;
    v16 = src;
LABEL_30:
    n = 0;
    *(_BYTE *)v16 = 0;
    if ( v23 != src )
      j_j___libc_free_0(v23, src[0] + 1LL);
  }
  v27 = v29;
  if ( dest == &v22 )
  {
    v29[0] = _mm_load_si128(&v22);
  }
  else
  {
    v27 = dest;
    *(_QWORD *)&v29[0] = v22.m128i_i64[0];
  }
  v22.m128i_i8[0] = 0;
  v9 = v21;
  dest = &v22;
  v21 = 0;
  v28 = v9;
  sub_C6B410(v8, (unsigned __int8 *)"file", 4u);
  sub_C6C710(v8, &v26, v10);
  sub_C6AE10(v8);
  sub_C6BC50(&v26);
  if ( dest != &v22 )
    j_j___libc_free_0(dest, v22.m128i_i64[0] + 1);
  v2 = *a1;
LABEL_3:
  result = *(int *)(v2 + 112);
  if ( (int)result > 0 )
  {
    v6 = a1[1];
    v27 = (_OWORD *)result;
    v26 = 3;
    sub_C6B410(v6, (unsigned __int8 *)"line", 4u);
    sub_C6C710(v6, &v26, v7);
    sub_C6AE10(v6);
    return sub_C6BC50(&v26);
  }
  return result;
}
