// Function: sub_1255B60
// Address: 0x1255b60
//
__int64 __fastcall sub_1255B60(__int64 a1, char *a2, unsigned __int64 a3)
{
  __int64 result; // rax
  size_t v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  size_t v9; // rdx
  void *dest; // [rsp+0h] [rbp-90h] BYREF
  size_t v11; // [rsp+8h] [rbp-88h]
  __m128i v12; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v13; // [rsp+20h] [rbp-70h] BYREF
  size_t n; // [rsp+28h] [rbp-68h]
  _QWORD src[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int16 v16; // [rsp+40h] [rbp-50h] BYREF
  __m128i *v17; // [rsp+48h] [rbp-48h]
  size_t v18; // [rsp+50h] [rbp-40h]
  __m128i v19; // [rsp+58h] [rbp-38h] BYREF

  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_WORD *)a1 = 5;
  result = sub_C6A630(a2, a3, 0);
  if ( (_BYTE)result )
    return result;
  sub_C6B0E0((__int64 *)&dest, (__int64)a2, a3);
  v16 = 6;
  if ( !(unsigned __int8)sub_C6A630((char *)dest, v11, 0) )
  {
    sub_C6B0E0((__int64 *)&v13, (__int64)dest, v11);
    v7 = dest;
    if ( v13 == src )
    {
      v9 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v9 = n;
        v7 = dest;
      }
      v11 = v9;
      *((_BYTE *)v7 + v9) = 0;
      v7 = v13;
      goto LABEL_13;
    }
    if ( dest == &v12 )
    {
      dest = v13;
      v11 = n;
      v12.m128i_i64[0] = src[0];
    }
    else
    {
      v8 = v12.m128i_i64[0];
      dest = v13;
      v11 = n;
      v12.m128i_i64[0] = src[0];
      if ( v7 )
      {
        v13 = v7;
        src[0] = v8;
        goto LABEL_13;
      }
    }
    v13 = src;
    v7 = src;
LABEL_13:
    n = 0;
    *(_BYTE *)v7 = 0;
    if ( v13 != src )
      j_j___libc_free_0(v13, src[0] + 1LL);
  }
  v17 = &v19;
  if ( dest == &v12 )
  {
    v19 = _mm_load_si128(&v12);
  }
  else
  {
    v17 = (__m128i *)dest;
    v19.m128i_i64[0] = v12.m128i_i64[0];
  }
  v6 = v11;
  dest = &v12;
  v11 = 0;
  v18 = v6;
  v12.m128i_i8[0] = 0;
  sub_C6BC50((unsigned __int16 *)a1);
  sub_C6A4F0(a1, &v16);
  result = sub_C6BC50(&v16);
  if ( dest != &v12 )
    return j_j___libc_free_0(dest, v12.m128i_i64[0] + 1);
  return result;
}
