// Function: sub_CB0630
// Address: 0xcb0630
//
__int64 __fastcall sub_CB0630(__int64 a1, unsigned __int8 **a2)
{
  __int64 result; // rax
  unsigned __int8 *v4; // rdi
  __int64 v5; // rcx
  unsigned __int8 *v6; // rdx
  unsigned __int8 *v7; // rsi
  __int64 v8; // rdx
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int8 src[32]; // [rsp+10h] [rbp-20h] BYREF

  sub_CA95D0(&v9, **(_QWORD **)(a1 + 672));
  result = v9.m128i_i64[0];
  v4 = *a2;
  if ( (unsigned __int8 *)v9.m128i_i64[0] == src )
  {
    v8 = v9.m128i_i64[1];
    if ( v9.m128i_i64[1] )
    {
      if ( v9.m128i_i64[1] == 1 )
      {
        result = src[0];
        *v4 = src[0];
      }
      else
      {
        result = (__int64)memcpy(v4, src, v9.m128i_u64[1]);
      }
      v8 = v9.m128i_i64[1];
      v4 = *a2;
    }
    a2[1] = (unsigned __int8 *)v8;
    v4[v8] = 0;
    v4 = (unsigned __int8 *)v9.m128i_i64[0];
  }
  else
  {
    v5 = v9.m128i_i64[1];
    v6 = *(unsigned __int8 **)src;
    if ( v4 == (unsigned __int8 *)(a2 + 2) )
    {
      *a2 = (unsigned __int8 *)v9.m128i_i64[0];
      a2[1] = (unsigned __int8 *)v5;
      a2[2] = v6;
    }
    else
    {
      v7 = a2[2];
      *a2 = (unsigned __int8 *)v9.m128i_i64[0];
      a2[1] = (unsigned __int8 *)v5;
      a2[2] = v6;
      if ( v4 )
      {
        v9.m128i_i64[0] = (__int64)v4;
        *(_QWORD *)src = v7;
        goto LABEL_5;
      }
    }
    v9.m128i_i64[0] = (__int64)src;
    v4 = src;
  }
LABEL_5:
  v9.m128i_i64[1] = 0;
  *v4 = 0;
  if ( (unsigned __int8 *)v9.m128i_i64[0] != src )
    return j_j___libc_free_0(v9.m128i_i64[0], *(_QWORD *)src + 1LL);
  return result;
}
