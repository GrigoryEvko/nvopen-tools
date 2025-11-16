// Function: sub_C6B270
// Address: 0xc6b270
//
__int64 *__fastcall sub_C6B270(__int64 **a1, __int64 a2)
{
  __m128i *v4; // rax
  char *m128i_i8; // rdi
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 *result; // rax
  __int64 *v9; // rbx
  __int64 *v10; // rdi
  __int64 v11; // rdx
  size_t v12; // rcx
  __int64 v13; // rsi
  size_t v14; // rdx
  __int64 *v15; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = (__m128i *)sub_22077B0(32);
  if ( v4 )
  {
    m128i_i8 = *(char **)a2;
    v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
    if ( (char *)(a2 + 16) == m128i_i8 )
    {
      m128i_i8 = v4[1].m128i_i8;
      v4[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 16);
      v4->m128i_i64[0] = (__int64)m128i_i8;
      v4[1].m128i_i64[0] = v6;
    }
    v7 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)a2 = a2 + 16;
    *(_QWORD *)(a2 + 8) = 0;
    v4->m128i_i64[1] = v7;
    *(_BYTE *)(a2 + 16) = 0;
  }
  else
  {
    m128i_i8 = (char *)MEMORY[0];
    v7 = MEMORY[8];
  }
  *a1 = (__int64 *)v4;
  a1[1] = 0;
  a1[2] = 0;
  if ( !(unsigned __int8)sub_C6A630(m128i_i8, v7, 0) )
  {
    sub_C6B0E0((__int64 *)&v15, **a1, (*a1)[1]);
    v9 = *a1;
    v10 = (__int64 *)**a1;
    if ( v15 == src )
    {
      v14 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v10 = src[0];
        else
          memcpy(v10, src, n);
        v14 = n;
        v10 = (__int64 *)*v9;
      }
      v9[1] = v14;
      *((_BYTE *)v10 + v14) = 0;
      v10 = v15;
      goto LABEL_12;
    }
    v11 = src[0];
    v12 = n;
    if ( v10 == v9 + 2 )
    {
      *v9 = (__int64)v15;
      v9[1] = v12;
      v9[2] = v11;
    }
    else
    {
      v13 = v9[2];
      *v9 = (__int64)v15;
      v9[1] = v12;
      v9[2] = v11;
      if ( v10 )
      {
        v15 = v10;
        src[0] = v13;
        goto LABEL_12;
      }
    }
    v15 = src;
    v10 = src;
LABEL_12:
    n = 0;
    *(_BYTE *)v10 = 0;
    if ( v15 != src )
      j_j___libc_free_0(v15, src[0] + 1LL);
  }
  result = (__int64 *)(*a1)[1];
  a1[1] = (__int64 *)**a1;
  a1[2] = result;
  return result;
}
