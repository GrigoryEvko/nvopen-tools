// Function: sub_16F25B0
// Address: 0x16f25b0
//
unsigned __int8 *__fastcall sub_16F25B0(unsigned __int8 ***a1, __int64 a2)
{
  __m128i *v4; // rax
  __int64 m128i_i64; // rcx
  int v6; // r8d
  unsigned __int8 *v7; // rdi
  __int64 v8; // rsi
  unsigned __int8 *result; // rax
  unsigned __int8 **v10; // rbx
  unsigned __int8 *v11; // rdi
  unsigned __int8 *v12; // rdx
  size_t v13; // rcx
  unsigned __int8 *v14; // rsi
  size_t v15; // rdx
  unsigned __int8 *v16; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = (__m128i *)sub_22077B0(32);
  if ( v4 )
  {
    v7 = *(unsigned __int8 **)a2;
    m128i_i64 = (__int64)v4[1].m128i_i64;
    v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
    if ( v7 == (unsigned __int8 *)(a2 + 16) )
    {
      v7 = (unsigned __int8 *)&v4[1];
      v4[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
    }
    else
    {
      m128i_i64 = *(_QWORD *)(a2 + 16);
      v4->m128i_i64[0] = (__int64)v7;
      v4[1].m128i_i64[0] = m128i_i64;
    }
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)a2 = a2 + 16;
    *(_QWORD *)(a2 + 8) = 0;
    v4->m128i_i64[1] = v8;
    *(_BYTE *)(a2 + 16) = 0;
  }
  else
  {
    v8 = MEMORY[8];
    v7 = (unsigned __int8 *)MEMORY[0];
  }
  *a1 = (unsigned __int8 **)v4;
  a1[1] = 0;
  a1[2] = 0;
  if ( !(unsigned __int8)sub_16F23B0(v7, v8, 0, m128i_i64, v6) )
  {
    sub_16F2420((__int64 *)&v16, **a1, (unsigned __int64)(*a1)[1]);
    v10 = *a1;
    v11 = **a1;
    if ( v16 == (unsigned __int8 *)src )
    {
      v15 = n;
      if ( n )
      {
        if ( n == 1 )
          *v11 = src[0];
        else
          memcpy(v11, src, n);
        v15 = n;
        v11 = *v10;
      }
      v10[1] = (unsigned __int8 *)v15;
      v11[v15] = 0;
      v11 = v16;
      goto LABEL_12;
    }
    v12 = (unsigned __int8 *)src[0];
    v13 = n;
    if ( v11 == (unsigned __int8 *)(v10 + 2) )
    {
      *v10 = v16;
      v10[1] = (unsigned __int8 *)v13;
      v10[2] = v12;
    }
    else
    {
      v14 = v10[2];
      *v10 = v16;
      v10[1] = (unsigned __int8 *)v13;
      v10[2] = v12;
      if ( v11 )
      {
        v16 = v11;
        src[0] = v14;
        goto LABEL_12;
      }
    }
    v16 = (unsigned __int8 *)src;
    v11 = (unsigned __int8 *)src;
LABEL_12:
    n = 0;
    *v11 = 0;
    if ( v16 != (unsigned __int8 *)src )
      j_j___libc_free_0(v16, src[0] + 1LL);
  }
  result = **a1;
  a1[2] = (unsigned __int8 **)(*a1)[1];
  a1[1] = (unsigned __int8 **)result;
  return result;
}
