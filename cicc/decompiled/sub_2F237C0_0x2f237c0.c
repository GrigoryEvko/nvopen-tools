// Function: sub_2F237C0
// Address: 0x2f237c0
//
void __fastcall sub_2F237C0(__int64 a1, __int64 *a2)
{
  _QWORD *v3; // rdi
  size_t v5; // rax
  __m128i *v6; // r14
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 *v9; // rax
  _BYTE *v10; // rax
  __m128i *v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // rdx
  size_t v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h]
  __m128i *p_src; // [rsp+8h] [rbp-58h]
  size_t n; // [rsp+10h] [rbp-50h]
  __m128i src; // [rsp+18h] [rbp-48h] BYREF
  __m128i v21; // [rsp+28h] [rbp-38h] BYREF
  int v22; // [rsp+38h] [rbp-28h]

  v3 = (_QWORD *)(a1 + 24);
  p_src = &src;
  v17 = *(v3 - 3);
  if ( (_QWORD *)*(v3 - 2) == v3 )
  {
    src = _mm_loadu_si128((const __m128i *)(a1 + 24));
  }
  else
  {
    p_src = (__m128i *)*(v3 - 2);
    src.m128i_i64[0] = *(_QWORD *)(a1 + 24);
  }
  v5 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = v3;
  v6 = (__m128i *)(a2 + 3);
  *(_QWORD *)(a1 + 16) = 0;
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  n = v5;
  LODWORD(v5) = *(_DWORD *)(a1 + 56);
  *(_BYTE *)(a1 + 24) = 0;
  v22 = v5;
  v8 = *a2;
  v21 = v7;
  *(_QWORD *)a1 = v8;
  v9 = (__int64 *)a2[1];
  if ( v9 == a2 + 3 )
  {
    v16 = a2[2];
    if ( v16 )
    {
      if ( v16 == 1 )
        *(_BYTE *)(a1 + 24) = *((_BYTE *)a2 + 24);
      else
        memcpy(v3, a2 + 3, v16);
      v16 = a2[2];
    }
    *(_QWORD *)(a1 + 16) = v16;
    *(_BYTE *)(a1 + v16 + 24) = 0;
    v10 = (_BYTE *)a2[1];
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = a2[2];
    *(_QWORD *)(a1 + 24) = a2[3];
    v10 = a2 + 3;
    a2[1] = (__int64)v6;
  }
  a2[2] = 0;
  *v10 = 0;
  *(__m128i *)(a1 + 40) = _mm_loadu_si128((const __m128i *)(a2 + 5));
  *(_DWORD *)(a1 + 56) = *((_DWORD *)a2 + 14);
  v11 = (__m128i *)a2[1];
  *a2 = v17;
  if ( p_src == &src )
  {
    v15 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        v11->m128i_i8[0] = src.m128i_i8[0];
        v15 = 1;
      }
      else
      {
        memcpy(v11, &src, n);
        v15 = n;
      }
      v11 = (__m128i *)a2[1];
    }
    a2[2] = v15;
    v11->m128i_i8[v15] = 0;
    v11 = p_src;
  }
  else
  {
    v12 = src.m128i_i64[0];
    if ( v11 == v6 )
    {
      a2[1] = (__int64)p_src;
      a2[2] = n;
      a2[3] = v12;
    }
    else
    {
      v13 = a2[3];
      a2[1] = (__int64)p_src;
      a2[2] = n;
      a2[3] = v12;
      if ( v11 )
      {
        p_src = v11;
        src.m128i_i64[0] = v13;
        goto LABEL_9;
      }
    }
    p_src = &src;
    v11 = &src;
  }
LABEL_9:
  v11->m128i_i8[0] = 0;
  v14 = v22;
  *(__m128i *)(a2 + 5) = _mm_loadu_si128(&v21);
  *((_DWORD *)a2 + 14) = v14;
  if ( p_src != &src )
    j_j___libc_free_0((unsigned __int64)p_src);
}
