// Function: sub_20E9B20
// Address: 0x20e9b20
//
__m128i *__fastcall sub_20E9B20(__m128i *a1, const char *a2, __int64 a3)
{
  __int64 v4; // rdx
  void **v5; // rdi
  __int64 v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  size_t v10; // rax
  _QWORD *v12; // rax
  __m128i *v13; // rdx
  __int64 v14; // rdi
  __m128i si128; // xmm0
  __int64 v16; // rax
  void *v17; // rdx
  size_t v18; // rdx
  unsigned int v19; // [rsp+2Ch] [rbp-C4h] BYREF
  void *dest; // [rsp+30h] [rbp-C0h] BYREF
  size_t v21; // [rsp+38h] [rbp-B8h]
  _QWORD v22[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i *v23; // [rsp+50h] [rbp-A0h] BYREF
  size_t v24; // [rsp+58h] [rbp-98h]
  __m128i v25; // [rsp+60h] [rbp-90h] BYREF
  void **p_dest; // [rsp+70h] [rbp-80h] BYREF
  size_t n; // [rsp+78h] [rbp-78h]
  _QWORD src[14]; // [rsp+80h] [rbp-70h] BYREF

  sub_16E2FC0((__int64 *)&dest, a3);
  v4 = v21;
  p_dest = (void **)src;
  if ( v21 > 0x8C )
    v4 = 140;
  sub_20E8F10((__int64 *)&p_dest, dest, (__int64)dest + v4);
  v5 = (void **)dest;
  if ( p_dest == src )
  {
    v18 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v18 = n;
      v5 = (void **)dest;
    }
    v21 = v18;
    *((_BYTE *)v5 + v18) = 0;
    v5 = p_dest;
  }
  else
  {
    if ( dest == v22 )
    {
      dest = p_dest;
      v21 = n;
      v22[0] = src[0];
    }
    else
    {
      v6 = v22[0];
      dest = p_dest;
      v21 = n;
      v22[0] = src[0];
      if ( v5 )
      {
        p_dest = v5;
        src[0] = v6;
        goto LABEL_7;
      }
    }
    p_dest = (void **)src;
    v5 = (void **)src;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v5 = 0;
  if ( p_dest != src )
    j_j___libc_free_0(p_dest, src[0] + 1LL);
  p_dest = &dest;
  LOWORD(src[0]) = 260;
  sub_16BEB10((__int64)&v23, (__int64)&p_dest, &v19);
  sub_16E8970((__int64)&p_dest, v19, 1, 0, v7);
  if ( v19 == -1 )
  {
    v12 = sub_16E8CB0();
    v13 = (__m128i *)v12[3];
    v14 = (__int64)v12;
    if ( v12[2] - (_QWORD)v13 <= 0x13u )
    {
      v14 = sub_16E7EE0((__int64)v12, "error opening file '", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v13[1].m128i_i32[0] = 656434540;
      *v13 = si128;
      v12[3] += 20LL;
    }
    v16 = sub_16E7EE0(v14, v23->m128i_i8, v24);
    v17 = *(void **)(v16 + 24);
    if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0xEu )
    {
      sub_16E7EE0(v16, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v17, "' for writing!\n", 15);
      *(_QWORD *)(v16 + 24) += 15LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    a2 = byte_3F871B3;
    sub_20E8F10(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    sub_20E9520((__int64)&p_dest, (__int64)a2);
    v8 = sub_16E8CB0();
    v9 = (_QWORD *)v8[3];
    if ( v8[2] - (_QWORD)v9 <= 7u )
    {
      a2 = " done. \n";
      sub_16E7EE0((__int64)v8, " done. \n", 8u);
    }
    else
    {
      *v9 = 0xA202E656E6F6420LL;
      v8[3] += 8LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v23 == &v25 )
    {
      a1[1] = _mm_load_si128(&v25);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v23;
      a1[1].m128i_i64[0] = v25.m128i_i64[0];
    }
    v10 = v24;
    v23 = &v25;
    v24 = 0;
    a1->m128i_i64[1] = v10;
    v25.m128i_i8[0] = 0;
  }
  sub_16E7C30((int *)&p_dest, (__int64)a2);
  if ( v23 != &v25 )
    j_j___libc_free_0(v23, v25.m128i_i64[0] + 1);
  if ( dest != v22 )
    j_j___libc_free_0(dest, v22[0] + 1LL);
  return a1;
}
