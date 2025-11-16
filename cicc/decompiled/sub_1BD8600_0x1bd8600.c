// Function: sub_1BD8600
// Address: 0x1bd8600
//
__m128i *__fastcall sub_1BD8600(__m128i *a1, const char ***a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // rdx
  void **v8; // rdi
  __int64 v9; // r9
  __int64 v10; // r8
  const char *v11; // rsi
  const char *v12; // rbx
  const char *i; // r15
  __int64 v14; // rdi
  _WORD *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  size_t v18; // rax
  _QWORD *v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rdi
  __m128i si128; // xmm0
  __int64 v24; // rax
  void *v25; // rdx
  size_t v26; // rdx
  unsigned int v28; // [rsp+2Ch] [rbp-104h] BYREF
  __int64 *p_p_dest; // [rsp+30h] [rbp-100h] BYREF
  const char ***v30; // [rsp+38h] [rbp-F8h]
  char v31; // [rsp+40h] [rbp-F0h]
  void *dest; // [rsp+50h] [rbp-E0h] BYREF
  size_t v33; // [rsp+58h] [rbp-D8h]
  _QWORD v34[2]; // [rsp+60h] [rbp-D0h] BYREF
  __m128i *v35; // [rsp+70h] [rbp-C0h] BYREF
  size_t v36; // [rsp+78h] [rbp-B8h]
  __m128i v37; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v38[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-90h] BYREF
  void **p_dest; // [rsp+B0h] [rbp-80h] BYREF
  size_t n; // [rsp+B8h] [rbp-78h]
  _QWORD src[14]; // [rsp+C0h] [rbp-70h] BYREF

  sub_16E2FC0((__int64 *)&dest, a3);
  v7 = v33;
  p_dest = (void **)src;
  if ( v33 > 0x8C )
    v7 = 140;
  sub_1BB98B0((__int64 *)&p_dest, dest, (__int64)dest + v7);
  v8 = (void **)dest;
  if ( p_dest == src )
  {
    v26 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v26 = n;
      v8 = (void **)dest;
    }
    v33 = v26;
    *((_BYTE *)v8 + v26) = 0;
    v8 = p_dest;
  }
  else
  {
    if ( dest == v34 )
    {
      dest = p_dest;
      v33 = n;
      v34[0] = src[0];
    }
    else
    {
      v9 = v34[0];
      dest = p_dest;
      v33 = n;
      v34[0] = src[0];
      if ( v8 )
      {
        p_dest = v8;
        src[0] = v9;
        goto LABEL_7;
      }
    }
    p_dest = (void **)src;
    v8 = (void **)src;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( p_dest != src )
    j_j___libc_free_0(p_dest, src[0] + 1LL);
  p_dest = &dest;
  LOWORD(src[0]) = 260;
  sub_16BEB10((__int64)&v35, (__int64)&p_dest, &v28);
  sub_16E8970((__int64)&p_dest, v28, 1, 0, v10);
  if ( v28 == -1 )
  {
    v20 = sub_16E8CB0();
    v21 = (__m128i *)v20[3];
    v22 = (__int64)v20;
    if ( v20[2] - (_QWORD)v21 <= 0x13u )
    {
      v22 = sub_16E7EE0((__int64)v20, "error opening file '", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v21[1].m128i_i32[0] = 656434540;
      *v21 = si128;
      v20[3] += 20LL;
    }
    v24 = sub_16E7EE0(v22, v35->m128i_i8, v36);
    v25 = *(void **)(v24 + 24);
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 0xEu )
    {
      sub_16E7EE0(v24, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v25, "' for writing!\n", 15);
      *(_QWORD *)(v24 + 24) += 15LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    v11 = byte_3F871B3;
    sub_1BB98B0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    p_p_dest = (__int64 *)&p_dest;
    v30 = a2;
    v31 = a4;
    sub_16E2FC0(v38, a5);
    v11 = (const char *)v38;
    sub_1BC61A0((__int64 *)&p_p_dest, v38);
    v12 = (*v30)[1];
    for ( i = **v30; v12 != i; i += 176 )
    {
      v11 = i;
      sub_1BCAF90((__int64)&p_p_dest, (__int64)v11);
    }
    v14 = (__int64)p_p_dest;
    v15 = (_WORD *)p_p_dest[3];
    if ( (unsigned __int64)(p_p_dest[2] - (_QWORD)v15) <= 1 )
    {
      v11 = "}\n";
      sub_16E7EE0((__int64)p_p_dest, "}\n", 2u);
    }
    else
    {
      *v15 = 2685;
      *(_QWORD *)(v14 + 24) += 2LL;
    }
    if ( (__int64 *)v38[0] != &v39 )
    {
      v11 = (const char *)(v39 + 1);
      j_j___libc_free_0(v38[0], v39 + 1);
    }
    v16 = sub_16E8CB0();
    v17 = (_QWORD *)v16[3];
    if ( v16[2] - (_QWORD)v17 <= 7u )
    {
      v11 = " done. \n";
      sub_16E7EE0((__int64)v16, " done. \n", 8u);
    }
    else
    {
      *v17 = 0xA202E656E6F6420LL;
      v16[3] += 8LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v35 == &v37 )
    {
      a1[1] = _mm_load_si128(&v37);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v35;
      a1[1].m128i_i64[0] = v37.m128i_i64[0];
    }
    v18 = v36;
    v35 = &v37;
    v36 = 0;
    a1->m128i_i64[1] = v18;
    v37.m128i_i8[0] = 0;
  }
  sub_16E7C30((int *)&p_dest, (__int64)v11);
  if ( v35 != &v37 )
    j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
  if ( dest != v34 )
    j_j___libc_free_0(dest, v34[0] + 1LL);
  return a1;
}
