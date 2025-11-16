// Function: sub_1DE2730
// Address: 0x1de2730
//
__m128i *__fastcall sub_1DE2730(__m128i *a1, __int64 *a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // rdx
  void **v8; // rdi
  __int64 v9; // r9
  __int64 v10; // r8
  const char *v11; // rsi
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 i; // r15
  __int64 *v15; // rdi
  _WORD *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  size_t v19; // rax
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i si128; // xmm0
  __int64 v25; // rax
  void *v26; // rdx
  size_t v27; // rdx
  unsigned int v29; // [rsp+2Ch] [rbp-134h] BYREF
  void *dest; // [rsp+30h] [rbp-130h] BYREF
  size_t v31; // [rsp+38h] [rbp-128h]
  _QWORD v32[2]; // [rsp+40h] [rbp-120h] BYREF
  __m128i *v33; // [rsp+50h] [rbp-110h] BYREF
  size_t v34; // [rsp+58h] [rbp-108h]
  __m128i v35; // [rsp+60h] [rbp-100h] BYREF
  __int64 v36[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v37; // [rsp+80h] [rbp-E0h] BYREF
  __int64 *p_p_dest; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v39; // [rsp+98h] [rbp-C8h]
  char v40; // [rsp+A0h] [rbp-C0h]
  __int64 v41; // [rsp+A8h] [rbp-B8h]
  __int64 v42; // [rsp+B0h] [rbp-B0h]
  __int64 v43; // [rsp+B8h] [rbp-A8h]
  __int64 v44; // [rsp+C0h] [rbp-A0h]
  __int64 v45; // [rsp+C8h] [rbp-98h]
  __int64 v46; // [rsp+D0h] [rbp-90h]
  void **p_dest; // [rsp+E0h] [rbp-80h] BYREF
  size_t n; // [rsp+E8h] [rbp-78h]
  _QWORD src[14]; // [rsp+F0h] [rbp-70h] BYREF

  sub_16E2FC0((__int64 *)&dest, a3);
  v7 = v31;
  p_dest = (void **)src;
  if ( v31 > 0x8C )
    v7 = 140;
  sub_1DDBCE0((__int64 *)&p_dest, dest, (__int64)dest + v7);
  v8 = (void **)dest;
  if ( p_dest == src )
  {
    v27 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v27 = n;
      v8 = (void **)dest;
    }
    v31 = v27;
    *((_BYTE *)v8 + v27) = 0;
    v8 = p_dest;
  }
  else
  {
    if ( dest == v32 )
    {
      dest = p_dest;
      v31 = n;
      v32[0] = src[0];
    }
    else
    {
      v9 = v32[0];
      dest = p_dest;
      v31 = n;
      v32[0] = src[0];
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
  sub_16BEB10((__int64)&v33, (__int64)&p_dest, &v29);
  sub_16E8970((__int64)&p_dest, v29, 1, 0, v10);
  if ( v29 == -1 )
  {
    v21 = sub_16E8CB0();
    v22 = (__m128i *)v21[3];
    v23 = (__int64)v21;
    if ( v21[2] - (_QWORD)v22 <= 0x13u )
    {
      v23 = sub_16E7EE0((__int64)v21, "error opening file '", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v22[1].m128i_i32[0] = 656434540;
      *v22 = si128;
      v21[3] += 20LL;
    }
    v25 = sub_16E7EE0(v23, v33->m128i_i8, v34);
    v26 = *(void **)(v25 + 24);
    if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 0xEu )
    {
      sub_16E7EE0(v25, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v26, "' for writing!\n", 15);
      *(_QWORD *)(v25 + 24) += 15LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    v11 = byte_3F871B3;
    sub_1DDBCE0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    p_p_dest = (__int64 *)&p_dest;
    v42 = 0;
    v39 = a2;
    v43 = 0;
    v40 = a4;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v41 = 0;
    j___libc_free_0(0);
    ++v43;
    v44 = 0;
    v45 = 0;
    LODWORD(v46) = 0;
    j___libc_free_0(0);
    sub_16E2FC0(v36, a5);
    v11 = (const char *)v36;
    sub_1DDD170(&p_p_dest, v36);
    v12 = v39;
    v13 = sub_1DDC510(*v39) + 320;
    for ( i = *(_QWORD *)(sub_1DDC510(*v12) + 328); i != v13; i = *(_QWORD *)(i + 8) )
    {
      v11 = (const char *)i;
      sub_1DDF8C0((__int64)&p_p_dest, i);
    }
    v15 = p_p_dest;
    v16 = (_WORD *)p_p_dest[3];
    if ( (unsigned __int64)(p_p_dest[2] - (_QWORD)v16) <= 1 )
    {
      v11 = "}\n";
      sub_16E7EE0((__int64)p_p_dest, "}\n", 2u);
    }
    else
    {
      *v16 = 2685;
      v15[3] += 2;
    }
    if ( (__int64 *)v36[0] != &v37 )
    {
      v11 = (const char *)(v37 + 1);
      j_j___libc_free_0(v36[0], v37 + 1);
    }
    j___libc_free_0(v44);
    v17 = sub_16E8CB0();
    v18 = (_QWORD *)v17[3];
    if ( v17[2] - (_QWORD)v18 <= 7u )
    {
      v11 = " done. \n";
      sub_16E7EE0((__int64)v17, " done. \n", 8u);
    }
    else
    {
      *v18 = 0xA202E656E6F6420LL;
      v17[3] += 8LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v33 == &v35 )
    {
      a1[1] = _mm_load_si128(&v35);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v33;
      a1[1].m128i_i64[0] = v35.m128i_i64[0];
    }
    v19 = v34;
    v33 = &v35;
    v34 = 0;
    a1->m128i_i64[1] = v19;
    v35.m128i_i8[0] = 0;
  }
  sub_16E7C30((int *)&p_dest, (__int64)v11);
  if ( v33 != &v35 )
    j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
  if ( dest != v32 )
    j_j___libc_free_0(dest, v32[0] + 1LL);
  return a1;
}
