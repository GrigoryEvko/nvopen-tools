// Function: sub_136FB20
// Address: 0x136fb20
//
__m128i *__fastcall sub_136FB20(__m128i *a1, __int64 **a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // rdx
  void **v8; // rdi
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 *v12; // rsi
  __int64 **v13; // r14
  __int64 v14; // rbx
  __int64 i; // r14
  __int64 **v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rdi
  __m128i si128; // xmm0
  __int64 v27; // rax
  void *v28; // rdx
  size_t v29; // rdx
  unsigned int v32; // [rsp+2Ch] [rbp-104h] BYREF
  void *dest; // [rsp+30h] [rbp-100h] BYREF
  size_t v34; // [rsp+38h] [rbp-F8h]
  _QWORD v35[2]; // [rsp+40h] [rbp-F0h] BYREF
  __m128i *v36; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-D8h]
  __m128i v38; // [rsp+60h] [rbp-D0h] BYREF
  __int64 **p_p_dest; // [rsp+70h] [rbp-C0h] BYREF
  __int64 **v40; // [rsp+78h] [rbp-B8h]
  char v41; // [rsp+80h] [rbp-B0h]
  __int64 v42; // [rsp+88h] [rbp-A8h]
  __int64 v43[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-90h] BYREF
  void **p_dest; // [rsp+B0h] [rbp-80h] BYREF
  size_t n; // [rsp+B8h] [rbp-78h]
  _QWORD src[14]; // [rsp+C0h] [rbp-70h] BYREF

  sub_16E2FC0(&dest, a3);
  v7 = v34;
  p_dest = (void **)src;
  if ( v34 > 0x8C )
    v7 = 140;
  sub_1367D20((__int64 *)&p_dest, dest, (__int64)dest + v7);
  v8 = (void **)dest;
  if ( p_dest == src )
  {
    v29 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v29 = n;
      v8 = (void **)dest;
    }
    v34 = v29;
    *((_BYTE *)v8 + v29) = 0;
    v8 = p_dest;
  }
  else
  {
    if ( dest == v35 )
    {
      dest = p_dest;
      v34 = n;
      v35[0] = src[0];
    }
    else
    {
      v9 = v35[0];
      dest = p_dest;
      v34 = n;
      v35[0] = src[0];
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
  sub_16BEB10(&v36, &p_dest, &v32);
  v10 = v32;
  sub_16E8970(&p_dest, v32, 1, 0);
  if ( v32 == -1 )
  {
    v23 = sub_16E8CB0(&p_dest, v10, v11);
    v24 = *(__m128i **)(v23 + 24);
    v25 = v23;
    if ( *(_QWORD *)(v23 + 16) - (_QWORD)v24 <= 0x13u )
    {
      v25 = sub_16E7EE0(v23, "error opening file '", 20);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v24[1].m128i_i32[0] = 656434540;
      *v24 = si128;
      *(_QWORD *)(v23 + 24) += 20LL;
    }
    v27 = sub_16E7EE0(v25, v36->m128i_i8, v37);
    v28 = *(void **)(v27 + 24);
    if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 0xEu )
    {
      sub_16E7EE0(v27, "' for writing!\n", 15);
    }
    else
    {
      qmemcpy(v28, "' for writing!\n", 15);
      *(_QWORD *)(v27 + 24) += 15LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_1367D20(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    p_p_dest = (__int64 **)&p_dest;
    v40 = a2;
    v42 = 0;
    v41 = a4;
    sub_16E2FC0(v43, a5);
    v12 = v43;
    sub_136A2B0(&p_p_dest, v43);
    v13 = v40;
    v14 = sub_1368BD0(*v40) + 72;
    for ( i = *(_QWORD *)(sub_1368BD0(*v13) + 80); i != v14; i = *(_QWORD *)(i + 8) )
    {
      v12 = (__int64 *)(i - 24);
      if ( !i )
        v12 = 0;
      sub_136C8B0((__int64 *)&p_p_dest, (__int64)v12);
    }
    v16 = p_p_dest;
    v17 = p_p_dest[3];
    if ( (unsigned __int64)((char *)p_p_dest[2] - (char *)v17) <= 1 )
    {
      v12 = (__int64 *)"}\n";
      sub_16E7EE0(p_p_dest, "}\n", 2);
    }
    else
    {
      *(_WORD *)v17 = 2685;
      v16[3] = (__int64 *)((char *)v16[3] + 2);
    }
    v18 = v43[0];
    if ( (__int64 *)v43[0] != &v44 )
    {
      v12 = (__int64 *)(v44 + 1);
      j_j___libc_free_0(v43[0], v44 + 1);
    }
    v19 = sub_16E8CB0(v18, v12, v17);
    v20 = *(_QWORD **)(v19 + 24);
    if ( *(_QWORD *)(v19 + 16) - (_QWORD)v20 <= 7u )
    {
      sub_16E7EE0(v19, " done. \n", 8);
    }
    else
    {
      *v20 = 0xA202E656E6F6420LL;
      *(_QWORD *)(v19 + 24) += 8LL;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v36 == &v38 )
    {
      a1[1] = _mm_load_si128(&v38);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v36;
      a1[1].m128i_i64[0] = v38.m128i_i64[0];
    }
    v21 = v37;
    v36 = &v38;
    v37 = 0;
    a1->m128i_i64[1] = v21;
    v38.m128i_i8[0] = 0;
  }
  sub_16E7C30(&p_dest);
  if ( v36 != &v38 )
    j_j___libc_free_0(v36, v38.m128i_i64[0] + 1);
  if ( dest != v35 )
    j_j___libc_free_0(dest, v35[0] + 1LL);
  return a1;
}
