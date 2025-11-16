// Function: sub_13BB510
// Address: 0x13bb510
//
__int64 __fastcall sub_13BB510(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v6; // rdx
  void **v7; // rdi
  __int64 v8; // r9
  __int64 v9; // rsi
  __int64 v10; // rdx
  const char *v11; // rsi
  __int64 v12; // rdi
  _WORD *v13; // rdx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rdi
  __m128i si128; // xmm0
  __int64 v23; // rax
  void *v24; // rdx
  size_t v25; // rdx
  unsigned int v29; // [rsp+2Ch] [rbp-104h] BYREF
  __int64 v30[2]; // [rsp+30h] [rbp-100h] BYREF
  char v31; // [rsp+40h] [rbp-F0h]
  void *dest; // [rsp+50h] [rbp-E0h] BYREF
  size_t v33; // [rsp+58h] [rbp-D8h]
  _QWORD v34[2]; // [rsp+60h] [rbp-D0h] BYREF
  __m128i *v35; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+78h] [rbp-B8h]
  __m128i v37; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v38[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-90h] BYREF
  void **p_dest; // [rsp+B0h] [rbp-80h] BYREF
  size_t n; // [rsp+B8h] [rbp-78h]
  _QWORD src[14]; // [rsp+C0h] [rbp-70h] BYREF

  sub_16E2FC0(&dest, a3);
  v6 = v33;
  p_dest = (void **)src;
  if ( v33 > 0x8C )
    v6 = 140;
  sub_13B5840((__int64 *)&p_dest, dest, (__int64)dest + v6);
  v7 = (void **)dest;
  if ( p_dest == src )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v25 = n;
      v7 = (void **)dest;
    }
    v33 = v25;
    *((_BYTE *)v7 + v25) = 0;
    v7 = p_dest;
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
      v8 = v34[0];
      dest = p_dest;
      v33 = n;
      v34[0] = src[0];
      if ( v7 )
      {
        p_dest = v7;
        src[0] = v8;
        goto LABEL_7;
      }
    }
    p_dest = (void **)src;
    v7 = (void **)src;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v7 = 0;
  if ( p_dest != src )
    j_j___libc_free_0(p_dest, src[0] + 1LL);
  p_dest = &dest;
  LOWORD(src[0]) = 260;
  sub_16BEB10(&v35, &p_dest, &v29);
  v9 = v29;
  sub_16E8970(&p_dest, v29, 1, 0);
  if ( v29 == -1 )
  {
    v19 = sub_16E8CB0(&p_dest, v9, v10);
    v20 = *(__m128i **)(v19 + 24);
    v21 = v19;
    if ( *(_QWORD *)(v19 + 16) - (_QWORD)v20 <= 0x13u )
    {
      v21 = sub_16E7EE0(v19, "error opening file '", 20);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v20[1].m128i_i32[0] = 656434540;
      *v20 = si128;
      *(_QWORD *)(v19 + 24) += 20LL;
    }
    v23 = sub_16E7EE0(v21, v35->m128i_i8, v36);
    v24 = *(void **)(v23 + 24);
    if ( *(_QWORD *)(v23 + 16) - (_QWORD)v24 <= 0xEu )
    {
      sub_16E7EE0(v23, "' for writing!\n", 15);
    }
    else
    {
      qmemcpy(v24, "' for writing!\n", 15);
      *(_QWORD *)(v23 + 24) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v30[0] = (__int64)&p_dest;
    v30[1] = a2;
    v31 = a4;
    sub_16E2FC0(v38, a5);
    v11 = (const char *)v38;
    sub_13B8800(v30, v38);
    sub_13BA9D0((__int64)v30);
    v12 = v30[0];
    v13 = *(_WORD **)(v30[0] + 24);
    if ( *(_QWORD *)(v30[0] + 16) - (_QWORD)v13 <= 1u )
    {
      v11 = "}\n";
      sub_16E7EE0(v30[0], "}\n", 2);
    }
    else
    {
      *v13 = 2685;
      *(_QWORD *)(v12 + 24) += 2LL;
    }
    v14 = v38[0];
    if ( v38[0] != &v39 )
    {
      v11 = (const char *)(v39 + 1);
      j_j___libc_free_0(v38[0], v39 + 1);
    }
    v15 = sub_16E8CB0(v14, v11, v13);
    v16 = *(_QWORD **)(v15 + 24);
    if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 7u )
    {
      sub_16E7EE0(v15, " done. \n", 8);
    }
    else
    {
      *v16 = 0xA202E656E6F6420LL;
      *(_QWORD *)(v15 + 24) += 8LL;
    }
    *(_QWORD *)a1 = a1 + 16;
    if ( v35 == &v37 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v37);
    }
    else
    {
      *(_QWORD *)a1 = v35;
      *(_QWORD *)(a1 + 16) = v37.m128i_i64[0];
    }
    v17 = v36;
    v35 = &v37;
    v36 = 0;
    *(_QWORD *)(a1 + 8) = v17;
    v37.m128i_i8[0] = 0;
  }
  sub_16E7C30(&p_dest);
  if ( v35 != &v37 )
    j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
  if ( dest != v34 )
    j_j___libc_free_0(dest, v34[0] + 1LL);
  return a1;
}
