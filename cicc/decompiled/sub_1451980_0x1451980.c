// Function: sub_1451980
// Address: 0x1451980
//
__int64 __fastcall sub_1451980(__int64 a1, __int64 *a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v6; // rdx
  void **v7; // rdi
  __int64 v8; // r9
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  __int64 v16; // rax
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __m128i si128; // xmm0
  __int64 v20; // rax
  void *v21; // rdx
  size_t v22; // rdx
  unsigned int v26; // [rsp+2Ch] [rbp-C4h] BYREF
  void *dest; // [rsp+30h] [rbp-C0h] BYREF
  size_t v28; // [rsp+38h] [rbp-B8h]
  _QWORD v29[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i *v30; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+58h] [rbp-98h]
  __m128i v32; // [rsp+60h] [rbp-90h] BYREF
  void **p_dest; // [rsp+70h] [rbp-80h] BYREF
  size_t n; // [rsp+78h] [rbp-78h]
  _QWORD src[14]; // [rsp+80h] [rbp-70h] BYREF

  sub_16E2FC0(&dest, a3);
  v6 = v28;
  p_dest = (void **)src;
  if ( v28 > 0x8C )
    v6 = 140;
  sub_144C6E0((__int64 *)&p_dest, dest, (__int64)dest + v6);
  v7 = (void **)dest;
  if ( p_dest == src )
  {
    v22 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v22 = n;
      v7 = (void **)dest;
    }
    v28 = v22;
    *((_BYTE *)v7 + v22) = 0;
    v7 = p_dest;
  }
  else
  {
    if ( dest == v29 )
    {
      dest = p_dest;
      v28 = n;
      v29[0] = src[0];
    }
    else
    {
      v8 = v29[0];
      dest = p_dest;
      v28 = n;
      v29[0] = src[0];
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
  sub_16BEB10(&v30, &p_dest, &v26);
  v9 = v26;
  sub_16E8970(&p_dest, v26, 1, 0);
  if ( v26 == -1 )
  {
    v16 = sub_16E8CB0(&p_dest, v9, v10);
    v17 = *(__m128i **)(v16 + 24);
    v18 = v16;
    if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0x13u )
    {
      v18 = sub_16E7EE0(v16, "error opening file '", 20);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v17[1].m128i_i32[0] = 656434540;
      *v17 = si128;
      *(_QWORD *)(v16 + 24) += 20LL;
    }
    v20 = sub_16E7EE0(v18, v30->m128i_i8, v31);
    v21 = *(void **)(v20 + 24);
    if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 0xEu )
    {
      sub_16E7EE0(v20, "' for writing!\n", 15);
    }
    else
    {
      qmemcpy(v21, "' for writing!\n", 15);
      *(_QWORD *)(v20 + 24) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    sub_14507E0((__int64)&p_dest, a2, a4, a5);
    v12 = sub_16E8CB0(&p_dest, a2, v11);
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 <= 7u )
    {
      sub_16E7EE0(v12, " done. \n", 8);
    }
    else
    {
      *v13 = 0xA202E656E6F6420LL;
      *(_QWORD *)(v12 + 24) += 8LL;
    }
    *(_QWORD *)a1 = a1 + 16;
    if ( v30 == &v32 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v32);
    }
    else
    {
      *(_QWORD *)a1 = v30;
      *(_QWORD *)(a1 + 16) = v32.m128i_i64[0];
    }
    v14 = v31;
    v30 = &v32;
    v31 = 0;
    *(_QWORD *)(a1 + 8) = v14;
    v32.m128i_i8[0] = 0;
  }
  sub_16E7C30(&p_dest);
  if ( v30 != &v32 )
    j_j___libc_free_0(v30, v32.m128i_i64[0] + 1);
  if ( dest != v29 )
    j_j___libc_free_0(dest, v29[0] + 1LL);
  return a1;
}
