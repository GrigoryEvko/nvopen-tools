// Function: sub_CA44E0
// Address: 0xca44e0
//
__int64 __fastcall sub_CA44E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r14d
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // rdx
  _BYTE *v8; // rsi
  __m128i *v9; // rcx
  __int64 v10; // rsi
  size_t v11; // rdx
  __m128i *p_src; // rax
  __m128i *v13; // r8
  __int64 v14; // rdi
  __m128i *v15; // rdi
  char v16; // bl
  _BYTE *v18; // rax
  __m128i v19; // xmm1
  int v20; // [rsp+Ch] [rbp-B4h]
  __m128i v21; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v22; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v23; // [rsp+30h] [rbp-90h] BYREF
  __m128i src; // [rsp+40h] [rbp-80h] BYREF
  __int128 v25; // [rsp+50h] [rbp-70h]
  __int128 v26; // [rsp+60h] [rbp-60h]
  __int128 v27; // [rsp+70h] [rbp-50h]
  int v28; // [rsp+80h] [rbp-40h]
  int v29; // [rsp+84h] [rbp-3Ch]

  v3 = sub_C82D80(*(DIR ***)(a1 + 48), a2);
  v4 = *(_QWORD *)(a1 + 48);
  if ( !v4 )
    goto LABEL_14;
  v28 = 0;
  v25 = 0;
  v23.m128i_i64[0] = (__int64)&src;
  v23.m128i_i64[1] = 0;
  src.m128i_i8[0] = 0;
  LODWORD(v25) = 9;
  BYTE4(v25) = 1;
  v29 = 0xFFFF;
  v26 = 0;
  v27 = 0;
  if ( !*(_QWORD *)(v4 + 16) )
  {
    sub_2240A30(&v23);
LABEL_14:
    v18 = *(_BYTE **)(a1 + 8);
    v23.m128i_i64[0] = (__int64)&src;
    *(_QWORD *)&v25 = 9;
    src.m128i_i8[0] = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *v18 = 0;
    v23.m128i_i64[1] = 0;
    *(_BYTE *)v23.m128i_i64[0] = 0;
    v15 = (__m128i *)v23.m128i_i64[0];
    *(_DWORD *)(a1 + 40) = v25;
    if ( v15 == &src )
      return v3;
    v16 = 0;
LABEL_11:
    j_j___libc_free_0(v15, src.m128i_i64[0] + 1);
    if ( !v16 )
      return v3;
    goto LABEL_18;
  }
  sub_2240A30(&v23);
  v5 = *(_QWORD *)(a1 + 48);
  v6 = *(_DWORD *)(v5 + 40);
  if ( v6 == 9 )
  {
    sub_C832B0(&v23, v5 + 8);
    v6 = 9;
    v5 = *(_QWORD *)(a1 + 48);
    if ( (v26 & 1) == 0 )
      v6 = DWORD2(v25);
  }
  v20 = v6;
  v7 = *(_QWORD *)(v5 + 8) + *(_QWORD *)(v5 + 16);
  v8 = *(_BYTE **)(v5 + 8);
  v21.m128i_i64[0] = (__int64)&v22;
  sub_CA1F00(v21.m128i_i64, v8, v7);
  v9 = (__m128i *)v21.m128i_i64[0];
  v23.m128i_i64[0] = (__int64)&src;
  if ( (__m128i *)v21.m128i_i64[0] == &v22 )
  {
    v11 = v21.m128i_u64[1];
    v19 = _mm_load_si128(&v22);
    LODWORD(v25) = v20;
    v21.m128i_i64[1] = 0;
    v13 = *(__m128i **)(a1 + 8);
    v23.m128i_i64[1] = v11;
    v22.m128i_i8[0] = 0;
    src = v19;
  }
  else
  {
    v10 = v22.m128i_i64[0];
    LODWORD(v25) = v20;
    v11 = v21.m128i_u64[1];
    p_src = *(__m128i **)(a1 + 8);
    v23 = v21;
    src.m128i_i64[0] = v22.m128i_i64[0];
    v13 = p_src;
    v21.m128i_i64[0] = (__int64)&v22;
    v21.m128i_i64[1] = 0;
    v22.m128i_i8[0] = 0;
    if ( v9 != &src )
    {
      if ( p_src == (__m128i *)(a1 + 24) )
      {
        *(_QWORD *)(a1 + 8) = v9;
        *(_QWORD *)(a1 + 16) = v11;
        *(_QWORD *)(a1 + 24) = v10;
      }
      else
      {
        v14 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 8) = v9;
        *(_QWORD *)(a1 + 16) = v11;
        *(_QWORD *)(a1 + 24) = v10;
        if ( p_src )
        {
          v23.m128i_i64[0] = (__int64)p_src;
          src.m128i_i64[0] = v14;
LABEL_9:
          v23.m128i_i64[1] = 0;
          goto LABEL_10;
        }
      }
      v23.m128i_i64[0] = (__int64)&src;
      p_src = &src;
      goto LABEL_9;
    }
  }
  if ( v11 )
  {
    if ( v11 == 1 )
      v13->m128i_i8[0] = src.m128i_i8[0];
    else
      memcpy(v13, &src, v11);
    v11 = v23.m128i_u64[1];
    v13 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 16) = v11;
  v13->m128i_i8[v11] = 0;
  p_src = (__m128i *)v23.m128i_i64[0];
  v23.m128i_i64[1] = 0;
LABEL_10:
  p_src->m128i_i8[0] = 0;
  v15 = (__m128i *)v23.m128i_i64[0];
  *(_DWORD *)(a1 + 40) = v25;
  v16 = 1;
  if ( v15 != &src )
    goto LABEL_11;
LABEL_18:
  if ( (__m128i *)v21.m128i_i64[0] != &v22 )
    j_j___libc_free_0(v21.m128i_i64[0], v22.m128i_i64[0] + 1);
  return v3;
}
