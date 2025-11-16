// Function: sub_38C1DA0
// Address: 0x38c1da0
//
unsigned __int64 __fastcall sub_38C1DA0(_QWORD *a1, __m128i *a2)
{
  __m128i *v2; // rax
  __m128i *v3; // rdx
  unsigned __int64 v4; // r14
  size_t v5; // r13
  __int32 v6; // eax
  __m128i v7; // xmm0
  __int32 v8; // eax
  __int64 v9; // r15
  const void *v10; // r12
  size_t v11; // rdx
  signed __int64 v12; // rax
  __int64 v13; // rax
  char v14; // dl
  size_t v15; // rbx
  const void *v16; // rsi
  size_t v17; // rcx
  size_t v18; // rbx
  const void *v19; // rdi
  const void *v20; // rsi
  char v21; // di
  bool v23; // al
  __int64 v24; // r12
  unsigned __int64 v25; // rdi
  __m128i *v26; // [rsp+0h] [rbp-70h]
  size_t v27; // [rsp+8h] [rbp-68h]
  unsigned __int32 v28; // [rsp+18h] [rbp-58h]
  __int32 v29; // [rsp+1Ch] [rbp-54h]
  __int64 m128i_i64; // [rsp+20h] [rbp-50h]
  _QWORD *v31; // [rsp+28h] [rbp-48h]
  size_t v32; // [rsp+30h] [rbp-40h]
  size_t v33; // [rsp+30h] [rbp-40h]

  v2 = (__m128i *)sub_22077B0(0x60u);
  v3 = (__m128i *)a2->m128i_i64[0];
  v4 = (unsigned __int64)v2;
  m128i_i64 = (__int64)v2[2].m128i_i64;
  v26 = v2 + 3;
  v2[2].m128i_i64[0] = (__int64)v2[3].m128i_i64;
  if ( v3 == &a2[1] )
  {
    v2[3] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    v2[2].m128i_i64[0] = (__int64)v3;
    v2[3].m128i_i64[0] = a2[1].m128i_i64[0];
  }
  v5 = a2->m128i_u64[1];
  a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
  v6 = a2[3].m128i_i32[0];
  a2[1].m128i_i8[0] = 0;
  a2->m128i_i64[1] = 0;
  v7 = _mm_loadu_si128(a2 + 2);
  v29 = v6;
  *(_DWORD *)(v4 + 80) = v6;
  v8 = a2[3].m128i_i32[1];
  *(_QWORD *)(v4 + 40) = v5;
  v28 = v8;
  *(_DWORD *)(v4 + 84) = v8;
  *(_QWORD *)(v4 + 88) = 0;
  v9 = a1[2];
  *(__m128i *)(v4 + 64) = v7;
  v31 = a1 + 1;
  if ( !v9 )
  {
    if ( v31 == (_QWORD *)a1[3] )
    {
      v9 = (__int64)(a1 + 1);
      v21 = 1;
      goto LABEL_29;
    }
    v9 = (__int64)(a1 + 1);
    goto LABEL_42;
  }
  v10 = *(const void **)(v4 + 32);
  while ( 1 )
  {
    v15 = *(_QWORD *)(v9 + 40);
    v16 = *(const void **)(v9 + 32);
    if ( v5 != v15 )
    {
      v11 = *(_QWORD *)(v9 + 40);
      if ( v5 <= v15 )
        v11 = v5;
      if ( !v11 || (LODWORD(v12) = memcmp(v10, v16, v11), !(_DWORD)v12) )
      {
        v12 = v5 - v15;
        if ( (__int64)(v5 - v15) >= 0x80000000LL )
          goto LABEL_23;
        if ( v12 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_12;
      }
LABEL_11:
      if ( (int)v12 >= 0 )
        goto LABEL_23;
      goto LABEL_12;
    }
    if ( v5 )
    {
      LODWORD(v12) = memcmp(v10, v16, v5);
      if ( (_DWORD)v12 )
        goto LABEL_11;
    }
    v17 = *(_QWORD *)(v4 + 72);
    v18 = *(_QWORD *)(v9 + 72);
    v19 = *(const void **)(v4 + 64);
    v20 = *(const void **)(v9 + 64);
    if ( v17 == v18 )
      break;
    if ( v17 <= v18 )
      goto LABEL_19;
    v33 = *(_QWORD *)(v4 + 72);
    if ( !v18 )
    {
LABEL_23:
      v13 = *(_QWORD *)(v9 + 24);
      v14 = 0;
      goto LABEL_24;
    }
    LODWORD(v12) = memcmp(v19, v20, v18);
    v17 = v33;
    if ( (_DWORD)v12 )
      goto LABEL_11;
LABEL_22:
    if ( v17 >= v18 )
      goto LABEL_23;
LABEL_12:
    v13 = *(_QWORD *)(v9 + 16);
    v14 = 1;
    if ( !v13 )
      goto LABEL_25;
LABEL_13:
    v9 = v13;
  }
  if ( v17 )
  {
    v27 = *(_QWORD *)(v4 + 72);
    if ( memcmp(v19, v20, v17) )
    {
      v17 = v27;
LABEL_19:
      if ( v17 )
      {
        v32 = v17;
        LODWORD(v12) = memcmp(v19, v20, v17);
        v17 = v32;
        if ( (_DWORD)v12 )
          goto LABEL_11;
      }
      if ( v17 == v18 )
        goto LABEL_23;
      goto LABEL_22;
    }
  }
  if ( v29 == *(_DWORD *)(v9 + 80) )
    v23 = v28 < *(_DWORD *)(v9 + 84);
  else
    v23 = v29 < *(_DWORD *)(v9 + 80);
  if ( v23 )
    goto LABEL_12;
  v13 = *(_QWORD *)(v9 + 24);
  v14 = 0;
LABEL_24:
  if ( v13 )
    goto LABEL_13;
LABEL_25:
  if ( v14 )
  {
    if ( a1[3] == v9 )
      goto LABEL_27;
LABEL_42:
    v24 = sub_220EF80(v9);
    if ( sub_38BCA60(v24 + 32, m128i_i64) )
      goto LABEL_27;
    v9 = v24;
    goto LABEL_44;
  }
  if ( !sub_38BCA60(v9 + 32, m128i_i64) )
  {
LABEL_44:
    v25 = *(_QWORD *)(v4 + 32);
    if ( v26 != (__m128i *)v25 )
      j_j___libc_free_0(v25);
    j_j___libc_free_0(v4);
    return v9;
  }
LABEL_27:
  v21 = 1;
  if ( v31 != (_QWORD *)v9 )
    v21 = sub_38BCA60(m128i_i64, v9 + 32);
LABEL_29:
  sub_220F040(v21, v4, (_QWORD *)v9, v31);
  ++a1[5];
  return v4;
}
