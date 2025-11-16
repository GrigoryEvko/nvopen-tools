// Function: sub_3942120
// Address: 0x3942120
//
__int64 __fastcall sub_3942120(_QWORD *a1, _QWORD *a2, __m128i **a3)
{
  _QWORD *v5; // rbx
  __int64 v6; // r12
  __m128i *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // r15
  _QWORD *v12; // r13
  _QWORD *v13; // rcx
  size_t v15; // r14
  size_t v16; // r15
  size_t v17; // rdx
  int v18; // eax
  unsigned int v19; // edi
  __int64 v20; // r14
  unsigned __int64 v21; // rdi

  v5 = a1;
  v6 = sub_22077B0(0xC0u);
  v7 = *a3;
  *(_QWORD *)(v6 + 32) = v6 + 48;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    *(__m128i *)(v6 + 48) = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v7->m128i_i64[0];
    *(_QWORD *)(v6 + 48) = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_i64[1];
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v7->m128i_i64[1] = 0;
  *(_QWORD *)(v6 + 40) = v8;
  v7[1].m128i_i8[0] = 0;
  memset((void *)(v6 + 64), 0, 0x80u);
  *(_QWORD *)(v6 + 120) = v6 + 104;
  *(_QWORD *)(v6 + 128) = v6 + 104;
  *(_QWORD *)(v6 + 168) = v6 + 152;
  *(_QWORD *)(v6 + 176) = v6 + 152;
  v9 = sub_3941E90(a1, a2, v6 + 32);
  v11 = v9;
  v12 = v10;
  if ( v10 )
  {
    v13 = a1 + 1;
    LOBYTE(a1) = 1;
    if ( v9 || v10 == v13 )
      goto LABEL_5;
    v15 = *(_QWORD *)(v6 + 40);
    v17 = v10[5];
    v16 = v17;
    if ( v15 <= v17 )
      v17 = *(_QWORD *)(v6 + 40);
    if ( !v17 || (v18 = memcmp(*(const void **)(v6 + 32), (const void *)v12[4], v17), v13 = v5 + 1, (v19 = v18) == 0) )
    {
      v20 = v15 - v16;
      LOBYTE(a1) = 0;
      if ( v20 > 0x7FFFFFFF )
      {
LABEL_5:
        sub_220F040((char)a1, v6, v12, v13);
        ++v5[5];
        return v6;
      }
      if ( v20 < (__int64)0xFFFFFFFF80000000LL )
      {
        LOBYTE(a1) = 1;
        goto LABEL_5;
      }
      v19 = v20;
    }
    LODWORD(a1) = v19 >> 31;
    goto LABEL_5;
  }
  sub_393DEF0(0);
  sub_393DB20(*(_QWORD *)(v6 + 112));
  v21 = *(_QWORD *)(v6 + 32);
  if ( v6 + 48 != v21 )
    j_j___libc_free_0(v21);
  j_j___libc_free_0(v6);
  return v11;
}
