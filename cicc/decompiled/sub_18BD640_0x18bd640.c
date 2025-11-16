// Function: sub_18BD640
// Address: 0x18bd640
//
__int64 __fastcall sub_18BD640(_QWORD *a1, _QWORD *a2, __m128i **a3)
{
  __int64 v5; // r12
  __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // r15
  _QWORD *v11; // r13
  _QWORD *v12; // rcx
  __int64 v13; // rdi
  size_t v15; // r14
  size_t v16; // r15
  size_t v17; // rdx
  int v18; // eax
  unsigned int v19; // edi
  __int64 v20; // r14
  __int64 v21; // rdi

  v5 = sub_22077B0(152);
  v6 = *a3;
  *(_QWORD *)(v5 + 32) = v5 + 48;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    *(__m128i *)(v5 + 48) = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v6->m128i_i64[0];
    *(_QWORD *)(v5 + 48) = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  *(_QWORD *)(v5 + 40) = v7;
  v6[1].m128i_i8[0] = 0;
  memset((void *)(v5 + 64), 0, 0x58u);
  *(_QWORD *)(v5 + 128) = v5 + 112;
  *(_QWORD *)(v5 + 136) = v5 + 112;
  v8 = sub_14F61B0(a1, a2, v5 + 32);
  v10 = v8;
  v11 = v9;
  if ( v9 )
  {
    v12 = a1 + 1;
    v13 = 1;
    if ( v8 || v9 == v12 )
      goto LABEL_5;
    v15 = *(_QWORD *)(v5 + 40);
    v17 = v9[5];
    v16 = v17;
    if ( v15 <= v17 )
      v17 = *(_QWORD *)(v5 + 40);
    if ( !v17 || (v18 = memcmp(*(const void **)(v5 + 32), (const void *)v11[4], v17), v12 = a1 + 1, (v19 = v18) == 0) )
    {
      v20 = v15 - v16;
      v13 = 0;
      if ( v20 > 0x7FFFFFFF )
      {
LABEL_5:
        sub_220F040(v13, v5, v11, v12);
        ++a1[5];
        return v5;
      }
      if ( v20 < (__int64)0xFFFFFFFF80000000LL )
      {
        v13 = 1;
        goto LABEL_5;
      }
      v19 = v20;
    }
    v13 = v19 >> 31;
    goto LABEL_5;
  }
  sub_18B5820(0);
  v21 = *(_QWORD *)(v5 + 32);
  if ( v5 + 48 != v21 )
    j_j___libc_free_0(v21, *(_QWORD *)(v5 + 48) + 1LL);
  j_j___libc_free_0(v5, 152);
  return v10;
}
