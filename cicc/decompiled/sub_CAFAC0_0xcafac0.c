// Function: sub_CAFAC0
// Address: 0xcafac0
//
__int64 __fastcall sub_CAFAC0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  const __m128i *v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  _BOOL8 v12; // rdi
  size_t v14; // r15
  size_t v15; // r14
  size_t v16; // rdx
  unsigned int v17; // eax

  v5 = sub_22077B0(64);
  v6 = *a3;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 56) = 0;
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(v6);
  v7 = sub_CAF3B0(a1, a2, v5 + 32);
  v9 = v7;
  if ( v8 )
  {
    v10 = v8;
    v11 = a1 + 1;
    v12 = 1;
    if ( !v7 && v8 != v11 )
    {
      v14 = *(_QWORD *)(v5 + 40);
      v16 = v8[5];
      v15 = v16;
      if ( v14 <= v16 )
        v16 = *(_QWORD *)(v5 + 40);
      if ( v16 && (v17 = memcmp(*(const void **)(v5 + 32), (const void *)v10[4], v16), v11 = a1 + 1, v17) )
      {
        v12 = v17 >> 31;
      }
      else
      {
        v12 = v14 < v15;
        if ( v14 == v15 )
          v12 = 0;
      }
    }
    sub_220F040(v12, v5, v10, v11);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5, 64);
    return v9;
  }
}
