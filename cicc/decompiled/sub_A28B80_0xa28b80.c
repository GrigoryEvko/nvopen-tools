// Function: sub_A28B80
// Address: 0xa28b80
//
__int64 __fastcall sub_A28B80(_QWORD *a1, _QWORD *a2, __m128i **a3)
{
  __int64 v5; // r12
  __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r13
  _QWORD *v12; // r15
  __int64 v13; // rdi
  __int64 v15; // rdi

  v5 = sub_22077B0(72);
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
  *(_DWORD *)(v5 + 64) = 0;
  v8 = sub_A288A0(a1, a2, v5 + 32);
  v10 = v8;
  v11 = v9;
  if ( v9 )
  {
    v12 = a1 + 1;
    v13 = 1;
    if ( !v8 && (_QWORD *)v9 != v12 )
      v13 = (unsigned int)sub_A15B80(
                            *(const void **)(v5 + 32),
                            *(_QWORD *)(v5 + 40),
                            *(const void **)(v9 + 32),
                            *(_QWORD *)(v9 + 40)) >> 31;
    sub_220F040(v13, v5, v11, a1 + 1);
    ++a1[5];
    return v5;
  }
  else
  {
    v15 = *(_QWORD *)(v5 + 32);
    if ( v5 + 48 != v15 )
      j_j___libc_free_0(v15, *(_QWORD *)(v5 + 48) + 1LL);
    j_j___libc_free_0(v5, 72);
    return v10;
  }
}
