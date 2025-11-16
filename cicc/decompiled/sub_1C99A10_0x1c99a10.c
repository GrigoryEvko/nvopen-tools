// Function: sub_1C99A10
// Address: 0x1c99a10
//
_QWORD *__fastcall sub_1C99A10(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  __m128i v5; // xmm0
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rcx
  _BOOL8 v11; // rdi

  v3 = sub_22077B0(56);
  v4 = *a2;
  v5 = _mm_loadu_si128((const __m128i *)(a2 + 1));
  v6 = v3;
  *(_QWORD *)(v3 + 32) = *a2;
  *(__m128i *)(v3 + 40) = v5;
  v7 = sub_1C99730(a1, (unsigned __int64 *)(v3 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 8;
    v11 = 1;
    if ( !v7 && v8 != v10 )
      v11 = v4 < *(_QWORD *)(v8 + 32);
    sub_220F040(v11, v6, v8, v10);
    ++*(_QWORD *)(a1 + 40);
    return (_QWORD *)v6;
  }
  else
  {
    j_j___libc_free_0(v6, 56);
    return v9;
  }
}
