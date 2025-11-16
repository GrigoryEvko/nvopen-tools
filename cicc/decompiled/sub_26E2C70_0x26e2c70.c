// Function: sub_26E2C70
// Address: 0x26e2c70
//
_QWORD *__fastcall sub_26E2C70(unsigned __int64 *a1, const __m128i *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r15
  _DWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13

  v2 = sub_22077B0(0x20u);
  v3 = (_QWORD *)v2;
  if ( v2 )
    *(_QWORD *)v2 = 0;
  *(__m128i *)(v2 + 8) = _mm_loadu_si128(a2);
  v4 = *(_QWORD *)(v2 + 8);
  v5 = v4 % a1[1];
  v6 = sub_26E2B00(a1, v5, (_DWORD *)(v2 + 8), v4);
  if ( !v6 )
    return sub_26DFEE0(a1, v5, v4, v3, 1);
  v7 = *(_QWORD *)v6;
  if ( !v7 )
    return sub_26DFEE0(a1, v5, v4, v3, 1);
  v8 = v7;
  j_j___libc_free_0((unsigned __int64)v3);
  return (_QWORD *)v8;
}
