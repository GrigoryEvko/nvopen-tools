// Function: sub_2CE1820
// Address: 0x2ce1820
//
unsigned __int64 __fastcall sub_2CE1820(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  __m128i v5; // xmm0
  unsigned __int64 v6; // r12
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // di

  v3 = sub_22077B0(0x38u);
  v4 = *a2;
  v5 = _mm_loadu_si128((const __m128i *)(a2 + 1));
  v6 = v3;
  *(_QWORD *)(v3 + 32) = *a2;
  *(__m128i *)(v3 + 40) = v5;
  v7 = sub_2CE1580(a1, (unsigned __int64 *)(v3 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = (_QWORD *)(a1 + 8);
    v11 = 1;
    if ( !v7 && v8 != v10 )
      v11 = v4 < v8[4];
    sub_220F040(v11, v6, v8, v10);
    ++*(_QWORD *)(a1 + 40);
    return v6;
  }
  else
  {
    j_j___libc_free_0(v6);
    return (unsigned __int64)v9;
  }
}
