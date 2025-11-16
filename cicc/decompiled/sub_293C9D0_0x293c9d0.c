// Function: sub_293C9D0
// Address: 0x293c9d0
//
unsigned __int64 *__fastcall sub_293C9D0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  unsigned __int64 *v5; // r12
  __m128i v6; // xmm0
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rax

  v5 = (unsigned __int64 *)sub_22077B0(0x80u);
  v6 = _mm_loadu_si128(*a3);
  v5[6] = (unsigned __int64)(v5 + 8);
  v5[7] = 0x800000000LL;
  *((__m128i *)v5 + 2) = v6;
  v7 = sub_293C880(a1, a2, v5 + 4);
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
    {
      v13 = v8[4];
      if ( v5[4] >= v13 )
      {
        v11 = 0;
        if ( v5[4] == v13 )
          v11 = v5[5] < v8[5];
      }
    }
    sub_220F040(v11, (__int64)v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0((unsigned __int64)v5);
    return v9;
  }
}
