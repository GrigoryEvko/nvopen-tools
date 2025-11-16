// Function: sub_2C96A50
// Address: 0x2c96a50
//
__int64 __fastcall sub_2C96A50(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  const __m128i *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rax

  v5 = sub_22077B0(0x38u);
  v6 = *a3;
  *(_BYTE *)(v5 + 48) = 0;
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(v6);
  v7 = sub_2C96900(a1, a2, (unsigned __int64 *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
    {
      v13 = v8[4];
      if ( *(_QWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_QWORD *)(v5 + 32) == v13 )
          v11 = *(_QWORD *)(v5 + 40) < v8[5];
      }
    }
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return (__int64)v9;
  }
}
