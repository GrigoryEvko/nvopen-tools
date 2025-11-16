// Function: sub_2D2C110
// Address: 0x2d2c110
//
__int64 __fastcall sub_2D2C110(_QWORD *a1, __int64 a2, const __m128i **a3)
{
  __int64 v5; // r12
  const __m128i *v6; // rax
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // r13
  _QWORD *v13; // rcx
  char v14; // di
  char v16; // al
  _QWORD *v17; // [rsp+8h] [rbp-28h]

  v5 = sub_22077B0(0x50u);
  v6 = *a3;
  *(_DWORD *)(v5 + 72) = 0;
  v7 = _mm_loadu_si128(v6);
  v8 = _mm_loadu_si128(v6 + 1);
  v9 = v6[2].m128i_i64[0];
  *(__m128i *)(v5 + 32) = v7;
  *(_QWORD *)(v5 + 64) = v9;
  *(__m128i *)(v5 + 48) = v8;
  v10 = sub_2D2BFE0(a1, a2, v5 + 32);
  v12 = v10;
  if ( v11 )
  {
    v13 = a1 + 1;
    v14 = 1;
    if ( !v10 && v11 != v13 )
    {
      v17 = v11;
      v16 = sub_2A4D650(v5 + 32, (__int64)(v11 + 4));
      v13 = a1 + 1;
      v11 = v17;
      v14 = v16;
    }
    sub_220F040(v14, v5, v11, v13);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v12;
  }
}
