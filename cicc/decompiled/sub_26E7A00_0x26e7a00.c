// Function: sub_26E7A00
// Address: 0x26e7a00
//
__int64 __fastcall sub_26E7A00(_QWORD *a1, __int64 a2, const __m128i *a3, char a4)
{
  __m128i v6; // xmm0
  unsigned __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v12; // [rsp+10h] [rbp-50h] BYREF
  __m128i v13[4]; // [rsp+18h] [rbp-48h] BYREF

  v6 = _mm_loadu_si128(a3);
  v12 = a2;
  v13[0] = v6;
  v7 = sub_26E11E0(&v12, (__int64)v13);
  v8 = sub_26E65E0(a1 + 18, v7 % a1[19], &v12, v7);
  if ( v8 && (v9 = *v8) != 0 )
  {
    return *(unsigned __int8 *)(v9 + 32);
  }
  else if ( !a4
         && (v10 = sub_26E6F90((__int64)a1, a2, (__int64)a3),
             v13[0] = _mm_loadu_si128(a3),
             v12 = a2,
             (*(_BYTE *)sub_26E66B0(a1 + 18, (__m128i *)&v12) = v10) != 0) )
  {
    v12 = a2;
    *(__m128i *)sub_26E17A0(a1 + 25, (unsigned __int64 *)&v12) = _mm_loadu_si128(a3);
  }
  else
  {
    return 0;
  }
  return v10;
}
