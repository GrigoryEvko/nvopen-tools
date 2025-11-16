// Function: sub_3185DC0
// Address: 0x3185dc0
//
__m128i *__fastcall sub_3185DC0(__m128i *a1, __int64 a2)
{
  void (*v2)(void); // rdx
  __m128i v3; // xmm0
  __m128i v5; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h]

  v2 = *(void (**)(void))(*(_QWORD *)a2 + 32LL);
  if ( (char *)v2 == (char *)sub_3184E50 )
  {
    (*(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 16LL))(&v5, a2, 0, 0);
    v3 = _mm_loadu_si128(&v5);
    a1[1].m128i_i64[0] = v6;
    *a1 = v3;
    return a1;
  }
  else
  {
    v2();
    return a1;
  }
}
