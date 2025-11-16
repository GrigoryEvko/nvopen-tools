// Function: sub_3185D40
// Address: 0x3185d40
//
__m128i *__fastcall sub_3185D40(__m128i *a1, _QWORD *a2)
{
  void (__fastcall *v2)(__m128i *, _QWORD *, __int64, _QWORD); // rbx
  __int64 (__fastcall *v3)(__int64); // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __m128i v6; // xmm0
  __m128i v8; // [rsp+10h] [rbp-30h] BYREF
  __int64 v9; // [rsp+20h] [rbp-20h]

  v2 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(*a2 + 16LL);
  v3 = *(__int64 (__fastcall **)(__int64))(*a2 + 64LL);
  if ( v3 == sub_3184E90 )
  {
    v4 = a2[2];
    v5 = 0;
    if ( (unsigned __int8)(*(_BYTE *)v4 - 22) > 6u )
      v5 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
  }
  else
  {
    v5 = (unsigned int)v3((__int64)a2);
  }
  v2(&v8, a2, v5, 0);
  v6 = _mm_loadu_si128(&v8);
  a1[1].m128i_i64[0] = v9;
  *a1 = v6;
  return a1;
}
