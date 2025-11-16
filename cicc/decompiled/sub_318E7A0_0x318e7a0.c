// Function: sub_318E7A0
// Address: 0x318e7a0
//
__int64 __fastcall sub_318E7A0(__int64 a1)
{
  __int64 v1; // r13
  void (__fastcall *v2)(__m128i *, __int64, _QWORD, _QWORD); // rbx
  int v3; // eax
  __m128i v4; // xmm0
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD))(*(_QWORD *)v1 + 16LL);
  v3 = sub_318E5E0(a1);
  v2(&v6, v1, (unsigned int)(v3 + 1), 0);
  v4 = _mm_loadu_si128(&v6);
  *(_QWORD *)(a1 + 16) = v7;
  *(__m128i *)a1 = v4;
  return a1;
}
