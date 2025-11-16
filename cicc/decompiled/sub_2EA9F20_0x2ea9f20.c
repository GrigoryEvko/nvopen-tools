// Function: sub_2EA9F20
// Address: 0x2ea9f20
//
__int64 __fastcall sub_2EA9F20(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 (__fastcall *v3)(const __m128i **, const __m128i *, int); // rax
  __m128i v4; // xmm0
  __int64 v5; // rcx
  __m128i v6; // xmm1
  void (__fastcall *v7)(_QWORD, _QWORD, _QWORD); // r8
  _BYTE *(__fastcall *v8)(__int64 *, __int64, char *, __int64 *, _QWORD *); // rdx
  __m128i v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 (__fastcall *v11)(const __m128i **, const __m128i *, int); // [rsp+10h] [rbp-40h]
  _BYTE *(__fastcall *v12)(__int64 *, __int64, char *, __int64 *, _QWORD *); // [rsp+18h] [rbp-38h]
  __m128i v13; // [rsp+20h] [rbp-30h] BYREF
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-20h]
  __int64 v15; // [rsp+38h] [rbp-18h]

  sub_2EA9F00(a1 + 176);
  v10.m128i_i64[1] = (__int64)a2;
  *(_QWORD *)(a1 + 2664) = a2;
  v2 = *a2;
  v14 = 0;
  v10.m128i_i64[0] = v2;
  v12 = sub_2EA9D30;
  v11 = sub_2EA9C30;
  sub_2EA9C30((const __m128i **)&v13, &v10, 2);
  v3 = v11;
  v4 = _mm_loadu_si128(&v13);
  v6 = _mm_loadu_si128((const __m128i *)(v5 + 120));
  v7 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(v5 + 136);
  v15 = *(_QWORD *)(v5 + 144);
  v8 = v12;
  *(_QWORD *)(v5 + 136) = v11;
  v14 = v7;
  *(_QWORD *)(v5 + 144) = v8;
  v13 = v6;
  *(__m128i *)(v5 + 120) = v4;
  if ( v7 )
  {
    v7(&v13, &v13, 3);
    v3 = v11;
  }
  if ( v3 )
    v3((const __m128i **)&v10, &v10, 3);
  return 0;
}
