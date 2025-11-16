// Function: sub_318B130
// Address: 0x318b130
//
__int64 __fastcall sub_318B130(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __m128i *v4; // rbx
  void (__fastcall *v5)(__m128i *, __int64, __int64); // rax
  __m128i v6; // xmm0
  void (__fastcall *v7)(_QWORD, _QWORD, _QWORD); // rdx
  void (__fastcall *v8)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-38h] BYREF
  __m128i v13; // [rsp+10h] [rbp-30h] BYREF
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-20h]
  __int64 v15; // [rsp+28h] [rbp-18h]

  v2 = a1 + 280;
  v12 = *(_QWORD *)(v2 + 96);
  *(_QWORD *)(v2 + 96) = v12 + 1;
  v3 = sub_318AF60(v2, &v12);
  v14 = 0;
  v4 = (__m128i *)v3;
  v5 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a2 + 16);
  if ( v5 )
  {
    v5(&v13, a2, 2);
    v15 = *(_QWORD *)(a2 + 24);
    v14 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a2 + 16);
  }
  v6 = _mm_loadu_si128(&v13);
  v7 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v4[1].m128i_i64[0];
  v13 = _mm_loadu_si128(v4);
  *v4 = v6;
  v8 = v14;
  v14 = v7;
  v9 = v4[1].m128i_i64[1];
  v4[1].m128i_i64[0] = (__int64)v8;
  v10 = v15;
  v15 = v9;
  v4[1].m128i_i64[1] = v10;
  if ( v14 )
    v14(&v13, &v13, 3);
  return v12;
}
