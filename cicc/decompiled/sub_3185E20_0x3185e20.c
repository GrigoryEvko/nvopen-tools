// Function: sub_3185E20
// Address: 0x3185e20
//
__m128i *__fastcall sub_3185E20(__m128i *a1, _QWORD *a2)
{
  __int64 v2; // rax
  void (*v3)(void); // rdx
  void (__fastcall *v4)(__m128i *, _QWORD *, __int64, _QWORD); // rbx
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __m128i v10; // [rsp+10h] [rbp-30h] BYREF
  __int64 v11; // [rsp+20h] [rbp-20h]

  v2 = *a2;
  v3 = *(void (**)(void))(*a2 + 40LL);
  if ( (char *)v3 == (char *)sub_3185D40 )
  {
    v4 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v2 + 16);
    v5 = *(__int64 (__fastcall **)(__int64))(v2 + 64);
    if ( v5 == sub_3184E90 )
    {
      v6 = a2[2];
      v7 = 0;
      if ( (unsigned __int8)(*(_BYTE *)v6 - 22) > 6u )
        v7 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
    }
    else
    {
      v7 = (unsigned int)v5((__int64)a2);
    }
    v4(&v10, a2, v7, 0);
    v8 = _mm_loadu_si128(&v10);
    a1[1].m128i_i64[0] = v11;
    *a1 = v8;
    return a1;
  }
  else
  {
    v3();
    return a1;
  }
}
