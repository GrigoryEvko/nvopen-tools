// Function: sub_31B8130
// Address: 0x31b8130
//
__int64 __fastcall sub_31B8130(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rsi
  __int64 v6; // rax
  void (__fastcall *v7)(__m128i *); // rdx
  void (__fastcall *v8)(__m128i *, _QWORD *, __int64, _QWORD); // r15
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  _QWORD *v12; // rsi
  __int64 v13; // rax
  void (__fastcall *v14)(__m128i *); // rdx
  void (__fastcall *v15)(__m128i *, _QWORD *, __int64, _QWORD); // r14
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __m128i v20; // xmm1
  __m128i v21; // xmm0
  __int64 v22; // rax
  unsigned int v24; // eax
  _QWORD *v25; // [rsp+8h] [rbp-98h]
  __m128i v26; // [rsp+10h] [rbp-90h] BYREF
  __int64 v27; // [rsp+20h] [rbp-80h]
  __m128i v28; // [rsp+30h] [rbp-70h] BYREF
  __int64 v29; // [rsp+40h] [rbp-60h]
  __m128i v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD **)(a2 + 8);
  v6 = *v5;
  v7 = *(void (__fastcall **)(__m128i *))(*v5 + 40LL);
  if ( (char *)v7 == (char *)sub_3185D40 )
  {
    v8 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v6 + 16);
    v9 = *(__int64 (__fastcall **)(__int64))(v6 + 64);
    if ( v9 == sub_3184E90 )
    {
      v10 = v5[2];
      v11 = 0;
      if ( (unsigned __int8)(*(_BYTE *)v10 - 22) > 6u )
        v11 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    }
    else
    {
      v11 = (unsigned int)v9((__int64)v5);
    }
    v8(&v30, v5, v11, 0);
    v29 = v31;
    v28 = _mm_loadu_si128(&v30);
  }
  else
  {
    v7(&v28);
  }
  v12 = *(_QWORD **)(a2 + 8);
  v13 = *v12;
  v14 = *(void (__fastcall **)(__m128i *))(*v12 + 40LL);
  if ( (char *)v14 == (char *)sub_3185D40 )
  {
    v15 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v13 + 16);
    v16 = *(__int64 (__fastcall **)(__int64))(v13 + 64);
    if ( v16 == sub_3184E90 )
    {
      v17 = v12[2];
      v18 = 0;
      if ( (unsigned __int8)(*(_BYTE *)v17 - 22) > 6u )
        v18 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
    }
    else
    {
      v25 = *(_QWORD **)(a2 + 8);
      v24 = v16((__int64)v12);
      v12 = v25;
      v18 = v24;
    }
    v15(&v30, v12, v18, 0);
    v27 = v31;
    v26 = _mm_loadu_si128(&v30);
  }
  else
  {
    v14(&v26);
  }
  v19 = v27;
  v20 = _mm_loadu_si128(&v28);
  *(_QWORD *)(a1 + 80) = a2;
  v21 = _mm_loadu_si128(&v26);
  *(_QWORD *)(a1 + 88) = a3;
  *(_QWORD *)(a1 + 16) = v19;
  v22 = v29;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = v22;
  *(_QWORD *)(a1 + 56) = -1;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(__m128i *)a1 = v21;
  *(__m128i *)(a1 + 24) = v20;
  return a1;
}
