// Function: sub_31B82E0
// Address: 0x31b82e0
//
__int64 __fastcall sub_31B82E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // rsi
  __int64 v8; // rax
  void (__fastcall *v9)(__m128i *); // rdx
  void (__fastcall *v10)(__m128i *, _QWORD *, __int64, _QWORD); // r10
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // rsi
  __int64 v15; // rax
  void (__fastcall *v16)(__m128i *); // rdx
  void (__fastcall *v17)(__m128i *, _QWORD *, __int64, _QWORD); // r13
  __int64 (__fastcall *v18)(__int64); // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __int64 v24; // rax
  unsigned int v26; // eax
  unsigned int v27; // eax
  void (__fastcall *v28)(__m128i *, _QWORD *, _QWORD, _QWORD); // [rsp+0h] [rbp-B0h]
  _QWORD *v29; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __m128i v32; // [rsp+20h] [rbp-90h] BYREF
  __int64 v33; // [rsp+30h] [rbp-80h]
  __m128i v34; // [rsp+40h] [rbp-70h] BYREF
  __int64 v35; // [rsp+50h] [rbp-60h]
  __m128i v36; // [rsp+60h] [rbp-50h] BYREF
  __int64 v37; // [rsp+70h] [rbp-40h]

  v3 = a2 + 56;
  v5 = *(_QWORD *)(a2 + 64) + 8LL * *(unsigned int *)(a2 + 80);
  v6 = *(_QWORD *)(a2 + 56);
  v7 = *(_QWORD **)(a2 + 8);
  v31 = v6;
  v8 = *v7;
  v9 = *(void (__fastcall **)(__m128i *))(*v7 + 40LL);
  if ( (char *)v9 == (char *)sub_3185D40 )
  {
    v10 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v8 + 16);
    v11 = *(__int64 (__fastcall **)(__int64))(v8 + 64);
    if ( v11 == sub_3184E90 )
    {
      v12 = v7[2];
      v13 = 0;
      if ( (unsigned __int8)(*(_BYTE *)v12 - 22) > 6u )
        v13 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
      v10(&v36, v7, v13, 0);
    }
    else
    {
      v28 = v10;
      v27 = v11((__int64)v7);
      v28(&v36, v7, v27, 0);
    }
    v35 = v37;
    v34 = _mm_loadu_si128(&v36);
  }
  else
  {
    v9(&v34);
  }
  v14 = *(_QWORD **)(a2 + 8);
  v15 = *v14;
  v16 = *(void (__fastcall **)(__m128i *))(*v14 + 40LL);
  if ( (char *)v16 == (char *)sub_3185D40 )
  {
    v17 = *(void (__fastcall **)(__m128i *, _QWORD *, __int64, _QWORD))(v15 + 16);
    v18 = *(__int64 (__fastcall **)(__int64))(v15 + 64);
    if ( v18 == sub_3184E90 )
    {
      v19 = v14[2];
      v20 = 0;
      if ( (unsigned __int8)(*(_BYTE *)v19 - 22) > 6u )
        v20 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
    }
    else
    {
      v29 = *(_QWORD **)(a2 + 8);
      v26 = v18((__int64)v14);
      v14 = v29;
      v20 = v26;
    }
    v17(&v36, v14, v20, 0);
    v33 = v37;
    v32 = _mm_loadu_si128(&v36);
  }
  else
  {
    v16(&v32);
  }
  v21 = v33;
  v22 = _mm_loadu_si128(&v32);
  *(_QWORD *)(a1 + 48) = v3;
  v23 = _mm_loadu_si128(&v34);
  *(_QWORD *)(a1 + 64) = v5;
  *(_QWORD *)(a1 + 16) = v21;
  v24 = v35;
  *(_QWORD *)(a1 + 72) = v5;
  *(_QWORD *)(a1 + 40) = v24;
  *(_QWORD *)(a1 + 80) = a2;
  *(_QWORD *)(a1 + 56) = v31;
  *(__m128i *)a1 = v22;
  *(_QWORD *)(a1 + 88) = a3;
  *(__m128i *)(a1 + 24) = v23;
  return a1;
}
