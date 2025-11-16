// Function: sub_31B8950
// Address: 0x31b8950
//
__int64 __fastcall sub_31B8950(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rsi
  __int64 v6; // rax
  void (__fastcall *v7)(__m128i *); // rdx
  void (__fastcall *v8)(_QWORD, _QWORD, _QWORD, _QWORD); // r8
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  _QWORD *v12; // rsi
  __int64 v13; // rax
  void (__fastcall *v14)(__m128i *); // rdx
  void (__fastcall *v15)(__m128i *, _QWORD *, __int64, _QWORD); // r15
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __m128i v19; // xmm3
  __int64 v20; // rsi
  __m128i *(__fastcall *v21)(__m128i *, __int64); // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __m128i v26; // xmm4
  __int64 v27; // rax
  __m128i v28; // xmm0
  __m128i v29; // xmm1
  __int64 v30; // rax
  unsigned int v32; // eax
  unsigned int v33; // eax
  void (__fastcall *v34)(_QWORD, _QWORD, _QWORD, _QWORD); // [rsp+0h] [rbp-C0h]
  _QWORD *v35; // [rsp+8h] [rbp-B8h]
  __m128i v36; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+20h] [rbp-A0h]
  __m128i v38; // [rsp+30h] [rbp-90h] BYREF
  __int64 v39; // [rsp+40h] [rbp-80h]
  __int128 v40; // [rsp+50h] [rbp-70h] BYREF
  __int64 v41; // [rsp+60h] [rbp-60h]
  __m128i v42; // [rsp+70h] [rbp-50h] BYREF
  __int64 v43; // [rsp+80h] [rbp-40h]

  v5 = *(_QWORD **)(a2 + 8);
  v6 = *v5;
  v7 = *(void (__fastcall **)(__m128i *))(*v5 + 40LL);
  if ( (char *)v7 == (char *)sub_3185D40 )
  {
    v8 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD))(v6 + 16);
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
      v34 = v8;
      v32 = v9((__int64)v5);
      v8 = v34;
      v11 = v32;
    }
    v8(&v42, v5, v11, 0);
    v37 = v43;
    v36 = _mm_loadu_si128(&v42);
  }
  else
  {
    v7(&v36);
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
      v35 = *(_QWORD **)(a2 + 8);
      v33 = v16((__int64)v12);
      v12 = v35;
      v18 = v33;
    }
    v15(&v42, v12, v18, 0);
    v19 = _mm_loadu_si128(&v42);
    v39 = v43;
    v38 = v19;
  }
  else
  {
    v14(&v38);
  }
  v20 = *(_QWORD *)(a2 + 8);
  v21 = *(__m128i *(__fastcall **)(__m128i *, __int64))(*(_QWORD *)v20 + 32LL);
  if ( v21 == sub_3184E50 )
  {
    (*(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD))(*(_QWORD *)v20 + 16LL))(&v42, v20, 0, 0);
    v26 = _mm_loadu_si128(&v42);
    v41 = v43;
    v40 = (__int128)v26;
  }
  else
  {
    ((void (__fastcall *)(__int128 *))v21)(&v40);
  }
  sub_31B8600(&v42, a3, v22, v23, v24, v25, v40, v41, v38.m128i_i64[0], v38.m128i_i64[1]);
  v27 = v43;
  v28 = _mm_loadu_si128(&v42);
  *(_QWORD *)(a1 + 80) = a2;
  v29 = _mm_loadu_si128(&v36);
  *(_QWORD *)(a1 + 88) = a3;
  *(_QWORD *)(a1 + 16) = v27;
  v30 = v37;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = v30;
  *(_QWORD *)(a1 + 56) = -1;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(__m128i *)a1 = v28;
  *(__m128i *)(a1 + 24) = v29;
  return a1;
}
