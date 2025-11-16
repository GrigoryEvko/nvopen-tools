// Function: sub_31B8730
// Address: 0x31b8730
//
__int64 __fastcall sub_31B8730(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rsi
  __int64 v6; // rax
  void (__fastcall *v7)(__m128i *); // rdx
  void (__fastcall *v8)(__m128i *, _QWORD *, __int64, _QWORD); // rbx
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  _QWORD *v12; // rbx
  _QWORD *v13; // r15
  __int64 v14; // rsi
  __m128i *(__fastcall *v15)(__m128i *, __int64); // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __m128i v20; // xmm3
  __int64 v21; // rax
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __int64 v24; // rax
  __int64 v26; // [rsp+18h] [rbp-98h]
  __m128i v27; // [rsp+20h] [rbp-90h] BYREF
  __int64 v28; // [rsp+30h] [rbp-80h]
  __int128 v29; // [rsp+40h] [rbp-70h] BYREF
  __int64 v30; // [rsp+50h] [rbp-60h]
  __m128i v31; // [rsp+60h] [rbp-50h] BYREF
  __int64 v32; // [rsp+70h] [rbp-40h]

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
    v8(&v31, v5, v11, 0);
    v28 = v32;
    v27 = _mm_loadu_si128(&v31);
  }
  else
  {
    v7(&v27);
  }
  v12 = *(_QWORD **)(a2 + 64);
  v13 = &v12[*(unsigned int *)(a2 + 80)];
  v26 = *(_QWORD *)(a2 + 56);
  if ( *(_DWORD *)(a2 + 72) )
  {
    for ( ; v12 != v13; ++v12 )
    {
      if ( *v12 != -4096 && *v12 != -8192 )
        break;
    }
  }
  else
  {
    v12 += *(unsigned int *)(a2 + 80);
  }
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(__m128i *(__fastcall **)(__m128i *, __int64))(*(_QWORD *)v14 + 32LL);
  if ( v15 == sub_3184E50 )
  {
    (*(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD))(*(_QWORD *)v14 + 16LL))(&v31, v14, 0, 0);
    v20 = _mm_loadu_si128(&v31);
    v30 = v32;
    v29 = (__int128)v20;
  }
  else
  {
    ((void (__fastcall *)(__int128 *))v15)(&v29);
  }
  sub_31B8600(&v31, a3, v16, v17, v18, v19, v29, v30, v27.m128i_i64[0], v27.m128i_i64[1]);
  v21 = v32;
  v22 = _mm_loadu_si128(&v31);
  *(_QWORD *)(a1 + 64) = v12;
  v23 = _mm_loadu_si128(&v27);
  *(_QWORD *)(a1 + 72) = v13;
  *(_QWORD *)(a1 + 16) = v21;
  v24 = v28;
  *(_QWORD *)(a1 + 80) = a2;
  *(_QWORD *)(a1 + 40) = v24;
  *(_QWORD *)(a1 + 88) = a3;
  *(_QWORD *)(a1 + 48) = a2 + 56;
  *(__m128i *)a1 = v22;
  *(_QWORD *)(a1 + 56) = v26;
  *(__m128i *)(a1 + 24) = v23;
  return a1;
}
