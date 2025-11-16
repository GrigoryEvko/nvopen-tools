// Function: sub_3908D50
// Address: 0x3908d50
//
__int64 __fastcall sub_3908D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rdx
  __m128i v15; // xmm1
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // r13d
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // r8
  __m128i v29; // [rsp+0h] [rbp-40h] BYREF
  __m128i v30; // [rsp+10h] [rbp-30h] BYREF

  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v9 = *(unsigned int *)(v6 + 120);
  if ( (_DWORD)v9 )
  {
    v10 = (__int64 *)(*(_QWORD *)(v6 + 112) + 32LL * (unsigned int)v9 - 32);
    v11 = v10[2];
    v12 = v10[3];
    v13 = *v10;
    v14 = v10[1];
  }
  else
  {
    v14 = 0;
    v13 = 0;
    v12 = 0;
    v11 = 0;
  }
  v29.m128i_i64[0] = v13;
  v29.m128i_i64[1] = v14;
  v30.m128i_i64[0] = v11;
  v30.m128i_i64[1] = v12;
  if ( *(_DWORD *)(v6 + 124) <= (unsigned int)v9 )
  {
    sub_16CD150(v6 + 112, (const void *)(v6 + 128), 0, 32, v7, v8);
    v9 = *(unsigned int *)(v6 + 120);
  }
  v15 = _mm_loadu_si128(&v30);
  v16 = *(_QWORD *)(v6 + 112) + 32 * v9;
  *(__m128i *)v16 = _mm_loadu_si128(&v29);
  *(__m128i *)(v16 + 16) = v15;
  ++*(_DWORD *)(v6 + 120);
  v21 = sub_39077D0(a1, 1, a4);
  if ( !(_BYTE)v21 )
    return v21;
  v23 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 56LL))(
          *(_QWORD *)(a1 + 8),
          1,
          v17,
          v18,
          v19,
          v20,
          v29.m128i_i64[0],
          v29.m128i_i64[1],
          v30.m128i_i64[0],
          v30.m128i_i64[1]);
  v24 = *(unsigned int *)(v23 + 120);
  v25 = v24;
  if ( (unsigned int)v24 <= 1 )
    return v21;
  v26 = *(_QWORD *)(v23 + 112) + 32 * v24;
  v27 = *(_QWORD *)(v26 - 64);
  v28 = *(_QWORD *)(v26 - 56);
  if ( *(_QWORD *)(v26 - 24) != v28 || *(_QWORD *)(v26 - 32) != v27 )
  {
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v23 + 152LL))(v23, v27, v28);
    v25 = *(_DWORD *)(v23 + 120);
  }
  *(_DWORD *)(v23 + 120) = v25 - 1;
  return v21;
}
