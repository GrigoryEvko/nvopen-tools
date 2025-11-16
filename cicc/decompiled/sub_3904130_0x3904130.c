// Function: sub_3904130
// Address: 0x3904130
//
__int64 __fastcall sub_3904130(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __m128i v11; // xmm1
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r13d
  __int64 v19; // rbx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r8
  __m128i v25; // [rsp+0h] [rbp-40h] BYREF
  __m128i v26; // [rsp+10h] [rbp-30h] BYREF

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v5 = *(unsigned int *)(v2 + 120);
  if ( (_DWORD)v5 )
  {
    v6 = (__int64 *)(*(_QWORD *)(v2 + 112) + 32LL * (unsigned int)v5 - 32);
    v7 = v6[2];
    v8 = v6[3];
    v9 = *v6;
    v10 = v6[1];
  }
  else
  {
    v10 = 0;
    v9 = 0;
    v8 = 0;
    v7 = 0;
  }
  v25.m128i_i64[0] = v9;
  v25.m128i_i64[1] = v10;
  v26.m128i_i64[0] = v7;
  v26.m128i_i64[1] = v8;
  if ( *(_DWORD *)(v2 + 124) <= (unsigned int)v5 )
  {
    v7 = v2 + 128;
    sub_16CD150(v2 + 112, (const void *)(v2 + 128), 0, 32, v3, v4);
    v5 = *(unsigned int *)(v2 + 120);
  }
  v11 = _mm_loadu_si128(&v26);
  v12 = *(_QWORD *)(v2 + 112) + 32 * v5;
  *(__m128i *)v12 = _mm_loadu_si128(&v25);
  *(__m128i *)(v12 + 16) = v11;
  ++*(_DWORD *)(v2 + 120);
  v17 = sub_39039A0(a1);
  if ( !(_BYTE)v17 )
    return v17;
  v19 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 56LL))(
          *(_QWORD *)(a1 + 8),
          v7,
          v13,
          v14,
          v15,
          v16,
          v25.m128i_i64[0],
          v25.m128i_i64[1],
          v26.m128i_i64[0],
          v26.m128i_i64[1]);
  v20 = *(unsigned int *)(v19 + 120);
  v21 = v20;
  if ( (unsigned int)v20 <= 1 )
    return v17;
  v22 = *(_QWORD *)(v19 + 112) + 32 * v20;
  v23 = *(_QWORD *)(v22 - 64);
  v24 = *(_QWORD *)(v22 - 56);
  if ( *(_QWORD *)(v22 - 24) != v24 || *(_QWORD *)(v22 - 32) != v23 )
  {
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v19 + 152LL))(v19, v23, v24);
    v21 = *(_DWORD *)(v19 + 120);
  }
  *(_DWORD *)(v19 + 120) = v21 - 1;
  return v17;
}
