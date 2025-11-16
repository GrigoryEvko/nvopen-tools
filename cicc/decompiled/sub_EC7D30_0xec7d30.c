// Function: sub_EC7D30
// Address: 0xec7d30
//
__int64 __fastcall sub_EC7D30(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // ecx
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  int v6; // edi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  const __m128i *v13; // r13
  __m128i *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r13d
  __int64 v20; // rax
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // r8
  char *v25; // r13
  __int64 v26; // [rsp+0h] [rbp-40h] BYREF
  int v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+10h] [rbp-30h]
  unsigned int v29; // [rsp+18h] [rbp-28h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v3 = *(_DWORD *)(v2 + 128);
  v4 = v2;
  v5 = *(_QWORD *)(v2 + 120);
  if ( v3 )
  {
    v8 = v3;
    v7 = 32LL * v3;
    v22 = v5 + v7 - 32;
    v10 = *(_QWORD *)(v22 + 16);
    v3 = *(_DWORD *)(v22 + 24);
    v9 = *(_QWORD *)v22;
    v6 = *(_DWORD *)(v22 + 8);
  }
  else
  {
    v6 = 0;
    v7 = 0;
    v8 = 0;
    v9 = 0;
    v10 = 0;
  }
  v29 = v3;
  v11 = *(unsigned int *)(v4 + 132);
  v12 = v8 + 1;
  v13 = (const __m128i *)&v26;
  v26 = v9;
  v27 = v6;
  v28 = v10;
  if ( v12 > v11 )
  {
    v23 = v4 + 120;
    v24 = v4 + 136;
    if ( v5 > (unsigned __int64)&v26 || (unsigned __int64)&v26 >= v5 + v7 )
    {
      sub_C8D5F0(v23, (const void *)(v4 + 136), v12, 0x20u, v24, v9);
      v5 = *(_QWORD *)(v4 + 120);
      v7 = 32LL * *(unsigned int *)(v4 + 128);
    }
    else
    {
      v25 = (char *)&v26 - v5;
      sub_C8D5F0(v23, (const void *)(v4 + 136), v12, 0x20u, v24, v9);
      v5 = *(_QWORD *)(v4 + 120);
      v13 = (const __m128i *)&v25[v5];
      v7 = 32LL * *(unsigned int *)(v4 + 128);
    }
  }
  v14 = (__m128i *)(v7 + v5);
  *v14 = _mm_loadu_si128(v13);
  v14[1] = _mm_loadu_si128(v13 + 1);
  ++*(_DWORD *)(v4 + 128);
  v19 = sub_EC74A0(a1);
  if ( (_BYTE)v19 )
  {
    v20 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, __int64, int, __int64, unsigned int))(**(_QWORD **)(a1 + 8) + 56LL))(
            *(_QWORD *)(a1 + 8),
            v7,
            v15,
            v16,
            v17,
            v18,
            v26,
            v27,
            v28,
            v29);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 168LL))(v20);
  }
  return v19;
}
