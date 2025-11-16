// Function: sub_E7E9B0
// Address: 0xe7e9b0
//
__int64 __fastcall sub_E7E9B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r13
  const __m128i *v4; // rbx
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  unsigned int v7; // ecx
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rax
  int v10; // edi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __m128i *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // r13
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r8
  char *v27; // rbx
  const char *v28; // [rsp+0h] [rbp-80h] BYREF
  char v29; // [rsp+20h] [rbp-60h]
  char v30; // [rsp+21h] [rbp-5Fh]
  __int64 v31; // [rsp+30h] [rbp-50h] BYREF
  int v32; // [rsp+38h] [rbp-48h]
  __int64 v33; // [rsp+40h] [rbp-40h]
  unsigned int v34; // [rsp+48h] [rbp-38h]
  __int16 v35; // [rsp+50h] [rbp-30h]

  result = sub_E7DDE0(a1);
  if ( *(_DWORD *)(result + 96) )
  {
    v3 = result;
    v4 = (const __m128i *)&v31;
    v5 = **(_QWORD **)(a1 + 296);
    v28 = ".llvm.call-graph-profile";
    v30 = 1;
    v29 = 3;
    v35 = 257;
    v6 = sub_E71CB0(v5, (size_t *)&v28, 1879002121, 0x80000000, 8, (__int64)&v31, 0, -1, 0);
    v7 = *(_DWORD *)(a1 + 128);
    v8 = v6;
    v9 = *(_QWORD *)(a1 + 120);
    if ( v7 )
    {
      v12 = v7;
      v11 = 32LL * v7;
      v24 = v9 + v11 - 32;
      v14 = *(_QWORD *)(v24 + 16);
      v7 = *(_DWORD *)(v24 + 24);
      v13 = *(_QWORD *)v24;
      v10 = *(_DWORD *)(v24 + 8);
    }
    else
    {
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v13 = 0;
      v14 = 0;
    }
    v34 = v7;
    v15 = *(unsigned int *)(a1 + 132);
    v16 = v12 + 1;
    v31 = v13;
    v32 = v10;
    v33 = v14;
    if ( v16 > v15 )
    {
      v25 = a1 + 120;
      v26 = a1 + 136;
      if ( v9 > (unsigned __int64)&v31 || (unsigned __int64)&v31 >= v9 + v11 )
      {
        sub_C8D5F0(v25, (const void *)(a1 + 136), v16, 0x20u, v26, v13);
        v9 = *(_QWORD *)(a1 + 120);
        v11 = 32LL * *(unsigned int *)(a1 + 128);
      }
      else
      {
        v27 = (char *)&v31 - v9;
        sub_C8D5F0(v25, (const void *)(a1 + 136), v16, 0x20u, v26, v13);
        v9 = *(_QWORD *)(a1 + 120);
        v4 = (const __m128i *)&v27[v9];
        v11 = 32LL * *(unsigned int *)(a1 + 128);
      }
    }
    v17 = (__m128i *)(v11 + v9);
    *v17 = _mm_loadu_si128(v4);
    v17[1] = _mm_loadu_si128(v4 + 1);
    v18 = *(_QWORD *)a1;
    ++*(_DWORD *)(a1 + 128);
    (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(v18 + 176))(a1, v8, 0);
    v19 = *(unsigned int *)(v3 + 96);
    v20 = *(__int64 **)(v3 + 88);
    v21 = 0;
    v22 = &v20[3 * v19];
    while ( v22 != v20 )
    {
      sub_E7E7A0(a1, v20, v21);
      sub_E7E7A0(a1, v20 + 1, v21);
      v23 = v20[2];
      v20 += 3;
      v21 += 8;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 536LL))(a1, v23, 8);
    }
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 168LL))(a1);
  }
  return result;
}
