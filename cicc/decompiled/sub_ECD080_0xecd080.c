// Function: sub_ECD080
// Address: 0xecd080
//
__int64 __fastcall sub_ECD080(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned int v7; // ecx
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  int v10; // edi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  const __m128i *v17; // r13
  __m128i *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // r13d
  __int64 v24; // rax
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // r8
  char *v29; // r13
  __int64 v30; // [rsp+0h] [rbp-40h] BYREF
  int v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+10h] [rbp-30h]
  unsigned int v33; // [rsp+18h] [rbp-28h]

  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v7 = *(_DWORD *)(v6 + 128);
  v8 = v6;
  v9 = *(_QWORD *)(v6 + 120);
  if ( v7 )
  {
    v12 = v7;
    v11 = 32LL * v7;
    v26 = v9 + v11 - 32;
    v14 = *(_QWORD *)(v26 + 16);
    v7 = *(_DWORD *)(v26 + 24);
    v13 = *(_QWORD *)v26;
    v10 = *(_DWORD *)(v26 + 8);
  }
  else
  {
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
  }
  v33 = v7;
  v15 = *(unsigned int *)(v8 + 132);
  v16 = v12 + 1;
  v17 = (const __m128i *)&v30;
  v30 = v13;
  v31 = v10;
  v32 = v14;
  if ( v16 > v15 )
  {
    v27 = v8 + 120;
    v28 = v8 + 136;
    if ( v9 > (unsigned __int64)&v30 || (unsigned __int64)&v30 >= v9 + v11 )
    {
      sub_C8D5F0(v27, (const void *)(v8 + 136), v16, 0x20u, v28, v13);
      v9 = *(_QWORD *)(v8 + 120);
      v11 = 32LL * *(unsigned int *)(v8 + 128);
    }
    else
    {
      v29 = (char *)&v30 - v9;
      sub_C8D5F0(v27, (const void *)(v8 + 136), v16, 0x20u, v28, v13);
      v9 = *(_QWORD *)(v8 + 120);
      v17 = (const __m128i *)&v29[v9];
      v11 = 32LL * *(unsigned int *)(v8 + 128);
    }
  }
  v18 = (__m128i *)(v11 + v9);
  *v18 = _mm_loadu_si128(v17);
  v18[1] = _mm_loadu_si128(v17 + 1);
  ++*(_DWORD *)(v8 + 128);
  v23 = sub_ECB300(a1, 1, a4);
  if ( (_BYTE)v23 )
  {
    v24 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, __int64, int, __int64, unsigned int))(**(_QWORD **)(a1 + 8) + 56LL))(
            *(_QWORD *)(a1 + 8),
            1,
            v19,
            v20,
            v21,
            v22,
            v30,
            v31,
            v32,
            v33);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 168LL))(v24);
  }
  return v23;
}
