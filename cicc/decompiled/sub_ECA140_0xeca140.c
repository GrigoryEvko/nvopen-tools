// Function: sub_ECA140
// Address: 0xeca140
//
__int64 __fastcall sub_ECA140(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // r14
  const __m128i *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // ecx
  unsigned __int64 v12; // rdx
  int v13; // esi
  unsigned __int64 v14; // r9
  __int64 v15; // r8
  const char *v16; // r11
  __int64 v17; // rdi
  unsigned __int64 v18; // r8
  __m128i *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdi
  const void *v30; // rsi
  char *v31; // r12
  __int64 v32; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v33; // [rsp+8h] [rbp-98h]
  const char *v34; // [rsp+10h] [rbp-90h] BYREF
  char v35; // [rsp+30h] [rbp-70h]
  char v36; // [rsp+31h] [rbp-6Fh]
  const char *v37; // [rsp+40h] [rbp-60h] BYREF
  int v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h]
  unsigned int v40; // [rsp+58h] [rbp-48h]
  __int16 v41; // [rsp+60h] [rbp-40h]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v37 = "expected string";
    v41 = 259;
    return sub_ECE0E0(v2, &v37, 0, 0);
  }
  v4 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v4 == 2 )
  {
    v5 = *(_QWORD *)(v4 + 16);
    v6 = *(_QWORD *)(v4 + 8);
    v7 = v5 + 1;
  }
  else
  {
    v5 = *(_QWORD *)(v4 + 16);
    v6 = *(_QWORD *)(v4 + 8);
    v7 = 1;
    if ( v5 )
    {
      v7 = v5 - 1;
      if ( v5 == 1 )
        v7 = 1;
      ++v6;
      v5 = v7 - 1;
    }
  }
  v8 = (const __m128i *)&v37;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v34 = ".note";
  v36 = 1;
  v35 = 3;
  v41 = 257;
  v33 = sub_E71CB0(v9, (size_t *)&v34, 7, 0, 0, (__int64)&v37, 0, -1, 0);
  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v11 = *(_DWORD *)(v10 + 128);
  v12 = *(_QWORD *)(v10 + 120);
  if ( v11 )
  {
    v15 = v11;
    v14 = 32LL * v11;
    v28 = v12 + v14 - 32;
    v17 = *(_QWORD *)(v28 + 16);
    v11 = *(_DWORD *)(v28 + 24);
    v16 = *(const char **)v28;
    v13 = *(_DWORD *)(v28 + 8);
  }
  else
  {
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
  }
  v37 = v16;
  v18 = v15 + 1;
  v38 = v13;
  v39 = v17;
  v40 = v11;
  if ( v18 > *(unsigned int *)(v10 + 132) )
  {
    v29 = v10 + 120;
    v30 = (const void *)(v10 + 136);
    if ( v12 > (unsigned __int64)&v37 )
    {
      v32 = v10;
    }
    else
    {
      v14 += v12;
      v32 = v10;
      if ( (unsigned __int64)&v37 < v14 )
      {
        v31 = (char *)&v37 - v12;
        sub_C8D5F0(v29, v30, v18, 0x20u, v18, v14);
        v10 = v32;
        v12 = *(_QWORD *)(v32 + 120);
        v8 = (const __m128i *)&v31[v12];
        v14 = 32LL * *(unsigned int *)(v32 + 128);
        goto LABEL_11;
      }
    }
    sub_C8D5F0(v29, v30, v18, 0x20u, v18, v14);
    v10 = v32;
    v12 = *(_QWORD *)(v32 + 120);
    v14 = 32LL * *(unsigned int *)(v32 + 128);
  }
LABEL_11:
  v19 = (__m128i *)(v14 + v12);
  *v19 = _mm_loadu_si128(v8);
  v19[1] = _mm_loadu_si128(v8 + 1);
  ++*(_DWORD *)(v10 + 128);
  v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v20 + 176LL))(v20, v33, 0);
  v21 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v21 + 536LL))(v21, v7, 4);
  v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v22 + 536LL))(v22, 0, 4);
  v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v23 + 536LL))(v23, 1, 4);
  v24 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v24 + 512LL))(v24, v6, v5);
  v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v25 + 536LL))(v25, 0, 1);
  v26 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v26 + 608LL))(v26, 2, 0, 1, 0);
  v27 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 168LL))(v27);
  return 0;
}
