// Function: sub_3906450
// Address: 0x3906450
//
__int64 __fastcall sub_3906450(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rsi
  __m128i *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 *v36; // rdx
  const char *v37; // [rsp+0h] [rbp-70h] BYREF
  char v38; // [rsp+10h] [rbp-60h]
  char v39; // [rsp+11h] [rbp-5Fh]
  __m128i v40; // [rsp+20h] [rbp-50h] BYREF
  __m128i v41; // [rsp+30h] [rbp-40h] BYREF

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 3 )
  {
    v6 = sub_3909460(*(_QWORD *)(a1 + 8));
    if ( *(_DWORD *)v6 == 2 )
    {
      v10 = *(_QWORD *)(v6 + 8);
      v8 = *(_QWORD *)(v6 + 16);
    }
    else
    {
      v7 = *(_QWORD *)(v6 + 16);
      v8 = 0;
      if ( v7 )
      {
        v9 = v7 - 1;
        if ( v7 == 1 )
          v9 = 1;
        if ( v9 <= v7 )
          v7 = v9;
        v8 = v7 - 1;
        v7 = 1;
      }
      v10 = *(_QWORD *)(v6 + 8) + v7;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v37 = ".note";
    v39 = 1;
    v38 = 3;
    v41.m128i_i16[0] = 257;
    v12 = sub_38C3B80(v11, (__int64)&v37, 7, 0, 0, (__int64)&v40, -1, 0);
    v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v16 = *(unsigned int *)(v15 + 120);
    if ( (_DWORD)v16 )
    {
      v36 = (__int64 *)(*(_QWORD *)(v15 + 112) + 32LL * (unsigned int)v16 - 32);
      v20 = v36[2];
      v19 = v36[3];
      v18 = *v36;
      v17 = v36[1];
    }
    else
    {
      v17 = 0;
      v18 = 0;
      v19 = 0;
      v20 = 0;
    }
    v40.m128i_i64[0] = v18;
    v40.m128i_i64[1] = v17;
    v41.m128i_i64[0] = v20;
    v41.m128i_i64[1] = v19;
    if ( *(_DWORD *)(v15 + 124) <= (unsigned int)v16 )
    {
      sub_16CD150(v15 + 112, (const void *)(v15 + 128), 0, 32, v13, v14);
      v16 = *(unsigned int *)(v15 + 120);
    }
    v21 = (__m128i *)(*(_QWORD *)(v15 + 112) + 32 * v16);
    *v21 = _mm_loadu_si128(&v40);
    v21[1] = _mm_loadu_si128(&v41);
    ++*(_DWORD *)(v15 + 120);
    v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v22 + 160LL))(v22, v12, 0);
    v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v23 + 424LL))(v23, v8 + 1, 4);
    v24 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v24 + 424LL))(v24, 0, 4);
    v25 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v25 + 424LL))(v25, 1, 4);
    v26 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64))(*(_QWORD *)v26 + 400LL))(v26, v10, v8);
    v27 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v27 + 424LL))(v27, 0, 1);
    v28 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v28 + 512LL))(v28, 4, 0, 1, 0);
    v29 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v30 = *(unsigned int *)(v29 + 120);
    v31 = v29;
    result = 0;
    v32 = v30;
    if ( (unsigned int)v30 > 1 )
    {
      v33 = *(_QWORD *)(v31 + 112) + 32 * v30;
      v34 = *(_QWORD *)(v33 - 64);
      v35 = *(_QWORD *)(v33 - 56);
      if ( *(_QWORD *)(v33 - 24) != v35 || *(_QWORD *)(v33 - 32) != v34 )
      {
        (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v31 + 152LL))(v31, v34, v35, v32);
        LODWORD(v32) = *(_DWORD *)(v31 + 120);
      }
      *(_DWORD *)(v31 + 120) = v32 - 1;
      return 0;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 8);
    v40.m128i_i64[0] = (__int64)"unexpected token in '.version' directive";
    v41.m128i_i16[0] = 259;
    return sub_3909CF0(v4, &v40, 0, 0, v2, v3);
  }
  return result;
}
