// Function: sub_3139E10
// Address: 0x3139e10
//
void __fastcall sub_3139E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  _QWORD *v9; // r13
  void (__fastcall *v10)(_QWORD *, __int64, __int64); // rax
  void *v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // r14d
  size_t v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __m128i *v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r13
  void (__fastcall *v23)(__int64 *, __int64, __int64); // rax
  void *v24; // rdi
  __int64 v25; // rax
  int v26; // r12d
  unsigned int v27; // [rsp+Ch] [rbp-44h]
  unsigned int v28; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v15 = a1 + 16;
    v19 = (__m128i *)sub_C8D7D0(a1, a1 + 16, 0, 0x58u, v29, a6);
    v20 = *(unsigned int *)(a1 + 8);
    v21 = 5 * v20;
    v22 = &v19->m128i_i64[11 * v20];
    if ( v22 )
    {
      v22[2] = 0;
      v23 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a2 + 16);
      if ( v23 )
      {
        v23(v22, a2, 2);
        v22[3] = *(_QWORD *)(a2 + 24);
        v22[2] = *(_QWORD *)(a2 + 16);
      }
      v24 = v22 + 9;
      v22[4] = *(_QWORD *)(a2 + 32);
      v22[5] = *(_QWORD *)(a2 + 40);
      v25 = *(_QWORD *)(a2 + 48);
      v22[7] = (__int64)(v22 + 9);
      v22[6] = v25;
      v22[8] = 0x200000000LL;
      v17 = *(unsigned int *)(a2 + 64);
      if ( (_DWORD)v17 )
      {
        v18 = (__int64)(v22 + 7);
        if ( v22 + 7 != (__int64 *)(a2 + 56) )
        {
          v21 = 8LL * (unsigned int)v17;
          if ( (unsigned int)v17 <= 2
            || (v28 = *(_DWORD *)(a2 + 64),
                sub_C8D5F0((__int64)(v22 + 7), v22 + 9, (unsigned int)v17, 8u, v17, v18),
                v24 = (void *)v22[7],
                v17 = v28,
                (v21 = 8LL * *(unsigned int *)(a2 + 64)) != 0) )
          {
            v27 = v17;
            memcpy(v24, *(const void **)(a2 + 56), v21);
            v17 = v27;
          }
          *((_DWORD *)v22 + 16) = v17;
        }
      }
    }
    sub_3139CC0(a1, v19, v21, v16, v17, v18);
    v26 = v29[0];
    if ( v15 != *(_QWORD *)a1 )
      _libc_free(*(_QWORD *)a1);
    *(_QWORD *)a1 = v19;
    *(_DWORD *)(a1 + 12) = v26;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 8);
    v9 = (_QWORD *)(*(_QWORD *)a1 + 88 * v7);
    if ( v9 )
    {
      v9[2] = 0;
      v10 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a2 + 16);
      if ( v10 )
      {
        v10(v9, a2, 2);
        v9[3] = *(_QWORD *)(a2 + 24);
        v9[2] = *(_QWORD *)(a2 + 16);
      }
      v11 = v9 + 9;
      v9[4] = *(_QWORD *)(a2 + 32);
      v9[5] = *(_QWORD *)(a2 + 40);
      v12 = *(_QWORD *)(a2 + 48);
      v9[7] = v9 + 9;
      v9[6] = v12;
      v9[8] = 0x200000000LL;
      v13 = *(_DWORD *)(a2 + 64);
      if ( v13 && v9 + 7 != (_QWORD *)(a2 + 56) )
      {
        v14 = 8LL * v13;
        if ( v13 <= 2
          || (sub_C8D5F0((__int64)(v9 + 7), v9 + 9, v13, 8u, (__int64)(v9 + 7), v13),
              v11 = (void *)v9[7],
              (v14 = 8LL * *(unsigned int *)(a2 + 64)) != 0) )
        {
          memcpy(v11, *(const void **)(a2 + 56), v14);
        }
        *((_DWORD *)v9 + 16) = v13;
      }
      v8 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + 1;
  }
}
