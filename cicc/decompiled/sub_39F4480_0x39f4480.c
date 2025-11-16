// Function: sub_39F4480
// Address: 0x39f4480
//
void __fastcall sub_39F4480(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r9
  __int64 v7; // r10
  const __m128i *v8; // r8
  __int64 v9; // rax
  __m128i *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  size_t v13; // r8
  void *v14; // r9
  int v15; // eax
  int v16; // r12d
  __int64 v17; // [rsp+8h] [rbp-208h]
  __int64 v18; // [rsp+10h] [rbp-200h]
  const __m128i *v19; // [rsp+18h] [rbp-1F8h]
  __int64 v20; // [rsp+20h] [rbp-1F0h]
  void *v21; // [rsp+20h] [rbp-1F0h]
  size_t v22; // [rsp+28h] [rbp-1E8h]
  _QWORD v23[4]; // [rsp+30h] [rbp-1E0h] BYREF
  int v24; // [rsp+50h] [rbp-1C0h]
  void **p_src; // [rsp+58h] [rbp-1B8h]
  _BYTE *v26; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-1A8h]
  _BYTE v28[96]; // [rsp+70h] [rbp-1A0h] BYREF
  void *src; // [rsp+D0h] [rbp-140h] BYREF
  size_t n; // [rsp+D8h] [rbp-138h]
  _BYTE v31[304]; // [rsp+E0h] [rbp-130h] BYREF

  v4 = *(_QWORD *)(a1 + 264);
  p_src = &src;
  v20 = v4;
  v27 = 0x400000000LL;
  v23[0] = &unk_49EFC48;
  n = 0x10000000000LL;
  v26 = v28;
  src = v31;
  v24 = 1;
  memset(&v23[1], 0, 24);
  sub_16E7A40((__int64)v23, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _BYTE **, __int64))(**(_QWORD **)(v20 + 16) + 24LL))(
    *(_QWORD *)(v20 + 16),
    a2,
    v23,
    &v26,
    a3);
  v5 = sub_38D4BB0(a1, 0);
  if ( (_DWORD)v27 )
  {
    v6 = 0;
    v7 = 24LL * (unsigned int)v27;
    do
    {
      v8 = (const __m128i *)&v26[v6];
      *(_DWORD *)&v26[v6 + 8] += *(_DWORD *)(v5 + 72);
      v9 = *(unsigned int *)(v5 + 120);
      if ( (unsigned int)v9 >= *(_DWORD *)(v5 + 124) )
      {
        v17 = v7;
        v18 = v6;
        v19 = v8;
        sub_16CD150(v5 + 112, (const void *)(v5 + 128), 0, 24, (int)v8, v6);
        v9 = *(unsigned int *)(v5 + 120);
        v7 = v17;
        v6 = v18;
        v8 = v19;
      }
      v6 += 24;
      v10 = (__m128i *)(*(_QWORD *)(v5 + 112) + 24 * v9);
      *v10 = _mm_loadu_si128(v8);
      v10[1].m128i_i64[0] = v8[1].m128i_i64[0];
      ++*(_DWORD *)(v5 + 120);
    }
    while ( v7 != v6 );
  }
  v11 = *(unsigned int *)(v5 + 72);
  v12 = *(unsigned int *)(v5 + 76);
  *(_QWORD *)(v5 + 56) = a3;
  v13 = (unsigned int)n;
  *(_BYTE *)(v5 + 17) = 1;
  v14 = src;
  v15 = v11;
  v16 = v13;
  if ( v13 > v12 - v11 )
  {
    v21 = src;
    v22 = v13;
    sub_16CD150(v5 + 64, (const void *)(v5 + 80), v13 + v11, 1, v13, (int)src);
    v11 = *(unsigned int *)(v5 + 72);
    v14 = v21;
    v13 = v22;
    v15 = *(_DWORD *)(v5 + 72);
  }
  if ( v16 )
  {
    memcpy((void *)(*(_QWORD *)(v5 + 64) + v11), v14, v13);
    v15 = *(_DWORD *)(v5 + 72);
  }
  *(_DWORD *)(v5 + 72) = v15 + v16;
  v23[0] = &unk_49EFD28;
  sub_16E7960((__int64)v23);
  if ( src != v31 )
    _libc_free((unsigned __int64)src);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
}
