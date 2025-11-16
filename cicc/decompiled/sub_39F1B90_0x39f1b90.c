// Function: sub_39F1B90
// Address: 0x39f1b90
//
void __fastcall sub_39F1B90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  int v5; // r9d
  const __m128i *v6; // rax
  const __m128i *v7; // r8
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdx
  size_t v13; // r8
  void *v14; // r9
  int v15; // eax
  int v16; // r12d
  const __m128i *v17; // [rsp+0h] [rbp-200h]
  const __m128i *v18; // [rsp+8h] [rbp-1F8h]
  void *v19; // [rsp+10h] [rbp-1F0h]
  size_t v20; // [rsp+18h] [rbp-1E8h]
  _QWORD v21[4]; // [rsp+20h] [rbp-1E0h] BYREF
  int v22; // [rsp+40h] [rbp-1C0h]
  void **p_src; // [rsp+48h] [rbp-1B8h]
  const __m128i *v24; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v25; // [rsp+58h] [rbp-1A8h]
  _BYTE v26[96]; // [rsp+60h] [rbp-1A0h] BYREF
  void *src; // [rsp+C0h] [rbp-140h] BYREF
  size_t n; // [rsp+C8h] [rbp-138h]
  _BYTE v29[304]; // [rsp+D0h] [rbp-130h] BYREF

  v4 = sub_38D4BB0(a1, 0);
  v25 = 0x400000000LL;
  n = 0x10000000000LL;
  v21[0] = &unk_49EFC48;
  p_src = &src;
  v24 = (const __m128i *)v26;
  src = v29;
  v22 = 1;
  memset(&v21[1], 0, 24);
  sub_16E7A40((__int64)v21, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, const __m128i **, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 264)
                                                                                             + 16LL)
                                                                               + 24LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 264) + 16LL),
    a2,
    v21,
    &v24,
    a3);
  v6 = v24;
  v7 = (const __m128i *)((char *)v24 + 24 * (unsigned int)v25);
  if ( v24 != v7 )
  {
    do
    {
      v6->m128i_i32[2] += *(_DWORD *)(v4 + 72);
      v8 = *(unsigned int *)(v4 + 120);
      if ( (unsigned int)v8 >= *(_DWORD *)(v4 + 124) )
      {
        v17 = v6;
        v18 = v7;
        sub_16CD150(v4 + 112, (const void *)(v4 + 128), 0, 24, (int)v7, v5);
        v8 = *(unsigned int *)(v4 + 120);
        v6 = v17;
        v7 = v18;
      }
      v9 = _mm_loadu_si128(v6);
      v6 = (const __m128i *)((char *)v6 + 24);
      v10 = *(_QWORD *)(v4 + 112) + 24 * v8;
      *(__m128i *)v10 = v9;
      *(_QWORD *)(v10 + 16) = v6[-1].m128i_i64[1];
      ++*(_DWORD *)(v4 + 120);
    }
    while ( v7 != v6 );
  }
  v11 = *(unsigned int *)(v4 + 72);
  v12 = *(unsigned int *)(v4 + 76);
  *(_QWORD *)(v4 + 56) = a3;
  v13 = (unsigned int)n;
  *(_BYTE *)(v4 + 17) = 1;
  v14 = src;
  v15 = v11;
  v16 = v13;
  if ( v13 > v12 - v11 )
  {
    v19 = src;
    v20 = v13;
    sub_16CD150(v4 + 64, (const void *)(v4 + 80), v13 + v11, 1, v13, (int)src);
    v11 = *(unsigned int *)(v4 + 72);
    v14 = v19;
    v13 = v20;
    v15 = *(_DWORD *)(v4 + 72);
  }
  if ( v16 )
  {
    memcpy((void *)(*(_QWORD *)(v4 + 64) + v11), v14, v13);
    v15 = *(_DWORD *)(v4 + 72);
  }
  *(_DWORD *)(v4 + 72) = v15 + v16;
  v21[0] = &unk_49EFD28;
  sub_16E7960((__int64)v21);
  if ( src != v29 )
    _libc_free((unsigned __int64)src);
  if ( v24 != (const __m128i *)v26 )
    _libc_free((unsigned __int64)v24);
}
