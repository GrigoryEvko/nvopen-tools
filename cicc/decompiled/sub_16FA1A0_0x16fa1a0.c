// Function: sub_16FA1A0
// Address: 0x16fa1a0
//
__int64 __fastcall sub_16FA1A0(__int64 a1, int a2, int a3, unsigned __int64 *a4, int a5, int a6)
{
  __int64 v7; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __m128i v18; // [rsp-60h] [rbp-60h] BYREF
  _QWORD *v19; // [rsp-50h] [rbp-50h]
  __int64 v20; // [rsp-48h] [rbp-48h]
  _QWORD v21[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 68) )
    return 1;
  if ( *(_DWORD *)(a1 + 56) < a2 )
  {
    v7 = *(unsigned int *)(a1 + 208);
    if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 212) )
    {
      sub_16CD150(a1 + 200, (const void *)(a1 + 216), 0, 4, a5, a6);
      v7 = *(unsigned int *)(a1 + 208);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 200) + 4 * v7) = *(_DWORD *)(a1 + 56);
    v10 = *(_QWORD *)(a1 + 40);
    ++*(_DWORD *)(a1 + 208);
    *(_DWORD *)(a1 + 56) = a2;
    v19 = v21;
    v20 = 0;
    LOBYTE(v21[0]) = 0;
    v18 = (__m128i)v10;
    v11 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
    v12 = _mm_loadu_si128(&v18);
    v13 = v11;
    *(_QWORD *)v11 = 0;
    v14 = v20;
    *(_QWORD *)(v11 + 8) = 0;
    *(__m128i *)(v11 + 24) = v12;
    *(_DWORD *)(v11 + 16) = a3;
    *(_QWORD *)(v11 + 40) = v11 + 56;
    sub_16F6740((__int64 *)(v11 + 40), v21, (__int64)v21 + v14);
    v15 = *a4;
    v16 = *(_QWORD *)v13;
    *(_QWORD *)(v13 + 8) = a4;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v13 = v15 | v16 & 7;
    *(_QWORD *)(v15 + 8) = v13;
    v17 = v19;
    *a4 = v13 | *a4 & 7;
    if ( v17 != v21 )
      j_j___libc_free_0(v17, v21[0] + 1LL);
  }
  return 1;
}
