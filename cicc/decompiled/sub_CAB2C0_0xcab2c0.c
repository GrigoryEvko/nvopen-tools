// Function: sub_CAB2C0
// Address: 0xcab2c0
//
__int64 __fastcall sub_CAB2C0(__int64 a1, int a2, int a3, unsigned __int64 *a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  __int64 v8; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rbx
  __m128i v15; // xmm0
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __m128i v19; // [rsp-70h] [rbp-70h] BYREF
  _BYTE *v20; // [rsp-60h] [rbp-60h]
  __int64 v21; // [rsp-58h] [rbp-58h]
  _QWORD v22[10]; // [rsp-50h] [rbp-50h] BYREF

  if ( *(_DWORD *)(a1 + 68) )
    return 1;
  v6 = *(_DWORD *)(a1 + 56);
  if ( v6 < a2 )
  {
    v8 = *(unsigned int *)(a1 + 200);
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 204) )
    {
      sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), v8 + 1, 4u, a5, a6);
      v8 = *(unsigned int *)(a1 + 200);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4 * v8) = v6;
    v11 = *(_QWORD *)(a1 + 40);
    v12 = *(_QWORD *)(a1 + 80);
    *(_DWORD *)(a1 + 56) = a2;
    v19.m128i_i64[0] = v11;
    ++*(_DWORD *)(a1 + 200);
    v13 = (v12 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    *(_QWORD *)(a1 + 160) += 72LL;
    v19.m128i_i64[1] = 0;
    v20 = v22;
    v21 = 0;
    LOBYTE(v22[0]) = 0;
    if ( *(_QWORD *)(a1 + 88) >= v13 + 72 && v12 )
    {
      *(_QWORD *)(a1 + 80) = v13 + 72;
      v14 = (v12 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v13 )
      {
        MEMORY[8] = a4;
        BUG();
      }
    }
    else
    {
      v14 = sub_9D1E70(a1 + 80, 72, 72, 4);
    }
    *(_QWORD *)v14 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_DWORD *)(v14 + 16) = a3;
    v15 = _mm_loadu_si128(&v19);
    *(_QWORD *)(v14 + 40) = v14 + 56;
    *(__m128i *)(v14 + 24) = v15;
    sub_CA64F0((__int64 *)(v14 + 40), v20, (__int64)&v20[v21]);
    v16 = *a4;
    v17 = *(_QWORD *)v14;
    *(_QWORD *)(v14 + 8) = a4;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v14 = v16 | v17 & 7;
    *(_QWORD *)(v16 + 8) = v14;
    v18 = v20;
    *a4 = *a4 & 7 | v14;
    if ( v18 != v22 )
      j_j___libc_free_0(v18, v22[0] + 1LL);
  }
  return 1;
}
