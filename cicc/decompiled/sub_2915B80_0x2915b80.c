// Function: sub_2915B80
// Address: 0x2915b80
//
void __fastcall sub_2915B80(__int64 a1, const void *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // r14
  unsigned __int64 v7; // r12
  unsigned int v8; // r13d
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // r12d
  __int64 v12; // rcx
  __int64 v13; // rdx
  __m128i *v14; // r13
  const __m128i *v15; // r12
  __m128i *v16; // r12
  __m128i *v17; // r10
  __int64 v18; // rbx
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  const __m128i *v21; // [rsp+18h] [rbp-48h]
  __m128i *v22; // [rsp+20h] [rbp-40h]

  v6 = 24 * a3;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((24 * a3) >> 3);
  v8 = *(_DWORD *)(a1 + 32);
  v9 = v8;
  if ( v7 + v8 > *(unsigned int *)(a1 + 36) )
  {
    sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v7 + v8, 0x18u, v7 + v8, a6);
    v9 = *(unsigned int *)(a1 + 32);
  }
  v10 = *(_QWORD *)(a1 + 24);
  if ( v6 )
  {
    memcpy((void *)(v10 + 24 * v9), a2, v6);
    v10 = *(_QWORD *)(a1 + 24);
    LODWORD(v9) = *(_DWORD *)(a1 + 32);
  }
  v11 = v9 + v7;
  *(_DWORD *)(a1 + 32) = v11;
  v12 = 24LL * (int)v8;
  v13 = 24LL * v11;
  v14 = (__m128i *)(v10 + v12);
  v15 = (const __m128i *)(v10 + v13);
  sub_2912200(&v20, (__m128i *)(v10 + v12), 0xAAAAAAAAAAAAAAABLL * ((v13 - v12) >> 3));
  if ( v22 )
    sub_2915A90(v14, v15, v22, v21);
  else
    sub_2914CE0(v14, v15);
  j_j___libc_free_0((unsigned __int64)v22);
  v16 = *(__m128i **)(a1 + 24);
  v17 = (__m128i *)((char *)v16 + 24 * *(unsigned int *)(a1 + 32));
  if ( v14 != v16 && v17 != v14 )
  {
    v19 = (__int64)&v16->m128i_i64[3 * *(unsigned int *)(a1 + 32)];
    v18 = 0xAAAAAAAAAAAAAAABLL * (((char *)v17 - (char *)v14) >> 3);
    sub_2912200(
      &v20,
      *(__m128i **)(a1 + 24),
      0xAAAAAAAAAAAAAAABLL * (((char *)v14 - (char *)v16) >> 3)
    - 0x5555555555555555LL * (((char *)v17 - (char *)v14) >> 3));
    if ( v22 )
      sub_2915570(v16, v14, v19, 0xAAAAAAAAAAAAAAABLL * (((char *)v14 - (char *)v16) >> 3), v18, v22, v21);
    else
      sub_2914B10(v16, v14, v19, 0xAAAAAAAAAAAAAAABLL * (((char *)v14 - (char *)v16) >> 3), v18);
    j_j___libc_free_0((unsigned __int64)v22);
  }
}
