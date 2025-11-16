// Function: sub_267AEB0
// Address: 0x267aeb0
//
__int64 __fastcall sub_267AEB0(__int64 a1, __int64 **a2, __int64 a3)
{
  __int8 *v5; // rsi
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // rax
  __int64 *i; // [rsp+28h] [rbp-98h]
  __int64 v22; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v23[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v24; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v25; // [rsp+60h] [rbp-60h]
  __int64 v26; // [rsp+70h] [rbp-50h] BYREF

  sub_B18290(a3, "Parallel region merged with parallel region", 0x2Bu);
  v5 = "s";
  if ( *((_DWORD *)*a2 + 2) <= 2u )
    v5 = (__int8 *)byte_3F871B3;
  sub_B18290(a3, v5, *((_DWORD *)*a2 + 2) >= 3u);
  sub_B18290(a3, " at ", 4u);
  v6 = *a2;
  v7 = **a2;
  v8 = (__int64 *)(v7 + 8LL * *((unsigned int *)v6 + 2));
  v9 = (__int64 *)(v7 + 8);
  for ( i = v8; i != v9; ++v9 )
  {
    v10 = *v9;
    v11 = *(_QWORD *)(*v9 + 48);
    v22 = v11;
    if ( v11 )
      sub_B96E90((__int64)&v22, v11, 1);
    sub_B16E20((__int64)v23, "OpenMPParallelMerge", 19, &v22);
    sub_23FD640(a3, (__int64)v23);
    if ( v25 != &v26 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( (__int64 *)v23[0] != &v24 )
      j_j___libc_free_0(v23[0]);
    if ( v22 )
      sub_B91220((__int64)&v22, v22);
    if ( *(_QWORD *)(**a2 + 8LL * *((unsigned int *)*a2 + 2) - 8) != v10 )
      sub_B18290(a3, ", ", 2u);
  }
  sub_B18290(a3, ".", 1u);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a3 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a3 + 12);
  v15 = _mm_loadu_si128((const __m128i *)(a3 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
  *(__m128i *)(a1 + 24) = v15;
  v16 = _mm_loadu_si128((const __m128i *)(a3 + 48));
  v17 = _mm_loadu_si128((const __m128i *)(a3 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v18 = *(_QWORD *)(a3 + 40);
  *(__m128i *)(a1 + 48) = v16;
  *(_QWORD *)(a1 + 40) = v18;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v17;
  if ( *(_DWORD *)(a3 + 88) )
    sub_26781A0(a1 + 80, a3 + 80, v12, a1, v13, v14);
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(a3 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(a3 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(a3 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  return a1;
}
