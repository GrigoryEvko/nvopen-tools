// Function: sub_1AFE150
// Address: 0x1afe150
//
__int64 __fastcall sub_1AFE150(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __m128i v6; // xmm0
  __int64 v7; // rax
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int64 *v10; // rdi
  char *v11; // rbx
  char *v12; // r12
  char *v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-288h]
  __int64 v16; // [rsp+18h] [rbp-278h] BYREF
  __m128i v17; // [rsp+20h] [rbp-270h] BYREF
  _QWORD v18[4]; // [rsp+30h] [rbp-260h] BYREF
  __int64 v19; // [rsp+50h] [rbp-240h] BYREF
  _QWORD v20[11]; // [rsp+80h] [rbp-210h] BYREF
  char *v21; // [rsp+D8h] [rbp-1B8h]
  unsigned int v22; // [rsp+E0h] [rbp-1B0h]
  char v23; // [rsp+E8h] [rbp-1A8h] BYREF

  v4 = **(_QWORD **)a2;
  v15 = **(_QWORD **)(v4 + 32);
  sub_13FD840(&v16, v4);
  sub_15C9090((__int64)&v17, &v16);
  sub_15CA330((__int64)v20, (__int64)"loop-unroll", (__int64)"PartialUnrolled", 15, &v17, v15);
  if ( v16 )
    sub_161E7C0((__int64)&v16, v16);
  sub_15CAB20((__int64)v20, "unrolled loop by a factor of ", 0x1Du);
  sub_15C9C50((__int64)&v17, "UnrollCount", 11, **(_DWORD **)(a2 + 8));
  v5 = sub_17C2270((__int64)v20, (__int64)&v17);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v5 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(v5 + 12);
  v6 = _mm_loadu_si128((const __m128i *)(v5 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v5 + 16);
  v7 = *(_QWORD *)(v5 + 40);
  *(__m128i *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 40) = v7;
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 56));
  *(_QWORD *)a1 = &unk_49ECF68;
  v9 = *(_QWORD *)(v5 + 48);
  *(__m128i *)(a1 + 56) = v8;
  *(_QWORD *)(a1 + 48) = v9;
  LOBYTE(v9) = *(_BYTE *)(v5 + 80);
  *(_BYTE *)(a1 + 80) = v9;
  if ( (_BYTE)v9 )
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(v5 + 72);
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  if ( *(_DWORD *)(v5 + 96) )
    sub_1AFDB00(a1 + 88, v5 + 88);
  v10 = (__int64 *)v18[2];
  *(_BYTE *)(a1 + 456) = *(_BYTE *)(v5 + 456);
  *(_DWORD *)(a1 + 460) = *(_DWORD *)(v5 + 460);
  *(_QWORD *)(a1 + 464) = *(_QWORD *)(v5 + 464);
  *(_QWORD *)a1 = &unk_49ECF98;
  if ( v10 != &v19 )
    j_j___libc_free_0(v10, v19 + 1);
  if ( (_QWORD *)v17.m128i_i64[0] != v18 )
    j_j___libc_free_0(v17.m128i_i64[0], v18[0] + 1LL);
  v11 = v21;
  v20[0] = &unk_49ECF68;
  v12 = &v21[88 * v22];
  if ( v21 != v12 )
  {
    do
    {
      v12 -= 88;
      v13 = (char *)*((_QWORD *)v12 + 4);
      if ( v13 != v12 + 48 )
        j_j___libc_free_0(v13, *((_QWORD *)v12 + 6) + 1LL);
      if ( *(char **)v12 != v12 + 16 )
        j_j___libc_free_0(*(_QWORD *)v12, *((_QWORD *)v12 + 2) + 1LL);
    }
    while ( v11 != v12 );
    v12 = v21;
  }
  if ( v12 != &v23 )
    _libc_free((unsigned __int64)v12);
  return a1;
}
