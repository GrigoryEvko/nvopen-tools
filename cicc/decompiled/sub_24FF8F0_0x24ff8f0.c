// Function: sub_24FF8F0
// Address: 0x24ff8f0
//
__int64 __fastcall sub_24FF8F0(__int64 a1, __int64 a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __int64 v14; // rax
  __int64 *v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v20[2]; // [rsp+0h] [rbp-270h] BYREF
  __int64 v21; // [rsp+10h] [rbp-260h] BYREF
  __int64 *v22; // [rsp+20h] [rbp-250h]
  __int64 v23; // [rsp+30h] [rbp-240h] BYREF
  __int64 v24[2]; // [rsp+50h] [rbp-220h] BYREF
  _QWORD v25[4]; // [rsp+60h] [rbp-210h] BYREF
  __int64 v26; // [rsp+80h] [rbp-1F0h] BYREF
  _QWORD v27[10]; // [rsp+A0h] [rbp-1D0h] BYREF
  unsigned __int64 *v28; // [rsp+F0h] [rbp-180h]
  unsigned int v29; // [rsp+F8h] [rbp-178h]
  char v30; // [rsp+100h] [rbp-170h] BYREF

  sub_B17560((__int64)v27, (__int64)"argpromotion", (__int64)"ArgumentRemoved", 15, **(_QWORD **)a2);
  sub_B18290((__int64)v27, "eliminating argument ", 0x15u);
  v3 = (char *)sub_BD5D20(**(_QWORD **)(a2 + 8));
  sub_B16430((__int64)v20, "ArgName", 7u, v3, v4);
  v5 = sub_23FD640((__int64)v27, (__int64)v20);
  sub_B18290(v5, "(", 1u);
  sub_B169E0(v24, "ArgIndex", 8, **(_DWORD **)(a2 + 16));
  v6 = sub_23FD640(v5, (__int64)v24);
  sub_B18290(v6, ")", 1u);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v6 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(v6 + 12);
  v11 = _mm_loadu_si128((const __m128i *)(v6 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v6 + 16);
  *(__m128i *)(a1 + 24) = v11;
  v12 = _mm_loadu_si128((const __m128i *)(v6 + 48));
  v13 = _mm_loadu_si128((const __m128i *)(v6 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v14 = *(_QWORD *)(v6 + 40);
  *(__m128i *)(a1 + 48) = v12;
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v13;
  if ( *(_DWORD *)(v6 + 88) )
    sub_24FF670(a1 + 80, v6 + 80, v7, v8, v9, v10);
  v15 = (__int64 *)v25[2];
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(v6 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(v6 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(v6 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  if ( v15 != &v26 )
    j_j___libc_free_0((unsigned __int64)v15);
  if ( (_QWORD *)v24[0] != v25 )
    j_j___libc_free_0(v24[0]);
  if ( v22 != &v23 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0]);
  v16 = v28;
  v27[0] = &unk_49D9D40;
  v17 = &v28[10 * v29];
  if ( v28 != v17 )
  {
    do
    {
      v17 -= 10;
      v18 = v17[4];
      if ( (unsigned __int64 *)v18 != v17 + 6 )
        j_j___libc_free_0(v18);
      if ( (unsigned __int64 *)*v17 != v17 + 2 )
        j_j___libc_free_0(*v17);
    }
    while ( v16 != v17 );
    v17 = v28;
  }
  if ( v17 != (unsigned __int64 *)&v30 )
    _libc_free((unsigned __int64)v17);
  return a1;
}
