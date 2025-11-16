// Function: sub_2790350
// Address: 0x2790350
//
__int64 __fastcall sub_2790350(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int64 v12; // rax
  __int64 *v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v18[2]; // [rsp+0h] [rbp-270h] BYREF
  __int64 v19; // [rsp+10h] [rbp-260h] BYREF
  __int64 *v20; // [rsp+20h] [rbp-250h]
  __int64 v21; // [rsp+30h] [rbp-240h] BYREF
  unsigned __int64 v22[2]; // [rsp+50h] [rbp-220h] BYREF
  _QWORD v23[4]; // [rsp+60h] [rbp-210h] BYREF
  __int64 v24; // [rsp+80h] [rbp-1F0h] BYREF
  _QWORD v25[10]; // [rsp+A0h] [rbp-1D0h] BYREF
  unsigned __int64 *v26; // [rsp+F0h] [rbp-180h]
  unsigned int v27; // [rsp+F8h] [rbp-178h]
  char v28; // [rsp+100h] [rbp-170h] BYREF

  sub_B174A0((__int64)v25, (__int64)"gvn", (__int64)"LoadElim", 8, **(_QWORD **)a2);
  sub_B18290((__int64)v25, "load of type ", 0xDu);
  sub_B16360((__int64)v18, "Type", 4, *(_QWORD *)(**(_QWORD **)a2 + 8LL));
  v3 = sub_23FD640((__int64)v25, (__int64)v18);
  sub_B18290(v3, " eliminated", 0xBu);
  sub_B17B50(v3);
  sub_B18290(v3, " in favor of ", 0xDu);
  sub_B16080((__int64)v22, "InfavorOfValue", 14, **(unsigned __int8 ***)(a2 + 8));
  v6 = sub_23FD640(v3, (__int64)v22);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v6 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(v6 + 12);
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v6 + 16);
  *(__m128i *)(a1 + 24) = v9;
  v10 = _mm_loadu_si128((const __m128i *)(v6 + 48));
  v11 = _mm_loadu_si128((const __m128i *)(v6 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v12 = *(_QWORD *)(v6 + 40);
  *(__m128i *)(a1 + 48) = v10;
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v11;
  if ( *(_DWORD *)(v6 + 88) )
    sub_27900D0(a1 + 80, v6 + 80, v4, v5, v7, v8);
  v13 = (__int64 *)v23[2];
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(v6 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(v6 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(v6 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  if ( v13 != &v24 )
    j_j___libc_free_0((unsigned __int64)v13);
  if ( (_QWORD *)v22[0] != v23 )
    j_j___libc_free_0(v22[0]);
  if ( v20 != &v21 )
    j_j___libc_free_0((unsigned __int64)v20);
  if ( (__int64 *)v18[0] != &v19 )
    j_j___libc_free_0(v18[0]);
  v14 = v26;
  v25[0] = &unk_49D9D40;
  v15 = &v26[10 * v27];
  if ( v26 != v15 )
  {
    do
    {
      v15 -= 10;
      v16 = v15[4];
      if ( (unsigned __int64 *)v16 != v15 + 6 )
        j_j___libc_free_0(v16);
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15);
    }
    while ( v14 != v15 );
    v15 = v26;
  }
  if ( v15 != (unsigned __int64 *)&v28 )
    _libc_free((unsigned __int64)v15);
  return a1;
}
