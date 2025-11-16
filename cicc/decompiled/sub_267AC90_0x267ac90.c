// Function: sub_267AC90
// Address: 0x267ac90
//
__int64 __fastcall sub_267AC90(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // r13d
  unsigned int v10; // r13d
  int v11; // eax
  __int8 *v12; // rsi
  size_t v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 v24[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v25[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+30h] [rbp-50h] BYREF

  sub_B18290(a3, "Replaced globalized variable with ", 0x22u);
  v5 = **a2;
  if ( *(_DWORD *)(v5 + 32) <= 0x40u )
    v6 = *(_QWORD *)(v5 + 24);
  else
    v6 = **(_QWORD **)(v5 + 24);
  sub_B16B10(v24, "SharedMemory", 12, v6);
  v7 = sub_23FD640(a3, (__int64)v24);
  v8 = **a2;
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
  {
    v12 = " bytes ";
    v13 = (*(_QWORD *)(v8 + 24) != 1) + 6LL;
    if ( *(_QWORD *)(v8 + 24) == 1 )
      v12 = " byte ";
  }
  else
  {
    v10 = v9 - 1;
    v11 = sub_C444A0(v8 + 24);
    v12 = " bytes ";
    v13 = (v11 != v10) + 6LL;
    if ( v11 == v10 )
      v12 = " byte ";
  }
  sub_B18290(v7, v12, v13);
  sub_B18290(v7, "of shared memory.", 0x11u);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(v7 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(v7 + 12);
  v18 = _mm_loadu_si128((const __m128i *)(v7 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(v7 + 16);
  *(__m128i *)(a1 + 24) = v18;
  v19 = _mm_loadu_si128((const __m128i *)(v7 + 48));
  v20 = _mm_loadu_si128((const __m128i *)(v7 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v21 = *(_QWORD *)(v7 + 40);
  *(__m128i *)(a1 + 48) = v19;
  *(_QWORD *)(a1 + 40) = v21;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v20;
  if ( *(_DWORD *)(v7 + 88) )
    sub_26781A0(a1 + 80, v7 + 80, v14, v15, v16, v17);
  v22 = (__int64 *)v25[2];
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(v7 + 416);
  *(_DWORD *)(a1 + 420) = *(_DWORD *)(v7 + 420);
  *(_QWORD *)(a1 + 424) = *(_QWORD *)(v7 + 424);
  *(_QWORD *)a1 = &unk_49D9D78;
  if ( v22 != &v26 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( (_QWORD *)v24[0] != v25 )
    j_j___libc_free_0(v24[0]);
  return a1;
}
