// Function: sub_38800E0
// Address: 0x38800e0
//
void __fastcall sub_38800E0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  double v18; // xmm4_8
  double v19; // xmm5_8
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // r13
  unsigned __int64 v24; // r8
  __int64 v25; // r13
  __int64 v26; // r12
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // r13
  __int64 v32; // rsi
  __int64 i; // r12

  v11 = a1 + 1480;
  v12 = *(_QWORD *)(a1 + 1464);
  if ( v12 != v11 )
  {
    a2 = *(_QWORD *)(a1 + 1480) + 1LL;
    j_j___libc_free_0(v12);
  }
  sub_387E620(*(_QWORD *)(a1 + 1408));
  sub_387EFE0(*(_QWORD **)(a1 + 1360));
  v13 = *(_QWORD *)(a1 + 1320);
  if ( v13 )
  {
    a2 = *(_QWORD *)(a1 + 1336) - v13;
    j_j___libc_free_0(v13);
  }
  sub_387F5B0(*(_QWORD **)(a1 + 1288));
  sub_387FB80(*(_QWORD **)(a1 + 1240));
  sub_387E1C0(*(_QWORD **)(a1 + 1192));
  sub_387ED10(*(_QWORD **)(a1 + 1144));
  sub_387FF90(*(_QWORD **)(a1 + 1088));
  sub_387F2B0(*(_QWORD **)(a1 + 1040));
  v14 = *(_QWORD *)(a1 + 1000);
  if ( v14 )
  {
    a2 = *(_QWORD *)(a1 + 1016) - v14;
    j_j___libc_free_0(v14);
  }
  sub_387E280(*(_QWORD *)(a1 + 968));
  sub_387F880(*(_QWORD **)(a1 + 920));
  sub_387E7F0(*(_QWORD **)(a1 + 872), a2, v15, v16, v17, a3, a4, a5, a6, v18, v19, a9, a10);
  v20 = *(_QWORD *)(a1 + 824);
  while ( v20 )
  {
    v21 = v20;
    sub_387EA70(*(_QWORD **)(v20 + 24));
    v22 = *(_QWORD *)(v20 + 40);
    v20 = *(_QWORD *)(v20 + 16);
    if ( v22 )
      sub_161E7C0(v21 + 40, v22);
    j_j___libc_free_0(v21);
  }
  sub_387E450(*(_QWORD *)(a1 + 776));
  if ( *(_DWORD *)(a1 + 740) )
  {
    v23 = *(unsigned int *)(a1 + 736);
    v24 = *(_QWORD *)(a1 + 728);
    if ( (_DWORD)v23 )
    {
      v25 = 8 * v23;
      v26 = 0;
      do
      {
        v27 = *(_QWORD *)(v24 + v26);
        if ( v27 && v27 != -8 )
        {
          _libc_free(v27);
          v24 = *(_QWORD *)(a1 + 728);
        }
        v26 += 8;
      }
      while ( v25 != v26 );
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 728);
  }
  _libc_free(v24);
  v28 = *(_QWORD *)(a1 + 200);
  if ( v28 != a1 + 216 )
    _libc_free(v28);
  if ( *(_DWORD *)(a1 + 160) > 0x40u )
  {
    v29 = *(_QWORD *)(a1 + 152);
    if ( v29 )
      j_j___libc_free_0_0(v29);
  }
  if ( *(void **)(a1 + 128) == sub_16982C0() )
  {
    v31 = *(_QWORD *)(a1 + 136);
    if ( v31 )
    {
      v32 = 32LL * *(_QWORD *)(v31 - 8);
      for ( i = v31 + v32; v31 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v31 - 8);
    }
  }
  else
  {
    sub_1698460(a1 + 128);
  }
  v30 = *(_QWORD *)(a1 + 72);
  if ( v30 != a1 + 88 )
    j_j___libc_free_0(v30);
}
