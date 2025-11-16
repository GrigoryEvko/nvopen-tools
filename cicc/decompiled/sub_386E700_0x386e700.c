// Function: sub_386E700
// Address: 0x386e700
//
__int64 __fastcall sub_386E700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 i; // r14
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  __int64 v19; // r8
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v15 = sub_1648700(i);
    if ( *((_BYTE *)v15 + 16) == 23 )
    {
      v26[0] = (unsigned __int64)v15;
      sub_386B860(a1 + 512, v26, v16, v17, v18);
    }
  }
  sub_164D160(a2, *(_QWORD *)(a2 - 24), a5, a6, a7, a8, a9, a10, a11, a12);
  sub_1426980(*(_QWORD *)a1, a2, a3, a4, v19);
  if ( *(_BYTE *)(a2 + 16) == 22 )
    sub_386DE70(a1, a2, 0, a5, a6, a7, a8, v20, v21, a11, a12);
  else
    sub_386D540(a1, a2, a5, a6, a7, a8, v20, v21, a11, a12);
  v22 = *(_QWORD *)(a1 + 608);
  *(_DWORD *)(a1 + 520) = 0;
  while ( v22 )
  {
    sub_386B1F0(*(_QWORD *)(v22 + 24));
    v23 = v22;
    v22 = *(_QWORD *)(v22 + 16);
    j_j___libc_free_0(v23);
  }
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = a1 + 600;
  *(_QWORD *)(a1 + 624) = a1 + 600;
  *(_QWORD *)(a1 + 632) = 0;
  return a1 + 600;
}
