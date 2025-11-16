// Function: sub_18C8660
// Address: 0x18c8660
//
__int64 __fastcall sub_18C8660(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // rbx
  __int64 v18; // r12
  unsigned int i; // r14d
  bool v20; // zf
  __int64 v21; // rdi
  int v22; // eax

  v9 = sub_15E0FD0(208);
  v11 = sub_16321A0(a1, (__int64)v9, v10);
  v12 = sub_15E0FD0(207);
  v14 = sub_16321A0(a1, (__int64)v12, v13);
  if ( (!v11 || !*(_QWORD *)(v11 + 8)) && (!v14 || !*(_QWORD *)(v14 + 8)) )
    return 0;
  v17 = *(_QWORD *)(a1 + 16);
  v18 = a1 + 8;
  for ( i = 0; v17 != v18; i |= v22 )
  {
    v20 = v17 == 0;
    v21 = v17 - 56;
    v17 = *(_QWORD *)(v17 + 8);
    if ( v20 )
      v21 = 0;
    LOBYTE(v22) = sub_18C7D70(v21, a2, a3, a4, a5, v15, v16, a8, a9);
  }
  return i;
}
