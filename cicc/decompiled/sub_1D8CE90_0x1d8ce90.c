// Function: sub_1D8CE90
// Address: 0x1d8ce90
//
__int64 __fastcall sub_1D8CE90(
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
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r8d
  int v18; // r9d
  double v19; // xmm4_8
  double v20; // xmm5_8

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9D3C0 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_6;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9D3C0);
  v14 = sub_14A4050(v13, a2);
  return sub_1D8CA60(a2, v14, a3, a4, a5, a6, v19, v20, a9, a10, v15, v16, v17, v18);
}
