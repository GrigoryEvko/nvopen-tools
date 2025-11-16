// Function: sub_1701F50
// Address: 0x1701f50
//
__int64 __fastcall sub_1701F50(
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
  __int64 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_14;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 360;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_13:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9E06C )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_13;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9E06C);
  return sub_1701250(a2, v15, v18 + 160, a3, a4, a5, a6, v19, v20, a9, a10);
}
