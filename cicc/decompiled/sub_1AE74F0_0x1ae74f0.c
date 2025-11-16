// Function: sub_1AE74F0
// Address: 0x1ae74f0
//
__int64 __fastcall sub_1AE74F0(
        __int64 a1,
        __int64 a2,
        double a3,
        __m128i a4,
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
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  double v17; // xmm4_8
  double v18; // xmm5_8

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_12;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8)
      + 360;
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v14 && (v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4F9E06C)) != 0 )
    v16 = v15 + 160;
  else
    v16 = 0;
  if ( (unsigned __int8)sub_1560180(a2 + 112, 34) )
    return 0;
  else
    return sub_1AE6420(a2, v13, v16, a3, a4, a5, a6, v17, v18, a9, a10);
}
