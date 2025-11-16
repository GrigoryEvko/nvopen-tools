// Function: sub_1A186D0
// Address: 0x1a186d0
//
__int64 __fastcall sub_1A186D0(
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
  __int64 *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9B6E8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_8;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9B6E8);
  return sub_1A182F0(a2, v13, v16 + 360, a3, a4, a5, a6, v17, v18, a9, a10);
}
