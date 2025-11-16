// Function: sub_18E5F80
// Address: 0x18e5f80
//
__int64 __fastcall sub_18E5F80(
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
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F98D2C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_8;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F98D2C);
  return sub_18E56C0(a2, v14 + 160, a3, a4, a5, a6, v15, v16, a9, a10);
}
