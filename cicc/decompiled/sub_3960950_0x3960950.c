// Function: sub_3960950
// Address: 0x3960950
//
__int64 __fastcall sub_3960950(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128 a5,
        __m128 a6,
        double a7,
        double a8,
        __m128 a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
    goto LABEL_14;
  while ( *(_UNKNOWN **)v12 != &unk_4F9E06C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_14;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9E06C);
  v15 = *(__int64 **)(a1 + 8);
  v16 = v14;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F99CCC )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_14;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F99CCC);
  if ( !byte_5054920 || *(_DWORD *)(a1 + 156) <= 0x3Cu )
    return 0;
  else
    return sub_395DD20(a2, v16 + 160, v19 + 160, a3, a4, a5, a6, v20, v21, a9, a10);
}
