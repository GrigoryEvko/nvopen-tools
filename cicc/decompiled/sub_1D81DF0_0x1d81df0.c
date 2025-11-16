// Function: sub_1D81DF0
// Address: 0x1d81df0
//
__int64 __fastcall sub_1D81DF0(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 (*v19)(); // rax
  __int64 v20; // rdi
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 (*v23)(); // rdx
  __int64 v24; // rax
  __int64 result; // rax

  v10 = (__int64 *)a1[1];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FCBA30 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_17;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FCBA30);
  v15 = (__int64 *)a1[1];
  v16 = *(_QWORD *)(v14 + 208);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_18:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9E06C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_18;
  }
  a1[21] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
             *(_QWORD *)(v17 + 8),
             &unk_4F9E06C)
         + 160;
  v19 = *(__int64 (**)())(*(_QWORD *)v16 + 16LL);
  if ( v19 == sub_16FF750 )
    BUG();
  v20 = ((__int64 (__fastcall *)(__int64, __int64))v19)(v16, a2);
  v23 = *(__int64 (**)())(*(_QWORD *)v20 + 56LL);
  v24 = 0;
  if ( v23 != sub_1D12D20 )
    v24 = ((__int64 (__fastcall *)(__int64))v23)(v20);
  a1[22] = v24;
  result = sub_1D81290(a1, a2, a3, a4, a5, a6, v21, v22, a9, a10);
  a1[21] = 0;
  a1[22] = 0;
  return result;
}
