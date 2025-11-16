// Function: sub_18F1640
// Address: 0x18f1640
//
__int64 __fastcall sub_18F1640(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
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
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  double v21; // xmm4_8
  double v22; // xmm5_8
  __m128i v23[6]; // [rsp+0h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_15:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F99130 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_15;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F99130);
  v15 = sub_13EB2D0(v14);
  v16 = *(__int64 **)(a1 + 8);
  v17 = (__int64 *)v15;
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F9E06C )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_16;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F9E06C);
  sub_13D1E20(v23[0].m128i_i64, a1, a2);
  return sub_18EED30(a2, v17, (__int64 **)(v20 + 160), v23, a3, a4, a5, a6, v21, v22, a9, a10);
}
