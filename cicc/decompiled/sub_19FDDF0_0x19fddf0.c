// Function: sub_19FDDF0
// Address: 0x19fddf0
//
__int64 __fastcall sub_19FDDF0(
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
  __int64 *v15; // rdx
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
    goto LABEL_12;
  while ( *(_UNKNOWN **)v12 != &unk_4F9B6E8 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_12;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9B6E8);
  v15 = *(__int64 **)(a1 + 8);
  v16 = (__int64 *)(v14 + 360);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9D3C0 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_12;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9D3C0);
  v20 = sub_14A4050(v19, a2);
  return sub_19FD070(a2, v16, v20, a3, a4, a5, a6, v21, v22, a9, a10);
}
