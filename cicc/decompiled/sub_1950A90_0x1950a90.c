// Function: sub_1950A90
// Address: 0x1950a90
//
__int64 __fastcall sub_1950A90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-58h] BYREF
  __int64 v35[10]; // [rsp+20h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v13 = *(__int64 **)(a1 + 8);
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9A488 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_27;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9A488);
  v17 = *(__int64 **)(a1 + 8);
  v18 = *(_QWORD *)(v16 + 160);
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F98724 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_28;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F98724);
  v22 = *(__int64 **)(a1 + 8);
  v23 = v21 + 160;
  v24 = *v22;
  v25 = v22[1];
  if ( v24 == v25 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v24 != &unk_4F9E06C )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_29;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_4F9E06C);
  v27 = *(__int64 **)(a1 + 8);
  v33 = v26 + 160;
  v28 = *v27;
  v29 = v27[1];
  if ( v28 == v29 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F9920C )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_30;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F9920C);
  v35[0] = v18;
  v35[2] = v33;
  v35[1] = v23;
  v34 = a3;
  v35[3] = v30 + 160;
  return sub_194D450(v35, a2, (__int64)sub_1948D60, (__int64)&v34, a4, a5, a6, a7, v31, v32, a10, a11);
}
