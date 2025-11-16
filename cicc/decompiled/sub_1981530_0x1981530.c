// Function: sub_1981530
// Address: 0x1981530
//
__int64 __fastcall sub_1981530(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128 a5,
        __m128 a6,
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
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_21:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9920C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_21;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9920C);
  v15 = *(__int64 **)(a1 + 8);
  v16 = v14 + 160;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_22:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_5051F8C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_22;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_5051F8C);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9E06C )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_23;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9E06C);
  v27[0] = v21;
  return sub_197EA40(
           v16,
           v24 + 160,
           (__int64 (__fastcall *)(__int64, __int64))sub_197CF50,
           (__int64)v27,
           a3,
           a4,
           a5,
           a6,
           v25,
           v26,
           a9,
           a10);
}
