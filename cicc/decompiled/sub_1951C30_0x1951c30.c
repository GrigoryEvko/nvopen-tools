// Function: sub_1951C30
// Address: 0x1951c30
//
__int64 __fastcall sub_1951C30(
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
  __int64 *v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  double v32; // xmm4_8
  double v33; // xmm5_8
  __m128i v34; // [rsp+0h] [rbp-60h] BYREF
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h]

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9E06C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_27;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9E06C);
  v15 = *(__int64 **)(a1 + 8);
  v16 = v14 + 160;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9B6E8 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_28;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9B6E8);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19 + 360;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9D764 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_29;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9D764);
  v25 = sub_14CF090(v24, a2);
  v26 = *(__int64 **)(a1 + 8);
  v27 = v25;
  v28 = *v26;
  v29 = v26[1];
  if ( v28 == v29 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F99CB0 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_30;
  }
  v30 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(
                      *(_QWORD *)(v28 + 8),
                      &unk_4F99CB0)
                  + 160);
  v31 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v34.m128i_i64[1] = v21;
  v35 = v16;
  v36 = v27;
  v34.m128i_i64[0] = v31;
  v37 = 0;
  return sub_1950FB0(a2, &v34, v30, a3, a4, a5, a6, v32, v33, a9, a10);
}
