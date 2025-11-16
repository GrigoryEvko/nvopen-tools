// Function: sub_1903B80
// Address: 0x1903b80
//
__int64 __fastcall sub_1903B80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
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
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  double v36; // xmm4_8
  double v37; // xmm5_8
  unsigned int v38; // r12d
  __int64 v40; // [rsp+0h] [rbp-2A0h]
  __int64 v41; // [rsp+8h] [rbp-298h]
  __m128i v42[41]; // [rsp+10h] [rbp-290h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_35;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 360;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D3C0 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_31;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D3C0);
  v19 = sub_14A4050(v18, a2);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9E06C )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_32;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9E06C);
  v25 = *(__int64 **)(a1 + 8);
  v26 = v24 + 160;
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9D764 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_33;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9D764);
  v30 = sub_14CF090(v29, a2);
  v31 = *(__int64 **)(a1 + 8);
  v32 = v30;
  v33 = *v31;
  v34 = v31[1];
  if ( v33 == v34 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F99768 )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_34;
  }
  v40 = v32;
  v41 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(
                      *(_QWORD *)(v33 + 8),
                      &unk_4F99768)
                  + 160);
  v35 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  sub_18FBF50((__int64)v42, v35, v15, v21, v26, v40, v41);
  v38 = sub_1900BB0(v42, a3, a4, a5, a6, v36, v37, a9, a10);
  sub_18FC980((__int64)v42);
  return v38;
}
