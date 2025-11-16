// Function: sub_1718CD0
// Address: 0x1718cd0
//
__int64 __fastcall sub_1718CD0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128i a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-40h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F96DB4 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_37;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F96DB4);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD *)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9D764 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_39;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9D764);
  v20 = sub_14CF090(v19, a2);
  v21 = *(__int64 **)(a1 + 8);
  v22 = v20;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_40:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F9B6E8 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_40;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F9B6E8);
  v26 = *(__int64 **)(a1 + 8);
  v40 = v25 + 360;
  v27 = *v26;
  v28 = v26[1];
  if ( v27 == v28 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9E06C )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_41;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9E06C);
  v30 = *(__int64 **)(a1 + 8);
  v31 = v29 + 160;
  v32 = *v30;
  v33 = v30[1];
  if ( v32 == v33 )
LABEL_38:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F99CB0 )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_38;
  }
  v39 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(
                      *(_QWORD *)(v32 + 8),
                      &unk_4F99CB0)
                  + 160);
  v34 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9920C, 1u);
  if ( v34 && (v37 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v34 + 104LL))(v34, &unk_4F9920C)) != 0 )
    v38 = v37 + 160;
  else
    v38 = 0;
  return sub_1717500(
           a2,
           a1 + 160,
           v16,
           v22,
           v40,
           v31,
           a3,
           a4,
           a5,
           a6,
           v35,
           v36,
           a9,
           a10,
           v39,
           *(_BYTE *)(a1 + 2256),
           *(_BYTE *)(a1 + 2257),
           v38);
}
