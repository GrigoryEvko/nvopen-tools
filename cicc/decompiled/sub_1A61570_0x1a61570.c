// Function: sub_1A61570
// Address: 0x1a61570
//
__int64 __fastcall sub_1A61570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned int v12; // r13d
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 **v34; // r14
  __int64 v35; // rax
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 v38; // rax
  __int64 v39; // rsi
  char v40; // r9
  __int64 v41; // [rsp+0h] [rbp-50h]
  __int64 v42; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v43[8]; // [rsp+10h] [rbp-40h] BYREF

  v42 = a2;
  if ( (unsigned __int8)sub_1404700(a1, a2) )
  {
    return 0;
  }
  else
  {
    v14 = *(__int64 **)(a1 + 8);
    v15 = *v14;
    v16 = v14[1];
    if ( v15 == v16 )
LABEL_32:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F9E06C )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_32;
    }
    v17 = *(_QWORD *)(**(_QWORD **)(v42 + 32) + 56LL);
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F9E06C);
    v19 = *(__int64 **)(a1 + 8);
    v20 = v18 + 160;
    v21 = *v19;
    v22 = v19[1];
    if ( v21 == v22 )
LABEL_35:
      BUG();
    while ( *(_UNKNOWN **)v21 != &unk_4F9920C )
    {
      v21 += 16;
      if ( v22 == v21 )
        goto LABEL_35;
    }
    v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
            *(_QWORD *)(v21 + 8),
            &unk_4F9920C);
    v24 = *(__int64 **)(a1 + 8);
    v25 = v23 + 160;
    v26 = *v24;
    v27 = v24[1];
    if ( v26 == v27 )
LABEL_34:
      BUG();
    while ( *(_UNKNOWN **)v26 != &unk_4F9D764 )
    {
      v26 += 16;
      if ( v27 == v26 )
        goto LABEL_34;
    }
    v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(
            *(_QWORD *)(v26 + 8),
            &unk_4F9D764);
    v29 = sub_14CF090(v28, v17);
    v30 = *(__int64 **)(a1 + 8);
    v41 = v29;
    v31 = *v30;
    v32 = v30[1];
    if ( v31 == v32 )
LABEL_33:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_4F9D3C0 )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_33;
    }
    v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
            *(_QWORD *)(v31 + 8),
            &unk_4F9D3C0);
    v34 = (__int64 **)sub_14A4050(v33, v17);
    v35 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
    if ( v35 && (v38 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v35 + 104LL))(v35, &unk_4F9A488)) != 0 )
      v39 = *(_QWORD *)(v38 + 160);
    else
      v39 = 0;
    v40 = *(_BYTE *)(a1 + 153);
    v43[0] = &v42;
    v43[1] = a3;
    v12 = sub_1A5F590(
            v42,
            v20,
            v25,
            v41,
            v34,
            v40,
            a4,
            a5,
            a6,
            a7,
            v36,
            v37,
            a10,
            a11,
            (void (__fastcall *)(__int64, __int64, char *, _QWORD))sub_1A4ED70,
            (__int64)v43,
            v39);
    sub_1404680(a3, v42);
  }
  return v12;
}
