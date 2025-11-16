// Function: sub_1AFB590
// Address: 0x1afb590
//
__int64 __fastcall sub_1AFB590(
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
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r13
  bool v27; // al
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 *v30; // r10
  __int64 *v31; // rcx
  unsigned int v32; // r12d
  _BOOL4 v33; // r9d
  __int64 *i; // rbx
  __int64 v35; // rdi
  int v36; // eax
  __int64 *v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  _BOOL4 v40; // [rsp+1Ch] [rbp-34h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9920C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_26;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9920C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13;
  v16 = v13 + 160;
  v17 = *v14;
  v18 = v14[1];
  if ( v17 == v18 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9E06C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_24;
  }
  v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9E06C)
      + 160;
  v19 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  if ( v19 && (v20 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v19 + 104LL))(v19, &unk_4F9A488)) != 0 )
    v21 = *(_QWORD *)(v20 + 160);
  else
    v21 = 0;
  v22 = *(__int64 **)(a1 + 8);
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F9D764 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_25;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F9D764);
  v26 = sub_14CF090(v25, a2);
  v27 = sub_1636850(a1, (__int64)&unk_4FB65F4);
  v30 = *(__int64 **)(v15 + 192);
  v31 = *(__int64 **)(v15 + 200);
  v32 = 0;
  v33 = v27;
  v38 = v31;
  for ( i = v30; v38 != i; v32 |= v36 )
  {
    v35 = *i;
    v40 = v33;
    ++i;
    v36 = sub_1AFB400(v35, v39, v16, v21, v26, v33, a3, a4, a5, a6, v28, v29, a9, a10);
    v33 = v40;
  }
  return v32;
}
