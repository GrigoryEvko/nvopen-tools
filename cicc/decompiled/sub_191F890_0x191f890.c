// Function: sub_191F890
// Address: 0x191f890
//
__int64 __fastcall sub_191F890(
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
  char v10; // r8
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // [rsp+0h] [rbp-50h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+18h] [rbp-38h]

  v10 = sub_1636880(a1, a2);
  result = 0;
  if ( v10 )
    return result;
  v12 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9920C, 1u);
  v13 = v12;
  if ( v12 )
    v13 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v12 + 104LL))(v12, &unk_4F9920C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
    goto LABEL_39;
  while ( *(_UNKNOWN **)v15 != &unk_4F99CB0 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_39;
  }
  v48 = 0;
  v45 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
                      *(_QWORD *)(v15 + 8),
                      &unk_4F99CB0)
                  + 160);
  if ( v13 )
    v13 += 160;
  if ( !*(_BYTE *)(a1 + 153) )
  {
    v39 = *(__int64 **)(a1 + 8);
    v40 = *v39;
    v41 = v39[1];
    if ( v41 == v40 )
      goto LABEL_39;
    while ( *(_UNKNOWN **)v40 != &unk_4F99308 )
    {
      v40 += 16;
      if ( v41 == v40 )
        goto LABEL_39;
    }
    v48 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v40 + 8) + 104LL))(
            *(_QWORD *)(v40 + 8),
            &unk_4F99308)
        + 160;
  }
  v17 = 0;
  if ( *(_BYTE *)(a1 + 154) )
  {
    v42 = *(__int64 **)(a1 + 8);
    v43 = *v42;
    v44 = v42[1];
    if ( v44 == v43 )
      goto LABEL_39;
    while ( *(_UNKNOWN **)v43 != &unk_4F97E48 )
    {
      v43 += 16;
      if ( v44 == v43 )
        goto LABEL_39;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v43 + 8) + 104LL))(
            *(_QWORD *)(v43 + 8),
            &unk_4F97E48)
        + 160;
  }
  v18 = *(__int64 **)(a1 + 8);
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
    goto LABEL_39;
  while ( *(_UNKNOWN **)v19 != &unk_4F96DB4 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_39;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F96DB4);
  v22 = *(__int64 **)(a1 + 8);
  v23 = *(_QWORD *)(v21 + 160);
  v24 = *v22;
  v25 = v22[1];
  if ( v24 == v25 )
    goto LABEL_39;
  while ( *(_UNKNOWN **)v24 != &unk_4F9B6E8 )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_39;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_4F9B6E8);
  v27 = *(__int64 **)(a1 + 8);
  v47 = v26 + 360;
  v28 = *v27;
  v29 = v27[1];
  if ( v28 == v29 )
    goto LABEL_39;
  while ( *(_UNKNOWN **)v28 != &unk_4F9E06C )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_39;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F9E06C);
  v31 = *(__int64 **)(a1 + 8);
  v32 = v30 + 160;
  v33 = *v31;
  v34 = v31[1];
  if ( v33 == v34 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F9D764 )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_39;
  }
  v46 = v32;
  v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(*(_QWORD *)(v33 + 8), &unk_4F9D764);
  v36 = sub_14CF090(v35, a2);
  return sub_191F4F0(a1 + 160, a2, v36, v46, v47, v23, a3, a4, a5, a6, v37, v38, a9, a10, v17, v48, v13, v45);
}
