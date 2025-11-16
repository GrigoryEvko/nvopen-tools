// Function: sub_19C86F0
// Address: 0x19c86f0
//
__int64 __fastcall sub_19C86F0(
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
  unsigned int v12; // eax
  unsigned int v13; // r14d
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  char v32; // al
  double v33; // xmm4_8
  double v34; // xmm5_8

  v12 = sub_1404700(a1, a2);
  if ( (_BYTE)v12 )
  {
    return 0;
  }
  else
  {
    v15 = *(__int64 **)(a1 + 8);
    v13 = v12;
    v16 = *v15;
    v17 = v15[1];
    if ( v16 == v17 )
LABEL_31:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_4F9D764 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_31;
    }
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_4F9D764);
    v19 = sub_14CF090(v18, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
    v20 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 176) = v19;
    v21 = *v20;
    v22 = v20[1];
    if ( v21 == v22 )
LABEL_34:
      BUG();
    while ( *(_UNKNOWN **)v21 != &unk_4F9920C )
    {
      v21 += 16;
      if ( v22 == v21 )
        goto LABEL_34;
    }
    v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
            *(_QWORD *)(v21 + 8),
            &unk_4F9920C);
    v24 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 168) = a3;
    *(_QWORD *)(a1 + 160) = v23 + 160;
    v25 = *v24;
    v26 = v24[1];
    if ( v25 == v26 )
LABEL_33:
      BUG();
    while ( *(_UNKNOWN **)v25 != &unk_4F9E06C )
    {
      v25 += 16;
      if ( v26 == v25 )
        goto LABEL_33;
    }
    v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(
            *(_QWORD *)(v25 + 8),
            &unk_4F9E06C);
    v28 = *(__int64 **)(a1 + 8);
    *(_QWORD *)(a1 + 304) = v27 + 160;
    v29 = *v28;
    v30 = v28[1];
    if ( v29 == v30 )
LABEL_32:
      BUG();
    while ( *(_UNKNOWN **)v29 != &unk_4FBA0D0 )
    {
      v29 += 16;
      if ( v30 == v29 )
        goto LABEL_32;
    }
    v31 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v29 + 8) + 104LL))(
            *(_QWORD *)(v29 + 8),
            &unk_4FBA0D0);
    *(_QWORD *)(a1 + 296) = a2;
    *(_QWORD *)(a1 + 184) = v31 + 160;
    v32 = sub_1560180(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL) + 112LL, 44);
    *(_BYTE *)(a1 + 328) = v32;
    if ( v32 )
      sub_1436EA0(a1 + 336, a2);
    do
    {
      *(_BYTE *)(a1 + 289) = 0;
      v13 |= sub_19C7120(a1, a4, a5, a6, a7, v33, v34, a10, a11);
    }
    while ( *(_BYTE *)(a1 + 289) );
  }
  return v13;
}
