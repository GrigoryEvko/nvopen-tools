// Function: sub_1A8FB10
// Address: 0x1a8fb10
//
__int64 __fastcall sub_1A8FB10(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r13d
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 *v35; // r8
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 v38; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v39[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v40)(_QWORD *, _QWORD *, int); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v41)(_QWORD **); // [rsp+28h] [rbp-38h]

  v10 = 0;
  if ( !(unsigned __int8)sub_1636880(a1, a2) )
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *v12;
    v14 = v12[1];
    if ( v13 == v14 )
LABEL_34:
      BUG();
    while ( *(_UNKNOWN **)v13 != &unk_4F9920C )
    {
      v13 += 16;
      if ( v14 == v13 )
        goto LABEL_34;
    }
    v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
            *(_QWORD *)(v13 + 8),
            &unk_4F9920C);
    v16 = *(__int64 **)(a1 + 8);
    v17 = v15 + 160;
    v18 = *v16;
    v19 = v16[1];
    if ( v18 == v19 )
LABEL_35:
      BUG();
    while ( *(_UNKNOWN **)v18 != &unk_5051F8C )
    {
      v18 += 16;
      if ( v19 == v18 )
        goto LABEL_35;
    }
    v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(
            *(_QWORD *)(v18 + 8),
            &unk_5051F8C);
    v21 = *(__int64 **)(a1 + 8);
    v38 = v20;
    v22 = *v21;
    v23 = v21[1];
    if ( v22 == v23 )
LABEL_36:
      BUG();
    while ( *(_UNKNOWN **)v22 != &unk_4F9E06C )
    {
      v22 += 16;
      if ( v23 == v22 )
        goto LABEL_36;
    }
    v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(
            *(_QWORD *)(v22 + 8),
            &unk_4F9E06C);
    v25 = *(__int64 **)(a1 + 8);
    v26 = v24 + 160;
    v27 = *v25;
    v28 = v25[1];
    if ( v27 == v28 )
LABEL_37:
      BUG();
    while ( *(_UNKNOWN **)v27 != &unk_4F9A488 )
    {
      v27 += 16;
      if ( v28 == v27 )
        goto LABEL_37;
    }
    v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
            *(_QWORD *)(v27 + 8),
            &unk_4F9A488);
    v30 = *(__int64 **)(a1 + 8);
    v31 = *(_QWORD *)(v29 + 160);
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
    v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(
            *(_QWORD *)(v32 + 8),
            &unk_4F99CB0);
    v41 = sub_1A891B0;
    v35 = *(__int64 **)(v34 + 160);
    v39[0] = &v38;
    v40 = sub_1A89220;
    v10 = sub_1A8CD80(a2, v17, v26, v31, v35, (__int64)v39, a3, a4, a5, a6, v36, v37, a9, a10);
    if ( v40 )
      v40(v39, v39, 3);
  }
  return v10;
}
