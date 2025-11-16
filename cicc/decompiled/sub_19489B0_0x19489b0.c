// Function: sub_19489B0
// Address: 0x19489b0
//
__int64 __fastcall sub_19489B0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
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
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rax
  unsigned int v33; // r14d
  double v34; // xmm4_8
  double v35; // xmm5_8
  _QWORD *v36; // r13
  _QWORD *v37; // r12
  __int64 v38; // rax
  __int64 v40; // [rsp+0h] [rbp-210h]
  __int64 v41; // [rsp+8h] [rbp-208h]
  _QWORD v42[6]; // [rsp+10h] [rbp-200h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-1D0h]
  __int64 v44; // [rsp+48h] [rbp-1C8h]
  _BYTE v45[448]; // [rsp+50h] [rbp-1C0h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9920C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_47;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9920C);
  v14 = *(__int64 **)(a1 + 8);
  v40 = v13 + 160;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_48:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9A488 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_48;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9A488);
  v18 = *(__int64 **)(a1 + 8);
  v19 = *(_QWORD **)(v17 + 160);
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_49:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F9E06C )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_49;
  }
  v41 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F9E06C)
      + 160;
  v22 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9B6E8, 1u);
  if ( v22 && (v23 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v22 + 104LL))(v22, &unk_4F9B6E8)) != 0 )
    v24 = v23 + 360;
  else
    v24 = 0;
  v25 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9D3C0, 1u);
  if ( v25 && (v26 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v25 + 104LL))(v25, &unk_4F9D3C0)) != 0 )
    v27 = sub_14A4050(v26, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  else
    v27 = 0;
  v28 = sub_157EB90(**(_QWORD **)(a2 + 32));
  v29 = sub_1632FA0(v28);
  if ( (unsigned int)sub_193DD90(a2) > LODWORD(qword_4FAF440[20]) )
    return 0;
  v30 = sub_1481F60(v19, a2, a3, a4);
  if ( !*(_WORD *)(v30 + 24) )
  {
    v31 = *(_QWORD *)(v30 + 32);
    v32 = *(_QWORD **)(v31 + 24);
    if ( *(_DWORD *)(v31 + 32) > 0x40u )
      v32 = (_QWORD *)*v32;
    if ( v32 != (_QWORD *)1 )
      goto LABEL_24;
    return 0;
  }
  if ( LOBYTE(qword_4FAF520[20])
    && ((int)sub_1CED350(a2) <= 1 || !(unsigned __int8)sub_1CED620(a2, **(_QWORD **)(a2 + 32))) )
  {
    return 0;
  }
LABEL_24:
  v42[4] = v24;
  v42[3] = v29;
  v42[0] = v40;
  v42[5] = v27;
  v42[2] = v41;
  v42[1] = v19;
  v43 = v45;
  v44 = 0x1000000000LL;
  v45[384] = 0;
  v33 = sub_13FCBF0(a2);
  if ( (_BYTE)v33 )
    v33 = sub_1945A50((__int64)v42, a2, a3, a4, a5, a6, v34, v35, a9, a10);
  v36 = v43;
  v37 = &v43[24 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v37 )
  {
    do
    {
      v38 = *(v37 - 1);
      v37 -= 3;
      if ( v38 != -8 && v38 != 0 && v38 != -16 )
        sub_1649B30(v37);
    }
    while ( v36 != v37 );
    v37 = v43;
  }
  if ( v37 != (_QWORD *)v45 )
    _libc_free((unsigned __int64)v37);
  return v33;
}
