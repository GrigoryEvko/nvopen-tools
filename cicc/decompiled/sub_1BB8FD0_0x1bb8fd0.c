// Function: sub_1BB8FD0
// Address: 0x1bb8fd0
//
__int64 __fastcall sub_1BB8FD0(
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
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // [rsp-10h] [rbp-B0h]
  __int64 v60; // [rsp+8h] [rbp-98h]
  __int64 v61; // [rsp+10h] [rbp-90h]
  __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 v63; // [rsp+28h] [rbp-78h]
  __int64 v64; // [rsp+30h] [rbp-70h]
  __int64 v65; // [rsp+38h] [rbp-68h]
  __int64 v66; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v67[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 (__fastcall *v68)(_QWORD *, _QWORD *, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v69)(_QWORD **); // [rsp+68h] [rbp-38h]

  v10 = 0;
  if ( !(unsigned __int8)sub_1636880(a1, a2) )
  {
    v12 = *(__int64 **)(a1 + 8);
    v13 = *v12;
    v14 = v12[1];
    if ( v13 == v14 )
LABEL_74:
      BUG();
    while ( *(_UNKNOWN **)v13 != &unk_4F9A488 )
    {
      v13 += 16;
      if ( v14 == v13 )
        goto LABEL_74;
    }
    v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
            *(_QWORD *)(v13 + 8),
            &unk_4F9A488);
    v16 = *(__int64 **)(a1 + 8);
    v61 = *(_QWORD *)(v15 + 160);
    v17 = *v16;
    v18 = v16[1];
    if ( v17 == v18 )
LABEL_69:
      BUG();
    while ( *(_UNKNOWN **)v17 != &unk_4F9920C )
    {
      v17 += 16;
      if ( v18 == v17 )
        goto LABEL_69;
    }
    v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
            *(_QWORD *)(v17 + 8),
            &unk_4F9920C);
    v20 = *(__int64 **)(a1 + 8);
    v21 = v19 + 160;
    v22 = *v20;
    v23 = v20[1];
    if ( v22 == v23 )
LABEL_70:
      BUG();
    while ( *(_UNKNOWN **)v22 != &unk_4F9D3C0 )
    {
      v22 += 16;
      if ( v23 == v22 )
        goto LABEL_70;
    }
    v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(
            *(_QWORD *)(v22 + 8),
            &unk_4F9D3C0);
    v25 = sub_14A4050(v24, a2);
    v26 = *(__int64 **)(a1 + 8);
    v27 = (__int64 *)v25;
    v28 = *v26;
    v29 = v26[1];
    if ( v28 == v29 )
LABEL_71:
      BUG();
    while ( *(_UNKNOWN **)v28 != &unk_4F9E06C )
    {
      v28 += 16;
      if ( v29 == v28 )
        goto LABEL_71;
    }
    v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(
            *(_QWORD *)(v28 + 8),
            &unk_4F9E06C);
    v31 = *(__int64 **)(a1 + 8);
    v64 = v30 + 160;
    v32 = *v31;
    v33 = v31[1];
    if ( v32 == v33 )
LABEL_68:
      BUG();
    while ( *(_UNKNOWN **)v32 != &unk_4F97E48 )
    {
      v32 += 16;
      if ( v33 == v32 )
        goto LABEL_68;
    }
    v63 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(
            *(_QWORD *)(v32 + 8),
            &unk_4F97E48)
        + 160;
    v34 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9B6E8, 1u);
    if ( v34 && (v35 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v34 + 104LL))(v34, &unk_4F9B6E8)) != 0 )
      v36 = v35 + 360;
    else
      v36 = 0;
    v37 = *(__int64 **)(a1 + 8);
    v38 = *v37;
    v39 = v37[1];
    if ( v38 == v39 )
LABEL_67:
      BUG();
    while ( *(_UNKNOWN **)v38 != &unk_4F96DB4 )
    {
      v38 += 16;
      if ( v39 == v38 )
        goto LABEL_67;
    }
    v40 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(
            *(_QWORD *)(v38 + 8),
            &unk_4F96DB4);
    v41 = *(__int64 **)(a1 + 8);
    v60 = *(_QWORD *)(v40 + 160);
    v42 = *v41;
    v43 = v41[1];
    if ( v42 == v43 )
LABEL_76:
      BUG();
    while ( *(_UNKNOWN **)v42 != &unk_4F9D764 )
    {
      v42 += 16;
      if ( v43 == v42 )
        goto LABEL_76;
    }
    v44 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v42 + 8) + 104LL))(
            *(_QWORD *)(v42 + 8),
            &unk_4F9D764);
    v45 = sub_14CF090(v44, a2);
    v46 = *(__int64 **)(a1 + 8);
    v65 = v45;
    v47 = *v46;
    v48 = v46[1];
    if ( v47 == v48 )
LABEL_75:
      BUG();
    while ( *(_UNKNOWN **)v47 != &unk_5051F8C )
    {
      v47 += 16;
      if ( v48 == v47 )
        goto LABEL_75;
    }
    v49 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v47 + 8) + 104LL))(
            *(_QWORD *)(v47 + 8),
            &unk_5051F8C);
    v50 = *(__int64 **)(a1 + 8);
    v66 = v49;
    v51 = *v50;
    v52 = v50[1];
    if ( v51 == v52 )
LABEL_72:
      BUG();
    while ( *(_UNKNOWN **)v51 != &unk_4F98D2C )
    {
      v51 += 16;
      if ( v52 == v51 )
        goto LABEL_72;
    }
    v53 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v51 + 8) + 104LL))(
            *(_QWORD *)(v51 + 8),
            &unk_4F98D2C);
    v54 = *(__int64 **)(a1 + 8);
    v62 = v53 + 160;
    v55 = *v54;
    v56 = v54[1];
    if ( v55 == v56 )
LABEL_73:
      BUG();
    while ( *(_UNKNOWN **)v55 != &unk_4F99CB0 )
    {
      v55 += 16;
      if ( v56 == v55 )
        goto LABEL_73;
    }
    v59 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v55 + 8) + 104LL))(
                        *(_QWORD *)(v55 + 8),
                        &unk_4F99CB0)
                    + 160);
    v67[0] = &v66;
    v69 = sub_1B8E120;
    v68 = sub_1B8E190;
    v10 = sub_1BB8E30(
            a1 + 160,
            a3,
            a4,
            a5,
            a6,
            v57,
            v58,
            a9,
            a10,
            a2,
            v61,
            v21,
            v27,
            v64,
            v63,
            v36,
            v62,
            v60,
            v65,
            (__int64)v67,
            v59);
    if ( v68 )
      v68(v67, v67, 3);
  }
  return v10;
}
