// Function: sub_288C520
// Address: 0x288c520
//
__int64 __fastcall sub_288C520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int8 v33; // al
  int v34; // ebx
  __int64 *v35; // r12
  __int64 v36; // [rsp+48h] [rbp-80h]
  __int64 v37; // [rsp+50h] [rbp-78h]
  __int64 v38; // [rsp+60h] [rbp-68h]
  __int64 **v39; // [rsp+68h] [rbp-60h]
  __int64 v40; // [rsp+70h] [rbp-58h]
  __int64 v41[2]; // [rsp+78h] [rbp-50h] BYREF
  __int64 *v42; // [rsp+88h] [rbp-40h]

  LODWORD(v3) = 0;
  if ( !(unsigned __int8)sub_D58140(a1, a2) )
  {
    v6 = *(__int64 **)(a1 + 8);
    v7 = *v6;
    v8 = v6[1];
    if ( v7 == v8 )
LABEL_42:
      BUG();
    while ( *(_UNKNOWN **)v7 != &unk_4F8144C )
    {
      v7 += 16;
      if ( v8 == v7 )
        goto LABEL_42;
    }
    v3 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
    v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F8144C);
    v10 = *(__int64 **)(a1 + 8);
    v36 = v9 + 176;
    v11 = *v10;
    v12 = v10[1];
    if ( v11 == v12 )
LABEL_47:
      BUG();
    while ( *(_UNKNOWN **)v11 != &unk_4F875EC )
    {
      v11 += 16;
      if ( v12 == v11 )
        goto LABEL_47;
    }
    v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
            *(_QWORD *)(v11 + 8),
            &unk_4F875EC);
    v14 = *(__int64 **)(a1 + 8);
    v40 = v13 + 176;
    v15 = *v14;
    v16 = v14[1];
    if ( v15 == v16 )
LABEL_46:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F881C8 )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_46;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F881C8);
    v18 = *(__int64 **)(a1 + 8);
    v19 = *(__int64 **)(v17 + 176);
    v20 = *v18;
    v21 = v18[1];
    if ( v20 == v21 )
LABEL_44:
      BUG();
    while ( *(_UNKNOWN **)v20 != &unk_4F89C28 )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_44;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
            *(_QWORD *)(v20 + 8),
            &unk_4F89C28);
    v23 = sub_DFED00(v22, v3);
    v24 = *(__int64 **)(a1 + 8);
    v39 = (__int64 **)v23;
    v25 = *v24;
    v26 = v24[1];
    if ( v25 == v26 )
LABEL_45:
      BUG();
    while ( *(_UNKNOWN **)v25 != &unk_4F8662C )
    {
      v25 += 16;
      if ( v26 == v25 )
        goto LABEL_45;
    }
    v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(
            *(_QWORD *)(v25 + 8),
            &unk_4F8662C);
    v28 = sub_CFFAC0(v27, v3);
    v29 = *(__int64 **)(a1 + 8);
    v30 = v28;
    v31 = *v29;
    v32 = v29[1];
    if ( v31 == v32 )
LABEL_43:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_4F8FC84 )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_43;
    }
    v37 = v30;
    v38 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
            *(_QWORD *)(v31 + 8),
            &unk_4F8FC84);
    sub_1049690(v41, v3);
    v33 = sub_BB9560(a1, (__int64)&unk_4F90E2C);
    v34 = sub_288A700(
            a2,
            v36,
            v40,
            v19,
            v39,
            v37,
            v41,
            0,
            0,
            (__int64 *)(v38 + 184),
            v33,
            *(_DWORD *)(a1 + 172),
            0,
            *(_BYTE *)(a1 + 176),
            *(_BYTE *)(a1 + 177),
            *(_QWORD *)(a1 + 180),
            *(_QWORD *)(a1 + 188),
            *(_WORD *)(a1 + 196),
            *(_WORD *)(a1 + 198),
            *(_WORD *)(a1 + 200),
            *(_WORD *)(a1 + 202),
            *(_WORD *)(a1 + 204),
            *(_QWORD *)(a1 + 208),
            0);
    if ( v34 == 2 )
      sub_D5B880(a3, a2);
    v35 = v42;
    LOBYTE(v3) = v34 != 0;
    if ( v42 )
    {
      sub_FDC110(v42);
      j_j___libc_free_0((unsigned __int64)v35);
    }
  }
  return (unsigned int)v3;
}
